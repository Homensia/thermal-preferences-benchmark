"""Tests of E9b, the NN analysis as in the paper (primary 1-NN, Case B, Case C on F12) restored from the
pre-revision code, "the snapshot" (analysis/nn_cases.py).

Tiny synthetic data only; no model is fitted. The restored functions are
compared with the snapshot source itself (the pre-revision ``analysis/nn_disagreement.py``, frozen verbatim in
``tests/data/snapshot_nn_disagreement.py.txt`` and identified by its git blob id) on data whose
categories are already canonical, so that the only intended change in the
definitions is the category-normalised data path.

Run from the repository root:
    .venv-revision/bin/python -m pytest tests/test_nn_cases.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "analysis"))
sys.path.insert(0, str(REPO_DIR))

import nn_cases as nc  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY  # noqa: E402

#: prefix of the "source" field written into the E9b detail files
SNAPSHOT_SOURCE_PREFIX = "pre-revision analysis/nn_disagreement.py (frozen copy: "
PROHIBITED = ("ceiling", "bound", "demonstrat", "informational")


# frozen copy of the snapshot file (a clone without the tag keeps the parity evidence); its git blob id
# is the one of the tag, checked below
SNAPSHOT_COPY = REPO_DIR / "tests" / "data" / "snapshot_nn_disagreement.py.txt"
SNAPSHOT_BLOB = "d0cee235b020751158cd5507ac7dcb090a29ffb7"


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _snapshot_source() -> str:
    """The snapshot nn_disagreement.py: the frozen copy, whose git blob id must be the snapshot's."""
    data = SNAPSHOT_COPY.read_bytes()
    assert _git_blob_id(data) == SNAPSHOT_BLOB, f"{SNAPSHOT_COPY} is not the snapshot file"
    return data.decode("utf-8")


def _snapshot_module():
    """The snapshot nn_disagreement.py executed in a namespace."""
    ns = {"__name__": "snapshot_nn_disagreement", "__file__": str(REPO_DIR / "analysis" / "_snapshot_nn.py")}
    exec(compile(_snapshot_source(), "snapshot_nn_disagreement.py", "exec"), ns)
    return ns


def test_frozen_snapshot_copy_is_the_tagged_file():
    assert "def case_b_identical_groups" in _snapshot_source()


def _features_frame(n, rng, dup_every=4):
    """F17-like frame: canonical categories, coarse numerical values (many groups and margin pairs)."""
    num, cat = nc.feature_set("features")
    df = pd.DataFrame({c: rng.integers(0, 3, n).astype(float) * 0.5 for c in num})
    df["Tair"] = 22 + rng.integers(0, 4, n) * 0.6
    df["RH"] = 40 + rng.integers(0, 4, n) * 3.0
    df["Âge"] = rng.choice([25.0, 35.0], n)
    for c in cat:                                              # one value for Climate / Building_type
        df[c] = rng.choice(GLOBAL_VOCABULARY[c][:1 if c in ("Climate", "Building_type") else 2], n)
    for i in range(0, n - 1, dup_every):                       # exact duplicates
        df.loc[i + 1, num + cat] = df.loc[i, num + cat]
    return df[num + cat]


# ================================================================ restored definitions = snapshot
def test_restored_constants_are_the_snapshot_constants():
    snap = _snapshot_module()
    assert nc.NUM_TOLERANCES == snap["NUM_TOLERANCES"]
    assert nc.CATEGORICAL_COLS == snap["CATEGORICAL_COLS"]


def test_e9b_feature_set_is_f12_the_papers_nn_feature_set():
    """The snapshot run cited by the paper used --features_key features_12 (its default)."""
    assert nc.FROZEN_FEATURES_KEY == "features_12"
    assert list(nc.FROZEN_F12) == list(nc.load_config(str(nc.CONFIG_PATH))["data"]["features_12"])
    assert '"--features_key",\n        type=str,\n        default="features_12"' in _snapshot_source()
    assert "--scope full --features_key features_12" in _snapshot_source()


def test_encoding_equals_the_snapshot_encoding():
    snap = _snapshot_module()
    rng = np.random.default_rng(4)
    X = _features_frame(60, rng)
    np.testing.assert_array_equal(nc.encode_features(X), snap["_encode_features"](X))


@pytest.mark.parametrize("seed", [0, 1])
def test_primary_equals_the_snapshot_without_duplicates(seed):
    """Without exact duplicates the snapshot primary_disagreement is unchanged by the self-exclusion fix."""
    snap = _snapshot_module()
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"Tair": rng.normal(24, 2, 120), "RH": rng.normal(50, 10, 120),
                      "Sexe": rng.choice(["male", "female"], 120)})
    y = rng.integers(-1, 2, len(X))
    Xs = nc.encode_features(X)
    a, sa = nc.primary_disagreement(Xs, y), snap["primary_disagreement"](Xs, y)
    for k in ("n_obs", "n_pairs_considered", "disagreement_rate", "agreement_rate", "n_disagree", "n_agree"):
        assert a[k] == sa[k], k
    assert a["n_zero_distance_neighbour"] == 0 and a["n_self_in_second_column"] == 0


def test_primary_excludes_the_query_by_row_position_with_duplicates():
    """With exact duplicates the search may return the duplicate first and the query second; the snapshot
    then compared the query with itself. The fix takes the first returned row that is not the query."""
    rng = np.random.default_rng(9)
    X = _features_frame(80, rng, dup_every=2)          # rows 2i and 2i+1 identical
    y = rng.integers(-1, 2, len(X))
    Xs = nc.encode_features(X)
    a = nc.primary_disagreement(Xs, y)
    # brute force: each query's neighbour is at the minimum distance over the other rows; with a duplicate
    # that distance is 0, so the neighbour of an exact twin is its twin (or another duplicate)
    D = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(Xs).kneighbors(Xs)
    own = np.arange(len(y))
    nb = np.where(idx[:, 0] == own, idx[:, 1], idx[:, 0])
    assert not np.any(nb == own)
    np.testing.assert_allclose(D[own, nb], D.min(axis=1))
    assert a["n_disagree"] == int(np.sum(y != y[nb]))
    assert a["n_zero_distance_neighbour"] == int(np.sum(D.min(axis=1) == 0)) >= 40
    assert a["n_self_in_second_column"] == int(np.sum(idx[:, 1] == own))
    # chance: 1 - sum_c p_c^2 over the query votes
    assert a["chance_pair_disagreement"] == pytest.approx(np.mean(y[:, None] != y[None, :]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_case_b_and_case_c_equal_the_snapshot_functions(seed):
    snap = _snapshot_module()
    rng = np.random.default_rng(seed)
    X = _features_frame(80, rng)
    y = rng.integers(-1, 2, len(X)).astype(float)
    b, sb = nc.case_b_identical_groups(X, y), snap["case_b_identical_groups"](X, y)
    for k in ("n_clusters", "n_clusters_all_agree", "cluster_agreement_rate", "n_pairs_considered",
              "n_disagree_pairs", "pair_agreement_rate", "pair_disagreement_rate"):
        assert b[k] == sb[k], k
    c, sc = nc.case_c_physio_margins(X, y), snap["case_c_physio_margins"](X, y)
    for k in ("n_pairs_considered", "agreement_rate", "disagreement_rate", "n_disagree", "n_agree"):
        assert c[k] == sc[k], k
    assert b["n_clusters"] > 0 and c["n_pairs_considered"] > b["n_pairs_considered"]


# ================================================================ hand examples and chance references
def test_case_b_hand_example_and_chance():
    X = pd.DataFrame({"Tair": [20.0001, 20.0004, 21.0, 21.0, 21.0, 25.0],
                      "Sexe": ["male", "male", "female", "female", "female", "male"]})
    y = np.array([1, 2, 1, 1, 2, 3])
    b = nc.case_b_identical_groups(X, y)
    # groups {0,1} (equal after rounding to 3 decimals) and {2,3,4}; row 5 alone
    assert b["n_clusters"] == 2 and b["n_rows_in_clusters"] == 5 and b["cluster_sizes"] == {"2": 1, "3": 1}
    assert b["n_clusters_all_agree"] == 0 and b["cluster_agreement_rate"] == 0.0
    assert b["n_pairs_considered"] == 1 + 3 and b["n_disagree_pairs"] == 1 + 2
    # members (each once): labels 1, 2, 1, 1, 2 -> q = (3/5, 2/5)
    q1, q2 = 3 / 5, 2 / 5
    assert b["chance_cluster_agreement"] == pytest.approx(((q1 ** 2 + q2 ** 2) + (q1 ** 3 + q2 ** 3)) / 2)
    # endpoints (once per pair): group 1 members x1, group 2 members x2 -> labels 1,2 + (1,1,2) x2
    p1, p2 = (1 + 4) / 8, (1 + 2) / 8
    assert b["chance_pair_disagreement"] == pytest.approx(1 - p1 ** 2 - p2 ** 2)
    shares = np.array([3, 2, 1]) / 6
    assert b["chance_disagreement_cohort"] == pytest.approx(1 - np.sum(shares ** 2))


def test_case_c_hand_example_inclusive_margins_and_unconstrained_variables():
    X = pd.DataFrame({"Tair": [20.0, 21.0, 21.0, 20.0], "RH": [50.0, 55.0, 50.0, 50.0],
                      "Trm_ema_28": [0.0, 100.0, -50.0, 3.0],          # no tolerance: never constrains
                      "Sexe": ["male", "male", "male", "female"], "Âge": [30.0, 30.0, 30.0, 30.0]})
    y = np.array([0, 0, 1, 0])
    c = nc.case_c_physio_margins(X, y)
    # pairs (0,1) |dTair| = 1, |dRH| = 5 (inclusive); (0,2); (1,2); row 3 differs by Sexe
    assert c["n_pairs_considered"] == 3 and c["n_disagree"] == 2 and c["n_rows_in_pairs"] == 3
    assert c["agreement_rate"] == pytest.approx(1 / 3)
    # endpoints: each of rows 0, 1, 2 enters 2 pairs -> labels 0, 0, 1 equally weighted
    assert c["chance_pair_disagreement"] == pytest.approx(1 - (2 / 3) ** 2 - (1 / 3) ** 2)
    assert nc.case_c_variables(X.columns) == (["Tair", "RH"], ["Sexe", "Âge"], ["Trm_ema_28"])
    X.loc[3, "Âge"] = 30.5                                          # Âge: strict equality (snapshot)
    X.loc[3, "Sexe"] = "male"
    assert nc.case_c_physio_margins(X, y)["n_pairs_considered"] == 3


def test_case_c_matches_brute_force_pairs():
    rng = np.random.default_rng(5)
    X = _features_frame(60, rng)
    y = rng.integers(0, 3, len(X))
    num, cat, _ = nc.case_c_variables(X.columns)
    pairs = [(i, j) for i, j in combinations(range(len(X)), 2)
             if all(X[k].iat[i] == X[k].iat[j] for k in cat)
             and all(abs(X[k].iat[i] - X[k].iat[j]) <= nc.NUM_TOLERANCES[k] for k in num)]
    c = nc.case_c_physio_margins(X, y)
    assert c["n_pairs_considered"] == len(pairs)
    assert c["n_disagree"] == sum(y[i] != y[j] for i, j in pairs)
    ends = np.array([y[i] for i, _ in pairs] + [y[j] for _, j in pairs])
    p = np.bincount(ends) / len(ends)
    assert c["chance_pair_disagreement"] == pytest.approx(1 - np.sum(p ** 2))


def test_empty_cases_give_not_available_values():
    X = pd.DataFrame({"Tair": [10.0, 20.0, 30.0], "Sexe": ["male", "female", "male"]})
    y = np.array([0, 1, 0])
    b, c = nc.case_b_identical_groups(X, y), nc.case_c_physio_margins(X, y)
    assert b["n_clusters"] == 0 and np.isnan(b["cluster_agreement_rate"]) and np.isnan(b["chance_pair_disagreement"])
    assert c["n_pairs_considered"] == 0 and np.isnan(c["agreement_rate"])


# ================================================================ category-normalised data path and outputs
def _cohort_csv(tmp_path, n=40, seed=3, raw_case=False):
    rng = np.random.default_rng(seed)
    df = _features_frame(n, rng)
    if raw_case:                       # raw spellings that the snapshot compared as different strings
        df.loc[1::2, "Sexe"] = df.loc[1::2, "Sexe"].str.upper()
        df.loc[1::2, "cooling type"] = df.loc[1::2, "cooling type"].str.title()
    for t in nc.TARGETS:
        df[t] = rng.integers(-1, 2, n)
    csv = tmp_path / ("cohort_raw.csv" if raw_case else "cohort.csv")
    df.to_csv(csv, index=False)
    return csv


def test_c2_categories_are_normalised_before_grouping(tmp_path):
    canonical = nc.run(_cohort_csv(tmp_path), tmp_path / "a", "features_12", list(nc.TARGETS))
    raw = nc.run(_cohort_csv(tmp_path, raw_case=True), tmp_path / "b", "features_12", list(nc.TARGETS))
    cols = ["case_b_clusters", "case_b_pairs", "case_b_cluster_agreement", "case_c_pairs", "case_c_agreement",
            "primary_disagreement"]
    pd.testing.assert_frame_equal(canonical[cols], raw[cols])
    assert (canonical.case_b_clusters > 0).all()
    # the snapshot string comparison separates the raw spellings: fewer groups and pairs on the raw file
    snap = _snapshot_module()
    df = pd.read_csv(tmp_path / "cohort_raw.csv")
    num, cat = nc.feature_set("features_12")
    sb = snap["case_b_identical_groups"](df[num + cat], df[nc.TARGETS[0]].to_numpy())
    assert sb["n_clusters"] < canonical.case_b_clusters.iloc[0]


def test_end_to_end_exploratory_outputs_labelled_and_never_overwritten(tmp_path):
    csv = _cohort_csv(tmp_path)
    out = tmp_path / "E9b"
    with pytest.raises(SystemExit):
        nc.main(["--data_csv", str(csv), "--output_dir", str(out)])          # --exploratory required
    assert not out.exists()
    nc.main(["--exploratory", "--data_csv", str(csv), "--output_dir", str(out)])
    s = pd.read_csv(out / "E9b_nn_cases_exploratory_summary.csv")
    assert (s.run_mode == "exploratory").all() and (s.experiment == "E9b_exploratory").all()
    assert s.target.tolist() == list(nc.TARGETS) and (s.n_features == 12).all() and (s.population == "full").all()
    assert (s.features_key == "features_12").all()
    assert s.frozen_definition.all()                  # mathematical definition flag only
    for col in ("primary_disagreement", "primary_agreement", "primary_chance_disagreement",
                "case_b_cluster_agreement", "case_b_chance_cluster_agreement", "case_c_disagreement",
                "case_c_chance_pair_disagreement", "case_b_chance_pair_disagreement", "chance_disagreement_cohort"):
        assert s[col].between(0, 1).all(), col
    d = json.loads((out / f"E9b_nn_cases_exploratory_detail_{nc.TARGETS[0]}.json").read_text())
    assert d["case_c_unconstrained_variables"] == []                     # F12: every variable constrained
    assert d["variables"] == list(nc.FROZEN_F12) and set(d["primary"]) >= {"disagreement_rate",
                                                                            "chance_pair_disagreement"}
    assert d["case_c_strict_equality"] == ["Âge", "Sexe", "Season", "Climate", "Building_type", "cooling type"]
    assert d["source"].startswith(SNAPSHOT_SOURCE_PREFIX) and "protocol" not in d["analysis"]
    assert (REPO_DIR / "tests" / "data" / "snapshot_nn_disagreement.py.txt").is_file()
    assert all(f.name.startswith("E9b_nn_cases_exploratory_") for f in out.iterdir())
    for f in out.iterdir():
        assert not any(w in f.read_text().lower() for w in PROHIBITED), f.name
    with pytest.raises(FileExistsError, match="never overwritten"):
        nc.run(csv, out, "features_12", list(nc.TARGETS))


def test_real_mode_refuses_non_frozen_parameters_and_unpinned_file(tmp_path):
    csv = _cohort_csv(tmp_path)
    with pytest.raises(ValueError, match="F12 .* and every target"):
        nc.run(csv, tmp_path / "r1", "features_12", ["TSV_3p"], mode="real")
    with pytest.raises(ValueError, match="F12 .* and every target"):           # F17 is not the paper's NN set
        nc.run(csv, tmp_path / "r3", "features", list(nc.TARGETS), mode="real")
    with pytest.raises(ValueError, match="pinned"):
        nc.run(csv, tmp_path / "r2", "features_12", list(nc.TARGETS), mode="real")
    assert not any((tmp_path / r).exists() for r in ("r1", "r2", "r3"))


def test_module_text_has_no_banned_wording():
    text = Path(nc.__file__).read_text().lower()
    assert not any(w in text for w in PROHIBITED)
