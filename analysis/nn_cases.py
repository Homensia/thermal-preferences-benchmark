"""NN analysis as in the paper: primary 1-NN, Case B and Case C, descriptive
(experiment E9b).

Restored from ``analysis/nn_disagreement.py`` of the pre-revision code (below:
"the snapshot"; ``_encode_features``, ``primary_disagreement``, ``case_b_identical_groups``,
``case_c_physio_margins``, ``NUM_TOLERANCES``, ``CATEGORICAL_COLS``), the code
of the nearest-neighbour numbers the paper cites (primary 57 / 44 / 41 %
disagreement, Case B 64 / 77 / 77 %, Case C 44 / 59 / 60 % agreement), with
minimal fixes. Supplementary descriptive statistics only: they say how often
votes differ between rows that are identical, or close, in the feature space;
they are never described as a limit on the accuracy of any model. No model is
fitted.

Definitions (snapshot, unchanged)
---------------------------------
Primary analysis (snapshot module docstring, quoted)
    "For each test-set observation find its single nearest neighbour (K=1, L2
    distance in the standardised feature space); report the fraction of pairs
    whose labels disagree." Run with ``--scope full``, every cohort row is a
    query and its neighbour is searched among the other cohort rows.
    Encoding (snapshot ``_encode_features``, quoted): "Integer-encode
    categoricals + standardize numerics" (categorical codes of the stripped
    strings, then ``StandardScaler`` on every column); search
    ``NearestNeighbors(n_neighbors=2, metric="euclidean")``.
Case B, strictly identical feature groups
    Rows are grouped by their feature vector, numerical variables rounded to 3
    decimals, categorical variables compared exactly. For the groups of size
    >= 2: ``cluster_agreement_rate`` = share of groups whose members all have
    the same label (the convention of the snapshot and of the original thesis, cited by the paper as
    Case B); ``pair_agreement_rate`` = share of within-group unordered pairs
    with the same label.
Case C, pairs within tight margins
    Every unordered pair of rows whose categorical variables of
    ``CATEGORICAL_COLS`` present in the feature set are equal (``Âge`` by strict
    equality, as in the snapshot) and whose numerical variables of
    ``NUM_TOLERANCES`` present in the feature set differ by at most Tair ±1 °C,
    Tout ±1 °C, RH ±5 %, vel ±0.3 m/s, clo ±0.3, Met ±0.2. Other variables of
    the feature set do not constrain the pairs (snapshot behaviour; listed in
    the output). ``agreement_rate`` = share of these pairs with the same label.
Feature space and population (the paper's method)
    F12 (``config.yaml`` key ``features_12``: Tair, RH, clo, vel, Âge, Tout,
    Met, Sexe, Season, Climate, Building_type, cooling type), the feature set
    of the snapshot run cited by the paper (``nn_disagreement.py --scope full
    --features_key features_12``); population = the full cohort (snapshot
    ``--scope full``: every non-null row of the cohort CSV, the methodology of
    the original thesis, which operated on the entire 15k-row cohort to
    maximise the chance of finding exact-feature duplicate clusters for Case B).
    No split is read.

Minimal fixes (relative to the snapshot)
----------------------------------------
* data: loaded through ``load_cohort`` (the revision data path): category
  normalisation (the snapshot compared raw strings, so
  categories differing only in case or spelling never matched), a missing or
  non-finite value raises instead of dropping the row, ``row_idx`` = 0-based
  position in the cohort CSV;
* primary: the query itself is excluded by its row position. The snapshot took
  the second column of ``kneighbors`` as "the neighbour" ("column 0 is the
  point itself"), which fails for exact duplicates: the search can return a
  duplicate first and the query second, and the query was then compared with
  itself. The neighbour is now the first returned row that is not the query (a
  duplicate at distance 0 is a legitimate neighbour); the number of queries
  affected is reported (``n_self_in_second_column``);
* chance references beside each value, on the same population (pairs:
  ``1 - Σ p_c²``; Case B groups: mean over the groups of ``Σ q_c^{n_g}``; see
  below);
* outputs labelled by run mode and never overwritten (as E9).

Chance references (same population as the value they accompany)
----------------------------------------------------------------
* ``primary`` ``chance_pair_disagreement``: ``1 - Σ p_c²``, p_c = class shares
  of the query votes (every cohort vote), the expected disagreement between a
  query vote and a vote drawn independently from the same distribution;
* ``chance_pair_disagreement`` (Case B pairs, Case C pairs): ``1 - Σ p_c²``,
  p_c = class shares of the votes entering the pairs, each vote counted once
  per pair it belongs to (expected disagreement of a pair whose two votes are
  drawn independently from that distribution);
* ``chance_cluster_agreement`` (Case B groups): mean over the groups of
  ``Σ_c q_c^{n_g}``, q_c = class shares of the group members (each once), n_g
  the group size (= ``Σ q_c²`` when every group is a pair);
* ``chance_disagreement_cohort``: ``1 - Σ p_c²`` over every cohort row.

Outputs (in the output directory, which must be new or empty)
--------------------------------------------------------------
    {label}_summary.csv                  one row per target
    {label}_detail_{target}.json         counts, variables, inputs and hashes
    {label}_category_coverage.json       category coverage log

``{label}``: ``E9b_nn_cases`` (real, only through ``run_revision.py e9b``),
``E9b_nn_cases_dry_run`` (runner dry run), ``E9b_nn_cases_exploratory``
(every standalone call).

Usage
-----
    python3 run_revision.py e9b --run_dir rerun_2026-10-XX_v2       # -> RUN/E9b_nn_cases
    python3 analysis/nn_cases.py --exploratory --output_dir <new directory>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_ANALYSIS_DIR = Path(__file__).resolve().parent
_TPB_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_TPB_ROOT), str(_ANALYSIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from category_normalisation import (  # noqa: E402
    CATEGORICAL_VARIABLES, normalise_categories, write_coverage_log,
)
from config_loader import load_config, read_consumed  # noqa: E402

CONFIG_PATH = _TPB_ROOT / "config.yaml"
TARGETS = ("thermal_sensation", "TSV_3p", "thermal_preference")
DEFAULT_DATA_CSV = _TPB_ROOT / "Data" / "ASHRAE_2022_Clean_api.csv"
# Archived artefact of the pre-revision run that must never be written to.
PROTECTED_DIR = _TPB_ROOT / "rerun_2026-04-28_no_reweighting"
# the paper's NN analysis: F12 (snapshot run ``--features_key features_12``), config.yaml order
FROZEN_FEATURES_KEY = "features_12"
FROZEN_F12 = ("Tair", "RH", "clo", "vel", "Âge", "Tout", "Met", "Sexe", "Season", "Climate", "Building_type",
              "cooling type")
POPULATION = "full"
ROUND_DECIMALS = 3                                   # snapshot case_b_identical_groups(round_decimals=3)

# --- cohort inputs (the data path of the E9 loader, unchanged) ---
def config_sha256(config) -> str:
    """SHA-256 of a configuration (sorted-key JSON; the digest of the run manifests)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def feature_set(features_key: str, config: dict | None = None) -> tuple[list[str], list[str]]:
    """Return (numerical, categorical) variables of a feature set of ``config``
    (default: config.yaml read now)."""
    config = load_config(str(CONFIG_PATH)) if config is None else config
    features = config["data"].get(features_key)
    if features is None:
        raise KeyError(f"Unknown features key '{features_key}' in config.yaml")
    categorical = [c for c in features if c in CATEGORICAL_VARIABLES]
    numerical = [c for c in features if c not in CATEGORICAL_VARIABLES]
    return numerical, categorical


def load_cohort(data_csv: Path, numerical: list[str], categorical: list[str],
                targets: list[str], raw: bytes | None = None) -> tuple[pd.DataFrame, list]:
    """Load the cohort; row_idx = 0-based row position in the CSV. No row filter:
    a missing or non-finite value in a used column raises; categories go through
    the shared category normalisation (unknown or missing value raises). Returns the
    frame and the coverage records. ``raw``: the bytes of ``data_csv``
    already read once by the caller, parsed from memory."""
    df = pd.read_csv(io.BytesIO(raw) if raw is not None else data_csv)
    missing = [c for c in numerical + categorical + targets if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {data_csv}: {missing}")
    df = df.reset_index(drop=True)
    for col in numerical + targets:
        values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{col}: missing or non-finite values in {data_csv}")
    df, coverage = normalise_categories(df, Path(data_csv).name, categorical)
    df.insert(0, "row_idx", np.arange(len(df), dtype=np.int64))
    return df, coverage


#: origin of the definitions, written into the E9b detail files and run manifest
SOURCE = "pre-revision analysis/nn_disagreement.py (frozen copy: tests/data/snapshot_nn_disagreement.py.txt)"


# --- snapshot constants, verbatim (snapshot nn_disagreement.py l. 75-83) ---
NUM_TOLERANCES = {
    "Tair": 1.0,
    "Tout": 1.0,
    "RH": 5.0,
    "vel": 0.3,
    "clo": 0.3,
    "Met": 0.2,
}
CATEGORICAL_COLS = {"Sexe", "Season", "Climate", "Building_type", "cooling type", "Âge"}

RUN_LABELS = {"real": "E9b_nn_cases", "dry_run": "E9b_nn_cases_dry_run", "exploratory": "E9b_nn_cases_exploratory"}
RUN_EXPERIMENTS = {"real": "E9b", "dry_run": "E9b_dry_run", "exploratory": "E9b_exploratory"}


# ---------------------------------------------------------------------------
# Chance references
# ---------------------------------------------------------------------------
def class_shares(labels, weights=None) -> dict:
    """Class -> share of ``labels`` (optionally weighted); empty -> {}."""
    s = pd.Series(np.asarray(labels))
    w = pd.Series(np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float))
    tot = float(w.sum())
    if tot <= 0:
        return {}
    return {k: float(v) / tot for k, v in w.groupby(s.to_numpy()).sum().items() if v > 0}


def chance_disagreement(shares: dict) -> float:
    """``1 - Σ p_c²``; NaN without votes."""
    return float(1.0 - sum(p * p for p in shares.values())) if shares else float("nan")


# ---------------------------------------------------------------------------
# Primary 1-NN (snapshot _encode_features / primary_disagreement)
# ---------------------------------------------------------------------------
def encode_features(X: pd.DataFrame) -> np.ndarray:
    """Integer-encode categoricals + standardize numerics; return dense matrix (snapshot, unchanged)."""
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            # Integer codes for categorical (stable across rows)
            X[col] = pd.Categorical(X[col].astype(str).str.strip()).codes.astype(float)
    scaler = StandardScaler()
    return scaler.fit_transform(X.to_numpy(dtype=float))


def primary_disagreement(X_std: np.ndarray, y: np.ndarray) -> dict:
    """K=1 nearest-neighbour disagreement rate in the standardized feature space (snapshot search,
    ``NearestNeighbors(n_neighbors=2, metric="euclidean")``). Fix: the neighbour is the first returned row
    that is not the query (self excluded by row position, not by column position)."""
    y = np.asarray(y)
    n = len(y)
    _, idx = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X_std).kneighbors(X_std)
    own = np.arange(n)
    neighbours = np.where(idx[:, 0] == own, idx[:, 1], idx[:, 0])
    if np.any(neighbours == own):
        raise AssertionError("a query is its own neighbour")
    disagree = int(np.sum(y != y[neighbours]))
    return {
        "n_obs": n,
        "n_pairs_considered": n,
        "disagreement_rate": disagree / n,
        "agreement_rate": (n - disagree) / n,
        "n_disagree": disagree,
        "n_agree": n - disagree,
        # neighbour with an identical encoded vector (exact comparison; the returned distances carry rounding)
        "n_zero_distance_neighbour": int(np.sum(np.all(X_std[neighbours] == X_std, axis=1))),
        # queries the snapshot compared with themselves (its idx[:, 1] was the query): exact duplicates only
        "n_self_in_second_column": int(np.sum(idx[:, 1] == own)),
        "chance_pair_disagreement": chance_disagreement(class_shares(y)),
    }


# ---------------------------------------------------------------------------
# Case B (snapshot case_b_identical_groups; rows keyed by row_idx)
# ---------------------------------------------------------------------------
def case_b_identical_groups(X: pd.DataFrame, y: np.ndarray, round_decimals: int = ROUND_DECIMALS) -> dict:
    """Cluster rows by rounded feature vectors; compute two disagreement metrics.

    Snapshot definition, unchanged: numerical variables rounded to
    ``round_decimals``, categorical variables compared as stripped strings
    (after the category normalisation of the loader); groups of size >= 2.

    * cluster_agreement_rate: fraction of clusters where all members share the
      same label (convention of the original thesis).
    * pair_agreement_rate: fraction of within-cluster unordered pairs whose
      labels agree (weights larger clusters more heavily).

    Added: the chance references on the same rows and the group sizes.
    """
    y = np.asarray(y)
    X_round = X.copy()
    for col in X_round.columns:
        if pd.api.types.is_numeric_dtype(X_round[col]):
            X_round[col] = X_round[col].round(round_decimals)
        else:
            X_round[col] = X_round[col].astype(str).str.strip()
    key = X_round.apply(lambda r: tuple(r.values), axis=1)
    df = pd.DataFrame({"key": key, "y": y, "pos": np.arange(len(y))})
    total_pairs = 0
    total_disagree = 0
    clusters_all_agree = 0
    clusters_total = 0
    sizes, member_pos = [], []
    for _key_val, group in df.groupby("key", sort=True):
        n = len(group)
        if n < 2:
            continue
        clusters_total += 1
        ys = group["y"].to_numpy()
        if len(set(ys)) == 1:
            clusters_all_agree += 1
        disagree = 0
        for i, j in combinations(range(n), 2):
            if ys[i] != ys[j]:
                disagree += 1
        pairs = n * (n - 1) // 2
        total_pairs += pairs
        total_disagree += disagree
        sizes.append(n)
        member_pos.append(group["pos"].to_numpy())
    cohort_chance = chance_disagreement(class_shares(y))
    if total_pairs == 0:
        nan = float("nan")
        return {"n_clusters": 0, "n_clusters_all_agree": 0, "n_rows_in_clusters": 0, "n_pairs_considered": 0,
                "n_disagree_pairs": 0, "cluster_sizes": {},
                "cluster_agreement_rate": nan, "cluster_disagreement_rate": nan,
                "pair_agreement_rate": nan, "pair_disagreement_rate": nan,
                "chance_cluster_agreement": nan, "chance_pair_disagreement": nan,
                "chance_disagreement_cohort": cohort_chance}
    members = np.concatenate(member_pos)
    q = class_shares(y[members])                                         # each member once
    endpoint_w = np.concatenate([np.full(len(m), len(m) - 1) for m in member_pos])
    p = class_shares(y[members], endpoint_w)                             # once per pair entered
    chance_cluster = float(np.mean([sum(v ** n for v in q.values()) for n in sizes]))
    return {
        "n_clusters": clusters_total,
        "n_clusters_all_agree": clusters_all_agree,
        "n_rows_in_clusters": int(len(members)),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(pd.Series(sizes).value_counts().items())},
        "cluster_agreement_rate": clusters_all_agree / clusters_total,
        "cluster_disagreement_rate": 1.0 - clusters_all_agree / clusters_total,
        "n_pairs_considered": total_pairs,
        "n_disagree_pairs": total_disagree,
        "pair_agreement_rate": 1.0 - total_disagree / total_pairs,
        "pair_disagreement_rate": total_disagree / total_pairs,
        "chance_cluster_agreement": chance_cluster,
        "chance_pair_disagreement": chance_disagreement(p),
        "chance_disagreement_cohort": cohort_chance,
    }


# ---------------------------------------------------------------------------
# Case C (snapshot case_c_physio_margins)
# ---------------------------------------------------------------------------
def case_c_variables(columns) -> tuple[list[str], list[str], list[str]]:
    """(numerical with a tolerance, strict-equality, unconstrained) variables of a feature set (snapshot rule)."""
    num = [c for c in columns if c in NUM_TOLERANCES]
    cat = [c for c in columns if c in CATEGORICAL_COLS]
    free = [c for c in columns if c not in NUM_TOLERANCES and c not in CATEGORICAL_COLS]
    return num, cat, free


def case_c_physio_margins(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Pairs within the snapshot per-feature tolerances (definition unchanged)."""
    y = np.asarray(y)
    n = len(X)
    num_cols, cat_cols, _ = case_c_variables(X.columns)
    num_arrays = {c: X[c].to_numpy(dtype=float) for c in num_cols}
    # strict equality: categorical as stripped strings (snapshot); a numerical variable of
    # CATEGORICAL_COLS (Âge) by exact value, equivalent to the snapshot string comparison of one float column
    cat_arrays = {c: (X[c].to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(X[c])
                      else X[c].astype(str).str.strip().to_numpy()) for c in cat_cols}

    total_pairs = 0
    total_disagree = 0
    degree = np.zeros(n, dtype=np.int64)          # pairs entered by each row (chance reference)
    for i in range(n):
        mask = np.ones(n - i - 1, dtype=bool)
        for c, arr in cat_arrays.items():
            mask &= (arr[i + 1:] == arr[i])
        for c, arr in num_arrays.items():
            mask &= (np.abs(arr[i + 1:] - arr[i]) <= NUM_TOLERANCES[c])
        if not mask.any():
            continue
        js = np.where(mask)[0] + i + 1
        total_pairs += len(js)
        total_disagree += int(np.sum(y[js] != y[i]))
        degree[i] += len(js)
        degree[js] += 1
    cohort_chance = chance_disagreement(class_shares(y))
    if total_pairs == 0:
        nan = float("nan")
        return {"n_pairs_considered": 0, "n_rows_in_pairs": 0, "n_disagree": 0, "n_agree": 0,
                "disagreement_rate": nan, "agreement_rate": nan, "chance_pair_disagreement": nan,
                "chance_disagreement_cohort": cohort_chance}
    inv = degree > 0
    return {
        "n_pairs_considered": int(total_pairs),
        "n_rows_in_pairs": int(inv.sum()),
        "disagreement_rate": total_disagree / total_pairs,
        "agreement_rate": 1.0 - total_disagree / total_pairs,
        "n_disagree": int(total_disagree),
        "n_agree": int(total_pairs - total_disagree),
        "chance_pair_disagreement": chance_disagreement(class_shares(y[inv], degree[inv])),
        "chance_disagreement_cohort": cohort_chance,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(data_csv: Path, output_dir: Path, features_key: str, targets, mode: str = "exploratory",
        config: dict | None = None) -> pd.DataFrame:
    """
    E9b into ``output_dir`` (labelled by ``mode``, ``RUN_LABELS``). ``"real"``
    and ``"dry_run"`` are passed by ``revision_runners.run_e9b`` only; a real
    run requires F12 (``features_12``, the paper's NN analysis), every target
    and the pinned cohort file (checked before any write). ``config``: the
    configuration consumed (default: config.yaml read now); its SHA-256 is
    recorded.
    """
    config = load_config(str(CONFIG_PATH)) if config is None else config
    config_used_sha256 = config_sha256(config)
    if mode not in RUN_LABELS:
        raise ValueError(f"mode must be one of {sorted(RUN_LABELS)}, got {mode!r}")
    label, experiment = RUN_LABELS[mode], RUN_EXPERIMENTS[mode]
    output_dir = Path(output_dir).resolve()
    prot = PROTECTED_DIR
    if output_dir == prot or prot in output_dir.parents or prot.name in output_dir.parts:
        raise ValueError(f"refusing to write inside the frozen artefact {prot}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"{output_dir} exists and is not an empty directory: E9b outputs are never "
                              "overwritten (use a new output directory)")
    targets = list(targets)
    frozen = features_key == FROZEN_FEATURES_KEY and sorted(targets) == sorted(TARGETS)
    data_raw, data_sha256 = read_consumed(Path(data_csv))       # one read: hashed and parsed bytes
    if mode == "real":
        import data_groups
        if not frozen:
            raise ValueError("a real E9b uses F12 (features_12, the paper's NN analysis) and every target")
        if data_sha256 != data_groups.PINNED_SHA256["canonical_csv"]:
            raise ValueError(f"{data_csv}: not the pinned ASHRAE-2022 cohort file")
        if list(config["data"].get(FROZEN_FEATURES_KEY) or []) != list(FROZEN_F12):
            raise ValueError("a real E9b uses F12 (the paper's NN analysis); the configuration "
                             f"consumed has features_12 = {config['data'].get(FROZEN_FEATURES_KEY)}")
    numerical, categorical = feature_set(features_key, config)
    df, coverage = load_cohort(Path(data_csv), numerical, categorical, targets, raw=data_raw)
    variables = numerical + categorical
    X = df[variables]
    X_std = encode_features(X)
    num_c, cat_c, free_c = case_c_variables(variables)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_coverage_log(coverage, output_dir / f"{label}_category_coverage.json")
    # the strings below are written into the E9b detail files
    common = {
        "analysis": "NN analysis as in the paper (primary 1-NN, Case B, Case C), label disagreement, "
                    "descriptive (restored from the pre-revision code, E9b)",
        "source": f"{SOURCE} (_encode_features, primary_disagreement, case_b_identical_groups, "
                  "case_c_physio_margins)",
        "primary_definition": "each cohort row against the other cohort rows: label of its nearest neighbour "
                              "(k = 1, Euclidean distance, categorical variables integer-encoded, every column "
                              "standardised); the query is excluded by row position",
        "run_mode": mode, "experiment": experiment, "population": POPULATION,
        "features_key": features_key, "frozen_definition": bool(frozen),
        "config_used_sha256": config_used_sha256, "variables": variables,
        "case_b_definition": f"groups of rows with identical feature vectors (numerical rounded to "
                             f"{ROUND_DECIMALS} decimals, categorical exact), groups of size >= 2",
        "case_c_tolerances": {c: NUM_TOLERANCES[c] for c in num_c},
        "case_c_strict_equality": cat_c,
        "case_c_unconstrained_variables": free_c,
        "chance_definitions": {
            "primary_chance_pair_disagreement": "1 - sum_c p_c^2, p_c = class shares of the query votes "
                                                "(every cohort vote)",
            "chance_pair_disagreement": "1 - sum_c p_c^2, p_c = class shares of the votes entering the pairs "
                                        "(each vote once per pair)",
            "chance_cluster_agreement": "mean over groups of sum_c q_c^n_g, q_c = class shares of the group members",
            "chance_disagreement_cohort": "1 - sum_c p_c^2 over every cohort row"},
        "data_csv": str(data_csv), "data_csv_sha256": data_sha256, "n_cohort_rows": int(len(df)),
    }
    rows = []
    for target in targets:
        y = df[target].to_numpy()
        a = primary_disagreement(X_std, y)
        b = case_b_identical_groups(X, y)
        c = case_c_physio_margins(X, y)
        detail = {**common, "target": target, "primary": a, "case_b_identical_groups": b,
                  "case_c_physio_margins": c}
        (output_dir / f"{label}_detail_{target}.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False))
        rows.append({
            "run_mode": mode, "experiment": experiment, "population": POPULATION, "target": target,
            "features_key": features_key, "frozen_definition": bool(frozen), "n_features": len(variables),
            "n": int(len(df)),
            "primary_disagreement": a["disagreement_rate"], "primary_agreement": a["agreement_rate"],
            "primary_chance_disagreement": a["chance_pair_disagreement"],
            "primary_n_zero_distance": a["n_zero_distance_neighbour"],
            "primary_n_self_in_second_column": a["n_self_in_second_column"],
            "case_b_clusters": b["n_clusters"], "case_b_rows": b["n_rows_in_clusters"],
            "case_b_pairs": b["n_pairs_considered"],
            "case_b_cluster_agreement": b["cluster_agreement_rate"],
            "case_b_chance_cluster_agreement": b["chance_cluster_agreement"],
            "case_b_pair_agreement": b["pair_agreement_rate"],
            "case_b_pair_disagreement": b["pair_disagreement_rate"],
            "case_b_chance_pair_disagreement": b["chance_pair_disagreement"],
            "case_c_pairs": c["n_pairs_considered"], "case_c_rows": c["n_rows_in_pairs"],
            "case_c_agreement": c["agreement_rate"], "case_c_disagreement": c["disagreement_rate"],
            "case_c_chance_pair_disagreement": c["chance_pair_disagreement"],
            "chance_disagreement_cohort": b["chance_disagreement_cohort"],
        })
        print(f"  {target:20s} primary 1-NN disagreement {a['disagreement_rate']:.3f} "
              f"(chance {a['chance_pair_disagreement']:.3f}); Case B {b['n_clusters']} groups, "
              f"agreement {b['cluster_agreement_rate']:.3f} "
              f"(chance {b['chance_cluster_agreement']:.3f}); Case C {c['n_pairs_considered']} pairs, "
              f"disagreement {c['disagreement_rate']:.3f} (chance {c['chance_pair_disagreement']:.3f})")
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"{label}_summary.csv", index=False)
    summary.attrs["config_used_sha256"] = config_used_sha256
    summary.attrs["variables"] = variables
    summary.attrs["data_csv_sha256"] = data_sha256
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="NN analysis as in the paper: primary 1-NN, Case B, Case C, "
                                            "descriptive (E9b; restored from the snapshot).")
    p.add_argument("--features_key", default=FROZEN_FEATURES_KEY,
                   help="config.yaml feature set: 'features_12' (F12, the paper's NN analysis, E9b) or another "
                        "key (exploratory)")
    p.add_argument("--data_csv", default=str(DEFAULT_DATA_CSV))
    p.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--exploratory", action="store_true",
                   help="Required: a standalone run is exploratory only (files and rows labelled "
                        "E9b_nn_cases_exploratory, no manifest); the real E9b runs through `run_revision.py e9b`.")
    args = p.parse_args(argv)
    if not args.exploratory:
        p.error("a real E9b runs only through `python3 run_revision.py e9b --run_dir RUN` (pinned cohort, run "
                "manifest); standalone nn_cases.py accepts --exploratory only")
    run(Path(args.data_csv), Path(args.output_dir), args.features_key, args.targets, mode="exploratory")


if __name__ == "__main__":
    main()
