"""Tests of analysis/paper_figures on synthetic inputs (no real run output is read).

* exact helpers: chance reference (closed form = brute force over all vote pairs),
  weighted-F1 from a confusion matrix (= sklearn), seed means;
* wording guard: every banned word is caught in text, in figure labels and by the
  output writer;
* UBJSON reader of the XGBoost boosters (hand-encoded bytes; equality with
  libxgboost on a model it can fit with the GPU hidden);
* input verification: a byte changed after the manifest was written is refused,
  a non-empty --out is refused, a pending experiment is refused unless allowed;
* end to end: a synthetic run directory with the layout of rerun_2026-10-XX_v2,
  including the E10 (hybrids) and E7b (SET/PET/ePMV) outputs, goes through every item.

Run from the repository root (GPU hidden):
    .venv-revision/bin/python -m pytest tests/test_paper_figures.py -q
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from analysis.paper_figures import cli, metrics, wording, xgb_ubj  # noqa: E402
from analysis.paper_figures.inputs import (InputHashError, OutputDir, PendingExperimentError,  # noqa: E402
                                           RunInputs)
from analysis.paper_figures.items import ITEMS  # noqa: E402

TARGETS = metrics.TARGETS
ONT = metrics.ONTOLOGY
E1_MODELS = ["RF", "XGB", "SVM", "ANN", "FTT"]
NUM = ["Tair", "Tout", "RH", "vel", "clo", "Met", "Âge", "Trm_ema_28", "RHout_ema_28", "precip_ema_28",
       "sunshine_ema_h_28", "wind_ema_28"]
CAT = ["Sexe", "Season", "Climate", "Building_type", "cooling type"]


# ------------------------------------------------------------------------------------------------
# Exact helpers
# ------------------------------------------------------------------------------------------------
def test_chance_disagreement_closed_form_equals_all_pairs():
    rng = np.random.default_rng(1)
    classes = ONT["thermal_sensation"]
    yq = rng.choice(classes, size=137, p=[.05, .1, .2, .3, .2, .1, .05])
    yc = rng.choice(classes, size=211, p=[.1, .1, .1, .4, .1, .1, .1])
    p_q, p_c = metrics.class_distribution(yq, classes), metrics.class_distribution(yc, classes)
    brute = np.mean(yq[:, None] != yc[None, :])
    assert metrics.chance_disagreement(p_q, p_c) == pytest.approx(brute, abs=1e-12)
    same = np.mean(yq[:, None] != yq[None, :])                       # pairs drawn with replacement
    assert metrics.chance_disagreement(p_q) == pytest.approx(same, abs=1e-12)
    assert metrics.chance_disagreement(p_q) == pytest.approx(1 - np.sum(p_q ** 2), abs=1e-15)


def test_chance_disagreement_known_values_and_guards():
    assert metrics.chance_disagreement([1 / 3] * 3) == pytest.approx(2 / 3)
    assert metrics.chance_disagreement([1.0, 0.0, 0.0]) == 0.0
    assert metrics.chance_disagreement([1, 0, 0], [0, 1, 0]) == 1.0
    with pytest.raises(ValueError):
        metrics.chance_disagreement([0.5, 0.6])
    with pytest.raises(ValueError):
        metrics.chance_disagreement([0.5, 0.5], [1 / 3] * 3)
    with pytest.raises(ValueError):
        metrics.class_distribution([0, 1, 5], (0, 1))


def test_weighted_f1_from_cm_equals_sklearn():
    from sklearn.metrics import confusion_matrix, f1_score
    rng = np.random.default_rng(2)
    labels = list(ONT["thermal_sensation"])
    for _ in range(20):
        y = rng.choice(labels[1:5], size=300)                         # absent true classes included
        p = np.where(rng.random(300) < .5, y, rng.choice(labels, size=300))
        cm = confusion_matrix(y, p, labels=labels)
        ref = f1_score(y, p, labels=labels, average="weighted", zero_division=0)
        assert metrics.weighted_f1_from_cm(cm.tolist()) == pytest.approx(ref, abs=1e-12)
        assert metrics.weighted_f1_from_cm(json.dumps(cm.tolist())) == pytest.approx(ref, abs=1e-12)
        assert metrics.accuracy_from_cm(cm) == pytest.approx(np.mean(y == p))


def test_seed_means():
    rows = pd.DataFrame({"t": ["a"] * 3 + ["b"], "m": ["x"] * 4, "seed": [42, 43, 44, 42], "acc": [.1, .2, .6, .5]})
    out = metrics.seed_means(rows, ["t", "m"], ["acc"]).set_index("t")
    assert out.loc["a", "acc"] == pytest.approx(0.3)
    assert (out.loc["a", "acc_min"], out.loc["a", "acc_max"]) == (.1, .6)
    assert out.loc["b", "seeds"] == [42] and out.loc["b", "n_seeds"] == 1


# ------------------------------------------------------------------------------------------------
# Wording guard
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["NN agreement ceiling", "an Upper Bound on accuracy", "this demonstrates",
                                  "Demonstrated", "informational rather than algorithmic", "Empirical models",
                                  "tsv7_cerema.png", "Mathilde", "upper-bound"])
def test_banned_wording_detected(text):
    assert wording.banned_in(text)
    with pytest.raises(wording.BannedWordingError):
        wording.check_text(text, "test")


def test_allowed_wording_passes():
    for text in ("1-NN vote agreement (1 − D$_1$)", "Chance agreement (two random votes)",
                 "Heat-balance indices", "Moujalled", "Hostein", "Test accuracy", "bounded"):
        wording.check_text(text, "test")


def test_banned_word_in_figure_labels_refused(tmp_path):
    import matplotlib.pyplot as plt
    out = OutputDir(tmp_path / "o")
    for setter in (lambda ax: ax.set_title("Performance ceiling"),
                   lambda ax: ax.set_xticks([0], ["empirical"]),
                   lambda ax: ax.plot([0], label="upper bound") and ax.legend()):
        fig, ax = plt.subplots()
        setter(ax)
        with pytest.raises(wording.BannedWordingError):
            out.savefig(fig, "f.png")
        plt.close(fig)
    with pytest.raises(wording.BannedWordingError):
        out.write_text("t.tex", "\\caption{NN ceiling}")
    with pytest.raises(wording.BannedWordingError):
        out.savefig(plt.figure(), "performance_ceiling.png")
    assert not any((tmp_path / "o").iterdir())


# ------------------------------------------------------------------------------------------------
# UBJSON / XGBoost importances
# ------------------------------------------------------------------------------------------------
def _ubj_key(s):
    """A UBJSON object key / string payload: int8 length then UTF-8 bytes."""
    b = s.encode()
    return b"i" + struct.pack(">b", len(b)) + b


def _ubj_typed(arr, code, fmt):
    return b"[$" + code + b"#i" + struct.pack(">b", len(arr)) + b"".join(struct.pack(fmt, v) for v in arr)


def _ubj_tree(split, left, loss, hess):
    return (b"{" + _ubj_key("split_indices") + _ubj_typed(split, b"l", ">i") + _ubj_key("left_children")
            + _ubj_typed(left, b"l", ">i") + _ubj_key("loss_changes") + _ubj_typed(loss, b"d", ">f")
            + _ubj_key("sum_hessian") + _ubj_typed(hess, b"d", ">f") + b"}")


def test_ubj_decoder_and_feature_score_hand_encoded():
    t1 = _ubj_tree([1, 0, 2], [1, -1, -1], [4.0, 0.0, 0.0], [10., 4., 6.])
    t2 = _ubj_tree([2, 0, 1, 0, 0], [1, 3, -1, -1, -1], [2.0, 6.0, 0.0, 0.0, 0.0], [8., 5., 3., 2., 3.])
    learner = (b"{" + _ubj_key("feature_names") + b"[]" + _ubj_key("learner_model_param") + b"{"
               + _ubj_key("num_feature") + b"S" + _ubj_key("3") + b"}"
               + _ubj_key("gradient_booster") + b"{" + _ubj_key("name") + b"S" + _ubj_key("gbtree")
               + _ubj_key("model") + b"{" + _ubj_key("trees") + b"[" + t1 + t2 + b"]" + b"}" + b"}" + b"}")
    doc = b"{" + _ubj_key("Config") + b"{}" + _ubj_key("Model") + b"{" + _ubj_key("learner") + learner + b"}}"
    decoded = xgb_ubj.decode_ubjson(doc)
    assert decoded["Model"]["learner"]["learner_model_param"]["num_feature"] == "3"
    # split nodes: tree 1 node 0 (f1, gain 4); tree 2 node 0 (f2, gain 2), node 1 (f0, gain 6)
    assert xgb_ubj.feature_score(decoded, "weight") == {"f0": 1.0, "f1": 1.0, "f2": 1.0}
    assert xgb_ubj.feature_score(decoded, "gain") == {"f0": 6.0, "f1": 4.0, "f2": 2.0}
    assert xgb_ubj.feature_score(decoded, "cover") == {"f0": 5.0, "f1": 10.0, "f2": 8.0}
    b = xgb_ubj.UBJBooster(doc)
    assert b.feature_names is None and b.num_features() == 3
    with pytest.raises(xgb_ubj.UBJDecodeError):
        xgb_ubj.decode_ubjson(doc + b"Z")


def test_ubj_booster_equals_libxgboost():
    import pickle
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = 2 * X[:, 0] + X[:, 3] + rng.normal(size=300) * .3
    try:      # single-output objective: fits with the GPU hidden (multi-class raises cudaErrorNoDevice)
        m = xgb.XGBRegressor(n_estimators=15, max_depth=4, device="cpu", tree_method="hist").fit(X, y)
    except xgb.core.XGBoostError as exc:
        pytest.skip(f"xgboost cannot fit here: {str(exc)[:80]}")
    ref = {t: m.get_booster().get_score(importance_type=t) for t in ("weight", "gain", "total_gain", "cover")}
    ref_fi = m.feature_importances_
    with xgb_ubj.booster_without_runtime():
        m2 = pickle.loads(pickle.dumps(m))
    assert isinstance(m2.get_booster(), xgb_ubj.UBJBooster)
    assert xgb.core.Booster.__setstate__ is not xgb_ubj._setstate_without_runtime      # restored
    for t, r in ref.items():
        got = m2.get_booster().get_score(importance_type=t)
        assert set(got) == set(r)
        for k in r:
            assert got[k] == pytest.approx(r[k], rel=1e-5)
    np.testing.assert_allclose(m2.feature_importances_, ref_fi, atol=1e-6)


# ------------------------------------------------------------------------------------------------
# Synthetic run directory
# ------------------------------------------------------------------------------------------------
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cm(y, p, labels):
    m = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {c: i for i, c in enumerate(labels)}
    for a, b in zip(y, p):
        m[idx[a], idx[b]] += 1
    return m


def _metric(y, p, labels):
    from sklearn.metrics import cohen_kappa_score, f1_score
    m = _cm(y, p, labels)
    present = sorted(set(y) | set(p))
    return {"n": len(y), "n_correct": int(np.trace(m)), "accuracy": float(np.mean(y == p)),
            "macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(y, p, labels=labels, weights="quadratic")),
            "confusion_matrix": json.dumps(m.tolist()),
            "support": json.dumps({str(c): int((y == c).sum()) for c in labels})}


def _noisy(rng, y, labels, acc):
    return np.where(rng.random(len(y)) < acc, y, rng.choice(labels, size=len(y)))


def _cohort(rng, n):
    df = pd.DataFrame({c: rng.normal(20, 5, n) for c in NUM})
    for c, vals in zip(CAT, (["Male", "Female"], ["Summer", "Winter"], ["Cfa", "Cfb"], ["Office", "Multifamily housing"],
                             ["Air Conditioned", "Naturally Ventilated"])):
        df[c] = rng.choice(vals, n)
    df["thermal_sensation"] = rng.choice(ONT["thermal_sensation"], n, p=[.04, .07, .16, .42, .2, .08, .03])
    df["TSV_3p"] = np.sign(df["thermal_sensation"]).astype(int)
    df["thermal_preference"] = rng.choice((-1, 0, 1), n, p=[.33, .5, .17])
    return df


def _fit_pipelines(rng, X, y, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import xgboost as xgb

    def pp():
        return ColumnTransformer([("num", StandardScaler(), NUM), ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT)])
    rf = Pipeline([("pp", pp()), ("clf", RandomForestClassifier(n_estimators=5, random_state=seed))]).fit(X, y)
    # regression head: XGBoost cannot fit a multi-class model with the GPU hidden; the importance path is the same
    xg = Pipeline([("pp", pp()), ("clf", xgb.XGBRegressor(n_estimators=5, max_depth=3, device="cpu",
                                                          tree_method="hist", random_state=seed))]).fit(X, y)
    return rf, xg


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """A synthetic run directory with the layout of rerun_2026-10-XX_v2 (predictions files, manifests, run
    index) and a checkout holding its inputs; the pinned cohort hash points at the synthetic cohort meanwhile."""
    import data_groups
    pinned = data_groups.PINNED_SHA256["canonical_csv"]
    try:
        yield _synthetic_run(tmp_path_factory, data_groups)
    finally:
        data_groups.PINNED_SHA256["canonical_csv"] = pinned


def _synthetic_run(tmp_path_factory, data_groups):
    import joblib
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("synthetic")
    data_root = root / "checkout"
    run = root / "rerun_synthetic"
    (data_root / "Data").mkdir(parents=True)
    (data_root / "splits" / "ASHRAE_2022").mkdir(parents=True)
    cwd = "/somewhere/run-checkout"
    n = 600
    cohort = _cohort(rng, n)
    # E9b: exact feature twins (Case B) and near twins (Case C) with their own votes
    cohort.loc[300:339, NUM + CAT] = cohort.loc[0:39, NUM + CAT].to_numpy()
    cohort.loc[320:339, "Tair"] += 0.5
    cohort.to_csv(data_root / "Data" / "ASHRAE_2022_Clean_api.csv", index=False)
    french = {"Moujalled": _cohort(rng, 61), "Hostein": _cohort(rng, 43)}
    for name, df in french.items():
        df.to_csv(data_root / "Data" / f"{name}_api.csv", index=False)
    perm = rng.permutation(n)
    te, tr = np.sort(perm[:120]), np.sort(perm[120:])
    split_paths = {}
    for t in TARGETS:
        p = data_root / "splits" / "ASHRAE_2022" / f"{t}_split_indices.json"
        p.write_text(json.dumps({"target": t, "train_idx": tr.tolist(), "test_idx": te.tolist()}))
        split_paths[t] = p
    cohort_inputs = {"cohort_csv": "Data/ASHRAE_2022_Clean_api.csv",
                     **{f"split:{t}": f"splits/ASHRAE_2022/{t}_split_indices.json" for t in TARGETS}}
    cohort_sha = {"cohort_csv": _sha(data_root / "Data/ASHRAE_2022_Clean_api.csv"),
                  **{f"split:{t}": _sha(split_paths[t]) for t in TARGETS}}
    data_groups.PINNED_SHA256["canonical_csv"] = cohort_sha["cohort_csv"]

    exps = {}
    bld = rng.integers(0, 30, n)                                     # building of each vote (bootstrap clusters)
    pub = np.array([f"Publication {i}" for i in rng.integers(0, 4, n)])

    def write_manifest(exp, dirname, outputs, inputs=None, inputs_sha=None, **extra):
        d = run / dirname
        d.mkdir(parents=True, exist_ok=True)
        man = {"format": "revision-2026-10/run-manifest-v1", "experiment": exp, "dry_run": False, "cwd": cwd,
               "inputs": inputs or {}, "inputs_sha256": inputs_sha or {},
               "outputs_sha256": {rel: _sha(d / rel) for rel in outputs}, **extra}
        (d / "run_manifest.json").write_text(json.dumps(man))
        exps[exp] = {"manifest": f"{dirname}/run_manifest.json", "manifest_sha256": _sha(d / "run_manifest.json"),
                     "status": "complete"}

    def write_predictions(dirname, name, outputs, **cols):
        path = run / dirname / "predictions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        outputs.append(f"predictions/{name}")
        return path

    def ashrae_cols(exp, t, model, seed, split_id, feature_set, rows, y_pred):
        rows = np.asarray(rows)
        return dict(experiment=exp, cohort="ASHRAE_2022", target=t, feature_set=feature_set, model=model, seed=seed,
                    split_id=split_id, row_idx=rows, publication=pub[rows], building_id=bld[rows],
                    y_true=cohort[t].to_numpy()[rows], y_pred=y_pred)

    # ---- E1 (predictions files + model files) ----
    e1_outputs, e1_preds = [], {}
    seeds_e1 = {m: ([42] if m == "SVM" else [42, 43, 44]) for m in E1_MODELS}
    for t in TARGETS:
        y = cohort[t].to_numpy()[te]
        for m in E1_MODELS:
            for s in seeds_e1[m]:
                p = _noisy(rng, y, ONT[t], .5 + .05 * E1_MODELS.index(m) / 5)
                e1_preds[(t, m, s)] = p
                write_predictions("E1_phase2", f"E1__ASHRAE_2022__{t}__{m}__seed{s}__archived_80_20.csv.gz", e1_outputs,
                                  **ashrae_cols("E1", t, m, s, "archived_80_20", "F17", te, p))
        X = cohort.iloc[tr][NUM + CAT]
        for s in (42,):
            rf, xg = _fit_pipelines(rng, X, cohort[t].to_numpy()[tr], s)
            for fam, pipe in (("RF", rf), ("XGB", xg)):
                rel = f"{t}/{fam}/seed_{s}/{t}__{fam}__seed{s}.joblib"
                (run / "E1_phase2" / rel).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipe, run / "E1_phase2" / rel)
                e1_outputs.append(rel)
    write_manifest("E1", "E1_phase2", e1_outputs, cohort_inputs, cohort_sha, seeds={"models": seeds_e1},
                   ml_inventory=sorted(f"{m}:{x}" for m, ss in seeds_e1.items() for x in ss))

    # ---- E4 (F12 predictions; the F17 side is the E1 fit) ----
    e4_outputs, pairs = [], []
    for t in TARGETS:
        y = cohort[t].to_numpy()[te]
        for fam in ("RF", "XGB"):
            for s in (42, 43, 44):
                name = f"E4__ASHRAE_2022__{t}__{fam}__seed{s}__archived_80_20.csv.gz"
                path = write_predictions("E4_ema", name, e4_outputs,
                                         **ashrae_cols("E4", t, fam, s, "archived_80_20", "F12", te,
                                                       _noisy(rng, y, ONT[t], .45)))
                e1_name = f"E1__ASHRAE_2022__{t}__{fam}__seed{s}__archived_80_20.csv.gz"
                pairs.append({"target": t, "model": fam, "seed": s, "base_feature_set": "F12",
                              "extended_feature_set": "F17", "base_file": f"../run/E4_ema/predictions/{name}",
                              "extended_file": f"../run/E1_phase2/predictions/{e1_name}", "base_sha256": _sha(path),
                              "extended_sha256": _sha(run / "E1_phase2" / "predictions" / e1_name)})
    pd.DataFrame(pairs).to_csv(run / "E4_ema" / "pairs.csv", index=False)
    write_manifest("E4", "E4_ema", e4_outputs + ["pairs.csv"])

    # ---- E2 (leave-one-publication-out predictions, folds, run metrics order) ----
    pubs = sorted(set(pub))
    e2_outputs, folds, e2_metrics = [], [], []
    for k, p_ in enumerate(pubs):
        rows_k = np.flatnonzero(pub == p_)
        folds.append({"fold": k, "split_id": f"lopo_{k:02d}", "publication": p_, "n_test": int(len(rows_k)),
                      "n_train": int(n - len(rows_k))})
    for t in TARGETS:
        for fam in ("RF", "XGB"):
            for k, p_ in enumerate(pubs):
                rows_k = np.flatnonzero(pub == p_)
                y = cohort[t].to_numpy()[rows_k]
                write_predictions("E2_lopo", f"E2__ASHRAE_2022__{t}__{fam}__seed42__lopo_{k:02d}.csv.gz", e2_outputs,
                                  **ashrae_cols("E2", t, fam, 42, f"lopo_{k:02d}", "F17", rows_k,
                                                _noisy(rng, y, ONT[t], .4)))
                e2_metrics.append({"target": t, "model": fam, "split_id": f"lopo_{k:02d}"})
    (run / "E2_lopo" / "lopo_folds.json").write_text(json.dumps({"publications": pubs, "folds": folds}))
    pd.DataFrame(e2_metrics).to_csv(run / "E2_lopo" / "metrics.csv", index=False)
    write_manifest("E2", "E2_lopo", e2_outputs + ["lopo_folds.json", "metrics.csv"])

    # ---- E10 (layout of revision_runners.run_e10: metrics.csv in the E1 run-metrics
    #      format + one predictions file per row under predictions/) ----
    (run / "E10_hybrids" / "predictions").mkdir(parents=True)
    rows, e10_outputs = [], ["metrics.csv"]
    for t in TARGETS:
        y = cohort[t].to_numpy()[te]
        for m in ("FTT_RF", "FTT_XGB"):
            for s in (42, 43, 44):
                pred = _noisy(rng, y, ONT[t], .45)
                name = f"E10__ASHRAE_2022__{t}__{m}__seed{s}__archived_80_20.csv.gz"
                pd.DataFrame({"experiment": "E10", "cohort": "ASHRAE_2022", "target": t, "feature_set": "FTTemb_F17",
                              "model": m, "seed": s, "split_id": "archived_80_20", "row_idx": te,
                              "building_id": bld[te], "y_true": y, "y_pred": pred}).to_csv(run / "E10_hybrids" / "predictions" / name, index=False,
                                                      compression={"method": "gzip", "mtime": 0})
                e10_outputs.append(f"predictions/{name}")
                rows.append({"experiment": "E10", "target": t, "feature_set": "FTTemb_F17", "split_id": "archived_80_20",
                             "model": m, "seed": s, **_metric(y, pred, list(ONT[t])), "predictions": name})
    pd.DataFrame(rows).to_csv(run / "E10_hybrids/metrics.csv", index=False)
    write_manifest("E10", "E10_hybrids", e10_outputs)

    # ---- E7 (predictions on the PMV validity mask) and E7b (expected layout: E7b_indices/metrics.csv) ----
    def index_metrics(methods, targets=TARGETS):
        out = []
        for t in targets:
            mask = rng.random(len(te)) < .8
            y = cohort[t].to_numpy()[te][mask]
            for meth in methods:
                out.append({"target": t, "pmv_variant": "primary", "method": meth, "subset": "on_mask",
                            "coverage": float(mask.mean()), "status": "ok",
                            **_metric(y, _noisy(rng, y, ONT[t], .3), list(ONT[t]))})
                out.append({"target": t, "pmv_variant": "dynamic", "method": meth, "subset": "on_mask",
                            "coverage": .5, "status": "ok", **_metric(y, y, list(ONT[t]))})
        return pd.DataFrame(out)
    e7_outputs = ["apmv_calibration.json"]
    (run / "E7_pmv").mkdir()
    (run / "E7_pmv" / "apmv_calibration.json").write_text(json.dumps({t: {"status": "ok"} for t in TARGETS}))
    for t in TARGETS:
        mask = rng.random(len(te)) < .8
        y = cohort[t].to_numpy()[te]
        for meth in ("PMV", "aPMV"):
            yp = _noisy(rng, y, ONT[t], .3).astype(float)
            yp[~mask] = np.nan
            cols = ashrae_cols("E7", t, meth, np.nan, "archived_80_20", "pmv_inputs", te, yp)
            write_predictions("E7_pmv", f"E7__ASHRAE_2022__{t}__{meth}.csv.gz", e7_outputs, **cols, pmv_valid=mask)
    write_manifest("E7", "E7_pmv", e7_outputs)
    (run / "E7b_indices").mkdir()
    # E7b layout of indicateurs_classiques.run_e7b: no pmv_variant column; each ML comparison row names its
    # index in a column "index" (on_mask / outside_mask), empty on the index rows
    e7b = index_metrics(["ePMV", "PTS_SET", "PTS_PET"], ("thermal_sensation", "TSV_3p"))   # no TPV
    e7b = e7b[e7b["pmv_variant"] == "primary"].drop(columns=["pmv_variant"])
    # e unbounded, flagged outside [0.5, 1.0] in its own column (TSV-7 flagged, TSV-3 not)
    is_epmv = e7b["method"] == "ePMV"
    e7b.loc[is_epmv, "epmv_e"] = e7b.loc[is_epmv, "target"].map({"thermal_sensation": 1.18, "TSV_3p": 0.74})
    e7b.loc[is_epmv, "epmv_e_flag_outside_range"] = e7b.loc[is_epmv, "target"].map(
        {"thermal_sensation": True, "TSV_3p": False})
    ml = []
    for t in ("thermal_sensation", "TSV_3p"):
        y = cohort[t].to_numpy()[te]
        for meth in ("ePMV", "PTS_SET", "PTS_PET"):
            for subset in ("on_mask", "outside_mask"):
                ml.append({"target": t, "method": "ML:model=RF/seed=42", "index": meth, "subset": subset,
                           "coverage": .8, "status": "ok", "model": "RF", "seed": 42,
                           **_metric(y, _noisy(rng, y, ONT[t], .5), list(ONT[t]))})
    pd.concat([e7b, pd.DataFrame(ml)], ignore_index=True).to_csv(run / "E7b_indices/metrics.csv", index=False)
    write_manifest("E7b", "E7b_indices", ["metrics.csv"])

    # ---- E6 (Phase 3 split files as inputs, predictions per model and split, full-cohort zero-shot) ----
    (data_root / "splits" / "phase3").mkdir(parents=True)
    e6_outputs = []
    fr_inputs = {c: f"{cwd}/Data/{c}_api.csv" for c in french}
    fr_sha = {c: _sha(data_root / "Data" / f"{c}_api.csv") for c in french}
    for c, df in french.items():
        for t in TARGETS:
            y = df[t].to_numpy()
            for m in ("RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot"):
                write_predictions("E6_phase3", f"E6__{c}__{t}__{m}__seed42__full_cohort.csv.gz", e6_outputs,
                                  experiment="E6", cohort=c, target=t, feature_set="F17", model=m, seed=42,
                                  split_id="full_cohort", row_idx=np.arange(len(y)), y_true=y,
                                  y_pred=_noisy(rng, y, ONT[t], .4))
            for regime in ("20-80", "80-20"):
                splits = []
                for r in range(5):
                    perm_c = rng.permutation(len(y))
                    cut = int(len(y) * (.2 if regime == "20-80" else .8))
                    splits.append({"split_id": f"{regime}_r{r:02d}", "dev_idx": np.sort(perm_c[:cut]).tolist(),
                                   "test_idx": np.sort(perm_c[cut:]).tolist()})
                name = f"{c}_{t}_{regime}.json"
                (data_root / "splits" / "phase3" / name).write_text(json.dumps(
                    {"cohort": c, "target": t, "regime": regime, "grouping": "date", "splits": splits}))
                fr_inputs[f"phase3_split:{name}"] = f"splits/phase3/{name}"
                fr_sha[f"phase3_split:{name}"] = _sha(data_root / "splits" / "phase3" / name)
                for sp in splits:
                    rows = np.asarray(sp["test_idx"])
                    for m in ("RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot", "RF_local", "XGB_local",
                              "FTT_head", "FTT_full"):
                        write_predictions("E6_phase3", f"E6__{c}__{t}__{m}__seed42__{sp['split_id']}.csv.gz",
                                          e6_outputs, experiment="E6", cohort=c, target=t, feature_set="F17", model=m,
                                          seed=42, split_id=sp["split_id"], row_idx=rows, y_true=y[rows],
                                          y_pred=_noisy(rng, y[rows], ONT[t], .5))
    write_manifest("E6", "E6_phase3", e6_outputs, fr_inputs, fr_sha)

    # ---- E9b (run summary + detail files written by analysis/nn_cases.run on the synthetic cohort) ----
    sys.path.insert(0, str(REPO_DIR / "analysis"))
    import nn_cases
    e9b = nn_cases.run(data_root / "Data/ASHRAE_2022_Clean_api.csv", run / "E9b_nn_cases", "features_12",
                       list(TARGETS), mode="real")
    write_manifest("E9b", "E9b_nn_cases", sorted(p.name for p in (run / "E9b_nn_cases").iterdir()),
                   {"cohort_csv": cohort_inputs["cohort_csv"]}, {"cohort_csv": cohort_sha["cohort_csv"]})

    # ---- Phase 1 grid (outside the run directory, pinned) ----
    p1 = data_root / "results" / "phase1"
    p1.mkdir(parents=True)
    from analysis.paper_figures.tables import PHASE1_HAGHIRAD_CONFIGS, PHASE1_HIGHCAP_CONFIGS
    grid = [{"config": cf, "target": t, "ros": ros, "encoding": enc, "dataset": ds,
             **{m: float(rng.uniform(.3, .7)) for m in ("accuracy", "macro_f1", "weighted_f1", "qwk")}}
            for cf in PHASE1_HAGHIRAD_CONFIGS + PHASE1_HIGHCAP_CONFIGS for t in TARGETS for ros in (False, True)
            for enc in ("onehot", "label") for ds in ("ASHRAE_2018", "ASHRAE_2022")]
    pd.DataFrame(grid).to_csv(p1 / "experiment_grid_full.csv", index=False)
    g_sha = _sha(p1 / "experiment_grid_full.csv")

    index = {"format": "revision-2026-10/run-index-v1", "experiments": exps,
             "phase1": {"path": "results/phase1", "experiment_grid_full_sha256": g_sha,
                        "experiment_grid_full_sha256_pinned": g_sha}}
    (run / "manifest.json").write_text(json.dumps(index))
    return {"run": run, "data_root": data_root, "cohort": cohort, "te": te, "tr": tr, "french": french,
            "root": root, "e9b": e9b, "e1_preds": e1_preds, "pub": pub, "bld": bld, "cohort_sha": cohort_sha}


@contextmanager
def _patched_output(sr, exp, dirname, rel, new_bytes):
    """Replace one run output, re-hash it in its run manifest and the run index (a consistent but different
    run); everything is restored afterwards."""
    run = sr["run"]
    files = [run / dirname / rel, run / dirname / "run_manifest.json", run / "manifest.json"]
    saved = {f: f.read_bytes() for f in files}
    try:
        files[0].write_bytes(new_bytes)
        man = json.loads(saved[files[1]])
        man["outputs_sha256"][rel] = _sha(files[0])
        files[1].write_text(json.dumps(man))
        index = json.loads(saved[files[2]])
        index["experiments"][exp]["manifest_sha256"] = _sha(files[1])
        files[2].write_text(json.dumps(index))
        yield
    finally:
        for f, b in saved.items():
            f.write_bytes(b)


@contextmanager
def _patched_manifest(sr, exp, dirname, new_bytes):
    """Replace one run manifest and re-hash it in the run index; restored afterwards."""
    run = sr["run"]
    files = [run / dirname / "run_manifest.json", run / "manifest.json"]
    saved = {f: f.read_bytes() for f in files}
    try:
        files[0].write_bytes(new_bytes)
        index = json.loads(saved[files[1]])
        index["experiments"][exp]["manifest_sha256"] = _sha(files[0])
        files[1].write_text(json.dumps(index))
        yield
    finally:
        for f, b in saved.items():
            f.write_bytes(b)


def _run_cli(sr, out, *extra):
    argv = ["--run_dir", str(sr["run"]), "--out", str(out), "--data_root", str(sr["data_root"]),
            "--fi_seeds", "42", *extra]
    assert cli.main(argv) == 0
    return out


@pytest.fixture(scope="module")
def all_outputs(synthetic_run, tmp_path_factory):
    return _run_cli(synthetic_run, tmp_path_factory.mktemp("figs") / "out")


def test_every_item_builds_and_manifest_lists_outputs(all_outputs):
    man = json.loads((all_outputs / "paper_figures_manifest.json").read_text())
    assert man["items"] == [n for n, i in ITEMS.items() if i.default] and "nn_e9" not in man["items"]
    assert man["pending_experiments_left_out"] == [] and man["mode"] == "final"
    written = {o["path"] for o in man["outputs"]}
    for p in all_outputs.rglob("*"):
        if p.is_file() and p.name != "paper_figures_manifest.json":
            assert str(p.relative_to(all_outputs)) in written
            assert hashlib.sha256(p.read_bytes()).hexdigest() == next(
                o["sha256"] for o in man["outputs"] if o["path"] == str(p.relative_to(all_outputs)))
    roles = {i["role"] for i in man["inputs"]}
    assert {"E10:metrics.csv", "E7b:metrics.csv",
            "E1:predictions/E1__ASHRAE_2022__thermal_sensation__RF__seed42__archived_80_20.csv.gz",
            "E6:predictions/E6__Hostein__TSV_3p__FTT_full__seed42__80-20_r04.csv.gz",
            "E7:predictions/E7__ASHRAE_2022__TSV_3p__aPMV.csv.gz", "E2:lopo_folds.json", "E4:pairs.csv",
            "E9b:E9b_nn_cases_summary.csv", "E9b:input:cohort_csv", "E1:input:cohort_csv", "cohort_csv",
            "E6:input:Moujalled", "E6:input:phase3_split:Hostein_TSV_3p_20-80.json",
            "phase1:experiment_grid_full.csv"} <= roles
    assert not any(r.startswith("report:") for r in roles)        # no aggregate file is read, all recomputed
    expected = ["phase1_haghirad_bars.pdf", "phase1_highcap_bars.png", "table_phase1_haghirad.tex",
                "phase2_indomain_bars.pdf", "table_phase2_indomain.tex", "heat_balance_indices_bars.pdf",
                "table_heat_balance_indices.tex", "phase3_zero_shot_bars.pdf", "table_phase3_adaptive.tex",
                "nn_agreement_vs_accuracy.pdf", "table_nn_disagreement.tex",
                "distributions/tsv7_hostein.png",
                "confusion_matrices/TSV_3p__FTTransformer_confusion_matrix.png",
                "feature_importance/feature_importance_heatmap.pdf",
                "feature_importance/thermal_preference_XGBoost_feature_importance.png"]
    for e in expected:
        assert (all_outputs / e).is_file(), e


def test_no_banned_wording_in_any_output(all_outputs):
    for p in all_outputs.rglob("*"):
        wording.check_text(str(p.relative_to(all_outputs)), "file name")
        if p.suffix in (".tex", ".csv", ".json"):
            wording.check_text(p.read_text(encoding="utf-8"), p.name)


def test_nn_figure_data_reads_e9b_full_cohort_k1_and_hybrids(all_outputs, synthetic_run):
    """The paper's NN figure reads E9b (F12, full cohort, k = 1)."""
    d = pd.read_csv(all_outputs / "nn_agreement_vs_accuracy_data.csv")
    assert set(d["population"]) == {"full"} and set(d["k"]) == {1} and set(d["features"]) == {"F12 (features_12)"}
    assert set(d["model"]) == set(E1_MODELS) | {"FTT_RF", "FTT_XGB"}
    assert set(d[d["model"].isin(["FTT_RF", "FTT_XGB"])]["kind"]) == {"hybrid"}
    e9b = synthetic_run["e9b"].set_index("target")
    for t in TARGETS:
        r = d[d["target"] == t].iloc[0]
        y = synthetic_run["cohort"][t].to_numpy()
        assert r["nn_agreement"] == pytest.approx(1 - e9b.loc[t, "primary_disagreement"], abs=1e-12)
        assert r["chance_agreement"] == pytest.approx(np.mean(y[:, None] == y[None, :]), abs=1e-12)
        assert r["n"] == len(y)
    np.testing.assert_allclose(d["nn_agreement"], 1 - d["primary_disagreement"])
    np.testing.assert_allclose(d["chance_agreement"], 1 - d["primary_chance_disagreement"])


def test_nn_figure_states_population():
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from analysis.paper_figures.nn_agreement_vs_accuracy import make_figure
    acc = pd.DataFrame({"target": list(TARGETS), "model_display": "RF", "kind": "standard",
                        "accuracy": [.5, .6, .7], "acc_low": [.49, .59, .69], "acc_high": [.51, .61, .71]})
    ref = pd.DataFrame({"target": list(TARGETS), "nn_agreement": [.4, .5, .6], "chance_agreement": [.2, .4, .5]})
    fig = make_figure(acc, ref, k=1, note="Reference lines (population: the 120 test votes)")
    assert any("population: the 120 test votes" in t.get_text() for t in fig.findobj(Text))
    plt.close(fig)


def test_phase2_table_seed_means_and_weighted_f1(all_outputs):
    tab = pd.read_csv(all_outputs / "table_phase2_indomain.csv")
    per = pd.read_csv(all_outputs / "table_phase2_indomain_per_seed.csv")
    assert set(tab["model"]) == set(E1_MODELS) | {"FTT_RF", "FTT_XGB"}
    g = per.groupby(["target", "model"])["weighted_f1"].mean()
    for _, r in tab.iterrows():
        assert r["weighted_f1"] == pytest.approx(g.loc[(r["target"], r["model"])])
    tex = (all_outputs / "table_phase2_indomain.tex").read_text()
    assert "FTT+RF" in tex and "FTT+XGBoost" in tex and "PENDING" not in tex
    # no "best" model in bold; point estimates, differences assessed by paired bootstrap
    assert "\\textbf" not in tex and "Best" not in tex and "paired bootstrap intervals" in tex
    assert "Mean over model seeds 42, 43 and 44 (SVM: seed 42 only); test set of 120 votes" in tex


def test_seeds_sentence_follows_the_seeds_averaged():
    from analysis.paper_figures.tables import seeds_sentence
    df = pd.DataFrame({"model": ["RF", "SVM", "ANN", "RF"], "seeds": [[42, 43], [42], [44], [42, 43]]})
    assert seeds_sentence(df) == "Mean over model seeds 42 and 43 (SVM: seed 42 only; ANN: seed 44 only)"
    assert seeds_sentence(pd.DataFrame({"model": ["RF"], "seeds": [[42, 43, 44]]})) == \
        "Mean over model seeds 42, 43 and 44"


def test_indices_include_e7b_and_drop_apts_epts(all_outputs):
    """Default = paper scope: every index on TSV-7 and TSV-3 only, no TPV row or panel."""
    tab = pd.read_csv(all_outputs / "table_heat_balance_indices.csv")
    assert set(tab["index"]) == {"PMV", "aPMV", "ePMV", "PTS_SET", "PTS_PET"}
    assert len(tab) == 10                     # primary variant, on_mask; 5 indices x TSV-7/TSV-3
    assert "thermal_preference" not in set(tab["target"])
    tex = (all_outputs / "table_heat_balance_indices.tex").read_text()
    assert "aPTS" not in tex and "ePTS" not in tex and "adaptive heat-balance" in tex
    assert "The models are evaluated on TSV-7 and TSV-3 only." in tex and "$x>0.5$" not in tex
    assert "Heat-balance models (PMV" in tex and "adaptive heat-balance models (aPMV" in tex
    assert "TPV" not in tex


def test_indices_table_reports_the_unbounded_epmv_factor_and_its_flag(all_outputs):
    """The ePMV e of each target is in the table CSV and caption, flagged outside [0.5, 1.0]."""
    tab = pd.read_csv(all_outputs / "table_heat_balance_indices.csv")
    ep = tab[tab["index"] == "ePMV"].set_index("target")
    assert ep.loc["thermal_sensation", "epmv_e"] == pytest.approx(1.18)
    assert bool(ep.loc["thermal_sensation", "epmv_e_flag_outside_range"]) is True
    assert bool(ep.loc["TSV_3p", "epmv_e_flag_outside_range"]) is False
    tex = (all_outputs / "table_heat_balance_indices.tex").read_text()
    assert "$e=1.180^{\\dagger}$" in tex and "$e=0.740$," not in tex and "$e=0.740$." in tex
    assert "not bounded" in tex and "Outside the range 0.5--1.0" in tex


@pytest.mark.parametrize("change", ["flag_wrong", "flag_missing", "no_e"])
def test_e7b_epmv_flag_must_match_e(synthetic_run, tmp_path, change):
    df = _e7b_frame(synthetic_run)
    sel = (df["method"] == "ePMV") & (df["target"] == "TSV_3p")
    if change == "flag_wrong":
        df.loc[sel, "epmv_e_flag_outside_range"] = True
    elif change == "flag_missing":
        df.loc[sel, "epmv_e_flag_outside_range"] = np.nan
    else:
        df = df.drop(columns=["epmv_e"])
    with _patched_output(synthetic_run, "E7b", "E7b_indices", "metrics.csv", df.to_csv(index=False).encode()):
        with pytest.raises(ValueError, match="E7b"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "indices")


def test_indices_tpv_opt_in(synthetic_run, tmp_path):
    """--indices_tpv (not the paper scope) adds the PMV / aPMV TPV rows; E7b stays TSV-7 / TSV-3."""
    out = _run_cli(synthetic_run, tmp_path / "o", "--only", "indices", "--indices_tpv")
    tab = pd.read_csv(out / "table_heat_balance_indices.csv")
    assert len(tab) == 12 and set(tab[tab["target"] == "thermal_preference"]["index"]) == {"PMV", "aPMV"}
    tex = (out / "table_heat_balance_indices.tex").read_text()
    assert "are evaluated on TSV-7 and TSV-3 only" in tex and "$-1$ if $x>0.5$" in tex


def test_documented_final_command_scope():
    """The final command of docs/PAPER_FIGURES.md produces the paper scope: no TPV index option in it."""
    doc = (Path(__file__).resolve().parents[1] / "docs" / "PAPER_FIGURES.md").read_text()
    finals = [line for line in doc.splitlines() if "-m analysis.paper_figures" in line and "--preview" not in line
              and "pytest" not in line]
    assert finals and all("--indices_tpv" not in line for line in finals)
    assert "--indices_no_tpv" not in doc


def _e7b_frame(sr):
    return pd.read_csv(sr["run"] / "E7b_indices" / "metrics.csv")


@pytest.mark.parametrize("change", ["add_tpv", "drop_tsv3", "drop_index"])
def test_e7b_unexpected_layout_refused(synthetic_run, tmp_path, change):
    df = _e7b_frame(synthetic_run)
    if change == "add_tpv":
        df = pd.concat([df, df[df["target"] == "TSV_3p"].assign(target="thermal_preference")])
    elif change == "drop_tsv3":
        df = df[df["target"] != "TSV_3p"]
    else:
        df = df[~((df["target"] == "TSV_3p") & (df["method"] == "PTS_PET"))]
    with _patched_output(synthetic_run, "E7b", "E7b_indices", "metrics.csv", df.to_csv(index=False).encode()):
        with pytest.raises(ValueError, match="E7b"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "indices")


def _e10_metrics(sr):
    return pd.read_csv(sr["run"] / "E10_hybrids" / "metrics.csv")


@pytest.mark.parametrize("change,match", [
    ("only_tsv7_seed42", "missing"), ("duplicate", "duplicated"), ("extra_seed", "unexpected"),
    ("feature_set", "feature_set"), ("split_id", "split_id"), ("support", "support"),
])
def test_e10_incomplete_or_inconsistent_metrics_refused(synthetic_run, tmp_path, change, match):
    df = _e10_metrics(synthetic_run)
    if change == "only_tsv7_seed42":
        df = df[(df["target"] == "thermal_sensation") & (df["seed"] == 42)]
    elif change == "duplicate":
        df = pd.concat([df, df.iloc[[3]]])
    elif change == "extra_seed":
        df = pd.concat([df, df.iloc[[0]].assign(seed=45)])
    elif change == "feature_set":
        df.loc[df.index[0], "feature_set"] = "F17"
    elif change == "split_id":
        df.loc[df.index[0], "split_id"] = "random_80_20"
    else:
        sup = json.loads(df.loc[df.index[0], "support"])
        k = next(iter(sup))
        sup[k] += 1
        df.loc[df.index[0], "support"] = json.dumps(sup)
    with _patched_output(synthetic_run, "E10", "E10_hybrids", "metrics.csv", df.to_csv(index=False).encode()):
        with pytest.raises(ValueError, match=match):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")


@pytest.mark.parametrize("change,match", [("y_pred", "confusion matrix"), ("row_idx", "archived test split"),
                                          ("y_true", "y_true"), ("seed", "seed")])
def test_e10_predictions_must_match_metrics_and_split(synthetic_run, tmp_path, change, match):
    m = _e10_metrics(synthetic_run).iloc[0]
    rel = f"predictions/{m['predictions']}"
    path = synthetic_run["run"] / "E10_hybrids" / rel
    pred = pd.read_csv(path)
    if change == "y_pred":
        pred.loc[0, "y_pred"] = next(c for c in ONT[m["target"]] if c != pred.loc[0, "y_pred"])
    elif change == "row_idx":
        pred.loc[0, "row_idx"] = int(synthetic_run["tr"][0])
    elif change == "y_true":
        pred["y_true"] = pred["y_true"].iloc[::-1].to_numpy()
    else:
        pred["seed"] = 43 if m["seed"] != 43 else 42
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(pred.to_csv(index=False).encode())
    with _patched_output(synthetic_run, "E10", "E10_hybrids", rel, buf.getvalue()):
        with pytest.raises(ValueError, match=match):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")


def test_phase3_tables_keep_paper_variants(all_outputs):
    ad = pd.read_csv(all_outputs / "table_phase3_adaptive.csv")
    assert set(ad["variant"]) == {"RF local retraining", "XGBoost local retraining", "FTT-Head-only fine-tuning",
                                  "FTT-Full fine-tuning", "Majority class (DEV)"}
    zs = (all_outputs / "table_phase3_zero_shot.tex").read_text()
    assert "Moujalled ($n=61$)" in zs and "Hostein ($n=43$)" in zs


def test_phase3_majority_class_rows(all_outputs, synthetic_run):
    """The zero-shot table / figure gain the
    cohort majority class (E6 Majority_descriptive_cohort), the adaptive table the DEV majority."""
    zs = pd.read_csv(all_outputs / "table_phase3_zero_shot.csv")
    maj = zs[zs["model"] == "Majority class"]
    assert len(maj) == 6 and maj["weighted_f1"].notna().all()
    for _, r in maj.iterrows():
        y = synthetic_run["french"][r["cohort"]][r["target"]]
        assert r["accuracy"] == pytest.approx(y.value_counts().max() / len(y))
    assert "Majority_deployable_ASHRAE_train" in set(zs["model"])         # other baselines kept in the CSV only
    tex = (all_outputs / "table_phase3_zero_shot.tex").read_text()
    assert tex.count(" & Majority class & ") == 6 and "most frequent vote of the cohort" in tex
    ad = pd.read_csv(all_outputs / "table_phase3_adaptive.csv")
    assert len(ad[ad["variant"] == "Majority class (DEV)"]) == 12 and "Majority_descriptive_TEST" not in set(ad["variant"])
    assert "most frequent vote of each development split" in (all_outputs / "table_phase3_adaptive.tex").read_text()


def test_phase3_majority_row_missing_refused(synthetic_run, tmp_path, monkeypatch):
    original = RunInputs.aggregate

    def without_majority(self, name):
        df = original(self, name)
        if name == "E6_full_cohort_zero_shot":
            df = df[~((df["model"] == "Majority_descriptive_cohort") & (df["cohort"] == "Hostein")
                      & (df["target"] == "TSV_3p"))]
        return df
    monkeypatch.setattr(RunInputs, "aggregate", without_majority)
    with pytest.raises(ValueError, match="majority class"):
        _run_cli(synthetic_run, tmp_path / "o", "--only", "phase3")


def test_nn_table_reads_e9b_recomputed_with_chance(all_outputs, synthetic_run):
    """tab:nn_disagreement = E9b: primary 1-NN, Case B and Case C, each beside its chance reference."""
    tab = pd.read_csv(all_outputs / "table_nn_disagreement.csv")
    run_summary = synthetic_run["e9b"].set_index("target")
    assert tab["target"].tolist() == list(TARGETS) and (tab["variables"] == ";".join(
        __import__("nn_cases").FROZEN_F12)).all()
    for _, r in tab.iterrows():
        s = run_summary.loc[r["target"]]
        assert r["primary_disagreement"] == pytest.approx(s["primary_disagreement"], abs=1e-12)
        assert r["case_b_clusters"] == s["case_b_clusters"] and r["case_c_pairs"] == s["case_c_pairs"]
        assert r["case_c_agreement"] == pytest.approx(s["case_c_agreement"], abs=1e-12)
        assert r["case_c_chance_agreement"] == pytest.approx(1 - s["case_c_chance_pair_disagreement"], abs=1e-12)
        y = synthetic_run["cohort"][r["target"]].to_numpy()
        assert r["chance_disagreement_cohort"] == pytest.approx(np.mean(y[:, None] != y[None, :]), abs=1e-12)
        assert r["primary_chance_disagreement"] == pytest.approx(np.mean(y[:, None] != y[None, :]), abs=1e-12)
    assert (tab["case_c_pairs"] >= 40).all() and (tab["case_b_clusters"] >= 20).all()
    assert (tab["primary_n_zero_distance"] >= 40).all()                     # the exact twins of the cohort
    tex = (all_outputs / "table_nn_disagreement.tex").read_text()
    assert "tab:nn_disagreement" in tex and "Case B: identical feature vectors" in tex and "600" in tex
    assert "Nearest neighbour" in tex and "12 variables" in tex and "Gower" not in tex
    for _, r in tab.iterrows():
        assert f"{100 * r['primary_disagreement']:.1f}" in tex and f"{100 * r['primary_chance_disagreement']:.1f}" in tex


def test_nn_e9b_must_be_f12(synthetic_run, tmp_path):
    """The paper's NN analysis is on F12: E9b detail files with another feature set are refused."""
    for t in TARGETS:
        rel = f"E9b_nn_cases_detail_{t}.json"
        d = json.loads((synthetic_run["run"] / "E9b_nn_cases" / rel).read_text())
        assert d["variables"] == list(__import__("nn_cases").FROZEN_F12) and d["features_key"] == "features_12"
    rel = f"E9b_nn_cases_detail_{TARGETS[0]}.json"
    d = json.loads((synthetic_run["run"] / "E9b_nn_cases" / rel).read_text())
    bad = {**d, "variables": d["variables"] + ["Trm_ema_28"]}
    with _patched_output(synthetic_run, "E9b", "E9b_nn_cases", rel, json.dumps(bad).encode()):
        with pytest.raises(ValueError, match="E9b"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "nn")


def test_nn_cases_run_summary_must_match_recomputation(synthetic_run, tmp_path):
    rel = "E9b_nn_cases_summary.csv"
    df = pd.read_csv(synthetic_run["run"] / "E9b_nn_cases" / rel)
    for col in ("case_c_agreement", "primary_disagreement"):
        bad = df.copy()
        bad.loc[0, col] = float(bad.loc[0, col]) + 0.01
        with _patched_output(synthetic_run, "E9b", "E9b_nn_cases", rel, bad.to_csv(index=False).encode()):
            with pytest.raises(ValueError, match="recomputed"):
                _run_cli(synthetic_run, tmp_path / f"o_{col}", "--only", "nn")


def test_nn_cases_in_figure_opt_in(synthetic_run, tmp_path):
    out = _run_cli(synthetic_run, tmp_path / "b", "--only", "nn", "--nn_cases_in_figure")
    assert (out / "nn_agreement_vs_accuracy.pdf").is_file() and (out / "table_nn_disagreement.tex").is_file()
    with pytest.raises(SystemExit):                                       # no population option on the paper item
        _run_cli(synthetic_run, tmp_path / "a", "--only", "nn", "--nn_population", "full")


def test_nn_cases_in_figure_carry_their_own_chance(synthetic_run, tmp_path, monkeypatch):
    """Case B / Case C in the figure: each value with its chance reference on the same groups / pairs
    (not the whole-cohort chance line), in the label, as an open-circle marker and in the data CSV."""
    from analysis.paper_figures import inputs as pf_inputs
    from analysis.paper_figures.wording import figure_texts
    seen = {}
    original = pf_inputs.OutputDir.savefig

    def spy(self, fig, rel, **kw):
        if rel.startswith("nn_agreement_vs_accuracy"):
            markers = []
            for ax in fig.axes:
                for line in ax.get_lines():
                    if line.get_marker() == "o":
                        markers.append((ax.get_title(loc="left"), line.get_markeredgecolor(),
                                        float(np.asarray(line.get_xdata())[0])))
            seen[rel] = (list(figure_texts(fig)), markers)
        return original(self, fig, rel, **kw)
    monkeypatch.setattr(pf_inputs.OutputDir, "savefig", spy)
    out = _run_cli(synthetic_run, tmp_path / "o", "--only", "nn", "--nn_cases_in_figure")
    texts, markers = seen["nn_agreement_vs_accuracy.pdf"]
    tab = pd.read_csv(out / "table_nn_disagreement.csv").set_index("target")
    data = pd.read_csv(out / "nn_agreement_vs_accuracy_data.csv")
    from analysis.paper_figures.metrics import TARGET_LABEL
    from analysis.paper_figures.nn_agreement_vs_accuracy import CASE_B_COLOUR, CASE_C_COLOUR
    import matplotlib.colors as mcolors
    for t in TARGETS:
        r = tab.loc[t]
        for value, chance, count, unit in ((r["case_b_cluster_agreement"], r["case_b_chance_cluster_agreement"],
                                            r["case_b_clusters"], "groups"),
                                           (r["case_c_agreement"], r["case_c_chance_agreement"], r["case_c_pairs"],
                                            "pairs")):
            label = f"{value*100:.0f}% (chance {chance*100:.0f}%; {int(count):,} {unit})".replace(",", " ")
            assert label in texts
        got = {mcolors.to_hex(c): x for title, c, x in markers if title == TARGET_LABEL[t]}
        assert got[mcolors.to_hex(CASE_B_COLOUR)] == pytest.approx(r["case_b_chance_cluster_agreement"], abs=1e-12)
        assert got[mcolors.to_hex(CASE_C_COLOUR)] == pytest.approx(r["case_c_chance_agreement"], abs=1e-12)
        # the same-population chance differs from the whole-cohort chance line of the panel
        d = data[data["target"] == t].iloc[0]
        assert d["case_b_chance_cluster_agreement"] == pytest.approx(r["case_b_chance_cluster_agreement"], abs=1e-12)
        assert d["case_c_chance_agreement"] == pytest.approx(r["case_c_chance_agreement"], abs=1e-12)
        assert abs(d["case_b_chance_cluster_agreement"] - d["chance_agreement"]) > 1e-6
    assert "own chance agreement on the same groups / pairs" in (out / "captions" / "nn_agreement_vs_accuracy.tex"
                                                                    ).read_text()


def test_distribution_counts(all_outputs, synthetic_run):
    c = pd.read_csv(all_outputs / "distributions/class_counts.csv")
    for name, df in [("ASHRAE-2022", synthetic_run["cohort"])] + list(synthetic_run["french"].items()):
        for t in TARGETS:
            got = c[(c["dataset"] == name) & (c["target"] == t)].set_index("class")["count"].to_dict()
            # every class of the scale, an empty one with count 0 (drawn as a zero bar)
            assert got == {int(k): int((df[t] == k).sum()) for k in ONT[t]}


def _read_pred(sr, dirname, name):
    return pd.read_csv(sr["run"] / dirname / "predictions" / name)


def test_aggregates_recomputed_from_the_predictions_files(all_outputs, synthetic_run):
    """The aggregate tables equal a direct computation from the synthetic predictions files."""
    sr = synthetic_run
    cohort, te = sr["cohort"], sr["te"]
    agg = {p.stem: pd.read_csv(p) for p in (all_outputs / "aggregates").glob("*.csv")}
    assert set(agg) == {"E1_metrics", "E1_confusion_matrices", "E6_full_cohort_zero_shot", "E6_strategy_metrics",
                        "E7_metrics", "E2_aggregates", "E4_ema_gains", "E1_model_differences"}
    # E1: one row per target x model x seed
    e1 = agg["E1_metrics"]
    assert len(e1) == 3 * 13
    for (t, m, s), p in sr["e1_preds"].items():
        r = e1[(e1["target"] == t) & (e1["model"] == m) & (e1["seed"] == s)].iloc[0]
        y = cohort[t].to_numpy()[te]
        assert r["accuracy"] == pytest.approx(np.mean(y == p), abs=1e-15)
        assert json.loads(r["confusion_matrix"]) == _cm(y, p, list(ONT[t])).tolist()
    cm = agg["E1_confusion_matrices"]
    assert cm["count"].sum() == 3 * 13 * len(te)
    # E6: each strategy pooled over the splits of a regime; development-split majority baseline
    st = agg["E6_strategy_metrics"]
    assert not set(st["strategy"]) & {"RF_pooled", "XGB_pooled", "RF_pooled_w1", "XGB_pooled_w1", "FTT_local"}
    c, t, reg = "Hostein", "TSV_3p", "20-80"
    payload = json.loads((sr["data_root"] / "splits" / "phase3" / f"{c}_{t}_{reg}.json").read_text())
    y_c = sr["french"][c][t].to_numpy()
    yt, yp, dev = [], [], []
    for sp in payload["splits"]:
        f = _read_pred(sr, "E6_phase3", f"E6__{c}__{t}__FTT_head__seed42__{sp['split_id']}.csv.gz")
        yt += f["y_true"].tolist()
        yp += f["y_pred"].tolist()
        d = pd.Series(y_c[sp["dev_idx"]]).value_counts()
        top = d[d == d.max()].index
        dev += [min(top, key=lambda v: (abs(v), v))] * len(sp["test_idx"])
    sel = (st["cohort"] == c) & (st["target"] == t) & (st["regime"] == reg)
    assert st[sel & (st["strategy"] == "FTT_head")]["accuracy"].iloc[0] == pytest.approx(
        np.mean(np.asarray(yt) == np.asarray(yp)), abs=1e-15)
    assert st[sel & (st["strategy"] == "Majority_deployable_DEV")]["accuracy"].iloc[0] == pytest.approx(
        np.mean(np.asarray(yt) == np.asarray(dev)), abs=1e-15)
    # E7: accuracy on the validity mask and its coverage
    e7 = agg["E7_metrics"]
    f = _read_pred(sr, "E7_pmv", "E7__ASHRAE_2022__TSV_3p__aPMV.csv.gz")
    mask = f["pmv_valid"].to_numpy(dtype=bool)
    r = e7[(e7["target"] == "TSV_3p") & (e7["method"] == "aPMV")].iloc[0]
    assert r["coverage"] == pytest.approx(mask.mean()) and int(r["n"]) == int(mask.sum())
    assert r["accuracy"] == pytest.approx(np.mean(f["y_true"][mask] == f["y_pred"][mask]), abs=1e-15)
    # E2: micro (all votes pooled) and macro (mean over the held-out publications) accuracy
    e2 = agg["E2_aggregates"]
    pubs = sorted(set(sr["pub"]))
    frames = [_read_pred(sr, "E2_lopo", f"E2__ASHRAE_2022__thermal_sensation__XGB__seed42__lopo_{k:02d}.csv.gz")
              for k in range(len(pubs))]
    allf = pd.concat(frames)
    g = e2[(e2["target"] == "thermal_sensation") & (e2["model"] == "XGB")].set_index("aggregate")["accuracy"]
    assert g["micro"] == pytest.approx(np.mean(allf["y_true"] == allf["y_pred"]), abs=1e-15)
    assert g["macro_over_publications"] == pytest.approx(np.mean([np.mean(f["y_true"] == f["y_pred"])
                                                                  for f in frames]), abs=1e-15)
    assert {"Majority_deployable", "Majority_descriptive"} <= set(e2["model"])
    # E4: gain = mean over the three seeds of acc(F17) - acc(F12), with a cluster bootstrap interval
    e4 = agg["E4_ema_gains"]
    assert len(e4) == 6 and not {"adj_low", "adj_high", "upper_below_margin"} & set(e4.columns)
    t = "TSV_3p"
    y = cohort[t].to_numpy()[te]
    acc12 = [np.mean(_read_pred(sr, "E4_ema", f"E4__ASHRAE_2022__{t}__RF__seed{s}__archived_80_20.csv.gz")["y_pred"]
                     == y) for s in (42, 43, 44)]
    acc17 = [np.mean(sr["e1_preds"][(t, "RF", s)] == y) for s in (42, 43, 44)]
    r = e4[(e4["family"] == "RF") & (e4["target"] == t)].iloc[0]
    assert r["gain"] == pytest.approx(np.mean(acc17) - np.mean(acc12), abs=1e-12)
    assert r["ci95_low"] <= r["gain"] <= r["ci95_high"]


def _gz_csv(df):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode())
    return buf.getvalue()


def test_e1_model_differences_equal_a_direct_computation(all_outputs, synthetic_run):
    """Difference from XGBoost = mean-over-seeds accuracy of the model - that of XGBoost on the same test votes
    (SVM: its single seed; hybrids: E10 files), with the frozen building-cluster bootstrap of revision_stats."""
    import revision_stats as rs
    sr = synthetic_run
    cohort, te = sr["cohort"], sr["te"]
    d = pd.read_csv(all_outputs / "aggregates" / "E1_model_differences.csv")
    assert list(d.columns) == ["target", "model", "reference", "n_test", "n_clusters", "acc_model", "acc_reference",
                               "difference", "ci95_low", "ci95_high", "n_bootstrap", "bootstrap_seed"]
    assert list(zip(d["target"], d["model"])) == [(t, m) for t in TARGETS for m in
                                                  ("RF", "SVM", "ANN", "FTT", "FTT_RF", "FTT_XGB")]
    assert set(d["reference"]) == {"XGB"} and set(d["n_test"]) == {len(te)}
    assert set(d["n_clusters"]) == {len(set(sr["bld"][te]))}
    assert (rs.BOOT_B, rs.BOOT_SEED) == (10_000, 20261008)
    assert set(d["n_bootstrap"]) == {10_000} and set(d["bootstrap_seed"]) == {20261008}
    w = rs.frozen_draw_weights(len(set(sr["bld"][te])))
    order, pos = rs.cluster_positions(sr["bld"][te])
    for _, r in d.iterrows():
        t, m = r["target"], r["model"]
        y = cohort[t].to_numpy()[te]
        if m in ("FTT_RF", "FTT_XGB"):
            preds = [_read_pred(sr, "E10_hybrids", f"E10__ASHRAE_2022__{t}__{m}__seed{s}__archived_80_20.csv.gz")[
                "y_pred"].to_numpy() for s in (42, 43, 44)]
        else:
            preds = [sr["e1_preds"][(t, m, s)] for s in ((42,) if m == "SVM" else (42, 43, 44))]
        ref = [sr["e1_preds"][(t, "XGB", s)] for s in (42, 43, 44)]
        acc_m, acc_ref = np.mean([np.mean(p == y) for p in preds]), np.mean([np.mean(p == y) for p in ref])
        assert r["acc_model"] == pytest.approx(acc_m, abs=1e-12)
        assert r["acc_reference"] == pytest.approx(acc_ref, abs=1e-12)
        assert r["difference"] == pytest.approx(acc_m - acc_ref, abs=1e-12)
        assert r["ci95_low"] <= r["difference"] <= r["ci95_high"] and r["ci95_low"] < r["ci95_high"]
        # paired: each replicate resamples the buildings once and recomputes both models on the same votes
        per_vote = np.mean([p == y for p in preds], axis=0) - np.mean([p == y for p in ref], axis=0)
        reps = (w @ np.bincount(pos, weights=per_vote, minlength=len(order))) / (
            w @ np.bincount(pos, minlength=len(order)).astype(float))
        assert (r["ci95_low"], r["ci95_high"]) == pytest.approx(tuple(np.quantile(reps, [.025, .975])), abs=1e-12)


def test_e1_model_differences_of_identical_predictions_are_exactly_zero(synthetic_run, tmp_path):
    """A model with the predictions of XGBoost, seed by seed: difference 0 and interval [0, 0] (paired)."""
    from contextlib import ExitStack
    sr, t = synthetic_run, "TSV_3p"
    with ExitStack() as stack:
        for s in (42, 43, 44):
            df = _read_pred(sr, "E1_phase2", f"E1__ASHRAE_2022__{t}__XGB__seed{s}__archived_80_20.csv.gz")
            df["model"] = "RF"
            stack.enter_context(_patched_output(
                sr, "E1", "E1_phase2", f"predictions/E1__ASHRAE_2022__{t}__RF__seed{s}__archived_80_20.csv.gz",
                _gz_csv(df)))
        # RunInputs alone: the E4 pairs of the unpatched run name the original RF files
        from analysis.paper_figures.aggregate import Aggregates
        d = Aggregates(RunInputs(sr["run"], data_root=sr["data_root"])).table("E1_model_differences")
    r = d[(d["target"] == t) & (d["model"] == "RF")].iloc[0]
    assert r["difference"] == 0 and r["ci95_low"] == 0 and r["ci95_high"] == 0
    assert r["acc_model"] == r["acc_reference"]
    other = d[(d["target"] == t) & (d["model"] == "ANN")].iloc[0]
    assert other["difference"] != 0


@pytest.mark.parametrize("change, match", [("reordered", "row_idx"), ("fewer", "row_idx"),
                                           ("other_buildings", "building_id")])
def test_e1_model_differences_refuse_models_on_other_test_rows(synthetic_run, change, match):
    """Every model must be predicted on the archived test rows, in their order, with the same buildings."""
    from analysis.paper_figures.aggregate import AggregateError, Aggregates
    sr, t = synthetic_run, "thermal_preference"
    name = f"E10__ASHRAE_2022__{t}__FTT_XGB__seed43__archived_80_20.csv.gz"
    df = _read_pred(sr, "E10_hybrids", name)
    if change == "reordered":                      # same votes, another order
        df = df.iloc[::-1].reset_index(drop=True)
    elif change == "fewer":
        df = df.iloc[:-1]
    else:
        df["building_id"] = df["building_id"].to_numpy()[::-1] + 1000
    with _patched_output(sr, "E10", "E10_hybrids", f"predictions/{name}", _gz_csv(df)):
        with pytest.raises(AggregateError, match=match):
            Aggregates(RunInputs(sr["run"], data_root=sr["data_root"])).frame("E1_model_differences")


def test_e1_model_differences_without_e10(synthetic_run, tmp_path):
    """E10 pending: refused in the final mode before any write; a preview leaves the hybrid rows out."""
    run = synthetic_run["run"]
    index_path = run / "manifest.json"
    original = index_path.read_bytes()
    index = json.loads(original)
    index["experiments"].pop("E10")
    try:
        index_path.write_text(json.dumps(index))
        with pytest.raises(PendingExperimentError):
            _run_cli(synthetic_run, tmp_path / "a", "--only", "aggregates")
        assert not (tmp_path / "a").exists()
        out = _run_cli(synthetic_run, tmp_path / "b", "--only", "aggregates", "--preview")
        d = pd.read_csv(out / "aggregates" / "E1_model_differences.csv")
        assert sorted(set(d["model"])) == ["ANN", "FTT", "RF", "SVM"] and len(d) == 12
        assert json.loads((out / "paper_figures_manifest.json").read_text())["pending_experiments_left_out"] == ["E10"]
    finally:
        index_path.write_bytes(original)


def test_aggregate_refuses_predictions_that_do_not_match_the_cohort(synthetic_run, tmp_path):
    rel = "predictions/E1__ASHRAE_2022__TSV_3p__ANN__seed43__archived_80_20.csv.gz"
    df = pd.read_csv(synthetic_run["run"] / "E1_phase2" / rel)
    df["y_true"] = df["y_true"].iloc[::-1].to_numpy()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode())
    with _patched_output(synthetic_run, "E1", "E1_phase2", rel, buf.getvalue()):
        with pytest.raises(ValueError, match="y_true"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")


def _e1_with_seeds(sr, seeds, inventory=None):
    """The E1 manifest of the synthetic run with other seeds (and inventory), re-hashed in the run index."""
    man = json.loads((sr["run"] / "E1_phase2" / "run_manifest.json").read_text())
    man["seeds"] = {"models": seeds}
    man["ml_inventory"] = sorted(f"{m}:{x}" for m, ss in seeds.items() for x in ss) if inventory is None else inventory
    return json.dumps(man).encode()


@pytest.mark.parametrize("change,match", [
    ({"ANN": [42]}, r"E1 ANN seeds \(42,\) != the paper seeds \(42, 43, 44\)"),
    ({"ANN": [42], "FTT": [42, 43]}, "the paper seeds"),
    ({"FTT": [42, 43, 45]}, r"E1 FTT seeds \(42, 43, 45\) != the paper seeds"),
    ({"RF": [42]}, r"E1 RF seeds \(42,\) != the paper seeds"),
    ({"SVM": [42, 43]}, r"E1 SVM seeds \(42, 43\) != the paper seeds \(42,\)"),
    ({"ANN": [42, 43, 43, 44]}, "lists a seed twice"),
    ({"FTT": [43, 44, 46]}, "the paper seeds"),
])
def test_e1_seeds_must_equal_the_paper_seeds(synthetic_run, tmp_path, change, match):
    """A reduced, substituted or duplicated E1 seed set is refused even if self-consistent (manifest and
    ml_inventory agree), before any table is written; the SVM is fitted once (seed 42)."""
    seeds = {m: ([42] if m == "SVM" else [42, 43, 44]) for m in E1_MODELS}
    seeds.update(change)
    inventory = None if "twice" not in match else sorted(f"{m}:{x}" for m, ss in seeds.items() for x in set(ss))
    with _patched_manifest(synthetic_run, "E1", "E1_phase2", _e1_with_seeds(synthetic_run, seeds, inventory)):
        with pytest.raises(ValueError, match=match):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")
    assert not (tmp_path / "o").exists() or not any((tmp_path / "o").rglob("table_*"))


@pytest.mark.parametrize("bad", [[42.1, 43.9, 44.2], [42.0, 43, 44], ["42", 43, 44], [True, 43, 44]])
def test_e1_non_integer_seeds_are_refused(synthetic_run, tmp_path, bad):
    """Seeds must be JSON integers: a float (even 42.0), a string or a bool is refused before int()."""
    seeds = {m: ([42] if m == "SVM" else [42, 43, 44]) for m in E1_MODELS}
    seeds["FTT"] = bad
    inventory = sorted(f"{m}:{x}" for m, ss in seeds.items() for x in ss)
    with _patched_manifest(synthetic_run, "E1", "E1_phase2", _e1_with_seeds(synthetic_run, seeds, inventory)):
        with pytest.raises(ValueError, match=r"non-integer seed for \['FTT'\]"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")
    assert not (tmp_path / "o").exists() or not any((tmp_path / "o").rglob("table_*"))


def test_e1_inventory_must_match_the_seeds(synthetic_run, tmp_path):
    seeds = {m: ([42] if m == "SVM" else [42, 43, 44]) for m in E1_MODELS}
    inventory = sorted(f"{m}:{x}" for m, ss in seeds.items() for x in ss if f"{m}:{x}" != "FTT:44")
    with _patched_manifest(synthetic_run, "E1", "E1_phase2", _e1_with_seeds(synthetic_run, seeds, inventory)):
        with pytest.raises(ValueError, match="ml_inventory"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")


def test_a_consistent_e1_run_with_a_deep_seed_removed_is_refused(synthetic_run, tmp_path):
    """Manifest, ml_inventory and predictions all consistent after removing FTT seed 44: still refused."""
    seeds = {m: ([42] if m == "SVM" else [42, 43, 44]) for m in E1_MODELS}
    seeds["FTT"] = [42, 43]
    man = json.loads(_e1_with_seeds(synthetic_run, seeds))
    for t in TARGETS:
        man["outputs_sha256"].pop(f"predictions/E1__ASHRAE_2022__{t}__FTT__seed44__archived_80_20.csv.gz")
    with _patched_manifest(synthetic_run, "E1", "E1_phase2", json.dumps(man).encode()):
        with pytest.raises(ValueError, match=r"E1 FTT seeds \(42, 43\) != the paper seeds \(42, 43, 44\)"):
            _run_cli(synthetic_run, tmp_path / "o", "--only", "phase2")


def test_pending_experiments_refused_then_left_out(synthetic_run, tmp_path):
    run = synthetic_run["run"]
    index_path = run / "manifest.json"
    original = index_path.read_bytes()
    index = json.loads(original)
    for e in ("E10", "E7b", "E9b"):
        index["experiments"].pop(e)
    try:
        index_path.write_text(json.dumps(index))
        with pytest.raises(PendingExperimentError):
            _run_cli(synthetic_run, tmp_path / "a", "--only", "phase2")
        with pytest.raises(PendingExperimentError):
            _run_cli(synthetic_run, tmp_path / "b", "--only", "distributions", "indices")
        assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()    # refused before any write
        with pytest.raises(SystemExit):                    # a preview never lands among the run outputs
            _run_cli(synthetic_run, run / "PAPER_FIGURES", "--only", "phase2", "--preview")
        assert not (run / "PAPER_FIGURES").exists()
        out = _run_cli(synthetic_run, tmp_path / "c", "--only", "phase2", "indices", "nn", "--preview")
        man = json.loads((out / "paper_figures_manifest.json").read_text())
        assert man["pending_experiments_left_out"] == ["E10", "E7b", "E9b"] and man["mode"] == "preview"
        assert not (out / "table_nn_disagreement.tex").exists()
        nnd = pd.read_csv(out / "nn_agreement_vs_accuracy_data.csv")      # chance line only, no NN line
        assert nnd["nn_agreement"].isna().all() and nnd["chance_agreement"].notna().all()
        assert "PENDING" in (out / "table_phase2_indomain.tex").read_text()
        assert set(pd.read_csv(out / "table_heat_balance_indices.csv")["index"]) == {"PMV", "aPMV"}
        assert "hybrid" not in set(pd.read_csv(out / "nn_agreement_vs_accuracy_data.csv")["kind"])
    finally:
        index_path.write_bytes(original)


def test_changed_input_refused(synthetic_run, tmp_path):
    p = (synthetic_run["run"] / "E1_phase2" / "predictions"
         / "E1__ASHRAE_2022__thermal_sensation__RF__seed42__archived_80_20.csv.gz")
    original = p.read_bytes()
    try:
        p.write_bytes(original + b"\n")
        with pytest.raises(InputHashError):
            _run_cli(synthetic_run, tmp_path / "x", "--only", "phase2")
    finally:
        p.write_bytes(original)
    cohort = synthetic_run["data_root"] / "Data" / "ASHRAE_2022_Clean_api.csv"
    original = cohort.read_bytes()
    try:
        cohort.write_bytes(original.replace(b"\n", b"\r\n", 1))
        with pytest.raises(InputHashError):
            _run_cli(synthetic_run, tmp_path / "y", "--only", "nn")
    finally:
        cohort.write_bytes(original)


def test_output_directory_must_be_new_or_empty(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x.txt").write_text("x")
    with pytest.raises(FileExistsError):
        OutputDir(tmp_path / "full")
    out = OutputDir(tmp_path / "empty")
    out.write_text("a.tex", "ok")
    with pytest.raises(FileExistsError):
        out.write_text("a.tex", "again")
    with pytest.raises(ValueError):
        OutputDir(tmp_path / "rerun_2026-04-28_no_reweighting" / "figs")


def test_list_items(capsys):
    assert cli.main(["--list"]) == 0
    text = capsys.readouterr().out
    for name, item in ITEMS.items():
        assert name in text
        for o in item.outputs + [item.paper]:
            wording.check_text(o, "--list output")
    wording.check_text(text, "--list output")


def test_regeneration_is_byte_identical(synthetic_run, all_outputs, tmp_path):
    again = _run_cli(synthetic_run, tmp_path / "again")
    m1 = json.loads((all_outputs / "paper_figures_manifest.json").read_text())
    m2 = json.loads((again / "paper_figures_manifest.json").read_text())
    h1 = {o["path"]: o["sha256"] for o in m1["outputs"]}
    h2 = {o["path"]: o["sha256"] for o in m2["outputs"]}
    assert h1.keys() == h2.keys()
    assert [p for p in h1 if h1[p] != h2[p]] == []


def test_feature_bars_use_english_labels(synthetic_run, tmp_path, monkeypatch):
    seen = {}
    original = OutputDir.savefig

    def recording(self, fig, rel, **kw):
        if rel.endswith("_feature_importance.png"):
            seen[rel] = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
        return original(self, fig, rel, **kw)
    monkeypatch.setattr(OutputDir, "savefig", recording)
    out = _run_cli(synthetic_run, tmp_path / "o", "--only", "feature_importance")
    assert len(seen) == 6
    for labels in seen.values():
        assert "gender" in labels and "age" in labels and "Sexe" not in labels and "Âge" not in labels
    groups = set(pd.read_csv(out / "feature_importance" / "feature_importance_per_seed.csv")["group"])
    assert {"Sexe", "Âge"} <= groups                      # CSV keeps the model feature names


# ------------------------------------------------------------------------------------------------
# Generated figures: print size, captions, names, no best-model bold
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def drawn(synthetic_run, tmp_path_factory):
    """One full build with every figure recorded at save time: size, texts (with font sizes), y limits
    and bar colours of each axes."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    seen = {}
    original = OutputDir.savefig

    def spy(self, fig, rel, **kw):
        if rel not in seen:
            fig.canvas.draw()
            texts = [(t.get_text(), float(t.get_fontsize())) for t in fig.findobj(Text)
                     if t.get_visible() and t.get_text().strip()]
            axes = []
            for ax in fig.axes:
                bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_width() > 0]
                axes.append({"ylim": ax.get_ylim(), "title": ax.get_title(),
                             "bars": [(p.get_x() + p.get_width() / 2, p.get_height(),
                                       mcolors.to_hex(p.get_facecolor())) for p in bars]})
            seen[rel] = {"size": tuple(fig.get_size_inches()), "texts": texts, "axes": axes}
        return original(self, fig, rel, **kw)
    mp = pytest.MonkeyPatch()
    mp.setattr(OutputDir, "savefig", spy)
    try:
        out = _run_cli(synthetic_run, tmp_path_factory.mktemp("drawn") / "out")
    finally:
        mp.undo()
    return out, seen


def test_every_figure_drawn_at_print_size_with_readable_text(drawn):
    """Each figure is drawn at the width at which it is included (at most the 510 pt text
    width), so its text is printed at the drawn size: no text below 7 pt."""
    import re
    from analysis.paper_figures.bar_charts import PAGE_WIDTH_IN
    out, seen = drawn
    assert any(r.startswith("nn_agreement_vs_accuracy") for r in seen)
    for rel, rec in seen.items():
        assert rec["size"][0] <= PAGE_WIDTH_IN + 1e-9, rel
        small = [(t, f) for t, f in rec["texts"] if f < 7.0]
        assert not small, (rel, small[:5])
    for pdf in out.rglob("*.pdf"):
        if pdf.name == "feature_importance_heatmap.pdf":     # analysis/feature_importance_heatmap.py, unchanged
            continue
        # the saved page (tight bounding box) is not wider than the width the figure is included at
        width = 510.24 * (0.32 if pdf.parent.name == "distributions" else
                          0.49 if pdf.parent.name in ("confusion_matrices", "feature_importance") else 1.0)
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
        assert box and float(box.group(1)) <= width + 0.5, (pdf.name, box and box.group(1), width)
    nn = seen["nn_agreement_vs_accuracy.pdf"]
    assert nn["size"][0] >= 0.95 * PAGE_WIDTH_IN
    # the population and definitions are in the caption, not in a note under the legend
    assert not any("Reference lines" in t for t, _ in nn["texts"])
    assert "Range over model seeds" in [t for t, _ in nn["texts"]]


def test_nn_figure_fixed_model_order_and_seed_range_columns(drawn):
    out, seen = drawn
    nn = seen["nn_agreement_vs_accuracy.pdf"]
    texts = [t for t, _ in nn["texts"]]
    order = ["RF", "XGBoost", "SVM", "ANN", "FTT", "FTT+RF", "FTT+XGBoost"]
    pos = [texts.index(m) for m in order]
    assert pos == sorted(pos)                         # the table order on every panel, top to bottom
    assert all(texts.count(m) == 1 for m in order)    # y labels on the first panel only
    d = pd.read_csv(out / "nn_agreement_vs_accuracy_data.csv")
    assert {"accuracy_seed_min", "accuracy_seed_max"} <= set(d.columns) and "acc_low" not in d.columns
    svm = d[d["model"] == "SVM"]
    assert (svm["accuracy_seed_min"] == svm["accuracy"]).all() and (svm["accuracy_seed_max"] == svm["accuracy"]).all()


def test_one_name_per_model_and_dataset(drawn):
    """One name per model and dataset: RF, XGBoost, SVM, ANN, FTT, FTT+RF, FTT+XGBoost; ASHRAE-2018 (Haghirad 2018 v2 release),
    ASHRAE-2022, Moujalled, Hostein; no D1-D4, no long or file-style model names in figures and LaTeX."""
    import re
    out, seen = drawn
    figure_text = "\n".join(t for rec in seen.values() for t, _ in rec["texts"])
    latex = "\n".join(line for p in out.rglob("*.tex") for line in p.read_text().splitlines()
                      if not line.startswith("%"))          # comments name the files (RandomForest, ...)
    for text in (figure_text, latex):
        assert "Random Forest" not in text and "RandomForest" not in text and "FTT + " not in text
        assert not re.search(r"Dataset D\d", text) and "db2.01" not in text and "FTT+XGB " not in text
    assert "FT-Transformer" not in figure_text
    assert all(m in figure_text for m in ("RF", "XGBoost", "SVM", "ANN", "FTT", "FTT+RF", "FTT+XGBoost"))
    for name in ("ASHRAE-2018\n(Haghirad 2018 v2 release)", "ASHRAE-2022", "Moujalled", "Hostein"):
        assert name in figure_text
    # FT-Transformer appears in LaTeX only where FTT is defined
    for line in latex.splitlines():
        if "FT-Transformer" in line and not line.startswith("%"):
            assert "FTT: FT-Transformer" in line or "FTT (FT-Transformer)" in line, line
    assert "ASHRAE-2018 (Haghirad 2018 v2 release)" in (out / "table_phase1_haghirad.tex").read_text()


def test_no_bold_in_any_table(drawn):
    out, _ = drawn
    for p in out.glob("table_*.tex"):
        assert "\\textbf" not in p.read_text(), p.name


def test_captions_state_what_the_reader_needs(drawn):
    """Captions: coverages, QWK = 0 of a constant predictor, seeds, TSV-3 / TPV codes, shared colour scale,
    RF MDI vs XGBoost gain, whiskers = seed range, heat-balance models."""
    out, _ = drawn
    cap = {p.stem: p.read_text() for p in (out / "captions").glob("*.tex")}
    assert set(cap) == {"distributions", "phase1_haghirad_bars", "phase1_highcap_bars", "phase2_indomain_bars",
                        "tsv7_confusion_matrices", "tsv3_confusion_matrices", "tpv_confusion_matrices",
                        "heat_balance_indices_bars", "feature_importance_heatmap", "feature_importance_bars",
                        "phase3_zero_shot_bars", "nn_agreement_vs_accuracy"}
    for k, v in cap.items():
        assert v.count("\\caption{") == 1 and "\\label{fig:" in v and "include it at width=" in v, k
        wording.check_text(v, k)
    assert "$-1$ want cooler, $0$ no change, $+1$ want warmer" in cap["distributions"]
    assert "TSV-3 $-1$ cool side" in cap["distributions"] and "empty bar at 0" in cap["distributions"]
    assert "$-1$ want cooler" in cap["tpv_confusion_matrices"] and "TSV-3 $-1$" in cap["tsv3_confusion_matrices"]
    for t in ("tsv7", "tsv3", "tpv"):
        assert "colour scale is shared by the five panels" in cap[f"{t}_confusion_matrices"]
    assert "coverage" in cap["heat_balance_indices_bars"] and "heat-balance models" in cap["heat_balance_indices_bars"]
    assert "QWK $=0$ by construction" in cap["phase3_zero_shot_bars"]
    assert "Mean over model seeds 42, 43 and 44 (SVM: seed 42 only)" in cap["phase2_indomain_bars"]
    assert "paired bootstrap intervals" in cap["phase2_indomain_bars"]
    for k in ("feature_importance_heatmap", "feature_importance_bars"):
        assert "MDI" in cap[k] and "gain-based" in cap[k]
    assert "EMA weather" in cap["feature_importance_heatmap"]
    nn = cap["nn_agreement_vs_accuracy"]
    assert "range of the test accuracy over the model seeds (not a confidence interval)" in nn
    assert "one seed has no whisker" in nn and "not accuracies of a model" in nn
    tab = (out / "table_nn_disagreement.tex").read_text()
    assert "Chance disagr." in tab and "Chance agr." in tab and "Case B rests on" in tab


def test_indices_ticks_carry_coverage(drawn):
    out, seen = drawn
    tab = pd.read_csv(out / "table_heat_balance_indices.csv")
    texts = [t for t, _ in seen["heat_balance_indices_bars.pdf"]["texts"]]
    for _, r in tab.iterrows():
        assert f"{r['index']} ({100 * r['coverage']:.0f}%)" in texts


def test_distribution_colours_fixed_per_vote_and_labels_clear(drawn):
    from analysis.paper_figures.items import DIST_COLOURS, DIST_FILE
    out, seen = drawn
    for t in TARGETS:
        for key in ("ashrae", "moujalled", "hostein"):
            rec = seen[f"distributions/{DIST_FILE[t]}_{key}.pdf"]
            (ax,) = rec["axes"]
            bars = sorted(ax["bars"])
            assert len(bars) == len(ONT[t])                    # empty classes kept as zero bars
            assert [c for _, _, c in bars] == [DIST_COLOURS[t][v].lower() for v in ONT[t]]
            assert ax["ylim"][1] >= 1.15 * max(h for _, h, _ in bars)   # value labels clear of the frame
        assert (out / "distributions" / f"{DIST_FILE[t]}_hostein.png").is_file()


def test_phase2_bars_axis_stops_at_one(drawn):
    _, seen = drawn
    assert {ax["ylim"][1] for ax in seen["phase2_indomain_bars.pdf"]["axes"]} == {1.0}


def test_confusion_matrices_share_one_scale_per_target(drawn):
    out, _ = drawn
    cm = pd.read_csv(out / "confusion_matrices" / "confusion_matrices.csv")
    for t, g in cm.groupby("target"):
        assert g["colour_scale_max"].nunique() == 1 and g["colour_scale_max"].iloc[0] == g["count"].max()


def test_phase3_csv_weighted_f1_filled_for_every_baseline(drawn, synthetic_run):
    """Every baseline row has its weighted-F1 (the ASHRAE-training majority rows once had a blank one)."""
    out, _ = drawn
    zs = pd.read_csv(out / "table_phase3_zero_shot.csv")
    assert zs["weighted_f1"].notna().all()
    rep = pd.read_csv(out / "aggregates" / "E6_full_cohort_zero_shot.csv")
    for _, r in rep[rep["model"] == "Majority_deployable_ASHRAE_train"].iterrows():
        got = zs[(zs["cohort"] == r["cohort"]) & (zs["target"] == r["target"]) & (zs["model"] == r["model"])]
        assert got["weighted_f1"].iloc[0] == pytest.approx(metrics.weighted_f1_from_cm(r["confusion_matrix"]),
                                                           abs=1e-12)


# ---------------------------------------------------------------- E6 / E9b from a later run directory
def _extra_run(sr, root, e1_sha=None, with_e1=False, grid_pin=None, e1_input=None, drop_e1_sha=False,
               phase1=None):
    """A later run directory holding copies of E6 and E9b (E6 records the E1 manifest it read)."""
    import shutil
    run = sr["run"]
    index = json.loads((run / "manifest.json").read_text())
    extra = root / "rerun_release"
    exps = {}
    for exp, d in (("E6", "E6_phase3"), ("E9b", "E9b_nn_cases")) + ((("E1", "E1_phase2"),) if with_e1 else ()):
        shutil.copytree(run / d, extra / d)
        if exp == "E6":
            man = json.loads((extra / d / "run_manifest.json").read_text())
            e1_recorded = index["experiments"]["E1"]["manifest_sha256"]
            man["e1_manifest_sha256"] = e1_sha or e1_recorded
            if drop_e1_sha:
                man.pop("e1_manifest_sha256")
            man.setdefault("inputs_sha256", {})["e1:run_manifest.json"] = e1_input or e1_recorded
            (extra / d / "run_manifest.json").write_text(json.dumps(man))
        exps[exp] = {"manifest": f"{d}/run_manifest.json", "manifest_sha256": _sha(extra / d / "run_manifest.json"),
                     "status": "complete"}
    phase1 = {**index["phase1"], "matches_pinned": True, **(phase1 or {})}
    if grid_pin:
        phase1["experiment_grid_full_sha256_pinned"] = grid_pin
    (extra / "manifest.json").write_text(json.dumps({"format": index["format"], "experiments": exps,
                                                     "phase1": phase1}))
    return extra


def test_e6_and_e9b_from_a_later_run_directory_give_the_same_outputs(synthetic_run, all_outputs, tmp_path):
    extra = _extra_run(synthetic_run, tmp_path)
    out = _run_cli(synthetic_run, tmp_path / "o", "--only", "phase3", "nn", "--extra_run_dir", str(extra))
    built = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file() and p.name != "paper_figures_manifest.json")
    assert built and all((out / p).read_bytes() == (all_outputs / p).read_bytes() for p in built)
    man = json.loads((out / "paper_figures_manifest.json").read_text())
    assert man["extra_run_dirs"][0]["replaces"] == ["E6", "E9b"]
    assert any(str(extra.name) in c["path"] and c["role"].startswith("E6:") for c in man["inputs"])


@pytest.mark.parametrize("case, exc, match", [
    ("other_e1", "InputHashError", "is not the E1 of"),
    ("no_e1_sha", "InputHashError", "is not the E1 of"),
    ("other_e1_input", "InputHashError", "is not the E1 of"),
    ("with_e1", "ValueError", "E1 must come from --run_dir"),
    ("grid_pin", "InputHashError", "Phase 1 grid hash, pin or matches_pinned"),
    ("grid_hash", "InputHashError", "Phase 1 grid hash, pin or matches_pinned"),
    ("not_matches_pinned", "InputHashError", "Phase 1 grid hash, pin or matches_pinned"),
])
def test_a_later_run_directory_must_extend_the_same_run(synthetic_run, tmp_path, case, exc, match):
    extra = _extra_run(synthetic_run, tmp_path, e1_sha="0" * 64 if case == "other_e1" else None,
                       with_e1=case == "with_e1", grid_pin="1" * 64 if case == "grid_pin" else None,
                       e1_input="2" * 64 if case == "other_e1_input" else None,
                       drop_e1_sha=case == "no_e1_sha",
                       phase1={"experiment_grid_full_sha256": "3" * 64} if case == "grid_hash" else
                       ({"matches_pinned": False} if case == "not_matches_pinned" else None))
    from analysis.paper_figures import inputs as pf_inputs
    with pytest.raises(getattr(pf_inputs, exc, ValueError), match=match):
        _run_cli(synthetic_run, tmp_path / "o", "--only", "phase3", "--extra_run_dir", str(extra))
    assert not (tmp_path / "o").exists() or not any((tmp_path / "o").rglob("table_*"))
