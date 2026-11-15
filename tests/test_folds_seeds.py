# -*- coding: utf-8 -*-
"""
Shared folds, frozen seeds and removal of the hybrids from the main pipeline.

Blocking tests:

* identical fold indices across families — every family (RF, XGB, SVM, ANN,
  FTT) receives the materialised folds of the target and records the same
  fold-vector hash; ``select_best`` uses them for the search and the reported
  folds (no ``RANDOM_SEED+1`` re-draw);
* seeds honoured — RF / XGB fitted with 42, 43, 44, SVM once, ANN / FTT with
  the frozen seeds; any other seed set is refused before data are loaded;
* hybrids absent from ``models.py`` — its CLI no longer offers them and an E1 run
  writes no hybrid output; the restored ``hybrid.py`` is
  imported only by the E10 runner, and ``config.yaml`` carries its ``hybrid``
  heads with the paper values; the fold-number Wilcoxon file is no longer produced.

Synthetic data only, except a read-only check of the archived ASHRAE-2022 split
files against the cohort labels (no model trained; folds written to tmp_path,
never to ``splits/``). Each test well under 60 s on CPU.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import revision_folds as rf  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY  # noqa: E402

TARGETS = ("thermal_sensation", "TSV_3p", "thermal_preference")
TINY_CLASSICAL = {
    "RandomForest": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2,
                     "min_samples_leaf": 1, "class_weight": None},
    "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3,
                "subsample": 1.0, "colsample_bytree": 1.0},
    "SVM": {"kernel": ["rbf", "linear"], "C": [0.5, 1], "gamma": ["scale"]},
}


@pytest.fixture(scope="module")
def models_module():
    import models
    return models


def _frame(features, y, seed=0):
    """Normalised-looking cohort: canonical categories, one informative numeric feature."""
    rng = np.random.default_rng(seed)
    n = len(y)
    df = pd.DataFrame(index=range(n))
    for i, f in enumerate(features):
        if f in GLOBAL_VOCABULARY:
            vocab = sorted(GLOBAL_VOCABULARY[f])
            df[f] = [vocab[(j + i) % len(vocab)] for j in range(n)]
        else:
            df[f] = rng.normal(size=n)
    df[features[0]] = np.asarray(y, dtype=float) + rng.normal(0, 0.3, n)
    return df


def _labels(n=150, classes=(-1, 0, 1), seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(np.resize(np.asarray(classes), n)))


def _write_split(split_dir: Path, target: str, y) -> None:
    tr, te = rf.redraw_split(y)
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{target}_split_indices.json").write_text(json.dumps(
        {"target": target, "train_idx": tr.tolist(), "test_idx": te.tolist(), "features": []}))


# ================================================================ config
#: SHA-256 of config.yaml recorded (``config_file_sha256``) by every manifest of the real run
#: rerun_2026-10-XX_v2, and of the release file (comment lines changed only)
RUN_V2_CONFIG_SHA256 = "771fef24e01ef5fa49dc5d43cb7677bb91c242625c9a12d699db1b465f987f4a"
RELEASE_CONFIG_SHA256 = "49053732523b5105b57ef4951e3bfe91ccadeee846458462852f7d11dd27b30c"
#: parsed configuration recorded by every manifest of the real run (``config_used_sha256``)
RUN_V2_CONFIG_USED_SHA256 = "e859f3e226de006d2609a08540d91e65e98e33993c404992d64728242ba6fda1"


def test_config_yaml_is_the_config_of_the_recorded_run():
    """config.yaml is the release file mapped from the recorded one, and parses to the recorded configuration."""
    import hashlib
    import revision_runners as rr
    sha = hashlib.sha256((REPO_DIR / "config.yaml").read_bytes()).hexdigest()
    assert sha == RELEASE_CONFIG_SHA256 == rr.release_file_sha256(RUN_V2_CONFIG_SHA256)
    config, file_sha = rr.read_config_file()
    assert file_sha == sha and rr._json_sha256(config) == RUN_V2_CONFIG_USED_SHA256
    manifests = sorted((REPO_DIR / "rerun_2026-10-XX_v2").glob("*/run_manifest.json"))
    if not manifests:
        pytest.skip("rerun_2026-10-XX_v2 not present in this checkout (pinned hashes checked above)")
    recorded = {m.parent.name: json.loads(m.read_text()) for m in manifests}
    assert {m["config_file_sha256"] for m in recorded.values()} == {RUN_V2_CONFIG_SHA256}
    assert {m["config_used_sha256"] for m in recorded.values()} == {RUN_V2_CONFIG_USED_SHA256}


def test_lock_file_is_the_recorded_lock_with_other_comments():
    """The lock file maps from the recorded one; its pins are those the recorded run checked."""
    import hashlib
    import revision_runners as rr
    lock = REPO_DIR / "requirements-revision-2026-10.lock"
    sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert sha == rr.release_file_sha256("53cbbdfeaea1e91ee42063addefbd6b37da07e56b37557e7a73264ac226943d0")
    assert rr._json_sha256(rr.lock_pins(lock)) == \
        "12dd5591fff14436b2574b27e722d91cadbee6eb68eca62b27bf5e05ac1545f8"
    assert set(rr.RELEASE_FILE_SHA256) == {RUN_V2_CONFIG_SHA256,
                                           "53cbbdfeaea1e91ee42063addefbd6b37da07e56b37557e7a73264ac226943d0"}
    assert rr.release_file_sha256("0" * 64) == "0" * 64          # any other hash is left unchanged


def test_config_yaml_folds_corrected_and_hybrids_removed():
    cfg = yaml.safe_load((REPO_DIR / "config.yaml").read_text(encoding="utf-8"))
    rf.check_cv_config(cfg)
    assert cfg["cross_validation"] == {"folds": 10, "shuffle": True}
    # hybrids: E10 heads = code constant rr.PAPER_HYBRID_HEADS (no config section, so that config.yaml
    # stays the run config), never a models.py family; stacking removed
    assert "hybrid" not in cfg
    assert "hybrid" not in cfg["models"] and "stacking" not in cfg["models"]
    reg = cfg["data"]["datasets"]
    assert reg["ASHRAE_2022"]["split_dir"] == "splits/ASHRAE_2022"
    assert reg["ASHRAE_2022"]["folds_dir"] == "splits"


@pytest.mark.parametrize("bad", [
    {"seed": 42, "cross_validation": {"folds": 5, "shuffle": True}},     # the snapshot value
    {"seed": 42, "cross_validation": {"folds": 10, "shuffle": False}},
    {"seed": 43, "cross_validation": {"folds": 10, "shuffle": True}},
])
def test_check_cv_config_rejects_other_settings(bad):
    with pytest.raises(rf.FoldIntegrityError):
        rf.check_cv_config(bad)


# ================================================================ archived split
def test_load_archived_split_accepts_the_redraw(tmp_path):
    y = _labels()
    _write_split(tmp_path, "TSV_3p", y)
    tr, te = rf.load_archived_split(tmp_path, "TSV_3p", y)
    tr2, te2 = rf.redraw_split(y)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2) and len(te) == 30


@pytest.mark.parametrize("tamper", ["reorder", "swap", "target", "drop"])
def test_load_archived_split_rejects_a_tampered_file(tmp_path, tamper):
    y = _labels()
    _write_split(tmp_path, "TSV_3p", y)
    p = tmp_path / "TSV_3p_split_indices.json"
    d = json.loads(p.read_text())
    if tamper == "reorder":                       # same sets, different order
        d["train_idx"] = sorted(d["train_idx"])
    elif tamper == "swap":                        # a train row moved to test and back
        d["train_idx"][0], d["test_idx"][0] = d["test_idx"][0], d["train_idx"][0]
    elif tamper == "target":
        d["target"] = "thermal_sensation"
    else:
        d["test_idx"] = d["test_idx"][:-1]
    p.write_text(json.dumps(d))
    with pytest.raises(rf.FoldIntegrityError):
        rf.load_archived_split(tmp_path, "TSV_3p", y)


@pytest.mark.parametrize("target", TARGETS)
def test_archived_ashrae_2022_splits_load_and_folds_materialise(tmp_path, target):
    """Read-only: the archived split equals the re-draw; folds go to tmp_path, never to splits/."""
    repo_folds = REPO_DIR / "splits" / f"{target}_folds.json"
    before = repo_folds.read_bytes() if repo_folds.exists() else None   # an E1 run may have written it
    y_all = pd.read_csv(REPO_DIR / "Data" / "ASHRAE_2022_Clean_api.csv", usecols=[target])[target]
    tr, te = rf.load_archived_split(REPO_DIR / "splits" / "ASHRAE_2022", target, y_all)
    assert (len(tr), len(te)) == (12179, 3045)
    folds = rf.ensure_folds(tmp_path, "ASHRAE_2022", target, tr, y_all.iloc[tr])
    counts = np.bincount(folds.fold_id)
    assert len(counts) == 10 and counts.min() >= 1217 and counts.max() <= 1218
    assert (repo_folds.read_bytes() if repo_folds.exists() else None) == before


# ================================================================ folds
def test_fold_ids_are_the_frozen_stratified_kfold():
    y = _labels(n=120)
    fold_id = rf.materialise_fold_ids(y)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for f, ((tr, va), (tr2, va2)) in enumerate(zip(skf.split(np.zeros(120), y), rf.cv_splits_from_fold_ids(fold_id))):
        assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
        assert sorted(np.bincount(y.iloc[va] + 1).tolist()) == [4, 4, 4]      # stratified
    assert np.array_equal(rf.fold_ids_from_splits(rf.cv_splits_from_fold_ids(fold_id), 120), fold_id)


def test_fold_ids_from_splits_rejects_a_non_partition():
    splits = rf.cv_splits_from_fold_ids(rf.materialise_fold_ids(_labels(n=60), k=3))
    tr0, va0 = splits[0]
    bad = [(tr0, va0[1:])] + splits[1:]                       # one row in no validation fold
    with pytest.raises(rf.FoldIntegrityError):
        rf.fold_ids_from_splits(bad, 60)
    with pytest.raises(rf.FoldIntegrityError):
        rf.fold_ids_from_splits([(tr0[1:], va0)] + splits[1:], 60)   # train ≠ complement


def test_ensure_folds_writes_once_and_rechecks(tmp_path):
    y = _labels(n=150)
    tr, _ = rf.redraw_split(y)
    a = rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    p = rf.folds_path(tmp_path, "TSV_3p")
    raw = p.read_bytes()
    stored = json.loads(raw)
    assert stored["k"] == 10 and stored["seed"] == 42 and stored["train_idx"] == tr.tolist()
    assert stored["fold_vector_sha256"] == a.sha256 == rf.fold_vector_sha256(stored["fold_id"])
    b = rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    assert b.sha256 == a.sha256 and p.read_bytes() == raw              # loaded, not rewritten
    with pytest.raises(rf.FoldIntegrityError, match="dataset"):
        rf.ensure_folds(tmp_path, "ASHRAE_2018_v2", "TSV_3p", tr, y.iloc[tr])
    with pytest.raises(rf.FoldIntegrityError, match="train_idx"):
        rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr[::-1], y.iloc[tr[::-1]])
    # a coherent but different fold vector (hash updated) is still refused
    stored["fold_id"] = np.roll(stored["fold_id"], 1).tolist()
    stored["fold_vector_sha256"] = rf.fold_vector_sha256(stored["fold_id"])
    p.write_text(json.dumps(stored))
    with pytest.raises(rf.FoldIntegrityError, match="materialisation"):
        rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    stored["fold_vector_sha256"] = "0" * 64
    p.write_text(json.dumps(stored))
    with pytest.raises(rf.FoldIntegrityError, match="recorded fold_vector_sha256"):
        rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    for bad in ("0" * 64, None):                                      # corrupt or missing train_idx hash
        good = json.loads(raw)
        if bad is None:
            del good["train_idx_sha256"]
        else:
            good["train_idx_sha256"] = bad
        p.write_text(json.dumps(good))
        with pytest.raises(rf.FoldIntegrityError, match="recorded train_idx_sha256"):
            rf.ensure_folds(tmp_path, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])


@pytest.mark.parametrize("sub", ["", "phase2_indomain", "a/../folds"])
def test_ensure_folds_refuses_the_frozen_artefact_before_any_write(monkeypatch, tmp_path, sub):
    import config_loader
    protected = tmp_path / "rerun_2026-04-28_no_reweighting"
    monkeypatch.setattr(config_loader, "PROTECTED_ARTEFACT_DIR", protected)
    y = _labels(n=150)
    tr, _ = rf.redraw_split(y)
    with pytest.raises(ValueError, match="frozen artefact"):
        rf.ensure_folds(protected / sub, "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    assert not protected.exists()


def test_models_cli_refuses_folds_in_the_frozen_artefact_before_loading_data(models_module, monkeypatch, tmp_path):
    monkeypatch.setattr(models_module, "UnifiedDataLoader",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    monkeypatch.chdir(REPO_DIR)
    monkeypatch.setattr(sys, "argv", ["models.py", "--models", "classical", "--output_dir", str(tmp_path / "out"),
                                      "--folds_from", "rerun_2026-04-28_no_reweighting/folds"])
    with pytest.raises(ValueError, match="frozen artefact"):
        models_module.main(unmanifested_test_run=True)
    assert not (REPO_DIR / "rerun_2026-04-28_no_reweighting" / "folds").exists()


# ================================================================ seeds
@pytest.mark.parametrize("model, seeds, ok", [
    ("RandomForest", (42, 43, 44), True),
    ("XGBoost", (42, 43, 44), True),
    ("RandomForest", (42,), False),
    ("XGBoost", (1, 2, 3), False),
    ("XGBoost", (44, 43, 42), False),
    ("SVM", (42,), True),
    ("SVM", (42, 43, 44), False),
    ("ANN", (42,), True),
    ("ANN", (42, 43, 44), True),
    ("FTTransformer", (42, 43), False),
    ("FTTransformer", (43,), False),
])
def test_check_model_seeds(model, seeds, ok):
    if ok:
        assert rf.check_model_seeds(model, seeds) == seeds
    else:
        with pytest.raises(ValueError, match="frozen seeds"):
            rf.check_model_seeds(model, seeds)


@pytest.mark.parametrize("extra, exc, match", [
    (["--seeds", "1", "2", "3"], ValueError, "frozen seeds"),
    (["--seeds", "42"], ValueError, "frozen seeds"),
    (["--cv_folds", "5"], ValueError, "K = 10"),
])
def test_models_cli_refuses_unfrozen_seeds_or_folds_before_loading_data(models_module, monkeypatch, tmp_path,
                                                                       extra, exc, match):
    monkeypatch.setattr(models_module, "UnifiedDataLoader",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    monkeypatch.setattr(sys, "argv", ["models.py", "--models", "classical",
                                      "--output_dir", str(tmp_path / "out"), *extra])
    with pytest.raises(exc, match=match):
        models_module.main(unmanifested_test_run=True)


def test_models_cli_refuses_the_revision_cohort_without_a_manifest(models_module, monkeypatch, tmp_path):
    """E1 runs through run_revision.py e1 (run manifest); models.py
    --dataset ASHRAE_2022 (its default) is refused before any data is loaded or anything written."""
    from models_deep import RevisionNotImplementedError
    monkeypatch.setattr(models_module, "UnifiedDataLoader",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    for extra in ([], ["--dataset", "ASHRAE_2022"]):
        monkeypatch.setattr(sys, "argv", ["models.py", "--models", "classical",
                                          "--output_dir", str(tmp_path / "out"), *extra])
        with pytest.raises(RevisionNotImplementedError, match="run_revision.py e1"):
            models_module.main()
    assert not (tmp_path / "out").exists()


# ================================================================ runner: blocking tests
def _deep_spy(models_module, seen):
    """Spy for select_best: records the folds and seed it receives (keeps the runner test
    fast; real ANN/FTT fits are exercised in test_early_stopping.py)."""
    def spy(target, model_name, est, pgrid, X_tr, y_tr, k=5, cv_splits=None, **kw):
        seen.append((model_name, est.random_state, [(np.asarray(a).copy(), np.asarray(b).copy())
                                                    for a, b in cv_splits]))
        majority = y_tr.mode().iloc[0]
        model = SimpleNamespace(predict=lambda X: np.full(len(X), majority),
                                history_={"epoch": [1], "train_loss": [0.0], "val_loss": [0.0], "val_f1": [0.0]})
        rows = [{"fold": i + 1, "experiment": target, "model": model_name, "val_accuracy": 0.0,
                 "folds_sha256": rf.splits_sha256(cv_splits, len(X_tr))} for i in range(len(cv_splits))]
        stats = pd.DataFrame([{"experiment": target, "model": model_name, "metric": "val_accuracy"}])
        return model, {}, pd.DataFrame(rows), stats, []
    return spy


def _patch_runner(models_module, monkeypatch, deep_seen, grid_seen, compute_cfg):
    import models_classic
    from sklearn.model_selection import GridSearchCV

    class Recording(GridSearchCV):
        def fit(self, X, y=None, **kw):
            grid_seen.append([(np.asarray(a).copy(), np.asarray(b).copy()) for a, b in self.cv])
            return super().fit(X, y, **kw)

    monkeypatch.setattr(models_classic, "GridSearchCV", Recording)
    monkeypatch.setitem(models_module.CONFIG, "classical_hparams", TINY_CLASSICAL)
    monkeypatch.setitem(models_module.CONFIG, "compute", compute_cfg)   # explicit CPU without a GPU
    monkeypatch.setattr(models_module, "select_best", _deep_spy(models_module, deep_seen))
    monkeypatch.setattr(models_module, "early_stopping_record", lambda model: {})
    monkeypatch.setattr(models_module, "plot_confusion_matrix", lambda *a, **k: None)


def _same_splits(a, b):
    return len(a) == len(b) and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                                    for x, y in zip(a, b))


def test_runner_shares_folds_and_honours_seeds_across_families(models_module, monkeypatch, tmp_path,
                                                                machine_compute, xgb_fits):
    feats = list(models_module.FEATURES_ALL)
    y = _labels(n=150)
    X = _frame(feats, y)
    tr, te = rf.redraw_split(y)
    folds = rf.ensure_folds(tmp_path / "folds", "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    from category_normalisation import split_feature_types, vocabulary_for
    num, cat = split_feature_types(X_train, X_train.columns)

    deep_seen, grid_seen = [], []
    _patch_runner(models_module, monkeypatch, deep_seen, grid_seen, machine_compute)
    args = SimpleNamespace(models=["all"], classical_models=["all"], deep_models=["all"],
                           skip_feature_importance=True)
    seeds = rf.model_seeds(ann_seeds=(42, 43, 44), ftt_seeds=(42,))
    best, fold_summaries, _, tests = models_module.run_experiment_for_target(
        "TSV_3p", X_train, y_train, X_test, y_test, num, cat, tmp_path / "out", args=args,
        cat_vocabulary=vocabulary_for(cat), cv_splits=folds.splits(), seeds_by_model=seeds)

    # --- identical fold indices across families (blocking)
    expected = folds.splits()
    assert len(grid_seen) == 3 + 3 + 1                                # RF ×3, XGB ×3, SVM ×1
    assert all(_same_splits(g, expected) for g in grid_seen)
    assert [(m, s) for m, s, _ in deep_seen] == [("ANN", 42), ("ANN", 43), ("ANN", 44), ("FTTransformer", 42)]
    assert all(_same_splits(d, expected) for *_, d in deep_seen)
    summary = pd.read_csv(tmp_path / "out" / "TSV_3p_ALLMODELS_fold_summary.csv")
    assert set(summary["folds_sha256"]) == {folds.sha256}
    assert set(summary.groupby(["model", "seed"]).size()) == {10}

    # --- seeds honoured (blocking)
    assert sorted(best["rf"]) == sorted(best["xgb"]) == [42, 43, 44] and sorted(best["svm"]) == [42]
    for s in (42, 43, 44):
        assert best["rf"][s].named_steps["clf"].random_state == s
        assert best["xgb"][s].named_steps["clf"].estimator_.get_params()["random_state"] == s
        for m in ("RandomForest", "XGBoost"):
            assert (tmp_path / "out" / m / f"seed_{s}" / f"TSV_3p__{m}__seed{s}_test_results.json").exists()
    scores = pd.DataFrame(tests)
    assert sorted(zip(scores["model"], scores["seed"])) == sorted(
        [("RandomForest", s) for s in (42, 43, 44)] + [("XGBoost", s) for s in (42, 43, 44)]
        + [("SVM", 42)] + [("ANN", s) for s in (42, 43, 44)] + [("FTTransformer", 42)])
    # seeds change the fitted RF (otherwise "3 seeds" would be one model three times)
    p = [best["rf"][s].predict_proba(X_test) for s in (42, 43)]
    assert not np.array_equal(p[0], p[1])

    # --- hybrids and the fold-number Wilcoxon are absent (blocking)
    names = {q.name for q in (tmp_path / "out").rglob("*")}
    assert not any("FTemb" in n or "as_features" in n or "wilcoxon" in n.lower() for n in names)
    assert not any("FTemb" in str(m) for m in scores["model"])


def test_runner_refuses_missing_or_divergent_folds(models_module, monkeypatch, tmp_path, machine_compute):
    feats = list(models_module.FEATURES_ALL)
    y = _labels(n=150)
    X = _frame(feats, y)
    tr, te = rf.redraw_split(y)
    folds = rf.ensure_folds(tmp_path / "folds", "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    from category_normalisation import split_feature_types, vocabulary_for
    num, cat = split_feature_types(X.iloc[tr], X.columns)
    args = SimpleNamespace(models=["classical"], classical_models=["RandomForest"], deep_models=["all"],
                           skip_feature_importance=True)
    with pytest.raises(ValueError, match="cv_splits"):
        models_module.run_experiment_for_target("TSV_3p", X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                                                num, cat, tmp_path / "a", args=args,
                                                cat_vocabulary=vocabulary_for(cat))
    # a family that used other folds makes the run fail
    deep_seen, grid_seen = [], []
    _patch_runner(models_module, monkeypatch, deep_seen, grid_seen, machine_compute)
    other = rf.cv_splits_from_fold_ids(rf.materialise_fold_ids(y.iloc[tr], seed=43))
    spy = models_module.select_best

    def divergent(*a, cv_splits=None, **k):
        return spy(*a, cv_splits=other, **k)
    monkeypatch.setattr(models_module, "select_best", divergent)
    args = SimpleNamespace(models=["classical", "deep"], classical_models=["RandomForest"],
                           deep_models=["ANN"], skip_feature_importance=True)
    with pytest.raises(rf.FoldIntegrityError, match="ANN__seed42: family used different folds"):
        models_module.run_experiment_for_target("TSV_3p", X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                                                num, cat, tmp_path / "b", args=args,
                                                cat_vocabulary=vocabulary_for(cat), cv_splits=folds.splits(),
                                                seeds_by_model=rf.model_seeds())


def test_grid_search_requires_the_materialised_folds(models_module):
    import models_classic
    X = pd.DataFrame({"x": np.arange(20.0)})
    with pytest.raises(ValueError, match="cv_splits"):
        models_classic.run_grid_search_on_train("TSV_3p", "RandomForest", None, {}, X, pd.Series([0, 1] * 10))


# ================================================================ select_best (deep) folds
class _SpyEst:
    """Minimal estimator: records the training rows of every fit."""
    fits: list = []

    def __init__(self, lr=1e-3):
        self.lr = lr

    def fit(self, X, y):
        _SpyEst.fits.append(np.asarray(X.index))
        self.c_ = pd.Series(y).mode().iloc[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.c_)


def test_select_best_uses_the_materialised_folds_for_search_and_report():
    """The deep selection uses the shared folds for both loops (no RANDOM_SEED+1 re-draw);
    only the fold plumbing is checked here (early stopping: test_early_stopping.py)."""
    import models_deep as md
    y = _labels(n=60)
    X = pd.DataFrame({"x": np.arange(60.0)})
    fold_id = rf.materialise_fold_ids(y, k=3)
    splits = rf.cv_splits_from_fold_ids(fold_id)
    _SpyEst.fits = []
    _, _, summary, *_ = md.select_best("TSV_3p", "ANN", _SpyEst(), {"lr": [1e-3]}, X, y,
                                       cv_splits=splits)
    # fits: 3 search folds (= the reported folds of a one-point grid), final model (all rows; no
    # auxiliary curves model)
    assert len(_SpyEst.fits) == 4
    train_parts = [tr for tr, _ in splits]
    assert all(np.array_equal(a, b) for a, b in zip(_SpyEst.fits[:3], train_parts))
    assert np.array_equal(_SpyEst.fits[3], X.index)
    # a two-point grid refits the reported folds on the same shared folds
    _SpyEst.fits = []
    md.select_best("TSV_3p", "ANN", _SpyEst(), {"lr": [1e-3, 1e-2]}, X, y, cv_splits=splits)
    assert len(_SpyEst.fits) == 2 * 3 + 1 + 3
    assert all(np.array_equal(a, b) for a, b in zip(_SpyEst.fits[7:], train_parts))
    assert len(summary) == 3 and set(summary["folds_sha256"]) == {rf.fold_vector_sha256(fold_id)}


# ================================================================ hybrids absent
def _revision_python_files():
    return sorted(set(REPO_DIR.glob("*.py")) | set((REPO_DIR / "analysis").glob("*.py"))
                  | set((REPO_DIR / "tests").glob("*.py")))


def test_hybrid_module_is_restored_and_imported_only_by_e10():
    """One copy of the hybrid code (repository root), used by the E10 runner only."""
    assert (REPO_DIR / "hybrid.py").exists() and not (REPO_DIR / "run_hybrid_only.py").exists()
    assert not (REPO_DIR / "archive" / "hybrid.py").exists()
    assert not (REPO_DIR / "archive" / "run_hybrid_only.py").exists()
    importers = set()
    for path in _revision_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom) else [])
            assert not any(m.split(".")[0] == "run_hybrid_only" for m in mods), path.name
            if any(m.split(".")[0] == "hybrid" for m in mods):
                importers.add(path.name)
    assert {p for p in importers if not p.startswith("test_")} == {"revision_runners.py"}


def test_fold_number_wilcoxon_is_removed():
    """No Wilcoxon over folds anywhere in the revision path."""
    import stats_utils
    assert not hasattr(stats_utils, "pairwise_wilcoxon")
    for name in ("models.py", "models_classic.py", "models_deep.py", "stats_utils.py",
                 "revision_stats.py", "revision_folds.py"):
        tree = ast.parse((REPO_DIR / name).read_text(encoding="utf-8"))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                names.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name for a in n.names)
            elif isinstance(n, ast.Call):
                names.add(getattr(n.func, "id", getattr(n.func, "attr", "")))
        assert not any("wilcoxon" in x.lower() for x in names), name


@pytest.mark.parametrize("argv", [
    ["--models", "hybrid"],
    ["--hybrid_heads", "RF"],
    ["--skip_embeddings"],
    ["--skip_visualizations"],
])
def test_models_cli_no_longer_offers_hybrids(models_module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["models.py", *argv])
    with pytest.raises(SystemExit):
        models_module.parse_args()


# ================================================================ end-to-end CLI on synthetic data
class _FakeLoader:
    frame: pd.DataFrame = None

    def __init__(self, *a, **k):
        self.category_coverage = {"ASHRAE_2022": []}

    def load_and_validate(self, path, name, category_normalisation=True):
        assert category_normalisation is True
        return _FakeLoader.frame.copy()

    def save_statistics(self, path):
        pass


def test_models_main_end_to_end_classical(models_module, monkeypatch, tmp_path, machine_compute, xgb_fits):
    feats = list(models_module.FEATURES_ALL)
    y = _labels(n=150)
    df = _frame(feats, y)
    for t in TARGETS:
        df[t] = y
    _FakeLoader.frame = df
    _write_split(tmp_path / "split", "TSV_3p", y)
    monkeypatch.setattr(models_module, "UnifiedDataLoader", _FakeLoader)
    monkeypatch.setitem(models_module.CONFIG, "classical_hparams", TINY_CLASSICAL)
    monkeypatch.setitem(models_module.CONFIG, "compute", machine_compute)
    monkeypatch.setattr(models_module, "plot_confusion_matrix", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    argv = ["models.py", "--models", "classical", "--targets", "TSV_3p", "--skip_feature_importance",
            "--split_dir", str(tmp_path / "split"), "--folds_from", str(tmp_path / "folds"),
            "--seeds", "42", "43", "44"]
    monkeypatch.setattr(sys, "argv", argv + ["--output_dir", str(tmp_path / "out")])
    models_module.main(unmanifested_test_run=True)

    folds = json.loads((tmp_path / "folds" / "TSV_3p_folds.json").read_text())
    out = tmp_path / "out"
    assert json.loads((out / "TSV_3p" / "splits" / "TSV_3p_folds.json").read_text()) == folds
    summary = pd.read_csv(out / "TSV_3p" / "TSV_3p_ALLMODELS_fold_summary.csv")
    assert set(summary["folds_sha256"]) == {folds["fold_vector_sha256"]}
    import joblib
    for s in (42, 43, 44):
        m = joblib.load(out / f"best_rf_by_target_seed{s}.joblib")["TSV_3p"]
        assert m.named_steps["clf"].random_state == s
    assert joblib.load(out / "best_rf_by_target.joblib")["TSV_3p"].named_steps["clf"].random_state == 42
    assert not (out / "best_svm_by_target_seed43.joblib").exists()
    comp = pd.read_csv(out / "comprehensive_experiment_comparison.csv")
    assert sorted(comp["seed"][comp["model"] == "XGBoost"]) == [42, 43, 44]
    assert not list(out.rglob("*wilcoxon*")) and not list(out.rglob("*FTemb*"))

    # second run: the folds file is re-used (unchanged), a tampered one stops the run
    raw = (tmp_path / "folds" / "TSV_3p_folds.json").read_bytes()
    monkeypatch.setattr(sys, "argv", argv + ["--output_dir", str(tmp_path / "out2"),
                                             "--classical_models", "RandomForest"])
    models_module.main(unmanifested_test_run=True)
    assert (tmp_path / "folds" / "TSV_3p_folds.json").read_bytes() == raw
    folds["fold_id"] = folds["fold_id"][::-1]
    folds["fold_vector_sha256"] = rf.fold_vector_sha256(folds["fold_id"])
    (tmp_path / "folds" / "TSV_3p_folds.json").write_text(json.dumps(folds))
    monkeypatch.setattr(sys, "argv", argv + ["--output_dir", str(tmp_path / "out3")])
    with pytest.raises(rf.FoldIntegrityError):
        models_module.main(unmanifested_test_run=True)
