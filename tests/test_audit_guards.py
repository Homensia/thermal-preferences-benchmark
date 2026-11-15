# -*- coding: utf-8 -*-
"""
Guards added during the revision of the code.

* Early stopping: the interim fail-closed guard on ANN/FTT training was
  lifted by the early-stopping fix (tests in ``test_early_stopping.py``);
  only the deep-stage selection rule is kept here.
* Persisted state after the category normalisation fix: archived FTT
  preprocessors and pre-revision one-hot pipelines are rejected with an
  actionable error before evaluation.
* Frozen artefact: no CLI writes inside ``rerun_2026-04-28_no_reweighting/``.
* XGB objective left to XGBoost: identical fits for >= 3 classes, a binary
  fit (instead of a crash) for 2 classes.

Synthetic data only, each test well under 60 s on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config_loader  # noqa: E402
from category_normalisation import (  # noqa: E402
    GLOBAL_VOCABULARY, categorical_features, vocabulary_for,
)

PROTECTED = REPO_DIR / "rerun_2026-04-28_no_reweighting"


@pytest.fixture(scope="module")
def md():
    import models_deep
    return models_deep


@pytest.fixture(scope="module")
def models_module():
    import models
    return models


def _frame(features, n=60, seed=0):
    """Normalised-looking cohort: every categorical value from the global vocabulary."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for f in features:
        if f in GLOBAL_VOCABULARY:
            vocab = sorted(GLOBAL_VOCABULARY[f])
            df[f] = [vocab[i % len(vocab)] for i in range(n)]
        else:
            df[f] = rng.normal(size=n)
    return df


# ---------------------------------------------------------------- deep-stage selection
@pytest.mark.parametrize("models, classical, expected", [
    (["all"], ["all"], True),                       # default CLI: deep stage selected
    (["deep"], ["all"], True),
    (["classical", "deep"], ["all"], True),
    (["classical"], ["all"], False),
    (["all"], ["RandomForest"], False),              # snapshot rule: a classical filter skips deep
])
def test_deep_models_requested(models_module, models, classical, expected):
    args = SimpleNamespace(models=models, classical_models=classical)
    assert models_module.deep_models_requested(args) is expected


# ---------------------------------------------------------------- frozen artefact
@pytest.mark.parametrize("path", [PROTECTED, PROTECTED / "phase3", PROTECTED / "a" / ".." / "b"])
def test_check_not_protected_rejects_the_frozen_artefact(path):
    with pytest.raises(ValueError, match="frozen artefact"):
        config_loader.check_not_protected(path)


def test_check_not_protected_rejects_the_artefact_of_another_checkout(tmp_path, monkeypatch):
    """A run started from a worktree must not write into the development
    checkout's frozen artefact (or a copy): any directory of that name is refused."""
    other = tmp_path / "thermal-preferences-benchmark" / "rerun_2026-04-28_no_reweighting"
    for path in (other, other / "E9_nn", other / "a" / ".." / "b"):
        with pytest.raises(ValueError, match="frozen artefact"):
            config_loader.check_not_protected(path)
    monkeypatch.syspath_prepend(str(REPO_DIR / "analysis"))
    import nn_cases
    with pytest.raises(ValueError, match="frozen artefact"):
        nn_cases.run(Path("unused.csv"), other / "E9b", "features_12", ["TSV_3p"], mode="exploratory",
                     config={"data": {"features_12": ["Tair"]}})
    assert not other.exists()


def test_check_not_protected_accepts_other_paths(tmp_path):
    config_loader.check_not_protected(tmp_path / "out")
    config_loader.check_not_protected(REPO_DIR / "rerun_2026-04-28_no_reweighting_copy")


def test_clis_refuse_to_write_into_the_frozen_artefact(models_module, monkeypatch):
    import transfert
    import indicateurs_classiques as ic
    monkeypatch.chdir(REPO_DIR)
    target = "rerun_2026-04-28_no_reweighting/phase3"                # relative, as in the README
    monkeypatch.setattr(transfert, "load_external_cohort",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    monkeypatch.setattr(sys, "argv", ["transfert.py", "--splits", "splits/phase3",
                                      "--models_dir", "rerun_2026-04-28_no_reweighting/phase2_indomain",
                                      "--output_dir", target])
    with pytest.raises(ValueError, match="frozen artefact"):
        transfert.main()
    monkeypatch.setattr(sys, "argv", ["models.py", "--models", "classical", "--output_dir",
                                      "rerun_2026-04-28_no_reweighting/phase2_indomain"])
    with pytest.raises(ValueError, match="frozen artefact"):
        models_module.main()
    with pytest.raises(ValueError, match="frozen artefact"):
        ic.main(["--pmv_only", "rerun_2026-04-28_no_reweighting/phase2_baselines"])


# ---------------------------------------------------------------- persisted model schema (category normalisation)
def _legacy_preprocessor(md):
    """Attribute set of the archived pre-revision TabPreprocessor."""
    pp = object.__new__(md.TabPreprocessor)
    pp.cat_cols, pp.num_cols = ["Sex"], ["Tair"]
    pp.encoders, pp.n_categories, pp.scaler = {}, [2], None
    return pp


def _tiny_ftt(md, models_module, **kw):
    feats = list(models_module.FEATURES_ALL)
    cat = categorical_features(feats)
    num = [f for f in feats if f not in cat]
    params = dict(num_cols=num, cat_cols=cat, cat_vocabulary=vocabulary_for(cat),
                  dim=8, depth=1, heads=2, dim_head=4, epochs=1, device="cpu")
    params.update(kw)
    X = _frame(feats, n=120)                 # >= 100 rows: internal validation subset
    return md.FTClassifier(**params).fit(X, [0, 1, 2] * 40), X


def test_legacy_preprocessor_transform_raises_actionable_error(md):
    with pytest.raises(md.LegacyModelStateError, match="pre-revision code that fitted them"):
        _legacy_preprocessor(md).transform(pd.DataFrame({"Sex": ["male"], "Tair": [20.0]}))


def test_ftt_schema_check(md, models_module):
    vocab = vocabulary_for(categorical_features(models_module.FEATURES_ALL))
    ftt, X = _tiny_ftt(md, models_module)
    md.check_ftt_revision_schema(ftt, vocab, "ok")
    with pytest.raises(md.LegacyModelStateError, match="vocabulary_"):
        md.check_ftt_revision_schema(SimpleNamespace(pp_=_legacy_preprocessor(md)), vocab, "old")
    with pytest.raises(md.LegacyModelStateError, match="pp_"):
        md.check_ftt_revision_schema(SimpleNamespace(), vocab, "none")
    ftt.pp_.oov_policy = "global_code"
    with pytest.raises(md.LegacyModelStateError, match="reserved_index"):
        md.check_ftt_revision_schema(ftt, vocab, "ann-policy")
    derived, _ = _tiny_ftt(md, models_module, cat_vocabulary=None)   # vocabulary from training rows
    with pytest.raises(md.LegacyModelStateError, match="global vocabulary"):
        md.check_ftt_revision_schema(derived, vocab, "derived")


def test_onehot_schema_check(models_module):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from models_classic import check_onehot_revision_schema, make_preprocess
    feats = list(models_module.FEATURES_ALL)
    cat = categorical_features(feats)
    num = [f for f in feats if f not in cat]
    vocab = vocabulary_for(cat)
    X, y = _frame(feats, n=30), [0, 1, 2] * 10

    def pipe(v):
        return Pipeline([("pp", make_preprocess(num, cat, cat_vocabulary=v)),
                         ("clf", RandomForestClassifier(n_estimators=2, random_state=0))]).fit(X, y)
    check_onehot_revision_schema(pipe(vocab), vocab, "revision")
    with pytest.raises(ValueError, match="pre-revision code that fitted them"):
        check_onehot_revision_schema(pipe(None), vocab, "snapshot")      # unknown -> all-zero


def test_load_model_dicts_rejects_legacy_and_accepts_revision_models(md, models_module, tmp_path):
    import joblib
    import transfert
    ftt, X = _tiny_ftt(md, models_module)
    good = tmp_path / "good"
    good.mkdir()
    joblib.dump({"TSV_3p": ftt}, good / "best_ft_by_target.joblib")
    ft_by_t, rf_by_t, xgb_by_t = transfert.load_model_dicts(good)
    assert len(ft_by_t["TSV_3p"].predict(X)) == len(X) and not rf_by_t and not xgb_by_t
    bad = tmp_path / "bad"
    bad.mkdir()
    joblib.dump({"TSV_3p": SimpleNamespace(pp_=_legacy_preprocessor(md))},
                bad / "best_ft_by_target.joblib")
    with pytest.raises(md.LegacyModelStateError, match="best_ft_by_target.joblib:TSV_3p"):
        transfert.load_model_dicts(bad)


ARCHIVED_FT = PROTECTED / "phase2_indomain" / "best_ft_by_target.joblib"


@pytest.mark.skipif(not ARCHIVED_FT.is_file(), reason="archived FTT models absent")
def test_archived_ftt_models_are_rejected_before_evaluation(md, tmp_path):
    """Archived preprocessors lack vocabulary_; read-only, no prediction computed."""
    import torch
    import transfert
    (tmp_path / "best_ft_by_target.joblib").symlink_to(ARCHIVED_FT)
    if not torch.cuda.is_available():
        # the archived FTT holds CUDA tensors: without a GPU it cannot even be unpickled
        # (pre-existing; still rejected before any evaluation)
        with pytest.raises(RuntimeError, match="deserialize object on a CUDA device"):
            transfert.load_model_dicts(tmp_path)
        return
    with pytest.raises(md.LegacyModelStateError, match="pre-revision code that fitted them"):
        transfert.load_model_dicts(tmp_path)


# ---------------------------------------------------------------- XGB objective
def test_xgb_proto_objective_default(models_module, machine_compute, xgb_fits):
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    import transfert
    from models_classic import LabelEncodedClassifier, make_preprocess
    feats = list(models_module.FEATURES_ALL)
    cat = categorical_features(feats)
    num = [f for f in feats if f not in cat]
    vocab = vocabulary_for(cat)
    X = _frame(feats, n=60)
    _, xgb = transfert.make_rf_xgb_protos(num, cat, cat_vocabulary=vocab, compute=machine_compute)
    xgb.set_params(clf__estimator__n_estimators=5)
    y3 = np.array([-1.0, 0.0, 1.0] * 20)
    p3 = xgb.fit(X, y3).predict_proba(X)
    assert xgb.named_steps["clf"].estimator_.get_xgb_params()["objective"] == "multi:softprob"
    explicit = Pipeline([("pp", make_preprocess(num, cat, cat_vocabulary=vocab)),
                         ("clf", LabelEncodedClassifier(XGBClassifier(
                             **{**xgb.named_steps["clf"].estimator.get_params(),
                                "objective": "multi:softprob"})))])
    np.testing.assert_array_equal(explicit.fit(X, y3).predict_proba(X), p3)   # identical for >= 3
    y2 = np.array([0.0, 1.0] * 30)
    xgb.fit(X, y2)                                                           # explicit softprob would raise
    assert xgb.named_steps["clf"].estimator_.get_xgb_params()["objective"] == "binary:logistic"
    assert set(xgb.predict(X)) <= {0.0, 1.0}


# ---------------------------------------------------------------- published tree
def test_snapshot_paper_scripts_absent_and_no_internal_report_in_the_code_tree():
    """Scripts emitting the pre-revision paper numbers ('ceiling' figure, fold Wilcoxon tables, snapshot
    table check) are not part of the published code; internal reports stay with the manuscript."""
    for name in ("performance_ceiling.py", "finalize_paper_rerun.py", "verify_paper_tables.py", "paper_values.csv"):
        assert not (REPO_DIR / "analysis" / name).exists()
        assert not (REPO_DIR / "archive" / "analysis" / name).exists()
    assert not list(REPO_DIR.glob("week*_*.md"))


def test_snapshot_revision_numbers_generator_is_never_published():
    """analysis/generate_revision_numbers.py (snapshot paper numbers, 'ceiling' section) is a
    local editorial tool of the manuscript pass: git-ignored and never tracked, so absent from the published
    tree; no tracked analysis script writes REVISION_NUMBERS.md or a 'ceiling' section."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", str(REPO_DIR), *a], capture_output=True, text=True)
    if git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")
    tracked = git("ls-files", "--", "analysis", "archive").stdout.split()
    assert tracked and not any(p.endswith("generate_revision_numbers.py") for p in tracked)
    assert git("check-ignore", "-q", "analysis/generate_revision_numbers.py").returncode == 0
    for rel in tracked:
        if rel.endswith(".py") and rel.startswith("analysis/"):
            text = (REPO_DIR / rel).read_text(encoding="utf-8")
            assert "REVISION_NUMBERS.md" not in text and "section_9_ceiling" not in text, rel


# ---------------------------------------------------------------- frozen artefact (Phase 1 and heatmap scripts)
def test_phase1_and_heatmap_scripts_refuse_the_frozen_artefact(tmp_path, monkeypatch):
    """haghirad_reproduction.py (README Phase 1 command) and feature_importance_heatmap.py refuse an output
    inside rerun_2026-04-28_no_reweighting/ before reading or writing anything."""
    import importlib
    monkeypatch.chdir(REPO_DIR)
    monkeypatch.syspath_prepend(str(REPO_DIR / "analysis"))
    hr = importlib.import_module("haghirad_reproduction")
    fi = importlib.import_module("feature_importance_heatmap")
    grid_dir = PROTECTED / "phase1_haghirad_grid"
    before = {p: p.stat().st_mtime_ns for p in grid_dir.glob("*")} if grid_dir.exists() else {}
    for out in (grid_dir, PROTECTED, PROTECTED / "x" / ".." / "phase1_haghirad_grid",
                Path("rerun_2026-04-28_no_reweighting") / "phase1_haghirad_grid"):
        with pytest.raises(ValueError, match="frozen artefact"):
            hr.main(["--output_dir", str(out), "--datasets", "ASHRAE_2022"])
    rel = Path("rerun_2026-04-28_no_reweighting") / "regenerated_figures" / "feature_importance_heatmap.pdf"
    for out in (PROTECTED / "regenerated_figures" / "feature_importance_heatmap.pdf", rel):
        with pytest.raises(ValueError, match="frozen artefact"):
            fi.main(["--results_dir", str(tmp_path / "absent"), "--out", str(out)])
        with pytest.raises(ValueError, match="frozen artefact"):
            fi.make_heatmap(pd.DataFrame(), out)
    with pytest.raises(FileNotFoundError):            # outside the artefact: the usual path is unchanged
        fi.generate(tmp_path / "absent", tmp_path / "fig.pdf")
    if before:
        assert {p: p.stat().st_mtime_ns for p in grid_dir.glob("*")} == before
