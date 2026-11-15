"""Tests of the revision statistics module.

Covers the label contract and the bootstrap core (cluster eligibility,
bootstrap intervals). Synthetic data only (no prediction is read,
no model is trained).

Run from the repository root:
    .venv-revision/bin/python -m pytest tests/test_revision_stats.py -q
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import revision_stats as rs  # noqa: E402

# =============================================================== label contract
def test_qwk_uses_full_ontology_protocol_example():
    y_true, y_pred = [-3, -3, -2, 3], [-3, -2, 3, 3]
    assert cohen_kappa_score(y_true, y_pred, weights="quadratic") == pytest.approx(0.6923, abs=1e-4)
    assert rs.quadratic_weighted_kappa(y_true, y_pred, labels=rs.ontology("thermal_sensation")) == \
        pytest.approx(0.5969, abs=1e-4)
    m = rs.classification_metrics(y_true, y_pred, "thermal_sensation")
    assert m["qwk"] == pytest.approx(0.5969, abs=1e-4)


def test_missing_middle_class_keeps_full_matrix_and_ontology_qwk():
    rng = np.random.default_rng(0)
    y_true = rng.choice([-3, -2, -1, 1, 2, 3], size=200)          # no class 0
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=200), -3, 3)
    m = rs.classification_metrics(y_true, y_pred, "thermal_sensation")
    assert np.asarray(m["confusion_matrix"]).shape == (7, 7)
    assert m["support"][0] == 0 and np.isnan(m["recall"][0])
    assert m["qwk"] == pytest.approx(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                                       labels=list(range(-3, 4))))
    cm3 = rs.confusion_matrix_ontology([-1, 1, 1], [1, 1, -1], "TSV_3p")
    assert cm3.shape == (3, 3) and list(cm3.index) == [-1, 0, 1] and cm3.loc[0].sum() == 0


def test_single_class_set_flags_qwk_undefined():
    m = rs.classification_metrics([0, 0, 0], [0, 0, 0], "thermal_preference")
    assert np.isnan(m["qwk"]) and m["qwk_undefined"]
    assert m["balanced_accuracy"] == 1.0 and m["balanced_accuracy_single_class"]
    # a single true class with varied predictions keeps a non-zero denominator: QWK = 0, defined
    m2 = rs.classification_metrics([0, 0, 0], [0, 1, -1], "thermal_preference")
    assert not m2["qwk_undefined"] and m2["qwk"] == pytest.approx(0.0)


def test_balanced_accuracy_and_macro_f1_over_present_classes():
    y_true, y_pred = [0, 0, 1, 1], [0, 2, 1, 1]
    m = rs.classification_metrics(y_true, y_pred, "thermal_sensation")
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    # F1(0) = 2/3, F1(1) = 1; class 2 (predicted, absent from y_true) is not in the denominator
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["classes_present"] == [0, 1]
    assert m["accuracy"] == 0.75 and m["n_correct"] == 3


def test_label_validation_never_drops_silently():
    with pytest.raises(ValueError, match="outside the ontology"):
        rs.classification_metrics([0, 1], [0, 4], "thermal_sensation")
    with pytest.raises(ValueError, match="NaN"):
        rs.classification_metrics([0, 1], [0, np.nan], "TSV_3p")
    with pytest.raises(ValueError, match="empty"):
        rs.classification_metrics([], [], "TSV_3p")
    with pytest.raises(ValueError, match="unknown target"):
        rs.ontology("TSV7")
    m = rs.classification_metrics(np.array([-1.0, 0.0, 1.0]), np.array([-1, 0, 0]), "thermal_preference")
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_stats_utils_qwk_requires_labels():
    import stats_utils
    with pytest.raises(TypeError):
        stats_utils.quadratic_weighted_kappa([0, 1], [0, 1])  # noqa
    assert stats_utils.quadratic_weighted_kappa([-3, -3, -2, 3], [-3, -2, 3, 3],
                                                labels=rs.ontology("thermal_sensation")) == \
        pytest.approx(0.5969, abs=1e-4)


def test_present_class_scores_is_the_shared_definition():
    rng = np.random.default_rng(21)
    y_true = rng.choice([-3, -1, 0, 2], size=300)
    y_pred = rng.integers(-3, 4, size=300)
    s = rs.present_class_scores(y_true, y_pred, labels=rs.ontology("thermal_sensation"))
    assert s["classes_present"] == [-3, -1, 0, 2] and not s["balanced_accuracy_single_class"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert s["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    from sklearn.metrics import f1_score
    assert s["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, labels=[-3, -1, 0, 2],
                                                   average="macro", zero_division=0))
    m = rs.classification_metrics(y_true, y_pred, "thermal_sensation")
    assert (m["macro_f1"], m["balanced_accuracy"]) == (s["macro_f1"], s["balanced_accuracy"])
    with pytest.raises(ValueError, match="outside the ontology"):
        rs.present_class_scores([0, 1], [0, 2], labels=rs.ontology("TSV_3p"))
    with pytest.raises(ValueError, match="differ in length"):
        rs.present_class_scores([0, 1], [0], labels=rs.ontology("TSV_3p"))


# ------------------------------------------------- QWK callers
class _FixedPredictor:
    """Stand-in for a fitted model: returns a fixed prediction vector."""

    def __init__(self, y_pred):
        self.y_pred = np.asarray(y_pred)

    def predict(self, X):
        assert len(X) == len(self.y_pred)
        return self.y_pred


@pytest.fixture(scope="module")
def models_module():
    import models
    return models


def test_eval_on_test_requires_and_uses_the_full_ontology(models_module, tmp_path):
    ev = models_module.eval_on_test
    y_true, y_pred = pd.Series([-3, -3, -2, 3]), [-3, -2, 3, 3]
    X = pd.DataFrame({"x": range(4)})
    with pytest.raises(ValueError, match="labels_order"):
        ev(_FixedPredictor(y_pred), X, y_true)
    res = ev(_FixedPredictor(y_pred), X, y_true, name="ex",
             labels_order=rs.ontology("thermal_sensation"), save_dir=tmp_path)
    # worked example: 0.5969 on the ontology, not the observed-label 0.6923
    assert res["qwk"] == pytest.approx(0.59689922, abs=1e-8) and not res["qwk_undefined"]
    assert res["labels"] == [str(c) for c in range(-3, 4)]
    assert np.asarray(res["confusion"]).shape == (7, 7)
    ref = rs.classification_metrics(y_true, y_pred, "thermal_sensation")
    assert res["f1_macro"] == pytest.approx(ref["macro_f1"])
    assert res["balanced_accuracy"] == pytest.approx(ref["balanced_accuracy"])
    saved = json.loads((tmp_path / "ex_test_results.json").read_text())
    assert saved["qwk"] == pytest.approx(res["qwk"])
    cm = pd.read_csv(tmp_path / "ex_confusion_matrix.csv", index_col=0)
    assert list(cm.index) == list(range(-3, 4)) and cm.shape == (7, 7)


def test_eval_on_test_missing_middle_single_class_and_invalid_predictions(models_module):
    ev = models_module.eval_on_test
    tpv = rs.ontology("thermal_preference")
    X3 = pd.DataFrame({"x": range(3)})
    # missing middle class: 3 x 3 matrix, QWK on the ontology
    res = ev(_FixedPredictor([-1, 1, 1]), X3, pd.Series([-1, 1, 1.0]), labels_order=tpv)
    assert np.asarray(res["confusion"]).shape == (3, 3) and np.asarray(res["confusion"])[1].sum() == 0
    assert res["qwk"] == pytest.approx(1.0)
    # single class in truth and predictions: undefined QWK, never an exception, 0 or 1
    single = ev(_FixedPredictor([0, 0, 0]), X3, pd.Series([0, 0, 0]), labels_order=tpv)
    assert np.isnan(single["qwk"]) and single["qwk_undefined"] and single["accuracy"] == 1.0
    # a prediction outside the ontology raises (never appended to the label list)
    with pytest.raises(ValueError, match="outside the ontology"):
        ev(_FixedPredictor([0, 2, 0]), X3, pd.Series([0, 1, 0]), labels_order=tpv)
    with pytest.raises(ValueError, match="outside the ontology"):
        ev(_FixedPredictor([0, 1, 0]), X3, pd.Series([0, -2, 0]), labels_order=tpv)
    with pytest.raises(ValueError, match="predictions"):
        ev(_FixedPredictor([0, 1]), X3.iloc[:2], pd.Series([0, 1, 0]), labels_order=tpv)


def test_every_evaluation_caller_passes_the_target_ontology():
    import ast
    for fname in ("models.py", "transfert.py"):
        tree = ast.parse((REPO_DIR / fname).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None)) == "eval_on_test"]
        assert calls, fname
        for c in calls:
            kw = {k.arg: k.value for k in c.keywords}
            assert "labels_order" in kw, f"{fname}:{c.lineno}"
            v = kw["labels_order"]
            assert isinstance(v, ast.Call) and getattr(v.func, "id", None) == "ontology", f"{fname}:{c.lineno}"
    for fname in ("models.py", "hybrid.py", "transfert.py", "stats_utils.py"):
        tree = ast.parse((REPO_DIR / fname).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) in (
                    "quadratic_weighted_kappa", "_qwk_on_labels", "cohen_kappa_score"):
                assert any(k.arg == "labels" for k in n.keywords), f"{fname}:{n.lineno}"


# Hybrid-head tests, adapted to the single fit (seed and compute arguments). A test that fits a
# multi-class XGBoost head needs the GPU visible with the CUDA build of xgboost (fixture xgb_fits).
_HEAD_COMPUTE = {"xgb_device": "cpu", "torch_device": "cpu", "cpu_jobs": 2, "gpu_concurrency": 1}


def _heads_or_skip(request, heads):
    if "XGBoost" in heads:
        request.getfixturevalue("xgb_fits")
    return heads


@pytest.mark.parametrize("heads", [["RF"], ["XGBoost"]])
def test_hybrid_heads_use_the_full_ontology(tmp_path, monkeypatch, request, heads):
    import hybrid
    heads = _heads_or_skip(request, heads)
    rng = np.random.default_rng(22)
    y_train = pd.Series(np.repeat([-1, 0, 1], 20))
    X_train = pd.DataFrame({"a": y_train + rng.normal(0, 0.1, 60), "b": rng.normal(0, 1, 60)})
    y_test = pd.Series([-1, -1, 1, 1, 1, 1.0])                         # no class 0
    X_test = pd.DataFrame({"a": y_test.to_numpy() + rng.normal(0, 0.1, 6), "b": rng.normal(0, 1, 6)})
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: X.to_numpy())
    hp = {"heads": {"RF": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
                    "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3,
                                "subsample": 1.0, "colsample_bytree": 1.0}}}
    rows = hybrid.run_heads_on_ft_and_save(None, X_train, y_train, X_test, y_test, tmp_path,
                                           target_name="thermal_preference", heads=heads, hparams=hp,
                                           seed=42, compute=_HEAD_COMPUTE)
    assert len(rows) == 1
    for head, tag in [(h, "RF" if h == "RF" else "XGB") for h in heads]:
        res = json.loads((tmp_path / head / f"thermal_preference_FTemb_{tag}_test_results.json").read_text())
        assert res["labels"] == ["-1", "0", "1"] and np.asarray(res["confusion"]).shape == (3, 3)
        ref = rs.classification_metrics(y_test, res["y_pred"], "thermal_preference")
        assert res["qwk"] == pytest.approx(ref["qwk"]) and res["f1_macro"] == pytest.approx(ref["macro_f1"])
        assert res["balanced_accuracy"] == pytest.approx(ref["balanced_accuracy"])
        assert set(res["report"]) >= {"-1", "0", "1"}
        assert not (tmp_path / head / f"thermal_preference_FTemb_{tag}_cv_best.json").exists()   # no head CV
    with pytest.raises(ValueError, match="outside the ontology"):
        hybrid.run_heads_on_ft_and_save(None, X_train, y_train, X_test, y_test.replace(1.0, 2.0),
                                        tmp_path / "bad", target_name="thermal_preference", heads=heads,
                                        hparams=hp, seed=42, compute=_HEAD_COMPUTE)


@pytest.mark.parametrize("heads", [["RF"], ["XGBoost"]])
def test_hybrid_heads_absent_training_class_kept_in_test(tmp_path, monkeypatch, request, heads):
    """A test class never seen in training is evaluated (as an error), not dropped."""
    import hybrid
    heads = _heads_or_skip(request, heads)
    rng = np.random.default_rng(23)
    y_train = pd.Series(np.repeat([-1, 0, 1], 20))                      # no class -2 in training
    X_train = pd.DataFrame({"a": y_train + rng.normal(0, 0.1, 60), "b": rng.normal(0, 1, 60)})
    y_test = pd.Series([-2, -2, -1, 0, 1, 1])
    X_test = pd.DataFrame({"a": y_test.to_numpy() + rng.normal(0, 0.1, 6), "b": rng.normal(0, 1, 6)})
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: X.to_numpy())
    hp = {"heads": {"RF": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
                    "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3,
                                "subsample": 1.0, "colsample_bytree": 1.0}}}
    hybrid.run_heads_on_ft_and_save(None, X_train, y_train, X_test, y_test, tmp_path,
                                    target_name="thermal_sensation", heads=heads, hparams=hp, seed=42,
                                    compute=_HEAD_COMPUTE)
    for head, tag in [(h, "RF" if h == "RF" else "XGB") for h in heads]:
        res = json.loads((tmp_path / head / f"thermal_sensation_FTemb_{tag}_test_results.json").read_text())
        cm = np.asarray(res["confusion"])
        assert res["n_test"] == 6 and len(res["y_pred"]) == 6 and cm.shape == (7, 7) and cm.sum() == 6
        assert cm[1].sum() == 2 and cm[1, 1] == 0                       # both -2 votes are errors
        assert res["n_correct"] <= 4
        assert res["report"]["macro avg"]["f1-score"] == pytest.approx(res["f1_macro"])
        assert res["report"]["macro avg classes"] == [-2, -1, 0, 1]


_TINY_HEADS = {"heads": {"RF": {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
                         "XGBoost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3,
                                     "subsample": 1.0, "colsample_bytree": 1.0}}}


@pytest.mark.parametrize("head,tag", [("RF", "RF"), ("XGBoost", "XGB")])
def test_hybrid_each_head_runs_alone(tmp_path, monkeypatch, request, head, tag):
    """The CV splitter must not depend on the RF head being selected."""
    import hybrid
    _heads_or_skip(request, [head])
    rng = np.random.default_rng(31)
    y_train = pd.Series(np.repeat([-1, 0, 1], 20))
    X_train = pd.DataFrame({"a": y_train + rng.normal(0, 0.1, 60), "b": rng.normal(0, 1, 60)})
    y_test = pd.Series([-1, 0, 1, 1.0])
    X_test = pd.DataFrame({"a": y_test.to_numpy() + rng.normal(0, 0.1, 4), "b": rng.normal(0, 1, 4)})
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: X.to_numpy())
    rows = hybrid.run_heads_on_ft_and_save(None, X_train, y_train, X_test, y_test, tmp_path,
                                           target_name="TSV_3p", heads=[head], hparams=_TINY_HEADS,
                                           seed=42, compute=_HEAD_COMPUTE)
    assert [r["model"] for r in rows] == [f"FTemb+{tag}"]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == [head]
    res = json.loads((tmp_path / head / f"TSV_3p_FTemb_{tag}_test_results.json").read_text())
    assert res["n_test"] == 4 and np.asarray(res["confusion"]).shape == (3, 3)


@pytest.mark.parametrize("head,tag", [("RF", "RF"), ("XGBoost", "XGB")])
def test_hybrid_head_trained_on_two_classes_evaluated_on_three(tmp_path, monkeypatch, request, head, tag):
    """A 3-class target with one class absent from training (XGB: binary objective)."""
    import hybrid
    _heads_or_skip(request, [head])
    rng = np.random.default_rng(32)
    y_train = pd.Series(np.repeat([0, 1], 20))                          # no class -1 in training
    X_train = pd.DataFrame({"a": y_train + rng.normal(0, 0.1, 40), "b": rng.normal(0, 1, 40)})
    y_test = pd.Series([-1, -1, 0, 0, 1, 1.0])
    X_test = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0], "b": rng.normal(0, 1, 6)})
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: X.to_numpy())
    hybrid.run_heads_on_ft_and_save(None, X_train, y_train, X_test, y_test, tmp_path,
                                    target_name="thermal_preference", heads=[head], hparams=_TINY_HEADS,
                                    seed=42, compute=_HEAD_COMPUTE)
    res = json.loads((tmp_path / head / f"thermal_preference_FTemb_{tag}_test_results.json").read_text())
    cm = np.asarray(res["confusion"])
    # predictions are mapped back to the original labels 0/1 (never encoded 0/1 or -1)
    assert res["y_pred"] == [0, 0, 0, 0, 1, 1]
    assert res["n_test"] == 6 and cm.shape == (3, 3) and cm.sum() == 6
    assert cm[0].tolist() == [0, 2, 0]                                  # both -1 votes are errors
    assert res["n_correct"] == 4 and res["accuracy"] == pytest.approx(4 / 6)
    ref = rs.classification_metrics(y_test, res["y_pred"], "thermal_preference")
    assert res["qwk"] == pytest.approx(ref["qwk"]) and res["f1_macro"] == pytest.approx(ref["macro_f1"])


# The legacy Phase 3 adaptation evaluator (single stratified row split, FT_head /
# FT_full re-initialising the network) and its interim guards were replaced by the
# E6 runner; its tests (every TEST row kept, unseen classes
# scored as errors, no CUDA needed for zero-shot) are in tests/test_phase3_runner.py.


# ------------------------------------- DEV/TEST partition check
@pytest.mark.parametrize("cohort, dev, test, match", [
    ([0, 1, 2, 3], [0, 0], [2, 3], "duplicate IDs in DEV"),
    ([0, 1, 2, 3], [0, 1], [2, 2, 3], "duplicate IDs in TEST"),
    ([0, 1, 2, 2], [0, 1], [2], "duplicate IDs in cohort"),
    ([0, 1, 2, 3], [0, 1], [1, 2, 3], "both DEV and TEST"),
    ([0, 1, 2, 3], [0], [2, 3], r"missing \[1\]"),
    ([0, 1, 2, 3], [0, 1], [2, 3, 9], r"foreign \[9\]"),
    ([0, 1, 2, 3], [0, 5], [2, 3], r"missing \[1\], foreign \[5\]"),  # equal sizes, wrong IDs
])
def test_check_partition_rejects_malformed_partitions(cohort, dev, test, match):
    import transfert
    with pytest.raises(ValueError, match=match):
        transfert.check_partition(cohort, dev, test, "Synthetic")


def test_check_partition_accepts_a_complete_partition():
    import transfert
    transfert.check_partition([10, 11, 12, 13], [12, 10], [13, 11], "Synthetic")
    transfert.check_partition(pd.Index([0, 1]), pd.Index([0, 1]), pd.Index([]), "Synthetic")


def test_zero_shot_evaluates_every_row_for_every_family(models_module, tmp_path):
    """FTT must not be evaluated on rows selected by their label."""
    import transfert
    target = "TSV_3p"
    feats = list(models_module.FEATURES_ALL)
    df = pd.DataFrame({f: [0.0, 0.0, 0.0] for f in feats}, index=[10, 11, 12])
    df[target] = [-1.0, 0.0, 1.0]
    seen = {}

    class _Recorder(_FixedPredictor):
        def __init__(self, name, y_pred, classes):
            super().__init__(y_pred)
            self.name, self.le_y_ = name, type("LE", (), {"classes_": np.asarray(classes)})()

        def predict(self, X):
            seen[self.name] = list(X.index)
            return super().predict(X)

    ft = _Recorder("FTT", [0, 0, 1], classes=[0, 1])                   # never learned -1
    rf = _Recorder("RF", [-1, 0, 1], classes=[-1, 0, 1])
    xgb = _Recorder("XGB", [0, 0, 0], classes=[-1, 0, 1])
    transfert.direct_eval_one_base("Synthetic", df, {target: ft}, {target: rf}, {target: xgb},
                                   output_dir=str(tmp_path))
    assert seen == {"FTT": [10, 11, 12], "RF": [10, 11, 12], "XGB": [10, 11, 12]}
    root = tmp_path / "Synthetic" / "A_direct" / target
    got = {}
    for sub, name in (("FTTransformer", "FTTransformer"), ("RandomForest", "RF"), ("XGBoost", "XGB")):
        got[name] = json.loads((root / sub / f"Synthetic__{target}__{name}_test_results.json").read_text())
    assert {k: v["n_test"] for k, v in got.items()} == {"FTTransformer": 3, "RF": 3, "XGB": 3}
    # the -1 vote the FTT never learned is kept and counts as one error
    ftt = got["FTTransformer"]
    assert ftt["n_correct"] == 2 and ftt["accuracy"] == pytest.approx(2 / 3)
    assert np.asarray(ftt["confusion"])[0].tolist() == [0, 1, 0]
    assert got["RF"]["accuracy"] == 1.0 and got["XGB"]["accuracy"] == pytest.approx(1 / 3)


def test_detailed_report_macro_matches_protocol_macro_f1(models_module, tmp_path, capsys):
    """Saved/printed report macro avg = truth-present macro-F1."""
    y_true, y_pred = pd.Series([0, 0, 1, 1]), [0, 2, 1, 1]
    lab = rs.ontology("thermal_sensation")
    rep = rs.ontology_classification_report(y_true, y_pred, labels=lab)
    assert [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg", "macro avg classes")] == \
        [str(c) for c in lab]
    assert rep["macro avg classes"] == [0, 1]
    assert rep["macro avg"]["f1-score"] == pytest.approx(0.8333333333333333)
    assert rep["macro avg"]["recall"] == pytest.approx(rs.present_class_scores(y_true, y_pred, labels=lab)
                                                        ["balanced_accuracy"])
    assert rep["2"] == {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    res = models_module.eval_on_test(_FixedPredictor(y_pred), pd.DataFrame({"x": range(4)}), y_true,
                                     name="rep", labels_order=lab, save_dir=tmp_path)
    saved = json.loads((tmp_path / "rep_test_report.json").read_text())
    for r in (res["report"], saved):
        assert r["macro avg"]["f1-score"] == pytest.approx(res["f1_macro"]) == pytest.approx(0.8333333333333333)
    assert res["recall_macro"] == pytest.approx(res["balanced_accuracy"])
    assert res["precision_macro"] == pytest.approx(rep["macro avg"]["precision"])
    printed = capsys.readouterr().out
    macro_line = next(l for l in printed.splitlines() if l.strip().startswith("macro avg "))
    assert macro_line.split()[4] == "0.8333"                           # f1-score column
    assert "0.2381" not in printed
    with pytest.raises(ValueError, match="target_names"):
        rs.ontology_classification_report(y_true, y_pred, labels=lab, target_names=["a"])


# =============================================================== majority baselines
def test_deployable_majority_tie_rules():
    assert rs.deployable_majority_class([-1, -1, 1, 1, 0], "thermal_preference") == -1
    assert rs.deployable_majority_class([2, 2, 0, 0, -1], "thermal_sensation") == 0
    assert rs.deployable_majority_class([1, 1, -1, -1, -2, -2], "thermal_sensation") == -1
    assert rs.deployable_majority_class([3, 3, 3, 0], "thermal_sensation") == 3


def test_majority_baselines_deployable_and_descriptive():
    b = rs.majority_baselines(y_train=[0, 0, 0, 1], y_eval=[1, 1, 1, 0, -1], target="TSV_3p")
    assert b["deployable_class"] == 0 and b["deployable_accuracy"] == pytest.approx(0.2)
    assert b["descriptive_class"] == 1 and b["descriptive_majority_rate"] == pytest.approx(0.6)
    # constant predictor: balanced accuracy = 1 / number of present classes
    assert b["deployable_balanced_accuracy"] == pytest.approx(1 / 3)
    assert b["descriptive_balanced_accuracy"] == pytest.approx(1 / 3)


# =============================================================== bootstrap core
def test_frozen_draw_is_the_declared_rng_call():
    g = 37
    w = rs.frozen_draw_weights(g)
    idx = np.random.default_rng(20261008).integers(0, g, size=(10_000, g))
    expected = np.stack([np.bincount(row, minlength=g) for row in idx])
    assert w.shape == (10_000, g) and np.array_equal(w, expected)
    assert np.all(w.sum(axis=1) == g)
    assert not w.flags.writeable
    assert rs.BOOT_B == 10_000 and rs.BOOT_SEED == 20261008


def _synthetic_votes(n_buildings=25, per=8, seed=1, ids=None):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_buildings) * 7 + 3 if ids is None else np.asarray(ids)
    clusters = np.repeat(ids, per)
    y = rng.integers(-1, 2, size=len(clusters))
    p = np.where(rng.random(len(clusters)) < 0.6, y, rng.integers(-1, 2, size=len(clusters)))
    return y, p, clusters


def test_replicates_equal_explicit_vote_resampling_on_sorted_ids():
    ids = np.array([90, 4, 17, 33, 2, 61, 8, 45, 12, 77, 26, 5])   # unsorted on purpose
    y, p, clusters = _synthetic_votes(ids=ids, per=5, seed=3)
    est = rs.accuracy_estimate(y, p, clusters, target="TSV_3p")
    assert est.available
    order = np.sort(ids)
    idx = np.random.default_rng(20261008).integers(0, len(order), size=(10_000, len(order)))
    correct = (y == p).astype(float)
    for b in (0, 1, 4321, 9999):
        votes = np.concatenate([np.flatnonzero(clusters == order[k]) for k in idx[b]])
        assert est.replicates[b] == pytest.approx(correct[votes].mean(), abs=1e-12)
    # row order of the evaluated set does not change the draw (it is over sorted IDs)
    perm = np.random.default_rng(9).permutation(len(y))
    est2 = rs.accuracy_estimate(y[perm], p[perm], clusters[perm], target="TSV_3p")
    assert np.allclose(est.replicates, est2.replicates, atol=1e-12)
    assert est.ci_low <= est.estimate <= est.ci_high


@pytest.mark.parametrize("n_buildings, available", [(1, False), (9, False), (10, True), (11, True)])
def test_cluster_eligibility_rule(n_buildings, available):
    """1 building -> "not available"; exactly 10 -> an interval."""
    y, p, clusters = _synthetic_votes(n_buildings=n_buildings, per=30, seed=n_buildings)
    est = rs.accuracy_estimate(y, p, clusters, target="TSV_3p")
    assert est.available is available
    assert est.estimate == pytest.approx(np.mean(y == p))
    assert est.n_clusters == n_buildings and est.n_denominator == len(y)
    if available:
        assert est.ci_low < est.ci_high
    else:
        assert est.interval == rs.NOT_AVAILABLE and np.isnan(est.ci_low) and est.replicates is None


def test_grouped_accuracy_per_publication_eligibility():
    y1, p1, c1 = _synthetic_votes(n_buildings=1, per=200, seed=1)
    y2, p2, c2 = _synthetic_votes(n_buildings=10, per=20, seed=2, ids=np.arange(100, 110))
    res = rs.grouped_accuracy(np.r_[y1, y2], np.r_[p1, p2], ["pubA"] * len(y1) + ["pubB"] * len(y2),
                              np.r_[c1, c2], target="TSV_3p")
    assert res["pubA"].interval == rs.NOT_AVAILABLE and res["pubA"].n_clusters == 1
    assert res["pubB"].available and res["pubB"].n_clusters == 10


def test_paired_contrast_with_seeds_recomputes_every_seed_on_same_resample():
    rng = np.random.default_rng(5)
    y, _, clusters = _synthetic_votes(n_buildings=30, per=10, seed=5)
    pa = np.stack([np.where(rng.random(len(y)) < 0.7, y, 0) for _ in range(3)])
    pb = np.where(rng.random(len(y)) < 0.6, y, 1)
    res = rs.paired_accuracy_contrast(y, pa, pb, clusters, target="TSV_3p")
    acc_a = np.mean([(s == y).mean() for s in pa])
    assert res["delta"].estimate == pytest.approx(acc_a - (pb == y).mean())
    seed_reps = [rs.accuracy_estimate(y, s, clusters, target="TSV_3p").replicates for s in pa]
    b_reps = rs.accuracy_estimate(y, pb, clusters, target="TSV_3p").replicates
    assert np.allclose(res["delta"].replicates, np.mean(seed_reps, axis=0) - b_reps, atol=1e-12)
    assert len(res["seed_accuracies_a"]) == 3 and res["seed_spread_b"] == 0.0


def test_percentile_interval_and_bonferroni_levels():
    reps = np.random.default_rng(2).normal(size=10_000)
    for m in (9, 12, 18):
        level = rs.bonferroni_level(m)
        assert level == pytest.approx(1 - 0.05 / m)
        lo, hi = rs.percentile_interval(reps, level)
        assert (lo, hi) == pytest.approx(tuple(np.quantile(reps, [0.025 / m, 1 - 0.025 / m])))
    with pytest.raises(ValueError, match="undefined replicate"):
        rs.percentile_interval(np.r_[reps[:10], np.nan], 0.95)


# =============================================================== zero denominators
def test_no_replicate_deletion_or_nan_skipping_in_source():
    src = inspect.getsource(rs)
    assert not re.search(r"np\.nan(mean|median|quantile|percentile|sum|std|var|min|max)", src)
    assert "dropna" not in src
    # an injected undefined replicate never yields an interval
    reps = np.r_[np.linspace(-0.01, 0.01, 99), np.nan]
    with pytest.raises(ValueError):
        rs.percentile_interval(reps, 0.95)


# =============================================================== predictions files
def test_predictions_files_alignment(tmp_path):
    base = pd.DataFrame({"row_idx": [5, 7, 9], "y_true": [0, 1, -1], "y_pred": [0, 1, 1],
                         "building_id": [1, 1, 2]})
    p = tmp_path / "a.csv.gz"
    base.to_csv(p, index=False, compression="gzip")
    a = rs.load_predictions(p)
    b = base.assign(y_pred=[0, 0, -1])
    y, preds, cl, rows = rs.align_predictions({"A": a, "B": b})
    assert list(rows) == [5, 7, 9] and list(preds["B"]) == [0, 0, -1]
    with pytest.raises(ValueError, match="row_idx"):
        rs.align_predictions({"A": a, "B": b.iloc[[1, 0, 2]].reset_index(drop=True)})
    with pytest.raises(ValueError, match="missing columns"):
        bad = tmp_path / "bad.csv.gz"
        base.drop(columns="y_pred").to_csv(bad, index=False, compression="gzip")
        rs.load_predictions(bad)


def test_predictions_alignment_one_prediction_per_vote():
    """Duplicate or missing row_idx, and serialised split IDs."""
    base = pd.DataFrame({"row_idx": [5, 7, 9], "y_true": [0, 1, -1], "y_pred": [0, 1, 1],
                         "building_id": [1, 1, 2]})
    dup = pd.DataFrame({"row_idx": [5, 5], "y_true": [0, 0], "y_pred": [0, 1], "building_id": [1, 1]})
    with pytest.raises(ValueError, match="duplicate row_idx"):
        rs.align_predictions({"A": dup, "B": dup.assign(y_pred=[1, 0])})
    with pytest.raises(ValueError, match="duplicate row_idx"):
        rs.align_predictions({"A": dup})
    missing = base.assign(row_idx=[5.0, np.nan, 9.0])
    with pytest.raises(ValueError, match="missing row_idx"):
        rs.align_predictions({"A": base, "B": missing})
    _, _, _, rows = rs.align_predictions({"A": base, "B": base}, expected_row_idx=[5, 7, 9])
    assert list(rows) == [5, 7, 9]
    for exp in ([5, 9, 7], [5, 7], [5, 7, 9, 11]):
        with pytest.raises(ValueError, match="serialised split IDs"):
            rs.align_predictions({"A": base, "B": base}, expected_row_idx=exp)
