# -*- coding: utf-8 -*-
"""
Internal validation and early stopping of the ANN and the FT-Transformer
(early-stopping fix of the revision).

Blocking tests:

* the restored checkpoint equals the weights of the best validation epoch
  (bitwise), and that epoch is the first minimum of the validation loss under
  the 1e-6 improvement threshold;
* the preprocessor (scaler, category codes, label encoder) is fitted on the
  internal training part (90 %) only;
* a fit whose internal validation subset cannot be built raises explicitly
  (< 100 rows, a class with one vote, more classes than validation votes, no
  finite validation loss); there is no fallback to the training loss.

Also: the internal split is the frozen draw (StratifiedShuffleSplit, 10 %,
seed 42, independent of the model seed); ``select_best`` fits the final model
once (no refit on 100 %, no auxiliary curves model) and every CV fit uses its
own internal split; ``models.py`` runs the deep stage end to end and writes the
final model's curves and early-stopping record.

Synthetic data only, tiny networks on CPU, each test well under 60 s.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import models_deep as md  # noqa: E402
import revision_folds as rf  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY, vocabulary_for  # noqa: E402

NUM = ["a", "b"]
CAT = ["Season"]
TINY = {
    "ANN": dict(hidden=(32,)),
    "FTT": dict(dim=8, depth=1, heads=2, dim_head=4),
}


def _data(n=200, seed=0, classes=(-1, 0, 1), noisy=True):
    """Labels unrelated to the features when ``noisy`` (the validation loss soon rises)."""
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.resize(np.asarray(classes), n))
    seasons = sorted(GLOBAL_VOCABULARY["Season"])
    X = pd.DataFrame({
        "a": rng.normal(size=n) + (0 if noisy else y),
        "b": rng.normal(size=n),
        "Season": [seasons[i % len(seasons)] for i in range(n)],
    })
    return X, y


def _make(kind, **kw):
    params = dict(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device="cpu",
                  lr=3e-2, batch_size=32, epochs=80, patience=4)
    params.update(kw)
    if kind == "ANN":
        return md.ANNClassifier(**TINY["ANN"], **params)
    return md.FTClassifier(**TINY["FTT"], **params)


def _logits(model, X):
    model.model_.eval()
    x_cat, x_num = model.pp_.transform(X)
    with torch.no_grad():
        if isinstance(model, md.ANNClassifier):
            return model.model_(torch.cat([x_num, x_cat.float()], dim=1))
        return model.model_(x_cat, x_num)


def _record_epochs(monkeypatch):
    """Wrap the shared loop so that every epoch's weights are recorded."""
    states = []
    orig = md._train_early_stopping

    def wrapped(*a, **k):
        return orig(*a, on_epoch_end=lambda ep, m: states.append(
            {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}), **k)
    monkeypatch.setattr(md, "_train_early_stopping", wrapped)
    return states


# ================================================================ internal split
def test_internal_split_is_the_frozen_draw():
    _, y = _data(n=200)
    tr, va = md.internal_validation_split(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr2, va2 = next(sss.split(np.zeros((200, 1)), y))
    assert np.array_equal(tr, np.sort(tr2)) and np.array_equal(va, np.sort(va2))
    assert len(va) == 20 and not set(tr) & set(va) and set(tr) | set(va) == set(range(200))
    assert (md.ES_VAL_FRACTION, md.ES_VAL_SEED, md.ES_MIN_FIT_ROWS, md.ES_MIN_DELTA) == (0.1, 42, 100, 1e-6)


@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_internal_split_does_not_depend_on_the_model_seed(kind):
    X, y = _data(n=150)
    X.index = np.arange(1000, 1150)[::-1]                # cohort row labels, not positions
    a = _make(kind, epochs=2, random_state=42).fit(X, y)
    b = _make(kind, epochs=2, random_state=43).fit(X, y)
    assert np.array_equal(a.es_val_idx_, b.es_val_idx_)
    tr, va = md.internal_validation_split(y)
    assert np.array_equal(a.es_val_idx_, X.index[va]) and np.array_equal(a.es_train_idx_, X.index[tr])


def test_snapshot_switches_that_disabled_validation_are_gone():
    for cls in (md.ANNClassifier, md.FTClassifier, md._TorchBase):
        params = inspect.signature(cls.__init__).parameters
        assert "curves_val_split" not in params and "record_curves" not in params
    with pytest.raises(TypeError):
        _make("ANN", curves_val_split=0.0)
    assert "legacy_snapshot_fit" not in inspect.signature(md.select_best).parameters
    assert not hasattr(md, "LEGACY_DEEP_FIT_MESSAGE")


# ================================================================ BLOCKING: restored checkpoint
@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_restored_checkpoint_equals_best_validation_epoch(kind, monkeypatch):
    X, y = _data(n=200)
    states = _record_epochs(monkeypatch)
    m = _make(kind).fit(X, y)
    h = m.history_
    assert len(states) == len(h["epoch"]) == m.n_epochs_run_

    # best epoch = first epoch whose validation loss beats the running best by > 1e-6
    best, best_ep = float("inf"), None
    for ep, v in zip(h["epoch"], h["val_loss"]):
        if v + 1e-6 < best:
            best, best_ep = v, ep
    assert m.best_epoch_ == best_ep and m.best_val_loss_ == best
    # early stop happened (patience 4), so the last epoch is not the best one
    assert m.stopped_early_ and m.n_epochs_run_ == best_ep + m.patience
    restored = {n: t.detach().cpu() for n, t in m.model_.state_dict().items()}
    assert restored.keys() == states[best_ep - 1].keys()
    assert all(torch.equal(restored[n], states[best_ep - 1][n]) for n in restored)
    assert any(not torch.equal(restored[n], states[-1][n]) for n in restored)

    # the monitored loss is the cross-entropy of the restored model on the internal validation rows
    X_val = X.loc[m.es_val_idx_]
    y_val = torch.tensor(m.le_y_.transform(np.asarray(y)[m.es_val_idx_]))
    ce = float(F.cross_entropy(_logits(m, X_val), y_val))
    assert ce == pytest.approx(m.best_val_loss_, abs=1e-6)


def test_best_epoch_can_be_the_last_epoch(monkeypatch):
    """Learnable labels, a short budget: the restored state is the last epoch when it is the best."""
    X, y = _data(n=200, noisy=False)
    states = _record_epochs(monkeypatch)
    m = _make("ANN", lr=1e-3, epochs=3).fit(X, y)
    assert m.best_epoch_ == 3 and not m.stopped_early_
    assert all(torch.equal(t.cpu(), states[-1][n]) for n, t in m.model_.state_dict().items())


# ================================================================ BLOCKING: preprocessing on the 90 %
@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_preprocessor_fitted_on_internal_training_part_only(kind, monkeypatch):
    X, y = _data(n=200)
    tr, va = md.internal_validation_split(y)
    X.loc[va, "a"] += 50.0                               # validation rows shifted: visible in the scaler
    X.loc[X["Season"] == "winter", "Season"] = "summer"
    X.loc[va[:3], "Season"] = "winter"                   # a season seen only in validation rows
    fitted = []
    orig_fit = md.TabPreprocessor.fit

    def spy(self, Xf):
        fitted.append(Xf.index.to_numpy().copy())
        return orig_fit(self, Xf)
    monkeypatch.setattr(md.TabPreprocessor, "fit", spy)
    labels_fitted = []

    class SpyLabelEncoder(md.LabelEncoder):
        def fit(self, yf):
            labels_fitted.append(np.asarray(yf).copy())
            return super().fit(yf)
    monkeypatch.setattr(md, "LabelEncoder", SpyLabelEncoder)
    m = _make(kind, epochs=2).fit(X, y)

    assert len(fitted) == 1 and np.array_equal(fitted[0], X.index[tr])
    assert np.allclose(m.pp_.scaler.mean_, X.loc[tr, NUM].mean().to_numpy())
    assert not np.allclose(m.pp_.scaler.mean_, X[NUM].mean().to_numpy())
    assert "winter" not in m.pp_.train_categories_["Season"]
    assert m.es_val_oov_counts_["Season"] == 3          # counted as OOV for the fit
    assert m.pp_.vocabulary_ == vocabulary_for(CAT)      # codes stay those of the global vocabulary
    assert len(labels_fitted) == 1 and np.array_equal(labels_fitted[0], np.asarray(y)[tr])   # labels: 90 % only
    assert list(m.classes_) == sorted(set(np.asarray(y)[tr]))


# ================================================================ BLOCKING: explicit failures
@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_fewer_than_100_rows_raises(kind):
    X, y = _data(n=99)
    m = _make(kind)
    with pytest.raises(md.EarlyStoppingValidationError, match="at least 100"):
        m.fit(X, y)
    assert not hasattr(m, "model_")
    md.internal_validation_split(_data(n=100)[1])        # the floor itself is accepted


@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_stratified_split_impossible_raises(kind):
    X, y = _data(n=150)
    y = y.copy()
    y[0] = 7                                             # a class with one vote
    with pytest.raises(md.EarlyStoppingValidationError, match="cannot be built"):
        _make(kind).fit(X, y)
    X, _ = _data(n=100)
    y = np.repeat(np.arange(20), 5)                      # 20 classes, 10 validation votes
    with pytest.raises(md.EarlyStoppingValidationError, match="cannot be built"):
        _make(kind).fit(X, y)


def test_no_finite_validation_loss_raises(monkeypatch):
    import types
    X, y = _data(n=150)
    monkeypatch.setattr(md, "F", types.SimpleNamespace(
        cross_entropy=lambda *a, **k: torch.tensor(float("nan"))))
    with pytest.raises(md.EarlyStoppingValidationError, match="finite validation loss"):
        _make("ANN", epochs=3, patience=10).fit(X, y)


def test_misaligned_or_duplicated_rows_raise():
    X, y = _data(n=150)
    with pytest.raises(ValueError, match="labels"):
        _make("ANN").fit(X, y[:-1])
    X.index = [0] * 150
    with pytest.raises(md.EarlyStoppingValidationError, match="unique"):
        _make("ANN").fit(X, y)


# ================================================================ select_best: final model, CV fits
def test_select_best_final_model_has_no_refit_and_curves_are_its_own(monkeypatch, tmp_path):
    X, y = _data(n=240, noisy=False)
    X.index = np.arange(5000, 5240)
    y = pd.Series(y, index=X.index)
    fold_id = rf.materialise_fold_ids(y, k=3)
    splits = rf.cv_splits_from_fold_ids(fold_id)
    fits = []
    orig = md._TorchBase._internal_split

    def spy(self, Xf, yf):
        out = orig(self, Xf, yf)
        fits.append((Xf.index.to_numpy().copy(), Xf.index[out[2]].to_numpy()))
        return out
    monkeypatch.setattr(md._TorchBase, "_internal_split", spy)
    est = _make("ANN", epochs=5)
    best, params, summary, stats, per_param = md.select_best(
        "TSV_3p", "ANN", est, {"lr": [3e-2]}, X, y, cv_splits=splits)

    assert len(fits) == 3 + 1                            # search folds (= reported folds, one-point grid), final
    for (rows, val), (tr, _) in zip(fits[:3], splits):
        assert np.array_equal(rows, X.index[tr]) and set(val) < set(rows)
    final_rows, final_val = fits[3]
    assert np.array_equal(final_rows, X.index)           # the final fit sees X_tr ...
    assert np.array_equal(best.es_val_idx_, final_val)   # ... and trains on its internal 90 % only
    assert len(best.es_train_idx_) == 216 and len(best.es_val_idx_) == 24
    assert set(best.es_train_idx_) | set(best.es_val_idx_) == set(X.index)

    md.save_torch_curves(best, tmp_path / "curves.csv")
    curves = pd.read_csv(tmp_path / "curves.csv")
    assert list(curves.columns) == ["epoch", "train_loss", "val_loss", "val_f1"]
    assert curves["val_loss"].notna().all() and len(curves) == best.n_epochs_run_
    assert np.allclose(curves["val_loss"], best.history_["val_loss"])
    rec = json.loads(json.dumps(md.early_stopping_record(best)))
    assert rec["es_val_idx"] == best.es_val_idx_.tolist() and rec["best_epoch"] == best.best_epoch_
    assert rec["refit_on_full_training_rows"] is False and rec["n_fit_rows"] == 240


def _fold_rows_by_refit(est, params, X, y, splits):
    """Reported fold metrics as the previous select_best computed them: a fresh fit per fold."""
    from sklearn.metrics import f1_score
    base = {k: getattr(est, k) for k in est.__dict__ if not k.endswith("_")}
    base.update(params)
    out = []
    for tr, va in splits:
        m = est.__class__(**base).fit(X.iloc[tr], y.iloc[tr])
        out.append((f1_score(y.iloc[tr], m.predict(X.iloc[tr]), average="macro", zero_division=0),
                    f1_score(y.iloc[va], m.predict(X.iloc[va]), average="macro", zero_division=0)))
    return out


@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_select_best_reports_the_search_fits_of_a_one_point_grid(kind, monkeypatch):
    """One-point grid -> K + 1 fits; reported folds equal a refit of each fold."""
    X, y = _data(n=180, noisy=False)
    y = pd.Series(y, index=X.index)
    splits = rf.cv_splits_from_fold_ids(rf.materialise_fold_ids(y, k=3))
    n_fits = []
    orig = md._TorchBase._internal_split
    monkeypatch.setattr(md._TorchBase, "_internal_split",
                        lambda self, Xf, yf: n_fits.append(1) or orig(self, Xf, yf))
    est = _make(kind, epochs=3, random_state=7)
    _, _, summary, _, per_param = md.select_best("TSV_3p", kind, est, {"lr": [3e-2]}, X, y, cv_splits=splits)
    assert len(n_fits) == 3 + 1
    monkeypatch.setattr(md._TorchBase, "_internal_split", orig)
    ref = _fold_rows_by_refit(est, {"lr": 3e-2}, X, y, splits)
    assert summary[["train_f1_macro", "val_f1_macro"]].to_numpy().tolist() == [list(r) for r in ref]
    assert np.isclose(per_param[0]["val_f1_macro_mean"], summary["val_f1_macro"].mean())
    assert (summary["training_time"] > 0).all()


def test_select_best_refits_the_folds_of_a_multi_point_grid(monkeypatch):
    """With several grid points the reported folds are refitted with the selected parameters (unchanged)."""
    X, y = _data(n=180, noisy=False)
    y = pd.Series(y, index=X.index)
    splits = rf.cv_splits_from_fold_ids(rf.materialise_fold_ids(y, k=3))
    n_fits = []
    orig = md._TorchBase._internal_split
    monkeypatch.setattr(md._TorchBase, "_internal_split",
                        lambda self, Xf, yf: n_fits.append(1) or orig(self, Xf, yf))
    est = _make("ANN", epochs=3, random_state=7)
    _, params, summary, _, _ = md.select_best("TSV_3p", "ANN", est, {"lr": [3e-2, 1e-3]}, X, y, cv_splits=splits)
    assert len(n_fits) == 2 * 3 + 1 + 3
    monkeypatch.setattr(md._TorchBase, "_internal_split", orig)
    ref = _fold_rows_by_refit(est, params, X, y, splits)
    assert summary[["train_f1_macro", "val_f1_macro"]].to_numpy().tolist() == [list(r) for r in ref]


def test_select_best_requires_the_materialised_folds():
    X, y = _data(n=150)
    with pytest.raises(ValueError, match="cv_splits"):
        md.select_best("TSV_3p", "ANN", _make("ANN"), {"lr": [1e-3]}, X, pd.Series(y))


def test_save_torch_curves_refuses_an_unfitted_model(tmp_path):
    with pytest.raises(ValueError, match="history"):
        md.save_torch_curves(_make("ANN"), tmp_path / "c.csv")


# ================================================================ models.py end to end (deep stage)
def test_models_main_runs_the_deep_stage_with_early_stopping(monkeypatch, tmp_path, machine_compute):
    import models as models_module
    import test_folds_seeds as tfs
    feats = list(models_module.FEATURES_ALL)
    y = tfs._labels(n=200)
    df = tfs._frame(feats, y)
    for t in tfs.TARGETS:
        df[t] = y
    tfs._FakeLoader.frame = df
    tfs._write_split(tmp_path / "split", "TSV_3p", y)
    tiny = {"ANN": dict(hidden_layers=[8], dropout=0.1, epochs=3, lr=1e-2, batch_size=64, patience=2,
                        focal_gamma=None, class_weights=None),
            "FTTransformer": dict(dim=8, depth=1, heads=2, dim_head=4, attn_dropout=0.1, ff_dropout=0.1,
                                  epochs=2, lr=1e-2, batch_size=64, patience=2,
                                  focal_gamma=None, class_weights=None)}
    monkeypatch.setattr(models_module, "UnifiedDataLoader", tfs._FakeLoader)
    monkeypatch.setitem(models_module.CONFIG, "deep_hparams", tiny)
    monkeypatch.setitem(models_module.CONFIG, "compute", machine_compute)   # explicit CPU without a GPU
    monkeypatch.setattr(models_module, "plot_confusion_matrix", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["models.py", "--models", "deep", "--targets", "TSV_3p",
                                      "--split_dir", str(tmp_path / "split"),
                                      "--folds_from", str(tmp_path / "folds"),
                                      "--output_dir", str(tmp_path / "out")])
    models_module.main(unmanifested_test_run=True)

    import joblib
    train_idx, _ = rf.redraw_split(y)
    out = tmp_path / "out"
    for name in ("ANN", "FTTransformer"):
        mdir = out / "TSV_3p" / name / "seed_42"
        rec = json.loads((mdir / f"TSV_3p_{name}__seed42_early_stopping.json").read_text())
        curves = pd.read_csv(mdir / f"TSV_3p_{name}__seed42_training_curves.csv")
        assert set(rec["es_val_idx"]) < set(train_idx.tolist())          # cohort rows of the training part
        assert rec["n_fit_rows"] == len(train_idx) and rec["n_internal_val"] == 16
        assert len(curves) == rec["n_epochs_run"] and curves["val_loss"].notna().all()
        assert rec["refit_on_full_training_rows"] is False
    ft = joblib.load(out / "best_ft_by_target.joblib")["TSV_3p"]
    assert ft.best_epoch_ == json.loads((out / "TSV_3p" / "FTTransformer" / "seed_42" /
                                         "TSV_3p_FTTransformer__seed42_early_stopping.json").read_text())["best_epoch"]
