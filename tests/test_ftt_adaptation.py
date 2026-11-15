# -*- coding: utf-8 -*-
"""
Phase 3 FT-Transformer adaptation (FT-Transformer fine-tuning fix of the
revision).

Blocking tests, numbered:

1. identity before adaptation (state dict bitwise equal to the source,
   predictions and probabilities equal to FTT zero-shot);
2. head-only isolation after one FT-head step (only ``to_logits.2.weight``
   and ``to_logits.2.bias`` change; the optimizer holds exactly these two);
3. full fine-tuning after one FT-full step (optimizer holds every parameter,
   lr 5e-5 vs 1e-4 for FT-head, encoder parameters change);
4. preprocessor and classes unchanged, ``FTClassifier.fit`` never called,
   DEV labels encoded with the source label encoder, unknown label raises;
5. save/reload reproduces predictions and probabilities exactly;
6. source untouched (state-dict SHA-256 and model-file SHA-256);
7. FTT-local: same parameter names and shapes, not the source weights,
   scaler fitted on DEV only, global vocabulary + ``<OOV>``, lr 3e-4;
8. budget and data: exactly 40 optimizer steps on exactly the DEV rows,
   budget read from the frozen config entry and asserted.

Synthetic data only, tiny networks on CPU, each test well under 60 s; the
device tests repeat 1, 2, 5, 6 and 8 on CUDA and are skipped
without a CUDA device.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import ftt_adaptation as fa  # noqa: E402
import models_deep as md  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY, vocabulary_for  # noqa: E402
from config_loader import load_config  # noqa: E402

NUM = ["a", "b"]
CAT = ["Season", "Sexe"]
ONTO = (-1, 0, 1)
TINY = dict(dim=8, depth=1, heads=2, dim_head=4)


def _data(n, seed, classes=ONTO, seasons=None, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.resize(np.asarray(classes, dtype=float), n))
    seasons = seasons or sorted(GLOBAL_VOCABULARY["Season"])
    X = pd.DataFrame({
        "a": rng.normal(size=n) + y + shift,
        "b": rng.normal(size=n) * 2 + shift,
        "Season": [seasons[i % len(seasons)] for i in range(n)],
        "Sexe": ["female" if i % 3 else "male" for i in range(n)],
    })
    return X, y


@pytest.fixture(scope="module")
def source():
    """A tiny revision E1-like FTT (global vocabulary + <OOV>, classes = ontology)."""
    X, y = _data(200, 0)
    m = md.FTClassifier(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device="cpu",
                        lr=1e-2, batch_size=64, epochs=3, patience=3, **TINY)
    return m.fit(X, y)


@pytest.fixture()
def dev_test():
    """French-like DEV (40 rows, shifted inputs, only summer/winter) and TEST sets."""
    Xd, yd = _data(40, 1, seasons=["summer", "winter"], shift=3.0)
    Xt, yt = _data(30, 2, seasons=["summer", "winter", "hot/wet"], shift=3.0)
    Xd.index = np.arange(500, 540)                       # cohort row labels, not positions
    Xt.index = np.arange(600, 630)
    return Xd, yd, Xt, yt


def _budget():
    return fa.load_budget(load_config())


def _prepare(source, strategy, Xd, yd):
    return fa.prepare_adaptation(source, strategy, Xd, yd, labels_order=ONTO, budget=_budget(), device="cpu")


def _state(model):
    return {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}


def _encoder_names(model):
    return [n for n, _ in model.named_parameters() if not n.startswith("to_logits.")]


# ================================================================ 8. frozen budget
def test_budget_is_read_from_the_frozen_config_entry_and_asserted():
    cfg = load_config()
    assert fa.load_budget(cfg) == fa.FROZEN_BUDGET == {
        "epochs": 40, "batch_size": 1024, "weight_decay": 1e-5, "grad_clip": 1.0, "seed": 42,
        "lr": {"ftt_head": 1e-4, "ftt_full": 5e-5, "ftt_local": 3e-4}}
    assert cfg["transfer"]["ftt_adaptation"]["lr"]["ftt_local"] == cfg["deep_hparams"]["FTTransformer"]["lr"]
    assert "finetune_params" not in cfg["transfer"]                  # snapshot entry (patience 6) removed
    assert set(cfg["transfer"]) == {"ftt_adaptation"}                # unused snapshot keys removed
    for path, value in [(("epochs",), 41), (("lr", "ftt_full"), 1e-4), (("lr", "ftt_head"), "1e-4"),
                        (("batch_size",), 512), (("seed",), 43), (("grad_clip",), None)]:
        bad = copy.deepcopy(cfg)
        entry = bad["transfer"]["ftt_adaptation"]
        for k in path[:-1]:
            entry = entry[k]
        entry[path[-1]] = value
        with pytest.raises(fa.FrozenBudgetError):
            fa.load_budget(bad)
    bad = copy.deepcopy(cfg); del bad["transfer"]["ftt_adaptation"]
    with pytest.raises(fa.FrozenBudgetError, match="missing"):
        fa.load_budget(bad)
    bad = copy.deepcopy(cfg); bad["deep_hparams"]["FTTransformer"]["lr"] = 1e-3
    with pytest.raises(fa.FrozenBudgetError, match="FTT-local lr / batch size"):
        fa.load_budget(bad)


@pytest.mark.parametrize("strategy", fa.STRATEGIES)
def test_exactly_forty_full_batch_steps_on_the_dev_rows_only(source, dev_test, strategy, monkeypatch):
    Xd, yd, Xt, yt = dev_test
    rows_seen, transformed = [], []
    orig_fwd = type(source.model_).forward

    def fwd(self, x_categ, x_numer, *a, **k):
        rows_seen.append(x_categ.shape[0])
        return orig_fwd(self, x_categ, x_numer, *a, **k)
    monkeypatch.setattr(type(source.model_), "forward", fwd)
    orig_tr = md.TabPreprocessor.transform

    def tr(self, X):
        transformed.append(X.index.to_numpy().copy())
        return orig_tr(self, X)
    monkeypatch.setattr(md.TabPreprocessor, "transform", tr)

    m = fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cpu")
    assert m.adapt_n_steps_ == m.adapt_epochs_ == 40 == len(m.adapt_history_["train_loss"])
    assert rows_seen == [len(Xd)] * 40                    # one full-batch step per epoch, DEV rows only
    assert len(transformed) == 1 and np.array_equal(transformed[0], Xd.index.to_numpy())
    assert np.array_equal(m.adapt_dev_idx_, Xd.index.to_numpy()) and m.adapt_n_dev_ == 40
    assert not hasattr(m, "es_val_idx_") and m.patience is None    # no validation subset, no early stopping
    assert not m.model_.training                                     # eval mode for prediction
    assert m.adapt_model_name_ == {"ftt_head": "FTT_head", "ftt_full": "FTT_full"}[strategy]
    rec = fa.adaptation_record(m)
    assert rec["n_optimizer_steps"] == 40 and len(rec["train_loss"]) == 40
    assert rec["adapted_state_sha256"] == fa.state_dict_sha256(m.model_) != rec["init_state_sha256"]


def test_dev_larger_than_the_batch_or_mismatched_raises(source):
    Xd, yd = _data(1025, 3)
    with pytest.raises(ValueError, match="full-batch"):
        fa.adapt_ftt(source, "ftt_head", Xd, yd, labels_order=ONTO, device="cpu")
    with pytest.raises(ValueError, match="rows"):
        fa.adapt_ftt(source, "ftt_head", Xd.iloc[:10], yd[:9], labels_order=ONTO, device="cpu")
    with pytest.raises(ValueError, match="strategy"):
        fa.adapt_ftt(source, "FT_head", Xd.iloc[:10], yd[:10], labels_order=ONTO, device="cpu")


# ================================================================ 1. identity before adaptation
@pytest.mark.parametrize("strategy", ["ftt_head", "ftt_full"])
def test_identity_before_adaptation(source, dev_test, strategy):
    Xd, yd, Xt, yt = dev_test
    est, trainable, inputs, y = _prepare(source, strategy, Xd, yd)
    src, cp = source.model_.state_dict(), est.model_.state_dict()
    assert src.keys() == cp.keys() and all(torch.equal(src[k], cp[k]) for k in src)   # params + buffers
    assert all(cp[k].data_ptr() != src[k].data_ptr() for k in src)                    # a deep copy
    assert fa.state_dict_sha256(est.model_) == fa.state_dict_sha256(source.model_)
    est.model_.eval()
    assert np.array_equal(est.predict(Xt), source.predict(Xt))                         # = FTT zero-shot
    assert np.array_equal(est.predict_proba(Xt), source.predict_proba(Xt))


# ================================================================ 2. head-only isolation
def test_one_ft_head_step_changes_only_the_output_head(source, dev_test, monkeypatch):
    Xd, yd, _, _ = dev_test
    est, trainable, inputs, y = _prepare(source, "ftt_head", Xd, yd)
    assert all(p.grad is None for p in est.model_.parameters())     # no gradient carried from the source
    flags = []
    orig_fwd = type(est.model_).forward

    def fwd(self, *a, **k):
        flags.append((self.training, {n for n, p in self.named_parameters() if p.requires_grad}))
        return orig_fwd(self, *a, **k)
    monkeypatch.setattr(type(est.model_), "forward", fwd)
    before = _state(est.model_)
    _, n_steps, opt = fa.train_fixed_budget(est, inputs, y, trainable, _budget(), n_epochs=1)
    after = _state(est.model_)
    assert n_steps == 1
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == set(fa.HEAD_PARAM_NAMES)
    held = [p for g in opt.param_groups for p in g["params"]]
    head = est.model_.to_logits[2]
    assert len(held) == 2 and held[0] is head.weight and held[1] is head.bias
    assert opt.param_groups[0]["lr"] == 1e-4 and opt.param_groups[0]["weight_decay"] == 1e-5
    # during training: train mode, only the head requires grad (LayerNorm of to_logits frozen)
    assert flags == [(True, set(fa.HEAD_PARAM_NAMES))]
    no_grad = {n for n, p in est.model_.named_parameters() if p.grad is None}
    assert no_grad == {n for n, _ in est.model_.named_parameters()} - set(fa.HEAD_PARAM_NAMES)
    assert "to_logits.0.weight" in no_grad and "to_logits.0.bias" in no_grad
    # deployed model: eval mode, requires_grad flags as on the source (prediction kernel path)
    assert not est.model_.training and all(p.requires_grad for p in est.model_.parameters())


# ================================================================ 3. full fine-tuning
def test_one_ft_full_step_trains_every_parameter_with_the_reduced_lr(source, dev_test):
    Xd, yd, _, _ = dev_test
    est, trainable, inputs, y = _prepare(source, "ftt_full", Xd, yd)
    before = _state(est.model_)
    _, n_steps, opt = fa.train_fixed_budget(est, inputs, y, trainable, _budget(), n_epochs=1)
    after = _state(est.model_)
    held = {id(p) for g in opt.param_groups for p in g["params"]}
    assert n_steps == 1 and held == {id(p) for p in est.model_.parameters()}
    assert opt.param_groups[0]["lr"] == 5e-5 != fa.FROZEN_BUDGET["lr"]["ftt_head"]   # snapshot bug guarded
    enc = _encoder_names(est.model_)
    assert any(not torch.equal(before[n], after[n]) for n in enc)                    # encoder moved
    assert all(p.requires_grad for p in est.model_.parameters())


@pytest.mark.parametrize("strategy,lr", [("ftt_head", 1e-4), ("ftt_full", 5e-5)])
def test_learning_rate_per_strategy_is_honoured(source, dev_test, strategy, lr, monkeypatch):
    Xd, yd, _, _ = dev_test
    seen = []
    orig = torch.optim.AdamW.__init__

    def spy(self, params, lr=1e-3, weight_decay=1e-2, **k):
        seen.append((lr, weight_decay))
        return orig(self, params, lr=lr, weight_decay=weight_decay, **k)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", spy)
    m = fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cpu")
    assert seen == [(lr, 1e-5)] and m.adapt_lr_ == m.lr == lr


# ================================================================ 4. preprocessor and classes unchanged
@pytest.mark.parametrize("strategy", ["ftt_head", "ftt_full"])
def test_source_preprocessor_and_classes_kept_and_fit_never_called(source, dev_test, strategy, monkeypatch):
    Xd, yd, _, _ = dev_test
    calls = []
    monkeypatch.setattr(md.FTClassifier, "fit", lambda self, X, y: calls.append("FTClassifier.fit"))
    monkeypatch.setattr(md.TabPreprocessor, "fit", lambda self, X: calls.append("TabPreprocessor.fit"))
    monkeypatch.setattr(fa.LabelEncoder, "fit", lambda self, y: calls.append("LabelEncoder.fit"))
    snap = (copy.deepcopy(source.pp_.vocabulary_), source.pp_.scaler.mean_.copy(),
            source.pp_.scaler.scale_.copy(), source.le_y_.classes_.copy(), np.asarray(source.classes_).copy())
    m = fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cpu")
    assert calls == []
    for pp in (source.pp_, m.pp_):
        assert pp.vocabulary_ == snap[0] and pp.oov_policy == "reserved_index"
        assert np.array_equal(pp.scaler.mean_, snap[1]) and np.array_equal(pp.scaler.scale_, snap[2])
        assert pp.train_categories_ == source.pp_.train_categories_ and pp.n_categories == source.pp_.n_categories
    assert m.pp_ is not source.pp_ and m.le_y_ is not source.le_y_
    for est in (source, m):
        assert np.array_equal(est.le_y_.classes_, snap[3]) and np.array_equal(est.classes_, snap[4])
    with pytest.raises(md.RevisionNotImplementedError, match="re-initialise the network"):
        fa.AdaptedFTClassifier.fit(m, Xd, yd)


@pytest.mark.parametrize("strategy", ["ftt_head", "ftt_full"])
def test_dev_labels_encoded_with_the_source_label_encoder(source, dev_test, strategy):
    Xd, yd, _, _ = dev_test
    sub = yd != 1                                                  # DEV without class 1
    est, _, _, y = _prepare(source, strategy, Xd[sub], yd[sub])
    assert torch.equal(y, torch.tensor(source.le_y_.transform(yd[sub]), dtype=torch.long))
    assert est.model_.to_logits[2].out_features == 3               # output over the source classes
    for bad in (2.0, np.nan, 0.5):
        y_bad = yd.copy(); y_bad[0] = bad
        with pytest.raises(ValueError, match="absent from the source classes"):
            fa.adapt_ftt(source, strategy, Xd, y_bad, labels_order=ONTO, device="cpu")


def test_source_whose_classes_are_not_the_ontology_is_rejected(source, dev_test):
    Xd, yd, _, _ = dev_test
    with pytest.raises(ValueError, match="ontology"):
        fa.adapt_ftt(source, "ftt_head", Xd, yd, labels_order=(-3, -2, -1, 0, 1, 2, 3), device="cpu")
    legacy = copy.deepcopy(source)
    del legacy.pp_.train_categories_                               # preprocessor state before the category fix
    with pytest.raises(md.LegacyModelStateError):
        fa.adapt_ftt(legacy, "ftt_head", Xd, yd, labels_order=ONTO, device="cpu")
    for seed in (43, 44, None, True):                              # E1 seed-42 source only
        other = copy.deepcopy(source)
        other.random_state = seed
        with pytest.raises(ValueError, match="seed"):
            fa.adapt_ftt(other, "ftt_full", Xd, yd, labels_order=ONTO, device="cpu")
    adapted = fa.adapt_ftt(source, "ftt_head", Xd, yd, labels_order=ONTO, device="cpu")
    assert fa.adaptation_record(adapted)["source_random_state"] == 42
    with pytest.raises(TypeError, match="not an adapted model"):
        fa.adapt_ftt(adapted, "ftt_full", Xd, yd, labels_order=ONTO, device="cpu")


# ================================================================ 5. save / reload
@pytest.mark.parametrize("strategy", fa.STRATEGIES)
def test_save_reload_reproduces_predictions_exactly(source, dev_test, strategy, tmp_path):
    Xd, yd, Xt, _ = dev_test
    m = fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cpu")
    path = tmp_path / f"{strategy}.joblib"
    joblib.dump(m, path)
    r = joblib.load(path)
    assert isinstance(r, fa.AdaptedFTClassifier) and fa.state_dict_sha256(r.model_) == m.adapted_state_sha256_
    assert np.array_equal(r.predict(Xt), m.predict(Xt))
    assert np.array_equal(r.predict_proba(Xt), m.predict_proba(Xt))
    assert r.adapt_history_ == m.adapt_history_


# ================================================================ 6. source untouched
def test_source_state_dict_and_model_file_untouched(source, dev_test, tmp_path):
    Xd, yd, Xt, _ = dev_test
    src_file = tmp_path / "best_ft_by_target.joblib"
    joblib.dump({"TSV_3p": source}, src_file)
    file_sha, state_sha = fa.file_sha256(src_file), fa.state_dict_sha256(source.model_)
    proba = source.predict_proba(Xt)
    models = [fa.adapt_ftt(source, s, Xd, yd, labels_order=ONTO, device="cpu") for s in fa.STRATEGIES]
    assert fa.state_dict_sha256(source.model_) == state_sha
    assert fa.file_sha256(src_file) == file_sha
    redump = tmp_path / "redump.joblib"                             # the in-memory object is unchanged too
    joblib.dump({"TSV_3p": source}, redump)
    assert fa.file_sha256(redump) == file_sha
    assert np.array_equal(source.predict_proba(Xt), proba)
    assert all(p.requires_grad for p in source.model_.parameters())
    assert all(m.source_state_sha256_ == state_sha != m.adapted_state_sha256_ for m in models)


def test_adaptation_raises_if_the_source_changes(source, dev_test, monkeypatch):
    Xd, yd, _, _ = dev_test
    src = copy.deepcopy(source)
    orig = fa.train_fixed_budget

    def corrupting(est, *a, **k):
        out = orig(est, *a, **k)
        with torch.no_grad():
            src.model_.to_logits[2].bias.add_(1.0)
        return out
    monkeypatch.setattr(fa, "train_fixed_budget", corrupting)
    with pytest.raises(RuntimeError, match="source"):
        fa.adapt_ftt(src, "ftt_full", Xd, yd, labels_order=ONTO, device="cpu")


# ================================================================ non-finite guards
@pytest.mark.parametrize("strategy", fa.STRATEGIES)
def test_non_finite_loss_stops_the_adaptation(source, dev_test, monkeypatch, strategy):
    """An injected NaN loss (diverged adaptation) never yields an adapted model, hence no prediction."""
    Xd, yd, _, _ = dev_test
    real = fa.F.cross_entropy
    monkeypatch.setattr(fa.F, "cross_entropy", lambda logits, y: real(logits, y) * float("nan"))
    with pytest.raises((RuntimeError, FloatingPointError), match="non-finite"):
        fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cpu")


def test_non_finite_loss_history_and_adapted_weights_are_refused(source, dev_test, monkeypatch):
    Xd, yd, _, _ = dev_test
    est, trainable, inputs, y = _prepare(source, "ftt_full", Xd, yd)
    monkeypatch.setattr(fa.torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)   # NaN reaches the history
    real = fa.F.cross_entropy
    monkeypatch.setattr(fa.F, "cross_entropy", lambda logits, y: real(logits, y) * float("nan"))
    with pytest.raises(FloatingPointError, match="training loss"):
        fa.train_fixed_budget(est, inputs, y, trainable, _budget(), n_epochs=2)
    monkeypatch.undo()
    orig = fa.train_fixed_budget

    def diverging(est, *a, **k):
        out = orig(est, *a, **k)
        with torch.no_grad():
            est.model_.to_logits[2].weight[0, 0] = float("inf")
        return out
    monkeypatch.setattr(fa, "train_fixed_budget", diverging)
    with pytest.raises(FloatingPointError, match="adapted weights"):
        fa.adapt_ftt(source, "ftt_head", Xd, yd, labels_order=ONTO, device="cpu")


def test_non_finite_logits_never_become_classes_or_probabilities(source, dev_test):
    """argmax over NaN logits would return class index 0 and softmax NaN probabilities (e.g. a
    missing numeric input); both raise, and the predictions export refuses invalid probabilities."""
    import phase3_runner as p3
    _, _, Xt, _ = dev_test
    bad = copy.deepcopy(source)
    with torch.no_grad():
        bad.model_.to_logits[2].bias[0] = float("nan")
    for call in (bad.predict, bad.predict_proba):
        with pytest.raises(md.NonFiniteOutputError, match="non-finite logits"):
            call(Xt)
    missing = Xt.copy()
    missing.loc[missing.index[0], "a"] = np.nan
    with pytest.raises(md.NonFiniteOutputError):
        source.predict(missing)
    good = source.predict_proba(Xt)
    p3.check_probabilities(good)
    for broken, match in ((np.where(np.arange(good.size).reshape(good.shape) == 0, np.nan, good), "non-finite"),
                          (good * 1.5, "sum to 1"), (good - 0.5, r"\[0, 1\]")):
        with pytest.raises((FloatingPointError, ValueError), match=match):
            p3.check_probabilities(broken)
    # component bounds are exact: the 1e-4 tolerance is for the row sum only
    for near in ([[-0.00005, 1.00005]], [[0.5, 0.5, -1e-12, 1e-12]], [[1.0 + 1e-9, 0.0]]):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            p3.check_probabilities(np.array(near))
    p3.check_probabilities(np.array([[0.0, 1.0], [0.5, 0.5 + 5e-5]]))   # sum within tolerance: accepted


def test_ftt_local_strategy_is_not_available(source, dev_test):
    """FTT-local (fresh initialisation) is not part of the published experiments."""
    Xd, yd, _, _ = dev_test
    assert fa.STRATEGIES == ("ftt_head", "ftt_full")
    with pytest.raises(ValueError, match="unknown FTT adaptation strategy"):
        fa.adapt_ftt(source, "ftt_local", Xd, yd, labels_order=ONTO, device="cpu")


# ================================================================ device (GPU / CPU)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")


def test_device_defaults_to_the_source_network_device(source, dev_test):
    Xd, yd, _, _ = dev_test
    for s in fa.STRATEGIES:
        assert fa.resolve_device(source, s) == "cpu"
        est, _, (x_cat, x_num), y = fa.prepare_adaptation(source, s, Xd, yd, labels_order=ONTO, budget=_budget())
        assert est.device == "cpu" and x_cat.device.type == x_num.device.type == y.device.type == "cpu"
        assert all(p.device.type == "cpu" for p in est.model_.parameters())


@pytest.mark.parametrize("strategy", ["ftt_head", "ftt_full"])
def test_ft_head_full_refuse_a_device_other_than_the_source(source, dev_test, strategy, monkeypatch):
    Xd, yd, _, _ = dev_test
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="source device"):
            fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        fa.adapt_ftt(source, strategy, Xd, yd, labels_order=ONTO, device="cuda")


def test_training_loss_history_is_the_per_step_loss(source, dev_test, monkeypatch):
    Xd, yd, _, _ = dev_test
    seen = []
    orig = torch.nn.functional.cross_entropy

    def spy(*a, **k):
        loss = orig(*a, **k)
        seen.append(float(loss.detach()))
        return loss
    monkeypatch.setattr(fa.F, "cross_entropy", spy)
    est, trainable, inputs, y = _prepare(source, "ftt_full", Xd, yd)
    hist, n, _ = fa.train_fixed_budget(est, inputs, y, trainable, _budget(), n_epochs=3)
    assert n == 3 and hist["epoch"] == [1, 2, 3]
    assert hist["train_loss"] == seen and all(isinstance(v, float) for v in hist["train_loss"])


@pytest.fixture(scope="module")
def cuda_source():
    """The tiny E1-like FTT trained on the GPU (as the E1 source)."""
    X, y = _data(200, 0)
    m = md.FTClassifier(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device="cuda",
                        lr=1e-2, batch_size=64, epochs=3, patience=3, **TINY)
    return m.fit(X, y)


@CUDA
@pytest.mark.parametrize("strategy", ["ftt_head", "ftt_full"])
def test_cuda_identity_before_adaptation(cuda_source, dev_test, strategy):
    Xd, yd, Xt, _ = dev_test
    est, _, inputs, y = fa.prepare_adaptation(cuda_source, strategy, Xd, yd, labels_order=ONTO, budget=_budget())
    assert est.device.startswith("cuda") and all(t.is_cuda for t in (*inputs, y))
    assert fa.state_dict_sha256(est.model_) == fa.state_dict_sha256(cuda_source.model_)
    est.model_.eval()
    assert np.array_equal(est.predict(Xt), cuda_source.predict(Xt))
    assert np.array_equal(est.predict_proba(Xt), cuda_source.predict_proba(Xt))


@CUDA
def test_cuda_one_ft_head_step_changes_only_the_output_head(cuda_source, dev_test):
    Xd, yd, _, _ = dev_test
    est, trainable, inputs, y = fa.prepare_adaptation(cuda_source, "ftt_head", Xd, yd, labels_order=ONTO,
                                                      budget=_budget())
    before = _state(est.model_)
    _, n_steps, opt = fa.train_fixed_budget(est, inputs, y, trainable, _budget(), n_epochs=1)
    after = _state(est.model_)
    assert n_steps == 1 and {k for k in before if not torch.equal(before[k], after[k])} == set(fa.HEAD_PARAM_NAMES)
    assert opt.param_groups[0]["lr"] == 1e-4


@CUDA
@pytest.mark.parametrize("strategy", fa.STRATEGIES)
def test_cuda_adaptation_budget_source_untouched_and_save_reload(cuda_source, dev_test, strategy, tmp_path):
    Xd, yd, Xt, _ = dev_test
    state_sha = fa.state_dict_sha256(cuda_source.model_)
    m = fa.adapt_ftt(cuda_source, strategy, Xd, yd, labels_order=ONTO)
    assert m.device.startswith("cuda") and all(p.is_cuda for p in m.model_.parameters())
    assert m.adapt_n_steps_ == 40 == len(m.adapt_history_["train_loss"])
    assert m.adapt_lr_ == fa.FROZEN_BUDGET["lr"][strategy]
    assert fa.state_dict_sha256(cuda_source.model_) == state_sha != m.adapted_state_sha256_
    path = tmp_path / f"{strategy}.joblib"
    joblib.dump(m, path)
    r = joblib.load(path)
    assert fa.state_dict_sha256(r.model_) == m.adapted_state_sha256_
    assert np.array_equal(r.predict(Xt), m.predict(Xt))
    assert np.array_equal(r.predict_proba(Xt), m.predict_proba(Xt))
