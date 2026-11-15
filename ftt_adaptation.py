"""
Phase 3 FT-Transformer adaptation (FT-head and FT-full fine-tuning).

Real adaptation of the E1 seed-42 FT-Transformer of a target (the *source*)
to a French DEV set, replacing the pre-revision ``clone_ft_for_finetune`` +
``FTClassifier.fit`` path, which re-initialised the network and the
preprocessing and used the head learning rate for both modes.

Two strategies, each fitted on the DEV rows of one serialised split only:

* ``ftt_head`` (FT-head): deep copy of the source network; the source
  preprocessor ``pp_``, label encoder ``le_y_`` and ``classes_`` are kept
  (never refitted); only the output head ``model_.to_logits[2]`` (weight and
  bias) is trainable and passed to the optimizer; lr 1e-4.
* ``ftt_full`` (FT-full): deep copy of the source network, source
  ``pp_``/``le_y_``/``classes_`` kept; every parameter trainable; lr 5e-5.

One fixed, pre-declared budget (no early stopping, unlike the from-scratch
FTT fits): 40 epochs, batch 1 024 (every DEV set is smaller: one full-batch step per
epoch, hence exactly 40 optimizer steps), a new AdamW (weight decay 1e-5)
over the trainable parameters only, gradient-norm clipping 1.0, unweighted
cross-entropy, ``train()`` mode for the whole network during training and
``eval()`` for prediction, seeds 42 (``random``, ``numpy``, ``torch``) set
immediately before each adaptation fit. No early stopping, no validation
subset, no checkpoint selection: the deployed model is the state after the
last epoch. The budget is read from the single frozen entry
``transfer.ftt_adaptation`` of ``config.yaml`` and asserted
(``load_budget``).

``FTClassifier.fit`` is never called; the adapted estimator is an
``AdaptedFTClassifier`` whose ``fit`` raises. The source estimator is never
modified (adaptation works on deep copies; its state-dict hash and its
preprocessing/label state are checked after every adaptation).
"""
from __future__ import annotations

import copy
import hashlib
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tab_transformer_pytorch import FTTransformer

from category_normalisation import vocabulary_for
from config_loader import load_config
from models_deep import (
    FTClassifier,
    RevisionNotImplementedError,
    check_ftt_revision_schema,
)


# ======================================================================================
# FROZEN CONSTANTS
# ======================================================================================
STRATEGIES = ("ftt_head", "ftt_full")

# Model names of the predictions files.
MODEL_NAMES = {"ftt_head": "FTT_head", "ftt_full": "FTT_full"}

# The one adaptation budget, identical across strategies, regimes, cohorts and targets. It equals the
# transfer.ftt_adaptation entry of config.yaml (asserted by load_budget), which also holds the lr of
# the FTT-local strategy (fresh initialisation), not part of the published experiments.
FROZEN_BUDGET = {
    "epochs": 40,
    "batch_size": 1024,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "seed": 42,
    "lr": {"ftt_head": 1e-4, "ftt_full": 5e-5, "ftt_local": 3e-4},
}

# Output head of tab_transformer_pytorch.FTTransformer 0.4.2
# (to_logits = Sequential(LayerNorm, ReLU, Linear)), identified by module path.
HEAD_INDEX = 2
HEAD_PARAM_NAMES = ("to_logits.2.weight", "to_logits.2.bias")

class FrozenBudgetError(ValueError):
    """The ``transfer.ftt_adaptation`` config entry differs from the frozen budget."""


# ======================================================================================
# BUDGET AND HASHES
# ======================================================================================
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_budget(config=None) -> dict:
    """
    Read the adaptation budget from ``config["transfer"]["ftt_adaptation"]``
    and assert it equals ``FROZEN_BUDGET``. The FTT-local learning rate and
    the batch size must also equal the from-scratch FTT values of
    ``deep_hparams.FTTransformer``.

    Raises
    ------
    FrozenBudgetError
        Missing entry, non-numeric value (e.g. YAML 1.1 reads ``1e-4`` as a
        string) or any value different from the frozen one.
    """
    cfg = load_config() if config is None else config
    try:
        entry = cfg["transfer"]["ftt_adaptation"]
    except (KeyError, TypeError):
        raise FrozenBudgetError("config entry transfer.ftt_adaptation is missing") from None
    leaves = [v for k, v in entry.items() if k != "lr"] + list(dict(entry.get("lr") or {}).values())
    if not all(_is_number(v) for v in leaves):
        raise FrozenBudgetError(f"transfer.ftt_adaptation holds non-numeric values: {entry!r}")
    if entry != FROZEN_BUDGET:
        raise FrozenBudgetError(
            f"transfer.ftt_adaptation = {entry!r} differs from the frozen budget "
            f"{FROZEN_BUDGET!r}.")
    ftt = cfg.get("deep_hparams", {}).get("FTTransformer", {})
    if ftt.get("lr") != FROZEN_BUDGET["lr"]["ftt_local"] or ftt.get("batch_size") != FROZEN_BUDGET["batch_size"]:
        raise FrozenBudgetError(
            "FTT-local lr / batch size must equal deep_hparams.FTTransformer lr / batch_size")
    return copy.deepcopy(FROZEN_BUDGET)


def state_dict_sha256(module: nn.Module) -> str:
    """SHA-256 of a state dict (parameters and buffers): names, dtypes, shapes and raw bytes, by sorted name."""
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        a = t.detach().cpu().contiguous()
        for part in (name, str(a.dtype), str(tuple(a.shape))):
            b = part.encode()
            h.update(len(b).to_bytes(8, "little")); h.update(b)
        raw = a.numpy().tobytes()
        h.update(len(raw).to_bytes(8, "little")); h.update(raw)
    return h.hexdigest()


def file_sha256(path) -> str:
    """SHA-256 of a file (e.g. the saved source model)."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_fingerprint(source) -> tuple:
    """State-dict hash + bytes of the source preprocessing and label state."""
    return (state_dict_sha256(source.model_),
            hashlib.sha256(pickle.dumps((source.pp_, source.le_y_, np.asarray(source.classes_)))).hexdigest())


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ======================================================================================
# ADAPTED ESTIMATOR
# ======================================================================================
class AdaptedFTClassifier(FTClassifier):
    """
    FT-Transformer adapted to a French DEV set.

    Built only by ``adapt_ftt``; ``predict`` / ``predict_proba`` are those of
    ``FTClassifier``. ``fit`` raises: refitting would re-initialise the network
    and the preprocessing.

    Adaptation attributes: ``adapt_strategy_``, ``adapt_model_name_``,
    ``adapt_budget_``, ``adapt_lr_``, ``adapt_epochs_``, ``adapt_n_steps_``,
    ``adapt_n_dev_``, ``adapt_dev_idx_`` (index labels of the DEV rows),
    ``adapt_trainable_params_``, ``adapt_history_`` (training loss per epoch),
    ``adapt_dev_oov_counts_``, ``init_state_sha256_``, ``adapted_state_sha256_``,
    ``source_state_sha256_``, ``source_random_state_``.
    """

    def fit(self, X, y):
        raise RevisionNotImplementedError(
            "AdaptedFTClassifier is built by ftt_adaptation.adapt_ftt only; FTClassifier.fit "
            "would re-initialise the network and the preprocessing.")


def output_head(model: nn.Module, n_classes: int) -> nn.Linear:
    """
    Output head ``model.to_logits[2]`` of an ``FTTransformer``, asserted to be
    an ``nn.Linear`` with ``out_features == n_classes`` (never located by
    searching for an ``out_features`` value).
    """
    if not isinstance(model, FTTransformer):
        raise TypeError(f"expected a tab_transformer_pytorch.FTTransformer, got {type(model).__name__}")
    try:
        head = model.to_logits[HEAD_INDEX]
    except (AttributeError, IndexError, TypeError):
        raise ValueError("FTTransformer has no to_logits[2] output head") from None
    if not isinstance(head, nn.Linear) or head.out_features != n_classes:
        raise ValueError(f"to_logits[2] is {head!r}; expected nn.Linear(dim, {n_classes})")
    return head


def _check_strategy(strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown FTT adaptation strategy {strategy!r}; expected one of {STRATEGIES}")


def _check_source(source, labels_order, seed: int) -> None:
    """The source must be the revision E1 seed-42 FTT, classes = full target ontology."""
    if not isinstance(source, FTClassifier) or isinstance(source, AdaptedFTClassifier):
        raise TypeError("the source must be a fitted E1 FTClassifier (not an adapted model)")
    src_seed = getattr(source, "random_state", None)
    if isinstance(src_seed, bool) or not isinstance(src_seed, (int, np.integer)) or int(src_seed) != seed:
        raise ValueError(f"source FTT model seed is {src_seed!r}; the adaptation starts from the "
                         f"E1 seed-{seed} FTT only")
    pp = getattr(source, "pp_", None)
    check_ftt_revision_schema(source, vocabulary_for(pp.cat_cols) if pp is not None else {}, "source")
    classes = np.asarray(source.classes_)
    expected = np.asarray(sorted(labels_order), dtype=float)
    if (classes.shape != expected.shape or not np.array_equal(classes.astype(float), expected)
            or not np.array_equal(np.asarray(source.le_y_.classes_), classes)):
        raise ValueError(f"source classes {classes.tolist()} differ from the target ontology "
                         f"{expected.tolist()} (source classes = full ontology)")
    output_head(source.model_, len(classes))


def _check_dev(X_dev: pd.DataFrame, y_dev, budget: dict) -> np.ndarray:
    y = pd.Series(y_dev).to_numpy()
    if len(X_dev) != len(y):
        raise ValueError(f"X_dev has {len(X_dev)} rows but y_dev has {len(y)} labels")
    if len(y) == 0:
        raise ValueError("empty DEV set")
    if not X_dev.index.is_unique:
        raise ValueError("X_dev index must be unique (adapt_dev_idx_ records index labels)")
    if len(y) > budget["batch_size"]:
        raise ValueError(
            f"DEV set of {len(y)} rows > batch size {budget['batch_size']}: the fixed budget "
            f"is one full-batch step per epoch (exactly {budget['epochs']} steps).")
    return y


def _encode_labels(le: LabelEncoder, y: np.ndarray) -> torch.Tensor:
    """Encode DEV labels with ``le``; a label absent from its classes raises (never added)."""
    classes = np.asarray(le.classes_)
    try:
        known = np.isin(np.asarray(y, dtype=float), classes.astype(float))
    except (TypeError, ValueError):
        raise ValueError("DEV labels must be numeric") from None
    if not known.all():
        raise ValueError(f"DEV labels {sorted(set(map(str, y[~known])))} are absent from the source "
                         f"classes {classes.tolist()}")
    return torch.tensor(le.transform(np.asarray(y, dtype=float).astype(classes.dtype)), dtype=torch.long)


def _new_estimator(source, strategy: str, budget: dict, device) -> AdaptedFTClassifier:
    """Unfitted adapted estimator carrying the source constructor parameters and the adaptation budget."""
    est = AdaptedFTClassifier.__new__(AdaptedFTClassifier)
    est.__dict__.update({k: copy.deepcopy(v) for k, v in vars(source).items() if not k.endswith("_")})
    est.lr = float(budget["lr"][strategy])
    est.epochs = int(budget["epochs"])
    est.batch_size = int(budget["batch_size"])
    est.patience = None                    # no early stopping
    est.focal_gamma = None                 # unweighted cross-entropy
    est.class_weights = None
    est.random_state = int(budget["seed"])
    est.device = device
    return est


def _canonical_device(device) -> torch.device:
    """``torch.device`` with an explicit CUDA index (``"cuda"`` → ``cuda:<current>``)."""
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        if d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
    return d


def resolve_device(source, strategy: str, device=None) -> str:
    """
    Device of an adaptation fit: the device holding the source network
    (the GPU when the E1 source was trained there, CPU otherwise).

    An explicit ``device`` must be the source's device, because the identity test compares
    the unadapted copy with FTT zero-shot bitwise, and CPU and CUDA kernels
    differ in the last bits.
    """
    try:
        src = _canonical_device(next(source.model_.parameters()).device)
    except StopIteration:
        raise ValueError("source FT-Transformer has no parameters") from None
    if device is None:
        return str(src)
    d = _canonical_device(device)
    if strategy in ("ftt_head", "ftt_full") and d != src:
        raise ValueError(f"{strategy} must run on the source device {src} (got {d}): identity with "
                         "FTT zero-shot is device-specific")
    return str(d)


def prepare_adaptation(source, strategy: str, X_dev: pd.DataFrame, y_dev, *, labels_order,
                       budget: dict, device=None):
    """
    Build the adapted estimator before any optimizer step.

    Returns
    -------
    est : AdaptedFTClassifier
        With ``pp_``, ``le_y_``, ``classes_`` and ``model_`` set; FT-head and
        FT-full: deep copies of the source (bitwise-identical weights).
    trainable : list of (name, parameter)
        Parameters passed to the optimizer (FT-head: the two head tensors).
    inputs : tuple(torch.LongTensor, torch.FloatTensor)
        DEV inputs on ``est.device`` (moved once; the 40 full-batch steps
        then run without host/device transfers).
    y : torch.LongTensor
        DEV labels encoded with ``est.le_y_``.
    """
    _check_strategy(strategy)
    _check_source(source, labels_order, budget["seed"])
    y_np = _check_dev(X_dev, y_dev, budget)
    device = resolve_device(source, strategy, device)
    _set_seeds(budget["seed"])                      # immediately before each adaptation fit
    est = _new_estimator(source, strategy, budget, device)

    # source preprocessing and labels kept, never refitted
    est.pp_ = copy.deepcopy(source.pp_)
    est.le_y_ = copy.deepcopy(source.le_y_)
    est.classes_ = est.le_y_.classes_
    est.model_ = copy.deepcopy(source.model_).to(device)

    x_cat, x_num = est.pp_.transform(X_dev)
    est.adapt_dev_oov_counts_ = dict(est.pp_.last_oov_counts_)
    y = _encode_labels(est.le_y_, y_np).to(device)

    head = output_head(est.model_, len(est.classes_))
    # No gradient is carried over from the source fit; requires_grad stays True
    # outside training, as on the source: torch.matmul chooses its kernel path
    # from requires_grad even under no_grad, so frozen flags on the deployed
    # model would change the last bits of its outputs (identity with FTT zero-shot).
    # The encoder is frozen during training only (train_fixed_budget).
    est.model_.zero_grad(set_to_none=True)
    for p in est.model_.parameters():
        p.requires_grad_(True)
    if strategy == "ftt_head":
        head_ids = {id(p) for p in head.parameters()}
        trainable = [(n, p) for n, p in est.model_.named_parameters() if id(p) in head_ids]
    else:
        trainable = list(est.model_.named_parameters())
    if strategy == "ftt_head" and tuple(n for n, _ in trainable) != HEAD_PARAM_NAMES:
        raise RuntimeError(f"FT-head trainable parameters {[n for n, _ in trainable]} != {HEAD_PARAM_NAMES}")
    return est, trainable, (x_cat.to(device), x_num.to(device)), y


def train_fixed_budget(est, inputs, y, trainable, budget: dict, *, n_epochs=None):
    """
    Fixed-budget training loop (no validation, last-epoch model).

    One full-batch step per epoch on the DEV rows; a new AdamW over the
    trainable parameters only (lr ``est.lr``, weight decay from the budget);
    gradient-norm clipping; unweighted cross-entropy; ``train()`` mode during
    training, ``eval()`` afterwards. ``n_epochs`` (tests only) overrides the
    budget's epoch count.

    Returns
    -------
    history : dict
        ``epoch``, ``train_loss``.
    n_steps : int
        Optimizer steps actually performed (counted by a step hook).
    optimizer : torch.optim.AdamW
    """
    epochs = budget["epochs"] if n_epochs is None else int(n_epochs)
    params = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(params, lr=est.lr, weight_decay=budget["weight_decay"])
    n_steps = [0]
    optimizer.register_step_post_hook(lambda opt, args, kwargs: n_steps.__setitem__(0, n_steps[0] + 1))
    losses = []
    model = est.model_
    trainable_ids = {id(p) for p in params}
    for p in model.parameters():                     # FT-head: encoder frozen during training
        p.requires_grad_(id(p) in trainable_ids)
    try:
        for ep in range(epochs):
            model.train()
            logits = model(*inputs)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            # a non-finite gradient norm (diverged loss or inputs) stops the fit
            torch.nn.utils.clip_grad_norm_(params, budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())             # read back once after training (no per-epoch GPU sync)
    finally:
        for p in model.parameters():                 # deployed model: flags as on the source
            p.requires_grad_(True)
        model.eval()
    train_loss = torch.stack(losses).cpu().tolist() if losses else []
    history = {"epoch": list(range(1, len(train_loss) + 1)), "train_loss": [float(v) for v in train_loss]}
    if not np.isfinite(history["train_loss"]).all():
        raise FloatingPointError(f"non-finite training loss during the FTT adaptation: {history['train_loss']}")
    return history, n_steps[0], optimizer


def adapt_ftt(source, strategy: str, X_dev: pd.DataFrame, y_dev, *, labels_order,
              device=None, config=None) -> AdaptedFTClassifier:
    """
    Adapt the source FT-Transformer to one DEV set with one strategy.

    Parameters
    ----------
    source : FTClassifier
        E1 seed-42 FTT of the target (revision preprocessor, classes = ontology).
    strategy : {"ftt_head", "ftt_full"}
    X_dev, y_dev :
        DEV rows of the serialised split only (no TEST row, no validation subset).
    labels_order : sequence
        Full target ontology (``revision_stats.ontology(target)``).
    device : str or None
        Defaults to the device holding the source network (``resolve_device``);
        FT-head and FT-full must run on it.
    config : dict or None
        Parsed ``config.yaml`` (default: loaded); its budget entry is asserted.

    Returns
    -------
    AdaptedFTClassifier
        The model after the last epoch, in ``eval`` mode.

    Raises
    ------
    RuntimeError
        The number of optimizer steps differs from the budget, or the source
        model changed during the adaptation.
    FloatingPointError
        A non-finite loss, gradient norm or adapted weight (no adapted model
        is returned, hence no prediction file is written).
    """
    budget = load_budget(config)
    before = _source_fingerprint(source)
    est, trainable, inputs, y = prepare_adaptation(source, strategy, X_dev, y_dev,
                                                   labels_order=labels_order, budget=budget, device=device)
    init_sha = state_dict_sha256(est.model_)
    history, n_steps, _ = train_fixed_budget(est, inputs, y, trainable, budget)
    if n_steps != budget["epochs"]:
        raise RuntimeError(f"{n_steps} optimizer steps instead of {budget['epochs']}")
    if _source_fingerprint(source) != before:
        raise RuntimeError("the source FT-Transformer was modified during adaptation")
    bad = [n for n, t in est.model_.state_dict().items()
           if torch.is_floating_point(t) and not bool(torch.isfinite(t).all())]
    if bad:
        raise FloatingPointError(f"non-finite adapted weights {bad[:5]} after the FTT adaptation ({strategy})")

    est.adapt_strategy_ = strategy
    est.adapt_model_name_ = MODEL_NAMES[strategy]
    est.adapt_budget_ = copy.deepcopy(budget)
    est.adapt_lr_ = est.lr
    est.adapt_epochs_ = budget["epochs"]
    est.adapt_n_steps_ = n_steps
    est.adapt_n_dev_ = int(len(X_dev))
    est.adapt_dev_idx_ = np.asarray(X_dev.index)
    est.adapt_trainable_params_ = [n for n, _ in trainable]
    est.adapt_history_ = history
    est.init_state_sha256_ = init_sha
    est.adapted_state_sha256_ = state_dict_sha256(est.model_)
    est.source_state_sha256_ = before[0]
    est.source_random_state_ = int(source.random_state)
    return est


def adaptation_record(est: AdaptedFTClassifier) -> dict:
    """JSON-serialisable record of one adaptation fit (training-loss history, hashes, budget)."""
    return {
        "model": est.adapt_model_name_,
        "strategy": est.adapt_strategy_,
        "budget": est.adapt_budget_,
        "lr": est.adapt_lr_,
        "epochs": est.adapt_epochs_,
        "n_optimizer_steps": est.adapt_n_steps_,
        "n_dev": est.adapt_n_dev_,
        "dev_idx": [x.item() if hasattr(x, "item") else x for x in est.adapt_dev_idx_],
        "trainable_params": list(est.adapt_trainable_params_),
        "dev_oov_counts": dict(est.adapt_dev_oov_counts_),
        "train_loss": list(est.adapt_history_["train_loss"]),
        "source_random_state": est.source_random_state_,
        "source_state_sha256": est.source_state_sha256_,
        "init_state_sha256": est.init_state_sha256_,
        "adapted_state_sha256": est.adapted_state_sha256_,
    }
