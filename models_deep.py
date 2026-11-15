"""
Deep learning models for thermal comfort prediction.

This module implements several PyTorch-based classifiers:
    • ANNClassifier         — Fully-connected MLP classifier
    • FTClassifier          — FT-Transformer (feature tokenization)
    • TabTransformerClassifier — TabTransformer variant
    • Preprocessing tools (TabPreprocessor)
    • Training utilities (FocalLoss, save_torch_curves, select_best)

It provides:
    - unified sklearn-compatible classifiers (fit(), predict())
    - early stopping, validation curves, cross-validation scoring
    - categorical label encoding + numeric standardization
    - GPU/CPU auto-selection

These models serve as the deep-learning backbone of the thermal comfort pipeline.
"""
import os, random, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, platform
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import  LabelEncoder
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer
)
from config_loader import load_config
from compute import configure_torch_determinism, torch_device
from category_normalisation import CategoryVocabularyError


CONFIG = load_config()
RANDOM_SEED    = CONFIG["seed"]
# FP32 (TF32 off, no AMP) and deterministic kernels for every ANN/FTT fit;
# set before any cuBLAS use.
configure_torch_determinism()


# ======================================================================================
# INTERIM GUARDS
# ======================================================================================
class RevisionNotImplementedError(NotImplementedError):
    """A legacy pre-revision path that would produce invalid revision results."""


class LegacyModelStateError(RuntimeError):
    """A persisted model whose state predates the revision category encoders."""


class NonFiniteOutputError(FloatingPointError):
    """A network produced non-finite logits: never turned into a class or a probability."""


def check_finite_logits(logits, what: str = "network"):
    """
    Refuse non-finite logits before ``argmax`` / ``softmax``: ``argmax`` over
    NaN logits would return a valid-looking
    class (index 0) and ``softmax`` NaN probabilities.
    """
    if not bool(torch.isfinite(logits).all()):
        n_bad = int((~torch.isfinite(logits)).any(dim=1).sum())
        raise NonFiniteOutputError(f"{what}: non-finite logits on {n_bad} of {logits.shape[0]} rows "
                                   "(diverged weights or non-finite inputs); no prediction is written")
    return logits


LEGACY_PREPROCESSOR_MESSAGE = (
    "This TabPreprocessor was fitted before the revision category encoders "
    "and lacks {missing}. Archived models (e.g. "
    "rerun_2026-04-28_no_reweighting/) can only be evaluated with the pre-revision "
    "code that fitted them (archived models); revision runs need E1 models "
    "fitted with this code. No OOV embedding is added to archived weights."
)

PREPROCESSOR_STATE = ("vocabulary", "oov_policy", "vocabulary_", "train_categories_")


def check_preprocessor_state(pp, label: str = "") -> None:
    """Raise ``LegacyModelStateError`` unless ``pp`` carries the revision state."""
    missing = [a for a in PREPROCESSOR_STATE if not hasattr(pp, a)]
    if missing:
        raise LegacyModelStateError(
            (f"[{label}] " if label else "") + LEGACY_PREPROCESSOR_MESSAGE.format(missing=missing))


def check_ftt_revision_schema(model, vocabulary, label: str = "") -> None:
    """
    Check that a persisted FT-Transformer was fitted by the revision code:
    a revision ``TabPreprocessor`` with the frozen global vocabulary
    (``vocabulary`` = ``category_normalisation.vocabulary_for(cat_cols)``) and
    the reserved ``<OOV>`` index. Raises ``LegacyModelStateError``.
    """
    pp = getattr(model, "pp_", None)
    if pp is None:
        raise LegacyModelStateError(f"[{label}] FT-Transformer has no fitted preprocessor (pp_)")
    check_preprocessor_state(pp, label)
    if pp.oov_policy != "reserved_index":
        raise LegacyModelStateError(
            f"[{label}] FT-Transformer oov_policy is {pp.oov_policy!r}, expected 'reserved_index'")
    expected = {c: tuple(sorted(vocabulary[c])) for c in pp.cat_cols}
    if pp.vocabulary is None or dict(pp.vocabulary_) != expected:
        raise LegacyModelStateError(
            f"[{label}] FT-Transformer vocabulary differs from the frozen global vocabulary")


# ======================================================================================
# INTERNAL VALIDATION AND EARLY STOPPING
# ======================================================================================
# Every ANN/FTT fit (final model and every CV fit) holds out a stratified 10 %
# internal validation subset of its own training rows (seed 42, whatever the
# model seed), fits its preprocessing on the remaining 90 %, monitors the
# validation cross-entropy and restores the best-validation checkpoint. There is
# no refit on 100 % and no fallback to the training loss: a fit whose validation
# subset cannot be built raises.
ES_VAL_FRACTION = 0.1      # internal validation fraction
ES_VAL_SEED = 42           # seed of the internal validation subset
ES_MIN_FIT_ROWS = 100      # fewer training rows → raise
ES_MIN_DELTA = 1e-6        # improvement threshold


class EarlyStoppingValidationError(ValueError):
    """The internal validation subset cannot be built or used for this fit."""


def internal_validation_split(y):
    """
    Internal validation subset of one ANN/FTT fit.

    ``StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)`` on the
    labels of the fit's training rows.

    Parameters
    ----------
    y : array-like
        Labels of the fit's training rows.

    Returns
    -------
    (train_pos, val_pos) : tuple of numpy.ndarray
        Sorted positions of the internal training part (90 %) and of the internal
        validation subset (10 %).

    Raises
    ------
    EarlyStoppingValidationError
        Fewer than ``ES_MIN_FIT_ROWS`` rows, a stratified split that cannot be
        drawn (e.g. a class with a single vote), or a class of the fit absent
        from the internal training part.
    """
    y = np.asarray(y)
    n = len(y)
    if n < ES_MIN_FIT_ROWS:
        raise EarlyStoppingValidationError(
            f"ANN/FTT fit on {n} rows: early stopping needs at least {ES_MIN_FIT_ROWS} training rows "
            f"for the internal validation subset; no fallback to the training loss.")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ES_VAL_FRACTION, random_state=ES_VAL_SEED)
    try:
        tr, va = next(sss.split(np.zeros((n, 1)), y))
    except ValueError as exc:
        raise EarlyStoppingValidationError(
            f"internal validation subset cannot be built on {n} rows: {exc}") from exc
    missing = sorted(map(str, set(np.unique(y)) - set(np.unique(y[tr]))))
    if missing:
        raise EarlyStoppingValidationError(
            f"classes {missing} have no vote in the internal training part.")
    return np.sort(tr), np.sort(va)


def _train_early_stopping(model, forward, train_xs, y_tr, val_xs, y_va, criterion, optimizer,
                          epochs, patience, batch_size, on_epoch_end=None):
    """
    Training loop shared by the ANN and the FT-Transformer.

    Monitor = cross-entropy on the internal validation subset (evaluated in
    ``eval`` mode after every epoch); an epoch improves when its validation loss
    is below the best one by more than ``ES_MIN_DELTA``; training stops after
    ``patience`` epochs without improvement or ``epochs`` epochs; the weights
    of the best epoch are restored before returning.

    ``on_epoch_end(epoch, model)``, if given, is called after the validation
    loss of each epoch is computed (used by the tests to record each epoch's
    weights).

    Returns
    -------
    history : dict
        ``epoch``, ``train_loss``, ``val_loss``, ``val_f1`` per epoch run.
    best_epoch : int
        1-based epoch whose weights were restored.
    best_loss : float
        Its validation cross-entropy.

    Raises
    ------
    EarlyStoppingValidationError
        No epoch produced a finite validation loss (no checkpoint to restore).
    """
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}
    best_loss, bad, best_state, best_epoch = float("inf"), 0, None, None
    n = y_tr.size(0)
    for ep in range(epochs):
        model.train()
        # Same CPU permutation stream as before (seeded run unchanged), moved to the
        # data's device once per epoch instead of once per batch.
        perm = torch.randperm(n).to(y_tr.device)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = forward(*(x[idx] for x in train_xs))
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())     # no per-batch host sync; read back once per epoch
        train_loss = float(np.mean(torch.stack(losses).cpu().tolist()))

        model.eval()
        with torch.no_grad():
            logits_va = forward(*val_xs)
            val_loss = float(F.cross_entropy(logits_va, y_va).item())
            val_f1 = f1_score(y_va.cpu().numpy(), torch.argmax(logits_va, dim=1).cpu().numpy(),
                              average="macro", zero_division=0)
        history["epoch"].append(ep+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        if on_epoch_end is not None:
            on_epoch_end(ep+1, model)

        if np.isfinite(val_loss) and val_loss + ES_MIN_DELTA < best_loss:
            best_loss, bad, best_epoch = val_loss, 0, ep+1
            # checkpoint kept on the model's device (no device-to-host copy; restored bitwise)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break

    if best_state is None:
        raise EarlyStoppingValidationError(
            "no epoch produced a finite validation loss: no checkpoint to restore.")
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return history, best_epoch, best_loss


def _record_early_stopping(est, X, tr_pos, va_pos, history, best_epoch, best_loss):
    """Store the internal split and the early-stopping outcome on a fitted estimator."""
    est.es_train_idx_ = np.asarray(X.index[tr_pos])
    est.es_val_idx_ = np.asarray(X.index[va_pos])
    est.history_ = history
    est.best_epoch_ = int(best_epoch)
    est.best_val_loss_ = float(best_loss)
    est.n_epochs_run_ = len(history["epoch"])
    est.stopped_early_ = est.n_epochs_run_ < est.epochs


# ======================================================================================
# PREPROCESSOR
# ======================================================================================
class TabPreprocessor:
    """
    Preprocess categorical + numerical features for the ANN and the FT-Transformer.

    Steps:
        1. Categorical columns → integer codes = position of the value in the
           sorted vocabulary of the column.
        2. Numerical columns → StandardScaler.
        3. Output tensors compatible with PyTorch models.

    Unknown values are **never** mapped to the first category:

    * a value outside the vocabulary raises ``CategoryVocabularyError``;
    * a value of the vocabulary absent from the training rows is OOV for the
      fit; it is counted in ``last_oov_counts_`` and encoded according to
      ``oov_policy``:
        - ``"global_code"`` (ANN): keeps its global code;
        - ``"reserved_index"`` (FTT): mapped to one reserved ``<OOV>``
          index per variable (= ``len(vocabulary)``), never seen in training.

    Parameters
    ----------
    cat_cols : list of str
        Names of categorical columns.
    num_cols : list of str
        Names of numerical columns.
    vocabulary : dict or None
        Mapping column → canonical values (``category_normalisation.GLOBAL_VOCABULARY``).
        ``None`` derives it from the training rows (sorted unique values, as
        the pre-revision ``LabelEncoder``).
    oov_policy : {"global_code", "reserved_index"}
        Encoding of values OOV for the fit (see above).

    Attributes
    ----------
    vocabulary_ : dict
        Column → tuple of sorted canonical values (code = position).
    train_categories_ : dict
        Column → set of values present in the training rows.
    n_categories : list[int]
        Embedding cardinality per categorical column (``+1`` with ``reserved_index``).
    scaler : StandardScaler
        Fitted numerical scaler.
    last_oov_counts_ : dict
        Column → number of OOV rows in the last ``transform`` call.
    """
    OOV_POLICIES = ("global_code", "reserved_index")

    def __init__(self, cat_cols, num_cols, vocabulary=None, oov_policy="global_code"):
        if oov_policy not in self.OOV_POLICIES:
            raise ValueError(f"oov_policy must be one of {self.OOV_POLICIES}, got {oov_policy!r}")
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.vocabulary = vocabulary
        self.oov_policy = oov_policy
        self.scaler = SK_StandardScaler()

    @staticmethod
    def _check_in_vocab(col, values, vocab):
        bad = ~values.isin(set(vocab))
        if bad.any():
            raise CategoryVocabularyError(
                f"{col}: values {sorted(map(str, values[bad].unique()))} are outside the "
                f"vocabulary {list(vocab)} (normalise with category_normalisation first)."
            )

    def fit(self, X: pd.DataFrame):
        """
        Fit the vocabulary codes and the scaler on the training rows.

        Parameters
        ----------
        X : pandas.DataFrame
            Training input dataframe.

        Returns
        -------
        self
        """
        self.vocabulary_ = {}
        self.train_categories_ = {}
        for c in self.cat_cols:
            values = X[c]
            if values.isna().any():
                raise CategoryVocabularyError(f"{c}: missing values in the training rows.")
            if self.vocabulary is None:
                values = values.astype(str)
                vocab = tuple(sorted(values.unique()))
            else:
                vocab = tuple(sorted(self.vocabulary[c]))
            self._check_in_vocab(c, values, vocab)
            self.vocabulary_[c] = vocab
            self.train_categories_[c] = set(values.unique())
        Xn = X[self.num_cols].copy()
        self.scaler.fit(Xn)
        extra = 1 if self.oov_policy == "reserved_index" else 0
        self.n_categories = [len(self.vocabulary_[c]) + extra for c in self.cat_cols]
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform dataframe into PyTorch tensors.

        Values outside the vocabulary raise; values OOV for the fit are counted
        (``last_oov_counts_``) and encoded according to ``oov_policy``.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        (x_cat, x_num) : tuple(torch.LongTensor, torch.FloatTensor)
        """
        check_preprocessor_state(self)
        codes = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
        self.last_oov_counts_ = {}
        for j, c in enumerate(self.cat_cols):
            vocab = self.vocabulary_[c]
            values = X[c] if self.vocabulary is not None else X[c].astype(str)
            self._check_in_vocab(c, values, vocab)
            index = {v: i for i, v in enumerate(vocab)}
            col_codes = values.map(index).to_numpy(dtype=np.int64)
            oov = ~values.isin(self.train_categories_[c]).to_numpy()
            if self.oov_policy == "reserved_index":
                col_codes[oov] = len(vocab)
            codes[:, j] = col_codes
            self.last_oov_counts_[c] = int(oov.sum())
        Xn = X[self.num_cols].copy()
        Xn = self.scaler.transform(Xn)
        x_cat = torch.tensor(codes, dtype=torch.long)
        x_num = torch.tensor(Xn, dtype=torch.float32)
        return x_cat, x_num

# ======================================================================================
# BASIC MLP
# ======================================================================================
class MLP(nn.Module):
    """
    Simple feed-forward neural network for tabular data.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    hidden : tuple of int, default=(256,128)
        Sizes of hidden layers.
    out_dim : int, default=3
        Number of output classes.
    dropout : float, default=0.1
        Dropout probability applied after each hidden layer.
    """
    def __init__(self, in_dim, hidden=(256, 128), out_dim=3, dropout=0.1):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)


    def forward(self, x): 
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Logit outputs.
        """
        return self.net(x)
# ======================================================================================
# FOCAL LOSS
# ======================================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    This loss down-weights easy examples and focuses learning on hard ones.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter.
    weight : torch.Tensor or None, optional
        Per-class weights.
    reduction : str, default="mean"
        Reduction mode ('mean', 'sum', or 'none').
    """
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute focal loss.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ======================================================================================
# SHARED TRAINING LOGIC
# ======================================================================================

class _TorchBase(BaseEstimator, ClassifierMixin):
    """
    Base class providing sklearn-like interface around PyTorch classifiers.

    Handles:
        - device selection (CPU/GPU)
        - seed setting
        - loss function selection (cross-entropy or focal loss)
        - common hyperparameters (lr, epochs, patience, etc.)

    Parameters
    ----------
    num_cols : list of str
        Numerical columns.
    cat_cols : list of str
        Categorical columns.
    lr : float, default=3e-4
        Learning rate.
    batch_size : int, default=1024
        Batch size for SGD.
    epochs : int, default=250
        Maximum training epochs.
    patience : int, default=10
        Early stopping patience.
    focal_gamma : float or None
        If provided, use FocalLoss with given gamma.
    class_weights : list or None
        Optional per-class weights.
    device : str or None
        'cuda' or 'cpu'. If None → auto-detect CUDA.
    random_state : int
        Model seed (weights, batch order). The internal validation subset always
        uses ``ES_VAL_SEED``.

    Every fit holds out the internal validation subset
    (``internal_validation_split``) and early-stops on it; the pre-revision
    parameters ``curves_val_split`` / ``record_curves``, which could disable it,
    are removed. After ``fit``: ``history_`` (per-epoch train loss,
    validation loss and macro-F1), ``es_train_idx_`` / ``es_val_idx_`` (index
    labels of ``X``), ``best_epoch_``, ``best_val_loss_``, ``n_epochs_run_``,
    ``stopped_early_``.
    """
    def __init__(self, num_cols, cat_cols=(), lr=3e-4, batch_size=1024, epochs=250, patience=10,
                  focal_gamma=None, class_weights=None, device=None, random_state=RANDOM_SEED):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lr = float(lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    @classmethod
    def _get_param_names(cls):
        """
        Constructor parameters of the subclass *and* of ``_TorchBase`` (passed
        through ``**kwargs``): without them ``get_params`` omitted ``epochs``,
        ``device``, ``random_state``, ... and ``sklearn.base.clone`` silently
        rebuilt the estimator with the default device and seed.
        """
        own = super()._get_param_names()
        if cls is _TorchBase:
            return own
        return sorted(set(own) | set(BaseEstimator._get_param_names.__func__(_TorchBase)))

    def _internal_split(self, X, y):
        """
        Internal validation subset of this fit and the labels.

        Returns ``(y_np, tr_pos, va_pos)``; the label encoder ``le_y_`` and
        ``classes_`` are fitted on the internal training part only.
        """
        y_np = pd.Series(y).to_numpy()
        if len(X) != len(y_np):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_np)} labels")
        if not X.index.is_unique:
            raise EarlyStoppingValidationError("X index must be unique (es_val_idx_ records index labels)")
        tr_pos, va_pos = internal_validation_split(y_np)
        self.le_y_ = LabelEncoder().fit(y_np[tr_pos])
        self.classes_ = self.le_y_.classes_
        return y_np, tr_pos, va_pos

    def _set_seeds(self):
        """
        Set Python, NumPy and PyTorch random seeds.

        Python's ``random`` is seeded too: ``hyper_connections`` (FTT with
        ``num_residual_streams=4``) draws the initial residual index of each
        layer with ``random.randrange``, so without it the FTT initialisation
        depended on earlier draws in the process (job order).
        """
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)



    def _criterion(self, out_dim):
        """
        Get appropriate loss function.

        Parameters
        ----------
        out_dim : int
            Number of output classes.

        Returns
        -------
        nn.Module
            CrossEntropyLoss or FocalLoss.
        """
        w = None
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)

        if self.focal_gamma is not None:
            return FocalLoss(gamma=self.focal_gamma, weight=w)
        return nn.CrossEntropyLoss(weight=w)

# ======================================================================================
# ANN CLASSIFIER
# ======================================================================================


class ANNClassifier(_TorchBase):
    """
    Multi-layer perceptron (MLP) classifier using PyTorch.

    Parameters
    ----------
    hidden : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    **kwargs :
        Passed to _TorchBase.
    """
    def __init__(self, num_cols, cat_cols=(), hidden=(256,128), dropout=0.1, lr=1e-3, batch_size=256, patience=10,
                 cat_vocabulary=None, **kwargs):
        super().__init__(num_cols, cat_cols, **kwargs)
        self.cat_vocabulary = cat_vocabulary
        self.hidden = hidden
        self.dropout = dropout
        self.lr = float(lr)              
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y):
        """
        Fit the MLP classifier.

        Includes:
            - internal stratified 10 % validation subset (seed 42)
            - preprocessing (label encoding + scaling) fitted on the other 90 % only
            - training loop with early stopping on the validation cross-entropy
            - best-validation checkpoint restoration (no refit on 100 %)

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        self._set_seeds()
        y_np, tr_pos, va_pos = self._internal_split(X, y)
        X_fit, X_val = X.iloc[tr_pos], X.iloc[va_pos]
        # Integer codes of the global vocabulary; OOV keeps its global code.
        # Preprocessing fitted on the internal training part (90 %) only.
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols),
                                   vocabulary=self.cat_vocabulary,
                                   oov_policy="global_code").fit(X_fit)
        x_cat_tr, x_num_tr = self.pp_.transform(X_fit)
        x_cat_va, x_num_va = self.pp_.transform(X_val)
        self.es_val_oov_counts_ = dict(self.pp_.last_oov_counts_)
        y_tr = torch.tensor(self.le_y_.transform(y_np[tr_pos]), dtype=torch.long)
        y_va = torch.tensor(self.le_y_.transform(y_np[va_pos]), dtype=torch.long)
        out_dim = len(self.le_y_.classes_)

        Xtr = torch.cat([x_num_tr, x_cat_tr.float()], dim=1)
        Xva = torch.cat([x_num_va, x_cat_va.float()], dim=1)

        self.model_ = MLP(in_dim=Xtr.shape[1], hidden=self.hidden, out_dim=out_dim, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(out_dim)

        Xtr, y_tr = Xtr.to(self.device), y_tr.to(self.device)
        Xva, y_va = Xva.to(self.device), y_va.to(self.device)
        history, best_epoch, best_loss = _train_early_stopping(
            self.model_, self.model_, (Xtr,), y_tr, (Xva,), y_va, criterion, optimizer,
            self.epochs, self.patience, self.batch_size)
        _record_early_stopping(self, X, tr_pos, va_pos, history, best_epoch, best_loss)
        return self

    def predict(self, X):
        """
        Predict class labels for new samples.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        array-like
            Decoded class labels.
        """
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        X_in = torch.cat([x_num, x_cat.float()], dim=1).to(self.device)
        with torch.no_grad():
            logits = check_finite_logits(self.model_(X_in), "ANN")
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.le_y_.inverse_transform(pred_idx)

    def predict_proba(self, X):
        """
        Class probabilities (softmax of the logits), columns in the order of
        ``classes_``. Needed for the ``p_<class>`` columns of the predictions
        files; the pre-revision ANN exposed only ``predict``.
        """
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        X_in = torch.cat([x_num, x_cat.float()], dim=1).to(self.device)
        with torch.no_grad():
            return F.softmax(check_finite_logits(self.model_(X_in), "ANN"), dim=1).cpu().numpy()




# ======================================================================================
# FT-TRANSFORMER CLASSIFIER
# ======================================================================================
class FTClassifier(_TorchBase):
    """
    FT-Transformer classifier for tabular data.

    Based on feature tokenization + attention over features.

    Parameters
    ----------
    dim, depth, heads, dim_head, attn_dropout, ff_dropout :
        Transformer architecture parameters.
    **kwargs :
        Passed to _TorchBase.
    """
    def __init__(self, num_cols, cat_cols, dim=256, depth=4, heads=8, dim_head=32,
                 attn_dropout=0.1, ff_dropout=0.1, lr=1e-4, batch_size=512, patience=10,
                 cat_vocabulary=None, **kwargs):

        """
            Fit the FT-Transformer classifier.

            Steps identical to ANNClassifier.fit but applied to FTTransformer.
        """
        super().__init__(num_cols, cat_cols, **kwargs)
        self.cat_vocabulary = cat_vocabulary
        self.dim, self.depth, self.heads, self.dim_head = dim, depth, heads, dim_head
        self.attn_dropout, self.ff_dropout = attn_dropout, ff_dropout
        self.lr = float(lr)              
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y):
        self._set_seeds()
        y_np, tr_pos, va_pos = self._internal_split(X, y)
        X_fit, X_val = X.iloc[tr_pos], X.iloc[va_pos]
        # Global vocabulary + one reserved <OOV> index per variable.
        # Preprocessing fitted on the internal training part (90 %) only.
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols),
                                   vocabulary=self.cat_vocabulary,
                                   oov_policy="reserved_index").fit(X_fit)
        x_cat_tr, x_num_tr = self.pp_.transform(X_fit)
        x_cat_va, x_num_va = self.pp_.transform(X_val)
        self.es_val_oov_counts_ = dict(self.pp_.last_oov_counts_)
        y_tr = torch.tensor(self.le_y_.transform(y_np[tr_pos]), dtype=torch.long)
        y_va = torch.tensor(self.le_y_.transform(y_np[va_pos]), dtype=torch.long)
        num_classes = len(self.le_y_.classes_)

        self.model_ = FTTransformer(
            categories=self.pp_.n_categories,
            num_continuous=x_num_tr.shape[1],
            dim=self.dim, depth=self.depth, heads=self.heads, dim_head=self.dim_head,
            dim_out=num_classes,
            attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout,
            num_residual_streams=4
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(num_classes)

        x_cat_tr, x_num_tr, y_tr = x_cat_tr.to(self.device), x_num_tr.to(self.device), y_tr.to(self.device)
        x_cat_va, x_num_va, y_va = x_cat_va.to(self.device), x_num_va.to(self.device), y_va.to(self.device)
        history, best_epoch, best_loss = _train_early_stopping(
            self.model_, self.model_, (x_cat_tr, x_num_tr), y_tr, (x_cat_va, x_num_va), y_va,
            criterion, optimizer, self.epochs, self.patience, self.batch_size)
        _record_early_stopping(self, X, tr_pos, va_pos, history, best_epoch, best_loss)
        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        array-like
        """
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
        with torch.no_grad():
            logits = check_finite_logits(self.model_(x_cat, x_num), type(self).__name__)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.le_y_.inverse_transform(pred_idx)
    
    def predict_proba(self, X):
        """
        Predict class-probabilities.

        Returns
        -------
        numpy.ndarray
            Probability vector for each sample.
        """
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
        with torch.no_grad():
            logits = check_finite_logits(self.model_(x_cat, x_num), type(self).__name__)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

# ======================================================================================
# MODEL REGISTRY
# ======================================================================================
def get_torch_models(num_cols, cat_cols, deep_hparams, seed=RANDOM_SEED, cat_vocabulary=None,
                     compute=None):
    """
    Registry of available deep-learning models.

    ``cat_vocabulary`` (column → canonical values) fixes the
    category codes of the ANN and FT-Transformer; ``None`` derives
    them from the training rows.

    ``compute`` (default: ``config.yaml``) fixes the device of both estimators
    (``compute.torch_device``; ``cuda`` without a CUDA device raises, no silent
    fallback to the CPU). Hyperparameters unchanged.

    Returns
    -------
    dict
        Mapping of model_name → (estimator_instance, hyperparameter_grid)
    """
    device = torch_device(compute)
    models = {
        "ANN": (
            ANNClassifier(
        num_cols=num_cols,
        cat_cols=cat_cols,
        hidden=deep_hparams["ANN"]["hidden_layers"],
        dropout=deep_hparams["ANN"]["dropout"],
        lr=deep_hparams["ANN"]["lr"],
        batch_size=deep_hparams["ANN"]["batch_size"],
        patience=deep_hparams["ANN"]["patience"],
        random_state=seed,
        focal_gamma=deep_hparams["ANN"].get("focal_gamma", None),
        class_weights=deep_hparams["ANN"].get("class_weights", None),
        cat_vocabulary=cat_vocabulary,
        device=device
    ),
            {
            "hidden": [deep_hparams["ANN"]["hidden_layers"]], 
            "dropout": [deep_hparams["ANN"]["dropout"]],
            "epochs": [deep_hparams["ANN"]["epochs"]],
            "lr": [deep_hparams["ANN"]["lr"]],
            "batch_size": [deep_hparams["ANN"]["batch_size"]],
            "patience": [deep_hparams["ANN"]["patience"]],
         }
        ),
        "FTTransformer": (
            FTClassifier(
            num_cols=num_cols,
            cat_cols=cat_cols,
            dim=deep_hparams["FTTransformer"]["dim"],
            depth=deep_hparams["FTTransformer"]["depth"],
            heads=deep_hparams["FTTransformer"]["heads"],
            dim_head=deep_hparams["FTTransformer"]["dim_head"],
            attn_dropout=deep_hparams["FTTransformer"]["attn_dropout"],
            ff_dropout=deep_hparams["FTTransformer"]["ff_dropout"],
            lr=deep_hparams["FTTransformer"]["lr"],
            batch_size=deep_hparams["FTTransformer"]["batch_size"],
            patience=deep_hparams["FTTransformer"]["patience"],
            random_state=seed,
            focal_gamma=deep_hparams["FTTransformer"].get("focal_gamma", None),
            class_weights=deep_hparams["FTTransformer"].get("class_weights", None),
            cat_vocabulary=cat_vocabulary,
            device=device
            ),
            {
            "dim": [deep_hparams["FTTransformer"]["dim"]],
            "depth": [deep_hparams["FTTransformer"]["depth"]],
            "heads": [deep_hparams["FTTransformer"]["heads"]],
            "dim_head": [deep_hparams["FTTransformer"]["dim_head"]],
            "attn_dropout": [deep_hparams["FTTransformer"]["attn_dropout"]],
            "ff_dropout": [deep_hparams["FTTransformer"]["ff_dropout"]],
            "epochs": [deep_hparams["FTTransformer"]["epochs"]],
            "lr": [deep_hparams["FTTransformer"]["lr"]],
            "batch_size": [deep_hparams["FTTransformer"]["batch_size"]],
           }
        )
    }
    return models

# ======================================================================================
# TRAINING CURVE SAVING
# ======================================================================================
def save_torch_curves(model, out_csv_path: Path):
    """
    Save the training history of a fitted ANN/FTT (epoch, train_loss, val_loss, val_f1).

    Called on the **final** model (the validation history of the
    final model is saved); its ``val_loss`` is the monitored internal-validation
    cross-entropy. Raises if the model has no history (never fitted).

    Parameters
    ----------
    model : PyTorch-based classifier
    out_csv_path : pathlib.Path
        Where to save the CSV file.
    """
    if not getattr(model, "history_", None) or len(model.history_.get("epoch", [])) == 0:
        raise ValueError("model has no training history (history_): fit it first")
    pd.DataFrame(model.history_).to_csv(out_csv_path, index=False)


def early_stopping_record(model) -> dict:
    """
    Early-stopping summary of a fitted ANN/FTT, JSON-ready.

    ``es_val_idx`` lists the index labels of the internal validation subset of
    the fit (row indices of the cohort when ``X`` carries them).
    """
    return {
        "monitor": "internal_validation_cross_entropy",
        "es_val_fraction": ES_VAL_FRACTION,
        "es_val_seed": ES_VAL_SEED,
        "min_delta": ES_MIN_DELTA,
        "min_fit_rows": ES_MIN_FIT_ROWS,
        "patience": int(model.patience),
        "max_epochs": int(model.epochs),
        "model_seed": int(model.random_state),
        "n_fit_rows": int(len(model.es_train_idx_) + len(model.es_val_idx_)),
        "n_internal_train": int(len(model.es_train_idx_)),
        "n_internal_val": int(len(model.es_val_idx_)),
        "best_epoch": int(model.best_epoch_),
        "best_val_loss": float(model.best_val_loss_),
        "n_epochs_run": int(model.n_epochs_run_),
        "stopped_early": bool(model.stopped_early_),
        "refit_on_full_training_rows": False,
        "es_val_oov_counts": {k: int(v) for k, v in model.es_val_oov_counts_.items()},
        "es_val_idx": [x.item() if hasattr(x, "item") else x for x in model.es_val_idx_],
    }



# ======================================================================================
# MANUAL GRID SEARCH FOR TORCH
# ======================================================================================

def select_best(target_name, model_name, est, param_grid, X_tr, y_tr, refit_metric="f1_macro",
                cv_splits=None):
    """
    Manual grid search for PyTorch models.

    Full sklearn GridSearchCV cannot be used because PyTorch estimators
    are not fully serializable. This function implements:

        - exhaustive hyperparameter search
        - k-fold cross-validation on the materialised folds
        - final model fit on ``X_tr``
        - return statistics and per-parameter performance

    Every fit (search folds, reported folds, final model) early-stops on its own
    internal validation subset (``internal_validation_split``). The final model
    is trained on the internal 90 % of ``X_tr`` and is
    the restored best-validation checkpoint; it is **not** refitted on 100 %
    (its ``es_val_idx_`` and ``history_`` are those of that fit). The pre-revision
    auxiliary ``model_for_curves`` fit is removed: the curves
    are those of the final model.

    With a one-point grid (every ``get_torch_models`` grid), the reported folds
    are the search fits themselves: same estimator parameters, same folds and
    same ``random_state``, so a refit would reproduce them (deterministic
    training); they are not fitted twice (K + 1 fits
    instead of 2K + 1). A grid with several points refits the K folds with the
    selected parameters, as before.

    Parameters
    ----------
    target_name : str
        Name of the task.
    model_name : str
        Model identifier.
    est : _TorchBase
        Prototype estimator instance.
    param_grid : dict
        Hyperparameter grid.
    X_tr : pandas.DataFrame
        Training features.
    y_tr : pandas.Series
        Training labels.
    refit_metric : str, default="f1_macro"
        Metric maximized during grid search.
    cv_splits : list of (array, array)
        Materialised folds of the target, positions in
        ``X_tr``, used for both the parameter search and the reported folds;
        their hash is written in the ``folds_sha256`` column. Required:
        ``None`` raises (no family draws its own folds).

    Returns
    -------
    best_model : _TorchBase
        Final fitted model.
    best_params : dict
        Best hyperparameter set.
    fold_summary_df : pandas.DataFrame
        Per-fold performance table.
    stats_df : pandas.DataFrame
        Summary statistics over folds.
    per_param_val : list of dict
        Mean/STD scores for each parameter combination.
    """
    if cv_splits is None:
        raise ValueError("select_best needs the materialised folds (cv_splits); "
                         "no family draws its own folds")
    from revision_folds import splits_sha256
    cv_splits = [(np.asarray(a), np.asarray(b)) for a, b in cv_splits]
    folds_sha256 = splits_sha256(cv_splits, len(X_tr))   # also checks the K-fold partition
    search_splits = report_splits = cv_splits
    keys = list(param_grid.keys())

# Generate all combinations
    def iter_param_sets(idx=0, current=None):
        if current is None: current = {}
        if idx == len(keys):
            yield current; return
        k_ = keys[idx]
        for v in param_grid[k_]:
            cc = dict(current); cc[k_] = v
            yield from iter_param_sets(idx+1, cc)

    best_score, best_params, best_model_proto = -np.inf, None, None
    per_param_val = []
    param_sets = list(iter_param_sets())
    # one-point grid: the search fits are the reported fits (same parameters, folds and seed)
    reuse_search_fits = len(param_sets) == 1 and search_splits is report_splits
    search_fits = []


# Evaluate parameter combinations
    for params in param_sets:
        base_params = {k: getattr(est, k) for k in est.__dict__.keys() if not k.endswith("_")}
        base_params.update(params)
        cv_scores = []
        for tr_idx, va_idx in search_splits:
            X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            m = est.__class__(**base_params)
            t0 = time.time()
            m.fit(X_tr_i, y_tr_i)
            fit_time = time.time() - t0
            y_pred = m.predict(X_va_i)
            cv_scores.append(f1_score(y_va_i, y_pred, average="macro", zero_division=0))
            if reuse_search_fits:
                search_fits.append((m.predict(X_tr_i), y_pred, fit_time))
        mean_cv = float(np.mean(cv_scores))
        per_param_val.append({"params": params, "val_f1_macro_mean": mean_cv, "val_f1_macro_std": float(np.std(cv_scores))})
        print(f"[{target_name} - {model_name}] params={params} | {refit_metric}={mean_cv:.4f}")
        if mean_cv > best_score:
            best_score, best_params = mean_cv, params
            best_model_proto = est.__class__(**base_params)

# Train final model (internal 90 %, restored best-validation checkpoint, no refit on 100 %)
    final_params = {k: getattr(best_model_proto, k) for k in best_model_proto.__dict__.keys() if not k.endswith("_")}
    best_model = est.__class__(**final_params)
    best_model.fit(X_tr, y_tr)


# Report final folds
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(report_splits, 1):
        X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        if reuse_search_fits:
            y_pred_tr, y_pred_va, train_time = search_fits[fold_id - 1]
        else:
            mparams = {k: getattr(best_model, k) for k in best_model.__dict__.keys() if not k.endswith("_")}
            m = est.__class__(**mparams)
            t0 = time.time()
            m.fit(X_tr_i, y_tr_i)
            train_time = time.time() - t0
            y_pred_tr = m.predict(X_tr_i)
            y_pred_va = m.predict(X_va_i)
        rows.append({
            "fold": fold_id,
            "experiment": target_name,
            "model": model_name,
            "train_f1_macro": f1_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_f1_macro":   f1_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "train_accuracy": accuracy_score(y_tr_i, y_pred_tr),
            "val_accuracy":   accuracy_score(y_va_i, y_pred_va),
            "train_precision_macro": precision_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_precision_macro":   precision_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "train_recall_macro": recall_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_recall_macro":   recall_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "training_time": train_time,
            "folds_sha256": folds_sha256,
        })
    fold_summary_df = pd.DataFrame(rows)

    def stats_of(col):
        arr = fold_summary_df[col].values
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "median": float(np.median(arr))}
    stats_df = pd.DataFrame([
        {"experiment": target_name, "model": model_name, "metric": "val_accuracy", **stats_of("val_accuracy")},
        {"experiment": target_name, "model": model_name, "metric": "val_f1_macro", **stats_of("val_f1_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_precision_macro", **stats_of("val_precision_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_recall_macro", **stats_of("val_recall_macro")},
    ])

    return best_model, best_params, fold_summary_df, stats_df, per_param_val

