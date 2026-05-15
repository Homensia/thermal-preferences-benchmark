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
import umap
import copy
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer
)
from config_loader import load_config


CONFIG = load_config()
RANDOM_SEED    = CONFIG["seed"]
# ======================================================================================
# PREPROCESSOR
# ======================================================================================
class TabPreprocessor:
    """
    Preprocess categorical + numerical features for Transformer-based tabular models.

    Steps:
        1. Categorical columns → LabelEncoder per column.
        2. Numerical columns → StandardScaler.
        3. Output tensors compatible with PyTorch models.

    Parameters
    ----------
    cat_cols : list of str
        Names of categorical columns.
    num_cols : list of str
        Names of numerical columns.

    Attributes
    ----------
    encoders : dict
        Mapping column → fitted LabelEncoder.
    scaler : StandardScaler
        Fitted numerical scaler.
    n_categories : list[int]
        Number of unique categories per categorical column.
    """
    def __init__(self, cat_cols, num_cols):  
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.encoders = {c: LabelEncoder() for c in self.cat_cols}
        self.scaler = SK_StandardScaler()

    def fit(self, X: pd.DataFrame):
        """
        Fit encoders and scaler.

        Parameters
        ----------
        X : pandas.DataFrame
            Training input dataframe.

        Returns
        -------
        self
        """
        Xc = X[self.cat_cols].astype(str).copy()
        for c in self.cat_cols:
            Xc[c] = self.encoders[c].fit_transform(Xc[c].fillna("Unknown"))
        Xn = X[self.num_cols].copy()
        Xn = self.scaler.fit(Xn)
        self.n_categories = [len(self.encoders[c].classes_) for c in self.cat_cols]
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform dataframe into PyTorch tensors.

        Unknown categories are mapped to the first seen category.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        (x_cat, x_num) : tuple(torch.LongTensor, torch.FloatTensor)
        """
        Xc = X[self.cat_cols].astype(str).copy()
        for c in self.cat_cols:
            known = set(self.encoders[c].classes_)
            Xc[c] = Xc[c].apply(lambda v: v if v in known else self.encoders[c].classes_[0])
            Xc[c] = self.encoders[c].transform(Xc[c])
        Xn = X[self.num_cols].copy()
        Xn = self.scaler.transform(Xn)
        x_cat = torch.tensor(Xc.values, dtype=torch.long)
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
        Random seed.
    curves_val_split : float
        Fraction for validation split (for plotting learning curves).
    record_curves : bool
        Whether to store loss/metrics during training.
    """
    def __init__(self, num_cols, cat_cols=(), lr=3e-4, batch_size=1024, epochs=250, patience=10,
                  focal_gamma=None, class_weights=None, device=None, random_state=RANDOM_SEED,
                  curves_val_split=0.1, record_curves=True):
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
        self.curves_val_split = curves_val_split
        self.record_curves = record_curves  


    def _set_seeds(self):
        """Set PyTorch + NumPy random seeds."""
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
    def __init__(self, num_cols, cat_cols=(), hidden=(256,128), dropout=0.1, lr=1e-3, batch_size=256, patience=10,**kwargs):
        super().__init__(num_cols, cat_cols, **kwargs)
        self.hidden = hidden
        self.dropout = dropout
        self.lr = float(lr)              
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y):
        """
        Fit the MLP classifier.

        Includes:
            - preprocessing (label encoding + scaling)
            - training loop with early stopping
            - validation split for curves
            - best checkpoint restoration

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        self._set_seeds()
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols)).fit(X)
        x_cat_all, x_num_all = self.pp_.transform(X)

        self.le_y_ = LabelEncoder()
        y_idx_np = self.le_y_.fit_transform(pd.Series(y).values)
        self.classes_ = self.le_y_.classes_
        y_all  = torch.tensor(y_idx_np, dtype=torch.long)
        out_dim = len(self.le_y_.classes_)

        if self.record_curves and x_cat_all.size(0) > 100 and self.curves_val_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.curves_val_split, random_state=self.random_state)
            tr_idx, va_idx = next(sss.split(x_cat_all.numpy(), y_all.numpy()))
            tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)
        else:
            tr_idx = torch.arange(x_cat_all.size(0)); va_idx = None

        x_cat_tr, x_num_tr, y_tr = x_cat_all[tr_idx], x_num_all[tr_idx], y_all[tr_idx]
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_all[va_idx], x_num_all[va_idx], y_all[va_idx]

        Xtr = torch.cat([x_num_tr, x_cat_tr.float()], dim=1)
        if va_idx is not None:
            Xva = torch.cat([x_num_va, x_cat_va.float()], dim=1)

        self.model_ = MLP(in_dim=Xtr.shape[1], hidden=self.hidden, out_dim=out_dim, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(out_dim)

        
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        best_loss, bad, best_state = float("inf"), 0, None
        Xtr, y_tr = Xtr.to(self.device), y_tr.to(self.device)
        if va_idx is not None:
            Xva, y_va = Xva.to(self.device), y_va.to(self.device)

        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(Xtr.size(0))
            losses = []
            for i in range(0, Xtr.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                logits = self.model_(Xtr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            val_loss = np.nan; val_f1 = np.nan
            if va_idx is not None:
                self.model_.eval()
                with torch.no_grad():
                    logits_va = self.model_(Xva)
                    val_loss = float(criterion(logits_va, y_va).item())
                    val_f1 = f1_score(y_va.cpu().numpy(),
                                      torch.argmax(logits_va, dim=1).cpu().numpy(),
                                      average="macro", zero_division=0)
            self.history_["epoch"].append(ep+1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_f1"].append(val_f1)

            monitor = val_loss if va_idx is not None else train_loss
            if monitor + 1e-6 < best_loss:
                best_loss, bad = monitor, 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
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
            logits = self.model_(X_in)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.le_y_.inverse_transform(pred_idx)




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
                 attn_dropout=0.1, ff_dropout=0.1, lr=1e-4, batch_size=512, patience=10, **kwargs):

        """
            Fit the FT-Transformer classifier.

            Steps identical to ANNClassifier.fit but applied to FTTransformer.
        """
        super().__init__(num_cols, cat_cols, **kwargs)
        self.dim, self.depth, self.heads, self.dim_head = dim, depth, heads, dim_head
        self.attn_dropout, self.ff_dropout = attn_dropout, ff_dropout
        self.lr = float(lr)              
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y):
        self._set_seeds()
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols)).fit(X)
        x_cat_all, x_num_all = self.pp_.transform(X)

        self.le_y_ = LabelEncoder()
        y_idx_np = self.le_y_.fit_transform(pd.Series(y).values)
        self.classes_ = self.le_y_.classes_  
        y_all  = torch.tensor(y_idx_np, dtype=torch.long)
        num_classes = len(self.le_y_.classes_)
        
        if self.record_curves and x_cat_all.size(0) > 100 and self.curves_val_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.curves_val_split, random_state=self.random_state)
            tr_idx, va_idx = next(sss.split(x_cat_all.numpy(), y_all.numpy()))
            tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)
        else:
            tr_idx = torch.arange(x_cat_all.size(0)); va_idx = None

        x_cat_tr, x_num_tr, y_tr = x_cat_all[tr_idx], x_num_all[tr_idx], y_all[tr_idx]
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_all[va_idx], x_num_all[va_idx], y_all[va_idx]

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

        self.history_ = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        best_loss, bad, best_state = float("inf"), 0, None
        x_cat_tr, x_num_tr, y_tr = x_cat_tr.to(self.device), x_num_tr.to(self.device), y_tr.to(self.device)
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_va.to(self.device), x_num_va.to(self.device), y_va.to(self.device)

        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(x_cat_tr.size(0))
            losses = []
            for i in range(0, x_cat_tr.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                logits = self.model_(x_cat_tr[idx], x_num_tr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            val_loss = np.nan; val_f1 = np.nan
            if va_idx is not None:
                self.model_.eval()
                with torch.no_grad():
                    logits_va = self.model_(x_cat_va, x_num_va)
                    val_loss = float(criterion(logits_va, y_va).item())
                    val_f1 = f1_score(y_va.cpu().numpy(),
                                      torch.argmax(logits_va, dim=1).cpu().numpy(),
                                      average="macro", zero_division=0)
            self.history_["epoch"].append(ep+1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_f1"].append(val_f1)

            monitor = val_loss if va_idx is not None else train_loss
            if monitor + 1e-6 < best_loss:
                best_loss, bad = monitor, 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
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
            logits = self.model_(x_cat, x_num)
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
            logits = self.model_(x_cat, x_num)          
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

# ======================================================================================
# MODEL REGISTRY
# ======================================================================================
def get_torch_models(num_cols, cat_cols, deep_hparams, seed=RANDOM_SEED):
    """
    Registry of available deep-learning models.

    Returns
    -------
    dict
        Mapping of model_name → (estimator_instance, hyperparameter_grid)
    """
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
        record_curves=True
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
            record_curves=True
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
    Save model training curves (epoch, train_loss, val_loss, val_f1).

    Parameters
    ----------
    model : PyTorch-based classifier
    out_csv_path : pathlib.Path
        Where to save the CSV file.

    Returns
    -------
    None
    """
    """Sauvegarde history_ (epoch, train_loss, val_loss, val_f1) si disponible."""
    if hasattr(model, "history_") and model.history_ and len(model.history_.get("epoch", [])) > 0:
        df = pd.DataFrame(model.history_)
        df.to_csv(out_csv_path, index=False)



# ======================================================================================
# MANUAL GRID SEARCH FOR TORCH
# ======================================================================================

def select_best(target_name, model_name, est, param_grid, X_tr, y_tr, k=5, refit_metric="f1_macro"):
    """
    Manual grid search for PyTorch models.

    Full sklearn GridSearchCV cannot be used because PyTorch estimators
    are not fully serializable. This function implements:

        - exhaustive hyperparameter search
        - k-fold cross-validation
        - final model retraining
        - optional learning-curves model
        - return statistics and per-parameter performance

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
    k : int, default=5
        CV folds.
    refit_metric : str, default="f1_macro"
        Metric maximized during grid search.

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
    model_for_curves : _TorchBase
        Fitted model with learning curves enabled.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
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


# Evaluate parameter combinations
    for params in iter_param_sets():
        base_params = {k: getattr(est, k) for k in est.__dict__.keys() if not k.endswith("_")}
        base_params.update(params)
        base_params.update({"record_curves": False, "curves_val_split": 0.0})
        cv_scores = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            m = est.__class__(**base_params)
            m.fit(X_tr_i, y_tr_i)
            y_pred = m.predict(X_va_i)
            cv_scores.append(f1_score(y_va_i, y_pred, average="macro", zero_division=0))
        mean_cv = float(np.mean(cv_scores))
        per_param_val.append({"params": params, "val_f1_macro_mean": mean_cv, "val_f1_macro_std": float(np.std(cv_scores))})
        print(f"[{target_name} - {model_name}] params={params} | {refit_metric}={mean_cv:.4f}")
        if mean_cv > best_score:
            best_score, best_params = mean_cv, params
            best_model_proto = est.__class__(**base_params)

# Train best model with full curves
    curve_params = {k: getattr(best_model_proto, k) for k in best_model_proto.__dict__.keys() if not k.endswith("_")}
    curve_params.update({"record_curves": True, "curves_val_split": 0.1})
    model_for_curves = est.__class__(**curve_params)
    model_for_curves.fit(X_tr, y_tr)

# Train final model
    final_params = {k: getattr(best_model_proto, k) for k in best_model_proto.__dict__.keys() if not k.endswith("_")}
    final_params.update({"record_curves": False, "curves_val_split": 0.0})
    best_model = est.__class__(**final_params)
    best_model.fit(X_tr, y_tr)


# Report final folds
    skf2 = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED+1)
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf2.split(X_tr, y_tr), 1):
        X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        mparams = {k: getattr(best_model, k) for k in best_model.__dict__.keys() if not k.endswith("_")}
        mparams.update({"record_curves": False, "curves_val_split": 0.0})
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
            "training_time": train_time
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

    return best_model, best_params, fold_summary_df, stats_df, per_param_val, model_for_curves

