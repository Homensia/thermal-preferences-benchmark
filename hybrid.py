"""
Hybrid modeling utilities: extracting deep embeddings from FTTransformers
and fitting classical ML heads (RF, XGBoost) on top.

This module implements:
    - ft_get_embeddings: extract penultimate-layer embeddings from FTTransformer
    - FTEmbeddingHead: sklearn-compatible wrapper for hybrid models
    - run_heads_on_ft_and_save: train classical ML heads on FT embeddings
    - visualize_ft_embeddings: UMAP visualization of embedding structure

It allows combining deep feature representation learning (Transformer-based)
with classical ML classifiers for improved robustness and interpretability.
"""
import os, random, time, json
from typing import Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import torch, platform
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer,
)
# Native statistical helpers shared with models.py (QWK + Wilson CI).
from stats_utils import quadratic_weighted_kappa, wilson_ci_accuracy
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import copy
import joblib
import torch.nn.functional as F
from config_loader import load_config




CONFIG = load_config()
RANDOM_SEED    = CONFIG["seed"]




# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================
def ensure_dir(p: Path):
    """
    Create directory if missing.

    Parameters
    ----------
    p : pathlib.Path

    Returns
    -------
    pathlib.Path
    """
    p.mkdir(parents=True, exist_ok=True); return p
# ======================================================================================
# INTERNAL UTILITY — LOCATE LAST LINEAR LAYER
# ======================================================================================
def _find_last_linear_for_out_dim(model: nn.Module, out_dim: int):
    """
    Locate the last ``nn.Linear`` layer whose output dimension equals ``out_dim``.

    This is used to insert a forward hook to extract penultimate-layer embeddings.

    Parameters
    ----------
    model : nn.Module
        The neural network model (FTTransformer or TabTransformer).
    out_dim : int
        Desired output dimension (typically number of classes).

    Returns
    -------
    nn.Linear or None
        The matching layer, if found.
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == out_dim:
            last = m
    return last


# ======================================================================================
# EMBEDDING EXTRACTION
# ======================================================================================
def ft_get_embeddings(ft_clf, X: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    """
    Extract embeddings from the FTTransformer's penultimate layer.

    A forward pre-hook is attached to the last linear layer (before the classifier head).
    This hook captures its input — i.e., the learned feature representation.

    Parameters
    ----------
    ft_clf : FTClassifier or TabTransformerClassifier
        A trained classifier containing:
            - pp_ : TabPreprocessor
            - model_ : FTTransformer
            - le_y_ : LabelEncoder
            - device : "cpu" or "cuda"
    X : pandas.DataFrame
        Input samples.
    batch_size : int, default=1024
        Batch size used to iterate through X.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (N_samples, embedding_dim).
    """
    x_cat_all, x_num_all = ft_clf.pp_.transform(X)
    x_cat_all, x_num_all = x_cat_all.to(ft_clf.device), x_num_all.to(ft_clf.device)

    num_classes = len(ft_clf.le_y_.classes_)
    last_lin = _find_last_linear_for_out_dim(ft_clf.model_, num_classes)
    if last_lin is None:
        raise RuntimeError("Dernière couche Linear (out_dim=num_classes) introuvable dans le FT.")

    penult_batches = []
    def pre_hook(mod, inputs):
        penult_batches.append(inputs[0].detach().cpu())

    h = last_lin.register_forward_pre_hook(pre_hook)
    try:
        ft_clf.model_.eval()
        with torch.no_grad():
            N = x_cat_all.size(0)
            for i in range(0, N, batch_size):
                x_cat = x_cat_all[i:i+batch_size]
                x_num = x_num_all[i:i+batch_size]
                _ = ft_clf.model_(x_cat, x_num)
    finally:
        h.remove()

    if not penult_batches:
        raise RuntimeError("Aucun embedding capturé (hook non déclenché).")
    return torch.cat(penult_batches, dim=0).numpy()


# ======================================================================================
# HYBRID HEAD WRAPPER
# ======================================================================================
class FTEmbeddingHead(BaseEstimator, ClassifierMixin):
    """
    Wrapper combining FTTransformer embeddings with a classical ML head.

    Steps:
        1. Extract embeddings from pretrained FTTransformer.
        2. Encode target labels via LabelEncoder.
        3. Train provided sklearn classifier on embeddings.

    Parameters
    ----------
    ft_clf : FTClassifier
        Pretrained transformer classifier.
    head_estimator : sklearn estimator
        Classifier applied on the extracted embeddings.

    Attributes
    ----------
    head_estimator_ : sklearn estimator
        Fitted classifier.
    le_y_ : LabelEncoder
        Encoder for target labels.
    """

    def __init__(self, ft_clf, head_estimator):
        self.ft_clf = ft_clf
        self.head_estimator = head_estimator

    def fit(self, X, y):
        """
        Fit the ML head on Transformer embeddings.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like

        Returns
        -------
        self
        """
        Z = ft_get_embeddings(self.ft_clf, X)
        self.le_y_ = LabelEncoder().fit(y)
        y_enc = self.le_y_.transform(y)
        self.head_estimator_ = clone(self.head_estimator)
        self.head_estimator_.fit(Z, y_enc)
        return self

    def predict(self, X):
        """
        Predict using ML head applied to Transformer embeddings.

        Returns
        -------
        array-like
            Decoded class predictions.
        """
        Z = ft_get_embeddings(self.ft_clf, X)
        y_pred_enc = self.head_estimator_.predict(Z)
        return self.le_y_.inverse_transform(y_pred_enc)

# ======================================================================================
# HYBRID TRAINING PIPELINE
# ======================================================================================
def run_heads_on_ft_and_save(ft_clf, X_train, y_train, X_test, y_test, out_dir: Path,
                             target_name: str, tag_prefix="FTemb",n_umap_components=32,
                             heads: Optional[List[str]] = None,hparams=None):
    hparams = hparams or {}
    
    """
    Train classical ML heads (RF, XGB) on FTTransformer embeddings and save results.

    For each requested head:
        - GridSearchCV is run on top of FT embeddings.
        - Best estimator is saved.
        - Confusion matrices + scores + reports are saved.
        - FT embeddings for train/test are computed once.

    Parameters
    ----------
    ft_clf : FTClassifier
        Pretrained FTTransformer-based classifier.
    X_train : pandas.DataFrame
    y_train : pandas.Series
    X_test : pandas.DataFrame
    y_test : pandas.Series
    out_dir : pathlib.Path
        Directory where results will be saved.
    target_name : str
        Name of prediction task.
    tag_prefix : str, default="FTemb"
        Prefix used in saved result filenames.
    n_umap_components : int, default=32
        (Unused here but kept for compatibility.)
    heads : list of str or None
        List of heads to train, among {"RF", "XGBoost"}.

    Returns
    -------
    list[dict]
        One result dict per trained head.
    """
    if heads is None:
        heads = ["RF", "XGBoost"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute embeddings once for efficiency
    Z_tr = ft_get_embeddings(ft_clf, X_train)
    Z_te = ft_get_embeddings(ft_clf, X_test)


    results_rows = []


    # ---------------------------------------------------------
    # RANDOM FOREST HEAD
    # ---------------------------------------------------------   
    def _as_list(v):
        # GridSearchCV.param_grid values must be list/tuple/ndarray. Accept
        # scalars from config.yaml by wrapping them in a one-element list.
        return v if isinstance(v, (list, tuple)) else [v]

    if "RF" in heads:
        rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced_subsample")
        rf_grid = {
            "n_estimators": _as_list(hparams["heads"]["RF"]["n_estimators"]),
            "max_depth": _as_list(hparams["heads"]["RF"]["max_depth"]),
            "min_samples_split": _as_list(hparams["heads"]["RF"]["min_samples_split"]),
            "min_samples_leaf": _as_list(hparams["heads"]["RF"]["min_samples_leaf"]),
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        rf_gs = GridSearchCV(rf, rf_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, return_train_score=True)
        rf_gs.fit(Z_tr, y_train.values)
        rf_best = rf_gs.best_estimator_
        y_pred_rf = rf_best.predict(Z_te)
    # Save directory structure
        rf_dir = ensure_dir(out_dir / "RF")
    # Save best params
        with open(rf_dir / f"{target_name}_{tag_prefix}_RF_cv_best.json", "w") as f:
            json.dump({"best_params": rf_gs.best_params_,
                    "cv_best_f1_macro": float(rf_gs.best_score_)}, f, indent=2)
        labels = sorted(pd.Series(y_test).unique())
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        pd.DataFrame(cm_rf, index=sorted(pd.Series(y_test).unique()), columns=sorted(pd.Series(y_test).unique())
                    ).to_csv(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.csv", index=True)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
        plt.title(f"{target_name} — {tag_prefix}+RF — Confusion"); plt.tight_layout()
        plt.savefig(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.png", dpi=300); plt.close()

    # Classification report
        report_rf = classification_report(y_test, y_pred_rf, zero_division=0, output_dict=True)
        with open(rf_dir / f"{target_name}_{tag_prefix}_RF_test_report.json", "w") as f:
            json.dump(report_rf, f, indent=2)

        # Native statistical enrichment (QWK + Wilson 95% CI on accuracy)
        acc_rf = float(accuracy_score(y_test, y_pred_rf))
        qwk_rf = quadratic_weighted_kappa(y_test, y_pred_rf)
        n_total_rf = int(len(y_test))
        n_correct_rf = int(
            (pd.Series(y_test).reset_index(drop=True)
             == pd.Series(y_pred_rf).reset_index(drop=True)).sum()
        )
        ci_low_rf, ci_high_rf = wilson_ci_accuracy(n_correct_rf, n_total_rf)
        result_rf = {
            "accuracy": acc_rf,
            "accuracy_wilson_ci95_low": float(ci_low_rf),
            "accuracy_wilson_ci95_high": float(ci_high_rf),
            "precision_macro": float(precision_score(y_test, y_pred_rf, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, y_pred_rf, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
            "qwk": qwk_rf,
            "n_test": n_total_rf,
            "n_correct": n_correct_rf,
            "labels": [str(l) for l in labels],
            "confusion": cm_rf.tolist(),
            "report": report_rf,
            "y_pred": pd.Series(y_pred_rf).tolist(),
        }
        with open(rf_dir / f"{target_name}_{tag_prefix}_RF_test_results.json", "w") as f:
            json.dump(result_rf, f, indent=2)

        results_rows.append({
            "experiment": target_name,
            "model": f"{tag_prefix}+RF",
            "test_accuracy": acc_rf,
            "test_f1_macro": result_rf["f1_macro"],
            "test_precision_macro": result_rf["precision_macro"],
            "test_recall_macro": result_rf["recall_macro"],
            "test_qwk": qwk_rf,
            "test_accuracy_ci95_low": float(ci_low_rf),
            "test_accuracy_ci95_high": float(ci_high_rf),
        })

        print(f"[RF sur embeddings] {target_name} — "
        f"Acc: {acc_rf:.4f} [{ci_low_rf:.3f};{ci_high_rf:.3f}], "
        f"QWK: {qwk_rf:.4f}, "
        f"F1_macro: {result_rf['f1_macro']:.4f}")
    

    # ---------------------------------------------------------
    # XGBOOST HEAD
    # ---------------------------------------------------------
    if "XGBoost" in heads:
        le_y = LabelEncoder().fit(y_train.values)
        y_train_enc = le_y.transform(y_train.values)
        y_test_enc  = le_y.transform(y_test.values)

        xgb = XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss",
                            tree_method="hist", objective="multi:softprob")
        xgb_grid = {
            "n_estimators": _as_list(hparams["heads"]["XGBoost"]["n_estimators"]),
            "max_depth": _as_list(hparams["heads"]["XGBoost"]["max_depth"]),
            "learning_rate": _as_list(hparams["heads"]["XGBoost"]["learning_rate"]),
            "subsample": _as_list(hparams["heads"]["XGBoost"]["subsample"]),
            "colsample_bytree": _as_list(hparams["heads"]["XGBoost"]["colsample_bytree"]),
        }
        xgb_gs = GridSearchCV(xgb, xgb_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, return_train_score=True)
        xgb_gs.fit(Z_tr, y_train_enc)
        xgb_best = xgb_gs.best_estimator_
        y_pred_enc = xgb_best.predict(Z_te)
        y_pred_xgb = le_y.inverse_transform(y_pred_enc)

        xgb_dir = ensure_dir(out_dir / "XGBoost")
        with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_cv_best.json", "w") as f:
            json.dump({"best_params": xgb_gs.best_params_,
                    "cv_best_f1_macro": float(xgb_gs.best_score_)}, f, indent=2)
        labels = sorted(pd.Series(y_test).unique())
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        pd.DataFrame(cm_xgb, index=sorted(pd.Series(y_test).unique()), columns=sorted(pd.Series(y_test).unique())
                    ).to_csv(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.csv", index=True)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",xticklabels=labels, yticklabels=labels)
        plt.title(f"{target_name} — {tag_prefix}+XGB — Confusion"); plt.tight_layout()
        plt.savefig(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.png", dpi=300); plt.close()

        report_xgb = classification_report(y_test, y_pred_xgb, zero_division=0, output_dict=True)
        with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_test_report.json", "w") as f:
            json.dump(report_xgb, f, indent=2)

        # Native statistical enrichment (QWK + Wilson 95% CI on accuracy)
        acc_xgb = float(accuracy_score(y_test, y_pred_xgb))
        qwk_xgb = quadratic_weighted_kappa(y_test, y_pred_xgb)
        n_total_xgb = int(len(y_test))
        n_correct_xgb = int(
            (pd.Series(y_test).reset_index(drop=True)
             == pd.Series(y_pred_xgb).reset_index(drop=True)).sum()
        )
        ci_low_xgb, ci_high_xgb = wilson_ci_accuracy(n_correct_xgb, n_total_xgb)
        result_xgb = {
            "accuracy": acc_xgb,
            "accuracy_wilson_ci95_low": float(ci_low_xgb),
            "accuracy_wilson_ci95_high": float(ci_high_xgb),
            "precision_macro": float(precision_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, y_pred_xgb, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0)),
            "qwk": qwk_xgb,
            "n_test": n_total_xgb,
            "n_correct": n_correct_xgb,
            "labels": [str(l) for l in labels],
            "confusion": cm_xgb.tolist(),
            "report": report_xgb,
            "y_pred": pd.Series(y_pred_xgb).tolist(),
        }
        with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_test_results.json", "w") as f:
            json.dump(result_xgb, f, indent=2)

        results_rows.append({
            "experiment": target_name,
            "model": f"{tag_prefix}+XGB",
            "test_accuracy": acc_xgb,
            "test_f1_macro": result_xgb["f1_macro"],
            "test_precision_macro": result_xgb["precision_macro"],
            "test_recall_macro": result_xgb["recall_macro"],
            "test_qwk": qwk_xgb,
            "test_accuracy_ci95_low": float(ci_low_xgb),
            "test_accuracy_ci95_high": float(ci_high_xgb),
        })

        print(f"[XGB sur embeddings] {target_name} — "
        f"Acc: {acc_xgb:.4f} [{ci_low_xgb:.3f};{ci_high_xgb:.3f}], "
        f"QWK: {qwk_xgb:.4f}, "
        f"F1_macro: {result_xgb['f1_macro']:.4f}")
    return results_rows



# ======================================================================================
# VISUALIZATION
# ======================================================================================
def _scatter_2d(X2, y, title, save_path):
    """
    2D scatterplot helper used for UMAP visualization.

    Parameters
    ----------
    X2 : numpy.ndarray
        2D coordinates after UMAP or PCA.
    y : array-like
        Color-coded point labels.
    title : str
        Plot title.
    save_path : pathlib.Path
        Output file path.
    """
    plt.figure(figsize=(7,6))
    classes = pd.Series(y).astype(str)
    labels = classes.unique().tolist()
    for lab in labels:
        m = (classes == lab).values
        plt.scatter(X2[m,0], X2[m,1], s=12, alpha=0.75, label=str(lab))
    plt.title(title)
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_ft_embeddings(ft_clf, X, y, out_dir: Path, tag="train", max_points=10000, random_state=42):
    """
    Visualize FTTransformer embeddings using UMAP projections.

    Produces:
        • A CSV file of full embeddings
        • UMAP 2D projection colored by target
        • UMAP 2D projection colored by each categorical feature

    Parameters
    ----------
    ft_clf : FTClassifier
        Pretrained FTTransformer-based model.
    X : pandas.DataFrame
    y : array-like
    out_dir : pathlib.Path
        Output directory for figures + CSV.
    tag : str, default="train"
        Name postfix for saved files.
    max_points : int, default=10000
        Limit visualization to a random subsample for performance.
    random_state : int
        Seed for subsampling.

    Returns
    -------
    None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = ft_get_embeddings(ft_clf, X,)
    y_arr = pd.Series(y).values


    # Optional subsampling for speed
    if Z.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(Z.shape[0], size=max_points, replace=False)
        Z_plot = Z[idx]; y_plot = y_arr[idx]
        X_plot = X.iloc[idx]
    else:
        Z_plot = Z; y_plot = y_arr; X_plot = X


    # Save full embedding matrix
    df_export = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])])
    df_export["target"] = y_arr
    df_export.to_csv(out_dir / f"FTemb_{tag}_full.csv", index=False)

    # Compute UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=random_state, metric="cosine")
    Z_umap2 = reducer.fit_transform(Z_plot)

    # Plot by target
    _scatter_2d(
        Z_umap2, y_plot,
        title=f"UMAP 2D par target ({tag})",
        save_path=out_dir / f"FTemb_{tag}_UMAP2_by_target.png"
    )

# Plot by categorical features
    for col in X_plot.select_dtypes(include=["object", "category"]).columns:
        vals = X_plot[col].astype(str).values
        _scatter_2d(
            Z_umap2, vals,
            title=f"UMAP 2D par {col} ({tag})",
            save_path=out_dir / f"FTemb_{tag}_UMAP2_by_{col}.png"
        )