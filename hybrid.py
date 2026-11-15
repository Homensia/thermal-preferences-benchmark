"""
Hybrid modeling utilities: extracting deep embeddings from FTTransformers
and fitting classical ML heads (RF, XGBoost) on top.

This module implements:
    - load_ftt_cpu: load a saved E1 FTTransformer from its bytes, on the CPU
    - ft_get_embeddings: extract penultimate-layer embeddings from FTTransformer
    - FTEmbeddingHead: sklearn-compatible wrapper for hybrid models
    - run_heads_on_ft_and_save: train classical ML heads on FT embeddings

It allows combining deep feature representation learning (Transformer-based)
with classical ML classifiers for improved robustness and interpretability.

Revision (experiment E10, ``revision_runners.run_e10``): the pre-revision
code restored with minimal changes only. The heads are fitted once on the
embeddings of the archived training rows, with the paper's hyperparameters
(``config.yaml`` ``hybrid.heads``) and the seed of the E1 FTT (no head
cross-validation);
the FTT is the trained E1 model, never refitted; extraction runs on the CPU.
"""
import contextlib, hashlib, io, json
from typing import Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
)
# Native statistical helpers shared with models.py (QWK + Wilson CI).
from stats_utils import quadratic_weighted_kappa, wilson_ci_accuracy
from revision_stats import ontology, check_labels, present_class_scores, ontology_classification_report
from compute import xgb_compute_params
from ftt_adaptation import output_head
from models_classic import ParallelFitRandomForestClassifier
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#: torch threads of the CPU extraction (recorded)
EMBEDDING_TORCH_THREADS = 8




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
# CPU LOADING OF A SAVED E1 FTT
# ======================================================================================
@contextlib.contextmanager
def _torch_storages_on_cpu():
    """
    While active, torch storages unpickled by ``joblib`` are mapped to the CPU
    (the E1 FTT files hold CUDA tensors; a plain load raises with no CUDA
    device). Private torch API (``torch.storage._load_from_bytes``), pinned by
    the locked torch version; the original is always restored.
    """
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


def load_ftt_cpu(raw_bytes: bytes, n_threads: int = EMBEDDING_TORCH_THREADS):
    """
    The FTClassifier saved by E1, parsed from ``raw_bytes`` (the bytes whose
    SHA-256 the caller validated against the E1 manifest; never a path: this
    function opens no file), every tensor on the CPU, ``device = "cpu"``;
    ``torch.set_num_threads(n_threads)``. Inference only.
    """
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError(f"load_ftt_cpu takes the model bytes, not {type(raw_bytes).__name__}")
    with _torch_storages_on_cpu():
        ft_clf = joblib.load(io.BytesIO(bytes(raw_bytes)))
    ft_clf.device = "cpu"
    if any(t.device.type != "cpu" for t in list(ft_clf.model_.parameters()) + list(ft_clf.model_.buffers())):
        raise RuntimeError("FTT tensors not on the CPU after load_ftt_cpu")
    torch.set_num_threads(int(n_threads))
    return ft_clf


def array_sha256(Z: np.ndarray) -> str:
    """SHA-256 of the C-contiguous bytes of an embedding matrix (recorded, never stored)."""
    return hashlib.sha256(np.ascontiguousarray(Z).tobytes()).hexdigest()


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
    # Guard: the layer found by its out_features is the output head to_logits[2]
    if last_lin is not output_head(ft_clf.model_, num_classes):
        raise RuntimeError("the last Linear(·, n_classes) is not the FTTransformer output head to_logits[2]")

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

    A head fitted by ``run_heads_on_ft_and_save`` is wrapped by
    ``from_fitted`` (the wrapper never fits):

        1. Extract embeddings from pretrained FTTransformer.
        2. Apply the fitted head; decode its predictions with the label
           encoder only when the head was fitted on encoded labels (XGB).

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
    le_y_ : LabelEncoder or None
        Encoder of the target labels used to fit the head (XGB), ``None`` when
        the head was fitted on the raw labels (RF).
    classes_ : numpy.ndarray
        Label of column j of ``predict_proba`` (RF: ``head.classes_``; XGB:
        ``le_y_.classes_``, code j = ``le_y_.classes_[j]``).
    last_Z_sha256_ : str
        SHA-256 of the last embedding matrix computed by ``predict`` /
        ``predict_proba``.
    """

    def __init__(self, ft_clf, head_estimator):
        self.ft_clf = ft_clf
        self.head_estimator = head_estimator

    @classmethod
    def from_fitted(cls, ft_clf, head, label_encoder: Optional[LabelEncoder] = None):
        """Wrap a head already fitted on the embeddings of ``ft_clf`` (raw labels, or ``label_encoder`` codes)."""
        obj = cls(ft_clf, head)
        obj.head_estimator_ = head
        obj.le_y_ = label_encoder
        if label_encoder is None:
            obj.classes_ = np.asarray(head.classes_)
        else:
            if not np.array_equal(np.asarray(head.classes_), np.arange(len(label_encoder.classes_))):
                raise ValueError("the head was not fitted on the codes 0..k-1 of label_encoder")
            obj.classes_ = np.asarray(label_encoder.classes_)
        return obj

    def _embeddings(self, X):
        Z = ft_get_embeddings(self.ft_clf, X)
        self.last_Z_sha256_ = array_sha256(Z)
        return Z

    def predict(self, X):
        """
        Predict using ML head applied to Transformer embeddings.

        Returns
        -------
        array-like
            Decoded class predictions.
        """
        Z = self._embeddings(X)
        y_pred_enc = self.head_estimator_.predict(Z)
        if self.le_y_ is None:
            return y_pred_enc
        return self.le_y_.inverse_transform(y_pred_enc)

    def predict_proba(self, X):
        """Head probabilities on the embeddings; column j is ``classes_[j]``."""
        return self.head_estimator_.predict_proba(self._embeddings(X))

# ======================================================================================
# HYBRID TRAINING PIPELINE
# ======================================================================================
def run_heads_on_ft_and_save(ft_clf, X_train, y_train, X_test, y_test, out_dir: Path,
                             target_name: str, tag_prefix="FTemb",
                             heads: Optional[List[str]] = None,hparams=None, *, seed: int, compute: dict):
    hparams = hparams or {}
    
    """
    Train classical ML heads (RF, XGB) on FTTransformer embeddings and save results.

    For each requested head:
        - The head is fitted once on the training embeddings with the single
          hyperparameter point of ``hparams`` and ``random_state = seed``
          (no head cross-validation).
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
    heads : list of str or None
        List of heads to train, among {"RF", "XGBoost"}.
    seed : int
        ``random_state`` of every head (the seed of the E1 FTT).
    compute : dict
        ``compute`` section of the run (RF / XGB host threads, XGB device).

    Returns
    -------
    list[dict]
        One result dict per trained head; besides the pre-revision keys, the
        fitted head (``estimator``), its label encoder (``label_encoder``,
        XGB only) and the SHA-256 of the training / test embeddings.
    """
    if heads is None:
        heads = ["RF", "XGBoost"]

    out_dir.mkdir(parents=True, exist_ok=True)
    # Full target ontology for QWK and confusion matrices;
    # labels outside it raise (never added, never dropped).
    labels = list(ontology(target_name))
    y_test = check_labels(y_test, labels, "y_test")

    # Compute embeddings once for efficiency
    Z_tr = ft_get_embeddings(ft_clf, X_train)
    Z_te = ft_get_embeddings(ft_clf, X_test)
    Z_hashes = {"Z_train_sha256": array_sha256(Z_tr), "Z_test_sha256": array_sha256(Z_te),
                "Z_train_shape": list(Z_tr.shape), "Z_test_shape": list(Z_te.shape)}


    results_rows = []


    # ---------------------------------------------------------
    # RANDOM FOREST HEAD
    # ---------------------------------------------------------   
    # Single fit with the one hyperparameter point (the pre-revision one-point
    # GridSearchCV refit, without its leaky 5-fold CV on in-sample embeddings).
    if "RF" in heads:
        rf_best = ParallelFitRandomForestClassifier(
            random_state=seed,
            class_weight=hparams["heads"]["RF"].get("class_weight", None),
            n_estimators=hparams["heads"]["RF"]["n_estimators"],
            max_depth=hparams["heads"]["RF"]["max_depth"],
            min_samples_split=hparams["heads"]["RF"]["min_samples_split"],
            min_samples_leaf=hparams["heads"]["RF"]["min_samples_leaf"],
            n_jobs=compute["cpu_jobs"],
        )
        rf_best.fit(Z_tr, y_train.values)
        y_pred_rf = check_labels(rf_best.predict(Z_te), labels, "y_pred_rf")
    # Save directory structure
        rf_dir = ensure_dir(out_dir / "RF")
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels)
        pd.DataFrame(cm_rf, index=labels, columns=labels
                    ).to_csv(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.csv", index=True)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
        plt.title(f"{target_name} — {tag_prefix}+RF — Confusion"); plt.tight_layout()
        plt.savefig(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.png", dpi=300); plt.close()

    # Classification report
        report_rf = ontology_classification_report(y_test, y_pred_rf, labels=labels)
        with open(rf_dir / f"{target_name}_{tag_prefix}_RF_test_report.json", "w") as f:
            json.dump(report_rf, f, indent=2)

        # Native statistical enrichment (QWK + Wilson 95% CI on accuracy)
        acc_rf = float(accuracy_score(y_test, y_pred_rf))
        qwk_rf = quadratic_weighted_kappa(y_test, y_pred_rf, labels=labels)
        n_total_rf = int(len(y_test))
        n_correct_rf = int(
            (pd.Series(y_test).reset_index(drop=True)
             == pd.Series(y_pred_rf).reset_index(drop=True)).sum()
        )
        ci_low_rf, ci_high_rf = wilson_ci_accuracy(n_correct_rf, n_total_rf)
        # Balanced accuracy and macro-F1 over the classes present in y_test.
        scores_rf = present_class_scores(y_test, y_pred_rf, labels=labels)
        result_rf = {
            "accuracy": acc_rf,
            "accuracy_wilson_ci95_low": float(ci_low_rf),
            "accuracy_wilson_ci95_high": float(ci_high_rf),
            "balanced_accuracy": scores_rf["balanced_accuracy"],
            "balanced_accuracy_single_class": scores_rf["balanced_accuracy_single_class"],
            "precision_macro": report_rf["macro avg"]["precision"],
            "recall_macro": report_rf["macro avg"]["recall"],
            "f1_macro": scores_rf["macro_f1"],
            "f1_micro": float(f1_score(y_test, y_pred_rf, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
            "qwk": qwk_rf,
            "qwk_undefined": bool(np.isnan(qwk_rf)),
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
            "head": "RF", "estimator": rf_best, "label_encoder": None, **Z_hashes,
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

        # Objective left to XGBoost: multi:softprob (with num_class) for >= 3
        # training classes, binary:logistic when only 2 remain (an explicit
        # multi:softprob fails on 2 classes); identical fits for >= 3 classes.
        # tree_method "hist", device and host threads from the run's compute section
        xgb_best = XGBClassifier(random_state=seed, eval_metric="mlogloss",
                                 n_estimators=hparams["heads"]["XGBoost"]["n_estimators"],
                                 max_depth=hparams["heads"]["XGBoost"]["max_depth"],
                                 learning_rate=hparams["heads"]["XGBoost"]["learning_rate"],
                                 subsample=hparams["heads"]["XGBoost"]["subsample"],
                                 colsample_bytree=hparams["heads"]["XGBoost"]["colsample_bytree"],
                                 **xgb_compute_params(compute))
        xgb_best.fit(Z_tr, y_train_enc)
        y_pred_enc = xgb_best.predict(Z_te)
        y_pred_xgb = check_labels(le_y.inverse_transform(y_pred_enc), labels, "y_pred_xgb")

        xgb_dir = ensure_dir(out_dir / "XGBoost")
        cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=labels)
        pd.DataFrame(cm_xgb, index=labels, columns=labels
                    ).to_csv(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.csv", index=True)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",xticklabels=labels, yticklabels=labels)
        plt.title(f"{target_name} — {tag_prefix}+XGB — Confusion"); plt.tight_layout()
        plt.savefig(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.png", dpi=300); plt.close()

        report_xgb = ontology_classification_report(y_test, y_pred_xgb, labels=labels)
        with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_test_report.json", "w") as f:
            json.dump(report_xgb, f, indent=2)

        # Native statistical enrichment (QWK + Wilson 95% CI on accuracy)
        acc_xgb = float(accuracy_score(y_test, y_pred_xgb))
        qwk_xgb = quadratic_weighted_kappa(y_test, y_pred_xgb, labels=labels)
        n_total_xgb = int(len(y_test))
        n_correct_xgb = int(
            (pd.Series(y_test).reset_index(drop=True)
             == pd.Series(y_pred_xgb).reset_index(drop=True)).sum()
        )
        ci_low_xgb, ci_high_xgb = wilson_ci_accuracy(n_correct_xgb, n_total_xgb)
        # Balanced accuracy and macro-F1 over the classes present in y_test.
        scores_xgb = present_class_scores(y_test, y_pred_xgb, labels=labels)
        result_xgb = {
            "accuracy": acc_xgb,
            "accuracy_wilson_ci95_low": float(ci_low_xgb),
            "accuracy_wilson_ci95_high": float(ci_high_xgb),
            "balanced_accuracy": scores_xgb["balanced_accuracy"],
            "balanced_accuracy_single_class": scores_xgb["balanced_accuracy_single_class"],
            "precision_macro": report_xgb["macro avg"]["precision"],
            "recall_macro": report_xgb["macro avg"]["recall"],
            "f1_macro": scores_xgb["macro_f1"],
            "f1_micro": float(f1_score(y_test, y_pred_xgb, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0)),
            "qwk": qwk_xgb,
            "qwk_undefined": bool(np.isnan(qwk_xgb)),
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
            "head": "XGBoost", "estimator": xgb_best, "label_encoder": le_y, **Z_hashes,
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
