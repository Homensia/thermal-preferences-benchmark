"""
Full unified training pipeline for thermal comfort prediction.

This module orchestrates the entire experimental workflow, combining:

    • Classical ML models (RandomForest, SVM, XGBoost)
    • Deep learning models (ANN, FTTransformer)
    • Cross-validation, grid-search, evaluation, visualization
    • Feature importance computation
    • Train/test split management
    • Logging, reproducibility
    • Unified data loading / validation

The FT-embedding hybrids (FTT+RF / FTT+XGB) are not run by this pipeline
(see ``hybrid.py``).

Experiment E1 of the revision (ASHRAE-2022) runs through
``python3 run_revision.py e1 --run_dir RUN`` (``revision_runners.run_e1``),
which writes the run manifest that the later experiments require; this CLI writes
no manifest and refuses ``--dataset ASHRAE_2022``. It remains the pre-revision
pipeline of the Phase 1 dataset (not rerun).

The pipeline:
    1. Loads and validates datasets via UnifiedDataLoader
    2. Loads the archived stratified train/test split and materialises the
       folds once per target (``revision_folds``)
    3. Runs selected models (classical, deep) with the frozen seeds
    4. Performs model selection via GridSearchCV or manual search
    5. Evaluates on held-out test data
    6. Saves every artifact (curves, confusion matrices, reports, models)
    7. Produces global comparison summary
"""
import os, random, time, json
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import logging
import pandas as pd
import torch, platform
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer,
)
from stats_utils import (
    quadratic_weighted_kappa,
    wilson_ci_accuracy,
)
from revision_stats import (
    ontology, check_labels, present_class_scores,
    ontology_classification_report, format_ontology_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import joblib
from category_normalisation import (
    oov_report, oov_summary, split_feature_types, vocabulary_for, write_coverage_log,
)
from unified_data_loader import ( UnifiedDataLoader, create_label_mappings_from_models)
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.nn.functional as F
from models_deep import (   select_best, 
                            get_torch_models,
                            save_torch_curves,
                            early_stopping_record,
                            RevisionNotImplementedError,
                         )
from models_classic import (    get_models, CLASSICAL_FAMILIES,
                                run_grid_search_on_train
                            )
from config_loader import load_config, check_not_protected
from compute import load_compute
import revision_folds




# ======================================================================================
# GLOBAL CONFIGURATION
# ======================================================================================
CONFIG = load_config()


RANDOM_SEED = CONFIG["seed"]
FEATURES_ALL = CONFIG["data"]["features"]
TARGETS_ALL = CONFIG["data"]["targets"]
base_out = Path("kfold_results_unified")


# ======================================================================================
# SCORERS FOR CROSS-VALIDATION
# ======================================================================================
SCORERS = {
    "f1_macro": make_scorer(lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
    "f1_micro": make_scorer(lambda yt, yp: f1_score(yt, yp, average="micro",  zero_division=0)),
    "f1_weighted": make_scorer(lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)),
    "precision_macro": make_scorer(lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0)),
    "recall_macro": make_scorer(lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0)),
    "accuracy": make_scorer(accuracy_score),
}

# =========================================================================================
# ===== REPRODUCIBILITY SEEDS =====
# =========================================================================================

def set_all_seeds(seed=RANDOM_SEED):
    """
    Set all possible seeds for reproducibility
    """
    print(f"🔧 Setting all seeds to {seed} for reproducibility...")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("   ✅ All seeds set successfully")

# Set seeds at import time
set_all_seeds()

print("PyTorch:", torch.__version__)
print("Python:", platform.python_version())
print("CUDA dispo ?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))




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
def plot_confusion_matrix(cm, labels, title="Matrice de confusion",save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.
    labels : list of str or int
        Row/column labels.
    title : str, optional
        Figure title.
    save_path : pathlib.Path or None
        If provided, saves the figure.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300) 
    plt.show()



# ======================================================================================
# FEATURE IMPORTANCE FOR PIPELINES
# ======================================================================================


def _get_transformed_feature_names_and_groups(pp: ColumnTransformer):
    """
    Retrieve transformed feature names and underlying original feature groups.

    Useful for aggregating OHE-expanded features back to original categorical variables.

    Parameters
    ----------
    pp : ColumnTransformer
        The preprocessing pipeline containing scalers and encoders.

    Returns
    -------
    feat_out : np.ndarray
        Names of transformed features.
    groups : np.ndarray
        Corresponding original feature groups.
    """
    feat_out, groups = [], []
    for name, trans, cols in pp.transformers_:
        if name == "remainder":
            continue

        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        else:
            names = np.array(cols, dtype=object)

        feat_out.extend(names.tolist())

        if name == "num":
            for c in names:
                groups.append(str(c).split("__", 1)[-1])
        elif name == "ohe" and hasattr(trans, "categories_"):
            for c in names:
                col_orig = str(c).split("__", 1)[-1]  
                groups.append(col_orig.split("_", 1)[0])
        else:
            for c in names:
                groups.append(str(c))

    return np.array(feat_out, dtype=object), np.array(groups, dtype=object)


def _extract_feature_importances_from_pipeline(pipe):
    """
    Extract feature importances from a sklearn Pipeline (pp + clf).

    Supports:
        - RandomForest.feature_importances_
        - XGBoost.feature_importances_
        - LabelEncodedClassifier(estimator_.feature_importances_)

    Parameters
    ----------
    pipe : sklearn.Pipeline

    Returns
    -------
    tuple or None
        (importances, feat_out, groups) or None if unavailable.
    """
    if not hasattr(pipe, "named_steps"):
        return None

    pp = pipe.named_steps.get("pp", None)
    clf = pipe.named_steps.get("clf", None)
    if pp is None or clf is None:
        return None

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "estimator_") and hasattr(clf.estimator_, "feature_importances_"):
        importances = clf.estimator_.feature_importances_

    if importances is None:
        return None

    try:
        feat_out, groups = _get_transformed_feature_names_and_groups(pp)
    except Exception:
        feat_out = getattr(pp, "get_feature_names_out", lambda: np.arange(len(importances)))()
        groups = np.array([str(f) for f in feat_out])

    if len(importances) != len(feat_out):
        print(f"[WARN] importances({len(importances)}) != features_out({len(feat_out)}). "
              f"Utilisation d'index générique.")
        feat_out = np.array([f"f{i}" for i in range(len(importances))])
        groups = feat_out.copy()

    return np.asarray(importances, dtype=float), feat_out, groups


def plot_feature_importance_pipeline(pipe, model_name, target_name, save_dir, top_n=None, save_csv=True):
    """
    Plot aggregated feature importances for tree-based models.

    Steps:
        1. Extract post-transformer features (num + OHE)
        2. Aggregate importances by original variable (sum of OHE parts)
        3. Normalize importances
        4. Save CSV and PNG bar plot

    Parameters
    ----------
    pipe : sklearn.Pipeline
        Model pipeline (preprocessing + classifier).
    model_name : str
    target_name : str
    save_dir : pathlib.Path
        Output directory for plots.
    top_n : int or None
        Display only top N features; if None, show all.
    save_csv : bool
        Whether to save CSV file.
    """
    extr = _extract_feature_importances_from_pipeline(pipe)
    if extr is None:
        print(f"[WARN] Aucune importance dispo pour {model_name}.")
        return

    importances, feat_out, groups = extr

    df_imp = pd.DataFrame({"feat_out": feat_out, "group": groups, "importance": importances})
    grouped = df_imp.groupby("group", as_index=True)["importance"].sum()

    grouped = grouped / grouped.sum()

    grouped = grouped.sort_values(ascending=False)
    if top_n is not None:
        grouped = grouped.head(top_n)

    save_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        grouped.to_csv(save_dir / f"{target_name}_{model_name}_feature_importance_aggregated.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.values, y=grouped.index, color="green")
    plt.xlabel("Feature Importances (normalized)")
    plt.ylabel("Features")
    plt.title(f"{model_name} – {target_name}")
    plt.tight_layout()
    plt.savefig(save_dir / f"{target_name}_{model_name}_feature_importance.png", dpi=300)
    plt.close()

    print(f"✅ Saved: {save_dir / f'{target_name}_{model_name}_feature_importance.png'}")


# ======================================================================================
# DATA SPLITTING
# ======================================================================================
def split_train_test_stratified(X, y, test_size=0.20, seed=RANDOM_SEED):
    """
    Perform a stratified train/test split.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    test_size : float
    seed : int

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of pandas.DataFrame/Series
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y))
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]


def eval_on_test(model, X_test, y_test, name="model", labels_order=None, target_names=None,save_dir: Path=None):
    """
    Evaluate a fitted model on held-out test data.

    Produces:
        • Accuracy / Precision / Recall / F1 (macro/micro/weighted)
        • Classification report
        • Confusion matrix (saved as CSV + PNG)
        • Prediction list

    Parameters
    ----------
    model : sklearn estimator or PyTorch wrapper
        Fitted model.
    X_test : pandas.DataFrame
    y_test : pandas.Series
    name : str
        Prefix name for saved files.
    labels_order : sequence of int, required
        Full target ontology: pass ``revision_stats.ontology(target)``.
        Used unchanged for QWK, the confusion matrix and the report; true or
        predicted labels outside it raise (never added, never dropped).
    target_names : list, optional
        Human-readable target class names.
    save_dir : pathlib.Path or None

    Returns
    -------
    dict
        Summary metrics, confusion matrix, report, and predictions.
    """
    if labels_order is None:
        raise ValueError(
            "eval_on_test needs labels_order = revision_stats.ontology(target); "
            "labels are never inferred from the evaluated set"
        )
    labels_order = [int(l) for l in labels_order]
    # Validate (finite, integral, inside the ontology) and cast to int: a label
    # outside the ontology raises instead of extending the label list.
    y_test = check_labels(y_test, labels_order, "y_test")
    y_pred = check_labels(model.predict(X_test), labels_order, "y_pred")
    if len(y_pred) != len(y_test):
        raise ValueError(f"{name}: {len(y_pred)} predictions for {len(y_test)} test rows")
    if target_names is None:
        target_names = [str(l) for l in labels_order]

    acc = float(accuracy_score(y_test, y_pred))
    # Balanced accuracy and macro-F1 over the classes present in y_test.
    present_scores = present_class_scores(y_test, y_pred, labels=labels_order)
    f1m = present_scores["macro_f1"]
    bal = present_scores["balanced_accuracy"]
    # Detailed report: every ontology class as a row; "macro avg" over the
    # classes present in y_test (same denominator as f1_macro).
    report = ontology_classification_report(
        y_test, y_pred, labels=labels_order, target_names=target_names
    )
    prec = report["macro avg"]["precision"]
    rec  = report["macro avg"]["recall"]
    f1mi = float(f1_score(y_test, y_pred, average="micro",   zero_division=0))
    f1w  = float(f1_score(y_test, y_pred, average="weighted",zero_division=0))

    # Native statistical enrichment (QWK + Wilson 95% CI on accuracy)
    # QWK on the full target ontology, unchanged; NaN + flag when undefined.
    qwk = quadratic_weighted_kappa(y_test, y_pred, labels=labels_order)
    n_total = int(len(y_test))
    n_correct = int(
        (pd.Series(y_test).reset_index(drop=True)
         == pd.Series(y_pred).reset_index(drop=True)).sum()
    )
    ci_low, ci_high = wilson_ci_accuracy(n_correct, n_total)

    per_class_f1 = {lbl: report[str(lbl)]["f1-score"] for lbl in labels_order if str(lbl) in report}
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    result = {
        "accuracy": acc,
        "accuracy_wilson_ci95_low": float(ci_low),
        "accuracy_wilson_ci95_high": float(ci_high),
        "balanced_accuracy": bal,
        "balanced_accuracy_single_class": present_scores["balanced_accuracy_single_class"],
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1m,
        "f1_micro": f1mi,
        "f1_weighted": f1w,
        "qwk": qwk,
        "qwk_undefined": bool(np.isnan(qwk)),
        "n_test": n_total,
        "n_correct": n_correct,
        "labels": [str(l) for l in labels_order],
        "confusion": cm.tolist(),
        "report": report,
        "y_pred": pd.Series(y_pred).tolist(),
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        cm_df.to_csv(save_dir / f"{name}_confusion_matrix.csv")
        plot_confusion_matrix(cm, labels_order, title=f"{name} — Confusion",
                              save_path=save_dir / f"{name}_confusion_matrix.png")
        with open(save_dir / f"{name}_test_report.json","w") as f:
            json.dump(report, f, indent=2)
        # Native test_results.json — self-contained for post-hoc tools.
        with open(save_dir / f"{name}_test_results.json", "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n--- TEST FINAL — {name} ---")
    print(f"Acc: {acc:.4f} [{ci_low:.3f};{ci_high:.3f}] | QWK: {qwk:.4f} | "
          f"F1_macro: {f1m:.4f}| F1_micro: {f1mi:.4f}| F1_weighted: {f1w:.4f}")
    print("F1 par classe   :", {str(k): f"{v:.3f}" for k, v in per_class_f1.items()})
    print("\n--- Rapport détaillé ---")
    print(format_ontology_report(report))

    return result









# ======================================================================================
# MAIN TRAINING WORKFLOW
# ======================================================================================

def deep_models_requested(args) -> bool:
    """True when the CLI arguments select the ANN/FTT stage."""
    return ("deep" in args.models) or \
           ("all" in args.models and args.classical_models == ["all"])


def run_experiment_for_target(
    TARGET: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    exp_dir: Path,
    args=None,
    cat_vocabulary=None,
    cv_splits=None,
    seeds_by_model=None) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]] : 


    """
    Run the full experimental workflow for a single target variable.

    Includes:
        - Classical ML (RF, SVM, XGBoost)
        - Deep models (ANN, FTTransformer)
        - CV, grid-search, metrics, visualizations

    Every family uses the same materialised folds (``cv_splits``)
    and the frozen model seeds (``seeds_by_model``); one fit, evaluation and
    output directory ``{model}/seed_{seed}/`` per model and seed. No test over
    folds is computed (the fold-number Wilcoxon pairing is removed).

    Parameters
    ----------
    TARGET : str
        Name of target variable.
    X_train, y_train : pandas.DataFrame / Series
        Training set.
    X_test, y_test : pandas.DataFrame / Series
        Test set.
    num_cols : list[str]
        Numerical columns.
    cat_cols : list[str]
        Categorical columns.
    exp_dir : pathlib.Path
        Directory where results for this target will be saved.
    args : argparse.Namespace or None
        CLI arguments controlling which models to run.
    cat_vocabulary : dict or None
        Frozen category vocabulary used by the one-hot, ANN and
        FTT encoders; ``None`` only for the Phase 1 ASHRAE-2018 v2 path.
    cv_splits : list of (array, array)
        Materialised folds of the target (``revision_folds.FoldAssignment.splits()``),
        positions in ``X_train``. Required.
    seeds_by_model : dict or None
        Model name → seeds (``revision_folds.model_seeds()``); ``None`` = the
        frozen defaults (RF/XGB 42, 43, 44; SVM, ANN, FTT 42).

    Returns
    -------
    tuple
        (best_models, fold_summaries, stats, test_results); ``best_models`` maps
        ``rf``/``xgb``/``svm``/``ft`` to ``{seed: fitted model}``.
    """
    print(f"\n{'='*120}")
    print(f"🎯 TARGET = {TARGET}")
    print(f"{'='*120}")

    if cv_splits is None:
        raise ValueError("run_experiment_for_target needs the materialised folds (cv_splits)")
    seeds_by_model = seeds_by_model or revision_folds.model_seeds()
    # Fold-vector hash of the shared folds (also checks they partition X_train);
    # every family's fold summary must carry it.
    folds_sha256 = revision_folds.splits_sha256(cv_splits, len(X_train))

    def check_same_folds(fold_summary_df, tag):
        used = set(fold_summary_df["folds_sha256"])
        if used != {folds_sha256}:
            raise revision_folds.FoldIntegrityError(
                f"{TARGET} {tag}: family used different folds {sorted(map(str, used))} != {folds_sha256}")

    best_models = {
    "rf": {},
    "xgb": {},
    "svm": {},
    "ft": {},
        }
    best_key = {"RandomForest": "rf", "XGBoost": "xgb", "SVM": "svm", "FTTransformer": "ft"}


    if args is None:
        class DefaultArgs:
            models = ["all"]
            classical_models = ["all"]
            deep_models = ["all"]
            skip_feature_importance = False
        args = DefaultArgs()


    # Determine which model types to run
    run_classical = ("classical" in args.models) or \
                ("all" in args.models and args.deep_models == ["all"])

    run_deep = deep_models_requested(args)


    

    # Get model configurations (one set per seed: the seed is the estimator's random_state)
    def classical_models_for(seed, family):
        # only the requested family is built: the XGB CUDA check runs only for XGB fits
        return get_models(num_cols, cat_cols, seed=seed, hparams=CONFIG["classical_hparams"],
                          cat_vocabulary=cat_vocabulary, compute=COMPUTE, families=[family])[family]

    def torch_models_for(seed):
        return get_torch_models(num_cols, cat_cols, CONFIG["deep_hparams"], seed=seed,
                                cat_vocabulary=cat_vocabulary, compute=COMPUTE)

    COMPUTE = load_compute(CONFIG)    # devices and CPU bounds

    MODELS_SK    = list(CLASSICAL_FAMILIES) if run_classical else []
    MODELS_TORCH = list(torch_models_for(RANDOM_SEED)) if run_deep else []


    # Filter classical models
    if run_classical and "all" not in args.classical_models:
        MODELS_SK = [k for k in MODELS_SK if k in args.classical_models]
    
    # Filter deep models
    if run_deep and "all" not in args.deep_models:
        MODELS_TORCH = [k for k in MODELS_TORCH if k in args.deep_models]



    # Storage for results
    per_model_fold_summaries = []
    per_model_stats = []
    per_model_test = []



    # ==================================================================================
    # CLASSICAL MODELS (RandomForest, SVM, XGBoost)
    # ==================================================================================



    if run_classical and MODELS_SK:
        print("\n" + "="*80)
        print("CLASSICAL MODELS")
        print("="*80)
        for model_name in MODELS_SK:
            for seed in seeds_by_model[model_name]:
                pipe, pgrid = classical_models_for(seed, model_name)
                best_est, best_params, fold_summary_df, stats_df, _ = run_grid_search_on_train(
                    TARGET, model_name, pipe, pgrid, X_train, y_train,
                    refit_metric="f1_macro", cv_splits=cv_splits, verbose=2, compute=COMPUTE
                            )

                tag = f"{model_name}__seed{seed}"
                check_same_folds(fold_summary_df, tag)
                mdir = ensure_dir(exp_dir / model_name / f"seed_{seed}")

                test_res = eval_on_test(
                    best_est, X_test, y_test,
                    name=f"{TARGET}__{tag}",
                    labels_order=ontology(TARGET),
                    save_dir=mdir
                    )
                test_res["seed"] = seed
                if model_name in ["RandomForest", "XGBoost"] and (not args.skip_feature_importance):
                    plot_feature_importance_pipeline(
                    pipe=best_est,
                    model_name=tag,
                    target_name=TARGET,
                    save_dir=mdir,
                    top_n=None,          
                    save_csv=True
                    )




                best_models[best_key[model_name]][seed] = best_est

                fold_summary_df = fold_summary_df.assign(seed=seed)
                stats_df = stats_df.assign(seed=seed)
                fold_summary_df.to_csv(mdir / f"{TARGET}_{tag}_fold_summary.csv", index=False)
                stats_df.to_csv(mdir / f"{TARGET}_{tag}_fold_statistics.csv", index=False)

                pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])\
                    .to_csv(mdir / f"{TARGET}_{tag}_confusion_matrix.csv")

                with open(mdir / f"{TARGET}_{tag}_test_results.json", "w") as f:
                    json.dump(test_res, f, indent=2)

                per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
                per_model_stats.append(stats_df.assign(model=model_name))
                per_model_test.append({
                 "experiment": TARGET, "model": model_name, "seed": seed,
                 "test_accuracy": test_res["accuracy"],
                 "test_f1_macro": test_res["f1_macro"],
                 "test_precision_macro": test_res["precision_macro"],
                    "test_recall_macro": test_res["recall_macro"]
                })
    else:
        print("\n⏭️  Skipping classical models")

    # ==================================================================================
    # DEEP LEARNING MODELS (ANN, FTTransformer)
    # ==================================================================================
    if run_deep and MODELS_TORCH:
        print("\n" + "="*80)
        print("DEEP LEARNING MODELS")
        print("="*80)

        for model_name in MODELS_TORCH:
            for seed in seeds_by_model[model_name]:
                est, pgrid = torch_models_for(seed)[model_name]
                best_model, best_params, fold_summary_df, stats_df, per_param_val = \
                    select_best(TARGET, model_name, est, pgrid, X_train, y_train, cv_splits=cv_splits)


                tag = f"{model_name}__seed{seed}"
                check_same_folds(fold_summary_df, tag)
                mdir = ensure_dir(exp_dir / model_name / f"seed_{seed}")
                # Validation history and internal validation subset
                # (es_val_idx, cohort row indices) of the final model itself.
                save_torch_curves(best_model, mdir / f"{TARGET}_{tag}_training_curves.csv")
                with open(mdir / f"{TARGET}_{tag}_early_stopping.json", "w") as f:
                    json.dump(early_stopping_record(best_model), f, indent=2)
                test_res = eval_on_test(best_model, X_test, y_test, name=f"{TARGET}__{tag}",
                                        labels_order=ontology(TARGET), save_dir=mdir)
                test_res["seed"] = seed

                if model_name in best_key:
                    best_models[best_key[model_name]][seed] = best_model


                fold_summary_df = fold_summary_df.assign(seed=seed)
                stats_df = stats_df.assign(seed=seed)
                fold_summary_df.to_csv(mdir / f"{TARGET}_{tag}_fold_summary.csv", index=False)
                stats_df.to_csv(mdir / f"{TARGET}_{tag}_fold_statistics.csv", index=False)
                with open(mdir / f"{TARGET}_{tag}_test_results.json", "w") as f:
                    json.dump(test_res, f, indent=2)
                with open(mdir / f"{TARGET}_{tag}_param_grid_cv.json", "w") as f:
                    json.dump(per_param_val, f, indent=2)
                cm_df = pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])
                cm_df.to_csv(mdir / f"{TARGET}_{tag}_confusion_matrix.csv")

                plot_confusion_matrix(test_res["confusion"], test_res["labels"], title=f"{TARGET} - {tag}",save_path=mdir / f"{TARGET}_{tag}_confusion_matrix.png")

                per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
                per_model_stats.append(stats_df.assign(model=model_name))
                per_model_test.append({
                    "experiment": TARGET, "model": model_name, "seed": seed,
                    "test_accuracy": test_res["accuracy"],
                    "test_f1_macro": test_res["f1_macro"],
                    "test_precision_macro": test_res["precision_macro"],
                    "test_recall_macro": test_res["recall_macro"]
                })
    else:
        print("\n⏭️  Skipping deep learning models")

    # ==================================================================================
    # SAVE AGGREGATED RESULTS
    # ==================================================================================
    
    if per_model_fold_summaries:
        # Fold scores are descriptive only; no test over folds.
        all_fold_df = pd.concat(per_model_fold_summaries, ignore_index=True)
        all_fold_df.to_csv(
            exp_dir / f"{TARGET}_ALLMODELS_fold_summary.csv", index=False
        )

    if per_model_stats:
        pd.concat(per_model_stats, ignore_index=True).to_csv(
            exp_dir / f"{TARGET}_ALLMODELS_fold_statistics.csv", index=False
        )
    
    pd.DataFrame(per_model_test).to_csv(
        exp_dir / f"{TARGET}_ALLMODELS_test_scores.csv", index=False
    )

    print(f"\n✅ Results saved to: {exp_dir}")
    
    return best_models, per_model_fold_summaries, per_model_stats, per_model_test


def setup_logging(verbose: bool) -> None:
    """
    Configure logging verbosity.

    Parameters
    ----------
    verbose : bool
        If True → DEBUG level, else INFO level.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args():
    """
    Parse command-line arguments controlling the pipeline.

    Supported options:
        - target selection
        - model selection
        - classical / deep specification
        - skipping feature importance
        - archived split, materialised folds, model seeds
        - output directory, verbosity

    Returns
    -------
    argparse.Namespace
    """
    import argparse
    
    p = argparse.ArgumentParser(
        description="Thermal Comfort Prediction - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python models.py
  
  # Run only thermal_sensation target
  python models.py --targets thermal_sensation
  
  # Run only classical models
  python models.py --models classical
  
  
  # Quick test with one target and classical models only
  python models.py --targets thermal_sensation --models classical --skip_feature_importance
        """
    )
     
    # Target selection
    p.add_argument(
        "--targets",
        type=str,
        nargs="+",
        choices=["thermal_sensation", "TSV_3p", "thermal_preference", "all"],
        default=["all"],
        help="Target variables to process (default: all)"
    )
    
    # Model selection
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["classical", "deep", "all"],
        default=["all"],
        help="Model types to train (default: all)"
    )
    
    # Classical models selection
    p.add_argument(
        "--classical_models",
        type=str,
        nargs="+",
        choices=["RandomForest", "SVM", "XGBoost", "all"],
        default=["all"],
        help="Which classical models to train (default: all)"
    )
    
    # Deep models selection
    p.add_argument(
        "--deep_models",
        type=str,
        nargs="+",
        choices=["ANN", "FTTransformer", "all"],
        default=["all"],
        help="Which deep models to train (default: all)"
    )

    
    # Features and outputs
    p.add_argument(
        "--skip_feature_importance",
        action="store_true",
        help="Skip feature importance plots for tree-based models"
    )
    
    # Cross-validation (K = 10 frozen)
    p.add_argument(
        "--cv_folds",
        type=int,
        default=revision_folds.FOLDS_K,
        help=f"Number of CV folds; must equal the frozen K = {revision_folds.FOLDS_K}"
    )

    # Archived split and materialised folds
    p.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory of the archived {target}_split_indices.json (default: the dataset's "
             "split_dir in config.yaml)"
    )
    p.add_argument(
        "--folds_from",
        type=str,
        default=None,
        help="Directory of {target}_folds.json: loaded and checked if present, written once "
             "otherwise (default: the dataset's folds_dir in config.yaml)"
    )
    
    # Dataset selection (Phase 1 replication uses ASHRAE_2018_v2, Phase 2 uses ASHRAE_2022)
    p.add_argument(
        "--dataset",
        type=str,
        choices=["ASHRAE_2022", "ASHRAE_2018_v2"],
        default="ASHRAE_2022",
        help="Which dataset to train on (default: ASHRAE_2022, 17 features). Use ASHRAE_2018_v2 for Phase 1 Haghirad reproduction (12 features)."
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="kfold_results_unified",
        help="Output directory for results (default: kfold_results_unified)"
    )
    
    # Model seeds (split, folds and validation subsets stay at 42)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(revision_folds.TREE_SEEDS),
        help="RF / XGB model seeds; must be 42 43 44. SVM runs once (seed 42)."
    )
    p.add_argument(
        "--ann_seeds",
        type=int,
        nargs="+",
        default=[42],
        help="ANN model seeds: 42, or 42 43 44"
    )
    p.add_argument(
        "--ftt_seeds",
        type=int,
        nargs="+",
        default=[42],
        help="FTT model seeds: 42, or 42 43 44"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return p.parse_args()



# ======================================================================================
# MAIN EXECUTION
# ======================================================================================

def main(unmanifested_test_run: bool = False):
    """
    Main execution function.

    ``--dataset ASHRAE_2022`` is refused before any data is loaded: E1 runs
    through ``run_revision.py e1``, which validates the environment and the
    checkout and writes the run manifest. ``unmanifested_test_run``
    (keyword, tests only, not exposed on the command line) lifts this guard
    for the synthetic end-to-end tests of this CLI.
    """

    # Parse arguments
    args = parse_args()
    setup_logging(args.verbose)
    check_not_protected(args.output_dir)  # never write into the frozen canonical artefact
    if args.dataset == "ASHRAE_2022" and not unmanifested_test_run:
        raise RevisionNotImplementedError(
            "models.py --dataset ASHRAE_2022 writes no run manifest: experiment E1 runs through "
            "`python3 run_revision.py e1 --run_dir RUN`")
    # Frozen folds and seeds: checked before loading any data.
    revision_folds.check_cv_config(CONFIG)
    if args.cv_folds != revision_folds.FOLDS_K:
        raise ValueError(f"--cv_folds {args.cv_folds}: the frozen protocol fixes K = {revision_folds.FOLDS_K}")
    seeds_by_model = revision_folds.model_seeds(args.seeds, args.ann_seeds, args.ftt_seeds)
    
    # Update global base_out if specified
    global base_out
    base_out = Path(args.output_dir)
    
    print("\n" + "="*120)
    print("THERMAL COMFORT PREDICTION - EXPERIMENTS")
    print("="*120)
    print(f"\n⚙️  Configuration:")
    print(f"   Targets: {args.targets}")
    print(f"   Models: {args.models}")
    print(f"   CV folds: {args.cv_folds} (materialised once per target, shared by every family)")
    print(f"   Model seeds: {seeds_by_model}")
    print(f"   Output: {args.output_dir}")
    print()
    
    # ==================================================================================
    # DATA LOADING AND VALIDATION
    # ==================================================================================
    
    print("📂 Loading and validating data...")

    Path("data_validation_reports").mkdir(parents=True, exist_ok=True)

    # Rebind global FEATURES_ALL so downstream code (split_*, ensure_dir, etc.) uses the right set
    global FEATURES_ALL

    # Resolve dataset tag → csv path + feature list from config.yaml registry
    dataset_registry = CONFIG["data"].get("datasets", {})
    ds_cfg = dataset_registry.get(args.dataset)
    if ds_cfg is None:
        raise ValueError(f"dataset {args.dataset!r} is not in the config.yaml registry")
    else:
        csv_path = ds_cfg["csv"]
        features_key = ds_cfg.get("features", "features")
        features_for_run = CONFIG["data"][features_key]
        # Category normalisation: on by default; off only for the Phase 1 ASHRAE-2018 v2 path (not rerun).
        normalise_cats = bool(ds_cfg.get("category_normalisation", True))
        split_dir = Path(args.split_dir or ds_cfg["split_dir"])
        folds_dir = Path(args.folds_from or ds_cfg["folds_dir"])
        check_not_protected(folds_dir)  # the fold files are written there

    FEATURES_ALL = features_for_run

    print(f"   Dataset: {args.dataset}  ({csv_path})")
    print(f"   Features ({len(features_for_run)}): {features_for_run}")
    print(f"   Archived split: {split_dir}   Folds: {folds_dir}")

    loader = UnifiedDataLoader(FEATURES_ALL, TARGETS_ALL)
    DATA = loader.load_and_validate(csv_path, args.dataset, category_normalisation=normalise_cats)
    if not DATA.index.equals(pd.RangeIndex(len(DATA))):
        raise ValueError("split indices are row positions of the cohort CSV: the loader must keep a RangeIndex")
    loader.save_statistics(Path(f"data_validation_reports/{args.dataset.lower()}_report.json"))

    # ==================================================================================
    # EXPERIMENT SETUP
    # ==================================================================================
    
    # Determine which targets to process
    if "all" in args.targets:
        targets_to_process = TARGETS_ALL
    else:
        targets_to_process = args.targets
    
    print(f"📊 Targets to process: {targets_to_process}\n")
    
    ensure_dir(base_out)
    if normalise_cats:
        write_coverage_log(loader.category_coverage[args.dataset],
                           base_out / f"category_coverage_{args.dataset}.json")
    
    # Storage for best models across all targets: model type → seed → target → model
    best_models_by_target: Dict[str, Dict[int, Dict[str, object]]] = {
        "rf": {},
        "xgb": {},
        "svm": {},
        "ft": {},
    }
    global_comparison_rows: List[Dict] = []

    # ==================================================================================
    # RUN EXPERIMENTS FOR EACH TARGET
    # ==================================================================================
    
    for TARGET in targets_to_process:
        
        # Prepare data
        X_all = DATA[FEATURES_ALL].copy()
        y_all = DATA[TARGET].copy()

        # Archived stratified split, asserted equal to the stratified re-draw
        train_idx, test_idx = revision_folds.load_archived_split(split_dir, TARGET, y_all)
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        # Folds materialised once per target, before any fit
        folds = revision_folds.ensure_folds(folds_dir, args.dataset, TARGET, train_idx, y_train)
        print(f"[FOLDS] {revision_folds.folds_path(folds_dir, TARGET)}  K={folds.k}  "
              f"fold_vector_sha256={folds.sha256}")

        # Save split indices
        out_split_dir = ensure_dir(base_out / TARGET / "splits")
        split_path = out_split_dir / f"{TARGET}_split_indices.json"
        with open(split_path, "w") as f:
            json.dump({
                "target": TARGET,
                "train_idx": X_train.index.astype(int).tolist(),
                "test_idx": X_test.index.astype(int).tolist(),
                "features": FEATURES_ALL
            }, f, indent=2)
        print(f"[SPLIT] Train/test indices saved → {split_path}")
        with open(out_split_dir / f"{TARGET}_folds.json", "w") as f:
            json.dump(folds.to_json(), f)

        # Identify numerical and categorical columns
        # Frozen categorical variables by name when normalised;
        # legacy dtype inference only on the Phase 1 opt-out path.
        if normalise_cats:
            num_cols, cat_cols = split_feature_types(X_train, X_train.columns)
        else:
            num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
            cat_cols = [c for c in X_train.columns if c not in num_cols]

        # Frozen vocabulary + OOV count of the test rows w.r.t. the training rows
        cat_vocabulary = vocabulary_for(cat_cols) if normalise_cats else None
        if normalise_cats:
            oov = oov_summary(oov_report(X_train, X_test, cat_cols))
            with open(out_split_dir / f"{TARGET}_oov_report.json", "w") as f:
                json.dump(oov, f, indent=2)
            print(f"[OOV] test votes with ≥ 1 OOV category: {oov['n_oov_votes']} "
                  f"{oov['counts_per_variable']}")

        # Create experiment directory
        exp_dir = ensure_dir(base_out / TARGET)

        # Run experiments
        best_models, fold_summaries, stats, test_results = run_experiment_for_target(
            TARGET=TARGET,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_cols=num_cols,
            cat_cols=cat_cols,
            exp_dir=exp_dir,
            args=args,
            cat_vocabulary=cat_vocabulary,
            cv_splits=folds.splits(),
            seeds_by_model=seeds_by_model,
        )

        # Store best models
        for model_type, by_seed in best_models.items():
            for seed, model in by_seed.items():
                best_models_by_target[model_type].setdefault(seed, {})[TARGET] = model

        # Aggregate global results
        for r in test_results:
            global_comparison_rows.append({
                "experiment": TARGET,
                "model": r["model"],
                "seed": r["seed"],
                "test_accuracy": r["test_accuracy"],
                "test_f1_macro": r["test_f1_macro"],
                "test_precision_macro": r["test_precision_macro"],
                "test_recall_macro": r["test_recall_macro"],
            })

    # ==================================================================================
    # SAVE BEST MODELS
    # ==================================================================================
    
    print("\n" + "="*120)
    print("SAVING BEST MODELS")
    print("="*120)
    
    # One file per seed; the unsuffixed file holds the seed-42 models, the zero-shot
    # and fine-tuning source of Phase 3.
    for model_type, by_seed in best_models_by_target.items():
        for seed, models_dict in sorted(by_seed.items()):
            save_path = base_out / f"best_{model_type}_by_target_seed{seed}.joblib"
            joblib.dump(models_dict, str(save_path))
            print(f"✅ Saved {model_type.upper()} models (seed {seed}): {save_path}")
            if seed == RANDOM_SEED:
                joblib.dump(models_dict, str(base_out / f"best_{model_type}_by_target.joblib"))


    # ==================================================================================
    # SAVE GLOBAL COMPARISON
    # ==================================================================================
    
    comparison_df = pd.DataFrame(global_comparison_rows)
    comparison_path = base_out / "comprehensive_experiment_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n✅ Global comparison saved: {comparison_path}")
    
    # Print summary statistics
    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY")
    print("="*120)
    
    summary = comparison_df.groupby('model').agg({
        'test_f1_macro': ['mean', 'std'],
        'test_accuracy': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    print("\n" + "="*120)
    print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*120 + "\n")


if __name__ == "__main__":
    main()
