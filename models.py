"""
Full unified training pipeline for thermal comfort prediction.

This module orchestrates the entire experimental workflow, combining:

    • Classical ML models (RandomForest, SVM, XGBoost)
    • Deep learning models (ANN, FTTransformer)
    • Hybrid models (FT embeddings → RF / XGBoost)
    • Cross-validation, grid-search, evaluation, visualization
    • Feature importance computation
    • Train/test split management
    • Logging, reproducibility, UMAP visualization, embeddings
    • Unified data loading / validation

The pipeline:
    1. Loads and validates datasets via UnifiedDataLoader
    2. Splits data stratified (train/test)
    3. Runs selected models (classical, deep, hybrid)
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
    classification_report, confusion_matrix, make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import copy
import joblib
from unified_data_loader import ( UnifiedDataLoader, create_label_mappings_from_models)
from tab_transformer_pytorch import FTTransformer, TabTransformer
import torch.nn.functional as F
from models_deep import (   select_best, 
                            get_torch_models,
                            save_torch_curves
                         )
from models_classic import (    get_models,
                                run_grid_search_on_train
                            )
from hybrid import (    _find_last_linear_for_out_dim,
                        ft_get_embeddings,
                        run_heads_on_ft_and_save,
                        visualize_ft_embeddings

)
from config_loader import load_config




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
    labels_order : list, optional
        Custom label ordering.
    target_names : list, optional
        Human-readable target class names.
    save_dir : pathlib.Path or None

    Returns
    -------
    dict
        Summary metrics, confusion matrix, report, and predictions.
    """
    y_pred = model.predict(X_test)
    if labels_order is None:
        labels_order = sorted(pd.Series(y_test).unique().tolist())
    if target_names is None:
        target_names = [str(l) for l in labels_order]

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    f1mi = float(f1_score(y_test, y_pred, average="micro",   zero_division=0))
    f1w  = float(f1_score(y_test, y_pred, average="weighted",zero_division=0))
    rec  = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_test, y_pred,
        labels=labels_order,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    per_class_f1 = {lbl: report[str(lbl)]["f1-score"] for lbl in labels_order if str(lbl) in report}
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        cm_df.to_csv(save_dir / f"{name}_confusion_matrix.csv")
        plot_confusion_matrix(cm, labels_order, title=f"{name} — Confusion",
                              save_path=save_dir / f"{name}_confusion_matrix.png")
        with open(save_dir / f"{name}_test_report.json","w") as f:
            json.dump(report, f, indent=2)

    

    print(f"\n--- TEST FINAL — {name} ---")
    print(f"Acc: {acc:.4f} | F1_macro: {f1m:.4f}| F1_micro: {f1mi:.4f}| F1_weighted: {f1w:.4f}")
    print("F1 par classe   :", {str(k): f"{v:.3f}" for k, v in per_class_f1.items()})
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_test, y_pred, zero_division=0))


    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1m,
        "f1_micro": f1mi,
        "f1_weighted": f1w,
        "labels": [str(l) for l in labels_order], 
        "confusion": cm.tolist(),                  
        "report": report,
        "y_pred": pd.Series(y_pred).tolist()
    }









# ======================================================================================
# MAIN TRAINING WORKFLOW
# ======================================================================================

def run_experiment_for_target(
    TARGET: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    exp_dir: Path,
    args=None) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]] : 


    """
    Run the full experimental workflow for a single target variable.

    Includes:
        - Classical ML (RF, SVM, XGBoost)
        - Deep models (ANN, FTTransformer)
        - Hybrid heads (FT embeddings → RF/XGB)
        - CV, grid-search, metrics, visualizations, embeddings

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

    Returns
    -------
    tuple
        (best_models, fold_summaries, stats, test_results)
    """
    print(f"\n{'='*120}")
    print(f"🎯 TARGET = {TARGET}")
    print(f"{'='*120}")

    best_models = {
    "rf": None,
    "xgb": None,
    "svm": None,
    "ft": None,
        }


    if args is None:
        class DefaultArgs:
            models = ["all"]
            classical_models = ["all"]
            deep_models = ["all"]
            hybrid_heads = ["all"]
            skip_feature_importance = False
            skip_visualizations = False
            skip_embeddings = False
            cv_folds = 5
            n_jobs = -1
        args = DefaultArgs()


    # Determine which model types to run
    run_classical = ("classical" in args.models) or \
                ("all" in args.models and args.deep_models == ["all"])

    run_deep = ("deep" in args.models) or \
           ("all" in args.models and args.classical_models == ["all"])

    run_hybrid = ("hybrid" in args.models)


    

    # Get model configurations
    MODELS_SK    = get_models(num_cols, cat_cols, seed=RANDOM_SEED, hparams=CONFIG["classical_hparams"] )
    MODELS_TORCH = get_torch_models(num_cols, cat_cols, CONFIG["deep_hparams"], seed=RANDOM_SEED)


    # Filter classical models
    if run_classical and "all" not in args.classical_models:
        MODELS_SK = {k: v for k, v in MODELS_SK.items() if k in args.classical_models}
    
    # Filter deep models
    if run_deep and "all" not in args.deep_models:
        MODELS_TORCH = {k: v for k, v in MODELS_TORCH.items() if k in args.deep_models}



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
        for model_name, (pipe, pgrid) in MODELS_SK.items():
            best_est, best_params, fold_summary_df, stats_df, _ = run_grid_search_on_train(
                TARGET, model_name, pipe, pgrid, X_train, y_train,
                refit_metric="f1_macro", k=args.cv_folds, verbose=2, n_jobs=-1
                        )

            mdir = ensure_dir(exp_dir / model_name)

            test_res = eval_on_test(
                best_est, X_test, y_test,
                name=f"{TARGET}__{model_name}",
                save_dir=mdir
                )
            if model_name in ["RandomForest", "XGBoost"] and (not args.skip_feature_importance):
                plot_feature_importance_pipeline(
                pipe=best_est,
                model_name=model_name,
                target_name=TARGET,
                save_dir=mdir,
                top_n=None,          
                save_csv=True
                )




            if model_name == "RandomForest":
                best_models["rf"] = best_est
            elif model_name == "XGBoost":
                best_models["xgb"] = best_est
            elif model_name == "SVM":
                best_models["svm"] = best_est

            fold_summary_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_summary.csv", index=False)
            stats_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_statistics.csv", index=False)

            pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])\
                .to_csv(mdir / f"{TARGET}_{model_name}_confusion_matrix.csv")

            with open(mdir / f"{TARGET}_{model_name}_test_results.json", "w") as f:
                json.dump(test_res, f, indent=2)

            per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
            per_model_stats.append(stats_df.assign(model=model_name))
            per_model_test.append({
             "experiment": TARGET, "model": model_name,
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

        for model_name, (est, pgrid) in MODELS_TORCH.items():
            best_model, best_params, fold_summary_df, stats_df, per_param_val, model_for_curves = \
                select_best(TARGET, model_name, est, pgrid, X_train, y_train, k=args.cv_folds)


            mdir = ensure_dir(exp_dir / model_name)          
            save_torch_curves(model_for_curves, mdir / f"{TARGET}_{model_name}_training_curves.csv")
            test_res = eval_on_test(best_model, X_test, y_test, name=f"{TARGET}__{model_name}",save_dir=ensure_dir(exp_dir / model_name ))

            if model_name == "FTTransformer":
                best_models["ft"] = best_model

                if (not args.skip_visualizations):
                    viz_dir = ensure_dir(exp_dir / f"{model_name}_viz")
                    visualize_ft_embeddings(best_model, X_train, y_train, viz_dir, tag="train", max_points=8000)
                    visualize_ft_embeddings(best_model, X_test,  y_test,  viz_dir, tag="test",  max_points=8000)
                
                
                if run_hybrid and (not args.skip_embeddings):
                    if "all" in args.hybrid_heads:
                        heads_to_train = ["RF", "XGBoost"]
                    else:
                        heads_to_train = [h for h in args.hybrid_heads if h in ["RF", "XGBoost"]]
                    try:
                        ft_head_dir = ensure_dir(exp_dir / f"{model_name}_as_features")
                        head_rows = run_heads_on_ft_and_save(
                            ft_clf=best_model,
                            X_train=X_train, y_train=y_train,
                            X_test=X_test,   y_test=y_test,
                            out_dir=ft_head_dir,
                            target_name=TARGET,
                            tag_prefix="FTemb",
                            heads=heads_to_train,
                            hparams=CONFIG["hybrid"]
                        )
                        for r in head_rows:
                            per_model_test.append(r)
                    except Exception as e:
                        print(f"[WARN] FT→(RF/XGB) embedding heads skipped: {e}")


            fold_summary_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_summary.csv", index=False)
            stats_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_statistics.csv", index=False)
            with open(mdir / f"{TARGET}_{model_name}_test_results.json", "w") as f:
                json.dump(test_res, f, indent=2)
            with open(mdir / f"{TARGET}_{model_name}_param_grid_cv.json", "w") as f:
                json.dump(per_param_val, f, indent=2)
            cm_df = pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])
            cm_df.to_csv(mdir / f"{TARGET}_{model_name}_confusion_matrix.csv")

            plot_confusion_matrix(test_res["confusion"], test_res["labels"], title=f"{TARGET} - {model_name}",save_path=mdir / f"{TARGET}_{model_name}_confusion_matrix.png")

            per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
            per_model_stats.append(stats_df.assign(model=model_name))
            per_model_test.append({
                "experiment": TARGET, "model": model_name,
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
        pd.concat(per_model_fold_summaries, ignore_index=True).to_csv(
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
        - classical / deep / hybrid specification
        - skipping feature importance, embeddings, visualizations
        - CV folds, output directory, seed, verbosity

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
        choices=["classical", "deep", "hybrid", "all"],
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

    #hybrid 
        # Hybrid heads selection (for FT embeddings -> RF/XGB)
    p.add_argument(
        "--hybrid_heads",
        type=str,
        nargs="+",
        choices=["RF", "XGBoost", "all"],
        default=["all"],
        help="Which hybrid heads to train on FT embeddings (default: all = RF and XGBoost)"
    )
    
    # Features and outputs
    p.add_argument(
        "--skip_feature_importance",
        action="store_true",
        help="Skip feature importance plots for tree-based models"
    )
    p.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip FT embeddings -> RF/XGB hybrid models"
    )
    p.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip UMAP visualizations for embeddings"
    )
    
    # Cross-validation
    p.add_argument(
        "--cv_folds",
        type=int,
        default=10,
        help="Number of CV folds (default: 10)"
    )
    
    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="kfold_results_unified",
        help="Output directory for results (default: kfold_results_unified)"
    )
    
    # Other
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
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

def main():
    """Main execution function."""

    # Parse arguments
    args = parse_args()
    setup_logging(args.verbose)
    
    # Update global seed if specified
    if args.seed != RANDOM_SEED:
        set_all_seeds(args.seed)
    
    # Update global base_out if specified
    global base_out
    base_out = Path(args.output_dir)
    
    print("\n" + "="*120)
    print("THERMAL COMFORT PREDICTION - EXPERIMENTS")
    print("="*120)
    print(f"\n⚙️  Configuration:")
    print(f"   Targets: {args.targets}")
    print(f"   Models: {args.models}")
    print(f"   CV folds: {args.cv_folds}")
    print(f"   Random seed: {args.seed}")
    print(f"   Output: {args.output_dir}")
    print()
    
    # ==================================================================================
    # DATA LOADING AND VALIDATION
    # ==================================================================================
    
    print("📂 Loading and validating data...")
    
    Path("data_validation_reports").mkdir(parents=True, exist_ok=True)
    loader = UnifiedDataLoader(FEATURES_ALL, TARGETS_ALL)
    DATA = loader.load_and_validate("Data/ASHRAE_2022_Clean_api.csv", "ASHRAE")
    loader.save_statistics(Path("data_validation_reports/ashrae_report.json"))

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
    
    # Storage for best models across all targets
    

    # Storage for best models across all targets
    best_models_by_target: Dict[str, Dict[str, object]] = {
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

        # Stratified split
        X_train, X_test, y_train, y_test = split_train_test_stratified(
            X_all, y_all, test_size=0.20, seed=RANDOM_SEED
        )

        # Save split indices
        split_dir = ensure_dir(base_out / TARGET / "splits")
        split_path = split_dir / f"{TARGET}_split_indices.json"
        with open(split_path, "w") as f:
            json.dump({
                "target": TARGET,
                "train_idx": X_train.index.astype(int).tolist(),
                "test_idx": X_test.index.astype(int).tolist(),
                "features": FEATURES_ALL
            }, f, indent=2)
        print(f"[SPLIT] Train/test indices saved → {split_path}")

        # Identify numerical and categorical columns
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        cat_cols = [c for c in X_train.columns if c not in num_cols]

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

        )

        # Store best models
        for model_type, model in best_models.items():
            if model is not None:
                best_models_by_target[model_type][TARGET] = model

        # Aggregate global results
        for r in test_results:
            global_comparison_rows.append({
                "experiment": TARGET,
                "model": r["model"],
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
    
    for model_type, models_dict in best_models_by_target.items():
        if models_dict:
            save_path = base_out / f"best_{model_type}_by_target.joblib"
            joblib.dump(models_dict, str(save_path))
            print(f"✅ Saved {model_type.upper()} models: {save_path}")


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
