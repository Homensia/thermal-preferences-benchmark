"""
Classical machine-learning models for thermal comfort prediction.

This module defines preprocessing utilities, model definitions (RandomForest,
SVM, XGBoost), and grid-search procedures used in the classical ML pipeline.

It provides:
    - LabelEncodedClassifier: wrapper that encodes/decode target labels
    - make_preprocess: numerical + categorical preprocessing pipeline
    - get_models: unified access to model definitions and grids
    - print_cv_fold_scores: helper for reporting CV metrics
    - run_grid_search_on_train: full grid-search with metrics export

These models serve as baselines for comparison with deep learning models.
"""
import os, random, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer
)
from config_loader import load_config




CONFIG = load_config()
RANDOM_SEED    = CONFIG["seed"]

# ======================================================================================
# LABEL-ENCODING WRAPPER
# ======================================================================================
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that automatically applies label encoding to ``y`` during ``fit()``,
    and decodes predictions back to original labels during ``predict()``.

    This is necessary for models such as XGBoost, which require integer-encoded
    labels for multi-class classification.

    Parameters
    ----------
    estimator : sklearn estimator, optional
        Base estimator wrapped by this class.
    **kwargs :
        Additional parameters forwarded to ``set_params()``, typically
        for configuring the underlying estimator.

    Attributes
    ----------
    le_ : LabelEncoder
        Fitted label encoder storing original class labels.
    estimator_ : sklearn estimator
        Cloned and fitted estimator.
    classes_ : array-like
        Original class names (before encoding).
    """
    def __init__(self, estimator=None, **kwargs):
        self.estimator = estimator
        if kwargs:
            self.set_params(**kwargs)

    def get_params(self, deep=True):
        """
        Return parameters of the wrapper and the underlying estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, include nested estimator parameters.

        Returns
        -------
        dict
            Mapping of parameter names to values.
        """
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=deep).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        """
        Set parameters of the wrapper and underlying estimator.

        Parameters
        ----------
        **params :
            Parameters of the wrapper or passed to the base estimator
            using the ``estimator__<param>`` prefix.

        Returns
        -------
        self
        """
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        est_params = {k.split("estimator__", 1)[1]: v
                      for k, v in params.items() if k.startswith("estimator__")}
        if est_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_params)
        return self

    def fit(self, X, y):
        """
        Fit the wrapped estimator using label-encoded targets.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        self
        """
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_enc)
        return self

    def predict(self, X):
        """
        Predict using the wrapped estimator and decode predictions.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        array-like
            Decoded class labels.
        """
        y_enc = self.estimator_.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        """
        Predict class probabilities if supported by underlying estimator.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        numpy.ndarray
            Predicted probability matrix.

        Raises
        ------
        AttributeError
            If the underlying estimator does not support ``predict_proba``.
        """
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)
        raise AttributeError("Base estimator has no predict_proba")

# ======================================================================================
# PREPROCESSING PIPELINE
# ======================================================================================
def make_preprocess(num_cols, cat_cols):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Numerical features → StandardScaler  
    Categorical features → OneHotEncoder

    Parameters
    ----------
    num_cols : list of str
        Names of numerical columns.
    cat_cols : list of str
        Names of categorical columns.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessing pipeline.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


# ======================================================================================
# MODEL DEFINITIONS
# ======================================================================================
def get_models(num_cols, cat_cols, seed=RANDOM_SEED,hparams=None):
    hparams = hparams or {} 
    """
    Build classical ML models with associated hyper-parameter grids.

    Models included:
        - RandomForest
        - SVM
        - XGBoost (wrapped with LabelEncodedClassifier)

    Parameters
    ----------
    num_cols : list of str
        Numerical feature columns.
    cat_cols : list of str
        Categorical feature columns.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict
        Dictionary mapping model names → (pipeline, param_grid).
    """
    pp = make_preprocess(num_cols, cat_cols)
    models = {
        "RandomForest": (
            Pipeline([("pp", pp), ("clf", RandomForestClassifier(random_state=seed))]),
            {
                "clf__n_estimators": [hparams["RandomForest"]["n_estimators"]],
                "clf__max_depth": [hparams["RandomForest"]["max_depth"]],
                "clf__min_samples_split": [hparams["RandomForest"]["min_samples_split"]],
                "clf__min_samples_leaf": [hparams["RandomForest"]["min_samples_leaf"]],
                "clf__class_weight": [hparams["RandomForest"]["class_weight"]]
            }
        ),
        "SVM": (
            Pipeline([("pp", pp), ("clf", SVC(probability=False, random_state=seed))]),
            {
                "clf__kernel": hparams["SVM"]["kernel"],
                "clf__C": hparams["SVM"]["C"],
                "clf__gamma": hparams["SVM"]["gamma"]
            }
        ),
        "XGBoost": (
    Pipeline([("pp", pp), 
              ("clf", LabelEncodedClassifier(
                  XGBClassifier(random_state=seed, eval_metric="mlogloss", tree_method="hist",
                                objective="multi:softprob")
              ))]),
            {
        "clf__estimator__n_estimators": [hparams["XGBoost"]["n_estimators"]],
        "clf__estimator__max_depth": [hparams["XGBoost"]["max_depth"]],
        "clf__estimator__learning_rate": [hparams["XGBoost"]["learning_rate"]],
        "clf__estimator__subsample": [hparams["XGBoost"]["subsample"]],
        "clf__estimator__colsample_bytree": [hparams["XGBoost"]["colsample_bytree"]],
            }
        ),
    }
    return models
# ======================================================================================
# REPORTING UTILITIES
# ======================================================================================
def print_cv_fold_scores(grid: GridSearchCV, metric="f1_macro"):
    """
    Print train/test cross-validation scores for the best parameter set.

    Parameters
    ----------
    grid : GridSearchCV
        Fitted grid-search object.
    metric : str, default="f1_macro"
        Metric used to extract per-fold scores.
    """
    best = grid.best_index_
    trains = sorted(k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith(f"_train_{metric}"))
    vals   = sorted(k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith(f"_test_{metric}"))
    print("\n[Scores par fold] (train | val):")
    for i, (kt, kv) in enumerate(zip(trains, vals), 1):
        print(f"  fold {i}: {grid.cv_results_[kt][best]:.4f} | {grid.cv_results_[kv][best]:.4f}")

# ======================================================================================
# GRID SEARCH EXECUTION
# ======================================================================================
def run_grid_search_on_train(target_name, model_name, pipe, param_grid, X_train, y_train,
                             refit_metric="f1_macro", k=5, n_jobs=-1, verbose=1):
    """
    Run a full GridSearchCV on the training dataset with multiple scoring metrics.

    Parameters
    ----------
    target_name : str
        Name of the prediction target (e.g. ``"thermal_sensation"``).
    model_name : str
        Model identifier (e.g. ``"RandomForest"``).
    pipe : sklearn.Pipeline
        Preprocessing + estimator pipeline.
    param_grid : dict
        Hyper-parameter grid.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series or array-like
        Training labels.
    refit_metric : str, default="f1_macro"
        Metric used to choose the best estimator.
    k : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Parallelism settings for GridSearchCV.
    verbose : int, default=1
        Level of verbosity.

    Returns
    -------
    best_estimator : sklearn estimator
        The best fitted estimator.
    best_params : dict
        Best hyper-parameter configuration.
    fold_summary_df : pandas.DataFrame
        Per-fold performance table.
    stats_df : pandas.DataFrame
        Summary statistics (mean, std, min, max, median).
    train_time : float
        Total training time (seconds).
    """
    from models import SCORERS
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SCORERS,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"\n[{target_name} - {model_name}] Best params:", grid.best_params_)
    print_cv_fold_scores(grid, metric=refit_metric)

    best = grid.best_index_
    rows = []
    for split_id in range(k):
        rows.append({
            "fold": split_id + 1,
            "experiment": target_name,
            "model": model_name,
            "train_f1_macro": grid.cv_results_[f"split{split_id}_train_f1_macro"][best],
            "val_f1_macro":   grid.cv_results_[f"split{split_id}_test_f1_macro"][best],
            "train_accuracy": grid.cv_results_[f"split{split_id}_train_accuracy"][best],
            "val_accuracy":   grid.cv_results_[f"split{split_id}_test_accuracy"][best],
            "train_precision_macro": grid.cv_results_[f"split{split_id}_train_precision_macro"][best],
            "val_precision_macro":   grid.cv_results_[f"split{split_id}_test_precision_macro"][best],
            "train_recall_macro": grid.cv_results_[f"split{split_id}_train_recall_macro"][best],
            "val_recall_macro":   grid.cv_results_[f"split{split_id}_test_recall_macro"][best],
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
    return grid.best_estimator_, grid.best_params_, fold_summary_df, stats_df, train_time