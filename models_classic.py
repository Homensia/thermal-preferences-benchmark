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
from compute import (load_compute, xgb_compute_params, grid_search_n_jobs, check_grid_search_jobs,
                     check_xgb_booster_device, bounded_cpu)




CONFIG = load_config()
RANDOM_SEED    = CONFIG["seed"]

# ======================================================================================
# RANDOM FOREST: PARALLEL FIT, SEQUENTIAL PREDICTION
# ======================================================================================
class ParallelFitRandomForestClassifier(RandomForestClassifier):
    """
    ``RandomForestClassifier`` that builds its trees on ``n_jobs`` threads but
    averages the tree probabilities sequentially.

    The fitted trees do not depend on ``n_jobs`` (per-tree seeds are drawn
    before the parallel loop). scikit-learn's threaded ``predict_proba``,
    however, adds the tree probabilities in thread-completion order, so with
    ``n_jobs > 1`` the probabilities can differ in the last bit between two
    executions and an exact tie between classes could flip the predicted label
    (CPU tree models need identical predictions on re-execution). Prediction
    therefore runs with ``n_jobs=1``, which gives the probabilities of a forest
    fitted and used with ``n_jobs=1``.
    """

    def predict_proba(self, X):
        n_jobs = self.n_jobs
        self.n_jobs = 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = n_jobs


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

    def fit(self, X, y, sample_weight=None):
        """
        Fit the wrapped estimator using label-encoded targets.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : array-like
            Target labels.
        sample_weight : array-like or None
            Per-row weights, forwarded unchanged to the wrapped estimator
            (e.g. a domain weight for pooled fits; the pre-revision
            wrapper could not receive them). ``None`` keeps the
            unweighted call ``estimator_.fit(X, y_enc)``.

        Returns
        -------
        self
        """
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator)
        if sample_weight is None:
            self.estimator_.fit(X, y_enc)
        else:
            self.estimator_.fit(X, y_enc, sample_weight=sample_weight)
        if isinstance(self.estimator_, XGBClassifier):
            check_xgb_booster_device(self.estimator_)    # GPU really used (no silent fallback)
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
def make_preprocess(num_cols, cat_cols, cat_vocabulary=None):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Numerical features → StandardScaler  
    Categorical features → OneHotEncoder

    With ``cat_vocabulary`` (category normalisation fix) the one-hot columns are
    the frozen global vocabulary and ``handle_unknown="error"``: a value OOV
    for the fit activates its own indicator (constant 0 in the training rows),
    and a value outside the vocabulary raises (it cannot pass the category
    normalisation anyway). ``None`` keeps the pre-revision encoder (categories from
    the training rows, unknown → all-zero); it is used only by the Phase 1
    ASHRAE-2018 v2 path, which is not rerun.

    Parameters
    ----------
    num_cols : list of str
        Names of numerical columns.
    cat_cols : list of str
        Names of categorical columns.
    cat_vocabulary : dict or None
        Column → canonical values (``category_normalisation.GLOBAL_VOCABULARY``).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessing pipeline.
    """
    if cat_vocabulary is None:
        ohe = OneHotEncoder(handle_unknown="ignore")
    else:
        ohe = OneHotEncoder(
            categories=[list(sorted(cat_vocabulary[c])) for c in cat_cols],
            handle_unknown="error",
        )
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("ohe", ohe, cat_cols),
    ])


def check_onehot_revision_schema(pipe, vocabulary, label: str = "") -> None:
    """
    Check that a persisted RF/XGB pipeline was fitted by the revision code
    (category normalisation fix): its ``pp`` one-hot encoder has the frozen
    global vocabulary as categories and ``handle_unknown="error"``. Archived
    pre-revision pipelines (categories from the training rows, unknown → all-zero)
    raise ``ValueError`` with the pre-revision-checkout instruction.
    """
    prefix = f"[{label}] " if label else ""
    try:
        ct = pipe.named_steps["pp"]
        ohe = ct.named_transformers_["ohe"]
        cols = next(c for name, _, c in ct.transformers_ if name == "ohe")
    except (AttributeError, KeyError, StopIteration) as exc:
        raise ValueError(f"{prefix}pipeline has no fitted 'pp' one-hot encoder ({exc!r})") from exc
    expected = [list(sorted(vocabulary[c])) for c in cols]
    if ohe.handle_unknown != "error" or [list(c) for c in ohe.categories_] != expected:
        raise ValueError(
            f"{prefix}one-hot encoder predates the revision encoders (handle_unknown="
            f"{ohe.handle_unknown!r}; categories must be the frozen global vocabulary). "
            "Archived models can only be evaluated with the pre-revision code that fitted them "
            "(archived models); revision runs need E1 models.")


# ======================================================================================
# MODEL DEFINITIONS
# ======================================================================================
CLASSICAL_FAMILIES = ("RandomForest", "SVM", "XGBoost")


def get_models(num_cols, cat_cols, seed=RANDOM_SEED,hparams=None, cat_vocabulary=None, compute=None,
               families=None):
    hparams = hparams or {} 
    compute = load_compute() if compute is None else compute
    families = CLASSICAL_FAMILIES if families is None else tuple(families)
    unknown = sorted(set(families) - set(CLASSICAL_FAMILIES))
    if unknown:
        raise ValueError(f"unknown classical families {unknown}")
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
    compute : dict or None
        ``compute`` section of the config (default: ``config.yaml``). Sets only
        where a fit runs: XGBoost ``device`` (``cpu`` by default; optional
        ``cuda`` raises if CUDA is unavailable), ``tree_method="hist"``
        and host threads; RF ``n_jobs = cpu_jobs`` (CPU, parallel over trees;
        predictions do not depend on ``n_jobs``). Hyperparameters unchanged.
    families : iterable of str or None
        Families to build (default: all of ``CLASSICAL_FAMILIES``). Only these
        are constructed, so the XGB device check (CUDA) runs only when XGBoost
        is requested.

    Returns
    -------
    dict
        Dictionary mapping model names → (pipeline, param_grid).
    """
    pp = make_preprocess(num_cols, cat_cols, cat_vocabulary=cat_vocabulary)
    builders = {
        "RandomForest": lambda: (
            Pipeline([("pp", pp), ("clf", ParallelFitRandomForestClassifier(random_state=seed, n_jobs=compute["cpu_jobs"]))]),
            {
                "clf__n_estimators": [hparams["RandomForest"]["n_estimators"]],
                "clf__max_depth": [hparams["RandomForest"]["max_depth"]],
                "clf__min_samples_split": [hparams["RandomForest"]["min_samples_split"]],
                "clf__min_samples_leaf": [hparams["RandomForest"]["min_samples_leaf"]],
                "clf__class_weight": [hparams["RandomForest"]["class_weight"]]
            }
        ),
        "SVM": lambda: (
            Pipeline([("pp", pp), ("clf", SVC(probability=False, random_state=seed))]),
            {
                "clf__kernel": hparams["SVM"]["kernel"],
                "clf__C": hparams["SVM"]["C"],
                "clf__gamma": hparams["SVM"]["gamma"]
            }
        ),
        "XGBoost": lambda: (
    Pipeline([("pp", pp), 
              ("clf", LabelEncodedClassifier(
                  XGBClassifier(random_state=seed, eval_metric="mlogloss",
                                objective="multi:softprob", **xgb_compute_params(compute))
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
    return {name: builders[name]() for name in CLASSICAL_FAMILIES if name in families}
# ======================================================================================
# REPORTING UTILITIES
# ======================================================================================
def check_complete_search(grid: GridSearchCV, k: int, what: str) -> None:
    """
    The frozen CV procedure completed: every
    candidate has a finite train and validation score for every scorer on
    each of the ``k`` folds; otherwise the run stops (no partial search is
    ever accepted).
    """
    res = grid.cv_results_
    n_cand = len(res["params"])
    keys = [f"split{i}_{part}_{name}" for i in range(k) for part in ("train", "test") for name in grid.scorer_]
    missing = [key for key in keys if key not in res]
    if missing:
        raise RuntimeError(f"[{what}] incomplete grid search: missing {missing[:3]}")
    bad = [key for key in keys if len(res[key]) != n_cand or not np.isfinite(np.asarray(res[key], float)).all()]
    if bad:
        raise RuntimeError(f"[{what}] grid search with non-finite or missing scores in {bad[:3]}: "
                           "a failed or incomplete CV fit invalidates the run")


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
                             refit_metric="f1_macro", cv_splits=None, n_jobs=None, verbose=1,
                             compute=None):
    """
    Run a full GridSearchCV on the training dataset with multiple scoring metrics.

    The folds are never drawn here: ``cv_splits`` are
    the positional ``(train, validation)`` pairs materialised once per target by
    ``revision_folds.ensure_folds`` and shared by every family. Their fold-vector
    hash is written in the ``folds_sha256`` column of the fold summary.

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
    cv_splits : list of (array, array)
        Materialised folds (``FoldAssignment.splits()``), positions in ``X_train``.
        Required; ``None`` raises.
    n_jobs : int or None, default=None
        Workers of GridSearchCV. ``None`` → ``compute.grid_search_n_jobs``
        (RF/XGB sequential, SVM over ``min(cpu_jobs, fits)`` workers; no nested
        parallelism). ``-1`` (all cores) or more than ``cpu_jobs`` is refused (compute rules),
        and so is ``n_jobs != 1`` for RF/XGB (nested with the estimator's own threads).
    compute : dict or None
        ``compute`` section bounding the workers (default: ``config.yaml``); the
        caller passes the one it built the estimators with.
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
    from revision_folds import splits_sha256
    if cv_splits is None:
        raise ValueError("run_grid_search_on_train needs the materialised folds (cv_splits); "
                         "no family draws its own folds")
    cv_splits = [(np.asarray(tr), np.asarray(va)) for tr, va in cv_splits]
    folds_sha256 = splits_sha256(cv_splits, len(X_train))   # also checks the K-fold partition
    k = len(cv_splits)
    compute = load_compute() if compute is None else compute
    if n_jobs is None:
        n_fits = k * int(np.prod([len(v) for v in param_grid.values()] or [1]))
        n_jobs = grid_search_n_jobs(model_name, n_fits, compute)
    clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
    inner = clf.estimator if isinstance(clf, LabelEncodedClassifier) else clf   # XGB inside the wrapper
    check_grid_search_jobs(model_name, n_jobs, getattr(inner, "n_jobs", None), compute,
                           estimator_device=getattr(inner, "device", None) or "cpu")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SCORERS,
        refit=refit_metric,
        cv=cv_splits,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        error_score="raise",          # a failed CV fit stops the run
    )
    t0 = time.time()
    with bounded_cpu(process_workers=n_jobs > 1):
        grid.fit(X_train, y_train)
    train_time = time.time() - t0
    check_complete_search(grid, k, f"{target_name} - {model_name}")

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
    return grid.best_estimator_, grid.best_params_, fold_summary_df, stats_df, train_time