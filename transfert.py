"""
External dataset evaluation script for thermal comfort prediction models.

This script evaluates previously trained models (FT-Transformer, RandomForest, XGBoost)
on external datasets (Moujalled, Hostein), using two complementary strategies:

1. **Zero-shot evaluation**  
   Models trained on ASHRAE are applied *directly* to the external datasets.
   No retraining is performed. Results reveal generalization capability.

2. **Fine-tuning evaluation**  
   Models are adapted to the external datasets using DEV/TEST splits, with four modes:
       • FT_head   : freeze FT backbone, train only final classification layer  
       • FT_full   : fine-tune entire FTTransformer  
       • RF_adapt  : retrain RandomForest on external DEV  
       • XGB_adapt : retrain XGBoost on external DEV  

Eval metrics include accuracy, precision, recall, F1 (macro/micro/weighted)
and confusion matrices for each target and evaluation mode.


"""



from typing import Dict, Tuple, Iterable
from typing import Optional, List
from pathlib import Path
import copy
import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from models import (
    ensure_dir,
    eval_on_test,
    FEATURES_ALL,
    TARGETS_ALL,
    set_all_seeds,
    base_out,
    RANDOM_SEED,
)
from models_classic import make_preprocess, LabelEncodedClassifier
from hybrid import _find_last_linear_for_out_dim
from config_loader import load_config







# ======================================================================================
# CONFIGURATION
# ======================================================================================

CONFIG = load_config()
set_all_seeds(RANDOM_SEED)


# ======================================================================================
# DATA SPLITTING
# ======================================================================================

def stratified_dev_test(df: pd.DataFrame, 
                       target: str, 
                       dev_ratio: float, 
                       seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified DEV/TEST split on an external dataset.

    Stratification ensures that the class distribution is preserved across
    development and testing subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        External dataset.
    target : str
        Name of the target variable.
    dev_ratio : float
        Development set proportion (e.g., 0.2 → 20% DEV / 80% TEST).
    seed : int, default=RANDOM_SEED
        Random seed.

    Returns
    -------
    (df_dev, df_test) : tuple of pandas.DataFrame
        Stratified development and test splits.
    """
    y = df[target].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0-dev_ratio, random_state=seed)
    dev_idx, test_idx = next(sss.split(df, y))
    return df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()


# ======================================================================================
# MODEL CREATION FOR FINE-TUNING
# ======================================================================================

def make_rf_xgb_protos(num_cols, cat_cols, seed: int = RANDOM_SEED):
    """
    Create reusable prototype pipelines for RandomForest and XGBoost.

    These prototypes are used for fine-tuning on the external datasets
    (RF_adapt, XGB_adapt modes).

    Parameters
    ----------
    num_cols : list of str
        Numerical columns of the dataset.
    cat_cols : list of str
        Categorical columns of the dataset.
    seed : int, default=RANDOM_SEED
        Random seed.

    Returns
    -------
    (rf_pipeline, xgb_pipeline) : tuple of sklearn.Pipeline
        Ready-to-train RandomForest and XGBoost pipelines.
    """
    pp = make_preprocess(num_cols, cat_cols)
    
    rf_proto = Pipeline([
        ("pp", pp),
        ("clf", RandomForestClassifier(
            n_estimators=600,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed
        ))
    ])
    
    xgb_proto = Pipeline([
        ("pp", pp),
        ("clf", LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=600,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                tree_method="hist",
                eval_metric="mlogloss",
                objective="multi:softprob",
                random_state=seed
            )
        ))
    ])
    
    return rf_proto, xgb_proto


def clone_ft_for_finetune(ft_model, 
                         lr: float = 5e-5, 
                         epochs: int = 50, 
                         patience: int = 6,
                         freeze_backbone: bool = True):
    """
    Clone and reconfigure a trained FTTransformer for fine-tuning.

    Depending on ``freeze_backbone``:
        • True  → Train only the final classification layer (FT_head)
        • False → Train entire FTTransformer model (FT_full)

    Parameters
    ----------
    ft_model : FTClassifier
        Trained FTTransformer model.
    lr : float, default=5e-5
        Learning rate.
    epochs : int, default=50
        Number of fine-tuning epochs.
    patience : int, default=6
        Early stopping patience.
    freeze_backbone : bool, default=True
        Whether to freeze all layers except the final linear head.

    Returns
    -------
    FTClassifier
        A cloned and fine-tuning-ready FT model.
    """
    m = copy.deepcopy(ft_model)
    m.lr = float(CONFIG["transfer"]["finetune_params"]["FT_head"]["lr"])
    m.epochs = CONFIG["transfer"]["finetune_params"]["FT_head"]["epochs"]
    m.patience = CONFIG["transfer"]["finetune_params"]["FT_head"]["patience"]
    m.record_curves = True
    m.curves_val_split = 0.1
    
    if freeze_backbone:
        # Freeze all parameters
        for p in m.model_.parameters():
            p.requires_grad = False
        
        # Unfreeze only the final layer
        last = _find_last_linear_for_out_dim(m.model_, len(m.le_y_.classes_))
        if last is None:
            raise RuntimeError("Cannot find final Linear layer in FTTransformer.")
        for p in last.parameters():
            p.requires_grad = True
    
    return m


# ======================================================================================
# ZERO-SHOT EVALUATION
# ======================================================================================

def direct_eval_one_base(base_name: str,
                        df: pd.DataFrame,
                        ft_by_target: dict,
                        rf_by_target: Optional[dict] = None,
                        xgb_by_target: Optional[dict] = None,
                        output_dir: str = "external_results") -> None:
    """
    Perform zero-shot evaluation on a given external dataset.

    Zero-shot = evaluating pre-trained models directly on external data
    *without* any adaptation or fine-tuning.

    Parameters
    ----------
    base_name : str
        Dataset name (e.g., "Moujalled", "Hostein").
    df : pandas.DataFrame
        External dataset.
    ft_by_target : dict
        Mapping target → trained FTTransformer model.
    rf_by_target : dict, optional
        Mapping target → trained RandomForest model.
    xgb_by_target : dict, optional
        Mapping target → trained XGB model.

    Notes
    -----
    For FTTransformer models, labels must match the trained model's class set.
    """
    out_root = ensure_dir(Path(output_dir) / base_name / "A_direct")

    for target in [t for t in TARGETS_ALL if t in df.columns]:
        X, y = df[FEATURES_ALL], df[target]

        # Harmonize types → float
        try:
            y = y.astype(float)
        except:
            y = pd.to_numeric(y, errors="coerce")

        # --- FTTransformer evaluation ---
        if target in ft_by_target:
            ft = ft_by_target[target]

            # Filter to classes known by FT model
            allowed = set(map(float, ft.le_y_.classes_))
            mask = y.astype(float).isin(allowed)

            if mask.sum() > 0:
                _ = eval_on_test(
                    ft, X[mask], y[mask],
                    name=f"{base_name}__{target}__FTTransformer",
                    save_dir=ensure_dir(out_root / target / "FTTransformer")
                )
            else:
                print(f"[{base_name}:{target}] No compatible labels with FT (skipped).")

        # --- RandomForest evaluation ---
        if rf_by_target and target in rf_by_target:
            rf = rf_by_target[target]
            _ = eval_on_test(
                rf, X, y,
                name=f"{base_name}__{target}__RF",
                save_dir=ensure_dir(out_root / target / "RandomForest")
            )

        # --- XGBoost evaluation ---
        if xgb_by_target and target in xgb_by_target:
            xgb = xgb_by_target[target]
            _ = eval_on_test(
                xgb, X, y,
                name=f"{base_name}__{target}__XGB",
                save_dir=ensure_dir(out_root / target / "XGBoost")
            )


# ======================================================================================
# FINE-TUNING EVALUATION
# ======================================================================================

def finetune_eval_one_base(base_name: str,
                          df: pd.DataFrame,
                          dev_ratio: float,
                          ft_model_by_target: dict,
                          rf_proto=None,
                          xgb_proto=None,
                          ft_lrs=(1e-4, 5e-5),
                          epochs: int = 40,
                          patience: int = 6,
                          modes=["all"],
                          seed: int = RANDOM_SEED,
                          output_dir: str = "external_results") -> None:
    """
    Perform fine-tuning evaluation on an external dataset.

    Steps
    -----
    1. Filter target labels to match FT class set.
    2. Stratified DEV/TEST split using ``dev_ratio``.
    3. Fine-tune using selected modes:

       - **FT_head** : freeze backbone, train final layer only  
       - **FT_full** : fine-tune entire FTTransformer  
       - **RF_adapt** : retrain RandomForest on DEV  
       - **XGB_adapt** : retrain XGBoost on DEV  

    Parameters
    ----------
    base_name : str
        Name of the external dataset.
    df : pandas.DataFrame
        External dataset.
    dev_ratio : float
        Development set proportion.
    ft_model_by_target : dict
        Mapping target → pre-trained FTTransformer.
    rf_proto : sklearn.Pipeline, optional
        Prototype RF pipeline to be retrained.
    xgb_proto : sklearn.Pipeline, optional
        Prototype XGB pipeline to be retrained.
    ft_lrs : tuple(float, float), default=(1e-4, 5e-5)
        Learning rates for FT_head and FT_full.
    epochs : int, default=40
        Number of fine-tuning epochs.
    patience : int, default=6
        Early stopping patience.
    modes : list of str, default=["all"]
        Which fine-tuning strategies to execute.
    seed : int, default=RANDOM_SEED

    Returns
    -------
    None
        Results are written to output folders.
    """
    out_root = ensure_dir(
        Path(output_dir) / base_name /
        f"B_finetune_{int(dev_ratio*100)}-{100 - int(dev_ratio*100)}"
    )

    for target in [t for t in TARGETS_ALL if t in df.columns]:
        ft_base = ft_model_by_target.get(target, None)
        if ft_base is None:
            print(f"[{base_name}:{target}] No reference FT model — skipped.")
            continue

        # --- Harmonize types: external y as float, FT classes as float ---
        allowed = set(map(float, ft_base.le_y_.classes_))
        df["_y_float"] = pd.to_numeric(df[target], errors="coerce")

        dff = df[df["_y_float"].isin(allowed)].copy()
        
        if dff["_y_float"].nunique() < 2:
            print(f"[{base_name}:{target}] Too few classes after filtering — skipped.")
            df.drop(columns=["_y_float"], inplace=True, errors="ignore")
            continue

        # Replace target with harmonized float version
        dff[target] = dff["_y_float"].astype(float)
        dff.drop(columns=["_y_float"], inplace=True, errors="ignore")

        # Split DEV/TEST
        df_dev, df_test = stratified_dev_test(dff, target, dev_ratio, seed=seed)
        Xd, yd = df_dev[FEATURES_ALL], df_dev[target].astype(float)
        Xt, yt = df_test[FEATURES_ALL], df_test[target].astype(float)

        # -------- FT_head (freeze backbone) --------
        if "all" in modes or "FT_head" in modes:
            print(f"\n🔧 Fine-tuning FT_head for {target}...")
            ft_head = clone_ft_for_finetune(
                ft_base,
                lr=ft_lrs[0],
                epochs=epochs,
                patience=patience,
                freeze_backbone=True
            )
            ft_head.fit(Xd, yd)
            _ = eval_on_test(
                ft_head, Xt, yt,
                name=f"{base_name}__{target}__FT_head",
                save_dir=ensure_dir(out_root / target / "FT_head")
            )

        # -------- FT_full (train all) --------
        if "all" in modes or "FT_full" in modes:
            print(f"\n🔧 Fine-tuning FT_full for {target}...")
            ft_full = clone_ft_for_finetune(
                ft_base,
                lr=ft_lrs[1],
                epochs=epochs,
                patience=patience,
                freeze_backbone=False
            )
            ft_full.fit(Xd, yd)
            _ = eval_on_test(
                ft_full, Xt, yt,
                name=f"{base_name}__{target}__FT_full",
                save_dir=ensure_dir(out_root / target / "FT_full")
            )

        # -------- RF adapt --------
        if "all" in modes or "RF_adapt" in modes:
            if rf_proto is not None:
                print(f"\n🔧 Adapting RandomForest for {target}...")
                rf_adapt = clone(rf_proto)
                rf_adapt.fit(Xd, yd)
                _ = eval_on_test(
                    rf_adapt, Xt, yt,
                    name=f"{base_name}__{target}__RF_adapt",
                    save_dir=ensure_dir(out_root / target / "RF_adapt")
                )

        # -------- XGB adapt --------
        if "all" in modes or "XGB_adapt" in modes:
            if xgb_proto is not None:
                print(f"\n🔧 Adapting XGBoost for {target}...")
                xgb_adapt = clone(xgb_proto)
                xgb_adapt.fit(Xd, yd)
                _ = eval_on_test(
                    xgb_adapt, Xt, yt,
                    name=f"{base_name}__{target}__XGB_adapt",
                    save_dir=ensure_dir(out_root / target / "XGB_adapt")
                )


# ======================================================================================
# COMPLETE EVALUATION SUITE
# ======================================================================================

def run_external_suite(
    Moujalled: Optional[pd.DataFrame],
    Hostein: Optional[pd.DataFrame],
    best_ft_by_target: Dict[str, object],
    best_rf_by_target: Optional[Dict[str, object]] = None,
    best_xgb_by_target: Optional[Dict[str, object]] = None,
    finetune_dev_ratios: Tuple[float, ...] = (0.2, 0.8),
    modes=["all"],
    seed: int = RANDOM_SEED,
    output_dir: str = "external_results",
) -> None:
    """
    Run the full external evaluation pipeline on both datasets.

    Includes:
        - Zero-shot evaluation
        - Optional fine-tuning evaluation for each ``dev_ratio``

    Parameters
    ----------
    Moujalled : pandas.DataFrame or None
        External dataset.
    Hostein : pandas.DataFrame or None
        External dataset.
    best_ft_by_target : dict
        FT models per target.
    best_rf_by_target : dict, optional
        RF models per target.
    best_xgb_by_target : dict, optional
        XGB models per target.
    finetune_dev_ratios : tuple of float, default=(0.2, 0.8)
        Different DEV/TEST splits for fine-tuning.
    modes : list of str
        Fine-tuning modes.
    seed : int

    Returns
    -------
    None
        Results saved under ``external_results/``.
    """
    def _protos(df):
        """Create RF and XGB prototypes."""
        cols = [c for c in FEATURES_ALL if c in df.columns]

        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if c not in num_cols]

        print(f"[protos] num_cols={num_cols}")
        print(f"[protos] cat_cols={cat_cols}")

        rf_p, xgb_p = make_rf_xgb_protos(num_cols=num_cols, cat_cols=cat_cols, seed=seed)
        return rf_p, xgb_p

    # --- Moujalled evaluation ---
    if Moujalled is not None:
        base = "Moujalled"
        print(f"\n{'='*80}")
        print(f"EVALUATING {base}")
        print(f"{'='*80}")

        # Zero-shot
        print(f"\n--- Zero-shot evaluation ---")
        direct_eval_one_base(
            base, Moujalled,
            best_ft_by_target,
            best_rf_by_target,
            best_xgb_by_target,
            output_dir=output_dir,
        )

        # Fine-tuning
        rf_p, xgb_p = _protos(Moujalled)
        for ratio in finetune_dev_ratios:
            print(f"\n--- Fine-tuning with {int(ratio*100)}/{int((1-ratio)*100)} split ---")
            finetune_eval_one_base(
                base, Moujalled,
                dev_ratio=ratio,
                ft_model_by_target=best_ft_by_target,
                rf_proto=rf_p,
                xgb_proto=xgb_p,
                modes=modes,
                seed=seed,
                output_dir=output_dir,
            )

    # --- Hostein evaluation ---
    if Hostein is not None:
        base = "Hostein"
        print(f"\n{'='*80}")
        print(f"EVALUATING {base}")
        print(f"{'='*80}")

        # Zero-shot
        print(f"\n--- Zero-shot evaluation ---")
        direct_eval_one_base(
            base, Hostein,
            best_ft_by_target,
            best_rf_by_target,
            best_xgb_by_target,
            output_dir=output_dir,
        )

        # Fine-tuning
        rf_p, xgb_p = _protos(Hostein)
        for ratio in finetune_dev_ratios:
            print(f"\n--- Fine-tuning with {int(ratio*100)}/{int((1-ratio)*100)} split ---")
            finetune_eval_one_base(
                base, Hostein,
                dev_ratio=ratio,
                ft_model_by_target=best_ft_by_target,
                rf_proto=rf_p,
                xgb_proto=xgb_p,
                modes=modes,
                seed=seed,
                output_dir=output_dir,
            )


# ======================================================================================
# CLI INTERFACE
# ======================================================================================

def setup_logging(verbose: bool) -> None:
    """
    Configure logging verbosity.

    Parameters
    ----------
    verbose : bool
        If True, enable DEBUG logging. Otherwise INFO logging.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_model_dicts(models_dir: Path):
    """
    Load stored model dictionaries from joblib files previously saved
    by ``models.py``.

    Parameters
    ----------
    models_dir : pathlib.Path
        Base directory containing joblib dictionaries.

    Returns
    -------
    (ft_by_target, rf_by_target, xgb_by_target) : tuple of dict
        Loaded model dictionaries for each model family.
    """
    from joblib import load
    
    ft_by_t, rf_by_t, xgb_by_t = {}, {}, {}
    
    ft_path = models_dir / "best_ft_by_target.joblib"
    rf_path = models_dir / "best_rf_by_target.joblib"
    xgb_path = models_dir / "best_xgb_by_target.joblib"
    
    if ft_path.exists():
        ft_by_t = load(ft_path)
        logging.info(f"Loaded FT models from {ft_path.name} ({len(ft_by_t)} targets)")
    
    if rf_path.exists():
        rf_by_t = load(rf_path)
        logging.info(f"Loaded RF models from {rf_path.name} ({len(rf_by_t)} targets)")
    
    if xgb_path.exists():
        xgb_by_t = load(xgb_path)
        logging.info(f"Loaded XGB models from {xgb_path.name} ({len(xgb_by_t)} targets)")
    
    return ft_by_t, rf_by_t, xgb_by_t


def parse_args():
    """
    Parse CLI arguments for external evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with:

        - ``finetune_ratios`` : list of development ratios  
        - ``finetune_modes``  : FT_head / FT_full / RF_adapt / XGB_adapt / all  
        - ``zero_shot_only``  : run only zero-shot  
        - ``seed``            : random seed  
        - ``verbose``         : verbose mode
    """
    p = argparse.ArgumentParser(
        description="Run external evaluations on Moujalled/Hostein datasets."
    )
    p.add_argument(
        "--finetune_ratios",
        type=str,
        default="0.2,0.8",
        help="Comma-separated DEV ratios for fine-tuning (e.g., '0.2,0.8')."
    )
    p.add_argument(
        "--finetune_modes",
        type=str,
        nargs="+",
        choices=["RF_adapt", "XGB_adapt", "FT_head", "FT_full", "all"],
        default=["all"],
        help="Which finetuning strategies to run on external data."
)

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    p.add_argument(
        "--zero_shot_only",
        type=int,
        default=0,
        help="Set to 1 to skip fine-tuning stage."
    )
    p.add_argument(
        "--models_dir",
        type=str,
        default="kfold_results_unified",
        help=(
            "Directory containing the joblib pickles of models trained on "
            "ASHRAE (best_ft_by_target.joblib, best_rf_by_target.joblib, "
            "best_xgb_by_target.joblib). Defaults to the in-domain output "
            "directory of models.py."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="external_results",
        help=(
            "Directory where external evaluation results are written "
            "(default: external_results)."
        ),
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    return p.parse_args()


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================

def main():
    """
    Entry point of the external evaluation script.

    Workflow
    --------
    1. Parse CLI arguments  
    2. Load Moujalled & Hostein datasets  
    3. Load stored model dictionaries from ``models.py``  
    4. Run zero-shot evaluation  
    5. Optionally run fine-tuning evaluation  
    6. Save all results under ``external_results/``  

    Raises
    ------
    FileNotFoundError
        If no trained model dictionaries are found.
    """
    args = parse_args()
    setup_logging(args.verbose)
    set_all_seeds(args.seed)

    # Paths (CLI-overridable: --models_dir, --output_dir).
    models_dir = Path(args.models_dir)
    output_dir = args.output_dir
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    print(f"   Models dir: {models_dir}")
    print(f"   Output dir: {output_dir}")

    # Load external datasets
    print("\n📂 Loading external datasets...")
    df_Moujalled = pd.read_csv("Data/Moujalled_api.csv")
    df_Hostein = pd.read_csv("Data/Hostein_api.csv")
    print(f"   Moujalled: {df_Moujalled.shape}")
    print(f"   Hostein: {df_Hostein.shape}")

    # Load trained models
    print("\n📦 Loading trained models...")
    ft_by_t, rf_by_t, xgb_by_t = load_model_dicts(models_dir)
    if not any([ft_by_t, rf_by_t, xgb_by_t]):
        raise RuntimeError(
            "No model dicts found. "
            "Expecting best_ft_by_target.joblib and optionally RF/XGB."
        )

    # Run evaluations
    if args.zero_shot_only:
        print("\n🎯 Running zero-shot evaluation only...")
        direct_eval_one_base("Moujalled", df_Moujalled, ft_by_t, rf_by_t, xgb_by_t,
                             output_dir=output_dir)
        direct_eval_one_base("Hostein", df_Hostein, ft_by_t, rf_by_t, xgb_by_t,
                             output_dir=output_dir)
    else:
        ratios = [float(x) for x in args.finetune_ratios.split(",") if x.strip()]
        print(f"\n🎯 Running full evaluation suite with ratios: {ratios}")
        run_external_suite(
            Moujalled=df_Moujalled,
            Hostein=df_Hostein,
            best_ft_by_target=ft_by_t,
            best_rf_by_target=rf_by_t,
            best_xgb_by_target=xgb_by_t,
            finetune_dev_ratios=tuple(ratios),
            modes=args.finetune_modes,
            seed=args.seed,
            output_dir=output_dir,
        )

    print("\n" + "="*80)
    print("✅ External evaluation completed successfully!")
    print("="*80)
    logging.info("Done.")


if __name__ == "__main__":
    main()