"""One-shot wrapper: train hybrid heads (FTemb->RF, FTemb->XGB) on top of the
already-trained FTTransformer backbones saved by models.py.

Use this when Phase 2 was launched with ``--models all`` against a pre-fix
build where the hybrid branch was silently skipped: it reloads
``best_ft_by_target.joblib``, reconstructs the exact train/test splits from
``{target}/splits/{target}_split_indices.json``, and calls the patched native
``run_heads_on_ft_and_save`` so that every per-target
``FTTransformer_as_features/{RF,XGBoost}/`` directory is populated with the
full QWK + Wilson CI + confusion-matrix + y_pred artefact bundle.

It does NOT retrain the FTTransformer — deterministic, ~1-2 min per target.

Usage
-----
    python run_hybrid_only.py \\
        --results_dir rerun_2026-04-21/kfold_results_unified
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config_loader import load_config
from hybrid import run_heads_on_ft_and_save
from unified_data_loader import UnifiedDataLoader
from models import ensure_dir  # reuse the canonical helper

CONFIG = load_config()
FEATURES_ALL = CONFIG["data"]["features"]
TARGETS_ALL = CONFIG["data"]["targets"]


def main():
    p = argparse.ArgumentParser(
        description="Rerun hybrid heads only, reusing saved FT backbones."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Phase-2 output dir containing best_ft_by_target.joblib and per-target splits/.",
    )
    p.add_argument(
        "--data_csv",
        type=str,
        default="Data/ASHRAE_2022_Clean_api.csv",
        help="ASHRAE cleaned dataset used at train time.",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    ft_joblib = results_dir / "best_ft_by_target.joblib"
    if not ft_joblib.exists():
        raise FileNotFoundError(f"Missing FT backbones: {ft_joblib}")

    print(f"📂 Loading ASHRAE: {args.data_csv}")
    loader = UnifiedDataLoader(FEATURES_ALL, TARGETS_ALL)
    DATA = loader.load_and_validate(args.data_csv, "ASHRAE")

    print(f"📂 Loading FT backbones: {ft_joblib}")
    ft_by_target = joblib.load(ft_joblib)

    for target, ft_clf in ft_by_target.items():
        print(f"\n{'='*80}\n🎯 TARGET = {target}\n{'='*80}")

        split_path = results_dir / target / "splits" / f"{target}_split_indices.json"
        if not split_path.exists():
            print(f"[SKIP] No split file at {split_path}")
            continue
        splits = json.loads(split_path.read_text())
        train_idx = splits["train_idx"]
        test_idx = splits["test_idx"]

        X_all = DATA[FEATURES_ALL].copy()
        y_all = DATA[target].copy()
        X_train = X_all.loc[train_idx]
        y_train = y_all.loc[train_idx]
        X_test = X_all.loc[test_idx]
        y_test = y_all.loc[test_idx]

        ft_head_dir = ensure_dir(results_dir / target / "FTTransformer_as_features")
        print(f"[HYBRID] train={len(X_train)}  test={len(X_test)}  -> {ft_head_dir}")

        head_rows = run_heads_on_ft_and_save(
            ft_clf=ft_clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            out_dir=ft_head_dir,
            target_name=target,
            tag_prefix="FTemb",
            heads=["RF", "XGBoost"],
            hparams=CONFIG["hybrid"],
        )
        for r in head_rows:
            name = r.get("model", "?")
            acc = r.get("test_accuracy", float("nan"))
            qwk = r.get("test_qwk", float("nan"))
            print(f"  {name}: Acc={acc:.4f}  QWK={qwk:.4f}")

    print("\n✅ Hybrid heads completed for all targets.")


if __name__ == "__main__":
    main()
