"""Verify that a rerun reproduces the numbers published in the paper.

This is a *manuscript-agnostic reproducibility check*, meant to be shipped with
the public repository. It does not embed manuscript-version-specific prose or
hardcoded "expected" metrics in the source. Instead:

  * `analysis/paper_values.csv` is a committed, frozen snapshot of the values
    reported in the paper's tables (one row per claim).
  * This script reads each claim, re-reads the *actual* value from the rerun
    artefacts, and compares them within a tolerance.
  * It prints a PASS / FAIL / MISSING table and exits non-zero on any failure,
    so it can be wired into CI or run by a reviewer in one command.

A reviewer reproduces the paper by:

    python models.py --dataset ASHRAE_2018_v2 --models classical \
        --classical_models RandomForest --cv_folds 10      # Phase 1
    python models.py --dataset ASHRAE_2022 --cv_folds 10   # Phase 2
    python transfert.py ...                                # Phase 3
    python indicateurs_classiques.py                       # empirical baselines
    python analysis/nn_disagreement.py --scope full \
        --features_key features_12 \
        --data_csv Data/ASHRAE_2022_Clean_api.csv \
        --output_dir <rerun>/diagnostics/nn_ashrae_2022/   # NN ceiling
    python analysis/verify_paper_tables.py --rerun <rerun>  # <-- this check

Usage
-----
    python analysis/verify_paper_tables.py                     # verify
    python analysis/verify_paper_tables.py --rerun <dir>       # custom rerun
    python analysis/verify_paper_tables.py --tol 0.015         # custom tolerance
    python analysis/verify_paper_tables.py --freeze            # (re)write the CSV

`--freeze` regenerates `paper_values.csv` from the current rerun. Use it once to
create the snapshot, or after an *intentional* change to the published numbers
(then commit the new CSV and review the git diff — it shows exactly which paper
values moved).

Dependencies: pandas, numpy (+ stdlib).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RERUN = "rerun_2026-04-28_no_reweighting"
PAPER_VALUES_CSV = Path(__file__).resolve().parent / "paper_values.csv"

DEFAULT_TOL = 0.01  # absolute tolerance on metrics in [0, 1] (1 percentage point)

TARGETS = ["thermal_sensation", "TSV_3p", "thermal_preference"]

# Structural enumeration used by --freeze. This encodes the *shape* of the
# benchmark (which models / cohorts / regimes / indices exist), never a
# manuscript value — the values are read live from the rerun.
PHASE2_MODELS = ["RandomForest", "XGBoost", "SVM", "ANN", "FTTransformer"]
PHASE2_HYBRIDS = [("FTT+RF", "RF"), ("FTT+XGBoost", "XGB")]
BASELINE_METHODS = ["pmv", "apmv", "epmv", "pts_set", "pts_pet", "apts", "epts"]
COHORTS = ["Moujalled", "Hostein"]
PHASE3_DIRECT_MODELS = ["RF", "XGB", "FTTransformer"]  # filename tag
PHASE3_DIRECT_FOLDER = {"RF": "RandomForest", "XGB": "XGBoost",
                        "FTTransformer": "FTTransformer"}
PHASE3_REGIMES = ["B_finetune_80-20", "B_finetune_20-80"]
PHASE3_VARIANTS = ["RF_adapt", "XGB_adapt", "FT_head", "FT_full"]
PHASE1_BASELINE_CONFIG = "Haghirad_NotOpt_NoCW"  # the config retained as Phase 2 baseline
NN_METRICS = ["primary_disagreement", "case_b_cluster_agreement", "case_c_agreement"]


# ---------------------------------------------------------------------------
#   IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
#   Loaders — each returns the actual metric value from the rerun, or None.
#   `key` encodes the block-specific locator (see freeze plan below).
# ---------------------------------------------------------------------------

def _load_phase1(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    df = _read_csv(rerun / "phase1_haghirad_grid" / "experiment_grid_full.csv")
    if df is None:
        return None
    ros_false = df["ros"].astype(str).str.lower() == "false"
    sub = df[(df["dataset"].astype(str) == "ASHRAE_2018")
             & (df["encoding"].astype(str) == "onehot")
             & ros_false
             & (df["config"].astype(str) == key)
             & (df["target"].astype(str) == target)]
    if sub.empty or metric not in sub.columns:
        return None
    return float(sub.iloc[0][metric])


def _load_phase2(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    base = rerun / "phase2_indomain" / target / key
    for name in (f"{target}__{key}_test_results.json",
                 f"{target}_{key}_test_results.json"):
        d = _read_json(base / name)
        if d is not None:
            return d.get(metric)
    return None


def _load_phase2_hybrid(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    # key = "RF" or "XGB" (the head tag); folder is FTTransformer_as_features/<Head>
    head_folder = {"RF": "RF", "XGB": "XGBoost"}[key]
    d = _read_json(rerun / "phase2_indomain" / target
                   / "FTTransformer_as_features" / head_folder
                   / f"{target}_FTemb_{key}_test_results.json")
    return None if d is None else d.get(metric)


def _load_baseline(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    df = _read_csv(rerun / "phase2_baselines" / "resume_indicateurs_par_target_sur_test.csv")
    if df is None:
        return None
    sub = df[(df["target"].astype(str) == target)
             & (df["method"].astype(str).str.lower() == key.lower())]
    if sub.empty or metric not in sub.columns:
        return None
    return float(sub.iloc[0][metric])


def _load_phase3_direct(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    # key = "<cohort>:<model_tag>", e.g. "Moujalled:RF"
    cohort, tag = key.split(":")
    folder = PHASE3_DIRECT_FOLDER[tag]
    d = _read_json(rerun / "phase3" / cohort / "A_direct" / target / folder
                   / f"{cohort}__{target}__{tag}_test_results.json")
    return None if d is None else d.get(metric)


def _load_phase3_adaptive(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    # key = "<cohort>:<regime>:<variant>", e.g. "Moujalled:B_finetune_80-20:RF_adapt"
    cohort, regime, variant = key.split(":")
    d = _read_json(rerun / "phase3" / cohort / regime / target / variant
                   / f"{cohort}__{target}__{variant}_test_results.json")
    return None if d is None else d.get(metric)


def _load_nn(rerun: Path, key: str, target: str, metric: str) -> Optional[float]:
    df = _read_csv(rerun / "diagnostics" / "nn_ashrae_2022" / "nn_disagreement_summary.csv")
    if df is None:
        return None
    sub = df[df["target"].astype(str) == target]
    if sub.empty or metric not in sub.columns:
        return None
    return float(sub.iloc[0][metric])


LOADERS: dict[str, Callable[[Path, str, str, str], Optional[float]]] = {
    "phase1": _load_phase1,
    "phase2": _load_phase2,
    "phase2_hybrid": _load_phase2_hybrid,
    "baseline": _load_baseline,
    "phase3_direct": _load_phase3_direct,
    "phase3_adaptive": _load_phase3_adaptive,
    "nn_ceiling": _load_nn,
}


# ---------------------------------------------------------------------------
#   Freeze — enumerate every claim and snapshot the current rerun values.
# ---------------------------------------------------------------------------

def _freeze_plan() -> list[tuple[str, str, str, str]]:
    """Yield (block, key, target, metric) for every checked claim."""
    plan: list[tuple[str, str, str, str]] = []
    for target in TARGETS:
        for metric in ("accuracy", "qwk"):
            plan.append(("phase1", PHASE1_BASELINE_CONFIG, target, metric))
        for model in PHASE2_MODELS:
            for metric in ("accuracy", "qwk"):
                plan.append(("phase2", model, target, metric))
        for _disp, tag in PHASE2_HYBRIDS:
            for metric in ("accuracy", "qwk"):
                plan.append(("phase2_hybrid", tag, target, metric))
        for method in BASELINE_METHODS:
            plan.append(("baseline", method, target, "accuracy"))
        for cohort in COHORTS:
            for tag in PHASE3_DIRECT_MODELS:
                plan.append(("phase3_direct", f"{cohort}:{tag}", target, "accuracy"))
            for regime in PHASE3_REGIMES:
                for variant in PHASE3_VARIANTS:
                    plan.append(("phase3_adaptive",
                                 f"{cohort}:{regime}:{variant}", target, "accuracy"))
        for metric in NN_METRICS:
            plan.append(("nn_ceiling", "-", target, metric))
    return plan


def freeze(rerun: Path) -> int:
    rows = []
    skipped = []
    for block, key, target, metric in _freeze_plan():
        actual = LOADERS[block](rerun, key, target, metric)
        if actual is None:
            skipped.append((block, key, target, metric))
            continue
        rows.append({
            "block": block,
            "key": key,
            "target": target,
            "metric": metric,
            "expected": round(float(actual), 3),
            "tol": "",
        })
    with open(PAPER_VALUES_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "key", "target", "metric",
                                           "expected", "tol"])
        w.writeheader()
        w.writerows(rows)
    print(f"[freeze] wrote {len(rows)} rows to {PAPER_VALUES_CSV.relative_to(_REPO_ROOT)}")
    if skipped:
        print(f"[freeze] {len(skipped)} claim(s) had no artefact and were skipped:")
        for block, key, target, metric in skipped:
            print(f"          - {block} {key} {target} {metric}")
    print("[freeze] review the git diff of paper_values.csv before committing.")
    return 0


# ---------------------------------------------------------------------------
#   Verify
# ---------------------------------------------------------------------------

def verify(rerun: Path, tol: float, strict_missing: bool) -> int:
    if not PAPER_VALUES_CSV.exists():
        print(f"[error] {PAPER_VALUES_CSV} not found — run with --freeze first.")
        return 2
    claims = pd.read_csv(PAPER_VALUES_CSV)

    n_pass = n_fail = n_missing = 0
    fail_rows: list[str] = []
    print(f"Verifying {len(claims)} paper claims against {rerun.name}/ "
          f"(tol = {tol:.3f})\n")
    header = f"{'STATUS':7} {'BLOCK':16} {'KEY':28} {'TARGET':19} {'METRIC':24} "\
             f"{'EXPECTED':>9} {'ACTUAL':>9} {'Δ':>8}"
    print(header)
    print("-" * len(header))
    for _, c in claims.iterrows():
        block, key = str(c["block"]), str(c["key"])
        target, metric = str(c["target"]), str(c["metric"])
        expected = float(c["expected"])
        row_tol = float(c["tol"]) if str(c.get("tol", "")).strip() not in ("", "nan") else tol
        loader = LOADERS.get(block)
        actual = loader(rerun, key, target, metric) if loader else None

        if actual is None:
            status, delta_s, actual_s = "MISSING", "—", "—"
            n_missing += 1
        else:
            delta = actual - expected
            delta_s = f"{delta:+.4f}"
            actual_s = f"{actual:.3f}"
            if abs(delta) <= row_tol:
                status = "PASS"
                n_pass += 1
            else:
                status = "FAIL"
                n_fail += 1
        line = (f"{status:7} {block:16} {key:28} {target:19} {metric:24} "
                f"{expected:9.3f} {actual_s:>9} {delta_s:>8}")
        print(line)
        if status in ("FAIL", "MISSING"):
            fail_rows.append(line)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} pass, {n_fail} fail, {n_missing} missing "
          f"({len(claims)} claims, tol {tol:.3f})")
    print("=" * 70)
    if fail_rows:
        print("\nClaims that did not reproduce:")
        for line in fail_rows:
            print("  " + line)

    if n_fail > 0:
        return 1
    if n_missing > 0 and strict_missing:
        return 1
    print("\n✅ All checked paper values reproduce within tolerance.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Verify a rerun reproduces the paper's published tables.")
    p.add_argument("--rerun", default=DEFAULT_RERUN,
                   help=f"Rerun directory under the repo root (default: {DEFAULT_RERUN}).")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help=f"Absolute tolerance on metrics (default: {DEFAULT_TOL}).")
    p.add_argument("--freeze", action="store_true",
                   help="Regenerate paper_values.csv from the rerun instead of verifying.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Do not fail the run when a claim's artefact is absent "
                        "(missing rows are still reported).")
    args = p.parse_args()

    rerun = _REPO_ROOT / args.rerun
    if not rerun.exists():
        print(f"[error] rerun directory not found: {rerun}")
        return 2

    if args.freeze:
        return freeze(rerun)
    return verify(rerun, tol=args.tol, strict_missing=not args.allow_missing)


if __name__ == "__main__":
    sys.exit(main())
