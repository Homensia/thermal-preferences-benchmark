# -*- coding: utf-8 -*-
"""
Fusion : indicateurs_classiques + métriques_classiques
- Lit la même base : ASHRAE_db2.01.1_clean.csv
- Calcule les indicateurs (PMV, PPD, SET, PET, PTS*, a/e*)
- Évalue les métriques de classification sur la même partie test que models.py
  en important kfold_results_unified/<TARGET>/test_indices.csv
- Sauvegarde :
    1) ASHRAE_db2.01.1_clean_with_indicators.csv  (base + colonnes indicateurs)
    2) resume_indicateurs_par_target_sur_test.csv (tableau récapitulatif)
    3) rapports par cible (txt) dans results_indicateurs/
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

from pythermalcomfort.models import pmv_ppd_iso, set_tmp, pet_steady, pmv_a, pmv_e

from stats_utils import quadratic_weighted_kappa, wilson_ci_accuracy

# Defaults align with the published tpb layout (ASHRAE_2022_Clean_api.csv is
# the cleaned cohort used by models.py / transfert.py). Overridable via CLI.
DEFAULT_BASE_CSV = "Data/ASHRAE_2022_Clean_api.csv"
DEFAULT_RESULTS_DIR = "kfold_results_unified"
DEFAULT_OUTPUT_DIR = "."

TARGETS = ["thermal_sensation", "TSV_3p", "thermal_preference"]
METHODS = ["pmv", "apmv", "epmv", "pts_pet", "pts_set", "apts", "epts"]

TPV_POS = 0.5
TPV_NEG = -0.5

rename_map = {
    "Tair": "ta",
    "RH": "rh",
    "Met": "met",
}

PTS_PET_A, PTS_PET_B, PTS_PET_C = -1.1704, 0.0494, None

# --- Coefficients PTS(SET) ---
PTS_SET_A, PTS_SET_B, PTS_SET_C = None, None, None
PTS_USE_QUADRATIC = False

def _as_float_field(res, key):
    if res is None:
        return np.nan
    if isinstance(res, dict) and key in res:
        return float(res[key])
    if hasattr(res, key):
        return float(getattr(res, key))
    try:
        return float(res)
    except Exception:
        return np.nan

def _nan_mask2(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = ~np.isnan(a) & ~np.isnan(b) & np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def _sanitize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "rh" in d:
        d["rh"] = pd.to_numeric(d["rh"], errors="coerce")
        frac = d["rh"] <= 1
        if frac.any():
            d.loc[frac, "rh"] = d.loc[frac, "rh"] * 100.0
        d["rh"] = d["rh"].clip(0, 100).fillna(50.0)
    else:
        d["rh"] = 50.0

    defaults = dict(vel=0.1, ta=25.0, tr=np.nan, met=1.1, clo=0.5, wme=0.0, tsv=0.0)
    for c, v in defaults.items():
        if c not in d.columns:
            d[c] = v
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["vel"] = d["vel"].clip(0.01, 3.0).fillna(0.1)
    d["met"] = d["met"].clip(0.7, 4.0).fillna(1.1)
    d["clo"] = d["clo"].clip(0.01, 2.0).fillna(0.5)
    d["ta"]  = d["ta"].clip(-20, 50).fillna(25.0)
    d["tr"]  = d["tr"].fillna(d["ta"]).clip(-20, 50)
    d["wme"] = d["wme"].fillna(0).clip(0, 2.0)
    return d

def _set_tmp_compat_scalar(tdb, tr, vel, rh, met, clo, wme=0.0):
    try:
        res = set_tmp(tdb=tdb, tr=tr, vr=vel, rh=rh, met=met, clo=clo, wme=wme)
    except TypeError:
        res = set_tmp(tdb=tdb, tr=tr, v=vel,  rh=rh, met=met, clo=clo, wme=wme)
    return _as_float_field(res, "set")

def _safe_set_row(rw):
    for k, vel_try in enumerate((rw.vel, max(rw.vel*1.05, 0.02))):
        try:
            return _set_tmp_compat_scalar(rw.ta, rw.tr, vel_try, rw.rh, rw.met, rw.clo, getattr(rw, "wme", 0.0))
        except Exception:
            if k == 1:
                return np.nan

def _safe_pet_row(rw):
    for v_try in (rw.vel, max(rw.vel*1.05, 0.02), 0.1):
        try:
            res = pet_steady(tdb=rw.ta, tr=rw.tr, v=v_try, rh=rw.rh, met=rw.met, clo=rw.clo)
            pet = _as_float_field(res, "pet")
            if pet is not None and not (isinstance(pet, float) and (math.isnan(pet) or math.isinf(pet))):
                return pet
        except Exception:
            continue
    return np.nan

def estimate_coeffs_pmv_df_py(df: pd.DataFrame):
    pmv, asv = _nan_mask2(df["pmv"], df.get("tsv", np.nan))
    if pmv.size == 0:
        return 0.0, 1.0
    a_pmv = float(np.mean(asv - pmv))
    denom = float(np.sum(pmv * pmv))
    e_pmv = float(np.sum(asv * pmv) / denom) if denom > 0 else 1.0
    return a_pmv, e_pmv

def estimate_coeffs_pts_df_py(df: pd.DataFrame):
    setv, asv = _nan_mask2(df["set"], df.get("tsv", np.nan))
    if setv.size == 0:
        return 0.0, 1.0
    a_pts = float(np.mean(asv - setv))
    denom = float(np.sum(setv * setv))
    e_pts = float(np.sum(asv * setv) / denom) if denom > 0 else 1.0
    return a_pts, e_pts

def fit_pts_coeffs(x_series, tsv_series, use_quadratic=False):
    x = np.asarray(x_series, dtype=float)
    y = np.asarray(tsv_series, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return 0.0, 0.2, None
    if use_quadratic:
        X = np.column_stack([np.ones_like(x), x, x**2])
    else:
        X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(beta[0]); b = float(beta[1])
    c = float(beta[2]) if use_quadratic and beta.size >= 3 else None
    return a, b, c

def add_apmv_epmv(df: pd.DataFrame, a_coeff: float, e_coeff: float):
    out = df.copy()
    apmv_vals, epmv_vals = [], []
    for rw in out.itertuples(index=False):
        res_a = pmv_a(tdb=rw.ta, tr=rw.tr, vr=rw.vel, rh=rw.rh, met=rw.met, clo=rw.clo,
                      wme=(rw.wme if hasattr(rw, "wme") else 0.0), a_coefficient=a_coeff)
        res_e = pmv_e(tdb=rw.ta, tr=rw.tr, vr=rw.vel, rh=rw.rh, met=rw.met, clo=rw.clo,
                      wme=(rw.wme if hasattr(rw, "wme") else 0.0), e_coefficient=e_coeff)
        apmv_vals.append(_as_float_field(res_a, "a_pmv"))
        epmv_vals.append(_as_float_field(res_e, "e_pmv"))
    out["apmv"] = np.array(apmv_vals, dtype=float)
    out["epmv"] = np.array(epmv_vals, dtype=float)
    return out

def apply_apts_epts_py(df: pd.DataFrame, a_pts: float, e_pts: float):
    out = df.copy()
    setv = out["set"].astype(float)
    out["apts"] = setv + float(a_pts)
    out["epts"] = setv * float(e_pts)
    return out

def compute_block(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    if "wme" not in df.columns:
        df["wme"] = 0.0
    df = _sanitize_inputs(df)

    # PMV / PPD
    pmv_vals, ppd_vals = [], []
    for rw in df.itertuples(index=False):
        res = pmv_ppd_iso(tdb=rw.ta, tr=rw.tr, vr=rw.vel, rh=rw.rh, met=rw.met, clo=rw.clo, wme=rw.wme)
        pmv_vals.append(_as_float_field(res, "pmv"))
        ppd_vals.append(_as_float_field(res, "ppd"))
    df["pmv"] = np.array(pmv_vals, dtype=float)
    df["ppd"] = np.array(ppd_vals, dtype=float)

    # SET
    df["set"] = [_safe_set_row(rw) for rw in df.itertuples(index=False)]

    # PET
    df["pet"] = [_safe_pet_row(rw) for rw in df.itertuples(index=False)]

    # PTS(PET)
    if any(v is None for v in (PTS_PET_A, PTS_PET_B)) and PTS_PET_C is None:
        a_pet, b_pet, c_pet = fit_pts_coeffs(df["pet"], df.get("tsv", np.nan), use_quadratic=False)
    else:
        a_pet, b_pet, c_pet = PTS_PET_A, PTS_PET_B, PTS_PET_C
    pet_np = df["pet"].to_numpy(dtype=float)
    df["pts_pet"] = (a_pet + b_pet * pet_np) if c_pet is None else (a_pet + b_pet * pet_np + c_pet * (pet_np**2))

    # PTS(SET)
    if any(v is None for v in (PTS_SET_A, PTS_SET_B)) and PTS_SET_C is None:
        a_set, b_set, c_set = fit_pts_coeffs(df["set"], df.get("tsv", np.nan), use_quadratic=PTS_USE_QUADRATIC)
    else:
        a_set, b_set, c_set = PTS_SET_A, PTS_SET_B, PTS_SET_C
    set_np = df["set"].to_numpy(dtype=float)
    df["pts_set"] = (a_set + b_set * set_np) if c_set is None else (a_set + b_set * set_np + c_set * (set_np**2))

    # Coeffs globaux a/e
    a_pmv, e_pmv = estimate_coeffs_pmv_df_py(df)
    a_pts, e_pts = estimate_coeffs_pts_df_py(df)

    # aPMV/ePMV
    df = add_apmv_epmv(df, a_pmv, e_pmv)
    # aPTS/ePTS
    df = apply_apts_epts_py(df, a_pts, e_pts)

    print(f"[Coeffs] aPMV={a_pmv:.4f}  ePMV={e_pmv:.4f}  aPTS={a_pts:.4f}  ePTS={e_pts:.4f}")

    return df

def round_clip7(y: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(y).astype(int), -3, 3)

def to_tsv3(y7: np.ndarray) -> np.ndarray:
    return np.where(y7 <= -1, -1, np.where(y7 >= 1, 1, 0)).astype(int)

def to_tpv_from_vote(y_cont: np.ndarray) -> np.ndarray:
    return np.where(y_cont > TPV_POS, -1, np.where(y_cont < TPV_NEG, 1, 0)).astype(int)

def eval_task(y_true, y_pred, labels, title, report_path=None):
    acc  = accuracy_score(y_true, y_pred)
    f1M  = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1mi = f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    qwk  = quadratic_weighted_kappa(y_true, y_pred)
    n_total = int(len(y_true))
    n_correct = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    ci_low, ci_high = wilson_ci_accuracy(n_correct, n_total)
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(f"\n=== {title} ===")
    print(rep)
    if report_path is not None:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n")
            f.write(rep)
    return {
        "accuracy": acc,
        "accuracy_wilson_ci95_low": float(ci_low),
        "accuracy_wilson_ci95_high": float(ci_high),
        "qwk": qwk,
        "f1_macro": f1M,
        "f1_micro": f1mi,
        "f1_weighted": f1w,
        "n_test": n_total,
        "n_correct": n_correct,
    }


def _load_test_indices(results_dir: Path, target: str):
    # Preferred: splits JSON (native tpb format)
    split_json = results_dir / target / "splits" / f"{target}_split_indices.json"
    if split_json.exists():
        d = json.loads(split_json.read_text())
        return np.asarray(d["test_idx"], dtype=int), f"splits JSON ({split_json.name})"
    # Legacy fallback: test_indices.csv
    legacy = results_dir / target / "test_indices.csv"
    if legacy.exists():
        return pd.read_csv(legacy)["row_id"].to_numpy(), "legacy test_indices.csv"
    return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Native empirical-indices pipeline (PMV/aPMV/ePMV/PTS_*)."
    )
    ap.add_argument("--data_csv", default=DEFAULT_BASE_CSV,
                    help="Cleaned ASHRAE CSV (same cohort used by models.py).")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                    help="Root containing <target>/splits/<target>_split_indices.json.")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                    help="Where to write the summary CSV and per-target text reports.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "results_indicateurs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    enriched_csv = output_dir / "ASHRAE_db2.01.1_clean_with_indicators.csv"
    summary_csv = output_dir / "resume_indicateurs_par_target_sur_test.csv"

    print(f"[INPUT]  data_csv    = {args.data_csv}")
    print(f"[INPUT]  results_dir = {results_dir}")
    print(f"[OUTPUT] output_dir  = {output_dir}")

    df_base = pd.read_csv(args.data_csv)

    df_calc = df_base.rename(columns={k: v for k, v in rename_map.items() if k in df_base.columns}).copy()

    df_feats = compute_block(df_calc)

    add_cols = ["pmv","ppd","set","pet","pts_pet","pts_set","apmv","epmv","apts","epts"]
    df_all = pd.concat([df_base.reset_index(drop=True),
                        df_feats[add_cols].reset_index(drop=True)], axis=1)

    df_all.to_csv(enriched_csv, index=False)
    print(f"[OK] Base enrichie écrite: {enriched_csv}")

    summary_rows = []
    for target in TARGETS:
        idx, src = _load_test_indices(results_dir, target)
        if idx is None:
            print(f"[WARN] indices test introuvables pour {target} dans {results_dir} -> évaluation sur TOUT le dataset.")
            df_eval = df_all.copy()
        else:
            print(f"[IDX] {target}: {len(idx)} rows from {src}")
            df_eval = df_all.loc[idx].copy()

        if target == "thermal_sensation":
            labels = [-3, -2, -1, 0, 1, 2, 3]
            y_true = pd.to_numeric(df_eval[target], errors="coerce").astype("Int64").to_numpy(dtype=int)
            for m in METHODS:
                y_cont = pd.to_numeric(df_eval[m], errors="coerce").to_numpy(dtype=float)
                y_pred = round_clip7(y_cont)
                res = eval_task(
                    y_true, y_pred, labels,
                    title=f"TSV7 — {m} — [{target}]",
                    report_path=reports_dir / f"report_{target}_{m}.txt"
                )
                summary_rows.append({"target": target, "method": m, **res})

        elif target == "TSV_3p":
            labels = [-1, 0, 1]
            y_true = pd.to_numeric(df_eval[target], errors="coerce").astype("Int64").to_numpy(dtype=int)
            for m in METHODS:
                y7 = round_clip7(pd.to_numeric(df_eval[m], errors="coerce").to_numpy(dtype=float))
                y_pred = to_tsv3(y7)
                res = eval_task(
                    y_true, y_pred, labels,
                    title=f"TSV3 — {m} — [{target}]",
                    report_path=reports_dir / f"report_{target}_{m}.txt"
                )
                summary_rows.append({"target": target, "method": m, **res})

        elif target == "thermal_preference":
            labels = [-1, 0, 1]
            y_true = pd.to_numeric(df_eval[target], errors="coerce").astype("Int64").to_numpy(dtype=int)
            for m in METHODS:
                y_cont = pd.to_numeric(df_eval[m], errors="coerce").to_numpy(dtype=float)
                y_pred = to_tpv_from_vote(y_cont)
                res = eval_task(
                    y_true, y_pred, labels,
                    title=f"TPV — {m} — [{target}]",
                    report_path=reports_dir / f"report_{target}_{m}.txt"
                )
                summary_rows.append({"target": target, "method": m, **res})

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n[OK] Résumé écrit: {summary_csv}")
    print(f"[OK] Rapports par cible dans: {reports_dir.resolve()}")
