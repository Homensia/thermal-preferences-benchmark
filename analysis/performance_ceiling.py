"""Native performance-ceiling figure generator for tpb artefacts.

Reads the k-fold results written by `models.py` and `hybrid.py` (the per-model
`*_test_results.json` files that now natively contain accuracy, QWK, Wilson CI
and y_pred) and produces a three-panel horizontal-bar figure showing, for each
target (TSV-7, TSV-3, TPV):

  * test accuracy of every model family trained in the run (typically 5
    standard models + 2 hybrid variants when present);
  * a red dashed line at the empirical near-neighbour agreement rate (the
    complement of the disagreement rate computed by the post-hoc K-NN analysis
    on the same feature space — these values are dataset-level constants
    provided via `--nn_agreement_tsv7/tsv3/tpv`);
  * a grey dotted line at the random-prior baseline accuracy (sum p_k^2 over
    the test-set class priors of RF, taken as the reference model).

No post-hoc reconstruction: every bar and CI is read directly from
`test_results.json` files already saved by the pipeline.

Usage
-----
    python analysis/performance_ceiling.py \\
        --results_dir kfold_results_unified \\
        --output figures/performance_ceiling.pdf

The script defaults to writing the PDF + a companion CSV of the raw values
(`figures/performance_ceiling_data.csv`).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


TARGETS = ("thermal_sensation", "TSV_3p", "thermal_preference")
TARGET_LABELS = {
    "thermal_sensation": "TSV-7",
    "TSV_3p": "TSV-3",
    "thermal_preference": "TPV",
}
STANDARD_MODELS = ("RandomForest", "XGBoost", "SVM", "ANN", "FTTransformer")
HYBRID_VARIANTS = ("FTemb_RF", "FTemb_XGB")
MODEL_DISPLAY = {
    "RandomForest": "RF",
    "XGBoost": "XGBoost",
    "SVM": "SVM",
    "ANN": "ANN",
    "FTTransformer": "FT-Transformer",
    "FTemb_RF": "FTT+RF",
    "FTemb_XGB": "FTT+XGB",
}


def _find_test_results(target_dir: Path, model: str) -> Path | None:
    """Locate the canonical test_results.json for a given model under a target."""
    # Standard models live one level deeper in `{ModelName}/`.
    std_path = target_dir / model / f"{target_dir.name}_{model}_test_results.json"
    if std_path.exists():
        return std_path
    # Hybrid variants live under FTTransformer_as_features/{RF,XGBoost}/.
    hybrid_map = {"FTemb_RF": "RF", "FTemb_XGB": "XGBoost"}
    if model in hybrid_map:
        p = (
            target_dir
            / "FTTransformer_as_features"
            / hybrid_map[model]
            / f"{target_dir.name}_FTemb_{hybrid_map[model].replace('Boost', '').replace('RF','RF').replace('XGBoost','XGB') if False else ('RF' if hybrid_map[model]=='RF' else 'XGB')}_test_results.json"
        )
        # Clean-up the over-clever f-string above:
        tag = "RF" if hybrid_map[model] == "RF" else "XGB"
        p = (
            target_dir
            / "FTTransformer_as_features"
            / hybrid_map[model]
            / f"{target_dir.name}_FTemb_{tag}_test_results.json"
        )
        if p.exists():
            return p
    return None


def load_model_test_results(results_dir: Path, target: str, model: str) -> dict | None:
    target_dir = results_dir / target
    if not target_dir.exists():
        return None
    path = _find_test_results(target_dir, model)
    if path is None or not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Harvest every test_results.json available under `results_dir`.

    Returns a long dataframe with columns: target, target_label, model,
    model_display, kind, accuracy, ci_low, ci_high, qwk, f1_macro,
    f1_weighted, n_test.
    """
    rows = []
    all_models = list(STANDARD_MODELS) + list(HYBRID_VARIANTS)
    for target in TARGETS:
        for model in all_models:
            d = load_model_test_results(results_dir, target, model)
            if d is None:
                continue
            kind = "hybrid" if "FTemb" in model else "standard"
            rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "model": model,
                    "model_display": MODEL_DISPLAY.get(model, model),
                    "kind": kind,
                    "accuracy": float(d.get("accuracy", float("nan"))),
                    "ci_low": float(d.get("accuracy_wilson_ci95_low", float("nan"))),
                    "ci_high": float(d.get("accuracy_wilson_ci95_high", float("nan"))),
                    "qwk": float(d.get("qwk", float("nan"))),
                    "f1_macro": float(d.get("f1_macro", float("nan"))),
                    "f1_weighted": float(d.get("f1_weighted", float("nan"))),
                    "n_test": int(d.get("n_test", 0)),
                }
            )
    return pd.DataFrame(rows)


def _random_baseline(results_dir: Path, target: str) -> float:
    """Baseline accuracy under random prediction weighted by class priors:
    sum_k p_k^2 with p_k from the RF confusion matrix row totals (test set).
    Falls back to the first available model if RF is missing."""
    target_dir = results_dir / target
    for model in ("RandomForest",) + STANDARD_MODELS:
        cm_path = target_dir / model / f"{target}_{model}_confusion_matrix.csv"
        if cm_path.exists():
            cm = pd.read_csv(cm_path, index_col=0).to_numpy()
            row = cm.sum(axis=1)
            if row.sum() > 0:
                p = row / row.sum()
                return float(np.sum(p ** 2))
    return float("nan")


def _panel(ax, df_t: pd.DataFrame, agreement_ceiling: float, baseline: float, title: str):
    d = df_t.sort_values("accuracy", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(d))
    colors = ["#4C72B0" if k == "standard" else "#8DA0CB" for k in d["kind"]]
    # Asymmetric error bars from Wilson CI when available.
    lo_err = (d["accuracy"] - d["ci_low"]).clip(lower=0).fillna(0)
    hi_err = (d["ci_high"] - d["accuracy"]).clip(lower=0).fillna(0)
    ax.barh(
        y_pos,
        d["accuracy"],
        xerr=[lo_err, hi_err],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        ecolor="#444",
        capsize=2,
    )
    for yi, v in zip(y_pos, d["accuracy"]):
        ax.text(v + 0.005, yi, f"{v*100:.0f}%", va="center", fontsize=8)
    ax.axvline(
        agreement_ceiling,
        color="#C44E52",
        linestyle="--",
        linewidth=1.6,
    )
    ax.axvline(
        baseline,
        color="#7F7F7F",
        linestyle=":",
        linewidth=1.6,
    )
    # Per-panel numeric annotations tucked at the top of each reference line
    # (keeps the value visible without re-introducing legend overlap).
    ax.text(agreement_ceiling, len(d) - 0.3,
            f"{agreement_ceiling*100:.0f}%",
            color="#C44E52", fontsize=7, ha="center", va="bottom")
    ax.text(baseline, len(d) - 0.3,
            f"{baseline*100:.0f}%",
            color="#7F7F7F", fontsize=7, ha="center", va="bottom")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["model_display"], fontsize=8)
    ax.set_xlabel("Test accuracy", fontsize=9)
    ax.set_xlim(0, max(d["accuracy"].max(), agreement_ceiling) + 0.12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left")


def main():
    p = argparse.ArgumentParser(
        description="Performance-ceiling figure from tpb k-fold artefacts."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="kfold_results_unified",
        help="Directory containing per-target subfolders with test_results.json.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="figures/performance_ceiling.pdf",
        help="Output PDF path (a companion .csv is also written alongside).",
    )
    p.add_argument(
        "--nn_agreement_tsv7",
        type=float,
        default=0.43,
        help="Empirical NN agreement rate for TSV-7 (default 0.43 from PFE Table 4.10).",
    )
    p.add_argument(
        "--nn_agreement_tsv3",
        type=float,
        default=0.56,
        help="Empirical NN agreement rate for TSV-3 (default 0.56 from PFE Table 4.10).",
    )
    p.add_argument(
        "--nn_agreement_tpv",
        type=float,
        default=0.59,
        help="Empirical NN agreement rate for TPV (default 0.59 from PFE Table 4.10).",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    df = load_all_results(results_dir)
    if df.empty:
        raise RuntimeError(
            f"No test_results.json files found under {results_dir}. "
            "Run models.py first."
        )

    agreement_by_target = {
        "thermal_sensation": args.nn_agreement_tsv7,
        "TSV_3p": args.nn_agreement_tsv3,
        "thermal_preference": args.nn_agreement_tpv,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        df_t = df[df["target"] == target]
        if df_t.empty:
            ax.set_visible(False)
            continue
        baseline = _random_baseline(results_dir, target)
        _panel(
            ax,
            df_t,
            agreement_ceiling=agreement_by_target[target],
            baseline=baseline,
            title=TARGET_LABELS[target],
        )

    # Shared legend below the three panels (no overlap with bars).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#4C72B0", edgecolor="black", label="Standard model"),
        Patch(facecolor="#8DA0CB", edgecolor="black", label="Hybrid model"),
        Line2D([0], [0], color="#C44E52", linestyle="--",
               linewidth=1.6, label="NN agreement ceiling"),
        Line2D([0], [0], color="#7F7F7F", linestyle=":",
               linewidth=1.6, label="Random baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    csv_out = out.with_suffix("").as_posix() + "_data.csv"
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
