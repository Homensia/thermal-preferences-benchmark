"""Aggregate feature-importance heatmap (17 features x 6 cells).

Reads the 6 *_feature_importance_aggregated.csv files from the Phase 2
rerun (thermal_sensation / TSV_3p / thermal_preference x RF / XGBoost)
and renders a single heatmap grouped by feature category:

    Environmental : Tair, Tout, RH, vel
    Personal      : clo, Met, Âge, Sexe
    Contextual    : Season, Climate, Building, cooling type
    EMA weather   : Trm_ema_28, RHout_ema_28, precip_ema_28,
                    sunshine_ema_h_28, wind_ema_28

Columns are split into two blocks (RF | XGBoost) with a visual
separator. Cell values are printed; the colour scale is normalised
per column so that the relative ranking within each model/target is
legible regardless of the absolute importance magnitude (RF and
XGBoost use different scoring conventions).

Usage
-----
    python analysis/feature_importance_heatmap.py \\
        --results_dir rerun_2026-04-21/kfold_results_unified \\
        --out <manuscript>/figures/feature_importance_heatmap.pdf

"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

TARGETS = [
    ("thermal_sensation", "TSV-7"),
    ("TSV_3p", "TSV-3"),
    ("thermal_preference", "TPV"),
]
MODELS = ["RandomForest", "XGBoost"]
MODEL_LABELS = {"RandomForest": "RF", "XGBoost": "XGBoost"}

# Feature order grouped by semantic category. The display label is the key.
FEATURE_GROUPS = [
    ("Environmental", ["Tair", "Tout", "RH", "vel"]),
    ("Personal", ["clo", "Met", "Âge", "Sexe"]),
    ("Contextual", ["Season", "Climate", "Building", "cooling type"]),
    (
        "EMA weather",
        [
            "Trm_ema_28",
            "RHout_ema_28",
            "precip_ema_28",
            "sunshine_ema_h_28",
            "wind_ema_28",
        ],
    ),
]
FEATURE_ORDER = [f for _, feats in FEATURE_GROUPS for f in feats]

# Pretty labels for axis.
FEATURE_LABELS = {
    "Tair": r"$T_{\mathrm{air}}$",
    "Tout": r"$T_{\mathrm{out}}$",
    "RH": "RH",
    "vel": "vel",
    "clo": "clo",
    "Met": "met",
    "Âge": "age",
    "Sexe": "gender",
    "Season": "season",
    "Climate": "climate",
    "Building": "building",
    "cooling type": "cooling",
    "Trm_ema_28": r"$\overline{T}_{\mathrm{out,28d}}$",
    "RHout_ema_28": r"$\overline{\mathrm{RH}}_{\mathrm{out,28d}}$",
    "precip_ema_28": r"precip$_{28\mathrm{d}}$",
    "sunshine_ema_h_28": r"sun$_{28\mathrm{d}}$",
    "wind_ema_28": r"wind$_{28\mathrm{d}}$",
}


def load_matrix(results_dir: Path) -> pd.DataFrame:
    """Assemble 17 x 6 matrix of aggregated feature importance."""
    cols = {}
    for model in MODELS:
        for target_key, target_label in TARGETS:
            p = (
                results_dir
                / target_key
                / model
                / f"{target_key}_{model}_feature_importance_aggregated.csv"
            )
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
            df = pd.read_csv(p).set_index("group")["importance"]
            cols[(MODEL_LABELS[model], target_label)] = df
    mat = pd.DataFrame(cols)
    # Reorder rows by our grouped feature order; fail hard if any feature is
    # missing so we don't silently publish a partial plot.
    missing = [f for f in FEATURE_ORDER if f not in mat.index]
    if missing:
        raise KeyError(
            f"Feature(s) missing from the aggregated CSVs: {missing}"
        )
    mat = mat.loc[FEATURE_ORDER]
    return mat


def make_heatmap(mat: pd.DataFrame, out_path: Path) -> None:
    """Render the heatmap, with per-column normalisation for readability."""
    # Per-column normalisation so RF and XGBoost are visually comparable,
    # even though their native scoring conventions differ.
    mat_norm = mat.div(mat.max(axis=0), axis=1)

    n_rows, n_cols = mat.shape

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    cmap = LinearSegmentedColormap.from_list(
        "tpb_or", ["#fff7ec", "#fdbb84", "#d7301f"]
    )
    im = ax.imshow(mat_norm.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # X-axis: model x target labels.
    col_labels = [f"{m}\n{t}" for (m, t) in mat.columns]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)

    # Y-axis: pretty feature labels.
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(f, f) for f in mat.index], fontsize=9
    )

    # Vertical separator between RF and XGBoost blocks.
    ax.axvline(x=2.5, color="black", linewidth=1.2)

    # Horizontal separators between feature groups.
    cum = 0
    group_centres = []
    for name, feats in FEATURE_GROUPS:
        if cum > 0:
            ax.axhline(y=cum - 0.5, color="black", linewidth=0.8)
        group_centres.append((name, cum + (len(feats) - 1) / 2))
        cum += len(feats)

    # Group labels on the LEFT, rotated vertically so they sit outside the
    # plotting area without clashing with the colorbar.
    for name, centre in group_centres:
        ax.text(
            -2.6,
            centre,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )

    # Cell annotations: raw importance value in percent.
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.values[i, j]
            shade = mat_norm.values[i, j]
            txt_color = "white" if shade > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{100 * v:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    # Block labels at the top.
    ax.text(
        1.0,
        -1.0,
        "Random Forest",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        4.0,
        -1.0,
        "XGBoost",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylim(n_rows - 0.5, -1.6)

    # Colour bar.
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.14)
    cbar.set_label("Per-column normalised importance", fontsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Expand the left margin so the rotated group labels printed at
    # x = -2.6 are not clipped by bbox_inches="tight".
    plt.subplots_adjust(left=0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    mat = load_matrix(args.results_dir)
    make_heatmap(mat, args.out)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.png')}")

    # Also persist the raw matrix (absolute importances, not normalised) next
    # to the figure, for reproducibility audits.
    csv_path = args.out.with_suffix(".csv")
    mat.to_csv(csv_path)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
