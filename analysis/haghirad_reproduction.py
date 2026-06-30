"""Controlled reproduction grid for Phase 1 Random-Forest baselines.

Runs a 216-cell grid isolating the contribution of every lever that
could explain the gap between our canonical Phase 1 rerun and Haghirad
et al. 2024:

    3 hp families      (Haghirad_NotOpt, Haghirad_Opt, HighCap)
  * 3 class_weight     (NoCW=None, Bal='balanced', BalSub='balanced_subsample')
  * 2 ROS conditions   (NoROS, ROS via imblearn.RandomOverSampler)
  * 2 encodings        (onehot, label)
  * 2 datasets         (ASHRAE_2018_v2, ASHRAE_2022)
  * 3 targets          (thermal_sensation / TSV_3p / thermal_preference)
  = 216 runs

All runs use the canonical 12-variable Haghirad feature space and the
frozen 80/20 train/test split (seed=42) committed under `splits/<dataset>/`
so that `HighCap_BalSub_NoROS` reproduces the published Phase 1 accuracies
bit-for-bit from a fresh clone — without depending on any prior rerun. If a
frozen split is absent, the split is regenerated deterministically
(`StratifiedShuffleSplit(test_size=0.2, random_state=42)`).

The "HighCap" hp family (n_estimators=600, max_depth=25, min_samples_split=10,
min_samples_leaf=2) corresponds to the high-capacity family of the manuscript;
internally it was previously called "TPB" — the archive
`rerun_2026-04-23_haghirad_grid/` still uses the TPB nomenclature.

Outputs a single consolidated directory:

    {output_dir}/
      experiment_grid_full.csv        (216 rows, Damien-spec columns)
      diagnostic_tables.md            (pivots + isolated-effect tables)
      README.md                       (short human-readable summary)
      per_run/{dataset}_{encoding}/{config}/{target}/
          test_results.json
          confusion_matrix.pdf / .png / .csv
          classification_report.csv

Usage
-----
    python analysis/haghirad_reproduction.py \\
        --datasets all --encodings all \\
        --output_dir rerun_2026-04-23_haghirad_grid
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config_loader import load_config  # noqa: E402

CONFIG = load_config()

TARGETS = [
    ("thermal_sensation", "TSV-7"),
    ("TSV_3p", "TSV-3"),
    ("thermal_preference", "TPV"),
]

CATEGORICAL_COLS = {"Sexe", "Season", "Climate", "Building_type", "cooling type"}

# Dataset registry: label -> (csv_path, splits_dir)
DATASET_REGISTRY = {
    "ASHRAE_2018": {
        "csv": "Data/ASHRAE_2018_v2.csv",
        "splits_dir": "splits/ASHRAE_2018",
    },
    "ASHRAE_2022": {
        "csv": "Data/ASHRAE_2022_Clean_api.csv",
        "splits_dir": "splits/ASHRAE_2022",
    },
}


# =============================================================================
#  HYPERPARAMETER FAMILIES x CLASS_WEIGHT = 9 configs
# =============================================================================
@dataclass(frozen=True)
class HPConfig:
    name: str            # e.g. 'Haghirad_NotOpt_Bal'
    hp_family: str       # Haghirad_NotOpt | Haghirad_Opt | HighCap
    class_weight: object  # None | 'balanced' | 'balanced_subsample'
    per_target: dict     # target_key -> RF kwargs (without class_weight)


def _hp_haghirad_notopt():
    return dict(n_estimators=225, max_depth=25, random_state=42)


def _hp_haghirad_opt_for(target):
    table = {
        "thermal_sensation": dict(n_estimators=220, max_depth=19, random_state=42),
        "TSV_3p": dict(n_estimators=221, max_depth=18, random_state=42),
        "thermal_preference": dict(n_estimators=225, max_depth=25, random_state=42),
    }
    return table[target]


def _hp_highcap():
    """High-capacity family (n_estimators=600, max_depth=25, min_samples_split=10,
    min_samples_leaf=2). Internally called 'TPB' in the archive
    `rerun_2026-04-23_haghirad_grid/`."""
    return dict(
        n_estimators=600,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=2,
        random_state=42,
    )


def _per_target_same(kw_builder):
    return {t: kw_builder() for t, _ in TARGETS}


def _per_target_haghirad_opt():
    return {t: _hp_haghirad_opt_for(t) for t, _ in TARGETS}


CW_LABELS = {None: "NoCW", "balanced": "Bal", "balanced_subsample": "BalSub"}


def _build_families() -> list[HPConfig]:
    out = []
    hp_specs = [
        ("Haghirad_NotOpt", _per_target_same, _hp_haghirad_notopt),
        ("Haghirad_Opt", _per_target_haghirad_opt, None),
        ("HighCap", _per_target_same, _hp_highcap),
    ]
    for hp_family, pt_builder, kw_builder in hp_specs:
        per_target = pt_builder() if kw_builder is None else pt_builder(kw_builder)
        for cw in [None, "balanced", "balanced_subsample"]:
            name = f"{hp_family}_{CW_LABELS[cw]}"
            out.append(
                HPConfig(
                    name=name,
                    hp_family=hp_family,
                    class_weight=cw,
                    per_target=per_target,
                )
            )
    return out


HP_FAMILIES = _build_families()
ROS_CONDITIONS = [False, True]


# =============================================================================
#  DATA LOADING & ENCODING
# =============================================================================
def load_cohort(csv_path: Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features in {csv_path}: {missing}")
    return df


def load_split(
    splits_dir: Path, target: str, y: "np.ndarray | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for `target`.

    Looks for a frozen split JSON in two layouts — the flat committed fixture
    `splits/<dataset>/<target>_split_indices.json` and the nested rerun layout
    `<target>/splits/<target>_split_indices.json`. If none is found and `y` is
    provided, the split is regenerated deterministically with
    `StratifiedShuffleSplit(test_size=0.2, random_state=42)` (the same scheme as
    Phase 2), so a fresh clone never hard-fails.
    """
    candidates = (
        splits_dir / f"{target}_split_indices.json",
        splits_dir / target / "splits" / f"{target}_split_indices.json",
    )
    for path in candidates:
        if path.exists():
            splits = json.loads(path.read_text())
            return (
                np.asarray(splits["train_idx"], dtype=int),
                np.asarray(splits["test_idx"], dtype=int),
            )
    if y is not None:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
        print(
            f"  [WARN] no frozen split for '{target}' under {splits_dir}; "
            f"regenerated deterministically (StratifiedShuffleSplit, seed=42)."
        )
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)
    raise FileNotFoundError(
        f"No split file for '{target}' under {splits_dir} "
        f"(looked for flat and nested layouts) and no labels supplied for the "
        f"deterministic fallback."
    )


def make_preprocess(
    num_cols: list[str], cat_cols: list[str], encoding: str = "onehot"
) -> ColumnTransformer:
    """StandardScaler(num) + {OneHot|StandardScaler-on-int-codes}(cat)."""
    if encoding == "onehot":
        return ColumnTransformer(
            [
                ("num", StandardScaler(), num_cols),
                ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
    if encoding == "label":
        return ColumnTransformer(
            [("all", StandardScaler(), num_cols + cat_cols)]
        )
    raise ValueError(f"Unknown encoding: {encoding!r}")


def split_feature_groups(features: list[str]) -> tuple[list[str], list[str]]:
    num_cols = [c for c in features if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in features if c in CATEGORICAL_COLS]
    return num_cols, cat_cols


# =============================================================================
#  METRICS
# =============================================================================
def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    if n_total == 0:
        return float("nan"), float("nan")
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = p + z**2 / (2 * n_total)
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    return (centre - spread) / denom, (centre + spread) / denom


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_correct = int((y_true == y_pred).sum())
    n_total = int(len(y_true))
    wl, wh = wilson_ci(n_correct, n_total)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = [int((y_true == c).sum()) for c in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_correct": n_correct,
        "n_total": n_total,
        "wilson_low": wl,
        "wilson_high": wh,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "labels": labels,
        "precision_per_class": {str(k): float(v) for k, v in zip(labels, prec)},
        "recall_per_class": {str(k): float(v) for k, v in zip(labels, rec)},
        "f1_per_class": {str(k): float(v) for k, v in zip(labels, f1)},
        "support_per_class": {str(k): int(v) for k, v in zip(labels, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# =============================================================================
#  RUN ONE CELL
# =============================================================================
def run_one(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hp_cfg: HPConfig,
    use_ros: bool,
    encoding: str,
) -> dict:
    num_cols, cat_cols = split_feature_groups(features)

    X_all = df[features].copy()
    if encoding == "label":
        # Integer-encode each categorical (deterministic codes fitted on the
        # whole column), the scaler will z-score these integers alongside
        # numerical features.
        for col in cat_cols:
            X_all[col] = pd.Categorical(
                X_all[col].astype(str).str.strip()
            ).codes.astype(float)

    X_train_raw = X_all.iloc[train_idx]
    X_test_raw = X_all.iloc[test_idx]
    y_train = df[target].to_numpy()[train_idx]
    y_test = df[target].to_numpy()[test_idx]

    pp = make_preprocess(num_cols, cat_cols, encoding=encoding)
    X_train = pp.fit_transform(X_train_raw)
    X_test = pp.transform(X_test_raw)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    if use_ros:
        X_train_fit, y_train_fit = RandomOverSampler(random_state=42).fit_resample(
            X_train, y_train
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    # Assemble final RF kwargs from hp + class_weight.
    rf_kwargs = dict(hp_cfg.per_target[target])
    if hp_cfg.class_weight is not None:
        rf_kwargs["class_weight"] = hp_cfg.class_weight

    clf = RandomForestClassifier(**rf_kwargs)
    clf.fit(X_train_fit, y_train_fit)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics.update(
        {
            "target": target,
            "config": hp_cfg.name,
            "hp_family": hp_cfg.hp_family,
            "class_weight": str(hp_cfg.class_weight) if hp_cfg.class_weight else "None",
            "n_estimators": rf_kwargs.get("n_estimators"),
            "max_depth": rf_kwargs.get("max_depth"),
            "ros": use_ros,
            "encoding": encoding,
            "n_train_original": int(len(y_train)),
            "n_train_after_ros": int(len(y_train_fit)),
            "n_test": int(len(y_test)),
            "n_features_after_preprocess": int(X_train.shape[1]),
            "rf_kwargs": rf_kwargs,
        }
    )
    return metrics


# =============================================================================
#  PER-RUN ARTEFACTS
# =============================================================================
def save_run_artefacts(run: dict, out_dir: Path, target: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. test_results.json
    (out_dir / "test_results.json").write_text(
        json.dumps(run, indent=2, default=str)
    )

    # 2. confusion matrix (pdf + png + csv)
    cm = np.asarray(run["confusion_matrix"])
    labels = run["labels"]
    pd.DataFrame(cm, index=[f"true={c}" for c in labels],
                 columns=[f"pred={c}" for c in labels]).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            shade = cm[i, j] / cm.max() if cm.max() > 0 else 0
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if shade > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{target} — {run['config']} — acc={run['accuracy']:.3f}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 3. classification_report.csv (per-class precision/recall/f1 + summaries)
    rows = []
    for c in labels:
        rows.append({
            "class": c,
            "support": run["support_per_class"][str(c)],
            "precision": run["precision_per_class"][str(c)],
            "recall": run["recall_per_class"][str(c)],
            "f1_score": run["f1_per_class"][str(c)],
        })
    # Summary rows
    n_total = sum(run["support_per_class"].values())
    macro_prec = np.mean([run["precision_per_class"][str(c)] for c in labels])
    macro_rec = np.mean([run["recall_per_class"][str(c)] for c in labels])
    weighted_prec = sum(
        run["precision_per_class"][str(c)] * run["support_per_class"][str(c)]
        for c in labels
    ) / n_total
    weighted_rec = sum(
        run["recall_per_class"][str(c)] * run["support_per_class"][str(c)]
        for c in labels
    ) / n_total
    rows.append({"class": "macro_avg", "support": n_total,
                 "precision": macro_prec, "recall": macro_rec,
                 "f1_score": run["macro_f1"]})
    rows.append({"class": "weighted_avg", "support": n_total,
                 "precision": weighted_prec, "recall": weighted_rec,
                 "f1_score": run["weighted_f1"]})
    rows.append({"class": "accuracy", "support": n_total,
                 "precision": "", "recall": "",
                 "f1_score": run["accuracy"]})
    rows.append({"class": "balanced_accuracy", "support": n_total,
                 "precision": "", "recall": "",
                 "f1_score": run["balanced_accuracy"]})
    pd.DataFrame(rows).to_csv(out_dir / "classification_report.csv", index=False)


# =============================================================================
#  GRID SUMMARY
# =============================================================================
COLUMNS_OUT = [
    "config", "target", "ros", "encoding", "dataset",
    "n_estimators", "max_depth", "class_weight",
    "accuracy", "balanced_accuracy", "wilson_low", "wilson_high",
    "macro_f1", "weighted_f1", "qwk",
    "n_train_original", "n_train_after_ros", "n_test",
    "n_features_after_preprocess", "encoding_runscript",
]


def build_grid_csv(all_runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in all_runs:
        rows.append({
            "config": r["config"],
            "target": r["target"],
            "ros": r["ros"],
            "encoding": r["encoding"],
            "dataset": r["dataset"],
            "n_estimators": r["n_estimators"],
            "max_depth": r["max_depth"],
            "class_weight": r["class_weight"],
            "accuracy": r["accuracy"],
            "balanced_accuracy": r["balanced_accuracy"],
            "wilson_low": r["wilson_low"],
            "wilson_high": r["wilson_high"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "qwk": r["qwk"],
            "n_train_original": r["n_train_original"],
            "n_train_after_ros": r["n_train_after_ros"],
            "n_test": r["n_test"],
            "n_features_after_preprocess": r["n_features_after_preprocess"],
            "encoding_runscript": r["encoding"],
        })
    return pd.DataFrame(rows, columns=COLUMNS_OUT)


def build_diagnostic_md(grid: pd.DataFrame) -> str:
    """Pivoted summaries + isolated-effect tables (ROS, class_weight, enc, ds)."""
    lines = ["# Diagnostic tables — full 216-cell reproduction grid", ""]

    # A. Consolidated pivot: config × target, 4 panels (ds × enc)
    lines.append("## A. Consolidated accuracy per config (pivoted)")
    lines.append("")
    for ds in ["ASHRAE_2018", "ASHRAE_2022"]:
        for enc in ["onehot", "label"]:
            sub = grid[(grid["dataset"] == ds) & (grid["encoding"] == enc)]
            if sub.empty:
                continue
            lines.append(f"### {ds} × encoding={enc}")
            lines.append("")
            lines.append("| Config | TSV-7 | TSV-3 | TPV |")
            lines.append("|---|---:|---:|---:|")
            # Sort by (hp_family, class_weight, ros) via parsed config name
            for hp in ["Haghirad_NotOpt", "Haghirad_Opt", "HighCap"]:
                for cw in ["NoCW", "Bal", "BalSub"]:
                    for ros in ["NoROS", "ROS"]:
                        cfg = f"{hp}_{cw}_{ros}" if False else f"{hp}_{cw}"
                        # config name excludes ros; we filter by ros flag
                        ros_bool = ros == "ROS"
                        cells = {}
                        for t, _ in TARGETS:
                            r = sub[
                                (sub["config"] == cfg)
                                & (sub["ros"] == ros_bool)
                                & (sub["target"] == t)
                            ]
                            cells[t] = r.iloc[0]["accuracy"] if not r.empty else float("nan")
                        lines.append(
                            f"| {cfg}_{ros} | {cells['thermal_sensation']:.3f} | "
                            f"{cells['TSV_3p']:.3f} | "
                            f"{cells['thermal_preference']:.3f} |"
                        )
            lines.append("")

    # B. Isolated class_weight effect (baseline = NoCW) per hp × ros × ds × enc
    lines.append("## B. Isolated class_weight effect (∆ = CW − NoCW, at fixed hp × ros × dataset × encoding)")
    lines.append("")
    lines.append("| hp_family | ROS | Dataset | Encoding | CW | ∆TSV-7 | ∆TSV-3 | ∆TPV |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for hp in ["Haghirad_NotOpt", "Haghirad_Opt", "HighCap"]:
        for ros_bool in [False, True]:
            for ds in ["ASHRAE_2018", "ASHRAE_2022"]:
                for enc in ["onehot", "label"]:
                    sub = grid[
                        (grid["hp_family"] == hp) if "hp_family" in grid.columns
                        else (grid["config"].str.startswith(hp + "_"))
                    ]
                    sub = grid[
                        grid["config"].str.startswith(hp + "_")
                        & (grid["ros"] == ros_bool)
                        & (grid["dataset"] == ds)
                        & (grid["encoding"] == enc)
                    ]
                    base = sub[sub["config"] == f"{hp}_NoCW"]
                    for cw_lab in ["Bal", "BalSub"]:
                        cfg = f"{hp}_{cw_lab}"
                        sub_cw = sub[sub["config"] == cfg]
                        deltas = []
                        for t, _ in TARGETS:
                            b = base[base["target"] == t]
                            c = sub_cw[sub_cw["target"] == t]
                            if b.empty or c.empty:
                                deltas.append(float("nan"))
                            else:
                                deltas.append(
                                    c.iloc[0]["accuracy"] - b.iloc[0]["accuracy"]
                                )
                        lines.append(
                            f"| {hp} | {'ROS' if ros_bool else 'noROS'} | {ds} | {enc} | {cw_lab} | "
                            f"{deltas[0]:+.3f} | {deltas[1]:+.3f} | {deltas[2]:+.3f} |"
                        )
    lines.append("")

    # C. Isolated ROS effect (∆ ROS − NoROS) per config × dataset × encoding
    lines.append("## C. Isolated ROS effect (∆ = ROS − NoROS)")
    lines.append("")
    lines.append("| Config | Dataset | Encoding | ∆TSV-7 | ∆TSV-3 | ∆TPV |")
    lines.append("|---|---|---|---:|---:|---:|")
    configs_unique = sorted(grid["config"].unique())
    for cfg in configs_unique:
        for ds in ["ASHRAE_2018", "ASHRAE_2022"]:
            for enc in ["onehot", "label"]:
                sub = grid[
                    (grid["config"] == cfg)
                    & (grid["dataset"] == ds)
                    & (grid["encoding"] == enc)
                ]
                deltas = []
                for t, _ in TARGETS:
                    r_ros = sub[(sub["ros"]) & (sub["target"] == t)]
                    r_no = sub[(~sub["ros"]) & (sub["target"] == t)]
                    if r_ros.empty or r_no.empty:
                        deltas.append(float("nan"))
                    else:
                        deltas.append(
                            r_ros.iloc[0]["accuracy"] - r_no.iloc[0]["accuracy"]
                        )
                lines.append(
                    f"| {cfg} | {ds} | {enc} | "
                    f"{deltas[0]:+.3f} | {deltas[1]:+.3f} | {deltas[2]:+.3f} |"
                )
    lines.append("")

    # D. Isolated encoding effect (∆ onehot − label) per config × ros × dataset
    lines.append("## D. Isolated encoding effect (∆ = onehot − label)")
    lines.append("")
    lines.append("| Config | ROS | Dataset | ∆TSV-7 | ∆TSV-3 | ∆TPV |")
    lines.append("|---|---|---|---:|---:|---:|")
    for cfg in configs_unique:
        for ros_bool in [False, True]:
            for ds in ["ASHRAE_2018", "ASHRAE_2022"]:
                sub = grid[
                    (grid["config"] == cfg)
                    & (grid["ros"] == ros_bool)
                    & (grid["dataset"] == ds)
                ]
                deltas = []
                for t, _ in TARGETS:
                    r_oh = sub[(sub["encoding"] == "onehot") & (sub["target"] == t)]
                    r_lb = sub[(sub["encoding"] == "label") & (sub["target"] == t)]
                    if r_oh.empty or r_lb.empty:
                        deltas.append(float("nan"))
                    else:
                        deltas.append(
                            r_oh.iloc[0]["accuracy"] - r_lb.iloc[0]["accuracy"]
                        )
                lines.append(
                    f"| {cfg} | {'ROS' if ros_bool else 'noROS'} | {ds} | "
                    f"{deltas[0]:+.3f} | {deltas[1]:+.3f} | {deltas[2]:+.3f} |"
                )
    lines.append("")

    # E. Isolated dataset effect (∆ 2018 − 2022) per config × ros × encoding
    lines.append("## E. Isolated dataset effect (∆ = ASHRAE_2018 − ASHRAE_2022)")
    lines.append("")
    lines.append("| Config | ROS | Encoding | ∆TSV-7 | ∆TSV-3 | ∆TPV |")
    lines.append("|---|---|---|---:|---:|---:|")
    for cfg in configs_unique:
        for ros_bool in [False, True]:
            for enc in ["onehot", "label"]:
                sub = grid[
                    (grid["config"] == cfg)
                    & (grid["ros"] == ros_bool)
                    & (grid["encoding"] == enc)
                ]
                deltas = []
                for t, _ in TARGETS:
                    r_18 = sub[(sub["dataset"] == "ASHRAE_2018") & (sub["target"] == t)]
                    r_22 = sub[(sub["dataset"] == "ASHRAE_2022") & (sub["target"] == t)]
                    if r_18.empty or r_22.empty:
                        deltas.append(float("nan"))
                    else:
                        deltas.append(
                            r_18.iloc[0]["accuracy"] - r_22.iloc[0]["accuracy"]
                        )
                lines.append(
                    f"| {cfg} | {'ROS' if ros_bool else 'noROS'} | {enc} | "
                    f"{deltas[0]:+.3f} | {deltas[1]:+.3f} | {deltas[2]:+.3f} |"
                )
    lines.append("")

    return "\n".join(lines)


def build_readme(grid: pd.DataFrame) -> str:
    lines = [
        "# Full reproduction grid — 216 runs",
        "",
        "3 hp families × 3 class_weight × 2 ROS × 2 encodings × 2 datasets × 3 targets.",
        "",
        "## Files",
        "",
        "- `experiment_grid_full.csv` — 216 rows with all metrics and metadata.",
        "- `diagnostic_tables.md` — pivoted summaries + isolated-effect tables.",
        "- `per_run/{dataset}_{encoding}/{config}/{target}/` — per-run artefacts:",
        "    - `test_results.json`",
        "    - `confusion_matrix.{pdf,png,csv}`",
        "    - `classification_report.csv` (precision/recall/f1 per class + macro/weighted + accuracy + balanced_accuracy)",
        "",
        "## Canonical sanity check",
        "",
        "`HighCap_BalSub_NoROS` × `thermal_sensation` × `onehot` × `ASHRAE_2018` must match",
        "the published Phase 1 accuracies (frozen split `splits/ASHRAE_2018/`):",
        "0.521 / 0.636 / 0.691 on TSV-7 / TSV-3 / TPV.",
        "",
        "## Quick extract",
        "",
    ]
    for ds in ["ASHRAE_2018", "ASHRAE_2022"]:
        for enc in ["onehot", "label"]:
            sub = grid[(grid["dataset"] == ds) & (grid["encoding"] == enc)]
            if sub.empty:
                continue
            lines.append(f"### {ds} × {enc}")
            lines.append("")
            lines.append("| Config | TSV-7 | TSV-3 | TPV |")
            lines.append("|---|---:|---:|---:|")
            for hp in ["Haghirad_NotOpt", "Haghirad_Opt", "HighCap"]:
                for cw in ["NoCW", "Bal", "BalSub"]:
                    cfg_name = f"{hp}_{cw}"
                    for ros_bool in [False, True]:
                        ros_s = "ROS" if ros_bool else "NoROS"
                        row_label = f"{cfg_name}_{ros_s}"
                        cells = {}
                        for t, _ in TARGETS:
                            r = sub[
                                (sub["config"] == cfg_name)
                                & (sub["ros"] == ros_bool)
                                & (sub["target"] == t)
                            ]
                            cells[t] = r.iloc[0]["accuracy"] if not r.empty else float("nan")
                        lines.append(
                            f"| {row_label} | {cells['thermal_sensation']:.3f} | "
                            f"{cells['TSV_3p']:.3f} | "
                            f"{cells['thermal_preference']:.3f} |"
                        )
            lines.append("")
    return "\n".join(lines)


# =============================================================================
#  MAIN
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="ASHRAE_2018 | ASHRAE_2022 | all",
    )
    p.add_argument(
        "--encodings",
        type=str,
        nargs="+",
        default=["all"],
        help="onehot | label | all",
    )
    p.add_argument(
        "--features_key",
        type=str,
        default="features_12",
        help="Config key in config.yaml (default features_12, the 12 Haghirad vars).",
    )
    args = p.parse_args()

    # Expand 'all'
    datasets = (
        list(DATASET_REGISTRY.keys())
        if args.datasets == ["all"]
        else list(args.datasets)
    )
    encodings = (
        ["onehot", "label"] if args.encodings == ["all"] else list(args.encodings)
    )

    features = CONFIG["data"].get(args.features_key)
    if features is None:
        raise KeyError(f"Unknown features key: {args.features_key}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_root = args.output_dir / "per_run"

    all_runs: list[dict] = []
    total = len(datasets) * len(encodings) * len(HP_FAMILIES) * 2 * len(TARGETS)
    counter = 0

    for ds_label in datasets:
        spec = DATASET_REGISTRY[ds_label]
        csv_path = Path(spec["csv"])
        splits_dir = Path(spec["splits_dir"])
        print(f"\n[{ds_label}] Loading {csv_path}")
        df = load_cohort(csv_path, features)
        print(f"  n_rows={len(df)}, features={len(features)}")

        for enc in encodings:
            print(f"\n  === Encoding: {enc} ===")
            for hp_cfg in HP_FAMILIES:
                for use_ros in ROS_CONDITIONS:
                    ros_label = "ROS" if use_ros else "NoROS"
                    print(
                        f"  [{ds_label} | {enc} | {hp_cfg.name}_{ros_label}]"
                    )
                    for target, target_label in TARGETS:
                        counter += 1
                        train_idx, test_idx = load_split(
                            splits_dir, target, y=df[target].to_numpy()
                        )
                        run = run_one(
                            df=df,
                            features=features,
                            target=target,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            hp_cfg=hp_cfg,
                            use_ros=use_ros,
                            encoding=enc,
                        )
                        run["dataset"] = ds_label
                        run["config_full"] = f"{hp_cfg.name}_{ros_label}"

                        # Save per-run artefacts
                        out_sub = (
                            per_run_root
                            / f"{ds_label}_{enc}"
                            / f"{hp_cfg.name}_{ros_label}"
                            / target
                        )
                        save_run_artefacts(run, out_sub, target)

                        print(
                            f"    [{counter:>3d}/{total}] {target_label:<6s} "
                            f"acc={run['accuracy']:.4f} balAcc={run['balanced_accuracy']:.4f} "
                            f"macroF1={run['macro_f1']:.4f} qwk={run['qwk']:.4f}"
                        )
                        all_runs.append(run)

    # Consolidated CSV
    grid = build_grid_csv(all_runs)
    grid_path = args.output_dir / "experiment_grid_full.csv"
    grid.to_csv(grid_path, index=False)
    print(f"\nWrote {grid_path}  ({len(grid)} rows)")

    # Add hp_family column to grid for diagnostic building (helper only)
    grid_for_diag = grid.copy()
    grid_for_diag["hp_family"] = grid_for_diag["config"].str.rsplit("_", n=1).str[0]
    # Build diagnostic + README
    diag_md = build_diagnostic_md(grid_for_diag)
    (args.output_dir / "diagnostic_tables.md").write_text(diag_md)
    (args.output_dir / "README.md").write_text(build_readme(grid))
    print(f"Wrote diagnostic_tables.md and README.md")


if __name__ == "__main__":
    main()
