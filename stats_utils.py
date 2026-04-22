"""Native statistical helpers used across the tpb pipeline.

Kept in a dedicated module so that `models.py`, `hybrid.py`, `transfert.py`,
and `indicateurs_classiques.py` can share the same primitives without the
circular-import hazards that would arise from hosting them inside `models.py`.

Exposes:
    * quadratic_weighted_kappa — ordinal-aware agreement (Cohen kappa with
      quadratic weights)
    * wilson_ci_accuracy        — two-sided Wilson score interval on a binomial
                                   accuracy
    * pairwise_wilcoxon         — paired Wilcoxon signed-rank tests across
                                   CV folds with Bonferroni correction
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Cohen's kappa with quadratic weights (ordinal-aware agreement).

    Weights distant misclassifications more heavily than adjacent ones; the
    natural complement of accuracy when the target has an intrinsic order
    (TSV-7, TSV-3, TPV).
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def wilson_ci_accuracy(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Exact for accuracy (fraction of correct predictions) given the test-set
    size; avoids the need for a bootstrap when raw ``y_pred`` is available
    (or can be reconstructed from a confusion matrix).
    """
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = n_correct / n_total
    denom = 1.0 + z ** 2 / n_total
    center = (p + z ** 2 / (2.0 * n_total)) / denom
    margin_sq = p * (1.0 - p) / n_total + z ** 2 / (4.0 * n_total ** 2)
    margin = (z * np.sqrt(margin_sq)) / denom
    return (center - margin, center + margin)


def pairwise_wilcoxon(
    fold_summary_df: pd.DataFrame,
    target: str,
    metric: str = "val_accuracy",
    alpha: float = 0.05,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests across CV folds, Bonferroni-corrected.

    Expects a long-format dataframe with at least ``fold``, ``model`` and the
    specified ``metric`` column. Returns one row per ordered model pair with
    columns ``target, model_a, model_b, delta_mean, delta_median, n_folds,
    p_raw, p_bonferroni, reject_at_{alpha}``. Writes the table to
    ``save_path`` when provided.
    """
    if "model" not in fold_summary_df.columns or metric not in fold_summary_df.columns:
        raise ValueError(
            f"fold_summary_df must contain 'model' and '{metric}' columns"
        )
    pivot = fold_summary_df.pivot(index="fold", columns="model", values=metric)
    models_list = list(pivot.columns)
    pairs = list(combinations(models_list, 2))
    n_pairs = max(1, len(pairs))
    records = []
    for a, b in pairs:
        x = pivot[a].to_numpy()
        y = pivot[b].to_numpy()
        d = x - y
        if np.allclose(d, 0):
            p_raw = 1.0
        else:
            try:
                p_raw = float(
                    wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
                )
            except ValueError:
                p_raw = 1.0
        p_adj = min(1.0, p_raw * n_pairs)
        records.append(
            {
                "target": target,
                "model_a": a,
                "model_b": b,
                "delta_mean": float(np.mean(d)),
                "delta_median": float(np.median(d)),
                "n_folds": int(len(d)),
                "p_raw": p_raw,
                "p_bonferroni": p_adj,
                f"reject_at_{alpha}": bool(p_adj < alpha),
            }
        )
    out = pd.DataFrame(records)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
    return out
