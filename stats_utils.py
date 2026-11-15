"""Native statistical helpers used across the tpb pipeline.

Kept in a dedicated module so that `models.py`, `transfert.py`,
and `indicateurs_classiques.py` can share the same primitives without the
circular-import hazards that would arise from hosting them inside `models.py`.

Exposes:
    * quadratic_weighted_kappa — ordinal-aware agreement (Cohen kappa with
      quadratic weights) on an explicit, required label list
    * wilson_ci_accuracy        — two-sided Wilson score interval on a binomial
                                   accuracy

The pre-revision ``pairwise_wilcoxon`` (Wilcoxon signed-rank tests pairing CV
folds by fold number across families) is removed: the folds were not shared
and the test is replaced by building-clustered paired bootstrap contrasts on the
test votes (``revision_stats``).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from revision_stats import quadratic_weighted_kappa as _qwk_on_labels


def quadratic_weighted_kappa(y_true, y_pred, labels) -> float:
    """Cohen's kappa with quadratic weights (ordinal-aware agreement).

    Weights distant misclassifications more heavily than adjacent ones; the
    natural complement of accuracy when the target has an intrinsic order
    (TSV-7, TSV-3, TPV).

    ``labels`` is required: pass the full target ontology
    (``revision_stats.ontology(target)``) so that a missing intermediate class
    does not compress the ordinal spacing. Returns NaN when the kappa
    denominator is zero (undefined, never 0 or 1).
    """
    return _qwk_on_labels(y_true, y_pred, labels=labels)


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
