"""Statistics of the revision rerun (frozen before the run).

Every function is pure: it takes per-observation arrays (read from the
predictions files: ``row_idx, y_true, y_pred, building_id,
publication, ...``) and returns numbers, intervals and decisions. Nothing here
trains a model or reads a test label to choose anything.

Contents:

* metrics on the full target ontology: accuracy, balanced
  accuracy and macro-F1 over the classes present in ``y_true``, QWK on the
  full ontology (undefined -> NaN with a flag), confusion matrices and
  per-class recall/support over the full ontology.
* majority baselines: deployable (train/DEV class, ties towards
  the class closest to 0, then the lower label) and descriptive (test rate).
* building-clustered paired bootstrap (B = 10 000, seed
  20261008, frozen draw over sorted cluster IDs, percentile intervals),
  cluster eligibility (>= 10 clusters contributing to the estimand's own
  denominator, no zero-denominator replicate; never dropped).

Undefined bootstrap replicates are never deleted and never averaged away: the
module contains no NaN-skipping reduction; an estimand with one undefined
replicate gets the interval "not available".
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

# --------------------------------------------------------------------------- constants
#: Target ontologies: complete ordered label lists, independent of any data subset.
ONTOLOGY: Dict[str, Tuple[int, ...]] = {
    "thermal_sensation": (-3, -2, -1, 0, 1, 2, 3),
    "TSV_3p": (-1, 0, 1),
    "thermal_preference": (-1, 0, 1),
}
TARGETS: Tuple[str, ...] = ("thermal_sensation", "TSV_3p", "thermal_preference")

BOOT_SEED = 20261008          # frozen bootstrap seed
BOOT_B = 10_000               # bootstrap replicates
MIN_CLUSTERS = 10             # minimum contributing clusters for an interval
MIN_DENOMINATOR_VOTES = 30    # minimum denominator votes, condition (b)
ALPHA = 0.05

ABLATION_FAMILIES: Tuple[str, ...] = ("RF", "XGB")
G3_COHORTS: Tuple[str, ...] = ("Moujalled", "Hostein")

NOT_AVAILABLE = "not available"


# --------------------------------------------------------------------------- labels
def ontology(target: str) -> Tuple[int, ...]:
    """Full ordered label list of ``target``."""
    try:
        return ONTOLOGY[target]
    except KeyError:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(ONTOLOGY)}") from None


def _as_labels(y, labels: Sequence[int], what: str) -> np.ndarray:
    """Validate a label vector against ``labels`` and return it as int64.

    Float labels (``-1.0``) are accepted when integral; NaN, non-integral values
    and values outside ``labels`` raise -- nothing is dropped silently.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {arr.shape}")
    try:
        f = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numeric labels") from None
    if not np.all(np.isfinite(f)):
        raise ValueError(f"{what} contains NaN or infinite values (never converted to a class)")
    if not np.all(f == np.round(f)):
        raise ValueError(f"{what} contains non-integral labels")
    out = f.astype(np.int64)
    bad = sorted(set(np.unique(out).tolist()) - set(labels))
    if bad:
        raise ValueError(f"{what} contains labels outside the ontology {tuple(labels)}: {bad}")
    return out


def check_labels(y, labels: Sequence[int], what: str = "labels") -> np.ndarray:
    """Public form of the ontology label validation, for the evaluation callers."""
    return _as_labels(y, labels, what)


def _confusion(yt: np.ndarray, yp: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.array([pos[v] for v in yt.tolist()], dtype=np.int64),
                   np.array([pos[v] for v in yp.tolist()], dtype=np.int64)), 1)
    return cm


def confusion_matrix_ontology(y_true, y_pred, target: str) -> pd.DataFrame:
    """Confusion matrix over the full ontology (rows = true, columns = predicted)."""
    labels = ontology(target)
    yt, yp = _as_labels(y_true, labels, "y_true"), _as_labels(y_pred, labels, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    return pd.DataFrame(_confusion(yt, yp, labels), index=list(labels), columns=list(labels))


def quadratic_weighted_kappa(y_true, y_pred, *, labels: Sequence[int]) -> float:
    """QWK on an explicit label list; NaN when the kappa denominator is zero.

    ``labels`` is required: without it a missing intermediate class compresses
    the ordinal spacing (example: 0.6923 without, 0.5969 with the
    TSV-7 ontology). Undefined QWK is returned as NaN, never as 0 or 1.
    """
    labels = tuple(labels)
    if len(labels) < 2:
        raise ValueError("QWK needs at least two labels")
    yt, yp = _as_labels(y_true, labels, "y_true"), _as_labels(y_pred, labels, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    if len(yt) == 0:
        return float("nan")
    cm = _confusion(yt, yp, labels).astype(float)
    k = len(labels)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    i, j = np.indices((k, k))
    weights = (i - j) ** 2 / (k - 1) ** 2
    if float(np.sum(weights * expected)) == 0.0:
        return float("nan")
    return float(cohen_kappa_score(yt, yp, labels=list(labels), weights="quadratic"))


def present_class_scores(y_true, y_pred, *, labels: Sequence[int]) -> Dict[str, Any]:
    """Balanced accuracy and macro-F1 over the classes present in ``y_true``.

    Shared by ``classification_metrics`` and the evaluation callers
    (``models.eval_on_test``, ``hybrid.run_heads_on_ft_and_save``) so that
    every reported macro-F1 has the same denominator: the classes of
    ``labels`` (ontology order) with non-zero support in ``y_true``.
    """
    labels = tuple(labels)
    yt, yp = _as_labels(y_true, labels, "y_true"), _as_labels(y_pred, labels, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    if len(yt) == 0:
        raise ValueError("empty evaluated set: balanced accuracy and macro-F1 are undefined")
    present = [c for c in labels if np.any(yt == c)]
    recall = [float(np.mean(yp[yt == c] == c)) for c in present]
    return {
        "classes_present": present,
        "balanced_accuracy": float(np.mean(recall)),
        "balanced_accuracy_single_class": len(present) == 1,
        "macro_f1": float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)),
    }


def ontology_classification_report(y_true, y_pred, *, labels: Sequence[int],
                                   target_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Detailed per-class report with the present-class macro denominator.

    One row per ontology class (absent classes kept, support 0), plus
    ``accuracy``, ``weighted avg`` (support-weighted, so absent classes weigh
    0) and ``macro avg`` = unweighted mean over the classes present in
    ``y_true`` -- the same denominator as ``present_class_scores``. The
    sklearn default (mean over every listed class) is not reported.
    ``macro avg classes`` lists the classes averaged.
    """
    from sklearn.metrics import precision_recall_fscore_support

    labels = tuple(labels)
    yt, yp = _as_labels(y_true, labels, "y_true"), _as_labels(y_pred, labels, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    if len(yt) == 0:
        raise ValueError("empty evaluated set: the report is undefined")
    names = [str(c) for c in labels] if target_names is None else [str(n) for n in target_names]
    if len(names) != len(labels):
        raise ValueError("target_names must have one entry per ontology label")
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=list(labels), zero_division=0)
    report: Dict[str, Any] = {
        n: {"precision": float(p[i]), "recall": float(r[i]), "f1-score": float(f[i]), "support": int(s[i])}
        for i, n in enumerate(names)
    }
    present = s > 0
    n = int(s.sum())
    report["accuracy"] = float(np.mean(yt == yp))
    report["macro avg"] = {"precision": float(np.mean(p[present])), "recall": float(np.mean(r[present])),
                           "f1-score": float(np.mean(f[present])), "support": n}
    report["weighted avg"] = {"precision": float(np.average(p, weights=s)),
                              "recall": float(np.average(r, weights=s)),
                              "f1-score": float(np.average(f, weights=s)), "support": n}
    report["macro avg classes"] = [c for c, keep in zip(labels, present) if keep]
    return report


def format_ontology_report(report: Mapping[str, Any]) -> str:
    """Plain-text rendering of ``ontology_classification_report`` (same numbers)."""
    lines = [f"{'':>14} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>8}"]
    for key, row in report.items():
        if isinstance(row, Mapping):
            lines.append(f"{key:>14} {row['precision']:9.4f} {row['recall']:9.4f} "
                         f"{row['f1-score']:9.4f} {row['support']:8d}")
    lines.append(f"{'accuracy':>14} {report['accuracy']:9.4f}")
    lines.append(f"macro avg over present classes: {list(report['macro avg classes'])}")
    return "\n".join(lines)


def classification_metrics(y_true, y_pred, target: str) -> Dict[str, Any]:
    """Ontology metrics of one evaluated set.

    Accuracy (primary); balanced accuracy = mean recall over the classes present
    in ``y_true`` (flagged when only one class is present); macro-F1 over the
    classes present in ``y_true`` (``zero_division=0``); QWK on the full
    ontology (NaN + ``qwk_undefined`` when its denominator is zero); confusion
    matrix and per-class recall/support over the full ontology.
    """
    labels = ontology(target)
    yt, yp = _as_labels(y_true, labels, "y_true"), _as_labels(y_pred, labels, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred differ in length")
    if len(yt) == 0:
        raise ValueError("empty evaluated set: accuracy is undefined")
    cm = _confusion(yt, yp, labels)
    support = cm.sum(axis=1)
    recall = {c: (float(cm[i, i] / support[i]) if support[i] > 0 else float("nan"))
              for i, c in enumerate(labels)}
    scores = present_class_scores(yt, yp, labels=labels)
    qwk = quadratic_weighted_kappa(yt, yp, labels=labels)
    return {
        "target": target,
        "n": int(len(yt)),
        "n_correct": int(np.trace(cm)),
        "accuracy": float(np.trace(cm) / len(yt)),
        "balanced_accuracy": scores["balanced_accuracy"],
        "balanced_accuracy_single_class": scores["balanced_accuracy_single_class"],
        "macro_f1": scores["macro_f1"],
        "classes_present": scores["classes_present"],
        "qwk": qwk,
        "qwk_undefined": bool(np.isnan(qwk)),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "support": {c: int(s) for c, s in zip(labels, support)},
        "n_predicted": {c: int(s) for c, s in zip(labels, cm.sum(axis=0))},
        "recall": recall,
    }


# --------------------------------------------------------------------------- baselines
def _majority_class(counts: Mapping[int, int]) -> int:
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    return min(tied, key=lambda c: (abs(c), c))


def deployable_majority_class(y_train, target: str) -> int:
    """Most frequent class of the strategy's training data.

    Ties go to the class closest to 0, then to the lower label.
    """
    labels = ontology(target)
    yt = _as_labels(y_train, labels, "y_train")
    if len(yt) == 0:
        raise ValueError("empty training data: no deployable majority class")
    return _majority_class({c: int(np.sum(yt == c)) for c in labels})


def majority_baselines(y_train, y_eval, target: str) -> Dict[str, Any]:
    """Deployable and descriptive majority baselines, with balanced accuracy.

    The deployable baseline predicts the training (or DEV) majority class; the
    descriptive baseline is the majority rate of the evaluated set itself (uses
    test labels; never presented as a model).
    """
    labels = ontology(target)
    ye = _as_labels(y_eval, labels, "y_eval")
    if len(ye) == 0:
        raise ValueError("empty evaluated set")
    dep = deployable_majority_class(y_train, target)
    dep_m = classification_metrics(ye, np.full(len(ye), dep), target)
    desc = _majority_class({c: int(np.sum(ye == c)) for c in labels})
    desc_m = classification_metrics(ye, np.full(len(ye), desc), target)
    return {
        "deployable_class": dep,
        "deployable_accuracy": dep_m["accuracy"],
        "deployable_balanced_accuracy": dep_m["balanced_accuracy"],
        "descriptive_class": desc,
        "descriptive_majority_rate": desc_m["accuracy"],
        "descriptive_balanced_accuracy": desc_m["balanced_accuracy"],
        "n_eval": int(len(ye)),
    }


# --------------------------------------------------------------------------- bootstrap core
@dataclass(frozen=True)
class Estimate:
    """Point estimate with a cluster-bootstrap percentile interval or "not available".

    ``n_denominator`` is the number of votes in the estimand's denominator set,
    ``n_clusters`` the clusters contributing to it, ``n_undefined_replicates``
    the replicates of the frozen draw whose denominator is zero (always
    published). When ``available`` is False the interval is NaN and
    ``reason`` says why; ``replicates`` is then None.
    """

    estimate: float
    ci_low: float
    ci_high: float
    level: float
    available: bool
    reason: str
    n_denominator: int
    n_clusters: int
    n_undefined_replicates: int
    replicates: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    details: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def interval(self):
        """``(low, high)`` or the string "not available" (never a degenerate interval)."""
        return (self.ci_low, self.ci_high) if self.available else NOT_AVAILABLE

    def interval_at(self, level: float) -> Tuple[float, float]:
        """Percentile interval of another level from the same replicates."""
        if not self.available:
            raise ValueError(f"interval not available: {self.reason}")
        return percentile_interval(self.replicates, level)

    def as_row(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "interval_available": self.available,
            "reason": self.reason,
            "n_denominator": self.n_denominator,
            "n_clusters": self.n_clusters,
            "n_undefined_replicates": self.n_undefined_replicates,
        }


def percentile_interval(replicates, level: float) -> Tuple[float, float]:
    """Two-sided percentile interval (numpy ``quantile``, linear interpolation).

    Raises on any NaN replicate: undefined replicates are never skipped.
    """
    reps = np.asarray(replicates, dtype=float)
    if reps.ndim != 1 or reps.size == 0:
        raise ValueError("replicates must be a non-empty 1-D array")
    if np.isnan(reps).any():
        raise ValueError("undefined replicate present: no interval (never dropped)")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    q = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [q, 1.0 - q])
    return float(lo), float(hi)


def bonferroni_level(m: int, alpha: float = ALPHA) -> float:
    """Level 1 - alpha/m of the Bonferroni-adjusted two-sided interval."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return 1.0 - alpha / m


@functools.lru_cache(maxsize=4)
def _draw_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))
    offsets = (np.arange(n_boot, dtype=np.int64) * n_clusters)[:, None]
    w = np.bincount((idx + offsets).ravel(), minlength=n_boot * n_clusters)
    w = w.reshape(n_boot, n_clusters).astype(float)
    w.setflags(write=False)
    return w


def frozen_draw_weights(n_clusters: int, n_boot: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """Multiplicity matrix (B, G) of the frozen draw.

    ``rng = numpy.random.default_rng(seed)``; ``rng.integers(0, G, size=(B, G))``
    drawn once; entry (b, g) = number of times cluster g (position in the sorted
    cluster list) is drawn in replicate b. Deterministic in (G, B, seed), hence
    shared by every estimand, model and contrast evaluated on the same cluster
    list. Read-only.
    """
    n_clusters, n_boot, seed = int(n_clusters), int(n_boot), int(seed)
    if n_clusters < 1 or n_boot < 1:
        raise ValueError("need at least one cluster and one replicate")
    return _draw_weights(n_clusters, n_boot, seed)


def cluster_positions(clusters, cluster_order=None) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted cluster list and the position of each observation's cluster in it.

    Without ``cluster_order`` the list is the sorted distinct values of
    ``clusters`` (buildings of the evaluated set). With it (Phase 3: the
    cohort's full group list), it must be unique and sorted ascending,
    and every observed cluster must belong to it.
    """
    c = np.asarray(clusters)
    if c.ndim != 1:
        raise ValueError("clusters must be one-dimensional")
    if pd.isna(c).any():
        raise ValueError("missing cluster identifier")
    if cluster_order is None:
        order = np.unique(c)
    else:
        order = np.asarray(cluster_order)
        if order.ndim != 1 or len(order) == 0:
            raise ValueError("cluster_order must be a non-empty 1-D list")
        if len(np.unique(order)) != len(order):
            raise ValueError("cluster_order contains duplicates")
        if not np.array_equal(np.sort(order), order):
            raise ValueError("cluster_order must be sorted ascending (frozen order)")
    pos = np.searchsorted(order, c)
    if len(c) and (np.any(pos >= len(order)) or np.any(order[np.minimum(pos, len(order) - 1)] != c)):
        raise ValueError("an observation's cluster is absent from cluster_order")
    return order, pos


def _not_available(estimate, level, reasons, n_den, n_clu, n_undef, details=None) -> Estimate:
    return Estimate(float(estimate), float("nan"), float("nan"), level, False, "; ".join(reasons),
                    int(n_den), int(n_clu), int(n_undef), None, dict(details or {}))


def cluster_bootstrap_ratio(
    numerator,
    denominator,
    clusters,
    *,
    cluster_order=None,
    level: float = 0.95,
    min_clusters: int = MIN_CLUSTERS,
    min_denominator: int = 0,
    gate: bool = True,
    gate_reason: str = "stratum-level support rule not met",
    n_boot: int = BOOT_B,
    seed: int = BOOT_SEED,
) -> Estimate:
    """Ratio estimand sum(numerator) / sum(denominator) with a cluster bootstrap.

    ``denominator`` is a 0/1 mask of the estimand's denominator set D;
    ``numerator`` is any per-observation value that is 0 outside D (an
    indicator for a rate, a paired difference for a contrast). Each replicate
    recomputes sum(w_g * num_g) / sum(w_g * den_g) with the frozen multiplicities
    w of :func:`frozen_draw_weights`, which equals keeping every vote of a
    cluster drawn m times m times.

    Eligibility: an interval is computed only if ``gate`` holds,
    |D| >= ``min_denominator``, at least ``min_clusters`` clusters contribute
    to D and no replicate has a zero denominator. Otherwise the result is
    descriptive (point estimate, |D|, contributing clusters and the number of
    undefined replicates) with the interval "not available".
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.ndim != 1 or num.shape != den.shape:
        raise ValueError("numerator and denominator must be 1-D arrays of equal length")
    if not np.all(np.isfinite(num)):
        raise ValueError("numerator contains NaN or infinite values")
    if not np.all((den == 0) | (den == 1)):
        raise ValueError("denominator must be a 0/1 mask")
    if np.any((den == 0) & (num != 0)):
        raise ValueError("numerator must be zero outside the denominator set")
    if len(np.asarray(clusters)) != len(num):
        raise ValueError("clusters must have one entry per observation")
    n_den = int(den.sum())
    if len(num) == 0 and cluster_order is None:
        return _not_available(float("nan"), level, ["empty evaluated set"], 0, 0, n_boot)
    order, pos = cluster_positions(clusters, cluster_order)
    g = len(order)
    num_g = np.bincount(pos, weights=num, minlength=g)
    den_g = np.bincount(pos, weights=den, minlength=g)
    contributing = int(np.sum(den_g > 0))
    estimate = float(num_g.sum() / den_g.sum()) if n_den > 0 else float("nan")
    w = frozen_draw_weights(g, n_boot, seed)
    rep_den = w @ den_g
    n_undef = int(np.sum(rep_den == 0))
    reasons = []
    if not gate:
        reasons.append(gate_reason)
    if n_den < min_denominator:
        reasons.append(f"{n_den} denominator votes < {min_denominator}")
    if contributing < min_clusters:
        reasons.append(f"{contributing} contributing clusters < {min_clusters}")
    if n_undef:
        reasons.append(f"{n_undef} of {n_boot} replicates with zero denominator")
    if reasons:
        return _not_available(estimate, level, reasons, n_den, contributing, n_undef)
    reps = (w @ num_g) / rep_den
    lo, hi = percentile_interval(reps, level)
    return Estimate(estimate, lo, hi, level, True, "", n_den, contributing, 0, reps)


def _prediction_matrix(preds, labels, n: int, what: str) -> np.ndarray:
    """(S, n) int label matrix from one prediction vector or one per seed."""
    arr = np.asarray(preds)
    rows = [arr] if arr.ndim == 1 else list(arr)
    if not rows:
        raise ValueError(f"{what}: no predictions")
    out = np.stack([_as_labels(r, labels, what) for r in rows])
    if out.shape[1] != n:
        raise ValueError(f"{what}: {out.shape[1]} predictions for {n} votes")
    return out


def accuracy_estimate(y_true, y_pred, clusters, *, target: str, level: float = 0.95,
                      cluster_order=None, min_clusters: int = MIN_CLUSTERS,
                      n_boot: int = BOOT_B, seed: int = BOOT_SEED) -> Estimate:
    """Accuracy of one model (seed-mean if ``y_pred`` is (S, n)) with a cluster interval."""
    labels = ontology(target)
    yt = _as_labels(y_true, labels, "y_true")
    correct = (_prediction_matrix(y_pred, labels, len(yt), "y_pred") == yt).mean(axis=0)
    return cluster_bootstrap_ratio(correct, np.ones(len(yt)), clusters, cluster_order=cluster_order,
                                   level=level, min_clusters=min_clusters, n_boot=n_boot, seed=seed)


def paired_accuracy_contrast(y_true, preds_a, preds_b, clusters, *, target: str,
                             level: float = 0.95, cluster_order=None,
                             min_clusters: int = MIN_CLUSTERS,
                             n_boot: int = BOOT_B, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Paired contrast Delta = acc(A) - acc(B) on the same votes.

    ``preds_a`` / ``preds_b`` are one prediction vector or one per seed (S, n);
    acc(model) is the mean over its seeds. Every replicate recomputes every
    seed-specific accuracy on the same resampled votes: with a common
    denominator this equals the ratio of the seed-averaged per-vote
    differences, which is what is bootstrapped. The seed spread is reported
    separately.
    """
    labels = ontology(target)
    yt = _as_labels(y_true, labels, "y_true")
    ca = (_prediction_matrix(preds_a, labels, len(yt), "preds_a") == yt).astype(float)
    cb = (_prediction_matrix(preds_b, labels, len(yt), "preds_b") == yt).astype(float)
    d = ca.mean(axis=0) - cb.mean(axis=0)
    est = cluster_bootstrap_ratio(d, np.ones(len(yt)), clusters, cluster_order=cluster_order,
                                  level=level, min_clusters=min_clusters, n_boot=n_boot, seed=seed)
    seed_a, seed_b = ca.mean(axis=1), cb.mean(axis=1)
    return {
        "delta": est,
        "accuracy_a": float(seed_a.mean()),
        "accuracy_b": float(seed_b.mean()),
        "seed_accuracies_a": seed_a.tolist(),
        "seed_accuracies_b": seed_b.tolist(),
        "seed_spread_a": float(seed_a.max() - seed_a.min()),
        "seed_spread_b": float(seed_b.max() - seed_b.min()),
        "n_votes": int(len(yt)),
    }


def balanced_accuracy_estimate(y_true, y_pred, clusters, *, target: str, level: float = 0.95,
                               gate: bool = True,
                               gate_reason: str = "stratum-level support rule not met",
                               cluster_order=None, min_clusters: int = MIN_CLUSTERS,
                               n_boot: int = BOOT_B, seed: int = BOOT_SEED) -> Estimate:
    """Balanced accuracy (mean recall over the classes present in ``y_true``) with interval.

    Eligibility: an interval requires ``gate``, and, for **every** present class,
    >= ``min_clusters`` contributing clusters and no zero-denominator replicate
    (a replicate lacking one present class is undefined). The 30-vote condition
    (b) is deliberately not applied to the component recalls.
    """
    labels = ontology(target)
    yt = _as_labels(y_true, labels, "y_true")
    yp = _prediction_matrix(y_pred, labels, len(yt), "y_pred")
    if len(yt) == 0:
        return _not_available(float("nan"), level, ["empty evaluated set"], 0, 0, n_boot)
    order, pos = cluster_positions(clusters, cluster_order)
    g = len(order)
    w = frozen_draw_weights(g, n_boot, seed)
    present = [c for c in labels if np.any(yt == c)]
    rec_num, rec_den, per_class = [], [], {}
    undefined = np.zeros(n_boot, dtype=bool)
    reasons = [] if gate else [gate_reason]
    for c in present:
        den_g = np.bincount(pos, weights=(yt == c).astype(float), minlength=g)
        num_g = np.bincount(pos, weights=((yp == c) & (yt == c)).mean(axis=0), minlength=g)
        rd = w @ den_g
        undefined |= rd == 0
        contributing = int(np.sum(den_g > 0))
        per_class[c] = {"n_denominator": int(den_g.sum()), "n_clusters": contributing,
                        "n_undefined_replicates": int(np.sum(rd == 0))}
        if contributing < min_clusters:
            reasons.append(f"class {c}: {contributing} contributing clusters < {min_clusters}")
        rec_num.append(w @ num_g)
        rec_den.append(rd)
        per_class[c]["recall"] = float(num_g.sum() / den_g.sum())
    estimate = float(np.mean([per_class[c]["recall"] for c in present]))
    n_undef = int(undefined.sum())
    if n_undef:
        reasons.append(f"{n_undef} of {n_boot} replicates lack a present class")
    details = {"classes_present": present, "per_class": per_class}
    if reasons:
        return _not_available(estimate, level, reasons, len(yt), g, n_undef, details)
    reps = np.mean([n_ / d_ for n_, d_ in zip(rec_num, rec_den)], axis=0)
    lo, hi = percentile_interval(reps, level)
    return Estimate(estimate, lo, hi, level, True, "", len(yt), g, 0, reps, details)


# --------------------------------------------------------------------------- grouped accuracy
def grouped_accuracy(y_true, y_pred, groups, clusters, *, target: str, level: float = 0.95,
                     n_boot: int = BOOT_B, seed: int = BOOT_SEED) -> Dict[Any, Estimate]:
    """Accuracy per group (e.g. per held-out publication, E2) with the cluster eligibility rule.

    The draw is over the sorted buildings of each group's own evaluated set; a
    group with fewer than 10 buildings is descriptive (interval "not available").
    """
    g = np.asarray(groups)
    cl = np.asarray(clusters)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return {k: accuracy_estimate(yt[g == k], yp[g == k], cl[g == k], target=target, level=level,
                                 n_boot=n_boot, seed=seed)
            for k in sorted(set(g.tolist()), key=str)}


# --------------------------------------------------------------------------- predictions files
PREDICTION_KEY_COLUMNS: Tuple[str, ...] = ("row_idx", "y_true", "y_pred")


def load_predictions(path) -> pd.DataFrame:
    """Read one predictions file (gzip CSV) and check its key columns."""
    df = pd.read_csv(Path(path))
    missing = [c for c in PREDICTION_KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def align_predictions(frames: Mapping[str, pd.DataFrame], cluster_col: str = "building_id",
                      expected_row_idx=None
                      ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Check that predictions files describe the same votes in the same order.

    Each frame must hold exactly one prediction per vote: ``row_idx`` without
    missing or duplicate values. Asserts identical ``row_idx`` sequence,
    ``y_true`` and ``cluster_col`` across every frame (paired contrasts need
    exactly the same votes) and, when ``expected_row_idx`` is given (the
    serialised split IDs), that this sequence equals it in order. Returns
    ``(y_true, {name: y_pred}, clusters, row_idx)``.
    """
    if not frames:
        raise ValueError("no predictions")
    names = list(frames)
    for name in names:
        idx = frames[name]["row_idx"]
        if idx.isna().any():
            raise ValueError(f"{name}: missing row_idx values")
        dup = idx[idx.duplicated()].unique().tolist()
        if dup:
            raise ValueError(f"{name}: duplicate row_idx values {dup[:10]} (one prediction per vote)")
    ref = frames[names[0]]
    if expected_row_idx is not None:
        exp = np.asarray(expected_row_idx)
        if len(exp) != len(ref) or not np.array_equal(ref["row_idx"].to_numpy(), exp):
            raise ValueError(f"{names[0]}: row_idx differs from the serialised split IDs "
                             f"({len(ref)} rows vs {len(exp)} expected)")
    for name in names[1:]:
        df = frames[name]
        for col in ("row_idx", "y_true", cluster_col):
            if len(df) != len(ref) or not np.array_equal(df[col].to_numpy(), ref[col].to_numpy()):
                raise ValueError(f"{name}: column {col!r} differs from {names[0]} (votes not paired)")
    return (ref["y_true"].to_numpy(), {n: frames[n]["y_pred"].to_numpy() for n in names},
            ref[cluster_col].to_numpy(), ref["row_idx"].to_numpy())
