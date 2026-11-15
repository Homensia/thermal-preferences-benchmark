"""Small exact computations used by the tables and figures (no model, no sampling).

* ``weighted_f1_from_cm``: weighted-F1 (sklearn ``average="weighted"`` on the
  full ontology, ``zero_division=0``) from a confusion matrix; the revision
  metrics files carry accuracy, macro-F1 and QWK but not weighted-F1, which
  the paper tables report.
* ``chance_disagreement``: expected disagreement between the vote of a query
  and the vote of a candidate drawn at random, independently, from the class
  distributions of the two pools: ``1 - sum_c p_query(c) * p_cand(c)``. When
  the query and candidate pools are the same population this is
  ``1 - sum_c p_c**2``.
* ``seed_means``: mean over seeds of per-seed metric rows.
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TARGETS = ("thermal_sensation", "TSV_3p", "thermal_preference")
TARGET_LABEL = {"thermal_sensation": "TSV-7", "TSV_3p": "TSV-3", "thermal_preference": "TPV"}
ONTOLOGY = {"thermal_sensation": (-3, -2, -1, 0, 1, 2, 3), "TSV_3p": (-1, 0, 1),
            "thermal_preference": (-1, 0, 1)}
# meaning of the vote codes (data_preparation.encode_targets: TPV {cooler: -1, no change: 0, warmer: +1};
# TSV-3 = TSV-7 collapsed {-3, -2, -1} -> -1, 0 -> 0, {1, 2, 3} -> +1, revision_runners.tsv7_to_tsv3)
VOTE_MEANING = {
    "thermal_sensation": {-3: "cold", -2: "cool", -1: "slightly cool", 0: "neutral", 1: "slightly warm",
                          2: "warm", 3: "hot"},
    "TSV_3p": {-1: "cool side", 0: "neutral", 1: "warm side"},
    "thermal_preference": {-1: "want cooler", 0: "no change", 1: "want warmer"},
}
# one name per model and per dataset in every figure, table and caption
MODEL_NAME = {"RF": "RF", "XGB": "XGBoost", "SVM": "SVM", "ANN": "ANN", "FTT": "FTT", "FTT_RF": "FTT+RF",
              "FTT_XGB": "FTT+XGBoost"}
DATASET_NAME = {"ASHRAE_2018": "ASHRAE-2018 (Haghirad 2018 v2 release)", "ASHRAE_2022": "ASHRAE-2022",
                "Moujalled": "Moujalled", "Hostein": "Hostein"}


def vote_code(v: int) -> str:
    """'+1', '0', '-1' (typographic minus) as printed on the axes."""
    return f"+{v}" if v > 0 else ("0" if v == 0 else f"\u2212{-v}")


def vote_codes_sentence(target: str) -> str:
    """'TPV: -1 want cooler, 0 no change, +1 want warmer' (LaTeX)."""
    parts = [f"${'+' if v > 0 else ''}{v}$ {m}" for v, m in VOTE_MEANING[target].items()]
    return f"{TARGET_LABEL[target]} codes: " + ", ".join(parts)


def as_matrix(cm) -> np.ndarray:
    """A confusion matrix given as a JSON string or nested list (rows = true, columns = predicted)."""
    if isinstance(cm, str):
        cm = json.loads(cm)
    m = np.asarray(cm, dtype=np.int64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {m.shape}")
    if (m < 0).any():
        raise ValueError("confusion matrix has negative counts")
    return m


def accuracy_from_cm(cm) -> float:
    m = as_matrix(cm)
    return float(np.trace(m) / m.sum())


def weighted_f1_from_cm(cm) -> float:
    m = as_matrix(cm).astype(float)
    tp = np.diag(m)
    n_true = m.sum(axis=1)
    n_pred = m.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp / n_pred, 0.0)
        recall = np.where(n_true > 0, tp / n_true, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    if n_true.sum() == 0:
        raise ValueError("empty confusion matrix")
    return float((f1 * n_true).sum() / n_true.sum())


def class_distribution(labels: Iterable, classes: Sequence) -> np.ndarray:
    """Class proportions of ``labels`` over ``classes`` (a label outside ``classes`` raises)."""
    y = pd.Series(list(labels))
    unknown = sorted(set(y.unique()) - set(classes))
    if unknown:
        raise ValueError(f"labels {unknown} outside the classes {list(classes)}")
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    if counts.sum() == 0:
        raise ValueError("no label")
    return counts / counts.sum()


def chance_disagreement(p_query, p_cand=None) -> float:
    """``1 - sum_c p_query(c) p_cand(c)``; ``p_cand=None`` means the same pool (``1 - sum_c p_c**2``)."""
    p_q = np.asarray(p_query, dtype=float)
    p_c = p_q if p_cand is None else np.asarray(p_cand, dtype=float)
    if p_q.shape != p_c.shape:
        raise ValueError("the two distributions must be over the same classes")
    for p in (p_q, p_c):
        if (p < 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-12):
            raise ValueError("a class distribution must be non-negative and sum to 1")
    return float(1.0 - np.dot(p_q, p_c))


def seed_means(rows: pd.DataFrame, keys: Sequence[str], metrics: Sequence[str],
               seed_col: str = "seed") -> pd.DataFrame:
    """Mean, min and max over seeds of ``metrics`` per ``keys``; ``seeds`` lists the seeds averaged."""
    out = []
    for key, g in rows.groupby(list(keys), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        rec["seeds"] = sorted(int(s) for s in g[seed_col])
        rec["n_seeds"] = len(g)
        for m in metrics:
            v = g[m].astype(float)
            rec[m] = float(v.mean())
            rec[f"{m}_min"] = float(v.min())
            rec[f"{m}_max"] = float(v.max())
        out.append(rec)
    return pd.DataFrame(out)


def check_close(a: Mapping, b: Mapping, keys: Sequence[str], what: str, tol: float = 1e-12) -> None:
    for k in keys:
        if not np.isclose(float(a[k]), float(b[k]), rtol=0, atol=tol):
            raise ValueError(f"{what}: {k} = {a[k]!r} here, {b[k]!r} in the report")
