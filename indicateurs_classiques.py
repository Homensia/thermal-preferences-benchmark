# -*- coding: utf-8 -*-
"""Physics-based index baselines of the revision (experiments E7 and E7b).

E7 (``--pmv_only``) computes PMV and its calibrated variant aPMV. E7b
(``--indices set_pet_epmv``) restores the pre-revision SET / PET with their
linear transforms PTS_SET / PTS_PET and ePMV, with the index-computation
fixes (inputs from the real columns, every
calibration on the archived training rows, NaN never a class), for TSV-7 and
TSV-3 only; E7 is unchanged. aPTS, ePTS and PPD are not restored.

Frozen PMV contract:

* ``pythermalcomfort.models.pmv_ppd_iso`` at the pinned version, whose version
  and source SHA-256 are checked before any computation (the code refuses to
  run otherwise);
* call ``pmv_ppd_iso(tdb=Tair, tr=tr, vr=vel, rh=RH, met=Met, clo=clo, wme=0,
  model="7730-2005", units="SI", limit_inputs=True, round_output=False)``;
* no default input value and no clipping: a missing column raises, a missing
  value gives a NaN PMV;
* a row is valid iff the returned PMV is finite (library mask: inclusive
  bounds 10 <= Tair <= 30, 10 <= tr <= 40, 0 <= vel <= 1, 0.8 <= Met <= 4,
  0 <= clo <= 2, -2 <= PMV <= 2); NaN is never converted to a class;
* conversion on the unrounded value: TSV-7 = floor(x + 0.5), TSV-3 by the
  7-to-3 map (``tsv7_to_tsv3``), TPV: x > 0.5 -> -1, x < -0.5 -> +1, else 0;
* sensitivity ``dynamic``: vr = v_relative(vel, Met) and
  clo = clo_dynamic_iso(clo, Met, vel), same call otherwise.

aPMV: aPMV = PMV / (1 + lam * PMV), lam in
[-0.2, 0.2] fitted by a deterministic bounded least-squares search on the
``train_idx`` rows of the evaluated target that lie inside the PMV mask;
"not available" on failure, never a fallback coefficient.

Before calibration, each target's split is checked to be the archived one
(complete partition of the cohort and ordered equality with the
stratified re-draw). ML predictions compared on the mask are validated
(``y_true`` required, both label columns finite, integral, in the ontology)
and, before any computation, their inventory is checked: every evaluated
target has the same ML groups, equal to ``--expected_ml``.

Command (E7; ``--expected_ml`` lists the E1 ``model:seed`` set):
    python3 indicateurs_classiques.py --pmv_only --splits splits/ASHRAE_2022 \\
        --ml_predictions RUN/E1_phase2/predictions \\
        --expected_ml RF:42 RF:43 RF:44 ... \\
        RUN/E7_pmv
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedShuffleSplit

import pythermalcomfort
from pythermalcomfort.models import pmv_ppd_iso, set_tmp, pet_steady, pmv_e
from pythermalcomfort.utilities import clo_dynamic_iso, v_relative

import revision_stats as rs
from config_loader import check_not_protected, read_consumed
# Label validation of the shared metrics (finite, integral, in ontology).
from revision_stats import _as_labels

# --- Pinned implementation ---------------------------------------------------
PINNED_PYTHERMALCOMFORT = "4.0.2"
# SHA-256 of pythermalcomfort/models/pmv_ppd_iso.py in 4.0.2 (same value as
# tests/test_environment.py).
PINNED_PMV_PPD_ISO_SHA256 = (
    "56ea982c98d1f22475cf0530f3ebbf25eecc4febda0038c2be5b7a873f792aa7"
)
# Library defaults inherited by the ``dynamic`` sensitivity.
PINNED_CLO_DYNAMIC_ISO_DEFAULTS = {"i_a": 0.7, "model": "9920-2007"}

# Frozen call arguments besides the six inputs.
PMV_CALL = {
    "wme": 0,
    "model": "7730-2005",
    "units": "SI",
    "limit_inputs": True,
    "round_output": False,
}

# Inclusive validity bounds of pmv_ppd_iso.py:171-189 (4.0.2).
INPUT_BOUNDS = {
    "Tair": (10.0, 30.0),
    "tr": (10.0, 40.0),
    "vel": (0.0, 1.0),
    "Met": (0.8, 4.0),
    "clo": (0.0, 2.0),
}
PMV_BOUNDS = (-2.0, 2.0)
INPUT_COLUMNS = ("Tair", "tr", "vel", "RH", "Met", "clo")
VARIANTS = ("primary", "dynamic")

# --- Targets -----------------------------------------------------------------
TARGETS = ["thermal_sensation", "TSV_3p", "thermal_preference"]
TSV7_COLUMN = "thermal_sensation"
ONTOLOGY = {t: list(labels) for t, labels in rs.ONTOLOGY.items()}
TPV_POS = 0.5
TPV_NEG = -0.5

# --- aPMV calibration contract ----------------------------------------------
APMV_LAMBDA_BOUND = 0.2
APMV_GRID = -0.2 + 1e-4 * np.arange(4001)  # lam_k = -0.2 + 1e-4 k, k = 0..4000
APMV_GRID_STEP = 1e-4
APMV_XATOL = 1e-10
APMV_MIN_ROWS = 100
NOT_AVAILABLE = "not available"

EXPERIMENT = "E7"
COHORT = "ASHRAE_2022"
FEATURE_SET = "pmv_inputs"
SPLIT_ID = "archived_80_20"
DEFAULT_DATA_CSV = "Data/ASHRAE_2022_Clean_api.csv"
DEFAULT_SPLITS_DIR = "splits/ASHRAE_2022"
# Archived external split: StratifiedShuffleSplit on the target.
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42
METHOD_NAMES = {"primary": "PMV", "dynamic": "PMV_dynamic"}

assert APMV_GRID[0] == -APMV_LAMBDA_BOUND and APMV_GRID[-1] == APMV_LAMBDA_BOUND
assert np.all(np.abs(APMV_GRID) <= APMV_LAMBDA_BOUND)


class PythermalcomfortPinError(RuntimeError):
    """The installed pythermalcomfort differs from the pinned one."""


class CalibrationLeakageError(ValueError):
    """A row outside train_idx ∩ PMV mask was offered to the aPMV fit."""


# ---------------------------------------------------------------------------
# Version and source-hash guard
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pythermalcomfort_provenance() -> dict:
    """Version, PMV source hash and inherited defaults, for the manifest."""
    pkg_dir = Path(pythermalcomfort.__file__).parent
    sig = inspect.signature(clo_dynamic_iso)
    return {
        "pythermalcomfort_version": importlib.metadata.version("pythermalcomfort"),
        "pmv_ppd_iso_sha256": _sha256_file(pkg_dir / "models" / "pmv_ppd_iso.py"),
        "shared_functions_sha256": _sha256_file(pkg_dir / "shared_functions.py"),
        "clo_dynamic_iso_defaults": {
            k: sig.parameters[k].default for k in PINNED_CLO_DYNAMIC_ISO_DEFAULTS
        },
        "pmv_call": dict(PMV_CALL),
        "input_bounds_inclusive": {k: list(v) for k, v in INPUT_BOUNDS.items()},
        "pmv_bounds_inclusive": list(PMV_BOUNDS),
    }


def check_pythermalcomfort_pin(provenance: Optional[dict] = None) -> dict:
    """Raise :class:`PythermalcomfortPinError` unless the pinned library is used."""
    prov = pythermalcomfort_provenance() if provenance is None else provenance
    problems = []
    if prov["pythermalcomfort_version"] != PINNED_PYTHERMALCOMFORT:
        problems.append(
            f"version {prov['pythermalcomfort_version']} != {PINNED_PYTHERMALCOMFORT}"
        )
    if prov["pmv_ppd_iso_sha256"] != PINNED_PMV_PPD_ISO_SHA256:
        problems.append(
            f"pmv_ppd_iso.py SHA-256 {prov['pmv_ppd_iso_sha256']} "
            f"!= {PINNED_PMV_PPD_ISO_SHA256}"
        )
    if prov["clo_dynamic_iso_defaults"] != PINNED_CLO_DYNAMIC_ISO_DEFAULTS:
        problems.append(
            f"clo_dynamic_iso defaults {prov['clo_dynamic_iso_defaults']} "
            f"!= {PINNED_CLO_DYNAMIC_ISO_DEFAULTS}"
        )
    if problems:
        raise PythermalcomfortPinError(
            "Refusing to compute PMV: " + "; ".join(problems)
        )
    return prov


# ---------------------------------------------------------------------------
# PMV and its validity mask
# ---------------------------------------------------------------------------
def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"PMV input column {col!r} is missing (no default is used)")
    return pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)


def input_predicates(inputs: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per input predicate of the validity mask (inclusive bounds)."""
    out = {}
    for col, (lo, hi) in INPUT_BOUNDS.items():
        x = _column(inputs, col)
        out[col] = np.isfinite(x) & (x >= lo) & (x <= hi)
    out["RH"] = np.isfinite(_column(inputs, "RH"))
    return pd.DataFrame(out, index=inputs.index)


def pmv_in_range(pmv) -> np.ndarray:
    pmv = np.asarray(pmv, dtype=float)
    return np.isfinite(pmv) & (pmv >= PMV_BOUNDS[0]) & (pmv <= PMV_BOUNDS[1])


def model_inputs(data: pd.DataFrame, variant: str = "primary") -> pd.DataFrame:
    """Inputs passed to pmv_ppd_iso: measured vel/clo, or the ``dynamic`` sensitivity."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown PMV variant {variant!r}")
    cols = {c: _column(data, c) for c in INPUT_COLUMNS}
    if variant == "dynamic":
        vel, met, clo = cols["vel"], cols["Met"], cols["clo"]
        cols["vel"] = np.asarray(v_relative(v=vel, met=met), dtype=float)
        cols["clo"] = np.asarray(clo_dynamic_iso(clo=clo, met=met, v=vel), dtype=float)
    return pd.DataFrame(cols, index=data.index)


def _call_pmv(inp: pd.DataFrame, limit_inputs: bool) -> np.ndarray:
    kwargs = dict(PMV_CALL, limit_inputs=limit_inputs)
    with warnings.catch_warnings():
        # The library warns on every out-of-range value; the mask is reported.
        warnings.simplefilter("ignore")
        res = pmv_ppd_iso(
            tdb=inp["Tair"].to_numpy(), tr=inp["tr"].to_numpy(),
            vr=inp["vel"].to_numpy(), rh=inp["RH"].to_numpy(),
            met=inp["Met"].to_numpy(), clo=inp["clo"].to_numpy(), **kwargs,
        )
    return np.asarray(res.pmv, dtype=float).reshape(-1)


def compute_pmv(data: pd.DataFrame, variant: str = "primary") -> np.ndarray:
    """Unrounded PMV per row; NaN marks a row outside the validity mask.

    Besides the pinned call, the PMV is recomputed with ``limit_inputs=False``
    to check that the library mask equals the frozen validity predicate
    (finite inputs, inclusive input bounds and -2 <= PMV <= 2); any difference
    raises instead of being silently absorbed.
    """
    check_pythermalcomfort_pin()
    if len(data) == 0:
        return np.empty(0, dtype=float)
    inp = model_inputs(data, variant)
    pmv = _call_pmv(inp, limit_inputs=True)
    raw = _call_pmv(inp, limit_inputs=False)
    expected = input_predicates(inp).all(axis=1).to_numpy() & pmv_in_range(raw)
    valid = np.isfinite(pmv)
    if not np.array_equal(valid, expected) or not np.array_equal(pmv[valid], raw[valid]):
        raise PythermalcomfortPinError(
            "pmv_ppd_iso mask differs from the frozen validity predicate"
        )
    return pmv


# ---------------------------------------------------------------------------
# Conversion: never from NaN, never clipped
# ---------------------------------------------------------------------------
def to_tsv7(values, valid_range: Tuple[int, int]) -> pd.arrays.IntegerArray:
    """floor(x + 0.5) on finite values; <NA> elsewhere; raises outside range."""
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    cls = np.floor(x[finite] + 0.5)
    lo, hi = valid_range
    if cls.size and (cls.min() < lo or cls.max() > hi):
        raise ValueError(
            f"TSV-7 class outside [{lo}, {hi}] (min {cls.min()}, max {cls.max()}); "
            "classes are never clipped"
        )
    out = pd.array([pd.NA] * len(x), dtype="Int64")
    out[finite] = cls.astype(int)
    return out


def tsv7_to_tsv3(tsv7) -> pd.arrays.IntegerArray:
    """{-3,-2,-1} -> -1, 0 -> 0, {1,2,3} -> +1; <NA> kept."""
    s = pd.array(tsv7, dtype="Int64")
    out = pd.array([pd.NA] * len(s), dtype="Int64")
    known = ~s.isna()
    v = s[known].to_numpy(dtype=int)
    out[known] = np.where(v <= -1, -1, np.where(v >= 1, 1, 0))
    return out


def to_tpv(values) -> pd.arrays.IntegerArray:
    """x > 0.5 -> -1 (cooler), x < -0.5 -> +1 (warmer), else 0; NaN -> <NA>."""
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = pd.array([pd.NA] * len(x), dtype="Int64")
    xf = x[finite]
    out[finite] = np.where(xf > TPV_POS, -1, np.where(xf < TPV_NEG, 1, 0))
    return out


def index_to_class(values, target: str, tsv7_range: Tuple[int, int]):
    if target == "thermal_sensation":
        return to_tsv7(values, tsv7_range)
    if target == "TSV_3p":
        return tsv7_to_tsv3(to_tsv7(values, tsv7_range))
    if target == "thermal_preference":
        return to_tpv(values)
    raise ValueError(f"unknown target {target!r}")


def pmv_to_class(pmv, target: str):
    """PMV -> class; valid PMV lies in [-2, 2], so TSV-7 is asserted in [-2, 2]."""
    return index_to_class(pmv, target, (-2, 2))


def apmv_to_class(apmv_values, target: str):
    """aPMV -> class; TSV-7 asserted in [-3, 3] (|lam| <= 0.2)."""
    return index_to_class(apmv_values, target, (-3, 3))


# ---------------------------------------------------------------------------
# aPMV
# ---------------------------------------------------------------------------
def apmv(pmv, lam: float) -> np.ndarray:
    """aPMV = PMV / (1 + lam * PMV); NaN PMV gives NaN; raises out of domain."""
    lam = float(lam)
    if not np.isfinite(lam) or abs(lam) > APMV_LAMBDA_BOUND:
        raise ValueError(f"lambda {lam} outside [-{APMV_LAMBDA_BOUND}, {APMV_LAMBDA_BOUND}]")
    pmv = np.asarray(pmv, dtype=float)
    finite = np.isfinite(pmv)
    denom = 1.0 + lam * pmv[finite]
    if np.any(denom <= 0):
        raise ValueError("aPMV denominator 1 + lambda * PMV <= 0")
    out = np.full(pmv.shape, np.nan)
    out[finite] = pmv[finite] / denom
    return out


@dataclass
class APMVCalibration:
    status: str  # "ok" or NOT_AVAILABLE
    lam: Optional[float]
    sse: Optional[float]
    n_calibration: int
    bound_active: Optional[bool]
    reason: Optional[str]
    calibration_idx_sha256: str

    @property
    def available(self) -> bool:
        return self.status == "ok"


def _index_array(idx) -> np.ndarray:
    arr = np.asarray(idx)
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError("indices must be a 1-D integer array")
    arr = arr.astype(np.int64)
    if np.unique(arr).size != arr.size:
        raise ValueError("indices contain duplicates")
    return arr


def apmv_calibration_rows(pmv, train_idx, test_idx) -> np.ndarray:
    """train_idx rows (in train_idx order) whose PMV is valid; asserts no overlap."""
    pmv = np.asarray(pmv, dtype=float)
    train_idx, test_idx = _index_array(train_idx), _index_array(test_idx)
    if np.intersect1d(train_idx, test_idx).size:
        raise CalibrationLeakageError("train_idx and test_idx overlap")
    return train_idx[np.isfinite(pmv[train_idx])]


def _sse(lam: float, p: np.ndarray, y: np.ndarray) -> float:
    r = y - apmv(p, lam)
    return float(np.sum(r * r))


def _fit_lambda(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Bounded grid search then bounded refinement (deterministic)."""
    grid_sse = np.array([_sse(lam, p, y) for lam in APMV_GRID])
    if not np.all(np.isfinite(grid_sse)):
        raise FloatingPointError("non-finite SSE on the lambda grid")
    cand = np.flatnonzero(grid_sse == grid_sse.min())
    # Ties: smallest |lam|, then lower lam.
    k = cand[np.lexsort((APMV_GRID[cand], np.abs(APMV_GRID[cand])))[0]]
    lam_k, sse_k = float(APMV_GRID[k]), float(grid_sse[k])

    def objective(lam: float) -> float:
        # Failure policy: any non-finite SSE met by the optimizer is a
        # failure (SciPy would otherwise only report success=False).
        s = _sse(lam, p, y)
        if not np.isfinite(s):
            raise FloatingPointError("non-finite SSE in the bounded refinement")
        return s

    res = minimize_scalar(
        objective,
        method="bounded",
        bounds=(max(-APMV_LAMBDA_BOUND, lam_k - APMV_GRID_STEP),
                min(APMV_LAMBDA_BOUND, lam_k + APMV_GRID_STEP)),
        options={"xatol": APMV_XATOL},
    )
    if not res.success:
        raise RuntimeError(f"bounded refinement failed: {res.message}")
    lam_r = float(res.x)
    sse_r = _sse(lam_r, p, y)
    # A failed or non-finite refinement is never a reason to fall back on the
    # grid coefficient. The grid candidate is kept only when the refinement
    # succeeded, is finite, and is not better.
    if not np.isfinite(sse_r):
        raise FloatingPointError("non-finite SSE in the bounded refinement")
    if sse_r <= sse_k:
        return lam_r, sse_r
    return lam_k, sse_k


def fit_apmv_lambda(pmv, tsv7, train_idx, test_idx,
                    calibration_idx=None) -> APMVCalibration:
    """Fit lam on train_idx ∩ PMV mask (response: TSV-7 label).

    ``pmv`` and ``tsv7`` are full-cohort arrays indexed by row position.
    ``calibration_idx`` may be passed only to be checked: it must be exactly
    train_idx ∩ PMV mask, otherwise :class:`CalibrationLeakageError` is
    raised (a test row is never fitted). Numerical failures and fewer than
    ``APMV_MIN_ROWS`` rows return status "not available" (no fallback).
    """
    pmv = np.asarray(pmv, dtype=float)
    expected = apmv_calibration_rows(pmv, train_idx, test_idx)
    if calibration_idx is None:
        rows = expected
    else:
        rows = _index_array(calibration_idx)
        if np.isin(rows, _index_array(test_idx)).any():
            raise CalibrationLeakageError("a test row was offered to the aPMV fit")
        if not np.array_equal(np.sort(rows), np.sort(expected)):
            raise CalibrationLeakageError(
                "calibration rows must be exactly train_idx ∩ PMV mask"
            )
    digest = hashlib.sha256(np.sort(rows).astype("<i8").tobytes()).hexdigest()
    n = int(rows.size)

    def not_available(reason: str) -> APMVCalibration:
        return APMVCalibration(NOT_AVAILABLE, None, None, n, None, reason, digest)

    if n < APMV_MIN_ROWS:
        return not_available(f"{n} calibration rows < {APMV_MIN_ROWS}")
    y = np.asarray(tsv7, dtype=float)[rows]
    if not np.all(np.isfinite(y)):
        raise ValueError("TSV-7 label missing on a calibration row")
    p = pmv[rows]
    try:
        lam, sse = _fit_lambda(p, y)
    except Exception as exc:  # failure policy: never a fallback coefficient
        return not_available(f"fit failed: {type(exc).__name__}: {exc}")
    if not np.isfinite(sse):
        return not_available("non-finite SSE")
    return APMVCalibration("ok", lam, sse, n, abs(lam) == APMV_LAMBDA_BOUND,
                           None, digest)


# ---------------------------------------------------------------------------
# Metrics (shared label contract)
# ---------------------------------------------------------------------------
METRIC_FIELDS = ("n", "n_correct", "accuracy", "balanced_accuracy",
                 "balanced_accuracy_single_class", "macro_f1", "qwk", "qwk_undefined")


def classification_metrics(y_true, y_pred, target: str) -> dict:
    """Scalar metrics from :func:`revision_stats.classification_metrics`.

    Both label vectors are validated there (finite, integral, inside the target
    ontology; NaN or out-of-ontology labels raise, nothing is cast silently).
    An empty evaluated set (e.g. no vote outside the mask) is reported
    explicitly: n = 0 and undefined metrics (NaN), never 0 or 1.
    """
    labels = ONTOLOGY[target]
    if len(y_true) == 0 or len(y_pred) == 0:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred differ in length")
        return {"n": 0, "n_correct": 0, "accuracy": np.nan, "balanced_accuracy": np.nan,
                "balanced_accuracy_single_class": False, "macro_f1": np.nan,
                "qwk": np.nan, "qwk_undefined": True,
                "confusion_matrix": json.dumps([[0] * len(labels)] * len(labels))}
    m = rs.classification_metrics(y_true, y_pred, target)
    out = {k: m[k] for k in METRIC_FIELDS}
    out["confusion_matrix"] = json.dumps(m["confusion_matrix"])  # full ontology
    return out


# ---------------------------------------------------------------------------
# E7 runner
# ---------------------------------------------------------------------------
def load_split(splits_dir, target: str, consumed_sha256: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The archived split of ``target``; ``consumed_sha256[target]`` receives the SHA-256 of the bytes parsed."""
    path = Path(splits_dir) / f"{target}_split_indices.json"
    raw, _ = read_consumed(path, consumed_sha256, target)          # one read (config_loader.read_consumed)
    d = json.loads(raw)
    if d.get("target", target) != target:
        raise ValueError(f"{path} is for target {d.get('target')!r}")
    return _index_array(d["train_idx"]), _index_array(d["test_idx"])


def check_archived_split(train_idx, test_idx, y, target: str) -> None:
    """Assert the archived split of ``target`` is the frozen one.

    The indices must partition all ``len(y)`` cohort rows (no overlap, no
    omission, no out-of-range row) and equal, **in order**, the re-draw
    ``StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`` on
    the target column. Any omitted, substituted or reordered index raises, so
    the aPMV calibration rows cannot silently change under ``archived_80_20``.
    """
    train_idx, test_idx = _index_array(train_idx), _index_array(test_idx)
    y = _as_labels(np.asarray(y), ONTOLOGY[target], f"{target} labels")
    n = len(y)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"{target}: train/test overlap")
    all_idx = np.concatenate([train_idx, test_idx])
    if (all_idx.size != n or (all_idx.size and (all_idx.min() < 0 or all_idx.max() >= n))
            or np.unique(all_idx).size != n):
        raise ValueError(f"{target}: split indices do not partition the {n} cohort rows")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_TEST_SIZE,
                                 random_state=SPLIT_SEED)
    redraw_train, redraw_test = next(sss.split(np.zeros(n), y))
    if not (np.array_equal(redraw_train, train_idx)
            and np.array_equal(redraw_test, test_idx)):
        raise ValueError(f"{target}: archived indices differ from the stratified re-draw")


def _subject_key(groups: pd.DataFrame) -> pd.Series:
    sid = groups["subject_id"]
    key = (groups["publication"].astype(str) + "|"
           + groups["building_id"].astype(str) + "|" + sid.astype(str))
    return key.where(sid.notna(), pd.NA)


def _prediction_rows(data, groups, target, method, rows, y_true, y_pred,
                     pmv, index_values) -> pd.DataFrame:
    n = len(rows)
    df = pd.DataFrame({
        "experiment": EXPERIMENT, "cohort": COHORT, "target": target,
        "feature_set": FEATURE_SET, "model": method,
        "seed": pd.array([pd.NA] * n, dtype="Int64"), "split_id": SPLIT_ID,
        "row_idx": rows,
    })
    for col in ("record_id", "publication", "building_id"):
        df[col] = (groups[col].to_numpy()[rows] if groups is not None
                   else pd.array([pd.NA] * n, dtype="object"))
    df["subject_key"] = (_subject_key(groups).to_numpy()[rows]
                         if groups is not None else pd.NA)
    df["date"] = data["Date"].to_numpy()[rows] if "Date" in data else pd.NA
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    for c in ONTOLOGY[target]:
        df[f"p_{c}"] = np.nan  # an index gives no probabilities
    df["oov_flag"] = False
    df["oov_vars"] = ""
    df["pmv"] = pmv
    df["pmv_valid"] = np.isfinite(pmv)
    df["index_value"] = index_values
    return df


def _metric_row(target, variant, method, subset, coverage, metrics, **extra):
    return {"target": target, "pmv_variant": variant, "method": method,
            "subset": subset, "coverage": coverage, **metrics, **extra}


def _ml_groups(ml_predictions: pd.DataFrame, target: str):
    sub = ml_predictions[ml_predictions["target"] == target]
    keys = [k for k in ("experiment", "model", "seed", "feature_set") if k in sub]
    for key, g in sub.groupby(keys, dropna=False, sort=True):
        yield dict(zip(keys, key if isinstance(key, tuple) else (key,))), g


def _model_seed(key: Mapping[str, object]) -> str:
    """``model:seed`` label of an ML group (``NA`` for a missing seed)."""
    seed = key.get("seed", None)
    if seed is None or pd.isna(seed):
        seed = "NA"
    elif float(seed).is_integer():
        seed = str(int(float(seed)))
    return f"{key['model']}:{seed}"


def check_ml_inventory(ml_predictions: pd.DataFrame, targets: Iterable[str],
                       expected_ml: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    """Completeness of the ML comparisons of E7 before any computation.

    Raises ``ValueError`` if a prediction targets a target that is not
    evaluated, if an evaluated target has no ML group, if the evaluated
    targets do not carry the same ML groups (``experiment``, ``model``,
    ``seed``, ``feature_set``), or, when ``expected_ml`` (``"model:seed"``
    labels, ``NA`` for a missing seed) is given, if any target's
    ``model:seed`` set differs from it. Returns ``{target: sorted labels}``.
    """
    targets = list(targets)
    present = set(ml_predictions["target"].astype(str))
    foreign = sorted(present - set(targets))
    if foreign:
        raise ValueError(f"ML predictions for targets that are not evaluated: {foreign}")
    groups = {t: [k for k, _ in _ml_groups(ml_predictions, t)] for t in targets}
    missing = [t for t in targets if not groups[t]]
    if missing:
        raise ValueError(f"no ML predictions for target(s) {missing}: every evaluated target "
                         "needs its ML comparison")
    canon = {t: sorted(repr(tuple(None if pd.isna(v) else v for v in k.values())) for k in g)
             for t, g in groups.items()}
    if len({tuple(v) for v in canon.values()}) > 1:
        raise ValueError(f"ML prediction groups differ between targets: {canon}")
    labels = {t: sorted(_model_seed(k) for k in g) for t, g in groups.items()}
    if expected_ml is not None:
        expected = sorted(expected_ml)
        for t, got in labels.items():
            if got != expected:
                raise ValueError(f"{t}: ML inventory {got} differs from the expected {expected}")
    return labels


def run_e7(data: pd.DataFrame, splits: Mapping[str, Tuple[np.ndarray, np.ndarray]],
           ml_predictions: Optional[pd.DataFrame] = None,
           groups: Optional[pd.DataFrame] = None,
           variants: Iterable[str] = VARIANTS,
           expected_ml: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """PMV/aPMV on the PMV mask, ML on the same mask and outside it.

    ``data`` keeps a RangeIndex (row positions of the split JSON files).
    When ``ml_predictions`` is given, its inventory is checked first
    (:func:`check_ml_inventory`, with ``expected_ml`` if given).
    Returns predictions, metrics, coverage, calibration and provenance.
    """
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError("data must keep a RangeIndex (canonical row positions)")
    if groups is not None and len(groups) != len(data):
        raise ValueError("group identifiers are not aligned with the data")
    if ml_predictions is not None:
        check_ml_inventory(ml_predictions, splits.keys(), expected_ml)
    elif expected_ml is not None:
        raise ValueError("expected_ml given without ml_predictions")
    provenance = check_pythermalcomfort_pin()
    variants = list(variants)
    pmv = {v: compute_pmv(data, v) for v in variants}
    tair = _column(data, "Tair")
    preds, metrics, coverage, calibration = [], [], [], {}

    for target, (train_idx, test_idx) in splits.items():
        train_idx, test_idx = _index_array(train_idx), _index_array(test_idx)
        ontology = ONTOLOGY[target]
        y_all = _as_labels(pd.to_numeric(data[target], errors="raise").to_numpy(),
                           ontology, f"{target} labels")
        # Before any calibration: the split is exactly the frozen one.
        check_archived_split(train_idx, test_idx, y_all, target)
        y_true = y_all[test_idx]

        ml_sets = []
        if ml_predictions is not None:
            for key, g in _ml_groups(ml_predictions, target):
                if "y_true" not in g:
                    raise ValueError(f"{target} {key}: ML predictions lack y_true "
                                     "(required to verify row alignment)")
                rows = _index_array(g["row_idx"].to_numpy())
                if rows.size != test_idx.size or set(rows.tolist()) != set(test_idx.tolist()):
                    raise ValueError(f"{target} {key}: ML rows differ from test_idx")
                g = g.set_index(pd.Index(rows, name="row_idx")).loc[test_idx]
                what = f"{target} {key}"
                if not np.array_equal(_as_labels(g["y_true"].to_numpy(), ontology,
                                                 f"{what} y_true"), y_true):
                    raise ValueError(f"{what}: ML y_true differs from the data")
                ml_sets.append((key, _as_labels(g["y_pred"].to_numpy(), ontology,
                                                f"{what} y_pred")))

        for variant in variants:
            p_test = pmv[variant][test_idx]
            mask = np.isfinite(p_test)
            hot = tair[test_idx] > INPUT_BOUNDS["Tair"][1]
            if (mask & hot).any():
                raise AssertionError("a vote with Tair > 30 °C is inside the PMV mask")
            cov = float(mask.mean())
            preds_ok = input_predicates(model_inputs(data.iloc[test_idx], variant))
            coverage.append({
                "target": target, "pmv_variant": variant, "n_test": int(test_idx.size),
                "n_valid": int(mask.sum()), "coverage": cov,
                "n_tair_gt_30": int(hot.sum()),
                **{f"n_fail_{c}": int(v) for c, v in (~preds_ok).sum().items()},
                "n_inputs_pass_pmv_out_of_range": int(
                    (preds_ok.all(axis=1).to_numpy() & ~mask).sum()),
            })

            methods = [(METHOD_NAMES[variant], p_test, pmv_to_class(p_test, target))]
            if variant == "primary":
                cal = fit_apmv_lambda(pmv[variant], data[TSV7_COLUMN].to_numpy(),
                                      train_idx, test_idx)
                calibration[target] = asdict(cal)
                if cal.available:
                    a_test = apmv(p_test, cal.lam)
                    if not np.array_equal(np.isfinite(a_test), mask):
                        raise AssertionError("aPMV mask differs from the PMV mask")
                    methods.append(("aPMV", a_test, apmv_to_class(a_test, target)))
                else:
                    metrics.append(_metric_row(target, variant, "aPMV", "on_mask", cov,
                                               {"n": int(mask.sum())}, status=NOT_AVAILABLE))

            for method, values, cls in methods:
                if not np.array_equal(~cls.isna(), mask):
                    raise AssertionError(f"{method}: classes outside the PMV mask")
                preds.append(_prediction_rows(data, groups, target, method, test_idx,
                                              y_true, cls, p_test, values))
                m = classification_metrics(y_true[mask], cls[mask].to_numpy(dtype=int), target)
                metrics.append(_metric_row(target, variant, method, "on_mask", cov, m, status="ok"))

            for key, y_ml in ml_sets:
                label = "ML:" + "/".join(f"{k}={v}" for k, v in key.items())
                for subset, sel in (("on_mask", mask), ("outside_mask", ~mask)):
                    m = classification_metrics(y_true[sel], y_ml[sel], target)
                    metrics.append(_metric_row(target, variant, label, subset, cov, m,
                                               status="ok", **key))

    return {
        "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "metrics": pd.DataFrame(metrics),
        "coverage": pd.DataFrame(coverage),
        "calibration": calibration,
        "provenance": provenance,
    }


# ===========================================================================
# E7b — SET, PET (PTS_SET, PTS_PET) and ePMV
# ===========================================================================
# Functions of the pre-revision ``indicateurs_classiques.py`` restored with the
# index-computation fixes, labelled S-1 ... S-9 below.
# SET, PET and PMV are heat-balance models; ePMV an adaptive heat-balance model.
E7B_EXPERIMENT = "E7b"
E7B_TARGETS = ["thermal_sensation", "TSV_3p"]      # as in the paper: no TPV
E7B_METHODS = ("PTS_SET", "PTS_PET", "ePMV")
#: pre-revision column names of the row-wise SET / PET calls (pre-revision ``rename_map``)
rename_map = {
    "Tair": "ta",
    "RH": "rh",
    "Met": "met",
}
# S-6: the pre-revision PTS constants (uncited PTS_PET_A, PTS_PET_B = -1.1704, 0.0494; PTS_SET
# constants None) and its constants-or-fit branch are removed, with the unused quadratic option
# (PTS_USE_QUADRATIC = False in the pre-revision code): PTS_SET and PTS_PET both take the pre-revision
# code's own linear fit (``fit_pts_coeffs``), the linear transform of the paper.
#: ePMV expectancy factor: the Fanger & Toftum (2002) range. e is applied UNBOUNDED and only flagged when it lies
#: outside this range (``e_flag_outside_range``); the range is never used to clip e
EPMV_E_FLAG_RANGE = (0.5, 1.0)
#: PTS saturation of the E7b adapter (S-8): the pre-revision ``round_clip7`` clip, before the class conversion
PTS_SATURATION = 3.5
#: scipy.optimize.fsolve status 2-5 messages (SciPy 1.17.1, ``_minpack_py._root_hybr``): non-convergence (S-4)
FSOLVE_NONCONVERGENCE = ("The number of calls to function has reached maxfev",
                         "xtol=",
                         "The iteration is not making good progress")
# SHA-256 of the pythermalcomfort 4.0.2 sources E7b calls (S-9), besides pmv_ppd_iso.py (E7 pin)
PINNED_E7B_SOURCES_SHA256 = {
    "models/set_tmp.py": "b53f8a514d3af191d0c5cac35aa6a577cd9d7de4f17a14fdc2a979f119c47488",
    "models/two_nodes_gagge.py": "f01dadd64a7199dde55d4afc1ded405cb819c1a39f798453cb6af19d76928379",
    "models/pet_steady.py": "dda6eb3d9cb61fdb95a500ed4a28623a02139eee277be1d719edcea014a7e034",
    "models/pmv_e.py": "3afaa439064379b3f220bd7e77a6ef0e69140467e5a01a465cbc65d1bfa0db4d",
    "utilities.py": "fd9ea09f7695e21c407675a6d043342fad4bfdeb2788bb5b59f706e534605612",
}


def e7b_provenance() -> dict:
    """E7 provenance plus the SHA-256 of the SET / PET / ePMV sources and the SciPy version (S-9)."""
    import scipy
    prov = pythermalcomfort_provenance()
    pkg_dir = Path(pythermalcomfort.__file__).parent
    prov["e7b_sources_sha256"] = {k: _sha256_file(pkg_dir / k) for k in PINNED_E7B_SOURCES_SHA256}
    prov["scipy_version"] = scipy.__version__
    prov["fsolve_nonconvergence_messages"] = list(FSOLVE_NONCONVERGENCE)
    return prov


def check_e7b_pin(provenance: Optional[dict] = None) -> dict:
    """The E7 pin (``check_pythermalcomfort_pin``) and the E7b source hashes (S-9); raises otherwise."""
    prov = e7b_provenance() if provenance is None else provenance
    check_pythermalcomfort_pin(prov)
    bad = [f"{k}: {prov['e7b_sources_sha256'].get(k)} != {v}" for k, v in PINNED_E7B_SOURCES_SHA256.items()
           if prov["e7b_sources_sha256"].get(k) != v]
    if bad:
        raise PythermalcomfortPinError("Refusing to compute SET/PET/ePMV (S-9): " + "; ".join(bad))
    return prov


class ResultFieldError(ValueError):
    """A library result without a parsable ``key`` field (counted as an exception of the row)."""


def _as_float_field(res, key):
    # a missing or unparsable result field raises (counted in the row's exception
    # log by ``_safe_set_row`` / ``_safe_pet_row``) instead of silently returning NaN
    if res is None:
        raise ResultFieldError(f"no result (field {key!r})")
    try:
        if isinstance(res, dict) and key in res:
            return float(res[key])
        if hasattr(res, key):
            return float(getattr(res, key))
        return float(res)
    except (TypeError, ValueError) as exc:
        raise ResultFieldError(f"result field {key!r} missing or not a number: {exc}") from exc


def _set_tmp_compat_scalar(tdb, tr, vel, rh, met, clo, wme=0.0):
    # S-2: pythermalcomfort 4.0.2 set_tmp takes v= (no vr=); no TypeError retry
    res = set_tmp(tdb=tdb, tr=tr, v=vel, rh=rh, met=met, clo=clo, wme=wme)
    return _as_float_field(res, "set")


def _is_nonconvergence(w) -> bool:
    return issubclass(w.category, RuntimeWarning) and str(w.message).startswith(FSOLVE_NONCONVERGENCE)


def _safe_set_row(rw, log: dict, i: int):
    # S-3: one call on the observed inputs (no perturbed-speed retry); an exception or a
    # non-finite value makes the row invalid; exceptions and warnings are flagged on row ``i``
    # of ``log`` (row-level diagnostics, aggregated per scope by ``summarise_log``)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v = _set_tmp_compat_scalar(rw.ta, rw.tr, rw.vel, rw.rh, rw.met, rw.clo, getattr(rw, "wme", 0.0))
        except Exception as exc:
            log["exception"][i] = type(exc).__name__
            v = np.nan
    log["warnings"][i] = tuple(w.category.__name__ for w in caught)
    return v if np.isfinite(v) else np.nan


def _safe_pet_row(rw, log: dict, i: int):
    # S-3 / S-4(b): one call; invalid if it raises, returns a non-finite value or if fsolve
    # did not converge (RuntimeWarning with a status 2-5 message, flagged on row ``i``)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = pet_steady(tdb=rw.ta, tr=rw.tr, v=rw.vel, rh=rw.rh, met=rw.met, clo=rw.clo)
            pet = _as_float_field(res, "pet")
        except Exception as exc:
            log["exception"][i] = type(exc).__name__
            pet = np.nan
    nonconv = False
    other = []
    for w in caught:
        if _is_nonconvergence(w):
            nonconv = True
        else:
            other.append(w.category.__name__)
    log["warnings"][i] = tuple(other)
    if nonconv:
        log["nonconvergence"][i] = True
        return np.nan
    return pet if np.isfinite(pet) else np.nan


def _new_log(n: int) -> dict:
    """
    Row-level diagnostics of one index over ``n`` cohort rows: exception
    class name (``""`` = none), fsolve non-convergence, other warning categories,
    non-finite input; ``call_warnings`` = warnings of a vectorised call (``pmv_e``),
    not attributable to a row.
    """
    return {"exception": np.full(n, "", dtype=object), "nonconvergence": np.zeros(n, dtype=bool),
            "warnings": [()] * n, "nonfinite_input": np.zeros(n, dtype=bool), "call_warnings": {}}


def summarise_log(log: dict, rows=None) -> dict:
    """Counts of the row-level diagnostics over ``rows`` (``None`` = every cohort row)."""
    idx = np.arange(len(log["exception"])) if rows is None else _index_array(rows)
    exc: Dict[str, int] = {}
    for name in log["exception"][idx]:
        if name:
            exc[name] = exc.get(name, 0) + 1
    warn: Dict[str, int] = {}
    for i in idx:
        for name in log["warnings"][i]:
            warn[name] = warn.get(name, 0) + 1
    return {"n_rows": int(idx.size), "n_nonfinite_inputs": int(log["nonfinite_input"][idx].sum()),
            "nonconvergence": int(log["nonconvergence"][idx].sum()), "exceptions": exc, "warnings": warn}


def snapshot_rows(data: pd.DataFrame) -> pd.DataFrame:
    """S-1: the six E7 inputs (``model_inputs``: missing column raises, no fill, no clip, no ``tsv``),
    pre-revision column names, ``wme`` = 0."""
    inp = model_inputs(data).rename(columns=rename_map)
    inp["wme"] = 0.0
    return inp


def finite_input_rows(inp: pd.DataFrame) -> np.ndarray:
    """S-4(a): a row with a non-finite input among the six is never sent to the library."""
    return np.isfinite(inp[["ta", "tr", "vel", "rh", "met", "clo"]].to_numpy(dtype=float)).all(axis=1)


def compute_set_pet(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    """SET and PET per row (pre-revision ``compute_block`` SET / PET lines, fixed S-1 ... S-4); NaN = invalid."""
    inp = snapshot_rows(data)
    ok = finite_input_rows(inp)
    logs = {"SET": _new_log(len(inp)), "PET": _new_log(len(inp))}
    set_v = np.full(len(inp), np.nan)
    pet_v = np.full(len(inp), np.nan)
    for i, rw in enumerate(inp.itertuples(index=False)):
        if not ok[i]:
            continue
        set_v[i] = _safe_set_row(rw, logs["SET"], i)
        pet_v[i] = _safe_pet_row(rw, logs["PET"], i)
    for lg in logs.values():
        lg["nonfinite_input"] = ~ok
    return set_v, pet_v, logs


def compute_pmv_e(data: pd.DataFrame, e_coefficient: float) -> Tuple[np.ndarray, dict]:
    """
    ``pmv_e`` of the pre-revision ``add_apmv_epmv`` (e-part) on the finite-input rows
    (S-1, S-4(a)): ``pmv_e(tdb, tr, vr=vel, rh, met, clo, e_coefficient, wme=0)``,
    library defaults (ISO 7730-2005 passes, ``limit_inputs=True``, output rounded
    to 0.01 by the library). NaN outside the library validity mask.
    """
    inp = snapshot_rows(data)
    ok = finite_input_rows(inp)
    log = _new_log(len(inp))
    log["nonfinite_input"] = ~ok
    out = np.full(len(inp), np.nan)
    if ok.any():
        r = inp[ok]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = pmv_e(tdb=r["ta"].to_numpy(), tr=r["tr"].to_numpy(), vr=r["vel"].to_numpy(),
                        rh=r["rh"].to_numpy(), met=r["met"].to_numpy(), clo=r["clo"].to_numpy(),
                        e_coefficient=float(e_coefficient), wme=0)
        for w in caught:
            log["call_warnings"][w.category.__name__] = log["call_warnings"].get(w.category.__name__, 0) + 1
        out[ok] = np.asarray(res.e_pmv, dtype=float).reshape(-1)
    return out, log


def fit_pts_coeffs(x_series, tsv_series):
    x = np.asarray(x_series, dtype=float)
    y = np.asarray(tsv_series, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    # S-5: the pre-revision fallback (x.size < 5 -> 0.0, 0.2) is replaced by "not available" (None)
    if x.size < APMV_MIN_ROWS:
        return None
    # S-6: linear only (the unused pre-revision quadratic option removed)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _apply_pts(x, a, b):
    """PTS = a + b·x of the pre-revision ``compute_block``."""
    return a + b * np.asarray(x, dtype=float)


def _calibration_rows(index_values, train_idx, test_idx, calibration_idx=None) -> np.ndarray:
    """train_idx ∩ index mask (S-5); a test row or other rows offered raise ``CalibrationLeakageError``."""
    expected = apmv_calibration_rows(index_values, train_idx, test_idx)
    if calibration_idx is None:
        return expected
    rows = _index_array(calibration_idx)
    if np.isin(rows, _index_array(test_idx)).any():
        raise CalibrationLeakageError("a test row was offered to the calibration")
    if not np.array_equal(np.sort(rows), np.sort(expected)):
        raise CalibrationLeakageError("calibration rows must be exactly train_idx ∩ index mask")
    return rows


def _digest(rows) -> str:
    return hashlib.sha256(np.sort(np.asarray(rows)).astype("<i8").tobytes()).hexdigest()


def fit_pts_calibration(index_values, tsv7, train_idx, test_idx, calibration_idx=None) -> dict:
    """
    PTS = a + b·x fitted (``fit_pts_coeffs``, linear) on train_idx ∩ index mask
    of the evaluated target, response = TSV-7 label (S-5); "not available"
    below ``APMV_MIN_ROWS`` rows or with constant labels / constant index.
    """
    x = np.asarray(index_values, dtype=float)
    rows = _calibration_rows(x, train_idx, test_idx, calibration_idx)
    n = int(rows.size)
    out = {"status": NOT_AVAILABLE, "a": None, "b": None, "sse": None, "n_calibration": n,
           "reason": None, "calibration_idx_sha256": _digest(rows)}
    y = np.asarray(tsv7, dtype=float)[rows]
    if n and not np.all(np.isfinite(y)):
        raise ValueError("TSV-7 label missing on a calibration row")
    if n < APMV_MIN_ROWS:
        return {**out, "reason": f"{n} calibration rows < {APMV_MIN_ROWS}"}
    if np.unique(y).size < 2:
        return {**out, "reason": "constant TSV-7 labels on the calibration rows"}
    if np.unique(x[rows]).size < 2:
        return {**out, "reason": "constant index on the calibration rows"}
    coeffs = fit_pts_coeffs(x[rows], y)
    if coeffs is None:
        return {**out, "reason": "fit_pts_coeffs: too few rows"}
    a, b = coeffs
    r = y - _apply_pts(x[rows], a, b)
    return {**out, "status": "ok", "a": a, "b": b, "sse": float(np.sum(r * r))}


def fit_epmv_e(x_e1, tsv7, train_idx, test_idx, calibration_idx=None) -> dict:
    """
    ePMV expectancy factor (S-7): x = ``pmv_e(..., e_coefficient=1)``
    on train_idx ∩ ePMV mask, y = TSV-7 label; ``e = Σxy / Σx²`` (pre-revision
    closed form, least squares through the origin), applied **unbounded**
    (no bounded fit). ``e_flag_outside_range``
    is True when e lies outside ``EPMV_E_FLAG_RANGE`` = [0.5, 1.0] (Fanger &
    Toftum range); the flag never changes e and never changes the status.
    Σx² = 0 or too few rows: "not available" (the pre-revision fallback 1.0 is
    removed).
    """
    x_all = np.asarray(x_e1, dtype=float)
    rows = _calibration_rows(x_all, train_idx, test_idx, calibration_idx)
    n = int(rows.size)
    out = {"status": NOT_AVAILABLE, "e": None, "e_flag_outside_range": None, "n_calibration": n,
           "sse_calibration": None, "sse_rounded_output": None,
           "e_flag_range": list(EPMV_E_FLAG_RANGE), "reason": None, "calibration_idx_sha256": _digest(rows)}
    y = np.asarray(tsv7, dtype=float)[rows]
    if n and not np.all(np.isfinite(y)):
        raise ValueError("TSV-7 label missing on a calibration row")
    if n < APMV_MIN_ROWS:
        return {**out, "reason": f"{n} calibration rows < {APMV_MIN_ROWS}"}
    x = x_all[rows]
    denom = float(np.sum(x * x))
    if not denom > 0:
        return {**out, "reason": "sum of squared PMV is 0"}
    e = float(np.sum(y * x) / denom)
    lo, hi = EPMV_E_FLAG_RANGE
    r = y - e * x
    rr_ = y - np.around(e * x, 2)
    return {**out, "status": "ok", "e": e, "e_flag_outside_range": bool(not lo <= e <= hi),
            "sse_calibration": float(np.sum(r * r)), "sse_rounded_output": float(np.sum(rr_ * rr_))}


def pts_to_class(values, target: str, tsv7_range: Tuple[int, int] = (-3, 3)):
    """
    E7b adapter for PTS_SET, PTS_PET and ePMV (S-8; ePMV since e is unbounded): on the finite values,
    x ≥ 3.5 → 3 and x < −3.5 → −3 (the pre-revision ``round_clip7`` clip, before the class
    conversion), NaN kept; then the E7 rule ``index_to_class`` (NaN → <NA>,
    never a class). Returns ``(classes, n_saturated_low, n_saturated_high)``.
    """
    if target not in E7B_TARGETS:
        raise ValueError(f"E7b evaluates {E7B_TARGETS} only, not {target!r}")
    x = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(x)
    high = finite & (x >= PTS_SATURATION)
    low = finite & (x < -PTS_SATURATION)
    x[high] = 3.0
    x[low] = -3.0
    return index_to_class(x, target, tsv7_range), int(low.sum()), int(high.sum())


def epmv_to_class(epmv_values, target: str):
    """
    ePMV → class: with e unbounded, ePMV = e·PMV can lie beyond the
    TSV-7 scale (e > 1), so ePMV follows the rule of the other restored indices
    (S-8, ``pts_to_class``): finite values ≥ 3.5 → 3 and < −3.5 → −3, counted;
    then the E7 rule on [−3, 3]; NaN never becomes a class. Returns
    ``(classes, n_saturated_low, n_saturated_high)``.
    """
    return pts_to_class(epmv_values, target)


def _log_counts(prefix: str, log: dict, rows=None) -> dict:
    """Coverage columns of ``summarise_log`` over ``rows``, prefixed (``cohort_`` / ``test_``)."""
    sm = summarise_log(log, rows)
    return {f"{prefix}n_nonfinite_inputs": sm["n_nonfinite_inputs"],
            f"{prefix}n_nonconvergence": sm["nonconvergence"],
            f"{prefix}exceptions": json.dumps(sm["exceptions"], sort_keys=True),
            f"{prefix}other_warnings": json.dumps(sm["warnings"], sort_keys=True)}


def _class_distribution(cls) -> str:
    """Predicted-class counts (JSON, class -> n) of the valid rows."""
    v = pd.Series(cls).dropna().astype(int).value_counts().sort_index()
    return json.dumps({str(k): int(n) for k, n in v.items()})


def run_e7b(data: pd.DataFrame, splits: Mapping[str, Tuple[np.ndarray, np.ndarray]],
            ml_predictions: Optional[pd.DataFrame] = None, groups: Optional[pd.DataFrame] = None,
            expected_ml: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """
    PTS_SET, PTS_PET and ePMV on their masks (TSV-7, TSV-3), the ML predictions
    on the same mask and outside it (the E7 comparison). Calibrations per
    evaluated target on its archived training rows inside the index mask.
    """
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError("data must keep a RangeIndex (canonical row positions)")
    if sorted(splits) != sorted(E7B_TARGETS):
        raise ValueError(f"E7b evaluates exactly {E7B_TARGETS}, got {sorted(splits)}")
    if groups is not None and len(groups) != len(data):
        raise ValueError("group identifiers are not aligned with the data")
    if ml_predictions is not None:
        check_ml_inventory(ml_predictions, splits.keys(), expected_ml)
    elif expected_ml is not None:
        raise ValueError("expected_ml given without ml_predictions")
    provenance = check_e7b_pin()
    set_v, pet_v, logs = compute_set_pet(data)
    x_e1, logs["ePMV"] = compute_pmv_e(data, 1.0)
    pmv_primary = compute_pmv(data, "primary")
    if (np.isfinite(x_e1) & ~np.isfinite(pmv_primary)).any():          # ePMV mask within the PMV mask
        raise AssertionError("the ePMV mask is not a subset of the E7 PMV mask")
    tsv7 = data[TSV7_COLUMN].to_numpy()
    preds, metrics, coverage, calibration = [], [], [], {}

    for target, (train_idx, test_idx) in splits.items():
        train_idx, test_idx = _index_array(train_idx), _index_array(test_idx)
        ontology = ONTOLOGY[target]
        y_all = _as_labels(pd.to_numeric(data[target], errors="raise").to_numpy(), ontology, f"{target} labels")
        check_archived_split(train_idx, test_idx, y_all, target)
        y_true = y_all[test_idx]
        ml_sets = []
        if ml_predictions is not None:
            for key, g in _ml_groups(ml_predictions, target):
                rows = _index_array(g["row_idx"].to_numpy())
                if rows.size != test_idx.size or set(rows.tolist()) != set(test_idx.tolist()):
                    raise ValueError(f"{target} {key}: ML rows differ from test_idx")
                g = g.set_index(pd.Index(rows, name="row_idx")).loc[test_idx]
                what = f"{target} {key}"
                if not np.array_equal(_as_labels(g["y_true"].to_numpy(), ontology, f"{what} y_true"), y_true):
                    raise ValueError(f"{what}: ML y_true differs from the data")
                ml_sets.append((key, _as_labels(g["y_pred"].to_numpy(), ontology, f"{what} y_pred")))

        calibration[target] = {}
        for method, raw, log in (("PTS_SET", set_v, logs["SET"]), ("PTS_PET", pet_v, logs["PET"]),
                                 ("ePMV", x_e1, logs["ePMV"])):
            if method == "ePMV":
                cal = fit_epmv_e(raw, tsv7, train_idx, test_idx)
                values = compute_pmv_e(data, cal["e"])[0] if cal["status"] == "ok" else None
            else:
                cal = fit_pts_calibration(raw, tsv7, train_idx, test_idx)
                values = _apply_pts(raw, cal["a"], cal["b"]) if cal["status"] == "ok" else None
            calibration[target][method] = cal
            raw_test = raw[test_idx]
            cov_row = {"target": target, "method": method, "n_rows": int(len(raw)),
                       "n_valid_rows": int(np.isfinite(raw).sum()), "n_test": int(test_idx.size),
                       "n_valid_test": int(np.isfinite(raw_test).sum()),
                       "calibration": cal["status"], **_log_counts("cohort_", log),
                       **_log_counts("test_", log, test_idx),
                       "call_warnings": json.dumps(log["call_warnings"], sort_keys=True)}
            if values is None:
                coverage.append({**cov_row, "coverage": float(np.isfinite(raw_test).mean()) if raw_test.size else np.nan})
                metrics.append({"target": target, "method": method, "subset": "on_mask", "status": NOT_AVAILABLE,
                                "n": int(np.isfinite(raw_test).sum())})
                continue
            if method == "ePMV":
                if not np.array_equal(np.isfinite(values), np.isfinite(raw)):
                    raise AssertionError("ePMV mask differs from the pmv_e(e=1) mask")
                cls_all, n_low_all, n_high_all = epmv_to_class(values, target)
                cls, n_low, n_high = epmv_to_class(values[test_idx], target)
            else:
                cls_all, n_low_all, n_high_all = pts_to_class(values, target)
                cls, n_low, n_high = pts_to_class(values[test_idx], target)
            mask = np.isfinite(values[test_idx])
            if not np.array_equal(~cls.isna(), mask):
                raise AssertionError(f"{method}: classes outside the index mask")
            cov = float(mask.mean()) if mask.size else np.nan
            coverage.append({**cov_row, "coverage": cov, "n_saturated_low": n_low_all,
                             "n_saturated_high": n_high_all, "n_saturated_low_test": n_low,
                             "n_saturated_high_test": n_high})
            fr = _prediction_rows(data, groups, target, method, test_idx, y_true, cls, raw_test, values[test_idx])
            fr = fr.rename(columns={"pmv": "index_raw", "pmv_valid": "index_valid"})
            fr["index_valid"] = mask
            fr["experiment"] = E7B_EXPERIMENT
            preds.append(fr)
            m = classification_metrics(y_true[mask], cls[mask].to_numpy(dtype=int), target)
            # transform and output beside the metrics: a, b (PTS) or e (ePMV), the range
            # of the index value on the evaluated rows and the predicted-class distribution
            v_test = values[test_idx][mask]
            metrics.append({"target": target, "method": method, "subset": "on_mask", "coverage": cov,
                            **m, "status": "ok", "pts_a": cal.get("a"), "pts_b": cal.get("b"),
                            "epmv_e": cal.get("e"),
                            # the flag sits in its own column, never in ``status``
                            "epmv_e_flag_outside_range": cal.get("e_flag_outside_range"),
                            "index_value_min": float(v_test.min()) if v_test.size else np.nan,
                            "index_value_max": float(v_test.max()) if v_test.size else np.nan,
                            "pred_class_distribution": _class_distribution(cls[mask])})
            for key, y_ml in ml_sets:
                label = "ML:" + "/".join(f"{k}={v}" for k, v in key.items())
                for subset, sel in (("on_mask", mask), ("outside_mask", ~mask)):
                    m = classification_metrics(y_true[sel], y_ml[sel], target)
                    metrics.append({"target": target, "method": label, "index": method, "subset": subset,
                                    "coverage": cov, **m, "status": "ok", **key})
    return {
        "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "metrics": pd.DataFrame(metrics),
        "coverage": pd.DataFrame(coverage),
        "calibration": calibration,
        "provenance": provenance,
    }


def _read_ml_predictions(path: Path, consumed_sha256: Optional[dict] = None) -> pd.DataFrame:
    """Every prediction file under ``path``, each parsed from one read; ``consumed_sha256`` receives
    ``e1:{file name}`` -> SHA-256 of the bytes parsed."""
    files = sorted(set(path.rglob("*.csv.gz")) | set(path.rglob("*.csv")))
    if not files:
        raise FileNotFoundError(f"no prediction file under {path}")
    frames = []
    for f in files:
        raw, _ = read_consumed(f, consumed_sha256, f"e1:{f.name}")
        frames.append(pd.read_csv(io.BytesIO(raw), compression="gzip" if f.name.endswith(".gz") else None))
    df = pd.concat(frames, ignore_index=True)
    missing = {"target", "model", "row_idx", "y_true", "y_pred"} - set(df.columns)
    if missing:
        raise ValueError(f"ML predictions lack columns {sorted(missing)}")
    return df


def _check_empty_output_dir(path) -> None:
    """One run per output directory: an existing non-empty directory is refused (never overwritten)."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} exists and is not an empty directory: E7 outputs are never "
                              "overwritten (use a new output directory)")


def build_parser() -> argparse.ArgumentParser:
    """
    The E7 command-line parser. ``--expected_ml`` takes one or more values
    (``nargs="+"``) and ``output_dir`` is positional: a caller that builds an
    argv puts ``output_dir`` first (``revision_runners.e7_argv``), or the
    open list would take it as one more MODEL:SEED when no flag follows it
    (an argument-order bug met in an earlier E7 run).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pmv_only", action="store_true",
                    help="E7: PMV and aPMV; required unless --indices set_pet_epmv.")
    ap.add_argument("--indices", choices=["set_pet_epmv"], default=None,
                    help="E7b: SET/PTS_SET, PET/PTS_PET and ePMV on TSV-7 and TSV-3; "
                         "excludes --pmv_only.")
    ap.add_argument("--data_csv", default=DEFAULT_DATA_CSV)
    ap.add_argument("--splits", default=DEFAULT_SPLITS_DIR,
                    help="Directory with <target>_split_indices.json (archived split).")
    ap.add_argument("--ml_predictions", default=None,
                    help="Directory of E1 prediction files (CSV / CSV.gz).")
    ap.add_argument("--expected_ml", nargs="+", default=None, metavar="MODEL:SEED",
                    help="Required with --ml_predictions: the exact E1 inventory per target "
                         "(e.g. RF:42 RF:43 RF:44 SVM:42 …; NA for a missing seed).")
    ap.add_argument("--skip_group_ids", action="store_true",
                    help="Do not attach record/publication/building identifiers.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Smoke check (tiny E1 models): files and rows labelled E7_dry_run, never results.")
    ap.add_argument("output_dir")
    return ap


def main(argv=None, *, e1_manifest=None, consumed_sha256: Optional[dict] = None,
         config: Optional[dict] = None) -> int:
    """
    E7 command line. A real E7 (no ``--dry_run``) runs only through
    ``python3 run_revision.py e7 --run_dir RUN`` (``revision_runners.run_e7``),
    which validates the E1 source and passes its manifest as ``e1_manifest``
    (a real, non-dry-run E1) and writes the run manifest;
    called directly, this CLI accepts dry runs only. ``--skip_group_ids`` is
    for dry runs only, and a non-empty output directory is refused.
    ``consumed_sha256`` (optional)
    receives, by input role of the E7 manifest, the SHA-256 of the bytes parsed
    (each input is read once and parsed from memory): ``data_csv``,
    ``split:{target}``, ``raw_csv`` / ``raw_metadata_xlsx`` (group identifiers)
    and ``e1:{file name}`` (ML predictions); ``run_e7`` compares them with the
    bound inputs. ``config``: the run configuration, for the identifier file
    paths (``data_groups``; not read again from ``config.yaml``).
    """
    ap = build_parser()
    args = ap.parse_args(argv)
    e7b = args.indices == "set_pet_epmv"
    if e7b and args.pmv_only:
        ap.error("--pmv_only (E7) and --indices set_pet_epmv (E7b) are separate experiments")
    if not args.pmv_only and not e7b:
        ap.error("--pmv_only is required (E7), or --indices set_pet_epmv (E7b); aPTS and ePTS were removed")
    if args.ml_predictions and not args.expected_ml:
        ap.error("--expected_ml is required with --ml_predictions (complete ML inventory)")
    if args.skip_group_ids and not args.dry_run:
        ap.error("--skip_group_ids drops record_id / publication / building_id: dry runs only (--dry_run)")
    check_not_protected(args.output_dir)  # never write into the frozen canonical artefact
    if not args.dry_run and (e1_manifest is None or e1_manifest.get("experiment") != "E1"
                             or e1_manifest.get("dry_run") is not False):
        ap.error("a real E7 / E7b runs only through `python3 run_revision.py e7` / `e7b --run_dir RUN` (validated "
                 "real E1 source, run manifest); standalone indicateurs_classiques.py accepts --dry_run only")
    _check_empty_output_dir(args.output_dir)

    consumed: dict = {}                       # input role -> SHA-256 of the bytes parsed
    data_raw, _ = read_consumed(args.data_csv, consumed, "data_csv")
    data = pd.read_csv(io.BytesIO(data_raw))
    parsed: dict = {}
    targets = E7B_TARGETS if e7b else TARGETS
    splits = {t: load_split(args.splits, t, parsed) for t in targets}
    consumed.update({f"split:{t}": sha for t, sha in parsed.items()})
    groups = None
    if not args.skip_group_ids:
        import data_groups
        seen: dict = {}
        groups = data_groups.load_group_ids(canonical_csv=args.data_csv, raw_bytes={"canonical_csv": data_raw},
                                            consumed_sha256=seen, config=config)
        consumed.update({k: seen[k] for k in ("raw_csv", "raw_metadata_xlsx")})
    ml = _read_ml_predictions(Path(args.ml_predictions), consumed) if args.ml_predictions else None
    if consumed_sha256 is not None:
        consumed_sha256.update(consumed)

    if e7b:
        if ml is not None:                    # every E1 file is parsed and bound (as E7); TSV-7 / TSV-3 evaluated
            ml = ml[ml["target"].isin(E7B_TARGETS)].reset_index(drop=True)
        res = run_e7b(data, splits, ml_predictions=ml, groups=groups, expected_ml=args.expected_ml)
    else:
        res = run_e7(data, splits, ml_predictions=ml, groups=groups, expected_ml=args.expected_ml)

    out = Path(args.output_dir)
    experiment = (E7B_EXPERIMENT if e7b else EXPERIMENT) + ("_dry_run" if args.dry_run else "")
    if args.dry_run and len(res["predictions"]):
        res["predictions"]["experiment"] = experiment
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    for (target, model), g in res["predictions"].groupby(["target", "model"], sort=False):
        # E7b: fixed gzip mtime (byte-reproducible files); E7 unchanged
        g.to_csv(out / "predictions" / f"{experiment}__{COHORT}__{target}__{model}.csv.gz",
                 index=False, **({"compression": {"method": "gzip", "mtime": 0}} if e7b else {}))
    res["metrics"].to_csv(out / "metrics.csv", index=False)
    res["coverage"].to_csv(out / "coverage.csv", index=False)
    (out / ("calibration.json" if e7b else "apmv_calibration.json")).write_text(
        json.dumps(res["calibration"], indent=2))
    prov = dict(res["provenance"])
    prov.update({
        "experiment": experiment, "dry_run": bool(args.dry_run),
        "command": [sys.executable, Path(__file__).name]
                   + (list(argv) if argv is not None else sys.argv[1:]),
        "data_csv": args.data_csv, "data_csv_sha256": consumed["data_csv"],      # bytes parsed
        "splits": {t: consumed[f"split:{t}"] for t in targets},
    })
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(res["coverage"].to_string(index=False))
    print(f"[OK] {'E7b' if e7b else 'E7'} outputs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
