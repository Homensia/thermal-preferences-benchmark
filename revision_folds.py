"""
Shared split, cross-validation folds and model seeds of the revision rerun.

The canonical run drew its folds inside each family (classical models:
``StratifiedKFold(random_state=42)``; reported deep-model folds:
``random_state=RANDOM_SEED+1``) and paired them by fold number in a Wilcoxon
test. In the revision:

* the archived 80/20 split of each target is **loaded** and asserted equal,
  in order, to the ``StratifiedShuffleSplit(test_size=0.2, random_state=42)``
  re-draw; a re-draw alone is never trusted;
* the K = 10 folds are materialised **once per target** on the archived
  ``train_idx`` with ``StratifiedKFold(shuffle=True, random_state=42)``,
  written to ``{folds_dir}/{target}_folds.json`` (one ``fold_id`` per training
  row) before any fit, never overwritten, and passed to every family; no family
  draws its own folds;
* the fold vector used by a family is identified by ``fold_vector_sha256``,
  so identity across families is checked byte for byte;
* model seeds are the frozen ones: RF / XGB {42, 43, 44}; ANN and FTT either
  {42} or {42, 43, 44} (fixed before the run); SVM is
  deterministic for prediction and runs once (seed 42).

No Wilcoxon test over folds exists any more; fold scores are descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from config_loader import check_not_protected, read_consumed

# ---------------------------------------------------------------- frozen values
SPLIT_TEST_SIZE = 0.2          # archived 80/20 split
SPLIT_SEED = 42                # split seed
FOLDS_K = 10                   # number of CV folds
FOLDS_SEED = 42                # fold seed
TREE_SEEDS = (42, 43, 44)      # RF, XGB
DEEP_SEED_OPTIONS = ((42,), (42, 43, 44))   # ANN, FTT
SVM_SEEDS = (42,)              # deterministic for prediction, one run

FAMILY_OF_MODEL = {
    "RandomForest": "tree",
    "XGBoost": "tree",
    "SVM": "svm",
    "ANN": "deep",
    "FTTransformer": "deep",
}

FOLDS_FORMAT = "revision-2026-10/folds-v1"


class FoldIntegrityError(ValueError):
    """An archived split or a materialised fold file disagrees with the frozen settings."""


# ---------------------------------------------------------------- configuration guard
def check_cv_config(config: dict) -> None:
    """``config.yaml`` must carry the frozen fold settings (K = 10, shuffled)."""
    cv = config.get("cross_validation", {})
    if cv.get("folds") != FOLDS_K or cv.get("shuffle") is not True:
        raise FoldIntegrityError(
            f"config.yaml cross_validation = {cv!r}; the frozen protocol fixes "
            f"folds = {FOLDS_K}, shuffle = true")
    if config.get("seed") != FOLDS_SEED:
        raise FoldIntegrityError(
            f"config.yaml seed = {config.get('seed')!r}; split, folds and validation "
            f"subsets use seed {FOLDS_SEED}")


# ---------------------------------------------------------------- seeds
def check_model_seeds(model_name: str, seeds: Iterable[int]) -> Tuple[int, ...]:
    """Return the seeds as a tuple, or raise if they are not the frozen ones."""
    seeds = tuple(int(s) for s in seeds)
    family = FAMILY_OF_MODEL.get(model_name)
    if family is None:
        raise ValueError(f"unknown model family {model_name!r}")
    allowed = {"tree": (TREE_SEEDS,), "svm": (SVM_SEEDS,), "deep": DEEP_SEED_OPTIONS}[family]
    if seeds not in allowed:
        raise ValueError(
            f"{model_name}: seeds {list(seeds)} are not frozen (the frozen seeds are "
            f"{[list(a) for a in allowed]})")
    return seeds


def model_seeds(tree_seeds=TREE_SEEDS, ann_seeds=(42,), ftt_seeds=(42,)) -> Dict[str, Tuple[int, ...]]:
    """Seeds per model name, each validated against the frozen seeds."""
    return {
        "RandomForest": check_model_seeds("RandomForest", tree_seeds),
        "XGBoost": check_model_seeds("XGBoost", tree_seeds),
        "SVM": check_model_seeds("SVM", SVM_SEEDS),
        "ANN": check_model_seeds("ANN", ann_seeds),
        "FTTransformer": check_model_seeds("FTTransformer", ftt_seeds),
    }


# ---------------------------------------------------------------- archived split
def redraw_split(y_all: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """The archived-split draw: ``StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)``."""
    y_all = np.asarray(y_all)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED)
    tr, te = next(sss.split(np.zeros(len(y_all)), y_all))
    return tr, te


def _int_vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise FoldIntegrityError(f"{name} must be a flat list of integers")
    return arr.astype(np.int64)


def read_json_consumed(path, consumed_sha256: Optional[MutableMapping[str, str]] = None):
    """
    Parse the JSON file ``path`` from one read; ``consumed_sha256[str(path)]``
    receives the SHA-256 of the bytes parsed (the hash bound in a manifest is that
    of the bytes consumed, not of a later read).
    """
    raw, _ = read_consumed(path, consumed_sha256)       # one read (config_loader.read_consumed)
    return json.loads(raw)


def load_archived_split(split_dir, target: str, y_all: Sequence,
                        consumed_sha256: Optional[MutableMapping[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ``{split_dir}/{target}_split_indices.json`` (0-based row positions of the
    cohort CSV) and assert it is a partition of the rows equal, in order, to the
    split re-draw on ``y_all``. Returns ``(train_idx, test_idx)``.
    ``consumed_sha256`` (optional) receives ``str(path) -> SHA-256`` of the bytes parsed.
    """
    path = Path(split_dir) / f"{target}_split_indices.json"
    payload = read_json_consumed(path, consumed_sha256)
    if payload.get("target") != target:
        raise FoldIntegrityError(f"{path}: target {payload.get('target')!r} != {target!r}")
    tr = _int_vector(payload.get("train_idx", []), "train_idx")
    te = _int_vector(payload.get("test_idx", []), "test_idx")
    n = len(y_all)
    both = np.concatenate([tr, te])
    if len(np.unique(both)) != len(both) or len(both) != n or not np.array_equal(np.sort(both), np.arange(n)):
        raise FoldIntegrityError(f"{path}: train/test indices are not a partition of the {n} rows")
    tr_new, te_new = redraw_split(y_all)
    if not (np.array_equal(tr, tr_new) and np.array_equal(te, te_new)):
        raise FoldIntegrityError(
            f"{path}: archived split differs from its stratified re-draw "
            "(StratifiedShuffleSplit test_size=0.2, random_state=42)")
    return tr, te


# ---------------------------------------------------------------- folds
def materialise_fold_ids(y_train: Sequence, k: int = FOLDS_K, seed: int = FOLDS_SEED) -> np.ndarray:
    """``fold_id`` (0 … k-1) of each training row, in the order of ``y_train``."""
    y_train = np.asarray(y_train)
    fold_id = np.full(len(y_train), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
        fold_id[va] = f
    assert (fold_id >= 0).all()
    return fold_id


def cv_splits_from_fold_ids(fold_id: Sequence) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Positional ``(train, validation)`` pairs, fold 0 … k-1, both sorted ascending."""
    fold_id = _int_vector(fold_id, "fold_id")
    k = int(fold_id.max()) + 1 if fold_id.size else 0
    if k < 2 or set(np.unique(fold_id).tolist()) != set(range(k)):
        raise FoldIntegrityError(f"fold_id must take every value 0 … k-1 with k >= 2 (got {np.unique(fold_id)})")
    return [(np.flatnonzero(fold_id != f), np.flatnonzero(fold_id == f)) for f in range(k)]


def fold_ids_from_splits(cv_splits, n: int) -> np.ndarray:
    """Inverse of :func:`cv_splits_from_fold_ids`; raises unless the splits are a K-fold partition of ``n`` rows."""
    fold_id = np.full(n, -1, dtype=np.int64)
    for f, (tr, va) in enumerate(cv_splits):
        tr, va = np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)
        if (fold_id[va] != -1).any():
            raise FoldIntegrityError("a training row is in the validation part of two folds")
        fold_id[va] = f
        expected_tr = np.setdiff1d(np.arange(n), va)
        if not np.array_equal(np.sort(tr), expected_tr):
            raise FoldIntegrityError(f"fold {f}: training part is not the complement of its validation part")
    if (fold_id == -1).any():
        raise FoldIntegrityError("some training rows are in no validation fold")
    return fold_id


def fold_vector_sha256(fold_id: Sequence) -> str:
    """SHA-256 of the fold vector (little-endian int64), the identity checked across families."""
    return hashlib.sha256(np.asarray(fold_id, dtype="<i8").tobytes()).hexdigest()


def splits_sha256(cv_splits, n: int) -> str:
    """Fold-vector hash of the positional splits actually used by a family."""
    return fold_vector_sha256(fold_ids_from_splits(cv_splits, n))


@dataclass(frozen=True)
class FoldAssignment:
    """The materialised folds of one target."""
    dataset: str
    target: str
    k: int
    seed: int
    train_idx: np.ndarray      # row positions of the cohort CSV, archived order
    fold_id: np.ndarray        # one fold per training row, aligned with train_idx

    @property
    def sha256(self) -> str:
        return fold_vector_sha256(self.fold_id)

    @property
    def train_idx_sha256(self) -> str:
        return fold_vector_sha256(self.train_idx)

    def splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return cv_splits_from_fold_ids(self.fold_id)

    def to_json(self) -> dict:
        return {
            "format": FOLDS_FORMAT,
            "dataset": self.dataset,
            "target": self.target,
            "k": self.k,
            "seed": self.seed,
            "method": "StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) on train_idx",
            "train_idx": self.train_idx.tolist(),
            "fold_id": self.fold_id.tolist(),
            "fold_vector_sha256": self.sha256,
            "train_idx_sha256": self.train_idx_sha256,
        }


def folds_path(folds_dir, target: str) -> Path:
    return Path(folds_dir) / f"{target}_folds.json"


def ensure_folds(folds_dir, dataset: str, target: str, train_idx: Sequence, y_train: Sequence,
                 k: int = FOLDS_K, seed: int = FOLDS_SEED,
                 consumed_sha256: Optional[MutableMapping[str, str]] = None) -> FoldAssignment:
    """
    Materialise the folds of ``target`` once and return them.

    If ``{folds_dir}/{target}_folds.json`` exists it is loaded and must match the
    dataset, target, K, seed, ``train_idx`` (in order), its recorded hash and the
    re-materialised fold vector; any difference raises (an existing file is never
    overwritten). Otherwise the file is written before any fit. A destination
    inside the frozen artefact ``rerun_2026-04-28_no_reweighting/`` raises before
    anything is read or written. ``consumed_sha256`` (optional) receives
    ``str(path) -> SHA-256`` of the bytes read (existing file) or written.
    """
    path = folds_path(folds_dir, target)
    check_not_protected(path)
    train_idx = _int_vector(train_idx, "train_idx")
    if len(train_idx) != len(y_train):
        raise FoldIntegrityError(f"{len(train_idx)} training indices for {len(y_train)} labels")
    fresh = FoldAssignment(dataset, target, int(k), int(seed), train_idx, materialise_fold_ids(y_train, k, seed))
    if path.exists():
        stored = read_json_consumed(path, consumed_sha256)
        for key in ("format", "dataset", "target", "k", "seed"):
            if stored.get(key) != fresh.to_json()[key]:
                raise FoldIntegrityError(f"{path}: {key} = {stored.get(key)!r}, expected {fresh.to_json()[key]!r}")
        stored_tr = _int_vector(stored.get("train_idx", []), "train_idx")
        stored_fold = _int_vector(stored.get("fold_id", []), "fold_id")
        if not np.array_equal(stored_tr, train_idx):
            raise FoldIntegrityError(f"{path}: train_idx differs from the archived split")
        if stored.get("train_idx_sha256") != fold_vector_sha256(stored_tr):
            raise FoldIntegrityError(f"{path}: recorded train_idx_sha256 does not match its train_idx")
        if stored.get("fold_vector_sha256") != fold_vector_sha256(stored_fold):
            raise FoldIntegrityError(f"{path}: recorded fold_vector_sha256 does not match its fold_id")
        if not np.array_equal(stored_fold, fresh.fold_id):
            raise FoldIntegrityError(f"{path}: stored folds differ from the fold materialisation")
        return fresh
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw = json.dumps(fresh.to_json()).encode()
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    if consumed_sha256 is not None:
        consumed_sha256[str(path)] = hashlib.sha256(raw).hexdigest()
    return fresh
