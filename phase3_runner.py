"""
Phase 3 runner — experiment E6 (transfer to the French cohorts).

Replaces the legacy adaptation evaluator of ``transfert.py`` (one stratified
row split; its FT-head / FT-full re-initialised the network) by the frozen design:

* **Splits.** Per cohort × target × regime, 20 group-disjoint
  splits. Groups are the vote dates (participant IDs were not received), listed
  as ISO strings in ascending order; split r permutes them with
  ``numpy.random.default_rng(1000 + r)`` and DEV is the shortest prefix of the
  permutation whose cumulative vote count is ≥ 20 % (resp. 80 %) of the
  cohort's votes; every other group is TEST. The splits are serialised to
  ``{splits_dir}/{cohort}_{target}_{regime}.json`` **before any fit**; an
  existing file is loaded and must equal the re-materialisation (never
  overwritten). With date groups the splits do not depend on the target: the
  three targets' files are asserted identical in indices.
* **Strategies**, all evaluated on exactly the TEST rows of the
  serialised split, in their serialised order (asserted with
  ``revision_stats.align_predictions``):

  - ``zero_shot``: the E1 seed-42 RF, XGB and FTT, no adaptation;
  - ``local``: RF and XGB fitted on DEV only; a DEV with fewer than two
    classes gets the DEV-majority constant predictor instead (flagged in
    the split summary);
  - ``ftt_head``, ``ftt_full``: ``ftt_adaptation.adapt_ftt`` from
    the E1 seed-42 FTT on the DEV rows with a fixed training budget.

* **No TEST vote is dropped**: a TEST class absent from DEV stays in
  TEST and is an error for a model that cannot output it; probability
  columns cover the full target ontology, mapped through each model's own
  ``classes_`` (0.0 for a class the model does not know).
* **Outputs** under ``output_dir``: one gzip CSV per (experiment, cohort,
  target, model, seed, split) in ``predictions/``, columns ``PREDICTION_COLUMNS``;
  ``ftt_adaptation/`` (training-loss history, budget and state-dict hashes of
  every FTT adaptation fit); ``split_summary.csv`` (DEV/TEST sizes, groups,
  class supports, constant-predictor flag, OOV counts per model); ``e6_run_record.json``
  (inputs, split-file and model-file hashes before/after the run, budget,
  compute section, git commit and package versions). One run per output
  directory: a non-empty ``output_dir`` is refused before any write
  (``check_output_dir``). Full-cohort zero-shot predictions (all votes, descriptive)
  use ``split_id = "full_cohort"``.

XGBoost runs on the device of ``config.yaml`` ``compute.xgb_device`` (CPU
by default); the FTT adaptation fits run on the device holding the
source network (``ftt_adaptation.resolve_device``). Every local and
FTT adaptation fit is a ``compute_scheduler`` job (RF
and XGB in the bounded CPU pool, XGB on the GPU only with ``xgb_device:
cuda``; the adaptations as GPU jobs, at most ``gpu_concurrency`` at once), with
model seed 42; its wall time and peak GPU memory are recorded in
``e6_run_record.json`` (``jobs``).

``dry_run`` (tests, smoke checks): only the first ``n_splits`` of the 20
serialised splits are evaluated and the files carry ``experiment =
"E6_dry_run"``, so that they cannot be mistaken for E6 results.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

import revision_stats as rs
from category_normalisation import split_feature_types, vocabulary_for
from compute import bounded_cpu, load_compute
from config_loader import check_not_protected, load_config
from models_classic import LabelEncodedClassifier, check_onehot_revision_schema


# ======================================================================================
# FROZEN CONSTANTS
# ======================================================================================
EXPERIMENT = "E6"
DRY_RUN_EXPERIMENT = "E6_dry_run"
FEATURE_SET = "F17"
N_SPLITS = 20
SPLIT_SEED_BASE = 1000              # split r: default_rng(1000 + r)
MODEL_SEED = 42                     # seed of the E1 source models
SOURCE_COHORT = "ASHRAE_2022"       # source cohort of the zero-shot models
GROUP_COLUMN = "Date"               # dates (no participant IDs received)
REGIMES = {"20-80": 0.2, "80-20": 0.8}          # DEV fraction of the votes
REGIME_LABELS = {"20-80": "20/80", "80-20": "80/20"}
MAIN_REGIME = "20-80"               # main regime; 80/20 supplement only
STRATEGIES = ("zero_shot", "local", "ftt_head", "ftt_full")
FTT_ADAPT_STRATEGIES = ("ftt_head", "ftt_full")
TREE_FAMILIES = ("RF", "XGB")
FULL_COHORT_SPLIT_ID = "full_cohort"
SPLITS_FORMAT = "revision-2026-10/phase3-splits-v1"
#: predictions-file family name -> compute_scheduler family name (job kind)
SCHEDULER_FAMILIES = {"RF": "RandomForest", "XGB": "XGBoost"}

PREDICTION_COLUMNS_HEAD = ("experiment", "cohort", "target", "feature_set", "model", "seed", "split_id",
                           "row_idx", "record_id", "publication", "building_id", "subject_key", "date",
                           "y_true", "y_pred")
PREDICTION_COLUMNS_TAIL = ("oov_flag", "oov_vars")


def prediction_columns(target: str) -> list:
    """Column list of a predictions file of ``target`` (``p_<class>`` over the full ontology)."""
    return (list(PREDICTION_COLUMNS_HEAD) + [f"p_{c}" for c in rs.ontology(target)]
            + list(PREDICTION_COLUMNS_TAIL))


class SplitIntegrityError(ValueError):
    """A serialised Phase 3 split differs from the frozen split procedure."""


def model_name(family: str, strategy: str) -> str:
    """Model name of the predictions files (``FTT_head`` etc. as in ``ftt_adaptation.MODEL_NAMES``)."""
    if strategy in FTT_ADAPT_STRATEGIES:
        return {"ftt_head": "FTT_head", "ftt_full": "FTT_full"}[strategy]
    return f"{family}_{strategy}"


def split_id(regime: str, r: int) -> str:
    return f"{regime}_r{r:02d}"


# ======================================================================================
# DEV/TEST PARTITION CHECK
# ======================================================================================
def check_partition(cohort_index, dev_index, test_index, label: str = "") -> None:
    """
    Assert that DEV and TEST partition the cohort: no duplicate ID within the
    cohort or either split, no ID shared by DEV and TEST, and DEV ∪ TEST equal
    to the cohort IDs exactly (nothing missing, nothing foreign).
    """
    cohort, dev, test = (pd.Index(i) for i in (cohort_index, dev_index, test_index))
    for name, idx in (("cohort", cohort), ("DEV", dev), ("TEST", test)):
        if idx.has_duplicates:
            raise ValueError(f"[{label}] duplicate IDs in {name}: "
                             f"{sorted(idx[idx.duplicated()].unique().tolist())}")
    shared = dev.intersection(test)
    if len(shared):
        raise ValueError(f"[{label}] IDs in both DEV and TEST: {sorted(shared.tolist())}")
    missing, foreign = cohort.difference(dev.union(test)), dev.union(test).difference(cohort)
    if len(missing) or len(foreign):
        raise ValueError(f"[{label}] DEV/TEST do not partition the cohort: "
                         f"missing {sorted(missing.tolist())}, foreign {sorted(foreign.tolist())}")


# ======================================================================================
# FROZEN GROUP-DISJOINT SPLITS
# ======================================================================================
def _row_groups(groups) -> np.ndarray:
    g = pd.Series(np.asarray(groups, dtype=object))
    if g.isna().any() or (g.astype(str).str.strip() == "").any():
        raise SplitIntegrityError("missing group identifier (date) in the cohort")
    return g.astype(str).to_numpy()


def group_order(groups) -> list:
    """Fixed initial order: distinct groups as strings, ascending."""
    return sorted(set(_row_groups(groups).tolist()))


def materialise_splits(groups, dev_fraction: float, n_splits: int = N_SPLITS) -> list:
    """
    The frozen split procedure, on the per-row group labels.

    Returns one dict per split r: ``r``, ``seed`` (1000 + r), ``dev_groups`` and
    ``test_groups`` (ascending), ``dev_idx`` and ``test_idx`` (ascending row
    positions of the cohort).
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction!r}")
    labels = _row_groups(groups)
    order_list = group_order(labels)
    position = {g: i for i, g in enumerate(order_list)}
    gid = np.asarray([position[g] for g in labels], dtype=np.int64)
    G, N = len(order_list), len(labels)
    n_g = np.bincount(gid, minlength=G)
    out = []
    for r in range(n_splits):
        seed = SPLIT_SEED_BASE + r
        order = np.random.default_rng(seed).permutation(G)
        cum = np.cumsum(n_g[order])
        k = int(np.searchsorted(cum, dev_fraction * N) + 1)   # shortest prefix with >= fraction of votes
        if k >= G:
            raise SplitIntegrityError(f"split {r}: DEV takes every group, TEST would be empty")
        dev_g = np.sort(order[:k])
        in_dev = np.isin(gid, dev_g)
        out.append({
            "r": r,
            "seed": seed,
            "dev_groups": [order_list[i] for i in dev_g],
            "test_groups": [order_list[i] for i in np.setdiff1d(np.arange(G), dev_g)],
            "dev_idx": np.flatnonzero(in_dev).tolist(),
            "test_idx": np.flatnonzero(~in_dev).tolist(),
        })
    return out


def _class_counts(y, target: str) -> Dict[str, int]:
    yy = rs.check_labels(y, rs.ontology(target), f"{target} labels")
    return {str(c): int(np.sum(yy == c)) for c in rs.ontology(target)}


def splits_payload(df: pd.DataFrame, cohort: str, target: str, regime: str, *,
                   group_col: str = GROUP_COLUMN, n_splits: int = N_SPLITS) -> dict:
    """JSON payload of ``{cohort}_{target}_{regime}.json`` (indices, groups, seeds, class presence)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(REGIMES)}")
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise SplitIntegrityError("split indices are row positions of the cohort CSV: RangeIndex required")
    labels = _row_groups(df[group_col])
    y = df[target].to_numpy()
    splits = materialise_splits(labels, REGIMES[regime], n_splits)
    for s in splits:
        s["split_id"] = split_id(regime, s["r"])
        s["dev_class_counts"] = _class_counts(y[s["dev_idx"]], target)
        s["test_class_counts"] = _class_counts(y[s["test_idx"]], target)
    return {
        "format": SPLITS_FORMAT,
        "cohort": cohort,
        "target": target,
        "regime": regime,
        "dev_fraction": REGIMES[regime],
        "grouping": "date",
        "group_column": group_col,
        "n_rows": int(len(df)),
        "row_groups_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest(),
        "groups": group_order(labels),
        "split_seeds": [SPLIT_SEED_BASE + r for r in range(n_splits)],
        "splits": splits,
    }


def splits_path(splits_dir, cohort: str, target: str, regime: str) -> Path:
    return Path(splits_dir) / f"{cohort}_{target}_{regime}.json"


def check_split(split: Mapping, row_groups: np.ndarray, label: str = "") -> None:
    """Partition of every cohort row, and group-disjointness of DEV and TEST."""
    n = len(row_groups)
    check_partition(range(n), split["dev_idx"], split["test_idx"], label)
    dev_g = set(row_groups[np.asarray(split["dev_idx"], dtype=int)].tolist())
    test_g = set(row_groups[np.asarray(split["test_idx"], dtype=int)].tolist())
    if dev_g & test_g:
        raise SplitIntegrityError(f"[{label}] groups in both DEV and TEST: {sorted(dev_g & test_g)}")
    if not split["dev_idx"] or not split["test_idx"]:
        raise SplitIntegrityError(f"[{label}] empty DEV or TEST")


def ensure_splits(splits_dir, df: pd.DataFrame, cohort: str, target: str, regime: str, *,
                  group_col: str = GROUP_COLUMN, consumed_sha256: Optional[dict] = None) -> dict:
    """
    Serialise the 20 splits of (cohort, target, regime) once, before any fit.
    An existing file is loaded and must equal the re-materialisation
    exactly (else ``SplitIntegrityError``); it is never overwritten.
    ``consumed_sha256[file name]`` receives the SHA-256 of the bytes read and
    compared (existing file) or written (new file), from that one read or
    write.
    """
    from config_loader import read_consumed
    check_not_protected(splits_dir)
    payload = json.loads(json.dumps(splits_payload(df, cohort, target, regime, group_col=group_col)))
    path = splits_path(splits_dir, cohort, target, regime)
    if path.exists():
        raw, _ = read_consumed(path, consumed_sha256, path.name)
        stored = json.loads(raw)
        if stored != payload:
            raise SplitIntegrityError(f"{path}: serialised splits differ from the frozen split procedure "
                                      "(or from this cohort); the file is never overwritten")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, indent=1).encode()
        with open(path, "xb") as fh:
            fh.write(raw)
        if consumed_sha256 is not None:
            consumed_sha256[path.name] = hashlib.sha256(raw).hexdigest()
    rows = _row_groups(df[group_col])
    for s in payload["splits"]:
        check_split(s, rows, f"{cohort}:{target}:{s['split_id']}")
    return payload


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================================
# FITS: LOCAL
# ======================================================================================
class DevMajorityClassifier:
    """
    DEV-majority constant predictor: replaces the local-only model when
    DEV holds fewer than two classes. Majority class, ties → class closest
    to 0, then the lower label; probability 1 on that class.
    """

    def __init__(self, target: str):
        self.target = target

    def fit(self, X, y):
        self.majority_ = rs.deployable_majority_class(y, self.target)
        self.classes_ = np.asarray([float(self.majority_)])
        return self

    def predict(self, X):
        return np.full(len(X), float(self.majority_))

    def predict_proba(self, X):
        return np.ones((len(X), 1))


def fit_local(proto, X_dev: pd.DataFrame, y_dev, target: str):
    """Local-only fit on DEV (seed 42 prototype); ``(model, is_constant_predictor)``."""
    y = np.asarray(y_dev, dtype=float)
    if len(np.unique(y)) < 2:
        return DevMajorityClassifier(target).fit(X_dev, y), True
    model = clone(proto)
    with bounded_cpu():
        model.fit(X_dev, y)
    return model, False


def tree_fit_job(seed, *, strategy: str, proto, X_dev, y_dev, target: str, frame_kw: Mapping) -> dict:
    """
    One E6 RF / XGB fit as a ``compute_scheduler`` job (the prototype carries
    model seed 42): ``local`` on DEV, evaluated on the TEST rows of
    ``frame_kw`` (``prediction_frame`` arguments). Returns the predictions frame
    and the constant-predictor flag (the model itself is not kept).
    """
    if strategy == "local":
        model, const = fit_local(proto, X_dev, y_dev, target)
    else:
        raise ValueError(f"tree_fit_job: unknown strategy {strategy!r}")
    return {"frame": prediction_frame(model, **frame_kw), "p20": bool(const)}


def ftt_adaptation_job(seed, *, source_blob: bytes, strategy: str, X_dev, y_dev, labels_order, config,
                       frame_kw: Mapping) -> dict:
    """
    One FTT adaptation as a GPU ``compute_scheduler`` job: the source
    FTT is unpickled from ``source_blob`` (a private copy; the E1 source is
    never touched), adapted on DEV with ``ftt_adaptation.adapt_ftt`` on the
    device holding the source network, and evaluated on the TEST rows of
    ``frame_kw``. Returns the predictions frame and the adaptation record.
    """
    from ftt_adaptation import adapt_ftt, adaptation_record
    source = pickle.loads(source_blob)
    est = adapt_ftt(source, strategy, X_dev, y_dev, labels_order=labels_order, config=config)
    frame = prediction_frame(est, categories=est.pp_.train_categories_, **frame_kw)
    return {"frame": frame, "record": adaptation_record(est)}


# ======================================================================================
# PREDICTIONS FILES
# ======================================================================================
def _model_classes(model) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError(f"{type(model).__name__} has no classes_")
    return np.asarray(classes)


PROBA_SUM_ATOL = 1e-4      # float32 softmax / XGB outputs sum to 1 within ~1e-6


def check_probabilities(proba: np.ndarray) -> None:
    """
    Probabilities exported to the predictions files are finite, each exactly in
    [0, 1], and sum to 1 per row within ``PROBA_SUM_ATOL`` (the tolerance applies
    to the row sum only); otherwise nothing is written.
    """
    if not np.isfinite(proba).all():
        raise FloatingPointError(f"non-finite probabilities on {int((~np.isfinite(proba)).any(axis=1).sum())} rows")
    if (proba < 0.0).any() or (proba > 1.0).any():
        raise ValueError("probabilities outside [0, 1]")
    sums = proba.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=PROBA_SUM_ATOL):
        raise ValueError(f"probability rows do not sum to 1 (range {sums.min():.6f}-{sums.max():.6f})")


def ontology_probabilities(model, X, target: str, n: int) -> np.ndarray:
    """
    ``predict_proba`` mapped to the full ontology through the model's own
    ``classes_`` (never by column position alone); a class absent from
    ``classes_`` gets 0.0.
    """
    labels = rs.ontology(target)
    classes = rs.check_labels(_model_classes(model), labels, "model classes_")
    if len(set(classes.tolist())) != len(classes):
        raise ValueError(f"duplicate model classes {classes.tolist()}")
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.shape != (n, len(classes)):
        raise ValueError(f"predict_proba shape {proba.shape} != ({n}, {len(classes)}) of classes_")
    check_probabilities(proba)
    out = np.zeros((n, len(labels)))
    for j, c in enumerate(classes):
        out[:, labels.index(int(c))] = proba[:, j]
    return out


def oov_columns(train_categories: Mapping[str, set], X_eval: pd.DataFrame, cat_cols) -> tuple:
    """``oov_flag`` / ``oov_vars`` of the evaluated rows w.r.t. a fit's training categories."""
    masks = {c: ~X_eval[c].isin(set(train_categories[c])).to_numpy() for c in cat_cols}
    flag = np.zeros(len(X_eval), dtype=bool)
    for m in masks.values():
        flag |= m
    oov_vars = [";".join(c for c in cat_cols if masks[c][i]) for i in range(len(X_eval))]
    return flag, oov_vars


def train_categories(X_train: pd.DataFrame, cat_cols) -> Dict[str, set]:
    return {c: set(X_train[c].unique()) for c in cat_cols}


def prediction_frame(model, X: pd.DataFrame, y_true, *, experiment: str, cohort: str, target: str,
                     model_name: str, seed: int, split_id: str, dates, categories: Mapping[str, set],
                     cat_cols) -> pd.DataFrame:
    """
    One predictions frame: one row per evaluated vote, in the order of
    ``X`` (= serialised split order); ``row_idx`` = ``X.index`` (row position
    in the cohort CSV). French cohorts carry no publication, building or
    participant identifier: ``record_id`` = ``"{cohort}:{row_idx}"``,
    ``publication`` / ``building_id`` / ``subject_key`` are missing.
    """
    labels = rs.ontology(target)
    n = len(X)
    yt = rs.check_labels(np.asarray(y_true), labels, f"{model_name} y_true")
    yp = rs.check_labels(np.asarray(model.predict(X)), labels, f"{model_name} y_pred")
    if len(yt) != n or len(yp) != n:
        raise ValueError(f"{model_name}: {len(yp)} predictions / {len(yt)} labels for {n} rows")
    proba = ontology_probabilities(model, X, target, n)
    flag, oov_vars = oov_columns(categories, X, cat_cols)
    row_idx = np.asarray(X.index, dtype=np.int64)
    df = pd.DataFrame({
        "experiment": experiment, "cohort": cohort, "target": target, "feature_set": FEATURE_SET,
        "model": model_name, "seed": int(seed), "split_id": split_id, "row_idx": row_idx,
        "record_id": [f"{cohort}:{i}" for i in row_idx],
        "publication": pd.array([pd.NA] * n, dtype="object"),
        "building_id": pd.array([pd.NA] * n, dtype="object"),
        "subject_key": pd.array([pd.NA] * n, dtype="object"),
        "date": np.asarray(dates, dtype=object),
        "y_true": yt, "y_pred": yp,
    })
    for j, c in enumerate(labels):
        df[f"p_{c}"] = proba[:, j]
    df["oov_flag"] = flag
    df["oov_vars"] = oov_vars
    return df[prediction_columns(target)]


def predictions_path(output_dir, frame: pd.DataFrame) -> Path:
    first = frame.iloc[0]
    name = (f"{first['experiment']}__{first['cohort']}__{first['target']}__{first['model']}"
            f"__seed{int(first['seed'])}__{first['split_id']}.csv.gz")
    return Path(output_dir) / "predictions" / name


def write_predictions(output_dir, frame: pd.DataFrame) -> Path:
    """Write a gzip CSV (fixed gzip mtime: byte-reproducible); an existing file is never overwritten."""
    if frame.empty:
        raise ValueError("empty predictions frame")
    path = predictions_path(output_dir, frame)
    if path.exists():
        raise FileExistsError(f"{path} exists: E6 outputs are never overwritten (use a new output_dir)")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


# ======================================================================================
# INPUT CHECKS
# ======================================================================================
def _seed_of(model, family: str):
    if family == "FTT":
        return getattr(model, "random_state", None)
    clf = model.named_steps["clf"]
    if isinstance(clf, LabelEncodedClassifier):
        return clf.estimator.get_params().get("random_state")
    return clf.get_params().get("random_state")


def check_zero_shot_models(models: Mapping[str, Mapping[str, object]], targets, strategies, cat_cols) -> None:
    """
    Every model needed by the requested strategies is present, is a revision
    model (global-vocabulary encoders) and has model seed 42.
    """
    from models_deep import check_ftt_revision_schema
    vocab = vocabulary_for(cat_cols)
    need = set()
    if "zero_shot" in strategies:
        need |= {"RF", "XGB", "FTT"}
    if set(strategies) & set(FTT_ADAPT_STRATEGIES):
        need.add("FTT")
    for t in targets:
        for fam in sorted(need):
            m = models.get(t, {}).get(fam)
            if m is None:
                raise ValueError(f"[{t}] E1 seed-42 {fam} model missing (needed by {sorted(strategies)})")
            if fam == "FTT":
                check_ftt_revision_schema(m, vocab, f"{t}:FTT")
            else:
                check_onehot_revision_schema(m, vocab, f"{t}:{fam}")
            seed = _seed_of(m, fam)
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) != MODEL_SEED:
                raise ValueError(f"[{t}] {fam} model seed is {seed!r}; zero-shot and FT sources are the "
                                 f"E1 seed-{MODEL_SEED} models")


def load_source_rows(csv_path, split_dir, targets, features, models_dir=None, return_coverage: bool = False,
                     consumed_sha256: Optional[dict] = None):
    """
    ASHRAE training rows of each target (zero-shot OOV reference): the cohort CSV
    normalised as in E1 and the archived split, asserted equal to its
    stratified re-draw (``revision_folds.load_archived_split``). With ``models_dir``,
    the split written by the E1 run (``{target}/splits/{target}_split_indices.json``)
    must equal it (the zero-shot models were trained on these rows).
    Returns ``{target: (X_train, y_train)}``; with ``return_coverage``,
    ``(rows, coverage)`` where ``coverage`` is the category-coverage record of the
    cohort normalisation (written by E6 as ``category_coverage_ASHRAE_2022.json``).
    ``consumed_sha256`` (optional) receives the SHA-256 of the bytes parsed,
    by input role of the E6 manifest: ``cohort_csv``,
    ``split:{target}`` (archived split) and
    ``e1:{target}/splits/{target}_split_indices.json`` (E1 copy).
    """
    import revision_folds
    from config_loader import read_consumed
    from unified_data_loader import UnifiedDataLoader
    loader = UnifiedDataLoader(list(features), list(rs.TARGETS))
    # one read of the cohort: parsed from these bytes, their hash reported as ``cohort_csv``
    raw, _ = read_consumed(csv_path, consumed_sha256, "cohort_csv")
    data = loader.load_and_validate(str(csv_path), SOURCE_COHORT, category_normalisation=True, raw=raw)
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError("split indices are row positions of the cohort CSV: the loader must keep a RangeIndex")
    out = {}
    for t in targets:
        seen: dict = {}
        tr, te = revision_folds.load_archived_split(split_dir, t, data[t], consumed_sha256=seen)
        if consumed_sha256 is not None:
            consumed_sha256[f"split:{t}"] = next(iter(seen.values()))
        if models_dir is not None:
            e1 = Path(models_dir) / t / "splits" / f"{t}_split_indices.json"
            if not e1.exists():
                raise FileNotFoundError(f"{e1}: E1 split file missing (zero-shot model provenance)")
            seen = {}
            payload = revision_folds.read_json_consumed(e1, seen)
            if consumed_sha256 is not None:
                consumed_sha256[f"e1:{t}/splits/{t}_split_indices.json"] = seen[str(e1)]
            if payload.get("train_idx") != tr.tolist() or payload.get("test_idx") != te.tolist():
                raise ValueError(f"{e1}: E1 split differs from the archived split of {split_dir}")
        out[t] = (data[list(features)].iloc[tr], data[t].iloc[tr].to_numpy(dtype=float))
    if return_coverage:
        return out, list(loader.category_coverage.get(SOURCE_COHORT) or [])
    return out


def config_sha256(config: Mapping) -> str:
    """SHA-256 of the configuration used (sorted-key JSON; same digest as the run manifests)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def check_output_dir(output_dir, splits_dir=None) -> None:
    """
    One E6 run per output directory: refuse an ``output_dir`` that already
    holds anything (run-level files such as ``split_summary.csv`` and
    ``e6_run_record.json`` would otherwise be overwritten and mix two runs).
    The serialised splits directory may sit inside ``output_dir`` (loaded and
    checked, never overwritten); it must not be ``output_dir`` itself.
    """
    out = Path(output_dir)
    if splits_dir is not None and Path(splits_dir).resolve() == out.resolve():
        raise ValueError(f"{out}: the splits directory must differ from the E6 output directory")
    if not out.exists():
        return
    if not out.is_dir():
        raise NotADirectoryError(f"{out} is not a directory")
    keep = Path(splits_dir).resolve() if splits_dir is not None else None
    others = sorted(p.name for p in out.iterdir()
                    if keep is None or not (p.resolve() == keep or p.resolve() in keep.parents))
    if others:
        raise FileExistsError(f"{out} is not empty ({others[:5]}): one E6 run per output directory, "
                              "outputs are never overwritten (use a new output_dir)")


PROVENANCE_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch")


def git_checkout(repo_dir=None) -> dict:
    """
    Kind of checkout the code runs from: a
    *linked worktree* (``git rev-parse --git-dir`` differs from
    ``--git-common-dir``; the documented launch procedure of real runs, a
    worktree fixed at the run commit) or the main development checkout.
    ``linked_worktree`` is ``None`` when git is unavailable.
    """
    import subprocess
    repo = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    git_dir, common = git("rev-parse", "--git-dir"), git("rev-parse", "--git-common-dir")
    if not git_dir or not common:
        return {"linked_worktree": None, "git_dir": None, "git_common_dir": None, "toplevel": None}
    git_dir, common = (repo / git_dir).resolve(), (repo / common).resolve()
    return {"linked_worktree": git_dir != common, "git_dir": str(git_dir), "git_common_dir": str(common),
            "toplevel": git("rev-parse", "--show-toplevel")}


def run_provenance() -> dict:
    """
    Code and environment of the run: git commit of this repository and
    whether tracked files differ from it, Python and package versions, torch
    CUDA build, and the kind of checkout (``git_worktree``: linked worktree or
    main checkout, ``git_checkout``). ``None`` where unavailable. The full
    ``pip freeze`` and the driver version belong to the rerun manifest.
    """
    import importlib.metadata as metadata
    import platform
    import subprocess

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True,
                                  text=True, check=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    versions = {}
    for pkg in PROVENANCE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        import torch
        cuda = torch.version.cuda
    except ImportError:
        cuda = None
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_tracked_changes": None if status is None else bool(status),
            "python": platform.python_version(), "packages": versions, "torch_cuda": cuda,
            "git_worktree": git_checkout()}


# ======================================================================================
# RUNNER
# ======================================================================================
def _ftt_state_hashes(models, targets) -> Dict[str, str]:
    from ftt_adaptation import state_dict_sha256
    return {t: state_dict_sha256(models[t]["FTT"].model_) for t in targets
            if models.get(t, {}).get("FTT") is not None}


def run_e6(*, cohorts: Mapping[str, pd.DataFrame], source_rows: Mapping[str, tuple],
           models: Mapping[str, Mapping[str, object]], make_protos: Callable, output_dir, splits_dir,
           targets: Sequence[str] = rs.TARGETS, regimes: Sequence[str] = (MAIN_REGIME,),
           strategies: Sequence[str] = STRATEGIES, features: Optional[Sequence[str]] = None,
           config: Optional[dict] = None, compute: Optional[dict] = None, n_splits: int = N_SPLITS,
           dry_run: bool = False, model_files: Optional[Mapping[str, object]] = None,
           cohort_files: Optional[Mapping[str, object]] = None, command: Optional[list] = None,
           gpu_concurrency: Optional[int] = None, source_files_sha256: Optional[Mapping[str, str]] = None,
           model_files_sha256: Optional[Mapping[str, str]] = None,
           cohort_files_sha256: Optional[Mapping[str, str]] = None) -> dict:
    """
    Run E6 and write its outputs (see the module docstring).

    Parameters
    ----------
    cohorts : {name: DataFrame}
        Normalised French cohorts (``transfert.load_external_cohort``), RangeIndex
        = row positions of the cohort CSV, with ``Date`` and the targets.
    source_rows : {target: (X_train, y_train)}
        ASHRAE training rows of each target (``load_source_rows``).
    models : {target: {"RF", "XGB", "FTT": fitted E1 seed-42 model}}
    make_protos : callable
        ``(num_cols, cat_cols, seed=, cat_vocabulary=, compute=) -> (rf, xgb)``
        prototype pipelines with the frozen hyperparameters
        (``transfert.make_rf_xgb_protos``).
    n_splits : int
        Splits evaluated: 20; fewer only with ``dry_run`` (all 20 are serialised).
    model_files : {name: path}
        Saved E1 model files; their SHA-256 must be unchanged at the end.
    model_files_sha256, cohort_files_sha256 : {name: SHA-256}
        Hashes of the bytes the caller actually loaded (``transfert.load_model_dicts``,
        ``load_external_cohort``), recorded instead of a
        later read of the files; the model files must still have them at the end.
    gpu_concurrency : int or None
        Override of ``compute.gpu_concurrency`` for this run (scheduling only; recorded);
        a real run refuses a value above ``compute.gpu_concurrency``.

    Returns
    -------
    dict
        The run record written to ``e6_run_record.json``.
    """
    import models as _models_module  # FEATURES_ALL (F17)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    features = list(_models_module.FEATURES_ALL if features is None else features)
    strategies, regimes, targets = list(strategies), list(regimes), list(targets)
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}; expected a subset of {STRATEGIES}")
    for reg in regimes:
        if reg not in REGIMES:
            raise ValueError(f"unknown regime {reg!r}; expected a subset of {sorted(REGIMES)}")
    for t in targets:
        rs.ontology(t)
    if not dry_run and n_splits != N_SPLITS:
        raise ValueError(f"E6 evaluates the {N_SPLITS} frozen splits; n_splits={n_splits} needs dry_run=True")
    if not 1 <= n_splits <= N_SPLITS:
        raise ValueError(f"n_splits must be in [1, {N_SPLITS}]")
    check_not_protected(output_dir)
    check_not_protected(splits_dir)
    check_output_dir(output_dir, splits_dir)   # one run per output directory, before any write
    output_dir = Path(output_dir)
    experiment = DRY_RUN_EXPERIMENT if dry_run else EXPERIMENT
    config = load_config() if config is None else config
    compute = load_compute(config) if compute is None else compute
    if gpu_concurrency is not None and (int(gpu_concurrency) < 1 or (
            not dry_run and int(gpu_concurrency) > int(compute["gpu_concurrency"]))):
        raise ValueError(f"gpu_concurrency {gpu_concurrency}: must be >= 1 and, in a real run, at most "
                         f"compute.gpu_concurrency = {compute['gpu_concurrency']} (one GPU job at a time "
                         "unless config.yaml says otherwise)")
    budget = None
    if set(strategies) & set(FTT_ADAPT_STRATEGIES):
        from ftt_adaptation import load_budget
        budget = load_budget(config)

    any_cohort = next(iter(cohorts.values()))
    num_cols, cat_cols = split_feature_types(any_cohort, features)
    vocab = vocabulary_for(cat_cols)
    check_zero_shot_models(models, targets, strategies, cat_cols)
    if "local" in strategies:
        rf_proto, xgb_proto = make_protos(num_cols, cat_cols, seed=MODEL_SEED, cat_vocabulary=vocab,
                                          compute=compute)
        protos = {"RF": rf_proto, "XGB": xgb_proto}
    if "zero_shot" in strategies:
        missing = [t for t in targets if t not in source_rows]
        if missing:
            raise ValueError(f"ASHRAE training rows missing for {missing} (zero-shot OOV reference)")

    provenance = run_provenance()   # code and environment at the start of the run
    if model_files_sha256 is not None:            # the bytes loaded
        if set(model_files_sha256) != set(model_files or {}):
            raise ValueError(f"model file hashes {sorted(model_files_sha256)} != model files {sorted(model_files or {})}")
        files_before = dict(model_files_sha256)
    else:
        files_before = {k: file_sha256(p) for k, p in (model_files or {}).items()}
    ftt_before = _ftt_state_hashes(models, targets)

    # ---------------------------------------------------- 1. splits, serialised before any fit
    split_files, payloads = {}, {}
    for cohort, df in cohorts.items():
        if not df.index.equals(pd.RangeIndex(len(df))):
            raise SplitIntegrityError(f"{cohort}: RangeIndex required (row_idx = row position in the CSV)")
        for reg in regimes:
            ref = None
            for t in targets:
                p = ensure_splits(splits_dir, df, cohort, t, reg, consumed_sha256=split_files)   # bytes used
                payloads[(cohort, t, reg)] = p
                idx = [(s["dev_idx"], s["test_idx"]) for s in p["splits"]]
                if ref is not None and idx != ref:
                    raise SplitIntegrityError(f"{cohort} {reg}: date splits differ between targets")
                ref = idx

    # ---------------------------------------------------- 2. fits (compute_scheduler jobs) and predictions
    # Every local / FTT adaptation fit is one compute_scheduler job:
    # RF and XGB on the CPU (XGB on the GPU only with compute.xgb_device = cuda), the FTT
    # adaptations as GPU jobs in worker processes, each from its own copy of the source FTT
    # (pickled once per target). The jobs return their TEST predictions frame (no model is
    # kept); frames are then aligned and written split by split, in the serialised order.
    from compute_scheduler import Job, job_kind, job_records, run_jobs
    adapt = [s_ for s_ in FTT_ADAPT_STRATEGIES if s_ in strategies]
    ftt_blobs = {t: pickle.dumps(models[t]["FTT"]) for t in targets} if adapt else {}
    plan, jobs, n_files = [], [], 0
    for cohort, df in cohorts.items():
        X_all = df[features]
        dates = df[GROUP_COLUMN].astype(str).to_numpy()
        for t in targets:
            y_all = df[t].to_numpy(dtype=float)
            rs.check_labels(y_all, rs.ontology(t), f"{cohort}:{t} labels")
            zs = models.get(t, {})
            X_src, y_src = source_rows.get(t, (None, None))
            src_cats = None if X_src is None else train_categories(X_src, cat_cols)

            def _kw(rows, name, sid, cats=None, *, _X=X_all, _y=y_all, _d=dates, _c=cohort, _t=t):
                kw = dict(X=_X.iloc[rows], y_true=_y[rows], experiment=experiment, cohort=_c, target=_t,
                          model_name=name, seed=MODEL_SEED, split_id=sid, dates=_d[rows], cat_cols=cat_cols)
                if cats is not None:
                    kw["categories"] = cats
                return kw

            # full-cohort zero-shot (descriptive, all votes)
            if "zero_shot" in strategies:
                every = np.arange(len(df))
                for fam in ("RF", "XGB", "FTT"):
                    cats = zs[fam].pp_.train_categories_ if fam == "FTT" else src_cats
                    write_predictions(output_dir, prediction_frame(
                        zs[fam], **_kw(every, model_name(fam, "zero_shot"), FULL_COHORT_SPLIT_ID, cats)))
                    n_files += 1

            for reg in regimes:
                for s in payloads[(cohort, t, reg)]["splits"][:n_splits]:
                    sid = s["split_id"]
                    dev, test = np.asarray(s["dev_idx"]), np.asarray(s["test_idx"])
                    X_dev, y_dev = X_all.iloc[dev], y_all[dev]
                    dev_cats = train_categories(X_dev, cat_cols)
                    entry = {"cohort": cohort, "target": t, "regime": reg, "split": s, "dev": dev, "test": test,
                             "frames": {}, "jobs": []}

                    def _job(name, slot, fn, kind, **kwargs):
                        key = f"{cohort}|{t}|{reg}|{sid}|{name}"
                        jobs.append(Job(key=key, kind=kind, fn=fn, seed=MODEL_SEED, kwargs=kwargs))
                        entry["jobs"].append((name, key, slot))

                    if "zero_shot" in strategies:
                        for fam in ("RF", "XGB", "FTT"):
                            cats = zs[fam].pp_.train_categories_ if fam == "FTT" else src_cats
                            name = model_name(fam, "zero_shot")
                            entry["frames"][name] = prediction_frame(zs[fam], **_kw(test, name, sid, cats))
                    if "local" in strategies:
                        for fam in TREE_FAMILIES:
                            name = model_name(fam, "local")
                            _job(name, "local", tree_fit_job, job_kind(SCHEDULER_FAMILIES[fam], compute),
                                 strategy="local", proto=protos[fam], X_dev=X_dev, y_dev=y_dev, target=t,
                                 frame_kw=_kw(test, name, sid, dev_cats))
                    for strategy in adapt:
                        name = model_name("FTT", strategy)
                        _job(name, strategy, ftt_adaptation_job, job_kind("FTT_adaptation", compute),
                             source_blob=ftt_blobs[t], strategy=strategy, X_dev=X_dev, y_dev=y_dev,
                             labels_order=rs.ontology(t), config=config, frame_kw=_kw(test, name, sid))
                    plan.append(entry)

    results = run_jobs(jobs, compute=compute, gpu_concurrency=gpu_concurrency) if jobs else {}

    summary_rows = []
    for entry in plan:
        cohort, t, reg, s = entry["cohort"], entry["target"], entry["regime"], entry["split"]
        sid, dev, test = s["split_id"], entry["dev"], entry["test"]
        frames, p20 = dict(entry["frames"]), False
        for name, key, strategy in entry["jobs"]:
            value = results[key].value
            frames[name] = value["frame"]
            if strategy == "local":
                p20 = p20 or value["p20"]
            if strategy in FTT_ADAPT_STRATEGIES:
                rec = value["record"]
                if rec["dev_idx"] != dev.tolist():
                    raise RuntimeError(f"{sid} {strategy}: adapted on rows other than the DEV rows")
                rec.update({"experiment": experiment, "cohort": cohort, "target": t, "regime": reg,
                            "split_id": sid, "split_seed": s["seed"]})
                rec_path = output_dir / "ftt_adaptation" / f"{experiment}__{cohort}__{t}__{sid}__{name}.json"
                if rec_path.exists():
                    raise FileExistsError(f"{rec_path} exists: E6 outputs are never overwritten")
                rec_path.parent.mkdir(parents=True, exist_ok=True)
                rec_path.write_text(json.dumps(rec, indent=1))

        if frames:
            # every strategy on exactly the same TEST rows, in the serialised order
            rs.align_predictions(frames, cluster_col="date", expected_row_idx=test)
        for fr in frames.values():
            write_predictions(output_dir, fr)
            n_files += 1
        dev_counts, test_counts = s["dev_class_counts"], s["test_class_counts"]
        summary_rows.append({
            "experiment": experiment, "cohort": cohort, "target": t, "regime": reg,
            "split_id": sid, "split_seed": s["seed"], "n_dev": len(dev), "n_test": len(test),
            "n_dev_groups": len(s["dev_groups"]), "n_test_groups": len(s["test_groups"]),
            "dev_class_counts": json.dumps(dev_counts), "test_class_counts": json.dumps(test_counts),
            "dev_n_classes": sum(v > 0 for v in dev_counts.values()),
            "test_classes_absent_from_dev": json.dumps(
                [int(c) for c in dev_counts if dev_counts[c] == 0 and test_counts[c] > 0]),
            "local_constant_p20": bool(p20),
            **{f"oov_test_votes__{k}": int(v["oov_flag"].sum()) for k, v in frames.items()},
        })

    # ---------------------------------------------------- 3. sources untouched
    files_after = {k: file_sha256(p) for k, p in (model_files or {}).items()}
    ftt_after = _ftt_state_hashes(models, targets)
    if files_after != files_before or ftt_after != ftt_before:
        raise RuntimeError("an E1 source model (file or FTT state dict) changed during E6")

    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    record = {
        "experiment": experiment,
        "dry_run": bool(dry_run),
        "started_utc": started,
        "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": list(command) if command is not None else None,
        "provenance": provenance,
        "cohorts": list(cohorts),
        "targets": targets,
        "regimes": regimes,
        "strategies": strategies,
        "n_splits_serialised": N_SPLITS,
        "n_splits_evaluated": n_splits,
        "model_seed": MODEL_SEED,
        "feature_set": FEATURE_SET,
        "features": features,
        "split_files_sha256": split_files,
        # ASHRAE split bytes parsed by load_source_rows, by input role
        "source_files_sha256": dict(source_files_sha256) if source_files_sha256 is not None else None,
        "cohort_files_sha256": (dict(cohort_files_sha256) if cohort_files_sha256 is not None
                                else {k: file_sha256(p) for k, p in (cohort_files or {}).items()}),
        "model_files_sha256": files_before,
        "ftt_source_state_sha256": ftt_before,
        "ftt_budget": budget,
        "config_sha256": config_sha256(config),
        "compute": dict(compute),
        "gpu_concurrency": compute["gpu_concurrency"] if gpu_concurrency is None else int(gpu_concurrency),
        "jobs": job_records(results),
        "n_prediction_files": n_files,
    }
    (output_dir / "e6_run_record.json").write_text(json.dumps(record, indent=1, default=str))
    return record
