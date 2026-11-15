"""
Runners of the revision rerun: experiments E1, E2, E4, the E6 and E7 drivers;
E10 (hybrids), the E7b driver (SET / PET / ePMV) and E9b (NN analysis),
restored from the pre-revision code.

Command line: ``run_revision.py`` (one entry point, one subcommand per
experiment). This module holds the library: data context, job functions,
orchestrators, predictions files and run manifests.

Common rules
------------
* **Run directory.** Every experiment writes under ``{run_dir}/{EXPERIMENT_DIRS[e]}/``
  (``E1_phase2``, ``E2_lopo``, ``E4_ema``, ``E6_phase3``, ``E7_pmv``,
  ``E10_hybrids``, ``E7b_indices``, ``E9b_nn_cases``). The run directory is an argument (placeholder
  ``rerun_2026-10-XX_v2``); the frozen canonical artefact
  ``rerun_2026-04-28_no_reweighting/`` is refused, and an experiment directory
  that already holds anything is refused before any write (one run per output
  directory, outputs never overwritten). Dry runs (tests, smoke checks) use
  ``{dir}_dry_run`` and label every file ``{experiment}_dry_run``.
* **Data.** The ASHRAE-2022 cohort (``Data/ASHRAE_2022_Clean_api.csv``) with the
  category normalisation, its group identifiers re-attached by
  ``data_groups.load_group_ids`` (``record_id``, ``publication``,
  ``building_id``, ``subject_id``, raw TSV vote) and the archived 80/20 split of
  each target, loaded and asserted equal to its re-draw.
* **Predictions.** One gzip CSV per (experiment, cohort, target, model,
  seed, split) in ``{exp_dir}/predictions/``, file name
  ``{experiment}__ASHRAE_2022__{target}__{model}__seed{seed}__{split_id}.csv.gz``
  (the E6 pattern), columns ``phase3_runner.prediction_columns``; one row per
  evaluated vote in the order of the serialised split; ``p_<class>`` over the
  full ontology through each model's ``classes_`` (NaN for SVM, which has no
  probabilities); ``oov_flag`` / ``oov_vars`` relative to each fit's training
  rows. Model names: ``RF``, ``XGB``, ``SVM``, ``ANN``, ``FTT``.
* **Scheduling.** Every fit is a ``compute_scheduler.Job`` (GPU jobs: ANN, FTT,
  and XGB only when ``compute.xgb_device`` is ``cuda``; CPU jobs: RF, SVM, XGB on
  the CPU), seeded with its own model seed, so results do not depend on the
  scheduling.
* **Manifest.** ``{exp_dir}/run_manifest.json``: command line, git commit,
  environment (lock-file hash, ``pip freeze`` and its hash, Python, CUDA runtime,
  NVIDIA driver), config hash, data/split/fold hashes, seeds, per-job wall time
  and peak GPU memory, every output file with its SHA-256. ``{run_dir}/manifest.json``
  indexes the experiments of the run (updated under a file lock) and carries the
  provenance of the unchanged Phase 1 artefact; the effective scheduling
  (``scheduler``: GPU jobs at once, override or config) and the untracked
  ``*.py`` files are recorded. A real (non-dry) run refuses an environment that
  differs from ``requirements-revision-2026-10.lock``, a configuration that
  differs from ``config.yaml`` and a git checkout with modified tracked files.
  **Startup binding**: ``_prepare`` keeps an
  immutable ``StartupRecord`` (code provenance, effective configuration hash,
  ``config.yaml`` hash), each runner binds to it the SHA-256 of the inputs it
  consumed (``bind_inputs``, before any fit; the cohort CSV hash is taken when
  it is loaded), and ``write_manifest`` re-checks the checkout (clean for a real
  run) and compares the ending commit / provenance, configuration and input
  hashes with that record: any drift aborts before the manifest and the run
  index entry are written. The manifest records the startup provenance.
  **One read per consumed input**: every
  input (cohort CSV, raw identifier files, French cohorts, split / fold JSONs,
  E1 predictions and model files, ``config.yaml``, lock file) is read once by
  ``config_loader.read_consumed`` and parsed from those bytes; the hash bound,
  pinned and recorded is that of the bytes parsed, and delegated readers (E6,
  E7, E9b) report it for ``check_consumed_inputs``.
* **Category coverage.** Every experiment on the cohort writes
  ``category_coverage_ASHRAE_2022.json`` (raw-value mapping / rejection log).
* **E1 as a source.** E4, E6, E7, E7b and E10 validate the E1 run they read
  (``load_e1_source``): manifest present, recorded output files with their
  SHA-256, same cohort CSV and archived splits; a real run refuses a dry-run E1;
  E4 also needs the same configuration, XGB device, environment and commit.

Experiments
-----------
* **E1**: RF, XGB (seeds 42 43 44), SVM (one run), ANN and FTT (seeds
  given by ``--ann_seeds`` / ``--ftt_seeds``; the paper run used 42 43 44) on the 12 179 training rows of each target, folds
  materialised once and shared by every family (hash checked), final
  evaluation on the 3 045 archived test rows. Also writes the files E6 reads:
  ``best_{rf,xgb,svm,ft}_by_target[_seed{s}].joblib`` and
  ``{target}/splits/{target}_split_indices.json``.
* **E2**: leave-one-publication-out, RF and XGB, seed 42; training
  rows = every row of the other publications; OOV votes kept and counted.
* **E4**: RF and XGB, seeds 42 43 44, constant hyperparameters;
  E4 fits F12; the F17 side of the contrast is the
  E1 fit (identical model: same rows, hyperparameters and seeds; tested), accepted
  only after ``load_e1_source`` (paired) and ``check_e1_predictions`` (identity
  columns, rows, labels, record ids), and referenced, with its SHA-256 and the
  E1 manifest SHA-256, in ``pairs.csv``. ``metrics.csv`` carries the
  majority baselines.
* **E6**: driver of ``transfert.py`` / ``phase3_runner`` with typed
  arguments (the wrapper owns the paths, the dry-run state and the configuration,
  whose SHA-256 must equal the ``config_sha256`` of ``e6_run_record.json``); every
  E6 fit is a scheduler job; the ASHRAE source rows get their coverage log.
* **E7**: driver of ``indicateurs_classiques`` (corrected indices module) on the archived
  test rows, with the E1 predictions as the ML comparison and the E1 inventory as
  ``--expected_ml``; a dry run labels its files ``E7_dry_run``.
* **E7b**: the same driver with ``--indices set_pet_epmv`` (SET /
  PTS_SET, PET / PTS_PET, ePMV; TSV-7 and TSV-3), directory ``E7b_indices``.
* **E10**: RF / XGBoost heads fitted once on the CPU-extracted
  embeddings of each E1 FTT (seeds 42 43 44, never refitted), paper head
  hyperparameters (code constant ``PAPER_HYBRID_HEADS``, recorded in the
  manifest; ``config.yaml`` is left byte-identical to the run config of the other experiments),
  head seed = FTT seed;
  predictions ``FTT_RF`` / ``FTT_XGB``.
* **E9b**: driver of ``analysis/nn_cases.py`` (primary 1-NN, Case B, Case C on F12,
  full cohort).

E2 and E4 fit RF/XGB directly with the single hyperparameter point of ``config.yaml``
(no descriptive CV: the folds serve the SVM search and the E1 supplement only);
a grid with more than one point raises.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import revision_folds
import revision_stats as rs
from category_normalisation import oov_report, oov_summary, split_feature_types, vocabulary_for
from compute import gpu_job_threads, load_compute, xgb_compute_params
from compute_scheduler import Job, job_kind, job_records, run_jobs
from config_loader import check_not_protected, load_config, read_consumed
from phase3_runner import (MODEL_SEED, file_sha256, oov_columns, ontology_probabilities, prediction_columns,
                           run_provenance, train_categories)

REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "config.yaml"
LOCK_FILE = REPO_DIR / "requirements-revision-2026-10.lock"
#: SHA-256 of config.yaml and of the lock file recorded by the manifests of the recorded run -> SHA-256 of the
#: same files in this release. Only comment lines changed: the parsed configuration (``config_used_sha256``
#: e859f3e226de006d2609a08540d91e65e98e33993c404992d64728242ba6fda1) and the lock-file pins are identical.
RELEASE_FILE_SHA256 = {
    "771fef24e01ef5fa49dc5d43cb7677bb91c242625c9a12d699db1b465f987f4a":      # config.yaml
        "49053732523b5105b57ef4951e3bfe91ccadeee846458462852f7d11dd27b30c",
    "53cbbdfeaea1e91ee42063addefbd6b37da07e56b37557e7a73264ac226943d0":      # requirements-revision-2026-10.lock
        "6e369df15041ca94cf3ab78946aa1fc29ca28933c4021c4ada8ac1e9ee1776f1",
}


def release_file_sha256(recorded_sha256: Optional[str]) -> Optional[str]:
    """A recorded config.yaml or lock-file SHA-256 expressed as the SHA-256 of the same file in this release
    (``RELEASE_FILE_SHA256``; any other value is returned unchanged)."""
    return RELEASE_FILE_SHA256.get(recorded_sha256, recorded_sha256)
PHASE1_DIR = REPO_DIR / "results" / "phase1"
# Pinned hash of the unchanged Phase 1 grid: the grid of the Phase 1 run
# (rerun_2026-04-28_no_reweighting/phase1_haghirad_grid/experiment_grid_full.csv), shipped as
# results/phase1/experiment_grid_full.csv.
PINNED_PHASE1_GRID_SHA256 = "8d520cf35363f885d16fcee0f6c588604c12cb30701a3c6f80ce1f19b551bede"

# ======================================================================================
# FROZEN CONSTANTS
# ======================================================================================
COHORT = "ASHRAE_2022"
DATASET = "ASHRAE_2022"
EXPERIMENT_DIRS = {"E1": "E1_phase2", "E2": "E2_lopo", "E4": "E4_ema", "E6": "E6_phase3", "E7": "E7_pmv",
                   "E10": "E10_hybrids", "E7b": "E7b_indices", "E9b": "E9b_nn_cases"}
DRY_RUN_SUFFIX = "_dry_run"
#: predictions-file model name -> family name of models_classic / models_deep / revision_folds
FAMILIES = {"RF": "RandomForest", "XGB": "XGBoost", "SVM": "SVM", "ANN": "ANN", "FTT": "FTTransformer"}
E1_FAMILIES = ("RF", "XGB", "SVM", "ANN", "FTT")
TREE_FAMILIES = ("RF", "XGB")
DEEP_FAMILIES = ("ANN", "FTT")
#: key of the E1 model dictionaries read by transfert.load_model_dicts (as models.py)
BEST_KEYS = {"RF": "rf", "XGB": "xgb", "SVM": "svm", "FTT": "ft"}
#: feature-set label (predictions files) -> config.yaml key
FEATURE_SETS = {"F17": "features", "F12": "features_12"}
ARCHIVED_SPLIT_ID = "archived_80_20"
LOPO_SEED = 42                      # E2 seed 42 only
TSV3_MAP = {-3: -1, -2: -1, -1: -1, 0: 0, 1: 1, 2: 1, 3: 1}     # TSV-7 -> TSV-3
#: E4: base and extended feature sets, and the set fitted here
#: (the F17 side is the E1 fit).
ABLATIONS = {"E4": {"base": "F12", "extended": "F17", "fitted": "F12"}}
MANIFEST_FORMAT = "revision-2026-10/run-manifest-v1"
#: installed distributions allowed outside the lock file (installer tooling, not imported by the runs)
UNPINNED_ALLOWED = ("pip", "setuptools", "wheel")
RUN_INDEX_FORMAT = "revision-2026-10/run-index-v1"

#: Tiny hyperparameters of a ``--dry_run`` (smoke checks of the pipeline, never results).
#: The FTT ``lr`` and ``batch_size`` equal the frozen FTT-local adaptation values
#: (``ftt_adaptation.load_budget``), so that the E6 dry run adapts FTT under
#: the same configuration that it records; the frozen adaptation budget is never changed.
DRY_RUN_HPARAMS = {
    "classical_hparams": {
        "RandomForest": {"n_estimators": 10, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 2,
                         "class_weight": None},
        "XGBoost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.3, "subsample": 0.7,
                    "colsample_bytree": 0.7},
        "SVM": {"kernel": ["rbf"], "C": [1], "gamma": ["scale"]},
    },
    "deep_hparams": {
        "ANN": {"hidden_layers": [16], "dropout": 0.1, "epochs": 3, "lr": 0.003, "batch_size": 1024,
                "patience": 2, "focal_gamma": None, "class_weights": None},
        "FTTransformer": {"dim": 8, "depth": 1, "heads": 2, "dim_head": 4, "attn_dropout": 0.1,
                          "ff_dropout": 0.1, "epochs": 2, "lr": 3e-4, "batch_size": 1024, "patience": 2,
                          "focal_gamma": None, "class_weights": None},
    },
    "hybrid": {"heads": {
        "RF": {"n_estimators": 10, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 2,
               "class_weight": None},
        "XGBoost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.3, "subsample": 0.7,
                    "colsample_bytree": 0.7},
    }},
}


class RunnerError(RuntimeError):
    """A runner input or output violates the frozen rules (nothing is written after it)."""


# ======================================================================================
# SMALL HELPERS
# ======================================================================================
def experiment_label(experiment: str, dry_run: bool) -> str:
    return experiment + (DRY_RUN_SUFFIX if dry_run else "")


def experiment_dir(run_dir, experiment: str, dry_run: bool = False) -> Path:
    return Path(run_dir) / (EXPERIMENT_DIRS[experiment] + (DRY_RUN_SUFFIX if dry_run else ""))


def check_empty_output_dir(path) -> Path:
    """Refuse the frozen artefact and any non-empty directory (one run per output directory)."""
    check_not_protected(path)
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        others = sorted(p.name for p in path.iterdir())
        if others:
            raise FileExistsError(f"{path} is not empty ({others[:5]}): one run per output directory, "
                                  "outputs are never overwritten (use a new run directory)")
    return path


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json_sha256(obj) -> str:
    return _sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fixed_params(param_grid: Mapping[str, Sequence]) -> dict:
    """The single hyperparameter point of a one-combination grid (E2, E4); else raise."""
    out = {}
    for k, v in param_grid.items():
        if len(v) != 1:
            raise RunnerError(f"{k}: {len(v)} values; E2 and E4 use the single config.yaml point "
                              "(constant hyperparameters), not a search")
        out[k] = v[0]
    return out


def features_for(config: dict, feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}")
    return list(config["data"][FEATURE_SETS[feature_set]])


def subject_keys(ids: pd.DataFrame) -> pd.Series:
    """Subject key ``publication|building_id|subject_id`` (<NA> when subject_id is missing), as E7."""
    sid = ids["subject_id"]
    key = ids["publication"].astype(str) + "|" + ids["building_id"].astype(str) + "|" + sid.astype(str)
    return key.where(sid.notna(), pd.NA)


def tsv7_to_tsv3(tsv7) -> np.ndarray:
    """TSV-7 -> TSV-3 map: {-3,-2,-1} -> -1, 0 -> 0, {1,2,3} -> +1 (raises outside the ontology)."""
    y = rs.check_labels(tsv7, rs.ontology("thermal_sensation"), "TSV-7")
    return np.asarray([TSV3_MAP[int(v)] for v in y], dtype=float)


# ======================================================================================
# DATA CONTEXT
# ======================================================================================
@dataclass
class AshraeContext:
    """The ASHRAE-2022 cohort as every runner sees it (row position = canonical row)."""
    data: pd.DataFrame          # normalised features (F17 ∪ F12 ∪ F18) + targets, RangeIndex
    ids: pd.DataFrame           # record_id, publication, building_id, subject_id, thermal_sensation_raw
    p26: pd.DataFrame           # record_id, publication, building_id, subject_key, date (predictions identifiers)
    csv_path: Path
    input_files: Dict[str, str]  # role -> path (hashed in the manifest)
    #: category coverage records of the normalisation (written by every experiment)
    category_coverage: List[dict] = field(default_factory=list)
    #: SHA-256 of the bytes of ``input_files`` that were parsed (one read each)
    input_sha256: Dict[str, str] = field(default_factory=dict)


def all_features(config: dict) -> List[str]:
    out: List[str] = []
    for key in FEATURE_SETS.values():
        for f in config["data"][key]:
            if f not in out:
                out.append(f)
    return out


def load_group_ids_for(csv_path, raw: Optional[bytes] = None,
                       consumed: Optional[Dict[str, str]] = None, config: Optional[dict] = None) -> pd.DataFrame:
    """
    ``data_groups.load_group_ids`` on ``csv_path`` (pinned provenance checked on
    the bytes consumed). ``raw``: the cohort bytes already read by the
    caller (not read again); ``consumed`` receives ``raw_csv`` /
    ``raw_metadata_xlsx`` / ``canonical_csv`` -> SHA-256 of the bytes consumed.
    """
    import data_groups
    return data_groups.load_group_ids(canonical_csv=csv_path,
                                      raw_bytes=None if raw is None else {"canonical_csv": raw},
                                      consumed_sha256=consumed, config=config)


def load_ashrae_context(config: dict, csv_path=None, ids: Optional[pd.DataFrame] = None) -> AshraeContext:
    """
    Load the cohort with the category normalisation (every feature set, every
    target) and its group identifiers. ``ids`` injects identifiers (tests);
    otherwise they are replayed from the raw files with the pinned provenance.

    Every input file is read once and parsed from those bytes; ``input_sha256``
    holds the SHA-256 of the bytes consumed,
    which ``bind_inputs`` then requires of the files.
    """
    from unified_data_loader import UnifiedDataLoader
    csv_path = Path(csv_path or config["data"]["datasets"][DATASET]["csv"])
    input_sha256: Dict[str, str] = {}
    raw, _ = read_consumed(csv_path, input_sha256, "cohort_csv")       # one read
    loader = UnifiedDataLoader(all_features(config), list(rs.TARGETS))
    data = loader.load_and_validate(str(csv_path), DATASET, category_normalisation=True, raw=raw)
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise RunnerError("split indices are row positions of the cohort CSV: the loader must keep a RangeIndex")
    dates = pd.read_csv(io.BytesIO(raw), usecols=["Date"])["Date"].astype(str).to_numpy()
    input_files = {"cohort_csv": str(csv_path)}
    if ids is None:
        import data_groups
        seen: Dict[str, str] = {}
        ids = load_group_ids_for(csv_path, raw, seen, config=config)
        paths = data_groups.default_paths(config)
        input_files.update({"raw_csv": str(paths["raw_csv"]), "raw_metadata_xlsx": str(paths["raw_metadata_xlsx"])})
        if seen.get("canonical_csv") != input_sha256["cohort_csv"]:
            raise RunnerError(f"{csv_path}: the identifiers were aligned with other cohort bytes than those parsed")
        input_sha256.update({k: seen[k] for k in ("raw_csv", "raw_metadata_xlsx")})
    ids = ids.reset_index(drop=True)
    if len(ids) != len(data) or len(dates) != len(data):
        raise RunnerError(f"identifiers ({len(ids)}) / dates ({len(dates)}) not aligned with the {len(data)} rows")
    for col in ("record_id", "publication", "building_id", "subject_id"):
        if col not in ids:
            raise RunnerError(f"group identifiers lack {col!r}")
    if ids["publication"].isna().any() or ids["building_id"].isna().any():
        raise RunnerError("every row needs a publication and a building_id")
    p26 = pd.DataFrame({
        "record_id": ids["record_id"].to_numpy(),
        "publication": ids["publication"].astype(str).to_numpy(),
        "building_id": ids["building_id"].to_numpy(),
        "subject_key": subject_keys(ids).to_numpy(),
        "date": dates,
    })
    return AshraeContext(data=data, ids=ids, p26=p26, csv_path=csv_path, input_files=input_files,
                         category_coverage=list(loader.category_coverage.get(DATASET) or []),
                         input_sha256=input_sha256)


def write_category_coverage(exp_dir, ctx: AshraeContext) -> Path:
    """Category coverage: raw-value mapping / rejection log of the cohort, hashed with the outputs."""
    return _write_json(Path(exp_dir) / f"category_coverage_{DATASET}.json", ctx.category_coverage)


# ======================================================================================
# PREDICTIONS FILES
# ======================================================================================
def predictions_name(experiment: str, target: str, model: str, seed: int, split_id: str) -> str:
    return f"{experiment}__{COHORT}__{target}__{model}__seed{int(seed)}__{split_id}.csv.gz"


def prediction_frame(model, X: pd.DataFrame, y_true, *, experiment: str, target: str, feature_set: str,
                     model_name: str, seed: int, split_id: str, p26_ids: pd.DataFrame,
                     categories: Mapping[str, set], cat_cols, probabilities: bool = True) -> pd.DataFrame:
    """
    Predictions frame of one evaluated set: one row per vote, in the order of ``X``
    (= serialised split order); ``row_idx`` = ``X.index`` (row position in the
    cohort CSV); ``p26_ids`` holds the identifiers of these rows (same order).
    ``probabilities=False`` (SVM) writes NaN in every ``p_<class>`` column.
    """
    labels = rs.ontology(target)
    n = len(X)
    row_idx = np.asarray(X.index, dtype=np.int64)
    if len(p26_ids) != n or not np.array_equal(np.asarray(p26_ids.index, dtype=np.int64), row_idx):
        raise RunnerError(f"{model_name}: identifiers not aligned with the evaluated rows")
    yt = rs.check_labels(np.asarray(y_true), labels, f"{model_name} y_true")
    yp = rs.check_labels(np.asarray(model.predict(X)), labels, f"{model_name} y_pred")
    if len(yt) != n or len(yp) != n:
        raise RunnerError(f"{model_name}: {len(yp)} predictions / {len(yt)} labels for {n} rows")
    proba = ontology_probabilities(model, X, target, n) if probabilities else np.full((n, len(labels)), np.nan)
    flag, oov_vars = oov_columns(categories, X, cat_cols)
    df = pd.DataFrame({
        "experiment": experiment, "cohort": COHORT, "target": target, "feature_set": feature_set,
        "model": model_name, "seed": int(seed), "split_id": split_id, "row_idx": row_idx,
        "record_id": p26_ids["record_id"].to_numpy(), "publication": p26_ids["publication"].to_numpy(),
        "building_id": p26_ids["building_id"].to_numpy(), "subject_key": p26_ids["subject_key"].to_numpy(),
        "date": p26_ids["date"].to_numpy(), "y_true": yt, "y_pred": yp,
    })
    for j, c in enumerate(labels):
        df[f"p_{c}"] = proba[:, j]
    df["oov_flag"] = flag
    df["oov_vars"] = oov_vars
    return df[prediction_columns(target)]


def write_predictions(exp_dir, frame: pd.DataFrame) -> Path:
    """Gzip CSV with a fixed gzip mtime (byte-reproducible); an existing file is never overwritten."""
    if frame.empty:
        raise RunnerError("empty predictions frame")
    first = frame.iloc[0]
    path = Path(exp_dir) / "predictions" / predictions_name(first["experiment"], first["target"], first["model"],
                                                            first["seed"], first["split_id"])
    if path.exists():
        raise FileExistsError(f"{path} exists: outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_json(path, obj) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))
    return path


def _write_csv(path, df: pd.DataFrame) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


METRIC_COLUMNS = ("n", "n_correct", "accuracy", "balanced_accuracy", "balanced_accuracy_single_class",
                  "macro_f1", "qwk", "qwk_undefined")


def metric_row(metrics: Mapping, **keys) -> dict:
    """Flat row of ``revision_stats.classification_metrics`` (confusion matrix as JSON)."""
    row = dict(keys)
    row.update({k: metrics[k] for k in METRIC_COLUMNS})
    row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    row["recall"] = json.dumps({str(k): v for k, v in metrics["recall"].items()})
    row["support"] = json.dumps({str(k): v for k, v in metrics["support"].items()})
    return row


def majority_rows(y_train, y_eval, target: str, **keys) -> List[dict]:
    """Deployable (training majority) and descriptive baselines as metrics rows."""
    b = rs.majority_baselines(y_train, y_eval, target)
    rows = []
    for name, cls in (("Majority_deployable", b["deployable_class"]),
                      ("Majority_descriptive", b["descriptive_class"])):
        m = rs.classification_metrics(y_eval, np.full(len(y_eval), cls), target)
        rows.append(metric_row(m, target=target, **keys, model=name, seed=pd.NA, majority_class=int(cls)))
    return rows


# ======================================================================================
# ESTIMATORS
# ======================================================================================
def build_classical(family: str, num_cols, cat_cols, seed: int, config: dict, compute: dict):
    """``(pipeline, grid)`` of a classical family with the config hyperparameters."""
    from models_classic import get_models
    name = FAMILIES[family]
    return get_models(num_cols, cat_cols, seed=seed, hparams=config["classical_hparams"],
                      cat_vocabulary=vocabulary_for(cat_cols), compute=compute, families=[name])[name]


def build_deep(family: str, num_cols, cat_cols, seed: int, config: dict, compute: dict):
    """``(estimator, grid)`` of ANN / FTT with the config hyperparameters on ``compute.torch_device``."""
    from models_deep import get_torch_models
    return get_torch_models(num_cols, cat_cols, config["deep_hparams"], seed=seed,
                            cat_vocabulary=vocabulary_for(cat_cols), compute=compute)[FAMILIES[family]]


def fit_fixed(family: str, num_cols, cat_cols, seed: int, config: dict, compute: dict, X, y):
    """RF / XGB fitted directly with the single config point (E2, E4); identical to the E1 grid refit."""
    if family not in TREE_FAMILIES:
        raise RunnerError(f"{family}: E2 and E4 fit RF and XGB only")
    pipe, grid = build_classical(family, num_cols, cat_cols, seed, config, compute)
    pipe.set_params(**fixed_params(grid))
    return pipe.fit(X, np.asarray(y, dtype=float))


def _model_dir(exp_dir, target: str, family: str, seed: int, feature_set: Optional[str] = None) -> Path:
    sub = f"{feature_set}/" if feature_set else ""
    return Path(exp_dir) / target / f"{sub}{family}" / f"seed_{seed}"


# ======================================================================================
# JOB FUNCTIONS (module level: picklable for the GPU worker processes)
# ======================================================================================
def e1_fit_job(seed, *, compute, family, target, X_train, y_train, X_test, y_test, p26_test, cv_splits,
               folds_sha256, config, exp_dir, experiment):
    """
    One E1 fit: search/CV on the shared folds (classical:
    ``run_grid_search_on_train``; ANN/FTT: ``select_best``, with early
    stopping), evaluation on the archived test rows, predictions file, per-model
    outputs and model file. Returns paths and metrics (no CUDA tensor).
    """
    import joblib
    feature_cols = list(X_train.columns)
    num_cols, cat_cols = split_feature_types(X_train, feature_cols)
    name = FAMILIES[family]
    mdir = _model_dir(exp_dir, target, family, seed)
    mdir.mkdir(parents=True, exist_ok=True)
    tag = f"{target}__{family}__seed{seed}"
    if family in DEEP_FAMILIES:
        from models_deep import early_stopping_record, save_torch_curves, select_best
        est, grid = build_deep(family, num_cols, cat_cols, seed, config, compute)
        model, best_params, fold_df, stats_df, per_param = select_best(target, name, est, grid, X_train, y_train,
                                                                        cv_splits=cv_splits)
        save_torch_curves(model, mdir / f"{tag}_training_curves.csv")
        _write_json(mdir / f"{tag}_early_stopping.json", early_stopping_record(model))
        _write_json(mdir / f"{tag}_param_grid_cv.json", per_param)
        categories = model.pp_.train_categories_
    else:
        from models_classic import run_grid_search_on_train
        pipe, grid = build_classical(family, num_cols, cat_cols, seed, config, compute)
        model, best_params, fold_df, stats_df, _ = run_grid_search_on_train(
            target, name, pipe, grid, X_train, y_train, refit_metric="f1_macro", cv_splits=cv_splits,
            verbose=0, compute=compute)
        categories = train_categories(X_train, cat_cols)
    used = set(fold_df["folds_sha256"])
    if used != {folds_sha256}:
        raise revision_folds.FoldIntegrityError(f"{tag}: folds {sorted(used)} != shared folds {folds_sha256}")
    frame = prediction_frame(model, X_test, y_test, experiment=experiment, target=target, feature_set="F17",
                             model_name=family, seed=seed, split_id=ARCHIVED_SPLIT_ID, p26_ids=p26_test,
                             categories=categories, cat_cols=cat_cols, probabilities=family != "SVM")
    pred_path = write_predictions(exp_dir, frame)
    metrics = rs.classification_metrics(frame["y_true"], frame["y_pred"], target)
    _write_json(mdir / f"{tag}_test_metrics.json", metrics)
    _write_json(mdir / f"{tag}_best_params.json", best_params)
    _write_csv(mdir / f"{tag}_fold_summary.csv", fold_df.assign(seed=seed))
    _write_csv(mdir / f"{tag}_fold_statistics.csv", stats_df.assign(seed=seed))
    model_path = mdir / f"{tag}.joblib"
    joblib.dump(model, model_path)
    return {"family": family, "target": target, "seed": int(seed), "predictions": str(pred_path),
            "model_file": str(model_path), "metrics": metric_row(metrics), "best_params": best_params,
            "folds_sha256": folds_sha256, "n_oov_test_votes": int(frame["oov_flag"].sum())}


def fixed_fit_job(seed, *, compute, family, target, feature_set, X_train, y_train, X_eval, y_eval, p26_eval,
                  config, exp_dir, experiment, split_id):
    """One RF / XGB fit with constant hyperparameters (E2, E4), evaluated on ``X_eval``."""
    num_cols, cat_cols = split_feature_types(X_train, list(X_train.columns))
    model = fit_fixed(family, num_cols, cat_cols, seed, config, compute, X_train, y_train)
    frame = prediction_frame(model, X_eval, y_eval, experiment=experiment, target=target, feature_set=feature_set,
                             model_name=family, seed=seed, split_id=split_id, p26_ids=p26_eval,
                             categories=train_categories(X_train, cat_cols), cat_cols=cat_cols)
    pred_path = write_predictions(exp_dir, frame)
    metrics = rs.classification_metrics(frame["y_true"], frame["y_pred"], target)
    return {"family": family, "target": target, "seed": int(seed), "feature_set": feature_set,
            "split_id": split_id, "predictions": str(pred_path), "metrics": metric_row(metrics),
            "n_train": int(len(X_train)), "n_eval": int(len(X_eval)),
            "n_oov_votes": int(frame["oov_flag"].sum())}


# ======================================================================================
# ENVIRONMENT AND MANIFEST
# ======================================================================================
def _normalise_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(lock_file=LOCK_FILE, text: Optional[str] = None) -> Dict[str, str]:
    """Exact pins of the lock file; ``text``: its content already read (one read)."""
    pins = {}
    for line in (Path(lock_file).read_text() if text is None else text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise RunnerError(f"lock line is not an exact pin: {line!r}")
        pins[_normalise_dist(name)] = version
    return pins


def pip_freeze() -> List[str]:
    """Installed distributions as sorted ``name==version`` lines (``pip freeze`` content)."""
    seen = {}
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if name:
            seen.setdefault(_normalise_dist(name), f"{name}=={d.version}")
    return sorted(seen.values(), key=str.lower)


def _nvidia_driver() -> Optional[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True,
                             timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"gpus": [line.strip() for line in out]}


def environment_record(lock_file=LOCK_FILE) -> dict:
    """Lock-file hash, ``pip freeze`` (+ hash), differences with the lock, Python, CUDA, driver."""
    freeze = pip_freeze()
    installed = {_normalise_dist(line.split("==")[0]): line.split("==", 1)[1] for line in freeze}
    lock_raw, lock_sha256 = read_consumed(lock_file)       # pins and hash from one read
    pins = lock_pins(lock_file, text=lock_raw.decode())
    diffs = sorted(f"{n}: lock {v}, installed {installed.get(n)}" for n, v in pins.items() if installed.get(n) != v)
    # installed but not pinned: recorded; only the installer tooling may be unpinned
    unpinned = sorted(f"{n}=={v}" for n, v in installed.items() if n not in pins)
    unexpected = [u for u in unpinned if u.split("==")[0] not in UNPINNED_ALLOWED]
    try:
        import torch
        cuda = {"torch_cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
                "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    except ImportError:
        cuda = {"torch_cuda_runtime": None, "cuda_available": False, "cudnn": None}
    return {
        "lock_file": str(lock_file),
        "lock_file_sha256": lock_sha256,
        "pip_freeze": freeze,
        "pip_freeze_sha256": _sha256_bytes("\n".join(freeze).encode()),
        "differences_from_lock": diffs,
        "unpinned_installed": unpinned,
        "unexpected_unpinned": unexpected,
        "matches_lock": not diffs and not unexpected,
        "python": platform.python_version(),
        "executable": sys.executable,
        **cuda,
        "nvidia": _nvidia_driver(),
    }


def check_locked_environment(env: dict, dry_run: bool) -> None:
    """A real run needs the locked environment (no difference with the lock file)."""
    if not dry_run and not env["matches_lock"]:
        raise RunnerError(f"environment differs from {env['lock_file']} (locked environment): "
                          f"{env['differences_from_lock'][:5]}; not in the lock: "
                          f"{(env.get('unexpected_unpinned') or [])[:5]}")


def check_config_unmodified(config: dict, dry_run: bool, file_config: Optional[dict] = None) -> None:
    """
    A real run uses ``config.yaml`` as is (overrides are for dry runs and tests
    only). ``file_config``: ``config.yaml`` parsed from the bytes whose hash the
    startup record keeps (``_prepare``); default: read it now.
    """
    file_config = load_config(CONFIG_PATH) if file_config is None else file_config
    if not dry_run and _json_sha256(config) != _json_sha256(file_config):
        raise RunnerError("the configuration differs from config.yaml: a real run uses config.yaml unchanged")


def phase1_provenance() -> dict:
    """
    Provenance of the unchanged Phase 1 artefact: read-only
    hash of the grid, compared with its pinned value (``matches_pinned``).
    """
    grid = PHASE1_DIR / "experiment_grid_full.csv"
    grid_sha = file_sha256(grid) if grid.exists() else None

    def rel(p: Path) -> str:
        return str(p.relative_to(REPO_DIR)) if REPO_DIR in p.parents else str(p)
    return {"path": rel(PHASE1_DIR),
            "experiment_grid_full_sha256": grid_sha,
            "experiment_grid_full_sha256_pinned": PINNED_PHASE1_GRID_SHA256,
            "matches_pinned": grid_sha == PINNED_PHASE1_GRID_SHA256,
            "rerun": False}


def check_phase1_artefact(dry_run: bool) -> dict:
    """
    A real run carries the provenance of the unchanged Phase 1 grid: the grid
    must have its pinned SHA-256 (a regenerated or missing grid is refused,
    never recorded silently). Checked
    before the experiment and again when its manifest is written.
    """
    prov = phase1_provenance()
    if not dry_run and not prov["matches_pinned"]:
        raise RunnerError(f"the Phase 1 artefact {prov['path']}/experiment_grid_full.csv does not have its pinned "
                          f"SHA-256 (grid {prov['experiment_grid_full_sha256']}, pinned "
                          f"{PINNED_PHASE1_GRID_SHA256}): Phase 1 is not rerun")
    return prov


def _job_records(results) -> List[dict]:
    return job_records(results)


def scheduler_record(compute: dict, gpu_concurrency: Optional[int]) -> dict:
    """Effective scheduling of a run: GPU jobs at once (override or config), threads, devices."""
    return {"gpu_concurrency": int(compute["gpu_concurrency"] if gpu_concurrency is None else gpu_concurrency),
            "gpu_concurrency_source": "config" if gpu_concurrency is None else "override",
            "gpu_concurrency_config": int(compute["gpu_concurrency"]), "gpu_job_threads": gpu_job_threads(compute),
            "cpu_jobs": int(compute["cpu_jobs"]), "xgb_device": compute["xgb_device"],
            "torch_device": compute["torch_device"]}


def untracked_python_files() -> Optional[List[str]]:
    """Untracked ``*.py`` files of the repository (invisible to the commit hash; recorded in the manifest)."""
    try:
        out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"], cwd=REPO_DIR,
                             capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted(line for line in out.splitlines() if line.strip())


def check_clean_checkout(provenance: dict, dry_run: bool) -> None:
    """A real run needs a checkout whose tracked files equal the recorded commit (clean checkout)."""
    if not dry_run and provenance.get("git_tracked_changes") is not False:
        state = "unknown" if provenance.get("git_tracked_changes") is None else "modified tracked files"
        raise RunnerError(f"git working tree {state}: a real run needs a clean checkout of the recorded commit "
                          "(commit first)")


def _output_hashes(exp_dir: Path) -> Dict[str, str]:
    return {str(p.relative_to(exp_dir)): file_sha256(p) for p in sorted(exp_dir.rglob("*"))
            if p.is_file() and p.name != "run_manifest.json"}


@dataclass(frozen=True)
class StartupRecord:
    """
    Immutable provenance of an experiment at its start: code provenance
    (``run_provenance``: commit, tracked changes, packages), SHA-256 of the
    effective configuration and of ``config.yaml``, and ``(role, path, sha256)``
    of every input consumed (``bind_inputs``). ``write_manifest`` compares the end state with it and records it.
    """
    provenance_json: str
    config_used_sha256: str
    config_file_sha256: str
    inputs: Tuple[Tuple[str, str, str], ...] = ()
    #: ``untracked_python_files()`` at the start (JSON): compared at the end, refused in a real run
    untracked_python_json: str = "[]"
    #: the configuration used equals ``config.yaml`` parsed from the bytes hashed in ``config_file_sha256``
    config_is_file: Optional[bool] = None
    #: ``_json_sha256`` of ``config.yaml`` as parsed from those bytes (import-time copies are compared with it)
    config_file_parsed_sha256: Optional[str] = None

    @property
    def untracked_python(self) -> Optional[List[str]]:
        return json.loads(self.untracked_python_json)

    @property
    def provenance(self) -> dict:
        return json.loads(self.provenance_json)

    @property
    def inputs_sha256(self) -> Dict[str, str]:
        return {role: sha for role, _, sha in self.inputs}


def check_no_untracked_python(untracked: Optional[List[str]], dry_run: bool) -> None:
    """A real run refuses untracked ``*.py`` files (importable code outside the recorded commit)."""
    if not dry_run and untracked != []:
        state = "unknown" if untracked is None else ", ".join(untracked)
        raise RunnerError(f"untracked Python files ({state}): a real run needs a clean checkout of the recorded "
                          "commit (commit, ignore or remove them)")


def read_config_file() -> Tuple[dict, str]:
    """
    ``config.yaml`` parsed, and the SHA-256 of the bytes parsed (one read).
    A run reads it **once**, where its configuration is first parsed
    (``run_revision._config``, or ``_run_config`` for a library call without a
    configuration), and carries this pair to ``_prepare`` (``config_file``):
    the recorded ``config_file_sha256`` is the hash of the bytes the
    configuration used was parsed from, never of a later read of the path.
    """
    raw, sha = read_consumed(CONFIG_PATH)
    return yaml.safe_load(raw), sha


def _run_config(config: Optional[dict], config_file: Optional[Tuple[dict, str]]
                ) -> Tuple[dict, Optional[Tuple[dict, str]]]:
    """
    The configuration of a ``run_e6`` / ``run_e7`` / ``run_e9b`` call and its
    ``config.yaml`` record: without ``config``, ``config.yaml`` is read once
    (``read_config_file``) and the configuration used is a copy of what those
    bytes parsed to; with ``config``, ``config_file`` is passed on
    (``None``: ``_prepare`` reads it, for direct library calls and tests).
    """
    if config is None:
        import copy
        config_file = read_config_file() if config_file is None else config_file
        config = copy.deepcopy(config_file[0])
    return config, config_file


#: modules that parse ``config.yaml`` at import time (relative to the working directory) into ``CONFIG``
IMPORT_TIME_CONFIG_MODULES = ("models_classic", "models_deep", "transfert", "models")


def check_import_time_configs(file_config_sha256: Optional[str], dry_run: bool) -> None:
    """
    A real run refuses an import-time ``CONFIG`` (``IMPORT_TIME_CONFIG_MODULES``,
    loaded when this is called) that is not ``config.yaml`` as parsed from the
    bytes recorded in ``config_file_sha256`` (``file_config_sha256``: hash of
    that parsed dictionary). The revision
    runners pass their configuration explicitly; this keeps a separately read
    copy from differing silently. Called by ``_prepare`` and at finalisation.
    """
    if dry_run or file_config_sha256 is None:
        return
    bad = [name for name in IMPORT_TIME_CONFIG_MODULES
           if name in sys.modules and isinstance(getattr(sys.modules[name], "CONFIG", None), dict)
           and _json_sha256(sys.modules[name].CONFIG) != file_config_sha256]
    if bad:
        raise RunnerError(f"import-time configuration of {', '.join(bad)} differs from config.yaml as parsed from "
                          "the recorded bytes: a real run starts from the worktree root, config.yaml "
                          "unchanged")


def startup_record(provenance: Mapping, config: dict,
                   config_file: Optional[Tuple[dict, str]] = None) -> StartupRecord:
    """``config_file``: ``read_config_file()`` already taken by the caller (default: read now)."""
    file_config, file_sha = read_config_file() if config_file is None else config_file
    return StartupRecord(provenance_json=json.dumps(provenance, sort_keys=True, default=str),
                         config_used_sha256=_json_sha256(config), config_file_sha256=file_sha,
                         untracked_python_json=json.dumps(untracked_python_files()),
                         config_is_file=_json_sha256(config) == _json_sha256(file_config),
                         config_file_parsed_sha256=_json_sha256(file_config))


def bind_inputs(startup: StartupRecord, inputs: Mapping[str, str],
                loaded_sha256: Optional[Mapping[str, str]] = None) -> StartupRecord:
    """
    A new ``StartupRecord`` with the SHA-256 of ``inputs`` (role -> path), taken
    when they are consumed (before any fit). ``loaded_sha256`` gives hashes taken
    when a file was loaded earlier (``AshraeContext.input_sha256``): the file must
    still have them. A role already bound must keep its path and hash.
    """
    bound = {role: (path, sha) for role, path, sha in startup.inputs}
    for role, path in sorted(inputs.items()):
        sha = file_sha256(path)
        if loaded_sha256 and role in loaded_sha256 and loaded_sha256[role] != sha:
            raise RunnerError(f"input {role} ({path}) changed after it was loaded or validated "
                              f"(loaded {loaded_sha256[role]}, now {sha})")
        if role in bound and bound[role] != (str(path), sha):
            raise RunnerError(f"input {role} rebound with another file or content: {bound[role]} -> {(str(path), sha)}")
        bound[role] = (str(path), sha)
    return replace(startup, inputs=tuple((role, path, sha) for role, (path, sha) in sorted(bound.items())))


def check_end_state(startup: StartupRecord, config: dict, inputs: Mapping[str, str], dry_run: bool,
                    env: Optional[Mapping] = None) -> dict:
    """
    Finalisation check, before any manifest or run-index write: a
    real run needs a clean checkout at the end too; for every run, the ending
    code provenance (commit, tracked changes, packages), the effective
    configuration, ``config.yaml`` and every recorded input must equal the
    startup record; the recorded inputs must be exactly the bound ones (an input
    not bound at consumption, or a bound one left out, is refused); with ``env``
    (``environment_record`` of the start), the lock file must still have the
    hash of the bytes whose pins were compared. Raises
    ``RunnerError`` listing every drift; returns the end-state record.
    """
    end = run_provenance()
    check_clean_checkout(end, dry_run)
    untracked = untracked_python_files()
    check_no_untracked_python(untracked, dry_run)
    start = startup.provenance
    drift = [f"provenance {k}: start {start.get(k)!r}, end {end.get(k)!r}"
             for k in sorted(set(start) | set(end)) if start.get(k) != end.get(k)]
    if untracked != startup.untracked_python:
        drift.append(f"untracked Python files: start {startup.untracked_python}, end {untracked}")
    if _json_sha256(config) != startup.config_used_sha256:
        drift.append(f"configuration used: start {startup.config_used_sha256}, end {_json_sha256(config)}")
    try:
        check_import_time_configs(startup.config_file_parsed_sha256, dry_run)
    except RunnerError as exc:
        drift.append(str(exc))
    config_file = file_sha256(CONFIG_PATH) if Path(CONFIG_PATH).exists() else None
    if config_file != startup.config_file_sha256:
        drift.append(f"{CONFIG_PATH}: start {startup.config_file_sha256}, end {config_file}")
    if env is not None and env.get("lock_file"):
        lock_now = file_sha256(env["lock_file"]) if Path(env["lock_file"]).is_file() else None
        if lock_now != env.get("lock_file_sha256"):
            drift.append(f"{env['lock_file']}: pins read from {env.get('lock_file_sha256')}, now {lock_now}")
    bound = {role: (path, sha) for role, path, sha in startup.inputs}
    # the recorded inventory is the bound one: every bound input is re-checked
    for role in sorted(set(bound) - set(inputs)):
        drift.append(f"input {role} ({bound[role][0]}) bound at consumption but missing from the recorded inputs")
    for role, path in sorted(inputs.items()):
        if role not in bound:
            drift.append(f"input {role} ({path}) not hashed when it was consumed")
            continue
        now = file_sha256(path) if Path(path).is_file() else None
        if (str(path), now) != bound[role]:
            drift.append(f"input {role}: start {bound[role]}, end {(str(path), now)}")
    if drift:
        raise RunnerError("the run state changed between its start and its end; no manifest written:\n  "
                          + "\n  ".join(drift))
    return {"git_commit": end.get("git_commit"), "git_tracked_changes": end.get("git_tracked_changes"),
            "checked_utc": _now(), "matches_startup": True}


def write_manifest(exp_dir: Path, run_dir: Path, *, experiment: str, dry_run: bool, command, started: str,
                   config: dict, compute: dict, env: dict, inputs: Mapping[str, str], seeds,
                   startup: StartupRecord, jobs=None, extra: Optional[dict] = None,
                   gpu_concurrency: Optional[int] = None) -> dict:
    """
    ``{exp_dir}/run_manifest.json`` and its entry in ``{run_dir}/manifest.json``
    ``startup`` is the record of ``_prepare`` with the consumed inputs
    bound (``bind_inputs``); any drift from it raises before anything is written
    (``check_end_state``), and the manifest records the startup provenance.
    """
    check_phase1_artefact(dry_run)
    end_state = check_end_state(startup, config, inputs, dry_run, env=env)
    inputs_sha256 = startup.inputs_sha256
    manifest = {
        "format": MANIFEST_FORMAT,
        "experiment": experiment_label(experiment, dry_run),
        "dry_run": bool(dry_run),
        "command": list(command) if command is not None else None,
        "cwd": os.getcwd(),
        "started_utc": started,
        "ended_utc": _now(),
        "provenance": startup.provenance,                 # code that ran (startup record)
        "provenance_end_check": end_state,
        "environment": env,
        "config_file": str(CONFIG_PATH),
        "config_file_sha256": startup.config_file_sha256,
        "config_used_sha256": startup.config_used_sha256,
        "config_is_file": startup.config_is_file,           # from the bytes hashed at the start
        "compute": dict(compute),
        "scheduler": scheduler_record(compute, gpu_concurrency),
        "untracked_python_files": startup.untracked_python,     # start = end (check_end_state)
        "inputs_sha256": {k: inputs_sha256[k] for k in sorted(inputs)},
        "inputs": dict(inputs),
        "seeds": seeds,
        "jobs": jobs or [],
        **(extra or {}),
        "outputs_sha256": _output_hashes(exp_dir),
    }
    # completion is marked only after the run-index registration: the
    # manifest is written under a pending name, registered (with its final name and SHA-256), then renamed
    # under the same index lock; a refused or failed registration, or a failed rename (the index entry is
    # then rolled back), leaves no run_manifest.json and no index entry
    path = exp_dir / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} exists: outputs are never overwritten")
    pending = _write_json(exp_dir / "run_manifest.json.pending", manifest)
    try:
        update_run_index(run_dir, manifest["experiment"], path, manifest_sha256=file_sha256(pending),
                         complete=lambda: os.replace(pending, path), undo=lambda: os.replace(path, pending))
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return manifest


def update_run_index(run_dir, experiment: str, manifest_path: Path, manifest_sha256: Optional[str] = None,
                     complete: Optional[Callable[[], None]] = None,
                     undo: Optional[Callable[[], None]] = None) -> dict:
    """
    Add one experiment to ``{run_dir}/manifest.json`` (created with the Phase 1
    provenance). ``manifest_sha256`` (default: hash of ``manifest_path``) lets
    ``write_manifest`` register a manifest before it gets its final name.

    With ``complete`` (that final rename), the entry is written with
    ``status: "registered"``, then ``complete`` runs and the entry is set to
    ``status: "complete"``, all under the same lock; if ``complete`` or the
    status write fails, ``undo`` reverts a completed rename and the index is
    restored to its previous bytes (or removed if it did not exist) before the
    error propagates. Without ``complete`` (the
    manifest already has its final name), the entry is written ``complete``.

    An entry already present is never overwritten, except one that never
    reached ``complete`` and whose manifest file does not exist (the process
    died between registration and completion); that entry is kept under
    ``superseded_unconfirmed`` with the time it was replaced. A ``complete``
    entry (or an entry without status) is refused even if its directory was
    moved or deleted: use a new run directory.
    """
    import fcntl
    index_path = Path(run_dir) / "manifest.json"
    # exclusive lock around the read-modify-write: experiments of one run directory may finish at once
    with open(Path(run_dir) / ".manifest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            before = index_path.read_bytes() if index_path.exists() else None
            if before is not None:
                index = json.loads(before)
            else:
                index = {"format": RUN_INDEX_FORMAT, "created_utc": _now(), "phase1": phase1_provenance(),
                         "experiments": {}}
            old = index["experiments"].get(experiment)
            if old is not None:
                unconfirmed = (isinstance(old, dict) and old.get("status") == "registered" and old.get("manifest")
                               and not (Path(run_dir) / old["manifest"]).exists())
                if not unconfirmed:
                    gone = (isinstance(old, dict) and old.get("manifest")
                            and not (Path(run_dir) / old["manifest"]).exists())
                    raise FileExistsError(
                        f"{index_path}: experiment {experiment} already recorded as complete (never overwritten)"
                        + ("; its manifest was moved or deleted: a completed experiment is never superseded, "
                           "use a new run directory" if gone else ""))
                index.setdefault("superseded_unconfirmed", []).append(
                    {"experiment": experiment, **old, "superseded_utc": _now()})
            entry = {"manifest": str(Path(manifest_path).relative_to(run_dir)),
                     "manifest_sha256": manifest_sha256 or file_sha256(manifest_path),
                     "recorded_utc": _now(), "status": "registered" if complete is not None else "complete"}
            index["experiments"][experiment] = entry
            tmp = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")

            def _write(obj):
                tmp.write_text(json.dumps(obj, indent=1))
                os.replace(tmp, index_path)
            _write(index)
            if complete is not None:
                renamed = False
                try:
                    complete()
                    renamed = True
                    entry.update(status="complete", completed_utc=_now())
                    _write(index)
                except BaseException:
                    try:                     # the index is restored even if the undo fails
                        if renamed and undo is not None:
                            undo()
                    finally:
                        if before is None:
                            index_path.unlink(missing_ok=True)
                        else:
                            tmp.write_bytes(before)
                            os.replace(tmp, index_path)
                    raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return index


@dataclass
class Prepared:
    exp_dir: Path
    compute: dict
    env: dict
    provenance: dict
    started: str
    startup: StartupRecord


def check_gpu_concurrency(compute: dict, gpu_concurrency: Optional[int], dry_run: bool) -> None:
    """
    One GPU job at a time unless ``config.yaml`` says otherwise
    (``compute.gpu_concurrency``, set before the run). A real run refuses an
    override above the configured value; dry runs and tests may use any value >= 1.
    """
    if gpu_concurrency is None:
        return
    if int(gpu_concurrency) < 1:
        raise RunnerError(f"gpu_concurrency must be >= 1, got {gpu_concurrency}")
    if not dry_run and int(gpu_concurrency) > int(compute["gpu_concurrency"]):
        raise RunnerError(f"--gpu_concurrency {gpu_concurrency} exceeds compute.gpu_concurrency = "
                          f"{compute['gpu_concurrency']} of config.yaml: the frozen schedule runs one GPU job at a time; "
                          "a higher value needs a change of config.yaml")


def _prepare(run_dir, experiment: str, config: dict, dry_run: bool, compute=None,
             gpu_concurrency: Optional[int] = None,
             config_file: Optional[Tuple[dict, str]] = None) -> Prepared:
    """
    Common checks before anything is written: frozen artefact, empty experiment
    directory, ``config.yaml`` unchanged, locked environment, clean
    checkout and no GPU concurrency above ``config.yaml`` — the last four
    for real runs only. ``config_file``: ``read_config_file()`` of the read
    ``config`` was parsed from (``run_revision``), recorded as is;
    ``None`` (direct library calls, tests): read now.
    """
    check_not_protected(run_dir)
    exp_dir = check_empty_output_dir(experiment_dir(run_dir, experiment, dry_run))
    if config_file is None:
        config_file = read_config_file()        # checked and hashed from one read
    check_config_unmodified(config, dry_run, file_config=config_file[0])
    env = environment_record()
    check_locked_environment(env, dry_run)
    prov = run_provenance()
    check_clean_checkout(prov, dry_run)
    startup = startup_record(prov, config, config_file)
    check_no_untracked_python(startup.untracked_python, dry_run)
    check_import_time_configs(startup.config_file_parsed_sha256, dry_run)
    check_phase1_artefact(dry_run)
    compute = load_compute(config) if compute is None else compute
    check_gpu_concurrency(compute, gpu_concurrency, dry_run)
    return Prepared(exp_dir, compute, env, prov, _now(), startup)


def _split(ctx: AshraeContext, split_dir, target: str, consumed: Optional[Dict[str, str]] = None):
    """
    The archived split of ``target``; ``consumed`` (role -> SHA-256) receives
    ``split:{target}`` = SHA-256 of the bytes parsed, which ``bind_inputs`` then
    requires of the file.
    """
    seen: Dict[str, str] = {}
    tr, te = revision_folds.load_archived_split(split_dir, target, ctx.data[target], consumed_sha256=seen)
    if consumed is not None:
        consumed[f"split:{target}"] = next(iter(seen.values()))
    return tr, te


def _split_hashes(split_dir, targets) -> Dict[str, str]:
    """Paths of the archived split files, by role ``split:{target}`` (hashed by ``bind_inputs``)."""
    return {f"split:{t}": str(Path(split_dir) / f"{t}_split_indices.json") for t in targets}


def check_consumed_inputs(startup: StartupRecord, consumed: Mapping[str, str], reader: str) -> None:
    """
    Inputs read by a delegated reader after they were bound (E6, E7, E9b): the
    SHA-256 of the bytes that reader parsed (role -> SHA-256) must be the bound
    one; a role it read but that was never
    bound is refused too. Raises ``RunnerError``; nothing is written.
    """
    bound = startup.inputs_sha256
    bad = [f"{role}: parsed {sha}, bound {bound.get(role)}" for role, sha in sorted(consumed.items())
           if bound.get(role) != sha]
    if bad:
        raise RunnerError(f"{reader} parsed inputs other than the bound ones; no manifest written:\n  "
                          + "\n  ".join(bad))


# ======================================================================================
# E1 AS A SOURCE (E4, E6, E7)
# ======================================================================================
def load_e1_source(e1_dir, *, dry_run: bool, files: Iterable = (), inputs: Optional[Mapping[str, str]] = None,
                   paired: Optional[Mapping[str, dict]] = None, validated: Optional[dict] = None) -> dict:
    """
    The E1 run read by a later experiment, validated before any write or fit:

    * ``{e1_dir}/run_manifest.json`` exists (E1 completed) and is an E1 manifest;
      a real run refuses a dry-run E1 (tiny models);
    * every file of ``files`` is an output recorded by that manifest, with the
      recorded SHA-256 (not modified after E1);
    * every input of ``inputs`` (role -> path, e.g. ``cohort_csv``, ``split:{t}``)
      has the SHA-256 recorded by E1 for that role (same data, same train/test rows);
    * ``paired`` (E4: the F17 side of a paired contrast) with keys
      ``config``, ``compute``, ``env``, ``provenance`` of the current run: same
      dry-run state, configuration used (hyperparameters, features), XGB device,
      lock file and ``pip freeze``, git commit; a real run also needs both
      checkouts clean.

    ``validated`` (optional dict) receives ``str(path) -> SHA-256`` of every
    file checked (the manifest: the hash of the bytes parsed; ``files`` and
    ``inputs``: the hashes recorded by E1). The caller binds these hashes, not
    a fresh hash taken after validation.

    Returns the E1 manifest; raises ``RunnerError`` otherwise.
    """
    e1_dir = Path(e1_dir)
    path = e1_dir / "run_manifest.json"
    if not path.exists():
        raise RunnerError(f"{path} missing: E1 did not complete (no recorded provenance); rerun E1")
    raw = path.read_bytes()                     # one read: the bytes hashed are the bytes validated
    man_sha256 = _sha256_bytes(raw)
    man = json.loads(raw)
    checked = {str(path): man_sha256}
    if man.get("format") != MANIFEST_FORMAT or man.get("experiment") != experiment_label("E1", bool(man.get("dry_run"))):
        raise RunnerError(f"{path}: not an E1 run manifest")
    if man["dry_run"] and not dry_run:
        raise RunnerError(f"{path}: E1 dry run (tiny models, never results); a real run needs a real E1 source")
    # the manifest is the one indexed by its run directory
    index_path = e1_dir.parent / "manifest.json"
    entry = None
    if index_path.exists():
        entry = (json.loads(index_path.read_text()).get("experiments") or {}).get(man["experiment"])
        if entry is not None and (e1_dir.parent / entry["manifest"]).resolve() != path.resolve():
            entry = None
    if entry is None:
        if not dry_run:
            raise RunnerError(f"{path}: not recorded in the run index {index_path}; a real run reads an E1 "
                              "written by run_revision.py in its run directory")
    elif man_sha256 != entry["manifest_sha256"]:
        raise RunnerError(f"{path}: SHA-256 differs from the run index {index_path} (modified after E1)")
    outs = man.get("outputs_sha256") or {}
    for f in files:
        f = Path(f)
        try:
            rel = str(f.relative_to(e1_dir))
        except ValueError:
            raise RunnerError(f"{f} is not inside the E1 directory {e1_dir}") from None
        if rel not in outs:
            raise RunnerError(f"{f}: not an output recorded by the E1 manifest {path}")
        if file_sha256(f) != outs[rel]:
            raise RunnerError(f"{f}: SHA-256 differs from the E1 manifest (modified after E1)")
        checked[str(f)] = outs[rel]
    recorded = man.get("inputs_sha256") or {}
    for role, p in (inputs or {}).items():
        if role not in recorded:
            raise RunnerError(f"{path}: no {role!r} input recorded (data / split provenance)")
        if file_sha256(p) != recorded[role]:
            raise RunnerError(f"{role}: {p} differs from the E1 input (other data or train/test rows)")
        checked[str(Path(p))] = recorded[role]
    if paired is not None:
        checks = {
            "dry_run": (man["dry_run"], bool(dry_run)),
            "config_used_sha256": (man.get("config_used_sha256"), _json_sha256(paired["config"])),
            "compute.xgb_device": ((man.get("compute") or {}).get("xgb_device"), paired["compute"]["xgb_device"]),
            "environment.lock_file_sha256": (
                release_file_sha256((man.get("environment") or {}).get("lock_file_sha256")),
                release_file_sha256(paired["env"]["lock_file_sha256"])),
            "environment.pip_freeze_sha256": ((man.get("environment") or {}).get("pip_freeze_sha256"),
                                              paired["env"]["pip_freeze_sha256"]),
            "provenance.git_commit": ((man.get("provenance") or {}).get("git_commit"),
                                      paired["provenance"].get("git_commit")),
        }
        bad = sorted(k for k, (a, b) in checks.items() if a != b)
        if bad:
            raise RunnerError(f"{path}: the E1 fits are not the paired F17 side of this run (differs in {bad}); "
                              "the paired comparison needs the same code, configuration, environment, rows and seeds")
        if not dry_run and (man.get("provenance") or {}).get("git_tracked_changes") is not False:
            raise RunnerError(f"{path}: E1 ran from a modified checkout; a paired real run needs a clean one")
    if validated is not None:
        validated.update(checked)
    return man


def expected_input_hashes(inputs: Mapping[str, str], validated: Mapping[str, str],
                          loaded: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """
    ``role -> SHA-256`` that ``bind_inputs`` must find for ``inputs``: the hashes
    taken when a file was loaded (``loaded``, by role) and those validated by
    ``load_e1_source`` (``validated``, by path). Two sources disagreeing for one
    role raise.
    """
    out = dict(loaded or {})
    for role, p in inputs.items():
        sha = validated.get(str(Path(p)))
        if sha is None:
            continue
        if role in out and out[role] != sha:
            raise RunnerError(f"input {role} ({p}): loaded with SHA-256 {out[role]}, validated against E1 as {sha}")
        out[role] = sha
    return out


def check_e1_predictions(path, *, experiment: str, target: str, model: str, seed: int, row_idx, y_true,
                         record_id, consumed: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    An E1 predictions file is the F17 fit of (target, model, seed) on these rows
    and labels. Parsed from one read; ``consumed`` receives
    ``e1:{file name}`` -> SHA-256 of the bytes checked. Returns the
    frame parsed from those bytes (E10 compares its predictions with it).
    """
    raw, _ = read_consumed(path, consumed, f"e1:{Path(path).name}")
    fr = pd.read_csv(io.BytesIO(raw), compression="gzip" if str(path).endswith(".gz") else None)
    expect = {"experiment": experiment, "cohort": COHORT, "target": target, "feature_set": "F17",
              "model": model, "seed": int(seed), "split_id": ARCHIVED_SPLIT_ID}
    for col, v in expect.items():
        if set(fr[col].tolist()) != {v}:
            raise RunnerError(f"{path}: {col} {sorted(set(fr[col].astype(str)))} != {v!r}")
    if fr["row_idx"].tolist() != [int(i) for i in row_idx]:
        raise RunnerError(f"{path}: not the archived test rows of {target}, in split order")
    if not np.array_equal(fr["y_true"].to_numpy(dtype=float), np.asarray(y_true, dtype=float)):
        raise RunnerError(f"{path}: y_true differs from the current {target} labels")
    if fr["record_id"].astype(str).tolist() != pd.Series(np.asarray(record_id)).astype(str).tolist():
        raise RunnerError(f"{path}: record_id differs from the current cohort rows")
    return fr


# ======================================================================================
# E1 — Phase 2 corrected, in-domain
# ======================================================================================
def run_e1(ctx: AshraeContext, *, run_dir, config: dict, split_dir, folds_dir=None,
           targets: Sequence[str] = rs.TARGETS, families: Sequence[str] = E1_FAMILIES,
           tree_seeds=revision_folds.TREE_SEEDS, ann_seeds=(42,), ftt_seeds=(42,),
           compute: Optional[dict] = None, gpu_concurrency: Optional[int] = None, dry_run: bool = False,
           command=None,
           config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """Run E1 and write its outputs (module docstring); returns the run manifest."""
    run_dir = Path(run_dir)
    pre = _prepare(run_dir, "E1", config, dry_run, compute, gpu_concurrency, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    experiment = experiment_label("E1", dry_run)
    revision_folds.check_cv_config(config)
    families = list(families)
    unknown = sorted(set(families) - set(E1_FAMILIES))
    if unknown or not families:
        raise RunnerError(f"E1 families must be a non-empty subset of {E1_FAMILIES}, got {families}")
    seeds_all = revision_folds.model_seeds(tree_seeds, ann_seeds, ftt_seeds)    # frozen seeds, raises otherwise
    seeds = {f: list(seeds_all[FAMILIES[f]]) for f in families}
    for t in targets:
        rs.ontology(t)
    folds_dir = Path(folds_dir) if folds_dir is not None else exp_dir / "folds"
    check_not_protected(folds_dir)
    features = features_for(config, "F17")
    data = ctx.data
    write_category_coverage(exp_dir, ctx)

    jobs, fold_info, inputs, baseline_rows = [], {}, dict(ctx.input_files), []
    inputs.update(_split_hashes(split_dir, targets))
    consumed: Dict[str, str] = {}                # role -> SHA-256 of the bytes parsed
    for t in targets:
        tr, te = _split(ctx, split_dir, t, consumed)
        X_all, y_all = data[features], data[t]
        X_train, X_test = X_all.iloc[tr], X_all.iloc[te]
        y_train, y_test = y_all.iloc[tr], y_all.iloc[te]
        seen: Dict[str, str] = {}
        folds = revision_folds.ensure_folds(folds_dir, DATASET, t, tr, y_train,       # once, before any fit
                                            consumed_sha256=seen)
        fpath = revision_folds.folds_path(folds_dir, t)
        consumed[f"folds:{t}"] = seen[str(fpath)]
        inputs[f"folds:{t}"] = str(fpath)
        fold_info[t] = {"path": str(fpath), "k": folds.k, "seed": folds.seed,
                        "fold_vector_sha256": folds.sha256, "train_idx_sha256": folds.train_idx_sha256}
        # E6 reads the split of the zero-shot / source models from the E1 directory
        sdir = exp_dir / t / "splits"
        _write_json(sdir / f"{t}_split_indices.json",
                    {"target": t, "train_idx": tr.tolist(), "test_idx": te.tolist(), "features": features})
        _write_json(sdir / f"{t}_folds.json", folds.to_json())
        num_cols, cat_cols = split_feature_types(X_train, features)
        _write_json(sdir / f"{t}_oov_report.json", oov_summary(oov_report(X_train, X_test, cat_cols)))
        baseline_rows += majority_rows(y_train, y_test, t, experiment=experiment, feature_set="F17",
                                       split_id=ARCHIVED_SPLIT_ID)
        p26_test = ctx.p26.iloc[te]
        for fam in families:
            for s in seeds[fam]:
                jobs.append(Job(key=f"{t}|{fam}|{s}", kind=job_kind(FAMILIES[fam], compute), fn=e1_fit_job, seed=s,
                                kwargs=dict(family=fam, target=t, X_train=X_train, y_train=y_train, X_test=X_test,
                                            y_test=y_test, p26_test=p26_test, cv_splits=folds.splits(),
                                            folds_sha256=folds.sha256, config=config, exp_dir=exp_dir,
                                            experiment=experiment)))

    # consumed inputs, before any fit, bound to the hashes of the bytes parsed
    startup = bind_inputs(pre.startup, inputs, {**ctx.input_sha256, **consumed})
    results = run_jobs(jobs, compute=compute, gpu_concurrency=gpu_concurrency)
    rows = []
    for r in results.values():
        v = r.value
        rows.append({"experiment": experiment, "target": v["target"], "feature_set": "F17",
                     "split_id": ARCHIVED_SPLIT_ID, "model": v["family"], "seed": v["seed"],
                     **{k: x for k, x in v["metrics"].items()}, "predictions": Path(v["predictions"]).name})
    _write_csv(exp_dir / "metrics.csv", pd.DataFrame(rows + baseline_rows))
    _assemble_best_models(exp_dir, results)

    inventory = sorted(f"{f}:{s}" for f in families for s in seeds[f])
    return write_manifest(exp_dir, run_dir, experiment="E1", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"split": revision_folds.SPLIT_SEED, "folds": revision_folds.FOLDS_SEED,
                                 "internal_validation": 42, "models": seeds},
                          jobs=_job_records(results),
                          extra={"targets": list(targets), "families": families, "folds": fold_info,
                                 "ml_inventory": inventory}, gpu_concurrency=gpu_concurrency)


def _assemble_best_models(exp_dir: Path, results) -> None:
    """``best_{rf,xgb,svm,ft}_by_target_seed{s}.joblib`` and the unsuffixed seed-42 files (E6 inputs)."""
    import joblib
    by = {}
    for r in results.values():
        v = r.value
        key = BEST_KEYS.get(v["family"])
        if key is not None:
            by.setdefault(key, {}).setdefault(v["seed"], {})[v["target"]] = v["model_file"]
    for key, by_seed in by.items():
        for seed, files in sorted(by_seed.items()):
            models = {t: joblib.load(p) for t, p in files.items()}
            joblib.dump(models, exp_dir / f"best_{key}_by_target_seed{seed}.joblib")
            if seed == MODEL_SEED:
                joblib.dump(models, exp_dir / f"best_{key}_by_target.joblib")


def _fixed_jobs_metrics(results, experiment: str, extra_keys: Iterable[str] = ()) -> List[dict]:
    rows = []
    for r in results.values():
        v = r.value
        rows.append({"experiment": experiment, "target": v["target"], "feature_set": v["feature_set"],
                     "split_id": v["split_id"], "model": v["family"], "seed": v["seed"],
                     **{k: v[k] for k in extra_keys}, **v["metrics"], "n_train": v["n_train"],
                     "n_oov_votes": v["n_oov_votes"], "predictions": Path(v["predictions"]).name})
    return rows


def _check_tree_seeds(seeds) -> List[int]:
    return list(revision_folds.check_model_seeds("RandomForest", seeds))


# ======================================================================================
# E2 — leave-one-publication-out
# ======================================================================================
def lopo_split_id(k: int) -> str:
    return f"lopo_{k:02d}"


def run_e2(ctx: AshraeContext, *, run_dir, config: dict, targets: Sequence[str] = rs.TARGETS,
           families: Sequence[str] = TREE_FAMILIES, compute: Optional[dict] = None,
           gpu_concurrency: Optional[int] = None, dry_run: bool = False, command=None,
           config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E2: one fold per publication (ascending names, ``lopo_{k:02d}``); training
    rows = every row of the other publications (not restricted to
    ``train_idx``); preprocessing fitted inside each fit; every row predicted
    once; OOV votes kept and counted per variable and publication; seed 42.
    """
    run_dir = Path(run_dir)
    pre = _prepare(run_dir, "E2", config, dry_run, compute, gpu_concurrency, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    experiment = experiment_label("E2", dry_run)
    families = list(families)
    if not families or set(families) - set(TREE_FAMILIES):
        raise RunnerError(f"E2 fits RF and XGB only, got {families}")
    features = features_for(config, "F17")
    pubs = ctx.p26["publication"].to_numpy()
    publications = sorted(set(pubs.tolist()))
    if len(publications) < 2:
        raise RunnerError("LOPO needs at least two publications")
    X_all = ctx.data[features]
    num_cols, cat_cols = split_feature_types(X_all, features)
    write_category_coverage(exp_dir, ctx)
    folds, jobs, baseline_rows = [], [], []
    for k, pub in enumerate(publications):
        test = np.flatnonzero(pubs == pub)
        train = np.flatnonzero(pubs != pub)
        rep = oov_report(X_all.iloc[train], X_all.iloc[test], cat_cols)
        folds.append({"fold": k, "split_id": lopo_split_id(k), "publication": pub, "n_train": int(len(train)),
                      "n_test": int(len(test)),
                      "n_test_buildings": int(ctx.p26["building_id"].iloc[test].nunique()),
                      "n_oov_votes": rep["n_oov_votes"], "oov_counts_per_variable": rep["counts_per_variable"],
                      "oov_values": rep["oov_values"]})
        for t in targets:
            y = ctx.data[t].to_numpy(dtype=float)
            baseline_rows += majority_rows(y[train], y[test], t, experiment=experiment,
                                           feature_set="F17", split_id=lopo_split_id(k), publication=pub)
            for fam in families:
                jobs.append(Job(key=f"{t}|{fam}|{lopo_split_id(k)}", kind=job_kind(FAMILIES[fam], compute),
                                fn=fixed_fit_job, seed=LOPO_SEED,
                                kwargs=dict(family=fam, target=t, feature_set="F17", X_train=X_all.iloc[train],
                                            y_train=y[train], X_eval=X_all.iloc[test], y_eval=y[test],
                                            p26_eval=ctx.p26.iloc[test], config=config, exp_dir=exp_dir,
                                            experiment=experiment, split_id=lopo_split_id(k))))
    _write_json(exp_dir / "lopo_folds.json", {"publications": publications, "folds": folds})
    inputs = dict(ctx.input_files)
    startup = bind_inputs(pre.startup, inputs, ctx.input_sha256)      # consumed inputs, before any fit
    results = run_jobs(jobs, compute=compute, gpu_concurrency=gpu_concurrency)
    pub_of = {f["split_id"]: f["publication"] for f in folds}
    rows = _fixed_jobs_metrics(results, experiment)
    for row in rows:
        row["publication"] = pub_of[row["split_id"]]
    _write_csv(exp_dir / "metrics.csv", pd.DataFrame(rows + baseline_rows))
    _write_csv(exp_dir / "aggregates.csv", _lopo_aggregates(exp_dir, rows, len(ctx.data)))
    return write_manifest(exp_dir, run_dir, experiment="E2", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"models": {f: [LOPO_SEED] for f in families}}, jobs=_job_records(results),
                          extra={"targets": list(targets), "families": families, "n_folds": len(publications)},
                          gpu_concurrency=gpu_concurrency)


def _lopo_aggregates(exp_dir: Path, rows: List[dict], n_rows: int) -> pd.DataFrame:
    """Micro (all predicted rows pooled) and macro (mean per-publication accuracy) aggregates, descriptive."""
    out = []
    df = pd.DataFrame(rows)
    for (t, fam), g in df.groupby(["target", "model"], sort=True):
        preds = pd.concat([pd.read_csv(exp_dir / "predictions" / f) for f in g["predictions"]], ignore_index=True)
        if len(preds) != n_rows or preds["row_idx"].duplicated().any() or set(preds["row_idx"]) != set(range(n_rows)):
            raise RunnerError(f"E2 {t} {fam}: every row must be predicted exactly once by the LOPO folds")
        m = rs.classification_metrics(preds["y_true"], preds["y_pred"], t)
        out.append({"target": t, "model": fam, "aggregate": "micro", "n": m["n"], "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]})
        out.append({"target": t, "model": fam, "aggregate": "macro_over_publications", "n": int(len(g)),
                    "accuracy": float(g["accuracy"].mean()), "balanced_accuracy": float(g["balanced_accuracy"].mean()),
                    "macro_f1": float(g["macro_f1"].mean())})
    return pd.DataFrame(out)


# ======================================================================================
# E4 — EMA ablation
# ======================================================================================
def _pair_sha256(path: Path, f17: Path, startup: StartupRecord) -> str:
    """SHA-256 of a pairs.csv file: the bound (consumed) hash for the E1 F17 side, else the output file's."""
    return startup.inputs_sha256[f"e1:{f17.name}"] if path == f17 else file_sha256(path)


def e1_prediction_file(e1_dir, target: str, family: str, seed: int, dry_run: bool) -> Path:
    return Path(e1_dir) / "predictions" / predictions_name(experiment_label("E1", dry_run), target, family, seed,
                                                           ARCHIVED_SPLIT_ID)


def run_ablation(ctx: AshraeContext, *, experiment: str, run_dir, e1_dir, config: dict, split_dir,
                 targets: Sequence[str] = rs.TARGETS, families: Sequence[str] = TREE_FAMILIES,
                 seeds=revision_folds.TREE_SEEDS, compute: Optional[dict] = None,
                 gpu_concurrency: Optional[int] = None, dry_run: bool = False, command=None,
                 config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E4 (F12 vs F17): fits the non-F17 set (RF, XGB × seeds ×
    targets, constant hyperparameters, archived split) and pairs it with the E1
    F17 predictions of the same target, family and seed (same rows, checked).
    """
    if experiment not in ABLATIONS:
        raise ValueError(f"unknown ablation {experiment!r}; expected one of {sorted(ABLATIONS)}")
    spec = ABLATIONS[experiment]
    run_dir = Path(run_dir)
    pre = _prepare(run_dir, experiment, config, dry_run, compute, gpu_concurrency, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    label = experiment_label(experiment, dry_run)
    families = list(families)
    if not families or set(families) - set(TREE_FAMILIES):
        raise RunnerError(f"{experiment} fits RF and XGB only, got {families}")
    seeds = _check_tree_seeds(seeds)
    features = features_for(config, spec["fitted"])
    # shared F17 side: the E1 run must be complete and comparable (manifest: same code, configuration,
    # XGB device, environment, data and split hashes), and every E1 predictions file must be the recorded
    # F17 fit of (target, family, seed) on the archived test rows with the current labels; checked before any fit
    inputs = dict(ctx.input_files)
    inputs.update(_split_hashes(split_dir, targets))
    consumed: Dict[str, str] = {}                # role -> SHA-256 of the bytes parsed
    splits = {t: _split(ctx, split_dir, t, consumed) for t in targets}
    e1_dir = Path(e1_dir)
    shared = {}
    for t in targets:
        for fam in families:
            for s in seeds:
                f = e1_prediction_file(e1_dir, t, fam, s, dry_run)
                if not f.exists():
                    raise RunnerError(f"{f}: E1 F17 predictions missing ({experiment} pairs its fits with E1)")
                shared[(t, fam, s)] = f
    validated: Dict[str, str] = {}
    e1_man = load_e1_source(e1_dir, dry_run=dry_run, files=shared.values(),
                            inputs={k: v for k, v in inputs.items() if k == "cohort_csv" or k.startswith("split:")},
                            paired={"config": config, "compute": compute, "env": env, "provenance": pre.provenance},
                            validated=validated)
    e1_manifest_sha = validated[str(e1_dir / "run_manifest.json")]     # the bytes validated
    for (t, fam, s), f in shared.items():
        te = splits[t][1]
        check_e1_predictions(f, experiment=e1_man["experiment"], target=t, model=fam, seed=s, row_idx=te,
                             y_true=ctx.data[t].to_numpy(dtype=float)[te], record_id=ctx.p26["record_id"].to_numpy()[te],
                             consumed=consumed)
        inputs[f"e1:{f.name}"] = str(f)
    inputs["e1:run_manifest.json"] = str(e1_dir / "run_manifest.json")
    # consumed inputs, before any fit, bound to the hashes validated against E1
    startup = bind_inputs(pre.startup, inputs,
                          expected_input_hashes(inputs, validated, {**ctx.input_sha256, **consumed}))
    write_category_coverage(exp_dir, ctx)
    jobs, baseline_rows = [], []
    for t in targets:
        tr, te = splits[t]
        X = ctx.data[features]
        y = ctx.data[t].to_numpy(dtype=float)
        baseline_rows += majority_rows(y[tr], y[te], t, experiment=label, feature_set=spec["fitted"],
                                       split_id=ARCHIVED_SPLIT_ID)
        for fam in families:
            for s in seeds:
                jobs.append(Job(key=f"{t}|{fam}|{s}", kind=job_kind(FAMILIES[fam], compute), fn=fixed_fit_job, seed=s,
                                kwargs=dict(family=fam, target=t, feature_set=spec["fitted"], X_train=X.iloc[tr],
                                            y_train=y[tr], X_eval=X.iloc[te], y_eval=y[te], p26_eval=ctx.p26.iloc[te],
                                            config=config, exp_dir=exp_dir, experiment=label,
                                            split_id=ARCHIVED_SPLIT_ID)))
    results = run_jobs(jobs, compute=compute, gpu_concurrency=gpu_concurrency)
    pairs = []
    for r in results.values():
        v = r.value
        f17 = shared[(v["target"], v["family"], v["seed"])]
        fitted = Path(v["predictions"])
        files = {"F17": f17, spec["fitted"]: fitted}
        pairs.append({"target": v["target"], "model": v["family"], "seed": v["seed"],
                      "base_feature_set": spec["base"], "extended_feature_set": spec["extended"],
                      "base_file": str(files[spec["base"]]), "extended_file": str(files[spec["extended"]]),
                      "base_sha256": _pair_sha256(files[spec["base"]], f17, startup),
                      "extended_sha256": _pair_sha256(files[spec["extended"]], f17, startup),
                      "e1_manifest_sha256": e1_manifest_sha})
    _write_csv(exp_dir / "pairs.csv", pd.DataFrame(pairs).sort_values(["target", "model", "seed"]))
    _write_csv(exp_dir / "metrics.csv", pd.DataFrame(_fixed_jobs_metrics(results, label) + baseline_rows))
    return write_manifest(exp_dir, run_dir, experiment=experiment, dry_run=dry_run, command=command,
                          started=started, config=config, compute=compute, env=env, inputs=inputs,
                          startup=startup,
                          seeds={"models": {f: seeds for f in families}}, jobs=_job_records(results),
                          extra={"targets": list(targets), "families": families, "ablation": spec,
                                 "fitted_features": features, "e1_dir": str(e1_dir),
                                 "e1_manifest_sha256": e1_manifest_sha,
                                 "shared_f17_predictions_sha256": {str(p): startup.inputs_sha256[f"e1:{p.name}"]
                                                                   for p in shared.values()}},
                          gpu_concurrency=gpu_concurrency)


# ======================================================================================
# E7 — driver of the corrected indices module
# ======================================================================================
def e7_argv(exp_dir, *, data_csv, split_dir, pred_dir, expected_ml: Sequence[str],
            skip_group_ids: bool = False, dry_run: bool = False) -> List[str]:
    """
    The argv of ``indicateurs_classiques.main`` for E7. The positional output
    directory comes first: ``--expected_ml`` takes one or more values, so a
    directory placed after the list is taken as one more MODEL:SEED whenever no
    flag closes the list, which is the case of a real run (no ``--dry_run``).
    """
    argv = [str(exp_dir), "--pmv_only", "--data_csv", str(data_csv), "--splits", str(split_dir),
            "--ml_predictions", str(pred_dir), "--expected_ml", *expected_ml]
    if skip_group_ids:
        argv.append("--skip_group_ids")
    if dry_run:
        argv.append("--dry_run")
    return argv


def run_e7(*, run_dir, e1_dir, data_csv, split_dir, expected_ml: Optional[Sequence[str]] = None,
           skip_group_ids: bool = False, dry_run: bool = False, command=None, config: Optional[dict] = None,
           config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E7 on the archived test rows (``indicateurs_classiques.main``): PMV/aPMV
    on the PMV mask, the E1 predictions on the same mask and outside it.

    The E1 source is validated first (``load_e1_source``), even when
    ``expected_ml`` is given: its manifest must exist, a real E7 refuses a
    dry-run E1, the predictions directory must hold exactly the predictions
    files recorded by the E1 manifest with their SHA-256, and the cohort CSV and
    archived splits must be those of E1. ``expected_ml`` defaults to the
    ``ml_inventory`` of the E1 manifest. A dry run labels its files and rows
    ``E7_dry_run``. ``skip_group_ids`` (synthetic tests) is refused in a real
    run: the building-clustered analyses need the identifiers.
    """
    import indicateurs_classiques as ic
    run_dir = Path(run_dir)
    if skip_group_ids and not dry_run:
        raise RunnerError("--skip_group_ids drops record_id / publication / building_id: dry runs only")
    config, config_file = _run_config(config, config_file)
    pre = _prepare(run_dir, "E7", config, dry_run, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    e1_dir = Path(e1_dir)
    pred_dir = e1_dir / "predictions"
    if not pred_dir.is_dir():
        raise RunnerError(f"{pred_dir}: E1 predictions missing (E7 compares the indices with E1)")
    pred_files = sorted(set(pred_dir.rglob("*.csv.gz")) | set(pred_dir.rglob("*.csv")))   # files E7 reads
    source_inputs = {"cohort_csv": str(data_csv), **_split_hashes(split_dir, rs.TARGETS)}
    validated: Dict[str, str] = {}
    e1_man = load_e1_source(e1_dir, dry_run=dry_run, files=pred_files, inputs=source_inputs, validated=validated)
    recorded = sorted(k for k in e1_man["outputs_sha256"] if k.startswith("predictions/"))
    if sorted(str(f.relative_to(e1_dir)) for f in pred_files) != recorded:
        raise RunnerError(f"{pred_dir}: predictions files differ from those recorded by the E1 manifest")
    if expected_ml is None:
        expected_ml = e1_man["ml_inventory"]
    argv = e7_argv(exp_dir, data_csv=data_csv, split_dir=split_dir, pred_dir=pred_dir,
                   expected_ml=expected_ml, skip_group_ids=skip_group_ids, dry_run=dry_run)
    inputs = {"data_csv": str(data_csv), **_split_hashes(split_dir, rs.TARGETS),
              "e1:run_manifest.json": str(e1_dir / "run_manifest.json")}
    if not skip_group_ids:                       # files replayed by data_groups for the identifiers
        import data_groups
        paths = data_groups.default_paths(config)
        inputs.update({"raw_csv": str(paths["raw_csv"]), "raw_metadata_xlsx": str(paths["raw_metadata_xlsx"])})
    inputs.update({f"e1:{p.name}": str(p) for p in pred_files})
    # the files E7 reads, bound before it reads them to the hashes validated against E1
    startup = bind_inputs(pre.startup, inputs, expected_input_hashes(inputs, validated))
    parsed: Dict[str, str] = {}
    if ic.main(argv, e1_manifest=e1_man, consumed_sha256=parsed, config=config) != 0:
        raise RunnerError("indicateurs_classiques.main failed")
    # every input E7 parsed (cohort, splits, identifier files, E1 predictions) is reported with the hash of the
    # bytes parsed, which must be the bound one
    expected_roles = sorted(set(inputs) - {"e1:run_manifest.json"})        # the manifest: load_e1_source
    if sorted(parsed) != expected_roles:
        raise RunnerError(f"indicateurs_classiques reports parsed hashes for {sorted(parsed)}, "
                          f"expected {expected_roles}")
    check_consumed_inputs(startup, parsed, "E7 (indicateurs_classiques)")
    return write_manifest(exp_dir, run_dir, experiment="E7", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"models": None},
                          extra={"c4_argv": argv, "expected_ml": list(expected_ml), "e1_dir": str(e1_dir),
                                 "e1_experiment": e1_man["experiment"],
                                 "e1_manifest_sha256": validated[str(e1_dir / "run_manifest.json")]})


# ======================================================================================
# E7b — SET / PET (PTS_SET, PTS_PET) and ePMV beside E7
# ======================================================================================
E7B_TARGETS = ("thermal_sensation", "TSV_3p")       # TSV-7 and TSV-3 only (as in the paper)


def e7b_argv(exp_dir, *, data_csv, split_dir, pred_dir, expected_ml: Sequence[str],
             skip_group_ids: bool = False, dry_run: bool = False) -> List[str]:
    """The argv of ``indicateurs_classiques.main`` for E7b (output directory first, as ``e7_argv``)."""
    argv = [str(exp_dir), "--indices", "set_pet_epmv", "--data_csv", str(data_csv), "--splits", str(split_dir),
            "--ml_predictions", str(pred_dir), "--expected_ml", *expected_ml]
    if skip_group_ids:
        argv.append("--skip_group_ids")
    if dry_run:
        argv.append("--dry_run")
    return argv


#: Decided before the run: the ePMV expectancy factor e is applied unbounded
#: (``indicateurs_classiques.fit_epmv_e``) and flagged outside [0.5, 1.0] (``e_flag_outside_range`` in
#: calibration.json, the ``epmv_e_flag_outside_range`` column of metrics.csv and the manifest ``epmv_e``);
#: an ePMV beyond the TSV-7 scale (e > 1) is saturated to the TSV-7 scale like the other restored indices.


def run_e7b(*, run_dir, e1_dir, data_csv, split_dir, expected_ml: Optional[Sequence[str]] = None,
            skip_group_ids: bool = False, dry_run: bool = False, command=None, config: Optional[dict] = None,
            config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E7b = ``run_e7`` with ``--indices set_pet_epmv``: SET with
    PTS_SET, PET with PTS_PET and ePMV on the archived test rows of TSV-7 and
    TSV-3, each on its validity mask, the E1 predictions on the same mask and
    outside it; calibrations on the training rows of the evaluated target.
    Same E1 validation, input binding and manifest as E7; directory
    ``E7b_indices``; run-index entry ``E7b``. The E1 predictions of every
    target are validated and bound (as E7); indicateurs_classiques evaluates TSV-7 and
    TSV-3 only.
    """
    import indicateurs_classiques as ic
    run_dir = Path(run_dir)
    if skip_group_ids and not dry_run:
        raise RunnerError("--skip_group_ids drops record_id / publication / building_id: dry runs only")
    config, config_file = _run_config(config, config_file)
    pre = _prepare(run_dir, "E7b", config, dry_run, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    e1_dir = Path(e1_dir)
    pred_dir = e1_dir / "predictions"
    if not pred_dir.is_dir():
        raise RunnerError(f"{pred_dir}: E1 predictions missing (E7b compares the indices with E1)")
    pred_files = sorted(set(pred_dir.rglob("*.csv.gz")) | set(pred_dir.rglob("*.csv")))   # files E7 reads
    source_inputs = {"cohort_csv": str(data_csv), **_split_hashes(split_dir, E7B_TARGETS)}
    validated: Dict[str, str] = {}
    e1_man = load_e1_source(e1_dir, dry_run=dry_run, files=pred_files, inputs=source_inputs, validated=validated)
    recorded = sorted(k for k in e1_man["outputs_sha256"] if k.startswith("predictions/"))
    if sorted(str(f.relative_to(e1_dir)) for f in pred_files) != recorded:
        raise RunnerError(f"{pred_dir}: predictions files differ from those recorded by the E1 manifest")
    if expected_ml is None:
        expected_ml = e1_man["ml_inventory"]
    argv = e7b_argv(exp_dir, data_csv=data_csv, split_dir=split_dir, pred_dir=pred_dir,
                    expected_ml=expected_ml, skip_group_ids=skip_group_ids, dry_run=dry_run)
    inputs = {"data_csv": str(data_csv), **_split_hashes(split_dir, E7B_TARGETS),
              "e1:run_manifest.json": str(e1_dir / "run_manifest.json")}
    if not skip_group_ids:                       # files replayed by data_groups for the identifiers
        import data_groups
        paths = data_groups.default_paths(config)
        inputs.update({"raw_csv": str(paths["raw_csv"]), "raw_metadata_xlsx": str(paths["raw_metadata_xlsx"])})
    inputs.update({f"e1:{p.name}": str(p) for p in pred_files})
    startup = bind_inputs(pre.startup, inputs, expected_input_hashes(inputs, validated))
    parsed: Dict[str, str] = {}
    if ic.main(argv, e1_manifest=e1_man, consumed_sha256=parsed, config=config) != 0:
        raise RunnerError("indicateurs_classiques.main failed")
    expected_roles = sorted(set(inputs) - {"e1:run_manifest.json"})        # the manifest: load_e1_source
    if sorted(parsed) != expected_roles:
        raise RunnerError(f"indicateurs_classiques reports parsed hashes for {sorted(parsed)}, "
                          f"expected {expected_roles}")
    check_consumed_inputs(startup, parsed, "E7b (indicateurs_classiques)")
    # the ePMV factor e of each target (unbounded) and its flag outside [0.5, 1.0], in the manifest
    calibration = json.loads((exp_dir / "calibration.json").read_text())
    epmv_e = {t: {k: calibration[t]["ePMV"].get(k) for k in ("status", "e", "e_flag_outside_range", "e_flag_range")}
              for t in sorted(calibration)}
    return write_manifest(exp_dir, run_dir, experiment="E7b", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"models": None},
                          extra={"epmv_e": epmv_e,
                                 "c4_argv": argv, "targets": list(E7B_TARGETS), "expected_ml": list(expected_ml),
                                 "e1_dir": str(e1_dir), "e1_experiment": e1_man["experiment"],
                                 "e1_manifest_sha256": validated[str(e1_dir / "run_manifest.json")]})


# ======================================================================================
# E10 — FTT-embedding + RF / XGBoost heads
# ======================================================================================
#: paper table (Method section): the head hyperparameters a real E10 must read from config.yaml
PAPER_HYBRID_HEADS = {
    "RF": {"n_estimators": 225, "max_depth": 25, "min_samples_split": 10, "min_samples_leaf": 2,
           "class_weight": None},
    "XGBoost": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.7,
                "colsample_bytree": 0.7},
}
E10_FTT_SEEDS = (42, 43, 44)                  # one head pair per E1 FTT seed
#: hybrid.run_heads_on_ft_and_save head -> (sub-directory, predictions-file model name)
E10_HEADS = {"RF": ("RF", "FTT_RF"), "XGBoost": ("XGBoost", "FTT_XGB")}
E10_FEATURE_SET = "FTTemb_F17"
#: tolerance of the CPU identity check against the E1 predictions file
E10_PROBA_ATOL = 1e-6
E10_LOGIT_ATOL = 1e-5


def e1_ftt_model_file(e1_dir, target: str, seed: int) -> Path:
    """Per-target E1 FTT model file (recorded in the E1 manifest ``outputs_sha256``)."""
    return Path(e1_dir) / target / "FTT" / f"seed_{seed}" / f"{target}__FTT__seed{seed}.joblib"


def check_paper_head_hparams(hparams: Optional[Mapping], dry_run: bool) -> dict:
    """
    Head hyperparameters of E10; returns them. ``config.yaml`` has no ``hybrid``
    section (kept byte-identical to the run config of the other experiments):
    a real run (``hparams`` None) takes
    the paper table ``PAPER_HYBRID_HEADS``; a dry run gets
    ``DRY_RUN_HPARAMS["hybrid"]`` injected by ``run_revision._config``. Values
    given to a real run must equal the paper table.
    """
    if hparams is None and not dry_run:
        hparams = {"heads": PAPER_HYBRID_HEADS}
    heads = (hparams or {}).get("heads") or {}
    if not dry_run and {k: dict(heads.get(k) or {}) for k in PAPER_HYBRID_HEADS} != PAPER_HYBRID_HEADS:
        raise RunnerError(f"hybrid.heads {heads} differs from the paper table {PAPER_HYBRID_HEADS}")
    if set(heads) != set(PAPER_HYBRID_HEADS):
        raise RunnerError(f"hybrid.heads must define exactly {sorted(PAPER_HYBRID_HEADS)}, got {sorted(heads)}")
    return {k: dict(v) for k, v in heads.items()}


def check_hybrid_rows(ft, train_idx, test_idx, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    No test row reaches the FTT or the heads: the FTT was fitted
    (internal training and early-stopping rows) on exactly ``train_idx``, the
    heads are fitted on exactly the ``train_idx`` rows in split order, and the
    evaluated rows are exactly ``test_idx``. Raises ``RunnerError``.
    """
    tr, te = np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)
    if np.intersect1d(tr, te).size:
        raise RunnerError("train_idx and test_idx overlap")
    if not np.array_equal(np.asarray(X_train.index, dtype=np.int64), tr):
        raise RunnerError("the head training rows are not the archived train_idx rows in split order")
    if not np.array_equal(np.asarray(X_test.index, dtype=np.int64), te):
        raise RunnerError("the evaluated rows are not the archived test_idx rows in split order")
    if np.intersect1d(np.asarray(X_train.index), te).size:
        raise RunnerError("a test row is in the head training rows")
    fit_rows = np.concatenate([np.asarray(ft.es_train_idx_), np.asarray(ft.es_val_idx_)]).astype(np.int64)
    if np.intersect1d(fit_rows, te).size or not np.array_equal(np.sort(fit_rows), np.sort(tr)):
        raise RunnerError("the E1 FTT was not fitted on exactly the archived training rows")
    return {"n_ftt_fit_rows": int(fit_rows.size), "n_ftt_internal_val": int(len(ft.es_val_idx_)),
            "n_head_train": int(len(tr)), "n_test": int(len(te))}


def _state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_ftt_identity(ft, X_test: pd.DataFrame, target: str, e1_frame: pd.DataFrame) -> dict:
    """
    The CPU-loaded FTT reproduces its E1 predictions file:
    ``y_pred`` exactly, ``p_<class>`` within ``E10_PROBA_ATOL``. Raises
    ``RunnerError`` (no GPU fallback).
    """
    labels = rs.ontology(target)
    yp = rs.check_labels(np.asarray(ft.predict(X_test)), labels, "FTT y_pred (CPU)")
    n_diff = int((yp != e1_frame["y_pred"].to_numpy(dtype=float)).sum())
    proba = ontology_probabilities(ft, X_test, target, len(X_test))
    ref = e1_frame[[f"p_{c}" for c in labels]].to_numpy(dtype=float)
    dp = float(np.max(np.abs(proba - ref))) if proba.size else 0.0
    if n_diff or not dp <= E10_PROBA_ATOL:
        raise RunnerError(f"{target}: the CPU-loaded FTT does not reproduce its E1 predictions "
                          f"({n_diff} y_pred differences, max |dp| {dp:.3g} > {E10_PROBA_ATOL}); E10 stops")
    return {"y_pred_differences": n_diff, "max_abs_dp": dp}


def check_hook(ft, X_test: pd.DataFrame, Z_test_sha256: str) -> dict:
    """
    The hooked tensor is the input of ``to_logits[2]``: a second
    extraction is bitwise equal to the one the heads used and
    ``to_logits[2](Z)`` reproduces the model logits within ``E10_LOGIT_ATOL``.
    """
    import torch
    import hybrid
    from ftt_adaptation import output_head
    Z = hybrid.ft_get_embeddings(ft, X_test)
    if hybrid.array_sha256(Z) != Z_test_sha256:
        raise RunnerError("a second extraction of the test embeddings differs from the first")
    x_cat, x_num = ft.pp_.transform(X_test)
    head = output_head(ft.model_, len(ft.le_y_.classes_))
    ft.model_.eval()
    with torch.no_grad():
        logits = torch.cat([ft.model_(x_cat[i:i + 1024], x_num[i:i + 1024]) for i in range(0, x_cat.size(0), 1024)])
        from_hook = head(torch.from_numpy(Z))
    dl = float((logits - from_hook).abs().max()) if len(Z) else 0.0
    if not dl <= E10_LOGIT_ATOL:
        raise RunnerError(f"to_logits[2](Z) differs from the model logits by {dl:.3g}")
    return {"hook": "forward pre-hook input of model_.to_logits[2] (" + repr(head) + ")",
            "max_abs_logit_from_hook": dl, "Z_test_repeat_equal": True}


def check_e10_cpu_only(stage: str, dry_run: bool) -> bool:
    """
    Nothing of E10 runs on the GPU: a real
    E10 is launched with the GPU visible (the CUDA build of xgboost refuses a
    multi-class CPU fit otherwise), so a real run raises if torch has
    initialised CUDA at ``stage``. Returns ``torch.cuda.is_initialized()``
    (recorded; a dry run only records it).
    """
    import torch
    initialised = bool(torch.cuda.is_initialized())
    if initialised and not dry_run:
        raise RunnerError(f"E10 {stage}: torch initialised CUDA; E10 runs on the CPU only (E10 stops)")
    return initialised


def check_e10_xgb_head_device(estimator, dry_run: bool) -> dict:
    """
    The XGB head was built for and trained on the CPU: a real run
    raises unless ``get_params()['device'] == 'cpu'`` and the booster config
    reports a CPU device. Returns ``{"params_device", "booster_device"}``
    (``None`` when the estimator does not expose them, dry-run stub only).
    """
    params = estimator.get_params().get("device") if hasattr(estimator, "get_params") else None
    booster = None
    if hasattr(estimator, "get_booster"):
        cfg = json.loads(estimator.get_booster().save_config())
        booster = str(cfg["learner"]["generic_param"].get("device", ""))
    if not dry_run and (params != "cpu" or booster is None or not booster.startswith("cpu")):
        raise RunnerError(f"E10 XGB head device: params {params!r}, booster {booster!r}; CPU required (E10 stops)")
    return {"params_device": params, "booster_device": booster}


def run_e10(ctx: AshraeContext, *, run_dir, e1_dir, config: dict, split_dir,
            targets: Sequence[str] = rs.TARGETS, compute: Optional[dict] = None, dry_run: bool = False,
            command=None, config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E10: RF and XGBoost heads (``hybrid.run_heads_on_ft_and_save``,
    restored pre-revision code, single fit) on the embeddings of each E1 FTT
    (one per FTT seed of the E1 inventory; a real run needs 42, 43 and 44 and
    every target), extracted on the CPU from the saved models (never refitted).

    E1 is validated first (``load_e1_source``: manifest in the run index, same
    cohort CSV and archived splits; a real run refuses a dry-run E1). Each model
    file is read once, its SHA-256 compared with the E1 manifest
    ``outputs_sha256`` entry before it is parsed, and those same bytes are
    parsed (``hybrid.load_ftt_cpu``). Before any head fit, for each FTT: the
    FTT fit rows are exactly ``train_idx`` and the head rows ``train_idx`` /
    ``test_idx`` in split order; the CPU model reproduces its E1 predictions
    file. After: the hook reproduces the logits, the network is unchanged and
    the torch thread count is the recorded one. Writes the pre-revision per-head
    files under ``{target}/seed_{s}/{RF,XGBoost}/``, the predictions
    (models ``FTT_RF`` / ``FTT_XGB``, feature set ``FTTemb_F17``), one
    embedding record per FTT, ``metrics.csv``, the manifest and the run-index
    entry ``E10``.
    """
    import torch
    import hybrid
    run_dir = Path(run_dir)
    pre = _prepare(run_dir, "E10", config, dry_run, compute, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    label = experiment_label("E10", dry_run)
    targets = list(targets)
    if not dry_run and sorted(targets) != sorted(rs.TARGETS):
        raise RunnerError(f"a real E10 covers every target {list(rs.TARGETS)}, got {targets}")
    for t in targets:
        rs.ontology(t)
    hparams = {"heads": check_paper_head_hparams(config.get("hybrid"), dry_run)}
    features = features_for(config, "F17")
    inputs = dict(ctx.input_files)
    inputs.update(_split_hashes(split_dir, targets))
    consumed: Dict[str, str] = {}                # role -> SHA-256 of the bytes parsed
    splits = {t: _split(ctx, split_dir, t, consumed) for t in targets}
    e1_dir = Path(e1_dir)
    validated: Dict[str, str] = {}
    e1_man = load_e1_source(e1_dir, dry_run=dry_run,
                            inputs={k: v for k, v in inputs.items() if k == "cohort_csv" or k.startswith("split:")},
                            validated=validated)
    e1_manifest_sha = validated[str(e1_dir / "run_manifest.json")]
    seeds = sorted(int(m.split(":")[1]) for m in e1_man.get("ml_inventory") or [] if m.startswith("FTT:"))
    if not seeds or (not dry_run and tuple(seeds) != E10_FTT_SEEDS):
        raise RunnerError(f"E1 FTT seeds {seeds}: a real E10 needs the E1 FTT of seeds {list(E10_FTT_SEEDS)}")
    outs = e1_man.get("outputs_sha256") or {}
    raw_models: Dict[Tuple[str, int], bytes] = {}
    e1_frames: Dict[Tuple[str, int], pd.DataFrame] = {}
    for t in targets:
        te = splits[t][1]
        for s in seeds:
            f = e1_ftt_model_file(e1_dir, t, s)
            rel = str(f.relative_to(e1_dir))
            if rel not in outs:
                raise RunnerError(f"{f}: not an output recorded by the E1 manifest")
            # one read: the bytes hashed are the bytes compared with E1 and then parsed
            raw, sha = read_consumed(f, consumed, f"e1:{f.name}")
            if sha != outs[rel]:
                raise RunnerError(f"{f}: SHA-256 {sha} differs from the E1 manifest ({outs[rel]}); not parsed")
            validated[str(f)] = outs[rel]
            raw_models[(t, s)] = raw
            inputs[f"e1:{f.name}"] = str(f)
            pf = e1_prediction_file(e1_dir, t, "FTT", s, dry_run)
            rel_p = str(pf.relative_to(e1_dir))
            if rel_p not in outs:
                raise RunnerError(f"{pf}: not an output recorded by the E1 manifest")
            e1_frames[(t, s)] = check_e1_predictions(
                pf, experiment=e1_man["experiment"], target=t, model="FTT", seed=s, row_idx=te,
                y_true=ctx.data[t].to_numpy(dtype=float)[te], record_id=ctx.p26["record_id"].to_numpy()[te],
                consumed=consumed)
            if consumed[f"e1:{pf.name}"] != outs[rel_p]:
                raise RunnerError(f"{pf}: SHA-256 differs from the E1 manifest (modified after E1)")
            validated[str(pf)] = outs[rel_p]
            inputs[f"e1:{pf.name}"] = str(pf)
    inputs["e1:run_manifest.json"] = str(e1_dir / "run_manifest.json")
    # consumed inputs, before any fit, bound to the hashes validated against E1
    startup = bind_inputs(pre.startup, inputs, expected_input_hashes(inputs, validated, {**ctx.input_sha256, **consumed}))
    xgb_params = xgb_compute_params(compute)
    if not dry_run and xgb_params["device"] != "cpu":
        raise RunnerError(f"compute.xgb_device {xgb_params['device']!r}: E10 heads run on the CPU only")
    cuda_initialized = {"start": check_e10_cpu_only("start", dry_run)}
    write_category_coverage(exp_dir, ctx)

    rows, records = [], {}
    for t in targets:
        tr, te = splits[t]
        X = ctx.data[features]
        y = ctx.data[t]
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        num_cols, cat_cols = split_feature_types(X_train, features)
        for s in seeds:
            f = e1_ftt_model_file(e1_dir, t, s)
            ft = hybrid.load_ftt_cpu(raw_models.pop((t, s)))
            threads = torch.get_num_threads()
            sd0 = _state_dict_sha256(ft.model_)
            rows_check = check_hybrid_rows(ft, tr, te, X_train, X_test)
            identity = check_ftt_identity(ft, X_test, t, e1_frames[(t, s)])
            head_rows = hybrid.run_heads_on_ft_and_save(
                ft_clf=ft, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                out_dir=exp_dir / t / f"seed_{s}", target_name=t, tag_prefix="FTemb",
                heads=["RF", "XGBoost"], hparams=hparams, seed=s, compute=compute)
            z = head_rows[0]
            xgb_device = {r["head"]: check_e10_xgb_head_device(r["estimator"], dry_run)
                          for r in head_rows if r["head"] == "XGBoost"}
            hook = check_hook(ft, X_test, z["Z_test_sha256"])
            for r in head_rows:
                sub, model_name = E10_HEADS[r["head"]]
                wrapper = hybrid.FTEmbeddingHead.from_fitted(ft, r["estimator"], r["label_encoder"])
                frame = prediction_frame(wrapper, X_test, y_test, experiment=label, target=t,
                                         feature_set=E10_FEATURE_SET, model_name=model_name, seed=s,
                                         split_id=ARCHIVED_SPLIT_ID, p26_ids=ctx.p26.iloc[te],
                                         categories=ft.pp_.train_categories_, cat_cols=cat_cols)
                if wrapper.last_Z_sha256_ != r["Z_test_sha256"]:
                    raise RunnerError(f"{t} seed {s} {model_name}: test embeddings re-extracted differ")
                tag = "RF" if r["head"] == "RF" else "XGB"
                res = json.loads((exp_dir / t / f"seed_{s}" / sub / f"{t}_FTemb_{tag}_test_results.json").read_text())
                if not np.array_equal(np.asarray(res["y_pred"], dtype=float), frame["y_pred"].to_numpy(dtype=float)):
                    raise RunnerError(f"{t} seed {s} {model_name}: predictions file differs from test_results.json")
                pred_path = write_predictions(exp_dir, frame)
                m = rs.classification_metrics(frame["y_true"], frame["y_pred"], t)
                rows.append({"experiment": label, "target": t, "feature_set": E10_FEATURE_SET,
                             "split_id": ARCHIVED_SPLIT_ID, "model": model_name, "seed": s, **metric_row(m),
                             "predictions": pred_path.name})
            if _state_dict_sha256(ft.model_) != sd0:
                raise RunnerError(f"{t} seed {s}: the FTT changed during E10 (frozen network)")
            if torch.get_num_threads() != threads or threads != hybrid.EMBEDDING_TORCH_THREADS:
                raise RunnerError(f"torch threads {torch.get_num_threads()} != recorded {threads}")
            rec = {"target": t, "seed": s, "source_file": str(f), "source_sha256": startup.inputs_sha256[f"e1:{f.name}"],
                   "device": "cpu", "torch_threads": threads, "torch_version": torch.__version__,
                   "torch_parallel_info": torch.__config__.parallel_info().splitlines(),
                   "torch_cuda_initialized": check_e10_cpu_only(f"{t} seed {s}", dry_run),
                   "xgb_head_device": xgb_device.get("XGBoost"),
                   "state_dict_sha256": sd0, "head_seed": s, "heads": hparams["heads"],
                   **{k: z[k] for k in ("Z_train_sha256", "Z_test_sha256", "Z_train_shape", "Z_test_shape")},
                   "rows": rows_check, "e1_identity": identity, **hook}
            _write_json(exp_dir / t / f"seed_{s}" / f"{t}__FTT__seed{s}_embedding_record.json", rec)
            records[f"{t}|{s}"] = rec
            del ft
    cuda_initialized["end"] = check_e10_cpu_only("end", dry_run)
    _write_csv(exp_dir / "metrics.csv", pd.DataFrame(rows))
    return write_manifest(exp_dir, run_dir, experiment="E10", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"ftt": seeds, "heads": "= FTT seed"},
                          extra={"targets": targets, "e1_dir": str(e1_dir), "e1_experiment": e1_man["experiment"],
                                 "e1_manifest_sha256": e1_manifest_sha, "head_hyperparameters": hparams["heads"],
                                 "embedding_device": "cpu", "torch_threads": hybrid.EMBEDDING_TORCH_THREADS,
                                 # torch.cuda.is_initialized() at the start and end guards (a real run raises if
                                 # True; recorded so that a dry run shows the guard result)
                                 "torch_cuda_initialized": cuda_initialized,
                                 "xgb_device": xgb_params["device"],
                                 "embedding_records": {k: {kk: v[kk] for kk in ("source_sha256", "Z_train_sha256",
                                                                                 "Z_test_sha256",
                                                                                 "torch_cuda_initialized")}
                                                       for k, v in records.items()}})


# ======================================================================================
# E9b — NN Case B / Case C, restored from the pre-revision code (analysis/nn_cases.py)
# ======================================================================================
def _nn_cases():
    analysis = str(REPO_DIR / "analysis")
    if analysis not in sys.path:
        sys.path.insert(0, analysis)
    import nn_cases
    return nn_cases


#: E9b feature set: F12, the feature set of the paper's NN analysis (pre-revision ``--features_key features_12``)
E9B_FEATURES_KEY = "features_12"


def run_e9b(*, run_dir, data_csv, targets: Sequence[str] = rs.TARGETS, dry_run: bool = False, command=None,
            config: Optional[dict] = None, config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E9b with the run layout and a manifest: ``analysis/nn_cases.run`` (the NN
    analysis as in the paper: pre-revision primary 1-NN, Case B and Case C,
    data path with the category normalisation fix, F12 = ``features_12`` (the paper's method), full
    cohort, chance reference beside each value) into
    ``{run_dir}/E9b_nn_cases[_dry_run]``,
    refused if that directory is not empty; run-index entry ``E9b``. A real run
    covers every target, reads the pinned cohort file and passes the real-run
    checks of ``_prepare``; a dry run may restrict the targets. No model is
    fitted, no split is read.
    """
    nc = _nn_cases()
    import data_groups
    run_dir = Path(run_dir)
    config, config_file = _run_config(config, config_file)
    targets = list(targets)
    if not dry_run:
        if sorted(targets) != sorted(rs.TARGETS):
            raise RunnerError("a real E9b covers every target")
        if file_sha256(data_csv) != data_groups.PINNED_SHA256["canonical_csv"]:
            raise RunnerError(f"{data_csv}: not the pinned ASHRAE-2022 cohort file")
        if list(config["data"].get(E9B_FEATURES_KEY) or []) != list(nc.FROZEN_F12):
            raise RunnerError("a real E9b uses F12 (features_12, the paper's NN analysis); the "
                              f"configuration has {E9B_FEATURES_KEY} = {config['data'].get(E9B_FEATURES_KEY)}")
    pre = _prepare(run_dir, "E9b", config, dry_run, config_file=config_file)
    mode = "dry_run" if dry_run else "real"
    inputs = {"cohort_csv": str(data_csv)}
    startup = bind_inputs(pre.startup, inputs)   # hashed before nn_cases.run reads it
    summary = nc.run(Path(data_csv), pre.exp_dir, E9B_FEATURES_KEY, targets, mode=mode, config=config)
    if summary.attrs.get("config_used_sha256") != _json_sha256(config) \
            or sorted(summary.attrs.get("variables") or []) != sorted(config["data"][E9B_FEATURES_KEY]):
        raise RunnerError("E9b ran with another configuration or feature set than the one this manifest records "
                          f"(analysis {summary.attrs.get('config_used_sha256')}, manifest {_json_sha256(config)})")
    if not summary.attrs.get("data_csv_sha256"):
        raise RunnerError("E9b does not report the hash of the cohort bytes it parsed")
    check_consumed_inputs(startup, {"cohort_csv": summary.attrs["data_csv_sha256"]}, "E9b (nn_cases)")
    if not (summary["run_mode"] == mode).all():
        raise RunnerError(f"E9b summary not labelled {mode!r}")
    if not dry_run and not bool(summary["frozen_definition"].all()):
        raise RunnerError("E9b summary not produced with the frozen definition")
    return write_manifest(pre.exp_dir, run_dir, experiment="E9b", dry_run=dry_run, command=command,
                          started=pre.started, config=config, compute=pre.compute, env=pre.env, inputs=inputs,
                          startup=startup, seeds={"models": None},
                          extra={"features_key": E9B_FEATURES_KEY, "population": nc.POPULATION, "targets": targets,
                                 "primary_definition": "k = 1, Euclidean distance on encode_features (categorical "
                                                       "integer codes, every column standardised), query "
                                                       "excluded by row position",
                                 "case_b_round_decimals": nc.ROUND_DECIMALS,
                                 "case_c_tolerances": dict(nc.NUM_TOLERANCES),
                                 "case_c_strict_equality": sorted(nc.CATEGORICAL_COLS),
                                 "source": nc.SOURCE,
                                 "n_summary_rows": int(len(summary))})


# ======================================================================================
# E6 — driver of transfert.py (phase3_runner)
# ======================================================================================
def bind_e6_record(record: Mapping, startup: StartupRecord, inputs: Mapping[str, str], splits_dir):
    """
    Inputs consumed inside ``transfert.main``:
    the serialised Phase 3 split files are bound (role ``phase3_split:{name}``)
    to the SHA-256 that ``phase3_runner.run_e6`` took before its first fit
    (``split_files_sha256``); the model and French cohort files it records must
    be the bound ones. Returns ``(inputs, startup)`` with the split files added;
    ``write_manifest`` then re-checks every one of them.
    """
    splits = record.get("split_files_sha256")
    if not isinstance(splits, Mapping) or not splits:
        raise RunnerError("E6 run record lists no serialised Phase 3 split file (split_files_sha256)")
    bound = startup.inputs_sha256
    bad = [f"e1:{n}" for n, sha in (record.get("model_files_sha256") or {}).items() if bound.get(f"e1:{n}") != sha]
    bad += [n for n, sha in (record.get("cohort_files_sha256") or {}).items() if bound.get(n) != sha]
    if bad or record.get("model_files_sha256") is None or record.get("cohort_files_sha256") is None:
        raise RunnerError(f"the E6 run record's model / cohort file hashes are not the bound inputs ({bad or 'missing'})")
    split_inputs = {f"phase3_split:{name}": str(Path(splits_dir) / name) for name in sorted(splits)}
    startup = bind_inputs(startup, split_inputs, {f"phase3_split:{n}": sha for n, sha in splits.items()})
    return {**inputs, **split_inputs}, startup



E6_MODEL_FILES = ("best_ft_by_target.joblib", "best_rf_by_target.joblib", "best_xgb_by_target.joblib")   # = transfert.E1_MODEL_FILES


def run_e6(*, run_dir, e1_dir, splits_dir=None, strategies: Optional[Sequence[str]] = None,
           regimes: Optional[Sequence[str]] = None, cohorts: Optional[Sequence[str]] = None,
           targets: Optional[Sequence[str]] = None, ashrae_csv=None, ashrae_split_dir=None,
           dry_run: bool = False, dry_run_splits: int = 1, gpu_concurrency: Optional[int] = None,
           command=None, config: Optional[dict] = None,
           config_file: Optional[Tuple[dict, str]] = None) -> dict:
    """
    E6 through ``transfert.main`` with the run layout
    and a manifest. The wrapper owns every path and the dry-run state: it builds
    the whole ``transfert.py`` command line from typed arguments (no free
    pass-through), so the manifest records what actually ran. A dry run writes
    ``E6_phase3_dry_run`` with ``--dry_run_splits`` (default 1) and serialises its
    Phase 3 splits inside that directory unless ``splits_dir`` is given; a real
    run evaluates the 20 splits. The E1 source is validated first
    (``load_e1_source``: manifest, recorded model and split files, cohort CSV
    and archived splits of E1; a real run refuses a dry-run E1). Every E6 fit
    is a ``compute_scheduler`` job (``phase3_runner.run_e6``); the job records
    go to the manifest.
    """
    import transfert
    run_dir = Path(run_dir)
    config, config_file = _run_config(config, config_file)
    ds = config["data"]["datasets"][DATASET]
    pre = _prepare(run_dir, "E6", config, dry_run, gpu_concurrency=gpu_concurrency, config_file=config_file)
    exp_dir, compute, env, started = pre.exp_dir, pre.compute, pre.env, pre.started
    if dry_run and not 1 <= int(dry_run_splits) <= 20:
        raise RunnerError(f"dry_run_splits must be in [1, 20], got {dry_run_splits}")
    if splits_dir is None:
        if not dry_run:
            raise RunnerError("a real E6 needs splits_dir (serialised Phase 3 splits, e.g. splits/phase3)")
        splits_dir = exp_dir / "splits"
    check_not_protected(splits_dir)
    e1_dir = Path(e1_dir)
    ashrae_csv = str(ashrae_csv or ds["csv"])
    ashrae_split_dir = str(ashrae_split_dir or ds["split_dir"])
    tg = list(targets or rs.TARGETS)
    files = [e1_dir / n for n in E6_MODEL_FILES if (e1_dir / n).exists()]
    files += [e1_dir / t / "splits" / f"{t}_split_indices.json" for t in tg]
    validated: Dict[str, str] = {}
    e1_man = load_e1_source(e1_dir, dry_run=dry_run, files=files,
                            inputs={"cohort_csv": ashrae_csv, **_split_hashes(ashrae_split_dir, tg)},
                            validated=validated)
    argv = ["--models_dir", str(e1_dir), "--splits", str(splits_dir), "--output_dir", str(exp_dir),
            "--ashrae_csv", ashrae_csv, "--ashrae_split_dir", ashrae_split_dir]
    for flag, values in (("--strategies", strategies), ("--regimes", regimes), ("--cohorts", cohorts),
                         ("--targets", targets)):
        if values:
            argv += [flag, *values]
    if dry_run:
        argv += ["--dry_run_splits", str(int(dry_run_splits))]
    if gpu_concurrency is not None:
        argv += ["--gpu_concurrency", str(int(gpu_concurrency))]
    # every file transfert.main may read, hashed before it reads them
    inputs = {f"e1:{f.relative_to(e1_dir)}": str(f) for f in files}
    inputs["e1:run_manifest.json"] = str(e1_dir / "run_manifest.json")
    inputs.update({"cohort_csv": ashrae_csv, **_split_hashes(ashrae_split_dir, tg)})
    french = {k: str(REPO_DIR / v) if not Path(v).is_absolute() else v
              for k, v in transfert.COHORT_FILES.items() if k in (cohorts or transfert.COHORT_FILES)}
    consumed = {**inputs, **{k: v for k, v in french.items() if Path(v).is_file()}}
    startup = bind_inputs(pre.startup, consumed, expected_input_hashes(consumed, validated))
    # the one configuration of the run (fits and manifest); the validated E1 manifest unlocks a real run
    record = transfert.main(argv, config=config, e1_manifest=e1_man)
    label = experiment_label("E6", dry_run)
    if not isinstance(record, dict) or record.get("experiment") != label or bool(record.get("dry_run")) != dry_run:
        raise RunnerError(f"transfert.main did not run {label} as requested "
                          f"(record experiment {None if not isinstance(record, dict) else record.get('experiment')})")
    if record.get("config_sha256") != _json_sha256(config):
        raise RunnerError("transfert.main ran with another configuration than the one this manifest records "
                          f"(record {record.get('config_sha256')}, manifest {_json_sha256(config)})")
    # the feature list fitted is the one of the configuration recorded
    if list(record.get("features") or []) != list(features_for(config, "F17")):
        raise RunnerError("transfert.main fitted another feature list than the configuration this manifest "
                          f"records (record {record.get('features')}, configuration {features_for(config, 'F17')})")
    # the ASHRAE cohort and split bytes load_source_rows parsed are the bound ones
    parsed = record.get("source_files_sha256")
    expected_roles = ({"cohort_csv"} | {f"split:{t}" for t in tg}
                      | {f"e1:{t}/splits/{t}_split_indices.json" for t in tg})
    if not isinstance(parsed, Mapping) or set(parsed) != expected_roles:
        raise RunnerError(f"the E6 run record does not list the ASHRAE cohort and split files it parsed "
                          f"(source_files_sha256 = {parsed!r})")
    check_consumed_inputs(startup, parsed, "E6 (transfert / load_source_rows)")
    coverage = exp_dir / f"category_coverage_{DATASET}.json"
    if not coverage.exists():
        raise RunnerError(f"{coverage} missing: E6 reads the ASHRAE cohort (coverage log)")
    inputs.update({k: v for k, v in french.items() if k in record.get("cohorts", transfert.COHORT_FILES)})
    inputs, startup = bind_e6_record(record, startup, inputs, splits_dir)
    return write_manifest(exp_dir, run_dir, experiment="E6", dry_run=dry_run, command=command, started=started,
                          config=config, compute=compute, env=env, inputs=inputs, startup=startup,
                          seeds={"models": [42], "phase3_splits": "numpy.random.default_rng(1000 + r)"},
                          jobs=record.get("jobs"),
                          extra={"transfert_argv": argv, "e1_dir": str(e1_dir), "e1_experiment": e1_man["experiment"],
                                 "e1_manifest_sha256": validated[str(e1_dir / "run_manifest.json")],
                                 "splits_dir": str(splits_dir), "n_splits_evaluated": record.get("n_splits_evaluated"),
                                 "strategies": record.get("strategies"), "regimes": record.get("regimes"),
                                 "targets": record.get("targets")},
                          gpu_concurrency=gpu_concurrency)
