"""
External dataset evaluation for thermal comfort prediction models (Phase 3, E6).

Models trained on ASHRAE (E1 seed-42 FT-Transformer, RandomForest, XGBoost)
are evaluated on the French cohorts (Moujalled, Hostein).

The command line runs experiment E6 through ``phase3_runner.run_e6``: 20 frozen date-disjoint splits
per cohort × target × regime, serialised before any fit; zero-shot and
local-only RF/XGB; FTT zero-shot, FT-head and FT-full
(``ftt_adaptation.adapt_ftt``); one predictions file per (cohort, target,
model, seed, split) with the columns of
``phase3_runner.prediction_columns``.

The legacy adaptation evaluator (one stratified row split, FT_head / FT_full
re-initialising the network) is removed and replaced by that runner.

``direct_eval_one_base`` (zero-shot metrics on every cohort row, legacy JSON
layout) is kept as a library function; E6 writes its full-cohort zero-shot
predictions itself (``split_id = "full_cohort"``).
"""



from typing import Dict, Tuple, Iterable
from typing import Optional, List
from pathlib import Path
import argparse
import json
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from models import (
    ensure_dir,
    eval_on_test,
    FEATURES_ALL,
    TARGETS_ALL,
    set_all_seeds,
    base_out,
    RANDOM_SEED,
)
from models_classic import (make_preprocess, LabelEncodedClassifier, check_onehot_revision_schema,
                            ParallelFitRandomForestClassifier)
from models_deep import check_ftt_revision_schema, RevisionNotImplementedError
from revision_stats import ontology
from category_normalisation import (
    categorical_features,
    load_csv_normalised,
    vocabulary_for,
    write_coverage_log,
)
from config_loader import load_config, check_not_protected
from compute import load_compute, xgb_compute_params
from phase3_runner import check_partition  # noqa: F401  (DEV/TEST partition check, re-exported)
from unified_data_loader import check_numeric_features







# ======================================================================================
# CONFIGURATION
# ======================================================================================

CONFIG = load_config()
set_all_seeds(RANDOM_SEED)


# ======================================================================================
# EXTERNAL COHORTS AND PROTOTYPES
# ======================================================================================

def load_external_cohort(csv_path, name: str, coverage_dir=None, return_coverage: bool = False,
                         consumed_sha256: dict = None):
    """
    Load a French cohort (Moujalled, Hostein) with the shared category
    normalisation (raises on any value outside the frozen
    table) and write its coverage log to ``coverage_dir`` if given. With
    ``return_coverage``, return ``(df, coverage)`` (the E6 CLI writes the log
    after the run's input checks). The file is read once and parsed from those
    bytes; ``consumed_sha256[name]`` receives their SHA-256.
    """
    import io
    from config_loader import read_consumed
    raw, _ = read_consumed(csv_path, consumed_sha256, name)
    # Frozen categorical variables of the feature set, by name: an absent column raises.
    df, coverage = load_csv_normalised(io.BytesIO(raw), name, categorical_features(FEATURES_ALL))
    cats = set(categorical_features(FEATURES_ALL))
    check_numeric_features(df, [f for f in FEATURES_ALL if f not in cats], name)
    if coverage_dir is not None:
        write_coverage_log(coverage, Path(coverage_dir) / f"category_coverage_{name}.json")
    return (df, coverage) if return_coverage else df


def make_rf_xgb_protos(num_cols, cat_cols, seed: int = RANDOM_SEED, cat_vocabulary=None, compute=None,
                       families=("RF", "XGB"), config=None):
    """
    Create reusable prototype pipelines for RandomForest and XGBoost.

    These prototypes are cloned for the local-only (DEV) and pooled
    (ASHRAE train + DEV) fits of E6 (``phase3_runner``).

    Parameters
    ----------
    num_cols : list of str
        Numerical columns of the dataset.
    cat_cols : list of str
        Categorical columns of the dataset.
    seed : int, default=RANDOM_SEED
        Random seed.
    cat_vocabulary : dict or None
        Frozen category vocabulary for the one-hot encoder.
    compute : dict or None
        ``compute`` section (default: ``config.yaml``): XGB device (CPU by default;
        optional GPU, no silent fallback), RF ``n_jobs = cpu_jobs`` (hyperparameters unchanged).
    families : iterable of {"RF", "XGB"}
        Prototypes to build; the other one is ``None`` (the XGB CUDA check runs
        only when XGB is requested).
    config : dict or None
        Configuration whose ``classical_hparams`` are used (default: ``config.yaml``,
        module ``CONFIG``); the E6 wrapper passes the configuration of its run
        (a dry run: the tiny hyperparameters it records).

    Both prototypes take the frozen hyperparameters of ``config.yaml``
    ``classical_hparams``, like the E1 fits.

    Returns
    -------
    (rf_pipeline, xgb_pipeline) : tuple of sklearn.Pipeline
        Ready-to-train RandomForest and XGBoost pipelines.
    """
    pp = make_preprocess(num_cols, cat_cols, cat_vocabulary=cat_vocabulary)
    compute = load_compute() if compute is None else compute
    families = set(families)
    if families - {"RF", "XGB"}:
        raise ValueError(f"unknown prototype families {sorted(families - {'RF', 'XGB'})}")
    rf_proto = xgb_proto = None

    hparams = (CONFIG if config is None else config)["classical_hparams"]
    rf_hp = hparams["RandomForest"]
    xgb_hp = hparams["XGBoost"]
    if "RF" in families:
        rf_proto = Pipeline([
            ("pp", pp),
            ("clf", ParallelFitRandomForestClassifier(
                n_estimators=rf_hp["n_estimators"],
                max_depth=rf_hp["max_depth"],
                min_samples_split=rf_hp["min_samples_split"],
                min_samples_leaf=rf_hp["min_samples_leaf"],
                class_weight=rf_hp.get("class_weight", None),
                random_state=seed,
                n_jobs=compute["cpu_jobs"]
            ))
        ])

    if "XGB" in families:
        xgb_proto = Pipeline([
            ("pp", pp),
            ("clf", LabelEncodedClassifier(
                XGBClassifier(
                    # frozen config values (the pre-revision prototype had 600 trees / depth 10)
                    n_estimators=xgb_hp["n_estimators"],
                    max_depth=xgb_hp["max_depth"],
                    learning_rate=xgb_hp["learning_rate"],
                    subsample=xgb_hp["subsample"],
                    colsample_bytree=xgb_hp["colsample_bytree"],
                    eval_metric="mlogloss",
                    # objective left to XGBoost: multi:softprob for >= 3 DEV classes,
                    # binary:logistic for 2 (an explicit multi:softprob fails on 2)
                    random_state=seed,
                    **xgb_compute_params(compute)        # device, tree_method="hist", host threads
                )
            ))
        ])

    return rf_proto, xgb_proto


# The pre-revision ``clone_ft_for_finetune`` (followed by ``FTClassifier.fit``, which
# re-initialised the network and the preprocessing) is replaced by
# ``ftt_adaptation.adapt_ftt``.


# ======================================================================================
# ZERO-SHOT EVALUATION
# ======================================================================================

def direct_eval_one_base(base_name: str,
                        df: pd.DataFrame,
                        ft_by_target: dict,
                        rf_by_target: Optional[dict] = None,
                        xgb_by_target: Optional[dict] = None,
                        output_dir: str = "external_results") -> None:
    """
    Perform zero-shot evaluation on a given external dataset.

    Zero-shot = evaluating pre-trained models directly on external data
    *without* any adaptation or fine-tuning.

    Parameters
    ----------
    base_name : str
        Dataset name (e.g., "Moujalled", "Hostein").
    df : pandas.DataFrame
        External dataset.
    ft_by_target : dict
        Mapping target → trained FTTransformer model.
    rf_by_target : dict, optional
        Mapping target → trained RandomForest model.
    xgb_by_target : dict, optional
        Mapping target → trained XGB model.

    Notes
    -----
    Every family is evaluated on all rows of ``df`` with the full target
    ontology; a true class absent from a model's training classes is kept
    (it is an error for that model), never filtered out.
    """
    out_root = ensure_dir(Path(output_dir) / base_name / "A_direct")

    for target in [t for t in TARGETS_ALL if t in df.columns]:
        X, y = df[FEATURES_ALL], df[target]

        # Harmonize types → float
        try:
            y = y.astype(float)
        except:
            y = pd.to_numeric(y, errors="coerce")

        # Zero-shot is evaluated on every row of the external cohort, for every
        # family: a true class the model never
        # learned stays in the evaluated set and counts as an error; rows are
        # never selected on their true label.
        if len(X) != len(df) or len(y) != len(df):
            raise ValueError(f"[{base_name}:{target}] evaluated rows differ from the cohort")

        # --- FTTransformer evaluation ---
        if target in ft_by_target:
            ft = ft_by_target[target]
            _ = eval_on_test(
                ft, X, y,
                name=f"{base_name}__{target}__FTTransformer",
                labels_order=ontology(target),
                save_dir=ensure_dir(out_root / target / "FTTransformer")
            )

        # --- RandomForest evaluation ---
        if rf_by_target and target in rf_by_target:
            rf = rf_by_target[target]
            _ = eval_on_test(
                rf, X, y,
                name=f"{base_name}__{target}__RF",
                labels_order=ontology(target),
                save_dir=ensure_dir(out_root / target / "RandomForest")
            )

        # --- XGBoost evaluation ---
        if xgb_by_target and target in xgb_by_target:
            xgb = xgb_by_target[target]
            _ = eval_on_test(
                xgb, X, y,
                name=f"{base_name}__{target}__XGB",
                labels_order=ontology(target),
                save_dir=ensure_dir(out_root / target / "XGBoost")
            )


# ======================================================================================
# CLI INTERFACE
# ======================================================================================

def setup_logging(verbose: bool) -> None:
    """
    Configure logging verbosity.

    Parameters
    ----------
    verbose : bool
        If True, enable DEBUG logging. Otherwise INFO logging.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_model_dicts(models_dir: Path, consumed_sha256: dict = None):
    """
    Load stored model dictionaries from joblib files previously saved
    by ``models.py``.

    Parameters
    ----------
    models_dir : pathlib.Path
        Base directory containing joblib dictionaries.

    Every loaded model is checked against the revision encoders before any
    evaluation (category normalisation fix): FTT models need the revision
    ``TabPreprocessor`` (global vocabulary + reserved ``<OOV>`` index), RF/XGB
    pipelines the global-vocabulary one-hot encoder. Archived pre-revision models
    raise with the pre-revision-checkout instruction.

    Each file is read once and unpickled from those bytes; ``consumed_sha256``
    receives ``{file name: SHA-256}`` of the bytes loaded.

    Returns
    -------
    (ft_by_target, rf_by_target, xgb_by_target) : tuple of dict
        Loaded model dictionaries for each model family.
    """
    import io
    from joblib import load as _joblib_load
    from config_loader import read_consumed

    def load(path):
        raw, _ = read_consumed(path, consumed_sha256, path.name)
        return _joblib_load(io.BytesIO(raw))
    vocabulary = vocabulary_for(categorical_features(FEATURES_ALL))
    
    ft_by_t, rf_by_t, xgb_by_t = {}, {}, {}
    
    ft_path = models_dir / "best_ft_by_target.joblib"
    rf_path = models_dir / "best_rf_by_target.joblib"
    xgb_path = models_dir / "best_xgb_by_target.joblib"
    
    if ft_path.exists():
        ft_by_t = load(ft_path)
        logging.info(f"Loaded FT models from {ft_path.name} ({len(ft_by_t)} targets)")
    
    if rf_path.exists():
        rf_by_t = load(rf_path)
        logging.info(f"Loaded RF models from {rf_path.name} ({len(rf_by_t)} targets)")
    
    if xgb_path.exists():
        xgb_by_t = load(xgb_path)
        logging.info(f"Loaded XGB models from {xgb_path.name} ({len(xgb_by_t)} targets)")

    for t, m in ft_by_t.items():
        check_ftt_revision_schema(m, vocabulary, f"{ft_path.name}:{t}")
    for path, models_ in ((rf_path, rf_by_t), (xgb_path, xgb_by_t)):
        for t, m in models_.items():
            check_onehot_revision_schema(m, vocabulary, f"{path.name}:{t}")

    return ft_by_t, rf_by_t, xgb_by_t


CLI_STRATEGIES = ("zero_shot", "local", "ftt_head", "ftt_full")
COHORT_FILES = {"Moujalled": "Data/Moujalled_api.csv", "Hostein": "Data/Hostein_api.csv"}
#: Full SHA-256 of the French cohort files (prefixes e13b6be4 / 791819bf);
#: a real E6 refuses any other file.
COHORT_SHA256 = {
    "Moujalled": "e13b6be45d225124b9c5fc0d6f3058e9e4a3bebfd24ebec55dcc1662743cb7fd",
    "Hostein": "791819bfcd671662bb285b93950fe610fb22c21604dfd738f6931c0d38b64e9f",
}
E1_MODEL_FILES = ("best_ft_by_target.joblib", "best_rf_by_target.joblib", "best_xgb_by_target.joblib")


def check_cohort_files(names, files=None, digests=None) -> None:
    """
    A real E6 reads the frozen French cohort files (full SHA-256 pinned).
    ``digests`` ({name: SHA-256}): the hashes of the bytes actually parsed
    (``load_external_cohort``); default: hash the files now.
    """
    from phase3_runner import file_sha256
    files = COHORT_FILES if files is None else files
    for name in names:
        digest = file_sha256(files[name]) if digests is None else digests.get(name)
        if digest != COHORT_SHA256[name]:
            raise ValueError(f"{files[name]}: SHA-256 {digest} != the frozen cohort file {name} "
                             f"({COHORT_SHA256[name]}); a real E6 reads the frozen cohort files only")


def check_real_e6_source(models_dir, e1_manifest) -> None:
    """
    A real E6 (all 20 splits) runs only through ``run_revision.py e6``
    (``revision_runners.run_e6``), which validates the E1 source
    (``load_e1_source``), the locked environment and the clean checkout,
    and writes the run manifest. ``e1_manifest`` is the E1 manifest that wrapper validated: it
    must be a real (non-dry-run) E1 and record every E1 model file of
    ``models_dir`` with its current SHA-256. A standalone ``transfert.py``
    call may only run a dry run (``--dry_run_splits N``, files marked
    ``E6_dry_run``).
    """
    from phase3_runner import file_sha256
    if e1_manifest is None:
        raise RevisionNotImplementedError(
            "a real E6 (--dry_run_splits 0) runs only through `python3 run_revision.py e6 --run_dir RUN`, "
            "which validates the E1 manifest, the locked environment and the checkout and writes the run "
            "manifest; standalone transfert.py accepts dry runs only (--dry_run_splits N)")
    if e1_manifest.get("experiment") != "E1" or e1_manifest.get("dry_run") is not False:
        raise RevisionNotImplementedError(
            f"a real E6 needs a real E1 source, got {e1_manifest.get('experiment')!r} "
            f"(dry_run={e1_manifest.get('dry_run')!r})")
    outs = e1_manifest.get("outputs_sha256") or {}
    for name in E1_MODEL_FILES:
        path = Path(models_dir) / name
        if path.exists() and outs.get(name) != file_sha256(path):
            raise RevisionNotImplementedError(f"{path}: not the model file recorded by the E1 manifest")


def _n_splits_arg(value: str) -> int:
    """``--dry_run_splits``: an integer in [0, 20] (0 = the full E6 run)."""
    import phase3_runner
    n = int(value)
    if not 0 <= n <= phase3_runner.N_SPLITS:
        raise argparse.ArgumentTypeError(f"must be in [0, {phase3_runner.N_SPLITS}], got {n}")
    return n


def parse_args(argv=None):
    """
    Parse the E6 command line:

    ``python3 transfert.py --models_dir …/E1_phase2 --splits splits/phase3
    --strategies zero_shot local ftt_head ftt_full
    --output_dir …/E6_phase3``

    The real E6 runs through ``python3 run_revision.py e6 --run_dir RUN``,
    which builds this command line; called directly, ``transfert.py`` accepts
    dry runs only (``--dry_run_splits N``; ``check_real_e6_source``).

    Returns
    -------
    argparse.Namespace
    """
    ds = CONFIG["data"]["datasets"]["ASHRAE_2022"]
    p = argparse.ArgumentParser(
        description="Run experiment E6 (Phase 3) on the Moujalled/Hostein cohorts."
    )
    p.add_argument("--models_dir", type=str, required=True,
                   help="E1 output directory: best_{rf,xgb,ft}_by_target.joblib (= seed 42) "
                        "and {target}/splits/{target}_split_indices.json.")
    p.add_argument("--splits", type=str, required=True,
                   help="Directory of the serialised Phase 3 splits ({cohort}_{target}_{regime}.json); "
                        "written once before any fit, then loaded and checked.")
    p.add_argument("--strategies", type=str, nargs="+", choices=CLI_STRATEGIES,
                   default=list(CLI_STRATEGIES), help="E6 strategies (default: all).")
    p.add_argument("--regimes", type=str, nargs="+", choices=["20-80", "80-20"], default=["20-80"],
                   help="DEV/TEST regimes; 20-80 is the main regime, 80-20 is supplement only.")
    p.add_argument("--cohorts", type=str, nargs="+", choices=list(COHORT_FILES),
                   default=list(COHORT_FILES))
    p.add_argument("--targets", type=str, nargs="+", choices=list(TARGETS_ALL), default=list(TARGETS_ALL))
    p.add_argument("--ashrae_csv", type=str, default=ds["csv"],
                   help="ASHRAE-2022 cohort CSV (source rows: zero-shot OOV reference).")
    p.add_argument("--ashrae_split_dir", type=str, default=ds["split_dir"],
                   help="Archived 80/20 split of the ASHRAE cohort.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="E6 output directory (predictions/, ftt_adaptation/, split_summary.csv, "
                        "e6_run_record.json).")
    p.add_argument("--dry_run_splits", type=_n_splits_arg, default=0,
                   help="Dry run: evaluate only the first N serialised splits (files marked E6_dry_run). "
                        "0 (default) = the full E6 run on the 20 splits, accepted only through "
                        "`run_revision.py e6` (validated E1 manifest, run manifest); standalone "
                        "transfert.py runs dry runs only.")
    p.add_argument("--gpu_concurrency", type=int, default=None,
                   help="Override compute.gpu_concurrency for this run (scheduling of the FTT "
                        "adaptation jobs only; recorded in e6_run_record.json).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return p.parse_args(argv)


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================

def main(argv=None, config=None, e1_manifest=None):
    """
    Entry point of experiment E6 (``phase3_runner.run_e6``).

    ``e1_manifest`` (library call by ``revision_runners.run_e6`` only): the
    validated E1 manifest; a real run (``--dry_run_splits 0``) refuses to start
    without it (``check_real_e6_source``), so standalone runs are dry runs.

    ``config`` (library call only, e.g. ``revision_runners.run_e6``): the one
    configuration of the run, used for the RF/XGB prototypes, the compute
    settings and the FTT adaptation budget, and recorded by its SHA-256 in
    ``e6_run_record.json``; default ``config.yaml`` (module ``CONFIG``).

    Workflow
    --------
    1. Parse CLI arguments; refuse any output or split directory inside the
       frozen canonical artefact before loading anything
    2. Load the French cohorts (category normalisation, coverage logs)
    3. Load the E1 seed-42 models (revision encoders checked)
    4. Load the ASHRAE training rows of each target (archived split, checked
       against the E1 split files)
    5. Run E6 and write its outputs under ``--output_dir`` (with the category
       coverage logs of the French cohorts and of the ASHRAE source rows);
       returns the run record (``e6_run_record.json``)
    """
    import functools
    import sys
    import phase3_runner
    config = CONFIG if config is None else config
    args = parse_args(argv)
    setup_logging(args.verbose)
    check_not_protected(args.output_dir)  # never write into the frozen canonical artefact
    check_not_protected(args.splits)
    phase3_runner.check_output_dir(args.output_dir, args.splits)  # one run per output directory
    dry_run = args.dry_run_splits > 0
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not dry_run:                       # before any data is loaded
        check_real_e6_source(models_dir, e1_manifest)
        check_cohort_files(args.cohorts)
    print(f"   Models dir: {models_dir}")
    print(f"   Splits dir: {args.splits}")
    print(f"   Output dir: {args.output_dir}   dry run: {dry_run}")

    # Category normalisation to the frozen ASHRAE vocabulary;
    # the coverage logs are written once run_e6 has passed its input checks and finished
    # every input is read once and parsed from those bytes; the hashes recorded are those of the bytes parsed
    cohort_sha256: dict = {}
    loaded = {name: load_external_cohort(COHORT_FILES[name], name, return_coverage=True,
                                         consumed_sha256=cohort_sha256)
              for name in args.cohorts}
    cohorts = {name: df for name, (df, _) in loaded.items()}
    if not dry_run:                       # the pins hold for the bytes parsed, not only for an earlier read
        check_cohort_files(args.cohorts, digests=cohort_sha256)

    model_sha256: dict = {}
    ft_by_t, rf_by_t, xgb_by_t = load_model_dicts(models_dir, consumed_sha256=model_sha256)
    models = {t: {"RF": rf_by_t.get(t), "XGB": xgb_by_t.get(t), "FTT": ft_by_t.get(t)}
              for t in args.targets}
    model_files = {name: models_dir / name for name in
                   ("best_ft_by_target.joblib", "best_rf_by_target.joblib", "best_xgb_by_target.joblib")
                   if (models_dir / name).exists()}

    source_files_sha256: dict = {}             # split bytes parsed, recorded for the wrapper
    source_rows, source_coverage = phase3_runner.load_source_rows(
        args.ashrae_csv, args.ashrae_split_dir, args.targets, FEATURES_ALL, models_dir=models_dir,
        return_coverage=True, consumed_sha256=source_files_sha256)

    record = phase3_runner.run_e6(
        cohorts=cohorts, source_rows=source_rows, models=models,
        make_protos=functools.partial(make_rf_xgb_protos, config=config),
        output_dir=args.output_dir, splits_dir=args.splits, targets=args.targets, regimes=args.regimes,
        strategies=args.strategies, features=FEATURES_ALL, config=config,
        n_splits=args.dry_run_splits if dry_run else phase3_runner.N_SPLITS, dry_run=dry_run,
        model_files=model_files, cohort_files={n: COHORT_FILES[n] for n in args.cohorts},
        command=list(sys.argv if argv is None else ["transfert.py", *argv]),
        gpu_concurrency=args.gpu_concurrency, source_files_sha256=source_files_sha256,
        model_files_sha256=model_sha256, cohort_files_sha256=cohort_sha256,
    )
    for name, (_, coverage) in loaded.items():
        write_coverage_log(coverage, Path(args.output_dir) / f"category_coverage_{name}.json")
    # the ASHRAE source rows (zero-shot OOV reference) are normalised too
    write_coverage_log(source_coverage, Path(args.output_dir) / f"category_coverage_{phase3_runner.SOURCE_COHORT}.json")
    print("\n" + "="*80)
    print(f"✅ {record['experiment']} completed: {record['n_prediction_files']} predictions files")
    print("="*80)
    logging.info("Done.")
    return record


if __name__ == "__main__":
    main()
