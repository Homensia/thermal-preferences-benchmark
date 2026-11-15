"""
Entry point of the revision rerun. One subcommand per experiment, all writing under
one run directory (placeholder ``rerun_2026-10-XX_v2``; the frozen canonical
artefact ``rerun_2026-04-28_no_reweighting/`` is refused):

    python3 run_revision.py e1    --run_dir RUN            # Phase 2 corrected   -> RUN/E1_phase2
    python3 run_revision.py e2    --run_dir RUN            # LOPO                -> RUN/E2_lopo
    python3 run_revision.py e4    --run_dir RUN            # EMA ablation F12/F17-> RUN/E4_ema
    python3 run_revision.py e6    --run_dir RUN            # Phase 3 (transfert) -> RUN/E6_phase3 (--splits splits/phase3)
    python3 run_revision.py e7    --run_dir RUN            # PMV/aPMV vs E1      -> RUN/E7_pmv
    python3 run_revision.py e10   --run_dir RUN            # FTT+RF / FTT+XGB    -> RUN/E10_hybrids
    python3 run_revision.py e7b   --run_dir RUN            # SET/PET/ePMV vs E1  -> RUN/E7b_indices
    python3 run_revision.py e9b   --run_dir RUN            # NN Case B / Case C  -> RUN/E9b_nn_cases

The real E1, E6 and E7 run only through these subcommands: ``models.py
--dataset ASHRAE_2022`` is refused, and ``transfert.py`` /
``indicateurs_classiques.py`` called directly accept dry runs only (no run
manifest otherwise).

E4, E6, E7, E7b and E10 read the E1 outputs of the same run directory, validated
against the E1 manifest (``revision_runners.load_e1_source``; a real run refuses
a dry-run E1). ``--dry_run`` is a smoke check: tiny
hyperparameters (``DRY_RUN_HPARAMS``), ``*_dry_run`` directories and experiment
labels, folds (E1) and Phase 3 splits (E6, ``--dry_run_splits`` splits evaluated)
written inside the dry-run directory; it never produces results. A real run
needs the locked environment, ``config.yaml`` unchanged, a clean checkout and
the Phase 1 artefact with its pinned SHA-256 (``revision_runners.check_phase1_artefact``).
A failed real experiment leaves its directory without ``run_manifest.json``
(never read by a consumer); to retry in the same run directory, move it aside
(e.g. ``RUN/failed/E6_phase3_<UTC>``, never delete or edit it), log the failure,
then rerun the subcommand (README, "A failed real experiment").
A real run is launched from the root of a clean checkout (a plain clone or a
git worktree) at the run commit: without ``--dry_run`` the entry point refuses
a working directory other than the repository root; every manifest records the
checkout kind (``provenance.git_worktree``) and the working directory (``cwd``).
``config.yaml`` is read once, where the configuration is parsed; its recorded
SHA-256 is that of those bytes.
See ``revision_runners`` for the outputs and the manifest.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import phase3_runner
import revision_folds
import revision_runners as rr
import revision_stats as rs
from config_loader import check_not_protected

E6_COHORTS = ("Moujalled", "Hostein")          # = transfert.COHORT_FILES (tested; transfert is slow to import)


def _config(dry_run: bool):
    """
    ``(config, config_file)``: ``config.yaml`` is read once (``rr.read_config_file``)
    and the configuration used (a copy, with ``DRY_RUN_HPARAMS`` for a dry run)
    is derived from the bytes parsed; ``config_file`` = (parsed, SHA-256 of those
    bytes) goes to ``_prepare`` and the manifest unchanged, never re-established
    by reopening the path.
    """
    config_file = rr.read_config_file()
    config = copy.deepcopy(config_file[0])
    if dry_run:
        for section, values in rr.DRY_RUN_HPARAMS.items():
            config[section] = copy.deepcopy(values)
    return config, config_file


def _dataset_cfg(config: dict) -> dict:
    return config["data"]["datasets"][rr.DATASET]


def check_launch_cwd(dry_run: bool, cwd: Path = None) -> Path:
    """
    A real run is launched from the root of the checkout (the directory of this
    file): the import-time ``config.yaml`` reads and the relative defaults
    (``Data/``, ``splits/``) then resolve inside the checkout whose code runs,
    never in another one. Dry runs skip it.
    Returns the working directory.
    """
    cwd = Path.cwd() if cwd is None else Path(cwd)
    if not dry_run and cwd.resolve() != rr.REPO_DIR.resolve():
        raise SystemExit(f"run_revision.py: a real run is launched from the repository root {rr.REPO_DIR} "
                         f"(cd there first), not from {cwd}")
    return cwd


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def launch(sp):
        return sp

    def common(sp, targets=rs.TARGETS, dry=True, data=True, concurrency=True):
        sp.add_argument("--run_dir", required=True, help="Run directory, e.g. rerun_2026-10-XX_v2")
        sp.add_argument("--targets", nargs="+", choices=list(rs.TARGETS), default=list(targets))
        if dry:
            sp.add_argument("--dry_run", action="store_true",
                            help="Smoke check: tiny hyperparameters, *_dry_run directories and labels; never results")
        if data:
            sp.add_argument("--data_csv", default=None, help="ASHRAE-2022 cohort CSV (default: config.yaml registry)")
        if concurrency:
            sp.add_argument("--gpu_concurrency", type=int, default=None,
                            help="Override compute.gpu_concurrency for this run (scheduling only; recorded)")
        return launch(sp)

    e1 = common(sub.add_parser("e1", help="E1: Phase 2 corrected, 5 families"))
    e1.add_argument("--families", nargs="+", default=list(rr.E1_FAMILIES), choices=list(rr.E1_FAMILIES))
    e1.add_argument("--seeds", nargs="+", type=int, default=list(revision_folds.TREE_SEEDS),
                    help="RF/XGB seeds (frozen: 42 43 44)")
    e1.add_argument("--ann_seeds", nargs="+", type=int, default=[42], help="ANN seeds (the paper run: 42 43 44)")
    e1.add_argument("--ftt_seeds", nargs="+", type=int, default=[42], help="FTT seeds (the paper run: 42 43 44)")
    e1.add_argument("--split_dir", default=None, help="Archived split (default: splits/ASHRAE_2022)")
    e1.add_argument("--folds_from", default=None,
                    help="Fold files directory (default: config folds_dir 'splits'; dry run: inside the run)")

    e2 = common(sub.add_parser("e2", help="E2: leave-one-publication-out"))
    e2.add_argument("--families", nargs="+", default=list(rr.TREE_FAMILIES), choices=list(rr.TREE_FAMILIES))

    e4 = common(sub.add_parser("e4", help="E4: EMA ablation, F12 vs F17"))
    e4.add_argument("--families", nargs="+", default=list(rr.TREE_FAMILIES), choices=list(rr.TREE_FAMILIES))
    e4.add_argument("--seeds", nargs="+", type=int, default=list(revision_folds.TREE_SEEDS))
    e4.add_argument("--split_dir", default=None)
    e4.add_argument("--e1_dir", default=None, help="E1 outputs (default: RUN/E1_phase2[_dry_run])")

    # E6: typed arguments only (no free pass-through: the wrapper owns every path and the dry-run state)
    e6 = launch(sub.add_parser("e6", help="E6: Phase 3 through transfert.py"))
    e6.add_argument("--run_dir", required=True)
    e6.add_argument("--e1_dir", default=None)
    e6.add_argument("--splits", default=None,
                    help="Serialised Phase 3 splits (default splits/phase3, dry run: inside the run)")
    e6.add_argument("--strategies", nargs="+", choices=list(phase3_runner.STRATEGIES), default=None)
    e6.add_argument("--regimes", nargs="+", choices=["20-80", "80-20"], default=None,
                    help="Default 20-80 (primary regime); 80-20 supplement")
    e6.add_argument("--cohorts", nargs="+", choices=list(E6_COHORTS), default=None)
    e6.add_argument("--targets", nargs="+", choices=list(rs.TARGETS), default=None)
    e6.add_argument("--gpu_concurrency", type=int, default=None,
                    help="Override compute.gpu_concurrency for this run (scheduling only; recorded)")
    e6.add_argument("--dry_run", action="store_true",
                    help="Smoke check on the E1 dry run: RUN/E6_phase3_dry_run, first --dry_run_splits splits")
    e6.add_argument("--dry_run_splits", type=int, default=1, help="Splits evaluated by a dry run (1-20)")

    e7 = launch(sub.add_parser("e7", help="E7: PMV/aPMV on the archived test rows vs E1"))
    e7.add_argument("--run_dir", required=True)
    e7.add_argument("--e1_dir", default=None)
    e7.add_argument("--data_csv", default=None)
    e7.add_argument("--split_dir", default=None)
    e7.add_argument("--expected_ml", nargs="+", default=None, metavar="MODEL:SEED",
                    help="Default: the ml_inventory of the E1 manifest")
    e7.add_argument("--skip_group_ids", action="store_true", help="Dry runs only (synthetic data)")
    e7.add_argument("--dry_run", action="store_true",
                    help="Compare with the E1 dry run (RUN/E1_phase2_dry_run); writes RUN/E7_pmv_dry_run")

    e7b = launch(sub.add_parser("e7b", help="E7b: SET/PTS_SET, PET/PTS_PET, ePMV vs E1, TSV-7 and TSV-3"))
    e7b.add_argument("--run_dir", required=True)
    e7b.add_argument("--e1_dir", default=None)
    e7b.add_argument("--data_csv", default=None)
    e7b.add_argument("--split_dir", default=None)
    e7b.add_argument("--skip_group_ids", action="store_true", help="Dry runs only (synthetic data)")
    e7b.add_argument("--dry_run", action="store_true",
                     help="Compare with the E1 dry run (RUN/E1_phase2_dry_run); writes RUN/E7b_indices_dry_run")

    e10 = launch(sub.add_parser("e10", help="E10: RF / XGBoost heads on the E1 FTT embeddings, CPU extraction"))
    e10.add_argument("--run_dir", required=True)
    e10.add_argument("--e1_dir", default=None, help="E1 outputs (default: RUN/E1_phase2[_dry_run])")
    e10.add_argument("--data_csv", default=None, help="ASHRAE-2022 cohort CSV (default: config.yaml registry)")
    e10.add_argument("--split_dir", default=None, help="Archived split (default: splits/ASHRAE_2022)")
    e10.add_argument("--targets", nargs="+", choices=list(rs.TARGETS), default=list(rs.TARGETS),
                     help="Dry runs only: a real E10 covers every target")
    e10.add_argument("--dry_run", action="store_true",
                     help="Smoke check on the E1 dry run: RUN/E10_hybrids_dry_run, tiny heads, never results")

    e9b = launch(sub.add_parser("e9b", help="E9b: NN Case B / Case C restored from the pre-revision code, "
                                            "descriptive"))
    e9b.add_argument("--run_dir", required=True)
    e9b.add_argument("--data_csv", default=None, help="ASHRAE-2022 cohort CSV (default: config.yaml registry)")
    e9b.add_argument("--targets", nargs="+", choices=list(rs.TARGETS), default=list(rs.TARGETS),
                     help="Dry runs only: a real E9b covers every target")
    e9b.add_argument("--dry_run", action="store_true", help="Smoke check: RUN/E9b_nn_cases_dry_run, never results")

    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    command = [sys.executable, Path(__file__).name, *(sys.argv[1:] if argv is None else argv)]
    check_not_protected(args.run_dir)
    dry_run = bool(getattr(args, "dry_run", False))
    check_launch_cwd(dry_run)                                     # before anything is read
    config, config_file = _config(dry_run)                        # the one config.yaml read
    ds = _dataset_cfg(config)
    split_dir = getattr(args, "split_dir", None) or ds["split_dir"]
    e1_dir = getattr(args, "e1_dir", None) or rr.experiment_dir(args.run_dir, "E1", dry_run)

    if args.command == "e6":
        return rr.run_e6(run_dir=args.run_dir, e1_dir=e1_dir,
                         splits_dir=args.splits or (None if dry_run else "splits/phase3"),
                         strategies=args.strategies, regimes=args.regimes, cohorts=args.cohorts, targets=args.targets,
                         dry_run=dry_run, dry_run_splits=args.dry_run_splits, gpu_concurrency=args.gpu_concurrency,
                         command=command, config=config, config_file=config_file)
    if args.command == "e9b":
        return rr.run_e9b(run_dir=args.run_dir, data_csv=args.data_csv or ds["csv"], targets=args.targets,
                          dry_run=dry_run, command=command, config=config, config_file=config_file)
    if args.command == "e7b":
        return rr.run_e7b(run_dir=args.run_dir, e1_dir=e1_dir, data_csv=args.data_csv or ds["csv"],
                          split_dir=split_dir, skip_group_ids=args.skip_group_ids, dry_run=dry_run,
                          command=command, config=config, config_file=config_file)
    if args.command == "e7":
        return rr.run_e7(run_dir=args.run_dir, e1_dir=e1_dir, data_csv=args.data_csv or ds["csv"],
                         split_dir=split_dir, expected_ml=args.expected_ml, skip_group_ids=args.skip_group_ids,
                         dry_run=dry_run, command=command, config=config, config_file=config_file)

    ctx = rr.load_ashrae_context(config, csv_path=args.data_csv)
    common = dict(run_dir=args.run_dir, config=config, config_file=config_file, dry_run=dry_run, command=command)
    if args.command == "e1":
        folds_dir = args.folds_from or (None if dry_run else ds["folds_dir"])
        return rr.run_e1(ctx, split_dir=split_dir, folds_dir=folds_dir, targets=args.targets, families=args.families,
                         tree_seeds=args.seeds, ann_seeds=args.ann_seeds, ftt_seeds=args.ftt_seeds,
                         gpu_concurrency=args.gpu_concurrency, **common)
    if args.command == "e2":
        return rr.run_e2(ctx, targets=args.targets, families=args.families, gpu_concurrency=args.gpu_concurrency,
                         **common)
    if args.command == "e10":
        return rr.run_e10(ctx, e1_dir=e1_dir, split_dir=split_dir, targets=args.targets, **common)
    if args.command == "e4":
        return rr.run_ablation(ctx, experiment="E4", e1_dir=e1_dir, split_dir=split_dir,
                               targets=args.targets, families=args.families, seeds=args.seeds,
                               gpu_concurrency=args.gpu_concurrency, **common)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
    print("[OK] run_revision completed")
