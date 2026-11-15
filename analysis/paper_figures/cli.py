"""Command line: ``python -m analysis.paper_figures --run_dir RUN --out NEW_DIR [--only ITEM ...] [--preview]``.

Two modes (docs/PAPER_FIGURES.md):

* final (default): every experiment an item needs must be complete in the run index; this is the mode
  of the published regeneration (``--out RUN/PAPER_FIGURES``);
* ``--preview``: pending experiments (E10 hybrids, E7b SET/PET/ePMV, E9b NN Case B/C) are left out of
  their items, the
  LaTeX tables carry a ``% PENDING`` comment and the manifest records ``mode: preview``; the output
  directory must lie outside ``--run_dir`` (a preview never lands among the run outputs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inputs import REPO_ROOT, OutputDir, RunInputs
from .items import ITEMS, Options


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m analysis.paper_figures", description=__doc__)
    p.add_argument("--run_dir", type=Path, help="revision run directory (rerun_2026-10-XX_v2)")
    p.add_argument("--extra_run_dir", type=Path, action="append", default=[],
                   help="later run directory whose experiments replace those of --run_dir (E1 always comes "
                        "from --run_dir); repeatable")
    p.add_argument("--out", type=Path, help="new or empty output directory")
    p.add_argument("--only", nargs="+", choices=sorted(ITEMS), help="items to build (default: every item)")
    p.add_argument("--data_root", type=Path, default=REPO_ROOT,
                   help="checkout holding Data/ and splits/ (run inputs, checked against the run manifests)")
    p.add_argument("--preview", action="store_true",
                   help="pending-preview mode: leave the rows of pending experiments (E10, E7b, E9b) out (recorded in "
                        "the manifest); --out must be outside --run_dir")
    p.add_argument("--cm_seed", default="42", choices=["42", "43", "44", "sum"],
                   help="model seed of the confusion matrices ('sum': counts summed over 42-44)")
    p.add_argument("--fi_seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="model seeds averaged in the feature-importance figures")
    p.add_argument("--nn_cases_in_figure", action="store_true",
                   help="also draw the E9b Case B / Case C agreements in nn_agreement_vs_accuracy")
    p.add_argument("--indices_tpv", action="store_true",
                   help="also show a TPV panel / rows (PMV, aPMV only) in the heat-balance index items; not the "
                        "paper scope (indices on TSV-7 and TSV-3 only, the default)")
    p.add_argument("--list", action="store_true", help="list the items and exit")
    args = p.parse_args(argv)
    if not args.list and (args.run_dir is None or args.out is None):
        p.error("--run_dir and --out are required")
    if args.preview and not args.list:
        out = args.out.resolve()
        for run in [args.run_dir.resolve(), *(d.resolve() for d in args.extra_run_dir)]:
            if out == run or run in out.parents:
                p.error(f"--preview output {out} must not be inside the run directory {run}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        for item in ITEMS.values():
            pend = (f"  [pending: {', '.join(item.pending)}]" if item.pending else "") + (
                "" if item.default else "  [not built by default]")
            print(f"{item.name:20s} {item.paper}{pend}")
            for o in item.outputs:
                print(f"{'':20s}   {o}")
        return 0
    names = args.only or [n for n, item in ITEMS.items() if item.default]
    inputs = RunInputs(args.run_dir, data_root=args.data_root, allow_pending=args.preview,
                       extra_run_dirs=args.extra_run_dir)
    for name in names:                       # fail before writing anything if an experiment is missing
        for exp in ITEMS[name].pending:
            inputs.require(exp)
    out = OutputDir(args.out)
    opt = Options(cm_seed=args.cm_seed, fi_seeds=tuple(args.fi_seeds), indices_tpv=args.indices_tpv,
                  nn_cases_in_figure=args.nn_cases_in_figure)
    for name in names:
        print(f"[paper_figures] {name}", flush=True)
        ITEMS[name].build(inputs, out, opt)
    command = [Path(sys.executable).name, "-m", "analysis.paper_figures"] + (
        list(argv) if argv is not None else sys.argv[1:])
    manifest = out.write_manifest(inputs, command, names, mode="preview" if args.preview else "final")
    if inputs.pending_used:
        print(f"[paper_figures] PENDING experiments left out: {', '.join(inputs.pending_used)}")
    print(f"[paper_figures] {len(out.written) - 1} files + {manifest.name} in {out.path}")
    return 0
