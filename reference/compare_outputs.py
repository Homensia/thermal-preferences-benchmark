"""Compare a reproduction with the reference SHA-256 of the paper outputs (standard library only).

    python reference/compare_outputs.py --run REPRO_RUN_DIR [--experiments E9b E2] [--match TEXT] [--partial]
    python reference/compare_outputs.py --figures FIGURES_DIR

``--run``: the reference rows of the selected experiments (default: those whose directory exists in the run)
are compared with the files of the run; ``--figures``: the rows of the figure regeneration. ``--match`` keeps
only the reference paths that contain TEXT (for example ``__RF__`` for an E2 run restricted to
``--families RF``).

Each row of ``reference/expected_outputs_sha256.csv`` has a comparison mode (column ``compare``):

* ``file``: SHA-256 of the file bytes;
* ``json_without:K1;K2``: SHA-256 of the JSON content without the top-level keys K1, K2 (keys sorted,
  compact separators): the declared fields that the release writes with other wording;
* ``csv_columns:C1;C2;...``: SHA-256 of the CSV restricted to the listed columns, in that order (the
  columns kept by the release; the other columns belong to strategies that the release does not run).

Status ``expected``: the file must exist and match. Status ``superseded``: a recorded output that the release
does not reproduce (its note says why); it is listed and never verified, and it never makes a check pass.

Result (last line) and exit status:

* ``RESULT: COMPLETE`` (0): every selected ``expected`` file exists and matches;
* ``RESULT: PARTIAL`` (2): with ``--partial``, every ``expected`` file present matches but some are absent
  (coverage printed);
* ``RESULT: FAIL`` (1): a mismatch, an absent ``expected`` file (without ``--partial``), an absent
  directory, an empty selection or no ``expected`` file compared.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "expected_outputs_sha256.csv"
COMPLETE, FAIL, PARTIAL = 0, 1, 2


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_without(raw: bytes, keys) -> bytes:
    obj = json.loads(raw.decode("utf-8"))
    for k in keys:
        obj.pop(k, None)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_csv_columns(raw: bytes, columns) -> bytes:
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    header = rows[0]
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"columns {missing} absent")
    idx = [header.index(c) for c in columns]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in rows:
        w.writerow([r[i] for i in idx])
    return out.getvalue().encode("utf-8")


def digest(path: Path, mode: str) -> str:
    """SHA-256 of ``path`` under the comparison mode of a reference row."""
    raw = path.read_bytes()
    if mode == "file":
        return sha256_bytes(raw)
    kind, _, arg = mode.partition(":")
    names = [x for x in arg.split(";") if x]
    if kind == "json_without":
        return sha256_bytes(canonical_json_without(raw, names))
    if kind == "csv_columns":
        return sha256_bytes(canonical_csv_columns(raw, names))
    raise ValueError(f"unknown comparison mode {mode!r}")


def compare(rows, root: Path, partial: bool) -> int:
    """Compare ``rows`` with the files under ``root``; prints every problem and a summary; returns the status."""
    expected = [r for r in rows if r["status"] == "expected"]
    superseded = [r for r in rows if r["status"] == "superseded"]
    same = differ = 0
    absent = []
    for r in expected:
        p = root / r["path"]
        if not p.is_file():
            absent.append(r["path"])
            continue
        try:
            ok = digest(p, r.get("compare") or "file") == r["sha256"]
        except (ValueError, UnicodeDecodeError) as e:
            ok = False
            print(f"UNREADABLE {r['path']}: {e}")
        if ok:
            same += 1
        else:
            differ += 1
            print(f"DIFFER   {r['path']} ({r.get('compare') or 'file'})")
    for path in absent:
        print(f"{'SKIPPED' if partial else 'MISSING'}  {path}")
    present_sup = sum((root / r["path"]).is_file() for r in superseded)
    print(f"expected: {len(expected)} selected, {same} match, {differ} differ, {len(absent)} absent; "
          f"superseded (not verified): {len(superseded)} selected, {present_sup} present")
    if differ or same == 0 or (absent and not partial):
        return FAIL
    if absent:
        print(f"coverage: {same} of {len(expected)} expected files verified")
        return PARTIAL
    return COMPLETE


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, help="reproduction run directory")
    p.add_argument("--experiments", nargs="+", help="experiments to compare (default: those present in --run)")
    p.add_argument("--figures", type=Path, help="figure regeneration directory (--out of analysis.paper_figures)")
    p.add_argument("--match", help="compare only the reference paths that contain this text")
    p.add_argument("--partial", action="store_true",
                   help="accept absent expected files (result PARTIAL, exit status 2, coverage printed)")
    p.add_argument("--reference", type=Path, default=REFERENCE, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.run and not args.figures:
        p.error("give --run and/or --figures")
    with open(args.reference, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not args.match or args.match in r["path"]]
    statuses = []
    if args.run:
        if not args.run.is_dir():
            print(f"FAIL: run directory {args.run} does not exist")
            statuses.append(FAIL)
        else:
            run_rows = [r for r in rows if r["scope"] == "run"]
            exps = args.experiments or sorted({r["experiment"] for r in run_rows
                                               if (args.run / r["path"].split("/")[0]).is_dir()})
            unknown = sorted(set(exps) - {r["experiment"] for r in run_rows})
            if unknown:
                print(f"FAIL: no reference rows for {unknown}" + (f" matching {args.match!r}" if args.match else ""))
                statuses.append(FAIL)
            if not exps:
                print(f"FAIL: no experiment directory of the reference in {args.run}")
                statuses.append(FAIL)
            for exp in exps:
                if exp in unknown:
                    continue
                print(f"== {exp}")
                statuses.append(compare([r for r in run_rows if r["experiment"] == exp], args.run, args.partial))
    if args.figures:
        print("== figures and tables")
        fig_rows = [r for r in rows if r["scope"] == "paper_figures"]
        if not args.figures.is_dir():
            print(f"FAIL: figure directory {args.figures} does not exist")
            statuses.append(FAIL)
        elif not fig_rows:
            print("FAIL: no reference rows selected")
            statuses.append(FAIL)
        else:
            statuses.append(compare(fig_rows, args.figures, args.partial))
    status = FAIL if FAIL in statuses or not statuses else (PARTIAL if PARTIAL in statuses else COMPLETE)
    print("RESULT:", {COMPLETE: "COMPLETE", PARTIAL: "PARTIAL", FAIL: "FAIL"}[status])
    return status


if __name__ == "__main__":
    sys.exit(main())
