"""Verified access to the outputs of a revision run, and the output directory.

Every file read by the paper-figure package goes through :class:`RunInputs`,
which checks its SHA-256 against the manifest that recorded it before parsing it:

* the run index ``<run_dir>/manifest.json`` records, per experiment, the path
  and SHA-256 of its ``run_manifest.json`` (status ``complete`` required);
* an experiment output (``E9b_nn_cases/E9b_nn_cases_summary.csv``, an E1 model file, ...) is
  checked against ``outputs_sha256`` of that experiment's ``run_manifest.json``;
* an experiment input (cohort CSV, archived split, French cohort) is checked
  against ``inputs_sha256`` of the experiment that consumed it, and read from
  the code checkout (``data_root``), where the run read it;
* the aggregate tables (E1, E6, E7 metrics, E2 and E4 aggregates) are recomputed
  from the verified predictions files (:mod:`.aggregate`);
* the cohort CSV of the distributions figure is checked against its pinned
  SHA-256 (``data_groups.PINNED_SHA256``);
* the Phase 1 grid (not rerun; ``results/phase1/experiment_grid_full.csv`` of
  ``data_root``) is checked against the pinned SHA-256 of the run index
  ``phase1`` entry.

Experiments E10 (hybrids), E7b (SET, PET, ePMV) and E9b (NN Case B / Case C)
have no output yet: they are *pending*. A figure that needs them raises :class:`PendingExperimentError`,
unless the caller passes ``allow_pending=True`` (CLI ``--preview``); the
missing rows are then left out and the omission is recorded in the output
manifest.

:class:`OutputDir` refuses an existing non-empty directory and creates every
file exclusively (never overwrites).
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

PENDING_EXPERIMENTS = {"E10": "E10_hybrids", "E7b": "E7b_indices", "E9b": "E9b_nn_cases"}
COHORT_CSV = "Data/ASHRAE_2022_Clean_api.csv"
PHASE1_GRID = "results/phase1/experiment_grid_full.csv"


def cohort_pinned_sha256() -> str:
    """Pinned SHA-256 of the ASHRAE-2022 cohort CSV (``data_groups.PINNED_SHA256["canonical_csv"]``)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import data_groups
    return data_groups.PINNED_SHA256["canonical_csv"]


class InputHashError(ValueError):
    """A file does not have the SHA-256 recorded by its manifest."""


class PendingExperimentError(RuntimeError):
    """An experiment needed by a figure has no completed output in the run index."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


#: experiments that read the E1 outputs: from an extra run directory they must prove which E1 they read
E1_CONSUMERS = ("E4", "E6", "E7", "E7b", "E10")


class RunInputs:
    """Hash-verified reader of one run directory, optionally completed by later run directories (see module
    docstring).

    ``extra_run_dirs``: run directories whose completed experiments replace those of ``run_dir`` (for
    example E6 and E9b rerun at a later commit against the E1 of ``run_dir``). E1 always comes from
    ``run_dir``; an extra experiment that records ``e1_manifest_sha256`` must have read that E1.
    """

    def __init__(self, run_dir, data_root=REPO_ROOT, allow_pending: bool = False, extra_run_dirs=()):
        self.run_dir = Path(run_dir).resolve()
        self.data_root = Path(data_root).resolve()
        self.allow_pending = bool(allow_pending)
        self.consumed: List[Dict[str, str]] = []   # every verified file: role, path, sha256
        self.pending_used: List[str] = []          # pending experiments left out of an output
        index_path = self.run_dir / "manifest.json"
        raw = self._read(index_path)
        self.run_index_sha256 = sha256_bytes(raw)
        self.run_index = json.loads(raw)
        self._record("run_index", index_path, self.run_index_sha256)
        #: experiment -> (run directory, run-index entry)
        self.sources: Dict[str, Tuple[Path, Dict[str, Any]]] = {
            exp: (self.run_dir, entry) for exp, entry in (self.run_index.get("experiments") or {}).items()}
        self.extra_indices: List[Dict[str, Any]] = []
        for extra in extra_run_dirs:
            extra = Path(extra).resolve()
            extra_path = extra / "manifest.json"
            extra_raw = self._read(extra_path)
            extra_index = json.loads(extra_raw)
            self._record("run_index", extra_path, sha256_bytes(extra_raw))
            base_pin = self.run_index["phase1"].get("experiment_grid_full_sha256_pinned")
            p1 = extra_index.get("phase1") or {}
            if not (base_pin and p1.get("experiment_grid_full_sha256") == p1.get("experiment_grid_full_sha256_pinned")
                    == base_pin and p1.get("matches_pinned") is True):
                raise InputHashError(f"{extra_path}: Phase 1 grid hash, pin or matches_pinned differ from the pin "
                                     f"of {index_path} ({base_pin})")
            replaced = []
            for exp, entry in (extra_index.get("experiments") or {}).items():
                if not (entry and entry.get("status") == "complete"):
                    continue
                if exp == "E1":
                    raise ValueError(f"{extra_path}: E1 must come from --run_dir {self.run_dir}")
                if exp in self.sources:
                    replaced.append(exp)
                self.sources[exp] = (extra, entry)
            self.extra_indices.append({"run_dir": str(extra), "run_index_sha256": sha256_bytes(extra_raw),
                                       "replaces": sorted(replaced)})
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._aggregates = None

    # -- low level ---------------------------------------------------------------------------------
    @staticmethod
    def _read(path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _record(self, role: str, path: Path, sha: str) -> None:
        try:
            shown = str(Path(path).resolve().relative_to(self.run_dir.parent))
        except ValueError:
            shown = str(path)
        self.consumed.append({"role": role, "path": shown, "sha256": sha})

    def _verified(self, role: str, path: Path, expected: Optional[str]) -> bytes:
        if not expected:
            raise InputHashError(f"{role}: no SHA-256 recorded for {path}")
        raw = self._read(path)
        got = sha256_bytes(raw)
        if got != expected:
            raise InputHashError(f"{role}: {path} has SHA-256 {got}, manifest records {expected}")
        self._record(role, path, got)
        return raw

    # -- experiments -------------------------------------------------------------------------------
    def has_experiment(self, exp: str) -> bool:
        entry = self.sources.get(exp, (None, None))[1]
        return bool(entry) and entry.get("status") == "complete"

    def require(self, exp: str) -> bool:
        """True if ``exp`` is available; False if it is pending and pending outputs are allowed."""
        if self.has_experiment(exp):
            return True
        if exp in PENDING_EXPERIMENTS and self.allow_pending:
            if exp not in self.pending_used:
                self.pending_used.append(exp)
            return False
        raise PendingExperimentError(
            f"experiment {exp} has no completed entry in {self.run_dir / 'manifest.json'}"
            + (" (pending: rerun with --preview to produce a preview without its rows)"
               if exp in PENDING_EXPERIMENTS else ""))

    def manifest(self, exp: str) -> Dict[str, Any]:
        if exp in self._manifests:
            return self._manifests[exp]
        if not self.has_experiment(exp):
            raise PendingExperimentError(f"experiment {exp} is not complete in the run index")
        run_dir, entry = self.sources[exp]
        path = run_dir / entry["manifest"]
        manifest = json.loads(self._verified(f"{exp}:run_manifest.json", path, entry.get("manifest_sha256")))
        if manifest.get("dry_run"):
            raise ValueError(f"{exp}: {path} is a dry run")
        if manifest.get("experiment") != exp:
            raise ValueError(f"{path}: experiment field {manifest.get('experiment')!r} != {exp!r}")
        if run_dir != self.run_dir and exp in E1_CONSUMERS:
            e1_sha = self.run_index["experiments"]["E1"]["manifest_sha256"]
            recorded = manifest.get("e1_manifest_sha256")
            bound = (manifest.get("inputs_sha256") or {}).get("e1:run_manifest.json")
            if not (recorded and bound and recorded == bound == e1_sha):
                raise InputHashError(f"{path}: E1 provenance (e1_manifest_sha256 {recorded}, input "
                                     f"e1:run_manifest.json {bound}) is not the E1 of {self.run_dir} ({e1_sha})")
        manifest["_dir"] = path.parent
        self._manifests[exp] = manifest
        return manifest

    def experiment_dir(self, exp: str) -> Path:
        return self.manifest(exp)["_dir"]

    def run_file(self, exp: str, rel: str) -> bytes:
        """An output file of experiment ``exp`` (path relative to its directory)."""
        m = self.manifest(exp)
        return self._verified(f"{exp}:{rel}", m["_dir"] / rel, m.get("outputs_sha256", {}).get(rel))

    def run_csv(self, exp: str, rel: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.run_file(exp, rel)))

    def input_file(self, exp: str, role: str) -> bytes:
        """An input of experiment ``exp`` (manifest ``inputs[role]``), read from ``data_root``."""
        m = self.manifest(exp)
        recorded = m["inputs"][role]
        p = Path(recorded)
        if p.is_absolute():
            p = p.relative_to(m["cwd"])        # the run read it inside its worktree
        return self._verified(f"{exp}:input:{role}", self.data_root / p, m["inputs_sha256"].get(role))

    # -- aggregates and cohort -----------------------------------------------------------------------
    def aggregate(self, name: str) -> pd.DataFrame:
        """An aggregate table recomputed from the verified predictions files (``aggregate.Aggregates``)."""
        if self._aggregates is None:
            from .aggregate import Aggregates
            self._aggregates = Aggregates(self)
        return self._aggregates.table(name)

    def cohort_csv(self) -> bytes:
        """The ASHRAE-2022 cohort CSV of the checkout (``data_root``), checked against its pinned SHA-256
        (``data_groups.PINNED_SHA256``) and against the cohort input recorded by E1."""
        pinned = cohort_pinned_sha256()
        e1 = self.manifest("E1")["inputs_sha256"].get("cohort_csv")
        if e1 != pinned:
            raise InputHashError(f"E1 read the cohort {e1}, the pinned cohort is {pinned}")
        return self._verified("cohort_csv", self.data_root / COHORT_CSV, pinned)

    # -- Phase 1 (not rerun) -----------------------------------------------------------------------
    def phase1_grid(self) -> pd.DataFrame:
        """The Phase 1 grid shipped with the code (``data_root/results/phase1``), checked against the SHA-256
        pinned in the run index (the grid every experiment of the run checked)."""
        entry = self.run_index["phase1"]
        if entry.get("experiment_grid_full_sha256") != entry.get("experiment_grid_full_sha256_pinned"):
            raise InputHashError("run index: Phase 1 grid SHA-256 differs from its pinned value")
        path = self.data_root / PHASE1_GRID
        raw = self._verified("phase1:experiment_grid_full.csv", path, entry.get("experiment_grid_full_sha256_pinned"))
        return pd.read_csv(io.BytesIO(raw))


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT)] + args, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class OutputDir:
    """A new or empty output directory; every file is created exclusively."""

    def __init__(self, path):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from config_loader import check_not_protected
        self.path = Path(path).resolve()
        check_not_protected(self.path)
        if self.path.exists():
            if not self.path.is_dir():
                raise NotADirectoryError(f"--out {self.path} exists and is not a directory")
            if any(self.path.iterdir()):
                raise FileExistsError(f"--out {self.path} is not empty: pass a new directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.written: List[Path] = []

    def new(self, rel: str) -> Path:
        p = self.path / rel
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.written.append(p)
        return p

    def write_text(self, rel: str, text: str) -> Path:
        from .wording import check_text
        check_text(text, rel)
        p = self.new(rel)
        with open(p, "x", encoding="utf-8") as f:
            f.write(text)
        return p

    def write_csv(self, rel: str, df: pd.DataFrame, index: bool = False) -> Path:
        return self.write_text(rel, df.to_csv(index=index, lineterminator="\n"))

    def savefig(self, fig, rel: str, **kw) -> Path:
        from .wording import check_figure, check_text
        check_text(rel, rel)
        check_figure(fig, rel)
        p = self.new(rel)
        fmt = p.suffix.lstrip(".")
        meta = {"Software": None} if fmt == "png" else ({"Creator": None, "Producer": None, "CreationDate": None}
                                                      if fmt == "pdf" else None)
        with open(p, "xb") as f:
            fig.savefig(f, format=fmt, bbox_inches="tight", metadata=meta, **kw)
        return p

    def write_manifest(self, inputs: RunInputs, command: List[str], items: List[str],
                       mode: str = "final") -> Path:
        record = {
            "format": "paper-figures-v1",
            "mode": mode,
            "command": command,
            "items": items,
            "code_git_commit": _git(["rev-parse", "HEAD"]),
            "code_git_tracked_changes": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
            "run_dir": str(inputs.run_dir),
            "run_index_sha256": inputs.run_index_sha256,
            "extra_run_dirs": inputs.extra_indices,
            "pending_experiments_left_out": list(inputs.pending_used),
            "inputs": inputs.consumed,
            "outputs": [{"path": str(p.relative_to(self.path)), "sha256": sha256_file(p)}
                        for p in self.written],
        }
        p = self.new("paper_figures_manifest.json")
        with open(p, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
            f.write("\n")
        return p
