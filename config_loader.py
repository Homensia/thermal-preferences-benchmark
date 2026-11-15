"""
Configuration Loader Module
===========================

This module provides a simple utility function to load experiment
and model configuration parameters from a YAML file. Centralizing
the configuration ensures reproducibility, easier experiment 
management, and decoupling between code and settings.

Functions
---------
load_config(path="config.yaml"):
    Load and parse a YAML configuration file, returning its content
    as a Python dictionary. The function uses `yaml.safe_load` to
    prevent execution of arbitrary YAML tags.
check_not_protected(path):
    Refuse any output path inside the frozen canonical artefact
    ``rerun_2026-04-28_no_reweighting/``.
check_new_data_output(path):
    Refuse, for the legacy data writers (``data_preparation.py``,
    ``feature_extraction_api.py``), an output that already exists or is one
    of the frozen input files.
read_consumed(path, consumed=None, role=None):
    One read per consumed input: every input
    file a revision run consumes (cohort CSVs, raw identifier files, French
    cohort CSVs, archived split / fold JSONs, E1 predictions and model files,
    ``config.yaml``, the lock file) is read **once**; the SHA-256 recorded for
    it is that of the bytes then parsed from memory (``io.BytesIO``), never of
    a second read of the path. A file swapped for other bytes while it is read
    and restored afterwards is thus recorded with the hash of the bytes actually
    consumed, which differs from the file on disk, and the binding / pin / end
    checks refuse the run.
"""


import hashlib
from pathlib import Path
from typing import MutableMapping, Optional, Tuple

import yaml

# Frozen canonical artefact of the pre-revision code: no script may write inside it.
PROTECTED_ARTEFACT_DIR = Path(__file__).resolve().parent / "rerun_2026-04-28_no_reweighting"


def check_not_protected(path) -> None:
    """Raise ``ValueError`` if ``path`` (resolved from the working directory)
    is the frozen canonical artefact or lies inside it, or inside any directory
    of that name: the copy of a run worktree and the development checkout's
    original alike (runs start from a worktree)."""
    p = Path(path).resolve()
    if p == PROTECTED_ARTEFACT_DIR or PROTECTED_ARTEFACT_DIR in p.parents or PROTECTED_ARTEFACT_DIR.name in p.parts:
        raise ValueError(f"refusing to write inside the frozen artefact {PROTECTED_ARTEFACT_DIR}")


# Frozen input files ("no existing file is modified"); their SHA-256 are listed in the README ("Data").
# ASHRAE_2022_Clean.csv is an intermediate output of data_preparation.py, not shipped and read by no experiment.
DATA_DIR = (Path(__file__).resolve().parent / "Data").resolve()
FROZEN_DATA_FILES = tuple(DATA_DIR / name for name in (
    "ASHRAE_09_06_2022.csv", "ASHRAE_09_06_2022_metadata.xlsx", "ASHRAE_2022_Clean.csv",
    "ASHRAE_2022_Clean_api.csv", "Moujalled_api.csv", "Hostein_api.csv", "ASHRAE_2018_v2.csv"))


def check_new_data_output(path) -> Path:
    """
    Output guard of the legacy data writers:
    raise ``FileExistsError`` if ``path`` already exists (never overwritten) and
    ``ValueError`` if it is one of the frozen input files (even when
    missing) or lies inside the frozen canonical artefact. Called before any
    computation or API call. Returns the resolved path.
    """
    p = Path(path).resolve()
    check_not_protected(p)
    if p in FROZEN_DATA_FILES:
        raise ValueError(f"refusing to write the frozen protocol input file {p}: pass a new --output path")
    if p.exists() or Path(path).is_symlink():
        raise FileExistsError(f"refusing to overwrite the existing file {p}: pass a new --output path")
    return p


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ConsumedInputError(ValueError):
    """The same input was consumed twice with different bytes."""


def read_consumed(path, consumed: Optional[MutableMapping[str, str]] = None,
                  role: Optional[str] = None) -> Tuple[bytes, str]:
    """
    ``(bytes, sha256)`` of one read of ``path``. ``consumed[role]`` (default
    key ``str(path)``) receives that SHA-256; a key already present with
    another hash raises ``ConsumedInputError`` (two reads of one input saw
    different bytes).
    """
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if consumed is not None:
        key = str(Path(path)) if role is None else role
        if key in consumed and consumed[key] != sha:
            raise ConsumedInputError(f"{path} ({key}) was consumed twice with different bytes "
                                     f"({consumed[key]}, then {sha})")
        consumed[key] = sha
    return raw, sha

