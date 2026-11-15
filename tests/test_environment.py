"""Tests that the interpreter is the locked revision environment.

The 2026-10 revision rerun runs in one locked virtualenv
(``docs/ENVIRONMENT_revision_2026-10.md``). These tests fail, on purpose, in any
other interpreter: every distribution pinned in
``requirements-revision-2026-10.lock`` must be installed at its pinned version,
the Python version must be the one the canonical run used, and the PMV source
file of pythermalcomfort must have the pinned SHA-256. No GPU is needed.

Run from the repository root:
    .venv-revision/bin/python -m pytest tests/test_environment.py -v
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_DIR / "requirements-revision-2026-10.lock"

PINNED_PYTHON = (3, 12, 3)  # rerun_2026-04-28_no_reweighting/audit_log.json:4
PINNED_PYTHERMALCOMFORT = "4.0.2"
# SHA-256 of pythermalcomfort/models/pmv_ppd_iso.py in pythermalcomfort 4.0.2.
PINNED_PMV_PPD_ISO_SHA256 = (
    "56ea982c98d1f22475cf0530f3ebbf25eecc4febda0038c2be5b7a873f792aa7"
)


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_pins() -> dict[str, str]:
    pins = {}
    for line in LOCK_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        assert sep, f"lock line is not an exact pin: {line!r}"
        pins[_normalise(name)] = version
    return pins


def test_lock_file_pins_the_required_versions():
    pins = _lock_pins()
    assert pins["scikit-learn"] == "1.8.0"
    assert pins["torch"] == "2.7.0+cu128"
    assert pins["pythermalcomfort"] == PINNED_PYTHERMALCOMFORT
    assert "pytest" in pins


def test_python_version_is_pinned():
    assert sys.version_info[:3] == PINNED_PYTHON


def test_installed_distributions_match_the_lock_file():
    pins = _lock_pins()
    installed = {
        _normalise(d.metadata["Name"]): d.version for d in metadata.distributions()
    }
    missing = sorted(set(pins) - set(installed))
    wrong = {n: (v, installed[n]) for n, v in pins.items()
             if n in installed and installed[n] != v}
    assert not missing, f"pinned but not installed: {missing}"
    assert not wrong, f"pinned != installed: {wrong}"


def test_pythermalcomfort_pmv_source_hash_is_pinned():
    pytest.importorskip("pythermalcomfort")
    import pythermalcomfort

    assert metadata.version("pythermalcomfort") == PINNED_PYTHERMALCOMFORT
    source = Path(pythermalcomfort.__file__).parent / "models" / "pmv_ppd_iso.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == PINNED_PMV_PPD_ISO_SHA256


def test_torch_is_the_cuda_12_8_build():
    torch = pytest.importorskip("torch")
    assert torch.__version__ == "2.7.0+cu128"
    assert torch.version.cuda == "12.8"
