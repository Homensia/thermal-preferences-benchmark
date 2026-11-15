# -*- coding: utf-8 -*-
"""
Shared fixtures.

``machine_compute``: the ``compute`` section of ``config.yaml`` (``xgb_device: cpu``,
``torch_device: cuda``) on a machine with a CUDA device;
on a machine without one, the same section with both devices set to ``cpu``
*explicitly*. Tests that exercise the runners (not the device rules) use it, so
that they also run without a GPU while the production guard stays strict
(``cuda`` in the config and no CUDA device still raises, ``tests/test_compute.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


@pytest.fixture
def machine_compute():
    import torch
    from compute import load_compute
    cfg = load_compute()
    if not torch.cuda.is_available():
        cfg = {**cfg, "xgb_device": "cpu", "torch_device": "cpu"}
    return cfg


@pytest.fixture
def xgb_fits(machine_compute):
    """
    Skip a test that *fits* XGBoost when this machine cannot fit it on
    ``machine_compute["xgb_device"]``: with the GPU hidden
    (``CUDA_VISIBLE_DEVICES=""``) the CUDA build of xgboost 3.2.0 raises
    ``cudaErrorNoDevice`` even for ``device="cpu"`` (driver present, no device).
    On a machine with a visible GPU and on a machine without a CUDA driver the test runs.
    """
    import numpy as np
    import xgboost
    # three classes: with the GPU hidden a binary CPU fit succeeds but a multi-class one
    # (every target of the benchmark) raises cudaErrorNoDevice
    X, y = np.arange(30.0).reshape(10, 3), np.arange(10) % 3
    try:
        xgboost.XGBClassifier(n_estimators=1, device=machine_compute["xgb_device"],
                              tree_method="hist").fit(X, y)
    except xgboost.core.XGBoostError as exc:
        if "cudaErrorNoDevice" in str(exc):
            pytest.skip("xgboost CUDA build cannot fit with the GPU hidden (cudaErrorNoDevice)")
        raise
    return machine_compute
