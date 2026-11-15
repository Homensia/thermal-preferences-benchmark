# Locked environment

Every experiment of the paper ran in one virtual environment whose packages are pinned in
`requirements-revision-2026-10.lock` (the output of `pip freeze`, 86 distributions). Every run manifest
records the SHA-256 of the lock file (`53cbbdfe…43d0` in the recorded run; `6e369df1…76f1` in this release,
where only the header comment changed, pins identical) and the `pip freeze` of the interpreter; a real run
refuses an interpreter whose packages differ from the lock file. `tests/test_environment.py` fails, on
purpose, in any other interpreter.

| Item | Value |
|---|---|
| Python | CPython 3.12.3 |
| torch | 2.7.0+cu128 (wheel from `https://download.pytorch.org/whl/cu128`; CUDA runtime 12.8, cuDNN 9.7.1) |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 (PyPI wheel, built with CUDA support) |
| numpy / pandas / scipy | 1.26.4 / 2.1.4 / 1.17.1 |
| pythermalcomfort | 4.0.2; SHA-256 of `pythermalcomfort/models/pmv_ppd_iso.py`: `56ea982c98d1f22475cf0530f3ebbf25eecc4febda0038c2be5b7a873f792aa7` (checked by the tests and by E7 / E7b before any computation) |
| pytest | 9.1.1 |

Hardware and system of the recorded run (from the run manifests): Ubuntu 24.04, one NVIDIA GeForce RTX 3090
(24 GB), driver 580.119.02.

## Install

From the repository root, with a CPython 3.12.3 interpreter:

```bash
python3.12 -m venv .venv-revision            # or: uv venv --python 3.12.3 --seed .venv-revision
.venv-revision/bin/python -m pip install --no-deps -r requirements-revision-2026-10.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128
.venv-revision/bin/python -m pip check
.venv-revision/bin/python -m pytest tests/test_environment.py -q
```

`--no-deps` installs exactly the distributions of the lock file. The torch wheel comes from the PyTorch
CUDA 12.8 index, every other distribution from PyPI. `.venv-revision/` is git-ignored.

## GPU and CPU

* The FT-Transformer and the ANN of E1 and the FTT adaptations of E6 ran on the GPU. RF, SVM and XGBoost
  ran on the CPU (`compute.xgb_device: cpu` in `config.yaml`).
* The xgboost 3.2.0 wheel is built with CUDA. On the recorded machine (NVIDIA driver installed), hiding the
  GPU (`CUDA_VISIBLE_DEVICES=`) makes every multi-class XGBoost fit fail with `cudaErrorNoDevice`, even with
  `device="cpu"`. The experiments that fit XGBoost (E1, E2, E4, E6, E10) were therefore run with the GPU
  visible. A machine without any NVIDIA driver has not been tested.
* E7, E7b, E9b and the regeneration of the figures and tables use the CPU only. E9b and the regeneration
  were checked with the GPU hidden and give byte-identical outputs.
* The tests run with the GPU hidden; the tests that need XGBoost multi-class fits or a CUDA device are
  skipped in that case, with the reason printed by `pytest -rs`.

## Configuration file

The values of `config.yaml` are those of the recorded run: its parsed content has the SHA-256 recorded by every
manifest (`config_used_sha256` `e859f3e2…6fda1`); only comment lines changed for the release (file SHA-256
`771fef24…` in the recorded run, `49053732…b30c` here; `revision_runners.RELEASE_FILE_SHA256`). Two entries are
leftovers of experiments that are not part of the paper: `features_18` (read by no code of this repository)
and `transfer.ftt_adaptation.lr.ftt_local` (only compared with the frozen adaptation budget; no FTT-local
strategy is run). They are kept so that the parsed configuration stays the recorded one.
