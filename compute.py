"""
Compute settings of the revision rerun.

Training of the deep models runs on a single GPU; the CPU is kept for the
CPU-optimised families. Nothing here changes a frozen model hyperparameter;
it only decides *where* and *with how many workers* a fit runs.

* **XGBoost** stays on the CPU by default (``compute.xgb_device: cpu``, as in
  the canonical run; the GPU is used only if it is useful, decided before the
  run). The optional GPU path (``xgb_device: cuda``) uses
  ``device="cuda"``, ``tree_method="hist"``; CUDA unavailable →
  ``ComputeConfigError`` (no silent fallback: XGBoost itself would warn and
  fall back to the CPU, whose sampling stream differs). The check runs
  when an XGB estimator is built (only the requested families are built, so
  RF/SVM-only or deep-only workflows do not need CUDA) and again after each fit
  (``check_xgb_booster_device``). A GPU XGB fit is a *GPU job* for the
  scheduler (it counts against ``gpu_concurrency``) and uses
  ``gpu_job_threads`` host threads; on the CPU it uses ``cpu_jobs`` threads.
* **RandomForest and SVM** stay on the CPU (scikit-learn), bounded by
  ``compute.cpu_jobs`` (default 20) with one level of parallelism
  only: RF parallelises over trees (``n_jobs = cpu_jobs``) inside a sequential
  grid search; SVM (no ``n_jobs``) parallelises the grid search over
  ``min(cpu_jobs, candidates x folds)`` workers; BLAS threads are capped at 1
  (``threadpoolctl`` in the parent, ``inner_max_num_threads=1`` in the joblib
  workers).
* **PyTorch** (ANN, FTT, FTT adaptation): FP32 numerics (TF32 disabled
  explicitly, no AMP) and deterministic kernels (``cudnn.deterministic``,
  ``benchmark=False``, ``CUBLAS_WORKSPACE_CONFIG``,
  ``torch.use_deterministic_algorithms(True)``); ``compute.torch_device``
  fixes the device of the runners' estimators (same no-fallback rule).

``compute.gpu_concurrency`` is read by ``compute_scheduler`` (number of
deep-model fits sharing the GPU in separate processes).
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from config_loader import load_config

CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
COMPUTE_KEYS = ("xgb_device", "torch_device", "cpu_jobs", "gpu_concurrency")
DEVICES = ("cuda", "cpu")
# cuBLAS needs a fixed workspace for deterministic GEMMs (PyTorch reproducibility notes).
CUBLAS_DETERMINISTIC_CONFIGS = (":4096:8", ":16:8")
CPU_FAMILIES = ("RandomForest", "SVM")


class ComputeConfigError(RuntimeError):
    """Invalid ``compute`` section, or a requested device that is not available."""


def load_compute(config: dict | None = None) -> dict:
    """
    The validated ``compute`` section of ``config`` (default: the repository
    ``config.yaml``). Every key of ``COMPUTE_KEYS`` is required; devices are
    ``cuda`` or ``cpu``; ``cpu_jobs`` is in [1, number of logical CPUs];
    ``gpu_concurrency`` >= 1.
    """
    config = load_config(CONFIG_PATH) if config is None else config
    section = config.get("compute")
    if not isinstance(section, dict):
        raise ComputeConfigError("config.yaml has no 'compute' section")
    missing = [k for k in COMPUTE_KEYS if k not in section]
    if missing:
        raise ComputeConfigError(f"compute section lacks {missing}")
    out = {k: section[k] for k in COMPUTE_KEYS}
    for k in ("xgb_device", "torch_device"):
        if out[k] not in DEVICES:
            raise ComputeConfigError(f"compute.{k} must be one of {DEVICES}, got {out[k]!r}")
    n_cpu = os.cpu_count() or 1
    for k, hi in (("cpu_jobs", n_cpu), ("gpu_concurrency", None)):
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1 or (hi is not None and v > hi):
            raise ComputeConfigError(f"compute.{k} must be an integer in [1, {hi or 'inf'}], got {v!r}")
    return out


def cuda_available() -> bool:
    """CUDA device visible to PyTorch (patched in the tests)."""
    import torch
    return torch.cuda.is_available()


def _require(device: str, what: str) -> str:
    if device == "cuda" and not cuda_available():
        raise ComputeConfigError(
            f"compute.{what} is 'cuda' but no CUDA device is available; set it to 'cpu' explicitly "
            "(no silent fallback)")
    return device


def xgb_device(compute: dict | None = None) -> str:
    """Device of every XGBoost fit of the runners."""
    compute = load_compute() if compute is None else compute
    device = _require(compute["xgb_device"], "xgb_device")
    if device == "cuda":
        import xgboost
        if not xgboost.build_info().get("USE_CUDA", False):
            raise ComputeConfigError("compute.xgb_device is 'cuda' but this xgboost build has no CUDA support")
    return device


def torch_device(compute: dict | None = None) -> str:
    """Device of the runners' ANN/FTT estimators."""
    compute = load_compute() if compute is None else compute
    return _require(compute["torch_device"], "torch_device")


def gpu_job_threads(compute: dict) -> int:
    """
    Host (CPU) threads of one GPU job: the cores left by the CPU pool, shared by
    the ``gpu_concurrency`` GPU jobs. It depends on the config only (not on the
    number of workers of a given scheduler call), so a job runs with the same
    thread count whatever the scheduling.
    """
    return max(1, ((os.cpu_count() or 1) - compute["cpu_jobs"]) // compute["gpu_concurrency"])


def xgb_compute_params(compute: dict | None = None) -> dict:
    """
    XGBoost compute parameters (hyperparameters untouched): device, hist, host
    threads (``cpu_jobs`` on the CPU; ``gpu_job_threads`` on the GPU, where the
    trees are built on the device and the predictions do not depend on the host
    thread count, test).
    """
    compute = load_compute() if compute is None else compute
    device = xgb_device(compute)
    n_jobs = compute["cpu_jobs"] if device == "cpu" else gpu_job_threads(compute)
    return {"device": device, "tree_method": "hist", "n_jobs": n_jobs}


def check_xgb_booster_device(xgb_estimator) -> None:
    """
    After a fit: the booster of an XGB estimator built for ``cuda`` really
    trained on a CUDA device (no silent fallback). No-op for a CPU estimator.
    """
    import json
    requested = str(xgb_estimator.get_params().get("device") or "cpu")
    if not requested.startswith("cuda"):
        return
    config = json.loads(xgb_estimator.get_booster().save_config())
    used = str(config["learner"]["generic_param"].get("device", ""))
    if not used.startswith("cuda"):
        raise ComputeConfigError(f"XGBoost was asked for device {requested!r} but its booster reports "
                                 f"{used!r} (silent fallback)")


def grid_search_n_jobs(model_name: str, n_fits: int, compute: dict | None = None) -> int:
    """
    ``GridSearchCV(n_jobs=...)`` for a classical family, without nested parallelism:
    RF (parallel over trees) and XGB (GPU, or its own threads) search sequentially;
    SVM searches over ``min(cpu_jobs, n_fits)`` workers.
    """
    compute = load_compute() if compute is None else compute
    if model_name == "SVM":
        return max(1, min(compute["cpu_jobs"], int(n_fits)))
    if model_name in ("RandomForest", "XGBoost"):
        return 1
    raise ValueError(f"unknown classical family {model_name!r}")


def check_grid_search_jobs(model_name: str, n_jobs, estimator_n_jobs, compute: dict,
                           estimator_device: str = "cpu") -> None:
    """
    Refuse a grid search that would nest parallelism or exceed the CPU budget
    (compute rules): ``n_jobs`` in [1, ``cpu_jobs``]; RF and XGB
    (internally parallel, or on the GPU) only with a sequential search
    (``n_jobs == 1``) and their own ``n_jobs`` in [1, ``cpu_jobs``]
    ([1, ``gpu_job_threads``] for a GPU XGB estimator).
    """
    cap = compute["cpu_jobs"]
    est_cap = gpu_job_threads(compute) if str(estimator_device).startswith("cuda") else cap
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= cap:
        raise ValueError(f"n_jobs={n_jobs!r}: CPU workers are bounded by compute.cpu_jobs={cap}")
    if model_name in ("RandomForest", "XGBoost"):
        if n_jobs != 1:
            raise ValueError(f"{model_name}: grid-search n_jobs={n_jobs} would nest parallelism with the "
                             "estimator's own threads (or GPU fits); RF/XGB search sequentially (n_jobs=1)")
        if (isinstance(estimator_n_jobs, bool) or not isinstance(estimator_n_jobs, int)
                or not 1 <= estimator_n_jobs <= est_cap):
            raise ValueError(f"{model_name}: estimator n_jobs={estimator_n_jobs!r} must be in "
                             f"[1, {est_cap}] (compute.cpu_jobs / gpu_job_threads)")


@contextlib.contextmanager
def bounded_cpu(process_workers: bool = False, blas_threads: int = 1):
    """
    Cap BLAS threads in this process and, when the grid search runs in
    ``process_workers`` (loky), in each worker too: no nested parallelism under
    a parallel grid search or a parallel forest. The loky backend is only forced
    for a process-parallel grid search, so that the forest keeps its thread
    backend (``prefer="threads"``) inside a sequential one.
    """
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits
    workers = (parallel_config(backend="loky", inner_max_num_threads=blas_threads)
               if process_workers else contextlib.nullcontext())
    with threadpool_limits(limits=blas_threads, user_api="blas"), workers:
        yield


def configure_torch_determinism() -> None:
    """
    FP32 and deterministic PyTorch settings.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before cuBLAS is first used: this
    function is called at import of ``models_deep`` and in every scheduler
    worker. A value other than the deterministic ones raises.
    """
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_DETERMINISTIC_CONFIGS[0])
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in CUBLAS_DETERMINISTIC_CONFIGS:
        raise ComputeConfigError(
            f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']!r} is not deterministic; "
            f"use one of {CUBLAS_DETERMINISTIC_CONFIGS}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def torch_determinism_state() -> dict:
    """Current settings, for the manifest and the tests."""
    import torch
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
