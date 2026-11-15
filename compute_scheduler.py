"""
Job scheduler of the revision rerun.

Runs independent fits concurrently on one workstation with a single GPU:

* **GPU jobs** -- every fit that uses the GPU: ANN, FTT, FTT adaptation and
  XGBoost when ``compute.xgb_device`` is ``cuda`` (``job_kind``). At most
  ``compute.gpu_concurrency`` of them run at once, in worker processes
  (``spawn``); ``gpu_concurrency: 1`` gives "one GPU job at a time".
  Each worker applies the FP32 / deterministic PyTorch settings
  (``compute.configure_torch_determinism``) and caps PyTorch and BLAS at
  ``compute.gpu_job_threads`` CPU threads ((cpus - cpu_jobs) // gpu_concurrency,
  >= 1, fixed by the config, not by the number of workers of a call), so that
  it does not compete with the CPU pool and a job runs with the same thread
  count whatever the scheduling.
* **CPU jobs** (RF, SVM, and XGB on the CPU) run **one at a time** in the
  calling process while the GPU jobs proceed, under ``compute.bounded_cpu``
  (BLAS capped at 1 thread): at most one heavy CPU pool (the job's own forest
  threads or grid-search workers, bounded by ``compute.cpu_jobs``) exists at any
  time. A job function with a ``compute`` parameter receives the scheduler's
  validated ``compute`` section, to build its estimators and grid search with
  the same budget (``get_models(compute=...)``, ``run_grid_search_on_train(compute=...)``).

Results do not depend on scheduling: every job carries its own fixed seed;
Python ``random``, NumPy and PyTorch are seeded with it immediately before the
job, and the job function receives it as ``seed=``. The tests check identical
predictions with ``gpu_concurrency`` 1 and 2 and with the job order reversed.

Each result records the wall time and, for GPU jobs on CUDA, the peak GPU
memory allocated/reserved by PyTorch in the job's process and the peak memory
used on the whole device during the job (sampled every 20 ms with
``cudaMemGetInfo``: includes XGBoost's allocations, every process's CUDA
context and the concurrent jobs), to choose ``gpu_concurrency``.

A failing job cancels the pending jobs; GPU jobs already running finish their
current fit before the error is raised (worker processes are not killed mid-fit).

Job functions must be module-level (picklable by reference) and return
picklable values without CUDA tensors (e.g. NumPy arrays, file paths).
"""
from __future__ import annotations

import inspect
import os
import pickle
import random
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Callable, Mapping

import numpy as np

from compute import bounded_cpu, gpu_job_threads, load_compute

KINDS = ("gpu", "cpu")
GPU_FAMILIES = ("ANN", "FTTransformer", "FTT_adaptation")
CPU_FAMILIES = ("RandomForest", "SVM")


class SchedulerError(RuntimeError):
    """Invalid job list."""


@dataclass(frozen=True)
class Job:
    """One independent fit. ``fn(seed=seed, **kwargs)`` is called in the worker."""
    key: str
    kind: str
    fn: Callable[..., Any]
    seed: int
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobResult:
    key: str
    kind: str
    seed: int
    value: Any
    seconds: float
    pid: int
    gpu_peak_allocated_bytes: int | None = None
    gpu_peak_reserved_bytes: int | None = None
    gpu_device_peak_used_bytes: int | None = None
    started_unix: float | None = None       # time.time() at the start / end of the job (comparable
    ended_unix: float | None = None         # across worker processes, e.g. overlap of concurrent jobs)


def job_records(results: Mapping[str, JobResult]) -> list[dict]:
    """JSON-serialisable per-job records (manifests): kind, seed, wall time, start/end, GPU peaks."""
    return [{"key": k, "kind": r.kind, "seed": int(r.seed), "seconds": r.seconds, "pid": r.pid,
             "started_unix": r.started_unix, "ended_unix": r.ended_unix,
             "gpu_peak_allocated_bytes": r.gpu_peak_allocated_bytes,
             "gpu_peak_reserved_bytes": r.gpu_peak_reserved_bytes,
             "gpu_device_peak_used_bytes": r.gpu_device_peak_used_bytes} for k, r in results.items()]


def job_kind(family: str, compute: dict) -> str:
    """Scheduler kind of a family's fit: ``gpu`` for every fit that uses the GPU."""
    if family in GPU_FAMILIES:
        return "gpu"
    if family == "XGBoost":
        return "gpu" if compute["xgb_device"] == "cuda" else "cpu"
    if family in CPU_FAMILIES:
        return "cpu"
    raise SchedulerError(f"unknown family {family!r}")


def seed_everything(seed: int) -> None:
    """Seed Python ``random``, NumPy and PyTorch (CPU and every CUDA device)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_WORKER_LIMITS = []      # keeps the worker's threadpoolctl limits alive


def _gpu_worker_init(torch_threads: int) -> None:
    import torch
    from threadpoolctl import threadpool_limits
    from compute import configure_torch_determinism
    configure_torch_determinism()
    torch.set_num_threads(torch_threads)
    _WORKER_LIMITS.append(threadpool_limits(limits=torch_threads, user_api="blas"))


class _DeviceMemorySampler:
    """Peak memory used on the whole CUDA device (all processes) while running."""

    def __init__(self, torch, period: float = 0.02):
        self.torch, self.period, self.peak = torch, period, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        free, total = self.torch.cuda.mem_get_info()
        self.peak = max(self.peak, total - free)

    def _loop(self):
        while not self._stop.wait(self.period):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def _call(job: Job, compute: dict):
    kwargs = dict(job.kwargs)
    if "compute" in inspect.signature(job.fn).parameters:
        kwargs["compute"] = compute
    return job.fn(seed=job.seed, **kwargs)


def _run_job(job: Job, gpu: bool, compute: dict) -> JobResult:
    cuda = False
    if gpu:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    seed_everything(job.seed)
    started = time.time()
    t0 = time.perf_counter()
    if cuda:
        with _DeviceMemorySampler(torch) as sampler:
            value = _call(job, compute)
            torch.cuda.synchronize()
    elif gpu:
        value = _call(job, compute)
    else:
        with bounded_cpu():                       # BLAS capped in the calling process
            value = _call(job, compute)
    seconds = time.perf_counter() - t0
    ended = time.time()
    peaks = ((torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), sampler.peak)
             if cuda else (None, None, None))
    return JobResult(job.key, job.kind, job.seed, value, seconds, os.getpid(), *peaks, started, ended)


def _validate(jobs) -> list[Job]:
    jobs = list(jobs)
    keys = [j.key for j in jobs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SchedulerError(f"duplicate job keys {dup}")
    for j in jobs:
        if not isinstance(j, Job):
            raise SchedulerError(f"not a Job: {j!r}")
        if j.kind not in KINDS:
            raise SchedulerError(f"[{j.key}] kind must be one of {KINDS}, got {j.kind!r}")
        if isinstance(j.seed, bool) or not isinstance(j.seed, (int, np.integer)):
            raise SchedulerError(f"[{j.key}] every job needs its own fixed integer seed, got {j.seed!r}")
        if "seed" in j.kwargs:
            raise SchedulerError(f"[{j.key}] the seed is passed by the scheduler, not in kwargs")
        if "compute" in j.kwargs:
            raise SchedulerError(f"[{j.key}] the compute section is passed by the scheduler, not in kwargs")
        if j.kind == "gpu":
            try:
                pickle.dumps(j)
            except Exception as exc:
                raise SchedulerError(f"[{j.key}] GPU jobs run in worker processes and must be "
                                     f"picklable (module-level function): {exc}") from exc
    return jobs


def gpu_worker_threads(compute: dict) -> int:
    """CPU threads (PyTorch, BLAS) per GPU worker: ``compute.gpu_job_threads`` (config only)."""
    return gpu_job_threads(compute)


def run_jobs(jobs, compute: dict | None = None, gpu_concurrency: int | None = None) -> dict[str, JobResult]:
    """
    Run ``jobs`` and return ``{key: JobResult}`` in the order of ``jobs``.

    GPU jobs are submitted first to a pool of ``gpu_concurrency`` processes
    (default ``compute.gpu_concurrency``); CPU jobs then run one at a time in
    this process. The first failing job (GPU failures are checked between CPU
    jobs) cancels the pending jobs and re-raises.
    """
    compute = load_compute() if compute is None else compute
    n_gpu = compute["gpu_concurrency"] if gpu_concurrency is None else int(gpu_concurrency)
    if n_gpu < 1:
        raise SchedulerError(f"gpu_concurrency must be >= 1, got {n_gpu}")
    jobs = _validate(jobs)
    gpu_jobs = [j for j in jobs if j.kind == "gpu"]
    cpu_jobs = [j for j in jobs if j.kind == "cpu"]
    results: dict[str, JobResult] = {}
    pool, futures = None, {}
    try:
        if gpu_jobs:
            n_workers = min(n_gpu, len(gpu_jobs))
            pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"),
                                       initializer=_gpu_worker_init,
                                       initargs=(gpu_worker_threads(compute),))
            futures = {pool.submit(_run_job, j, True, compute): j.key for j in gpu_jobs}
        for j in cpu_jobs:
            results[j.key] = _run_job(j, False, compute)
            for f in [f for f in futures if f.done() and f.exception() is not None]:
                raise f.exception()
        if futures:
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for f in futures:
                if f in done and f.exception() is not None:
                    raise f.exception()
            for f, key in futures.items():
                results[key] = f.result()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return {j.key: results[j.key] for j in jobs}
