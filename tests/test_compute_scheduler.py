# -*- coding: utf-8 -*-
"""
Job scheduler (``compute_scheduler.py``).

* results do not depend on scheduling: identical predictions with
  ``gpu_concurrency`` 1 and 2, and with the job order reversed (each job seeds
  Python ``random``, NumPy and PyTorch with its own seed);
* GPU jobs run in separate worker processes with the deterministic PyTorch
  settings, CPU jobs one at a time in the calling process;
* GPU jobs really overlap (N = 2) or not (N = 1); GPU XGB counts as a GPU job;
* thread limits inside CPU jobs (BLAS 1) and GPU workers (gpu_job_threads);
* GPU time and peak memory (process and device-wide) are recorded per job (CUDA only);
* invalid job lists and failing jobs raise.

Tiny ANN / FT-Transformer / RF fits on synthetic data; on the GPU when
available (CPU otherwise). The worker processes import ``models_deep`` (about
8 s each).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import compute  # noqa: E402
import compute_scheduler as cs  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY, vocabulary_for  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM = ["a", "b"]
CAT = ["Season", "Sexe"]
CFG = {"xgb_device": "cpu", "torch_device": DEVICE, "cpu_jobs": 2, "gpu_concurrency": 2}


def _data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.resize(np.array([-1.0, 0.0, 1.0]), n))
    X = pd.DataFrame({
        "a": rng.normal(size=n) + y,
        "b": rng.normal(size=n),
        "Season": rng.choice(sorted(GLOBAL_VOCABULARY["Season"]), n),
        "Sexe": rng.choice(sorted(GLOBAL_VOCABULARY["Sexe"]), n),
    })
    return X, y


# ---------------------------------------------------------------- job functions (module level)
def deep_fit_job(seed, kind, device):
    import models_deep as md
    X, y = _data()
    params = dict(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device=device,
                  epochs=3, batch_size=32, random_state=seed)
    m = (md.ANNClassifier(hidden=(8,), **params) if kind == "ANN"
         else md.FTClassifier(dim=8, depth=2, heads=2, dim_head=4, **params)).fit(X, y)
    out = {"pred": m.predict(X), "pid": os.getpid(),
           "deterministic": torch.are_deterministic_algorithms_enabled()}
    if kind == "FTT":
        out["proba"] = m.predict_proba(X)
    return out


def rf_fit_job(seed):
    import models_classic as mc
    X, y = _data()
    pipe = mc.ParallelFitRandomForestClassifier(n_estimators=16, random_state=seed, n_jobs=2)
    Xn = X[NUM].to_numpy()
    return {"proba": pipe.fit(Xn, y).predict_proba(Xn), "pid": os.getpid()}


def failing_job(seed):
    raise ArithmeticError(f"job with seed {seed} failed")


def timed_job(seed, pause):
    import time
    t0 = time.time()
    time.sleep(pause)
    return {"start": t0, "end": time.time(), "pid": os.getpid()}


def threads_job(seed, compute):
    """Reports the thread limits it runs under and the compute section it received."""
    from threadpoolctl import threadpool_info
    return {"blas": sorted({i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"}),
            "torch": torch.get_num_threads(), "compute": compute, "pid": os.getpid()}


def _jobs():
    return [cs.Job("ANN__s42", "gpu", deep_fit_job, 42, {"kind": "ANN", "device": DEVICE}),
            cs.Job("FTT__s42", "gpu", deep_fit_job, 42, {"kind": "FTT", "device": DEVICE}),
            cs.Job("FTT__s43", "gpu", deep_fit_job, 43, {"kind": "FTT", "device": DEVICE}),
            cs.Job("RF__s42", "cpu", rf_fit_job, 42)]


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k].value, b[k].value
        for f in ("pred", "proba"):
            if f in va:
                np.testing.assert_array_equal(va[f], vb[f], err_msg=f"{k}.{f}")


# ---------------------------------------------------------------- scheduling independence
def test_identical_predictions_with_one_or_two_gpu_workers_and_any_order():
    jobs = _jobs()
    one = cs.run_jobs(jobs, compute=CFG, gpu_concurrency=1)
    two = cs.run_jobs(jobs[::-1], compute=CFG, gpu_concurrency=2)
    assert list(one) == [j.key for j in jobs]                # results in job order
    assert list(two) == [j.key for j in jobs[::-1]]
    _same(one, two)
    # the seed is honoured (otherwise identity would be trivial)
    assert not np.array_equal(one["FTT__s42"].value["proba"], one["FTT__s43"].value["proba"])
    # same answer as a direct call in this process, with the workers' thread count
    # (on the CPU the float reductions depend on the number of PyTorch threads)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(cs.gpu_worker_threads(CFG))
    try:
        direct = deep_fit_job(42, "FTT", DEVICE)
    finally:
        torch.set_num_threads(n_threads)
    np.testing.assert_array_equal(direct["proba"], one["FTT__s42"].value["proba"])

    # GPU jobs in spawned workers with deterministic settings; CPU jobs in this process
    gpu_pids_one = {r.pid for r in one.values() if r.kind == "gpu"}
    gpu_pids_two = {r.pid for r in two.values() if r.kind == "gpu"}
    assert len(gpu_pids_one) == 1 and os.getpid() not in gpu_pids_one | gpu_pids_two
    assert len(gpu_pids_two) <= 2
    assert all(r.value["deterministic"] for r in one.values() if r.kind == "gpu")
    assert one["RF__s42"].pid == two["RF__s42"].pid == os.getpid()
    for r in one.values():
        assert r.seconds > 0
        if r.kind == "gpu" and DEVICE == "cuda":
            assert r.gpu_peak_allocated_bytes > 0 and r.gpu_peak_reserved_bytes >= r.gpu_peak_allocated_bytes
            # device-wide peak (CUDA context, other processes, non-PyTorch allocations) is larger
            assert r.gpu_device_peak_used_bytes > r.gpu_peak_reserved_bytes
        else:
            assert r.gpu_peak_allocated_bytes is None and r.gpu_device_peak_used_bytes is None


def test_gpu_jobs_really_run_concurrently_up_to_n():
    """With N = 2 two GPU jobs overlap in time; with N = 1 they never do."""
    jobs = [cs.Job(f"t{i}", "gpu", timed_job, i, {"pause": 1.5}) for i in range(2)]
    for n, overlap in ((2, True), (1, False)):
        res = cs.run_jobs(jobs, compute={**CFG, "gpu_concurrency": n})
        a, b = (res[f"t{i}"].value for i in range(2))
        assert (a["start"] < b["end"] and b["start"] < a["end"]) is overlap
        assert len({a["pid"], b["pid"]}) == n


def test_thread_limits_inside_jobs_and_compute_passed():
    """CPU jobs run under the bounded-CPU context (BLAS 1 thread) and receive
    the scheduler's compute section; GPU workers cap PyTorch and BLAS at gpu_job_threads."""
    res = cs.run_jobs([cs.Job("cpu", "cpu", threads_job, 1), cs.Job("gpu", "gpu", threads_job, 2)],
                      compute=CFG)
    cpu, gpu = res["cpu"].value, res["gpu"].value
    assert cpu["blas"] == [1] and cpu["compute"] == CFG and cpu["pid"] == os.getpid()
    n = cs.gpu_worker_threads(CFG)
    assert gpu["torch"] == n and gpu["blas"] == [n] and gpu["compute"] == CFG
    from threadpoolctl import threadpool_info          # the parent's limits are restored
    assert any(i["num_threads"] > 1 for i in threadpool_info() if i["user_api"] == "blas") or os.cpu_count() == 1


def test_job_kind_counts_gpu_xgb_as_a_gpu_job():
    """XGB on the GPU counts against gpu_concurrency (1 -> one GPU job at a time)."""
    gpu, cpu = {**CFG, "xgb_device": "cuda"}, {**CFG, "xgb_device": "cpu"}
    assert [cs.job_kind(f, gpu) for f in ("ANN", "FTTransformer", "FTT_adaptation", "XGBoost")] == ["gpu"] * 4
    assert [cs.job_kind(f, gpu) for f in ("RandomForest", "SVM")] == ["cpu"] * 2
    assert cs.job_kind("XGBoost", cpu) == "cpu"
    with pytest.raises(cs.SchedulerError, match="unknown family"):
        cs.job_kind("TabPFN", gpu)


def test_cpu_jobs_only_run_in_process():
    res = cs.run_jobs([cs.Job("RF__s42", "cpu", rf_fit_job, 42), cs.Job("RF__s43", "cpu", rf_fit_job, 43)],
                      compute=CFG)
    np.testing.assert_array_equal(res["RF__s42"].value["proba"], rf_fit_job(42)["proba"])
    assert not np.array_equal(res["RF__s42"].value["proba"], res["RF__s43"].value["proba"])


def test_gpu_worker_threads_leave_the_cpu_pool_its_cores():
    """Fixed by the config (cpu_jobs, gpu_concurrency), not by the N of a call."""
    n = os.cpu_count()
    assert cs.gpu_worker_threads({"cpu_jobs": n - 4, "gpu_concurrency": 2}) == 2
    assert cs.gpu_worker_threads({"cpu_jobs": n - 4, "gpu_concurrency": 1}) == 4
    assert cs.gpu_worker_threads({"cpu_jobs": n, "gpu_concurrency": 2}) == 1


# ---------------------------------------------------------------- validation and failures
@pytest.mark.parametrize("jobs, match", [
    ([cs.Job("a", "cpu", rf_fit_job, 1), cs.Job("a", "cpu", rf_fit_job, 2)], "duplicate"),
    ([cs.Job("a", "tpu", rf_fit_job, 1)], "kind"),
    ([cs.Job("a", "cpu", rf_fit_job, None)], "seed"),
    ([cs.Job("a", "cpu", rf_fit_job, True)], "seed"),
    ([cs.Job("a", "cpu", rf_fit_job, 1, {"seed": 2})], "seed"),
    ([cs.Job("a", "cpu", threads_job, 1, {"compute": {}})], "compute"),
    ([cs.Job("a", "gpu", lambda seed: seed, 1)], "picklable"),
])
def test_invalid_jobs_raise(jobs, match):
    with pytest.raises(cs.SchedulerError, match=match):
        cs.run_jobs(jobs, compute=CFG)


def test_failing_jobs_raise():
    with pytest.raises(ArithmeticError, match="seed 7"):
        cs.run_jobs([cs.Job("bad", "cpu", failing_job, 7)], compute=CFG)
    with pytest.raises(ArithmeticError, match="seed 8"):
        cs.run_jobs([cs.Job("bad", "gpu", failing_job, 8)], compute=CFG, gpu_concurrency=1)


def test_default_concurrency_comes_from_the_config():
    assert compute.load_compute()["gpu_concurrency"] == 1      # one GPU job at a time by default
    with pytest.raises(cs.SchedulerError, match="gpu_concurrency"):
        cs.run_jobs([], compute=CFG, gpu_concurrency=0)
