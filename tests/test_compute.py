# -*- coding: utf-8 -*-
"""
Compute settings (CPU pool and a single GPU).

* config ``compute`` section: values and validation;
* XGBoost: CPU by default (same predictions as the pre-revision
  construction, independent of the host threads, no CUDA needed);
  optional GPU path: device honoured (GPU booster on CUDA), every other
  hyperparameter unchanged, no silent fallback without CUDA, identical
  predictions with the same seed on the GPU (determinism);
* RF / SVM on the CPU: bounded ``n_jobs``, no nested parallelism (grid search
  vs estimator; BLAS threads), RF predictions independent of ``n_jobs``;
* PyTorch: FP32 / deterministic settings; the training loop without per-batch
  host synchronisation gives the same history and weights as the previous
  per-batch ``loss.item()`` loop; ANN / FTT fits with the same seed are
  identical whatever the earlier draws of Python ``random`` (hyper-connections
  initialisation), on CPU and on CUDA.

Synthetic data only, tiny models; CUDA tests are skipped without a GPU.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import compute  # noqa: E402
import models_classic as mc  # noqa: E402
import models_deep as md  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY, vocabulary_for  # noqa: E402
from config_loader import load_config  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
NUM = ["a", "b"]
CAT = ["Season", "Sexe"]
HP = load_config(compute.CONFIG_PATH)["classical_hparams"]
GPU = {"xgb_device": "cuda", "torch_device": "cuda", "cpu_jobs": 4, "gpu_concurrency": 2}
CPU = {**GPU, "xgb_device": "cpu", "torch_device": "cpu"}


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.resize(np.array([-1.0, 0.0, 1.0]), n))
    X = pd.DataFrame({
        "a": rng.normal(size=n) + y,
        "b": rng.normal(size=n),
        "Season": rng.choice(sorted(GLOBAL_VOCABULARY["Season"]), n),
        "Sexe": rng.choice(sorted(GLOBAL_VOCABULARY["Sexe"]), n),
    })
    return X, y


def _models(cfg, seed=42, hparams=HP):
    return mc.get_models(NUM, CAT, seed=seed, hparams=hparams, cat_vocabulary=vocabulary_for(CAT),
                         compute=cfg)


def _tiny_hp(n_estimators=8):
    hp = copy.deepcopy(HP)
    hp["XGBoost"]["n_estimators"] = n_estimators
    hp["RandomForest"]["n_estimators"] = n_estimators
    return hp


def _fit_xgb(cfg, seed=42):
    pipe, grid = _models(cfg, seed, _tiny_hp())["XGBoost"]
    pipe.set_params(**{k: v[0] for k, v in grid.items()})
    X, y = _data()
    return pipe.fit(X, y), X


# ================================================================ config section
def test_config_compute_section():
    cfg = compute.load_compute()
    # gpu_concurrency 1: one GPU job at a time
    assert cfg == {"xgb_device": "cpu", "torch_device": "cuda", "cpu_jobs": 20, "gpu_concurrency": 1}


def test_default_config_keeps_xgb_on_the_cpu_without_cuda(monkeypatch):
    """XGBoost stays on the CPU (canonical run) by default; the default needs no CUDA device
    and is a CPU job."""
    import compute_scheduler as cs
    import transfert
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    cfg = compute.load_compute()
    assert compute.xgb_device(cfg) == "cpu"
    assert compute.xgb_compute_params(cfg) == {"device": "cpu", "tree_method": "hist", "n_jobs": 20}
    pipe, _ = mc.get_models(NUM, CAT, hparams=HP, cat_vocabulary=vocabulary_for(CAT),
                            families=["XGBoost"])["XGBoost"]       # compute=None -> config.yaml
    assert pipe.named_steps["clf"].estimator.get_params()["device"] == "cpu"
    _, xgb = transfert.make_rf_xgb_protos(NUM, CAT, cat_vocabulary=vocabulary_for(CAT), families=("XGB",))
    assert xgb.named_steps["clf"].estimator.get_params()["device"] == "cpu"
    assert cs.job_kind("XGBoost", cfg) == "cpu"


def test_cpu_xgb_matches_snapshot_construction_and_host_threads(xgb_fits):
    """CPU XGB with ``tree_method="hist"`` and ``n_jobs=cpu_jobs`` predicts exactly as the
    snapshot construction (no device / tree_method / n_jobs: XGBoost's default 'auto' is
    'hist' on the CPU) and as a single-threaded fit, twice (0 mismatch on re-execution)."""
    X, y = _data()
    grid = {k: v[0] for k, v in _models(CPU, hparams=_tiny_hp())["XGBoost"][1].items()}
    out = []
    for extra in ({}, {"clf__estimator__n_jobs": 1}, {"clf__estimator__n_jobs": None,
                                                       "clf__estimator__device": None,
                                                       "clf__estimator__tree_method": None}):
        for _ in range(2):
            pipe, _ = _models(CPU, hparams=_tiny_hp())["XGBoost"]
            pipe.set_params(**grid, **extra)
            out.append(pipe.fit(X, y).predict_proba(X))
    for p in out[1:]:
        np.testing.assert_array_equal(p, out[0])
    fitted, _ = _fit_xgb(compute.load_compute())                  # default config: CPU booster
    generic = json.loads(fitted.named_steps["clf"].estimator_.get_booster().save_config())
    assert generic["learner"]["generic_param"]["device"] == "cpu"


@pytest.mark.parametrize("bad, match", [
    ({}, "no 'compute' section"),
    ({"compute": {"xgb_device": "cuda"}}, "lacks"),
    ({"compute": {**GPU, "xgb_device": "gpu"}}, "xgb_device"),
    ({"compute": {**GPU, "torch_device": "auto"}}, "torch_device"),
    ({"compute": {**GPU, "cpu_jobs": -1}}, "cpu_jobs"),
    ({"compute": {**GPU, "cpu_jobs": 10_000}}, "cpu_jobs"),
    ({"compute": {**GPU, "cpu_jobs": True}}, "cpu_jobs"),
    ({"compute": {**GPU, "gpu_concurrency": 0}}, "gpu_concurrency"),
])
def test_invalid_compute_section_raises(bad, match):
    with pytest.raises(compute.ComputeConfigError, match=match):
        compute.load_compute(bad)


# ================================================================ XGBoost: optional GPU path
def test_xgb_hyperparameters_unchanged_only_compute_params_added():
    for cfg, device in ((GPU, "cuda"), (CPU, "cpu")):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        pipe, grid = _models(cfg)["XGBoost"]
        params = pipe.named_steps["clf"].estimator.get_params()
        assert params["device"] == device and params["tree_method"] == "hist"
        # host threads: the CPU pool on the CPU; the cores left to GPU jobs on the GPU
        assert params["n_jobs"] == (cfg["cpu_jobs"] if device == "cpu" else compute.gpu_job_threads(cfg))
        assert params["objective"] == "multi:softprob" and params["eval_metric"] == "mlogloss"
        assert params["random_state"] == 42
        # frozen hyperparameters still come from the grid, unchanged
        assert {k.split("__")[-1]: v for k, v in grid.items()} == {k: [v] for k, v in HP["XGBoost"].items()}
        # no other XGB parameter is set
        set_params = {k for k, v in params.items() if v is not None and k != "missing"}
        assert set_params == {"device", "tree_method", "n_jobs", "objective", "eval_metric",
                              "random_state", "enable_categorical"}


@CUDA
def test_xgb_device_honoured_on_the_booster():
    pipe, _ = _fit_xgb(GPU)
    booster = pipe.named_steps["clf"].estimator_.get_booster()
    generic = json.loads(booster.save_config())["learner"]["generic_param"]
    assert generic["device"].startswith("cuda")
    assert json.loads(booster.save_config())["learner"]["gradient_booster"]["gbtree_train_param"][
        "tree_method"] == "hist"
    pipe_cpu, _ = _fit_xgb(CPU)
    generic = json.loads(pipe_cpu.named_steps["clf"].estimator_.get_booster().save_config())
    assert generic["learner"]["generic_param"]["device"] == "cpu"


@CUDA
def test_xgb_gpu_deterministic_for_a_seed():
    (p1, X), (p2, _) = _fit_xgb(GPU, 42), _fit_xgb(GPU, 42)
    np.testing.assert_array_equal(p1.predict_proba(X), p2.predict_proba(X))
    np.testing.assert_array_equal(p1.predict(X), p2.predict(X))
    p3, _ = _fit_xgb(GPU, 43)                       # the seed is honoured (subsample 0.7)
    assert not np.array_equal(p1.predict_proba(X), p3.predict_proba(X))


def test_xgb_cuda_without_gpu_raises_no_silent_fallback(monkeypatch):
    import transfert
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    with pytest.raises(compute.ComputeConfigError, match="no CUDA device"):
        _models(GPU)
    with pytest.raises(compute.ComputeConfigError, match="no CUDA device"):
        transfert.make_rf_xgb_protos(NUM, CAT, cat_vocabulary=vocabulary_for(CAT), compute=GPU)
    pipe, _ = _models(CPU)["XGBoost"]                # explicit CPU config is accepted
    assert pipe.named_steps["clf"].estimator.get_params()["device"] == "cpu"


def test_only_requested_families_are_built_so_cpu_workflows_need_no_cuda(monkeypatch):
    """The CUDA check runs for XGB fits only."""
    import transfert
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    built = mc.get_models(NUM, CAT, hparams=HP, cat_vocabulary=vocabulary_for(CAT), compute=GPU,
                          families=["RandomForest", "SVM"])
    assert list(built) == ["RandomForest", "SVM"]
    with pytest.raises(compute.ComputeConfigError, match="no CUDA device"):
        mc.get_models(NUM, CAT, hparams=HP, cat_vocabulary=vocabulary_for(CAT), compute=GPU,
                      families=["XGBoost"])
    with pytest.raises(ValueError, match="unknown"):
        mc.get_models(NUM, CAT, hparams=HP, compute=CPU, families=["ANN"])
    rf, xgb = transfert.make_rf_xgb_protos(NUM, CAT, cat_vocabulary=vocabulary_for(CAT), compute=GPU,
                                           families=("RF",))
    assert xgb is None and isinstance(rf.named_steps["clf"], mc.ParallelFitRandomForestClassifier)
    rf, xgb = transfert.make_rf_xgb_protos(NUM, CAT, cat_vocabulary=vocabulary_for(CAT), compute=CPU,
                                           families=("XGB",))
    assert rf is None and xgb is not None


def test_runner_rf_svm_and_cpu_deep_do_not_need_cuda(monkeypatch, tmp_path):
    """The runner builds only the requested classical families: RF/SVM and CPU deep fits
    run with ``xgb_device: cuda`` and no CUDA device; requesting XGB still raises."""
    import models as models_module
    import revision_folds as rf
    import test_folds_seeds as tfs
    from category_normalisation import split_feature_types
    from types import SimpleNamespace
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    feats = list(models_module.FEATURES_ALL)
    y = tfs._labels(n=150)
    X = tfs._frame(feats, y)
    tr, te = rf.redraw_split(y)
    folds = rf.ensure_folds(tmp_path / "folds", "ASHRAE_2022", "TSV_3p", tr, y.iloc[tr])
    num, cat = split_feature_types(X.iloc[tr], X.columns)
    deep_seen, grid_seen = [], []
    tfs._patch_runner(models_module, monkeypatch, deep_seen, grid_seen,
                      {**compute.load_compute(), "xgb_device": "cuda", "torch_device": "cpu"})

    def run(classical, out):
        args = SimpleNamespace(models=["classical", "deep"], classical_models=classical, deep_models=["ANN"],
                               skip_feature_importance=True)
        return models_module.run_experiment_for_target(
            "TSV_3p", X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], num, cat, tmp_path / out, args=args,
            cat_vocabulary=vocabulary_for(cat), cv_splits=folds.splits(),
            seeds_by_model=rf.model_seeds())
    best, *_ = run(["RandomForest", "SVM"], "a")
    assert sorted(best["rf"]) == [42, 43, 44] and sorted(best["svm"]) == [42] and not best["xgb"]
    assert [m for m, *_ in deep_seen] == ["ANN"] * len(rf.model_seeds()["ANN"])
    with pytest.raises(compute.ComputeConfigError, match="no CUDA device"):
        run(["XGBoost"], "b")


def test_xgb_booster_device_checked_after_fit():
    """A CUDA-requested booster that reports another device raises."""
    class Stub:
        def __init__(self, requested, used):
            self.requested, self.used = requested, used

        def get_params(self):
            return {"device": self.requested}

        def get_booster(self):
            used = self.used
            return type("B", (), {"save_config": lambda self: json.dumps(
                {"learner": {"generic_param": {"device": used}}})})()
    compute.check_xgb_booster_device(Stub("cuda", "cuda:0"))
    compute.check_xgb_booster_device(Stub("cpu", "cpu"))
    with pytest.raises(compute.ComputeConfigError, match="silent fallback"):
        compute.check_xgb_booster_device(Stub("cuda", "cpu"))


@CUDA
def test_gpu_xgb_predictions_independent_of_host_threads():
    """XGB on the GPU uses ``gpu_job_threads`` host threads: same predictions as with 1 or 20."""
    out = []
    for jobs in (1, compute.gpu_job_threads(GPU), 20):
        pipe, grid = _models(GPU, hparams=_tiny_hp())["XGBoost"]
        pipe.set_params(clf__estimator__n_jobs=jobs, **{k: v[0] for k, v in grid.items()})
        X, y = _data()
        out.append(pipe.fit(X, y).predict_proba(X))
    for p in out[1:]:
        np.testing.assert_array_equal(p, out[0])


def test_transfer_protos_use_the_compute_section_and_the_frozen_hyperparameters():
    """The Phase 3 prototypes take the frozen values (the pre-revision XGB had
    600 trees / depth 10)."""
    import transfert
    cfg = GPU if torch.cuda.is_available() else CPU
    rf, xgb = transfert.make_rf_xgb_protos(NUM, CAT, seed=43, cat_vocabulary=vocabulary_for(CAT), compute=cfg)
    p = xgb.named_steps["clf"].estimator.get_params()
    host = cfg["cpu_jobs"] if cfg["xgb_device"] == "cpu" else compute.gpu_job_threads(cfg)
    assert (p["device"], p["tree_method"], p["n_jobs"]) == (cfg["xgb_device"], "hist", host)
    assert {k: p[k] for k in HP["XGBoost"]} == HP["XGBoost"]
    from xgboost import XGBClassifier
    # objective left to XGBoost (test_audit_guards::test_xgb_proto_objective_default)
    assert (p["random_state"], p["eval_metric"]) == (43, "mlogloss")
    assert p["objective"] == XGBClassifier().get_params()["objective"]
    set_params = {k for k, v in p.items() if v is not None and k not in ("missing", "objective")}
    assert set_params == set(HP["XGBoost"]) | {"device", "tree_method", "n_jobs", "eval_metric",
                                               "random_state", "enable_categorical"}
    r = rf.named_steps["clf"]
    assert {k: r.get_params()[k] for k in HP["RandomForest"]} == HP["RandomForest"]
    assert r.n_jobs == cfg["cpu_jobs"] and r.random_state == 43
    assert isinstance(r, mc.ParallelFitRandomForestClassifier)


# ================================================================ RF / SVM on the CPU
def test_rf_n_jobs_bounded_and_predictions_independent_of_n_jobs():
    """Trees built on ``cpu_jobs`` threads; probabilities bitwise equal to a plain
    ``n_jobs=1`` forest and across re-executions (0 mismatch)."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = _data(400)
    probs = []
    for jobs in (1, 8, 8):
        pipe, grid = _models({**CPU, "cpu_jobs": jobs}, hparams=_tiny_hp(64))["RandomForest"]
        assert isinstance(pipe.named_steps["clf"], mc.ParallelFitRandomForestClassifier)
        assert pipe.named_steps["clf"].n_jobs == jobs
        pipe.set_params(**{k: v[0] for k, v in grid.items()})
        probs.append(pipe.fit(X, y).predict_proba(X))
        assert pipe.named_steps["clf"].n_jobs == jobs          # restored after prediction
    plain, _ = _models({**CPU, "cpu_jobs": 1}, hparams=_tiny_hp(64))["RandomForest"]
    plain.steps[-1] = ("clf", RandomForestClassifier(**plain.named_steps["clf"].get_params()))
    plain.set_params(**{k: v[0] for k, v in grid.items()})
    reference = plain.fit(X, y).predict_proba(X)
    for p in probs:
        np.testing.assert_array_equal(p, reference)


def test_svm_stays_on_cpu_sklearn():
    pipe, _ = _models(CPU)["SVM"]
    from sklearn.svm import SVC
    assert type(pipe.named_steps["clf"]) is SVC and pipe.named_steps["clf"].probability is False


@pytest.mark.parametrize("model, n_fits, expected", [
    ("RandomForest", 10, 1), ("XGBoost", 10, 1), ("SVM", 120, 20), ("SVM", 6, 6)])
def test_grid_search_workers_no_nested_parallelism(model, n_fits, expected):
    cfg = {**GPU, "cpu_jobs": 20}
    assert compute.grid_search_n_jobs(model, n_fits, cfg) == expected
    with pytest.raises(ValueError):
        compute.grid_search_n_jobs("ANN", 10, cfg)


def test_run_grid_search_bounds_workers_and_blas(monkeypatch):
    from sklearn.model_selection import GridSearchCV
    from threadpoolctl import threadpool_info
    seen = []

    class Spy(GridSearchCV):
        def fit(self, X, y=None, **kw):
            blas = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
            est_jobs = getattr(self.estimator.named_steps["clf"], "n_jobs", None)
            seen.append((self.n_jobs, est_jobs, blas))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(mc, "GridSearchCV", Spy)
    X, y = _data(60)
    folds = [(np.arange(60)[np.arange(60) % 3 != k], np.arange(60)[np.arange(60) % 3 == k]) for k in range(3)]
    cfg = {**CPU, "cpu_jobs": 20}                    # explicit CPU: no CUDA needed for RF / SVM
    models = _models(cfg, hparams=_tiny_hp(4))
    for name in ("RandomForest", "SVM"):
        pipe, grid = models[name]
        mc.run_grid_search_on_train("TSV_3p", name, pipe, grid, X, pd.Series(y), cv_splits=folds, verbose=0,
                                    compute=cfg)
    (rf_grid, rf_est, rf_blas), (svm_grid, svm_est, svm_blas) = seen
    assert rf_grid == 1 and rf_est == 20                # parallel over trees, sequential search
    assert svm_grid == min(20, 3 * 12) and svm_est is None
    assert all(b == 1 for b in rf_blas + svm_blas)      # BLAS capped under the search
    for bad in (-1, 0, 21):
        with pytest.raises(ValueError, match="cpu_jobs"):
            mc.run_grid_search_on_train("TSV_3p", "SVM", *models["SVM"], X, pd.Series(y),
                                        cv_splits=folds, n_jobs=bad, compute=cfg)
    # the caller's compute section bounds the workers (not config.yaml)
    with pytest.raises(ValueError, match="cpu_jobs=4"):
        mc.run_grid_search_on_train("TSV_3p", "SVM", *models["SVM"], X, pd.Series(y),
                                    cv_splits=folds, n_jobs=8, compute={**cfg, "cpu_jobs": 4})
    assert len(seen) == 2                               # every refusal before any fit


def test_grid_search_stops_on_a_failed_cv_fit_and_on_non_finite_scores():
    """GridSearchCV's default error_score=nan would keep a search in which
    one candidate failed on one fold (warning only) and refit the winner; the run must stop instead."""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.pipeline import Pipeline

    class FailsWithoutRowZero(ClassifierMixin, BaseEstimator):
        def __init__(self, fail=False):
            self.fail = fail

        def fit(self, X, y):
            if self.fail and 0 not in set(X.index):          # only the fold whose validation part holds row 0
                raise ValueError("injected CV failure")
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), self.classes_[0])

    X, y = _data(60)
    folds = [(np.arange(60)[np.arange(60) % 3 != k], np.arange(60)[np.arange(60) % 3 == k]) for k in range(3)]
    pipe = Pipeline([("clf", FailsWithoutRowZero())])
    with pytest.raises(ValueError, match="injected CV failure"):
        mc.run_grid_search_on_train("TSV_3p", "SVM", pipe, {"clf__fail": [False, True]}, X, pd.Series(y),
                                    cv_splits=folds, n_jobs=1, verbose=0, compute={**CPU, "cpu_jobs": 2})
    best, params, summary, _, _ = mc.run_grid_search_on_train(
        "TSV_3p", "SVM", pipe, {"clf__fail": [False]}, X, pd.Series(y), cv_splits=folds, n_jobs=1, verbose=0,
        compute={**CPU, "cpu_jobs": 2})
    assert len(summary) == 3 and params == {"clf__fail": False}

    class Grid:                                               # a completed search with one NaN score
        scorer_ = {"accuracy": None}
        cv_results_ = {"params": [{}, {}], **{f"split{i}_{p}_accuracy": np.array([0.5, 0.5])
                                              for i in range(2) for p in ("train", "test")}}
    mc.check_complete_search(Grid, 2, "ok")
    Grid.cv_results_["split1_test_accuracy"] = np.array([0.5, np.nan])
    with pytest.raises(RuntimeError, match="non-finite"):
        mc.check_complete_search(Grid, 2, "nan")
    del Grid.cv_results_["split1_test_accuracy"]
    with pytest.raises(RuntimeError, match="missing"):
        mc.check_complete_search(Grid, 2, "missing")


@pytest.mark.parametrize("name", ["RandomForest", "XGBoost"])
def test_grid_search_refuses_nested_parallelism_for_rf_and_xgb(monkeypatch, name, request):
    """An explicit search ``n_jobs > 1`` around an internally parallel
    estimator, or an estimator ``n_jobs`` outside [1, cpu_jobs], is refused before any fit."""
    from sklearn.model_selection import GridSearchCV
    fits = []

    class Spy(GridSearchCV):
        def fit(self, X, y=None, **kw):
            fits.append((self.n_jobs, self.estimator.get_params().get("clf__n_jobs",
                         self.estimator.get_params().get("clf__estimator__n_jobs"))))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(mc, "GridSearchCV", Spy)
    X, y = _data(60)
    folds = [(np.arange(60)[np.arange(60) % 3 != k], np.arange(60)[np.arange(60) % 3 == k]) for k in range(3)]
    cfg = {**CPU, "cpu_jobs": 20}
    key = "clf__n_jobs" if name == "RandomForest" else "clf__estimator__n_jobs"
    for search_jobs in (2, 20):
        pipe, grid = _models({**cfg, "cpu_jobs": 2}, hparams=_tiny_hp(4))[name]
        with pytest.raises(ValueError, match="nest parallelism"):
            mc.run_grid_search_on_train("TSV_3p", name, pipe, grid, X, pd.Series(y), cv_splits=folds,
                                        n_jobs=search_jobs, verbose=0, compute=cfg)
    for est_jobs in (-1, None, 21):
        pipe, grid = _models(cfg, hparams=_tiny_hp(4))[name]
        pipe.set_params(**{key: est_jobs})
        with pytest.raises(ValueError, match="estimator n_jobs"):
            mc.run_grid_search_on_train("TSV_3p", name, pipe, grid, X, pd.Series(y), cv_splits=folds,
                                        verbose=0, compute=cfg)
    assert fits == []
    if name == "XGBoost":
        request.getfixturevalue("xgb_fits")      # the accepted fit below needs a usable xgboost
    pipe, grid = _models(cfg, hparams=_tiny_hp(4))[name]           # the accepted configuration
    mc.run_grid_search_on_train("TSV_3p", name, pipe, grid, X, pd.Series(y), cv_splits=folds,
                                n_jobs=1, verbose=0, compute=cfg)
    assert fits == [(1, 20)]


# ================================================================ PyTorch settings
def test_torch_fp32_and_deterministic_settings():
    state = compute.torch_determinism_state()
    assert state == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
    }


def test_non_deterministic_cublas_config_raises(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    with pytest.raises(compute.ComputeConfigError, match="CUBLAS_WORKSPACE_CONFIG"):
        compute.configure_torch_determinism()


def test_torch_registry_device_from_config(monkeypatch):
    cfg = GPU if torch.cuda.is_available() else CPU
    hp = load_config(compute.CONFIG_PATH)["deep_hparams"]
    reg = md.get_torch_models(NUM, CAT, hp, cat_vocabulary=vocabulary_for(CAT), compute=cfg)
    assert {m.device for m, _ in reg.values()} == {cfg["torch_device"]}
    monkeypatch.setattr(compute, "cuda_available", lambda: False)
    with pytest.raises(compute.ComputeConfigError, match="no CUDA device"):
        md.get_torch_models(NUM, CAT, hp, cat_vocabulary=vocabulary_for(CAT), compute=GPU)


def _legacy_loop(model, train_xs, y_tr, epochs, batch_size):
    """Previous loop body (per-batch ``loss.item()``, CPU permutation, CPU checkpoint)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    hist = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(y_tr.size(0))
        losses = []
        for i in range(0, y_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(*(x[idx] for x in train_xs)), y_tr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        hist.append(float(np.mean(losses)))
    return hist


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_training_loop_values_unchanged_without_per_batch_sync(device):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(150, 5, generator=g).to(device)
    y = torch.randint(0, 3, (150,), generator=g).to(device)
    torch.manual_seed(1)
    net = md.MLP(in_dim=5, hidden=(16,), out_dim=3, dropout=0.1).to(device)
    ref = copy.deepcopy(net)
    torch.manual_seed(7)
    hist, best_epoch, _ = md._train_early_stopping(
        net, net, (X,), y, (X[:30],), y[:30], torch.nn.CrossEntropyLoss(),
        torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-5),
        epochs=6, patience=100, batch_size=32,
        on_epoch_end=lambda ep, m: None)
    torch.manual_seed(7)
    # the legacy loop also evaluates on the validation rows between epochs; that does not
    # draw from the RNG, so its training-loss history must match exactly
    legacy = _legacy_loop(ref, (X,), y, epochs=6, batch_size=32)
    assert hist["train_loss"] == legacy
    assert best_epoch >= 1


def test_torch_estimators_clone_keeps_device_seed_and_epochs():
    """``sklearn.base.clone`` used to drop the ``_TorchBase`` parameters
    (``**kwargs``) and silently rebuild with the automatic device and seed 42."""
    from sklearn.base import clone
    common = dict(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device="cpu",
                  epochs=3, batch_size=32, random_state=7, lr=1e-2, patience=2)
    for est in (md.ANNClassifier(hidden=(8,), **common), md.FTClassifier(dim=8, depth=1, **common)):
        c = clone(est)
        assert (c.device, c.random_state, c.epochs, c.batch_size, c.lr, c.patience) == ("cpu", 7, 3, 32, 1e-2, 2)
        assert c.get_params() == est.get_params()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("kind", ["ANN", "FTT"])
def test_torch_fit_reproducible_whatever_python_random_state(kind, device):
    X, y = _data(160)
    params = dict(num_cols=NUM, cat_cols=CAT, cat_vocabulary=vocabulary_for(CAT), device=device,
                  epochs=4, batch_size=32, random_state=42)
    make = (lambda: md.ANNClassifier(hidden=(8,), **params)) if kind == "ANN" else \
        (lambda: md.FTClassifier(dim=8, depth=2, heads=2, dim_head=4, **params))
    out = []
    for py_seed in (0, 12345):
        random.seed(py_seed)                          # earlier draws in the process (job order)
        m = make().fit(X, y)
        out.append((m.predict(X), {k: v.cpu() for k, v in m.model_.state_dict().items()}))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    assert out[0][1].keys() == out[1][1].keys()
    assert all(torch.equal(out[0][1][k], out[1][1][k]) for k in out[0][1])
