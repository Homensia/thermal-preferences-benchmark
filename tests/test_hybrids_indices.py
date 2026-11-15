# -*- coding: utf-8 -*-
"""
Hybrids (E10) and SET / PET / ePMV indices (E7b), restored from the
pre-revision code with minimal fixes.

E10: CPU loading from the validated bytes (one read, tampered bytes refused
before parsing), hook = ``to_logits[2]`` input, bitwise-repeatable and
thread-invariant CPU extraction, frozen network, no test row reaching the FTT
or the heads, paper head hyperparameters for a real run, label handling of
the RF / XGB wrappers, dry run end to end (predictions, embedding records,
manifest, run index), CLI wiring.

E7b: inputs from the real columns (no default, no clip, no ``tsv``), a NaN
input never sent to the library, no perturbed-input retry, PET
non-convergence invalidates the row (two consecutive rows caught), a
compliance warning does not, calibrations on ``train_idx`` ∩ mask only, no
pre-revision PTS_PET constants, unbounded ePMV factor flagged outside
[0.5, 1.0], the PTS adapter (NaN never a class, saturation counted) also
used for ePMV, TPV refused, ``pmv_e`` first pass = E7 PMV, dry run end to
end, CLI wiring.

Synthetic data only; CPU only (the GPU hidden). A multi-class XGBoost fit
needs the GPU visible with the CUDA build of xgboost (``conftest.xgb_fits``):
the E10 dry run replaces the XGBoost class by a CPU stub with the same
interface, so that the runner path is exercised with the GPU hidden.
"""
from __future__ import annotations

import copy
import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

REPO_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_DIR, REPO_DIR / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hybrid  # noqa: E402
import indicateurs_classiques as ic  # noqa: E402
import revision_folds  # noqa: E402
import revision_runners as rr  # noqa: E402
import revision_stats as rs  # noqa: E402
from config_loader import load_config  # noqa: E402
from phase3_runner import ontology_probabilities  # noqa: E402

E1_TARGETS = ("thermal_sensation", "TSV_3p")


# ================================================================ fixtures
@pytest.fixture(scope="module")
def compute():
    from compute import load_compute
    return {**load_compute(), "cpu_jobs": 2, "gpu_concurrency": 1, "xgb_device": "cpu", "torch_device": "cpu"}


@pytest.fixture(scope="module")
def config():
    cfg = copy.deepcopy(load_config(rr.CONFIG_PATH))
    for section, values in rr.DRY_RUN_HPARAMS.items():
        cfg[section] = copy.deepcopy(values)
    return cfg


@pytest.fixture(scope="module")
def synth(tmp_path_factory, config):
    import test_revision_runners as trr
    base = tmp_path_factory.mktemp("synth_e10")
    df, ids = trr._synthetic_cohort()
    csv = base / "cohort_api.csv"
    df.to_csv(csv, index=False)
    split_dir = base / "split"
    split_dir.mkdir()
    for t in rs.TARGETS:
        tr, te = revision_folds.redraw_split(df[t])
        (split_dir / f"{t}_split_indices.json").write_text(
            json.dumps({"target": t, "train_idx": tr.tolist(), "test_idx": te.tolist()}))
    ctx = rr.load_ashrae_context(config, csv_path=csv, ids=ids)
    return {"ctx": ctx, "csv": csv, "split_dir": split_dir, "df": df, "ids": ids}


@pytest.fixture(scope="module")
def e1_ftt(tmp_path_factory, synth, config, compute):
    """E1 dry run, RF (42 43 44) and FTT (42), TSV-7 and TSV-3, CPU only."""
    run = tmp_path_factory.mktemp("run_e1_ftt")
    man = rr.run_e1(synth["ctx"], run_dir=run, config=config, split_dir=synth["split_dir"], targets=E1_TARGETS,
                    families=("RF", "FTT"), compute=compute, gpu_concurrency=1, dry_run=True,
                    command=["pytest", "e1_ftt"])
    return run, man


def _split(synth, t):
    return revision_folds.load_archived_split(synth["split_dir"], t, synth["df"][t])


def _e1_model(e1_ftt, t="TSV_3p", s=42):
    run, _ = e1_ftt
    return rr.e1_ftt_model_file(run / "E1_phase2_dry_run", t, s)


class _StubXGB:
    """CPU stand-in for ``xgboost.XGBClassifier`` (same constructor keywords; fitted on codes 0..k-1)."""

    def __init__(self, **kw):
        self.kw = kw
        self._m = DecisionTreeClassifier(max_depth=kw.get("max_depth"), random_state=kw.get("random_state"))

    def fit(self, X, y):
        y = np.asarray(y)
        assert np.array_equal(np.unique(y), np.arange(np.unique(y).size)), "XGB head must get label codes"
        self._m.fit(X, y)
        self.classes_ = self._m.classes_
        return self

    def predict(self, X):
        return self._m.predict(X)

    def predict_proba(self, X):
        return self._m.predict_proba(X)


# ================================================================ E10: loading, hook, determinism
def test_load_ftt_cpu_takes_the_bytes_maps_to_cpu_and_restores_torch(e1_ftt):
    f = _e1_model(e1_ftt)
    raw = f.read_bytes()
    original = torch.storage._load_from_bytes
    ft = hybrid.load_ftt_cpu(raw)
    assert torch.storage._load_from_bytes is original                     # private hook restored
    assert ft.device == "cpu" and all(p.device.type == "cpu" for p in ft.model_.parameters())
    assert torch.get_num_threads() == hybrid.EMBEDDING_TORCH_THREADS
    with pytest.raises(TypeError, match="bytes"):
        hybrid.load_ftt_cpu(f)                                             # a path is never opened


def test_cpu_embeddings_are_repeatable_and_the_hook_is_the_output_head(e1_ftt, synth):
    ft = hybrid.load_ftt_cpu(_e1_model(e1_ftt).read_bytes())
    tr, te = _split(synth, "TSV_3p")
    X_test = synth["ctx"].data[rr.features_for(load_config(rr.CONFIG_PATH), "F17")].iloc[te]
    sd0 = rr._state_dict_sha256(ft.model_)
    Za = hybrid.ft_get_embeddings(ft, X_test)
    Zb = hybrid.ft_get_embeddings(ft, X_test)
    # bitwise repeatable at the recorded thread count (E10 pins it). Thread invariance (8 vs 1),
    # measured on the nine real E1 FTT, does not hold for this tiny dry-run
    # network (last-bit differences): it is not relied upon, the thread count is fixed and recorded.
    assert torch.get_num_threads() == hybrid.EMBEDDING_TORCH_THREADS
    assert np.array_equal(Za, Zb)
    assert Za.shape == (len(te), ft.model_.to_logits[2].in_features)
    rec = rr.check_hook(ft, X_test, hybrid.array_sha256(Za))              # to_logits[2](Z) = logits
    assert rec["max_abs_logit_from_hook"] <= rr.E10_LOGIT_ATOL
    assert rr._state_dict_sha256(ft.model_) == sd0                        # frozen network


def test_extraction_refuses_a_layer_other_than_to_logits_2(e1_ftt, synth, monkeypatch):
    ft = hybrid.load_ftt_cpu(_e1_model(e1_ftt).read_bytes())
    tr, te = _split(synth, "TSV_3p")
    X_test = synth["ctx"].data[rr.features_for(load_config(rr.CONFIG_PATH), "F17")].iloc[te]
    monkeypatch.setattr(hybrid, "_find_last_linear_for_out_dim", lambda m, k: torch.nn.Linear(4, k))
    with pytest.raises(RuntimeError, match="to_logits"):
        hybrid.ft_get_embeddings(ft, X_test)


def test_hybrid_module_keeps_only_code_used_by_e10():
    """No dead import, no UMAP view, no head CV, no fit on the wrapper."""
    import ast
    tree = ast.parse((REPO_DIR / "hybrid.py").read_text(encoding="utf-8"))
    imported = {(a.asname or a.name).split(".")[0] for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert imported - used == set()
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not names & {"visualize_ft_embeddings", "_scatter_2d", "fit"}
    assert not {"umap", "GridSearchCV", "StratifiedKFold"} & imported


# ================================================================ E10: rows (leakage) and hyperparameters
def test_no_test_row_reaches_the_ftt_or_the_heads(e1_ftt, synth):
    ft = hybrid.load_ftt_cpu(_e1_model(e1_ftt).read_bytes())
    tr, te = _split(synth, "TSV_3p")
    X = synth["ctx"].data
    rec = rr.check_hybrid_rows(ft, tr, te, X.iloc[tr], X.iloc[te])
    assert rec["n_ftt_fit_rows"] == len(tr) and rec["n_head_train"] == len(tr)
    with pytest.raises(rr.RunnerError, match="head training rows"):       # a test row in the head fit set
        rr.check_hybrid_rows(ft, tr, te, X.iloc[np.r_[tr[:-1], te[:1]]], X.iloc[te])
    with pytest.raises(rr.RunnerError, match="overlap"):
        rr.check_hybrid_rows(ft, np.r_[tr, te[:1]], te, X.iloc[np.r_[tr, te[:1]]], X.iloc[te])
    ft.es_val_idx_ = np.r_[np.asarray(ft.es_val_idx_), te[:1]]          # an FTT that saw a test row
    with pytest.raises(rr.RunnerError, match="FTT was not fitted"):
        rr.check_hybrid_rows(ft, tr, te, X.iloc[tr], X.iloc[te])


def test_real_e10_uses_the_paper_head_hyperparameters():
    file_cfg = load_config(rr.CONFIG_PATH)
    assert "hybrid" not in file_cfg                                       # no hybrid section in config.yaml
    assert rr.check_paper_head_hparams(file_cfg.get("hybrid"), dry_run=False) == rr.PAPER_HYBRID_HEADS
    assert rr.check_paper_head_hparams({"heads": rr.PAPER_HYBRID_HEADS}, dry_run=False) == rr.PAPER_HYBRID_HEADS
    bad = {"heads": copy.deepcopy(rr.PAPER_HYBRID_HEADS)}
    bad["heads"]["RF"]["n_estimators"] = 600                               # the pre-revision value
    with pytest.raises(rr.RunnerError, match="paper table"):
        rr.check_paper_head_hparams(bad, dry_run=False)
    assert rr.check_paper_head_hparams(rr.DRY_RUN_HPARAMS["hybrid"], dry_run=True)["RF"]["n_estimators"] == 10


# ================================================================ E10: wrapper label handling
def _tsv7_embeddings(n=140, seed=3, classes=(-3, -2, -1, 0, 1, 2)):
    rng = np.random.default_rng(seed)
    y = np.resize(np.asarray(classes, dtype=float), n)                    # class +3 absent from training
    Z = np.column_stack([y + rng.normal(0, 0.05, n), rng.normal(0, 1, n)])
    return Z, y


def test_rf_wrapper_returns_raw_labels_and_never_decodes(monkeypatch):
    Z, y = _tsv7_embeddings()
    head = RandomForestClassifier(n_estimators=5, random_state=0).fit(Z, y)
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: np.asarray(X, dtype=float))
    w = hybrid.FTEmbeddingHead.from_fitted(None, head)
    monkeypatch.setattr(LabelEncoder, "inverse_transform", lambda *a: (_ for _ in ()).throw(AssertionError))
    X = pd.DataFrame(Z[:20])
    assert np.array_equal(w.predict(X), head.predict(Z[:20])) and w.le_y_ is None
    assert np.array_equal(w.classes_, head.classes_)
    p = ontology_probabilities(w, X, "thermal_sensation", 20)
    assert np.array_equal(p[:, :6], head.predict_proba(Z[:20])) and np.all(p[:, 6] == 0)   # p_3 = 0 (absent)
    assert w.last_Z_sha256_ == hybrid.array_sha256(Z[:20])


def test_xgb_wrapper_decodes_codes_with_the_runner_encoder(monkeypatch):
    Z, y = _tsv7_embeddings()
    le = LabelEncoder().fit(y)
    head = _StubXGB(max_depth=4, random_state=0).fit(Z, le.transform(y))
    monkeypatch.setattr(hybrid, "ft_get_embeddings", lambda clf, X: np.asarray(X, dtype=float))
    w = hybrid.FTEmbeddingHead.from_fitted(None, head, le)
    X = pd.DataFrame(Z[:20])
    assert np.array_equal(w.predict(X), le.inverse_transform(head.predict(Z[:20])))
    assert np.array_equal(w.classes_, le.classes_) and set(w.predict(X)) <= set(y)
    p = ontology_probabilities(w, X, "thermal_sensation", 20)
    assert np.array_equal(p[:, :6], head.predict_proba(Z[:20])) and np.all(p[:, 6] == 0)
    with pytest.raises(ValueError, match="codes"):
        hybrid.FTEmbeddingHead.from_fitted(None, RandomForestClassifier(n_estimators=2).fit(Z, y), le)


# ================================================================ E10: dry run end to end
def _spy_reads(monkeypatch, paths):
    import config_loader
    seen = {str(p): 0 for p in paths}
    real = config_loader.read_consumed

    def spy(path, consumed=None, role=None):
        if str(Path(path)) in seen:
            seen[str(Path(path))] += 1
        return real(path, consumed, role)
    monkeypatch.setattr(rr, "read_consumed", spy)
    return seen


def test_e10_dry_run_writes_predictions_records_and_manifest(tmp_path, synth, e1_ftt, config, compute, monkeypatch):
    # the CUDA state of torch is process-wide: an earlier test of the suite may have initialised it, so this
    # in-process run only checks that it is recorded; "E10 never initialises CUDA" is checked in a fresh
    # process (test_e10_fresh_process_never_initialises_cuda)
    run, e1_man = e1_ftt
    e1_dir = run / "E1_phase2_dry_run"
    monkeypatch.setattr(hybrid, "XGBClassifier", _StubXGB)
    import models_deep
    monkeypatch.setattr(models_deep.FTClassifier, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    models = [rr.e1_ftt_model_file(e1_dir, t, 42) for t in E1_TARGETS]
    seen = _spy_reads(monkeypatch, models)
    loads = []
    real_load = joblib.load
    monkeypatch.setattr(hybrid.joblib, "load", lambda f, *a, **k: loads.append(type(f)) or real_load(f, *a, **k))
    man = rr.run_e10(synth["ctx"], run_dir=tmp_path, e1_dir=e1_dir, config=config, split_dir=synth["split_dir"],
                     targets=E1_TARGETS, compute=compute, dry_run=True, command=["pytest", "e10"])
    assert seen == {str(m): 1 for m in models}                            # one read per model file
    assert loads == [io.BytesIO] * len(models)                            # parsed from those bytes
    exp = tmp_path / "E10_hybrids_dry_run"
    assert man["experiment"] == "E10_dry_run" and man["seeds"]["ftt"] == [42]
    assert man["e1_manifest_sha256"] == rr.file_sha256(e1_dir / "run_manifest.json")
    for t in E1_TARGETS:
        tr, te = _split(synth, t)
        rec = json.loads((exp / t / "seed_42" / f"{t}__FTT__seed42_embedding_record.json").read_text())
        assert rec["device"] == "cpu" and rec["torch_threads"] == hybrid.EMBEDDING_TORCH_THREADS
        assert isinstance(rec["torch_cuda_initialized"], bool) and "xgb_head_device" in rec
        assert rec["source_sha256"] == e1_man["outputs_sha256"][f"{t}/FTT/seed_42/{t}__FTT__seed42.joblib"]
        assert rec["rows"]["n_head_train"] == len(tr) and rec["e1_identity"]["y_pred_differences"] == 0
        assert rec["Z_train_shape"][0] == len(tr) and rec["Z_test_shape"][0] == len(te)
        for model, sub, tag in (("FTT_RF", "RF", "RF"), ("FTT_XGB", "XGBoost", "XGB")):
            fr = pd.read_csv(exp / "predictions" / f"E10_dry_run__ASHRAE_2022__{t}__{model}__seed42__archived_80_20.csv.gz")
            assert fr["row_idx"].tolist() == te.tolist() and set(fr["feature_set"]) == {"FTTemb_F17"}
            labels = rs.ontology(t)
            assert np.allclose(fr[[f"p_{c}" for c in labels]].to_numpy().sum(axis=1), 1.0)
            res = json.loads((exp / t / "seed_42" / sub / f"{t}_FTemb_{tag}_test_results.json").read_text())
            assert res["y_pred"] == fr["y_pred"].tolist()
            assert not (exp / t / "seed_42" / sub / f"{t}_FTemb_{tag}_cv_best.json").exists()
    met = pd.read_csv(exp / "metrics.csv")
    assert sorted(set(met["model"])) == ["FTT_RF", "FTT_XGB"] and len(met) == 2 * len(E1_TARGETS)
    index = json.loads((tmp_path / "manifest.json").read_text())
    assert index["experiments"]["E10_dry_run"]["status"] == "complete"
    with pytest.raises(FileExistsError):                                  # one run per output directory
        rr.run_e10(synth["ctx"], run_dir=tmp_path, e1_dir=e1_dir, config=config, split_dir=synth["split_dir"],
                   targets=E1_TARGETS, compute=compute, dry_run=True)


def test_e10_refuses_a_tampered_model_before_parsing(tmp_path, synth, e1_ftt, config, compute, monkeypatch):
    import shutil
    run, _ = e1_ftt
    e1_dir = tmp_path / "copy" / "E1_phase2_dry_run"
    shutil.copytree(run / "E1_phase2_dry_run", e1_dir)
    shutil.copy(run / "manifest.json", tmp_path / "copy" / "manifest.json")
    f = rr.e1_ftt_model_file(e1_dir, "TSV_3p", 42)
    f.write_bytes(f.read_bytes() + b"\0")
    monkeypatch.setattr(hybrid, "load_ftt_cpu", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed")))
    with pytest.raises(rr.RunnerError, match="differs from the E1 manifest"):
        rr.run_e10(synth["ctx"], run_dir=tmp_path / "out", e1_dir=e1_dir, config=config,
                   split_dir=synth["split_dir"], targets=["TSV_3p"], compute=compute, dry_run=True)
    assert not (tmp_path / "out" / "E10_hybrids_dry_run" / "run_manifest.json").exists()


def _e1_frame_of(ft, X_test, t):
    labels = rs.ontology(t)
    fr = pd.DataFrame(ontology_probabilities(ft, X_test, t, len(X_test)), columns=[f"p_{c}" for c in labels])
    fr["y_pred"] = np.asarray(ft.predict(X_test), dtype=float)
    return fr


def test_ftt_identity_check_stops_e10_on_a_prediction_or_probability_difference(e1_ftt, synth):
    """Stop path: one y_pred or one p_<class> beyond 1e-6 differs -> E10 stops."""
    t = "TSV_3p"
    ft = hybrid.load_ftt_cpu(_e1_model(e1_ftt).read_bytes())
    _, te = _split(synth, t)
    X_test = synth["ctx"].data[rr.features_for(load_config(rr.CONFIG_PATH), "F17")].iloc[te]
    ref = _e1_frame_of(ft, X_test, t)
    assert rr.check_ftt_identity(ft, X_test, t, ref)["y_pred_differences"] == 0
    bad_p = ref.copy()
    bad_p.loc[0, f"p_{rs.ontology(t)[0]}"] += 2 * rr.E10_PROBA_ATOL
    with pytest.raises(rr.RunnerError, match="E10 stops"):
        rr.check_ftt_identity(ft, X_test, t, bad_p)
    bad_y = ref.copy()
    labels = rs.ontology(t)
    bad_y.loc[0, "y_pred"] = float(next(c for c in labels if c != bad_y.loc[0, "y_pred"]))
    with pytest.raises(rr.RunnerError, match="E10 stops"):
        rr.check_ftt_identity(ft, X_test, t, bad_y)


def _e10_tamper_after_last_head(monkeypatch, action):
    """Run ``action(ft)`` after the metrics of the last head (XGB) of the single FTT of the run."""
    seen = {"ft": None, "n": 0}
    real_load, real_metric_row = hybrid.load_ftt_cpu, rr.metric_row

    def load(*a, **k):
        seen["ft"] = real_load(*a, **k)
        return seen["ft"]

    def metric_row(m):
        seen["n"] += 1
        if seen["n"] == 2:
            action(seen["ft"])
        return real_metric_row(m)
    monkeypatch.setattr(hybrid, "load_ftt_cpu", load)
    monkeypatch.setattr(rr, "metric_row", metric_row)


@pytest.mark.parametrize("what", ["state_dict", "threads"])
def test_e10_stops_if_the_network_or_the_thread_count_changes(tmp_path, synth, e1_ftt, config, compute,
                                                              monkeypatch, what):
    run, _ = e1_ftt
    monkeypatch.setattr(hybrid, "XGBClassifier", _StubXGB)
    threads0 = torch.get_num_threads()
    if what == "state_dict":
        _e10_tamper_after_last_head(monkeypatch, lambda ft: ft.model_.register_buffer("tamper", torch.zeros(1)))
        match = "frozen network"
    else:
        _e10_tamper_after_last_head(monkeypatch, lambda ft: torch.set_num_threads(1))
        match = "torch threads"
    try:
        with pytest.raises(rr.RunnerError, match=match):
            rr.run_e10(synth["ctx"], run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", config=config,
                       split_dir=synth["split_dir"], targets=["TSV_3p"], compute=compute, dry_run=True)
    finally:
        torch.set_num_threads(threads0)
    assert not (tmp_path / "E10_hybrids_dry_run" / "run_manifest.json").exists()


def _fresh_python(code: str, timeout: int = 900) -> dict:
    """Run ``code`` in a fresh interpreter (its own torch CUDA state, independent of the tests run before in
    this process) with the GPU visible (``CUDA_VISIBLE_DEVICES`` removed); ``code`` prints one JSON line
    last. Returns that JSON."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR), str(REPO_DIR / "tests")])
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_DIR, env=env, capture_output=True, text=True,
                         timeout=timeout)
    assert res.returncode == 0, res.stderr[-4000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_e10_cpu_only_guard_in_a_fresh_process():
    """The real-run start guard, in a fresh process with the GPU visible: torch has not initialised CUDA
    (importing the runner does not), so the guard passes (returns False) instead of raising."""
    out = _fresh_python(
        "import json, torch, revision_runners as rr\n"
        "before = bool(torch.cuda.is_initialized())\n"
        "guard = rr.check_e10_cpu_only('start', dry_run=False)\n"
        "print(json.dumps({'cuda_available': bool(torch.cuda.is_available()), 'before': before, 'guard': guard,"
        " 'after': bool(torch.cuda.is_initialized())}))\n", timeout=300)
    assert out["before"] is False and out["guard"] is False and out["after"] is False


def test_e10_fresh_process_never_initialises_cuda(tmp_path, synth, e1_ftt):
    """A whole E10 dry run (the code path of a real E10) in a fresh process with the GPU visible: torch CUDA
    is never initialised (every embedding record False, the real-run guard passes after the run)."""
    run, _ = e1_ftt
    code = f"""
import copy, json, torch
import revision_runners as rr, hybrid
import test_revision_runners as trr
import test_hybrids_indices as thi
from compute import load_compute
from config_loader import load_config
cfg = copy.deepcopy(load_config(rr.CONFIG_PATH))
for section, values in rr.DRY_RUN_HPARAMS.items():
    cfg[section] = copy.deepcopy(values)
compute = {{**load_compute(), "cpu_jobs": 2, "gpu_concurrency": 1, "xgb_device": "cpu", "torch_device": "cpu"}}
df, ids = trr._synthetic_cohort()
ctx = rr.load_ashrae_context(cfg, csv_path={str(synth["csv"])!r}, ids=ids)
hybrid.XGBClassifier = thi._StubXGB
before = bool(torch.cuda.is_initialized())
man = rr.run_e10(ctx, run_dir={str(tmp_path / "fresh")!r}, e1_dir={str(run / "E1_phase2_dry_run")!r}, config=cfg,
                 split_dir={str(synth["split_dir"])!r}, targets={list(E1_TARGETS)!r}, compute=compute,
                 dry_run=True, command=["pytest", "e10-fresh"])
print(json.dumps({{"cuda_available": bool(torch.cuda.is_available()), "before": before,
                   "manifest": man["torch_cuda_initialized"],
                   "records": [r["torch_cuda_initialized"] for r in man["embedding_records"].values()],
                   "guard_after": rr.check_e10_cpu_only("after", dry_run=False),
                   "after": bool(torch.cuda.is_initialized())}}))
"""
    out = _fresh_python(code)
    assert out["before"] is False and out["after"] is False and out["guard_after"] is False
    assert out["manifest"] == {"start": False, "end": False}
    assert out["records"] and not any(out["records"])


def test_e10_cpu_only_guards(monkeypatch):
    """Nothing of a real E10 runs on the GPU: torch CUDA initialised or an XGB head not on the CPU
    stops it. The CUDA state is forced here (it is process-wide; the real state is checked in a fresh
    process by the two tests above)."""
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert rr.check_e10_cpu_only("start", dry_run=False) is False
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(rr.RunnerError, match="CPU only"):
        rr.check_e10_cpu_only("end", dry_run=False)
    assert rr.check_e10_cpu_only("end", dry_run=True) is True             # a dry run records it

    class _Booster:
        def __init__(self, device):
            self.device = device

        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": self.device}}})

    class _Head:
        def __init__(self, params_device, booster_device):
            self.p, self.b = params_device, booster_device

        def get_params(self):
            return {"device": self.p}

        def get_booster(self):
            return _Booster(self.b)
    assert rr.check_e10_xgb_head_device(_Head("cpu", "cpu"), dry_run=False) == {"params_device": "cpu",
                                                                                 "booster_device": "cpu"}
    for head in (_Head("cuda", "cuda:0"), _Head("cpu", "cuda:0"), _Head("cuda", "cpu")):
        with pytest.raises(rr.RunnerError, match="CPU required"):
            rr.check_e10_xgb_head_device(head, dry_run=False)
    with pytest.raises(rr.RunnerError, match="CPU required"):            # a real head must expose its device
        rr.check_e10_xgb_head_device(_StubXGB(max_depth=2), dry_run=False)
    assert rr.check_e10_xgb_head_device(_StubXGB(max_depth=2), dry_run=True) == {"params_device": None,
                                                                                 "booster_device": None}


def test_real_e10_refuses_a_dry_run_e1(tmp_path, synth, e1_ftt, compute, monkeypatch):
    monkeypatch.setattr(rr, "run_provenance", lambda: {"git_commit": "c" * 40, "git_tracked_changes": False})
    monkeypatch.setattr(rr, "check_locked_environment", lambda env, dry_run: None)
    monkeypatch.setattr(rr, "check_phase1_artefact", lambda dry_run: rr.phase1_provenance())
    monkeypatch.setattr(rr, "untracked_python_files", lambda: [])
    run, _ = e1_ftt
    with pytest.raises(rr.RunnerError, match="dry run"):
        rr.run_e10(synth["ctx"], run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run",
                   config=load_config(rr.CONFIG_PATH), split_dir=synth["split_dir"], compute=compute,
                   dry_run=False)
    with pytest.raises(rr.RunnerError, match="every target"):
        rr.run_e10(synth["ctx"], run_dir=tmp_path / "b", e1_dir=run / "E1_phase2_dry_run",
                   config=load_config(rr.CONFIG_PATH), split_dir=synth["split_dir"], targets=["TSV_3p"],
                   compute=compute, dry_run=False)


# ================================================================ E7b: restored functions
def _inputs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ta = rng.uniform(19, 29, n)
    return pd.DataFrame({"Tair": ta, "tr": ta + rng.normal(0, 1, n), "vel": rng.uniform(0.05, 0.3, n),
                         "RH": rng.uniform(30, 70, n), "Met": rng.uniform(1.0, 1.3, n), "clo": rng.uniform(0.4, 0.9, n)})


def test_index_inputs_have_no_default_and_a_nan_row_is_never_sent(monkeypatch):
    df = _inputs(20)
    with pytest.raises(KeyError, match="no default"):
        ic.compute_set_pet(df.drop(columns=["clo"]))
    df.loc[3, "vel"] = np.nan
    df.loc[7, "RH"] = np.nan
    sent = {"set": [], "pet": [], "pmv_e": []}
    real_set, real_pet, real_pmv_e = ic.set_tmp, ic.pet_steady, ic.pmv_e
    monkeypatch.setattr(ic, "set_tmp", lambda **k: sent["set"].append(k) or real_set(**k))
    monkeypatch.setattr(ic, "pet_steady", lambda **k: sent["pet"].append(k) or real_pet(**k))
    monkeypatch.setattr(ic, "pmv_e", lambda **k: sent["pmv_e"].append(k) or real_pmv_e(**k))
    s, p, logs = ic.compute_set_pet(df)
    e, _ = ic.compute_pmv_e(df, 1.0)
    assert np.isnan(s[[3, 7]]).all() and np.isnan(p[[3, 7]]).all() and np.isnan(e[[3, 7]]).all()
    assert len(sent["set"]) == len(sent["pet"]) == 18 and ic.summarise_log(logs["SET"])["n_nonfinite_inputs"] == 2
    for k in sent["set"] + sent["pet"]:
        assert all(np.isfinite(float(v)) for kk, v in k.items() if kk in ("tdb", "tr", "v", "rh", "met", "clo"))
    assert all(np.isfinite(np.asarray(v, dtype=float)).all() for k in sent["pmv_e"]
               for kk, v in k.items() if kk in ("tdb", "tr", "vr", "rh", "met", "clo"))
    # observed inputs, never a perturbed air speed: vel passed as is
    assert [k["v"] for k in sent["set"]] == [v for i, v in enumerate(df["vel"]) if i not in (3, 7)]


def test_failing_calls_are_invalid_without_retry(monkeypatch):
    df = _inputs(4)
    calls = []

    def boom(**k):
        calls.append(k["v"])
        raise ValueError("library failure")
    monkeypatch.setattr(ic, "set_tmp", boom)
    monkeypatch.setattr(ic, "pet_steady", boom)
    s, p, logs = ic.compute_set_pet(df)
    assert np.isnan(s).all() and np.isnan(p).all() and len(calls) == 8      # one call per row and index
    assert ic.summarise_log(logs["SET"])["exceptions"] == {"ValueError": 4}
    assert ic.summarise_log(logs["PET"])["exceptions"] == {"ValueError": 4}


def test_a_missing_result_field_is_counted_as_an_exception_not_a_silent_nan(monkeypatch):
    from types import SimpleNamespace
    df = _inputs(3)
    monkeypatch.setattr(ic, "set_tmp", lambda **k: SimpleNamespace(other=1.0))       # no "set" field
    monkeypatch.setattr(ic, "pet_steady", lambda **k: {"pet": "n/a"})               # unparsable "pet"
    s, p, logs = ic.compute_set_pet(df)
    assert np.isnan(s).all() and np.isnan(p).all()
    assert ic.summarise_log(logs["SET"])["exceptions"] == {"ResultFieldError": 3}
    assert ic.summarise_log(logs["PET"])["exceptions"] == {"ResultFieldError": 3}
    with pytest.raises(ic.ResultFieldError):
        ic._as_float_field(None, "set")


def test_diagnostics_are_row_level_and_aggregated_per_scope(monkeypatch):
    """Failures confined to training rows never show in the test-row columns, and conversely."""
    df = _inputs(10)
    tr, te = np.arange(0, 7), np.arange(7, 10)
    fail = {}
    real_pet = ic.pet_steady

    def flaky(**k):
        i = int(np.flatnonzero(np.isclose(df["vel"].to_numpy(), k["v"]))[0])
        if i in fail["exc"]:
            raise ValueError("library failure")
        if i in fail["nonconv"]:
            warnings.warn("The iteration is not making good progress, as measured by the", RuntimeWarning)
        return real_pet(**k)
    monkeypatch.setattr(ic, "pet_steady", flaky)
    for where, rows in (("train", tr), ("test", te)):
        fail.update(exc={int(rows[0])}, nonconv={int(rows[1])})
        _, p, logs = ic.compute_set_pet(df)
        assert np.isnan(p[[rows[0], rows[1]]]).all()
        cohort = ic._log_counts("cohort_", logs["PET"])
        test = ic._log_counts("test_", logs["PET"], te)
        assert cohort["cohort_exceptions"] == json.dumps({"ValueError": 1}) and cohort["cohort_n_nonconvergence"] == 1
        if where == "train":
            assert test["test_exceptions"] == "{}" and test["test_n_nonconvergence"] == 0
        else:
            assert test["test_exceptions"] == json.dumps({"ValueError": 1}) and test["test_n_nonconvergence"] == 1


def test_pet_nonconvergence_invalidates_each_row_and_compliance_warnings_do_not(monkeypatch):
    df = _inputs(4)
    from types import SimpleNamespace

    def fake_pet(**k):
        i = int(round(k["v"] * 1e6))
        if i in nonconv:
            warnings.warn("The iteration is not making good progress, as measured by the \n improvement",
                          RuntimeWarning, stacklevel=2)
        else:
            warnings.warn("ASHRAE-55 compliance: value out of range", UserWarning, stacklevel=2)
        return SimpleNamespace(pet=21.0)
    nonconv = {int(round(df.loc[i, "vel"] * 1e6)) for i in (1, 2)}         # two consecutive rows
    monkeypatch.setattr(ic, "pet_steady", fake_pet)
    _, p, logs = ic.compute_set_pet(df)
    assert np.isnan(p[[1, 2]]).all() and np.isfinite(p[[0, 3]]).all()
    sm = ic.summarise_log(logs["PET"])
    assert sm["nonconvergence"] == 2 and sm["warnings"] == {"UserWarning": 2}
    assert logs["PET"]["nonconvergence"].tolist() == [False, True, True, False]       # flagged per row


def test_calibrations_use_train_rows_inside_the_mask_only():
    n = 400
    rng = np.random.default_rng(1)
    x = rng.uniform(20, 30, n)
    x[:10] = np.nan
    y = np.clip(np.floor(-6 + 0.25 * x + rng.normal(0, 0.5, n) + 0.5), -3, 3)
    y[:10] = 0
    tr, te = np.arange(0, 300), np.arange(300, n)
    cal = ic.fit_pts_calibration(x, y, tr, te)
    assert cal["status"] == "ok" and cal["n_calibration"] == 290
    m = np.isfinite(x[tr])
    beta = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x[tr][m]]), y[tr][m], rcond=None)[0]
    assert cal["a"] == pytest.approx(beta[0]) and cal["b"] == pytest.approx(beta[1])
    with pytest.raises(ic.CalibrationLeakageError):
        ic.fit_pts_calibration(x, y, tr, te, calibration_idx=np.r_[tr[10:], te[:1]])
    with pytest.raises(ic.CalibrationLeakageError):
        ic.fit_epmv_e(x - 25, y, tr, te, calibration_idx=np.r_[tr[10:], te[:1]])
    assert ic.fit_pts_calibration(x, np.zeros(n), tr, te)["status"] == ic.NOT_AVAILABLE    # constant labels
    assert ic.fit_pts_calibration(x, y, tr[:105], te)["status"] == ic.NOT_AVAILABLE       # 95 rows < 100
    # no pre-revision PTS constants, no constants-or-fit branch, no quadratic option
    assert not {n for n in dir(ic) if n.startswith("PTS_") and n[-2:] in ("_A", "_B", "_C")}
    assert not hasattr(ic, "PTS_USE_QUADRATIC") and set(cal) >= {"a", "b"} and "c" not in cal
    import ast
    consts = {n.value for n in ast.walk(ast.parse(Path(ic.__file__).read_text())) if isinstance(n, ast.Constant)}
    assert not {1.1704, -1.1704, 0.0494} & consts


@pytest.mark.parametrize("slope, flagged", [(0.3, True), (0.8, False), (1.6, True), (-0.4, True)])
def test_epmv_factor_is_the_unbounded_least_squares_fit_through_the_origin(slope, flagged):
    """e = Σxy / Σx² applied as is (never clipped), flagged outside [0.5, 1.0]."""
    n = 300
    rng = np.random.default_rng(2)
    x = np.round(rng.uniform(-1.9, 1.9, n), 2)
    y = np.clip(np.round(slope * x + rng.normal(0, 0.3, n)), -3, 3)
    tr, te = np.arange(0, 250), np.arange(250, n)
    cal = ic.fit_epmv_e(x, y, tr, te)
    e_ls = float(np.linalg.lstsq(x[tr, None], y[tr], rcond=None)[0][0])
    assert cal["status"] == "ok" and cal["e"] == pytest.approx(e_ls)          # unbounded: the LS value itself
    assert cal["e_flag_outside_range"] is flagged and cal["e_flag_range"] == [0.5, 1.0]
    assert cal["e_flag_outside_range"] is (not 0.5 <= cal["e"] <= 1.0)
    assert "bound_active" not in cal and "e_unbounded" not in cal
    e = cal["e"]
    assert cal["sse_calibration"] == pytest.approx(np.sum((y[tr] - e * x[tr]) ** 2))
    assert cal["sse_rounded_output"] == pytest.approx(np.sum((y[tr] - np.around(e * x[tr], 2)) ** 2))


def test_epmv_factor_is_never_clipped():
    assert ic.EPMV_E_FLAG_RANGE == (0.5, 1.0) and not hasattr(ic, "EPMV_E_BOUNDS")
    import inspect
    src = inspect.getsource(ic.fit_epmv_e)
    assert "clip" not in src.split('"""')[2]                                  # no clip in the code body


def test_sse_of_the_rounded_output_differs_from_the_calibration_sse():
    x = np.full(200, 0.15)
    y = np.full(200, 0.1)
    y[::2] = 0.03
    cal = ic.fit_epmv_e(x, y, np.arange(150), np.arange(150, 200))
    assert cal["e"] == pytest.approx(0.065 / 0.15) and cal["e_flag_outside_range"] is True   # e·x = 0.065
    assert cal["sse_calibration"] != pytest.approx(cal["sse_rounded_output"])


def test_pts_adapter_saturates_counts_and_never_makes_a_class_from_nan():
    v = np.array([np.nan, 3.49, 3.5, 7.0, -3.5, -3.51, -9.0, 0.4, np.nan])
    cls, low, high = ic.pts_to_class(v, "thermal_sensation")
    assert cls.isna().tolist() == [True] + [False] * 7 + [True]
    assert cls[1:8].tolist() == [3, 3, 3, -3, -3, -3, 0] and (low, high) == (2, 2)
    with pytest.raises(ValueError, match="never clipped"):                # E7's rule is unchanged
        ic.to_tsv7(v, (-3, 3))
    cls3, _, _ = ic.pts_to_class(v, "TSV_3p")
    assert cls3[1:8].tolist() == [1, 1, 1, -1, -1, -1, 0] and cls3.isna().sum() == 2
    with pytest.raises(ValueError, match="TSV-7 and TSV-3|thermal_preference"):
        ic.pts_to_class(v, "thermal_preference")


def test_epmv_follows_the_saturation_rule_of_the_other_restored_indices():
    """e > 1 can put ePMV beyond the scale; like PTS_SET / PTS_PET, finite values >= 3.5
    -> 3 and < -3.5 -> -3, counted; never an error; NaN never a class."""
    ok, low, high = ic.epmv_to_class(np.array([-2.0, -1.5, 0.49, 1.49, 2.0, np.nan]), "thermal_sensation")
    assert ok.tolist()[:5] == [-2, -1, 0, 1, 2] and ok.isna().tolist()[5] and (low, high) == (0, 0)
    v = np.array([2.6, 3.49, 3.5, 4.2, -3.5, -3.6, np.nan])                  # e > 1: beyond [-2, 2] and the scale
    cls, low, high = ic.epmv_to_class(v, "thermal_sensation")
    assert cls[:6].tolist() == [3, 3, 3, 3, -3, -3] and cls.isna().tolist() == [False] * 6 + [True]
    assert (low, high) == (1, 2)
    p_cls, p_low, p_high = ic.pts_to_class(v, "thermal_sensation")
    assert p_cls.tolist() == cls.tolist() and (p_low, p_high) == (low, high)   # the same rule
    cls3, _, _ = ic.epmv_to_class(v, "TSV_3p")
    assert cls3[:6].tolist() == [1, 1, 1, 1, -1, -1] and cls3.isna().sum() == 1
    with pytest.raises(ValueError):
        ic.epmv_to_class(np.array([0.0]), "thermal_preference")


def test_e7b_with_e_above_one_converts_and_counts_out_of_scale_epmv(monkeypatch):
    """The E7b path applies the saturation rule to ePMV (no raise) and records the counts and the e flag."""
    df = _inputs(200, seed=7)
    rng = np.random.default_rng(7)
    df["thermal_sensation"] = np.clip(np.round((df["Tair"] - 24) / 2 + rng.normal(0, 0.7, len(df))), -3, 3)
    df["TSV_3p"] = np.sign(df["thermal_sensation"])
    splits = {t: (np.arange(0, 150), np.arange(150, 200)) for t in E1_TARGETS}
    monkeypatch.setattr(ic, "check_archived_split", lambda *a, **k: None)
    real_fit = ic.fit_epmv_e

    def fit_e_3(*a, **k):                     # a calibration giving e = 3 (unbounded, flagged)
        cal = real_fit(*a, **k)
        return {**cal, "e": 3.0, "e_flag_outside_range": True}
    monkeypatch.setattr(ic, "fit_epmv_e", fit_e_3)
    res = ic.run_e7b(df, splits)
    cov = res["coverage"].set_index(["target", "method"])
    met = res["metrics"]
    for t in E1_TARGETS:
        c = cov.loc[(t, "ePMV")]
        assert c["n_saturated_low"] + c["n_saturated_high"] > 0                 # out-of-scale values counted
        row = met[(met["target"] == t) & (met["method"] == "ePMV")].iloc[0]
        assert row["status"] == "ok" and row["epmv_e"] == 3.0 and bool(row["epmv_e_flag_outside_range"])
        assert res["calibration"][t]["ePMV"]["e_flag_outside_range"] is True
    fr = res["predictions"]
    ep = fr[fr["model"] == "ePMV"]
    assert len(ep) and ep["y_pred"].dropna().astype(int).between(-3, 3).all()
    assert (ep["y_pred"].isna() == ~ep["index_valid"].astype(bool)).all()      # NaN never a class


def test_pmv_e_first_pass_is_the_e7_pmv_and_its_mask_a_subset():
    df = _inputs(300, seed=4)
    df.loc[:9, "Tair"] = 31.0                                             # outside the PMV mask
    x, _ = ic.compute_pmv_e(df, 1.0)
    pmv = ic.compute_pmv(df, "primary")
    assert not (np.isfinite(x) & ~np.isfinite(pmv)).any()
    neutral_or_cool = np.isfinite(x) & (pmv <= 0)                         # met not adjusted: e = 1 gives PMV
    assert neutral_or_cool.any()
    assert np.allclose(x[neutral_or_cool], np.round(pmv[neutral_or_cool], 2), atol=0.0100001)


def test_e7b_refuses_tpv():
    df = _inputs(50)
    with pytest.raises(ValueError, match="exactly"):
        ic.run_e7b(df, {"thermal_preference": (np.arange(40), np.arange(40, 50))})


def test_e7b_pin_covers_the_set_pet_and_pmv_e_sources():
    prov = ic.check_e7b_pin()
    assert set(prov["e7b_sources_sha256"]) == set(ic.PINNED_E7B_SOURCES_SHA256) and prov["scipy_version"]
    bad = copy.deepcopy(prov)
    bad["e7b_sources_sha256"]["models/pet_steady.py"] = "0" * 64
    with pytest.raises(ic.PythermalcomfortPinError):
        ic.check_e7b_pin(bad)


# ================================================================ E7b: dry run end to end
def test_e7b_dry_run_writes_indices_calibrations_and_manifest(tmp_path, synth, e1_ftt):
    run, e1_man = e1_ftt
    man = rr.run_e7b(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", data_csv=synth["csv"],
                     split_dir=synth["split_dir"], skip_group_ids=True, dry_run=True)
    exp = tmp_path / "E7b_indices_dry_run"
    assert man["experiment"] == "E7b_dry_run" and man["targets"] == list(E1_TARGETS)
    assert man["expected_ml"] == e1_man["ml_inventory"]
    cal = json.loads((exp / "calibration.json").read_text())
    assert sorted(cal) == sorted(E1_TARGETS)
    for t in E1_TARGETS:
        ep = cal[t]["ePMV"]                                              # unbounded, flagged
        assert ep["status"] == "ok" and ep["e_flag_outside_range"] is (not 0.5 <= ep["e"] <= 1.0)
        assert cal[t]["PTS_SET"]["status"] == "ok" and cal[t]["PTS_PET"]["status"] == "ok"
        tr, te = _split(synth, t)
        for m in ic.E7B_METHODS:
            fr = pd.read_csv(exp / "predictions" / f"E7b_dry_run__ASHRAE_2022__{t}__{m}.csv.gz")
            assert fr["row_idx"].tolist() == te.tolist() and set(fr["experiment"]) == {"E7b_dry_run"}
            assert (fr["y_pred"].isna() == ~fr["index_valid"]).all()           # NaN never a class
            assert {"index_raw", "index_value", "index_valid"} <= set(fr.columns)
    cov = pd.read_csv(exp / "coverage.csv")
    assert set(cov["method"]) == set(ic.E7B_METHODS) and set(cov["target"]) == set(E1_TARGETS)
    for scope in ("cohort_", "test_"):                                     # diagnostics per scope
        assert {f"{scope}exceptions", f"{scope}n_nonconvergence", f"{scope}other_warnings"} <= set(cov.columns)
    assert cov.loc[cov["method"] == "ePMV", ["n_saturated_low", "n_saturated_high"]].notna().all().all()
    met = pd.read_csv(exp / "metrics.csv")
    assert set(met["target"]) == set(E1_TARGETS) and "thermal_preference" not in set(met["target"])
    idx = met[met["method"].isin(ic.E7B_METHODS)]                          # transform beside the metrics
    pts = idx[idx["method"].str.startswith("PTS_")]
    assert pts[["pts_a", "pts_b"]].notna().all().all() and idx.loc[idx["method"] == "ePMV", "epmv_e"].notna().all()
    for _, r in idx.iterrows():
        dist = json.loads(r["pred_class_distribution"])
        assert sum(dist.values()) == r["n"] and r["index_value_min"] <= r["index_value_max"]
    ml = met[met["method"].str.startswith("ML:")]
    assert set(ml["subset"]) == {"on_mask", "outside_mask"} and set(ml["model"]) == {"RF", "FTT"}
    index = json.loads((tmp_path / "manifest.json").read_text())
    assert index["experiments"]["E7b_dry_run"]["status"] == "complete"


def test_paper_figures_accept_the_e10_and_e7b_runner_outputs(tmp_path, synth, e1_ftt, config, compute,
                                                             monkeypatch):
    """Producer -> consumer contract: the outputs of run_e10 / run_e7b themselves (dry runs,
    relabelled as real) pass the checks of analysis/paper_figures (``check_e10_metrics``, the E10 rows of
    Figure_Table5; ``pick_index_rows``, the E7b rows of Figure_Table6), so a change of either producer
    before the real run is caught here and not only on the real outputs."""
    from analysis.paper_figures import items
    run, _ = e1_ftt
    e1_dir = run / "E1_phase2_dry_run"
    monkeypatch.setattr(hybrid, "XGBClassifier", _StubXGB)
    rr.run_e10(synth["ctx"], run_dir=tmp_path / "e10", e1_dir=e1_dir, config=config, split_dir=synth["split_dir"],
               targets=E1_TARGETS, compute=compute, dry_run=True, command=["pytest", "e10"])
    exp = tmp_path / "e10" / "E10_hybrids_dry_run"
    met = pd.read_csv(exp / "metrics.csv")
    assert set(met["experiment"]) == {"E10_dry_run"}
    met["experiment"] = "E10"                                            # relabelled as real

    def read_predictions(name):
        fr = pd.read_csv(exp / "predictions" / name, compression="gzip")
        assert set(fr["experiment"]) == {"E10_dry_run"}
        return fr.assign(experiment="E10")
    cohort = pd.read_csv(synth["csv"])
    test_idx = {t: np.asarray(_split(synth, t)[1], dtype=np.int64) for t in E1_TARGETS}
    e1 = pd.read_csv(e1_dir / "metrics.csv")
    rows = items.check_e10_metrics(met, e1, cohort, test_idx, read_predictions, targets=E1_TARGETS, seeds=(42,))
    assert len(rows) == len(E1_TARGETS) * len(items.E10_MODELS) and set(rows["model"]) == set(items.E10_MODELS)
    assert rows["weighted_f1"].notna().all()
    # a tampered prediction is still refused on the producer's own layout
    bad = met.copy()
    bad.loc[0, "predictions"] = bad.loc[1, "predictions"]
    with pytest.raises(ValueError):
        items.check_e10_metrics(bad, e1, cohort, test_idx, read_predictions, targets=E1_TARGETS, seeds=(42,))

    rr.run_e7b(run_dir=tmp_path / "e7b", e1_dir=e1_dir, data_csv=synth["csv"], split_dir=synth["split_dir"],
               skip_group_ids=True, dry_run=True)
    e7b = pd.read_csv(tmp_path / "e7b" / "E7b_indices_dry_run" / "metrics.csv")
    assert tuple(items.E7B_TARGETS) == E1_TARGETS                        # TSV-7 and TSV-3 only
    picked = items.pick_index_rows(e7b, items.E7B_INDICES, items.E7B_TARGETS, "E7b")
    assert sorted(zip(picked["target"], picked["method"])) == sorted(
        (t, m) for t in E1_TARGETS for m in items.E7B_INDICES)
    assert picked[["coverage", "n", "accuracy", "macro_f1", "weighted_f1", "qwk"]].notna().all().all()


def test_epmv_range_gate_removed_and_e7b_manifest_records_the_epmv_flag(tmp_path, synth, e1_ftt):
    """No refusal of a real E7b over the ePMV factor e; the manifest records e and its flag."""
    assert not hasattr(rr, "E7B_D3_IMPLEMENTED") and not hasattr(rr, "E7B_D3_PENDING")
    run, _ = e1_ftt
    man = rr.run_e7b(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", data_csv=synth["csv"],
                     split_dir=synth["split_dir"], skip_group_ids=True, dry_run=True)
    cal = json.loads((tmp_path / "E7b_indices_dry_run" / "calibration.json").read_text())
    assert sorted(man["epmv_e"]) == sorted(E1_TARGETS)
    for t in E1_TARGETS:
        rec = man["epmv_e"][t]
        assert rec["e"] == cal[t]["ePMV"]["e"] and rec["e_flag_range"] == [0.5, 1.0]
        assert rec["e_flag_outside_range"] is (not 0.5 <= rec["e"] <= 1.0)


# ================================================================ CLI wiring
def test_cli_dispatches_e10_and_e7b(tmp_path, synth, monkeypatch):
    import run_revision
    seen = {}
    monkeypatch.setattr(rr, "load_ashrae_context", lambda config, csv_path=None: synth["ctx"])
    for name in ("run_e10", "run_e7b"):
        monkeypatch.setattr(rr, name, lambda *a, _n=name, **kw: seen.setdefault(_n, []).append(kw) or kw)
    kw = run_revision.main(["e10", "--run_dir", str(tmp_path), "--dry_run", "--targets", "TSV_3p"])
    assert kw["dry_run"] and kw["targets"] == ["TSV_3p"] and kw["e1_dir"] == tmp_path / "E1_phase2_dry_run"
    assert kw["split_dir"] == "splits/ASHRAE_2022" and kw["config"]["hybrid"] == rr.DRY_RUN_HPARAMS["hybrid"]
    kw = run_revision.main(["e7b", "--run_dir", str(tmp_path), "--dry_run", "--skip_group_ids"])
    assert kw["dry_run"] and kw["skip_group_ids"] and kw["e1_dir"] == tmp_path / "E1_phase2_dry_run"
    sub = run_revision.parse_args(["e10", "--run_dir", "x"])
    assert sub.targets == list(rs.TARGETS) and not sub.dry_run
