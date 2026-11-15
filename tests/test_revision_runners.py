# -*- coding: utf-8 -*-
"""
Runners of the revision rerun (``revision_runners.py``, ``run_revision.py``).

Dry runs on a tiny synthetic cohort (300 rows, 4 publications, tiny models):

* E1: five families, frozen seeds, shared folds (one hash for every family),
  per-observation predictions on the archived test rows in split order (SVM: NaN
  probabilities; ANN/FTT: probabilities), early-stopping records, the
  ``best_*_by_target`` files and split copies read by E6, majority baselines,
  manifest (commit, env lock hash, data / config hashes, seeds, command) and
  run index; byte-identical re-execution of the CPU families;
* E2: one fold per publication, training rows = the other publications, every
  row predicted once, OOV votes kept and flagged;
* E4: F12 fits paired with the E1 F17 fits (a direct fit equals the E1
  grid refit; the fitted columns are observed), majority baselines; E1 sources
  refused when missing, incomplete (no manifest), modified, swapped, with other
  labels, another configuration / XGB device / environment, or a dry run for a
  real run (``load_e1_source``);
* E6: typed driver (no path override, one dry-run state, E1 source checked,
  scheduler job records in the manifest);
* E7: the indices driver (``indicateurs_classiques.py``) on the E1 dry-run
  predictions with the E1 inventory, dry-run label in files and rows; a real E7
  refuses dry-run E1 predictions and ``skip_group_ids``; the indices-driver argv
  keeps the output directory out of the open ``--expected_ml`` list in real and
  dry runs (a bug found in a real run);
* guards: frozen artefact, non-empty output, seeds, modified config, unlocked
  environment and modified checkout refused for real runs; run index under a
  lock; effective scheduling recorded; CLI dispatch of every subcommand.

A second E1 on a folds directory that already holds the fold files loads and
checks them (never rewrites): ``tests/test_folds_seeds.py::
test_ensure_folds_writes_once_and_rechecks``.
"""
from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import revision_folds  # noqa: E402
import revision_runners as rr  # noqa: E402
import revision_stats as rs  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY  # noqa: E402
from config_loader import load_config  # noqa: E402
from phase3_runner import file_sha256, prediction_columns  # noqa: E402

N = 300
PUBS = ("P0", "P1", "P2", "P3")


# ================================================================ fixtures
@pytest.fixture(scope="module")
def compute():
    import torch
    import xgboost
    from compute import load_compute
    cfg = {**load_compute(), "cpu_jobs": 2, "gpu_concurrency": 1}
    if not torch.cuda.is_available():
        cfg.update(xgb_device="cpu", torch_device="cpu")
    try:   # hidden GPU with the CUDA build of xgboost (see conftest.xgb_fits)
        xgboost.XGBClassifier(n_estimators=1, device="cpu", tree_method="hist").fit(
            np.arange(30.0).reshape(10, 3), np.arange(10) % 3)
    except xgboost.core.XGBoostError as exc:
        if "cudaErrorNoDevice" in str(exc):
            pytest.skip("xgboost CUDA build cannot fit with the GPU hidden")
        raise
    return cfg


@pytest.fixture(scope="module")
def config():
    cfg = copy.deepcopy(load_config(rr.CONFIG_PATH))
    for section, values in rr.DRY_RUN_HPARAMS.items():
        cfg[section] = copy.deepcopy(values)
    return cfg


def _synthetic_cohort(n=N, seed=0):
    """Cohort CSV columns, group identifiers (P3 alone has Climate Dfb) and raw votes with halves."""
    rng = np.random.default_rng(seed)
    pub = np.asarray(PUBS)[np.arange(n) % 4]
    raw = rng.choice(np.arange(-3.0, 3.01, 0.5), size=n)
    raw[:6] = [-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]
    tsv7 = np.floor(raw + 0.5)
    tair = rng.uniform(18, 32, n)
    other_climates = [c for c in sorted(GLOBAL_VOCABULARY["Climate"]) if c != "Dfb"]
    df = pd.DataFrame({
        "Date": [f"2020-01-{1 + i % 28:02d}" for i in range(n)],
        "Season": np.asarray(sorted(GLOBAL_VOCABULARY["Season"]))[(np.arange(n) // 4) % 4],   # not tied to pub
        "Âge": rng.uniform(20, 60, n),
        "Sexe": np.resize(sorted(GLOBAL_VOCABULARY["Sexe"]), n),
        "Tair": tair + 0.3 * tsv7,
        "tr": tair + rng.normal(0, 1, n),
        "RH": rng.uniform(30, 70, n), "vel": rng.uniform(0.05, 0.3, n), "Met": rng.uniform(1.0, 1.4, n),
        "clo": rng.uniform(0.4, 1.0, n),
        "thermal_sensation": tsv7,
        "thermal_preference": np.clip(-np.sign(tsv7) + rng.integers(-1, 2, n) * (rng.random(n) < 0.2), -1, 1),
        "Tout": rng.normal(20, 5, n),
        "Building_type": np.resize(sorted(GLOBAL_VOCABULARY["Building_type"]), n),
        "cooling type": np.resize(sorted(GLOBAL_VOCABULARY["cooling type"]), n),
        "TSV_3p": rr.tsv7_to_tsv3(tsv7),
        "Climate": np.where(pub == "P3", "Dfb", np.resize(other_climates, n)),
    })
    for c in ("Trm_ema_28", "RHout_ema_28", "precip_ema_28", "sunshine_ema_h_28", "wind_ema_28"):
        df[c] = rng.normal(size=n)
    df.loc[:4, "tr"] = df.loc[:4, "Tair"] + 12.0          # 5 rows with |tr - Tair| > 10 (P0, P1, ...)
    ids = pd.DataFrame({"record_id": np.arange(1000, 1000 + n), "publication": pub,
                        "building_id": np.arange(n) % 20,
                        "subject_id": pd.array([pd.NA if i % 7 == 0 else i for i in range(n)], dtype="Int64"),
                        "thermal_sensation_raw": raw})
    return df, ids


@pytest.fixture(scope="module")
def synth(tmp_path_factory, config):
    base = tmp_path_factory.mktemp("synth")
    df, ids = _synthetic_cohort()
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
def e1_trees(tmp_path_factory, synth, config, compute):
    """E1 dry run, RF + XGB (seeds 42 43 44), all three targets, CPU only."""
    run = tmp_path_factory.mktemp("run_trees")
    man = rr.run_e1(synth["ctx"], run_dir=run, config=config, split_dir=synth["split_dir"],
                    families=("RF", "XGB"), compute=compute, gpu_concurrency=1, dry_run=True,
                    command=["pytest", "e1_trees"])
    return run, man


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path)


def _split(synth, t):
    return revision_folds.load_archived_split(synth["split_dir"], t, synth["df"][t])


# ================================================================ config and small helpers
def test_fixed_params_needs_a_single_point():
    assert rr.fixed_params({"a": [1], "b": ["x"]}) == {"a": 1, "b": "x"}
    with pytest.raises(rr.RunnerError):
        rr.fixed_params({"kernel": ["rbf", "linear"]})


def test_ann_predict_proba_is_consistent_with_predict():
    import models_deep as md
    from category_normalisation import vocabulary_for
    df, _ = _synthetic_cohort(n=200, seed=3)
    num, cat = ["Tair", "RH"], ["Sexe", "Season"]
    m = md.ANNClassifier(num_cols=num, cat_cols=cat, hidden=(8,), epochs=3, batch_size=64, device="cpu",
                         cat_vocabulary=vocabulary_for(cat), random_state=42).fit(df[num + cat], df["TSV_3p"])
    p = m.predict_proba(df[num + cat])
    assert p.shape == (200, 3) and np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(m.classes_[p.argmax(axis=1)], m.predict(df[num + cat]))


# ================================================================ E1
def test_e1_tree_dry_run_predictions_manifest_and_e6_inputs(synth, e1_trees):
    run, man = e1_trees
    exp = run / "E1_phase2_dry_run"
    assert man["experiment"] == "E1_dry_run" and man["dry_run"]
    preds = exp / "predictions"
    for t in rs.TARGETS:
        tr, te = _split(synth, t)
        for fam in ("RF", "XGB"):
            for s in (42, 43, 44):
                fr = _read(preds / f"E1_dry_run__ASHRAE_2022__{t}__{fam}__seed{s}__archived_80_20.csv.gz")
                assert list(fr.columns) == prediction_columns(t)
                assert fr["row_idx"].tolist() == te.tolist()                       # split order
                assert fr["y_true"].tolist() == synth["df"][t].iloc[te].astype(int).tolist()
                assert (fr["model"] == fam).all() and (fr["seed"] == s).all() and (fr["feature_set"] == "F17").all()
                assert fr["record_id"].tolist() == synth["ids"]["record_id"].iloc[te].tolist()
                assert fr["publication"].tolist() == synth["ids"]["publication"].iloc[te].tolist()
                assert fr["building_id"].tolist() == synth["ids"]["building_id"].iloc[te].tolist()
                assert fr["date"].tolist() == synth["df"]["Date"].iloc[te].tolist()
                sid = synth["ids"]["subject_id"].iloc[te]
                assert fr["subject_key"].isna().tolist() == sid.isna().tolist()
                p = fr[[f"p_{c}" for c in rs.ontology(t)]].to_numpy()
                assert np.allclose(p.sum(axis=1), 1.0)
                assert not fr["oov_flag"].any()                                      # no OOV in E1
        # files read by E6 (transfert.load_model_dicts, phase3_runner.load_source_rows)
        split = json.loads((exp / t / "splits" / f"{t}_split_indices.json").read_text())
        assert split["train_idx"] == tr.tolist() and split["test_idx"] == te.tolist()
        folds = json.loads((exp / t / "splits" / f"{t}_folds.json").read_text())
        assert folds["fold_vector_sha256"] == man["folds"][t]["fold_vector_sha256"]
    rf = joblib.load(exp / "best_rf_by_target.joblib")
    assert set(rf) == set(rs.TARGETS) and rf["TSV_3p"].named_steps["clf"].random_state == 42
    assert joblib.load(exp / "best_xgb_by_target_seed44.joblib")["TSV_3p"].named_steps["clf"].estimator.random_state == 44
    from category_normalisation import categorical_features, vocabulary_for
    from models_classic import check_onehot_revision_schema
    vocab = vocabulary_for(categorical_features(rr.features_for(load_config(rr.CONFIG_PATH), "F17")))
    for m in rf.values():
        check_onehot_revision_schema(m, vocab)
    # metrics table with the majority baselines
    met = pd.read_csv(exp / "metrics.csv")
    assert set(met["model"]) == {"RF", "XGB", "Majority_deployable", "Majority_descriptive"}
    assert len(met) == 3 * (6 + 2)
    # manifest
    assert man["ml_inventory"] == sorted(f"{f}:{s}" for f in ("RF", "XGB") for s in (42, 43, 44))
    assert man["seeds"]["models"] == {"RF": [42, 43, 44], "XGB": [42, 43, 44]}
    assert man["command"] == ["pytest", "e1_trees"]
    assert man["provenance"]["git_commit"] and len(man["environment"]["lock_file_sha256"]) == 64
    assert man["environment"]["pip_freeze"] and man["config_file_sha256"] == file_sha256(rr.CONFIG_PATH)
    assert man["config_is_file"] is False                                          # dry-run overrides
    assert man["inputs_sha256"]["cohort_csv"] == file_sha256(synth["csv"])
    assert {f"split:{t}" for t in rs.TARGETS} <= set(man["inputs_sha256"])
    assert len(man["jobs"]) == 18 and all(j["kind"] == "cpu" for j in man["jobs"])
    assert man["scheduler"]["gpu_concurrency"] == 1 and man["scheduler"]["gpu_concurrency_source"] == "override"
    cov = json.loads((exp / "category_coverage_ASHRAE_2022.json").read_text())         # category coverage log
    assert {c["variable"] for c in cov} == {"Season", "Sexe", "Building_type", "cooling type", "Climate"}
    assert all(c["n_rows"] == N and c["rows_rejected"] == 0 for c in cov)
    assert man["outputs_sha256"]["category_coverage_ASHRAE_2022.json"] == file_sha256(
        exp / "category_coverage_ASHRAE_2022.json")
    on_disk = json.loads((exp / "run_manifest.json").read_text())
    assert on_disk["outputs_sha256"]["metrics.csv"] == file_sha256(exp / "metrics.csv")
    index = json.loads((run / "manifest.json").read_text())
    assert "E1_dry_run" in index["experiments"] and index["phase1"]["rerun"] is False
    with pytest.raises(FileExistsError):
        rr.update_run_index(run, "E1_dry_run", exp / "run_manifest.json")


def test_e1_five_families_share_folds_and_write_deep_records(tmp_path, synth, config, compute):
    t = "TSV_3p"
    man = rr.run_e1(synth["ctx"], run_dir=tmp_path, config=config, split_dir=synth["split_dir"], targets=[t],
                    compute=compute, gpu_concurrency=1, dry_run=True)
    exp = tmp_path / "E1_phase2_dry_run"
    tr, te = _split(synth, t)
    assert man["ml_inventory"] == sorted(["RF:42", "RF:43", "RF:44", "XGB:42", "XGB:43", "XGB:44", "SVM:42",
                                          "ANN:42", "FTT:42"])
    kinds = {j["key"].split("|")[1]: j["kind"] for j in man["jobs"]}
    assert kinds["ANN"] == kinds["FTT"] == "gpu" and kinds["RF"] == kinds["SVM"] == "cpu"
    assert kinds["XGB"] == ("gpu" if compute["xgb_device"] == "cuda" else "cpu")
    hashes = set()
    for fam, seeds in (("RF", (42, 43, 44)), ("XGB", (42, 43, 44)), ("SVM", (42,)), ("ANN", (42,)), ("FTT", (42,))):
        for s in seeds:
            fr = _read(exp / "predictions" / f"E1_dry_run__ASHRAE_2022__{t}__{fam}__seed{s}__archived_80_20.csv.gz")
            assert fr["row_idx"].tolist() == te.tolist() and len(fr) == len(te)
            p = fr[["p_-1", "p_0", "p_1"]].to_numpy()
            if fam == "SVM":
                assert np.isnan(p).all()                                  # SVM: no probabilities
            else:
                assert np.allclose(p.sum(axis=1), 1.0)
            fold = pd.read_csv(exp / t / fam / f"seed_{s}" / f"{t}__{fam}__seed{s}_fold_summary.csv")
            hashes |= set(fold["folds_sha256"])
    assert hashes == {man["folds"][t]["fold_vector_sha256"]}             # one fold vector for every family
    es = json.loads((exp / t / "FTT" / "seed_42" / f"{t}__FTT__seed42_early_stopping.json").read_text())
    assert es["refit_on_full_training_rows"] is False and es["n_fit_rows"] == len(tr)
    assert set(es["es_val_idx"]) <= set(tr.tolist())                     # internal validation inside train_idx
    curves = pd.read_csv(exp / t / "ANN" / "seed_42" / f"{t}__ANN__seed42_training_curves.csv")
    assert {"epoch", "train_loss", "val_loss"} <= set(curves.columns)
    ft = joblib.load(exp / "best_ft_by_target.joblib")[t]
    from category_normalisation import categorical_features, vocabulary_for
    from models_deep import check_ftt_revision_schema
    check_ftt_revision_schema(ft, vocabulary_for(categorical_features(rr.features_for(config, "F17"))))
    assert ft.random_state == 42 and (exp / "best_svm_by_target.joblib").exists()


def test_e1_cpu_families_are_byte_identical_on_re_execution(tmp_path, synth, config, compute):
    """CPU tree models and SVM, identical per-observation predictions (byte-identical files)."""
    outs = []
    for name in ("a", "b"):
        rr.run_e1(synth["ctx"], run_dir=tmp_path / name, config=config, split_dir=synth["split_dir"],
                  targets=["thermal_preference"], families=("RF", "XGB", "SVM"), compute=compute,
                  gpu_concurrency=1, dry_run=True)
        outs.append(tmp_path / name / "E1_phase2_dry_run" / "predictions")
    files = sorted(p.name for p in outs[0].iterdir())
    assert len(files) == 7
    for f in files:
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes()


def test_guards_refuse_before_any_write(tmp_path, synth, config, compute):
    ctx, split = synth["ctx"], synth["split_dir"]
    with pytest.raises(ValueError, match="frozen artefact"):
        rr.run_e1(ctx, run_dir=REPO_DIR / "rerun_2026-04-28_no_reweighting" / "x", config=config, split_dir=split,
                  compute=compute, dry_run=True)
    with pytest.raises(ValueError, match="not frozen"):
        rr.run_e1(ctx, run_dir=tmp_path / "s", config=config, split_dir=split, tree_seeds=(1, 2, 3),
                  compute=compute, dry_run=True)
    assert not (tmp_path / "s" / "E1_phase2_dry_run").exists()
    busy = tmp_path / "busy" / "E2_lopo_dry_run"
    busy.mkdir(parents=True)
    (busy / "old.csv").write_text("x")
    with pytest.raises(FileExistsError):
        rr.run_e2(ctx, run_dir=tmp_path / "busy", config=config, compute=compute, dry_run=True)
    # a real run needs config.yaml unchanged and the locked environment
    with pytest.raises(rr.RunnerError, match="config.yaml"):
        rr.run_e2(ctx, run_dir=tmp_path / "real", config=config, compute=compute, dry_run=False)
    assert not (tmp_path / "real").exists()
    env = {"matches_lock": False, "lock_file": "lock", "differences_from_lock": ["torch: lock 2.7.0, installed 2.9"]}
    with pytest.raises(rr.RunnerError, match="locked environment"):
        rr.check_locked_environment(env, dry_run=False)
    rr.check_locked_environment(env, dry_run=True)
    rr.check_config_unmodified(load_config(rr.CONFIG_PATH), dry_run=False)


def test_environment_record_lists_differences_from_the_lock(tmp_path):
    env = rr.environment_record()
    assert env["pip_freeze_sha256"] == rr._sha256_bytes("\n".join(env["pip_freeze"]).encode())
    fake = tmp_path / "lock"
    fake.write_text("# comment\nnumpy==0.0.1\n")
    env2 = rr.environment_record(fake)
    assert env2["matches_lock"] is False and env2["differences_from_lock"][0].startswith("numpy: lock 0.0.1")


# ================================================================ E2 (leave-one-publication-out)
def test_e2_lopo_predicts_every_row_once_and_keeps_oov_votes(tmp_path, synth, config, compute):
    t = "TSV_3p"
    man = rr.run_e2(synth["ctx"], run_dir=tmp_path, config=config, targets=[t], compute=compute,
                    gpu_concurrency=3, dry_run=True)       # CPU jobs only: the override changes nothing but is recorded
    assert man["scheduler"]["gpu_concurrency"] == 3 and man["scheduler"]["gpu_concurrency_config"] == 1
    assert man["scheduler"]["gpu_concurrency_source"] == "override"
    assert man["scheduler"]["xgb_device"] == compute["xgb_device"]
    exp = tmp_path / "E2_lopo_dry_run"
    folds = json.loads((exp / "lopo_folds.json").read_text())
    assert folds["publications"] == list(PUBS) and man["n_folds"] == 4
    pubs = synth["ids"]["publication"].to_numpy()
    for f in folds["folds"]:
        assert f["n_test"] == int((pubs == f["publication"]).sum()) and f["n_train"] == N - f["n_test"]
        assert f["n_oov_votes"] == (f["n_test"] if f["publication"] == "P3" else 0)
    for fam in ("RF", "XGB"):
        frames = [_read(exp / "predictions" / f"E2_dry_run__ASHRAE_2022__{t}__{fam}__seed42__lopo_{k:02d}.csv.gz")
                  for k in range(4)]
        allp = pd.concat(frames, ignore_index=True)
        assert sorted(allp["row_idx"]) == list(range(N))                      # every row once
        for k, fr in enumerate(frames):
            assert set(fr["publication"]) == {PUBS[k]}
            assert fr["row_idx"].tolist() == np.flatnonzero(pubs == PUBS[k]).tolist()
            if PUBS[k] == "P3":                                               # OOV kept, counted, flagged
                assert fr["oov_flag"].all() and set(fr["oov_vars"]) == {"Climate"}
            else:
                assert not fr["oov_flag"].any()
    met = pd.read_csv(exp / "metrics.csv")
    fits = met[met["model"].isin(["RF", "XGB"])]
    assert len(fits) == 8 and (fits["n_train"] == N - fits["n"]).all()
    assert fits.loc[fits["publication"] == "P3", "n_oov_votes"].tolist() == [75, 75]
    assert (met["model"] == "Majority_deployable").sum() == 4
    agg = pd.read_csv(exp / "aggregates.csv")
    assert set(agg["aggregate"]) == {"micro", "macro_over_publications"}
    assert agg.loc[agg["aggregate"] == "micro", "n"].tolist() == [N, N]


# ================================================================ E4 (F12 / F17 ablation)
def test_direct_fit_equals_the_e1_grid_refit(synth, e1_trees, config, compute):
    """The F17 side of E4 is the E1 fit: a direct fit with the config point gives the same predictions."""
    from category_normalisation import split_feature_types
    run, _ = e1_trees
    t = "thermal_sensation"
    tr, te = _split(synth, t)
    feats = rr.features_for(config, "F17")
    X, y = synth["ctx"].data[feats], synth["ctx"].data[t]
    num, cat = split_feature_types(X, feats)
    for fam in ("RF", "XGB"):
        m = rr.fit_fixed(fam, num, cat, 43, config, compute, X.iloc[tr], y.iloc[tr])
        e1 = _read(rr.e1_prediction_file(run / "E1_phase2_dry_run", t, fam, 43, True))
        assert np.array_equal(m.predict(X.iloc[te]).astype(int), e1["y_pred"].to_numpy())
        assert np.allclose(m.predict_proba(X.iloc[te]), e1[[f"p_{c}" for c in rs.ontology(t)]].to_numpy(),
                           rtol=0, atol=1e-12)


EMA_VARS = ("Trm_ema_28", "RHout_ema_28", "precip_ema_28", "sunshine_ema_h_28", "wind_ema_28")


def test_e4_pairs_new_fits_with_e1(tmp_path, synth, e1_trees, config, compute, monkeypatch):
    run, _ = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    fitted_columns = []
    real_fit_fixed = rr.fit_fixed

    def spy(family, num_cols, cat_cols, seed, config_, compute_, X, y):     # CPU jobs run in this process
        fitted_columns.append(list(X.columns))
        return real_fit_fixed(family, num_cols, cat_cols, seed, config_, compute_, X, y)
    if compute["xgb_device"] == "cpu":
        monkeypatch.setattr(rr, "fit_fixed", spy)
    for exp_name, fitted, base, ext in (("E4", "F12", "F12", "F17"),):
        fitted_columns.clear()
        man = rr.run_ablation(synth["ctx"], experiment=exp_name, run_dir=tmp_path, e1_dir=e1_dir, config=config,
                              split_dir=synth["split_dir"], compute=compute, gpu_concurrency=1, dry_run=True)
        exp = rr.experiment_dir(tmp_path, exp_name, True)
        if compute["xgb_device"] == "cpu":                                  # the columns actually fitted
            assert len(fitted_columns) == 18
            assert all(c == rr.features_for(config, fitted) for c in fitted_columns)
            assert all(not set(EMA_VARS) & set(c) for c in fitted_columns)
        assert man["fitted_features"] == rr.features_for(config, fitted)
        pairs = pd.read_csv(exp / "pairs.csv")
        assert len(pairs) == 3 * 2 * 3
        assert set(pairs["e1_manifest_sha256"]) == {file_sha256(e1_dir / "run_manifest.json")}
        met = pd.read_csv(exp / "metrics.csv")                                # majority baselines
        assert (met["model"] == "Majority_deployable").sum() == 3 and (met["model"] == "Majority_descriptive").sum() == 3
        assert set(pairs["base_feature_set"]) == {base} and set(pairs["extended_feature_set"]) == {ext}
        for _, p in pairs.iterrows():
            b, e = _read(p["base_file"]), _read(p["extended_file"])
            assert b["row_idx"].tolist() == e["row_idx"].tolist() and b["y_true"].equals(e["y_true"])
            assert set(b["feature_set"]) == {base} and set(e["feature_set"]) == {ext}
            assert (b["seed"] == p["seed"]).all() and (e["model"] == p["model"]).all()
            assert file_sha256(p["base_file"]) == p["base_sha256"]
        new = list((exp / "predictions").iterdir())
        assert len(new) == 18 and all(f"{exp_name}_dry_run__" in f.name for f in new)
        assert len(man["shared_f17_predictions_sha256"]) == 18
    assert not (tmp_path / "E3_tr_dry_run").exists() and set(rr.ABLATIONS) == {"E4"}


def test_ablation_refuses_missing_e1_predictions(tmp_path, synth, config, compute):
    with pytest.raises(rr.RunnerError, match="E1 F17 predictions missing"):
        rr.run_ablation(synth["ctx"], experiment="E4", run_dir=tmp_path, e1_dir=tmp_path / "nope", config=config,
                        split_dir=synth["split_dir"], compute=compute, dry_run=True)
    assert not rr.experiment_dir(tmp_path, "E4", True).exists()
    with pytest.raises(ValueError, match="unknown ablation"):                 # a removed ablation is refused
        rr.run_ablation(synth["ctx"], experiment="E3", run_dir=tmp_path, e1_dir=tmp_path / "nope", config=config,
                        split_dir=synth["split_dir"], compute=compute, dry_run=True)


def _e1_copy(e1_trees, dst) -> Path:
    import shutil
    run, _ = e1_trees
    return Path(shutil.copytree(run / "E1_phase2_dry_run", dst))


def _rewrite_predictions(path, **changes):
    fr = pd.read_csv(path)
    for col, fn in changes.items():
        fr[col] = fn(fr[col])
    fr.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def _manifest_edit(e1_dir, fn):
    man = json.loads((e1_dir / "run_manifest.json").read_text())
    fn(man)
    (e1_dir / "run_manifest.json").write_text(json.dumps(man))


def _ablation(synth, config, compute, run_dir, e1_dir, **kw):
    return rr.run_ablation(synth["ctx"], experiment="E4", run_dir=run_dir, e1_dir=e1_dir, config=config,
                           split_dir=synth["split_dir"], targets=["TSV_3p"], families=["RF"], compute=compute,
                           gpu_concurrency=1, dry_run=True, **kw)


def test_ablation_accepts_the_recorded_lock_hash_mapped_to_the_release_lock(tmp_path, synth, e1_trees, config,
                                                                            compute):
    """An E1 manifest that recorded the lock file of the recorded run pairs with the release lock (comment-only
    change, ``RELEASE_FILE_SHA256``); any other lock hash is refused."""
    current = rr.environment_record()["lock_file_sha256"]
    recorded = {v: k for k, v in rr.RELEASE_FILE_SHA256.items()}.get(current)
    if recorded is None:
        pytest.skip("lock file is not the release lock")
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    _manifest_edit(e1, lambda m: m["environment"].__setitem__("lock_file_sha256", recorded))
    _ablation(synth, config, compute, tmp_path / "run", e1)
    e1b = _e1_copy(e1_trees, tmp_path / "E1b")
    _manifest_edit(e1b, lambda m: m["environment"].__setitem__("lock_file_sha256", "0" * 64))
    with pytest.raises(rr.RunnerError, match="lock_file_sha256"):
        _ablation(synth, config, compute, tmp_path / "run_b", e1b)


@pytest.mark.parametrize("case", ["no_manifest", "modified_predictions", "swapped_seed", "changed_truth",
                                  "other_config", "other_xgb_device", "other_environment", "other_commit",
                                  "other_split"])
def test_ablation_refuses_an_e1_source_that_is_not_the_paired_f17_side(tmp_path, synth, e1_trees, config,
                                                                       compute, case, monkeypatch):
    """E1 must be complete, unmodified and from the same code / config / device / environment / rows."""
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    f42 = rr.e1_prediction_file(e1, "TSV_3p", "RF", 42, True)
    if case == "no_manifest":                                   # E1 crashed before writing its manifest
        (e1 / "run_manifest.json").unlink()
        match = "E1 did not complete"
    elif case == "modified_predictions":
        _rewrite_predictions(f42, y_pred=lambda c: -c)
        match = "modified after E1"
    elif case == "swapped_seed":                                # seed-43 content under the seed-42 name, hash updated
        f43 = rr.e1_prediction_file(e1, "TSV_3p", "RF", 43, True)
        f42.write_bytes(f43.read_bytes())
        _manifest_edit(e1, lambda m: m["outputs_sha256"].__setitem__(f"predictions/{f42.name}", file_sha256(f42)))
        match = "seed"
    elif case == "changed_truth":                               # aligned rows, other labels, hash updated
        _rewrite_predictions(f42, y_true=lambda c: np.where(c == 1, -1, c))
        _manifest_edit(e1, lambda m: m["outputs_sha256"].__setitem__(f"predictions/{f42.name}", file_sha256(f42)))
        match = "y_true differs"
    elif case == "other_config":                                # e.g. classical hyperparameters changed after E1
        _manifest_edit(e1, lambda m: m.__setitem__("config_used_sha256", "0" * 64))
        match = "config_used_sha256"
    elif case == "other_xgb_device":
        _manifest_edit(e1, lambda m: m["compute"].__setitem__("xgb_device", "cuda" if compute["xgb_device"] == "cpu"
                                                                else "cpu"))
        match = "xgb_device"
    elif case == "other_environment":
        _manifest_edit(e1, lambda m: m["environment"].__setitem__("pip_freeze_sha256", "0" * 64))
        match = "pip_freeze"
    elif case == "other_commit":
        _manifest_edit(e1, lambda m: m["provenance"].__setitem__("git_commit", "0" * 40))
        match = "git_commit"
    else:                                                       # other train/test rows than E1
        _manifest_edit(e1, lambda m: m["inputs_sha256"].__setitem__("split:TSV_3p", "0" * 64))
        match = "differs from the E1 input"
    with pytest.raises(rr.RunnerError, match=match):
        _ablation(synth, config, compute, tmp_path / "run", e1)
    assert not (tmp_path / "run" / "E4_ema_dry_run").exists()                 # refused before any write


def test_load_e1_source_refuses_a_dry_run_e1_for_a_real_run(e1_trees):
    run, _ = e1_trees
    e1 = run / "E1_phase2_dry_run"
    assert rr.load_e1_source(e1, dry_run=True)["experiment"] == "E1_dry_run"
    with pytest.raises(rr.RunnerError, match="dry run"):
        rr.load_e1_source(e1, dry_run=False)
    with pytest.raises(rr.RunnerError, match="not an output recorded"):
        rr.load_e1_source(e1, dry_run=True, files=[e1 / "run_manifest.json"])


# ================================================================ E7 driver
def test_e7_driver_compares_the_indices_with_the_e1_inventory(tmp_path, synth, e1_trees):
    run, e1_man = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    man = rr.run_e7(run_dir=tmp_path, e1_dir=e1_dir, data_csv=synth["csv"], split_dir=synth["split_dir"],
                    skip_group_ids=True, dry_run=True)
    exp = tmp_path / "E7_pmv_dry_run"
    assert man["expected_ml"] == e1_man["ml_inventory"] and man["e1_experiment"] == "E1_dry_run"
    met = pd.read_csv(exp / "metrics.csv")
    assert {"PMV", "PMV_dynamic"} <= set(met["method"])
    ml = met[met["method"].str.startswith("ML:")]
    assert set(ml["model"]) == {"RF", "XGB"} and set(ml["subset"]) == {"on_mask", "outside_mask"}
    for t in rs.TARGETS:                                                     # dry-run label in files and rows
        fr = pd.read_csv(exp / "predictions" / f"E7_dry_run__ASHRAE_2022__{t}__PMV.csv.gz")
        assert set(fr["experiment"]) == {"E7_dry_run"}
        assert not (exp / "predictions" / f"E7__ASHRAE_2022__{t}__PMV.csv.gz").exists()
    prov = json.loads((exp / "provenance.json").read_text())
    assert prov["dry_run"] is True and prov["experiment"] == "E7_dry_run"
    with pytest.raises(ValueError, match="differs from the expected"):
        rr.run_e7(run_dir=tmp_path / "b", e1_dir=e1_dir, data_csv=synth["csv"], split_dir=synth["split_dir"],
                  expected_ml=["RF:42"], skip_group_ids=True, dry_run=True)


def _as_real_run(monkeypatch):
    """Lift the real-run environment / checkout guards (tested on their own) to reach the source checks."""
    monkeypatch.setattr(rr, "run_provenance", lambda: {"git_commit": "c" * 40, "git_tracked_changes": False})
    monkeypatch.setattr(rr, "check_locked_environment", lambda env, dry_run: None)
    monkeypatch.setattr(rr, "check_phase1_artefact", lambda dry_run: rr.phase1_provenance())


def test_phase1_artefact_hashes_are_pinned_and_checked(tmp_path, monkeypatch):
    """A real run (start and manifest) refuses a regenerated or missing
    Phase 1 grid instead of recording its new hash silently; dry runs record it. The shipped grid
    (results/phase1/experiment_grid_full.csv) has the pinned hash."""
    shipped = rr.REPO_DIR / "results" / "phase1" / "experiment_grid_full.csv"
    assert rr.PHASE1_DIR == shipped.parent
    assert file_sha256(shipped) == rr.PINNED_PHASE1_GRID_SHA256
    assert rr.check_phase1_artefact(dry_run=False)["matches_pinned"] is True
    p1 = tmp_path / "phase1"
    p1.mkdir()
    grid = p1 / "experiment_grid_full.csv"
    grid.write_text("cell,acc\n1,0.5\n")
    monkeypatch.setattr(rr, "PHASE1_DIR", p1)
    monkeypatch.setattr(rr, "PINNED_PHASE1_GRID_SHA256", file_sha256(grid))
    assert rr.check_phase1_artefact(dry_run=False)["matches_pinned"] is True
    monkeypatch.setattr(rr, "run_provenance", lambda: {"git_commit": "c" * 40, "git_tracked_changes": False})
    monkeypatch.setattr(rr, "check_locked_environment", lambda env, dry_run: None)
    for tamper in (lambda: grid.write_text("cell,acc\n1,0.6\n"), lambda: grid.unlink()):
        grid.write_text("cell,acc\n1,0.5\n")
        tamper()
        prov = rr.check_phase1_artefact(dry_run=True)                  # a dry run records the mismatch
        assert prov["matches_pinned"] is False
        with pytest.raises(rr.RunnerError, match="Phase 1 artefact"):
            rr.check_phase1_artefact(dry_run=False)
        with pytest.raises(rr.RunnerError, match="Phase 1 artefact"):
            rr._prepare(tmp_path / "run", "E9b", load_config(rr.CONFIG_PATH), False)
        with pytest.raises(rr.RunnerError, match="Phase 1 artefact"):
            rr.write_manifest(tmp_path / "exp", tmp_path / "run", experiment="E9b", dry_run=False, command=None,
                              started="t", config={}, compute={}, env={}, inputs={}, seeds=None,
                              startup=rr.startup_record(rr.run_provenance(), {}))
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_real_e7_refuses_dry_run_e1_predictions_even_with_an_explicit_inventory(tmp_path, synth, e1_trees,
                                                                               monkeypatch):
    """Valid, correctly aligned dry-run E1 predictions never reach a production E7."""
    _as_real_run(monkeypatch)
    run, e1_man = e1_trees
    with pytest.raises(rr.RunnerError, match="dry run"):
        rr.run_e7(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", data_csv=synth["csv"],
                  split_dir=synth["split_dir"], expected_ml=e1_man["ml_inventory"], dry_run=False)
    with pytest.raises(rr.RunnerError, match="dry runs only"):
        rr.run_e7(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", data_csv=synth["csv"],
                  split_dir=synth["split_dir"], skip_group_ids=True, dry_run=False)
    assert not (tmp_path / "E7_pmv").exists()


def test_e7_refuses_predictions_not_recorded_by_e1(tmp_path, synth, e1_trees):
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    extra = e1 / "predictions" / "E1_dry_run__ASHRAE_2022__TSV_3p__RF__seed45__archived_80_20.csv.gz"
    extra.write_bytes(rr.e1_prediction_file(e1, "TSV_3p", "RF", 44, True).read_bytes())
    with pytest.raises(rr.RunnerError, match="not an output recorded"):
        rr.run_e7(run_dir=tmp_path / "run", e1_dir=e1, data_csv=synth["csv"], split_dir=synth["split_dir"],
                  skip_group_ids=True, dry_run=True)
    assert not (tmp_path / "run" / "E7_pmv_dry_run").exists()


# ---------------------------------------------------------------- E7 argv (bug found in a real run)
REAL_E1_INVENTORY = ["ANN:42", "ANN:43", "ANN:44", "FTT:42", "FTT:43", "FTT:44", "RF:42", "RF:43", "RF:44",
                     "SVM:42", "XGB:42", "XGB:43", "XGB:44"]


@pytest.mark.parametrize("dry_run,skip_group_ids", [(False, False), (True, False), (True, True)])
def test_e7_argv_keeps_the_output_directory_in_real_and_dry_runs(tmp_path, dry_run, skip_group_ids):
    """The positional output directory is never taken into the open ``--expected_ml`` list, real run included
    (no ``--dry_run`` after the list): an E7 argument bug found in a real run."""
    import indicateurs_classiques as ic
    exp_dir = tmp_path / "RUN" / ("E7_pmv_dry_run" if dry_run else "E7_pmv")
    argv = rr.e7_argv(exp_dir, data_csv="Data/x_api.csv", split_dir="splits/ASHRAE_2022",
                      pred_dir=tmp_path / "RUN" / "E1_phase2" / "predictions", expected_ml=REAL_E1_INVENTORY,
                      skip_group_ids=skip_group_ids, dry_run=dry_run)
    args = ic.build_parser().parse_args(argv)
    assert args.output_dir == str(exp_dir)
    assert args.expected_ml == REAL_E1_INVENTORY
    assert args.pmv_only is True and args.dry_run is dry_run and args.skip_group_ids is skip_group_ids
    assert args.data_csv == "Data/x_api.csv" and args.splits == "splits/ASHRAE_2022"
    assert args.ml_predictions == str(tmp_path / "RUN" / "E1_phase2" / "predictions")


def test_e7_directory_after_the_open_inventory_list_fails_to_parse(tmp_path):
    """The pre-fix order (directory last, no flag after ``--expected_ml``) fails to parse: the case e7_argv avoids."""
    import indicateurs_classiques as ic
    legacy = ["--pmv_only", "--expected_ml", *REAL_E1_INVENTORY, str(tmp_path / "E7_pmv")]
    with pytest.raises(SystemExit) as exc:
        ic.build_parser().parse_args(legacy)
    assert exc.value.code == 2


def test_real_e7_passes_the_indices_driver_the_output_directory_and_the_e1_inventory(tmp_path, synth, monkeypatch):
    """run_e7 with dry_run=False builds the argv the indices driver receives, parsed here by the real parser
    (the driver itself is stopped): before the fix, the E7 directory was read as a 14th MODEL:SEED and parsing
    exited with code 2.
    The real-run guards and the E1 source validation are stubbed (tested on their own) to reach the call;
    no model is fitted, so the test also runs with the GPU hidden."""
    import data_groups
    import indicateurs_classiques as ic
    orig_prepare = rr._prepare
    e1_dir = tmp_path / "E1_phase2"
    (e1_dir / "predictions").mkdir(parents=True)
    recorded = {}
    for name in ("E1__ASHRAE_2022__TSV_3p__RF__seed42__archived_80_20.csv.gz",
                 "E1__ASHRAE_2022__TSV_3p__XGB__seed42__archived_80_20.csv.gz"):
        f = e1_dir / "predictions" / name
        f.write_bytes(gzip.compress(b"row,pred\n0,1\n"))
        recorded[f"predictions/{name}"] = file_sha256(f)
    e1_man = {"experiment": "E1", "dry_run": False, "outputs_sha256": recorded, "ml_inventory": REAL_E1_INVENTORY}
    (e1_dir / "run_manifest.json").write_text(json.dumps(e1_man))

    def prepare(run_dir, experiment, config, dry_run, **kw):
        pre = orig_prepare(run_dir, experiment, config, True, **kw)
        return dataclasses.replace(pre, exp_dir=rr.experiment_dir(run_dir, experiment, dry_run))

    def load(e1_dir, *, dry_run, **kw):
        assert dry_run is False
        return copy.deepcopy(e1_man)

    raw_csv, raw_xlsx = tmp_path / "raw.csv", tmp_path / "raw_metadata.xlsx"
    raw_csv.write_text("a\n1\n")
    raw_xlsx.write_bytes(b"x")
    seen = {}

    class _Stop(Exception):
        pass

    def c4_main(argv, **kw):
        seen["args"] = ic.build_parser().parse_args(argv)
        seen["e1_manifest"] = kw["e1_manifest"]
        raise _Stop

    monkeypatch.setattr(rr, "_prepare", prepare)
    monkeypatch.setattr(rr, "load_e1_source", load)
    monkeypatch.setattr(data_groups, "default_paths", lambda config: {"raw_csv": raw_csv,
                                                                      "raw_metadata_xlsx": raw_xlsx})
    monkeypatch.setattr(ic, "main", c4_main)
    with pytest.raises(_Stop):
        rr.run_e7(run_dir=tmp_path / "run", e1_dir=e1_dir, data_csv=synth["csv"],
                  split_dir=synth["split_dir"], dry_run=False)
    args = seen["args"]
    assert args.output_dir == str(tmp_path / "run" / "E7_pmv")
    assert args.expected_ml == REAL_E1_INVENTORY
    assert args.ml_predictions == str(e1_dir / "predictions")
    assert args.dry_run is False and args.skip_group_ids is False and args.pmv_only is True
    assert seen["e1_manifest"]["dry_run"] is False
    assert not (tmp_path / "run" / "E7_pmv").exists()


# ================================================================ E9b: the paper's NN analysis (pre-revision code)
def test_e9b_runner_writes_a_manifest_and_never_overwrites(tmp_path, synth, monkeypatch):
    man = rr.run_e9b(run_dir=tmp_path, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True,
                     command=["pytest", "e9b"])
    exp = tmp_path / "E9b_nn_cases_dry_run"
    assert man["experiment"] == "E9b_dry_run" and man["features_key"] == "features_12" and man["population"] == "full"
    assert rr.E9B_FEATURES_KEY == "features_12" and "Euclidean" in man["primary_definition"]
    assert man["case_c_tolerances"] == {"Tair": 1.0, "Tout": 1.0, "RH": 5.0, "vel": 0.3, "clo": 0.3, "Met": 0.2}
    assert man["case_b_round_decimals"] == 3 and man["source"] == rr._nn_cases().SOURCE
    assert set(man["inputs_sha256"]) == {"cohort_csv"}
    assert man["outputs_sha256"]["E9b_nn_cases_dry_run_summary.csv"] == file_sha256(
        exp / "E9b_nn_cases_dry_run_summary.csv")
    assert all(n.startswith("E9b_nn_cases_dry_run_") for n in man["outputs_sha256"])
    summary = pd.read_csv(exp / "E9b_nn_cases_dry_run_summary.csv")
    assert (summary.run_mode == "dry_run").all() and summary.target.tolist() == ["TSV_3p"]
    assert {"case_b_chance_cluster_agreement", "case_c_chance_pair_disagreement", "primary_disagreement",
            "primary_chance_disagreement"} <= set(summary.columns) and (summary.n_features == 12).all()
    index = json.loads((tmp_path / "manifest.json").read_text())["experiments"]
    assert index["E9b_dry_run"]["status"] == "complete"
    with pytest.raises(FileExistsError, match="not empty"):
        rr.run_e9b(run_dir=tmp_path, data_csv=synth["csv"], dry_run=True)
    # a real E9b: every target, pinned cohort file, F12 (the paper's NN feature set); refused before any write
    _as_real_run(monkeypatch)
    with pytest.raises(rr.RunnerError, match="every target"):
        rr.run_e9b(run_dir=tmp_path / "r", data_csv=synth["csv"], targets=["TSV_3p"])
    with pytest.raises(rr.RunnerError, match="pinned"):
        rr.run_e9b(run_dir=tmp_path / "r", data_csv=synth["csv"])
    import data_groups
    monkeypatch.setitem(data_groups.PINNED_SHA256, "canonical_csv", file_sha256(synth["csv"]))
    cfg = copy.deepcopy(load_config(rr.CONFIG_PATH))
    cfg["data"]["features_12"] = ["Tair"]
    with pytest.raises(rr.RunnerError, match="F12"):
        rr.run_e9b(run_dir=tmp_path / "r", data_csv=synth["csv"], config=cfg)
    assert not (tmp_path / "r").exists()


def test_e9b_consumes_the_configuration_its_manifest_records(tmp_path, synth, monkeypatch):
    nc = rr._nn_cases()
    cfg = copy.deepcopy(load_config(rr.CONFIG_PATH))
    cfg["data"]["features_12"] = ["Tair", "Sexe"]
    man = rr.run_e9b(run_dir=tmp_path / "a", data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True, config=cfg)
    summary = pd.read_csv(tmp_path / "a" / "E9b_nn_cases_dry_run" / "E9b_nn_cases_dry_run_summary.csv")
    assert (summary.n_features == 2).all() and man["config_used_sha256"] == rr._json_sha256(cfg)
    orig = nc.run
    monkeypatch.setattr(nc, "run", lambda *a, **k: orig(*a, **{**k, "config": load_config(rr.CONFIG_PATH)}))
    with pytest.raises(rr.RunnerError, match="another configuration"):
        rr.run_e9b(run_dir=tmp_path / "b", data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True, config=cfg)
    _no_completed_manifest(tmp_path / "b" / "E9b_nn_cases_dry_run", tmp_path / "b")


def test_e9b_subcommand_is_wired(tmp_path):
    import run_revision
    a = run_revision.parse_args(["e9b", "--run_dir", str(tmp_path), "--dry_run", "--targets", "TSV_3p"])
    assert a.command == "e9b" and a.dry_run and a.targets == ["TSV_3p"]
    a = run_revision.parse_args(["e9b", "--run_dir", str(tmp_path)])
    assert a.targets == list(rs.TARGETS) and not a.dry_run
    assert rr.EXPERIMENT_DIRS["E9b"] == "E9b_nn_cases" and "E9" not in rr.EXPERIMENT_DIRS


# ================================================================ E6 driver
def _fake_transfert(monkeypatch, seen, record=None, after=None):
    import transfert

    def main(argv, config=None, e1_manifest=None):
        import phase3_runner
        seen["argv"] = list(argv)
        seen["config"] = config
        seen["e1_manifest"] = e1_manifest
        a = transfert.parse_args(argv)
        dry = a.dry_run_splits > 0
        Path(a.splits).mkdir(parents=True, exist_ok=True)              # one serialised split file
        split = Path(a.splits) / "Hostein_TSV_3p_20-80.json"
        split.write_text(json.dumps({"splits": []}))
        french = {n: rr.REPO_DIR / v if not Path(v).is_absolute() else Path(v)
                  for n, v in transfert.COHORT_FILES.items() if n in a.cohorts}
        rec = {"experiment": "E6_dry_run" if dry else "E6", "dry_run": dry, "cohorts": a.cohorts,
               "n_splits_evaluated": a.dry_run_splits or 20, "strategies": a.strategies, "regimes": a.regimes,
               "targets": a.targets, "jobs": [{"key": "Hostein|TSV_3p|20-80|r00|FTT_head", "kind": "gpu"}],
               "config_sha256": phase3_runner.config_sha256(config),
               "split_files_sha256": {split.name: file_sha256(split)},
               "model_files_sha256": {n: file_sha256(Path(a.models_dir) / n) for n in rr.E6_MODEL_FILES
                                      if (Path(a.models_dir) / n).exists()},
               "cohort_files_sha256": {n: file_sha256(f) for n, f in french.items() if f.is_file()},
               "features": list(config["data"]["features"]),
               # the ASHRAE split bytes load_source_rows parses
               "source_files_sha256": {
                   "cohort_csv": file_sha256(a.ashrae_csv),
                   **{f"split:{t}": file_sha256(Path(a.ashrae_split_dir) / f"{t}_split_indices.json")
                      for t in a.targets},
                   **{f"e1:{t}/splits/{t}_split_indices.json":
                      file_sha256(Path(a.models_dir) / t / "splits" / f"{t}_split_indices.json") for t in a.targets}}}
        if after is not None:
            after(a)
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.output_dir) / "e6_run_record.json").write_text(json.dumps(rec))
        (Path(a.output_dir) / "category_coverage_ASHRAE_2022.json").write_text("[]")
        return rec if record is None else record
    monkeypatch.setattr(transfert, "main", main)


def test_e6_driver_owns_paths_and_dry_run_state(tmp_path, synth, e1_trees, monkeypatch):
    """Typed arguments, one dry-run state for directory, splits, inner runner and manifest."""
    import transfert
    run, _ = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    seen = {}
    _fake_transfert(monkeypatch, seen)
    man = rr.run_e6(run_dir=tmp_path, e1_dir=e1_dir, regimes=["20-80", "80-20"], strategies=["zero_shot", "local"],
                    ashrae_csv=synth["csv"], ashrae_split_dir=synth["split_dir"], dry_run=True, dry_run_splits=2,
                    gpu_concurrency=1)
    exp = tmp_path / "E6_phase3_dry_run"
    a = transfert.parse_args(seen["argv"])
    assert a.output_dir == str(exp) and a.models_dir == str(e1_dir) and a.splits == str(exp / "splits")
    assert a.dry_run_splits == 2 and a.gpu_concurrency == 1 and a.ashrae_csv == str(synth["csv"])
    assert a.regimes == ["20-80", "80-20"] and a.strategies == ["zero_shot", "local"]
    assert man["experiment"] == "E6_dry_run" and man["dry_run"] and man["transfert_argv"] == seen["argv"]
    assert seen["e1_manifest"]["experiment"] == "E1_dry_run"          # the validated E1 manifest is passed on
    assert seen["config"] is not None and man["config_used_sha256"] == rr._json_sha256(seen["config"])
    assert man["jobs"] == [{"key": "Hostein|TSV_3p|20-80|r00|FTT_head", "kind": "gpu"}]
    assert {"Moujalled", "Hostein", "e1:run_manifest.json", "e1:best_rf_by_target.joblib"} <= set(man["inputs_sha256"])
    assert man["scheduler"]["gpu_concurrency"] == 1
    assert "E6_dry_run" in json.loads((tmp_path / "manifest.json").read_text())["experiments"]
    # an inner run in another dry-run state than the wrapper's is refused
    _fake_transfert(monkeypatch, seen, record={"experiment": "E6", "dry_run": False})
    with pytest.raises(rr.RunnerError, match="as requested"):
        rr.run_e6(run_dir=tmp_path / "b", e1_dir=e1_dir, ashrae_csv=synth["csv"],
                  ashrae_split_dir=synth["split_dir"], dry_run=True)
    # a real E6 refuses the dry-run E1 models before calling transfert
    _as_real_run(monkeypatch)
    seen.clear()
    with pytest.raises(rr.RunnerError, match="dry run"):
        rr.run_e6(run_dir=tmp_path / "c", e1_dir=e1_dir, splits_dir=tmp_path / "s3", ashrae_csv=synth["csv"],
                  ashrae_split_dir=synth["split_dir"], dry_run=False)
    assert "argv" not in seen


def test_e6_dry_run_fits_with_the_recorded_configuration(tmp_path, synth, e1_trees, config, monkeypatch):
    """Unmocked transfert.main: the RF/XGB fits use the
    configuration that the manifest hashes, and the ASHRAE coverage log is written and hashed."""
    import transfert
    from ftt_adaptation import load_budget
    load_budget(config)                      # the dry configuration keeps the FTT adaptation budget
    run, _ = e1_trees
    built = []
    orig = transfert.make_rf_xgb_protos

    def spy(*a, **kw):
        protos = orig(*a, **kw)
        built.append(protos)
        return protos
    monkeypatch.setattr(transfert, "make_rf_xgb_protos", spy)
    man = rr.run_e6(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", strategies=["local"],
                    cohorts=["Hostein"], targets=["TSV_3p"], ashrae_csv=synth["csv"],
                    ashrae_split_dir=synth["split_dir"], dry_run=True, dry_run_splits=1, gpu_concurrency=1,
                    config=config)
    assert len(built) == 1
    rf, xgb = built[0]
    hp = config["classical_hparams"]
    assert rf.named_steps["clf"].n_estimators == hp["RandomForest"]["n_estimators"] == 10
    assert rf.named_steps["clf"].max_depth == hp["RandomForest"]["max_depth"]
    est = xgb.named_steps["clf"].estimator
    assert est.get_params()["n_estimators"] == hp["XGBoost"]["n_estimators"] == 10
    assert est.get_params()["max_depth"] == hp["XGBoost"]["max_depth"]
    exp = tmp_path / "E6_phase3_dry_run"
    record = json.loads((exp / "e6_run_record.json").read_text())
    assert record["config_sha256"] == man["config_used_sha256"] == rr._json_sha256(config)
    assert len(man["jobs"]) == 2                               # RF/XGB local
    cov = exp / "category_coverage_ASHRAE_2022.json"
    assert isinstance(json.loads(cov.read_text()), list)
    assert man["outputs_sha256"]["category_coverage_ASHRAE_2022.json"] == file_sha256(cov)
    assert "category_coverage_Hostein.json" in man["outputs_sha256"]


def test_e6_wrapper_refuses_a_record_from_another_configuration(tmp_path, e1_trees, synth, config, monkeypatch):
    seen = {}
    _fake_transfert(monkeypatch, seen)
    import transfert
    inner = transfert.main
    monkeypatch.setattr(transfert, "main", lambda argv, config=None, e1_manifest=None:
                        inner(argv, config=rr.load_config(rr.CONFIG_PATH), e1_manifest=e1_manifest))
    run, _ = e1_trees
    with pytest.raises(rr.RunnerError, match="another configuration"):
        rr.run_e6(run_dir=tmp_path, e1_dir=run / "E1_phase2_dry_run", ashrae_csv=synth["csv"],
                  ashrae_split_dir=synth["split_dir"], dry_run=True, config=config)
    assert not (tmp_path / "E6_phase3_dry_run" / "run_manifest.json").exists()


def test_e6_cli_has_no_pass_through(tmp_path):
    import run_revision
    import transfert
    assert set(run_revision.E6_COHORTS) == set(transfert.COHORT_FILES)
    for extra in (["--output_dir", "/tmp/other"], ["--models_dir", "/tmp/m"], ["--", "--dry_run_splits", "1"]):
        with pytest.raises(SystemExit):
            run_revision.parse_args(["e6", "--run_dir", str(tmp_path), *extra])


def test_standalone_transfert_refuses_a_real_e6(tmp_path, synth, e1_trees, monkeypatch):
    """Called directly, transfert.py runs dry runs only; a real E6
    needs the validated E1 manifest of run_revision.py e6 (real E1, recorded model files). Refused
    before any data is loaded."""
    import transfert
    from models_deep import RevisionNotImplementedError
    monkeypatch.setattr(transfert, "load_external_cohort",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    run, e1_man = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    argv = ["--models_dir", str(e1_dir), "--splits", str(tmp_path / "s3"), "--output_dir", str(tmp_path / "out"),
            "--ashrae_csv", str(synth["csv"]), "--ashrae_split_dir", str(synth["split_dir"])]
    with pytest.raises(RevisionNotImplementedError, match="run_revision.py e6"):
        transfert.main(argv)                                         # no manifest (the documented CLI)
    with pytest.raises(RevisionNotImplementedError, match="real E1"):
        transfert.main(argv, e1_manifest=e1_man)                     # dry-run E1 manifest
    real = {**e1_man, "experiment": "E1", "dry_run": False}
    with pytest.raises(RevisionNotImplementedError, match="recorded by the E1 manifest"):
        transfert.main(argv, e1_manifest={**real, "outputs_sha256": {}})   # model files not recorded
    assert not (tmp_path / "out").exists() and not (tmp_path / "s3").exists()
    transfert.check_real_e6_source(e1_dir, real)                     # recorded files with their SHA-256: accepted


def test_real_e6_reads_the_french_cohort_files_of_the_protocol(tmp_path, monkeypatch):
    """Full SHA-256 of the French cohort files (prefixes e13b6be4 / 791819bf) checked for a real E6."""
    import transfert
    assert transfert.COHORT_SHA256["Moujalled"].startswith("e13b6be4")
    assert transfert.COHORT_SHA256["Hostein"].startswith("791819bf")
    monkeypatch.chdir(rr.REPO_DIR)
    if all(Path(f).is_file() for f in transfert.COHORT_FILES.values()):
        transfert.check_cohort_files(list(transfert.COHORT_FILES))
    other = tmp_path / "Hostein_api.csv"
    other.write_text("Tair\n21.0\n")
    with pytest.raises(ValueError, match="frozen cohort file"):
        transfert.check_cohort_files(["Hostein"], files={"Hostein": str(other)})


# ================================================================ CLI
def test_cli_dispatches_dry_runs_inside_the_run_directory(tmp_path, synth, monkeypatch, xgb_fits):
    """xgb_fits: the e2 dry run fits XGBoost (skipped when the CUDA build cannot fit with the GPU hidden)."""
    import run_revision

    def _ctx(config):
        return rr.AshraeContext(data=synth["ctx"].data, ids=synth["ctx"].ids, p26=synth["ctx"].p26,
                                csv_path=synth["csv"], input_files={"cohort_csv": str(synth["csv"])})
    monkeypatch.setattr(rr, "load_ashrae_context", lambda config, csv_path=None: _ctx(config))
    splits_before = sorted(p.name for p in (REPO_DIR / "splits").rglob("*"))
    man = run_revision.main(["e1", "--run_dir", str(tmp_path), "--dry_run", "--targets", "TSV_3p",
                             "--families", "RF", "--split_dir", str(synth["split_dir"]), "--gpu_concurrency", "1"])
    assert man["experiment"] == "E1_dry_run" and man["command"][1:3] == ["run_revision.py", "e1"]
    assert (tmp_path / "E1_phase2_dry_run" / "folds" / "TSV_3p_folds.json").exists()     # never splits/ in a dry run
    assert sorted(p.name for p in (REPO_DIR / "splits").rglob("*")) == splits_before
    man2 = run_revision.main(["e2", "--run_dir", str(tmp_path), "--dry_run", "--targets", "TSV_3p",
                              "--families", "XGB", "--gpu_concurrency", "1"])
    assert man2["experiment"] == "E2_dry_run"
    index = json.loads((tmp_path / "manifest.json").read_text())
    assert set(index["experiments"]) == {"E1_dry_run", "E2_dry_run"}
    with pytest.raises(SystemExit):
        run_revision.main(["e1", "--run_dir", str(tmp_path), "--families", "GBM"])


def test_cli_dispatches_every_subcommand(tmp_path, synth, monkeypatch):
    """Arguments reaching the library for e4, e6 and e7 (library calls mocked); removed subcommands refused."""
    import run_revision
    seen = {}
    monkeypatch.setattr(rr, "load_ashrae_context", lambda config, csv_path=None: synth["ctx"])
    monkeypatch.setattr(run_revision.phase3_runner, "git_checkout", lambda: {"linked_worktree": True})
    for name in ("run_ablation", "run_e6", "run_e7"):
        monkeypatch.setattr(rr, name, lambda *a, _n=name, **kw: seen.setdefault(_n, []).append(kw) or kw)
    run = str(tmp_path)
    kw = run_revision.main(["e4", "--run_dir", run, "--dry_run", "--seeds", "42", "--targets", "TSV_3p"])
    assert kw["experiment"] == "E4" and kw["e1_dir"] == tmp_path / "E1_phase2_dry_run" and kw["seeds"] == [42]
    assert kw["targets"] == ["TSV_3p"] and kw["dry_run"] and kw["gpu_concurrency"] is None
    kw = run_revision.main(["e4", "--run_dir", run, "--e1_dir", "/x/E1", "--gpu_concurrency", "1"])
    assert kw["experiment"] == "E4" and kw["e1_dir"] == "/x/E1" and kw["gpu_concurrency"] == 1 and not kw["dry_run"]
    assert kw["split_dir"] == "splits/ASHRAE_2022" and kw["families"] == ["RF", "XGB"]
    kw = run_revision.main(["e6", "--run_dir", run, "--regimes", "20-80", "80-20"])
    assert kw["splits_dir"] == "splits/phase3" and kw["regimes"] == ["20-80", "80-20"] and not kw["dry_run"]
    assert kw["e1_dir"] == tmp_path / "E1_phase2"
    kw = run_revision.main(["e6", "--run_dir", run, "--dry_run", "--dry_run_splits", "3"])
    assert kw["splits_dir"] is None and kw["dry_run"] and kw["dry_run_splits"] == 3
    assert kw["e1_dir"] == tmp_path / "E1_phase2_dry_run"
    kw = run_revision.main(["e7", "--run_dir", run, "--dry_run", "--skip_group_ids"])
    assert kw["skip_group_ids"] and kw["dry_run"] and kw["e1_dir"] == tmp_path / "E1_phase2_dry_run"
    assert kw["expected_ml"] is None
    for removed in ("pilot", "e3", "e5", "e9"):                          # not part of the release
        with pytest.raises(SystemExit):
            run_revision.main([removed, "--run_dir", run, "--dry_run"])


def test_default_gpu_concurrency_is_one_gpu_job_at_a_time():
    """The unqualified documented E1 command runs one GPU job at a time."""
    from compute import load_compute
    compute = load_compute(load_config(rr.CONFIG_PATH))
    assert compute["gpu_concurrency"] == 1
    rec = rr.scheduler_record(compute, None)
    assert rec["gpu_concurrency"] == 1 and rec["gpu_concurrency_source"] == "config"


def test_real_runs_refuse_gpu_concurrency_above_config(tmp_path, synth, config, compute, monkeypatch):
    """One GPU job at a time: a real run cannot override compute.gpu_concurrency upwards."""
    one = {**compute, "gpu_concurrency": 1}
    rr.check_gpu_concurrency(one, None, dry_run=False)
    rr.check_gpu_concurrency(one, 1, dry_run=False)
    rr.check_gpu_concurrency(one, 2, dry_run=True)
    rr.check_gpu_concurrency({**compute, "gpu_concurrency": 2}, 2, dry_run=False)
    with pytest.raises(rr.RunnerError, match="one GPU job at a time"):
        rr.check_gpu_concurrency(one, 2, dry_run=False)
    with pytest.raises(rr.RunnerError, match=">= 1"):
        rr.check_gpu_concurrency(one, 0, dry_run=True)
    _as_real_run(monkeypatch)
    with pytest.raises(rr.RunnerError, match="one GPU job at a time"):
        rr.run_e2(synth["ctx"], run_dir=tmp_path, config=load_config(rr.CONFIG_PATH), gpu_concurrency=2,
                  dry_run=False)
    import phase3_runner
    with pytest.raises(ValueError, match="at most"):
        phase3_runner.run_e6(cohorts={}, source_rows={}, models={}, make_protos=None, output_dir=tmp_path / "o",
                             splits_dir=tmp_path / "s", config=load_config(rr.CONFIG_PATH), gpu_concurrency=2)


def test_unpinned_packages_break_the_lock_except_installer_tooling(tmp_path):
    """A distribution installed but absent from the lock fails a real run (pip/setuptools/wheel aside)."""
    env = rr.environment_record()
    pins = rr.lock_pins()
    assert env["unexpected_unpinned"] == [] or not env["matches_lock"]
    assert all(u.split("==")[0] not in pins for u in env["unpinned_installed"])
    lock = tmp_path / "lock"
    lock.write_text("\n".join(f"{n}=={v}" for n, v in pins.items() if n != "numpy") + "\n")
    env2 = rr.environment_record(lock)
    assert env2["matches_lock"] is False and any(u.startswith("numpy==") for u in env2["unexpected_unpinned"])
    with pytest.raises(rr.RunnerError, match="not in the lock"):
        rr.check_locked_environment(env2, dry_run=False)


def test_e1_source_must_match_the_run_index(tmp_path, e1_trees, monkeypatch):
    """The E1 manifest read by a later experiment is the one indexed by its run directory."""
    import shutil
    run, _ = e1_trees
    dst = tmp_path / "run"
    shutil.copytree(run, dst)
    e1 = dst / "E1_phase2_dry_run"
    rr.load_e1_source(e1, dry_run=True)
    man = json.loads((e1 / "run_manifest.json").read_text())
    man["seeds"] = {"tampered": True}
    (e1 / "run_manifest.json").write_text(json.dumps(man, indent=1))
    with pytest.raises(rr.RunnerError, match="run index"):
        rr.load_e1_source(e1, dry_run=True)
    # a real run needs an indexed E1 (a copied E1 outside its run directory is refused)
    real = tmp_path / "real" / "E1_phase2"
    shutil.copytree(run / "E1_phase2_dry_run", real)
    m = json.loads((real / "run_manifest.json").read_text())
    m.update(dry_run=False, experiment="E1")
    (real / "run_manifest.json").write_text(json.dumps(m))
    with pytest.raises(rr.RunnerError, match="not recorded in the run index"):
        rr.load_e1_source(real, dry_run=False)


# ================================================================ startup binding
def _stateful_provenance(monkeypatch, commit="a" * 40, dirty=False):
    """``run_provenance`` reading a mutable state: an event during the run changes what the end check sees."""
    state = {"git_commit": commit, "git_tracked_changes": dirty}
    monkeypatch.setattr(rr, "run_provenance", lambda: {**state, "python": "3.x", "packages": {"numpy": "1"}})
    return state


def _no_completed_manifest(exp_dir: Path, run_dir: Path) -> None:
    assert not (exp_dir / "run_manifest.json").exists()
    index = run_dir / "manifest.json"
    assert not index.exists() or not json.loads(index.read_text())["experiments"]


def _drift_events(state, cfg, cfg_file, csv):
    return {
        "commit_a_to_b_clean": lambda: state.update(git_commit="b" * 40),
        "dirty_tracked_files": lambda: state.update(git_tracked_changes=True),
        "configuration_used": lambda: cfg["compute"].update(cpu_jobs=99),
        "config_yaml_edited": lambda: cfg_file.write_text(cfg_file.read_text() + "\n# edited\n"),
        "input_replaced": lambda: csv.write_bytes(csv.read_bytes().replace(b"2020-01-01", b"2020-01-02", 1)),
    }


@pytest.mark.parametrize("event", ["commit_a_to_b_clean", "dirty_tracked_files", "configuration_used",
                                   "config_yaml_edited", "input_replaced"])
def test_drift_during_a_run_leaves_no_completed_manifest(tmp_path, synth, monkeypatch, event):
    """A commit (A -> B, both clean), tracked-file, configuration or input change between the start
    and the end of an experiment aborts it before its manifest and its run-index entry are written."""
    import shutil
    state = _stateful_provenance(monkeypatch)
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    cfg = copy.deepcopy(load_config(cfg_file))
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    nn = rr._nn_cases()
    orig = nn.run

    def run_then_event(*a, **k):
        out = orig(*a, **k)
        _drift_events(state, cfg, cfg_file, csv)[event]()
        return out

    monkeypatch.setattr(nn, "run", run_then_event)
    run = tmp_path / "run"
    # the configuration change is caught by the E9b consumed-configuration check or the end check
    with pytest.raises(rr.RunnerError, match="changed between its start and its end|another configuration"):
        rr.run_e9b(run_dir=run, data_csv=csv, targets=["TSV_3p"], dry_run=True, config=cfg)
    _no_completed_manifest(run / "E9b_nn_cases_dry_run", run)


def test_no_drift_records_the_startup_provenance(tmp_path, synth, monkeypatch):
    state = _stateful_provenance(monkeypatch)
    man = rr.run_e9b(run_dir=tmp_path, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    assert man["provenance"]["git_commit"] == state["git_commit"]
    assert man["provenance_end_check"]["matches_startup"] is True
    assert man["inputs_sha256"] == {"cohort_csv": file_sha256(synth["csv"])}
    assert "E9b_dry_run" in json.loads((tmp_path / "manifest.json").read_text())["experiments"]


@pytest.mark.parametrize("event", ["commit_a_to_b_clean", "dirty_tracked_files", "configuration_used",
                                   "config_yaml_edited", "input_replaced"])
def test_real_run_finalisation_refuses_drift(tmp_path, monkeypatch, event):
    """The same events for a real run (start checks passed): the end check (clean checkout re-checked, commit,
    configuration and input hashes compared with the startup record) refuses before any manifest write."""
    import shutil
    state = _stateful_provenance(monkeypatch)
    monkeypatch.setattr(rr, "check_locked_environment", lambda env, dry_run: None)
    monkeypatch.setattr(rr, "check_phase1_artefact", lambda dry_run: rr.phase1_provenance())
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    cfg = load_config(cfg_file)
    csv = tmp_path / "input.csv"
    csv.write_text("Date\n2020-01-01\n")
    run = tmp_path / "run"
    pre = rr._prepare(run, "E9b", cfg, False)
    startup = rr.bind_inputs(pre.startup, {"cohort_csv": str(csv)})
    _drift_events(state, cfg, cfg_file, csv)[event]()
    match = "clean checkout" if event == "dirty_tracked_files" else "changed between its start and its end"
    with pytest.raises(rr.RunnerError, match=match):
        rr.write_manifest(pre.exp_dir, run, experiment="E9b", dry_run=False, command=None, started=pre.started,
                          config=cfg, compute=pre.compute, env=pre.env, inputs={"cohort_csv": str(csv)},
                          seeds=None, startup=startup)
    _no_completed_manifest(pre.exp_dir, run)


def test_input_replaced_after_loading_aborts_the_experiment(tmp_path, synth, config, compute, monkeypatch):
    """The cohort CSV is hashed when it is loaded; a replacement before the fits (bind) or during
    them (end check) leaves no completed manifest or run-index entry."""
    import shutil
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    ctx = rr.load_ashrae_context(config, csv_path=csv, ids=synth["ids"])
    assert ctx.input_sha256 == {"cohort_csv": file_sha256(csv)}
    kw = dict(config=config, targets=("TSV_3p",), families=("RF",), compute=compute, dry_run=True)
    original = csv.read_bytes()
    csv.write_bytes(original.replace(b"2020-01-01", b"2020-01-02", 1))     # after loading, before the run
    with pytest.raises(rr.RunnerError, match="changed after it was loaded"):
        rr.run_e2(ctx, run_dir=tmp_path / "r1", **kw)
    _no_completed_manifest(tmp_path / "r1" / "E2_lopo_dry_run", tmp_path / "r1")
    csv.write_bytes(original)
    orig_jobs = rr.run_jobs

    def jobs_then_replace(*a, **k):
        out = orig_jobs(*a, **k)
        csv.write_bytes(original.replace(b"2020-01-01", b"2020-01-02", 1))  # during the fits
        return out

    monkeypatch.setattr(rr, "run_jobs", jobs_then_replace)
    with pytest.raises(rr.RunnerError, match="changed between its start and its end"):
        rr.run_e2(ctx, run_dir=tmp_path / "r2", **kw)
    _no_completed_manifest(tmp_path / "r2" / "E2_lopo_dry_run", tmp_path / "r2")


def test_startup_record_is_immutable_and_inputs_must_be_bound(tmp_path, monkeypatch):
    import dataclasses
    _stateful_provenance(monkeypatch)
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    start = rr.startup_record(rr.run_provenance(), {"k": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        start.config_used_sha256 = "x"
    bound = rr.bind_inputs(start, {"a": str(f)})
    assert start.inputs == () and bound.inputs_sha256 == {"a": file_sha256(f)}
    assert rr.bind_inputs(bound, {"a": str(f)}) == bound                     # same file, same bytes: idempotent
    with pytest.raises(rr.RunnerError, match="changed after it was loaded"):
        rr.bind_inputs(start, {"a": str(f)}, {"a": "0" * 64})
    with pytest.raises(rr.RunnerError, match="not hashed when it was consumed"):
        rr.check_end_state(bound, {"k": 1}, {"a": str(f), "b": str(f)}, dry_run=True)
    f.write_text("x\n2\n")
    with pytest.raises(rr.RunnerError, match="rebound"):
        rr.bind_inputs(bound, {"a": str(f)})
    with pytest.raises(rr.RunnerError, match="input a"):
        rr.check_end_state(bound, {"k": 1}, {"a": str(f)}, dry_run=True)


def test_real_runs_refuse_a_modified_checkout():
    rr.check_clean_checkout({"git_tracked_changes": False}, dry_run=False)
    rr.check_clean_checkout({"git_tracked_changes": True}, dry_run=True)
    for state in (True, None):
        with pytest.raises(rr.RunnerError, match="clean checkout"):
            rr.check_clean_checkout({"git_tracked_changes": state}, dry_run=False)


def test_run_index_keeps_every_entry_under_concurrent_writers(tmp_path):
    """Concurrent read-modify-write of {run_dir}/manifest.json loses no experiment."""
    import threading
    paths = []
    for i in range(8):
        d = tmp_path / f"X{i}"
        d.mkdir()
        (d / "run_manifest.json").write_text(json.dumps({"i": i}))
        paths.append(d / "run_manifest.json")
    threads = [threading.Thread(target=rr.update_run_index, args=(tmp_path, f"X{i}", p)) for i, p in enumerate(paths)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(json.loads((tmp_path / "manifest.json").read_text())["experiments"]) == {f"X{i}" for i in range(8)}


# ================================================================ input inventory
def test_end_check_covers_every_bound_input(tmp_path, monkeypatch):
    """A bound input left out of the recorded inputs (and changed) is drift, as is an unbound recorded input."""
    import shutil
    _stateful_provenance(monkeypatch)
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    cfg = load_config(cfg_file)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("1")
    b.write_text("2")
    startup = rr.bind_inputs(rr.startup_record(rr.run_provenance(), cfg), {"a": str(a), "split": str(b)})
    b.write_text("3")
    with pytest.raises(rr.RunnerError, match="split .* missing from the recorded inputs"):
        rr.check_end_state(startup, cfg, {"a": str(a)}, dry_run=True)
    with pytest.raises(rr.RunnerError, match="input split: start"):
        rr.check_end_state(startup, cfg, {"a": str(a), "split": str(b)}, dry_run=True)
    b.write_text("2")
    assert rr.check_end_state(startup, cfg, {"a": str(a), "split": str(b)}, dry_run=True)["matches_startup"]


def _e6_kw(synth, **kw):
    return dict(ashrae_csv=synth["csv"], ashrae_split_dir=synth["split_dir"], dry_run=True, cohorts=["Hostein"],
                **kw)


def test_e6_binds_the_phase3_split_files_it_consumed(tmp_path, synth, e1_trees, monkeypatch):
    """The serialised Phase 3 split files are bound to the hashes taken before the first fit and
    re-checked at the end; a change after serialisation, after binding, or a model-file record that is not
    the bound file leaves no completed manifest."""
    run, _ = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    seen = {}
    _fake_transfert(monkeypatch, seen)
    man = rr.run_e6(run_dir=tmp_path / "ok", e1_dir=e1_dir, **_e6_kw(synth))
    split = tmp_path / "ok" / "E6_phase3_dry_run" / "splits" / "Hostein_TSV_3p_20-80.json"
    assert man["inputs_sha256"]["phase3_split:Hostein_TSV_3p_20-80.json"] == file_sha256(split)
    assert man["inputs"]["phase3_split:Hostein_TSV_3p_20-80.json"] == str(split)
    # (1) changed after serialisation (hash recorded by phase3_runner), before the wrapper binds it
    _fake_transfert(monkeypatch, seen, after=lambda a: None)
    import transfert
    inner = transfert.main

    def edit_split_after(argv, config=None, e1_manifest=None):
        rec = inner(argv, config=config, e1_manifest=e1_manifest)
        (Path(transfert.parse_args(argv).splits) / "Hostein_TSV_3p_20-80.json").write_text('{"splits": [1]}')
        return rec
    monkeypatch.setattr(transfert, "main", edit_split_after)
    with pytest.raises(rr.RunnerError, match="changed after it was loaded or validated"):
        rr.run_e6(run_dir=tmp_path / "r1", e1_dir=e1_dir, **_e6_kw(synth))
    _no_completed_manifest(tmp_path / "r1" / "E6_phase3_dry_run", tmp_path / "r1")
    # (2) changed after binding, before the manifest
    _fake_transfert(monkeypatch, seen)
    orig_end = rr.check_end_state

    def edit_then_check(startup, *a, **k):
        for role, path, _ in startup.inputs:
            if role.startswith("phase3_split:"):
                Path(path).write_text('{"splits": [2]}')
        return orig_end(startup, *a, **k)
    monkeypatch.setattr(rr, "check_end_state", edit_then_check)
    with pytest.raises(rr.RunnerError, match="input phase3_split:Hostein_TSV_3p_20-80.json: start"):
        rr.run_e6(run_dir=tmp_path / "r2", e1_dir=e1_dir, **_e6_kw(synth))
    _no_completed_manifest(tmp_path / "r2" / "E6_phase3_dry_run", tmp_path / "r2")
    monkeypatch.setattr(rr, "check_end_state", orig_end)
    # (3) the inner run recorded another model file than the bound one
    _fake_transfert(monkeypatch, seen, after=lambda a: None)
    inner = transfert.main

    def other_model(argv, config=None, e1_manifest=None):
        rec = inner(argv, config=config, e1_manifest=e1_manifest)
        rec["model_files_sha256"] = {n: "0" * 64 for n in rec["model_files_sha256"]}
        return rec
    monkeypatch.setattr(transfert, "main", other_model)
    with pytest.raises(rr.RunnerError, match="not the bound inputs"):
        rr.run_e6(run_dir=tmp_path / "r3", e1_dir=e1_dir, **_e6_kw(synth))
    _no_completed_manifest(tmp_path / "r3" / "E6_phase3_dry_run", tmp_path / "r3")


@pytest.mark.parametrize("what", ["predictions", "manifest"])
def test_e1_source_replaced_after_validation_is_not_bound(tmp_path, synth, e1_trees, config, compute,
                                                          monkeypatch, what):
    """An E1 file replaced between its validation against the E1 manifest and its binding is
    refused (the validated hash is bound, not a fresh one); nothing is written."""
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    f42 = rr.e1_prediction_file(e1, "TSV_3p", "RF", 42, True)
    orig = rr.load_e1_source

    def validate_then_replace(*a, **k):
        man = orig(*a, **k)
        if what == "predictions":                       # other probabilities, same rows and labels
            _rewrite_predictions(f42, p_0=lambda c: c * 0 + 0.5)
        else:                                           # same content, other bytes
            (e1 / "run_manifest.json").write_text(json.dumps(json.loads((e1 / "run_manifest.json").read_text()),
                                                             indent=3))
        return man
    monkeypatch.setattr(rr, "load_e1_source", validate_then_replace)
    # predictions: the bytes check_e1_predictions parsed are not the validated ones; manifest: bind
    with pytest.raises(rr.RunnerError, match="changed after it was loaded or validated|validated against E1 as"):
        _ablation(synth, config, compute, tmp_path / "run", e1)
    _no_completed_manifest(tmp_path / "run" / "E4_ema_dry_run", tmp_path / "run")
    assert not (tmp_path / "run" / "E4_ema_dry_run").exists()


def test_e6_e1_model_replaced_after_validation_is_not_bound(tmp_path, synth, e1_trees, monkeypatch):
    run, _ = e1_trees
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    seen = {}
    _fake_transfert(monkeypatch, seen)
    orig = rr.load_e1_source

    def validate_then_replace(*a, **k):
        man = orig(*a, **k)
        (e1 / "best_rf_by_target.joblib").write_bytes((e1 / "best_xgb_by_target.joblib").read_bytes())
        return man
    monkeypatch.setattr(rr, "load_e1_source", validate_then_replace)
    with pytest.raises(rr.RunnerError, match="changed after it was loaded or validated"):
        rr.run_e6(run_dir=tmp_path / "r", e1_dir=e1, **_e6_kw(synth))
    assert "argv" not in seen                          # refused before transfert.main
    _no_completed_manifest(tmp_path / "r" / "E6_phase3_dry_run", tmp_path / "r")


# ================================================================ completion after registration
@pytest.mark.parametrize("failure", ["duplicate", "index_write"])
def test_index_failure_leaves_no_completed_manifest(tmp_path, synth, monkeypatch, failure):
    run = tmp_path / "run"
    if failure == "duplicate":                  # the experiment is already registered in the run index
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"format": rr.RUN_INDEX_FORMAT, "created_utc": "x",
                                                       "phase1": {}, "experiments": {"E9b_dry_run": {}}}))
        exc, match = FileExistsError, "already recorded"
    else:
        def broken(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(rr, "update_run_index", broken)
        exc, match = OSError, "disk full"
    with pytest.raises(exc, match=match):
        rr.run_e9b(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    exp = run / "E9b_nn_cases_dry_run"
    assert exp.is_dir() and not (exp / "run_manifest.json").exists()
    assert not (exp / "run_manifest.json.pending").exists()


def test_registered_manifest_hash_is_the_final_file(tmp_path, synth):
    man = rr.run_e9b(run_dir=tmp_path, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["experiments"][man["experiment"]]
    final = tmp_path / entry["manifest"]
    assert final.name == "run_manifest.json" and entry["manifest_sha256"] == file_sha256(final)
    assert not list((tmp_path / "E9b_nn_cases_dry_run").glob("*.pending"))


# ================================================================ untracked Python files
def test_untracked_python_files_are_in_the_startup_record_and_compared(tmp_path, synth, monkeypatch):
    state = {"files": []}
    monkeypatch.setattr(rr, "untracked_python_files", lambda: list(state["files"]))
    nn = rr._nn_cases()
    orig = nn.run

    def run_then_new_file(*a, **k):
        out = orig(*a, **k)
        state["files"] = ["shadow.py"]
        return out
    monkeypatch.setattr(nn, "run", run_then_new_file)
    with pytest.raises(rr.RunnerError, match="untracked Python files: start \\[\\], end \\['shadow.py'\\]"):
        rr.run_e9b(run_dir=tmp_path / "a", data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    _no_completed_manifest(tmp_path / "a" / "E9b_nn_cases_dry_run", tmp_path / "a")
    monkeypatch.setattr(nn, "run", orig)
    man = rr.run_e9b(run_dir=tmp_path / "b", data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)       # a dry run records them
    assert man["untracked_python_files"] == ["shadow.py"]
    _as_real_run(monkeypatch)                                   # a real run refuses them at start
    cfg = load_config(rr.CONFIG_PATH)
    for files in (["shadow.py"], None):
        state["files"] = files
        monkeypatch.setattr(rr, "untracked_python_files", lambda: None if state["files"] is None
                            else list(state["files"]))
        with pytest.raises(rr.RunnerError, match="untracked Python files"):
            rr._prepare(tmp_path / "r", "E9b", cfg, False)
    assert not (tmp_path / "r").exists()


# ================================================================ bytes consumed = bytes bound
def _copy_split_dir(synth, dest: Path) -> Path:
    import shutil
    shutil.copytree(synth["split_dir"], dest)
    return dest


def _same_content_other_bytes(path: Path) -> None:
    """Rewrite a JSON file with the same parsed content and other bytes (a swap no parser sees)."""
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1))


def test_e1_split_replaced_after_it_was_parsed_leaves_no_manifest(tmp_path, synth, config, monkeypatch):
    """An external split file replaced right after E1 parsed it (before binding) is refused: the
    bound hash is that of the bytes parsed; no manifest, no run-index entry."""
    split_dir = _copy_split_dir(synth, tmp_path / "splits")
    path = split_dir / "TSV_3p_split_indices.json"
    orig = rr._split

    def read_then_replace(*a, **k):
        out = orig(*a, **k)
        payload = json.loads(path.read_text())
        payload["train_idx"], payload["test_idx"] = payload["test_idx"], payload["train_idx"]
        path.write_text(json.dumps(payload))
        return out
    monkeypatch.setattr(rr, "_split", read_then_replace)
    compute = {**rr.load_compute(config), "xgb_device": "cpu", "torch_device": "cpu", "cpu_jobs": 1}
    with pytest.raises(rr.RunnerError, match="split:TSV_3p .* changed after it was loaded"):
        rr.run_e1(synth["ctx"], run_dir=tmp_path / "run", config=config, split_dir=split_dir, targets=["TSV_3p"],
                  families=["RF"], compute=compute, dry_run=True)
    _no_completed_manifest(tmp_path / "run" / "E1_phase2_dry_run", tmp_path / "run")


def test_e1_external_folds_replaced_after_they_were_read_leaves_no_manifest(tmp_path, synth, config, monkeypatch):
    """Fold reader: an existing external folds file replaced after ensure_folds read it is refused."""
    split_dir = _copy_split_dir(synth, tmp_path / "splits")
    folds_dir = tmp_path / "folds"
    tr, _ = revision_folds.load_archived_split(split_dir, "TSV_3p", synth["df"]["TSV_3p"])
    revision_folds.ensure_folds(folds_dir, rr.DATASET, "TSV_3p", tr, synth["df"]["TSV_3p"].iloc[tr])
    orig = revision_folds.ensure_folds

    def read_then_replace(*a, **k):
        out = orig(*a, **k)
        _same_content_other_bytes(revision_folds.folds_path(folds_dir, "TSV_3p"))
        return out
    monkeypatch.setattr(revision_folds, "ensure_folds", read_then_replace)
    compute = {**rr.load_compute(config), "xgb_device": "cpu", "torch_device": "cpu", "cpu_jobs": 1}
    with pytest.raises(rr.RunnerError, match="folds:TSV_3p .* changed after it was loaded"):
        rr.run_e1(synth["ctx"], run_dir=tmp_path / "run", config=config, split_dir=split_dir, folds_dir=folds_dir,
                  targets=["TSV_3p"], families=["RF"], compute=compute, dry_run=True)
    _no_completed_manifest(tmp_path / "run" / "E1_phase2_dry_run", tmp_path / "run")


def test_readers_report_the_hash_of_the_bytes_they_parse(tmp_path, synth):
    import indicateurs_classiques as ic
    import phase3_runner
    path = synth["split_dir"] / "TSV_3p_split_indices.json"
    seen = {}
    revision_folds.load_archived_split(synth["split_dir"], "TSV_3p", synth["df"]["TSV_3p"], consumed_sha256=seen)
    assert seen == {str(path): file_sha256(path)}
    seen = {}
    ic.load_split(synth["split_dir"], "TSV_3p", seen)
    assert seen == {"TSV_3p": file_sha256(path)}
    seen = {}
    phase3_runner.load_source_rows(synth["csv"], synth["split_dir"], ["TSV_3p"], ["Tair", "RH"],
                                   consumed_sha256=seen)
    assert seen == {"split:TSV_3p": file_sha256(path), "cohort_csv": file_sha256(synth["csv"])}
    folds_dir = tmp_path / "folds"
    tr, _ = revision_folds.redraw_split(synth["df"]["TSV_3p"])
    for _ in range(2):                                   # written, then read back
        seen = {}
        revision_folds.ensure_folds(folds_dir, rr.DATASET, "TSV_3p", tr, synth["df"]["TSV_3p"].iloc[tr],
                                    consumed_sha256=seen)
        f = revision_folds.folds_path(folds_dir, "TSV_3p")
        assert seen == {str(f): file_sha256(f)}


def _swap_during_parse(monkeypatch, module, name, path: Path):
    """Around ``module.name``: other bytes (same content) while it parses ``path``, original bytes restored after
    (the end check alone cannot see such a swap)."""
    orig = getattr(module, name)

    def swapped(*a, **k):
        before = path.read_bytes()
        _same_content_other_bytes(path)
        try:
            return orig(*a, **k)
        finally:
            path.write_bytes(before)
    monkeypatch.setattr(module, name, swapped)


def test_e7_split_swapped_while_the_indices_driver_parses_it_leaves_no_manifest(tmp_path, synth, e1_trees,
                                                                                 monkeypatch):
    import indicateurs_classiques as ic
    run, _ = e1_trees
    split_dir = _copy_split_dir(synth, tmp_path / "splits")
    _swap_during_parse(monkeypatch, ic, "load_split", split_dir / "TSV_3p_split_indices.json")
    with pytest.raises(rr.RunnerError, match="parsed inputs other than the bound ones"):
        rr.run_e7(run_dir=tmp_path / "run", e1_dir=run / "E1_phase2_dry_run", data_csv=synth["csv"],
                  split_dir=split_dir, skip_group_ids=True, dry_run=True)
    _no_completed_manifest(tmp_path / "run" / "E7_pmv_dry_run", tmp_path / "run")


def test_e6_split_swapped_while_load_source_rows_parses_it_leaves_no_manifest(tmp_path, synth, e1_trees, config,
                                                                             monkeypatch):
    run, _ = e1_trees
    split_dir = _copy_split_dir(synth, tmp_path / "splits")
    _swap_during_parse(monkeypatch, revision_folds, "load_archived_split", split_dir / "TSV_3p_split_indices.json")
    with pytest.raises(rr.RunnerError, match="parsed inputs other than the bound ones"):
        rr.run_e6(run_dir=tmp_path / "run", e1_dir=run / "E1_phase2_dry_run", strategies=["local"],
                  cohorts=["Hostein"], targets=["TSV_3p"], ashrae_csv=synth["csv"], ashrae_split_dir=split_dir,
                  dry_run=True, dry_run_splits=1, gpu_concurrency=1, config=config)
    _no_completed_manifest(tmp_path / "run" / "E6_phase3_dry_run", tmp_path / "run")


def test_e6_wrapper_refuses_a_record_without_parsed_split_hashes_or_with_other_features(tmp_path, synth, e1_trees,
                                                                                      config, monkeypatch):
    """The E6 record must list the split bytes parsed, equal to the bound ones,
    and the feature list fitted must be that of the configuration the manifest records."""
    run, _ = e1_trees
    e1_dir = run / "E1_phase2_dry_run"
    cases = {
        "missing": (lambda rec: rec.pop("source_files_sha256"), "does not list the ASHRAE cohort and split files"),
        "other_bytes": (lambda rec: rec["source_files_sha256"].update({"split:TSV_3p": "0" * 64}),
                        "parsed inputs other than the bound ones"),
        "features": (lambda rec: rec.update(features=list(config["data"]["features"])[:-1]),
                     "another feature list"),
    }
    for name, (edit, match) in cases.items():
        seen = {}
        _fake_transfert(monkeypatch, seen)
        import transfert
        inner = transfert.main

        def edited(argv, config=None, e1_manifest=None, _inner=inner, _edit=edit):
            rec = _inner(argv, config=config, e1_manifest=e1_manifest)
            _edit(rec)
            return rec
        monkeypatch.setattr(transfert, "main", edited)
        with pytest.raises(rr.RunnerError, match=match):
            rr.run_e6(run_dir=tmp_path / name, e1_dir=e1_dir, targets=["TSV_3p"], config=config, **_e6_kw(synth))
        _no_completed_manifest(tmp_path / name / "E6_phase3_dry_run", tmp_path / name)


# ================================================================ final rename failure
def test_final_rename_failure_rolls_back_the_index_and_allows_the_documented_retry(tmp_path, synth, monkeypatch):
    """A failed final rename leaves no run_manifest.json, no pending file and no index entry (the
    index is restored under its lock), so the documented recovery (move aside, rerun) works."""
    import shutil
    run = tmp_path / "run"
    orig = rr.os.replace

    def fail_final(src, dst):
        if str(src).endswith("run_manifest.json.pending"):
            raise OSError("injected final rename failure")
        return orig(src, dst)
    monkeypatch.setattr(rr.os, "replace", fail_final)
    with pytest.raises(OSError, match="injected"):
        rr.run_e9b(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    exp = run / "E9b_nn_cases_dry_run"
    assert exp.is_dir()
    assert not (exp / "run_manifest.json").exists() and not (exp / "run_manifest.json.pending").exists()
    assert not (run / "manifest.json").exists()                    # created by this registration: removed
    monkeypatch.setattr(rr.os, "replace", orig)
    shutil.move(str(exp), str(tmp_path / "failed_E9"))             # documented recovery: move aside, rerun
    man = rr.run_e9b(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    entry = json.loads((run / "manifest.json").read_text())["experiments"][man["experiment"]]
    assert entry["manifest_sha256"] == file_sha256(run / entry["manifest"])
    # an existing index is restored byte for byte when a completion fails
    before = (run / "manifest.json").read_bytes()

    def failing_completion():
        raise OSError("injected final rename failure")
    with pytest.raises(OSError, match="injected"):
        rr.update_run_index(run, "E2_dry_run", run / "E2_lopo_dry_run" / "run_manifest.json",
                            manifest_sha256="0" * 64, complete=failing_completion)
    assert (run / "manifest.json").read_bytes() == before


def test_dangling_index_entry_is_superseded_by_the_retry(tmp_path, synth):
    """A process killed between registration and completion leaves a "registered"
    entry whose manifest does not exist; the documented retry replaces it (kept under superseded_unconfirmed);
    the new entry ends "complete"; a completed entry never is replaced."""
    run = tmp_path / "run"
    run.mkdir()
    dangling = {"manifest": "E9b_nn_cases_dry_run/run_manifest.json", "manifest_sha256": "0" * 64, "recorded_utc": "t",
                "status": "registered"}
    (run / "manifest.json").write_text(json.dumps({"format": rr.RUN_INDEX_FORMAT, "created_utc": "x", "phase1": {},
                                                   "experiments": {"E9b_dry_run": dangling}}))
    man = rr.run_e9b(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    index = json.loads((run / "manifest.json").read_text())
    assert index["superseded_unconfirmed"][0]["manifest_sha256"] == "0" * 64
    assert "superseded_dangling" not in index
    entry = index["experiments"][man["experiment"]]
    assert entry["manifest_sha256"] == file_sha256(run / "E9b_nn_cases_dry_run" / "run_manifest.json")
    assert entry["status"] == "complete" and entry["completed_utc"]
    with pytest.raises(FileExistsError, match="already recorded"):   # completed: never overwritten
        rr.update_run_index(run, "E9b_dry_run", run / "E9b_nn_cases_dry_run" / "run_manifest.json", manifest_sha256="1" * 64)


@pytest.mark.parametrize("old_status", ["complete", None])
def test_completed_experiment_moved_aside_is_refused_on_retry(tmp_path, synth, old_status):
    """A completed experiment whose directory is moved aside (or an entry without status, the
    conservative reading) is never superseded: the retry is refused, the index is unchanged, no manifest is
    left in the new directory; the message points to a new run directory."""
    import shutil
    run = tmp_path / "run"
    kw = dict(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    rr.run_e9b(**kw)
    if old_status is None:                                      # legacy entry without status
        index = json.loads((run / "manifest.json").read_text())
        del index["experiments"]["E9b_dry_run"]["status"]
        (run / "manifest.json").write_text(json.dumps(index))
    before = (run / "manifest.json").read_bytes()
    shutil.move(str(run / "E9b_nn_cases_dry_run"), str(tmp_path / "moved_E9"))
    with pytest.raises(FileExistsError, match="never superseded, use a new run directory"):
        rr.run_e9b(**kw)
    assert (run / "manifest.json").read_bytes() == before
    assert not (run / "E9b_nn_cases_dry_run" / "run_manifest.json").exists()
    assert not (run / "E9b_nn_cases_dry_run" / "run_manifest.json.pending").exists()


def test_failed_status_write_reverts_the_rename_and_the_index(tmp_path, synth, monkeypatch):
    """If the "complete" status cannot be written after the rename, the rename is undone and the
    index restored: no run_manifest.json, no pending file, no index entry."""
    run = tmp_path / "run"
    orig = rr.os.replace
    calls = {"index": 0}

    def fail_second_index_write(src, dst):
        if Path(dst).name == "manifest.json":
            calls["index"] += 1
            if calls["index"] == 2:                              # registered, then the status write
                raise OSError("injected status write failure")
        return orig(src, dst)
    monkeypatch.setattr(rr.os, "replace", fail_second_index_write)
    with pytest.raises(OSError, match="injected status"):
        rr.run_e9b(run_dir=run, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    exp = run / "E9b_nn_cases_dry_run"
    assert not (exp / "run_manifest.json").exists() and not (exp / "run_manifest.json.pending").exists()
    assert not (run / "manifest.json").exists()


# ================================================================ consumed inputs
def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _swap_on_read(monkeypatch, path, mutate):
    """Restored substitution around the read of ``path``: the file holds ``mutate(bytes)`` only while
    ``Path.read_bytes`` (the one read of ``config_loader.read_consumed``) reads it, and the original bytes are
    written back right after, so any later hash of the path sees the original file. Returns the list of
    substituted byte strings (one per read)."""
    import pathlib
    target = Path(path).resolve()
    orig = pathlib.Path.read_bytes
    seen = []

    def read_bytes(self):
        if Path(self).resolve() != target:
            return orig(self)
        with open(target, "rb") as fh:
            before = fh.read()
        swapped = mutate(before)
        assert swapped != before
        with open(target, "wb") as fh:
            fh.write(swapped)
        try:
            seen.append(swapped)
            return orig(self)
        finally:
            with open(target, "wb") as fh:
                fh.write(before)
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    return seen


def _no_path_parse(monkeypatch, *paths):
    """Fail if pandas or joblib parse one of ``paths`` from its path (a second read, not the bytes hashed)."""
    import joblib as _joblib
    targets = {Path(p).resolve() for p in paths}

    def guard(fn, name):
        def wrapped(src, *a, **k):
            if isinstance(src, (str, Path)) and Path(src).resolve() in targets:
                raise AssertionError(f"{name} re-read {src} from its path")
            return fn(src, *a, **k)
        return wrapped
    monkeypatch.setattr(pd, "read_csv", guard(pd.read_csv, "pandas.read_csv"))
    monkeypatch.setattr(pd, "read_excel", guard(pd.read_excel, "pandas.read_excel"))
    monkeypatch.setattr(_joblib, "load", guard(_joblib.load, "joblib.load"))


def _plus_newline(b: bytes) -> bytes:
    return b + b"\n"                                        # same parsed CSV, other bytes


def _regzip(b: bytes) -> bytes:
    return gzip.compress(gzip.decompress(b), mtime=12345)   # same predictions, other bytes


def test_read_consumed_hashes_the_bytes_it_returns_and_refuses_two_different_reads(tmp_path):
    from config_loader import ConsumedInputError, read_consumed
    f = tmp_path / "x.csv"
    f.write_bytes(b"a\n1\n")
    seen = {}
    raw, sha = read_consumed(f, seen, "x")
    assert raw == b"a\n1\n" and sha == _sha(raw) and seen == {"x": sha}
    assert read_consumed(f, seen, "x")[1] == sha                       # same bytes again: accepted
    f.write_bytes(b"a\n2\n")
    with pytest.raises(ConsumedInputError, match="consumed twice with different bytes"):
        read_consumed(f, seen, "x")


def test_cohort_swapped_while_it_is_read_records_the_bytes_parsed_and_is_refused(tmp_path, synth, config, compute,
                                                                                monkeypatch):
    """load_ashrae_context: the cohort is read once; a substitution restored right after that read
    is what is parsed AND hashed (no second path read), so binding refuses the run; no completed manifest."""
    import shutil
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    with monkeypatch.context() as m:
        _no_path_parse(m, csv)
        clean = rr.load_ashrae_context(config, csv_path=csv, ids=synth["ids"])
    assert clean.input_sha256 == {"cohort_csv": file_sha256(csv)}
    with monkeypatch.context() as m:
        swapped = _swap_on_read(m, csv, lambda b: b.replace(b"2020-01-01", b"2020-01-02", 1))
        ctx = rr.load_ashrae_context(config, csv_path=csv, ids=synth["ids"])
    assert len(swapped) == 1                                   # one read
    assert ctx.input_sha256["cohort_csv"] == _sha(swapped[0]) != file_sha256(csv)
    assert ctx.p26["date"].iloc[0] == "2020-01-02"             # the bytes hashed are the bytes parsed
    with pytest.raises(rr.RunnerError, match="cohort_csv .* changed after it was loaded"):
        rr.run_e2(ctx, run_dir=tmp_path / "run", config=config, targets=("TSV_3p",), families=("RF",),
                  compute=compute, dry_run=True)
    _no_completed_manifest(tmp_path / "run" / "E2_lopo_dry_run", tmp_path / "run")


def test_e9b_cohort_swapped_while_nn_reads_it_leaves_no_manifest_and_fails_the_pin(tmp_path, synth, monkeypatch):
    """E9b: nn_cases.run reads the cohort once; a restored substitution is reported as the bytes parsed
    and refused against the bound hash (dry run), and fails the pinned-cohort check of a real run."""
    import data_groups
    import shutil
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    with monkeypatch.context() as m:
        _swap_on_read(m, csv, _plus_newline)
        with pytest.raises(rr.RunnerError, match="parsed inputs other than the bound ones"):
            rr.run_e9b(run_dir=tmp_path / "run", data_csv=csv, targets=["TSV_3p"], dry_run=True)
    _no_completed_manifest(tmp_path / "run" / "E9b_nn_cases_dry_run", tmp_path / "run")
    with monkeypatch.context() as m:
        _no_path_parse(m, csv)
        man = rr.run_e9b(run_dir=tmp_path / "ok", data_csv=csv, targets=["TSV_3p"], dry_run=True)
    assert man["inputs_sha256"]["cohort_csv"] == file_sha256(csv)
    # a real E9b: the pin is checked on the bytes read, not on an earlier hash of the path
    _as_real_run(monkeypatch)
    monkeypatch.setitem(data_groups.PINNED_SHA256, "canonical_csv", file_sha256(csv))
    _swap_on_read(monkeypatch, csv, _plus_newline)
    with pytest.raises(ValueError, match="pinned"):
        rr.run_e9b(run_dir=tmp_path / "real", data_csv=csv)
    _no_completed_manifest(tmp_path / "real" / "E9b_nn_cases", tmp_path / "real")


@pytest.mark.parametrize("what", ["data_csv", "predictions"])
def test_e7_input_swapped_while_the_indices_driver_reads_it_leaves_no_manifest(tmp_path, synth, e1_trees,
                                                                                monkeypatch, what):
    """E7: the cohort and the E1 predictions are parsed by the indices driver from one read each and reported;
    a restored substitution is refused against the bound hash."""
    import shutil
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    kw = dict(e1_dir=e1, data_csv=csv, split_dir=synth["split_dir"], skip_group_ids=True, dry_run=True)
    with monkeypatch.context() as m:
        if what == "data_csv":
            _swap_on_read(m, csv, _plus_newline)
        else:
            _swap_on_read(m, rr.e1_prediction_file(e1, "TSV_3p", "RF", 42, True), _regzip)
        with pytest.raises(rr.RunnerError, match="parsed inputs other than the bound ones"):
            rr.run_e7(run_dir=tmp_path / "run", **kw)
    _no_completed_manifest(tmp_path / "run" / "E7_pmv_dry_run", tmp_path / "run")
    with monkeypatch.context() as m:
        _no_path_parse(m, csv, *sorted((e1 / "predictions").iterdir()))
        man = rr.run_e7(run_dir=tmp_path / "ok", **kw)
    prov = json.loads((tmp_path / "ok" / "E7_pmv_dry_run" / "provenance.json").read_text())
    assert prov["data_csv_sha256"] == man["inputs_sha256"]["data_csv"] == file_sha256(csv)


def test_e4_e1_predictions_swapped_while_checked_are_refused(tmp_path, synth, e1_trees, config, compute,
                                                             monkeypatch):
    """E4: check_e1_predictions parses the bytes it hashes; a restored substitution differs from
    the hash validated against E1 and is refused before any fit."""
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    f = rr.e1_prediction_file(e1, "TSV_3p", "RF", 42, True)
    with monkeypatch.context() as m:
        _swap_on_read(m, f, _regzip)
        with pytest.raises(rr.RunnerError, match="validated against E1 as"):
            _ablation(synth, config, compute, tmp_path / "run", e1)
    assert not (tmp_path / "run" / "E4_ema_dry_run").exists()
    with monkeypatch.context() as m:
        _no_path_parse(m, *sorted((e1 / "predictions").iterdir()))
        man = _ablation(synth, config, compute, tmp_path / "ok", e1)
    pairs = pd.read_csv(tmp_path / "ok" / "E4_ema_dry_run" / "pairs.csv")
    row = pairs[pairs["seed"] == 42].iloc[0]                     # E4: the F17 side (extended) is the E1 file
    assert row["extended_sha256"] == man["inputs_sha256"][f"e1:{f.name}"] == file_sha256(f)


@pytest.mark.parametrize("what", ["cohort_csv", "french_cohort", "model_file"])
def test_e6_input_swapped_while_transfert_reads_it_leaves_no_manifest(tmp_path, synth, e1_trees, config,
                                                                     monkeypatch, what):
    """E6: the ASHRAE cohort, the French cohort and the E1 model files are read once by transfert /
    phase3_runner and recorded with the hash of the bytes loaded; a restored substitution is refused (parsed
    hash != bound hash, or source model hash changed); no completed manifest."""
    import shutil
    import transfert
    hostein = rr.REPO_DIR / transfert.COHORT_FILES["Hostein"]
    if not hostein.is_file():
        pytest.skip("French cohort absent")
    french = tmp_path / "Hostein_api.csv"
    shutil.copy(hostein, french)
    monkeypatch.setattr(transfert, "COHORT_FILES", {**transfert.COHORT_FILES, "Hostein": str(french)})
    csv = tmp_path / "cohort_api.csv"
    shutil.copy(synth["csv"], csv)
    e1 = _e1_copy(e1_trees, tmp_path / "E1")
    model = e1 / "best_rf_by_target.joblib"

    def rejoblib(b):
        import io
        buf = io.BytesIO()
        joblib.dump(joblib.load(io.BytesIO(b)), buf, compress=1)
        return buf.getvalue()
    kw = dict(e1_dir=e1, strategies=["local"], cohorts=["Hostein"], targets=["TSV_3p"], ashrae_csv=csv,
              ashrae_split_dir=synth["split_dir"], dry_run=True, dry_run_splits=1, gpu_concurrency=1, config=config)
    with monkeypatch.context() as m:
        _swap_on_read(m, {"cohort_csv": csv, "french_cohort": french, "model_file": model}[what],
                      {"cohort_csv": _plus_newline, "french_cohort": _plus_newline, "model_file": rejoblib}[what])
        with pytest.raises((rr.RunnerError, RuntimeError),
                           match="parsed inputs other than the bound ones|not the bound inputs|changed during E6"):
            rr.run_e6(run_dir=tmp_path / "run", **kw)
    _no_completed_manifest(tmp_path / "run" / "E6_phase3_dry_run", tmp_path / "run")
    with monkeypatch.context() as m:
        _no_path_parse(m, csv, french, *sorted(e1.glob("best_*_by_target*.joblib")))
        man = rr.run_e6(run_dir=tmp_path / "ok", **kw)
    rec = json.loads((tmp_path / "ok" / "E6_phase3_dry_run" / "e6_run_record.json").read_text())
    assert rec["source_files_sha256"]["cohort_csv"] == man["inputs_sha256"]["cohort_csv"] == file_sha256(csv)
    assert rec["cohort_files_sha256"] == {"Hostein": file_sha256(french)}
    assert rec["model_files_sha256"]["best_rf_by_target.joblib"] == file_sha256(model)


def test_config_and_lock_file_swapped_while_read_are_refused(tmp_path, synth, monkeypatch):
    """config.yaml and lock file: checked and hashed from one read; a restored substitution leaves the
    recorded hash different from the file, which the end check refuses."""
    import shutil
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    with monkeypatch.context() as m:
        _swap_on_read(m, cfg_file, lambda b: b + b"\n# swapped\n")
        with pytest.raises(rr.RunnerError, match="config.yaml: start .* end"):
            rr.run_e9b(run_dir=tmp_path / "run", data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True, config=load_config(cfg_file))
    _no_completed_manifest(tmp_path / "run" / "E9b_nn_cases_dry_run", tmp_path / "run")
    lock = tmp_path / "requirements.lock"
    shutil.copy(rr.LOCK_FILE, lock)
    with monkeypatch.context() as m:
        _swap_on_read(m, lock, lambda b: b + b"# swapped\n")
        env = rr.environment_record(lock)
    assert env["lock_file_sha256"] != file_sha256(lock)
    startup = rr.startup_record(rr.run_provenance(), load_config(cfg_file))
    with pytest.raises(rr.RunnerError, match="pins read from"):
        rr.check_end_state(startup, load_config(cfg_file), {}, dry_run=True, env=env)
    assert rr.check_end_state(startup, load_config(cfg_file), {}, dry_run=True,
                              env=rr.environment_record(lock))["matches_startup"]


def _swap_first_access(monkeypatch, path, mutate):
    """Restored substitution around the FIRST access of ``path``, whatever the reader (``open`` or
    ``Path.read_bytes``): the file holds ``mutate(bytes)`` during that access only (for ``open``, until the
    handle is closed), then the original bytes are written back. Returns the list of substituted bytes."""
    import builtins
    import pathlib
    target = Path(path).resolve()
    orig_open, orig_read = builtins.open, pathlib.Path.read_bytes
    seen = []

    def _hit(p):
        try:
            return not seen and isinstance(p, (str, os.PathLike)) and Path(p).resolve() == target
        except (TypeError, ValueError, OSError):
            return False

    def _swap():
        with orig_open(target, "rb") as fh:
            before = fh.read()
        swapped = mutate(before)
        assert swapped != before
        with orig_open(target, "wb") as fh:
            fh.write(swapped)
        seen.append(swapped)
        return before

    def _restore(before):
        with orig_open(target, "wb") as fh:
            fh.write(before)

    class _Restoring:
        def __init__(self, fh, before):
            self._fh, self._before = fh, before

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def close(self):
            try:
                self._fh.close()
            finally:
                _restore(self._before)

        def __getattr__(self, name):
            return getattr(self._fh, name)

        def __iter__(self):
            return iter(self._fh)

    def fake_open(file, *a, **k):
        if not _hit(file):
            return orig_open(file, *a, **k)
        before = _swap()
        return _Restoring(orig_open(file, *a, **k), before)

    def fake_read_bytes(self):
        if not _hit(self):
            return orig_read(self)
        before = _swap()
        try:
            return orig_read(self)
        finally:
            _restore(before)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", fake_read_bytes)
    return seen


def _cli_e9b_dry(synth, run):
    import run_revision
    return run_revision.main(["e9b", "--run_dir", str(run), "--dry_run", "--data_csv", str(synth["csv"]),
                              "--targets", "TSV_3p"])


@pytest.mark.parametrize("mutate", [lambda b: b + b"\n# swapped comment\n",
                                    lambda b: b.replace(b"seed:", b"seed: 7 #", 1)], ids=["comment", "value"])
def test_initial_config_parse_swapped_is_refused(tmp_path, synth, monkeypatch, mutate):
    """The CLI parses config.yaml once and records the hash of THOSE bytes (dry-run derivation
    included); a restored substitution around that initial parse leaves the recorded hash different from the
    file, which the end check refuses: no completed manifest. Before the fix the CLI parsed the swapped bytes
    and recorded the hash of a second read (the restored file), and the run completed."""
    import shutil
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    with monkeypatch.context() as m:
        seen = _swap_first_access(m, cfg_file, mutate)
        with pytest.raises(rr.RunnerError, match="config.yaml: start .* end"):
            _cli_e9b_dry(synth, tmp_path / "run")
    assert len(seen) == 1
    _no_completed_manifest(tmp_path / "run" / "E9b_nn_cases_dry_run", tmp_path / "run")
    # no swap: the recorded hash is the file's, and config.yaml is read exactly once by the run
    reads = []
    import config_loader
    orig = config_loader.read_consumed
    monkeypatch.setattr(config_loader, "read_consumed", lambda p, *a, **k: (reads.append(Path(p)), orig(p, *a, **k))[1])
    monkeypatch.setattr(rr, "read_consumed", config_loader.read_consumed)
    import data_groups
    monkeypatch.setattr(data_groups, "CONFIG_PATH", tmp_path / "absent.yaml")      # never re-read for paths
    man = _cli_e9b_dry(synth, tmp_path / "ok")
    assert man["config_file_sha256"] == file_sha256(cfg_file) and man["config_is_file"] is False
    assert reads.count(cfg_file) == 1


def test_cli_config_is_derived_from_the_one_read(tmp_path, monkeypatch):
    """_config returns the configuration used (dry-run hyperparameters on a copy) and the
    (parsed, SHA-256) record of the single read; the parsed record is not altered by the dry-run derivation."""
    import run_revision
    config, (parsed, sha) = run_revision._config(dry_run=True)
    assert sha == file_sha256(rr.CONFIG_PATH) and parsed == load_config(rr.CONFIG_PATH)
    assert config["classical_hparams"] == rr.DRY_RUN_HPARAMS["classical_hparams"] != parsed["classical_hparams"]
    real, (parsed2, _) = run_revision._config(dry_run=False)
    assert real == parsed2 and real is not parsed2


def test_library_calls_without_config_parse_the_bytes_they_record(tmp_path, synth, monkeypatch):
    """run_e9b (likewise run_e6 / run_e7) without a configuration reads config.yaml once and records
    the hash of the bytes parsed; a substitution around that read is refused."""
    import shutil
    cfg_file = tmp_path / "config.yaml"
    shutil.copy(rr.CONFIG_PATH, cfg_file)
    monkeypatch.setattr(rr, "CONFIG_PATH", cfg_file)
    kw = dict(data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    with monkeypatch.context() as m:
        _swap_first_access(m, cfg_file, lambda b: b + b"\n# swapped\n")
        with pytest.raises(rr.RunnerError, match="config.yaml: start .* end"):
            rr.run_e9b(run_dir=tmp_path / "run", **kw)
    _no_completed_manifest(tmp_path / "run" / "E9b_nn_cases_dry_run", tmp_path / "run")
    man = rr.run_e9b(run_dir=tmp_path / "ok", **kw)
    assert man["config_file_sha256"] == file_sha256(cfg_file) and man["config_is_file"] is True


def test_real_runs_refuse_an_import_time_config_other_than_the_recorded_one(monkeypatch):
    """Other configuration reads: an import-time CONFIG differing from config.yaml as parsed
    from the recorded bytes is refused in a real run (at _prepare and at finalisation); dry runs skip it."""
    import models_classic
    parsed, _ = rr.read_config_file()
    ref = rr._json_sha256(parsed)
    rr.check_import_time_configs(ref, dry_run=False)                       # the modules' copies agree
    bad = copy.deepcopy(models_classic.CONFIG)
    bad["seed"] = int(bad["seed"]) + 1
    monkeypatch.setattr(models_classic, "CONFIG", bad)
    with pytest.raises(rr.RunnerError, match="import-time configuration of models_classic"):
        rr.check_import_time_configs(ref, dry_run=False)
    rr.check_import_time_configs(ref, dry_run=True)
    startup = rr.startup_record(rr.run_provenance(), parsed)
    assert startup.config_file_parsed_sha256 == ref


def test_group_id_paths_come_from_the_run_configuration(tmp_path, synth, config, monkeypatch):
    """data_groups.default_paths: the identifier file paths are taken from the run configuration,
    not from a second parse of config.yaml."""
    import data_groups
    monkeypatch.setattr(data_groups, "CONFIG_PATH", tmp_path / "absent.yaml")
    paths = data_groups.default_paths(config)
    assert paths["raw_csv"] == data_groups._resolve(config["data"]["raw"]["ashrae_csv"])
    with pytest.raises(FileNotFoundError):
        data_groups.default_paths()
    seen = {}

    def fake_load_group_ids(**kw):
        seen.update(kw)
        kw["consumed_sha256"].update(canonical_csv=rr.hashlib.sha256(kw["raw_bytes"]["canonical_csv"]).hexdigest(),
                                     raw_csv="a" * 64, raw_metadata_xlsx="b" * 64)
        return synth["ids"]
    monkeypatch.setattr(data_groups, "load_group_ids", fake_load_group_ids)
    ctx = rr.load_ashrae_context(config, csv_path=synth["csv"])
    assert seen["config"] is config and ctx.input_files["raw_csv"] == str(paths["raw_csv"])


def test_real_runs_refuse_a_working_directory_other_than_the_repository_root(tmp_path, monkeypatch):
    """A real run is launched from the repository root (import-time config.yaml reads and the
    relative Data/ and splits/ defaults resolve in the checkout whose code runs)."""
    import run_revision
    with pytest.raises(SystemExit, match="repository root"):
        run_revision.check_launch_cwd(False, tmp_path)
    assert run_revision.check_launch_cwd(False, rr.REPO_DIR) == rr.REPO_DIR
    assert run_revision.check_launch_cwd(True, tmp_path) == tmp_path                   # dry runs
    monkeypatch.setattr(rr, "read_config_file", lambda: pytest.fail("config read"))
    monkeypatch.chdir(tmp_path)
    for argv in (["e9b"], ["e1"], ["e6"], ["e7"]):
        with pytest.raises(SystemExit, match="repository root"):
            run_revision.main([*argv, "--run_dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_index_is_restored_even_if_the_undo_fails(tmp_path, monkeypatch):
    """A failing undo of the completion rename does not skip the index restore."""
    run = tmp_path / "run"
    (run / "E9b_nn_cases").mkdir(parents=True)
    pending = run / "E9b_nn_cases" / "run_manifest.json.pending"
    pending.write_text("{}")
    final = run / "E9b_nn_cases" / "run_manifest.json"
    monkeypatch.setattr(rr, "phase1_provenance", lambda: {})
    real_write = rr.Path.write_text
    calls = {"n": 0}

    def write_text(self, *a, **k):                       # the second index write (status complete) fails
        if self.name.startswith("manifest.json.") and self.name.endswith(".tmp"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
        return real_write(self, *a, **k)
    monkeypatch.setattr(rr.Path, "write_text", write_text)

    def undo():
        raise OSError("undo failed")
    with pytest.raises(OSError, match="undo failed"):
        rr.update_run_index(run, "E9b", final, manifest_sha256="c" * 64,
                            complete=lambda: os.replace(pending, final), undo=undo)
    assert not (run / "manifest.json").exists()             # restored to "absent" despite the failed undo


# ================================================================ launch checkout
def test_git_checkout_tells_a_linked_worktree_from_the_main_checkout(tmp_path):
    import subprocess
    import phase3_runner

    def git(*a, cwd):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd, check=True,
                       capture_output=True)
    main = tmp_path / "main"
    main.mkdir()
    git("init", "-q", cwd=main)
    (main / "a.py").write_text("x = 1\n")
    git("add", "a.py", cwd=main)
    git("commit", "-q", "-m", "c", cwd=main)
    git("worktree", "add", "-q", "--detach", str(tmp_path / "wt"), "HEAD", cwd=main)
    m, w = phase3_runner.git_checkout(main), phase3_runner.git_checkout(tmp_path / "wt")
    assert m["linked_worktree"] is False and w["linked_worktree"] is True
    assert w["git_common_dir"] == m["git_common_dir"] == m["git_dir"] != w["git_dir"]
    assert phase3_runner.git_checkout(tmp_path)["linked_worktree"] is None        # not a repository


def test_real_runs_accept_a_plain_clone(tmp_path, monkeypatch):
    """A real run starts from a plain clone (the main checkout) as from a linked worktree: the checkout kind is
    recorded in the manifest, never refused; the clean-tree, lock and config checks still apply."""
    import run_revision
    assert not hasattr(run_revision, "check_launch_checkout")
    for kind in ({"linked_worktree": False, "toplevel": "/x/clone"}, {"linked_worktree": True, "toplevel": "/x/wt"}):
        monkeypatch.setattr(run_revision.phase3_runner, "git_checkout", lambda kind=kind: kind)
        monkeypatch.chdir(rr.REPO_DIR)
        monkeypatch.setattr(rr, "read_config_file", lambda: (_ for _ in ()).throw(RuntimeError("config read")))
        with pytest.raises(RuntimeError, match="config read"):          # past the launch checks
            run_revision.main(["e1", "--run_dir", str(tmp_path / "run")])
    with pytest.raises(SystemExit):
        run_revision.parse_args(["e1", "--run_dir", "r", "--allow-main-checkout"])
    assert not (tmp_path / "run").exists()


def test_manifest_records_the_checkout_kind(tmp_path, synth):
    import phase3_runner
    man = rr.run_e9b(run_dir=tmp_path, data_csv=synth["csv"], targets=["TSV_3p"], dry_run=True)
    assert man["provenance"]["git_worktree"] == phase3_runner.git_checkout()
    assert man["provenance"]["git_worktree"]["linked_worktree"] in (True, False)
