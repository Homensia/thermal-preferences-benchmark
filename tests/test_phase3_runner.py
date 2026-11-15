# -*- coding: utf-8 -*-
"""
Phase 3 runner, experiment E6.

* frozen date-disjoint splits: procedure, sizes on the real cohorts
  (labels and dates only, no fit), serialisation before any fit,
  load-and-check, identical across targets;
* ``LabelEncodedClassifier`` forwards a sample weight unchanged;
* DEV with one class → DEV-majority constant predictor, flagged; and the
  missing-DEV-class contract (no TEST vote dropped, ``p_<class>`` keyed
  by label value through ``classes_``);
* a full dry run on tiny synthetic data (prediction columns, row order, every
  strategy on the same TEST rows, 40-step FTT adaptation records, sources
  untouched, byte-identical re-execution, outputs never overwritten);
* the transfert.py CLI now runs E6; the legacy adaptation evaluator is gone.

Synthetic data and tiny models on CPU; each test well under 60 s.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import phase3_runner as p3  # noqa: E402
import revision_stats as rs  # noqa: E402
from category_normalisation import GLOBAL_VOCABULARY, split_feature_types, vocabulary_for  # noqa: E402
from models_classic import LabelEncodedClassifier  # noqa: E402

MOUJALLED = REPO_DIR / "Data" / "Moujalled_api.csv"
HOSTEIN = REPO_DIR / "Data" / "Hostein_api.csv"
TARGETS = ("TSV_3p", "thermal_preference")
TINY = dict(dim=8, depth=1, heads=2, dim_head=4)


@pytest.fixture(scope="module")
def tr():
    import transfert
    return transfert


@pytest.fixture(scope="module")
def features():
    import models
    return list(models.FEATURES_ALL)


# ================================================================ splits
def _reference_split(groups, frac, r):
    """Independent re-implementation of the split procedure."""
    dates = np.sort(pd.unique(np.asarray(groups)))
    n_g = pd.Series(groups).value_counts().reindex(dates).to_numpy()
    order = np.random.default_rng(1000 + r).permutation(len(dates))
    k = int(np.searchsorted(np.cumsum(n_g[order]), frac * n_g.sum()) + 1)
    return sorted(dates[order[:k]].tolist())


def test_materialise_splits_follows_the_frozen_procedure():
    groups = np.repeat([f"2020-01-{d:02d}" for d in range(1, 13)], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
    groups = np.random.default_rng(0).permutation(groups)            # row order is irrelevant
    n = len(groups)
    for frac in (0.2, 0.8):
        splits = p3.materialise_splits(groups, frac)
        assert len(splits) == 20 and [s["seed"] for s in splits] == list(range(1000, 1020))
        for s in splits:
            assert s["dev_groups"] == _reference_split(groups, frac, s["r"])
            dev, test = np.asarray(s["dev_idx"]), np.asarray(s["test_idx"])
            p3.check_partition(range(n), dev, test)
            assert not set(groups[dev]) & set(groups[test])                    # date-disjoint
            assert set(groups[dev]) == set(s["dev_groups"]) and set(groups[test]) == set(s["test_groups"])
            # shortest prefix with >= frac of the votes: dropping the last DEV group goes below it
            order = np.random.default_rng(s["seed"]).permutation(12)
            sizes = pd.Series(groups).value_counts().reindex(sorted(set(groups))).to_numpy()[order]
            k = len(s["dev_groups"])
            assert sizes[:k].sum() >= frac * n > sizes[:k - 1].sum()


def test_group_order_is_ascending_strings_and_missing_groups_raise():
    assert p3.group_order(["2004-08-10", "2004-08-09", "2004-08-10"]) == ["2004-08-09", "2004-08-10"]
    for bad in (["2004-08-09", None], ["2004-08-09", " "]):
        with pytest.raises(p3.SplitIntegrityError):
            p3.materialise_splits(bad, 0.2)


@pytest.mark.skipif(not (MOUJALLED.exists() and HOSTEIN.exists()), reason="French cohorts absent")
@pytest.mark.parametrize("cohort, path, regime, dev_votes, test_dates, never_in_test", [
    ("Moujalled", MOUJALLED, "20-80", (51, 65), (22, 26), 0),
    ("Moujalled", MOUJALLED, "80-20", (201, 216), (4, 8), 1),
    ("Hostein", HOSTEIN, "20-80", (33, 39), (40, 48), 0),
    ("Hostein", HOSTEIN, "80-20", (131, 139), (7, 14), 0),
])
def test_real_cohort_splits_match_the_reference_procedure(tr, tmp_path, cohort, path, regime, dev_votes,
                                                          test_dates, never_in_test):
    """Dates and labels only (no fit); written to tmp_path, never to splits/."""
    df = tr.load_external_cohort(path, cohort)
    payloads = [p3.ensure_splits(tmp_path, df, cohort, t, regime) for t in rs.TARGETS]
    splits = payloads[0]["splits"]
    assert [len(s["dev_idx"]) for s in splits] and (
        min(len(s["dev_idx"]) for s in splits), max(len(s["dev_idx"]) for s in splits)) == dev_votes
    assert (min(len(s["test_groups"]) for s in splits), max(len(s["test_groups"]) for s in splits)) == test_dates
    ever_test = set().union(*(s["test_groups"] for s in splits))
    assert len(set(payloads[0]["groups"]) - ever_test) == never_in_test
    for p in payloads[1:]:                                             # same date splits for every target
        assert [(s["dev_idx"], s["test_idx"]) for s in p["splits"]] == \
               [(s["dev_idx"], s["test_idx"]) for s in splits]
    for s in splits:
        assert s["dev_groups"] == _reference_split(df["Date"].astype(str).to_numpy(), p3.REGIMES[regime], s["r"])


def _cohort(n_dates=12, per_date=5, seed=0, labels=None):
    rng = np.random.default_rng(seed)
    import models
    n = n_dates * per_date
    df = pd.DataFrame(index=pd.RangeIndex(n))
    for i, f in enumerate(models.FEATURES_ALL):
        if f in GLOBAL_VOCABULARY:
            vocab = sorted(GLOBAL_VOCABULARY[f])
            df[f] = [vocab[(j + i) % len(vocab)] for j in range(n)]
        else:
            df[f] = rng.normal(size=n)
    df["Date"] = np.repeat([f"2023-07-{d + 1:02d}" for d in range(n_dates)], per_date)
    for t in TARGETS:
        df[t] = (rng.integers(-1, 2, size=n) if labels is None else labels).astype(float)
        df[models.FEATURES_ALL[0]] += df[t]                           # learnable signal
    return df


def test_ensure_splits_writes_once_then_loads_and_checks(tmp_path):
    df = _cohort()
    p = p3.ensure_splits(tmp_path, df, "Synth", "TSV_3p", "20-80")
    path = p3.splits_path(tmp_path, "Synth", "TSV_3p", "20-80")
    raw = path.read_bytes()
    assert p3.ensure_splits(tmp_path, df, "Synth", "TSV_3p", "20-80") == p and path.read_bytes() == raw
    assert p["format"] == p3.SPLITS_FORMAT and p["split_seeds"] == list(range(1000, 1020))
    assert p["splits"][3]["split_id"] == "20-80_r03" and p["grouping"] == "date"
    for s in p["splits"]:                                             # class presence logged per split
        assert sum(s["dev_class_counts"].values()) == len(s["dev_idx"])
        assert sum(s["test_class_counts"].values()) == len(s["test_idx"])
    tampered = json.loads(raw)
    tampered["splits"][0]["dev_idx"] = tampered["splits"][0]["dev_idx"][:-1]
    path.write_text(json.dumps(tampered))
    with pytest.raises(p3.SplitIntegrityError, match="never overwritten"):
        p3.ensure_splits(tmp_path, df, "Synth", "TSV_3p", "20-80")
    assert json.loads(path.read_text()) == tampered                   # not overwritten
    other = df.copy()
    other.loc[0, "Date"] = "2023-07-02"                               # a different cohort
    with pytest.raises(p3.SplitIntegrityError):
        p3.ensure_splits(tmp_path / "b", other.iloc[::-1], "Synth", "TSV_3p", "20-80")   # not a RangeIndex
    with pytest.raises(ValueError, match="frozen artefact"):
        p3.ensure_splits(REPO_DIR / "rerun_2026-04-28_no_reweighting" / "x", df, "Synth", "TSV_3p", "20-80")


# ================================================================ sample weights through LabelEncodedClassifier
class _Spy(BaseEstimator, ClassifierMixin):
    """Records the rows, labels and weights it is fitted with."""

    def fit(self, X, y, sample_weight=None):
        self.X_ = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
        self.y_ = np.asarray(y)
        self.sample_weight_ = None if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(y)
        return self


def test_label_encoded_classifier_forwards_sample_weight_unchanged():
    w = np.asarray([0.5, 2.0, 1.0, 3.0])
    m = LabelEncodedClassifier(_Spy()).fit(np.zeros((4, 1)), [-1.0, 0.0, 1.0, 1.0], sample_weight=w)
    assert m.estimator_.sample_weight_ is not None and np.array_equal(m.estimator_.sample_weight_, w)
    assert list(m.estimator_.y_) == [0, 1, 2, 2]
    m = LabelEncodedClassifier(_Spy()).fit(np.zeros((4, 1)), [-1.0, 0.0, 1.0, 1.0])
    assert m.estimator_.sample_weight_ is None


# ================================================================ single-class DEV and missing DEV class
def test_single_class_dev_gets_the_dev_majority_constant_predictor(tr, machine_compute):
    rf, _ = tr.make_rf_xgb_protos(["a"], ["Season"], cat_vocabulary=vocabulary_for(["Season"]),
                                  compute=machine_compute, families=("RF",))
    X = pd.DataFrame({"a": np.arange(6.0), "Season": "summer"})
    m, const = p3.fit_local(rf, X, np.zeros(6), "TSV_3p")
    assert const and isinstance(m, p3.DevMajorityClassifier)
    X_test = pd.DataFrame({"a": [0.0, 1.0, 2.0, 9.0], "Season": "summer"}, index=[10, 11, 12, 13])
    fr = p3.prediction_frame(m, X_test, [-1, 0, 1, 1], experiment="E6", cohort="C", target="TSV_3p",
                             model_name="RF_local", seed=42, split_id="20-80_r00", dates=["d"] * 4,
                             categories=p3.train_categories(X, ["Season"]), cat_cols=["Season"])
    assert len(fr) == 4 and list(fr["row_idx"]) == [10, 11, 12, 13]          # no TEST vote dropped
    assert list(fr["y_pred"]) == [0, 0, 0, 0] and list(fr["p_0"]) == [1.0] * 4
    assert list(fr["p_-1"]) == [0.0] * 4 and list(fr["p_1"]) == [0.0] * 4
    # the tie rule inside the constant predictor is not needed with one class; two classes → real fit
    m2, const2 = p3.fit_local(rf, X, np.asarray([0, 1, 0, 1, 0, 1.0]), "TSV_3p")
    assert not const2 and not isinstance(m2, p3.DevMajorityClassifier)


def test_missing_dev_class_keeps_every_test_row_and_maps_probabilities_by_label(tr, xgb_fits):
    """DEV without class -1: every TEST row is kept and probabilities are mapped by label."""
    rf, xgb = tr.make_rf_xgb_protos(["a"], ["Season"], cat_vocabulary=vocabulary_for(["Season"]),
                                    compute=xgb_fits)
    X = pd.DataFrame({"a": np.r_[np.zeros(10), np.ones(10)], "Season": "summer"})
    y = np.r_[np.zeros(10), np.ones(10)]
    X_test = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 0.5], "Season": "summer"}, index=[7, 3, 9, 1, 4])
    y_test = [-1, -1, 0, 1, -1]
    for proto in (rf, xgb):
        m, const = p3.fit_local(proto, X, y, "TSV_3p")
        assert not const
        fr = p3.prediction_frame(m, X_test, y_test, experiment="E6", cohort="C", target="TSV_3p",
                                 model_name="M", seed=42, split_id="s", dates=["d"] * 5,
                                 categories=p3.train_categories(X, ["Season"]), cat_cols=["Season"])
        assert len(fr) == 5 and list(fr["row_idx"]) == [7, 3, 9, 1, 4]
        assert (fr["p_-1"] == 0.0).all()
        proba = m.predict_proba(X_test)
        classes = list(np.asarray(m.classes_, dtype=float))
        assert classes == [0.0, 1.0]
        np.testing.assert_array_equal(fr["p_0"], proba[:, classes.index(0.0)])
        np.testing.assert_array_equal(fr["p_1"], proba[:, classes.index(1.0)])
        acc = (fr["y_true"] == fr["y_pred"]).mean()
        assert acc <= 2 / 5                                             # the three -1 votes are errors


def test_prediction_frame_flags_oov_relative_to_the_fit():
    m = p3.DevMajorityClassifier("TSV_3p").fit(None, [1.0])
    X = pd.DataFrame({"Season": ["summer", "winter"], "Sexe": ["male", "female"]}, index=[0, 1])
    fr = p3.prediction_frame(m, X, [1, 1], experiment="E6", cohort="C", target="TSV_3p", model_name="M",
                             seed=42, split_id="s", dates=["a", "b"],
                             categories={"Season": {"summer"}, "Sexe": {"male"}}, cat_cols=["Season", "Sexe"])
    assert list(fr["oov_flag"]) == [False, True] and list(fr["oov_vars"]) == ["", "Season;Sexe"]
    assert list(fr.columns) == p3.prediction_columns("TSV_3p")
    assert fr["record_id"].tolist() == ["C:0", "C:1"] and fr["publication"].isna().all()


# ================================================================ dry run on tiny data
def _ashrae(features, n=150, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=pd.RangeIndex(n))
    for i, f in enumerate(features):
        if f in GLOBAL_VOCABULARY:
            vocab = sorted(GLOBAL_VOCABULARY[f])
            df[f] = [vocab[(j + i) % len(vocab)] for j in range(n)]
        else:
            df[f] = rng.normal(size=n)
    ys = {t: np.resize(np.asarray([-1.0, 0.0, 1.0]), n) for t in TARGETS}
    for t in TARGETS:
        df[features[0]] += ys[t]
    return df, ys


@pytest.fixture(scope="module")
def e1_models(tr, features):
    """Tiny E1-like seed-42 models (revision encoders, classes = ontology) and their source rows."""
    import torch
    import models_deep as md
    from compute import load_compute
    compute = load_compute()
    if not torch.cuda.is_available():
        compute = {**compute, "xgb_device": "cpu", "torch_device": "cpu"}
    import xgboost
    df, ys = _ashrae(features)
    num, cat = split_feature_types(df, features)
    vocab = vocabulary_for(cat)
    models, source = {}, {}
    for t in TARGETS:
        rf, xgb = tr.make_rf_xgb_protos(num, cat, seed=42, cat_vocabulary=vocab, compute=compute)
        rf.set_params(clf__n_estimators=10)
        xgb.set_params(clf__estimator__n_estimators=10)
        ft = md.FTClassifier(num_cols=num, cat_cols=cat, cat_vocabulary=vocab, device="cpu", lr=1e-2,
                             batch_size=64, epochs=2, patience=2, random_state=42, **TINY)
        try:
            xgb_fit = xgb.fit(df[features], ys[t])
        except xgboost.core.XGBoostError as exc:       # as the conftest fixture xgb_fits
            if "cudaErrorNoDevice" in str(exc):
                pytest.skip("xgboost CUDA build cannot fit with the GPU hidden (cudaErrorNoDevice)")
            raise
        models[t] = {"RF": rf.fit(df[features], ys[t]), "XGB": xgb_fit, "FTT": ft.fit(df[features], ys[t])}
        source[t] = (df[features], ys[t])
    return models, source, compute


def _small_protos(tr):
    def make(num_cols, cat_cols, **kw):
        rf, xgb = tr.make_rf_xgb_protos(num_cols, cat_cols, **kw)
        rf.set_params(clf__n_estimators=10)
        xgb.set_params(clf__estimator__n_estimators=10)
        return rf, xgb
    return make


def _cohorts():
    a = _cohort(seed=3)
    # second cohort: one class except on one date that is in TEST of split 0 → single-class DEV on split 0
    b = _cohort(n_dates=10, per_date=4, seed=4, labels=np.zeros(40))
    dev0 = set(p3.materialise_splits(b["Date"], 0.2)[0]["dev_groups"])
    other = next(d for d in sorted(b["Date"].unique()) if d not in dev0)
    for t in TARGETS:
        b.loc[b["Date"] == other, t] = [-1.0, 1.0, 1.0, -1.0]
    return {"CohortA": a, "CohortB": b}


def _run(tr, e1_models, features, out, splits, **kw):
    models, source, compute = e1_models
    args = dict(cohorts=_cohorts(), source_rows=source, models=models, make_protos=_small_protos(tr),
                output_dir=out, splits_dir=splits, targets=TARGETS, features=features, compute=compute,
                n_splits=2, dry_run=True)
    args.update(kw)
    return p3.run_e6(**args)


def test_dry_run_writes_predictions_files_for_every_strategy(tr, e1_models, features, tmp_path):
    models, _, _ = e1_models
    files = {}
    for name, fam in (("best_ft_by_target.joblib", "FTT"), ("best_rf_by_target.joblib", "RF")):
        files[name] = tmp_path / name
        joblib.dump({t: models[t][fam] for t in TARGETS}, files[name])
    rec = _run(tr, e1_models, features, tmp_path / "out", tmp_path / "splits", model_files=files)
    out = tmp_path / "out"
    names = ["RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot", "RF_local", "XGB_local", "FTT_head", "FTT_full"]
    cohorts = _cohorts()
    assert rec["experiment"] == "E6_dry_run" and rec["dry_run"] and rec["n_splits_evaluated"] == 2
    assert rec["n_prediction_files"] == 2 * len(TARGETS) * (3 + 2 * len(names))
    assert rec["model_files_sha256"] and rec["ftt_budget"]["epochs"] == 40
    # every fit went through compute_scheduler: trees as CPU jobs in this process,
    # FTT adaptations as GPU jobs in worker processes; zero-shot predictions are not fits
    import os
    jobs = {j["key"]: j for j in rec["jobs"]}
    n_fit_splits = 2 * len(TARGETS) * 2
    assert len(jobs) == n_fit_splits * (2 + 2)
    for key, j in jobs.items():
        model = key.split("|")[-1]
        assert j["seed"] == 42 and j["seconds"] > 0
        if model.startswith("FTT_"):
            assert j["kind"] == "gpu" and j["pid"] != os.getpid()
        else:
            assert j["kind"] == ("gpu" if model.startswith("XGB") and e1_models[2]["xgb_device"] == "cuda" else "cpu")
            if j["kind"] == "cpu":
                assert j["pid"] == os.getpid()
    assert not any("zero_shot" in k for k in jobs)
    assert rec["gpu_concurrency"] == e1_models[2]["gpu_concurrency"]
    for cohort, df in cohorts.items():
        for t in TARGETS:
            payload = json.loads(p3.splits_path(tmp_path / "splits", cohort, t, "20-80").read_text())
            for s in payload["splits"][:2]:
                test = s["test_idx"]
                frames = {}
                for m in names:
                    f = out / "predictions" / f"E6_dry_run__{cohort}__{t}__{m}__seed42__{s['split_id']}.csv.gz"
                    fr = pd.read_csv(f)
                    assert list(fr.columns) == p3.prediction_columns(t)
                    assert fr["row_idx"].tolist() == test                       # serialised order
                    assert fr["y_true"].tolist() == df[t].iloc[test].astype(int).tolist()
                    assert (fr["experiment"] == "E6_dry_run").all() and (fr["seed"] == 42).all()
                    assert (fr["model"] == m).all() and (fr["split_id"] == s["split_id"]).all()
                    assert fr["date"].tolist() == df["Date"].iloc[test].tolist()
                    np.testing.assert_allclose(fr[[f"p_{c}" for c in rs.ontology(t)]].sum(axis=1), 1.0,
                                               rtol=1e-5)
                    frames[m] = fr
                rs.align_predictions(frames, cluster_col="date", expected_row_idx=test)
                for m in ("FTT_head", "FTT_full"):
                    r = json.loads((out / "ftt_adaptation" /
                                    f"E6_dry_run__{cohort}__{t}__{s['split_id']}__{m}.json").read_text())
                    assert r["n_optimizer_steps"] == 40 and r["dev_idx"] == s["dev_idx"]
                    assert len(r["train_loss"]) == 40 and r["source_random_state"] == 42
            # serialised splits identical across targets (date groups)
            assert payload["splits"][0]["dev_idx"] == json.loads(
                p3.splits_path(tmp_path / "splits", cohort, TARGETS[0], "20-80").read_text())["splits"][0]["dev_idx"]
            full = pd.read_csv(out / "predictions" / f"E6_dry_run__{cohort}__{t}__FTT_zero_shot__seed42__full_cohort.csv.gz")
            assert full["row_idx"].tolist() == list(range(len(df)))
    summary = pd.read_csv(out / "split_summary.csv")
    assert len(summary) == 2 * len(TARGETS) * 2
    assert (summary["local_constant_p20"] == (summary["dev_n_classes"] < 2)).all()
    flagged = summary[summary["local_constant_p20"]]
    assert len(flagged) and set(flagged["cohort"]) == {"CohortB"}
    for _, row in flagged.iterrows():
        fr = pd.read_csv(out / "predictions" /
                         f"E6_dry_run__CohortB__{row['target']}__RF_local__seed42__{row['split_id']}.csv.gz")
        assert fr["y_pred"].nunique() == 1 and len(fr) == row["n_test"]     # constant, no vote dropped
        assert json.loads(row["test_classes_absent_from_dev"])              # TEST classes absent from DEV kept
    assert (summary.filter(like="oov_test_votes__RF_zero_shot") == 0).all().all()


def test_diverged_ftt_writes_no_e6_output(tr, e1_models, features, tmp_path):
    """A source FTT with non-finite weights stops E6 before any FTT
    predictions file, split summary or run record is written (no valid-looking class from NaN logits);
    the run has no record, so it is never a completed E6 (the wrapper writes no manifest)."""
    import copy
    import torch
    import models_deep as md
    models, source, compute = e1_models
    bad = {t: dict(m) for t, m in models.items()}
    ft = copy.deepcopy(models["TSV_3p"]["FTT"])
    with torch.no_grad():
        ft.model_.to_logits[2].bias[0] = float("nan")
    bad["TSV_3p"]["FTT"] = ft
    out = tmp_path / "out"
    with pytest.raises(md.NonFiniteOutputError):
        _run(tr, (bad, source, compute), features, out, tmp_path / "splits", targets=("TSV_3p",),
             strategies=("zero_shot", "ftt_head"))
    written = sorted(f.name for f in (out / "predictions").glob("*")) if (out / "predictions").exists() else []
    assert not [f for f in written if "__FTT_" in f]
    assert not (out / "split_summary.csv").exists() and not (out / "e6_run_record.json").exists()


def test_dry_run_is_reproducible_and_never_overwrites(tr, e1_models, features, tmp_path):
    _run(tr, e1_models, features, tmp_path / "a", tmp_path / "splits", targets=("TSV_3p",),
         strategies=("zero_shot", "local", "ftt_full"))
    _run(tr, e1_models, features, tmp_path / "b", tmp_path / "splits", targets=("TSV_3p",),
         strategies=("zero_shot", "local", "ftt_full"))
    fa = sorted((tmp_path / "a" / "predictions").glob("*.csv.gz"))
    assert fa and [f.name for f in fa] == [f.name for f in sorted((tmp_path / "b" / "predictions").glob("*.csv.gz"))]
    for f in fa:                                                        # identical bytes (gzip mtime 0)
        assert f.read_bytes() == (tmp_path / "b" / "predictions" / f.name).read_bytes(), f.name
        assert gzip.decompress(f.read_bytes())
    with pytest.raises(FileExistsError):
        _run(tr, e1_models, features, tmp_path / "a", tmp_path / "splits", targets=("TSV_3p",),
             strategies=("zero_shot",))


def test_run_guards(tr, e1_models, features, tmp_path):
    models, _, _ = e1_models
    with pytest.raises(ValueError, match="dry_run"):                    # a real run evaluates the 20 splits
        _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", dry_run=False)
    with pytest.raises(ValueError, match="unknown strategy"):
        _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", strategies=("FT_head",))
    for removed in ("pooled", "ftt_local"):                             # not part of the published E6
        with pytest.raises(ValueError, match="unknown strategy"):
            _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", strategies=(removed,))
    with pytest.raises(ValueError, match="unknown regime"):
        _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", regimes=("50-50",))
    with pytest.raises(ValueError, match="frozen artefact"):
        _run(tr, e1_models, features, REPO_DIR / "rerun_2026-04-28_no_reweighting" / "E6", tmp_path / "s")
    bad = {t: dict(m) for t, m in models.items()}
    bad["TSV_3p"]["FTT"] = None
    with pytest.raises(ValueError, match="FTT model missing"):
        _run(tr, (bad, *e1_models[1:]), features, tmp_path / "o", tmp_path / "s")
    import copy
    bad = {t: dict(m) for t, m in models.items()}
    bad["TSV_3p"]["RF"] = copy.deepcopy(models["TSV_3p"]["RF"]).set_params(clf__random_state=43)
    with pytest.raises(ValueError, match="zero-shot and FT sources are the"):
        _run(tr, (bad, *e1_models[1:]), features, tmp_path / "o", tmp_path / "s")
    assert not (tmp_path / "o").exists() and not (tmp_path / "s").exists()   # raised before any write


def test_load_source_rows_checks_the_e1_split(tmp_path, features, monkeypatch):
    import revision_folds
    df, ys = _ashrae(features, n=60)
    for t in rs.TARGETS:
        df[t] = np.resize(np.asarray(rs.ontology(t), dtype=float), len(df))
    csv = tmp_path / "ashrae.csv"
    df.to_csv(csv, index=False)
    split_dir, models_dir = tmp_path / "split", tmp_path / "e1"
    for t in TARGETS:
        trn, te = revision_folds.redraw_split(df[t])
        payload = {"target": t, "train_idx": trn.tolist(), "test_idx": te.tolist()}
        (split_dir).mkdir(exist_ok=True)
        (split_dir / f"{t}_split_indices.json").write_text(json.dumps(payload))
        (models_dir / t / "splits").mkdir(parents=True, exist_ok=True)
        (models_dir / t / "splits" / f"{t}_split_indices.json").write_text(json.dumps(payload))
    rows = p3.load_source_rows(csv, split_dir, TARGETS, features, models_dir=models_dir)
    trn, _ = revision_folds.redraw_split(df["TSV_3p"])
    assert rows["TSV_3p"][0].index.tolist() == trn.tolist() and len(rows["TSV_3p"][1]) == len(trn)
    e1 = models_dir / "TSV_3p" / "splits" / "TSV_3p_split_indices.json"
    payload = json.loads(e1.read_text())
    payload["train_idx"] = payload["train_idx"][::-1]
    e1.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="E1 split differs"):
        p3.load_source_rows(csv, split_dir, TARGETS, features, models_dir=models_dir)
    e1.unlink()
    with pytest.raises(FileNotFoundError):
        p3.load_source_rows(csv, split_dir, TARGETS, features, models_dir=models_dir)


# ================================================================ CLI (legacy evaluator replaced)
def test_legacy_adaptation_evaluator_is_replaced_by_the_e6_runner(tr):
    for name in ("stratified_dev_test", "finetune_eval_one_base", "run_external_suite",
                 "check_legacy_adaptation_request", "clone_ft_for_finetune", "LEGACY_SPLIT_MESSAGE"):
        assert not hasattr(tr, name), name
    assert tr.check_partition is p3.check_partition
    args = tr.parse_args(["--models_dir", "m", "--splits", "splits/phase3", "--output_dir", "o"])
    assert args.strategies == list(p3.STRATEGIES) and args.regimes == ["20-80"] and args.dry_run_splits == 0
    assert set(args.cohorts) == {"Moujalled", "Hostein"}
    with pytest.raises(SystemExit):                                     # legacy modes no longer parse
        tr.parse_args(["--models_dir", "m", "--splits", "s", "--output_dir", "o", "--finetune_modes", "FT_head"])


def test_cli_refuses_the_frozen_artefact_before_loading_anything(tr, monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_DIR)
    monkeypatch.setattr(tr, "load_external_cohort",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    for out, spl in (("rerun_2026-04-28_no_reweighting/E6", str(tmp_path / "s")),
                     (str(tmp_path / "o"), "rerun_2026-04-28_no_reweighting/splits")):
        with pytest.raises(ValueError, match="frozen artefact"):
            tr.main(["--models_dir", str(tmp_path), "--splits", spl, "--output_dir", out])
    with pytest.raises(FileNotFoundError, match="Models directory"):
        tr.main(["--models_dir", str(tmp_path / "absent"), "--splits", str(tmp_path / "s"),
                 "--output_dir", str(tmp_path / "o")])


def test_zero_shot_only_builds_no_prototype_and_needs_no_cuda(tr, e1_models, features, tmp_path, monkeypatch):
    """Zero-shot alone fits nothing: no prototype is built, so ``compute.xgb_device: cuda``
    on a machine without CUDA does not raise (as for the replaced legacy path)."""
    import compute as compute_module
    monkeypatch.setattr(compute_module, "cuda_available", lambda: False)

    def no_protos(*a, **k):
        raise AssertionError("prototype built for zero-shot")
    rec = _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", strategies=("zero_shot",),
               make_protos=no_protos, compute={**e1_models[2], "xgb_device": "cuda"})
    assert rec["n_prediction_files"] == 2 * len(TARGETS) * (3 + 2 * 3)


def test_missing_source_rows_raise_before_any_write(tr, e1_models, features, tmp_path):
    with pytest.raises(ValueError, match="ASHRAE training rows missing"):
        _run(tr, e1_models, features, tmp_path / "o", tmp_path / "s", source_rows={},
             strategies=("zero_shot",))
    assert not (tmp_path / "o").exists() and not (tmp_path / "s").exists()


# ================================================================ output-directory guards and CLI checks
def test_check_output_dir_allows_only_an_empty_directory_or_the_splits_inside(tmp_path):
    p3.check_output_dir(tmp_path / "absent", tmp_path / "s")
    out = tmp_path / "out"
    (out / "splits").mkdir(parents=True)
    p3.check_output_dir(out, out / "splits")                           # pre-serialised splits inside: allowed
    (out / "splits" / "x.json").write_text("{}")
    p3.check_output_dir(out, out / "splits")
    with pytest.raises(FileExistsError, match="not empty"):
        p3.check_output_dir(out, tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="must differ"):
        p3.check_output_dir(out, out)
    (out / "split_summary.csv").write_text("x")
    with pytest.raises(FileExistsError, match="split_summary.csv"):
        p3.check_output_dir(out, out / "splits")


def test_run_outputs_are_never_overwritten(tr, e1_models, features, tmp_path):
    import re
    import xgboost
    out, splits = tmp_path / "out", tmp_path / "splits"
    rec = _run(tr, e1_models, features, out, splits, targets=("TSV_3p",), strategies=("zero_shot", "local"))
    # provenance in the run record
    prov = rec["provenance"]
    assert prov["packages"]["xgboost"] == xgboost.__version__ and prov["python"]
    if (REPO_DIR / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"]) and isinstance(prov["git_tracked_changes"], bool)
    assert json.loads((out / "e6_run_record.json").read_text())["provenance"] == prov
    # a second run with other strategies into the same folder is refused before any write
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="one E6 run per output directory"):
        _run(tr, e1_models, features, out, splits, targets=("TSV_3p",), strategies=("local",))
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_cli_rejects_out_of_range_dry_run_splits(tr):
    base = ["--models_dir", "m", "--splits", "s", "--output_dir", "o"]
    assert tr.parse_args(base + ["--dry_run_splits", "3"]).dry_run_splits == 3
    for bad in ("-1", "21", "x"):
        with pytest.raises(SystemExit):
            tr.parse_args(base + ["--dry_run_splits", bad])


def test_cli_refuses_a_non_empty_output_dir_before_loading_anything(tr, monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "load_external_cohort",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("data loaded")))
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "e6_run_record.json").write_text("{}")
    with pytest.raises(FileExistsError, match="not empty"):
        tr.main(["--models_dir", str(tmp_path), "--splits", str(tmp_path / "s"),
                 "--output_dir", str(tmp_path / "o")])
    assert (tmp_path / "o" / "e6_run_record.json").read_text() == "{}"


@pytest.mark.skipif(not MOUJALLED.exists(), reason="French cohort absent")
def test_cli_writes_coverage_logs_only_after_the_run_succeeds(tr, monkeypatch, tmp_path):
    """Labels and categories of the real cohort only: models, source rows and run_e6 are stubs."""
    monkeypatch.chdir(REPO_DIR)
    monkeypatch.setattr(tr, "load_model_dicts", lambda d, **k: ({}, {}, {}))
    monkeypatch.setattr(p3, "load_source_rows", lambda *a, **k: ({}, [{"variable": "Season", "stub": True}]))
    # standalone transfert.py runs dry runs only
    args = ["--models_dir", str(tmp_path), "--splits", str(tmp_path / "s"), "--cohorts", "Moujalled",
            "--dry_run_splits", "1"]

    def fails(**kw):
        raise ValueError("input check failed")
    monkeypatch.setattr(p3, "run_e6", fails)
    with pytest.raises(ValueError, match="input check failed"):
        tr.main(args + ["--output_dir", str(tmp_path / "o1")])
    assert not (tmp_path / "o1").exists()                              # nothing written by a refused run
    monkeypatch.setattr(p3, "run_e6", lambda **kw: {"experiment": "E6_dry_run", "n_prediction_files": 0})
    tr.main(args + ["--output_dir", str(tmp_path / "o2")])
    assert (tmp_path / "o2" / "category_coverage_Moujalled.json").exists()
    # the ASHRAE source rows reloaded by E6 get their coverage log too
    assert json.loads((tmp_path / "o2" / "category_coverage_ASHRAE_2022.json").read_text()) == [
        {"variable": "Season", "stub": True}]
