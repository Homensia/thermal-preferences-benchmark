"""Blocking tests of the PMV / aPMV contract (experiment E7; PMV indices fix).

All data are synthetic; no model is trained. Run from the repository root:
    .venv-revision/bin/python -m pytest tests/test_pmv_indices.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedShuffleSplit

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import indicateurs_classiques as ic  # noqa: E402
from pythermalcomfort.models import pmv_ppd_iso  # noqa: E402

BASE = {"Tair": 24.0, "tr": 24.0, "vel": 0.1, "RH": 50.0, "Met": 1.2, "clo": 0.7}


def frame(*rows) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **r} for r in rows])


def raw_pmv(row: dict) -> float:
    """PMV without the applicability mask (limit_inputs=False)."""
    r = {**BASE, **row}
    return float(pmv_ppd_iso(tdb=r["Tair"], tr=r["tr"], vr=r["vel"], rh=r["RH"],
                             met=r["Met"], clo=r["clo"], wme=0, model="7730-2005",
                             units="SI", limit_inputs=False, round_output=False).pmv)


# ---------------------------------------------------------------------------
# Pinned version and source hash
# ---------------------------------------------------------------------------
def test_pin_guard_accepts_the_locked_library():
    prov = ic.check_pythermalcomfort_pin()
    assert prov["pythermalcomfort_version"] == "4.0.2"
    assert prov["pmv_ppd_iso_sha256"] == ic.PINNED_PMV_PPD_ISO_SHA256
    assert prov["clo_dynamic_iso_defaults"] == {"i_a": 0.7, "model": "9920-2007"}
    assert prov["pmv_call"] == {"wme": 0, "model": "7730-2005", "units": "SI",
                                "limit_inputs": True, "round_output": False}


@pytest.mark.parametrize("field,value", [
    ("pythermalcomfort_version", "3.9.3"),
    ("pmv_ppd_iso_sha256", "0" * 64),
    ("clo_dynamic_iso_defaults", {"i_a": 0.7, "model": "other"}),
])
def test_pin_guard_refuses_a_different_library(field, value):
    prov = dict(ic.pythermalcomfort_provenance(), **{field: value})
    with pytest.raises(ic.PythermalcomfortPinError):
        ic.check_pythermalcomfort_pin(prov)


def test_compute_pmv_refuses_to_run_on_hash_mismatch(monkeypatch):
    monkeypatch.setattr(ic, "PINNED_PMV_PPD_ISO_SHA256", "f" * 64)
    with pytest.raises(ic.PythermalcomfortPinError):
        ic.compute_pmv(frame({}))


# ---------------------------------------------------------------------------
# Individual inclusive input predicates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col", list(ic.INPUT_BOUNDS))
def test_each_input_predicate_is_inclusive(col):
    lo, hi = ic.INPUT_BOUNDS[col]
    eps = 1e-9
    df = frame({col: lo}, {col: hi}, {col: lo - eps}, {col: hi + eps}, {col: np.nan})
    assert ic.input_predicates(df)[col].tolist() == [True, True, False, False, False]


def test_pmv_range_predicate_is_inclusive():
    assert ic.pmv_in_range([-2.0, 2.0, -2.0 - 1e-9, 2.0 + 1e-9, np.nan]).tolist() == [
        True, True, False, False, False]


def test_frozen_bounds_match_the_pinned_library_source():
    src = (Path(ic.pythermalcomfort.__file__).parent / "models" / "pmv_ppd_iso.py").read_text()
    for snippet in ("valid_range(tdb, (10.0, 30.0))", "valid_range(tr, (10.0, 40.0))",
                    "valid_range(vr, (0.0, 1.0))", "valid_range(met, (0.8, 4.0))",
                    "valid_range(clo, (0.0, 2.0))", "valid_range(pmv, (-2, 2))"):
        assert snippet in src


# ---------------------------------------------------------------------------
# End-to-end validity on each bound (PMV itself inside [-2, 2])
# ---------------------------------------------------------------------------
BOUND_FIXTURES = [
    {"Tair": 10.0, "tr": 30.0, "clo": 1.5, "Met": 1.6},
    {"Tair": 30.0, "tr": 24.0, "clo": 0.3, "Met": 1.0, "vel": 0.3},
    {"tr": 10.0, "Tair": 26.0, "clo": 1.0},
    {"tr": 40.0, "Tair": 20.0, "clo": 0.5},
    {"vel": 0.0},
    {"vel": 1.0, "Tair": 27.0, "tr": 27.0, "clo": 0.5},
    {"Met": 0.8, "Tair": 26.0, "tr": 26.0, "clo": 1.0},
    {"Met": 4.0, "Tair": 18.0, "tr": 18.0, "clo": 0.3, "vel": 0.8},
    {"clo": 0.0, "Tair": 28.0, "tr": 28.0, "Met": 1.4},
    {"clo": 2.0, "Tair": 18.0, "tr": 18.0, "Met": 1.0},
]


@pytest.mark.parametrize("fixture", BOUND_FIXTURES)
def test_input_exactly_on_its_bound_is_valid_end_to_end(fixture):
    expected = raw_pmv(fixture)
    assert -2.0 <= expected <= 2.0  # the fixture satisfies the PMV predicate too
    assert ic.input_predicates(frame(fixture)).all(axis=1).item()
    pmv = ic.compute_pmv(frame(fixture))
    assert np.isfinite(pmv[0]) and pmv[0] == expected  # unrounded


def test_inputs_pass_but_pmv_out_of_range_is_invalid_not_clipped():
    row = {"Tair": 10.0, "tr": 25.0, "vel": 0.1, "RH": 50.0, "Met": 1.2, "clo": 0.5}
    assert ic.input_predicates(frame(row)).all(axis=1).item()
    assert raw_pmv(row) < -2.0  # ≈ -2.71 with limit_inputs=False
    assert np.isnan(ic.compute_pmv(frame(row))[0])


def test_every_vote_above_30C_is_invalid():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"Tair": rng.uniform(30.0 + 1e-9, 40.0, n),
                       "tr": rng.uniform(10, 40, n), "vel": rng.uniform(0, 1, n),
                       "RH": rng.uniform(10, 90, n), "Met": rng.uniform(0.8, 2, n),
                       "clo": rng.uniform(0, 1.5, n)})
    assert np.isnan(ic.compute_pmv(df)).all()


def test_no_default_and_no_clipping_of_inputs():
    with pytest.raises(KeyError):
        ic.compute_pmv(frame({}).drop(columns="clo"))
    df = frame({"RH": np.nan}, {"vel": 1.5}, {"Met": 0.7}, {"clo": 2.1}, {"tr": np.nan})
    assert np.isnan(ic.compute_pmv(df)).all()  # never filled, never clipped


def test_dynamic_sensitivity_has_its_own_mask():
    df = frame({}, {"vel": 0.95, "Met": 2.0})  # v_relative = 0.95 + 0.3 > 1
    prim, dyn = ic.compute_pmv(df, "primary"), ic.compute_pmv(df, "dynamic")
    assert np.isfinite(prim).all()
    assert np.isfinite(dyn[0]) and np.isnan(dyn[1])


# ---------------------------------------------------------------------------
# Conversion on the unrounded PMV
# ---------------------------------------------------------------------------
def test_unrounded_floor_rule_and_nan_never_a_class():
    x = [0.499, 0.5, -0.5, -0.501, 1.5, -1.5, np.nan]
    assert list(ic.pmv_to_class(x, "thermal_sensation")) == [0, 1, 0, -1, 2, -1, pd.NA]
    assert list(ic.pmv_to_class(x, "TSV_3p")) == [0, 1, 0, -1, 1, -1, pd.NA]
    tpv = ic.pmv_to_class([0.5, 0.5001, -0.5, -0.5001, 0.0, np.nan], "thermal_preference")
    assert list(tpv) == [0, -1, 0, 1, 0, pd.NA]


def test_pmv_class_outside_ontology_raises_instead_of_clipping():
    with pytest.raises(ValueError):
        ic.pmv_to_class([2.6], "thermal_sensation")


# ---------------------------------------------------------------------------
# aPMV
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pmv,lam,value,cls,tpv", [
    (-2.0, 0.2, -10 / 3, -3, 1),
    (2.0, -0.2, 10 / 3, 3, -1),
    (2.0, 0.2, 10 / 7, 1, -1),
])
def test_apmv_values_outside_the_pmv_range(pmv, lam, value, cls, tpv):
    a = ic.apmv([pmv], lam)
    assert a[0] == pytest.approx(value, abs=1e-12)
    assert ic.apmv_to_class(a, "thermal_sensation")[0] == cls
    assert ic.apmv_to_class(a, "thermal_preference")[0] == tpv


@pytest.mark.parametrize("lam", [0.4, 0.5, -0.4, -0.5, 0.2000001, np.nan])
def test_apmv_out_of_domain_lambda_raises(lam):
    with pytest.raises(ValueError):
        ic.apmv([-2.0, 0.0, 2.0], lam)


def test_apmv_class_outside_ontology_raises():
    with pytest.raises(ValueError):
        ic.apmv_to_class([3.6], "thermal_sensation")


def test_apmv_nan_pmv_gives_nan_and_no_class():
    a = ic.apmv([np.nan, 1.0], 0.1)
    assert np.isnan(a[0]) and np.isfinite(a[1])
    assert ic.apmv_to_class(a, "thermal_sensation").isna().tolist() == [True, False]


def _synthetic_pmv(n=500, seed=1):
    rng = np.random.default_rng(seed)
    pmv = rng.uniform(-2, 2, n)
    pmv[rng.random(n) < 0.3] = np.nan  # outside the mask
    idx = rng.permutation(n)
    return pmv, idx[: int(0.8 * n)], idx[int(0.8 * n):]


def test_fit_constrained_at_bound_when_unconstrained_optimum_beyond():
    rng = np.random.default_rng(2)
    p = rng.uniform(-1.0, 2.0, 300)
    y = p / (1 + 0.6 * p)  # exact fit at lam = 0.6
    unc = minimize_scalar(lambda l: np.sum((y - p / (1 + l * p)) ** 2),
                          method="bounded", bounds=(0.21, 0.95), options={"xatol": 1e-10})
    assert unc.x >= 0.5
    cal = ic.fit_apmv_lambda(p, y, np.arange(300), np.array([], dtype=int))
    assert cal.available and cal.lam == 0.2 and cal.bound_active is True
    assert np.isfinite(cal.sse) and np.isfinite(ic.apmv(p, cal.lam)).all()


def test_fit_interior_solution_not_flagged():
    rng = np.random.default_rng(3)
    p = rng.uniform(-2, 2, 400)
    y = p / (1 + 0.05 * p) + rng.normal(0, 0.05, 400)
    cal = ic.fit_apmv_lambda(p, y, np.arange(400), np.array([], dtype=int))
    assert cal.available and abs(cal.lam - 0.05) < 0.01 and cal.bound_active is False
    grid_best = min(ic._sse(l, p, y) for l in ic.APMV_GRID)
    assert cal.sse <= grid_best


def test_only_train_rows_inside_mask_enter_the_fit():
    pmv, train, test = _synthetic_pmv()
    y = np.floor(np.nan_to_num(pmv) + 0.5)
    rows = ic.apmv_calibration_rows(pmv, train, test)
    assert set(rows) == {i for i in train if np.isfinite(pmv[i])}
    assert not set(rows) & set(test)
    cal = ic.fit_apmv_lambda(pmv, y, train, test)
    assert cal.n_calibration == len(rows)
    # Labels of test rows and of out-of-mask training rows are never read.
    y2 = y.copy()
    y2[test] = 99.0
    y2[train[~np.isfinite(pmv[train])]] = np.nan
    assert ic.fit_apmv_lambda(pmv, y2, train, test).lam == cal.lam
    # An injected test row raises.
    test_valid = test[np.isfinite(pmv[test])]
    with pytest.raises(ic.CalibrationLeakageError):
        ic.fit_apmv_lambda(pmv, y, train, test, calibration_idx=np.r_[rows, test_valid[:1]])
    # An out-of-mask or missing row also raises.
    with pytest.raises(ic.CalibrationLeakageError):
        ic.fit_apmv_lambda(pmv, y, train, test, calibration_idx=rows[1:])
    with pytest.raises(ic.CalibrationLeakageError):
        ic.apmv_calibration_rows(pmv, train, np.r_[test, train[:1]])


def test_fit_is_bitwise_deterministic():
    pmv, train, test = _synthetic_pmv(seed=4)
    rng = np.random.default_rng(5)
    y = np.clip(np.floor(np.nan_to_num(pmv) + rng.normal(0, 0.8, pmv.size) + 0.5), -3, 3)
    a = ic.fit_apmv_lambda(pmv, y, train, test)
    b = ic.fit_apmv_lambda(pmv, y, train, test)
    assert np.float64(a.lam).tobytes() == np.float64(b.lam).tobytes()
    assert a.sse == b.sse and a.calibration_idx_sha256 == b.calibration_idx_sha256


def test_failure_path_returns_not_available(monkeypatch):
    pmv, train, test = _synthetic_pmv(n=100)  # < 100 calibration rows
    cal = ic.fit_apmv_lambda(pmv, np.zeros(100), train, test)
    assert cal.status == ic.NOT_AVAILABLE and cal.lam is None and not cal.available
    pmv, train, test = _synthetic_pmv()

    def boom(p, y):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(ic, "_fit_lambda", boom)
    cal = ic.fit_apmv_lambda(pmv, np.zeros(pmv.size), train, test)
    assert cal.status == ic.NOT_AVAILABLE and cal.lam is None and "synthetic" in cal.reason



@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_refinement_sse_is_not_available(monkeypatch, bad):
    """Finite grid SSEs then a non-finite refinement SSE: no grid fallback."""
    pmv, train, test = _synthetic_pmv()
    y = np.clip(np.floor(np.nan_to_num(pmv) + 0.5), -3, 3)
    n_grid = ic.APMV_GRID.size
    calls = {"n": 0}

    def fake_sse(lam, p, yy):
        calls["n"] += 1
        return 1.0 if calls["n"] <= n_grid else float(bad)

    monkeypatch.setattr(ic, "_sse", fake_sse)
    monkeypatch.setattr(ic, "minimize_scalar",
                        lambda f, **kw: type("R", (), {"x": 0.0, "success": True})())
    cal = ic.fit_apmv_lambda(pmv, y, train, test)
    assert cal.status == ic.NOT_AVAILABLE and not cal.available
    assert cal.lam is None and cal.sse is None and cal.bound_active is None
    assert "non-finite SSE" in cal.reason


def test_finite_but_worse_refinement_keeps_the_grid_candidate(monkeypatch):
    pmv, train, test = _synthetic_pmv()
    y = np.clip(np.floor(np.nan_to_num(pmv) + 0.5), -3, 3)
    n_grid = ic.APMV_GRID.size
    calls = {"n": 0}

    def fake_sse(lam, p, yy):
        calls["n"] += 1
        return 1.0 if calls["n"] <= n_grid else 2.0

    monkeypatch.setattr(ic, "_sse", fake_sse)
    monkeypatch.setattr(ic, "minimize_scalar",
                        lambda f, **kw: type("R", (), {"x": 0.013, "success": True})())
    cal = ic.fit_apmv_lambda(pmv, y, train, test)
    assert cal.status == "ok" and cal.lam == 0.0 and cal.sse == 1.0


def _nan_on_first_refinement_eval(monkeypatch):
    """Real SSE, except NaN at the first off-grid evaluation after a grid sweep.

    Reproduces the failure mode: with the real SciPy optimizer, a single NaN
    inside the refinement makes SciPy report success=False while every later
    evaluation (including a fresh SSE at the returned x) is finite.
    """
    real_sse, grid = ic._sse, set(ic.APMV_GRID.tolist())
    state = {"armed": False, "injected": 0}

    def sse(lam, p, y):
        if float(lam) in grid:
            state["armed"] = True
        elif state["armed"]:
            state["armed"] = False
            state["injected"] += 1
            return float("nan")
        return real_sse(lam, p, y)

    monkeypatch.setattr(ic, "_sse", sse)
    return state


def test_nan_inside_real_refinement_is_not_available(monkeypatch):
    pmv, train, test = _synthetic_pmv()
    y = np.clip(np.floor(np.nan_to_num(pmv) + 0.5), -3, 3)
    assert ic.fit_apmv_lambda(pmv, y, train, test).status == "ok"
    state = _nan_on_first_refinement_eval(monkeypatch)
    cal = ic.fit_apmv_lambda(pmv, y, train, test)
    assert state["injected"] == 1
    assert cal.status == ic.NOT_AVAILABLE and not cal.available
    assert cal.lam is None and cal.sse is None and cal.bound_active is None
    assert "non-finite SSE" in cal.reason


def test_unsuccessful_optimizer_is_not_available(monkeypatch):
    """success=False with a finite x and SSE: no grid fallback either."""
    pmv, train, test = _synthetic_pmv()
    y = np.clip(np.floor(np.nan_to_num(pmv) + 0.5), -3, 3)
    monkeypatch.setattr(ic, "minimize_scalar", lambda f, **kw: type(
        "R", (), {"x": 0.0, "success": False, "message": "synthetic failure"})())
    cal = ic.fit_apmv_lambda(pmv, y, train, test)
    assert cal.status == ic.NOT_AVAILABLE and cal.lam is None
    assert "synthetic failure" in cal.reason


# ---------------------------------------------------------------------------
# E7 runner on a synthetic cohort: identical masks, ML on the same rows
# ---------------------------------------------------------------------------
def _cohort(n=600, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Tair": rng.uniform(15, 34, n), "tr": rng.uniform(12, 38, n),
        "vel": rng.uniform(0, 1.3, n), "RH": rng.uniform(20, 80, n),
        "Met": rng.uniform(0.9, 1.8, n), "clo": rng.uniform(0.3, 1.2, n),
        "Date": "2020-01-01",
    })
    raw = np.asarray(pmv_ppd_iso(tdb=df.Tair, tr=df.tr, vr=df.vel, rh=df.RH, met=df.Met,
                                 clo=df.clo, limit_inputs=False, round_output=False).pmv)
    tsv = np.clip(np.floor(raw + rng.normal(0, 0.7, n) + 0.5), -3, 3).astype(int)
    df["thermal_sensation"] = tsv
    df["TSV_3p"] = np.sign(tsv).astype(int)
    df["thermal_preference"] = rng.integers(-1, 2, n)
    return df, _redraw_splits(df)


def _redraw_splits(df):
    """The frozen split: StratifiedShuffleSplit(test_size=0.2, seed 42)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return {t: next(sss.split(np.zeros(len(df)), df[t].to_numpy())) for t in ic.TARGETS}


def _ml(df, splits, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t, (_, test) in splits.items():
        out.append(pd.DataFrame({"experiment": "E1", "target": t, "model": "RF", "seed": 42,
                                 "row_idx": test[::-1], "y_true": df[t].to_numpy()[test[::-1]],
                                 "y_pred": rng.choice(ic.ONTOLOGY[t], test.size)}))
    return pd.concat(out, ignore_index=True)


def test_run_e7_masks_and_ml_comparison_rows():
    df, splits = _cohort()
    res = ic.run_e7(df, splits, ml_predictions=_ml(df, splits))
    pred, met, cov = res["predictions"], res["metrics"], res["coverage"]
    assert set(pred["model"]) == {"PMV", "aPMV", "PMV_dynamic"}  # no ePMV / PTS_*
    for t, (_, test) in splits.items():
        p = pred[(pred.target == t) & (pred.model == "PMV")]
        a = pred[(pred.target == t) & (pred.model == "aPMV")]
        assert p.row_idx.tolist() == test.tolist() == a.row_idx.tolist()
        assert (p.pmv_valid.to_numpy() == a.pmv_valid.to_numpy()).all()  # aPMV mask = PMV mask
        assert (p.y_pred.isna().to_numpy() == ~p.pmv_valid.to_numpy()).all()
        assert (a.y_pred.isna().to_numpy() == ~p.pmv_valid.to_numpy()).all()
        assert not p.loc[df.Tair.to_numpy()[test] > 30, "pmv_valid"].any()
        n_valid = int(p.pmv_valid.sum())
        c = cov[(cov.target == t) & (cov.pmv_variant == "primary")].iloc[0]
        assert c.n_valid == n_valid and c.n_test == test.size
        m = met[(met.target == t) & (met.pmv_variant == "primary")]
        on = m[m.subset == "on_mask"].set_index("method")
        assert on.loc["PMV", "n"] == on.loc["aPMV", "n"] == n_valid
        assert on.loc["ML:experiment=E1/model=RF/seed=42", "n"] == n_valid
        out = m[(m.subset == "outside_mask")].iloc[0]
        assert out.n == test.size - n_valid
        assert res["calibration"][t]["status"] == "ok"
        assert abs(res["calibration"][t]["lam"]) <= 0.2


def test_run_e7_calibration_ignores_test_labels():
    df, splits = _cohort()
    lam = {t: c["lam"] for t, c in ic.run_e7(df, splits)["calibration"].items()}
    # TSV-7 test labels of the TSV-7 target cannot change without changing its
    # stratified re-draw (the split guard then raises); test-label isolation for
    # that target is covered by test_only_train_rows_inside_mask_enter_the_fit.
    for t in ("TSV_3p", "thermal_preference"):
        test = splits[t][1]
        # Flip the TSV-7 labels (aPMV response) of this target's test rows only.
        df2 = df.copy()
        df2.loc[test, "thermal_sensation"] = -df2.loc[test, "thermal_sensation"]
        res = ic.run_e7(df2, {t: splits[t]}, variants=["primary"])
        assert res["calibration"][t]["lam"] == lam[t]


def test_run_e7_rejects_ml_rows_not_matching_the_test_set():
    df, splits = _cohort()
    ml = _ml(df, splits)
    ml = ml.drop(index=ml.index[0])
    with pytest.raises(ValueError):
        ic.run_e7(df, splits, ml_predictions=ml)


# ---------------------------------------------------------------------------
# Archived split identity, checked before calibration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corruption", ["omitted", "substituted", "reordered_train",
                                        "reordered_test", "overlap", "out_of_range"])
def test_archived_split_must_equal_the_frozen_redraw(corruption):
    df, splits = _cohort(n=300)
    t = "thermal_sensation"
    train, test = (a.copy() for a in splits[t])
    ic.check_archived_split(train, test, df[t], t)  # the frozen split passes
    if corruption == "omitted":          # a valid training row silently removed
        train = train[1:]
    elif corruption == "substituted":    # one train row swapped with one test row
        train[0], test[0] = test[0], train[0]
    elif corruption == "reordered_train":
        train = train[::-1].copy()
    elif corruption == "reordered_test":
        test = np.roll(test, 1)
    elif corruption == "overlap":
        test = np.r_[test, train[:1]]
    else:
        train = np.r_[train[1:], len(df)]
    with pytest.raises(ValueError, match="re-draw|partition|overlap"):
        ic.check_archived_split(train, test, df[t], t)
    with pytest.raises(ValueError):  # run_e7 refuses before any calibration
        ic.run_e7(df, {t: (train, test)}, variants=["primary"])


def test_run_e7_rejects_a_split_missing_training_rows(monkeypatch):
    df, splits = _cohort()
    called = []
    monkeypatch.setattr(ic, "fit_apmv_lambda", lambda *a, **k: called.append(1))
    train, test = splits["thermal_sensation"]
    with pytest.raises(ValueError, match="partition"):
        ic.run_e7(df, {"thermal_sensation": (train[:-10], test)}, variants=["primary"])
    assert not called


# ---------------------------------------------------------------------------
# ML comparison: malformed predictions raise instead of becoming classes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("y_pred", [[-1, 0, np.nan], [-1, 0, 99], [-1.9, 0.8, 1.2],
                                    [-1, 0, np.inf]])
def test_metrics_reject_malformed_predictions(y_pred):
    with pytest.raises(ValueError):
        ic.classification_metrics(np.array([-1, 0, 1]), np.array(y_pred), "TSV_3p")


def test_metrics_valid_float_labels_and_empty_set():
    m = ic.classification_metrics(np.array([-1, 0, 1]), np.array([-1.0, 0.0, 0.0]), "TSV_3p")
    assert m["n"] == 3 and m["accuracy"] == pytest.approx(2 / 3) and m["qwk"] < 1
    e = ic.classification_metrics(np.array([], dtype=int), np.array([], dtype=int), "TSV_3p")
    assert e["n"] == 0 and np.isnan(e["accuracy"]) and e["qwk_undefined"]
    with pytest.raises(ValueError):
        ic.classification_metrics(np.array([], dtype=int), np.array([0]), "TSV_3p")


@pytest.mark.parametrize("bad", [np.nan, 99, 0.8, 5])
@pytest.mark.parametrize("column", ["y_pred", "y_true"])
def test_run_e7_rejects_malformed_ml_labels(bad, column):
    df, splits = _cohort()
    ml = _ml(df, splits)
    ml[column] = ml[column].astype(float)
    ml.loc[ml.index[ml.target == "TSV_3p"][0], column] = bad
    with pytest.raises(ValueError):
        ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"])


# --------------------------------------------- ML inventory
def _ml_multi(df, splits):
    """Two ML groups per target (RF seed 42, XGB seed 42)."""
    a = _ml(df, splits, seed=0)
    b = _ml(df, splits, seed=1).assign(model="XGB")
    return pd.concat([a, b], ignore_index=True)


def test_run_e7_rejects_a_missing_target():
    df, splits = _cohort(n=120)
    ml = _ml(df, splits)
    ml = ml[ml.target != "TSV_3p"]                                 # reproduction
    with pytest.raises(ValueError, match=r"no ML predictions for target\(s\) \['TSV_3p'\]"):
        ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"])


def test_run_e7_rejects_a_model_missing_for_one_target():
    df, splits = _cohort(n=120)
    ml = _ml_multi(df, splits)
    ml = ml[~((ml.target == "thermal_preference") & (ml.model == "XGB"))]
    with pytest.raises(ValueError, match="differ between targets"):
        ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"])


def test_run_e7_rejects_a_seed_missing_for_one_target():
    df, splits = _cohort(n=120)
    ml = _ml_multi(df, splits)
    extra = ml[ml.model == "RF"].assign(seed=43)
    extra = extra[extra.target != "TSV_3p"]                        # seed 43 absent for TSV-3
    with pytest.raises(ValueError, match="differ between targets"):
        ic.run_e7(df, splits, ml_predictions=pd.concat([ml, extra]), variants=["primary"])


def test_run_e7_checks_the_expected_inventory():
    df, splits = _cohort(n=120)
    ml = _ml_multi(df, splits)
    with pytest.raises(ValueError, match="differs from the expected"):   # a model absent everywhere
        ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"],
                  expected_ml=["RF:42", "XGB:42", "SVM:42"])
    res = ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"],
                    expected_ml=["XGB:42", "RF:42"])
    on = res["metrics"][res["metrics"].subset == "on_mask"]
    assert on.groupby("target")["method"].apply(lambda m: m.str.startswith("ML:").sum()).eq(2).all()
    with pytest.raises(ValueError, match="without ml_predictions"):
        ic.run_e7(df, splits, variants=["primary"], expected_ml=["RF:42"])


def test_run_e7_rejects_ml_for_a_target_not_evaluated():
    df, splits = _cohort(n=120)
    with pytest.raises(ValueError, match="not evaluated"):
        ic.run_e7(df, {"TSV_3p": splits["TSV_3p"]}, ml_predictions=_ml(df, splits),
                  variants=["primary"])


def test_check_ml_inventory_seed_labels():
    ml = pd.DataFrame({"target": ["TSV_3p"] * 2, "model": ["SVM", "RF"],
                       "seed": [np.nan, 42.0], "row_idx": [0, 0]})
    assert ic.check_ml_inventory(ml, ["TSV_3p"]) == {"TSV_3p": ["RF:42", "SVM:NA"]}


def test_cli_requires_expected_ml_with_ml_predictions(tmp_path):
    with pytest.raises(SystemExit):
        ic.main(["--pmv_only", "--ml_predictions", str(tmp_path), str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_run_e7_requires_ml_y_true():
    df, splits = _cohort()
    with pytest.raises(ValueError, match="y_true"):
        ic.run_e7(df, splits, ml_predictions=_ml(df, splits).drop(columns="y_true"),
                  variants=["primary"])


def test_run_e7_rejects_misaligned_ml_rows():
    df, splits = _cohort()
    ml = _ml(df, splits)
    sub = ml.index[ml.target == "TSV_3p"]
    # Same row set, but y_true/y_pred shifted by one row: alignment check fails.
    ml.loc[sub, "row_idx"] = np.roll(ml.loc[sub, "row_idx"].to_numpy(), 1)
    if (df["TSV_3p"].to_numpy()[ml.loc[sub, "row_idx"]] == ml.loc[sub, "y_true"]).all():
        pytest.skip("synthetic labels coincide after the shift")
    with pytest.raises(ValueError, match="y_true differs"):
        ic.run_e7(df, splits, ml_predictions=ml, variants=["primary"])


def test_run_e7_not_available_gives_no_apmv_class():
    df, splits = _cohort(n=120)  # < 100 valid calibration rows
    res = ic.run_e7(df, splits, variants=["primary"])
    assert "aPMV" not in set(res["predictions"]["model"])
    na = res["metrics"][res["metrics"].method == "aPMV"]
    assert len(na) == 3 and (na.status == ic.NOT_AVAILABLE).all()
    assert all(c["status"] == ic.NOT_AVAILABLE for c in res["calibration"].values())


def test_run_e7_refinement_nan_gives_no_apmv_predictions(monkeypatch):
    df, splits = _cohort()
    state = _nan_on_first_refinement_eval(monkeypatch)
    res = ic.run_e7(df, splits, variants=["primary"])
    assert state["injected"] == 3  # one failed fit per target
    assert "aPMV" not in set(res["predictions"]["model"])
    na = res["metrics"][res["metrics"].method == "aPMV"]
    assert len(na) == 3 and (na.status == ic.NOT_AVAILABLE).all()
    for c in res["calibration"].values():
        assert c["status"] == ic.NOT_AVAILABLE and c["lam"] is None
        assert "non-finite SSE" in c["reason"]


def test_cli_requires_pmv_only_and_writes_outputs(tmp_path):
    df, splits = _cohort(n=300)
    csv = tmp_path / "cohort.csv"
    df.to_csv(csv, index=False)
    sdir = tmp_path / "splits"
    sdir.mkdir()
    import json
    for t, (tr, te) in splits.items():
        (sdir / f"{t}_split_indices.json").write_text(
            json.dumps({"target": t, "train_idx": tr.tolist(), "test_idx": te.tolist()}))
    with pytest.raises(SystemExit):
        ic.main(["--data_csv", str(csv), "--splits", str(sdir), "--skip_group_ids",
                 str(tmp_path / "out")])
    assert ic.main(["--pmv_only", "--data_csv", str(csv), "--splits", str(sdir),
                    "--skip_group_ids", "--dry_run", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    for f in ("metrics.csv", "coverage.csv", "apmv_calibration.json", "provenance.json"):
        assert (out / f).is_file()
    pred = pd.read_csv(out / "predictions" / "E7_dry_run__ASHRAE_2022__thermal_sensation__PMV.csv.gz")
    for col in ("experiment", "row_idx", "y_true", "y_pred", "p_-3", "p_3", "pmv", "pmv_valid"):
        assert col in pred.columns
    # an occupied output directory is never overwritten
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="never overwritten"):
        ic.main(["--pmv_only", "--data_csv", str(csv), "--splits", str(sdir),
                 "--skip_group_ids", "--dry_run", str(out)])
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_standalone_cli_runs_dry_runs_only(tmp_path):
    """A real E7 needs run_revision.py e7 (validated real E1 manifest); --skip_group_ids is for dry
    runs only; both refused before any data is read."""
    for extra in (["--skip_group_ids"], []):                    # real run: refused without the E1 manifest
        with pytest.raises(SystemExit):
            ic.main(["--pmv_only", "--data_csv", str(tmp_path / "absent.csv"), *extra, str(tmp_path / "out")])
    for man in ({"experiment": "E1_dry_run", "dry_run": True}, {"experiment": "E1"}):
        with pytest.raises(SystemExit):
            ic.main(["--pmv_only", "--data_csv", str(tmp_path / "absent.csv"), str(tmp_path / "out")],
                    e1_manifest=man)
    with pytest.raises(SystemExit):                              # never with a real E1 either
        ic.main(["--pmv_only", "--skip_group_ids", str(tmp_path / "out")],
                e1_manifest={"experiment": "E1", "dry_run": False})
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Tiny real-data subset (no result computed): the frozen-predicate guard holds
# ---------------------------------------------------------------------------
REAL_CSV = REPO_DIR / "Data" / "ASHRAE_2022_Clean_api.csv"


@pytest.mark.skipif(not REAL_CSV.is_file(), reason="canonical cohort CSV absent")
def test_guard_holds_on_a_tiny_real_subset():
    df = pd.read_csv(REAL_CSV, nrows=200)
    pmv = ic.compute_pmv(df)  # raises if the library mask != frozen predicate
    assert pmv.shape == (200,)
    assert np.isnan(pmv[df["Tair"].to_numpy() > 30]).all()
    ic.compute_pmv(df, "dynamic")


@pytest.mark.skipif(not REAL_CSV.is_file(), reason="canonical cohort CSV absent")
def test_archived_splits_pass_the_guard_on_the_real_cohort():
    df = pd.read_csv(REAL_CSV)
    for t in ic.TARGETS:
        train, test = ic.load_split(REPO_DIR / ic.DEFAULT_SPLITS_DIR, t)
        ic.check_archived_split(train, test, df[t], t)  # read-only, no PMV computed
        assert (train.size, test.size) == (12179, 3045)
