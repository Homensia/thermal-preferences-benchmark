"""Aggregate tables of the paper, computed from the hash-verified per-vote predictions of the run.

Each table is recomputed from the predictions files that the experiments wrote (one file per model, seed and
split, listed with its SHA-256 in the experiment's ``run_manifest.json``) and from the run inputs (cohort CSV,
archived split, Phase 3 split files, French cohorts), with the metric functions of ``revision_stats``:

* ``E1_metrics`` / ``E1_confusion_matrices``: one row per target x model x seed on the archived test votes
  (``revision_stats.classification_metrics``), and the same confusion matrices in long form;
* ``E6_full_cohort_zero_shot``: the zero-shot models on every vote of each French cohort, with the ASHRAE
  training majority class and the cohort majority class;
* ``E6_strategy_metrics``: each Phase 3 strategy pooled over the 20 date-grouped (development, test) splits of
  each regime, with the majority-class baselines (ASHRAE training majority, majority of each development split,
  majority of the test votes);
* ``E7_metrics``: PMV and aPMV (primary variant) on their validity mask;
* ``E2_aggregates``: leave-one-publication-out accuracy, pooled over every vote (micro) and averaged over the
  held-out publications (macro), with both majority baselines;
* ``E4_ema_gains``: paired accuracy gain of the 17-variable set over the 12-variable set (RF, XGBoost; mean over
  the three seeds) with its building-cluster bootstrap 95 % interval (``revision_stats.paired_accuracy_contrast``);
* ``E1_model_differences``: paired accuracy difference of each Phase 2 model (E1 families and, when E10 is
  complete, the two hybrids) from XGBoost on the archived test votes, each model averaged over its seeds, with
  the same building-cluster bootstrap 95 % interval as the E4 gain.

Every predictions file is checked before use: experiment, cohort, target, model, seed, split and feature set
columns, ``row_idx`` equal to the serialised split rows (order included) and ``y_true`` equal to the cohort
votes at those rows.

The tables are returned as they are read back from CSV (``to_csv`` then ``read_csv``), so that the figures and
tables see the same parsed values whether the aggregate is computed here or read from a file.
"""
from __future__ import annotations

import io
import json
import math
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .inputs import REPO_ROOT, RunInputs

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import revision_stats as rs  # noqa: E402

COHORT = "ASHRAE_2022"
ARCHIVED = "archived_80_20"
E1_MODELS = ("RF", "XGB", "SVM", "ANN", "FTT")
#: E1 seeds of the paper: three seeds per family, SVM fitted once (seed 42); run with
#: ``run_revision.py e1 --ann_seeds 42 43 44 --ftt_seeds 42 43 44``
E1_SEEDS = {"RF": (42, 43, 44), "XGB": (42, 43, 44), "SVM": (42,), "ANN": (42, 43, 44), "FTT": (42, 43, 44)}
TREE = ("RF", "XGB")
NA_TEXT = rs.NOT_AVAILABLE
DESCRIPTIVE_LEVEL = 0.95
#: E6 model names reported (phase3_runner.model_name), zero-shot first
E6_MODELS = ("RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot", "RF_local", "XGB_local", "FTT_head", "FTT_full")
E6_ZERO_SHOT = ("RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot")
E6_FTT_ADAPTED = ("FTT_head", "FTT_full")
E6_REGIMES = ("20-80", "80-20")
E6_SEED = 42
E6_FEATURE_SET = "F17"
E6_FULL_COHORT_SPLIT_ID = "full_cohort"
FTT_FIXED_BUDGET_NOTE = ("FTT adaptation with a fixed pre-declared budget "
                         "(40 epochs, no validation subset, last-epoch model)")
E6_DESCRIPTIVE_NOTE = "accuracy and balanced accuracy only; macro-F1 and QWK not computed"
E7_METHODS = ("PMV", "aPMV")
E7_FEATURE_SET = "pmv_inputs"
LOPO_SEED = 42
E4_SPEC = {"base": "F12", "extended": "F17", "fitted": "F12"}
E4_SEEDS = (42, 43, 44)
#: E10 hybrids (``revision_runners.run_e10``): model names, seeds (one head pair per E1 FTT seed), feature set
E10_MODELS = ("FTT_RF", "FTT_XGB")
E10_SEEDS = (42, 43, 44)
E10_FEATURE_SET = "FTTemb_F17"
#: E1_model_differences: reference model and the models compared with it, in row order
DIFF_REFERENCE = "XGB"
DIFF_MODELS = ("RF", "SVM", "ANN", "FTT") + E10_MODELS


class AggregateError(ValueError):
    """A predictions file or input does not describe what the aggregate expects."""


def predictions_name(experiment: str, target: str, model: str, seed: int, split_id: str) -> str:
    """File name of an ASHRAE predictions file (``revision_runners.predictions_name``)."""
    return f"{experiment}__{COHORT}__{target}__{model}__seed{int(seed)}__{split_id}.csv.gz"


def lopo_split_id(k: int) -> str:
    return f"lopo_{k:02d}"


def _flat_metrics(m: Mapping[str, Any]) -> Dict[str, Any]:
    return {"n": m["n"], "n_correct": m["n_correct"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "balanced_accuracy_single_class": m["balanced_accuracy_single_class"], "macro_f1": m["macro_f1"],
            "qwk": m["qwk"], "qwk_undefined": m["qwk_undefined"],
            "classes_present": json.dumps(m["classes_present"]),
            "recall": json.dumps({str(k): (None if math.isnan(v) else v) for k, v in m["recall"].items()}),
            "support": json.dumps({str(k): v for k, v in m["support"].items()}),
            "confusion_matrix": json.dumps(m["confusion_matrix"])}


def _metrics_or_empty(y_true, y_pred, target: str) -> Dict[str, Any]:
    """classification_metrics, or n = 0 with NaN metrics for an empty evaluated set."""
    if len(y_true) == 0:
        k = len(rs.ontology(target))
        return {"n": 0, "n_correct": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan"),
                "balanced_accuracy_single_class": False, "macro_f1": float("nan"), "qwk": float("nan"),
                "qwk_undefined": True, "classes_present": [], "recall": {c: float("nan") for c in rs.ontology(target)},
                "support": {c: 0 for c in rs.ontology(target)}, "confusion_matrix": [[0] * k for _ in range(k)]}
    return rs.classification_metrics(y_true, y_pred, target)


def _descriptive_baseline(y_eval, target: str) -> Tuple[Optional[int], Dict[str, Any]]:
    """Majority class of the evaluated set itself and its metrics (n = 0 if empty)."""
    ye = np.asarray(y_eval, dtype=np.int64)
    if not len(ye):
        return None, _metrics_or_empty(ye, ye, target)
    cls = int(rs.majority_baselines(ye, ye, target)["descriptive_class"])
    return cls, rs.classification_metrics(ye, np.full(len(ye), cls), target)


def _baseline_pair(y_eval, deployable_pred, target: str) -> Dict[str, Any]:
    """Deployable (training or development majority, given per vote) and descriptive (majority of the evaluated
    votes) majority-class baselines on one evaluated set."""
    ye = np.asarray(y_eval, dtype=np.int64)
    dp = np.asarray(deployable_pred, dtype=np.int64)
    if len(dp) != len(ye):
        raise AggregateError("deployable baseline predictions and evaluated votes differ in length")
    dep = _metrics_or_empty(ye, dp, target)
    desc_cls, desc = _descriptive_baseline(ye, target)
    dep_classes = sorted(set(dp.tolist()))
    return {"Majority_deployable": dep, "Majority_descriptive": desc, "descriptive_class": desc_cls,
            "deployable_class": dep_classes[0] if len(dep_classes) == 1 else None}


def _labels(values, target: str, what: str) -> np.ndarray:
    return rs.check_labels(np.asarray(values), rs.ontology(target), what)


def round_trip(df: pd.DataFrame) -> pd.DataFrame:
    """The frame as written to CSV and parsed back (pandas defaults)."""
    return pd.read_csv(io.BytesIO(df.to_csv(index=False).encode()))


class Aggregates:
    """Lazily computed aggregate tables of one run (see the module docstring)."""

    TABLES = ("E1_metrics", "E1_confusion_matrices", "E6_full_cohort_zero_shot", "E6_strategy_metrics",
              "E7_metrics", "E2_aggregates", "E4_ema_gains", "E1_model_differences")

    def __init__(self, inputs: RunInputs):
        self.inputs = inputs
        self._raw: Dict[str, pd.DataFrame] = {}
        self._parsed: Dict[str, pd.DataFrame] = {}
        self._cohort: Optional[pd.DataFrame] = None
        self._splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._e1_pred: Dict[Tuple[str, str, int], np.ndarray] = {}

    # ------------------------------------------------------------------ public
    def table(self, name: str) -> pd.DataFrame:
        """The aggregate ``name`` (one of ``TABLES``), as parsed back from its CSV form."""
        if name not in self._parsed:
            self._parsed[name] = round_trip(self.frame(name))
        return self._parsed[name].copy()

    def frame(self, name: str) -> pd.DataFrame:
        """The aggregate ``name`` as computed (before the CSV round trip)."""
        if name not in self.TABLES:
            raise KeyError(f"unknown aggregate {name!r}; known: {list(self.TABLES)}")
        if name not in self._raw:
            builder: Callable[[], None] = {
                "E1_metrics": self._e1, "E1_confusion_matrices": self._e1,
                "E6_full_cohort_zero_shot": self._e6, "E6_strategy_metrics": self._e6,
                "E7_metrics": self._e7, "E2_aggregates": self._e2, "E4_ema_gains": self._e4,
                "E1_model_differences": self._e1_differences}[name]
            builder()
        return self._raw[name]

    # ------------------------------------------------------------------ inputs
    def cohort(self) -> pd.DataFrame:
        if self._cohort is None:
            self._cohort = pd.read_csv(io.BytesIO(self.inputs.input_file("E1", "cohort_csv")))
        return self._cohort

    def y(self, t: str) -> np.ndarray:
        return _labels(self.cohort()[t].to_numpy(dtype=float), t, f"cohort {t}")

    def split(self, t: str) -> Tuple[np.ndarray, np.ndarray]:
        if t not in self._splits:
            js = json.loads(self.inputs.input_file("E1", f"split:{t}"))
            if js.get("target") != t:
                raise AggregateError(f"split file of {t} names target {js.get('target')!r}")
            tr, te = np.asarray(js["train_idx"], dtype=np.int64), np.asarray(js["test_idx"], dtype=np.int64)
            n = len(self.cohort())
            if np.intersect1d(tr, te).size or len(np.unique(tr)) != len(tr) or len(np.unique(te)) != len(te) \
                    or len(tr) + len(te) != n:
                raise AggregateError(f"split of {t} does not partition the {n} cohort votes")
            self._splits[t] = (tr, te)
        return self._splits[t]

    def predictions(self, exp: str, name: str) -> pd.DataFrame:
        # round_trip parser: the floats written by the runners are parsed exactly
        return pd.read_csv(io.BytesIO(self.inputs.run_file(exp, f"predictions/{name}")), compression="gzip",
                           float_precision="round_trip")

    @staticmethod
    def check_frame(df: pd.DataFrame, *, exp: str, cohort: str, target: str, model: str, seed: Optional[int],
                    split_id: str, feature_set: str, rows, y_true, pred_mask=None, what: str = "") -> np.ndarray:
        """Checks of one predictions file; returns ``y_pred`` (int, or float with NaN outside ``pred_mask``)."""
        what = what or f"{exp} {cohort} {target} {model} {seed} {split_id}"
        for col, val in (("experiment", exp), ("cohort", cohort), ("target", target), ("model", model),
                         ("split_id", split_id), ("feature_set", feature_set)):
            if col not in df or not (df[col].astype(str) == str(val)).all():
                raise AggregateError(f"{what}: column {col!r} is not constant {val!r}")
        if seed is None:
            if not df["seed"].isna().all():
                raise AggregateError(f"{what}: seed should be missing")
        elif not (pd.to_numeric(df["seed"], errors="coerce") == seed).all():
            raise AggregateError(f"{what}: seed is not {seed}")
        rows = np.asarray(rows, dtype=np.int64)
        if len(df) != len(rows) or not np.array_equal(df["row_idx"].to_numpy(), rows):
            raise AggregateError(f"{what}: row_idx differs from the serialised split rows (order included)")
        yt = _labels(df["y_true"].to_numpy(), target, f"{what} y_true")
        if not np.array_equal(yt, np.asarray(y_true)):
            raise AggregateError(f"{what}: y_true differs from the cohort votes at row_idx")
        if pred_mask is None:
            return _labels(df["y_pred"].to_numpy(), target, f"{what} y_pred")
        raw = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(raw), np.asarray(pred_mask, dtype=bool)):
            raise AggregateError(f"{what}: y_pred is not missing exactly outside the validity mask")
        _labels(raw[pred_mask], target, f"{what} y_pred")
        return raw

    def e1_inventory(self) -> List[Tuple[str, int]]:
        """The (model, seed) pairs of E1, checked against the seeds of the paper (``E1_SEEDS``; never inferred
        from the outputs): no seed listed twice, ``ml_inventory`` equal to ``seeds.models``, every family with
        exactly its paper seeds. A reduced, substituted or duplicated seed set is refused."""
        man = self.inputs.manifest("E1")
        raw_seeds = (man.get("seeds") or {}).get("models") or {}
        not_int = sorted(m for m, v in raw_seeds.items()
                         if any(isinstance(s, bool) or not isinstance(s, int) for s in v))
        if not_int:
            raise AggregateError(f"E1 seeds.models holds a non-integer seed for {not_int}")
        dup = sorted(m for m, v in raw_seeds.items() if len(list(v)) != len({int(s) for s in v}))
        if dup:
            raise AggregateError(f"E1 seeds.models lists a seed twice for {dup}")
        seeds = {m: tuple(sorted(int(s) for s in v)) for m, v in raw_seeds.items()}
        inventory = sorted(man.get("ml_inventory") or [])
        if sorted(f"{m}:{s}" for m, ss in seeds.items() for s in ss) != inventory:
            raise AggregateError(f"E1 ml_inventory {inventory} differs from seeds.models {seeds}")
        if set(seeds) != set(E1_MODELS):
            raise AggregateError(f"E1 families {sorted(seeds)} != {list(E1_MODELS)}")
        for m in E1_MODELS:
            if seeds[m] != E1_SEEDS[m]:
                raise AggregateError(f"E1 {m} seeds {seeds[m]} != the paper seeds {E1_SEEDS[m]}")
        return [(m, s) for m in E1_MODELS for s in seeds[m]]

    def e1_prediction(self, t: str, m: str, s: int) -> np.ndarray:
        key = (t, m, s)
        if key not in self._e1_pred:
            tr, te = self.split(t)
            df = self.predictions("E1", predictions_name("E1", t, m, s, ARCHIVED))
            self._e1_pred[key] = self.check_frame(df, exp="E1", cohort=COHORT, target=t, model=m, seed=s,
                                                  split_id=ARCHIVED, feature_set="F17", rows=te,
                                                  y_true=self.y(t)[te])
        return self._e1_pred[key]

    # ------------------------------------------------------------------ E1
    def _e1(self) -> None:
        rows, cms = [], []
        for t in rs.TARGETS:
            tr, te = self.split(t)
            y = self.y(t)[te]
            for m, s in self.e1_inventory():
                met = rs.classification_metrics(y, self.e1_prediction(t, m, s), t)
                rows.append({"target": t, "model": m, "seed": s, "row_type": "seed", **_flat_metrics(met)})
                labels = rs.ontology(t)
                for i, a in enumerate(labels):
                    for j, b in enumerate(labels):
                        cms.append({"target": t, "model": m, "seed": s, "true": a, "pred": b,
                                    "count": met["confusion_matrix"][i][j]})
        self._raw["E1_metrics"] = pd.DataFrame(rows)
        self._raw["E1_confusion_matrices"] = pd.DataFrame(cms)

    # ------------------------------------------------------------------ E6
    def french(self, c: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.inputs.input_file("E6", c)))

    def phase3_splits(self, c: str, t: str, reg: str) -> List[Dict[str, Any]]:
        name = f"{c}_{t}_{reg}.json"
        payload = json.loads(self.inputs.input_file("E6", f"phase3_split:{name}"))
        if payload.get("grouping") != "date" or payload.get("cohort", c) != c or payload.get("target", t) != t:
            raise AggregateError(f"Phase 3 split file {name}: grouping / cohort / target differ")
        return list(payload["splits"])

    def _e6(self) -> None:
        rows, full = [], []
        french = {c: self.french(c) for c in rs.G3_COHORTS}
        # rows in sorted (cohort, target, regime) order, as the tables were first written
        for c in sorted(rs.G3_COHORTS):
            for t in sorted(rs.TARGETS):
                y_c = _labels(french[c][t].to_numpy(dtype=float), t, f"{c} {t}")
                for reg in sorted(E6_REGIMES):
                    splits = self.phase3_splits(c, t, reg)
                    preds = {}
                    for m in E6_MODELS:
                        parts = []
                        for s in splits:
                            rows_s = np.asarray(s["test_idx"], dtype=np.int64)
                            df = self.predictions("E6", f"E6__{c}__{t}__{m}__seed{E6_SEED}__{s['split_id']}.csv.gz")
                            parts.append(self.check_frame(df, exp="E6", cohort=c, target=t, model=m, seed=E6_SEED,
                                                          split_id=s["split_id"], feature_set=E6_FEATURE_SET,
                                                          rows=rows_s, y_true=y_c[rows_s]))
                        preds[m] = np.concatenate(parts)
                    y_pool = np.concatenate([y_c[np.asarray(s["test_idx"], dtype=np.int64)] for s in splits])
                    for m in E6_MODELS:
                        met = rs.classification_metrics(y_pool, preds[m], t)
                        rows.append({"cohort": c, "target": t, "regime": reg, "strategy": m, "row_type": "strategy",
                                     **_flat_metrics(met),
                                     "cooler_recall": met["recall"].get(-1) if t == "thermal_preference" else None,
                                     "ftt_note": FTT_FIXED_BUDGET_NOTE if m in E6_FTT_ADAPTED else ""})
                    # baselines: ASHRAE training rows (zero-shot); development votes of each split (adaptation)
                    src_cls = rs.deployable_majority_class(self.y(t)[self.split(t)[0]], t)
                    dev_pred = np.concatenate([np.full(len(s["test_idx"]), rs.deployable_majority_class(
                        y_c[np.asarray(s["dev_idx"], dtype=np.int64)], t)) for s in splits])
                    for kind, pred in (("Majority_deployable_ASHRAE_train", np.full(len(y_pool), src_cls)),
                                       ("Majority_deployable_DEV", dev_pred)):
                        met = rs.classification_metrics(y_pool, pred, t)
                        rows.append({"cohort": c, "target": t, "regime": reg, "strategy": kind, "row_type": "baseline",
                                     **_flat_metrics(met)})
                    desc = rs.majority_baselines(y_pool, y_pool, t)
                    rows.append({"cohort": c, "target": t, "regime": reg, "strategy": "Majority_descriptive_TEST",
                                 "row_type": "baseline", "n": len(y_pool), "accuracy": desc["descriptive_majority_rate"],
                                 "balanced_accuracy": desc["descriptive_balanced_accuracy"],
                                 "majority_class": desc["descriptive_class"], "baseline_note": E6_DESCRIPTIVE_NOTE})
        # full-cohort zero-shot (point estimates) and its two majority-class baselines
        for c in rs.G3_COHORTS:
            n = len(french[c])
            for t in rs.TARGETS:
                y_c = _labels(french[c][t].to_numpy(dtype=float), t, f"{c} {t}")
                for m in E6_ZERO_SHOT:
                    df = self.predictions("E6", f"E6__{c}__{t}__{m}__seed{E6_SEED}__{E6_FULL_COHORT_SPLIT_ID}.csv.gz")
                    yp = self.check_frame(df, exp="E6", cohort=c, target=t, model=m, seed=E6_SEED,
                                          split_id=E6_FULL_COHORT_SPLIT_ID, feature_set=E6_FEATURE_SET,
                                          rows=np.arange(n), y_true=y_c)
                    met = rs.classification_metrics(y_c, yp, t)
                    full.append({"cohort": c, "target": t, "model": m, **_flat_metrics(met),
                                 "note": "descriptive; never compared with an ASHRAE accuracy"})
                bp = _baseline_pair(y_c, np.full(n, rs.deployable_majority_class(self.y(t)[self.split(t)[0]], t)), t)
                for kind, cls in (("Majority_deployable_ASHRAE_train", bp["deployable_class"]),
                                  ("Majority_descriptive_cohort", bp["descriptive_class"])):
                    mm = bp["Majority_deployable" if "deployable" in kind else "Majority_descriptive"]
                    full.append({"cohort": c, "target": t, "model": kind, **_flat_metrics(mm), "majority_class": cls,
                                 "note": "majority-class baseline"})
        self._raw["E6_strategy_metrics"] = pd.DataFrame(rows)
        self._raw["E6_full_cohort_zero_shot"] = pd.DataFrame(full)

    # ------------------------------------------------------------------ E7
    def _e7(self) -> None:
        cal = json.loads(self.inputs.run_file("E7", "apmv_calibration.json"))
        rows = []
        for t in rs.TARGETS:
            tr, te = self.split(t)
            y = self.y(t)[te]
            c = cal.get(t) or {}
            for method in E7_METHODS:
                if method == "aPMV" and c.get("status") != "ok":
                    rows.append({"target": t, "pmv_variant": "primary", "method": method, "subset": "on_mask",
                                 "status": NA_TEXT, "reason": str(c.get("reason"))})
                    continue
                df = self.predictions("E7", f"E7__{COHORT}__{t}__{method}.csv.gz")
                mask = df["pmv_valid"].map(lambda v: v if isinstance(v, (bool, np.bool_))
                                           else str(v).strip().lower() == "true").to_numpy(dtype=bool)
                yp = self.check_frame(df, exp="E7", cohort=COHORT, target=t, model=method, seed=None,
                                      split_id=ARCHIVED, feature_set=E7_FEATURE_SET, rows=te, y_true=y, pred_mask=mask)
                m = _metrics_or_empty(y[mask], yp[mask].astype(np.int64), t)
                rows.append({"target": t, "pmv_variant": "primary", "method": method, "subset": "on_mask",
                             "coverage": float(mask.mean()), **_flat_metrics(m), "status": "ok"})
        self._raw["E7_metrics"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------ E2
    def _e2(self) -> None:
        folds = json.loads(self.inputs.run_file("E2", "lopo_folds.json"))
        pubs = list(folds["publications"])
        if len(folds["folds"]) != len(pubs):
            raise AggregateError("E2 lopo_folds.json: one fold per publication expected")
        metrics_csv = pd.read_csv(io.BytesIO(self.inputs.run_file("E2", "metrics.csv")), float_precision="round_trip")
        n = len(self.cohort())
        families = tuple(f for f in TREE if f in set(metrics_csv["model"].astype(str)))
        # rows of each held-out publication, from the predictions files of the first family and target
        pub = np.full(n, None, dtype=object)
        fold_rows: Dict[int, np.ndarray] = {}
        for k, f in enumerate(folds["folds"]):
            if f["publication"] != pubs[k] or f["split_id"] != lopo_split_id(k) or int(f["fold"]) != k:
                raise AggregateError(f"E2 lopo_folds.json: fold {k} does not describe publication {k}")
            df = self.predictions("E2", predictions_name("E2", rs.TARGETS[0], families[0], LOPO_SEED, lopo_split_id(k)))
            r = df["row_idx"].to_numpy(dtype=np.int64)
            if not (df["publication"].astype(str) == pubs[k]).all() or len(r) != int(f["n_test"]) \
                    or not np.array_equal(r, np.sort(r)) or (pub[r] != None).any():  # noqa: E711
                raise AggregateError(f"E2 fold {k}: predicted rows are not the votes of publication {k}")
            pub[r] = pubs[k]
            fold_rows[k] = r
        if (pub == None).any():  # noqa: E711
            raise AggregateError("E2: the LOPO folds do not cover every cohort vote")
        pub = pub.astype(str)
        agg_rows = []
        for t in rs.TARGETS:
            y_t = self.y(t)
            dep = np.full(n, -99, dtype=np.int64)
            for p in pubs:
                sel = pub == p
                dep[sel] = rs.deployable_majority_class(y_t[~sel], t)
            base_pub = {p: _baseline_pair(y_t[pub == p], dep[pub == p], t) for p in pubs}
            for fam in families:
                y_all = np.full(n, -99, dtype=np.int64)
                for k in range(len(pubs)):
                    rows_k = np.flatnonzero(pub == pubs[k])
                    df = self.predictions("E2", predictions_name("E2", t, fam, LOPO_SEED, lopo_split_id(k)))
                    y_all[rows_k] = self.check_frame(df, exp="E2", cohort=COHORT, target=t, model=fam, seed=LOPO_SEED,
                                                     split_id=lopo_split_id(k), feature_set="F17", rows=rows_k,
                                                     y_true=y_t[rows_k])
                if (y_all == -99).any():
                    raise AggregateError(f"E2 {t} {fam}: the LOPO folds do not predict every row exactly once")
                vals: Dict[str, List[float]] = {}
                for p in pubs:
                    sel = pub == p
                    met = rs.classification_metrics(y_t[sel], y_all[sel], t)
                    bp = base_pub[p]
                    for kind, mm in (("model", met), ("Majority_deployable", bp["Majority_deployable"]),
                                     ("Majority_descriptive", bp["Majority_descriptive"])):
                        for col in ("accuracy", "balanced_accuracy", "macro_f1"):
                            vals.setdefault(f"{kind}:{col}", []).append(mm[col])
                micro = rs.classification_metrics(y_t, y_all, t)
                # order of the per-publication rows of the run's metrics.csv, so that the mean is the same sum
                order = [int(str(s).split("_")[-1]) for s in metrics_csv[(metrics_csv["target"] == t)
                                                                          & (metrics_csv["model"] == fam)]["split_id"]]
                if sorted(order) != list(range(len(pubs))):
                    order = list(range(len(pubs)))
                macro = {col: float(np.mean(np.asarray([vals[f"model:{col}"][i] for i in order])))
                         for col in ("accuracy", "balanced_accuracy", "macro_f1")}
                for agg, v, nn in (("micro", micro, micro["n"]), ("macro_over_publications", macro, len(pubs))):
                    agg_rows.append({"target": t, "model": fam, "aggregate": agg, "n": nn, "accuracy": v["accuracy"],
                                     "balanced_accuracy": v["balanced_accuracy"], "macro_f1": v["macro_f1"]})
                if fam == families[0]:          # baseline aggregates, once per target
                    bmicro = _baseline_pair(y_t, dep, t)
                    for kind in ("Majority_deployable", "Majority_descriptive"):
                        bmacro = {col: float(np.mean(np.asarray([vals[f"{kind}:{col}"][i] for i in order])))
                                  for col in ("accuracy", "balanced_accuracy", "macro_f1")}
                        for agg, v, nn in (("micro", bmicro[kind], n), ("macro_over_publications", bmacro, len(pubs))):
                            agg_rows.append({"target": t, "model": kind, "aggregate": agg, "n": nn,
                                             "accuracy": v["accuracy"], "balanced_accuracy": v["balanced_accuracy"],
                                             "macro_f1": v["macro_f1"]})
        self._raw["E2_aggregates"] = pd.DataFrame(agg_rows)

    # ------------------------------------------------------------------ E4
    def _e4(self) -> None:
        pairs = pd.read_csv(io.BytesIO(self.inputs.run_file("E4", "pairs.csv")))
        e1_out = self.inputs.manifest("E1").get("outputs_sha256") or {}
        e4_out = self.inputs.manifest("E4").get("outputs_sha256") or {}
        side_files: Dict[Tuple[str, str, int], Dict[str, Tuple[str, str]]] = {}
        for _, r in pairs.iterrows():
            t, fam, s = str(r["target"]), str(r["model"]), int(r["seed"])
            if r["base_feature_set"] != E4_SPEC["base"] or r["extended_feature_set"] != E4_SPEC["extended"]:
                raise AggregateError(f"E4 pairs.csv: row {t} {fam} {s} does not describe {E4_SPEC}")
            files = {}
            for side in ("base", "extended"):
                src = ("E1", predictions_name("E1", t, fam, s, ARCHIVED)) if E4_SPEC[side] == "F17" else \
                    ("E4", predictions_name("E4", t, fam, s, ARCHIVED))
                outputs = e1_out if src[0] == "E1" else e4_out
                if str(r[f"{side}_file"]).rsplit("/", 1)[-1] != src[1] \
                        or outputs.get(f"predictions/{src[1]}") != r[f"{side}_sha256"]:
                    raise AggregateError(f"E4 pairs.csv: {side} file of {t} {fam} {s} is not {src[0]} {src[1]}")
                files[side] = src
            side_files[(t, fam, s)] = files
        if set(side_files) != {(t, f, s) for t in rs.TARGETS for f in TREE for s in E4_SEEDS} \
                or len(pairs) != len(side_files):
            raise AggregateError("E4 pairs.csv: (target, family, seed) set differs")
        gains = {}
        for t in rs.TARGETS:
            tr, te = self.split(t)
            y = self.y(t)[te]
            for fam in TREE:
                frames = {}
                for s in E4_SEEDS:
                    for side in ("base", "extended"):
                        src_exp, fname = side_files[(t, fam, s)][side]
                        df = self.predictions(src_exp, fname)
                        self.check_frame(df, exp=src_exp, cohort=COHORT, target=t, model=fam, seed=s,
                                         split_id=ARCHIVED, feature_set=E4_SPEC[side], rows=te, y_true=y)
                        frames[f"{side}:{s}"] = df
                yy, preds, cl, _ = rs.align_predictions(frames, cluster_col="building_id", expected_row_idx=te)
                p_ext = np.stack([preds[f"extended:{s}"] for s in E4_SEEDS])
                p_base = np.stack([preds[f"base:{s}"] for s in E4_SEEDS])
                gains[(fam, t)] = rs.paired_accuracy_contrast(
                    yy, p_ext, p_base, cl, target=t, level=DESCRIPTIVE_LEVEL, cluster_order=None,
                    min_clusters=rs.MIN_CLUSTERS, n_boot=rs.BOOT_B, seed=rs.BOOT_SEED)
        rows = []
        for fam in TREE:
            for t in rs.TARGETS:
                g = gains[(fam, t)]
                est = g["delta"]
                rows.append({"family": fam, "target": t, "gain": est.estimate,
                             "ci95_low": est.ci_low if est.available else None,
                             "ci95_high": est.ci_high if est.available else None,
                             "interval_available": est.available, "reason": est.reason,
                             "n_clusters": est.n_clusters,
                             "base_feature_set": E4_SPEC["base"], "extended_feature_set": E4_SPEC["extended"],
                             "seed_accuracies_base": json.dumps(g["seed_accuracies_b"]),
                             "seed_accuracies_ext": json.dumps(g["seed_accuracies_a"]),
                             "seed_spread_base": g["seed_spread_b"], "seed_spread_ext": g["seed_spread_a"]})
        self._raw["E4_ema_gains"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------ E1 model differences
    def _e1_differences(self) -> None:
        """Paired accuracy difference of each Phase 2 model from ``DIFF_REFERENCE`` on the archived test votes.

        Same conventions as the E4 gain: every predictions file checked against the serialised test rows (order
        included) and the cohort votes, the files of the two models aligned on ``row_idx`` / ``y_true`` /
        ``building_id`` (``revision_stats.align_predictions``), accuracy of a model = mean over its seeds (SVM:
        its single seed), and the building-cluster bootstrap of the seed-averaged per-vote differences
        (``revision_stats.paired_accuracy_contrast``, frozen draw ``BOOT_B`` / ``BOOT_SEED``). The hybrids are
        read from E10 and left out when E10 is pending in a preview.
        """
        e1_seeds: Dict[str, List[int]] = {}
        for m, s in self.e1_inventory():
            e1_seeds.setdefault(m, []).append(s)
        models = {m: ("E1", "F17", tuple(e1_seeds[m])) for m in E1_MODELS}
        if self.inputs.require("E10"):
            models.update({m: ("E10", E10_FEATURE_SET, E10_SEEDS) for m in E10_MODELS})
        rows = []
        for t in rs.TARGETS:
            tr, te = self.split(t)
            y = self.y(t)[te]
            frames: Dict[str, List[str]] = {}
            all_frames: Dict[str, pd.DataFrame] = {}
            for m, (exp, feature_set, seeds) in models.items():
                for s in seeds:
                    df = self.predictions(exp, predictions_name(exp, t, m, s, ARCHIVED))
                    self.check_frame(df, exp=exp, cohort=COHORT, target=t, model=m, seed=s, split_id=ARCHIVED,
                                     feature_set=feature_set, rows=te, y_true=y)
                    if "building_id" not in df.columns:
                        raise AggregateError(f"{exp} {t} {m} {s}: predictions file without building_id")
                    all_frames[f"{m}:{s}"] = df
                    frames.setdefault(m, []).append(f"{m}:{s}")
            # every model on the same votes, in the same order, with the same buildings (paired contrasts)
            try:
                yy, preds, cl, _ = rs.align_predictions(all_frames, cluster_col="building_id", expected_row_idx=te)
            except ValueError as e:
                raise AggregateError(f"E1_model_differences {t}: {e}") from e
            p_ref = np.stack([preds[k] for k in frames[DIFF_REFERENCE]])
            for m in DIFF_MODELS:
                if m not in models:
                    continue
                g = rs.paired_accuracy_contrast(
                    yy, np.stack([preds[k] for k in frames[m]]), p_ref, cl, target=t, level=DESCRIPTIVE_LEVEL,
                    cluster_order=None, min_clusters=rs.MIN_CLUSTERS, n_boot=rs.BOOT_B, seed=rs.BOOT_SEED)
                est = g["delta"]
                rows.append({"target": t, "model": m, "reference": DIFF_REFERENCE, "n_test": g["n_votes"],
                             "n_clusters": est.n_clusters, "acc_model": g["accuracy_a"],
                             "acc_reference": g["accuracy_b"], "difference": est.estimate,
                             "ci95_low": est.ci_low if est.available else None,
                             "ci95_high": est.ci_high if est.available else None,
                             "n_bootstrap": rs.BOOT_B, "bootstrap_seed": rs.BOOT_SEED})
        self._raw["E1_model_differences"] = pd.DataFrame(rows)
