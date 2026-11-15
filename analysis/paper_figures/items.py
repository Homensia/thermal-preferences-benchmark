"""One builder per paper figure / table (``ITEMS``), reading only hash-verified run outputs."""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import bar_charts, captions, tables  # noqa: E402
from .inputs import OutputDir, RunInputs  # noqa: E402
from .metrics import (DATASET_NAME, MODEL_NAME, ONTOLOGY, TARGET_LABEL, TARGETS, accuracy_from_cm,  # noqa: E402
                      as_matrix, chance_disagreement, class_distribution, seed_means, vote_code,
                      weighted_f1_from_cm)

E1_MODELS = ["RF", "XGB", "SVM", "ANN", "FTT"]
E10_MODELS = ["FTT_RF", "FTT_XGB"]          # E10_hybrids/metrics.csv model names
E10_SEEDS = (42, 43, 44)                    # one head pair per E1 FTT seed
E10_FEATURE_SET = "FTTemb_F17"              # revision_runners.E10_FEATURE_SET
ARCHIVED_SPLIT_ID = "archived_80_20"
MODEL_DISPLAY = MODEL_NAME          # one name per model in every figure, table and caption
MODEL_FILE = {"RF": "RandomForest", "XGB": "XGBoost", "SVM": "SVM", "ANN": "ANN", "FTT": "FTTransformer"}
E7_INDICES = ["PMV", "aPMV"]
E7B_INDICES = ["ePMV", "PTS_SET", "PTS_PET"]
EPMV_E_FLAG_RANGE = (0.5, 1.0)          # indicateurs_classiques.EPMV_E_FLAG_RANGE (flag, no clip)
E7_TARGETS = TARGETS                                                 # E7 evaluated PMV / aPMV on all three
E7B_TARGETS = ("thermal_sensation", "TSV_3p")                        # no TPV in E7b (the paper reports TSV-7, TSV-3)
INDEX_ORDER = ["PMV", "aPMV", "ePMV", "PTS_SET", "PTS_PET"]       # Figure_Table6 order without aPTS / ePTS
METRICS4 = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted-F1"), ("qwk", "QWK")]
PHASE3_ZERO_SHOT = {"RF_zero_shot": MODEL_NAME["RF"], "XGB_zero_shot": MODEL_NAME["XGB"],
                    "FTT_zero_shot": MODEL_NAME["FTT"]}
PHASE3_COLOURS = {MODEL_NAME["RF"]: "#0072B2", MODEL_NAME["XGB"]: "#D55E00", MODEL_NAME["FTT"]: "#009E73"}
# RF / XGBoost are cloned and refitted on the target-domain development split (phase3_runner): local
# retraining; "fine-tuning" is kept for the FTT adaptations of the Phase 2 network
PHASE3_ADAPTIVE = {"RF_local": "RF local retraining", "XGB_local": "XGBoost local retraining",
                   "FTT_head": "FTT-Head-only fine-tuning", "FTT_full": "FTT-Full fine-tuning"}
# majority-class rows: zero-shot: the
# most frequent vote of the cohort predicted for all its votes (E6 Majority_descriptive_cohort); adaptive:
# the most frequent vote of the development split predicted for the test votes (E6 Majority_deployable_DEV)
PHASE3_MAJORITY_ZERO_SHOT = {"Majority_descriptive_cohort": "Majority class"}
PHASE3_MAJORITY_ADAPTIVE = {"Majority_deployable_DEV": "Majority class (DEV)"}
MAJORITY_COLOUR = "#9E9E9E"
# one colour per vote value in every distribution panel: diverging cold (blue) to
# hot (red) for the sensation votes; TPV coloured by the change wanted (-1 cooler: blue, +1 warmer: red)
VOTE_COLOURS = {-3: "#2166AC", -2: "#4393C3", -1: "#92C5DE", 0: "#BDBDBD", 1: "#F4A582", 2: "#D6604D",
                3: "#B2182B"}
DIST_COLOURS = {"thermal_sensation": VOTE_COLOURS,
                "TSV_3p": {-1: VOTE_COLOURS[-2], 0: VOTE_COLOURS[0], 1: VOTE_COLOURS[2]},
                "thermal_preference": {-1: VOTE_COLOURS[-2], 0: VOTE_COLOURS[0], 1: VOTE_COLOURS[2]}}
DIST_TICK_WORD = {"TSV_3p": {-1: "cool", 0: "neutral", 1: "warm"},
                  "thermal_preference": {-1: "cooler", 0: "no change", 1: "warmer"}}
# print sizes: every figure is drawn at the width at which it is
# included, so that no text falls below 7 pt (Elsevier artwork guidance)
# (the saved page is the tight bounding box plus PAD_IN on each side: the canvas is narrower by 2 PAD_IN)
PAD_IN = 0.02
TEXT_WIDTH_IN = bar_charts.PAGE_WIDTH_IN                  # 510 pt: the text width of the 5p layout
HALF_WIDTH_IN = 0.49 * TEXT_WIDTH_IN - 2 * PAD_IN     # subfigure at 0.49 text width
THIRD_WIDTH_IN = 0.32 * TEXT_WIDTH_IN - 2 * PAD_IN    # subfigure at 0.32 text width
DIST_FILE = {"thermal_sensation": "tsv7", "TSV_3p": "tsv3", "thermal_preference": "tpv"}


@dataclass
class Options:
    cm_seed: str = "42"
    fi_seeds: Sequence[int] = (42, 43, 44)
    indices_tpv: bool = False       # indices on TSV-7 and TSV-3 only; True (CLI --indices_tpv, not the
                                    # paper scope) adds a TPV panel / rows of PMV and aPMV
    nn_cases_in_figure: bool = False  # also draw the E9b Case B / Case C agreements in the NN figure


@dataclass
class Item:
    name: str
    paper: str
    outputs: List[str]
    build: Callable[[RunInputs, OutputDir, Options], None]
    pending: List[str] = field(default_factory=list)
    default: bool = True            # built without --only (False: not in the paper, built only on request)


# ---------------------------------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------------------------------
def _metric_rows(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add weighted-F1 from the confusion matrix; check the recorded accuracy against it."""
    df = df.copy()
    acc_cm = df["confusion_matrix"].map(accuracy_from_cm)
    bad = ~np.isclose(acc_cm, df["accuracy"].astype(float), rtol=0, atol=1e-12)
    if bad.any():
        raise ValueError(f"{what}: accuracy differs from its confusion matrix in {int(bad.sum())} rows")
    df["weighted_f1"] = df["confusion_matrix"].map(weighted_f1_from_cm)
    return df


def phase2_rows(inputs: RunInputs) -> tuple:
    """Per-seed rows (E1 and, if complete, E10) and their seed means; the E1 rows are recomputed from the E1
    predictions files (``aggregate.Aggregates``)."""
    e1 = inputs.aggregate("E1_metrics")
    seeds = _metric_rows(e1[(e1["row_type"] == "seed") & e1["model"].isin(E1_MODELS)], "E1")
    seeds["seed"] = seeds["seed"].astype(int)
    seeds["experiment"] = "E1"
    means = seed_means(seeds, ["target", "model"], ["accuracy", "macro_f1", "qwk", "weighted_f1"])
    note = None
    if inputs.require("E10"):
        e10 = e10_rows(inputs, seeds)
        seeds = pd.concat([seeds, e10[seeds.columns.intersection(e10.columns)]], ignore_index=True)
        means = pd.concat([means, seed_means(e10, ["target", "model"],
                                             ["accuracy", "macro_f1", "qwk", "weighted_f1"])], ignore_index=True)
    else:
        note = "E10 (FTT+RF, FTT+XGB) has no output yet: hybrid rows left out"
    means["model_display"] = means["model"].map(MODEL_DISPLAY)
    return seeds, means, note


def _int_seeds(col: pd.Series, what: str) -> pd.Series:
    v = pd.to_numeric(col, errors="raise").astype(float)
    if v.isna().any() or not np.equal(np.mod(v, 1), 0).all():
        raise ValueError(f"{what}: non-integer seed values {sorted(set(col))}")
    return v.astype(int)


def e10_rows(inputs: RunInputs, e1_seed_rows: pd.DataFrame) -> pd.DataFrame:
    """E10 metrics rows, accepted only if complete and on the E1 test votes.

    Exactly one row per target x {FTT_RF, FTT_XGB} x seed {42, 43, 44}; experiment ``E10``, feature set
    ``FTTemb_F17``, split ``archived_80_20``; ``n`` and the class support equal to those of the E1 test
    votes; and each row's confusion matrix recomputed from its predictions file (hash-verified), whose
    ``row_idx`` must be exactly the archived test indices of the target and whose ``y_true`` must be the
    cohort votes at those rows (archived split and cohort: inputs of E1, hash-verified).
    """
    cohort = pd.read_csv(io.BytesIO(inputs.input_file("E1", "cohort_csv")))
    test_idx = {}
    for t in TARGETS:
        split = json.loads(inputs.input_file("E1", f"split:{t}"))
        if split.get("target") != t:
            raise ValueError(f"E1 split file of {t}: target {split.get('target')!r}")
        test_idx[t] = np.asarray(split["test_idx"], dtype=np.int64)
    return check_e10_metrics(
        inputs.run_csv("E10", "metrics.csv"), e1_seed_rows, cohort, test_idx,
        lambda name: pd.read_csv(io.BytesIO(inputs.run_file("E10", f"predictions/{name}")), compression="gzip"))


def check_e10_metrics(e10: pd.DataFrame, e1_seed_rows: pd.DataFrame, cohort: pd.DataFrame,
                      test_idx: Dict[str, np.ndarray], read_predictions: Callable[[str], pd.DataFrame],
                      targets=TARGETS, seeds=E10_SEEDS) -> pd.DataFrame:
    """The checks of ``e10_rows`` on plain inputs (``read_predictions``: predictions file name -> frame), so
    that the tests can apply them to the outputs of ``revision_runners.run_e10`` itself."""
    for col in ("experiment", "target", "feature_set", "split_id", "model", "seed", "n", "accuracy", "macro_f1",
                "qwk", "confusion_matrix", "support", "predictions"):
        if col not in e10.columns:
            raise ValueError(f"E10 metrics.csv: column {col!r} missing")
    e10 = e10.copy()
    e10["seed"] = _int_seeds(e10["seed"], "E10 metrics.csv")
    keys = list(zip(e10["target"], e10["model"], e10["seed"]))
    expected = {(t, m, s) for t in targets for m in E10_MODELS for s in seeds}
    dup = sorted({k for k in keys if keys.count(k) > 1})
    missing, extra = sorted(expected - set(keys)), sorted(set(keys) - expected)
    if dup or missing or extra:
        raise ValueError(f"E10 metrics.csv: expected one row per target x {E10_MODELS} x seeds {list(seeds)}; "
                         f"missing {missing}, duplicated {dup}, unexpected {extra}")
    for col, want in (("experiment", "E10"), ("feature_set", E10_FEATURE_SET), ("split_id", ARCHIVED_SPLIT_ID)):
        bad = sorted(set(e10[col].astype(str)) - {want})
        if bad:
            raise ValueError(f"E10 metrics.csv: {col} {bad}, expected {want!r}")
    e10 = _metric_rows(e10, "E10")
    for t in targets:
        te = test_idx[t]
        y = pd.to_numeric(cohort[t], errors="raise").to_numpy()
        e1_support = {json.dumps(json.loads(s), sort_keys=True)
                      for s in e1_seed_rows[e1_seed_rows["target"] == t]["support"]}
        if len(e1_support) != 1:
            raise ValueError(f"E1 {t}: the seed rows do not share one test support")
        e1_support = json.loads(e1_support.pop())
        labels = list(ONTOLOGY[t])
        for _, r in e10[e10["target"] == t].iterrows():
            what = f"E10 {t} {r['model']} seed {r['seed']}"
            if int(r["n"]) != len(te) or json.loads(r["support"]) != e1_support:
                raise ValueError(f"{what}: n {r['n']} / support {r['support']} differ from the E1 test votes "
                                 f"({len(te)}, {e1_support})")
            pred = read_predictions(r["predictions"])
            for col, want in (("experiment", "E10"), ("target", t), ("feature_set", E10_FEATURE_SET),
                              ("model", r["model"]), ("split_id", ARCHIVED_SPLIT_ID)):
                if set(pred[col].astype(str)) != {want}:
                    raise ValueError(f"{what}: predictions file {col} {sorted(set(pred[col].astype(str)))}")
            if set(_int_seeds(pred["seed"], what)) != {int(r["seed"])}:
                raise ValueError(f"{what}: predictions file seed {sorted(set(pred['seed']))}")
            idx = pred["row_idx"].to_numpy(dtype=np.int64)
            if len(idx) != len(te) or not np.array_equal(np.sort(idx), np.sort(te)):
                raise ValueError(f"{what}: predicted rows are not the archived test split of {t}")
            yt, yp = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
            if not np.array_equal(yt, y[idx]):
                raise ValueError(f"{what}: y_true differs from the cohort votes")
            pos = {c: i for i, c in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
            for a, b in zip(yt, yp):
                if a not in pos or b not in pos:
                    raise ValueError(f"{what}: vote {a} / prediction {b} outside the ontology")
                cm[pos[a], pos[b]] += 1
            if not np.array_equal(cm, as_matrix(r["confusion_matrix"])):
                raise ValueError(f"{what}: confusion matrix differs from its predictions file")
    e10["experiment"] = "E10"
    return e10


def pick_index_rows(df: pd.DataFrame, methods, targets, what: str) -> pd.DataFrame:
    """Index rows of an E7 / E7b metrics file (primary PMV variant, on the validity mask), one per index and
    target, every row with status ok."""
    sel = df["method"].isin(methods) & (df["subset"] == "on_mask")
    if "pmv_variant" in df.columns:
        sel &= df["pmv_variant"].fillna("primary") == "primary"
    out = df[sel].copy()
    # the E7b metrics.csv names the index of each ML comparison row in a column "index" (empty on the
    # index rows kept here): dropped, the index name is the method of the kept rows
    out = out.drop(columns=["index"], errors="ignore")
    if "status" in out.columns and (out["status"].fillna("ok") != "ok").any():
        raise ValueError(f"{what}: rows with status != ok: {out[out['status'] != 'ok'][['target', 'method']]}")
    if set(out["target"]) != set(targets):
        raise ValueError(f"{what}: targets {sorted(set(out['target']))}, expected {sorted(targets)}")
    for t in targets:
        got = sorted(out[out["target"] == t]["method"])
        if got != sorted(methods):
            raise ValueError(f"{what}: {t} has indices {got}, expected {sorted(methods)}")
    if "ePMV" in methods:
        # e unbounded, its flag outside [0.5, 1.0] in its own column (never in status)
        ep = out[out["method"] == "ePMV"]
        if not {"epmv_e", "epmv_e_flag_outside_range"} <= set(out.columns):
            raise ValueError(f"{what}: ePMV rows without epmv_e / epmv_e_flag_outside_range")
        e = pd.to_numeric(ep["epmv_e"], errors="raise").to_numpy(dtype=float)
        flag = ep["epmv_e_flag_outside_range"].map(
            lambda v: v if isinstance(v, (bool, np.bool_)) else {"True": True, "False": False}.get(str(v))).to_numpy()
        lo, hi = EPMV_E_FLAG_RANGE
        if not np.isfinite(e).all() or any(f is None for f in flag) or \
                [bool(f) for f in flag] != [not (lo <= v <= hi) for v in e]:
            raise ValueError(f"{what}: ePMV flag outside [{lo}, {hi}] inconsistent with e = {e.tolist()}")
        out.loc[ep.index, "epmv_e_flag_outside_range"] = [bool(f) for f in flag]
    return _metric_rows(out, what)


def index_rows(inputs: RunInputs, include_tpv: bool = False) -> tuple:
    pick = pick_index_rows
    e7 = pick(inputs.aggregate("E7_metrics"), E7_INDICES, E7_TARGETS, "E7")
    if not include_tpv:
        e7 = e7[e7["target"] != "thermal_preference"]
    note = None
    parts = [e7]
    if inputs.require("E7b"):
        parts.append(pick(inputs.run_csv("E7b", "metrics.csv"), E7B_INDICES, E7B_TARGETS, "E7b"))
    else:
        note = "E7b (ePMV, PTS_SET, PTS_PET) has no output yet: those indices left out"
    rows = pd.concat(parts, ignore_index=True).rename(columns={"method": "index"})
    cols = ["index", "target", "coverage", "n", "accuracy", "macro_f1", "weighted_f1", "qwk"]
    if "epmv_e" in rows.columns:
        cols += ["epmv_e", "epmv_e_flag_outside_range"]
    return rows[cols], note


E9B_SUMMARY = "E9b_nn_cases_summary.csv"
E9B_COUNT_COLUMNS = ("case_b_clusters", "case_b_rows", "case_b_pairs", "case_c_pairs", "case_c_rows",
                     "primary_n_zero_distance", "primary_n_self_in_second_column")
E9B_RATE_COLUMNS = ("primary_disagreement", "primary_agreement", "primary_chance_disagreement",
                    "case_b_cluster_agreement", "case_b_chance_cluster_agreement", "case_b_pair_agreement",
                    "case_b_pair_disagreement", "case_b_chance_pair_disagreement", "case_c_agreement",
                    "case_c_disagreement", "case_c_chance_pair_disagreement", "chance_disagreement_cohort")


def _nn_cases_module():
    import sys
    from .inputs import REPO_ROOT
    for path in (str(REPO_ROOT), str(REPO_ROOT / "analysis")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import nn_cases
    return nn_cases


def nn_case_rows(inputs: RunInputs) -> pd.DataFrame:
    """E9b, the paper's NN analysis (pre-revision definitions: primary 1-NN, Case B, Case C; F12, full
    cohort), recomputed here from the E9b cohort input (hash-verified) and the variables of the E9b detail
    files, and required equal to the run summary (counts exactly, rates to 1e-12). The cohort must be the
    cohort of the accuracies drawn beside it (E1, same SHA-256)."""
    nc = _nn_cases_module()
    summ = inputs.run_csv("E9b", E9B_SUMMARY)
    if not ((summ["run_mode"] == "real").all() and summ["frozen_definition"].astype(bool).all()
            and (summ["population"] == "full").all()):
        raise ValueError(f"{E9B_SUMMARY}: not a real full-cohort run with the frozen definition")
    if sorted(summ["target"]) != sorted(TARGETS) or summ["target"].duplicated().any():
        raise ValueError(f"{E9B_SUMMARY}: targets {list(summ['target'])}, expected one row per {list(TARGETS)}")
    if inputs.manifest("E9b")["inputs_sha256"].get("cohort_csv") != inputs.manifest("E1")["inputs_sha256"].get(
            "cohort_csv"):
        raise ValueError("E9b and E1 did not read the same cohort file")
    raw = inputs.input_file("E9b", "cohort_csv")
    variables = None
    for t in TARGETS:
        d = json.loads(inputs.run_file("E9b", f"E9b_nn_cases_detail_{t}.json"))
        if variables is None:
            variables = list(d["variables"])
        if list(d["variables"]) != variables:
            raise ValueError(f"E9b detail files of {t}: other variables")
    if variables != list(nc.FROZEN_F12):
        raise ValueError(f"E9b variables {variables} are not F12, the feature set of the paper's NN analysis")
    numerical = [v for v in variables if v not in nc.CATEGORICAL_VARIABLES]
    categorical = [v for v in variables if v in nc.CATEGORICAL_VARIABLES]
    df, _ = nc.load_cohort(Path("cohort_csv"), numerical, categorical, list(TARGETS), raw=raw)
    X = df[numerical + categorical]
    X_std = nc.encode_features(X)
    rows = []
    for t in TARGETS:
        r = summ[summ["target"] == t].iloc[0]
        if int(r["n"]) != len(df):
            raise ValueError(f"E9b {t}: n = {r['n']}, cohort has {len(df)} votes")
        y = df[t].to_numpy()
        a = nc.primary_disagreement(X_std, y)
        b, c = nc.case_b_identical_groups(X, y), nc.case_c_physio_margins(X, y)
        got = {"primary_disagreement": a["disagreement_rate"], "primary_agreement": a["agreement_rate"],
               "primary_chance_disagreement": a["chance_pair_disagreement"],
               "primary_n_zero_distance": a["n_zero_distance_neighbour"],
               "primary_n_self_in_second_column": a["n_self_in_second_column"],
               "case_b_clusters": b["n_clusters"], "case_b_rows": b["n_rows_in_clusters"],
               "case_b_pairs": b["n_pairs_considered"], "case_c_pairs": c["n_pairs_considered"],
               "case_c_rows": c["n_rows_in_pairs"],
               "case_b_cluster_agreement": b["cluster_agreement_rate"],
               "case_b_chance_cluster_agreement": b["chance_cluster_agreement"],
               "case_b_pair_agreement": b["pair_agreement_rate"], "case_b_pair_disagreement": b["pair_disagreement_rate"],
               "case_b_chance_pair_disagreement": b["chance_pair_disagreement"],
               "case_c_agreement": c["agreement_rate"], "case_c_disagreement": c["disagreement_rate"],
               "case_c_chance_pair_disagreement": c["chance_pair_disagreement"],
               "chance_disagreement_cohort": b["chance_disagreement_cohort"]}
        for col in E9B_COUNT_COLUMNS:
            if int(r[col]) != int(got[col]):
                raise ValueError(f"E9b {t}: {col} = {r[col]} in the run, {got[col]} recomputed")
        for col in E9B_RATE_COLUMNS:
            a_, e = float(r[col]), float(got[col])
            if not ((np.isnan(a_) and np.isnan(e)) or np.isclose(a_, e, rtol=0, atol=1e-12)):
                raise ValueError(f"E9b {t}: {col} = {a_} in the run, {e} recomputed")
        rows.append({"target": t, "n": len(df), **got,
                     "nn_agreement": got["primary_agreement"],
                     "chance_agreement": 1 - got["primary_chance_disagreement"],
                     "case_b_chance_pair_agreement": 1 - got["case_b_chance_pair_disagreement"],
                     "case_c_chance_agreement": 1 - got["case_c_chance_pair_disagreement"],
                     "variables": ";".join(variables)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------------
def _save_fig(out: OutputDir, fig, stem: str, png_dpi: int = 300, pdf: bool = True, **kw) -> None:
    if pdf:
        out.savefig(fig, f"{stem}.pdf", **kw)
    out.savefig(fig, f"{stem}.png", dpi=png_dpi, **kw)
    plt.close(fig)


def _caption(out: OutputDir, stem: str, label: str, files: str, width: str, text: str) -> None:
    out.write_text(f"captions/{stem}.tex", captions.caption_file(label, files, width, text))


def _final_size_rc():
    return {"font.family": "sans-serif", "font.sans-serif": bar_charts.FONT_FAMILY, "pdf.fonttype": 42,
            "ps.fonttype": 42}


def build_distributions(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    ashrae = pd.read_csv(io.BytesIO(inputs.cohort_csv()))
    french = {c: pd.read_csv(io.BytesIO(inputs.input_file("E6", c))) for c in ("Moujalled", "Hostein")}
    zs = inputs.aggregate("E6_full_cohort_zero_shot")
    datasets = {"ashrae": ("ASHRAE_2022", ashrae), "moujalled": ("Moujalled", french["Moujalled"]),
                "hostein": ("Hostein", french["Hostein"])}
    records, sizes = [], {}
    for key, (ds, df) in datasets.items():
        name = DATASET_NAME[ds]
        sizes[ds] = len(df)
        for t in TARGETS:
            y = pd.to_numeric(df[t], errors="raise")
            if y.isna().any():
                raise ValueError(f"{name} {t}: missing votes")
            # every class of the scale, an empty one as a zero bar: one position and colour per vote value
            counts = {int(c): int((y == c).sum()) for c in ONTOLOGY[t]}
            if sum(counts.values()) != len(y):
                raise ValueError(f"{name} {t}: votes outside the ontology")
            if key != "ashrae":
                sup = json.loads(zs[(zs["cohort"] == name) & (zs["target"] == t)]["support"].iloc[0])
                if {int(k): v for k, v in sup.items() if v} != {c: v for c, v in counts.items() if v}:
                    raise ValueError(f"{name} {t}: counts {counts} differ from the E6 support {sup}")
            for c, n in counts.items():
                records.append({"dataset": name, "target": t, "class": c, "count": n})
            rc = {**_final_size_rc(), "axes.spines.top": False, "axes.spines.right": False}
            with plt.rc_context(rc):
                fig, ax = plt.subplots(figsize=(THIRD_WIDTH_IN, 1.95), layout="constrained")
                xs = np.arange(len(counts))
                ax.bar(xs, list(counts.values()), color=[DIST_COLOURS[t][c] for c in counts], width=0.8,
                       edgecolor="#444444", linewidth=0.3)
                top = max(counts.values())
                for x, v in zip(xs, counts.values()):
                    ax.text(x, v + 0.02 * top, f"{v}", ha="center", va="bottom", fontsize=7)
                ax.set_ylim(0, 1.2 * top)                 # value labels clear of the frame
                ax.set_xticks(xs)
                words = DIST_TICK_WORD.get(t)
                ax.set_xticklabels([vote_code(c) + (f"\n{words[c]}" if words else "") for c in counts],
                                   fontsize=7.5, linespacing=1.0)
                ax.tick_params(axis="y", labelsize=7.5)
                ax.tick_params(length=2, width=0.5)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_xlabel(f"{TARGET_LABEL[t]} vote", fontsize=8)
                ax.set_title(f"{TARGET_LABEL[t]} — {name} (n = {len(y):,})".replace(",", " "),
                             fontsize=7.5, fontweight="bold")
                _save_fig(out, fig, f"distributions/{DIST_FILE[t]}_{key}", pad_inches=PAD_IN)
    out.write_csv("distributions/class_counts.csv", pd.DataFrame(records))
    _caption(out, "distributions", "fig:distributions",
             "distributions/{tsv7,tsv3,tpv}_{ashrae,moujalled,hostein}.{pdf,png}", "0.32\\linewidth",
             captions.distributions(sizes))


def phase1_tpv_identical(sub: pd.DataFrame) -> List[str]:
    """Datasets on which every Haghirad Opt row of TPV equals its NotOpt row (same class weighting) on the four
    metrics: Haghirad's TPV optimum is the NotOpt setting, so the equal bars are genuine."""
    out = []
    for ds in ("ASHRAE_2018", "ASHRAE_2022"):
        g = sub[(sub["dataset"] == ds) & (sub["target"] == "thermal_preference")].set_index("config")
        pairs = [(f"Haghirad_NotOpt_{w}", f"Haghirad_Opt_{w}") for w in ("NoCW", "Bal", "BalSub")]
        if all(a in g.index and b in g.index for a, b in pairs) and all(
                np.allclose(g.loc[a, [m for m, _ in METRICS4]].astype(float),
                            g.loc[b, [m for m, _ in METRICS4]].astype(float), rtol=0, atol=0) for a, b in pairs):
            out.append(DATASET_NAME[ds])
    return out


PHASE1_ROW_LABELS = {"ASHRAE_2018": "ASHRAE-2018\n(Haghirad 2018 v2 release)", "ASHRAE_2022": "ASHRAE-2022"}


def build_phase1(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    grid = inputs.phase1_grid()
    specs = [("haghirad", tables.PHASE1_HAGHIRAD_CONFIGS, "tab:phase1_haghirad",
              bar_charts.PHASE1_HAGHIRAD_COLOURS, "Model Configuration (Haghirad):", 6.49, 0.1),
             ("highcap", tables.PHASE1_HIGHCAP_CONFIGS, "tab:phase1_highcap",
              bar_charts.PHASE1_HIGHCAP_COLOURS, "Model Configuration:", 6.3, 0.2)]
    for key, configs, label, colours, legend, height, ystep in specs:
        sub = grid[(grid["encoding"] == "onehot") & (grid["ros"].astype(str) == "False")
                   & grid["config"].isin(configs)].copy()
        identical = phase1_tpv_identical(sub) if key == "haghirad" else []
        caption = captions.phase1(key, identical)
        out.write_text(f"table_phase1_{key}.tex", tables.phase1_table(grid, configs, label,
                                                                     captions.phase1(key, identical, kind="rows")))
        out.write_csv(f"table_phase1_{key}.csv", sub)
        sub["x"] = sub["target"].map(TARGET_LABEL)
        long = bar_charts.long_from_wide(sub, METRICS4, "x", "config", "dataset")
        fig = bar_charts.facet_bars(long, panel_order=[m for _, m in METRICS4], x_order=list(TARGET_LABEL.values()),
                                fill_order=configs, colours=colours, legend_title=legend,
                                row_order=["ASHRAE_2018", "ASHRAE_2022"], row_labels=PHASE1_ROW_LABELS,
                                ystep=ystep, height_in=height, legend_ncol=3)
        _save_fig(out, fig, f"phase1_{key}_bars")
        _caption(out, f"phase1_{key}_bars", f"fig:phase1_{key}", f"phase1_{key}_bars.{{pdf,png}}", "\\linewidth",
                 caption)


def build_phase2(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    seeds, means, note = phase2_rows(inputs)
    order = [MODEL_DISPLAY[m] for m in E1_MODELS + E10_MODELS]
    tab = means.copy()
    tab["model"] = tab["model_display"]
    n_test = sorted({int(v) for v in seeds["n"]})
    if len(n_test) != 1:
        raise ValueError(f"Phase 2 rows on different test sets: n = {n_test}")
    out.write_text("table_phase2_indomain.tex", tables.phase2_table(tab, order, note, n_test=n_test[0]))
    out.write_csv("table_phase2_indomain.csv", means.assign(seeds=means["seeds"].map(json.dumps)))
    out.write_csv("table_phase2_indomain_per_seed.csv",
                  seeds[["experiment", "target", "model", "seed", "n", "accuracy", "macro_f1", "weighted_f1", "qwk"]])
    present = [m for m in order if m in set(means["model_display"])]
    long = bar_charts.long_from_wide(means.assign(x=means["target"].map(TARGET_LABEL)), METRICS4, "x", "model_display")
    fig = bar_charts.facet_bars(long, panel_order=[m for _, m in METRICS4], x_order=list(TARGET_LABEL.values()),
                            fill_order=present, colours=bar_charts.MODEL_COLOURS, legend_title="Machine Learning Model:",
                            ncols=2, ystep=0.2, ymax=1.0, height_in=6.3,
                            legend_ncol=int(np.ceil(len(present) / 2)), legend_byrow=False,
                            strip_fill="#F2F2F2")
    _save_fig(out, fig, "phase2_indomain_bars")
    _caption(out, "phase2_indomain_bars", "fig:phase2_indomain", "phase2_indomain_bars.{pdf,png}", "\\linewidth",
             captions.phase2(tables.seeds_sentence(tab), n_test[0], hybrids=any(m in E10_MODELS for m in means["model"])))


def build_confusion_matrices(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    """One matrix per target x model, drawn at 0.49 text width; the colour scale is shared by the five models of
    a target, so the shading compares across panels."""
    import seaborn as sns
    cm_long = inputs.aggregate("E1_confusion_matrices")
    e1 = inputs.aggregate("E1_metrics")
    seeds_wanted = [42, 43, 44] if opt.cm_seed == "sum" else [int(opt.cm_seed)]
    records = []
    for t in TARGETS:
        labels = list(ONTOLOGY[t])
        mats = {}
        for m in E1_MODELS:
            sub = cm_long[(cm_long["target"] == t) & (cm_long["model"] == m)]
            avail = sorted(int(s) for s in sub["seed"].unique())
            use = [s for s in seeds_wanted if s in avail]
            if not use:
                raise ValueError(f"{t} {m}: no confusion matrix for seed(s) {seeds_wanted} (available {avail})")
            total = np.zeros((len(labels), len(labels)), dtype=np.int64)
            for s in use:
                g = sub[sub["seed"] == s]
                mat = g.pivot(index="true", columns="pred", values="count").reindex(index=labels, columns=labels)
                mat = mat.fillna(0).astype(np.int64).to_numpy()
                ref = e1[(e1["row_type"] == "seed") & (e1["target"] == t) & (e1["model"] == m)
                         & (e1["seed"].astype(float) == s)]["confusion_matrix"]
                if not np.array_equal(mat, as_matrix(ref.iloc[0])):
                    raise ValueError(f"{t} {m} seed {s}: E1 confusion matrices differ from the E1 metrics")
                total += mat
            mats[m] = (total, "seed " + "+".join(str(s) for s in use))
        vmax = int(max(int(mat.max()) for mat, _ in mats.values()))
        ticks = [vote_code(c) for c in labels]
        for m, (total, seed_txt) in mats.items():
            with plt.rc_context(_final_size_rc()):
                fig, ax = plt.subplots(figsize=(HALF_WIDTH_IN, 2.85), layout="constrained")
                sns.heatmap(total, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax, xticklabels=ticks,
                            yticklabels=ticks, ax=ax, annot_kws={"fontsize": 7.5 if len(labels) > 3 else 8.5},
                            cbar_kws={"label": "Count (scale shared by the five models)"})
                ax.tick_params(labelsize=8, length=2)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xlabel("Predicted vote", fontsize=8)
                ax.set_ylabel("True vote", fontsize=8)
                ax.set_title(f"{TARGET_LABEL[t]} — {MODEL_DISPLAY[m]} ({seed_txt})", fontsize=8.5,
                             fontweight="bold")
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=7, length=2)
                cbar.ax.yaxis.label.set_size(7)
                out.savefig(fig, f"confusion_matrices/{t}__{MODEL_FILE[m]}_confusion_matrix.pdf", pad_inches=PAD_IN)
                out.savefig(fig, f"confusion_matrices/{t}__{MODEL_FILE[m]}_confusion_matrix.png", dpi=300,
                            pad_inches=PAD_IN)
                plt.close(fig)
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    records.append({"target": t, "model": m, "seeds": seed_txt, "true": a, "pred": b,
                                    "count": int(total[i, j]), "colour_scale_max": vmax})
        n_test = {int(v) for v in e1[(e1["row_type"] == "seed") & (e1["target"] == t)]["n"]}
        if len(n_test) != 1:
            raise ValueError(f"{t}: E1 seed rows on different test sets {sorted(n_test)}")
        stem = {"thermal_sensation": "tsv7", "TSV_3p": "tsv3", "thermal_preference": "tpv"}[t]
        seed_txts = sorted({txt for _, txt in mats.values()})
        _caption(out, f"{stem}_confusion_matrices", f"fig:{stem}_confusion_matrices",
                 f"confusion_matrices/{t}__{{{','.join(MODEL_FILE[m] for m in E1_MODELS)}}}_confusion_matrix"
                 ".{pdf,png}", "0.49\\linewidth",
                 captions.confusion(t, " / ".join(seed_txts), vmax, n_test.pop()))
    out.write_csv("confusion_matrices/confusion_matrices.csv", pd.DataFrame(records))


def build_indices(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    rows, note = index_rows(inputs, include_tpv=opt.indices_tpv)
    present = [i for i in INDEX_ORDER if i in set(rows["index"])]
    out.write_text("table_heat_balance_indices.tex", tables.indices_table(rows, present, note))
    out.write_csv("table_heat_balance_indices.csv", rows)
    long = pd.DataFrame({"panel": rows["target"].map(TARGET_LABEL), "x": rows["index"], "value": rows["accuracy"]})
    panels = [TARGET_LABEL[t] for t in TARGETS if t in set(rows["target"])]
    # coverage under each index: the bars rest on different subsets of votes
    ticks = {(TARGET_LABEL[r["target"]], r["index"]): f"{r['index']} ({100 * float(r['coverage']):.0f}%)"
             for _, r in rows.iterrows()}
    fig = bar_charts.index_bars(long, panel_order=panels, x_order=present, colours=bar_charts.INDEX_COLOURS,
                            tick_labels=ticks)
    _save_fig(out, fig, "heat_balance_indices_bars")
    coverage = {}
    for _, r in rows.iterrows():
        coverage.setdefault(r["index"], {})[r["target"]] = float(r["coverage"])
    e1 = inputs.aggregate("E1_metrics")
    n_test = sorted({int(v) for v in e1[e1["row_type"] == "seed"]["n"]})
    if len(n_test) != 1:
        raise ValueError(f"E1 seed rows on different test sets: {n_test}")
    n_test = n_test[0]
    _caption(out, "heat_balance_indices_bars", "fig:heat_balance_indices", "heat_balance_indices_bars.{pdf,png}",
             "\\linewidth", captions.indices({i: coverage[i] for i in present}, n_test))


def _fi_long(inputs: RunInputs, seeds: Sequence[int]) -> pd.DataFrame:
    import joblib
    import sys
    from .inputs import REPO_ROOT
    from .xgb_ubj import booster_without_runtime
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    # a numba JIT cache, if any library compiles one, must be writable: default it under TMPDIR
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tpb_numba_cache"))
    import models  # noqa: E402  (restored functions: _extract_feature_importances_from_pipeline)
    recs = []
    for t in TARGETS:
        for fam, model_name in (("RF", "RandomForest"), ("XGB", "XGBoost")):
            for s in seeds:
                raw = inputs.run_file("E1", f"{t}/{fam}/seed_{s}/{t}__{fam}__seed{s}.joblib")
                with booster_without_runtime():
                    pipe = joblib.load(io.BytesIO(raw))
                extr = models._extract_feature_importances_from_pipeline(pipe)
                if extr is None:
                    raise ValueError(f"{t} {fam} seed {s}: no feature importance")
                importances, feat_out, groups = extr
                if len(set(feat_out)) != len(feat_out) or any(str(f).startswith("f") and str(f)[1:].isdigit()
                                                               for f in feat_out):
                    raise ValueError(f"{t} {fam} seed {s}: transformed feature names not recovered")
                # models.plot_feature_importance_pipeline: sum over the one-hot parts, normalise to 1
                grouped = pd.DataFrame({"group": groups, "importance": importances}).groupby("group")["importance"].sum()
                grouped = grouped / grouped.sum()
                for g, v in grouped.items():
                    recs.append({"target": t, "model": model_name, "seed": s, "group": g, "importance": float(v)})
    return pd.DataFrame(recs)


FI_MEASURE = {"RandomForest": "impurity-based (MDI)", "XGBoost": "gain-based"}


def build_feature_importance(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    import seaborn as sns
    import sys
    from .inputs import REPO_ROOT
    if str(REPO_ROOT / "analysis") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "analysis"))
    import feature_importance_heatmap as fih  # restored heatmap (analysis/feature_importance_heatmap.py)
    long = _fi_long(inputs, opt.fi_seeds)
    out.write_csv("feature_importance/feature_importance_per_seed.csv", long)
    mean = long.groupby(["target", "model", "group"], as_index=False)["importance"].mean()
    cols = {}
    for model in fih.MODELS:
        for t, tl in fih.TARGETS:
            cols[(fih.MODEL_LABELS[model], tl)] = mean[(mean["target"] == t) & (mean["model"] == model)].set_index(
                "group")["importance"]
    mat = pd.DataFrame(cols)
    missing = [f for f in fih.FEATURE_ORDER if f not in mat.index]
    if missing or len(mat.index) != len(fih.FEATURE_ORDER):
        raise KeyError(f"feature groups {sorted(mat.index)} do not match the heatmap order (missing {missing})")
    mat = mat.loc[fih.FEATURE_ORDER]
    pdf = out.new("feature_importance/feature_importance_heatmap.pdf")
    png = pdf.with_suffix(".png")
    out.written.remove(pdf)                      # make_heatmap creates both files exclusively itself
    fih.make_heatmap(mat, pdf)
    out.written += [pdf, png]
    out.write_csv("feature_importance/feature_importance_heatmap.csv", mat, index=True)
    _caption(out, "feature_importance_heatmap", "fig:feature_importance",
             "feature_importance/feature_importance_heatmap.{pdf,png}", "\\linewidth",
             captions.feature_importance_heatmap(opt.fi_seeds))
    for t in TARGETS:
        for model in fih.MODELS:
            grouped = mean[(mean["target"] == t) & (mean["model"] == model)].set_index("group")["importance"]
            grouped = grouped.sort_values(ascending=False)
            name = MODEL_NAME["RF" if model == "RandomForest" else "XGB"]
            # models.plot_feature_importance_pipeline layout, drawn at 0.49 text width
            with plt.rc_context(_final_size_rc()):
                fig, ax = plt.subplots(figsize=(HALF_WIDTH_IN, 3.0), layout="constrained")
                # display labels of the heatmap (English: Sexe -> gender, Âge -> age); CSV group names unchanged
                sns.barplot(x=grouped.values, y=[fih.FEATURE_LABELS.get(g, g) for g in grouped.index],
                            color="green", ax=ax)
                ax.set_xlabel(f"Normalised {FI_MEASURE[model]} importance", fontsize=8)
                ax.set_ylabel("Feature", fontsize=8)
                ax.tick_params(labelsize=7.5, length=2)
                ax.set_title(f"{name} — {TARGET_LABEL[t]}", fontsize=8.5, fontweight="bold")
                out.savefig(fig, f"feature_importance/{t}_{model}_feature_importance.pdf", pad_inches=PAD_IN)
                out.savefig(fig, f"feature_importance/{t}_{model}_feature_importance.png", dpi=300,
                            pad_inches=PAD_IN)
                plt.close(fig)
    _caption(out, "feature_importance_bars", "fig:feature_importance_bars",
             "feature_importance/{target}_{RandomForest,XGBoost}_feature_importance.{pdf,png}", "0.49\\linewidth",
             captions.feature_importance_bars(opt.fi_seeds))


def _one_row_per(df: pd.DataFrame, keys: List[str], expected: set, what: str) -> None:
    got = list(map(tuple, df[keys].astype(str).to_numpy()))
    if len(got) != len(set(got)) or set(got) != expected:
        raise ValueError(f"{what}: expected one row per {keys} in {sorted(expected)}, got {sorted(got)}")


def build_phase3(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    zs = inputs.aggregate("E6_full_cohort_zero_shot")
    zs_models = zs[zs["model"].isin(PHASE3_ZERO_SHOT)]
    zs_models = _metric_rows(zs_models, "E6 zero-shot")
    zs_models["model"] = zs_models["model"].map(PHASE3_ZERO_SHOT)
    zs_major = zs[zs["model"].isin(PHASE3_MAJORITY_ZERO_SHOT)]
    _one_row_per(zs_major, ["cohort", "target"], {(c, t) for c in tables.COHORTS for t in TARGETS},
                 "E6 zero-shot majority class")
    zs_major = _metric_rows(zs_major, "E6 zero-shot majority class")
    zs_major["model"] = zs_major["model"].map(PHASE3_MAJORITY_ZERO_SHOT)
    zs_rows = pd.concat([zs_models, zs_major], ignore_index=True)
    keep = ["cohort", "target", "model", "n", "accuracy", "macro_f1", "weighted_f1", "qwk"]
    order = list(PHASE3_ZERO_SHOT.values()) + list(PHASE3_MAJORITY_ZERO_SHOT.values())
    out.write_text("table_phase3_zero_shot.tex", tables.phase3_direct_table(zs_rows[keep], order))
    # the other E6 baselines stay in the CSV only; weighted-F1 from their confusion matrix
    base = zs[~zs["model"].isin(list(PHASE3_ZERO_SHOT) + list(PHASE3_MAJORITY_ZERO_SHOT))]
    if "confusion_matrix" in base.columns and base["confusion_matrix"].notna().all():
        base = _metric_rows(base, "E6 other baselines")
    out.write_csv("table_phase3_zero_shot.csv", pd.concat([zs_rows[keep], base[[c for c in keep if c in base.columns]]],
                                                          ignore_index=True))
    long = bar_charts.long_from_wide(zs_rows.assign(x=zs_rows["target"].map(TARGET_LABEL)), METRICS4, "x", "model",
                                 "cohort")
    fig = bar_charts.facet_bars(long, panel_order=[m for _, m in METRICS4], x_order=list(TARGET_LABEL.values()),
                            fill_order=order, colours={**PHASE3_COLOURS, "Majority class": MAJORITY_COLOUR},
                            legend_title="Model:", row_order=["Moujalled", "Hostein"],
                            row_labels={c: DATASET_NAME[c] for c in ("Moujalled", "Hostein")}, ystep=0.1,
                            height_in=6.1, legend_ncol=4)
    _save_fig(out, fig, "phase3_zero_shot_bars")
    n = {c: int(zs_rows[zs_rows["cohort"] == c]["n"].iloc[0]) for c in tables.COHORTS}
    _caption(out, "phase3_zero_shot_bars", "fig:phase3_zero_shot", "phase3_zero_shot_bars.{pdf,png}", "\\linewidth",
             captions.phase3(n))
    st_all = inputs.aggregate("E6_strategy_metrics")
    st = st_all[(st_all["row_type"] == "strategy") & st_all["strategy"].isin(PHASE3_ADAPTIVE)]
    st = _metric_rows(st, "E6 strategies")
    st["variant"] = st["strategy"].map(PHASE3_ADAPTIVE)
    st_major = st_all[(st_all["row_type"] == "baseline") & st_all["strategy"].isin(PHASE3_MAJORITY_ADAPTIVE)]
    _one_row_per(st_major, ["cohort", "target", "regime"],
                 {(c, t, r) for c, t, r in st[["cohort", "target", "regime"]].astype(str).itertuples(index=False)},
                 "E6 adaptive majority class (DEV)")
    st_major = _metric_rows(st_major, "E6 adaptive majority class (DEV)")
    st_major["variant"] = st_major["strategy"].map(PHASE3_MAJORITY_ADAPTIVE)
    st = pd.concat([st, st_major], ignore_index=True)
    keep = ["cohort", "target", "regime", "variant", "n", "accuracy", "macro_f1", "weighted_f1", "qwk"]
    out.write_text("table_phase3_adaptive.tex", tables.phase3_adaptive_table(
        st[keep], list(PHASE3_ADAPTIVE.values()) + list(PHASE3_MAJORITY_ADAPTIVE.values())))
    out.write_csv("table_phase3_adaptive.csv", st[keep])


# the E9b values drawn with --nn_cases_in_figure, each beside its chance reference on the same population
NN_CASE_FIGURE_COLUMNS = ("case_b_cluster_agreement", "case_b_chance_cluster_agreement", "case_b_clusters",
                          "case_c_agreement", "case_c_chance_agreement", "case_c_pairs")


def _cohort_chance_agreement(inputs: RunInputs) -> pd.DataFrame:
    """Σ p_c² over the cohort votes of the accuracies (E1 cohort input), per target."""
    cohort = pd.read_csv(io.BytesIO(inputs.input_file("E1", "cohort_csv")))
    rows = []
    for t in TARGETS:
        y = pd.to_numeric(cohort[t], errors="raise").to_numpy()
        rows.append({"target": t, "n": len(y), "chance_agreement":
                     1 - chance_disagreement(class_distribution(y, ONTOLOGY[t]))})
    return pd.DataFrame(rows)


def build_nn(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    """The paper's NN figure and table, from E9b (the NN analysis as in the paper: F12, full cohort)."""
    from .nn_agreement_vs_accuracy import make_figure
    cohort_chance = _cohort_chance_agreement(inputs)
    if inputs.require("E9b"):
        cases = nn_case_rows(inputs)
        cc = cohort_chance.set_index("target")["chance_agreement"]
        for _, r in cases.iterrows():                    # same cohort, same reference
            if not np.isclose(r["chance_agreement"], cc[r["target"]], rtol=0, atol=1e-12):
                raise ValueError(f"E9b {r['target']}: chance agreement differs from the E1 cohort")
        out.write_text("table_nn_disagreement.tex", tables.nn_analysis_table(cases))
        out.write_csv("table_nn_disagreement.csv", cases)
        ref = cases[["target", "n", "primary_disagreement", "nn_agreement", "primary_chance_disagreement",
                     "chance_agreement"]]
    elif opt.nn_cases_in_figure:
        raise ValueError("--nn_cases_in_figure needs the E9b outputs")
    else:                                                # preview: no NN line, chance line only
        cases = None
        ref = cohort_chance.assign(primary_disagreement=np.nan, nn_agreement=np.nan,
                                   primary_chance_disagreement=1 - cohort_chance["chance_agreement"])
    seeds, means, note = phase2_rows(inputs)
    acc = means.copy()
    acc["model_display"] = acc["model"].map(MODEL_DISPLAY)
    acc["kind"] = np.where(acc["model"].isin(E10_MODELS), "hybrid", "standard")
    # whiskers: the range of the accuracy over the model seeds (not a confidence interval; one seed: none)
    acc["acc_low"], acc["acc_high"] = acc["accuracy_min"], acc["accuracy_max"]
    n = int(ref["n"].iloc[0])
    extra = None
    if opt.nn_cases_in_figure:
        extra = cases.set_index("target")[list(NN_CASE_FIGURE_COLUMNS)]
    order = [MODEL_DISPLAY[m] for m in E1_MODELS + E10_MODELS]
    fig = make_figure(acc, ref[["target", "nn_agreement", "chance_agreement"]], k=1, cases=extra, order=order)
    _save_fig(out, fig, "nn_agreement_vs_accuracy", pad_inches=PAD_IN)
    _caption(out, "nn_agreement_vs_accuracy", "fig:nn_agreement_vs_accuracy", "nn_agreement_vs_accuracy.{pdf,png}",
             "\\linewidth", captions.nn(n, tables.seeds_sentence(acc.assign(model=acc["model_display"])),
                                        cases=extra is not None, hybrids=bool((acc["kind"] == "hybrid").any()),
                                        nn_line=bool(ref["nn_agreement"].notna().any())))
    data = acc[["target", "model", "model_display", "kind", "seeds", "accuracy", "acc_low", "acc_high"]].rename(
        columns={"acc_low": "accuracy_seed_min", "acc_high": "accuracy_seed_max"}).merge(
        ref.assign(population="full", k=1, features="F12 (features_12)",
                   chance_definition=f"sum_c p_c^2, p_c = class shares of the {n} cohort votes"), on="target")
    if extra is not None:
        data = data.merge(extra.reset_index(), on="target")
    out.write_csv("nn_agreement_vs_accuracy_data.csv", data.assign(seeds=data["seeds"].map(json.dumps)))


def build_aggregates(inputs: RunInputs, out: OutputDir, opt: Options) -> None:
    """The aggregate tables read by the other items and the numbers cited in the text (E2 leave-one-publication-out
    aggregates, E4 EMA gains, paired Phase 2 accuracy differences from XGBoost), as recomputed from the run's
    predictions files."""
    from .aggregate import Aggregates
    for name in Aggregates.TABLES:
        inputs.aggregate(name)                       # computed once, cached
        out.write_csv(f"aggregates/{name}.csv", inputs._aggregates.frame(name))


ITEMS: Dict[str, Item] = {i.name: i for i in [
    Item("distributions", "fig:distributions (Method)",
         ["distributions/{tsv7,tsv3,tpv}_{ashrae,moujalled,hostein}.{pdf,png}", "distributions/class_counts.csv",
          "captions/distributions.tex"], build_distributions),
    Item("phase1", "Figure_Table3 + tab:phase1_haghirad; Figure_Table4 + tab:phase1_highcap (Results)",
         ["phase1_haghirad_bars.{pdf,png}", "table_phase1_haghirad.{tex,csv}", "phase1_highcap_bars.{pdf,png}",
          "table_phase1_highcap.{tex,csv}", "captions/phase1_{haghirad,highcap}_bars.tex"], build_phase1),
    Item("phase2", "Figure_Table5 + tab:phase2_indomain (Results)",
         ["phase2_indomain_bars.{pdf,png}", "table_phase2_indomain.{tex,csv}", "table_phase2_indomain_per_seed.csv",
          "captions/phase2_indomain_bars.tex"],
         build_phase2, pending=["E10"]),
    Item("confusion_matrices", "fig:tsv7/tsv3/tpv_confusion_matrices (Results)",
         ["confusion_matrices/{target}__{RandomForest,XGBoost,SVM,ANN,FTTransformer}_confusion_matrix.{pdf,png}",
          "confusion_matrices/confusion_matrices.csv", "captions/{tsv7,tsv3,tpv}_confusion_matrices.tex"],
         build_confusion_matrices),
    Item("indices", "Figure_Table6 + tab:heat_balance_indices (Results)",
         ["heat_balance_indices_bars.{pdf,png}", "table_heat_balance_indices.{tex,csv}",
          "captions/heat_balance_indices_bars.tex"], build_indices,
         pending=["E7b"]),
    Item("feature_importance", "fig:feature_importance (+ per-model importance bars)",
         ["feature_importance/feature_importance_heatmap.{pdf,png,csv}",
          "feature_importance/{target}_{RandomForest,XGBoost}_feature_importance.{pdf,png}",
          "feature_importance/feature_importance_per_seed.csv",
          "captions/feature_importance_{heatmap,bars}.tex"], build_feature_importance),
    Item("phase3", "Figure_Table7 + tab:phase3_direct; tab:phase3_adaptive (Results)",
         ["phase3_zero_shot_bars.{pdf,png}", "table_phase3_zero_shot.{tex,csv}", "table_phase3_adaptive.{tex,csv}",
          "captions/phase3_zero_shot_bars.tex"],
         build_phase3),
    Item("nn", "fig:nn_agreement_vs_accuracy; tab:nn_disagreement (E9b, the paper's NN analysis: primary 1-NN, "
         "Case B, Case C, each beside its chance reference)",
         ["nn_agreement_vs_accuracy.{pdf,png}", "nn_agreement_vs_accuracy_data.csv", "table_nn_disagreement.{tex,csv}",
          "captions/nn_agreement_vs_accuracy.tex"],
         build_nn, pending=["E10", "E9b"]),
    Item("aggregates", "aggregate tables (E1, E6, E7 metrics) and the numbers cited in the text: E2 leave-one-"
         "publication-out aggregates, E4 EMA gains, E1 paired accuracy differences from XGBoost (hybrids: E10)",
         ["aggregates/{E1_metrics,E1_confusion_matrices,E6_full_cohort_zero_shot,E6_strategy_metrics,E7_metrics,"
          "E2_aggregates,E4_ema_gains,E1_model_differences}.csv"], build_aggregates, pending=["E10"]),
]}
