"""Tests for the category normalisation and OOV policy (category normalisation fix).

Blocking tests covered here:

* frozen vocabulary — 100 % of the French values are recognised (251/251 and
  163/163 rows on all five variables); a synthetic unknown value raises; a
  missing value raises.
* OOV values are counted and never mapped to the first class:
  one-hot (global vocabulary, own indicator constant 0 in training), ANN
  (global code kept), FTT (reserved ``<OOV>`` index).
* on the E1 training rows the ANN codes (and the whole ANN input
  tensor) equal the snapshot ``TabPreprocessor`` codes; the one-hot matrix
  equals the snapshot ``make_preprocess`` matrix.

The snapshot oracles are the class / function source of the pre-revision code
read with ``git show`` (skipped when the pre-revision tag is not available).
Real-data tests are skipped when a data file is absent.

Run from the repository root:
    .venv-revision/bin/python -m pytest tests/test_category_normalisation.py -v
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import category_normalisation as cn  # noqa: E402
from category_normalisation import (  # noqa: E402
    CATEGORICAL_VARIABLES,
    GLOBAL_VOCABULARY,
    CategoryVocabularyError,
    coverage_table,
    load_csv_normalised,
    normalise_categories,
    oov_report,
    oov_summary,
)

DATA = REPO_DIR / "Data"
ASHRAE_API = DATA / "ASHRAE_2022_Clean_api.csv"
MOUJALLED = DATA / "Moujalled_api.csv"
HOSTEIN = DATA / "Hostein_api.csv"
ASHRAE_2018 = DATA / "ASHRAE_2018_v2.csv"
SPLITS = REPO_DIR / "splits" / "ASHRAE_2022"
TARGETS = ("thermal_sensation", "TSV_3p", "thermal_preference")

# Frozen vocabulary, copied literally (order as in the reference table).
FROZEN_VOCABULARY = {
    "Sexe": ["female", "male"],
    "Season": ["summer", "winter", "hot/wet", "cool/dry"],
    "Climate": ["Aw", "Am", "BSh", "BSk", "BWh", "Cfa", "Cfb", "Dfb"],
    "Building_type": ["office", "multifamily housing"],
    "cooling type": ["naturally ventilated", "air conditioned", "mixed mode"],
}
FROZEN_EXTERNAL = [
    ("Sexe", "Female", "female"),
    ("Sexe", "Male", "male"),
    ("Season", "Summer", "summer"),
    ("Season", "Winter", "winter"),
    ("Climate", "Cfb", "Cfb"),
    ("Building_type", "Office", "office"),
    ("Building_type", "Multifamily housing", "multifamily housing"),
    ("cooling type", "Naturally Ventilated", "naturally ventilated"),
    ("cooling type", "Naturally ventilated", "naturally ventilated"),
    ("cooling type", "Mixed Mode", "mixed mode"),
]
NUM_COLS = ["Tair", "RH"]
CAT_COLS = list(CATEGORICAL_VARIABLES)


def needs(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"absent: {missing}")


@contextlib.contextmanager
def in_repo_dir():
    """The repo modules read ``config.yaml`` from the working directory at import."""
    old = os.getcwd()
    os.chdir(REPO_DIR)
    try:
        yield
    finally:
        os.chdir(old)


def import_repo(name):
    with in_repo_dir():
        return importlib.import_module(name)


def snapshot_object(filename, name, namespace):
    """Exec the pre-revision source of top-level ``name`` of ``filename``, frozen verbatim in
    ``tests/data/pre_revision_<name>.py.txt``."""
    src = (REPO_DIR / "tests" / "data" / f"pre_revision_{name}.py.txt").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(ast.get_source_segment(src, node), namespace)
    return namespace[name]


def synthetic(rows):
    """Canonical synthetic rows: list of (Sexe, Season, Climate, Building_type, cooling type)."""
    df = pd.DataFrame(rows, columns=CAT_COLS)
    rng = np.random.default_rng(0)
    df.insert(0, "Tair", rng.normal(24, 2, len(df)))
    df.insert(1, "RH", rng.normal(50, 10, len(df)))
    return df


TRAIN_NO_WINTER = synthetic([
    ("female", "summer", "Aw", "office", "air conditioned"),
    ("male", "hot/wet", "Cfa", "office", "mixed mode"),
    ("female", "cool/dry", "Cfb", "multifamily housing", "naturally ventilated"),
    ("male", "summer", "BSh", "office", "naturally ventilated"),
])
EVAL_WITH_WINTER = synthetic([
    ("female", "winter", "Aw", "office", "air conditioned"),   # Season OOV
    ("male", "summer", "Cfa", "office", "mixed mode"),         # no OOV
    ("male", "winter", "Dfb", "office", "mixed mode"),         # Season + Climate OOV
])


# ======================================================================================
# Frozen tables and normalisation
# ======================================================================================

def test_global_vocabulary_is_the_frozen_protocol_table_sorted():
    assert set(GLOBAL_VOCABULARY) == set(CATEGORICAL_VARIABLES)
    for var, values in FROZEN_VOCABULARY.items():
        assert GLOBAL_VOCABULARY[var] == tuple(sorted(values)), var


def test_external_mapping_is_the_frozen_protocol_table():
    flat = [(v, raw, can) for v, t in cn.EXTERNAL_MAPPING.items() for raw, can in t.items()]
    assert sorted(flat) == sorted(FROZEN_EXTERNAL)


@pytest.mark.parametrize("var, raw, canonical", FROZEN_EXTERNAL)
def test_each_french_label_maps_to_its_canonical_value(var, raw, canonical):
    df = pd.DataFrame({var: [raw, f"  {raw} ", canonical]})
    out, coverage = normalise_categories(df, "synthetic", [var])
    assert out[var].tolist() == [canonical] * 3
    assert coverage[0]["rows_mapped"] == 3 and coverage[0]["rows_rejected"] == 0


def test_canonical_values_map_to_themselves_and_whitespace_is_collapsed():
    for var, vocab in GLOBAL_VOCABULARY.items():
        df = pd.DataFrame({var: list(vocab)})
        out, _ = normalise_categories(df, "ashrae", [var])
        assert out[var].tolist() == list(vocab)
    out, _ = normalise_categories(
        pd.DataFrame({"cooling type": ["Naturally   Ventilated", "AIR CONDITIONED"]}), "x"
    )
    assert out["cooling type"].tolist() == ["naturally ventilated", "air conditioned"]


@pytest.mark.parametrize("var, value", [
    ("Season", "Autumn"), ("Season", "spring"), ("Sexe", "undefined"),
    ("Building_type", "Classroom"), ("Climate", "Humid subtropical"),
    ("cooling type", "Unknown"),
    ("Climate", "cfb"),  # Köppen codes are case-sensitive
])
def test_synthetic_unknown_value_raises(var, value):
    df = pd.DataFrame({var: ["Cfb" if var == "Climate" else GLOBAL_VOCABULARY[var][0], value]})
    with pytest.raises(CategoryVocabularyError, match=var):
        normalise_categories(df, "synthetic", [var])


@pytest.mark.parametrize("missing", [np.nan, None, "", "nan", "  "])
def test_missing_value_raises(missing):
    df = pd.DataFrame({"Sexe": ["female", missing]}, dtype=object)
    with pytest.raises(CategoryVocabularyError, match="Sexe"):
        normalise_categories(df, "synthetic", ["Sexe"])


def test_absent_column_and_unregistered_variable_raise():
    with pytest.raises(CategoryVocabularyError):
        normalise_categories(pd.DataFrame({"Sexe": ["male"]}), "x", ["Season"])
    with pytest.raises(CategoryVocabularyError):
        normalise_categories(pd.DataFrame({"Foo": ["a"]}), "x", ["Foo"])


def test_coverage_record_counts_values_and_rows():
    df = pd.DataFrame({"Season": ["Summer", "Summer", "Winter"], "Sexe": ["Male", "Female", "Male"]})
    out, coverage = normalise_categories(df, "toy")
    assert df["Season"].tolist() == ["Summer", "Summer", "Winter"]  # input untouched
    by_var = {c["variable"]: c for c in coverage}
    assert by_var["Season"]["values_seen"] == {"Summer": 2, "Winter": 1}
    assert by_var["Season"]["mapping"] == {"Summer": "summer", "Winter": "winter"}
    assert by_var["Season"]["canonical_counts"] == {"summer": 2, "winter": 1}
    assert by_var["Season"]["rows_mapped"] == 3 and by_var["Season"]["rows_rejected"] == 0
    table = coverage_table(coverage)
    assert (table["coverage_pct"] == 100.0).all()


# ======================================================================================
# OOV relative to a training set
# ======================================================================================

def test_oov_report_counts_flags_and_groups():
    rep = oov_report(TRAIN_NO_WINTER, EVAL_WITH_WINTER, CAT_COLS,
                     groups=pd.Series(["pubA", "pubA", "pubB"]))
    assert rep["oov_flag"].tolist() == [True, False, True]
    assert rep["oov_vars"] == ["Season", "", "Season;Climate"]
    assert rep["n_oov_votes"] == 2 and rep["n_eval"] == 3
    assert rep["counts_per_variable"] == {
        "Sexe": 0, "Season": 2, "Climate": 1, "Building_type": 0, "cooling type": 0,
    }
    assert rep["oov_values"]["Season"] == ["winter"] and rep["oov_values"]["Climate"] == ["Dfb"]
    assert rep["counts_per_group"]["pubA"]["any"] == 1
    assert rep["counts_per_group"]["pubB"] == {
        "Sexe": 0, "Season": 1, "Climate": 1, "Building_type": 0, "cooling type": 0, "any": 1,
    }
    json.dumps(oov_summary(rep))  # serialisable


def test_oov_report_rejects_non_canonical_values():
    raw = EVAL_WITH_WINTER.copy()
    raw.loc[0, "Season"] = "Winter"
    with pytest.raises(CategoryVocabularyError):
        oov_report(TRAIN_NO_WINTER, raw, CAT_COLS)


def test_one_hot_uses_global_vocabulary_and_never_drops_oov():
    mc = import_repo("models_classic")
    vocab = cn.vocabulary_for(CAT_COLS)
    pp = mc.make_preprocess(NUM_COLS, CAT_COLS, cat_vocabulary=vocab).fit(TRAIN_NO_WINTER)
    ohe = pp.named_transformers_["ohe"]
    assert [tuple(c) for c in ohe.categories_] == [vocab[c] for c in CAT_COLS]
    names = list(ohe.get_feature_names_out(CAT_COLS))
    winter = 2 + names.index("Season_winter")
    dfb = 2 + names.index("Climate_Dfb")
    first_season = 2 + names.index("Season_cool/dry")
    Xtr = pp.transform(TRAIN_NO_WINTER)
    Xev = pp.transform(EVAL_WITH_WINTER)
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
    Xev = Xev.toarray() if hasattr(Xev, "toarray") else Xev
    assert (Xtr[:, winter] == 0).all() and (Xtr[:, dfb] == 0).all()   # constant 0 in training
    assert Xev[:, winter].tolist() == [1, 0, 1]                         # own indicator, not dropped
    assert Xev[:, dfb].tolist() == [0, 0, 1]
    assert Xev[[0, 2], first_season].tolist() == [0, 0]                 # not the first class
    # every one-hot block sums to 1 (no all-zero row, no silent drop)
    ohe_block = Xev[:, 2:]
    assert ohe_block.sum(axis=1).tolist() == [len(CAT_COLS)] * 3
    # a string outside the vocabulary cannot be encoded silently
    bad = EVAL_WITH_WINTER.copy()
    bad.loc[0, "Season"] = "autumn"
    with pytest.raises(ValueError):
        pp.transform(bad)


def test_one_hot_equals_snapshot_when_all_categories_are_in_training():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    snap = snapshot_object("models_classic.py", "make_preprocess", {
        "ColumnTransformer": ColumnTransformer, "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    })
    mc = import_repo("models_classic")
    full = synthetic([
        (s, se, cl, b, co)
        for s, se, cl, b, co in zip(
            np.resize(GLOBAL_VOCABULARY["Sexe"], 24), np.resize(GLOBAL_VOCABULARY["Season"], 24),
            np.resize(GLOBAL_VOCABULARY["Climate"], 24), np.resize(GLOBAL_VOCABULARY["Building_type"], 24),
            np.resize(GLOBAL_VOCABULARY["cooling type"], 24))
    ])
    new = mc.make_preprocess(NUM_COLS, CAT_COLS, cn.vocabulary_for(CAT_COLS)).fit_transform(full)
    old = snap(NUM_COLS, CAT_COLS).fit_transform(full)
    np.testing.assert_array_equal(np.asarray(new.todense() if hasattr(new, "todense") else new),
                                  np.asarray(old.todense() if hasattr(old, "todense") else old))


def test_ann_preprocessor_oov_keeps_global_code_and_is_counted():
    md = import_repo("models_deep")
    vocab = cn.vocabulary_for(CAT_COLS)
    pp = md.TabPreprocessor(CAT_COLS, NUM_COLS, vocabulary=vocab, oov_policy="global_code")
    pp.fit(TRAIN_NO_WINTER)
    x_cat, _ = pp.transform(EVAL_WITH_WINTER)
    season = CAT_COLS.index("Season")
    climate = CAT_COLS.index("Climate")
    assert x_cat[:, season].tolist() == [vocab["Season"].index("winter"),
                                         vocab["Season"].index("summer"),
                                         vocab["Season"].index("winter")]
    assert vocab["Season"].index("winter") != 0
    assert x_cat[2, climate].item() == vocab["Climate"].index("Dfb") != 0
    assert pp.n_categories == [len(vocab[c]) for c in CAT_COLS]
    assert pp.last_oov_counts_ == {
        "Sexe": 0, "Season": 2, "Climate": 1, "Building_type": 0, "cooling type": 0,
    }


def test_ftt_preprocessor_maps_oov_to_reserved_index_never_first_class():
    md = import_repo("models_deep")
    vocab = cn.vocabulary_for(CAT_COLS)
    pp = md.TabPreprocessor(CAT_COLS, NUM_COLS, vocabulary=vocab, oov_policy="reserved_index")
    pp.fit(TRAIN_NO_WINTER)
    assert pp.n_categories == [len(vocab[c]) + 1 for c in CAT_COLS]
    x_cat, _ = pp.transform(EVAL_WITH_WINTER)
    season, climate = CAT_COLS.index("Season"), CAT_COLS.index("Climate")
    oov_season = len(vocab["Season"])
    assert x_cat[:, season].tolist() == [oov_season, vocab["Season"].index("summer"), oov_season]
    assert x_cat[2, climate].item() == len(vocab["Climate"])
    # the reserved index is never produced for training rows
    x_tr, _ = pp.transform(TRAIN_NO_WINTER)
    assert (x_tr < torch_tensor(pp.n_categories) - 1).all()
    assert pp.last_oov_counts_["Season"] == 0


def torch_tensor(values):
    import torch
    return torch.tensor(values)


@pytest.mark.parametrize("policy", ["global_code", "reserved_index"])
def test_preprocessor_raises_on_value_outside_vocabulary(policy):
    md = import_repo("models_deep")
    pp = md.TabPreprocessor(CAT_COLS, NUM_COLS, vocabulary=cn.vocabulary_for(CAT_COLS),
                            oov_policy=policy).fit(TRAIN_NO_WINTER)
    bad = EVAL_WITH_WINTER.copy()
    bad.loc[1, "Sexe"] = "Female"   # not normalised → outside vocabulary
    with pytest.raises(CategoryVocabularyError):
        pp.transform(bad)
    # derived vocabulary (vocabulary=None): unseen value raises, never classes_[0]
    pp0 = md.TabPreprocessor(CAT_COLS, NUM_COLS, oov_policy=policy).fit(TRAIN_NO_WINTER)
    with pytest.raises(CategoryVocabularyError):
        pp0.transform(EVAL_WITH_WINTER)
    with pytest.raises(CategoryVocabularyError):
        md.TabPreprocessor(CAT_COLS, NUM_COLS, vocabulary=cn.vocabulary_for(CAT_COLS)).fit(bad)


def test_ann_and_ftt_carry_the_vocabulary_through_fit_and_clone():
    from sklearn.base import clone
    md = import_repo("models_deep")
    vocab = cn.vocabulary_for(CAT_COLS)
    train = pd.concat([TRAIN_NO_WINTER] * 25, ignore_index=True)   # >= 100 rows
    y = np.array([0, 1, 2, 1] * 25)
    common = dict(num_cols=NUM_COLS, cat_cols=CAT_COLS, cat_vocabulary=vocab,
                  epochs=1, device="cpu")
    ann = md.ANNClassifier(hidden=(4,), **common)
    ftt = md.FTClassifier(dim=8, depth=1, heads=2, dim_head=4, **common)
    for est in (ann, ftt):
        assert clone(est).cat_vocabulary == vocab
    ann.fit(train, y)
    ftt.fit(train, y)
    assert ann.pp_.oov_policy == "global_code" and ann.pp_.vocabulary_ == vocab
    assert ftt.pp_.oov_policy == "reserved_index"
    assert ftt.pp_.n_categories == [len(vocab[c]) + 1 for c in CAT_COLS]
    # OOV rows are scored (kept), not dropped
    assert len(ann.predict(EVAL_WITH_WINTER)) == 3
    assert len(ftt.predict(EVAL_WITH_WINTER)) == 3
    assert ftt.pp_.last_oov_counts_["Season"] == 2


def test_config_registry_flags():
    import yaml
    cfg = yaml.safe_load((REPO_DIR / "config.yaml").read_text())
    ds = cfg["data"]["datasets"]
    assert ds["ASHRAE_2022"].get("category_normalisation", True) is True
    assert ds["ASHRAE_2018_v2"]["category_normalisation"] is False


# ======================================================================================
# Real data
# ======================================================================================

@needs(MOUJALLED, HOSTEIN)
@pytest.mark.parametrize("path, name, n_rows", [(MOUJALLED, "Moujalled", 251),
                                                (HOSTEIN, "Hostein", 163)])
def test_french_cohorts_recognised_at_100_percent(path, name, n_rows):
    df, coverage = load_csv_normalised(path, name)
    assert len(df) == n_rows
    assert sorted(c["variable"] for c in coverage) == sorted(CATEGORICAL_VARIABLES)
    for c in coverage:
        assert c["rows_mapped"] == n_rows and c["rows_rejected"] == 0, c
    for var in CATEGORICAL_VARIABLES:
        assert set(df[var]) <= set(GLOBAL_VOCABULARY[var])


@needs(MOUJALLED, HOSTEIN, ASHRAE_API)
def test_unified_loader_normalises_french_and_ashrae_rows():
    udl = import_repo("unified_data_loader")
    cfg = yaml_config()
    feats, targets = cfg["data"]["features"], cfg["data"]["targets"]
    loader = udl.UnifiedDataLoader(feats, targets)
    for path, name, n in [(MOUJALLED, "Moujalled", 251), (HOSTEIN, "Hostein", 163)]:
        df = loader.load_and_validate(str(path), name)
        cov = loader.category_coverage[name]
        assert len(cov) == 5 and all(c["rows_mapped"] == n for c in cov)
        assert df["cooling type"].isin(GLOBAL_VOCABULARY["cooling type"]).all()
    raw = pd.read_csv(ASHRAE_API)
    df = loader.load_and_validate(str(ASHRAE_API), "ASHRAE_2022")
    # ASHRAE-2022 values are already canonical: normalisation is the identity
    for var in CATEGORICAL_VARIABLES:
        assert df[var].tolist() == raw[var].tolist()
    assert all(c["rows_rejected"] == 0 and c["rows_mapped"] == 15_224
               for c in loader.category_coverage["ASHRAE_2022"])


@needs(ASHRAE_2018)
def test_phase1_2018_v2_vocabulary_is_outside_the_frozen_table():
    udl = import_repo("unified_data_loader")
    cfg = yaml_config()
    loader = udl.UnifiedDataLoader(cfg["data"]["features_12"], cfg["data"]["targets"])
    with pytest.raises(CategoryVocabularyError):
        loader.load_and_validate(str(ASHRAE_2018), "ASHRAE_2018_v2")
    df = loader.load_and_validate(str(ASHRAE_2018), "ASHRAE_2018_v2", category_normalisation=False)
    assert len(df) == 17_801


def _loader_csv(tmp_path, sexe_values):
    """Two-row CSV: canonical categories except ``Sexe``; numeric ``Tair``; one target."""
    df = synthetic([
        ("female", "summer", "Aw", "office", "air conditioned"),
        ("male", "winter", "Cfb", "office", "mixed mode"),
    ])
    df["Sexe"] = sexe_values
    df["target"] = [0, 1]
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("sexe_values", [[np.nan, np.nan], [0, 1], [0.0, np.nan]],
                         ids=["all_missing", "numeric_codes", "numeric_and_missing"])
def test_loader_rejects_frozen_categorical_with_numeric_dtype(tmp_path, sexe_values):
    """A frozen categorical inferred numeric must not bypass the vocabulary check."""
    udl = import_repo("unified_data_loader")
    path = _loader_csv(tmp_path, sexe_values)
    assert pd.api.types.is_numeric_dtype(pd.read_csv(path)["Sexe"])
    loader = udl.UnifiedDataLoader(["Sexe", "Tair"], ["target"])
    with pytest.raises(CategoryVocabularyError, match="Sexe"):
        loader.load_and_validate(str(path), "synthetic")


def test_loader_coverage_lists_every_configured_frozen_categorical(tmp_path):
    udl = import_repo("unified_data_loader")
    path = _loader_csv(tmp_path, ["Female", "Male"])
    feats = NUM_COLS + CAT_COLS
    loader = udl.UnifiedDataLoader(feats, ["target"])
    df = loader.load_and_validate(str(path), "synthetic")
    cov = loader.category_coverage["synthetic"]
    assert [c["variable"] for c in cov] == CAT_COLS
    assert all(c["rows_mapped"] == 2 and c["rows_rejected"] == 0 for c in cov)
    assert df["Sexe"].tolist() == ["female", "male"]
    assert pd.api.types.is_float_dtype(df["Tair"])
    # Phase 1 opt-out keeps the legacy dtype routing (no validation, no coverage)
    legacy = udl.UnifiedDataLoader(["Sexe", "Tair"], ["target"])
    out = legacy.load_and_validate(str(_loader_csv(tmp_path, [0, 1])), "p1",
                                   category_normalisation=False)
    assert "p1" not in legacy.category_coverage and out["Sexe"].tolist() == [0, 1]


def test_loader_refuses_missing_or_non_numeric_numerical_features(tmp_path):
    """No family imputes; a missing numerical value would reach the
    networks as NaN logits. The revision loaders refuse it; the Phase 1 opt-out keeps the legacy path."""
    udl = import_repo("unified_data_loader")
    good = pd.read_csv(_loader_csv(tmp_path, ["Female", "Male"]))
    for value, match in ((np.nan, "missing or non-finite"), (np.inf, "missing or non-finite"),
                         ("warm", "not numeric")):
        df = good.copy()
        df["Tair"] = df["Tair"].astype(object) if isinstance(value, str) else df["Tair"]
        df.loc[0, "Tair"] = value
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        loader = udl.UnifiedDataLoader(["Sexe", "Tair"], ["target"])
        with pytest.raises(ValueError, match=match):
            loader.load_and_validate(str(path), "synthetic")
        if not isinstance(value, str):
            legacy = udl.UnifiedDataLoader(["Sexe", "Tair"], ["target"])
            legacy.load_and_validate(str(path), "p1", category_normalisation=False)
    with pytest.raises(KeyError, match="missing"):
        udl.check_numeric_features(good, ["absent"], "synthetic")
    udl.check_numeric_features(good, ["Tair"], "synthetic")


def test_split_feature_types_routes_frozen_categoricals_by_name():
    df = pd.DataFrame({"Tair": [24.0], "Sexe": [0], "Season": ["summer"], "Other": ["x"]})
    num_cols, cat_cols = cn.split_feature_types(df, ["Tair", "Sexe", "Season", "Other"])
    assert num_cols == ["Tair"]
    assert cat_cols == ["Sexe", "Season", "Other"]
    # a non-frozen non-numeric column is categorical and then has no frozen vocabulary
    with pytest.raises(CategoryVocabularyError):
        cn.vocabulary_for(cat_cols)
    assert cn.categorical_features(["Tair", "cooling type", "Sexe"]) == ["cooling type", "Sexe"]


@needs(MOUJALLED, HOSTEIN)
def test_transfert_loader_normalises_and_writes_coverage(tmp_path, machine_compute):
    tr = import_repo("transfert")
    for path, name, n in [(MOUJALLED, "Moujalled", 251), (HOSTEIN, "Hostein", 163)]:
        df = tr.load_external_cohort(path, name, coverage_dir=tmp_path)
        assert len(df) == n
        log = json.loads((tmp_path / f"category_coverage_{name}.json").read_text())
        assert all(c["rows_mapped"] == n and c["rows_rejected"] == 0 for c in log)
        cat_cols = [c for c in CATEGORICAL_VARIABLES]
        rf, xgb = tr.make_rf_xgb_protos(["Tair"], cat_cols, cat_vocabulary=cn.vocabulary_for(cat_cols),
                                        compute=machine_compute)
        ohe = rf.named_steps["pp"].transformers[1][1]
        assert ohe.handle_unknown == "error"


def yaml_config():
    import yaml
    return yaml.safe_load((REPO_DIR / "config.yaml").read_text())


def _e1_train_rows(target):
    split = json.loads((SPLITS / f"{target}_split_indices.json").read_text())
    return split["train_idx"], split["test_idx"]


@needs(ASHRAE_API, SPLITS)
@pytest.mark.parametrize("target", TARGETS)
def test_e1_ann_codes_equal_snapshot_codes_P33(target):
    import torch
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    md = import_repo("models_deep")
    Snap = snapshot_object("models_deep.py", "TabPreprocessor", {
        "pd": pd, "np": np, "torch": torch, "LabelEncoder": LabelEncoder,
        "SK_StandardScaler": StandardScaler,
    })
    cfg = yaml_config()
    feats = cfg["data"]["features"]
    raw = pd.read_csv(ASHRAE_API)
    norm, _ = normalise_categories(raw, "ASHRAE_2022")
    train_idx, test_idx = _e1_train_rows(target)
    num_cols = [c for c in feats if c not in CATEGORICAL_VARIABLES]
    cat_cols = [c for c in feats if c in CATEGORICAL_VARIABLES]

    snap = Snap(cat_cols, num_cols).fit(raw.loc[train_idx, feats])
    ann = md.TabPreprocessor(cat_cols, num_cols, vocabulary=cn.vocabulary_for(cat_cols),
                             oov_policy="global_code").fit(norm.loc[train_idx, feats])
    ftt = md.TabPreprocessor(cat_cols, num_cols, vocabulary=cn.vocabulary_for(cat_cols),
                             oov_policy="reserved_index").fit(norm.loc[train_idx, feats])
    assert ann.n_categories == snap.n_categories
    assert ftt.n_categories == [n + 1 for n in snap.n_categories]
    for rows in (train_idx, test_idx):
        s_cat, s_num = snap.transform(raw.loc[rows, feats])
        a_cat, a_num = ann.transform(norm.loc[rows, feats])
        f_cat, _ = ftt.transform(norm.loc[rows, feats])
        assert torch.equal(a_cat, s_cat)
        assert torch.equal(f_cat, s_cat)          # no OOV in E1: same codes
        assert torch.equal(torch.cat([a_num, a_cat.float()], 1),
                           torch.cat([s_num, s_cat.float()], 1))   # ANN input unchanged
        assert sum(ann.last_oov_counts_.values()) == 0


@needs(ASHRAE_API, SPLITS)
@pytest.mark.parametrize("target", TARGETS)
def test_e1_one_hot_equals_snapshot_and_has_no_oov(target):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    snap = snapshot_object("models_classic.py", "make_preprocess", {
        "ColumnTransformer": ColumnTransformer, "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    })
    mc = import_repo("models_classic")
    feats = yaml_config()["data"]["features"]
    raw = pd.read_csv(ASHRAE_API)
    norm, _ = normalise_categories(raw, "ASHRAE_2022")
    train_idx, test_idx = _e1_train_rows(target)
    num_cols = [c for c in feats if c not in CATEGORICAL_VARIABLES]
    cat_cols = [c for c in feats if c in CATEGORICAL_VARIABLES]
    old = snap(num_cols, cat_cols).fit(raw.loc[train_idx, feats])
    new = mc.make_preprocess(num_cols, cat_cols, cn.vocabulary_for(cat_cols)).fit(norm.loc[train_idx, feats])
    for rows in (train_idx, test_idx):
        a = old.transform(raw.loc[rows, feats])
        b = new.transform(norm.loc[rows, feats])
        a = a.toarray() if hasattr(a, "toarray") else a
        b = b.toarray() if hasattr(b, "toarray") else b
        np.testing.assert_array_equal(a, b)
    rep = oov_report(norm.loc[train_idx], norm.loc[test_idx], cat_cols)
    assert rep["n_oov_votes"] == 0   # E1 OOV = 0


@needs(ASHRAE_API, MOUJALLED, HOSTEIN, SPLITS)
@pytest.mark.parametrize("target", TARGETS)
def test_french_values_occur_in_every_e1_training_set(target):
    """Zero-shot / pooled OOV = 0 (expected counts)."""
    norm, _ = normalise_categories(pd.read_csv(ASHRAE_API), "ASHRAE_2022")
    train_idx, _ = _e1_train_rows(target)
    for path, name in [(MOUJALLED, "Moujalled"), (HOSTEIN, "Hostein")]:
        fr, _ = load_csv_normalised(path, name)
        rep = oov_report(norm.loc[train_idx], fr, list(CATEGORICAL_VARIABLES))
        assert rep["n_oov_votes"] == 0, (name, rep["counts_per_variable"])
