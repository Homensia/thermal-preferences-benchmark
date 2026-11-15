"""
Category normalisation and out-of-vocabulary (OOV) policy.

Single shared module used by every loader (``UnifiedDataLoader``, the Phase 3
external-cohort loader of ``transfert.py``) and every model family (one-hot
preprocessing of ``models_classic.make_preprocess``, ``TabPreprocessor`` of the
ANN and FT-Transformer in ``models_deep.py``).

Frozen content (fixed before the revision runs):

* ``GLOBAL_VOCABULARY``: the categories of the ASHRAE-2022 cohort features,
  stored **sorted** so that the position of a value is its integer code
  (identical to the pre-revision ``LabelEncoder`` codes whenever every category
  occurs in the training rows).
* ``EXTERNAL_MAPPING``: the raw French values (as stored) and their canonical
  value.

Normalisation:
    1. ``str.strip()``, internal whitespace collapsed; ``casefold()`` for every
       variable except ``Climate`` (Köppen codes are case-sensitive).
    2. Look-up in the frozen table. A value absent from the table, or a missing
       value (NaN, empty string, the string ``"nan"``), is a vocabulary error:
       :class:`CategoryVocabularyError` is raised and the run stops.
    3. A coverage record is produced per dataset × variable (values seen, rows
       mapped, rows rejected — always 0 when no exception is raised).

OOV: a value is OOV *for a fit* when it belongs to the global
vocabulary but does not occur in that fit's training rows. It is never mapped
to the first class; :func:`oov_report` counts it per variable (and per group),
and gives the ``oov_flag`` / ``oov_vars`` columns of the predictions files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# FROZEN TABLES
# ======================================================================================

CATEGORICAL_VARIABLES: Tuple[str, ...] = (
    "Sexe", "Season", "Climate", "Building_type", "cooling type",
)

#: Variables compared case-sensitively (step 1 skips ``casefold``).
CASE_SENSITIVE_VARIABLES = frozenset({"Climate"})

#: Frozen global vocabulary, sorted (position = integer code).
GLOBAL_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "Sexe": tuple(sorted(["female", "male"])),
    "Season": tuple(sorted(["summer", "winter", "hot/wet", "cool/dry"])),
    "Climate": tuple(sorted(["Aw", "Am", "BSh", "BSk", "BWh", "Cfa", "Cfb", "Dfb"])),
    "Building_type": tuple(sorted(["office", "multifamily housing"])),
    "cooling type": tuple(sorted(["naturally ventilated", "air conditioned", "mixed mode"])),
}

#: Frozen mapping table for external (French) values, raw value as stored.
EXTERNAL_MAPPING: Dict[str, Dict[str, str]] = {
    "Sexe": {"Female": "female", "Male": "male"},
    "Season": {"Summer": "summer", "Winter": "winter"},
    "Climate": {"Cfb": "Cfb"},
    "Building_type": {"Office": "office", "Multifamily housing": "multifamily housing"},
    "cooling type": {
        "Naturally Ventilated": "naturally ventilated",
        "Naturally ventilated": "naturally ventilated",
        "Mixed Mode": "mixed mode",
    },
}

_MISSING_KEYS = frozenset({"", "nan"})


class CategoryVocabularyError(ValueError):
    """A categorical value is missing or absent from the frozen table."""


# ======================================================================================
# STEP 1 + LOOK-UP TABLE
# ======================================================================================

def normalise_key(variable: str, value) -> Optional[str]:
    """Normalisation step 1. Returns ``None`` for a missing value (NaN/None)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    key = re.sub(r"\s+", " ", str(value).strip())
    if variable not in CASE_SENSITIVE_VARIABLES:
        key = key.casefold()
    return key


def _build_lookup() -> Dict[str, Dict[str, str]]:
    """Frozen table keyed by step-1-normalised values (canonical values map to themselves)."""
    lookup: Dict[str, Dict[str, str]] = {}
    for var, vocab in GLOBAL_VOCABULARY.items():
        table = {normalise_key(var, v): v for v in vocab}
        for raw, canonical in EXTERNAL_MAPPING.get(var, {}).items():
            if canonical not in vocab:
                raise AssertionError(f"{var}: mapping target {canonical!r} not in the vocabulary")
            key = normalise_key(var, raw)
            if table.get(key, canonical) != canonical:
                raise AssertionError(f"{var}: conflicting entries for {key!r}")
            table[key] = canonical
        lookup[var] = table
    return lookup


LOOKUP: Dict[str, Dict[str, str]] = _build_lookup()


def categorical_features(features: Iterable[str]) -> List[str]:
    """Frozen categorical variables among ``features``, selected by NAME (never by dtype)."""
    return [c for c in features if c in CATEGORICAL_VARIABLES]


def split_feature_types(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[List[str], List[str]]:
    """
    ``(num_cols, cat_cols)`` for normalised data: a frozen categorical variable
    is categorical whatever its dtype; any other non-numeric column is also
    categorical (and then fails :func:`vocabulary_for`, never silently numeric).
    """
    cols = list(cols)
    cat_cols = [c for c in cols
                if c in CATEGORICAL_VARIABLES or not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols


def vocabulary_for(cat_cols: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    """Global vocabulary restricted to ``cat_cols`` (raises for an unknown variable)."""
    out = {}
    for c in cat_cols:
        if c not in GLOBAL_VOCABULARY:
            raise CategoryVocabularyError(
                f"Categorical variable {c!r} has no frozen vocabulary."
            )
        out[c] = GLOBAL_VOCABULARY[c]
    return out


# ======================================================================================
# NORMALISATION + COVERAGE
# ======================================================================================

def normalise_categories(
    df: pd.DataFrame,
    dataset_name: str,
    variables: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Normalise the categorical variables of ``df`` to the canonical vocabulary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows (left unmodified).
    dataset_name : str
        Name written in the coverage records.
    variables : sequence of str, optional
        Variables to normalise; default: those of ``CATEGORICAL_VARIABLES``
        present in ``df``. A requested variable absent from ``df`` raises.

    Returns
    -------
    (df_normalised, coverage) : tuple
        A copy of ``df`` with canonical values, and one coverage record per
        variable (see :func:`coverage_table`).

    Raises
    ------
    CategoryVocabularyError
        If any value is missing or absent from the frozen table; the message
        lists the rejected values and their counts.
    """
    if variables is None:
        variables = [v for v in CATEGORICAL_VARIABLES if v in df.columns]
    out = df.copy()
    coverage: List[dict] = []
    errors: List[str] = []
    for var in variables:
        if var not in df.columns:
            raise CategoryVocabularyError(f"{dataset_name}: column {var!r} absent.")
        if var not in LOOKUP:
            raise CategoryVocabularyError(
                f"{dataset_name}: variable {var!r} has no frozen vocabulary."
            )
        table = LOOKUP[var]
        raw = df[var]
        keys = raw.map(lambda v: normalise_key(var, v))
        mapped = keys.map(lambda k: None if (k is None or k in _MISSING_KEYS) else table.get(k))
        rejected = mapped.isna()

        raw_str = raw.map(lambda v: "<NaN>" if (v is None or (not isinstance(v, str) and pd.isna(v))) else str(v))
        values_seen = {str(k): int(n) for k, n in raw_str.value_counts(sort=False).sort_index().items()}
        mapping = {
            str(r): m for r, m in sorted(
                set(zip(raw_str[~rejected], mapped[~rejected]))
            )
        }
        rejected_values = {str(k): int(n) for k, n in raw_str[rejected].value_counts().sort_index().items()}
        coverage.append({
            "dataset": dataset_name,
            "variable": var,
            "n_rows": int(len(df)),
            "rows_mapped": int((~rejected).sum()),
            "rows_rejected": int(rejected.sum()),
            "values_seen": values_seen,
            "mapping": mapping,
            "canonical_counts": {str(k): int(n) for k, n in mapped[~rejected].value_counts().sort_index().items()},
            "rejected_values": rejected_values,
        })
        if rejected.any():
            errors.append(f"{var}: {rejected_values}")
        else:
            out[var] = mapped.astype(object).values

    if errors:
        raise CategoryVocabularyError(
            f"{dataset_name}: values missing or absent from the frozen vocabulary "
            f"— " + "; ".join(errors)
        )
    return out, coverage


def assert_canonical(df: pd.DataFrame, variables: Iterable[str]) -> None:
    """Raise unless every value of ``variables`` is a canonical vocabulary value."""
    for var in variables:
        vocab = set(vocabulary_for([var])[var])
        bad = ~df[var].isin(vocab)
        if bad.any():
            raise CategoryVocabularyError(
                f"{var}: non-canonical values {sorted(map(str, df.loc[bad, var].unique()))} "
                f"(normalise with normalise_categories first)."
            )


def coverage_table(coverage: List[dict]) -> pd.DataFrame:
    """Coverage records as a table: dataset, variable, rows, mapped, rejected, coverage %."""
    rows = []
    for c in coverage:
        rows.append({
            "dataset": c["dataset"],
            "variable": c["variable"],
            "n_rows": c["n_rows"],
            "rows_mapped": c["rows_mapped"],
            "rows_rejected": c["rows_rejected"],
            "coverage_pct": 100.0 * c["rows_mapped"] / c["n_rows"] if c["n_rows"] else float("nan"),
            "values_seen": json.dumps(c["values_seen"], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def write_coverage_log(coverage: List[dict], path) -> Path:
    """Write the coverage records as JSON (normalisation step 3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    return path


def load_csv_normalised(
    csv_path,
    dataset_name: str,
    variables: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Read a CSV and normalise its categorical variables (raises on any vocabulary error)."""
    df = pd.read_csv(csv_path)
    return normalise_categories(df, dataset_name, variables)


# ======================================================================================
# OOV RELATIVE TO A TRAINING SET
# ======================================================================================

def oov_report(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    groups: Optional[pd.Series] = None,
) -> dict:
    """
    Count the evaluation votes whose categories do not occur in the training rows.

    Both frames must already be normalised (canonical values only). No label is used.

    Parameters
    ----------
    train_df, eval_df : pandas.DataFrame
        Training rows of the fit and evaluated rows.
    variables : sequence of str, optional
        Categorical variables; default: ``CATEGORICAL_VARIABLES`` present in ``eval_df``.
    groups : pandas.Series, optional
        Group label of each evaluated row (publication, cohort, ...), aligned
        with ``eval_df``; adds per-group counts.

    Returns
    -------
    dict
        ``oov_flag`` (bool array, one per evaluated row), ``oov_vars`` (list of
        ``";"``-joined variable names, ``""`` if none), ``n_eval``,
        ``n_oov_votes``, ``counts_per_variable`` {var: n votes},
        ``oov_values`` {var: sorted OOV values}, and ``counts_per_group``
        {group: {var: n votes, "any": n}} when ``groups`` is given.
    """
    if variables is None:
        variables = [v for v in CATEGORICAL_VARIABLES if v in eval_df.columns]
    variables = list(variables)
    assert_canonical(train_df, variables)
    assert_canonical(eval_df, variables)

    masks = {}
    oov_values = {}
    for var in variables:
        seen = set(train_df[var].unique())
        m = ~eval_df[var].isin(seen).to_numpy()
        masks[var] = m
        oov_values[var] = sorted(map(str, eval_df.loc[m, var].unique()))

    n = len(eval_df)
    flag = np.zeros(n, dtype=bool)
    for m in masks.values():
        flag |= m
    oov_vars = [
        ";".join(v for v in variables if masks[v][i]) for i in range(n)
    ]
    report = {
        "oov_flag": flag,
        "oov_vars": oov_vars,
        "n_eval": int(n),
        "n_oov_votes": int(flag.sum()),
        "counts_per_variable": {v: int(masks[v].sum()) for v in variables},
        "oov_values": oov_values,
    }
    if groups is not None:
        g = pd.Series(np.asarray(groups), index=range(n))
        per_group = {}
        for key in sorted(g.unique(), key=str):
            sel = (g == key).to_numpy()
            per_group[str(key)] = {v: int(masks[v][sel].sum()) for v in variables}
            per_group[str(key)]["any"] = int(flag[sel].sum())
        report["counts_per_group"] = per_group
    return report


def oov_summary(report: dict) -> dict:
    """JSON-serialisable summary of :func:`oov_report` (per-row arrays dropped)."""
    return {k: v for k, v in report.items() if k not in ("oov_flag", "oov_vars")}
