"""Tests for data_groups.load_group_ids.

Two kinds of tests:

* Data-free tests (always run): the value guard ``assert_aligned``, the
  provenance helpers, and an end-to-end run of ``load_group_ids`` on a small
  synthetic raw CSV / metadata workbook in which discarded rows sit between
  retained rows, so the pairing of votes and identifiers is checked without
  the real database.
* Real-data tests (skipped when a raw / canonical file is absent, e.g. in a
  fresh clone without the raw database). They compare the loader with an
  independent oracle: the raw CSV and the metadata are read and cleaned here
  with plain pandas, without calling ``data_preparation`` or the replay
  helpers of ``data_groups``.

Run from the repository root:
    python -m pytest tests/test_group_ids.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import data_groups  # noqa: E402
from data_groups import (  # noqa: E402
    FLOAT_ATOL,
    OUTPUT_COLUMNS,
    PINNED_RECORD_SEQUENCE_SHA256,
    PINNED_SHA256,
    SOURCE_COLUMNS,
    GroupAlignmentError,
    ProvenanceError,
    assert_aligned,
    default_paths,
    load_group_ids,
    record_sequence_sha256,
    sha256_file,
)

EXPECTED_ROWS = 15_224
EXPECTED_PUBLICATIONS = 10
EXPECTED_BUILDINGS = 367
EXPECTED_SUBJECT_NON_MISSING = 11_754
EXPECTED_HALF_VOTES = {-2.5: 35, -1.5: 61, -0.5: 261, 0.5: 241, 1.5: 119, 2.5: 62}

PATHS = default_paths()
_missing = [str(p) for p in PATHS.values() if not Path(p).is_file()]
needs_data = pytest.mark.skipif(
    bool(_missing), reason=f"raw / canonical data files absent: {_missing}"
)

# Independent restatement of the cleaning rules (NOT imported from
# data_preparation): raw -> canonical column names, and the complete-case filter.
ORACLE_RENAME = {
    "ta": "Tair", "t_out": "Tout", "rh": "RH", "gender": "Sexe",
    "cooling_type": "cooling type", "building_type": "Building_type", "timestamp": "Date",
}
ORACLE_REQUIRED = [
    "Date", "tr", "Tair", "lat", "lon", "year", "RH", "clo", "vel", "Âge", "Tout", "Met",
    "Sexe", "Season", "Building_type", "cooling type", "thermal_sensation", "thermal_preference",
]


# ----------------------------------------------------------------------------
# Data-free tests: value guard
# ----------------------------------------------------------------------------

def test_canonical_csv_is_the_registry_entry_used_by_models_py():
    # models.py --dataset ASHRAE_2022 reads config.yaml data.datasets.ASHRAE_2022.csv
    assert PATHS["canonical_csv"].name == "ASHRAE_2022_Clean_api.csv"


def _toy_frame() -> pd.DataFrame:
    n = 5
    return pd.DataFrame(
        {
            "Tair": [20.0, 21.0, 22.0, 23.0, 24.0],
            "tr": [20.5, 21.5, 22.5, 23.5, 24.5],
            "RH": [40.0, 45.0, 50.0, 55.0, 60.0],
            "vel": [0.1, 0.1, 0.2, 0.2, 0.3],
            "Met": [1.1, 1.2, 1.1, 1.2, 1.1],
            "clo": [0.5, 0.6, 0.7, 0.8, 0.9],
            "Tout": [10.0, 11.0, 12.0, 13.0, 14.0],
            "lat": [45.0] * n,
            "lon": [5.0] * n,
            "Âge": [30.0, 31.0, 32.0, 33.0, 34.0],
            "thermal_sensation": [-1, 0, 1, 2, 0],
            "TSV_3p": [-1, 0, 1, 1, 0],
            "thermal_preference": [1, 0, -1, -1, 0],
            "year": [2020] * n,
            "Date": ["2020-01-01"] * n,
            "Season": ["winter"] * n,
            "Sexe": ["female", "male", "female", "male", "female"],
            "Building_type": ["office"] * n,
            "cooling type": ["mixed mode"] * n,
        }
    )


def test_toy_frame_covers_every_source_column():
    assert set(_toy_frame().columns) == set(SOURCE_COLUMNS)
    assert len(SOURCE_COLUMNS) == 19


def test_assert_aligned_accepts_identical_frames():
    assert_aligned(_toy_frame(), _toy_frame())


def _changed(value):
    return value + 7 if isinstance(value, (int, float, np.integer, np.floating)) else str(value) + "_x"


@pytest.mark.parametrize("column", SOURCE_COLUMNS)
def test_assert_aligned_rejects_a_change_in_any_source_column(column):
    other = _toy_frame()
    other.loc[3, column] = _changed(other.loc[3, column])
    with pytest.raises(GroupAlignmentError, match=r"Values differ at row 3 \(" + column):
        assert_aligned(_toy_frame(), other)


def test_float_tolerance_is_absolute_and_documented():
    other = _toy_frame()
    other.loc[2, "Tair"] += 0.5 * FLOAT_ATOL
    assert_aligned(_toy_frame(), other)
    other.loc[2, "Tair"] += 2 * FLOAT_ATOL
    with pytest.raises(GroupAlignmentError, match="row 2"):
        assert_aligned(_toy_frame(), other)


def test_assert_aligned_reports_earliest_row_over_all_columns():
    other = _toy_frame()
    other.loc[3, "Tair"] = 99.0              # Tair is checked first ...
    other.loc[0, "thermal_sensation"] = 3    # ... but this row comes earlier
    with pytest.raises(GroupAlignmentError, match=r"row 0 \(thermal_sensation"):
        assert_aligned(_toy_frame(), other)


def test_assert_aligned_missing_column_is_an_alignment_error():
    with pytest.raises(GroupAlignmentError, match="Missing columns"):
        assert_aligned(_toy_frame(), _toy_frame().drop(columns=["Date"]))


def test_assert_aligned_rejects_row_count_and_order_changes():
    with pytest.raises(GroupAlignmentError, match="Row count mismatch"):
        assert_aligned(_toy_frame(), _toy_frame().iloc[:-1])
    swapped = _toy_frame().iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
    with pytest.raises(GroupAlignmentError):
        assert_aligned(_toy_frame(), swapped)


def test_value_guard_cannot_see_swaps_of_identical_rows():
    # Documented limitation: identity is enforced by provenance, not values.
    base = pd.concat([_toy_frame(), _toy_frame().iloc[[4]]], ignore_index=True)
    swapped = base.iloc[[0, 1, 2, 3, 5, 4]].reset_index(drop=True)
    assert_aligned(base, swapped)


def test_record_sequence_digest_is_order_sensitive():
    assert record_sequence_sha256([1, 2, 3]) != record_sequence_sha256([2, 1, 3])
    assert record_sequence_sha256([1, 2, 3]) == record_sequence_sha256(np.array([1, 2, 3]))


# ----------------------------------------------------------------------------
# Data-free tests: synthetic end-to-end run of load_group_ids
# ----------------------------------------------------------------------------

def _synthetic_inputs(tmp_path: Path):
    """Write a tiny raw CSV, metadata workbook and expected canonical CSV.

    Raw records 11..19; retained = 11, 13, 16, 19. Discarded rows sit between
    retained rows, for different reasons. Votes include half-votes so that a
    misaligned raw vote would be visible.
    """
    pytest.importorskip("openpyxl")
    ts = "2020-01-0{}T00:00:00Z"
    raw = pd.DataFrame(
        {
            "record_id": [11, 12, 13, 14, 15, 16, 17, 18, 19],
            "building_id": [1, 1, 2, 2, 3, 2, 1, 2, 1],
            "timestamp": [ts.format(d) for d in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
            "Season": ["winter"] * 9,
            "subject_id": [101, 102, np.nan, 104, 105, 7, 107, 108, 101],
            "Âge": [30, 31, 32, 33, 34, 35, 36, 37, 38],
            "gender": ["female", "male", "male", "female", "female", "female", "male", "male", "female"],
            "ta": [20.0, 21.0, 22.0, np.nan, 24.0, 25.0, 26.0, 27.0, 28.0],
            "tr": [20.5] * 9,
            "rh": [50.0] * 9,
            "vel": [0.1] * 9,
            "Met": [1.1] * 9,
            "clo": [0.5] * 9,
            # record 12: no vote; record 18: unparseable vote
            "thermal_sensation": [-2.5, np.nan, -0.5, 1.0, 0.0, 1.5, 2.0, "x", 0.4],
            # record 17: no preference
            "thermal_preference": ["warmer", "cooler", "no change", "cooler", "cooler",
                                   "cooler", np.nan, "cooler", "no change"],
            "t_out": [10.0] * 9,
        }
    )
    meta = pd.DataFrame(
        {
            "building_id": [1, 2, 3],
            "publication": ["pub-A", "pub-B", "pub-C"],
            "lat": [45.0, 46.0, np.nan],  # building 3 has no coordinates: record 15 dropped
            "lon": [5.0, 6.0, np.nan],
            "building_type": ["office"] * 3,
            "cooling_type": ["mixed mode"] * 3,
            "year": [2020] * 3,
        }
    )
    # Expected canonical rows, written out by hand (records 11, 13, 16, 19).
    canonical = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-03", "2020-01-06", "2020-01-09"],
            "Season": ["winter"] * 4,
            "Âge": [30, 32, 35, 38],
            "Sexe": ["female", "male", "female", "female"],
            "Tair": [20.0, 22.0, 25.0, 28.0],
            "tr": [20.5] * 4, "RH": [50.0] * 4, "vel": [0.1] * 4,
            "Met": [1.1] * 4, "clo": [0.5] * 4,
            "thermal_sensation": [-2, 0, 2, 0],   # floor(x + 0.5)
            "thermal_preference": [1, 0, -1, 0],
            "Tout": [10.0] * 4,
            "lat": [45.0, 46.0, 46.0, 45.0], "lon": [5.0, 6.0, 6.0, 5.0],
            "Building_type": ["office"] * 4, "cooling type": ["mixed mode"] * 4,
            "year": [2020] * 4,
            "TSV_3p": [-1, 0, 1, 0],
            "Climate": ["Cfb"] * 4,               # post-filter column, not compared
        }
    )
    p = {
        "raw_csv": tmp_path / "raw.csv",
        "raw_metadata_xlsx": tmp_path / "meta.xlsx",
        "canonical_csv": tmp_path / "canonical.csv",
    }
    raw.to_csv(p["raw_csv"], index=False)
    meta.to_excel(p["raw_metadata_xlsx"], index=False)
    canonical.to_csv(p["canonical_csv"], index=False)
    return p, raw, meta


def test_synthetic_pairs_votes_and_ids_across_discarded_rows(tmp_path):
    p, _, _ = _synthetic_inputs(tmp_path)
    out = load_group_ids(**p, verify_provenance=False)
    assert tuple(out.columns) == OUTPUT_COLUMNS
    assert out["record_id"].tolist() == [11, 13, 16, 19]
    assert out["thermal_sensation_raw"].tolist() == [-2.5, -0.5, 1.5, 0.4]
    assert out["building_id"].tolist() == [1, 2, 2, 1]
    assert out["publication"].tolist() == ["pub-A", "pub-B", "pub-B", "pub-A"]
    assert out["subject_id"].astype("object").where(out["subject_id"].notna(), None).tolist() == [
        101, None, 7, 101
    ]
    assert out.attrs["verified_provenance"] is False


def test_synthetic_inputs_fail_provenance(tmp_path):
    p, _, _ = _synthetic_inputs(tmp_path)
    with pytest.raises(ProvenanceError, match="not the pinned canonical inputs"):
        load_group_ids(**p)


def test_synthetic_duplicate_metadata_building_is_rejected(tmp_path):
    p, _, meta = _synthetic_inputs(tmp_path)
    pd.concat([meta, meta.iloc[[1]]], ignore_index=True).to_excel(p["raw_metadata_xlsx"], index=False)
    with pytest.raises(GroupAlignmentError, match="duplicated building_id"):
        load_group_ids(**p, verify_provenance=False)


def test_synthetic_duplicate_record_id_is_rejected(tmp_path):
    p, raw, _ = _synthetic_inputs(tmp_path)
    raw.loc[2, "record_id"] = 11
    raw.to_csv(p["raw_csv"], index=False)
    with pytest.raises(GroupAlignmentError, match="duplicated record_id"):
        load_group_ids(**p, verify_provenance=False)


def test_synthetic_canonical_with_swapped_rows_is_rejected(tmp_path):
    p, _, _ = _synthetic_inputs(tmp_path)
    canonical = pd.read_csv(p["canonical_csv"])
    canonical.iloc[[1, 2]] = canonical.iloc[[2, 1]].to_numpy()
    canonical.to_csv(p["canonical_csv"], index=False)
    with pytest.raises(GroupAlignmentError, match="Values differ at row 1"):
        load_group_ids(**p, verify_provenance=False)


@pytest.mark.parametrize("key", ["raw_csv", "raw_metadata_xlsx", "canonical_csv"])
def test_inputs_swapped_while_read_fail_the_pins_and_are_never_re_read(tmp_path, monkeypatch, key):
    """Each of the three inputs is read once; the pins are checked on those
    bytes, every frame is parsed from them (no second read of the path), and the hashes reported are those of
    the bytes consumed. A substitution restored right after the read fails the pin check."""
    import hashlib
    import pathlib
    p, _, _ = _synthetic_inputs(tmp_path)
    monkeypatch.setattr(data_groups, "PINNED_SHA256", {k: sha256_file(v) for k, v in p.items()})
    monkeypatch.setattr(data_groups, "PINNED_ROW_COUNT", 4)
    monkeypatch.setattr(data_groups, "PINNED_RECORD_SEQUENCE_SHA256", record_sequence_sha256([11, 13, 16, 19]))
    targets = {Path(v).resolve() for v in p.values()}

    def guard(fn):
        def wrapped(src, *a, **k):
            if isinstance(src, (str, Path)) and Path(src).resolve() in targets:
                raise AssertionError(f"{src} parsed from its path (second read)")
            return fn(src, *a, **k)
        return wrapped
    with monkeypatch.context() as m:
        m.setattr(pd, "read_csv", guard(pd.read_csv))
        m.setattr(pd, "read_excel", guard(pd.read_excel))
        seen = {}
        out = load_group_ids(**p, consumed_sha256=seen)
    assert out["record_id"].tolist() == [11, 13, 16, 19]
    assert seen == {k: sha256_file(v) for k, v in p.items()} == out.attrs["sha256"]
    # restored substitution around the one read of `key`
    target, orig = Path(p[key]).resolve(), pathlib.Path.read_bytes
    swapped = []

    def read_bytes(self):
        if Path(self).resolve() != target:
            return orig(self)
        with target.open("rb") as fh:
            before = fh.read()
        with target.open("wb") as fh:
            fh.write(before + b"\0")
        try:
            swapped.append(orig(self))
            return swapped[-1]
        finally:
            with target.open("wb") as fh:
                fh.write(before)
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    seen = {}
    with pytest.raises(ProvenanceError, match=key):
        load_group_ids(**p, consumed_sha256=seen)
    assert len(swapped) == 1 and seen[key] == hashlib.sha256(swapped[0]).hexdigest() != sha256_file(p[key])


# ----------------------------------------------------------------------------
# Real-data tests
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def groups() -> pd.DataFrame:
    pytest.importorskip("openpyxl")
    return load_group_ids()


@pytest.fixture(scope="module")
def canonical() -> pd.DataFrame:
    return pd.read_csv(PATHS["canonical_csv"])


@pytest.fixture(scope="module")
def source():
    """Raw CSV and metadata read directly, with validated keys."""
    pytest.importorskip("openpyxl")
    raw = pd.read_csv(PATHS["raw_csv"], low_memory=False)
    meta = pd.read_excel(PATHS["raw_metadata_xlsx"])
    assert raw["record_id"].is_unique
    assert meta["building_id"].is_unique
    return raw, meta


@pytest.fixture(scope="module")
def oracle(source) -> pd.DataFrame:
    """Independent reconstruction of the canonical rows (no data_preparation)."""
    raw, meta = source
    d = raw.merge(meta, on="building_id", how="left", validate="many_to_one")
    assert len(d) == len(raw)
    d = d.rename(columns=ORACLE_RENAME)
    d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce").dt.strftime("%Y-%m-%d")
    d["vote_raw"] = pd.to_numeric(d["thermal_sensation"], errors="coerce")
    d["thermal_sensation"] = np.floor(d["vote_raw"] + 0.5)
    d["thermal_preference"] = d["thermal_preference"].map({"cooler": -1, "no change": 0, "warmer": 1})
    d = d.dropna(subset=ORACLE_REQUIRED).reset_index(drop=True)
    d["TSV_3p"] = np.sign(d["thermal_sensation"])
    return d


def _check_against_source(groups: pd.DataFrame, raw: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Every returned value must equal its source record, looked up by record_id."""
    sel = raw.set_index("record_id", verify_integrity=True).loc[groups["record_id"].to_numpy()]
    np.testing.assert_array_equal(
        groups["thermal_sensation_raw"].to_numpy(dtype=float),
        pd.to_numeric(sel["thermal_sensation"], errors="coerce").to_numpy(dtype=float),
        err_msg="raw vote differs from its source record",
    )
    np.testing.assert_array_equal(
        groups["building_id"].to_numpy(), sel["building_id"].to_numpy(),
        err_msg="building_id differs from its source record",
    )
    src_subject = pd.to_numeric(sel["subject_id"], errors="coerce").to_numpy(dtype=float)
    out_subject = groups["subject_id"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    np.testing.assert_array_equal(out_subject, src_subject, err_msg="subject_id differs from its source record")
    pub = meta.set_index("building_id", verify_integrity=True)["publication"]
    np.testing.assert_array_equal(
        groups["publication"].to_numpy(dtype=object),
        pub.loc[groups["building_id"].to_numpy()].to_numpy(dtype=object),
        err_msg="publication differs from the metadata of its building",
    )


@needs_data
def test_input_files_have_pinned_digests():
    for key, digest in PINNED_SHA256.items():
        assert sha256_file(PATHS[key]) == digest, key


@needs_data
def test_row_count_columns_and_provenance(groups):
    assert len(groups) == EXPECTED_ROWS
    assert tuple(groups.columns) == OUTPUT_COLUMNS
    assert groups.index.equals(pd.RangeIndex(EXPECTED_ROWS))
    assert groups.attrs["verified_provenance"] is True
    assert groups.attrs["sha256"] == PINNED_SHA256
    assert record_sequence_sha256(groups["record_id"]) == PINNED_RECORD_SEQUENCE_SHA256


@needs_data
def test_independent_oracle_reproduces_canonical_csv(oracle, canonical):
    assert len(oracle) == EXPECTED_ROWS == len(canonical)
    for col in SOURCE_COLUMNS:
        a, b = oracle[col], canonical[col]
        if pd.api.types.is_numeric_dtype(b):
            np.testing.assert_array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), err_msg=col)
        else:
            np.testing.assert_array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy(), err_msg=col)
    # The pinned record sequence is the oracle's, not only the replay's.
    assert record_sequence_sha256(oracle["record_id"]) == PINNED_RECORD_SEQUENCE_SHA256


@needs_data
def test_loader_matches_independent_oracle_row_by_row(groups, oracle):
    np.testing.assert_array_equal(groups["record_id"].to_numpy(), oracle["record_id"].to_numpy())
    np.testing.assert_array_equal(
        groups["thermal_sensation_raw"].to_numpy(dtype=float), oracle["vote_raw"].to_numpy(dtype=float)
    )
    np.testing.assert_array_equal(groups["building_id"].to_numpy(), oracle["building_id"].to_numpy())


@needs_data
def test_ids_and_raw_votes_match_source_records(groups, source):
    raw, meta = source
    assert groups["record_id"].is_unique
    _check_against_source(groups, raw, meta)


@needs_data
@pytest.mark.parametrize(
    "column, rows",
    [
        ("thermal_sensation_raw", [1109, 1110]),  # -2.5 vs -2.0, same rounded labels
        ("building_id", [11448, 11723]),          # buildings 181 vs 182
        ("subject_id", [1608, 1611]),             # rows indistinguishable on all 19 columns
    ],
)
def test_source_check_detects_misassigned_values(groups, source, column, rows):
    raw, meta = source
    mutated = groups.copy()
    mutated.loc[rows, column] = groups.loc[rows[::-1], column].to_numpy()
    assert not mutated[column].equals(groups[column])
    with pytest.raises(AssertionError):
        _check_against_source(mutated, raw, meta)


@needs_data
@pytest.mark.parametrize("rows", [(11448, 11723), (1109, 1110)])
def test_value_guard_rejects_counterexample_swaps(canonical, rows):
    # Swaps that passed the earlier Tair/lat/lon/Date/label guard.
    i, j = rows
    replayed = canonical.copy()
    swapped = canonical.copy()
    swapped.iloc[[i, j]] = canonical.iloc[[j, i]].to_numpy()
    with pytest.raises(GroupAlignmentError, match=f"Values differ at row {i}"):
        assert_aligned(replayed, swapped)


@needs_data
def test_reordered_canonical_file_fails_provenance(canonical, tmp_path):
    # Byte-preserving reorder: the raw text lines of the canonical CSV are
    # copied verbatim and two data lines are exchanged, so the file differs
    # from the pinned one only by row order (no pandas re-serialisation).
    # The two rows must be distinguishable: exchanging identical records
    # (e.g. rows 1608 and 1611, equal on all 19 compared columns and
    # byte-identical as CSV lines) yields the same bytes and the same digest,
    # i.e. it is unobservable from the file and not a reorder at all.
    i, j = 11448, 11723  # buildings 181 vs 182
    text = PATHS["canonical_csv"].read_bytes()
    lines = text.splitlines(keepends=True)
    assert len(lines) == 1 + len(canonical)  # header + one line per row, no embedded newlines
    assert b"".join(lines) == text
    assert lines[1 + i] != lines[1 + j]
    assert not canonical.iloc[i].equals(canonical.iloc[j])
    permuted = list(lines)
    permuted[1 + i], permuted[1 + j] = lines[1 + j], lines[1 + i]
    assert sorted(permuted) == sorted(lines)  # same multiset of lines, only order changed
    path = tmp_path / "ASHRAE_2022_Clean_api_reordered.csv"
    path.write_bytes(b"".join(permuted))
    reread = pd.read_csv(path)
    pd.testing.assert_frame_equal(
        reread, canonical.iloc[[*range(i), j, *range(i + 1, j), i, *range(j + 1, len(canonical))]]
        .reset_index(drop=True)
    )
    assert sha256_file(PATHS["canonical_csv"]) == PINNED_SHA256["canonical_csv"]
    assert sha256_file(path) != PINNED_SHA256["canonical_csv"]
    with pytest.raises(ProvenanceError, match="canonical_csv"):
        load_group_ids(canonical_csv=path)


@needs_data
def test_group_counts(groups):
    assert groups["publication"].notna().all()
    assert groups["building_id"].notna().all()
    assert groups["publication"].nunique() == EXPECTED_PUBLICATIONS
    assert groups["building_id"].nunique() == EXPECTED_BUILDINGS
    assert int(groups["subject_id"].notna().sum()) == EXPECTED_SUBJECT_NON_MISSING


@needs_data
def test_each_building_belongs_to_one_publication(groups):
    assert (groups.groupby("building_id")["publication"].nunique() == 1).all()


@needs_data
def test_half_votes(groups):
    counts = groups["thermal_sensation_raw"].value_counts()
    observed = {k: int(counts.get(k, 0)) for k in EXPECTED_HALF_VOTES}
    assert observed == EXPECTED_HALF_VOTES
