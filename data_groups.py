"""
Group identifiers for the canonical ASHRAE-2022 cohort (15 224 rows).

The cleaned CSV used by the paper (``Data/ASHRAE_2022_Clean_api.csv``, the
``ASHRAE_2022`` entry of the ``data.datasets`` registry in ``config.yaml``)
does not keep the study / building / subject identifiers of the raw ASHRAE
Global Thermal Comfort Database II. Grouped analyses (leave-one-publication-out,
bootstrap grouped by building, subject-level checks) and the M20 rounding
sensitivity (raw, unrounded thermal sensation votes) need them.

This module rebuilds them by replaying the row-filtering steps of the cleaning
pipeline in :mod:`data_preparation` (``load_and_merge`` ->
``normalize_columns`` -> ``encode_targets`` -> ``drop_incomplete_rows``) on the
raw CSV and the metadata workbook. The remaining steps of
:func:`data_preparation.process_dataset` (Köppen lookup, column selection) and
the weather enrichment of :mod:`feature_extraction_api` never add, drop or
reorder rows.

What the row alignment rests on
-------------------------------
Row identity is NOT inferred from the cleaned values. Many canonical rows share
identical values (on all 19 source-derived columns, eight pairs of rows are
indistinguishable, e.g. canonical rows 1608 and 1611 = records 3444 and 3447,
with different subjects and raw votes). A reordered cleaned file therefore
cannot be realigned from its values alone. The alignment contract is:

1. **Pinned provenance.** By default the raw CSV, the metadata workbook and
   the canonical CSV must have the SHA-256 digests in :data:`PINNED_SHA256`,
   and the replayed ``record_id`` sequence must have the digest
   :data:`PINNED_RECORD_SEQUENCE_SHA256`. Any other file (including a
   reordered copy of the canonical CSV) raises :class:`ProvenanceError`.
2. **Deterministic source order.** The replay only filters rows of the raw
   CSV; it never sorts or reorders them, exactly as the canonical pipeline.
3. **Merge cardinality.** ``record_id`` is unique in the raw CSV,
   ``building_id`` is unique in the metadata workbook (so the left join is
   many-to-one and keeps the raw row count), and every retained row has a
   publication.
4. **Consistency guard.** :func:`assert_aligned` compares all 19 source-derived
   columns of the canonical CSV with the replay, row by row. This catches a
   wrong file or a broken replay; it is a necessary check, not a proof of
   identity (see the duplicate rows above).

Nothing is written to disk. The canonical pipeline functions are only called,
never modified.

Example
-------
>>> from data_groups import load_group_ids
>>> groups = load_group_ids()           # paths taken from config.yaml
>>> groups.publication.nunique()        # 10
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from config_loader import read_consumed

REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "config.yaml"

#: Dataset tag of the canonical Phase 2 / Phase 3 cohort in ``config.yaml``.
CANONICAL_DATASET = "ASHRAE_2022"

#: Group identifier columns returned by :func:`load_group_ids`.
GROUP_COLUMNS = ("publication", "building_id", "subject_id")

#: Raw (unrounded) thermal sensation vote, for the M20 rounding sensitivity.
RAW_TSV_COLUMN = "thermal_sensation_raw"

#: All columns returned by :func:`load_group_ids`, in order.
OUTPUT_COLUMNS = ("record_id",) + GROUP_COLUMNS + (RAW_TSV_COLUMN,)

#: SHA-256 of the input files the canonical cohort was built from
#: (keys match :func:`default_paths`). Checked when ``verify_provenance=True``.
PINNED_SHA256: Dict[str, str] = {
    "raw_csv": "1bb6b1d84434f0b0604d61ede8304f4b586f5ab649c6fe1b991b0effd6b227f3",
    "raw_metadata_xlsx": "d16324e2f26d1a8b0e567914acd016a79c4d4433b4f05d9593111e42ffe0eb32",
    "canonical_csv": "ce71980083ce5b2c9e8374d070a5daef25e7805153bfc569691c5cb20f184bc9",
}

#: Number of rows of the canonical cohort.
PINNED_ROW_COUNT = 15_224

#: SHA-256 of the retained ``record_id`` sequence, as little-endian int64
#: bytes (see :func:`record_sequence_sha256`). It pins the identity AND order
#: of the 15 224 raw records behind the canonical rows. Obtained identically
#: from this replay and from an independent pandas reconstruction that does
#: not use :mod:`data_preparation` (see ``tests/test_group_ids.py``).
PINNED_RECORD_SEQUENCE_SHA256 = "283665bb4b8bb89c4b193a75b249552275919cf4689283bce56714951dd73246"

# Source-derived columns of the canonical CSV, compared row by row.
# Every column the canonical CSV shares with the replay is listed here
# (the other canonical columns, Climate and the *_ema_28 weather features,
# are computed after the row filter and are not compared).
#: Real-valued columns, compared with ``|a - b| <= FLOAT_ATOL`` (NaN == NaN).
FLOAT_COLUMNS = ("Tair", "tr", "RH", "vel", "Met", "clo", "Tout", "lat", "lon", "Âge")
#: Integer-valued columns (labels and year), compared exactly.
EXACT_NUMERIC_COLUMNS = ("thermal_sensation", "TSV_3p", "thermal_preference", "year")
#: Text columns, compared exactly as strings.
STRING_COLUMNS = ("Date", "Season", "Sexe", "Building_type", "cooling type")
SOURCE_COLUMNS = FLOAT_COLUMNS + EXACT_NUMERIC_COLUMNS + STRING_COLUMNS

#: Absolute tolerance for :data:`FLOAT_COLUMNS` (no relative tolerance).
#: The canonical values match the replay exactly; the tolerance only absorbs
#: float round-trip noise from re-saving a CSV (the sister file
#: ``ASHRAE_2022_Clean.csv`` differs from the API file by <= 2.2e-47 on ``vel``).
FLOAT_ATOL = 1e-9

PathLike = Union[str, os.PathLike]


class GroupAlignmentError(AssertionError):
    """Raised when the replayed raw rows do not match the canonical CSV."""


class ProvenanceError(GroupAlignmentError):
    """Raised when an input file or the record sequence is not the pinned one."""


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _preparation_module():
    """Import :mod:`data_preparation`.

    The module loads ``config.yaml`` with a path relative to the current
    working directory at import time, so the first import is done from the
    repository root.
    """
    with _working_directory(REPO_DIR):
        return importlib.import_module("data_preparation")


def _resolve(path: PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def default_paths(config: Optional[dict] = None) -> Dict[str, Path]:
    """Return the raw CSV, metadata workbook and canonical CSV from config.yaml.

    ``config``: the configuration of the calling run, already parsed from the
    bytes it records; ``None``: read
    ``config.yaml`` now (standalone use).
    """
    if config is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    data = config["data"]
    return {
        "raw_csv": _resolve(data["raw"]["ashrae_csv"]),
        "raw_metadata_xlsx": _resolve(data["raw"]["ashrae_meta"]),
        "canonical_csv": _resolve(data["datasets"][CANONICAL_DATASET]["csv"]),
    }


def sha256_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def record_sequence_sha256(record_ids) -> str:
    """Return the SHA-256 of a ``record_id`` sequence (little-endian int64 bytes)."""
    arr = np.asarray(record_ids, dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_input_provenance(paths: Dict[str, PathLike],
                            observed: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Check the input files against :data:`PINNED_SHA256`.

    ``observed`` (key -> SHA-256) gives the digests of the bytes actually
    consumed; default: hash the files now.
    Returns the observed digests. Raises :class:`ProvenanceError` listing every
    file whose digest differs.
    """
    observed = dict(observed) if observed is not None else {key: sha256_file(paths[key]) for key in PINNED_SHA256}
    bad = [
        f"{key} ({paths[key]}): sha256 {observed[key]} != pinned {PINNED_SHA256[key]}"
        for key in PINNED_SHA256
        if observed[key] != PINNED_SHA256[key]
    ]
    if bad:
        raise ProvenanceError("Input files are not the pinned canonical inputs:\n  " + "\n  ".join(bad))
    return observed


def _source(src):
    """A path as a string, or a fresh in-memory reader over consumed bytes."""
    return io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)


def check_merge_inputs(raw_csv, raw_metadata_xlsx) -> None:
    """Check the key cardinality the replayed left join relies on.

    ``record_id`` must be present and unique in the raw CSV, and
    ``building_id`` present and unique in the metadata workbook, so that
    ``load_and_merge`` (a left join on ``building_id``) is many-to-one and
    keeps exactly one merged row per raw record, in raw order.

    Raises
    ------
    GroupAlignmentError
        On a missing or duplicated key.
    """
    raw = pd.read_csv(_source(raw_csv), usecols=["record_id", "building_id"], low_memory=False)
    meta = pd.read_excel(_source(raw_metadata_xlsx), usecols=["building_id"])
    problems: List[str] = []
    if raw["record_id"].isna().any():
        problems.append(f"{int(raw['record_id'].isna().sum())} raw rows have no record_id")
    if raw["record_id"].duplicated().any():
        problems.append(f"{int(raw['record_id'].duplicated().sum())} duplicated record_id in the raw CSV")
    if meta["building_id"].isna().any():
        problems.append(f"{int(meta['building_id'].isna().sum())} metadata rows have no building_id")
    if meta["building_id"].duplicated().any():
        dup = meta.loc[meta["building_id"].duplicated(), "building_id"].iloc[0]
        problems.append(
            f"{int(meta['building_id'].duplicated().sum())} duplicated building_id in the "
            f"metadata (e.g. {dup!r}): the left join would duplicate raw records"
        )
    if problems:
        raise GroupAlignmentError("Merge cardinality check failed: " + "; ".join(problems))


def replay_cleaning_rows(raw_csv, raw_metadata_xlsx) -> pd.DataFrame:
    """Replay the row-level steps of the cleaning pipeline on the raw data.

    Uses the functions of :mod:`data_preparation` in the order of
    :func:`data_preparation.process_dataset`, stopping before the steps that
    do not affect rows. Before targets are encoded, the raw thermal sensation
    vote is copied to :data:`RAW_TSV_COLUMN` (as float, same numeric coercion
    as ``encode_targets``). No step sorts or reorders rows.

    Returns
    -------
    pandas.DataFrame
        All merged columns for the retained rows, in raw-file order, with a
        fresh RangeIndex.
    """
    prep = _preparation_module()
    df = prep.load_and_merge(_source(raw_csv), _source(raw_metadata_xlsx))
    df = prep.normalize_columns(df)
    df[RAW_TSV_COLUMN] = pd.to_numeric(df["thermal_sensation"], errors="coerce")
    df = prep.encode_targets(df)
    df = prep.drop_incomplete_rows(df)
    return df.reset_index(drop=True)


def assert_aligned(replayed: pd.DataFrame, canonical: pd.DataFrame) -> None:
    """Consistency guard: compare every source-derived column row by row.

    Compares the columns in :data:`SOURCE_COLUMNS` (all 19 columns the
    canonical CSV shares with the replay): :data:`FLOAT_COLUMNS` within
    :data:`FLOAT_ATOL` (absolute), :data:`EXACT_NUMERIC_COLUMNS` and
    :data:`STRING_COLUMNS` exactly.

    This detects a wrong file or a broken replay. It cannot prove row
    identity: rows with identical values on all compared columns exist, and
    swapping them passes this check. Identity is enforced by the provenance
    checks of :func:`load_group_ids`.

    Raises
    ------
    GroupAlignmentError
        On a missing column, a row-count mismatch, or a value mismatch. The
        message gives the earliest mismatching row over all columns, the
        columns that differ at that row, and the mismatch count per column.
    """
    missing = {
        name: [c for c in SOURCE_COLUMNS if c not in frame.columns]
        for name, frame in (("replayed", replayed), ("canonical", canonical))
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise GroupAlignmentError(f"Missing columns for the alignment check: {missing}")

    if len(replayed) != len(canonical):
        raise GroupAlignmentError(
            f"Row count mismatch: replayed {len(replayed)} vs canonical {len(canonical)}"
        )

    bad_masks: Dict[str, np.ndarray] = {}
    for col in FLOAT_COLUMNS + EXACT_NUMERIC_COLUMNS:
        a = pd.to_numeric(replayed[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(canonical[col], errors="coerce").to_numpy(dtype=float)
        if col in FLOAT_COLUMNS:
            ok = np.isclose(a, b, rtol=0.0, atol=FLOAT_ATOL, equal_nan=True)
        else:
            ok = (a == b) | (np.isnan(a) & np.isnan(b))
        if not ok.all():
            bad_masks[col] = ~ok
    for col in STRING_COLUMNS:
        ok = replayed[col].astype(str).to_numpy() == canonical[col].astype(str).to_numpy()
        if not ok.all():
            bad_masks[col] = ~ok

    if bad_masks:
        first = min(int(np.flatnonzero(m)[0]) for m in bad_masks.values())
        at_first = [c for c, m in bad_masks.items() if m[first]]
        details = ", ".join(
            f"{c}: replayed {replayed[c].iloc[first]!r} vs canonical {canonical[c].iloc[first]!r}"
            for c in at_first
        )
        counts = {c: int(m.sum()) for c, m in bad_masks.items()}
        raise GroupAlignmentError(
            f"Values differ at row {first} ({details}); mismatching rows per column: {counts}"
        )


def load_group_ids(
    raw_csv: Optional[PathLike] = None,
    raw_metadata_xlsx: Optional[PathLike] = None,
    canonical_csv: Optional[PathLike] = None,
    verify_provenance: bool = True,
    raw_bytes: Optional[Dict[str, bytes]] = None,
    consumed_sha256: Optional[Dict[str, str]] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Return per-row group identifiers aligned with the canonical CSV.

    Parameters
    ----------
    raw_csv, raw_metadata_xlsx, canonical_csv : path-like, optional
        Default to the paths declared in ``config.yaml``
        (``data.raw.ashrae_csv``, ``data.raw.ashrae_meta`` and
        ``data.datasets.ASHRAE_2022.csv``).
    verify_provenance : bool, default True
        Require the three files to match :data:`PINNED_SHA256` and the
        replayed record sequence to match :data:`PINNED_RECORD_SEQUENCE_SHA256`.
        Only set to False for synthetic test fixtures: without it, row
        identity rests on the value guard alone, which cannot detect swaps of
        rows with identical values.
    raw_bytes : {key: bytes}, optional
        Bytes of some of the three inputs already read once by the caller
        (e.g. ``{"canonical_csv": ...}``, the cohort the caller parsed).
    consumed_sha256 : dict, optional
        Receives ``key -> SHA-256`` of the bytes consumed for each of the three
        inputs. Each input is read once; the provenance pins are checked on
        those bytes and every frame is parsed from them in memory.
    config : dict, optional
        Configuration of the calling run, for the default paths (not read
        again from ``config.yaml``).

    Returns
    -------
    pandas.DataFrame
        One row per canonical row, same order, RangeIndex = canonical row
        position (the positions used by the serialized split indices).
        Columns: ``record_id`` (raw database record id), ``publication``,
        ``building_id``, ``subject_id`` (nullable Int64) and
        ``thermal_sensation_raw`` (unrounded vote).

        ``subject_id`` is missing for 3 of the 10 publications (3 470 rows)
        and is NOT a global key: the same number is reused across
        publications and, in some studies, across buildings. Use it only
        together with ``publication`` and ``building_id``; even then it is a
        source identifier, not a verified person identity.

    Raises
    ------
    ProvenanceError
        If ``verify_provenance`` and an input file or the record sequence is
        not the pinned one.
    GroupAlignmentError
        On a merge-cardinality violation, a duplicated retained record, a
        retained row without publication, a non-integer subject_id, or a
        failed :func:`assert_aligned` check.
    """
    defaults = default_paths(config)
    paths = {
        "raw_csv": _resolve(raw_csv) if raw_csv is not None else defaults["raw_csv"],
        "raw_metadata_xlsx": (
            _resolve(raw_metadata_xlsx) if raw_metadata_xlsx is not None
            else defaults["raw_metadata_xlsx"]
        ),
        "canonical_csv": (
            _resolve(canonical_csv) if canonical_csv is not None else defaults["canonical_csv"]
        ),
    }

    blobs = dict(raw_bytes or {})
    observed = {}
    for key, path in paths.items():                       # one read per input
        if key not in blobs:
            blobs[key], _ = read_consumed(path)
        observed[key] = hashlib.sha256(blobs[key]).hexdigest()
    if consumed_sha256 is not None:
        consumed_sha256.update(observed)
    digests = verify_input_provenance(paths, observed) if verify_provenance else {}
    check_merge_inputs(blobs["raw_csv"], blobs["raw_metadata_xlsx"])

    replayed = replay_cleaning_rows(blobs["raw_csv"], blobs["raw_metadata_xlsx"])
    if replayed["record_id"].duplicated().any():
        raise GroupAlignmentError("Retained rows contain duplicated record_id")
    if replayed["publication"].isna().any():
        raise GroupAlignmentError(
            f"{int(replayed['publication'].isna().sum())} retained rows have no publication"
        )

    canonical = pd.read_csv(io.BytesIO(blobs["canonical_csv"]))
    assert_aligned(replayed, canonical)

    if verify_provenance:
        if len(replayed) != PINNED_ROW_COUNT:
            raise ProvenanceError(f"Replayed {len(replayed)} rows, pinned {PINNED_ROW_COUNT}")
        seq = record_sequence_sha256(replayed["record_id"])
        if seq != PINNED_RECORD_SEQUENCE_SHA256:
            raise ProvenanceError(
                f"Replayed record_id sequence sha256 {seq} != pinned {PINNED_RECORD_SEQUENCE_SHA256}"
            )

    # subject_id is stored as text in the raw CSV (numeric strings or empty).
    # Convert to nullable Int64, refusing any value that is not an integer.
    subject_raw = replayed["subject_id"]
    subject_num = pd.to_numeric(subject_raw, errors="coerce")
    bad = subject_raw.notna() & (subject_num.isna() | (subject_num % 1 != 0))
    if bad.any():
        raise GroupAlignmentError(
            f"{int(bad.sum())} subject_id values are not integers, e.g. "
            f"{subject_raw[bad].iloc[0]!r}"
        )

    out = pd.DataFrame(
        {
            "record_id": replayed["record_id"].astype("int64"),
            "publication": replayed["publication"].astype("string"),
            "building_id": replayed["building_id"].astype("int64"),
            "subject_id": subject_num.astype("Int64"),
            RAW_TSV_COLUMN: replayed[RAW_TSV_COLUMN].astype("float64"),
        },
        columns=list(OUTPUT_COLUMNS),
    )
    out.index = pd.RangeIndex(len(out), name="row")
    for key, path in paths.items():
        out.attrs[key] = str(path)
    out.attrs["verified_provenance"] = bool(verify_provenance)
    out.attrs["sha256"] = digests
    return out
