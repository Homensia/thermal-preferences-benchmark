"""
Clean and normalize the ASHRAE 2022 dataset.

This module provides a complete preprocessing pipeline for the ASHRAE Global
Thermal Comfort Database II (2022). It merges the raw CSV with the metadata
Excel file, standardizes column names, encodes target variables, removes
incomplete rows, and computes the Köppen climate classification.

The final cleaned dataset is written to disk.

Main steps:
    - Load raw CSV and metadata Excel
    - Merge on 'building_id'
    - Rename columns to unified schema
    - Normalize date format (YYYY-MM-DD)
    - Encode thermal_sensation and thermal_preference
    - Create TSV_3p (7-point → 3 classes)
    - Drop incomplete samples
    - Add climate classification (kgcpy.lookupCZ)
    - Keep only a curated subset of columns
    - Save the cleaned dataset

"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from kgcpy import lookupCZ 
from config_loader import load_config





# ====== CONFIGURATION ======
CONFIG = load_config()
DEFAULT_CONFIG = {
    "raw_csv": CONFIG["data"]["raw"]["ashrae_csv"],
    "raw_metadata_xlsx":  CONFIG["data"]["raw"]["ashrae_meta"],
    "output_clean_csv": CONFIG["data"]["clean"]["ashrae"],
}

RENAME_MAP: Dict[str, str] = {
    "ta": "Tair",
    "t_out": "Tout",
    "rh": "RH",
    "gender": "Sexe",
    "cooling_type": "cooling type",
    "building_type": "Building_type",
    "timestamp": "Date",
}


REQUIRED_FOR_DROPNA: List[str] = [
    "Date", "tr", "Tair", "lat", "lon", "year", "RH", "clo", "vel", "Âge",
    "Tout", "Met", "Sexe", "Season", "Building_type",
    "cooling type", "thermal_sensation", "thermal_preference"
]
COLUMNS_TO_KEEP  : List[str] = [
    "Date", "tr", "Tair", "lat", "lon", "year", "RH", "clo", "vel", "Âge",
    "Tout", "Met", "Sexe","Climate" , "Season", "Building_type",
    "cooling type", "thermal_sensation", "thermal_preference","TSV_3p"
]


def setup_logger() -> logging.Logger:
    """
    Configure and return a logger for the module.

    Returns
    -------
    logging.Logger
        A configured logger instance using INFO verbosity.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def add_koppen_climate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Köppen climate class for each sample.

    Climate classification is computed using :func:`kgcpy.lookupCZ`,
    which takes geographical coordinates (latitude, longitude).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least ``lat`` and ``lon`` columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional column ``Climate``.
    """
    logger.info("Computing Köppen climate classification using kgcpy.lookupCZ")
    df["Climate"] = df.apply(lambda r: lookupCZ(r["lat"], r["lon"]), axis=1)
    return df



def load_and_merge(raw_csv: str, raw_metadata_xlsx: str) -> pd.DataFrame:
    """
    Load the raw ASHRAE CSV and metadata Excel file, then merge them.

    Merge is performed on the column ``building_id`` using a left join.

    Parameters
    ----------
    raw_csv : str
        Path to the raw main CSV file.
    raw_metadata_xlsx : str
        Path to the metadata Excel file.

    Returns
    -------
    pandas.DataFrame
        Merged dataframe.
    """
    logger.info(f"Lecture CSV principal: {raw_csv}")
    df1 = pd.read_csv(raw_csv, low_memory=False)

    logger.info(f"Lecture Excel metadata: {raw_metadata_xlsx}")
    df2 = pd.read_excel(raw_metadata_xlsx)

    logger.info("Fusion sur 'building_id' (left join)")
    df_merged = pd.merge(df1, df2, on="building_id", how="left")
    logger.info(f"Après merge: {df_merged.shape[0]} lignes, {df_merged.shape[1]} colonnes")
    return df_merged


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and rename columns to a unified naming scheme.

    This includes:
      - renaming columns using :data:`RENAME_MAP`
      - converting timestamp → ``YYYY-MM-DD`` format

    Parameters
    ----------
    df : pandas.DataFrame
        Raw merged dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with renamed and normalized columns.
    """
    logger.info("Renommage des colonnes")
    df = df.rename(columns=RENAME_MAP)

    # Date → YYYY-MM-DD
    if "Date" in df.columns:
        logger.info("Normalisation de la Date (YYYY-MM-DD)")
        dt = pd.to_datetime(df["Date"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
        df["Date"] = dt.dt.strftime("%Y-%m-%d")

    return df


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variables: thermal preference and thermal sensation.

    Steps:
    -------
    1. ``thermal_preference``:
         - Map {``cooler``: -1, ``no change``: 0, ``warmer``: 1}
         - Convert to Int64
    2. ``thermal_sensation``:
         - Convert to float
         - Round using ``floor(x + 0.5)`` (7-point integer scale)
    3. ``TSV_3p``:
         - Derived from ``thermal_sensation`` (7 → 3 classes: -1, 0, 1)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Updated dataframe with encoded targets.
    """
    logger.info("Encodage des cibles thermal_sensation / thermal_preference")

    df["thermal_preference"] = (
        pd.to_numeric(
            df["thermal_preference"].replace({"no change": 0, "cooler": -1, "warmer": 1}),
            errors="coerce"
        )
        .astype("Int64")
    )

    # thermal_sensation → arrondi à l'entier (7 points) via floor(x+0.5)
    if "thermal_sensation" in df.columns:
        df["thermal_sensation"] = (
            np.floor(pd.to_numeric(df["thermal_sensation"], errors="coerce") + 0.5)
            .astype("Int64")
        )

    # TSV_3p dérivé de thermal_sensation (7→3)
    if "thermal_sensation" in df.columns:
        map_7_to_3 = {-3: -1, -2: -1, -1: -1, 0: 0, 1: 1, 2: 1, 3: 1}
        df["TSV_3p"] = df["thermal_sensation"].map(map_7_to_3).astype("Int64")

    return df

def drop_unused_columns(df: pd.DataFrame, columns_to_keep: list) -> pd.DataFrame:
    """
    Remove all columns except a curated subset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    columns_to_keep : list of str
        List of columns that must appear in the final dataset.

    Returns
    -------
    pandas.DataFrame
        Dataframe reduced to the specified set of columns.
    """
    # Colonnes réellement présentes dans le DF et à garder
    valid_keep = [c for c in columns_to_keep if c in df.columns]

    # Colonnes à supprimer = tout le reste
    cols_to_drop = [c for c in df.columns if c not in valid_keep]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} unused columns: {cols_to_drop}")
    else:
        logger.info("No unused columns to drop.")

    return df



def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values on critical variables.

    Critical variables are defined in :data:`REQUIRED_FOR_DROPNA`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with incomplete rows removed.
    """
    logger.info("Drop des lignes incomplètes sur les colonnes critiques")
    required = [c for c in REQUIRED_FOR_DROPNA if c in df.columns]
    before = df.shape[0]
    df = df.dropna(subset=required)
    after = df.shape[0]
    logger.info(f"Lignes avant drop: {before} → après drop: {after} (supprimées: {before - after})")
    return df



def process_dataset(raw_csv: str, raw_metadata_xlsx: str, output_clean_csv: str) -> None:
    """
    Full preprocessing pipeline for the ASHRAE dataset.

    Steps:
    -------
    - Load raw CSV and metadata
    - Merge on building_id
    - Standardize column names
    - Encode targets
    - Drop incomplete rows
    - Compute Köppen climate classification
    - Keep only relevant columns
    - Save the cleaned dataset

    Parameters
    ----------
    raw_csv : str
        Path to the raw ASHRAE CSV file.
    raw_metadata_xlsx : str
        Path to the metadata Excel file.
    output_clean_csv : str
        Path where the processed dataset should be saved.

    Returns
    -------
    None
    """
    df = load_and_merge(raw_csv, raw_metadata_xlsx)
    df = normalize_columns(df)
    df = encode_targets(df)
    df = drop_incomplete_rows(df)
    df = add_koppen_climate(df)
    df = drop_unused_columns(df, COLUMNS_TO_KEEP)
    

    logger.info(f"Sauvegarde: {output_clean_csv}")
    Path(output_clean_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_clean_csv, index=False)
    logger.info("✅ Nettoyage terminé.")


if __name__ == "__main__":
    process_dataset(
        raw_csv=DEFAULT_CONFIG["raw_csv"],
        raw_metadata_xlsx=DEFAULT_CONFIG["raw_metadata_xlsx"],
        output_clean_csv=DEFAULT_CONFIG["output_clean_csv"],
    )
