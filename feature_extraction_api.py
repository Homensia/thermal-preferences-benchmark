"""
Fetch and compute meteorological data from the Open-Meteo API.

This module enriches thermal comfort datasets (ASHRAE, Moujalled, Hostein)
with long-term meteorological features derived from historical daily weather data.

For every unique (latitude, longitude, date) triplet, the script:

1. Fetches daily historical weather data from the Open-Meteo archive API
   (temperature, relative humidity, sunshine duration, wind speed, precipitation).

2. Computes Exponential Moving Averages (EMA) over a configurable number of days
   (default: 28 days) for:
        - Trm_ema       (°C)       → running mean dry-bulb temperature
        - RHout_ema     (%)        → mean relative humidity
        - precip_ema    (mm/day)   → precipitation sum
        - sunshine_ema_h(h/day)    → sunshine duration (converted from seconds)
        - wind_ema      (m/s)      → 10 m mean wind speed

3. Stores the enriched dataset `<name>_api.csv`.

This file is part of the thermal comfort prediction pipeline.
"""

import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from config_loader import load_config




CONFIG = load_config()["weather"]


# ====== CONFIGURATION ======
DEFAULT_CONFIG = {
    'windows': CONFIG["ema_windows"],        # EMA windows (days)
    'days_fetch': CONFIG["days_fetch"],       # how many past days to fetch from Open-Meteo
    'cache_dir': CONFIG["cache_dir"]   # folder where API cache is stored
}


# ====== EMA FUNCTION ======
def ema(values: np.ndarray, span: int) -> float:
    """
    Compute the Exponential Moving Average (EMA) of a vector.

    The EMA is computed using the recursive formula:

    .. math::
        EMA_t = \\alpha x_t + (1 - \\alpha) EMA_{t-1}

    with :math:`\\alpha = 2 / (span + 1)`.

    Parameters
    ----------
    values : numpy.ndarray
        Input array of daily weather values.
    span : int
        Window size used to compute the EMA.

    Returns
    -------
    float
        The computed EMA value, or ``np.nan`` if input is empty.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    ema_val = v[0]
    for x in v[1:]:
        ema_val = alpha * x + (1 - alpha) * ema_val
    return float(ema_val)


def get_ema_columns(window: int) -> Dict[str, float]:
    """
    Generate default EMA column names for a given window.

    Parameters
    ----------
    window : int
        The EMA window (e.g., 28).

    Returns
    -------
    dict
        Mapping of column names → default ``np.nan`` values.
    """
    return {
        f"Trm_ema_{window}": np.nan,
        f"RHout_ema_{window}": np.nan,
        f"precip_ema_{window}": np.nan,
        f"sunshine_ema_h_{window}": np.nan,
        f"wind_ema_{window}": np.nan,
    }


def ensure_columns(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Ensure that the dataframe contains all EMA feature columns.

    Missing columns are created automatically and filled with NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    windows : list of int
        EMA windows for which columns must exist.

    Returns
    -------
    pandas.DataFrame
        Updated dataframe containing all required columns.
    """
    for window in windows:
        for col, val in get_ema_columns(window).items():
            if col not in df.columns:
                df[col] = val
    return df


# ======================================================================================
# OPEN-METEO CLIENT INITIALIZATION
# ======================================================================================
def setup_openmeteo_client(cache_dir: str = '.cache'):
    """
    Create an Open-Meteo client with caching and automatic retry.

    Parameters
    ----------
    cache_dir : str, optional
        Path to the directory storing cached API responses.

    Returns
    -------
    openmeteo_requests.Client
        A configured Open-Meteo API client.
    """
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ======================================================================================
# WEATHER DATA FETCHING
# ======================================================================================
def fetch_weather_data(openmeteo_client, lat: float, lon: float, date_iso: str, days_fetch: int = 30) -> Optional[Dict[str, np.ndarray]]:
    """
    Fetch historical daily weather data from the Open-Meteo archive API.

    The function retrieves all daily variables needed for EMA calculations:
    temperature, humidity, sunshine, precipitation, and wind speed.

    Parameters
    ----------
    openmeteo_client : openmeteo_requests.Client
        The API client created with :func:`setup_openmeteo_client`.
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    date_iso : str
        ISO-formatted date (``YYYY-MM-DD``) for which history must be fetched.
    days_fetch : int, optional
        Number of past days to retrieve (default: 30).

    Returns
    -------
    dict or None
        Dictionary containing numpy arrays for each weather variable,
        or ``None`` if the API request failed.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    daily_vars = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "sunshine_duration",
        "precipitation_sum",
        "wind_speed_10m_mean"
    ]

    try:
        target_date = pd.to_datetime(date_iso)
        start_date = (target_date - pd.Timedelta(days=days_fetch - 1)).strftime("%Y-%m-%d")
        end_date = target_date.strftime("%Y-%m-%d")

        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": start_date,
            "end_date": end_date,
            "daily": daily_vars,
        }

        response = openmeteo_client.weather_api(url, params=params)[0]
        daily = response.Daily()

        weather_data = {
            'temperature': daily.Variables(0).ValuesAsNumpy(),
            'humidity': daily.Variables(1).ValuesAsNumpy(),
            'sunshine': daily.Variables(2).ValuesAsNumpy(),
            'precipitation': daily.Variables(3).ValuesAsNumpy(),
            'wind': daily.Variables(4).ValuesAsNumpy(),
        }

        lengths = [len(v) for v in weather_data.values()]
        if not all(length == lengths[0] for length in lengths):
            logger.warning(f"Inconsistent data lengths for {lat}, {lon}, {date_iso}")
            return None

        return weather_data

    except Exception as e:
        logger.warning(f"Failed to fetch data for {lat}, {lon}, {date_iso}: {str(e)}")
        return None


# ======================================================================================
# EMA FEATURE COMPUTATION
# ======================================================================================
def compute_ema_features(weather_data: Dict[str, np.ndarray], windows: List[int]) -> Dict[str, float]:
    """
    Compute EMA-based meteorological features for a given weather history.

    Parameters
    ----------
    weather_data : dict
        Dictionary mapping weather variables → numpy arrays.
        Must contain keys: ``temperature``, ``humidity``, ``sunshine``,
        ``precipitation``, ``wind``.
    windows : list of int
        EMA window sizes.

    Returns
    -------
    dict
        A dictionary containing computed EMA features for each requested window.
    """
    features = {}

    for window in windows:
        if len(weather_data['temperature']) < window + 1:
            logger.warning(f"Not enough data for window {window}")
            continue

        slice_obj = slice(-(window + 1), -1)
        temp_window = weather_data['temperature'][slice_obj]
        humidity_window = weather_data['humidity'][slice_obj]
        sunshine_window = weather_data['sunshine'][slice_obj]
        precip_window = weather_data['precipitation'][slice_obj]
        wind_window = weather_data['wind'][slice_obj]

        features.update({
            f"Trm_ema_{window}": ema(temp_window, span=window),
            f"RHout_ema_{window}": ema(humidity_window, span=window),
            f"precip_ema_{window}": ema(precip_window, span=window),
            f"sunshine_ema_h_{window}": ema(sunshine_window, span=window) / 3600.0,
            f"wind_ema_{window}": ema(wind_window, span=window),
        })

    return features


# ======================================================================================
# MAIN PROCESSING PIPELINE
# ======================================================================================
def process_thermal_comfort_data(input_path: str, output_path: str, windows: List[int], days_fetch: int = 30, cache_dir: str = '.cache') -> None:
    """
    Enrich a thermal comfort dataset with EMA-based meteorological features.

    Steps
    -----
    1. Load input CSV
    2. Normalize date column
    3. Ensure output EMA columns exist
    4. For each unique (lat, lon, date):
         - Fetch historical weather data
         - Compute EMA features
    5. Inject computed features back into the dataset
    6. Save enriched CSV to disk

    Parameters
    ----------
    input_path : str
        Path to the input dataset.
    output_path : str
        Path where the enriched dataset will be saved.
    windows : list of int
        EMA window sizes (e.g., [28]).
    days_fetch : int, optional
        Number of past days to fetch from the API (default: 30).
    cache_dir : str, optional
        Directory where API responses are cached.

    Returns
    -------
    None
    """
    logger.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path)
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    logger.info(f"Ensuring EMA columns for windows: {windows}")
    data = ensure_columns(data, windows)

    logger.info("Setting up Open-Meteo client")
    openmeteo_client = setup_openmeteo_client(cache_dir)

    unique_keys = data[["lat", "lon", "Date"]].drop_duplicates()
    logger.info(f"Processing {len(unique_keys)} unique location-date combinations")

    weather_cache = {}
    for idx, (_, row) in enumerate(unique_keys.iterrows(), 1):
        lat, lon, date_iso = row["lat"], row["lon"], row["Date"]

        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(unique_keys)}")

        weather_data = fetch_weather_data(openmeteo_client, lat, lon, date_iso, days_fetch)
        weather_cache[(lat, lon, date_iso)] = compute_ema_features(weather_data, windows) if weather_data else None

    logger.info("Filling dataframe with computed features")
    def fill_row(row):
        features = weather_cache.get((row["lat"], row["lon"], row["Date"]))
        if features:
            for col, value in features.items():
                row[col] = value
        return row

    data = data.apply(fill_row, axis=1)

    logger.info(f"Saving results to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"✅ Done: {output_path}")

# ======================================================================================
# BATCH PROCESSING FOR ALL DATASETS
# ======================================================================================
DATASETS = [
    "Data/ASHRAE_2022_Clean.csv",
    "Data/Moujalled.csv",
    "Data/Hostein.csv",
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=== Starting batch processing ===")

    for input_path in DATASETS:
        output_path = str(Path(input_path).with_name(Path(input_path).stem + "_api.csv"))
        logger.info(f"--> Processing: {Path(input_path).name}")

        try:
            process_thermal_comfort_data(
                input_path=input_path,
                output_path=output_path,
                windows=DEFAULT_CONFIG['windows'],
                days_fetch=DEFAULT_CONFIG['days_fetch'],
                cache_dir=DEFAULT_CONFIG['cache_dir']
            )
        except Exception as e:
            logger.error(f"❌ Failed on {input_path}: {e}", exc_info=True)

    logger.info("=== All datasets processed ===")
