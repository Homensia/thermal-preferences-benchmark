# -*- coding: utf-8 -*-
"""
Legacy data writers never overwrite the frozen cohort files ("no existing file
is modified").

``data_preparation.process_dataset`` and
``feature_extraction_api.process_thermal_comfort_data`` refuse, before any
read, computation or Open-Meteo call, an output that already exists or is one
of the frozen data files; a new output path is written. The API is mocked;
temporary files only (the frozen files are hashed, never written).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config_loader  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def in_repo(monkeypatch):
    monkeypatch.chdir(REPO_DIR)          # the modules read config.yaml relative to the working directory


@pytest.fixture
def fea(in_repo):
    import feature_extraction_api
    return feature_extraction_api


@pytest.fixture
def dp(in_repo):
    import data_preparation
    return data_preparation


def _fail(*_a, **_k):
    raise AssertionError("called although the output is refused")


def _small_input(path: Path) -> Path:
    pd.DataFrame({"lat": [45.0, 45.0, 48.8], "lon": [5.0, 5.0, 2.3],
                  "Date": ["2020-07-01", "2020-07-01", "2020-07-03"], "Tair": [24.0, 25.0, 26.0]}).to_csv(path,
                                                                                                         index=False)
    return path


def _fake_weather(days):
    def fetch(client, lat, lon, date_iso, days_fetch=30):
        base = np.arange(days_fetch, dtype=float)
        return {"temperature": 20 + base / 10, "humidity": 50 + base, "sunshine": 3600 * np.ones(days_fetch),
                "precipitation": np.zeros(days_fetch), "wind": 2 + 0 * base}
    return fetch


# ------------------------------------------------------------------ the guard itself
def test_frozen_data_files_are_the_pinned_input_files():
    names = {p.name for p in config_loader.FROZEN_DATA_FILES}
    assert names == {"ASHRAE_09_06_2022.csv", "ASHRAE_09_06_2022_metadata.xlsx", "ASHRAE_2022_Clean.csv",
                     "ASHRAE_2022_Clean_api.csv", "Moujalled_api.csv", "Hostein_api.csv", "ASHRAE_2018_v2.csv"}
    assert all(p.parent == (REPO_DIR / "Data").resolve() for p in config_loader.FROZEN_DATA_FILES)


def test_check_new_data_output(tmp_path, monkeypatch):
    new = tmp_path / "sub" / "new.csv"
    assert config_loader.check_new_data_output(new) == new.resolve()
    existing = tmp_path / "old.csv"
    existing.write_text("a\n1\n")
    with pytest.raises(FileExistsError, match="existing file"):
        config_loader.check_new_data_output(existing)
    link = tmp_path / "link.csv"                     # a dangling symlink is not a new file either
    link.symlink_to(tmp_path / "nowhere.csv")
    with pytest.raises(FileExistsError):
        config_loader.check_new_data_output(link)
    monkeypatch.chdir(REPO_DIR)
    for frozen in config_loader.FROZEN_DATA_FILES:   # absolute, relative, and even when missing
        with pytest.raises(ValueError, match="frozen protocol"):
            config_loader.check_new_data_output(frozen)
        with pytest.raises(ValueError, match="frozen protocol"):
            config_loader.check_new_data_output(Path("Data") / frozen.name)
        with pytest.raises(ValueError, match="frozen protocol"):
            config_loader.check_new_data_output(Path("Data") / ".." / "Data" / frozen.name)
    with pytest.raises(ValueError, match="frozen artefact"):
        config_loader.check_new_data_output(config_loader.PROTECTED_ARTEFACT_DIR / "x.csv")


# ------------------------------------------------------------------ feature_extraction_api
def test_feature_extraction_refuses_an_existing_output_before_any_read_or_api_call(tmp_path, fea, monkeypatch):
    monkeypatch.setattr(fea, "setup_openmeteo_client", _fail)
    monkeypatch.setattr(fea, "fetch_weather_data", _fail)
    monkeypatch.setattr(fea.pd, "read_csv", _fail)
    out = tmp_path / "out_api.csv"
    out.write_text("keep,me\n1,2\n")
    before = _sha(out)
    with pytest.raises(FileExistsError):
        fea.process_thermal_comfort_data(str(tmp_path / "missing_input.csv"), str(out), [28])
    assert _sha(out) == before


@pytest.mark.parametrize("name", ["ASHRAE_2022_Clean_api.csv", "Moujalled_api.csv", "Hostein_api.csv"])
def test_feature_extraction_refuses_the_frozen_api_files(tmp_path, fea, monkeypatch, name):
    monkeypatch.setattr(fea, "setup_openmeteo_client", _fail)
    monkeypatch.setattr(fea, "fetch_weather_data", _fail)
    frozen = REPO_DIR / "Data" / name
    before = _sha(frozen) if frozen.exists() else None
    with pytest.raises(ValueError, match="frozen protocol"):
        fea.process_thermal_comfort_data(str(_small_input(tmp_path / "in.csv")), f"Data/{name}", [28])
    assert (_sha(frozen) if frozen.exists() else None) == before


def test_feature_extraction_writes_a_new_output(tmp_path, fea, monkeypatch):
    calls = []
    monkeypatch.setattr(fea, "setup_openmeteo_client", lambda cache_dir=".cache": "client")
    fetch = _fake_weather(30)
    monkeypatch.setattr(fea, "fetch_weather_data", lambda *a, **k: calls.append(a[1:4]) or fetch(*a, **k))
    out = tmp_path / "new" / "in_api.csv"
    fea.process_thermal_comfort_data(str(_small_input(tmp_path / "in.csv")), str(out), [28], days_fetch=30,
                                     cache_dir=str(tmp_path / "cache"))
    df = pd.read_csv(out)
    assert len(df) == 3 and len(calls) == 2                      # one API call per unique (lat, lon, date)
    assert {"Trm_ema_28", "RHout_ema_28", "precip_ema_28", "sunshine_ema_h_28", "wind_ema_28"} <= set(df)
    assert df["sunshine_ema_h_28"].eq(1.0).all() and df["Trm_ema_28"].notna().all()
    with pytest.raises(FileExistsError):                         # a second run never overwrites it
        fea.process_thermal_comfort_data(str(tmp_path / "in.csv"), str(out), [28])


def test_feature_extraction_cli_needs_an_explicit_new_output(tmp_path, fea, monkeypatch):
    monkeypatch.setattr(fea, "setup_openmeteo_client", _fail)
    assert not hasattr(fea, "DATASETS")                          # the batch over the frozen files is gone
    with pytest.raises(SystemExit):
        fea.parse_args(["--input", "Data/Moujalled.csv"])
    with pytest.raises(ValueError, match="frozen protocol"):
        fea.main(["--input", "Data/Moujalled.csv", "--output", "Data/Moujalled_api.csv"])
    existing = tmp_path / "x.csv"
    existing.write_text("a\n")
    with pytest.raises(FileExistsError):
        fea.main(["--input", "Data/Moujalled.csv", "--output", str(existing)])
    assert existing.read_text() == "a\n"


# ------------------------------------------------------------------ data_preparation
def _raw_like():
    return pd.DataFrame({"ta": [22.0, 23.0], "t_out": [10.0, 11.0], "rh": [40.0, 45.0], "gender": ["Male", "Female"],
                         "cooling_type": ["Mixed Mode"] * 2, "building_type": ["Office"] * 2,
                         "timestamp": ["2012-01-01T10:00:00Z", "2012-01-02T10:00:00Z"], "tr": [22.0, 23.0],
                         "lat": [45.0, 45.0], "lon": [5.0, 5.0], "year": [2012, 2012], "clo": [0.8, 0.9],
                         "vel": [0.1, 0.1], "Âge": [30, 40], "Met": [1.1, 1.2], "Season": ["Winter"] * 2,
                         "thermal_sensation": [-0.5, 1.0], "thermal_preference": ["warmer", "no change"],
                         "building_id": [1, 1]})


def test_data_preparation_refuses_an_existing_output_before_reading(tmp_path, dp, monkeypatch):
    monkeypatch.setattr(dp, "load_and_merge", _fail)
    monkeypatch.setattr(dp, "add_koppen_climate", _fail)
    out = tmp_path / "clean.csv"
    out.write_text("keep\n")
    with pytest.raises(FileExistsError):
        dp.process_dataset("raw.csv", "meta.xlsx", str(out))
    assert out.read_text() == "keep\n"
    frozen = REPO_DIR / dp.DEFAULT_CONFIG["output_clean_csv"]
    assert frozen.resolve() in config_loader.FROZEN_DATA_FILES      # config.yaml data.clean.ashrae is frozen
    before = _sha(frozen) if frozen.exists() else None
    with pytest.raises(ValueError, match="frozen protocol"):
        dp.process_dataset("raw.csv", "meta.xlsx", dp.DEFAULT_CONFIG["output_clean_csv"])
    assert (_sha(frozen) if frozen.exists() else None) == before


def test_data_preparation_writes_a_new_output(tmp_path, dp, monkeypatch):
    monkeypatch.setattr(dp, "load_and_merge", lambda raw_csv, meta: _raw_like())
    monkeypatch.setattr(dp, "add_koppen_climate", lambda df: df.assign(Climate="Cfb"))
    out = tmp_path / "new" / "clean.csv"
    dp.process_dataset("raw.csv", "meta.xlsx", str(out))
    df = pd.read_csv(out)
    assert list(df["thermal_sensation"]) == [0, 1] and list(df["TSV_3p"]) == [0, 1]
    assert list(df["thermal_preference"]) == [1, 0] and set(df["Climate"]) == {"Cfb"}
    with pytest.raises(FileExistsError):
        dp.process_dataset("raw.csv", "meta.xlsx", str(out))


def test_data_preparation_cli_needs_an_explicit_new_output(tmp_path, dp, monkeypatch):
    monkeypatch.setattr(dp, "load_and_merge", _fail)
    with pytest.raises(SystemExit):
        dp.parse_args([])
    with pytest.raises(ValueError, match="frozen protocol"):
        dp.main(["--output", "Data/ASHRAE_2022_Clean.csv"])
    with pytest.raises(ValueError, match="frozen protocol"):
        dp.main(["--output", "Data/ASHRAE_2022_Clean_api.csv"])
    assert dp.parse_args(["--output", str(tmp_path / "n.csv")]).raw_csv == dp.DEFAULT_CONFIG["raw_csv"]


# ---------------------------------------------------------------- heatmap sibling outputs
@pytest.fixture
def fi(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(REPO_DIR / "analysis"))
    return importlib.import_module("feature_importance_heatmap")


def _importance_dir(fi, root: Path) -> Path:
    for model in fi.MODELS:
        for target_key, _ in fi.TARGETS:
            d = root / target_key / model
            d.mkdir(parents=True)
            pd.DataFrame({"group": fi.FEATURE_ORDER,
                          "importance": np.linspace(0.01, 0.2, len(fi.FEATURE_ORDER))}).to_csv(
                d / f"{target_key}_{model}_feature_importance_aggregated.csv", index=False)
    return root


@pytest.mark.parametrize("name", ["ASHRAE_2022_Clean", "ASHRAE_2022_Clean_api", "Moujalled_api", "Hostein_api"])
def test_heatmap_refuses_a_frozen_csv_sibling_before_reading(fi, monkeypatch, name):
    """``--out Data/<frozen>.pdf`` derives the frozen ``Data/<frozen>.csv``: refused before any read or write."""
    frozen = REPO_DIR / "Data" / f"{name}.csv"
    before = _sha(frozen) if frozen.exists() else None
    monkeypatch.setattr(fi, "load_matrix", _fail)
    monkeypatch.setattr(fi, "make_heatmap", _fail)
    for out in (REPO_DIR / "Data" / f"{name}.pdf", Path("Data") / f"{name}.pdf"):
        monkeypatch.chdir(REPO_DIR)
        with pytest.raises(ValueError, match="frozen protocol"):
            fi.generate(Path("unused"), out)
    assert not (REPO_DIR / "Data" / f"{name}.pdf").exists() and not (REPO_DIR / "Data" / f"{name}.png").exists()
    assert (_sha(frozen) if frozen.exists() else None) == before


@pytest.mark.parametrize("existing", ["fig.pdf", "fig.png", "fig.csv"])
def test_heatmap_refuses_any_existing_destination(tmp_path, fi, monkeypatch, existing):
    results = _importance_dir(fi, tmp_path / "res")
    (tmp_path / existing).write_text("keep")
    monkeypatch.setattr(fi, "load_matrix", _fail)
    with pytest.raises(FileExistsError):
        fi.generate(results, tmp_path / "fig.pdf")
    assert (tmp_path / existing).read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["res", existing])


def test_heatmap_refuses_colliding_destinations(tmp_path, fi):
    for out in (tmp_path / "fig.png", tmp_path / "fig.csv"):
        with pytest.raises(ValueError, match="collide"):
            fi.generate(tmp_path / "absent", out)


def test_heatmap_writes_three_new_files(tmp_path, fi):
    results = _importance_dir(fi, tmp_path / "res")
    pdf, png, csv = fi.generate(results, tmp_path / "out" / "fig.pdf")
    assert pdf.read_bytes().startswith(b"%PDF") and png.read_bytes().startswith(b"\x89PNG")
    assert pd.read_csv(csv, index_col=0, header=[0, 1]).shape == (17, 6)
    with pytest.raises(FileExistsError):              # a second run never overwrites
        fi.generate(results, tmp_path / "out" / "fig.pdf")
