"""Regression coverage for selected production-feature missingness."""

import numpy as np
import pandas as pd
import pytest

from src.training.train_production import load_data


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=600, freq="h"),
        "power_demand": np.arange(600, dtype=float) + 1000,
        "기온": 15.0,
        "clfmAbbrCd": np.nan,
        "solar": np.nan,
    })


@pytest.mark.parametrize("include_weather", [False, True])
def test_unused_missing_columns_do_not_discard_observations(tmp_path, hourly_frame, include_weather):
    path = tmp_path / "data.csv"
    hourly_frame.to_csv(path, index=False)
    result, features = load_data(str(path), include_weather=include_weather)
    assert len(result) == 600 - 168
    assert not result[features].isna().any().any()
    assert (result.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize("column,include_weather", [("power_demand", False), ("기온", True)])
def test_missing_selected_value_cannot_create_silent_gap(tmp_path, hourly_frame, column, include_weather):
    hourly_frame.loc[300, column] = np.nan
    path = tmp_path / "data.csv"
    hourly_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)hourly.*after.*dropna"):
        load_data(str(path), include_weather=include_weather)


def test_selected_missing_value_at_end_is_removed(tmp_path, hourly_frame):
    hourly_frame.loc[599, "기온"] = np.nan
    path = tmp_path / "data.csv"
    hourly_frame.to_csv(path, index=False)
    result, features = load_data(str(path), include_weather=True)
    assert result.index[-1] == hourly_frame.loc[598, "datetime"]
    assert not result[features].isna().any().any()


@pytest.mark.parametrize("column,include_weather", [("power_demand", False), ("기온", True)])
def test_nonfinite_selected_training_input_is_rejected(tmp_path, hourly_frame, column, include_weather):
    hourly_frame.loc[300, column] = np.inf
    path = tmp_path / "nonfinite.csv"
    hourly_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_data(str(path), include_weather=include_weather)


@pytest.mark.parametrize("column", ["temp_mean", "dewpoint_mean"])
def test_nonfinite_weather_derivation_input_is_rejected(tmp_path, hourly_frame, column):
    hourly_frame["temp_mean"] = 10.0
    hourly_frame["dewpoint_mean"] = 5.0
    hourly_frame.loc[300, column] = np.inf
    path = tmp_path / "nonfinite_weather.csv"
    hourly_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_data(str(path), include_weather=True)


@pytest.mark.parametrize("column", ["THI", "HDD", "CDD"])
def test_overwritten_weather_input_does_not_block_training(tmp_path, hourly_frame, column):
    hourly_frame["temp_mean"] = 10.0
    hourly_frame["dewpoint_mean"] = 5.0
    hourly_frame[column] = np.inf
    path = tmp_path / "overwritten.csv"
    hourly_frame.to_csv(path, index=False)
    result, features = load_data(str(path), include_weather=True)
    assert column in features
    assert np.isfinite(result[column]).all()


def test_unrecomputed_weather_input_still_rejects_infinity(tmp_path, hourly_frame):
    hourly_frame["THI"] = np.inf
    path = tmp_path / "retained.csv"
    hourly_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_data(str(path), include_weather=True)
