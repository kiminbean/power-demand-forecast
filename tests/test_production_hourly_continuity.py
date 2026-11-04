"""Hourly continuity regressions for production training data."""

import numpy as np
import pandas as pd
import pytest

from src.training.train_production import load_data


@pytest.fixture
def hourly_data() -> pd.DataFrame:
    """Provide enough observations for lag and sequence creation."""
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=600, freq="h"),
        "power_demand": np.arange(600, dtype=float) + 1000,
        "auxiliary": 1.0,
    })


def test_missing_input_hour_is_rejected_before_lags(tmp_path, hourly_data):
    path = tmp_path / "missing.csv"
    hourly_data.drop(index=300).to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)hourly.*before.*lag"):
        load_data(str(path))


def test_dropna_internal_gap_is_rejected(tmp_path, hourly_data):
    hourly_data.loc[300, "power_demand"] = np.nan
    path = tmp_path / "missing_value.csv"
    hourly_data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)hourly.*after.*dropna"):
        load_data(str(path))


def test_continuous_data_keeps_correct_hourly_lags(tmp_path, hourly_data):
    path = tmp_path / "hourly.csv"
    hourly_data.sample(frac=1, random_state=42).to_csv(path, index=False)
    result, _ = load_data(str(path))
    assert len(result) > 0
    assert (result.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    original = hourly_data.set_index("datetime")["power_demand"]
    stamp = result.index[0]
    assert result.loc[stamp, "demand_lag_24"] == original.loc[stamp - pd.Timedelta(hours=24)]
