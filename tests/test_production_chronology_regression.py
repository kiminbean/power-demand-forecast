"""Regression tests for chronological production-training ingestion."""

import numpy as np
import pandas as pd
import pytest

from src.training.train_production import TimeSeriesDataset, load_data, split_data_by_time


def _hourly_data() -> pd.DataFrame:
    """Create enough hours to populate the longest lag and a training sequence."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2022-12-10", periods=600, freq="h"),
            "power_demand": np.arange(600, dtype=float) + 1000,
            "기온": np.arange(600, dtype=float) / 100 + 10,
        }
    )


@pytest.mark.parametrize("include_weather", [False, True])
def test_shuffled_input_preserves_causal_features_and_sequences(tmp_path, include_weather):
    data = _hourly_data()
    ordered_path = tmp_path / "ordered.csv"
    shuffled_path = tmp_path / "shuffled.csv"
    data.to_csv(ordered_path, index=False)
    data.sample(frac=1, random_state=42).to_csv(shuffled_path, index=False)

    ordered, features = load_data(str(ordered_path), include_weather)
    shuffled, shuffled_features = load_data(str(shuffled_path), include_weather)

    assert shuffled_features == features
    pd.testing.assert_frame_equal(shuffled, ordered)
    assert shuffled.index.is_monotonic_increasing
    stamp = shuffled.index[0]
    original = data.set_index("datetime")["power_demand"]
    assert shuffled.loc[stamp, "demand_lag_1"] == original.loc[stamp - pd.Timedelta(hours=1)]
    train, _, _ = split_data_by_time(shuffled)
    sequence, target = TimeSeriesDataset(train.values, seq_length=24, horizon=1)[0]
    np.testing.assert_array_equal(sequence[:, 0].numpy(), train.iloc[:24, 0].to_numpy())
    assert target.item() == train.iloc[24, 0]
    assert np.all(np.diff(sequence[:, 0].numpy()) > 0)


@pytest.mark.parametrize("bad_datetime", ["not-a-date", ""])
def test_invalid_or_missing_datetime_is_rejected(tmp_path, bad_datetime):
    data = _hourly_data().astype({"datetime": "object"})
    data.loc[0, "datetime"] = bad_datetime
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)datetime"):
        load_data(str(path))


def test_duplicate_timestamps_are_rejected(tmp_path):
    data = _hourly_data()
    data.loc[1, "datetime"] = data.loc[0, "datetime"]
    path = tmp_path / "duplicate.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        load_data(str(path))
