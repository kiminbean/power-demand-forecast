"""Regression coverage for inference time-axis safety."""

import numpy as np
import pandas as pd
import pytest

from api.schemas import TimeSeriesData
from api.service import PredictionService
from inference.feature_preprocessing import prepare_inference_features


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    """Build sufficient history for lag features and a prediction sequence."""
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    return pd.DataFrame({"power_demand": np.arange(400, dtype=float)}, index=index)


@pytest.mark.parametrize("case", ["reverse", "partial_reverse", "duplicate", "gap", "half_hour", "nat"])
def test_invalid_time_axis_is_rejected_before_feature_generation(
    hourly_frame: pd.DataFrame, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row-based lag and filling must not run on an invalid time axis."""
    if case == "reverse":
        hourly_frame = hourly_frame.iloc[::-1]
    elif case == "partial_reverse":
        positions = np.arange(len(hourly_frame))
        positions[200], positions[201] = positions[201], positions[200]
        hourly_frame = hourly_frame.iloc[positions]
    elif case == "duplicate":
        index = list(hourly_frame.index)
        index[200] = index[199]
        hourly_frame.index = pd.DatetimeIndex(index)
    elif case == "gap":
        hourly_frame = hourly_frame.drop(hourly_frame.index[200])
    elif case == "half_hour":
        hourly_frame.index = pd.date_range("2024-01-01", periods=400, freq="30min")
    else:
        index = list(hourly_frame.index)
        index[200] = pd.NaT
        hourly_frame.index = pd.DatetimeIndex(index)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("Feature generation ran before time-axis validation")

    monkeypatch.setattr("inference.feature_preprocessing.add_time_features", fail_if_called)
    with pytest.raises(ValueError, match="(?i)(hour|timestamp|datetime)"):
        prepare_inference_features(hourly_frame)


def test_non_datetime_index_is_rejected(hourly_frame: pd.DataFrame) -> None:
    """The direct interface must not silently interpret integer indexes as dates."""
    with pytest.raises(ValueError, match="DatetimeIndex"):
        prepare_inference_features(hourly_frame.reset_index(drop=True))


def test_datetime_column_is_supported(hourly_frame: pd.DataFrame) -> None:
    """Existing DataFrame datetime-column input remains supported."""
    result = prepare_inference_features(hourly_frame.rename_axis("datetime").reset_index())
    assert len(result) == 232


def test_api_preserves_sorting_compatibility(hourly_frame: pd.DataFrame) -> None:
    """API continues to sort unique hourly observations before preprocessing."""
    records = [
        TimeSeriesData(datetime=index.to_pydatetime(), power_demand=value)
        for index, value in hourly_frame["power_demand"].items()
    ]
    service = PredictionService()
    ordered = service._convert_to_dataframe(records)
    reversed_input = service._convert_to_dataframe(records[::-1])
    pd.testing.assert_frame_equal(
        service._prepare_features_for_api(ordered),
        service._prepare_features_for_api(reversed_input),
    )


def test_aware_hourly_index_accepts_elapsed_time_across_dst() -> None:
    """Hourly continuity uses elapsed time for timezone-aware timestamps."""
    index = pd.date_range("2024-03-01", periods=400, freq="h", tz="America/New_York")
    frame = pd.DataFrame({"power_demand": np.arange(400, dtype=float)}, index=index)
    assert len(prepare_inference_features(frame)) == 232


def test_calendar_features_match_between_utc_and_jeju_local_time(hourly_frame: pd.DataFrame) -> None:
    local = hourly_frame.copy()
    local.index = local.index.tz_localize("Asia/Seoul")
    utc = local.copy()
    utc.index = utc.index.tz_convert("UTC")
    left = prepare_inference_features(local)
    right = prepare_inference_features(utc)
    pd.testing.assert_frame_equal(left, right)


def test_api_accepts_mixed_dst_offsets_as_aware_hourly_data() -> None:
    index = pd.date_range("2024-03-01", periods=400, freq="h", tz="America/New_York")
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=float(i))
               for i, ts in enumerate(index)]
    result = PredictionService()._convert_to_dataframe(records)
    assert str(result.index.tz) == "Asia/Seoul"
    assert len(prepare_inference_features(result)) == 232


def test_api_rejects_mixed_naive_and_aware_timestamps(hourly_frame: pd.DataFrame) -> None:
    index = hourly_frame.index
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=float(i))
               for i, ts in enumerate(index)]
    records[5].datetime = records[5].datetime.replace(tzinfo=__import__("datetime").timezone.utc)
    with pytest.raises(ValueError, match="(?i)(naive|timezone|aware)"):
        PredictionService()._convert_to_dataframe(records)
