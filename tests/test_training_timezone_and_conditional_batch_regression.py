"""Regression tests for training timezones and conditional batch inference."""

import numpy as np
import pandas as pd
import pytest

from api.schemas import ModelType, TimeSeriesData
from api.service import PredictionService
from src.data.dataset import split_data_by_time
from src.inference.feature_preprocessing import prepare_inference_features
from src.training.train_production import load_data
from tests.test_batch_insufficient_history_regression import configured_predictor, frame_for


def test_utc_training_calendar_features_match_inference(tmp_path) -> None:
    """An aware instant must use the same Jeju calendar in both paths."""
    frame = frame_for(400)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize("UTC")
    frame.index.name = "datetime"
    path = tmp_path / "aware.csv"
    frame.rename_axis("datetime").to_csv(path)
    training, features = load_data(str(path))
    inference = prepare_inference_features(frame, selected_features=features)
    pd.testing.assert_frame_equal(training, inference.loc[training.index, features])


def test_aware_loaded_training_data_splits_at_local_boundaries(tmp_path) -> None:
    frame = frame_for(400)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize("UTC")
    path = tmp_path / "aware_split.csv"
    frame.rename_axis("datetime").to_csv(path)
    training, _ = load_data(str(path))
    local_times = pd.DatetimeIndex(training.index).tz_localize(None)
    train_end = local_times[100]
    val_end = local_times[150]
    train, val, test = split_data_by_time(training, str(train_end), str(val_end))
    assert [len(train), len(val), len(test)] == [101, 50, len(training) - 151]
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), training)


@pytest.mark.parametrize("month", [1, 7])
@pytest.mark.parametrize("weather_loaded", [False, True])
def test_conditional_batch_matches_api_demand_only_contract(
    configured_predictor, month: int, weather_loaded: bool
) -> None:
    """Conditional batch keeps the API's demand-only behavior in all seasons."""
    frame = frame_for(400)
    frame.index = pd.date_range(f"2024-{month:02d}-01", periods=400, freq="h")
    if weather_loaded:
        configured_predictor.model_weather = configured_predictor.model_demand
        configured_predictor.scaler_weather = configured_predictor.scaler_demand
        configured_predictor.config_weather = {
            "features": ["power_demand", "기온"],
            "training_config": {"seq_length": 168},
        }
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=stamp.to_pydatetime(), power_demand=value)
               for stamp, value in frame["power_demand"].items()]
    direct = configured_predictor.predict_batch(frame, model="conditional", step=24)
    api = service.predict_batch(records, model_type=ModelType.CONDITIONAL, step=24)
    np.testing.assert_allclose(direct.predictions, [item["prediction"] for item in api.predictions])
    assert direct.model_used == "demand_only"


def test_conditional_batch_uses_demand_even_with_distinct_weather_model(configured_predictor) -> None:
    """Batch remains demand-only even when weather predictions would differ."""
    import torch

    class ConstantWeather(torch.nn.Module):
        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            return torch.full((sequence.shape[0], 1), 10.0, device=sequence.device)

    frame = frame_for(400)
    frame["기온"] = 8.0
    configured_predictor.model_weather = ConstantWeather()
    configured_predictor.scaler_weather = configured_predictor.scaler_demand
    configured_predictor.config_weather = {
        "features": ["power_demand", "기온"], "training_config": {"seq_length": 168}
    }
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=stamp.to_pydatetime(), power_demand=row.power_demand,
                              temperature=row["기온"])
               for stamp, row in frame.iterrows()]
    direct = configured_predictor.predict_batch(frame, model="conditional", step=24)
    api = service.predict_batch(records, model_type=ModelType.CONDITIONAL, step=24)
    np.testing.assert_allclose(direct.predictions, [item["prediction"] for item in api.predictions])
    assert direct.model_used == api.model_used == "demand_only"
    assert configured_predictor.predict_conditional(frame).predicted_demand != direct.predictions[-1]
