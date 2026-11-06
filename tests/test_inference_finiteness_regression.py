"""Regression coverage for non-finite values across inference paths."""

import numpy as np
import pytest
import torch

from api.schemas import ModelType, TimeSeriesData
from api.service import PredictionService
from tests.test_batch_insufficient_history_regression import configured_predictor, frame_for


class NonfiniteModel(torch.nn.Module):
    """Model returning a non-finite output for validation tests."""

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.full((sequence.shape[0], 1), float("nan"), device=sequence.device)


class NonfiniteInverseScaler:
    """Finite scaled input but non-finite inverse-transformed output."""

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        return np.full_like(values, np.inf)


class LargeFiniteInverseScaler(NonfiniteInverseScaler):
    """Produce finite values whose naive aggregate overflows."""

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        return np.full(values.shape, 1e308, dtype=np.float64)


def run_path(predictor, frame, path: str) -> None:
    service = PredictionService()
    service.predictor = predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value)
               for ts, value in frame["power_demand"].items()]
    if path == "direct_single":
        predictor.predict_demand_only(frame)
    elif path == "direct_batch":
        predictor.predict_batch(frame, step=24)
    elif path == "api_single":
        service._run_prediction(frame, "demand_only")
    else:
        service.predict_batch(records, model_type=ModelType.DEMAND_ONLY, step=24)


@pytest.mark.parametrize("path", ["direct_single", "direct_batch", "api_single", "api_batch"])
def test_overflow_after_scaling_is_rejected(configured_predictor, path: str) -> None:
    frame = frame_for(400)
    frame["power_demand"] = 1e39
    with pytest.raises(ValueError, match="(?i)finite"):
        run_path(configured_predictor, frame, path)


@pytest.mark.parametrize("path", ["direct_single", "direct_batch", "api_single", "api_batch"])
@pytest.mark.parametrize("failure", ["model", "inverse"])
def test_nonfinite_output_is_rejected(configured_predictor, path: str, failure: str) -> None:
    if failure == "model":
        configured_predictor.model_demand = NonfiniteModel()
    else:
        configured_predictor.scaler_demand = NonfiniteInverseScaler()
    with pytest.raises(ValueError, match="(?i)finite"):
        run_path(configured_predictor, frame_for(400), path)


def test_derived_only_demand_rejects_infinite_raw_input(configured_predictor) -> None:
    configured_predictor.config_demand["features"] = ["demand_ma_6h"]
    frame = frame_for(400)
    frame.iloc[-2, frame.columns.get_loc("power_demand")] = np.inf
    with pytest.raises(ValueError, match="(?i)finite"):
        configured_predictor.predict_demand_only(frame)


@pytest.mark.parametrize("step", [0, -1])
@pytest.mark.parametrize("path", ["direct_batch", "api_batch"])
def test_nonpositive_batch_step_is_rejected(configured_predictor, path: str, step: int) -> None:
    frame = frame_for(400)
    if path == "direct_batch":
        with pytest.raises(ValueError, match="(?i)step.*positive"):
            configured_predictor.predict_batch(frame, step=step)
    else:
        service = PredictionService()
        service.predictor = configured_predictor
        service._is_initialized = True
        records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value)
                   for ts, value in frame["power_demand"].items()]
        with pytest.raises(ValueError, match="(?i)step.*positive"):
            service.predict_batch(records, model_type=ModelType.DEMAND_ONLY, step=step)


@pytest.mark.parametrize("path", ["direct_batch", "api_batch"])
def test_numpy_integer_step_is_accepted(configured_predictor, path: str) -> None:
    frame = frame_for(400)
    if path == "direct_batch":
        result = configured_predictor.predict_batch(frame, step=np.int64(24))
        assert len(result.predictions) > 0
    else:
        service = PredictionService()
        service.predictor = configured_predictor
        service._is_initialized = True
        records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value)
                   for ts, value in frame["power_demand"].items()]
        assert service.predict_batch(records, model_type=ModelType.DEMAND_ONLY,
                                     step=np.int64(24)).total_predictions > 0


def test_batch_statistics_never_return_nonfinite_values(configured_predictor) -> None:
    configured_predictor.scaler_demand = LargeFiniteInverseScaler()
    frame = frame_for(400)
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value)
               for ts, value in frame["power_demand"].items()]
    with pytest.raises(ValueError, match="(?i)statistics.*finite"):
        service.predict_batch(records, model_type=ModelType.DEMAND_ONLY, step=24)
