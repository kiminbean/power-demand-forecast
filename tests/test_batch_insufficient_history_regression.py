"""Regression tests for insufficient preprocessed batch history."""

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app, get_service
from api.service import PredictionService
from inference.predict import ProductionPredictor


class IdentityScaler:
    """Provide an identity transform for a deterministic test model."""

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        return values


class LastDemand(torch.nn.Module):
    """Return the last demand in each sequence."""

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence[:, -1, :1]


@pytest.fixture
def configured_predictor() -> ProductionPredictor:
    """Use real preprocessing with a lightweight deterministic model."""
    predictor = ProductionPredictor(device=torch.device("cpu"))
    predictor.is_loaded = True
    predictor.model_demand = LastDemand()
    predictor.scaler_demand = IdentityScaler()
    predictor.config_demand = {
        "features": ["power_demand", "demand_lag_168"],
        "training_config": {"seq_length": 168},
    }
    return predictor


def frame_for(n_rows: int) -> pd.DataFrame:
    """Create a complete chronological demand history."""
    return pd.DataFrame(
        {"power_demand": np.arange(n_rows, dtype=float)},
        index=pd.date_range("2024-01-01", periods=n_rows, freq="h"),
    )


@pytest.mark.parametrize("seq_length,n_rows", [(168, 168), (168, 335), (48, 215)])
def test_direct_batch_reports_insufficient_processed_history(
    configured_predictor: ProductionPredictor, seq_length: int, n_rows: int
) -> None:
    configured_predictor.config_demand["training_config"]["seq_length"] = seq_length
    actual = max(n_rows - 168, 0)
    with pytest.raises(ValueError, match=rf"(?i)insufficient.*{seq_length}.*{actual}"):
        configured_predictor.predict_batch(frame_for(n_rows))


@pytest.mark.parametrize("seq_length,n_rows,step,expected", [
    (168, 336, 1, 1), (168, 337, 1, 2), (168, 337, 24, 1),
    (48, 216, 1, 1), (48, 217, 1, 2), (48, 217, 24, 1),
])
def test_batch_boundary_is_consistent_across_direct_and_api(
    configured_predictor: ProductionPredictor,
    seq_length: int,
    n_rows: int,
    step: int,
    expected: int,
) -> None:
    configured_predictor.config_demand["training_config"]["seq_length"] = seq_length
    frame = frame_for(n_rows)
    direct = configured_predictor.predict_batch(frame, step=step)
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    app.dependency_overrides[get_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/predict/batch",
            json={"data": [{"datetime": ts.isoformat(), "power_demand": val}
                           for ts, val in frame["power_demand"].items()], "step": step},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)
    assert response.status_code == 200, response.text
    assert len(direct.predictions) == expected
    assert response.json()["total_predictions"] == expected
    assert [item["prediction"] for item in response.json()["predictions"]] == direct.predictions.tolist()


@pytest.mark.parametrize("n_rows,expected_status,actual", [(167, 422, None), (168, 400, 0), (335, 400, 167)])
def test_http_batch_insufficient_history(
    configured_predictor: ProductionPredictor, n_rows: int, expected_status: int, actual: int | None
) -> None:
    frame = frame_for(n_rows)
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    app.dependency_overrides[get_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/predict/batch",
            json={"data": [{"datetime": ts.isoformat(), "power_demand": val}
                           for ts, val in frame["power_demand"].items()]},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)
    assert response.status_code == expected_status
    if actual is not None:
        detail = response.json()["detail"]
        assert "insufficient" in detail.lower()
        assert "168" in detail
        assert str(actual) in detail
