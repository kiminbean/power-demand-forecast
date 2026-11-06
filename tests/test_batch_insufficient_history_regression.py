"""Regression tests for insufficient preprocessed batch history."""

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from api.config import settings
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
                           for ts, val in frame["power_demand"].items()],
                  "model_type": "demand_only", "step": step},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)
    assert response.status_code == 200, response.text
    assert len(direct.predictions) == expected
    assert response.json()["total_predictions"] == expected
    assert [item["prediction"] for item in response.json()["predictions"]] == direct.predictions.tolist()


def test_http_batch_without_model_type_falls_back_to_lstm_when_ar_gbm_is_missing(
    configured_predictor: ProductionPredictor, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "DEFAULT_MODEL", "ar_gbm")
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True

    def post(frame: pd.DataFrame):
        app.dependency_overrides[get_service] = lambda: service
        try:
            return TestClient(app).post(
                "/predict/batch",
                json={"data": [{"datetime": ts.isoformat(), "power_demand": val}
                               for ts, val in frame["power_demand"].items()], "step": 24},
            )
        finally:
            app.dependency_overrides.pop(get_service, None)

    frame = frame_for(400)
    response = post(frame)
    # ar_gbm을 불러오지 못한 서버에서 생략 요청은 LSTM 값 그대로이고 대체 이유가 표시된다
    assert response.status_code == 200, response.text
    assert response.json()["model_used"] == "demand_only (fallback: ar_gbm unavailable)"
    direct = configured_predictor.predict_batch(frame, step=24)
    assert [item["prediction"] for item in response.json()["predictions"]] == direct.predictions.tolist()
    # 대체 모델도 336시간이 필요하므로 그보다 짧은 생략 요청은 예전처럼 400이다
    short = post(frame_for(335))
    assert short.status_code == 400 and "insufficient" in short.json()["detail"].lower()


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
                           for ts, val in frame["power_demand"].items()],
                  "model_type": "demand_only"},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)
    assert response.status_code == expected_status
    if actual is not None:
        detail = response.json()["detail"]
        assert "insufficient" in detail.lower()
        assert "168" in detail
        assert str(actual) in detail
