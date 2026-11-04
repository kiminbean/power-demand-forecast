"""Regression tests for non-finite prediction input values."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_service


@pytest.mark.parametrize("field", ["power_demand", "temperature", "humidity", "wind_speed", "precipitation"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_nonfinite_observations(field: str, value: str) -> None:
    """Non-finite observations must not be passed to the prediction service."""
    service = Mock()
    service.predict.return_value = {
        "success": True, "prediction": 750.0, "model_used": "demand_only",
        "timestamp": "2024-01-01T00:00:00", "processing_time_ms": 1.0,
    }
    app.dependency_overrides[get_service] = lambda: service
    start = datetime(2024, 1, 1)
    data = [
        {"datetime": (start + timedelta(hours=i)).isoformat(), "power_demand": 750.0}
        for i in range(168)
    ]
    data[-1][field] = value
    try:
        response = TestClient(app).post("/predict", json={"data": data})
    finally:
        app.dependency_overrides.pop(get_service, None)
    assert response.status_code == 422
    service.predict.assert_not_called()
