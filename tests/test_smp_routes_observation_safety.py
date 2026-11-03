"""Do not publish model estimates or synthetic SMP as observations."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import smp_routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(smp_routes.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("region", ["jeju", "mainland"])
@pytest.mark.parametrize("model_used", ["v3.2", "fallback"])
def test_current_rejects_predictions_as_observations(client, monkeypatch, region, model_used):
    prediction = Mock(return_value={"q50": [150.0], "model_used": model_used})
    monkeypatch.setattr(smp_routes, "get_real_smp_prediction", prediction)

    response = client.get("/smp/current", params={"region": region})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
    prediction.assert_not_called()


@pytest.mark.parametrize("model_used", ["v3.2", "fallback"])
def test_comparison_does_not_present_predictions_as_actual_prices(client, monkeypatch, model_used):
    prediction = Mock(return_value={"q50": [150.0], "model_used": model_used})
    monkeypatch.setattr(smp_routes, "get_real_smp_prediction", prediction)

    response = client.get("/smp/compare")

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
    prediction.assert_not_called()


@pytest.mark.parametrize("region", [None, "jeju", "mainland"])
def test_historical_does_not_publish_random_data(client, region):
    params = {"start_date": "2025-01-01T00:00:00", "end_date": "2025-01-02T00:00:00"}
    if region:
        params["region"] = region

    response = client.get("/smp/historical", params=params)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "start,end",
    [
        ("2025-01-02T00:00:00", "2025-01-01T00:00:00"),
        ("2025-01-01T00:00:00", "2025-02-01T00:00:00"),
    ],
)
def test_historical_preserves_date_range_validation(client, start, end):
    response = client.get("/smp/historical", params={"start_date": start, "end_date": end})
    assert response.status_code == 400


def test_prediction_retains_explicit_demo_mode(client, monkeypatch):
    now = datetime(2025, 1, 1)
    monkeypatch.setattr(
        smp_routes,
        "get_real_smp_prediction",
        lambda hours: {
            "times": [now], "q10": [90.0], "q50": [100.0], "q90": [110.0],
            "model_used": "fallback",
        },
    )

    response = client.post("/smp/predict", json={"region": "mainland", "hours": 1})

    assert response.status_code == 200
    assert response.json()["model_info"]["mode"] == "demo"
    assert response.json()["predictions"][0]["predicted_smp"] == 100.0
