"""Regression coverage for bidding quantities and optimizer fallback."""

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ai_bidding_optimizer import AIBiddingOptimizer
from api.main import app as v6_app
from src.rebms.api.main import app as v5_app


CAPACITIES = [0.0001, 0.01, 0.03, 1.01, 50.0, 500.0]
RISKS = ["conservative", "moderate", "aggressive"]
URL = "/api/v1/bidding/optimized-segments"


def assert_quantities(data: dict, capacity: float) -> None:
    """Check each response segment and its reported total against capacity."""
    assert data["capacity_mw"] == capacity
    assert len(data["hourly_bids"]) == 24
    assert len({bid["hour"] for bid in data["hourly_bids"]}) == 24
    for bid in data["hourly_bids"]:
        assert len(bid["segments"]) == 10
        quantities = [segment["quantity_mw"] for segment in bid["segments"]]
        assert all(q >= 0 for q in quantities)
        assert math.fsum(quantities) == pytest.approx(capacity, rel=1e-12, abs=0)
        assert bid["total_mw"] == pytest.approx(capacity, rel=1e-12, abs=0)
        expected_price = math.fsum(
            s["price_krw_mwh"] * (s["quantity_mw"] / math.fsum(quantities))
            for s in bid["segments"]
        )
        assert bid["avg_price"] == pytest.approx(round(expected_price, 1))
    assert data["total_daily_mwh"] > 0


@pytest.fixture
def forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real optimizer with a deterministic forecast."""
    monkeypatch.setattr(AIBiddingOptimizer, "get_smp_forecast", lambda self, hours: {
        "q10": [80.0] * hours, "q50": [100.0] * hours,
        "q90": [120.0] * hours, "times": list(range(hours)),
        "model_used": "test forecast",
    })


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("risk", RISKS)
def test_v6_optimizer_preserves_capacity(forecast: None, capacity: float, risk: str) -> None:
    response = TestClient(v6_app).get(URL, params={"capacity_mw": capacity, "risk_level": risk})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "test forecast"
    assert_quantities(data, capacity)


@pytest.mark.parametrize("capacity", CAPACITIES + [math.ulp(0.0) * units for units in range(1, 21)])
def test_v6_optimizer_failure_falls_back_for_all_hours(monkeypatch: pytest.MonkeyPatch, capacity: float) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced optimizer failure")

    monkeypatch.setattr(AIBiddingOptimizer, "optimize_hourly_bids", fail)
    response = TestClient(v6_app).get(URL, params={"capacity_mw": capacity})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "Fallback (rule-based)"
    assert {bid["hour"] for bid in data["hourly_bids"]} == set(range(24))
    assert_quantities(data, capacity)


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("risk", RISKS)
def test_v5_sibling_preserves_capacity(monkeypatch: pytest.MonkeyPatch, capacity: float, risk: str) -> None:
    from src.smp.models import smp_predictor

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced predictor failure")

    monkeypatch.setattr(smp_predictor, "get_smp_predictor", fail)
    response = TestClient(v5_app).get(URL, params={"capacity_mw": capacity, "risk_level": risk})
    assert response.status_code == 200
    assert_quantities(response.json(), capacity)


@pytest.mark.parametrize("app", [v6_app, v5_app])
@pytest.mark.parametrize("risk", RISKS)
@pytest.mark.parametrize("units", range(1, 21))
def test_subnormal_capacities_preserve_nonnegative_quantities(
    app: FastAPI, risk: str, units: int, forecast: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.smp.models import smp_predictor

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced predictor failure")

    monkeypatch.setattr(smp_predictor, "get_smp_predictor", fail)
    capacity = math.ulp(0.0) * units
    response = TestClient(app).get(URL, params={"capacity_mw": capacity, "risk_level": risk})
    assert response.status_code == 200
    data = response.json()
    for bid in data["hourly_bids"]:
        quantities = [segment["quantity_mw"] for segment in bid["segments"]]
        assert all(q >= 0 for q in quantities)
        assert math.fsum(quantities) == capacity
        assert bid["total_mw"] == capacity


@pytest.mark.parametrize("app", [v6_app, v5_app])
def test_smallest_positive_capacity_is_preserved(
    app: FastAPI, forecast: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.smp.models import smp_predictor

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced predictor failure")

    monkeypatch.setattr(smp_predictor, "get_smp_predictor", fail)
    capacity = math.ulp(0.0)
    response = TestClient(app).get(URL, params={"capacity_mw": capacity})
    assert response.status_code == 200
    data = response.json()
    assert len(data["hourly_bids"]) == 24
    for bid in data["hourly_bids"]:
        quantities = [segment["quantity_mw"] for segment in bid["segments"]]
        assert all(q >= 0 for q in quantities)
        assert math.fsum(quantities) == capacity
        assert bid["total_mw"] == capacity
        assert bid["avg_price"] > 0


@pytest.mark.parametrize("app", [v6_app, v5_app])
@pytest.mark.parametrize("capacity", ["0", "-1", "500.01", "nan", "inf"])
def test_invalid_capacity_rejected(app: FastAPI, capacity: str) -> None:
    response = TestClient(app).get(URL, params={"capacity_mw": capacity})
    assert response.status_code == 422


@pytest.mark.parametrize("capacity", [0, -1, 500.01, float("nan"), float("inf")])
def test_optimizer_rejects_invalid_capacity(capacity: float) -> None:
    with pytest.raises(ValueError):
        AIBiddingOptimizer().optimize_hourly_bids(capacity_mw=capacity)
