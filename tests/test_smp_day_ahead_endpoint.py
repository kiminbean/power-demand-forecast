"""The day-ahead SMP is explicitly distinguished from real-time prices."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api import day_ahead_routes
from api.kpx_jeju_day_ahead import JejuDayAheadSmp


def test_day_ahead_endpoint_identifies_market_and_source(monkeypatch):
    quote = JejuDayAheadSmp(trade_day="20261103", trade_hour=8, price=154.08)
    monkeypatch.setattr(day_ahead_routes, "fetch_jeju_day_ahead", AsyncMock(return_value=quote))
    with TestClient(app) as client:
        response = client.get("/api/v1/smp/day-ahead")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["market_type"] == "day_ahead"
    assert body["current_smp"] == 154.08
    assert body["trade_day"] == "20261103"
    assert body["trade_hour"] == 8
    assert body["source_url"] == quote.source_url


def test_day_ahead_endpoint_handles_absent_source(monkeypatch):
    monkeypatch.setattr(day_ahead_routes, "fetch_jeju_day_ahead", AsyncMock(return_value=None))
    with TestClient(app) as client:
        response = client.get("/api/v1/smp/day-ahead")
    assert response.status_code == 200
    assert response.json() == {
        "status": "no_data", "current_smp": None, "trade_day": None,
        "trade_hour": None, "market_type": "day_ahead", "data_source": None,
        "source_url": None,
    }
