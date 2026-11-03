"""The day-ahead SMP is explicitly distinguished from real-time prices."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import kpx_jeju_day_ahead
from api.kpx_jeju_day_ahead import JejuDayAheadSmp


@pytest.mark.parametrize("price", [0.0, 154.08])
def test_unverified_day_ahead_quote_is_not_published(monkeypatch, price):
    quote = JejuDayAheadSmp(trade_day="20261103", trade_hour=8, price=price)
    fetch = AsyncMock(return_value=quote)
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)
    with TestClient(app) as client:
        response = client.get("/api/v1/smp/day-ahead")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_data"
    assert body["market_type"] == "day_ahead"
    assert body["current_smp"] is None
    assert body["trade_day"] is None
    assert body["trade_hour"] is None
    assert body["source_url"] is None
    fetch.assert_not_awaited()


def test_day_ahead_endpoint_handles_absent_source(monkeypatch):
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", AsyncMock(return_value=None))
    with TestClient(app) as client:
        response = client.get("/api/v1/smp/day-ahead")
    assert response.status_code == 200
    assert response.json() == {
        "status": "no_data", "current_smp": None, "trade_day": None,
        "trade_hour": None, "market_type": "day_ahead", "data_source": None,
        "source_url": None,
    }
