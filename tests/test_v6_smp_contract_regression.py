"""Regression tests for v6 SMP provenance and missing-value handling."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI

from api import v6_routes
from api.realtime_api import SMPData


@pytest.fixture
def api_client_dependencies(monkeypatch):
    """Replace external APIs and models with deterministic responses."""
    smp = AsyncMock(return_value=None)
    monkeypatch.setattr(v6_routes.realtime_client, "get_smp_realtime", smp)
    monkeypatch.setattr(v6_routes.realtime_client, "get_power_supply_realtime", AsyncMock(return_value=None))
    monkeypatch.setattr(v6_routes.realtime_client, "get_weather_realtime", AsyncMock(return_value=None))
    monkeypatch.setattr(v6_routes, "load_power_demand_data", lambda: pd.DataFrame())
    monkeypatch.setattr(v6_routes, "load_weather_data", lambda: pd.DataFrame())
    monkeypatch.setattr(
        v6_routes, "load_smp_data",
        lambda: pd.DataFrame({"smp_jeju": [999.0], "hour": [0]}),
    )
    prediction = SimpleNamespace(solar_mw=10.0, wind_mw=20.0, method="test", confidence=1.0)
    monkeypatch.setattr(v6_routes, "get_ml_predictor", lambda: SimpleNamespace(predict_hour=lambda **kwargs: prediction))
    return smp


def fetch(path):
    """Exercise actual response model serialization through ASGI."""
    app = FastAPI()
    app.include_router(v6_routes.router)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(path)
            assert response.status_code == 200, response.text
            return response.json()

    return asyncio.run(request())


def smp_data(trade_day, price):
    return SMPData(trade_day=trade_day, trade_hour=1, smp_land=0, smp_jeju=price,
                   timestamp=datetime.now(), data_source="test source")


def test_missing_smp_is_null_in_both_routes(api_client_dependencies):
    current = fetch("/api/v1/smp/current")
    kpi = fetch("/api/v1/dashboard/kpis")
    assert current["current_smp"] is None
    assert current["comparison"] is None
    assert current["data_date_status"] == "no_data"
    assert current["trade_day"] is None
    assert current["data_source"] is None
    assert current["latest_smp"] is None
    assert kpi["current_smp"] is None
    assert kpi["smp_change_pct"] is None
    assert kpi["daily_revenue_million"] is None
    assert kpi["revenue_change_pct"] is None


@pytest.mark.parametrize("day_offset,expected", [(-1, "historical"), (1, "future")])
def test_noncurrent_smp_not_reported_as_current(api_client_dependencies, day_offset, expected):
    day = (datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(days=day_offset)).strftime("%Y%m%d")
    api_client_dependencies.return_value = smp_data(day, 123.4)
    current = fetch("/api/v1/smp/current")
    kpi = fetch("/api/v1/dashboard/kpis")
    assert current["current_smp"] is None
    assert current["comparison"] is None
    assert current["data_date_status"] == expected
    assert current["trade_day"] == day
    assert current["data_source"] == "test source"
    assert current["latest_smp"] == {
        "price": 123.4, "trade_day": day, "trade_hour": 1,
        "data_source": "test source", "market": "unknown", "region": "jeju",
    }
    assert kpi["current_smp"] is None
    assert kpi["smp_change_pct"] is None
    assert kpi["daily_revenue_million"] is None


def test_unknown_trade_date_is_not_a_current_price(api_client_dependencies):
    api_client_dependencies.return_value = smp_data("bad-date", 123.4)
    current = fetch("/api/v1/smp/current")
    assert current["current_smp"] is None
    assert current["data_date_status"] == "unknown"
    assert current["trade_day"] == "bad-date"
    assert fetch("/api/v1/dashboard/kpis")["current_smp"] is None


@pytest.mark.parametrize("price", [123.4, 0.0])
def test_current_date_is_dated_not_live_including_zero(api_client_dependencies, price):
    day = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    api_client_dependencies.return_value = smp_data(day, price)
    current = fetch("/api/v1/smp/current")
    kpi = fetch("/api/v1/dashboard/kpis")
    assert current["current_smp"] is None
    assert current["comparison"] is None
    assert current["data_date_status"] == "current_date"
    assert current["trade_day"] == day
    assert current["data_source"] == "test source"
    assert current["latest_smp"] == {
        "price": price, "trade_day": day, "trade_hour": 1,
        "data_source": "test source", "market": "unknown", "region": "jeju",
    }
    assert kpi["current_smp"] is None
    assert kpi["smp_change_pct"] is None
    assert kpi["daily_revenue_million"] is None
    assert kpi["revenue_change_pct"] is None
