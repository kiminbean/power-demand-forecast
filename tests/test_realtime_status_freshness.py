"""Tests for SMP observation-date disclosure in the realtime status API."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from api import v6_routes


@pytest.mark.parametrize(
    ("trade_day", "expected"),
    [("20261102", "current_date"), ("20261031", "historical"),
     ("20261103", "future"), ("not-a-date", "unknown"), ("", "unknown")],
)
def test_smp_trade_date_status(trade_day: str, expected: str) -> None:
    """Classify the trade date without assuming an hourly publication schedule."""
    now = datetime(2026, 11, 2, 10, tzinfo=ZoneInfo("Asia/Seoul"))
    assert v6_routes._smp_trade_date_status(trade_day, now) == expected


def test_status_discloses_old_smp_even_if_transport_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An old SMP response remains connected but is identified as historical."""
    smp = SimpleNamespace(smp_jeju=106.6, trade_hour=1, trade_day="20000101")
    monkeypatch.setattr(v6_routes.realtime_client, "get_smp_realtime", AsyncMock(return_value=smp))
    monkeypatch.setattr(v6_routes.realtime_client, "get_power_supply_realtime", AsyncMock(return_value=None))
    monkeypatch.setattr(v6_routes.realtime_client, "get_weather_realtime", AsyncMock(return_value=None))
    result = asyncio.run(v6_routes.get_realtime_api_status())
    assert result.smp_api["status"] == "connected"
    assert result.smp_api["data"]["trade_day"] == "20000101"
    assert result.smp_api["data_date_status"] == "historical"
