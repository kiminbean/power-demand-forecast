"""Serve Jeju SMP only from the verified Jeju pilot API, never from unverified fallbacks."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from api import realtime_api


def test_jeju_realtime_returns_cached_verified_price_without_refetch(monkeypatch):
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    price = realtime_api.SMPData(
        trade_day="20261103", trade_hour=4, smp_land=150.0,
        smp_jeju=120.0, timestamp=datetime.now(),
    )
    client._smp_cache["smp_9"] = (price, datetime.now())
    source = AsyncMock(return_value=price)
    monkeypatch.setattr(client, "_get_jeju_smp_via_api", source)
    old_source = Mock()
    monkeypatch.setattr(realtime_api.httpx, "AsyncClient", old_source)

    assert asyncio.run(client.get_smp_realtime(area_code="9")) is price
    source.assert_not_called()
    old_source.assert_not_called()


def test_jeju_realtime_does_not_fallback_to_unverified_old_api(monkeypatch):
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    source = AsyncMock(return_value=None)
    monkeypatch.setattr(client, "_get_jeju_smp_via_api", source)
    old_source = Mock()
    monkeypatch.setattr(realtime_api.httpx, "AsyncClient", old_source)

    assert asyncio.run(client.get_smp_realtime(area_code="9")) is None
    source.assert_awaited_once()
    old_source.assert_not_called()
    assert "smp_9" not in client._smp_cache
