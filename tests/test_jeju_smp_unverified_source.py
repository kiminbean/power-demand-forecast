"""Serve Jeju SMP only from the verified Jeju pilot API, never from unverified fallbacks."""

import asyncio
from datetime import datetime, timedelta
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


def test_jeju_realtime_keeps_last_verified_price_when_api_fails(monkeypatch):
    """data.go.kr 일일 한도(429) 등으로 API가 실패해도 24시간 이내 마지막 검증값은 계속 보여 준다."""
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    price = realtime_api.SMPData(
        trade_day="20261103", trade_hour=24, smp_land=124.93,
        smp_jeju=124.93, timestamp=datetime.now(),
    )
    client._smp_cache["smp_9"] = (price, datetime.now() - timedelta(hours=2))  # TTL(5분) 지남
    monkeypatch.setattr(client, "_get_jeju_smp_via_api", AsyncMock(return_value=None))

    assert asyncio.run(client.get_smp_realtime(area_code="9")) is price


def test_jeju_realtime_does_not_serve_price_older_than_a_day(monkeypatch):
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    price = realtime_api.SMPData(
        trade_day="20261101", trade_hour=24, smp_land=100.0,
        smp_jeju=100.0, timestamp=datetime.now(),
    )
    client._smp_cache["smp_9"] = (price, datetime.now() - timedelta(hours=25))
    monkeypatch.setattr(client, "_get_jeju_smp_via_api", AsyncMock(return_value=None))

    assert asyncio.run(client.get_smp_realtime(area_code="9")) is None


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
