"""Serve Jeju SMP only from the verified Jeju pilot API, never from unverified fallbacks."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

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


def test_last_verified_price_survives_restart_via_disk(monkeypatch, tmp_path):
    """재시작으로 메모리 캐시가 비고 API도 실패하면 디스크에 남긴 마지막 검증값을 쓴다."""
    path = tmp_path / "jeju_rt_smp_last.json"
    first = realtime_api.RealtimeAPIClient(api_key="test-key")
    first.jeju_smp_last_path = path
    price = realtime_api.SMPData(trade_day="20261103", trade_hour=24, smp_land=124.93,
                                 smp_jeju=124.93, timestamp=datetime.now(), data_source="KPX 제주시범사업 API")
    monkeypatch.setattr(first, "_get_jeju_smp_via_api", AsyncMock(return_value=price))
    asyncio.run(first.get_smp_realtime(area_code="9"))
    assert path.exists()

    restarted = realtime_api.RealtimeAPIClient(api_key="test-key")
    restarted.jeju_smp_last_path = path
    monkeypatch.setattr(restarted, "_get_jeju_smp_via_api", AsyncMock(return_value=None))
    result = asyncio.run(restarted.get_smp_realtime(area_code="9"))
    assert (result.trade_day, result.trade_hour, result.smp_jeju) == ("20261103", 24, 124.93)


@pytest.mark.parametrize("content", [
    '{"trade_day": "20261103", "trade_hour": 24, "smp_land": 1, "smp_jeju": 1, "data_source": "x", "saved_at": "2000-01-01T00:00:00"}',
    '{"trade_day": "20261103", "trade_hour": 24, "smp_jeju": "NaN", "saved_at": "%s"}',
    "not json",
])
def test_stale_or_invalid_disk_price_is_ignored(monkeypatch, tmp_path, content):
    path = tmp_path / "jeju_rt_smp_last.json"
    path.write_text(content.replace("%s", datetime.now().isoformat()), encoding="utf-8")
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    client.jeju_smp_last_path = path
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
