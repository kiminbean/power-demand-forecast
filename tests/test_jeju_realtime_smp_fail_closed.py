"""Prevent day-ahead prices from masquerading as real-time Jeju SMP."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from api import realtime_api


def _item(**overrides):
    item = {
        "date": "20261103",
        "hour": 4,
        "jsmp": 150.0,
        "jsmpRt": 120.0,
        "jlfd": 600.0,
        "jlfdRt": 610.0,
    }
    item.update(overrides)
    return item


def _mock_jeju_api(monkeypatch, items):
    response = Mock()
    response.json.return_value = {
        "response": {
            "header": {"resultCode": "00"},
            "body": {"items": {"item": items}},
        }
    }
    client = AsyncMock()
    client.get.return_value = response
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(realtime_api.httpx, "AsyncClient", Mock(return_value=context))
    return client


@pytest.mark.parametrize("value", [None, "abc", True, float("nan"), float("inf")])
def test_invalid_realtime_price_never_substitutes_day_ahead(monkeypatch, value):
    item = _item(jsmpRt=value)
    if value is None:
        del item["jsmpRt"]
    _mock_jeju_api(monkeypatch, [item])
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    assert asyncio.run(client._get_jeju_smp_via_api()) is None


@pytest.mark.parametrize("value", [None, "abc", True, float("nan"), float("inf")])
def test_incomplete_hour_is_not_averaged_as_realtime(monkeypatch, value):
    invalid = _item(jsmpRt=value)
    if value is None:
        del invalid["jsmpRt"]
    _mock_jeju_api(monkeypatch, [_item(), invalid])
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    assert asyncio.run(client._get_jeju_smp_via_api()) is None


def test_valid_realtime_hour_keeps_realtime_average(monkeypatch):
    _mock_jeju_api(monkeypatch, [_item(jsmpRt=100.0), _item(jsmpRt=120.0)])
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    result = asyncio.run(client._get_jeju_smp_via_api())

    assert result is not None
    assert result.smp_jeju == 110.0
    assert result.smp_land == 150.0
    assert result.trade_day == "20261103"
    assert result.trade_hour == 4


def test_rejected_jeju_value_is_not_cached_when_other_source_fails(monkeypatch):
    upstream = _mock_jeju_api(monkeypatch, [_item(jsmpRt=float("nan"))])
    upstream.get.side_effect = [upstream.get.return_value, realtime_api.httpx.ConnectError("offline")]
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    assert asyncio.run(client.get_smp_realtime(area_code="9")) is None
    assert "smp_9" not in client._smp_cache
