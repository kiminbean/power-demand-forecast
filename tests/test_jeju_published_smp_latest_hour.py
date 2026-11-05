"""제주시범사업 API의 실제 응답 형태로 공표 SMP의 최신 시간 선택과 음수 가격을 검증한다.

2026-11-05 실측 응답: 날짜는 내림차순, 같은 날짜 안에서 hour(1~24)·gugan(15분 4구간)은
오름차순이다. jsmp는 하루전, jsmpRt는 실시간 가격이며 jsmpRt는 0 또는 음수일 수 있다.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

from api import realtime_api


def _rows(date, hour, rt_prices, da=100.0):
    return [
        {"date": date, "hour": hour, "gugan": f"{i + 1}구간", "jsmp": da,
         "jsmpRt": price, "jlfd": 600.0, "jlfdRt": 610.0}
        for i, price in enumerate(rt_prices)
    ]


def _api_order():
    # 실제 API 정렬: 최신 날짜가 먼저, 날짜 안에서는 1시부터 24시까지
    items = []
    for date in ("20261103", "20261102"):
        for hour in range(1, 25):
            items += _rows(date, hour, [80.0 + hour] * 4)
    return items


def _mock_jeju_api(monkeypatch, items):
    response = Mock()
    response.json.return_value = {
        "response": {"header": {"resultCode": "00"}, "body": {"items": {"item": items}}}
    }
    client = AsyncMock()
    client.get.return_value = response
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    factory = Mock(return_value=context)
    monkeypatch.setattr(realtime_api.httpx, "AsyncClient", factory)
    return factory


def test_latest_hour_of_latest_date_is_selected(monkeypatch):
    _mock_jeju_api(monkeypatch, _api_order())
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    result = asyncio.run(client._get_jeju_smp_via_api())

    assert result is not None
    assert (result.trade_day, result.trade_hour) == ("20261103", 24)
    assert result.smp_jeju == 104.0


def test_negative_and_zero_realtime_prices_are_valid(monkeypatch):
    _mock_jeju_api(monkeypatch, _rows("20261102", 12, [-60.0, -71.43, -71.43, 0.0], da=0.0))
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    result = asyncio.run(client._get_jeju_smp_via_api())

    assert result is not None
    assert result.trade_hour == 12
    assert round(result.smp_jeju, 4) == round((-60.0 - 71.43 - 71.43 + 0.0) / 4, 4)


def test_jeju_published_smp_is_served_from_verified_api(monkeypatch):
    factory = _mock_jeju_api(monkeypatch, _api_order())
    client = realtime_api.RealtimeAPIClient(api_key="test-key")

    result = asyncio.run(client.get_smp_realtime(area_code="9"))

    assert result is not None
    assert (result.trade_day, result.trade_hour) == ("20261103", 24)
    assert factory.call_count == 1
    assert client._smp_cache["smp_9"][0] is result
