"""The day-ahead SMP is explicitly distinguished from real-time prices.

2026-11-05 교차 검증: KPX 포털 제주 하루전시장 표(10-30~11-03, 120시간)가 제주시범사업 API의
jsmp(하루전)와 전부 일치했고, jsmp≠jsmpRt인 8개 시간에서도 모두 하루전 값과 일치했다.
"""

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import day_ahead_routes, kpx_jeju_day_ahead
from api.kpx_jeju_day_ahead import JejuDayAheadSmp

SEOUL = ZoneInfo("Asia/Seoul")


@pytest.fixture(autouse=True)
def clear_quote_cache():
    day_ahead_routes._quote_cache.clear()
    yield
    day_ahead_routes._quote_cache.clear()


def _at(monkeypatch, *args):
    now = datetime(*args, tzinfo=SEOUL)
    monkeypatch.setattr(day_ahead_routes, "_now", lambda: now)
    return now


def _get():
    with TestClient(app) as client:
        response = client.get("/api/v1/smp/day-ahead")
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("price", [0.0, -10.5, 154.08])
def test_verified_day_ahead_quote_is_published(monkeypatch, price):
    _at(monkeypatch, 2026, 11, 5, 15, 54)
    quote = JejuDayAheadSmp(trade_day="20261105", trade_hour=16, price=price)
    fetch = AsyncMock(return_value=quote)
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)

    assert _get() == {
        "status": "available", "current_smp": price, "trade_day": "20261105",
        "trade_hour": 16, "market_type": "day_ahead", "data_source": "KPX 제주 하루전시장",
        "source_url": kpx_jeju_day_ahead.SOURCE_URL,
    }
    fetch.assert_awaited_once()


def test_day_ahead_endpoint_handles_absent_source(monkeypatch):
    _at(monkeypatch, 2026, 11, 5, 15, 54)
    fetch = AsyncMock(return_value=None)
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)

    assert _get() == {
        "status": "no_data", "current_smp": None, "trade_day": None,
        "trade_hour": None, "market_type": "day_ahead", "data_source": None,
        "source_url": None,
    }
    _get()
    assert fetch.await_count == 2  # 실패는 캐시하지 않는다


def test_quote_is_reused_within_its_trading_hour(monkeypatch):
    _at(monkeypatch, 2026, 11, 5, 15, 1)
    fetch = AsyncMock(return_value=JejuDayAheadSmp("20261105", 16, 179.89))
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)

    _get()
    _at(monkeypatch, 2026, 11, 5, 15, 9)
    assert _get()["current_smp"] == 179.89
    fetch.assert_awaited_once()


@pytest.mark.parametrize("later", [(2026, 11, 5, 16, 0, 1), (2026, 11, 5, 15, 12)])
def test_quote_is_refetched_after_hour_change_or_ttl(monkeypatch, later):
    _at(monkeypatch, 2026, 11, 5, 15, 1)
    fetch = AsyncMock(side_effect=[
        JejuDayAheadSmp("20261105", 16, 179.89),
        JejuDayAheadSmp("20261105", 17, 150.0),
    ])
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)

    _get()
    _at(monkeypatch, *later)
    _get()
    assert fetch.await_count == 2


def test_exact_hour_still_belongs_to_cached_period(monkeypatch):
    # 정각 16:00은 16h 구간(15:00 초과~16:00 이하)의 끝점이다.
    _at(monkeypatch, 2026, 11, 5, 15, 55)
    fetch = AsyncMock(return_value=JejuDayAheadSmp("20261105", 16, 179.89))
    monkeypatch.setattr(kpx_jeju_day_ahead, "fetch_jeju_day_ahead", fetch)

    _get()
    _at(monkeypatch, 2026, 11, 5, 16, 0)
    assert _get()["trade_hour"] == 16
    fetch.assert_awaited_once()
