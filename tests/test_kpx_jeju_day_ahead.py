"""Tests for the official Jeju day-ahead SMP reader."""

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from api.kpx_jeju_day_ahead import (
    SOURCE_URL,
    fetch_jeju_day_ahead,
    parse_jeju_day_ahead,
)

SEOUL = ZoneInfo("Asia/Seoul")
NOW = datetime(2026, 11, 3, 12, 30, tzinfo=SEOUL)


def page(now=NOW, price="165.47"):
    """Produce a structurally representative official portal page."""
    dates = [now.date() - timedelta(days=i) for i in range(6, -1, -1)]
    headers = "".join(f"<th>{day:%m.%d}<br>(화)</th>" for day in dates)
    rows = "".join(
        f"<tr><td>{hour}h</td>" + "<td>100.0</td>" * 6
        + f"<td>{price if hour == now.hour + 1 else '120.0'}</td></tr>"
        for hour in range(1, 25)
    )
    return (
        "<html><head><title>제주(하루전시장) &lt; 계통한계가격(SMP)"
        " : 전력거래소</title></head><body>"
        f"<h5><a>이전</a>{now:%Y년 %m월 %d일} 거래분<a>다음</a></h5>"
        "<p class='info_top'>단위 : 원/kWh</p><table>"
        "<caption>제주 SMP</caption><thead><tr><th>구분</th>"
        f"{headers}</tr></thead><tbody>{rows}"
        "<tr><td>최대</td>" + "<td>120</td>" * 7 + "</tr>"
        "</tbody></table></body></html>"
    )


def test_valid_day_ahead_and_provenance():
    result = parse_jeju_day_ahead(page(), NOW)
    assert result is not None
    assert (result.trade_day, result.trade_hour, result.price) == ("20261103", 13, 165.47)
    assert result.market_type == "day_ahead"
    assert result.data_source == "KPX 제주 하루전시장"
    assert result.source_url == SOURCE_URL


@pytest.mark.parametrize("value, expected", [("0.0", 0.0), ("-15.23", -15.23)])
def test_zero_and_negative_prices(value, expected):
    assert parse_jeju_day_ahead(page(price=value), NOW).price == expected


@pytest.mark.parametrize("value", ["", "-", "NaN", "inf", "1e3", "1,2", "12원"])
def test_malformed_prices_fail_closed(value):
    assert parse_jeju_day_ahead(page(price=value), NOW) is None


@pytest.mark.parametrize("transform", [
    lambda s: s.replace("제주 SMP</caption>", "육지 SMP</caption>"),
    lambda s: s.replace("제주(하루전시장)", "제주(실시간시장)"),
    lambda s: s.replace("원/kWh", "원/MWh"),
    lambda s: s.replace("2026년 11월 03일 거래분", "11월 03일 거래분"),
    lambda s: s.replace("2026년 11월 03일 거래분", "2025년 11월 03일 거래분"),
    lambda s: s.replace("11.03<br>(화)", "11.02<br>(화)"),
    lambda s: s.replace("11.02<br>(화)", "11.03<br>(화)"),
    lambda s: s.replace("<td>13h</td>", "<td>12h</td>"),
    lambda s: s.replace("<td>13h</td>", "<td>13 hours</td>"),
    lambda s: s.replace("<td>165.47</td>", ""),
    lambda s: s.replace("<td>13h</td>", "<td rowspan='2'>13h</td>"),
    lambda s: s.replace("</body>", "<h5>2025년 11월 03일 거래분</h5></body>"),
    lambda s: s.replace("</body>", "<div>접근이 차단되었습니다</div></body>"),
])
def test_untrusted_or_malformed_pages_fail_closed(transform):
    assert parse_jeju_day_ahead(transform(page()), NOW) is None


def test_block_page_fails_closed():
    assert parse_jeju_day_ahead("<title>Security check</title>접근 차단", NOW) is None


def test_target_date_absent_fails_closed():
    assert parse_jeju_day_ahead(page(NOW - timedelta(days=1)), NOW) is None


@pytest.mark.parametrize("now", [
    datetime(2027, 1, 1, 0, 15, tzinfo=SEOUL),
    datetime(2026, 12, 31, 23, 15, tzinfo=SEOUL),
])
def test_year_rollover_and_endpoint_hour(now):
    result = parse_jeju_day_ahead(page(now), now)
    assert result is not None
    assert result.trade_day == now.strftime("%Y%m%d")
    assert result.trade_hour == now.hour + 1


def test_timezone_conversion_and_naive_rejection():
    assert parse_jeju_day_ahead(page(), NOW.astimezone(timezone.utc)).trade_hour == 13
    assert parse_jeju_day_ahead(page(), NOW.replace(tzinfo=None)) is None


def test_exact_hour_is_end_of_previous_trading_period():
    now = NOW.replace(minute=0)
    result = parse_jeju_day_ahead(page(now), now)
    assert result is not None
    assert result.trade_hour == 12


def test_midnight_boundary_fails_closed_without_previous_day_table():
    now = NOW.replace(hour=0, minute=0)
    assert parse_jeju_day_ahead(page(now), now) is None


def test_async_fetch_success_and_timeout(monkeypatch):
    calls = []
    original_client = httpx.AsyncClient

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=page(), request=request)

    def client_factory(**kwargs):
        assert kwargs["timeout"] is not None
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    result = asyncio.run(fetch_jeju_day_ahead(now=NOW))
    assert result.price == 165.47
    assert str(calls[0].url) == SOURCE_URL


@pytest.mark.parametrize("status", [403, 404, 500])
def test_async_http_errors_return_none(monkeypatch, status):
    original_client = httpx.AsyncClient
    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, request=request)
        ), **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    assert asyncio.run(fetch_jeju_day_ahead(now=NOW)) is None


def test_async_network_failure_returns_none(monkeypatch):
    original_client = httpx.AsyncClient
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)
    def client_factory(**kwargs):
        return original_client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    assert asyncio.run(fetch_jeju_day_ahead(now=NOW)) is None
