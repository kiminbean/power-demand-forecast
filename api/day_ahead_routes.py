"""Official Jeju day-ahead SMP endpoint, separate from real-time SMP."""

from datetime import datetime, timedelta
from typing import Dict, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from pydantic import BaseModel

from api import kpx_jeju_day_ahead
from api.kpx_jeju_day_ahead import JejuDayAheadSmp

router = APIRouter(prefix="/api/v1/smp", tags=["SMP"])

SEOUL = ZoneInfo("Asia/Seoul")
# 공개 대시보드의 반복 새로고침이 KPX 포털을 매번 조회하지 않도록 짧게 캐시하되,
# 정정 공표를 반영할 수 있도록 같은 거래시간 안에서도 주기적으로 다시 읽는다.
QUOTE_CACHE_TTL = timedelta(minutes=10)
_quote_cache: Dict[str, Tuple[JejuDayAheadSmp, datetime]] = {}


class JejuDayAheadResponse(BaseModel):
    """Price for the present Seoul trade hour in the day-ahead market."""

    status: Literal["available", "no_data"]
    current_smp: Optional[float] = None
    trade_day: Optional[str] = None
    trade_hour: Optional[int] = None
    market_type: Literal["day_ahead"] = "day_ahead"
    data_source: Optional[str] = None
    source_url: Optional[str] = None


def _now() -> datetime:
    return datetime.now(SEOUL)


def _covers(quote: JejuDayAheadSmp, now: datetime) -> bool:
    """거래시간 H는 (H-1)시 초과 ~ H시 이하 구간이다(정각은 직전 구간의 끝점)."""
    end = datetime.strptime(quote.trade_day, "%Y%m%d").replace(tzinfo=SEOUL) + timedelta(hours=quote.trade_hour)
    return end - timedelta(hours=1) < now <= end


async def _current_quote(now: datetime) -> Optional[JejuDayAheadSmp]:
    cached = _quote_cache.get("jeju")
    if cached is not None:
        quote, fetched_at = cached
        if _covers(quote, now) and now - fetched_at < QUOTE_CACHE_TTL:
            return quote
    # 검증 실패나 네트워크 오류는 캐시하지 않고 다음 요청에서 다시 시도한다.
    quote = await kpx_jeju_day_ahead.fetch_jeju_day_ahead(now)
    if quote is None:
        _quote_cache.pop("jeju", None)
        return None
    _quote_cache["jeju"] = (quote, now)
    return quote


@router.get("/day-ahead", response_model=JejuDayAheadResponse)
async def get_jeju_day_ahead() -> JejuDayAheadResponse:
    """Return the verified KPX Jeju day-ahead price for the present Seoul trade hour."""
    quote = await _current_quote(_now())
    if quote is None:
        return JejuDayAheadResponse(status="no_data")
    return JejuDayAheadResponse(
        status="available",
        current_smp=quote.price,
        trade_day=quote.trade_day,
        trade_hour=quote.trade_hour,
        market_type="day_ahead",
        data_source=quote.data_source,
        source_url=quote.source_url,
    )
