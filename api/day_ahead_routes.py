"""Official Jeju day-ahead SMP endpoint, separate from real-time SMP."""

from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from api.kpx_jeju_day_ahead import fetch_jeju_day_ahead

router = APIRouter(prefix="/api/v1/smp", tags=["SMP"])


class JejuDayAheadResponse(BaseModel):
    """Price for the present Seoul trade hour in the day-ahead market."""

    status: Literal["available", "no_data"]
    current_smp: Optional[float] = None
    trade_day: Optional[str] = None
    trade_hour: Optional[int] = None
    market_type: Literal["day_ahead"] = "day_ahead"
    data_source: Optional[str] = None
    source_url: Optional[str] = None


@router.get("/day-ahead", response_model=JejuDayAheadResponse)
async def get_jeju_day_ahead() -> JejuDayAheadResponse:
    """Return only an authenticated-source day-ahead price for today's hour."""
    quote = await fetch_jeju_day_ahead()
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
