"""Official Jeju day-ahead SMP endpoint, separate from real-time SMP."""

from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

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
    """Withhold prices until publication and column identity can be verified."""
    # 공표 여부와 날짜별 열의 신뢰성을 확인할 때까지 가격을 공개하지 않는다.
    return JejuDayAheadResponse(status="no_data")
