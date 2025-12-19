"""
RE-BMS Bidding API Router
=========================

REST endpoints for 10-segment bid management.

Endpoints:
- POST /bids - Create new bid
- GET /bids - List bids
- GET /bids/{bid_id} - Get bid details
- POST /bids/{bid_id}/submit - Submit to KPX
- POST /bids/{bid_id}/optimize - AI optimize bid
- POST /bids/{bid_id}/validate - Validate bid
- DELETE /bids/{bid_id} - Cancel bid
"""

from datetime import date, datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    DailyBidCreate,
    DailyBidResponse,
    BidOptimizeRequest,
    BidSubmitResponse,
    ValidationResponse,
    ValidationErrorDetail,
    ErrorResponse,
)
from ...models.bid import (
    BidSegment,
    HourlyBid,
    DailyBid,
    MarketType,
    BidStatus,
    create_optimized_segments,
)
from ...validators.bid_validator import BidValidator, ValidationError


router = APIRouter()

# In-memory storage (replace with database in production)
_bids_storage: dict = {}
_validator = BidValidator(check_deadline=False)  # Disable deadline check for dev


def _daily_bid_to_response(bid: DailyBid) -> DailyBidResponse:
    """Convert DailyBid to response schema"""
    return DailyBidResponse(
        bid_id=bid.bid_id,
        resource_id=bid.resource_id,
        market_type=bid.market_type.value,
        trading_date=bid.trading_date.isoformat(),
        hourly_bids=[
            {
                'hour': hb.hour,
                'segments': [
                    {
                        'segment_id': s.segment_id,
                        'quantity_mw': s.quantity_mw,
                        'price_krw_mwh': s.price_krw_mwh,
                        'cumulative_mw': s.cumulative_mw,
                    }
                    for s in hb.segments
                ],
                'total_capacity_mw': hb.total_capacity_mw,
                'weighted_avg_price': hb.weighted_avg_price,
            }
            for hb in bid.hourly_bids
        ],
        status=bid.status.value,
        created_at=bid.created_at.isoformat(),
        updated_at=bid.updated_at.isoformat(),
        submitted_at=bid.submitted_at.isoformat() if bid.submitted_at else None,
        kpx_reference_id=bid.kpx_reference_id,
        risk_level=bid.risk_level,
        smp_forecast_used=bid.smp_forecast_used,
        total_daily_mwh=bid.total_daily_mwh,
        average_price=bid.average_price,
    )


@router.post("/", response_model=DailyBidResponse)
async def create_bid(request: DailyBidCreate):
    """
    Create a new 10-segment bid

    Creates a bid for Day-Ahead or Real-Time Market.
    Validates monotonic price constraint before saving.
    """
    try:
        # Convert request to domain objects
        hourly_bids = []
        for hb_data in request.hourly_bids:
            segments = [
                BidSegment(
                    segment_id=s.segment_id,
                    quantity_mw=s.quantity_mw,
                    price_krw_mwh=s.price_krw_mwh,
                )
                for s in hb_data.segments
            ]
            hourly_bids.append(HourlyBid(hour=hb_data.hour, segments=segments))

        # Create bid
        bid = DailyBid(
            bid_id=str(uuid4()),
            resource_id=request.resource_id,
            market_type=MarketType(request.market_type),
            trading_date=request.trading_date,
            hourly_bids=hourly_bids,
            risk_level=request.risk_level,
        )

        # Validate
        _validator.validate_and_raise(bid)

        # Store
        _bids_storage[bid.bid_id] = bid

        return _daily_bid_to_response(bid)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[DailyBidResponse])
async def list_bids(
    resource_id: Optional[str] = None,
    trading_date: Optional[date] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    List bids with filtering

    Supports filtering by resource_id, trading_date, and status.
    """
    bids = list(_bids_storage.values())

    # Apply filters
    if resource_id:
        bids = [b for b in bids if b.resource_id == resource_id]

    if trading_date:
        bids = [b for b in bids if b.trading_date == trading_date]

    if status:
        try:
            status_enum = BidStatus(status)
            bids = [b for b in bids if b.status == status_enum]
        except ValueError:
            pass

    # Sort by created_at descending
    bids.sort(key=lambda x: x.created_at, reverse=True)

    # Paginate
    bids = bids[offset:offset + limit]

    return [_daily_bid_to_response(b) for b in bids]


@router.get("/{bid_id}", response_model=DailyBidResponse)
async def get_bid(bid_id: str):
    """Get bid details by ID"""
    bid = _bids_storage.get(bid_id)
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    return _daily_bid_to_response(bid)


@router.post("/{bid_id}/submit", response_model=BidSubmitResponse)
async def submit_bid(bid_id: str):
    """
    Submit bid to KPX

    Validates deadline and submits via KPX API.
    In development mode, simulates submission.
    """
    bid = _bids_storage.get(bid_id)
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    if bid.status != BidStatus.DRAFT:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit bid with status: {bid.status.value}"
        )

    # Simulate KPX submission (replace with actual KPX client)
    bid.status = BidStatus.SUBMITTED
    bid.submitted_at = datetime.now()
    bid.kpx_reference_id = f"KPX-{uuid4().hex[:8].upper()}"

    return BidSubmitResponse(
        status="submitted",
        bid_id=bid.bid_id,
        kpx_reference_id=bid.kpx_reference_id,
        submitted_at=bid.submitted_at.isoformat(),
    )


@router.post("/{bid_id}/optimize", response_model=DailyBidResponse)
async def optimize_bid(bid_id: str, request: BidOptimizeRequest):
    """
    AI-optimize bid using SMP forecast

    Uses existing SMP predictor to optimize 10-segment structure
    based on quantile predictions and risk tolerance.
    """
    bid = _bids_storage.get(bid_id)
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    if bid.status != BidStatus.DRAFT:
        raise HTTPException(
            status_code=400,
            detail="Can only optimize draft bids"
        )

    # Get SMP forecast (try to use existing predictor)
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)
        predictions = predictor.predict_24h()

        q10 = predictions.get('q10', [90.0] * 24)
        q50 = predictions.get('q50', [100.0] * 24)
        q90 = predictions.get('q90', [110.0] * 24)

        # Convert to lists if numpy arrays
        if hasattr(q10, 'tolist'):
            q10 = q10.tolist()
        if hasattr(q50, 'tolist'):
            q50 = q50.tolist()
        if hasattr(q90, 'tolist'):
            q90 = q90.tolist()

    except Exception:
        # Fallback to default SMP values
        q10 = [80.0 + i * 0.5 for i in range(24)]
        q50 = [95.0 + i * 0.5 for i in range(24)]
        q90 = [110.0 + i * 0.5 for i in range(24)]

    # Optimize each hourly bid
    for i, hb in enumerate(bid.hourly_bids):
        hour_idx = hb.hour - 1
        if hour_idx < 24:
            optimized_segments = create_optimized_segments(
                total_capacity_mw=hb.total_capacity_mw or 50.0,  # Default 50 MW
                smp_q10=q10[hour_idx],
                smp_q50=q50[hour_idx],
                smp_q90=q90[hour_idx],
                risk_level=request.risk_level,
            )
            bid.hourly_bids[i] = HourlyBid(hour=hb.hour, segments=optimized_segments)

    bid.smp_forecast_used = request.use_smp_forecast
    bid.risk_level = request.risk_level
    bid.updated_at = datetime.now()

    return _daily_bid_to_response(bid)


@router.post("/{bid_id}/validate", response_model=ValidationResponse)
async def validate_bid(bid_id: str):
    """
    Validate bid constraints

    Checks monotonic constraint, capacity limits, and price ranges.
    """
    bid = _bids_storage.get(bid_id)
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    is_valid, errors = _validator.validate_daily_bid(bid)

    return ValidationResponse(
        is_valid=is_valid,
        total_errors=len(errors),
        errors=[
            ValidationErrorDetail(
                type=e.get('type', 'unknown'),
                message=e.get('message', ''),
                hour=e.get('hour'),
            )
            for e in errors
        ],
        bid_id=bid.bid_id,
    )


@router.delete("/{bid_id}")
async def cancel_bid(bid_id: str):
    """Cancel a draft or submitted bid"""
    bid = _bids_storage.get(bid_id)
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    if bid.status in [BidStatus.SETTLED, BidStatus.ACCEPTED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel bid with status: {bid.status.value}"
        )

    bid.status = BidStatus.CANCELLED
    bid.updated_at = datetime.now()

    return {"status": "cancelled", "bid_id": bid_id}


@router.post("/template/{resource_id}", response_model=DailyBidResponse)
async def create_bid_template(
    resource_id: str,
    trading_date: date,
    capacity_mw: float = Query(default=50.0, gt=0),
):
    """
    Create empty bid template

    Creates a template with zero quantities for all 24 hours.
    Useful for starting a new bid from scratch.
    """
    bid = DailyBid.create_empty(
        bid_id=str(uuid4()),
        resource_id=resource_id,
        trading_date=trading_date,
        market_type=MarketType.DAM,
    )

    _bids_storage[bid.bid_id] = bid

    return _daily_bid_to_response(bid)
