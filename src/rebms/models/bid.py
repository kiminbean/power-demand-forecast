"""
RE-BMS Bid Models
=================

10-segment bid structures for KPX Day-Ahead and Real-Time Markets.

KPX Market Rules:
- DAM (Day-Ahead Market): Submit by D-1 10:00, hourly granularity
- RTM (Real-Time Market): Submit 15 minutes before, 15-minute granularity
- Monotonic Constraint: Prices must be non-decreasing across segments
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class MarketType(Enum):
    """Market type enumeration"""
    DAM = "day_ahead"     # Day-Ahead Market (D-1 by 10:00)
    RTM = "real_time"     # Real-Time Market (15-minute intervals)


class BidStatus(Enum):
    """Bid lifecycle status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_ACCEPTED = "partially_accepted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    SETTLED = "settled"


@dataclass
class BidSegment:
    """Single segment of 10-segment bid

    KPX requires monotonically increasing prices across segments.
    Each segment represents a price-quantity pair.

    Attributes:
        segment_id: Segment number (1-10)
        quantity_mw: Generation quantity in MW
        price_krw_mwh: Price in KRW/MWh
        cumulative_mw: Cumulative quantity up to this segment

    Example:
        >>> seg = BidSegment(segment_id=1, quantity_mw=10.0, price_krw_mwh=80.0)
        >>> seg.segment_id
        1
    """
    segment_id: int              # 1-10
    quantity_mw: float           # Generation quantity (MW)
    price_krw_mwh: float         # Price (KRW/MWh)
    cumulative_mw: float = 0.0   # Cumulative quantity (auto-calculated)

    def __post_init__(self):
        if not 1 <= self.segment_id <= 10:
            raise ValueError(f"Segment ID must be 1-10, got {self.segment_id}")
        if self.quantity_mw < 0:
            raise ValueError("Quantity cannot be negative")
        if self.price_krw_mwh < 0:
            raise ValueError("Price cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'segment_id': self.segment_id,
            'quantity_mw': self.quantity_mw,
            'price_krw_mwh': self.price_krw_mwh,
            'cumulative_mw': self.cumulative_mw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BidSegment':
        """Create from dictionary"""
        return cls(
            segment_id=data['segment_id'],
            quantity_mw=data['quantity_mw'],
            price_krw_mwh=data['price_krw_mwh'],
            cumulative_mw=data.get('cumulative_mw', 0.0),
        )


@dataclass
class HourlyBid:
    """Hourly bid with 10 segments

    Represents bidding for a single hour with up to 10 price-quantity pairs.
    Must satisfy monotonic price increase constraint.

    Attributes:
        hour: Hour of the day (1-24 for DAM, 1-96 for RTM 15-min)
        segments: List of bid segments (up to 10)
        total_capacity_mw: Total bid capacity (auto-calculated)
        weighted_avg_price: Weighted average price (auto-calculated)

    Example:
        >>> segments = [
        ...     BidSegment(1, 10.0, 80.0),
        ...     BidSegment(2, 8.0, 85.0),
        ... ]
        >>> hb = HourlyBid(hour=1, segments=segments)
        >>> hb.total_capacity_mw
        18.0
    """
    hour: int                               # 1-24 for DAM, 1-96 for RTM (15-min)
    segments: List[BidSegment] = field(default_factory=list)
    total_capacity_mw: float = 0.0          # Total bid capacity
    weighted_avg_price: float = 0.0         # Weighted average price

    def __post_init__(self):
        if self.segments:
            self._validate_monotonic_constraint()
            self._calculate_aggregates()

    def _validate_monotonic_constraint(self):
        """Validate KPX monotonic increase constraint

        Raises:
            ValueError: If prices are not monotonically increasing
        """
        if len(self.segments) > 10:
            raise ValueError(f"Maximum 10 segments allowed, got {len(self.segments)}")

        # Get non-zero quantity segments
        prices = [s.price_krw_mwh for s in self.segments if s.quantity_mw > 0]

        for i in range(1, len(prices)):
            if prices[i] < prices[i-1]:
                raise ValueError(
                    f"Monotonic constraint violated at hour {self.hour}: "
                    f"segment {i+1} price ({prices[i]}) < segment {i} price ({prices[i-1]})"
                )

    def _calculate_aggregates(self):
        """Calculate total capacity and weighted average price"""
        self.total_capacity_mw = sum(s.quantity_mw for s in self.segments)

        if self.total_capacity_mw > 0:
            self.weighted_avg_price = sum(
                s.quantity_mw * s.price_krw_mwh for s in self.segments
            ) / self.total_capacity_mw

        # Calculate cumulative MW
        cumulative = 0.0
        for segment in self.segments:
            cumulative += segment.quantity_mw
            segment.cumulative_mw = cumulative

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hour': self.hour,
            'segments': [s.to_dict() for s in self.segments],
            'total_capacity_mw': self.total_capacity_mw,
            'weighted_avg_price': self.weighted_avg_price,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HourlyBid':
        """Create from dictionary"""
        segments = [BidSegment.from_dict(s) for s in data.get('segments', [])]
        return cls(
            hour=data['hour'],
            segments=segments,
        )

    @classmethod
    def create_empty(cls, hour: int, num_segments: int = 10) -> 'HourlyBid':
        """Create empty hourly bid with zero-quantity segments"""
        segments = [
            BidSegment(segment_id=i, quantity_mw=0.0, price_krw_mwh=0.0)
            for i in range(1, num_segments + 1)
        ]
        return cls(hour=hour, segments=segments)


@dataclass
class DailyBid:
    """Complete daily bid submission for DAM

    Contains 24 hourly bids with 10-segment structure each.
    Submitted by D-1 10:00 for Day-Ahead Market.

    Attributes:
        bid_id: Unique identifier (UUID)
        resource_id: Generator resource ID
        market_type: DAM or RTM
        trading_date: Trading date (D)
        hourly_bids: List of 24 hourly bids
        status: Current bid status
        risk_level: Risk tolerance (conservative/moderate/aggressive)

    Example:
        >>> bid = DailyBid(
        ...     bid_id="bid-001",
        ...     resource_id="solar-001",
        ...     market_type=MarketType.DAM,
        ...     trading_date=date(2025, 12, 20),
        ...     hourly_bids=[HourlyBid.create_empty(h) for h in range(1, 25)],
        ... )
        >>> len(bid.hourly_bids)
        24
    """
    bid_id: str                             # Unique identifier (UUID)
    resource_id: str                        # Generator resource ID
    market_type: MarketType                 # DAM or RTM
    trading_date: date                      # Trading date (D)
    hourly_bids: List[HourlyBid] = field(default_factory=list)

    # Metadata
    status: BidStatus = BidStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    kpx_reference_id: Optional[str] = None  # KPX transaction reference

    # Risk parameters
    risk_level: str = "moderate"            # conservative/moderate/aggressive
    smp_forecast_used: bool = False         # Whether AI forecast was used

    # Summary (auto-calculated)
    total_daily_mwh: float = 0.0
    average_price: float = 0.0

    def __post_init__(self):
        if self.hourly_bids:
            self._calculate_summary()

    def _calculate_summary(self):
        """Calculate daily summary statistics"""
        self.total_daily_mwh = sum(hb.total_capacity_mw for hb in self.hourly_bids)

        total_value = sum(
            hb.total_capacity_mw * hb.weighted_avg_price
            for hb in self.hourly_bids
        )

        if self.total_daily_mwh > 0:
            self.average_price = total_value / self.total_daily_mwh

    def get_hourly_bid(self, hour: int) -> Optional[HourlyBid]:
        """Get hourly bid by hour number"""
        for hb in self.hourly_bids:
            if hb.hour == hour:
                return hb
        return None

    def update_hourly_bid(self, hour: int, hourly_bid: HourlyBid):
        """Update hourly bid for specific hour"""
        for i, hb in enumerate(self.hourly_bids):
            if hb.hour == hour:
                self.hourly_bids[i] = hourly_bid
                self.updated_at = datetime.now()
                self._calculate_summary()
                return

        # Add if not exists
        self.hourly_bids.append(hourly_bid)
        self.hourly_bids.sort(key=lambda x: x.hour)
        self.updated_at = datetime.now()
        self._calculate_summary()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API response"""
        return {
            'bid_id': self.bid_id,
            'resource_id': self.resource_id,
            'market_type': self.market_type.value,
            'trading_date': self.trading_date.isoformat(),
            'hourly_bids': [hb.to_dict() for hb in self.hourly_bids],
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'kpx_reference_id': self.kpx_reference_id,
            'risk_level': self.risk_level,
            'smp_forecast_used': self.smp_forecast_used,
            'total_daily_mwh': self.total_daily_mwh,
            'average_price': self.average_price,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DailyBid':
        """Create from dictionary"""
        hourly_bids = [HourlyBid.from_dict(hb) for hb in data.get('hourly_bids', [])]

        return cls(
            bid_id=data['bid_id'],
            resource_id=data['resource_id'],
            market_type=MarketType(data['market_type']),
            trading_date=date.fromisoformat(data['trading_date']),
            hourly_bids=hourly_bids,
            status=BidStatus(data.get('status', 'draft')),
            risk_level=data.get('risk_level', 'moderate'),
            smp_forecast_used=data.get('smp_forecast_used', False),
        )

    @classmethod
    def create_empty(
        cls,
        bid_id: str,
        resource_id: str,
        trading_date: date,
        market_type: MarketType = MarketType.DAM,
        num_hours: int = 24,
    ) -> 'DailyBid':
        """Create empty daily bid template"""
        hourly_bids = [
            HourlyBid.create_empty(hour=h)
            for h in range(1, num_hours + 1)
        ]

        return cls(
            bid_id=bid_id,
            resource_id=resource_id,
            market_type=market_type,
            trading_date=trading_date,
            hourly_bids=hourly_bids,
        )


def create_optimized_segments(
    total_capacity_mw: float,
    smp_q10: float,
    smp_q50: float,
    smp_q90: float,
    risk_level: str = "moderate",
) -> List[BidSegment]:
    """Create optimized 10-segment structure based on SMP forecast

    Args:
        total_capacity_mw: Total capacity to allocate
        smp_q10: SMP 10th percentile (conservative price)
        smp_q50: SMP median (moderate price)
        smp_q90: SMP 90th percentile (aggressive price)
        risk_level: "conservative", "moderate", or "aggressive"

    Returns:
        List of 10 BidSegment objects with monotonically increasing prices
    """
    # Calculate price ladder (monotonically increasing)
    prices = np.linspace(smp_q10 * 0.9, smp_q90 * 1.1, 10)

    # Ensure strict monotonic increase
    for i in range(1, len(prices)):
        if prices[i] <= prices[i-1]:
            prices[i] = prices[i-1] + 0.01

    # Calculate quantity distribution based on risk level
    if risk_level == "conservative":
        # Front-loaded: more in lower-priced segments
        weights = np.array([0.20, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05])
    elif risk_level == "aggressive":
        # Back-loaded: more in higher-priced segments
        weights = np.array([0.05, 0.05, 0.07, 0.08, 0.08, 0.10, 0.10, 0.12, 0.15, 0.20])
    else:
        # Moderate: even distribution
        weights = np.array([0.10] * 10)

    quantities = total_capacity_mw * weights

    # Create segments
    segments = [
        BidSegment(
            segment_id=i + 1,
            quantity_mw=round(quantities[i], 2),
            price_krw_mwh=round(prices[i], 2),
        )
        for i in range(10)
    ]

    return segments
