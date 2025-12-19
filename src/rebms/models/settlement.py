"""
RE-BMS Settlement Models
========================

Settlement and penalty calculation models for KPX market.

Settlement Rules:
- Generation Revenue = Actual Generation × Clearing Price (SMP)
- Imbalance Penalty = |Deviation| × Penalty Price

Jeju Pilot Imbalance Rules:
- ±12% tolerance band (no penalty within band)
- Over-generation: 80% of SMP
- Under-generation: 120% of SMP
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP


class SettlementType(Enum):
    """Settlement type"""
    GENERATION = "generation"       # Normal generation settlement
    IMBALANCE = "imbalance"         # Imbalance penalty
    CURTAILMENT = "curtailment"     # Curtailment compensation


class PenaltyType(Enum):
    """Penalty type for imbalance"""
    NO_PENALTY = "no_penalty"           # Within tolerance band
    OVER_GENERATION = "over_generation"  # Generated more than cleared
    UNDER_GENERATION = "under_generation"  # Generated less than cleared


# Jeju Pilot Imbalance Rules
JEJU_TOLERANCE_PERCENT = 12.0  # ±12% tolerance band
OVER_GEN_PENALTY_RATE = 0.80   # 80% of SMP for over-generation
UNDER_GEN_PENALTY_RATE = 1.20  # 120% of SMP for under-generation


@dataclass
class HourlySettlement:
    """Hourly settlement record

    Calculates revenue and penalties for a single hour.

    Attributes:
        hour: Hour number (1-24)
        bid_quantity_mw: Bid amount from submission
        actual_generation_mw: Metered actual generation
        cleared_quantity_mw: Amount cleared by auction
        clearing_price_krw: SMP (System Marginal Price)
        bid_price_krw: Submitted bid price

    Example:
        >>> hs = HourlySettlement(
        ...     hour=12,
        ...     bid_quantity_mw=50.0,
        ...     actual_generation_mw=48.0,
        ...     cleared_quantity_mw=50.0,
        ...     clearing_price_krw=95.0,
        ...     bid_price_krw=90.0,
        ... )
        >>> hs.calculate()
        >>> hs.net_revenue_krw
        Decimal('4560.00')
    """
    # Required fields (no defaults)
    hour: int                               # 1-24
    bid_quantity_mw: float                  # Bid amount
    actual_generation_mw: float             # Metered generation
    cleared_quantity_mw: float              # Cleared by auction
    clearing_price_krw: float               # Market clearing price (SMP)
    bid_price_krw: float                    # Submitted bid price

    # Optional fields (with defaults)
    imbalance_mw: float = 0.0               # Difference (actual - cleared)
    imbalance_price_krw: float = 0.0        # Penalty price

    # Amounts (KRW) - Using Decimal for precision
    generation_revenue_krw: Decimal = field(default=Decimal('0'))
    imbalance_charge_krw: Decimal = field(default=Decimal('0'))
    net_revenue_krw: Decimal = field(default=Decimal('0'))

    # Penalty
    penalty_type: PenaltyType = PenaltyType.NO_PENALTY
    penalty_rate: float = 0.0               # Penalty multiplier
    deviation_percent: float = 0.0          # Deviation from cleared

    def calculate(self, tolerance_percent: float = JEJU_TOLERANCE_PERCENT):
        """Calculate settlement amounts with Jeju pilot rules

        Args:
            tolerance_percent: Tolerance band percentage (default: 12%)
        """
        # Calculate imbalance
        self.imbalance_mw = self.actual_generation_mw - self.cleared_quantity_mw

        # Calculate deviation percentage
        if self.cleared_quantity_mw > 0:
            self.deviation_percent = abs(self.imbalance_mw / self.cleared_quantity_mw) * 100
        else:
            self.deviation_percent = 0.0

        # Generation revenue (always calculated on actual generation)
        self.generation_revenue_krw = Decimal(str(
            self.actual_generation_mw * self.clearing_price_krw
        )).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        # Determine penalty type and calculate charges
        if self.deviation_percent <= tolerance_percent:
            # Within tolerance band - no penalty
            self.penalty_type = PenaltyType.NO_PENALTY
            self.penalty_rate = 0.0
            self.imbalance_price_krw = 0.0
            self.imbalance_charge_krw = Decimal('0')

        elif self.imbalance_mw > 0:
            # Over-generation: gets less than full SMP
            self.penalty_type = PenaltyType.OVER_GENERATION
            self.penalty_rate = OVER_GEN_PENALTY_RATE

            # Excess generation only gets 80% of SMP
            excess_mw = abs(self.imbalance_mw) - (self.cleared_quantity_mw * tolerance_percent / 100)
            if excess_mw > 0:
                self.imbalance_price_krw = self.clearing_price_krw * (1 - self.penalty_rate)
                self.imbalance_charge_krw = Decimal(str(
                    excess_mw * self.imbalance_price_krw
                )).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        else:
            # Under-generation: pays penalty at 120% of SMP
            self.penalty_type = PenaltyType.UNDER_GENERATION
            self.penalty_rate = UNDER_GEN_PENALTY_RATE

            # Shortfall penalized at 120% of SMP
            shortfall_mw = abs(self.imbalance_mw) - (self.cleared_quantity_mw * tolerance_percent / 100)
            if shortfall_mw > 0:
                self.imbalance_price_krw = self.clearing_price_krw * self.penalty_rate
                self.imbalance_charge_krw = Decimal(str(
                    shortfall_mw * self.imbalance_price_krw
                )).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        # Net revenue
        if self.penalty_type == PenaltyType.OVER_GENERATION:
            # Over-gen: charge represents reduced revenue
            self.net_revenue_krw = self.generation_revenue_krw - self.imbalance_charge_krw
        elif self.penalty_type == PenaltyType.UNDER_GENERATION:
            # Under-gen: charge is a penalty
            self.net_revenue_krw = self.generation_revenue_krw - self.imbalance_charge_krw
        else:
            # No penalty
            self.net_revenue_krw = self.generation_revenue_krw

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hour': self.hour,
            'bid_quantity_mw': self.bid_quantity_mw,
            'actual_generation_mw': self.actual_generation_mw,
            'cleared_quantity_mw': self.cleared_quantity_mw,
            'imbalance_mw': self.imbalance_mw,
            'deviation_percent': round(self.deviation_percent, 2),
            'clearing_price_krw': self.clearing_price_krw,
            'bid_price_krw': self.bid_price_krw,
            'imbalance_price_krw': self.imbalance_price_krw,
            'generation_revenue_krw': float(self.generation_revenue_krw),
            'imbalance_charge_krw': float(self.imbalance_charge_krw),
            'net_revenue_krw': float(self.net_revenue_krw),
            'penalty_type': self.penalty_type.value,
            'penalty_rate': self.penalty_rate,
        }


@dataclass
class DailySettlement:
    """Daily settlement summary

    Aggregates 24 hourly settlements with performance metrics.

    Attributes:
        settlement_id: Unique identifier
        resource_id: Generator resource ID
        trading_date: Trading date
        bid_id: Reference to submitted bid
        hourly_settlements: List of 24 hourly settlements

    Example:
        >>> ds = DailySettlement(
        ...     settlement_id="settle-001",
        ...     resource_id="solar-001",
        ...     trading_date=date(2025, 12, 19),
        ...     bid_id="bid-001",
        ...     hourly_settlements=hourly_list,
        ... )
        >>> ds.calculate_totals()
        >>> print(f"Net Revenue: {ds.net_daily_revenue}")
    """
    settlement_id: str
    resource_id: str
    trading_date: date
    bid_id: str                             # Reference to submitted bid

    hourly_settlements: List[HourlySettlement] = field(default_factory=list)

    # Daily aggregates (KRW)
    total_generation_revenue: Decimal = field(default=Decimal('0'))
    total_imbalance_charges: Decimal = field(default=Decimal('0'))
    net_daily_revenue: Decimal = field(default=Decimal('0'))

    # Generation quantities (MWh)
    total_bid_mwh: float = 0.0
    total_cleared_mwh: float = 0.0
    total_actual_mwh: float = 0.0
    total_imbalance_mwh: float = 0.0

    # Performance metrics
    accuracy_percent: float = 0.0           # Forecast accuracy
    average_clearing_price: float = 0.0     # Average SMP
    average_deviation: float = 0.0          # Average deviation %

    # Penalty counts
    hours_over_generation: int = 0
    hours_under_generation: int = 0
    hours_no_penalty: int = 0

    # Status
    is_finalized: bool = False
    finalized_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_totals(self, tolerance_percent: float = JEJU_TOLERANCE_PERCENT):
        """Calculate daily totals from hourly settlements"""
        if not self.hourly_settlements:
            return

        # Calculate each hourly settlement
        for hs in self.hourly_settlements:
            hs.calculate(tolerance_percent)

        # Aggregate revenues and charges
        self.total_generation_revenue = sum(
            hs.generation_revenue_krw for hs in self.hourly_settlements
        )
        self.total_imbalance_charges = sum(
            hs.imbalance_charge_krw for hs in self.hourly_settlements
        )
        self.net_daily_revenue = self.total_generation_revenue - self.total_imbalance_charges

        # Aggregate quantities
        self.total_bid_mwh = sum(hs.bid_quantity_mw for hs in self.hourly_settlements)
        self.total_cleared_mwh = sum(hs.cleared_quantity_mw for hs in self.hourly_settlements)
        self.total_actual_mwh = sum(hs.actual_generation_mw for hs in self.hourly_settlements)
        self.total_imbalance_mwh = sum(abs(hs.imbalance_mw) for hs in self.hourly_settlements)

        # Accuracy
        if self.total_cleared_mwh > 0:
            deviation = abs(self.total_actual_mwh - self.total_cleared_mwh)
            self.accuracy_percent = 100 * (1 - deviation / self.total_cleared_mwh)
        else:
            self.accuracy_percent = 0.0

        # Average clearing price
        self.average_clearing_price = sum(
            hs.clearing_price_krw for hs in self.hourly_settlements
        ) / len(self.hourly_settlements)

        # Average deviation
        self.average_deviation = sum(
            hs.deviation_percent for hs in self.hourly_settlements
        ) / len(self.hourly_settlements)

        # Penalty counts
        self.hours_over_generation = sum(
            1 for hs in self.hourly_settlements
            if hs.penalty_type == PenaltyType.OVER_GENERATION
        )
        self.hours_under_generation = sum(
            1 for hs in self.hourly_settlements
            if hs.penalty_type == PenaltyType.UNDER_GENERATION
        )
        self.hours_no_penalty = sum(
            1 for hs in self.hourly_settlements
            if hs.penalty_type == PenaltyType.NO_PENALTY
        )

    def finalize(self):
        """Mark settlement as finalized"""
        self.is_finalized = True
        self.finalized_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'settlement_id': self.settlement_id,
            'resource_id': self.resource_id,
            'trading_date': self.trading_date.isoformat(),
            'bid_id': self.bid_id,
            'hourly_settlements': [hs.to_dict() for hs in self.hourly_settlements],
            'total_generation_revenue': float(self.total_generation_revenue),
            'total_imbalance_charges': float(self.total_imbalance_charges),
            'net_daily_revenue': float(self.net_daily_revenue),
            'total_bid_mwh': round(self.total_bid_mwh, 2),
            'total_cleared_mwh': round(self.total_cleared_mwh, 2),
            'total_actual_mwh': round(self.total_actual_mwh, 2),
            'total_imbalance_mwh': round(self.total_imbalance_mwh, 2),
            'accuracy_percent': round(self.accuracy_percent, 2),
            'average_clearing_price': round(self.average_clearing_price, 2),
            'average_deviation': round(self.average_deviation, 2),
            'hours_over_generation': self.hours_over_generation,
            'hours_under_generation': self.hours_under_generation,
            'hours_no_penalty': self.hours_no_penalty,
            'is_finalized': self.is_finalized,
            'finalized_at': self.finalized_at.isoformat() if self.finalized_at else None,
            'created_at': self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DailySettlement':
        """Create from dictionary"""
        hourly_settlements = []
        for hs_data in data.get('hourly_settlements', []):
            hs = HourlySettlement(
                hour=hs_data['hour'],
                bid_quantity_mw=hs_data['bid_quantity_mw'],
                actual_generation_mw=hs_data['actual_generation_mw'],
                cleared_quantity_mw=hs_data['cleared_quantity_mw'],
                clearing_price_krw=hs_data['clearing_price_krw'],
                bid_price_krw=hs_data['bid_price_krw'],
            )
            hourly_settlements.append(hs)

        settlement = cls(
            settlement_id=data['settlement_id'],
            resource_id=data['resource_id'],
            trading_date=date.fromisoformat(data['trading_date']),
            bid_id=data['bid_id'],
            hourly_settlements=hourly_settlements,
        )
        settlement.calculate_totals()
        return settlement


def create_settlement_from_bid_and_actual(
    settlement_id: str,
    bid,  # DailyBid
    actual_generation: List[float],  # 24 hourly values
    clearing_prices: List[float],    # 24 hourly SMP values
) -> DailySettlement:
    """Create settlement from bid and actual generation data

    Args:
        settlement_id: Unique settlement ID
        bid: DailyBid object
        actual_generation: List of 24 actual generation values (MW)
        clearing_prices: List of 24 clearing prices (KRW/MWh)

    Returns:
        DailySettlement with calculated totals
    """
    hourly_settlements = []

    for i, hb in enumerate(bid.hourly_bids):
        hour = hb.hour
        actual_mw = actual_generation[hour - 1] if hour <= len(actual_generation) else 0.0
        smp = clearing_prices[hour - 1] if hour <= len(clearing_prices) else 0.0

        hs = HourlySettlement(
            hour=hour,
            bid_quantity_mw=hb.total_capacity_mw,
            actual_generation_mw=actual_mw,
            cleared_quantity_mw=hb.total_capacity_mw,  # Assume full clearing
            clearing_price_krw=smp,
            bid_price_krw=hb.weighted_avg_price,
        )
        hourly_settlements.append(hs)

    settlement = DailySettlement(
        settlement_id=settlement_id,
        resource_id=bid.resource_id,
        trading_date=bid.trading_date,
        bid_id=bid.bid_id,
        hourly_settlements=hourly_settlements,
    )
    settlement.calculate_totals()

    return settlement
