"""
RE-BMS Bid Validator
====================

Validation logic for 10-segment bids and market constraints.

Validation Rules:
1. Monotonic Constraint: Prices must be non-decreasing across segments
2. Capacity Constraint: Total bid ≤ installed capacity
3. Deadline Constraint: DAM by D-1 10:00, RTM 15 min before
4. Segment Constraint: Maximum 10 segments per hour
5. Price Cap: Prices must be within market cap
"""

from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple, Dict, Any

from ..models.bid import BidSegment, HourlyBid, DailyBid, MarketType


class ValidationError(Exception):
    """Bid validation error with details"""

    def __init__(self, message: str, errors: List[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': self.message,
            'errors': self.errors,
        }


# Market Configuration
DAM_DEADLINE_HOUR = 10   # D-1 10:00
RTM_LEAD_TIME_MIN = 15   # 15 minutes before interval
MAX_SEGMENTS = 10
MAX_PRICE_KRW_MWH = 500  # Price cap (configurable)
MIN_PRICE_KRW_MWH = 0


def validate_monotonic_constraint(segments: List[BidSegment]) -> Tuple[bool, Optional[str]]:
    """Validate monotonic price increase constraint

    Args:
        segments: List of bid segments

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not segments:
        return True, None

    if len(segments) > MAX_SEGMENTS:
        return False, f"Maximum {MAX_SEGMENTS} segments allowed, got {len(segments)}"

    # Get non-zero quantity segments
    active_segments = [s for s in segments if s.quantity_mw > 0]

    for i in range(1, len(active_segments)):
        prev_price = active_segments[i-1].price_krw_mwh
        curr_price = active_segments[i].price_krw_mwh

        if curr_price < prev_price:
            return False, (
                f"Monotonic constraint violated: segment {active_segments[i].segment_id} "
                f"price ({curr_price:.2f}) < segment {active_segments[i-1].segment_id} "
                f"price ({prev_price:.2f})"
            )

    return True, None


def validate_deadline(
    market_type: MarketType,
    trading_date: date,
    current_time: Optional[datetime] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate bid submission deadline

    Args:
        market_type: DAM or RTM
        trading_date: Trading date
        current_time: Current time (default: now)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if current_time is None:
        current_time = datetime.now()

    if market_type == MarketType.DAM:
        # DAM deadline: D-1 10:00
        deadline = datetime.combine(
            trading_date - timedelta(days=1),
            time(DAM_DEADLINE_HOUR, 0)
        )

        if current_time > deadline:
            return False, (
                f"DAM deadline passed: {deadline.strftime('%Y-%m-%d %H:%M')}. "
                f"Current time: {current_time.strftime('%Y-%m-%d %H:%M')}"
            )

    elif market_type == MarketType.RTM:
        # RTM deadline: 15 minutes before interval
        interval_start = datetime.combine(trading_date, time(0, 0))
        deadline = interval_start - timedelta(minutes=RTM_LEAD_TIME_MIN)

        if current_time > deadline:
            return False, (
                f"RTM deadline passed. Must submit at least {RTM_LEAD_TIME_MIN} "
                f"minutes before interval"
            )

    return True, None


def validate_capacity(
    segments: List[BidSegment],
    installed_capacity_mw: float,
) -> Tuple[bool, Optional[str]]:
    """Validate total bid capacity against installed capacity

    Args:
        segments: List of bid segments
        installed_capacity_mw: Installed capacity in MW

    Returns:
        Tuple of (is_valid, error_message)
    """
    total_bid_mw = sum(s.quantity_mw for s in segments)

    if total_bid_mw > installed_capacity_mw:
        return False, (
            f"Total bid ({total_bid_mw:.2f} MW) exceeds "
            f"installed capacity ({installed_capacity_mw:.2f} MW)"
        )

    return True, None


def validate_price_range(segments: List[BidSegment]) -> Tuple[bool, Optional[str]]:
    """Validate prices are within market cap

    Args:
        segments: List of bid segments

    Returns:
        Tuple of (is_valid, error_message)
    """
    for segment in segments:
        if segment.price_krw_mwh < MIN_PRICE_KRW_MWH:
            return False, (
                f"Segment {segment.segment_id} price ({segment.price_krw_mwh:.2f}) "
                f"below minimum ({MIN_PRICE_KRW_MWH})"
            )

        if segment.price_krw_mwh > MAX_PRICE_KRW_MWH:
            return False, (
                f"Segment {segment.segment_id} price ({segment.price_krw_mwh:.2f}) "
                f"exceeds market cap ({MAX_PRICE_KRW_MWH})"
            )

    return True, None


class BidValidator:
    """Comprehensive bid validator

    Validates all aspects of 10-segment bids:
    - Monotonic price constraint
    - Deadline constraints
    - Capacity limits
    - Price ranges

    Example:
        >>> validator = BidValidator()
        >>> is_valid, errors = validator.validate_daily_bid(bid)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """

    def __init__(
        self,
        max_price_cap: float = MAX_PRICE_KRW_MWH,
        check_deadline: bool = True,
    ):
        """Initialize validator

        Args:
            max_price_cap: Maximum allowed price
            check_deadline: Whether to check deadline constraints
        """
        self.max_price_cap = max_price_cap
        self.check_deadline = check_deadline

    def validate_segments(
        self,
        segments: List[BidSegment],
        installed_capacity_mw: Optional[float] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate segment list

        Args:
            segments: List of segments to validate
            installed_capacity_mw: Optional capacity limit

        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []

        # Monotonic constraint
        is_valid, error = validate_monotonic_constraint(segments)
        if not is_valid:
            errors.append({'type': 'monotonic', 'message': error})

        # Price range
        is_valid, error = validate_price_range(segments)
        if not is_valid:
            errors.append({'type': 'price_range', 'message': error})

        # Capacity constraint
        if installed_capacity_mw is not None:
            is_valid, error = validate_capacity(segments, installed_capacity_mw)
            if not is_valid:
                errors.append({'type': 'capacity', 'message': error})

        return len(errors) == 0, errors

    def validate_hourly_bid(
        self,
        hourly_bid: HourlyBid,
        installed_capacity_mw: Optional[float] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate hourly bid

        Args:
            hourly_bid: HourlyBid to validate
            installed_capacity_mw: Optional capacity limit

        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []

        # Validate hour range
        if not 1 <= hourly_bid.hour <= 24:
            errors.append({
                'type': 'hour_range',
                'message': f"Hour must be 1-24, got {hourly_bid.hour}",
                'hour': hourly_bid.hour,
            })

        # Validate segments
        is_valid, segment_errors = self.validate_segments(
            hourly_bid.segments,
            installed_capacity_mw,
        )
        for error in segment_errors:
            error['hour'] = hourly_bid.hour
            errors.append(error)

        return len(errors) == 0, errors

    def validate_daily_bid(
        self,
        daily_bid: DailyBid,
        installed_capacity_mw: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate complete daily bid

        Args:
            daily_bid: DailyBid to validate
            installed_capacity_mw: Optional capacity limit
            current_time: Current time for deadline check

        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []

        # Check deadline
        if self.check_deadline:
            is_valid, error = validate_deadline(
                daily_bid.market_type,
                daily_bid.trading_date,
                current_time,
            )
            if not is_valid:
                errors.append({'type': 'deadline', 'message': error})

        # Validate all hourly bids
        for hourly_bid in daily_bid.hourly_bids:
            is_valid, hourly_errors = self.validate_hourly_bid(
                hourly_bid,
                installed_capacity_mw,
            )
            errors.extend(hourly_errors)

        # Check for duplicate hours
        hours = [hb.hour for hb in daily_bid.hourly_bids]
        if len(hours) != len(set(hours)):
            duplicates = [h for h in hours if hours.count(h) > 1]
            errors.append({
                'type': 'duplicate_hours',
                'message': f"Duplicate hours found: {set(duplicates)}",
            })

        return len(errors) == 0, errors

    def validate_and_raise(
        self,
        daily_bid: DailyBid,
        installed_capacity_mw: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ):
        """Validate and raise ValidationError if invalid

        Args:
            daily_bid: DailyBid to validate
            installed_capacity_mw: Optional capacity limit
            current_time: Current time for deadline check

        Raises:
            ValidationError: If validation fails
        """
        is_valid, errors = self.validate_daily_bid(
            daily_bid,
            installed_capacity_mw,
            current_time,
        )

        if not is_valid:
            raise ValidationError(
                f"Bid validation failed with {len(errors)} error(s)",
                errors,
            )

    def quick_validate_monotonic(self, segments: List[BidSegment]) -> bool:
        """Quick monotonic check without error details"""
        is_valid, _ = validate_monotonic_constraint(segments)
        return is_valid

    def get_validation_summary(
        self,
        daily_bid: DailyBid,
        installed_capacity_mw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get validation summary for UI display

        Args:
            daily_bid: DailyBid to validate
            installed_capacity_mw: Optional capacity limit

        Returns:
            Summary dict with counts and status
        """
        is_valid, errors = self.validate_daily_bid(
            daily_bid,
            installed_capacity_mw,
            current_time=None,  # Skip deadline for UI
        )

        error_types = {}
        for error in errors:
            error_type = error.get('type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'is_valid': is_valid,
            'total_errors': len(errors),
            'error_counts': error_types,
            'hours_with_errors': list(set(
                e.get('hour') for e in errors if e.get('hour')
            )),
            'bid_id': daily_bid.bid_id,
        }
