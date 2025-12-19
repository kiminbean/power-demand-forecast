"""
RE-BMS v5.0 - Renewable Energy Bidding Management System
=========================================================

Mobile-first bidding system for KPX Day-Ahead Market (DAM) and Real-Time Market (RTM).

Features:
- 10-segment bidding with monotonic price constraint
- AI-powered bid optimization using SMP predictor
- KPX API integration (XML/JSON)
- Command Center Dashboard
- Settlement and penalty management

Usage:
    # API Server
    uvicorn src.rebms.api.main:app --port 8506

    # Dashboard
    streamlit run src/rebms/dashboard/app.py --server.port 8507
"""

__version__ = "5.0.0"
__author__ = "Power Demand Forecast Team"

from .models.bid import (
    BidSegment,
    HourlyBid,
    DailyBid,
    MarketType,
    BidStatus,
)
from .models.resource import (
    RenewableResource,
    ResourcePortfolio,
    ResourceType,
)
from .models.settlement import (
    HourlySettlement,
    DailySettlement,
    PenaltyType,
)

__all__ = [
    # Version
    "__version__",
    # Bid models
    "BidSegment",
    "HourlyBid",
    "DailyBid",
    "MarketType",
    "BidStatus",
    # Resource models
    "RenewableResource",
    "ResourcePortfolio",
    "ResourceType",
    # Settlement models
    "HourlySettlement",
    "DailySettlement",
    "PenaltyType",
]
