"""
RE-BMS KPX Client
=================

KPX API client for bid submission and market data retrieval.

Note: Actual KPX API integration requires:
- NPKI certificate for digital signature
- VPN or dedicated network connection
- Registered participant credentials

This module provides a mock client for development and testing.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import hashlib
from uuid import uuid4

from ..models.bid import DailyBid


class KPXClient:
    """KPX API Client (Abstract Base)

    Production implementation would include:
    - NPKI digital signature
    - XML/JSON message formatting
    - Certificate management
    - Retry logic with exponential backoff
    """

    # KPX API Endpoints (placeholder)
    ENDPOINTS = {
        "bid_submit": "/api/v1/market/bid",
        "bid_status": "/api/v1/market/bid/{bid_id}",
        "auction_result": "/api/v1/market/auction/{trading_date}",
        "settlement": "/api/v1/settlement/{resource_id}",
        "smp": "/api/v1/market/smp",
    }

    def __init__(
        self,
        base_url: str = "https://api.kpx.or.kr",
        cert_path: Optional[Path] = None,
        key_path: Optional[Path] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url
        self.cert_path = cert_path
        self.key_path = key_path
        self.timeout = timeout

    async def submit_bid(self, bid: DailyBid) -> Dict[str, Any]:
        """Submit bid to KPX

        Args:
            bid: DailyBid to submit

        Returns:
            Submission result with reference ID
        """
        raise NotImplementedError("Use KPXMockClient for development")

    async def get_auction_result(self, trading_date: str) -> Dict[str, Any]:
        """Get auction clearing results"""
        raise NotImplementedError("Use KPXMockClient for development")

    async def get_smp(self, date_from: str, date_to: str) -> Dict[str, Any]:
        """Get SMP data from KPX"""
        raise NotImplementedError("Use KPXMockClient for development")

    def _to_kpx_xml(self, bid: DailyBid) -> str:
        """Convert bid to KPX XML format"""
        root = ET.Element("Bid")

        # Header
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "MessageId").text = bid.bid_id
        ET.SubElement(header, "Timestamp").text = datetime.now().isoformat()
        ET.SubElement(header, "SenderId").text = bid.resource_id

        # Body
        body = ET.SubElement(root, "Body")
        ET.SubElement(body, "ResourceId").text = bid.resource_id
        ET.SubElement(body, "TradingDate").text = bid.trading_date.isoformat()
        ET.SubElement(body, "MarketType").text = bid.market_type.value.upper()

        hourly_bids_elem = ET.SubElement(body, "HourlyBids")

        for hb in bid.hourly_bids:
            hb_elem = ET.SubElement(hourly_bids_elem, "HourlyBid")
            hb_elem.set("hour", str(hb.hour))

            segments_elem = ET.SubElement(hb_elem, "Segments")

            for seg in hb.segments:
                seg_elem = ET.SubElement(segments_elem, "Segment")
                seg_elem.set("id", str(seg.segment_id))
                ET.SubElement(seg_elem, "Quantity").text = f"{seg.quantity_mw:.2f}"
                ET.SubElement(seg_elem, "Price").text = f"{seg.price_krw_mwh:.2f}"

        return ET.tostring(root, encoding="unicode")

    def _sign_xml(self, xml_content: str) -> str:
        """Apply NPKI digital signature (placeholder)"""
        # TODO: Implement actual NPKI signing
        signature = hashlib.sha256(xml_content.encode()).hexdigest()

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<SignedDocument>
    <Signature algorithm="SHA256">{signature}</Signature>
    {xml_content}
</SignedDocument>"""


class KPXMockClient(KPXClient):
    """Mock KPX Client for development and testing

    Simulates KPX API responses without actual network calls.
    """

    def __init__(self):
        super().__init__()
        self._submitted_bids: Dict[str, Dict] = {}
        self._auction_results: Dict[str, Dict] = {}

    async def submit_bid(self, bid: DailyBid) -> Dict[str, Any]:
        """Simulate bid submission

        Returns:
            Mock submission result
        """
        reference_id = f"KPX-{uuid4().hex[:8].upper()}"

        result = {
            "status": "accepted",
            "reference_id": reference_id,
            "bid_id": bid.bid_id,
            "resource_id": bid.resource_id,
            "trading_date": bid.trading_date.isoformat(),
            "submitted_at": datetime.now().isoformat(),
            "validation": {
                "monotonic_check": "passed",
                "capacity_check": "passed",
                "deadline_check": "passed",
            },
        }

        self._submitted_bids[bid.bid_id] = result
        return result

    async def get_bid_status(self, bid_id: str) -> Dict[str, Any]:
        """Get submitted bid status"""
        if bid_id in self._submitted_bids:
            return self._submitted_bids[bid_id]

        return {
            "status": "not_found",
            "bid_id": bid_id,
        }

    async def get_auction_result(self, trading_date: str) -> Dict[str, Any]:
        """Simulate auction clearing results"""
        import numpy as np

        # Generate mock clearing results
        hourly_results = []
        for hour in range(1, 25):
            hourly_results.append({
                "hour": hour,
                "clearing_price_krw": round(80 + np.sin(hour * np.pi / 12) * 20 + np.random.randn() * 5, 2),
                "total_cleared_mw": round(500 + np.random.randn() * 50, 2),
                "total_bids": round(550 + np.random.randn() * 50, 2),
            })

        return {
            "trading_date": trading_date,
            "market_type": "day_ahead",
            "status": "cleared",
            "cleared_at": datetime.now().isoformat(),
            "hourly_results": hourly_results,
            "summary": {
                "average_clearing_price": round(sum(h["clearing_price_krw"] for h in hourly_results) / 24, 2),
                "total_cleared_mwh": round(sum(h["total_cleared_mw"] for h in hourly_results), 2),
            },
        }

    async def get_smp(self, date_from: str, date_to: str) -> Dict[str, Any]:
        """Get historical SMP data"""
        import numpy as np
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(date_from)
        end = datetime.fromisoformat(date_to)

        smp_data = []
        current = start

        while current <= end:
            for hour in range(24):
                smp_data.append({
                    "timestamp": (current + timedelta(hours=hour)).isoformat(),
                    "hour": hour + 1,
                    "smp_mainland": round(90 + np.sin(hour * np.pi / 12) * 15 + np.random.randn() * 3, 2),
                    "smp_jeju": round(95 + np.sin(hour * np.pi / 12) * 15 + np.random.randn() * 3, 2),
                })
            current += timedelta(days=1)

        return {
            "date_from": date_from,
            "date_to": date_to,
            "count": len(smp_data),
            "data": smp_data,
        }

    async def simulate_settlement(
        self,
        bid: DailyBid,
        actual_generation: List[float],
    ) -> Dict[str, Any]:
        """Simulate settlement calculation

        Args:
            bid: Submitted bid
            actual_generation: 24-hour actual generation (MW)

        Returns:
            Settlement summary
        """
        from decimal import Decimal

        # Get clearing prices
        auction = await self.get_auction_result(bid.trading_date.isoformat())
        clearing_prices = [h["clearing_price_krw"] for h in auction["hourly_results"]]

        total_revenue = Decimal(0)
        total_penalty = Decimal(0)

        hourly_settlements = []

        for i, hb in enumerate(bid.hourly_bids):
            hour = hb.hour
            actual = actual_generation[hour - 1] if hour <= len(actual_generation) else 0
            cleared = hb.total_capacity_mw
            smp = clearing_prices[hour - 1]

            revenue = Decimal(str(actual * smp))
            imbalance = actual - cleared

            # Penalty calculation (Jeju pilot: ±12% tolerance)
            tolerance = cleared * 0.12
            penalty = Decimal(0)

            if abs(imbalance) > tolerance:
                excess = abs(imbalance) - tolerance
                if imbalance > 0:  # Over-generation
                    penalty = Decimal(str(excess * smp * 0.2))  # 20% penalty
                else:  # Under-generation
                    penalty = Decimal(str(excess * smp * 0.2))  # 20% penalty

            total_revenue += revenue
            total_penalty += penalty

            hourly_settlements.append({
                "hour": hour,
                "cleared_mw": cleared,
                "actual_mw": actual,
                "imbalance_mw": imbalance,
                "clearing_price": smp,
                "revenue_krw": float(revenue),
                "penalty_krw": float(penalty),
                "net_krw": float(revenue - penalty),
            })

        return {
            "bid_id": bid.bid_id,
            "trading_date": bid.trading_date.isoformat(),
            "total_revenue_krw": float(total_revenue),
            "total_penalty_krw": float(total_penalty),
            "net_revenue_krw": float(total_revenue - total_penalty),
            "hourly_settlements": hourly_settlements,
        }
