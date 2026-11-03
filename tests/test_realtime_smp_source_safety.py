"""Guard against reporting unverified crawler data as Jeju SMP."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from api import realtime_api


def test_realtime_jeju_does_not_use_unverified_crawler(monkeypatch):
    """A crawler demand value or estimate must not become a Jeju SMP price."""
    client = realtime_api.RealtimeAPIClient(api_key="test-key")
    client.use_crawlers = True
    monkeypatch.setattr(client, "_get_jeju_smp_via_api", AsyncMock(return_value=None))
    crawler = Mock(return_value=realtime_api.SMPData(
        trade_day="20261103", trade_hour=4, smp_land=629.49,
        smp_jeju=616.90, timestamp=datetime.now(),
    ))
    monkeypatch.setattr(client, "_get_smp_via_crawler", crawler)
    monkeypatch.setattr(realtime_api, "KPX_SMP_API", "http://127.0.0.1:1")

    result = asyncio.run(client.get_smp_realtime(area_code="9"))

    assert result is None
    crawler.assert_not_called()
