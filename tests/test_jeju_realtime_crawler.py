"""
제주 실시간 전력수급 크롤러 테스트
==================================

JejuRealtimeCrawler 및 JejuRealtimeData 클래스 테스트
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from tools.crawlers.jeju_realtime_crawler import (
    JejuRealtimeData,
    JejuRealtimeCrawler,
)


# ============================================================================
# JejuRealtimeData Tests
# ============================================================================

class TestJejuRealtimeData:
    """JejuRealtimeData 데이터클래스 테스트"""

    def test_data_creation(self):
        """기본 데이터 생성 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1435.0,
            current_demand=625.0,
            supply_reserve=810.0,
            operation_reserve=477.0,
        )
        assert data.timestamp == "2025-12-18 10:30"
        assert data.supply_capacity == 1435.0
        assert data.current_demand == 625.0
        assert data.supply_reserve == 810.0
        assert data.operation_reserve == 477.0

    def test_reserve_rate_calculation(self):
        """예비율 자동 계산 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1400.0,
            current_demand=700.0,
            supply_reserve=700.0,
            operation_reserve=350.0,
        )
        # 예비율 = 공급예비력 / 현재수요 * 100
        expected_rate = (700.0 / 700.0) * 100
        assert data.reserve_rate == expected_rate

    def test_reserve_rate_zero_demand(self):
        """수요가 0일 때 예비율 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1400.0,
            current_demand=0.0,
            supply_reserve=700.0,
            operation_reserve=350.0,
        )
        assert data.reserve_rate == 0.0

    def test_utilization_rate_calculation(self):
        """이용률 계산 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1400.0,
            current_demand=700.0,
            supply_reserve=700.0,
            operation_reserve=350.0,
        )
        # 이용률 = 현재수요 / 공급능력 * 100
        expected_rate = (700.0 / 1400.0) * 100
        assert data.utilization_rate == expected_rate

    def test_utilization_rate_zero_capacity(self):
        """공급능력이 0일 때 이용률 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=0.0,
            current_demand=700.0,
            supply_reserve=700.0,
            operation_reserve=350.0,
        )
        assert data.utilization_rate == 0.0

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1435.0,
            current_demand=625.0,
            supply_reserve=810.0,
            operation_reserve=477.0,
        )
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d['timestamp'] == "2025-12-18 10:30"
        assert d['supply_capacity'] == 1435.0
        assert 'reserve_rate' in d
        assert 'utilization_rate' in d
        assert 'fetched_at' in d

    def test_fetched_at_auto_set(self):
        """fetched_at 자동 설정 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1435.0,
            current_demand=625.0,
            supply_reserve=810.0,
            operation_reserve=477.0,
        )
        assert data.fetched_at != ""
        # 날짜 형식 확인
        datetime.strptime(data.fetched_at, "%Y-%m-%d %H:%M:%S")

    def test_source_default(self):
        """source 기본값 테스트"""
        data = JejuRealtimeData(
            timestamp="2025-12-18 10:30",
            supply_capacity=1435.0,
            current_demand=625.0,
            supply_reserve=810.0,
            operation_reserve=477.0,
        )
        assert data.source == "kpx.or.kr"


# ============================================================================
# JejuRealtimeCrawler Tests
# ============================================================================

class TestJejuRealtimeCrawler:
    """JejuRealtimeCrawler 클래스 테스트"""

    def test_crawler_init(self):
        """크롤러 초기화 테스트"""
        crawler = JejuRealtimeCrawler()
        assert crawler.timeout == 30
        assert crawler.session is not None
        crawler.close()

    def test_crawler_init_custom_timeout(self):
        """크롤러 커스텀 타임아웃 초기화 테스트"""
        crawler = JejuRealtimeCrawler(timeout=60)
        assert crawler.timeout == 60
        crawler.close()

    def test_context_manager(self):
        """컨텍스트 매니저 테스트"""
        with JejuRealtimeCrawler() as crawler:
            assert crawler.session is not None

    def test_session_headers(self):
        """HTTP 세션 헤더 테스트"""
        crawler = JejuRealtimeCrawler()
        assert 'User-Agent' in crawler.session.headers
        crawler.close()


class TestJejuRealtimeCrawlerDataFetching:
    """데이터 수집 테스트"""

    @pytest.fixture
    def mock_html_response(self):
        """테스트용 HTML 응답"""
        return """
        <html>
        <body>
            <div>공급능력: 1,435 MW</div>
            <div>현재부하: 625 MW</div>
            <div>공급예비력: 810 MW</div>
            <div>운영예비력: 477 MW</div>
            <div>2025.12.18(수) 10:30</div>
        </body>
        </html>
        """

    def test_fetch_realtime_success(self, mock_html_response):
        """실시간 데이터 수집 성공 테스트"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_html_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            data = crawler.fetch_realtime()

            assert data is not None
            assert data.supply_capacity == 1435.0
            assert data.current_demand == 625.0
            assert data.supply_reserve == 810.0
            assert data.operation_reserve == 477.0

        crawler.close()

    def test_fetch_realtime_network_error(self):
        """네트워크 오류 시 None 반환"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler.session, 'get') as mock_get:
            import requests
            mock_get.side_effect = requests.RequestException("Network error")

            data = crawler.fetch_realtime()
            assert data is None

        crawler.close()

    def test_fetch_realtime_insufficient_data(self):
        """데이터 부족 시 None 반환"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.text = "<html>1,000 MW</html>"  # 1개만 있음
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            data = crawler.fetch_realtime()
            assert data is None

        crawler.close()


class TestJejuRealtimeCrawlerTimestamp:
    """타임스탬프 추출 테스트"""

    def test_extract_timestamp_standard_format(self):
        """표준 형식 타임스탬프 추출"""
        crawler = JejuRealtimeCrawler()

        html = "2025.12.18(수) 10:30 기준"
        timestamp = crawler._extract_timestamp(html)

        assert timestamp == "2025-12-18 10:30"
        crawler.close()

    def test_extract_timestamp_fallback(self):
        """타임스탬프 없을 때 현재 시간 사용"""
        crawler = JejuRealtimeCrawler()

        html = "<html>No timestamp here</html>"
        timestamp = crawler._extract_timestamp(html)

        # 현재 시간 기반 (형식만 확인)
        assert len(timestamp) == 16  # "YYYY-MM-DD HH:MM"
        crawler.close()


class TestJejuRealtimeCrawlerStatus:
    """상태 조회 테스트"""

    def test_get_status_safe(self):
        """정상 상태 테스트"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler, 'fetch_realtime') as mock_fetch:
            mock_data = JejuRealtimeData(
                timestamp="2025-12-18 10:30",
                supply_capacity=1400.0,
                current_demand=600.0,
                supply_reserve=800.0,  # 예비율 133%
                operation_reserve=400.0,
            )
            mock_fetch.return_value = mock_data

            status = crawler.get_status()

            assert status['status'] == 'safe'
            assert status['status_text'] == '정상'

        crawler.close()

    def test_get_status_warning(self):
        """주의 상태 테스트"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler, 'fetch_realtime') as mock_fetch:
            mock_data = JejuRealtimeData(
                timestamp="2025-12-18 10:30",
                supply_capacity=1000.0,
                current_demand=900.0,
                supply_reserve=70.0,  # 예비율 7.8%
                operation_reserve=35.0,
            )
            mock_fetch.return_value = mock_data

            status = crawler.get_status()

            assert status['status'] == 'warning'
            assert status['status_text'] == '주의'

        crawler.close()

    def test_get_status_danger(self):
        """위험 상태 테스트"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler, 'fetch_realtime') as mock_fetch:
            mock_data = JejuRealtimeData(
                timestamp="2025-12-18 10:30",
                supply_capacity=1000.0,
                current_demand=950.0,
                supply_reserve=40.0,  # 예비율 4.2%
                operation_reserve=20.0,
            )
            mock_fetch.return_value = mock_data

            status = crawler.get_status()

            assert status['status'] == 'danger'
            assert status['status_text'] == '위험'

        crawler.close()

    def test_get_status_error(self):
        """에러 상태 테스트"""
        crawler = JejuRealtimeCrawler()

        with patch.object(crawler, 'fetch_realtime') as mock_fetch:
            mock_fetch.return_value = None

            status = crawler.get_status()

            assert status['status'] == 'error'

        crawler.close()


# ============================================================================
# Integration Tests
# ============================================================================

class TestJejuRealtimeCrawlerIntegration:
    """실제 KPX 연동 통합 테스트 (네트워크 필요)"""

    @pytest.mark.skipif(
        not pytest.importorskip("requests"),
        reason="requests not available"
    )
    def test_fetch_realtime_live(self):
        """실제 KPX 실시간 데이터 조회"""
        crawler = JejuRealtimeCrawler(timeout=10)

        try:
            data = crawler.fetch_realtime()

            if data:
                # 데이터가 있으면 유효성 검사
                assert data.supply_capacity > 0
                assert data.current_demand >= 0
                assert data.supply_reserve >= 0
                assert 0 <= data.reserve_rate <= 500  # 합리적인 범위
                assert 0 <= data.utilization_rate <= 100
            # 네트워크 문제로 None일 수도 있음 (테스트 통과)
        finally:
            crawler.close()

    def test_data_consistency(self):
        """데이터 일관성 테스트"""
        crawler = JejuRealtimeCrawler(timeout=10)

        try:
            data = crawler.fetch_realtime()

            if data:
                # 공급예비력 = 공급능력 - 현재수요 (대략)
                expected_reserve = data.supply_capacity - data.current_demand
                # 10% 오차 허용
                assert abs(data.supply_reserve - expected_reserve) < expected_reserve * 0.1
        finally:
            crawler.close()
