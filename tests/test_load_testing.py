"""
부하 테스트 모듈 테스트 (Task 21)
=================================
Load testing 헬퍼 함수 테스트
"""

import pytest
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 테스트 데이터 생성 테스트
# ============================================================================

class TestDataGeneration:
    """테스트 데이터 생성 테스트"""

    def test_generate_prediction_request(self):
        """예측 요청 생성"""
        from tests.load_testing import generate_prediction_request

        request = generate_prediction_request()

        assert 'location' in request
        assert 'horizons' in request
        assert 'model_type' in request
        assert isinstance(request['horizons'], list)

    def test_generate_historical_request(self):
        """과거 데이터 요청 생성"""
        from tests.load_testing import generate_historical_request

        request = generate_historical_request()

        assert 'location' in request
        assert 'start_date' in request
        assert 'end_date' in request
        assert 'resolution' in request

    def test_generate_forecast_request(self):
        """예보 요청 생성"""
        from tests.load_testing import generate_forecast_request

        request = generate_forecast_request()

        assert 'location' in request
        assert 'hours_ahead' in request
        assert 'include_weather' in request

    def test_request_randomness(self):
        """요청 랜덤성 확인"""
        from tests.load_testing import generate_prediction_request

        # 여러 요청 생성
        requests = [generate_prediction_request() for _ in range(10)]

        # 최소한 일부는 다른 값을 가져야 함
        horizons = [str(r['horizons']) for r in requests]
        assert len(set(horizons)) > 1, "Requests should have some variation"


# ============================================================================
# 성능 기준 테스트
# ============================================================================

class TestPerformanceCriteria:
    """성능 기준 테스트"""

    def test_criteria_defaults(self):
        """기본 기준값"""
        from tests.load_testing import PerformanceCriteria

        assert PerformanceCriteria.MAX_RESPONSE_TIME_MS > 0
        assert PerformanceCriteria.MAX_P90_MS > 0
        assert PerformanceCriteria.MAX_FAILURE_RATE >= 0
        assert PerformanceCriteria.MIN_RPS > 0

    def test_validate_passing(self):
        """성공하는 검증"""
        from tests.load_testing import PerformanceCriteria

        analysis = {
            "summary": {"failure_rate": 0.5},
            "response_times": {
                "avg_response_time_ms": 100,
                "p90_ms": 150
            },
            "throughput": {"requests_per_second": 100}
        }

        results = PerformanceCriteria.validate(analysis)

        assert results['avg_response_time'] == True
        assert results['p90_response_time'] == True
        assert results['failure_rate'] == True
        assert results['throughput'] == True

    def test_validate_failing(self):
        """실패하는 검증"""
        from tests.load_testing import PerformanceCriteria

        analysis = {
            "summary": {"failure_rate": 10.0},  # 높은 실패율
            "response_times": {
                "avg_response_time_ms": 1000,  # 높은 응답 시간
                "p90_ms": 500
            },
            "throughput": {"requests_per_second": 10}  # 낮은 처리량
        }

        results = PerformanceCriteria.validate(analysis)

        assert results['avg_response_time'] == False
        assert results['p90_response_time'] == False
        assert results['failure_rate'] == False
        assert results['throughput'] == False

    def test_get_summary(self):
        """검증 요약"""
        from tests.load_testing import PerformanceCriteria

        results = {
            'avg_response_time': True,
            'p90_response_time': True,
            'failure_rate': False,
            'throughput': True
        }

        summary = PerformanceCriteria.get_summary(results)

        assert '3/4' in summary
        assert '✓' in summary
        assert '✗' in summary


# ============================================================================
# 분석기 테스트
# ============================================================================

class TestLoadTestAnalyzer:
    """부하 테스트 분석기 테스트"""

    def test_analyzer_creation(self):
        """분석기 생성"""
        from tests.load_testing import LoadTestAnalyzer

        analyzer = LoadTestAnalyzer()
        assert analyzer is not None

    def test_analyze_empty(self):
        """빈 분석"""
        from tests.load_testing import LoadTestAnalyzer

        analyzer = LoadTestAnalyzer()
        result = analyzer.analyze_stats()

        assert result == {}

    def test_analyze_with_csv(self):
        """CSV 파일 분석"""
        import pandas as pd
        from tests.load_testing import LoadTestAnalyzer

        # 테스트용 CSV 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Type,Name,Request Count,Failure Count,Average Response Time,Min Response Time,Max Response Time,50%,90%,Requests/s\n")
            f.write("GET,/health,100,2,50.5,10,200,45,80,25.5\n")
            f.write("POST,/predict,500,10,100.2,20,500,80,150,50.0\n")
            f.write(",Aggregated,600,12,90.0,10,500,70,130,75.5\n")
            csv_path = f.name

        analyzer = LoadTestAnalyzer(stats_file=csv_path)
        result = analyzer.analyze_stats()

        assert 'summary' in result
        assert result['summary']['total_requests'] > 0
        assert result['summary']['total_failures'] >= 0

        assert 'response_times' in result
        assert 'throughput' in result
        assert 'by_endpoint' in result

        # 정리
        Path(csv_path).unlink()

    def test_generate_report(self):
        """리포트 생성"""
        import pandas as pd
        from tests.load_testing import LoadTestAnalyzer

        # 테스트용 CSV 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Type,Name,Request Count,Failure Count,Average Response Time,Min Response Time,Max Response Time,50%,90%,Requests/s\n")
            f.write("GET,/health,100,2,50.5,10,200,45,80,25.5\n")
            csv_path = f.name

        analyzer = LoadTestAnalyzer(stats_file=csv_path)
        report = analyzer.generate_report()

        assert "Load Test Report" in report
        assert "Summary" in report
        assert "Response Times" in report

        # 정리
        Path(csv_path).unlink()


# ============================================================================
# Locust 클래스 테스트 (Locust 미설치 시 스킵)
# ============================================================================

class TestLocustClasses:
    """Locust 클래스 테스트"""

    def test_locust_availability(self):
        """Locust 가용성 확인"""
        from tests.load_testing import LOCUST_AVAILABLE

        # Locust 설치 여부에 따라 값이 달라짐
        assert isinstance(LOCUST_AVAILABLE, bool)

    @pytest.mark.skipif(
        not __import__('tests.load_testing', fromlist=['LOCUST_AVAILABLE']).LOCUST_AVAILABLE,
        reason="Locust not installed"
    )
    def test_user_classes_exist(self):
        """사용자 클래스 존재"""
        from tests.load_testing import PowerDemandAPIUser, HeavyUser, LightUser

        assert PowerDemandAPIUser is not None
        assert HeavyUser is not None
        assert LightUser is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
