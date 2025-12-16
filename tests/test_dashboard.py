"""
Dashboard 테스트 (Task 14)
==========================
"""

import pytest
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock streamlit before importing dashboard
sys.modules['streamlit'] = MagicMock()

from src.dashboard.app import (
    Dashboard,
    DashboardConfig,
    DataFetcher,
    MockDataGenerator,
    ChartFactory,
    DashboardComponents,
    get_config
)


# ============================================================================
# DashboardConfig Tests
# ============================================================================

class TestDashboardConfig:
    """대시보드 설정 테스트"""

    def test_default_config(self):
        """기본 설정"""
        config = DashboardConfig()
        assert config.api_url == "http://localhost:8000"
        assert config.refresh_interval == 60
        assert config.default_location == "jeju"
        assert config.theme == "light"
        assert config.chart_height == 400
        assert config.max_history_days == 365

    def test_custom_config(self):
        """커스텀 설정"""
        config = DashboardConfig(
            api_url="http://custom:9000",
            refresh_interval=30,
            theme="dark"
        )
        assert config.api_url == "http://custom:9000"
        assert config.refresh_interval == 30
        assert config.theme == "dark"

    def test_get_config(self):
        """설정 로드 함수"""
        config = get_config()
        assert isinstance(config, DashboardConfig)


# ============================================================================
# DataFetcher Tests
# ============================================================================

class TestDataFetcher:
    """데이터 수집기 테스트"""

    @pytest.fixture
    def fetcher(self):
        """데이터 수집기 픽스처"""
        return DataFetcher("http://localhost:8000")

    def test_init(self, fetcher):
        """초기화"""
        assert fetcher.api_url == "http://localhost:8000"
        assert fetcher._cache == {}

    @patch("src.dashboard.app.requests.post")
    def test_get_predictions_success(self, mock_post, fetcher):
        """예측 데이터 조회 성공"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "predictions": [{"horizon": "1h", "prediction": 1000.0}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = fetcher.get_predictions()
        assert result is not None
        assert "predictions" in result

    @patch("src.dashboard.app.requests.post")
    def test_get_predictions_error(self, mock_post, fetcher):
        """예측 데이터 조회 실패"""
        mock_post.side_effect = Exception("Connection error")

        result = fetcher.get_predictions()
        assert result is None

    @patch("src.dashboard.app.requests.get")
    def test_get_historical_data_success(self, mock_get, fetcher):
        """과거 데이터 조회 성공"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"timestamp": "2025-01-01T00:00:00", "demand": 1000}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetcher.get_historical_data(
            location="jeju",
            start_date="2025-01-01",
            end_date="2025-01-02"
        )
        assert result is not None
        assert "data" in result

    @patch("src.dashboard.app.requests.get")
    def test_get_historical_data_error(self, mock_get, fetcher):
        """과거 데이터 조회 실패"""
        mock_get.side_effect = Exception("Connection error")

        result = fetcher.get_historical_data(
            location="jeju",
            start_date="2025-01-01",
            end_date="2025-01-02"
        )
        assert result is None

    @patch("src.dashboard.app.requests.get")
    def test_get_health_success(self, mock_get, fetcher):
        """상태 확인 성공"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy", "uptime": 3600}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetcher.get_health()
        assert result is not None
        assert result["status"] == "healthy"

    @patch("src.dashboard.app.requests.get")
    def test_get_health_error(self, mock_get, fetcher):
        """상태 확인 실패"""
        mock_get.side_effect = Exception("Connection error")

        result = fetcher.get_health()
        assert result is None

    @patch("src.dashboard.app.requests.get")
    def test_get_models_success(self, mock_get, fetcher):
        """모델 목록 성공"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "lstm_v1", "type": "lstm"}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetcher.get_models()
        assert result is not None
        assert "models" in result

    @patch("src.dashboard.app.requests.get")
    def test_get_metrics_success(self, mock_get, fetcher):
        """메트릭 조회 성공"""
        mock_response = Mock()
        mock_response.json.return_value = {"total_predictions": 100}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetcher.get_metrics()
        assert result is not None
        assert result["total_predictions"] == 100


# ============================================================================
# MockDataGenerator Tests
# ============================================================================

class TestMockDataGenerator:
    """모의 데이터 생성기 테스트"""

    def test_generate_predictions_default(self):
        """기본 예측 생성"""
        result = MockDataGenerator.generate_predictions()

        assert "request_id" in result
        assert "location" in result
        assert "predictions" in result
        assert len(result["predictions"]) == 3  # default horizons

    def test_generate_predictions_custom_horizons(self):
        """커스텀 시간대 예측 생성"""
        result = MockDataGenerator.generate_predictions(["1h", "6h", "12h", "24h", "48h"])

        assert len(result["predictions"]) == 5

    def test_predictions_have_required_fields(self):
        """예측 필수 필드"""
        result = MockDataGenerator.generate_predictions(["1h"])
        pred = result["predictions"][0]

        assert "timestamp" in pred
        assert "horizon" in pred
        assert "prediction" in pred
        assert "lower_bound" in pred
        assert "upper_bound" in pred
        assert "confidence" in pred

    def test_generate_historical_data_hourly(self):
        """시간별 과거 데이터 생성"""
        start = date(2025, 1, 1)
        end = date(2025, 1, 2)

        result = MockDataGenerator.generate_historical_data(start, end, "hourly")

        assert "data" in result
        assert result["count"] >= 24
        assert result["location"] == "jeju"

    def test_generate_historical_data_daily(self):
        """일별 과거 데이터 생성"""
        start = date(2025, 1, 1)
        end = date(2025, 1, 7)

        result = MockDataGenerator.generate_historical_data(start, end, "daily")

        assert result["count"] == 7

    def test_historical_data_has_required_fields(self):
        """과거 데이터 필수 필드"""
        start = date(2025, 1, 1)
        end = date(2025, 1, 1)

        result = MockDataGenerator.generate_historical_data(start, end, "daily")
        record = result["data"][0]

        assert "timestamp" in record
        assert "demand" in record
        assert "temperature" in record
        assert "humidity" in record

    def test_historical_data_realistic_values(self):
        """과거 데이터 현실적인 값"""
        start = date(2025, 1, 1)
        end = date(2025, 1, 7)

        result = MockDataGenerator.generate_historical_data(start, end)

        for record in result["data"]:
            # 수요는 양수
            assert record["demand"] > 0
            # 기온은 -30~50 범위
            assert -30 < record["temperature"] < 50
            # 습도는 0~100
            assert 0 <= record["humidity"] <= 100


# ============================================================================
# ChartFactory Tests
# ============================================================================

class TestChartFactory:
    """차트 팩토리 테스트"""

    def test_create_prediction_chart_empty(self):
        """빈 예측 차트"""
        fig = ChartFactory.create_prediction_chart([])
        assert fig is not None

    def test_create_prediction_chart_with_data(self):
        """데이터 있는 예측 차트"""
        predictions = [
            {
                "timestamp": "2025-01-01T12:00:00",
                "prediction": 1000.0,
                "lower_bound": 950.0,
                "upper_bound": 1050.0
            }
        ]
        fig = ChartFactory.create_prediction_chart(predictions)

        assert fig is not None
        assert len(fig.data) >= 1  # at least prediction line

    def test_create_historical_chart_empty(self):
        """빈 과거 데이터 차트"""
        fig = ChartFactory.create_historical_chart([])
        assert fig is not None

    def test_create_historical_chart_with_data(self):
        """데이터 있는 과거 데이터 차트"""
        data = [
            {
                "timestamp": "2025-01-01T12:00:00",
                "demand": 1000.0,
                "temperature": 20.0
            }
        ]
        fig = ChartFactory.create_historical_chart(data)

        assert fig is not None
        assert len(fig.data) >= 1

    def test_create_demand_pattern_chart_empty(self):
        """빈 수요 패턴 차트"""
        fig = ChartFactory.create_demand_pattern_chart([])
        assert fig is not None

    def test_create_demand_pattern_chart_with_data(self):
        """데이터 있는 수요 패턴 차트"""
        data = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "demand": 800.0
            },
            {
                "timestamp": "2025-01-01T12:00:00",
                "demand": 1200.0
            }
        ]
        fig = ChartFactory.create_demand_pattern_chart(data)

        assert fig is not None
        assert len(fig.data) >= 1

    def test_create_model_comparison_chart_empty(self):
        """빈 모델 비교 차트"""
        fig = ChartFactory.create_model_comparison_chart([])
        assert fig is not None

    def test_create_model_comparison_chart_with_data(self):
        """데이터 있는 모델 비교 차트"""
        models = [
            {"name": "lstm", "metrics": {"rmse": 45, "mape": 3.5}},
            {"name": "ensemble", "metrics": {"rmse": 40, "mape": 3.0}}
        ]
        fig = ChartFactory.create_model_comparison_chart(models)

        assert fig is not None
        assert len(fig.data) >= 2

    def test_create_gauge_chart(self):
        """게이지 차트"""
        fig = ChartFactory.create_gauge_chart(75, "System Load", max_value=100)

        assert fig is not None


# ============================================================================
# DashboardComponents Tests
# ============================================================================

class TestDashboardComponents:
    """대시보드 컴포넌트 테스트"""

    # Note: Streamlit 컴포넌트 테스트는 통합 테스트에서 수행
    # 여기서는 로직만 테스트

    def test_components_class_exists(self):
        """컴포넌트 클래스 존재"""
        components = DashboardComponents()
        assert components is not None

    def test_render_header_is_callable(self):
        """render_header 호출 가능"""
        assert callable(DashboardComponents.render_header)

    def test_render_sidebar_is_callable(self):
        """render_sidebar 호출 가능"""
        assert callable(DashboardComponents.render_sidebar)

    def test_render_status_cards_is_callable(self):
        """render_status_cards 호출 가능"""
        assert callable(DashboardComponents.render_status_cards)


# ============================================================================
# Dashboard Tests
# ============================================================================

class TestDashboard:
    """대시보드 테스트"""

    def test_init_default_config(self):
        """기본 설정으로 초기화"""
        dashboard = Dashboard()
        assert dashboard.config is not None
        assert dashboard.fetcher is not None
        assert dashboard.mock_generator is not None
        assert dashboard.chart_factory is not None

    def test_init_custom_config(self):
        """커스텀 설정으로 초기화"""
        config = DashboardConfig(api_url="http://custom:9000")
        dashboard = Dashboard(config)

        assert dashboard.config.api_url == "http://custom:9000"

    @patch.object(DataFetcher, "get_health")
    def test_check_api_status_online(self, mock_health):
        """API 온라인 상태"""
        mock_health.return_value = {"status": "healthy"}
        dashboard = Dashboard()

        assert dashboard.check_api_status() is True

    @patch.object(DataFetcher, "get_health")
    def test_check_api_status_offline(self, mock_health):
        """API 오프라인 상태"""
        mock_health.return_value = None
        dashboard = Dashboard()

        assert dashboard.check_api_status() is False

    def test_dashboard_has_run_method(self):
        """run 메서드 존재"""
        dashboard = Dashboard()
        assert callable(dashboard.run)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_mock_data_to_chart_flow(self):
        """모의 데이터 → 차트 흐름"""
        # 모의 데이터 생성
        predictions = MockDataGenerator.generate_predictions()
        historical = MockDataGenerator.generate_historical_data(
            date(2025, 1, 1),
            date(2025, 1, 7)
        )

        # 차트 생성
        pred_chart = ChartFactory.create_prediction_chart(predictions["predictions"])
        hist_chart = ChartFactory.create_historical_chart(historical["data"])

        assert pred_chart is not None
        assert hist_chart is not None

    def test_full_mock_workflow(self):
        """전체 모의 워크플로우"""
        # 1. 설정
        config = DashboardConfig()

        # 2. 모의 데이터
        predictions = MockDataGenerator.generate_predictions(["1h", "6h", "24h"])
        historical = MockDataGenerator.generate_historical_data(
            date(2025, 1, 1),
            date(2025, 1, 7),
            "hourly"
        )

        # 3. 데이터 검증
        assert len(predictions["predictions"]) == 3
        assert len(historical["data"]) >= 24 * 7

        # 4. 차트 생성
        charts = [
            ChartFactory.create_prediction_chart(predictions["predictions"]),
            ChartFactory.create_historical_chart(historical["data"]),
            ChartFactory.create_demand_pattern_chart(historical["data"]),
        ]

        for chart in charts:
            assert chart is not None


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_predictions_list(self):
        """빈 예측 리스트"""
        result = MockDataGenerator.generate_predictions([])
        assert result["predictions"] == []

    def test_single_day_historical(self):
        """1일 과거 데이터"""
        result = MockDataGenerator.generate_historical_data(
            date(2025, 1, 1),
            date(2025, 1, 1),
            "daily"
        )
        assert result["count"] == 1

    def test_large_horizon_value(self):
        """큰 시간대 값"""
        result = MockDataGenerator.generate_predictions(["48h"])
        pred = result["predictions"][0]

        assert pred["horizon"] == "48h"
        # 신뢰도가 낮아야 함
        assert pred["confidence"] < 0.95

    def test_chart_with_none_values(self):
        """None 값이 있는 차트"""
        predictions = [
            {
                "timestamp": "2025-01-01T12:00:00",
                "prediction": 1000.0,
                "lower_bound": None,
                "upper_bound": None
            }
        ]
        # None 처리가 잘 되어야 함
        fig = ChartFactory.create_prediction_chart(predictions)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
