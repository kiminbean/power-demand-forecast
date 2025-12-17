"""
Dashboard 테스트
================

현재 대시보드 구조 (app.py, app_v1.py)에 맞춘 테스트
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock streamlit before importing dashboard
sys.modules['streamlit'] = MagicMock()


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """대시보드 설정 테스트"""

    def test_config_import(self):
        """Config 클래스 import 테스트"""
        from src.dashboard.app import Config
        assert Config is not None

    def test_config_attributes(self):
        """Config 속성 테스트"""
        from src.dashboard.app import Config
        assert hasattr(Config, 'API_URL')
        assert hasattr(Config, 'RENEWABLE_API_URL')
        assert hasattr(Config, 'RENEWABLE_COLORS')

    def test_config_api_url(self):
        """API URL 설정 테스트"""
        from src.dashboard.app import Config
        assert Config.API_URL == "http://localhost:8000"

    def test_config_renewable_api_url(self):
        """신재생 API URL 설정 테스트"""
        from src.dashboard.app import Config
        assert Config.RENEWABLE_API_URL == "http://localhost:8001"

    def test_config_renewable_colors(self):
        """신재생에너지 색상 설정 테스트"""
        from src.dashboard.app import Config
        assert isinstance(Config.RENEWABLE_COLORS, dict)
        assert 'solar' in Config.RENEWABLE_COLORS
        assert 'wind' in Config.RENEWABLE_COLORS
        assert 'total' in Config.RENEWABLE_COLORS


# ============================================================================
# APIClient Tests
# ============================================================================

class TestAPIClient:
    """API 클라이언트 테스트"""

    def test_api_client_import(self):
        """APIClient 클래스 import 테스트"""
        from src.dashboard.app import APIClient
        assert APIClient is not None

    def test_api_client_init(self):
        """APIClient 초기화 테스트"""
        from src.dashboard.app import APIClient
        client = APIClient()
        assert client.base_url == "http://localhost:8000"

    def test_api_client_custom_url(self):
        """APIClient 커스텀 URL 테스트"""
        from src.dashboard.app import APIClient
        client = APIClient(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"

    @patch('requests.get')
    def test_health_check_returns_dict(self, mock_get):
        """헬스 체크가 dict를 반환하는지 테스트"""
        from src.dashboard.app import APIClient
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "healthy"}

        client = APIClient()
        result = client.health_check()
        assert isinstance(result, dict)
        assert 'status' in result

    @patch('requests.get')
    def test_health_check_offline_on_error(self, mock_get):
        """헬스 체크 실패 시 offline 상태 반환 테스트"""
        from src.dashboard.app import APIClient
        mock_get.side_effect = Exception("Connection error")

        client = APIClient()
        result = client.health_check()
        assert isinstance(result, dict)
        assert result.get('status') == 'offline'


# ============================================================================
# RenewableAPIClient Tests
# ============================================================================

class TestRenewableAPIClient:
    """신재생에너지 API 클라이언트 테스트"""

    def test_renewable_api_client_import(self):
        """RenewableAPIClient 클래스 import 테스트"""
        from src.dashboard.app import RenewableAPIClient
        assert RenewableAPIClient is not None

    def test_renewable_api_client_init(self):
        """RenewableAPIClient 초기화 테스트"""
        from src.dashboard.app import RenewableAPIClient
        client = RenewableAPIClient()
        assert client.base_url == "http://localhost:8001"

    @patch('requests.get')
    def test_renewable_health_check_returns_dict(self, mock_get):
        """신재생 API 헬스 체크가 dict를 반환하는지 테스트"""
        from src.dashboard.app import RenewableAPIClient
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "healthy"}

        client = RenewableAPIClient()
        result = client.health_check()
        assert isinstance(result, dict)


# ============================================================================
# Charts Tests
# ============================================================================

class TestCharts:
    """차트 클래스 테스트"""

    def test_charts_import(self):
        """Charts 클래스 import 테스트"""
        from src.dashboard.app import Charts
        assert Charts is not None

    def test_charts_has_realtime_method(self):
        """Charts 실시간 예측 차트 메서드 존재 테스트"""
        from src.dashboard.app import Charts
        assert hasattr(Charts, 'create_realtime_prediction_chart')

    def test_charts_has_batch_method(self):
        """Charts 배치 예측 차트 메서드 존재 테스트"""
        from src.dashboard.app import Charts
        assert hasattr(Charts, 'create_batch_prediction_chart')

    def test_charts_has_scenario_method(self):
        """Charts 시나리오 비교 차트 메서드 존재 테스트"""
        from src.dashboard.app import Charts
        assert hasattr(Charts, 'create_scenario_comparison_chart')

    def test_charts_has_renewable_method(self):
        """Charts 신재생 예측 차트 메서드 존재 테스트"""
        from src.dashboard.app import Charts
        assert hasattr(Charts, 'create_renewable_prediction_chart')


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """유틸리티 함수 테스트"""

    def test_create_sample_weather_import(self):
        """create_sample_weather 함수 import 테스트"""
        from src.dashboard.app import create_sample_weather
        assert create_sample_weather is not None

    def test_create_sample_weather(self):
        """샘플 날씨 데이터 생성 테스트"""
        from src.dashboard.app import create_sample_weather
        weather = create_sample_weather(datetime.now(), hours=24)
        assert isinstance(weather, list)
        assert len(weather) == 24

    def test_create_sample_weather_keys(self):
        """샘플 날씨 데이터 키 테스트"""
        from src.dashboard.app import create_sample_weather
        weather = create_sample_weather(datetime.now(), hours=1)
        assert len(weather) > 0
        item = weather[0]
        assert 'datetime' in item
        assert 'temperature' in item
        assert 'humidity' in item


# ============================================================================
# Module Import Tests
# ============================================================================

class TestModuleImports:
    """모듈 import 테스트"""

    def test_dashboard_module_import(self):
        """대시보드 모듈 import 테스트"""
        from src.dashboard import Config, APIClient, RenewableAPIClient, Charts
        assert Config is not None
        assert APIClient is not None
        assert RenewableAPIClient is not None
        assert Charts is not None

    def test_all_exports(self):
        """__all__ export 테스트"""
        from src import dashboard
        assert hasattr(dashboard, '__all__')
        assert 'Config' in dashboard.__all__
        assert 'APIClient' in dashboard.__all__
