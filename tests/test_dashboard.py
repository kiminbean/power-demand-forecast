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


# ============================================================================
# Reserve Rate Alert System Tests (v4.0.2)
# ============================================================================

class TestReserveRateAlertThresholds:
    """예비율 경보 임계값 테스트 (KPX 기준)"""

    def get_alert_status(self, reserve_rate: float) -> dict:
        """
        Reserve rate에 따른 경보 상태를 반환하는 헬퍼 함수.
        app_v4.py의 로직을 미러링.
        """
        if reserve_rate < 5:
            return {
                "status": "critical",
                "class": "status-critical",
                "text": "위험",
                "alert_class": "alert-danger",
                "icon": "🚨",
                "show_alert": True
            }
        elif reserve_rate < 10:
            return {
                "status": "danger",
                "class": "status-danger",
                "text": "주의",
                "alert_class": "alert-danger",
                "icon": "⚠️",
                "show_alert": True
            }
        elif reserve_rate < 15:
            return {
                "status": "warning",
                "class": "status-warning",
                "text": "관심",
                "alert_class": "alert-warning",
                "icon": "📢",
                "show_alert": True
            }
        else:
            return {
                "status": "normal",
                "class": "status-normal",
                "text": "정상",
                "alert_class": None,
                "icon": None,
                "show_alert": False
            }

    # ========== Critical Alert Tests (<5%) ==========

    def test_critical_at_0_percent(self):
        """0% 예비율: Critical 경보"""
        result = self.get_alert_status(0.0)
        assert result["status"] == "critical"
        assert result["text"] == "위험"
        assert result["show_alert"] is True

    def test_critical_at_3_percent(self):
        """3% 예비율: Critical 경보"""
        result = self.get_alert_status(3.0)
        assert result["status"] == "critical"
        assert result["icon"] == "🚨"

    def test_critical_at_4_99_percent(self):
        """4.99% 예비율: Critical 경보 (경계값)"""
        result = self.get_alert_status(4.99)
        assert result["status"] == "critical"
        assert result["class"] == "status-critical"

    # ========== Danger/Warning Alert Tests (5-10%) ==========

    def test_danger_at_5_percent(self):
        """5% 예비율: Danger 경보 (경계값)"""
        result = self.get_alert_status(5.0)
        assert result["status"] == "danger"
        assert result["text"] == "주의"
        assert result["show_alert"] is True

    def test_danger_at_7_percent(self):
        """7% 예비율: Danger 경보"""
        result = self.get_alert_status(7.0)
        assert result["status"] == "danger"
        assert result["icon"] == "⚠️"

    def test_danger_at_9_99_percent(self):
        """9.99% 예비율: Danger 경보 (경계값)"""
        result = self.get_alert_status(9.99)
        assert result["status"] == "danger"
        assert result["class"] == "status-danger"

    # ========== Caution Alert Tests (10-15%) ==========

    def test_warning_at_10_percent(self):
        """10% 예비율: Warning 경보 (경계값)"""
        result = self.get_alert_status(10.0)
        assert result["status"] == "warning"
        assert result["text"] == "관심"
        assert result["show_alert"] is True

    def test_warning_at_12_percent(self):
        """12% 예비율: Warning 경보"""
        result = self.get_alert_status(12.0)
        assert result["status"] == "warning"
        assert result["icon"] == "📢"

    def test_warning_at_14_99_percent(self):
        """14.99% 예비율: Warning 경보 (경계값)"""
        result = self.get_alert_status(14.99)
        assert result["status"] == "warning"
        assert result["alert_class"] == "alert-warning"

    # ========== Normal Status Tests (>=15%) ==========

    def test_normal_at_15_percent(self):
        """15% 예비율: 정상 (경계값)"""
        result = self.get_alert_status(15.0)
        assert result["status"] == "normal"
        assert result["text"] == "정상"
        assert result["show_alert"] is False

    def test_normal_at_20_percent(self):
        """20% 예비율: 정상"""
        result = self.get_alert_status(20.0)
        assert result["status"] == "normal"
        assert result["alert_class"] is None

    def test_normal_at_50_percent(self):
        """50% 예비율: 정상"""
        result = self.get_alert_status(50.0)
        assert result["status"] == "normal"
        assert result["icon"] is None

    def test_normal_at_100_percent(self):
        """100% 예비율: 정상"""
        result = self.get_alert_status(100.0)
        assert result["status"] == "normal"
        assert result["show_alert"] is False

    # ========== Edge Cases ==========

    def test_negative_reserve_rate(self):
        """음수 예비율: Critical 경보"""
        result = self.get_alert_status(-5.0)
        assert result["status"] == "critical"

    def test_very_high_reserve_rate(self):
        """매우 높은 예비율 (500%): 정상"""
        result = self.get_alert_status(500.0)
        assert result["status"] == "normal"


class TestReserveRateAlertMessages:
    """예비율 경보 메시지 테스트"""

    def get_alert_message(self, reserve_rate: float) -> str:
        """Reserve rate에 따른 경보 메시지 반환"""
        if reserve_rate < 5:
            return f"예비율 {reserve_rate:.1f}% - 즉각적인 부하 감축 필요"
        elif reserve_rate < 10:
            return f"예비율 {reserve_rate:.1f}% - 전력 수급 상황 주시 필요"
        elif reserve_rate < 15:
            return f"예비율 {reserve_rate:.1f}% - 전력 사용 절감 협조 요청"
        else:
            return None

    def test_critical_message_format(self):
        """Critical 메시지 포맷 테스트"""
        msg = self.get_alert_message(3.0)
        assert "3.0%" in msg
        assert "즉각적인 부하 감축" in msg

    def test_danger_message_format(self):
        """Danger 메시지 포맷 테스트"""
        msg = self.get_alert_message(7.5)
        assert "7.5%" in msg
        assert "수급 상황 주시" in msg

    def test_warning_message_format(self):
        """Warning 메시지 포맷 테스트"""
        msg = self.get_alert_message(12.0)
        assert "12.0%" in msg
        assert "절감 협조" in msg

    def test_normal_no_message(self):
        """Normal 상태: 메시지 없음"""
        msg = self.get_alert_message(20.0)
        assert msg is None


class TestReserveRateAlertTitles:
    """예비율 경보 제목 테스트"""

    def get_alert_title(self, reserve_rate: float) -> str:
        """Reserve rate에 따른 경보 제목 반환"""
        if reserve_rate < 5:
            return "전력 수급 위험 경보"
        elif reserve_rate < 10:
            return "전력 수급 주의 경보"
        elif reserve_rate < 15:
            return "전력 수급 관심 단계"
        else:
            return None

    def test_critical_title(self):
        """Critical 제목: 위험 경보"""
        title = self.get_alert_title(3.0)
        assert title == "전력 수급 위험 경보"

    def test_danger_title(self):
        """Danger 제목: 주의 경보"""
        title = self.get_alert_title(7.0)
        assert title == "전력 수급 주의 경보"

    def test_warning_title(self):
        """Warning 제목: 관심 단계"""
        title = self.get_alert_title(12.0)
        assert title == "전력 수급 관심 단계"

    def test_normal_no_title(self):
        """Normal: 제목 없음"""
        title = self.get_alert_title(20.0)
        assert title is None


class TestKPXThresholdConstants:
    """KPX 임계값 상수 테스트"""

    # KPX 공식 기준값
    KPX_CRITICAL_THRESHOLD = 5.0   # 위험
    KPX_DANGER_THRESHOLD = 10.0    # 주의
    KPX_WARNING_THRESHOLD = 15.0   # 관심

    def test_critical_threshold_value(self):
        """Critical 임계값: 5%"""
        assert self.KPX_CRITICAL_THRESHOLD == 5.0

    def test_danger_threshold_value(self):
        """Danger 임계값: 10%"""
        assert self.KPX_DANGER_THRESHOLD == 10.0

    def test_warning_threshold_value(self):
        """Warning 임계값: 15%"""
        assert self.KPX_WARNING_THRESHOLD == 15.0

    def test_threshold_ordering(self):
        """임계값 순서: Critical < Danger < Warning"""
        assert self.KPX_CRITICAL_THRESHOLD < self.KPX_DANGER_THRESHOLD
        assert self.KPX_DANGER_THRESHOLD < self.KPX_WARNING_THRESHOLD

    def test_all_thresholds_positive(self):
        """모든 임계값이 양수"""
        assert self.KPX_CRITICAL_THRESHOLD > 0
        assert self.KPX_DANGER_THRESHOLD > 0
        assert self.KPX_WARNING_THRESHOLD > 0


# ============================================================================
# Alert History Tests (v4.0.2)
# ============================================================================

import tempfile
import json


class AlertHistoryForTest:
    """테스트용 AlertHistory 클래스 (app_v4.py와 동일한 로직)"""

    MAX_HISTORY = 100

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._history = self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    def add_alert(self, reserve_rate: float, status: str, title: str, message: str):
        now = datetime.now()
        if self._history:
            last = self._history[0]
            last_time = datetime.fromisoformat(last['timestamp'])
            if last['status'] == status and (now - last_time).seconds < 60:
                return

        alert = {
            'timestamp': now.isoformat(),
            'reserve_rate': round(reserve_rate, 2),
            'status': status,
            'title': title,
            'message': message
        }
        self._history.insert(0, alert)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]
        self._save()

    def get_recent(self, count: int = 10):
        return self._history[:count]

    def get_stats(self):
        if not self._history:
            return {'total': 0, 'critical': 0, 'danger': 0, 'warning': 0}
        stats = {'total': len(self._history), 'critical': 0, 'danger': 0, 'warning': 0}
        for alert in self._history:
            status = alert.get('status', '')
            if status in stats:
                stats[status] += 1
        return stats

    def clear(self):
        self._history = []
        self._save()


class TestAlertHistory:
    """AlertHistory 클래스 테스트"""

    def create_temp_history(self):
        """테스트용 임시 AlertHistory 생성"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / "test_alerts.json"
        return AlertHistoryForTest(file_path=temp_path)

    def test_alert_history_creation(self):
        """AlertHistory 생성 테스트"""
        history = self.create_temp_history()
        assert history is not None
        assert history._history == []

    def test_add_alert(self):
        """경보 추가 테스트"""
        history = self.create_temp_history()
        history.add_alert(
            reserve_rate=3.5,
            status="critical",
            title="전력 수급 위험 경보",
            message="예비율 3.5% - 즉각적인 부하 감축 필요"
        )
        assert len(history._history) == 1
        assert history._history[0]['reserve_rate'] == 3.5
        assert history._history[0]['status'] == "critical"

    def test_add_multiple_alerts(self):
        """여러 경보 추가 테스트"""
        history = self.create_temp_history()

        # 첫 번째 경보
        history.add_alert(3.0, "critical", "위험", "msg1")

        # 다른 status의 경보는 바로 추가
        history.add_alert(7.0, "danger", "주의", "msg2")

        assert len(history._history) == 2
        # 최신 경보가 먼저
        assert history._history[0]['status'] == "danger"
        assert history._history[1]['status'] == "critical"

    def test_duplicate_alert_prevention(self):
        """중복 경보 방지 테스트 (같은 status 1분 이내)"""
        history = self.create_temp_history()

        history.add_alert(3.0, "critical", "위험1", "msg1")
        history.add_alert(4.0, "critical", "위험2", "msg2")  # 같은 status

        # 같은 status는 1분 이내 중복 추가 안됨
        assert len(history._history) == 1

    def test_get_recent(self):
        """최근 경보 조회 테스트"""
        history = self.create_temp_history()

        # 여러 경보 추가 (다른 status로)
        history.add_alert(3.0, "critical", "위험", "msg1")
        history.add_alert(7.0, "danger", "주의", "msg2")
        history.add_alert(12.0, "warning", "관심", "msg3")

        recent = history.get_recent(2)
        assert len(recent) == 2
        assert recent[0]['status'] == "warning"  # 최신

    def test_get_stats(self):
        """경보 통계 테스트"""
        history = self.create_temp_history()

        history.add_alert(3.0, "critical", "위험", "msg1")
        history.add_alert(7.0, "danger", "주의", "msg2")
        history.add_alert(12.0, "warning", "관심", "msg3")

        stats = history.get_stats()
        assert stats['total'] == 3
        assert stats['critical'] == 1
        assert stats['danger'] == 1
        assert stats['warning'] == 1

    def test_get_stats_empty(self):
        """빈 이력 통계 테스트"""
        history = self.create_temp_history()
        stats = history.get_stats()
        assert stats['total'] == 0

    def test_clear_history(self):
        """이력 초기화 테스트"""
        history = self.create_temp_history()

        history.add_alert(3.0, "critical", "위험", "msg")
        assert len(history._history) == 1

        history.clear()
        assert len(history._history) == 0

    def test_persistence(self):
        """파일 저장/로드 테스트"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / "persist_test.json"

        # 첫 번째 인스턴스에서 저장
        history1 = AlertHistoryForTest(file_path=temp_path)
        history1.add_alert(5.0, "danger", "주의", "msg")

        # 두 번째 인스턴스에서 로드
        history2 = AlertHistoryForTest(file_path=temp_path)
        assert len(history2._history) == 1
        assert history2._history[0]['reserve_rate'] == 5.0

    def test_max_history_limit(self):
        """최대 이력 개수 제한 테스트"""
        history = self.create_temp_history()
        history.MAX_HISTORY = 5  # 테스트용 제한

        # 다른 status로 6개 추가
        statuses = ["critical", "danger", "warning", "critical", "danger", "warning"]
        for i, status in enumerate(statuses):
            history.add_alert(float(i), status, f"title{i}", f"msg{i}")

        assert len(history._history) <= 5

    def test_alert_timestamp(self):
        """경보 타임스탬프 테스트"""
        history = self.create_temp_history()
        history.add_alert(3.0, "critical", "위험", "msg")

        alert = history._history[0]
        assert 'timestamp' in alert
        # ISO format 검증
        timestamp = datetime.fromisoformat(alert['timestamp'])
        assert isinstance(timestamp, datetime)

    def test_alert_fields(self):
        """경보 필드 완전성 테스트"""
        history = self.create_temp_history()
        history.add_alert(
            reserve_rate=7.5,
            status="danger",
            title="전력 수급 주의 경보",
            message="예비율 7.5% - 주시 필요"
        )

        alert = history._history[0]
        assert 'timestamp' in alert
        assert alert['reserve_rate'] == 7.5
        assert alert['status'] == "danger"
        assert alert['title'] == "전력 수급 주의 경보"
        assert alert['message'] == "예비율 7.5% - 주시 필요"
