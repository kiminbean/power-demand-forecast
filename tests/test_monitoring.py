"""
모니터링 시스템 테스트 (Task 18)
================================
모니터링 모듈 테스트
"""

import pytest
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Prometheus Metrics Tests
# ============================================================================

class TestPrometheusMetrics:
    """Prometheus 메트릭 테스트"""

    def test_counter_basic(self):
        """카운터 기본 동작"""
        from src.monitoring.prometheus_metrics import Counter

        counter = Counter('test_counter', 'Test counter')
        assert counter.get() == 0.0

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5.0)
        assert counter.get() == 6.0

    def test_counter_with_labels(self):
        """라벨이 있는 카운터"""
        from src.monitoring.prometheus_metrics import Counter

        counter = Counter('test_counter', 'Test counter', labels=['method', 'status'])

        counter.inc(labels={'method': 'GET', 'status': '200'})
        counter.inc(labels={'method': 'GET', 'status': '200'})
        counter.inc(labels={'method': 'POST', 'status': '201'})

        assert counter.get(labels={'method': 'GET', 'status': '200'}) == 2.0
        assert counter.get(labels={'method': 'POST', 'status': '201'}) == 1.0

    def test_gauge_basic(self):
        """게이지 기본 동작"""
        from src.monitoring.prometheus_metrics import Gauge

        gauge = Gauge('test_gauge', 'Test gauge')

        gauge.set(50.0)
        assert gauge.get() == 50.0

        gauge.inc(10.0)
        assert gauge.get() == 60.0

        gauge.dec(20.0)
        assert gauge.get() == 40.0

    def test_histogram_basic(self):
        """히스토그램 기본 동작"""
        from src.monitoring.prometheus_metrics import Histogram

        histogram = Histogram(
            'test_histogram',
            'Test histogram',
            buckets=[0.1, 0.5, 1.0, 2.0]
        )

        histogram.observe(0.3)
        histogram.observe(0.7)
        histogram.observe(1.5)

        metrics = histogram.collect()
        assert len(metrics) > 0

    def test_timer_context_manager(self):
        """타이머 컨텍스트 매니저"""
        from src.monitoring.prometheus_metrics import Histogram, Timer

        histogram = Histogram('test_latency', 'Test latency')

        with Timer(histogram):
            time.sleep(0.1)

        metrics = histogram.collect()
        sum_metric = next(m for m in metrics if m.name == 'test_latency_sum')
        assert sum_metric.value >= 0.1

    def test_prediction_metrics(self):
        """예측 메트릭"""
        from src.monitoring.prometheus_metrics import PredictionMetrics

        metrics = PredictionMetrics()

        metrics.record_prediction(
            location='jeju',
            horizon='1h',
            model_type='lstm',
            value=850.0,
            latency=0.5
        )

        collected = metrics.collect()
        assert len(collected) > 0

        # 요청 카운터 확인
        request_metrics = [m for m in collected if 'requests' in m.name]
        assert len(request_metrics) > 0

    def test_system_metrics(self):
        """시스템 메트릭"""
        from src.monitoring.prometheus_metrics import SystemMetrics

        metrics = SystemMetrics()
        metrics.update()

        collected = metrics.collect()
        assert len(collected) > 0

        # CPU 메트릭 확인
        cpu_metrics = [m for m in collected if 'cpu' in m.name]
        assert len(cpu_metrics) > 0

    def test_metrics_collector(self):
        """메트릭 수집기"""
        from src.monitoring.prometheus_metrics import MetricsCollector

        collector = MetricsCollector()

        # 커스텀 메트릭 등록
        counter = collector.register_counter('custom_counter', 'Custom counter')
        counter.inc(10)

        # Prometheus 형식 내보내기
        prometheus_output = collector.export_prometheus()
        assert 'custom_counter' in prometheus_output

        # JSON 형식 내보내기
        json_output = collector.export_json()
        assert isinstance(json_output, dict)


# ============================================================================
# Alerting Tests
# ============================================================================

class TestAlerting:
    """알림 시스템 테스트"""

    def test_alert_creation(self):
        """알림 생성"""
        from src.monitoring.alerting import Alert, AlertLevel, AlertState

        alert = Alert(
            name='test_alert',
            level=AlertLevel.WARNING,
            message='Test alert message'
        )

        assert alert.name == 'test_alert'
        assert alert.level == AlertLevel.WARNING
        assert alert.state == AlertState.FIRING

        alert_dict = alert.to_dict()
        assert alert_dict['name'] == 'test_alert'
        assert alert_dict['level'] == 'warning'

    def test_threshold_rule_gt(self):
        """임계값 규칙 (greater than)"""
        from src.monitoring.alerting import ThresholdRule, AlertLevel

        rule = ThresholdRule(
            name='high_value',
            threshold=100.0,
            comparison='gt',
            level=AlertLevel.WARNING
        )

        # 임계값 이하 - 알림 없음
        alert = rule.evaluate(50.0)
        assert alert is None

        # 임계값 초과 - 알림 발생
        alert = rule.evaluate(150.0)
        assert alert is not None
        assert 'high_value' in alert.name

    def test_threshold_rule_lt(self):
        """임계값 규칙 (less than)"""
        from src.monitoring.alerting import ThresholdRule, AlertLevel

        rule = ThresholdRule(
            name='low_value',
            threshold=50.0,
            comparison='lt',
            level=AlertLevel.ERROR
        )

        # 임계값 이상 - 알림 없음
        alert = rule.evaluate(60.0)
        assert alert is None

        # 임계값 미만 - 알림 발생
        alert = rule.evaluate(30.0)
        assert alert is not None

    def test_threshold_rule_with_duration(self):
        """지속 시간 있는 임계값 규칙"""
        from src.monitoring.alerting import ThresholdRule, AlertLevel

        rule = ThresholdRule(
            name='sustained_high',
            threshold=100.0,
            comparison='gt',
            for_duration=timedelta(seconds=1)
        )

        # 첫 번째 초과 - 대기 상태
        alert = rule.evaluate(150.0)
        assert alert is None

        # 시간 경과 후 재평가
        time.sleep(1.1)
        alert = rule.evaluate(150.0)
        assert alert is not None

    def test_anomaly_rule(self):
        """이상 탐지 규칙"""
        from src.monitoring.alerting import AnomalyRule, AlertLevel

        rule = AnomalyRule(
            name='anomaly',
            window_size=20,
            std_threshold=2.0
        )

        # 정상 범위 데이터
        for i in range(20):
            alert = rule.evaluate(100.0 + i * 0.1)
            assert alert is None

        # 이상치
        alert = rule.evaluate(200.0)
        assert alert is not None

    def test_alert_manager(self):
        """알림 매니저"""
        from src.monitoring.alerting import (
            AlertManager, ThresholdRule, AlertLevel
        )

        manager = AlertManager()

        # 규칙 추가
        manager.add_rule(ThresholdRule(
            name='test_rule',
            threshold=100.0,
            comparison='gt',
            labels={'metric': 'test_metric'}
        ))

        # 알림 평가
        alerts = manager.evaluate('test_metric', 150.0)
        assert len(alerts) > 0

        # 활성 알림 조회
        active = manager.get_active_alerts()
        assert len(active) > 0

    def test_alert_manager_resolve(self):
        """알림 해제"""
        from src.monitoring.alerting import (
            AlertManager, ThresholdRule, AlertState
        )

        manager = AlertManager()

        manager.add_rule(ThresholdRule(
            name='test_rule',
            threshold=100.0,
            comparison='gt',
            labels={'metric': 'test_metric'}
        ))

        # 알림 발생
        manager.evaluate('test_metric', 150.0)
        assert len(manager.get_active_alerts()) == 1

        # 알림 해제
        alerts = manager.evaluate('test_metric', 50.0)
        resolved = [a for a in alerts if a.state == AlertState.RESOLVED]
        assert len(resolved) > 0
        assert len(manager.get_active_alerts()) == 0


# ============================================================================
# Logging Tests
# ============================================================================

class TestLogging:
    """로깅 테스트"""

    def test_json_formatter(self):
        """JSON 포매터"""
        import logging
        from src.monitoring.logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert 'Test message' in formatted
        assert 'timestamp' in formatted

        import json
        parsed = json.loads(formatted)
        assert parsed['message'] == 'Test message'
        assert parsed['level'] == 'INFO'

    def test_log_context(self):
        """로그 컨텍스트"""
        from src.monitoring.logging_config import LogContext

        LogContext.clear()
        LogContext.set(request_id='123', user='test')

        context = LogContext.get()
        assert context['request_id'] == '123'
        assert context['user'] == 'test'

        LogContext.clear()
        assert LogContext.get() == {}

    def test_log_context_scope(self):
        """로그 컨텍스트 스코프"""
        from src.monitoring.logging_config import LogContext

        LogContext.clear()
        LogContext.set(outer='value')

        with LogContext.scope(inner='scoped'):
            context = LogContext.get()
            assert context['inner'] == 'scoped'

        context = LogContext.get()
        assert 'inner' not in context

    def test_structured_logger(self):
        """구조화된 로거"""
        from src.monitoring.logging_config import StructuredLogger

        logger = StructuredLogger('test')

        # 예외 발생하지 않음
        logger.info('Test message', key='value')
        logger.warning('Warning message')
        logger.error('Error message')

    def test_setup_logging(self):
        """로깅 설정"""
        from src.monitoring.logging_config import setup_logging, LogConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(
                log_dir=tmpdir,
                format='json',
                output='file'
            )
            setup_logging(config)

            import logging
            logging.info('Test log message')

            # 로그 파일 생성 확인
            log_files = list(Path(tmpdir).glob('*.log'))
            assert len(log_files) > 0


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthChecks:
    """헬스 체크 테스트"""

    def test_system_health_check(self):
        """시스템 헬스 체크"""
        from src.monitoring.health_checks import SystemHealthCheck, HealthStatus

        check = SystemHealthCheck()
        result = check.check()

        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert 'cpu_percent' in result.details
        assert 'memory_percent' in result.details

    def test_model_health_check(self):
        """모델 헬스 체크"""
        from src.monitoring.health_checks import ModelHealthCheck, HealthStatus

        check = ModelHealthCheck()
        result = check.check()

        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def test_model_health_with_metrics(self):
        """메트릭이 있는 모델 헬스 체크"""
        from src.monitoring.health_checks import ModelHealthCheck, HealthStatus

        check = ModelHealthCheck(min_accuracy=0.8, max_latency_ms=1000.0)

        # 정상 메트릭
        check.update_metrics(latency_ms=100.0, accuracy=0.95)
        result = check.check()
        assert result.status == HealthStatus.HEALTHY

        # 낮은 정확도
        check.update_metrics(latency_ms=100.0, accuracy=0.5)
        result = check.check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_dependency_health_check(self):
        """의존성 헬스 체크"""
        from src.monitoring.health_checks import DependencyHealthCheck, HealthStatus

        # 성공하는 체크
        check = DependencyHealthCheck('success', lambda: True)
        result = check.check()
        assert result.status == HealthStatus.HEALTHY

        # 실패하는 체크
        check = DependencyHealthCheck('fail', lambda: False)
        result = check.check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_health_checker(self):
        """헬스 체커"""
        from src.monitoring.health_checks import (
            HealthChecker, SystemHealthCheck, HealthStatus
        )

        checker = HealthChecker()
        checker.add_check(SystemHealthCheck())

        # 모든 체크 실행
        results = checker.run_all_checks()
        assert len(results) > 0
        assert 'system' in results

    def test_health_report(self):
        """헬스 리포트"""
        from src.monitoring.health_checks import HealthChecker, SystemHealthCheck

        checker = HealthChecker()
        checker.add_check(SystemHealthCheck())

        report = checker.get_health_report()

        assert 'status' in report
        assert 'timestamp' in report
        assert 'checks' in report
        assert 'summary' in report
        assert 'total' in report['summary']

    def test_overall_status(self):
        """전체 상태"""
        from src.monitoring.health_checks import (
            HealthChecker, DependencyHealthCheck, HealthStatus
        )

        checker = HealthChecker()

        # 모두 정상
        checker.add_check(DependencyHealthCheck('check1', lambda: True))
        checker.add_check(DependencyHealthCheck('check2', lambda: True))
        assert checker.get_overall_status() == HealthStatus.HEALTHY

        # 하나 실패
        checker.add_check(DependencyHealthCheck('check3', lambda: False))
        assert checker.get_overall_status() == HealthStatus.UNHEALTHY


# ============================================================================
# Integration Tests
# ============================================================================

class TestMonitoringIntegration:
    """모니터링 통합 테스트"""

    def test_full_monitoring_flow(self):
        """전체 모니터링 흐름"""
        from src.monitoring import (
            MetricsCollector,
            AlertManager,
            ThresholdRule,
            HealthChecker,
            SystemHealthCheck
        )

        # 메트릭 수집기
        collector = MetricsCollector()

        # 알림 매니저
        alert_manager = AlertManager()
        alert_manager.add_rule(ThresholdRule(
            name='high_cpu',
            threshold=90.0,
            comparison='gt'
        ))

        # 헬스 체커
        health_checker = HealthChecker()
        health_checker.add_check(SystemHealthCheck())

        # 시스템 메트릭 수집
        collector.system_metrics.update()
        system_metrics = collector.system_metrics.collect()
        assert len(system_metrics) > 0

        # 헬스 체크 실행
        report = health_checker.get_health_report()
        assert report['status'] in ['healthy', 'degraded', 'unhealthy']

    def test_metrics_and_alerting(self):
        """메트릭과 알림 통합"""
        from src.monitoring import (
            MetricsCollector,
            AlertManager,
            ThresholdRule,
            AlertLevel
        )

        collector = MetricsCollector()
        alert_manager = AlertManager()

        # 높은 레이턴시 알림 규칙
        alert_manager.add_rule(ThresholdRule(
            name='high_latency',
            threshold=1.0,
            comparison='gt',
            level=AlertLevel.WARNING,
            labels={'metric': 'latency'}
        ))

        # 예측 수행 (높은 레이턴시)
        collector.prediction_metrics.record_prediction(
            location='jeju',
            horizon='1h',
            model_type='lstm',
            value=850.0,
            latency=2.0  # 높은 레이턴시
        )

        # 알림 평가
        alerts = alert_manager.evaluate('latency', 2.0)
        assert len(alerts) > 0

    def test_logging_with_context(self):
        """컨텍스트 로깅 통합"""
        from src.monitoring import (
            setup_logging, LogContext, create_logger, LogConfig
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(log_dir=tmpdir, output='file')
            setup_logging(config)

            logger = create_logger('test')

            with LogContext.scope(request_id='abc123'):
                logger.info('Request started')
                logger.info('Request completed')

            # 로그 파일 확인
            log_files = list(Path(tmpdir).glob('*.log'))
            assert len(log_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
