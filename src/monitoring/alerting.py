"""
알림 시스템 (Task 18)
======================
임계값 및 이상 탐지 기반 알림을 관리합니다.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from collections import deque
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """알림 수준"""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class AlertState(Enum):
    """알림 상태"""
    PENDING = 'pending'
    FIRING = 'firing'
    RESOLVED = 'resolved'


@dataclass
class Alert:
    """알림"""
    name: str
    level: AlertLevel
    message: str
    state: AlertState = AlertState.FIRING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    fired_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'state': self.state.value,
            'labels': self.labels,
            'annotations': self.annotations,
            'value': self.value,
            'threshold': self.threshold,
            'fired_at': self.fired_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertRule:
    """알림 규칙 베이스 클래스"""

    def __init__(
        self,
        name: str,
        level: AlertLevel = AlertLevel.WARNING,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        for_duration: timedelta = None
    ):
        self.name = name
        self.level = level
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.for_duration = for_duration or timedelta(seconds=0)

        self._pending_since: Optional[datetime] = None
        self._is_firing = False

    def evaluate(self, value: float) -> Optional[Alert]:
        """규칙 평가 - 서브클래스에서 구현"""
        raise NotImplementedError

    def _create_alert(self, message: str, value: float = None, threshold: float = None) -> Alert:
        """알림 생성"""
        return Alert(
            name=self.name,
            level=self.level,
            message=message,
            labels=self.labels,
            annotations=self.annotations,
            value=value,
            threshold=threshold
        )

    def reset(self) -> Optional[Alert]:
        """규칙 리셋 (알림 해제)"""
        if self._is_firing:
            self._is_firing = False
            self._pending_since = None
            return Alert(
                name=self.name,
                level=self.level,
                message=f'{self.name} resolved',
                state=AlertState.RESOLVED,
                labels=self.labels,
                resolved_at=datetime.now()
            )
        return None


class ThresholdRule(AlertRule):
    """임계값 기반 알림 규칙"""

    def __init__(
        self,
        name: str,
        threshold: float,
        comparison: str = 'gt',  # gt, lt, gte, lte, eq
        level: AlertLevel = AlertLevel.WARNING,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        for_duration: timedelta = None,
        message_template: str = None
    ):
        super().__init__(name, level, labels, annotations, for_duration)
        self.threshold = threshold
        self.comparison = comparison
        self.message_template = message_template or f'{name}: {{value}} {{comparison}} {{threshold}}'

    def evaluate(self, value: float) -> Optional[Alert]:
        """임계값 평가"""
        triggered = False

        if self.comparison == 'gt':
            triggered = value > self.threshold
            comp_str = '>'
        elif self.comparison == 'lt':
            triggered = value < self.threshold
            comp_str = '<'
        elif self.comparison == 'gte':
            triggered = value >= self.threshold
            comp_str = '>='
        elif self.comparison == 'lte':
            triggered = value <= self.threshold
            comp_str = '<='
        elif self.comparison == 'eq':
            triggered = value == self.threshold
            comp_str = '=='
        else:
            comp_str = '?'

        if triggered:
            if self._pending_since is None:
                self._pending_since = datetime.now()

            elapsed = datetime.now() - self._pending_since
            if elapsed >= self.for_duration:
                if not self._is_firing:
                    self._is_firing = True
                    message = self.message_template.format(
                        value=value,
                        threshold=self.threshold,
                        comparison=comp_str
                    )
                    return self._create_alert(message, value, self.threshold)
        else:
            if self._is_firing:
                return self.reset()
            self._pending_since = None

        return None


class AnomalyRule(AlertRule):
    """이상 탐지 기반 알림 규칙"""

    def __init__(
        self,
        name: str,
        window_size: int = 100,
        std_threshold: float = 3.0,
        level: AlertLevel = AlertLevel.WARNING,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        for_duration: timedelta = None
    ):
        super().__init__(name, level, labels, annotations, for_duration)
        self.window_size = window_size
        self.std_threshold = std_threshold
        self._history = deque(maxlen=window_size)

    def evaluate(self, value: float) -> Optional[Alert]:
        """이상 탐지 평가"""
        self._history.append(value)

        if len(self._history) < 10:
            return None

        values = np.array(list(self._history))
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return None

        z_score = abs(value - mean) / std

        if z_score > self.std_threshold:
            if self._pending_since is None:
                self._pending_since = datetime.now()

            elapsed = datetime.now() - self._pending_since
            if elapsed >= self.for_duration:
                if not self._is_firing:
                    self._is_firing = True
                    message = (
                        f'{self.name}: Anomaly detected - '
                        f'value={value:.2f}, z-score={z_score:.2f} > {self.std_threshold}'
                    )
                    return self._create_alert(message, value, z_score)
        else:
            if self._is_firing:
                return self.reset()
            self._pending_since = None

        return None


class RateOfChangeRule(AlertRule):
    """변화율 기반 알림 규칙"""

    def __init__(
        self,
        name: str,
        threshold: float,
        window_size: int = 10,
        level: AlertLevel = AlertLevel.WARNING,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        for_duration: timedelta = None
    ):
        super().__init__(name, level, labels, annotations, for_duration)
        self.threshold = threshold
        self.window_size = window_size
        self._history = deque(maxlen=window_size)

    def evaluate(self, value: float) -> Optional[Alert]:
        """변화율 평가"""
        if len(self._history) > 0:
            previous = self._history[-1]
            if previous != 0:
                rate_of_change = abs((value - previous) / previous) * 100
            else:
                rate_of_change = 0.0
        else:
            rate_of_change = 0.0

        self._history.append(value)

        if rate_of_change > self.threshold:
            if self._pending_since is None:
                self._pending_since = datetime.now()

            elapsed = datetime.now() - self._pending_since
            if elapsed >= self.for_duration:
                if not self._is_firing:
                    self._is_firing = True
                    message = (
                        f'{self.name}: Rapid change detected - '
                        f'rate={rate_of_change:.2f}% > {self.threshold}%'
                    )
                    return self._create_alert(message, value, rate_of_change)
        else:
            if self._is_firing:
                return self.reset()
            self._pending_since = None

        return None


class AlertHandler:
    """알림 핸들러 베이스 클래스"""

    def handle(self, alert: Alert) -> None:
        """알림 처리"""
        raise NotImplementedError


class LogAlertHandler(AlertHandler):
    """로그 기반 알림 핸들러"""

    def __init__(self, logger_name: str = 'alerts'):
        self._logger = logging.getLogger(logger_name)

    def handle(self, alert: Alert) -> None:
        """알림 로깅"""
        log_method = {
            AlertLevel.INFO: self._logger.info,
            AlertLevel.WARNING: self._logger.warning,
            AlertLevel.ERROR: self._logger.error,
            AlertLevel.CRITICAL: self._logger.critical
        }.get(alert.level, self._logger.warning)

        log_method(f'[{alert.state.value.upper()}] {alert.name}: {alert.message}')


class WebhookAlertHandler(AlertHandler):
    """웹훅 기반 알림 핸들러"""

    def __init__(self, webhook_url: str, timeout: int = 10):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def handle(self, alert: Alert) -> None:
        """웹훅 호출"""
        try:
            import urllib.request
            data = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=self.timeout)
            logger.info(f'Alert sent to webhook: {alert.name}')
        except Exception as e:
            logger.error(f'Failed to send alert to webhook: {e}')


class CallbackAlertHandler(AlertHandler):
    """콜백 기반 알림 핸들러"""

    def __init__(self, callback: Callable[[Alert], None]):
        self.callback = callback

    def handle(self, alert: Alert) -> None:
        """콜백 호출"""
        try:
            self.callback(alert)
        except Exception as e:
            logger.error(f'Alert callback failed: {e}')


class AlertManager:
    """알림 매니저"""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._handlers: List[AlertHandler] = []
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

        # 기본 로그 핸들러 추가
        self._handlers.append(LogAlertHandler())

    def add_rule(self, rule: AlertRule) -> None:
        """알림 규칙 추가"""
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """알림 규칙 제거"""
        with self._lock:
            if name in self._rules:
                del self._rules[name]

    def add_handler(self, handler: AlertHandler) -> None:
        """알림 핸들러 추가"""
        self._handlers.append(handler)

    def evaluate(self, metric_name: str, value: float) -> List[Alert]:
        """메트릭 평가"""
        triggered_alerts = []

        with self._lock:
            for rule_name, rule in self._rules.items():
                # 규칙과 메트릭 매칭 (라벨 기반)
                if rule.labels.get('metric') == metric_name or not rule.labels.get('metric'):
                    alert = rule.evaluate(value)
                    if alert:
                        triggered_alerts.append(alert)

                        if alert.state == AlertState.FIRING:
                            self._alerts[alert.name] = alert
                        elif alert.state == AlertState.RESOLVED:
                            if alert.name in self._alerts:
                                del self._alerts[alert.name]

                        self._alert_history.append(alert)
                        self._notify(alert)

        return triggered_alerts

    def evaluate_all(self, metrics: Dict[str, float]) -> List[Alert]:
        """모든 메트릭 평가"""
        triggered_alerts = []
        for name, value in metrics.items():
            alerts = self.evaluate(name, value)
            triggered_alerts.extend(alerts)
        return triggered_alerts

    def _notify(self, alert: Alert) -> None:
        """알림 전송"""
        for handler in self._handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                logger.error(f'Alert handler failed: {e}')

    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        with self._lock:
            return list(self._alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """알림 이력 조회"""
        with self._lock:
            return list(self._alert_history)[-limit:]

    def silence_alert(self, name: str, duration: timedelta = None) -> None:
        """알림 일시 중지"""
        # 구현: 알림 규칙에 silence 플래그 추가
        pass

    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        with self._lock:
            return {
                'rules_count': len(self._rules),
                'handlers_count': len(self._handlers),
                'active_alerts': len(self._alerts),
                'history_size': len(self._alert_history),
                'active_alert_names': list(self._alerts.keys())
            }


def create_alert_manager() -> AlertManager:
    """알림 매니저 생성"""
    manager = AlertManager()

    # 기본 알림 규칙 추가
    manager.add_rule(ThresholdRule(
        name='high_prediction_latency',
        threshold=1.0,
        comparison='gt',
        level=AlertLevel.WARNING,
        labels={'metric': 'prediction_latency'},
        message_template='Prediction latency too high: {value:.2f}s > {threshold}s'
    ))

    manager.add_rule(ThresholdRule(
        name='high_cpu_usage',
        threshold=90.0,
        comparison='gt',
        level=AlertLevel.WARNING,
        labels={'metric': 'cpu_usage'},
        for_duration=timedelta(minutes=5),
        message_template='CPU usage too high: {value:.1f}% > {threshold}%'
    ))

    manager.add_rule(ThresholdRule(
        name='high_memory_usage',
        threshold=85.0,
        comparison='gt',
        level=AlertLevel.WARNING,
        labels={'metric': 'memory_usage'},
        for_duration=timedelta(minutes=5),
        message_template='Memory usage too high: {value:.1f}% > {threshold}%'
    ))

    manager.add_rule(ThresholdRule(
        name='low_model_accuracy',
        threshold=0.8,
        comparison='lt',
        level=AlertLevel.ERROR,
        labels={'metric': 'model_r2'},
        message_template='Model accuracy too low: R²={value:.4f} < {threshold}'
    ))

    manager.add_rule(ThresholdRule(
        name='high_mape',
        threshold=10.0,
        comparison='gt',
        level=AlertLevel.WARNING,
        labels={'metric': 'model_mape'},
        message_template='MAPE too high: {value:.2f}% > {threshold}%'
    ))

    manager.add_rule(AnomalyRule(
        name='prediction_anomaly',
        window_size=100,
        std_threshold=3.0,
        level=AlertLevel.WARNING,
        labels={'metric': 'prediction_value'}
    ))

    return manager
