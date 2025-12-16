"""
Prometheus 메트릭 수집기 (Task 18)
==================================
시스템 및 예측 성능 메트릭을 Prometheus 형식으로 수집합니다.
"""

import time
import threading
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    SUMMARY = 'summary'


@dataclass
class MetricValue:
    """메트릭 값"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """히스토그램 버킷"""
    le: float
    count: int = 0


class Counter:
    """카운터 메트릭"""

    def __init__(self, name: str, description: str = '', labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """카운터 증가"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] += value

    def get(self, labels: Dict[str, str] = None) -> float:
        """카운터 값 조회"""
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0.0)

    def _get_label_key(self, labels: Dict[str, str] = None) -> tuple:
        """라벨 키 생성"""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """메트릭 수집"""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                result.append(MetricValue(
                    name=self.name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.COUNTER
                ))
        return result


class Gauge:
    """게이지 메트릭"""

    def __init__(self, name: str, description: str = '', labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """게이지 값 설정"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """게이지 증가"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + value

    def dec(self, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """게이지 감소"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) - value

    def get(self, labels: Dict[str, str] = None) -> float:
        """게이지 값 조회"""
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0.0)

    def _get_label_key(self, labels: Dict[str, str] = None) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """메트릭 수집"""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                result.append(MetricValue(
                    name=self.name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.GAUGE
                ))
        return result


class Histogram:
    """히스토그램 메트릭"""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str = '',
        labels: List[str] = None,
        buckets: List[float] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._data: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {
            'buckets': {b: 0 for b in self.buckets + [float('inf')]},
            'sum': 0.0,
            'count': 0
        })
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """값 관측"""
        label_key = self._get_label_key(labels)
        with self._lock:
            data = self._data[label_key]
            data['sum'] += value
            data['count'] += 1
            for bucket in self.buckets + [float('inf')]:
                if value <= bucket:
                    data['buckets'][bucket] += 1

    def _get_label_key(self, labels: Dict[str, str] = None) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """메트릭 수집"""
        result = []
        with self._lock:
            for label_key, data in self._data.items():
                base_labels = dict(label_key) if label_key else {}

                # 버킷별 카운트
                for bucket, count in data['buckets'].items():
                    bucket_labels = {**base_labels, 'le': str(bucket)}
                    result.append(MetricValue(
                        name=f'{self.name}_bucket',
                        value=count,
                        labels=bucket_labels,
                        metric_type=MetricType.HISTOGRAM
                    ))

                # 합계와 개수
                result.append(MetricValue(
                    name=f'{self.name}_sum',
                    value=data['sum'],
                    labels=base_labels,
                    metric_type=MetricType.HISTOGRAM
                ))
                result.append(MetricValue(
                    name=f'{self.name}_count',
                    value=data['count'],
                    labels=base_labels,
                    metric_type=MetricType.HISTOGRAM
                ))

        return result


class Timer:
    """타이머 컨텍스트 매니저"""

    def __init__(self, histogram: Histogram, labels: Dict[str, str] = None):
        self.histogram = histogram
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.histogram.observe(duration, self.labels)
        return False


class PredictionMetrics:
    """예측 관련 메트릭"""

    def __init__(self):
        # 예측 요청 카운터
        self.prediction_requests = Counter(
            'power_demand_prediction_requests_total',
            'Total number of prediction requests',
            labels=['location', 'horizon', 'model_type']
        )

        # 예측 레이턴시 히스토그램
        self.prediction_latency = Histogram(
            'power_demand_prediction_latency_seconds',
            'Prediction latency in seconds',
            labels=['location', 'horizon'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        # 예측 오류 카운터
        self.prediction_errors = Counter(
            'power_demand_prediction_errors_total',
            'Total number of prediction errors',
            labels=['location', 'error_type']
        )

        # 예측 값 게이지
        self.prediction_value = Gauge(
            'power_demand_prediction_value_mw',
            'Current prediction value in MW',
            labels=['location', 'horizon']
        )

        # 모델 정확도 게이지
        self.model_accuracy = Gauge(
            'power_demand_model_accuracy',
            'Model accuracy (R-squared)',
            labels=['model_type', 'horizon']
        )

        # 모델 MAPE 게이지
        self.model_mape = Gauge(
            'power_demand_model_mape',
            'Model MAPE (%)',
            labels=['model_type', 'horizon']
        )

    def record_prediction(
        self,
        location: str,
        horizon: str,
        model_type: str,
        value: float,
        latency: float
    ) -> None:
        """예측 기록"""
        labels = {'location': location, 'horizon': horizon, 'model_type': model_type}
        self.prediction_requests.inc(labels=labels)
        self.prediction_latency.observe(latency, labels={'location': location, 'horizon': horizon})
        self.prediction_value.set(value, labels={'location': location, 'horizon': horizon})

    def record_error(self, location: str, error_type: str) -> None:
        """오류 기록"""
        self.prediction_errors.inc(labels={'location': location, 'error_type': error_type})

    def update_accuracy(self, model_type: str, horizon: str, accuracy: float, mape: float) -> None:
        """정확도 업데이트"""
        labels = {'model_type': model_type, 'horizon': horizon}
        self.model_accuracy.set(accuracy, labels=labels)
        self.model_mape.set(mape, labels=labels)

    def collect(self) -> List[MetricValue]:
        """모든 메트릭 수집"""
        result = []
        result.extend(self.prediction_requests.collect())
        result.extend(self.prediction_latency.collect())
        result.extend(self.prediction_errors.collect())
        result.extend(self.prediction_value.collect())
        result.extend(self.model_accuracy.collect())
        result.extend(self.model_mape.collect())
        return result


class SystemMetrics:
    """시스템 메트릭"""

    def __init__(self):
        # CPU 사용률
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage'
        )

        # 메모리 사용률
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage'
        )

        # 메모리 사용량 (바이트)
        self.memory_bytes = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            labels=['type']
        )

        # 디스크 사용률
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            labels=['mountpoint']
        )

        # 프로세스 수
        self.process_count = Gauge(
            'system_process_count',
            'Number of running processes'
        )

        # 업타임
        self.uptime = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds'
        )

        self._start_time = time.time()

    def update(self) -> None:
        """시스템 메트릭 업데이트"""
        # CPU
        self.cpu_usage.set(psutil.cpu_percent())

        # 메모리
        mem = psutil.virtual_memory()
        self.memory_usage.set(mem.percent)
        self.memory_bytes.set(mem.used, labels={'type': 'used'})
        self.memory_bytes.set(mem.available, labels={'type': 'available'})
        self.memory_bytes.set(mem.total, labels={'type': 'total'})

        # 디스크
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                self.disk_usage.set(usage.percent, labels={'mountpoint': partition.mountpoint})
            except PermissionError:
                continue

        # 프로세스 수
        self.process_count.set(len(psutil.pids()))

        # 업타임
        self.uptime.set(time.time() - self._start_time)

    def collect(self) -> List[MetricValue]:
        """모든 메트릭 수집"""
        self.update()
        result = []
        result.extend(self.cpu_usage.collect())
        result.extend(self.memory_usage.collect())
        result.extend(self.memory_bytes.collect())
        result.extend(self.disk_usage.collect())
        result.extend(self.process_count.collect())
        result.extend(self.uptime.collect())
        return result


class MetricsCollector:
    """통합 메트릭 수집기"""

    def __init__(self):
        self.prediction_metrics = PredictionMetrics()
        self.system_metrics = SystemMetrics()
        self._custom_metrics: Dict[str, Any] = {}

    def register_counter(self, name: str, description: str = '', labels: List[str] = None) -> Counter:
        """커스텀 카운터 등록"""
        counter = Counter(name, description, labels)
        self._custom_metrics[name] = counter
        return counter

    def register_gauge(self, name: str, description: str = '', labels: List[str] = None) -> Gauge:
        """커스텀 게이지 등록"""
        gauge = Gauge(name, description, labels)
        self._custom_metrics[name] = gauge
        return gauge

    def register_histogram(
        self,
        name: str,
        description: str = '',
        labels: List[str] = None,
        buckets: List[float] = None
    ) -> Histogram:
        """커스텀 히스토그램 등록"""
        histogram = Histogram(name, description, labels, buckets)
        self._custom_metrics[name] = histogram
        return histogram

    def get_metric(self, name: str):
        """메트릭 조회"""
        return self._custom_metrics.get(name)

    def collect_all(self) -> List[MetricValue]:
        """모든 메트릭 수집"""
        result = []
        result.extend(self.prediction_metrics.collect())
        result.extend(self.system_metrics.collect())
        for metric in self._custom_metrics.values():
            result.extend(metric.collect())
        return result

    def export_prometheus(self) -> str:
        """Prometheus 형식으로 내보내기"""
        lines = []
        metrics = self.collect_all()

        for metric in metrics:
            # 라벨 포맷팅
            if metric.labels:
                labels_str = ','.join(f'{k}="{v}"' for k, v in metric.labels.items())
                line = f'{metric.name}{{{labels_str}}} {metric.value}'
            else:
                line = f'{metric.name} {metric.value}'

            lines.append(line)

        return '\n'.join(lines)

    def export_json(self) -> Dict[str, Any]:
        """JSON 형식으로 내보내기"""
        metrics = self.collect_all()
        result = {}

        for metric in metrics:
            name = metric.name
            if metric.labels:
                label_str = '_'.join(f'{k}_{v}' for k, v in sorted(metric.labels.items()))
                name = f'{name}_{label_str}'

            result[name] = {
                'value': metric.value,
                'labels': metric.labels,
                'type': metric.metric_type.value,
                'timestamp': metric.timestamp.isoformat()
            }

        return result


def create_metrics_collector() -> MetricsCollector:
    """메트릭 수집기 생성"""
    return MetricsCollector()
