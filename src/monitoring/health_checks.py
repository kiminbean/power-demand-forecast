"""
헬스 체크 시스템 (Task 18)
===========================
시스템, 모델, 의존성 상태를 모니터링합니다.
"""

import time
import threading
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """헬스 상태"""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'


@dataclass
class HealthCheckResult:
    """헬스 체크 결과"""
    name: str
    status: HealthStatus
    message: str = ''
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'latency_ms': self.latency_ms,
            'checked_at': self.checked_at.isoformat()
        }


class HealthCheck:
    """헬스 체크 베이스 클래스"""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout

    def check(self) -> HealthCheckResult:
        """헬스 체크 수행"""
        start = time.time()
        try:
            result = self._perform_check()
            latency_ms = (time.time() - start) * 1000
            result.latency_ms = latency_ms
            return result
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency_ms
            )

    def _perform_check(self) -> HealthCheckResult:
        """실제 체크 수행 - 서브클래스에서 구현"""
        raise NotImplementedError


class SystemHealthCheck(HealthCheck):
    """시스템 헬스 체크"""

    def __init__(
        self,
        name: str = 'system',
        cpu_threshold: float = 90.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0
    ):
        super().__init__(name)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def _perform_check(self) -> HealthCheckResult:
        """시스템 상태 체크"""
        issues = []
        status = HealthStatus.HEALTHY

        # CPU 체크
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.cpu_threshold:
            issues.append(f'CPU usage high: {cpu_percent:.1f}%')
            status = HealthStatus.DEGRADED

        # 메모리 체크
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            issues.append(f'Memory usage high: {memory.percent:.1f}%')
            status = HealthStatus.DEGRADED

        # 디스크 체크
        disk = psutil.disk_usage('/')
        if disk.percent > self.disk_threshold:
            issues.append(f'Disk usage high: {disk.percent:.1f}%')
            status = HealthStatus.DEGRADED

        if len(issues) >= 3:
            status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            name=self.name,
            status=status,
            message='; '.join(issues) if issues else 'All systems operational',
            details={
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 ** 3)
            }
        )


class ModelHealthCheck(HealthCheck):
    """모델 헬스 체크"""

    def __init__(
        self,
        name: str = 'model',
        model_path: Path = None,
        min_accuracy: float = 0.8,
        max_latency_ms: float = 1000.0
    ):
        super().__init__(name)
        self.model_path = model_path
        self.min_accuracy = min_accuracy
        self.max_latency_ms = max_latency_ms
        self._last_prediction_time: Optional[datetime] = None
        self._last_latency_ms: float = 0.0
        self._last_accuracy: float = 1.0

    def update_metrics(
        self,
        latency_ms: float,
        accuracy: float = None
    ) -> None:
        """메트릭 업데이트"""
        self._last_prediction_time = datetime.now()
        self._last_latency_ms = latency_ms
        if accuracy is not None:
            self._last_accuracy = accuracy

    def _perform_check(self) -> HealthCheckResult:
        """모델 상태 체크"""
        issues = []
        status = HealthStatus.HEALTHY

        # 모델 파일 존재 체크
        if self.model_path and not self.model_path.exists():
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f'Model file not found: {self.model_path}'
            )

        # 최근 예측 체크
        if self._last_prediction_time:
            since_last = datetime.now() - self._last_prediction_time
            if since_last > timedelta(hours=1):
                issues.append(f'No recent predictions ({since_last.total_seconds()/3600:.1f}h)')
                status = HealthStatus.DEGRADED

        # 레이턴시 체크
        if self._last_latency_ms > self.max_latency_ms:
            issues.append(f'High latency: {self._last_latency_ms:.1f}ms')
            status = HealthStatus.DEGRADED

        # 정확도 체크
        if self._last_accuracy < self.min_accuracy:
            issues.append(f'Low accuracy: {self._last_accuracy:.4f}')
            status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            name=self.name,
            status=status,
            message='; '.join(issues) if issues else 'Model healthy',
            details={
                'last_prediction': self._last_prediction_time.isoformat() if self._last_prediction_time else None,
                'last_latency_ms': self._last_latency_ms,
                'last_accuracy': self._last_accuracy,
                'model_path': str(self.model_path) if self.model_path else None
            }
        )


class DependencyHealthCheck(HealthCheck):
    """의존성 헬스 체크"""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 5.0
    ):
        super().__init__(name, timeout)
        self.check_func = check_func

    def _perform_check(self) -> HealthCheckResult:
        """의존성 상태 체크"""
        try:
            is_healthy = self.check_func()
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message='Connected' if is_healthy else 'Connection failed'
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f'Check failed: {e}'
            )


class DatabaseHealthCheck(DependencyHealthCheck):
    """데이터베이스 헬스 체크"""

    def __init__(
        self,
        name: str = 'database',
        connection_string: str = None,
        timeout: float = 5.0
    ):
        self.connection_string = connection_string

        def check_db():
            # 간단한 연결 테스트 (실제로는 DB 드라이버 사용)
            return True

        super().__init__(name, check_db, timeout)


class RedisHealthCheck(DependencyHealthCheck):
    """Redis 헬스 체크"""

    def __init__(
        self,
        name: str = 'redis',
        host: str = 'localhost',
        port: int = 6379,
        timeout: float = 5.0
    ):
        self.host = host
        self.port = port

        def check_redis():
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except Exception:
                return False

        super().__init__(name, check_redis, timeout)


class APIHealthCheck(DependencyHealthCheck):
    """외부 API 헬스 체크"""

    def __init__(
        self,
        name: str,
        url: str,
        timeout: float = 5.0,
        expected_status: int = 200
    ):
        self.url = url
        self.expected_status = expected_status

        def check_api():
            import urllib.request
            try:
                req = urllib.request.Request(url, method='HEAD')
                response = urllib.request.urlopen(req, timeout=timeout)
                return response.status == expected_status
            except Exception:
                return False

        super().__init__(name, check_api, timeout)


class HealthChecker:
    """통합 헬스 체커"""

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

    def add_check(self, check: HealthCheck) -> None:
        """헬스 체크 추가"""
        with self._lock:
            self._checks[check.name] = check

    def remove_check(self, name: str) -> None:
        """헬스 체크 제거"""
        with self._lock:
            if name in self._checks:
                del self._checks[name]

    def run_check(self, name: str) -> HealthCheckResult:
        """특정 체크 실행"""
        with self._lock:
            if name not in self._checks:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message='Check not found'
                )

            result = self._checks[name].check()
            self._results[name] = result
            return result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """모든 체크 실행"""
        results = {}
        with self._lock:
            for name, check in self._checks.items():
                result = check.check()
                self._results[name] = result
                results[name] = result
        return results

    def get_overall_status(self) -> HealthStatus:
        """전체 상태 조회"""
        results = self.run_all_checks()

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_report(self) -> Dict[str, Any]:
        """헬스 리포트 생성"""
        results = self.run_all_checks()
        overall = self.get_overall_status()

        return {
            'status': overall.value,
            'timestamp': datetime.now().isoformat(),
            'checks': {name: result.to_dict() for name, result in results.items()},
            'summary': {
                'total': len(results),
                'healthy': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                'unknown': sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN)
            }
        }

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """마지막 결과 조회"""
        with self._lock:
            return self._results.get(name)


class HealthCheckScheduler:
    """헬스 체크 스케줄러"""

    def __init__(
        self,
        health_checker: HealthChecker,
        interval_seconds: float = 30.0
    ):
        self.health_checker = health_checker
        self.interval_seconds = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """스케줄러 시작"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f'Health check scheduler started (interval: {self.interval_seconds}s)')

    def stop(self) -> None:
        """스케줄러 중지"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info('Health check scheduler stopped')

    def _run(self) -> None:
        """스케줄러 루프"""
        while self._running:
            try:
                self.health_checker.run_all_checks()
            except Exception as e:
                logger.error(f'Health check failed: {e}')

            time.sleep(self.interval_seconds)


def create_health_checker() -> HealthChecker:
    """기본 헬스 체커 생성"""
    checker = HealthChecker()

    # 시스템 헬스 체크
    checker.add_check(SystemHealthCheck())

    # 모델 헬스 체크
    checker.add_check(ModelHealthCheck())

    return checker


def create_full_health_checker(
    redis_host: str = 'localhost',
    redis_port: int = 6379
) -> HealthChecker:
    """전체 헬스 체커 생성"""
    checker = create_health_checker()

    # Redis 헬스 체크
    checker.add_check(RedisHealthCheck(host=redis_host, port=redis_port))

    return checker
