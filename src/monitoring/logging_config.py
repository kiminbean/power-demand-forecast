"""
구조화 로깅 설정 (Task 18)
===========================
JSON 형식의 구조화된 로깅과 로그 컨텍스트를 제공합니다.
"""

import logging
import logging.handlers
import json
import sys
import traceback
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
import os


class LogLevel:
    """로그 레벨 상수"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class JSONFormatter(logging.Formatter):
    """JSON 형식 포매터"""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_thread: bool = False,
        include_process: bool = False,
        extra_fields: Dict[str, Any] = None
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_thread = include_thread
        self.include_process = include_process
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON으로 포맷"""
        log_data = {}

        # 기본 필드
        if self.include_timestamp:
            log_data['timestamp'] = datetime.fromtimestamp(record.created).isoformat()

        if self.include_level:
            log_data['level'] = record.levelname

        if self.include_logger:
            log_data['logger'] = record.name

        # 메시지
        log_data['message'] = record.getMessage()

        # 스레드/프로세스 정보
        if self.include_thread:
            log_data['thread'] = record.threadName
            log_data['thread_id'] = record.thread

        if self.include_process:
            log_data['process'] = record.processName
            log_data['process_id'] = record.process

        # 위치 정보
        log_data['module'] = record.module
        log_data['function'] = record.funcName
        log_data['line'] = record.lineno

        # 예외 정보
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }

        # 컨텍스트 정보 추가
        if hasattr(record, 'context') and record.context:
            log_data['context'] = record.context

        # 추가 필드
        for key, value in self.extra_fields.items():
            log_data[key] = value

        # extra에서 추가 필드 가져오기
        if hasattr(record, 'extra_data') and record.extra_data:
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """컬러 콘솔 포매터"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: str = None):
        super().__init__(fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def format(self, record: logging.LogRecord) -> str:
        """컬러 포맷 적용"""
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        return super().format(record)


class LogContext:
    """스레드 로컬 로그 컨텍스트"""

    _context = threading.local()

    @classmethod
    def set(cls, **kwargs) -> None:
        """컨텍스트 설정"""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        cls._context.data.update(kwargs)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """컨텍스트 조회"""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        return cls._context.data.copy()

    @classmethod
    def clear(cls) -> None:
        """컨텍스트 초기화"""
        cls._context.data = {}

    @classmethod
    @contextmanager
    def scope(cls, **kwargs):
        """컨텍스트 스코프"""
        old_data = cls.get()
        cls.set(**kwargs)
        try:
            yield
        finally:
            cls._context.data = old_data


class ContextFilter(logging.Filter):
    """컨텍스트 필터"""

    def filter(self, record: logging.LogRecord) -> bool:
        """로그 레코드에 컨텍스트 추가"""
        record.context = LogContext.get()
        return True


class StructuredLogger:
    """구조화된 로거"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addFilter(ContextFilter())

    def _log(
        self,
        level: int,
        message: str,
        extra: Dict[str, Any] = None,
        exc_info: bool = False
    ) -> None:
        """로그 기록"""
        extra_data = extra or {}
        record = self._logger.makeRecord(
            self.name,
            level,
            '',
            0,
            message,
            (),
            exc_info if exc_info else None,
        )
        record.extra_data = extra_data
        self._logger.handle(record)

    def debug(self, message: str, **kwargs) -> None:
        """DEBUG 로그"""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs) -> None:
        """INFO 로그"""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """WARNING 로그"""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """ERROR 로그"""
        self._log(logging.ERROR, message, kwargs, exc_info)

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """CRITICAL 로그"""
        self._log(logging.CRITICAL, message, kwargs, exc_info)

    def exception(self, message: str, **kwargs) -> None:
        """예외 로그"""
        self._log(logging.ERROR, message, kwargs, exc_info=True)


@dataclass
class LogConfig:
    """로그 설정"""
    level: int = logging.INFO
    format: str = 'json'  # 'json' or 'text'
    output: str = 'both'  # 'console', 'file', or 'both'
    log_dir: str = './logs'
    app_name: str = 'power-demand-forecast'
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_thread: bool = False
    include_process: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict)


def setup_logging(config: LogConfig = None) -> None:
    """
    로깅 설정

    Args:
        config: 로그 설정 (None이면 기본값 사용)
    """
    if config is None:
        config = LogConfig()

    # 로그 디렉토리 생성
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 포매터 생성
    if config.format == 'json':
        formatter = JSONFormatter(
            include_thread=config.include_thread,
            include_process=config.include_process,
            extra_fields={
                'app': config.app_name,
                **config.extra_fields
            }
        )
    else:
        formatter = ColoredFormatter()

    # 콘솔 핸들러
    if config.output in ['console', 'both']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(ContextFilter())
        root_logger.addHandler(console_handler)

    # 파일 핸들러
    if config.output in ['file', 'both']:
        # 일반 로그
        log_file = log_dir / f'{config.app_name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ContextFilter())
        root_logger.addHandler(file_handler)

        # 에러 로그 (별도 파일)
        error_file = log_dir / f'{config.app_name}.error.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(ContextFilter())
        root_logger.addHandler(error_handler)

    # 컨텍스트 필터 추가
    root_logger.addFilter(ContextFilter())

    logging.info(f'Logging configured: level={logging.getLevelName(config.level)}, format={config.format}')


def create_logger(name: str, level: int = None) -> StructuredLogger:
    """구조화된 로거 생성"""
    return StructuredLogger(name, level or logging.INFO)


def log_execution(logger: StructuredLogger = None, level: int = logging.INFO):
    """함수 실행 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            _logger = logger or create_logger(func.__module__)
            func_name = f'{func.__module__}.{func.__name__}'

            _logger.info(f'Starting {func_name}', function=func_name)
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                _logger.info(
                    f'Completed {func_name}',
                    function=func_name,
                    elapsed_seconds=elapsed
                )
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                _logger.exception(
                    f'Failed {func_name}: {e}',
                    function=func_name,
                    elapsed_seconds=elapsed,
                    error=str(e)
                )
                raise

        return wrapper
    return decorator


class RequestLogger:
    """HTTP 요청 로깅"""

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or create_logger('http')

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: str = None,
        **kwargs
    ) -> None:
        """요청 로깅"""
        self.logger.info(
            f'{method} {path} {status_code}',
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            request_id=request_id,
            **kwargs
        )


class MetricsLogger:
    """메트릭 로깅"""

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or create_logger('metrics')

    def log_metric(
        self,
        name: str,
        value: float,
        unit: str = None,
        tags: Dict[str, str] = None
    ) -> None:
        """메트릭 로깅"""
        self.logger.info(
            f'Metric: {name}={value}',
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            metric_tags=tags or {}
        )


class AuditLogger:
    """감사 로깅"""

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or create_logger('audit')

    def log_action(
        self,
        action: str,
        user: str = None,
        resource: str = None,
        details: Dict[str, Any] = None
    ) -> None:
        """작업 로깅"""
        self.logger.info(
            f'Audit: {action}',
            audit_action=action,
            audit_user=user,
            audit_resource=resource,
            audit_details=details or {}
        )
