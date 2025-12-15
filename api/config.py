"""
API Configuration
==================

환경 변수 및 설정 관리
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API 설정"""

    # pydantic-settings v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # API 기본 설정
    APP_NAME: str = "Power Demand Forecast API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "제주도 전력 수요 예측 서비스"
    DEBUG: bool = False

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False

    # 모델 설정
    MODEL_DIR: Optional[str] = None
    DEFAULT_MODEL: str = "conditional"
    SEQ_LENGTH: int = 168

    # CORS 설정
    CORS_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # 타임아웃 설정
    REQUEST_TIMEOUT: int = 30  # seconds
    PREDICTION_TIMEOUT: int = 60  # seconds

    @property
    def model_path(self) -> Path:
        """모델 디렉토리 경로"""
        if self.MODEL_DIR:
            return Path(self.MODEL_DIR)
        # 기본 경로: 프로젝트 루트의 models/production
        project_root = Path(__file__).parent.parent
        return project_root / "models" / "production"

    @property
    def cors_origins_list(self) -> list:
        """CORS origins 리스트"""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤 반환"""
    return Settings()


# 편의를 위한 직접 접근
settings = get_settings()
