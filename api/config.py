"""
API Configuration
==================

환경 변수 및 설정 관리
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schemas import ModelType


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
    AR_GBM_MODEL_DIR: Optional[str] = None
    # model_type을 생략한 /predict·/predict/batch 요청이 쓰는 모델 (ModelType 값).
    # 2025년 시간외 비교에서 ar_gbm이 서빙 LSTM보다 정확했다 (docs/2026-11-06-demand-retrain-raw-target.md 12절)
    DEFAULT_MODEL: str = ModelType.AR_GBM.value
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

    @field_validator("DEFAULT_MODEL", mode="before")
    @classmethod
    def _validate_default_model(cls, value):
        # 환경변수 끝의 줄바꿈·공백을 지우고, 없는 모델 이름이면 시작할 때 바로 실패한다.
        # (str, Enum) 멤버의 str()은 'ModelType.AR_GBM'이라 값을 직접 꺼낸다
        text = (value.value if isinstance(value, ModelType) else str(value)).strip()
        allowed = sorted(m.value for m in ModelType)
        if text not in allowed:
            raise ValueError(f"DEFAULT_MODEL must be one of {allowed}, got {text!r}")
        return text

    @property
    def model_path(self) -> Path:
        """모델 디렉토리 경로"""
        if self.MODEL_DIR:
            return Path(self.MODEL_DIR)
        # 기본 경로: 프로젝트 루트의 models/production
        project_root = Path(__file__).parent.parent
        return project_root / "models" / "production"

    @property
    def ar_gbm_model_path(self) -> Path:
        """선택형 AR + GBM 모델 디렉토리 경로 (운영 기본 모델과 따로 둔다)"""
        # 환경변수 끝의 줄바꿈·공백이 경로에 섞이지 않게 한다
        if self.AR_GBM_MODEL_DIR and self.AR_GBM_MODEL_DIR.strip():
            return Path(self.AR_GBM_MODEL_DIR.strip())
        return Path(__file__).parent.parent / "models" / "demand_ar_gbm"

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
