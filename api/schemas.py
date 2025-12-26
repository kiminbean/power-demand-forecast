"""
API Schemas (Pydantic Models)
=============================

요청/응답 데이터 검증을 위한 Pydantic 모델 정의
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================
# Enums
# ============================================================

class ModelType(str, Enum):
    """사용 가능한 모델 타입"""
    DEMAND_ONLY = "demand_only"
    WEATHER_FULL = "weather_full"
    CONDITIONAL = "conditional"


class ConditionalMode(str, Enum):
    """Conditional 모델 모드"""
    SOFT = "soft"
    HARD = "hard"


# ============================================================
# Input Schemas
# ============================================================

class TimeSeriesData(BaseModel):
    """시계열 데이터 포인트"""
    datetime: datetime
    power_demand: float = Field(..., description="전력 수요 (MW)")
    temperature: Optional[float] = Field(None, alias="기온", description="기온 (°C)")
    humidity: Optional[float] = Field(None, alias="습도", description="습도 (%)")
    wind_speed: Optional[float] = Field(None, alias="풍속", description="풍속 (m/s)")
    precipitation: Optional[float] = Field(None, alias="강수량", description="강수량 (mm)")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "datetime": "2024-01-15T14:00:00",
                "power_demand": 850.5,
                "기온": 5.2,
                "습도": 65.0,
                "풍속": 3.5,
                "강수량": 0.0
            }
        }


class PredictionRequest(BaseModel):
    """단일 예측 요청"""
    model_config = ConfigDict(protected_namespaces=())

    data: List[TimeSeriesData] = Field(
        ...,
        min_length=168,
        description="시계열 데이터 (최소 168시간 = 7일)"
    )
    model_type: ModelType = Field(
        default=ModelType.DEMAND_ONLY,
        description="사용할 모델 타입"
    )

    @field_validator('data')
    @classmethod
    def validate_data_length(cls, v):
        if len(v) < 168:
            raise ValueError(f"최소 168개의 데이터 포인트가 필요합니다. (현재: {len(v)}개)")
        return v


class ConditionalPredictionRequest(BaseModel):
    """조건부 예측 요청"""
    data: List[TimeSeriesData] = Field(
        ...,
        min_length=168,
        description="시계열 데이터 (최소 168시간)"
    )
    mode: ConditionalMode = Field(
        default=ConditionalMode.SOFT,
        description="예측 모드: soft (확률적 블렌딩) 또는 hard (이진 선택)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"datetime": "2024-01-08T00:00:00", "power_demand": 750.0, "기온": 3.5},
                ],
                "mode": "soft"
            }
        }


class BatchPredictionRequest(BaseModel):
    """배치 예측 요청"""
    model_config = ConfigDict(protected_namespaces=())

    data: List[TimeSeriesData] = Field(
        ...,
        min_length=168,
        description="시계열 데이터"
    )
    model_type: ModelType = Field(
        default=ModelType.DEMAND_ONLY,
        description="사용할 모델 타입"
    )
    step: int = Field(
        default=1,
        ge=1,
        le=24,
        description="슬라이딩 윈도우 스텝 (1-24)"
    )


# ============================================================
# Output Schemas
# ============================================================

class PredictionResponse(BaseModel):
    """단일 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    prediction: float = Field(..., description="예측 전력 수요 (MW)")
    model_used: str = Field(..., description="사용된 모델")
    timestamp: datetime = Field(..., description="예측 대상 시간")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")


class ConditionalPredictionResponse(BaseModel):
    """조건부 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    prediction: float = Field(..., description="예측 전력 수요 (MW)")
    model_used: str = Field(..., description="사용된 모델/모드")
    timestamp: datetime = Field(..., description="예측 대상 시간")
    context: Dict = Field(..., description="예측 컨텍스트 정보")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="요청 성공 여부")
    predictions: List[Dict] = Field(..., description="예측 결과 목록")
    model_used: str = Field(..., description="사용된 모델")
    total_predictions: int = Field(..., description="총 예측 수")
    statistics: Dict = Field(..., description="예측 통계")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    models_loaded: bool = Field(..., description="모델 로드 여부")
    device: str = Field(..., description="연산 디바이스")
    uptime_seconds: float = Field(..., description="서비스 가동 시간 (초)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "device": "mps",
                "uptime_seconds": 3600.5
            }
        }


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    models: List[Dict] = Field(..., description="로드된 모델 목록")
    default_model: str = Field(..., description="기본 모델")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "demand_only",
                        "type": "LSTM",
                        "n_features": 25,
                        "seq_length": 168,
                        "status": "loaded"
                    },
                    {
                        "name": "weather_full",
                        "type": "LSTM",
                        "n_features": 35,
                        "seq_length": 168,
                        "status": "loaded"
                    }
                ],
                "default_model": "conditional"
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답"""
    success: bool = Field(default=False, description="요청 성공 여부")
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "INVALID_DATA",
                "error_message": "입력 데이터가 유효하지 않습니다",
                "detail": "최소 168개의 데이터 포인트가 필요합니다"
            }
        }


# ============================================================
# Daily Prediction Schemas (BiLSTM v18)
# ============================================================

class DailyTimeSeriesData(BaseModel):
    """일간 시계열 데이터 포인트"""
    date: datetime = Field(..., description="날짜")
    power_mwh: float = Field(..., description="일간 전력 수요 (MWh)")
    avg_temp: Optional[float] = Field(None, description="평균 기온 (°C)")
    min_temp: Optional[float] = Field(None, description="최저 기온 (°C)")
    max_temp: Optional[float] = Field(None, description="최고 기온 (°C)")
    humidity: Optional[float] = Field(None, description="습도 (%)")
    sunlight: Optional[float] = Field(None, description="일조량 (h)")
    dew_point: Optional[float] = Field(None, description="이슬점 (°C)")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-15",
                "power_mwh": 20500.5,
                "avg_temp": 5.2,
                "min_temp": -1.0,
                "max_temp": 10.5,
                "humidity": 65.0,
                "sunlight": 6.5
            }
        }


class DailyPredictionRequest(BaseModel):
    """일간 예측 요청"""
    data: List[DailyTimeSeriesData] = Field(
        ...,
        min_length=7,
        description="일간 시계열 데이터 (최소 7일)"
    )

    @field_validator('data')
    @classmethod
    def validate_data_length(cls, v):
        if len(v) < 7:
            raise ValueError(f"최소 7일의 데이터가 필요합니다. (현재: {len(v)}일)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"date": "2024-01-08", "power_mwh": 19800.0, "avg_temp": 3.5},
                    {"date": "2024-01-09", "power_mwh": 20100.0, "avg_temp": 4.2},
                    {"date": "2024-01-10", "power_mwh": 20300.0, "avg_temp": 5.0},
                    {"date": "2024-01-11", "power_mwh": 19500.0, "avg_temp": 2.8},
                    {"date": "2024-01-12", "power_mwh": 18200.0, "avg_temp": 1.5},
                    {"date": "2024-01-13", "power_mwh": 17800.0, "avg_temp": 0.8},
                    {"date": "2024-01-14", "power_mwh": 20200.0, "avg_temp": 3.2}
                ]
            }
        }


class DailyPredictionResponse(BaseModel):
    """일간 예측 응답"""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "success": True,
                "prediction": 20850.5,
                "prediction_date": "2024-01-15",
                "model_used": "BiLSTM (v18)",
                "model_info": {
                    "input_days": 7,
                    "features_used": 31,
                    "device": "mps"
                },
                "processing_time_ms": 15.32
            }
        }
    )

    success: bool = Field(..., description="요청 성공 여부")
    prediction: float = Field(..., description="예측 일간 전력 수요 (MWh)")
    prediction_date: datetime = Field(..., description="예측 대상 날짜")
    model_used: str = Field(..., description="사용된 모델")
    model_info: Dict = Field(..., description="모델 메타데이터")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")
