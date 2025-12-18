"""
SMP API Schemas
================

SMP 예측 및 분석을 위한 Pydantic 모델 정의
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================
# Enums
# ============================================================

class SMPRegion(str, Enum):
    """SMP 지역"""
    MAINLAND = "mainland"
    JEJU = "jeju"


class QuantileLevel(str, Enum):
    """Quantile 수준"""
    Q10 = "q10"
    Q50 = "q50"
    Q90 = "q90"


# ============================================================
# SMP Data Schemas
# ============================================================

class SMPDataPoint(BaseModel):
    """SMP 데이터 포인트"""
    timestamp: datetime
    hour: int = Field(..., ge=1, le=24, description="시간 (1-24)")
    smp_mainland: float = Field(..., description="육지 SMP (원/kWh)")
    smp_jeju: float = Field(..., description="제주 SMP (원/kWh)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-12-18T14:00:00",
                "hour": 14,
                "smp_mainland": 165.5,
                "smp_jeju": 158.2
            }
        }
    )


class SMPCurrentResponse(BaseModel):
    """현재 SMP 응답"""
    region: SMPRegion
    current_smp: float = Field(..., description="현재 SMP (원/kWh)")
    hour: int
    timestamp: datetime
    comparison: Dict[str, float] = Field(
        default_factory=dict,
        description="비교 정보 (일평균, 주평균 등)"
    )


# ============================================================
# SMP Prediction Schemas
# ============================================================

class SMPPredictionRequest(BaseModel):
    """SMP 예측 요청"""
    model_config = ConfigDict(protected_namespaces=())

    region: SMPRegion = Field(
        default=SMPRegion.JEJU,
        description="예측 대상 지역"
    )
    hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="예측 시간 (1-168시간)"
    )
    include_confidence: bool = Field(
        default=True,
        description="신뢰구간 포함 여부"
    )
    model_type: Optional[str] = Field(
        default=None,
        description="모델 타입 (lstm, tft, ensemble)"
    )


class SMPPredictionPoint(BaseModel):
    """SMP 예측 포인트"""
    hour: int = Field(..., description="시간 (1-24)")
    timestamp: datetime
    predicted_smp: float = Field(..., description="예측 SMP (원/kWh)")
    lower_bound: Optional[float] = Field(None, description="하한 (Q10)")
    upper_bound: Optional[float] = Field(None, description="상한 (Q90)")
    confidence: Optional[float] = Field(None, description="신뢰도")


class SMPPredictionResponse(BaseModel):
    """SMP 예측 응답"""
    region: SMPRegion
    predictions: List[SMPPredictionPoint]
    summary: Dict[str, float] = Field(
        default_factory=dict,
        description="요약 통계 (평균, 최대, 최소 등)"
    )
    model_info: Dict[str, str] = Field(
        default_factory=dict,
        description="사용된 모델 정보"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="예측 생성 시간"
    )


# ============================================================
# SMP Analysis Schemas
# ============================================================

class SMPHistoricalRequest(BaseModel):
    """과거 SMP 조회 요청"""
    start_date: datetime
    end_date: datetime
    region: Optional[SMPRegion] = Field(
        default=None,
        description="지역 (None이면 전체)"
    )


class SMPHistoricalResponse(BaseModel):
    """과거 SMP 조회 응답"""
    data: List[SMPDataPoint]
    summary: Dict[str, float]
    period: Dict[str, datetime]


class SMPComparisonResponse(BaseModel):
    """SMP 비교 응답"""
    mainland_current: float
    jeju_current: float
    difference: float = Field(..., description="육지 - 제주 차이")
    difference_percent: float = Field(..., description="차이 비율 (%)")
    historical_avg_diff: float = Field(..., description="과거 평균 차이")


# ============================================================
# Error Response
# ============================================================

class SMPErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
