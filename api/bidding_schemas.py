"""
Bidding API Schemas
====================

입찰 전략 및 수익 시뮬레이션을 위한 Pydantic 모델 정의
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================
# Enums
# ============================================================

class RiskLevel(str, Enum):
    """리스크 수준"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class EnergyType(str, Enum):
    """발전 유형"""
    SOLAR = "solar"
    WIND = "wind"
    HYBRID = "hybrid"


# ============================================================
# Bidding Strategy Schemas
# ============================================================

class BiddingStrategyRequest(BaseModel):
    """입찰 전략 요청"""
    model_config = ConfigDict(protected_namespaces=())

    capacity_kw: float = Field(
        ...,
        gt=0,
        description="설비 용량 (kW)"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.SOLAR,
        description="발전 유형"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.MODERATE,
        description="리스크 허용도"
    )
    location: Optional[Dict[str, float]] = Field(
        default=None,
        description="위치 정보 {'latitude': 33.5, 'longitude': 126.5}"
    )
    prediction_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="예측 시간"
    )


class BiddingHourDetail(BaseModel):
    """시간별 입찰 상세"""
    hour: int = Field(..., ge=1, le=24)
    smp_predicted: float = Field(..., description="예측 SMP (원/kWh)")
    smp_lower: float = Field(..., description="SMP 하한 (Q10)")
    smp_upper: float = Field(..., description="SMP 상한 (Q90)")
    generation_kw: float = Field(..., description="예상 발전량 (kW)")
    expected_revenue: float = Field(..., description="예상 수익 (원)")
    rank: int = Field(..., description="수익 순위")
    recommended: bool = Field(..., description="추천 여부")


class BiddingStrategyResponse(BaseModel):
    """입찰 전략 응답"""
    risk_level: RiskLevel
    recommended_hours: List[int] = Field(..., description="추천 입찰 시간대")
    total_hours: int
    total_generation_kwh: float = Field(..., description="총 예상 발전량 (kWh)")
    total_revenue: float = Field(..., description="총 예상 수익 (원)")
    average_smp: float = Field(..., description="평균 SMP (원/kWh)")
    revenue_per_kwh: float = Field(..., description="kWh당 수익 (원)")
    hourly_details: List[BiddingHourDetail]
    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# Revenue Simulation Schemas
# ============================================================

class RevenueSimulationRequest(BaseModel):
    """수익 시뮬레이션 요청"""
    capacity_kw: float = Field(..., gt=0, description="설비 용량 (kW)")
    energy_type: EnergyType = Field(default=EnergyType.SOLAR)
    hours: int = Field(default=24, ge=1, le=168)
    scenarios: Optional[List[str]] = Field(
        default=None,
        description="시나리오 목록 (기본: q10, q50, q90)"
    )


class ScenarioResult(BaseModel):
    """시나리오별 결과"""
    scenario_name: str
    total_revenue: float
    average_hourly: float
    best_hour: int
    worst_hour: int


class RevenueSimulationResponse(BaseModel):
    """수익 시뮬레이션 응답"""
    expected_revenue: float = Field(..., description="기대 수익")
    best_case: float = Field(..., description="최선 시나리오")
    worst_case: float = Field(..., description="최악 시나리오")
    risk_adjusted: float = Field(..., description="리스크 조정 수익")
    revenue_range: float = Field(..., description="수익 범위")
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# Generation Prediction Schemas
# ============================================================

class GenerationPredictionRequest(BaseModel):
    """발전량 예측 요청"""
    capacity_kw: float = Field(..., gt=0, description="설비 용량 (kW)")
    energy_type: EnergyType = Field(default=EnergyType.SOLAR)
    hours: int = Field(default=24, ge=1, le=168)
    weather: Optional[Dict[str, float]] = Field(
        default=None,
        description="기상 조건 {'temperature': 25, 'cloud_cover': 30, 'wind_speed': 5}"
    )
    location: Optional[Dict[str, float]] = Field(
        default=None,
        description="위치 정보 {'latitude': 33.5, 'longitude': 126.5}"
    )


class GenerationPredictionPoint(BaseModel):
    """발전량 예측 포인트"""
    hour: int = Field(..., ge=1, le=24)
    timestamp: datetime
    generation_kw: float = Field(..., description="예상 발전량 (kW)")
    capacity_factor: float = Field(..., description="이용률 (%)")
    uncertainty: Optional[float] = Field(None, description="불확실성 (%)")


class GenerationPredictionResponse(BaseModel):
    """발전량 예측 응답"""
    energy_type: EnergyType
    capacity_kw: float
    predictions: List[GenerationPredictionPoint]
    summary: Dict[str, float] = Field(
        default_factory=dict,
        description="요약 (총 발전량, 평균 이용률 등)"
    )
    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# Combined Request for Dashboard
# ============================================================

class FullBiddingAnalysisRequest(BaseModel):
    """종합 입찰 분석 요청 (대시보드용)"""
    capacity_kw: float = Field(..., gt=0)
    energy_type: EnergyType = Field(default=EnergyType.SOLAR)
    risk_level: RiskLevel = Field(default=RiskLevel.MODERATE)
    location: Optional[Dict[str, float]] = None
    weather: Optional[Dict[str, float]] = None
    include_simulation: bool = Field(default=True)
    include_generation: bool = Field(default=True)


class FullBiddingAnalysisResponse(BaseModel):
    """종합 입찰 분석 응답"""
    strategy: BiddingStrategyResponse
    simulation: Optional[RevenueSimulationResponse] = None
    generation: Optional[GenerationPredictionResponse] = None
    generated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# Error Response
# ============================================================

class BiddingErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
