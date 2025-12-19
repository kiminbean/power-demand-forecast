"""
RE-BMS API Schemas
==================

Pydantic models for API request/response validation.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Bid Schemas
# ============================================================================

class BidSegmentCreate(BaseModel):
    """Schema for creating a bid segment"""
    segment_id: int = Field(..., ge=1, le=10, description="Segment ID (1-10)")
    quantity_mw: float = Field(..., ge=0, description="Quantity in MW")
    price_krw_mwh: float = Field(..., ge=0, description="Price in KRW/MWh")

    @field_validator('price_krw_mwh')
    @classmethod
    def validate_price_cap(cls, v):
        if v > 500:  # Market cap
            raise ValueError('Price exceeds market cap (500 KRW/MWh)')
        return v


class BidSegmentResponse(BaseModel):
    """Schema for bid segment response"""
    segment_id: int
    quantity_mw: float
    price_krw_mwh: float
    cumulative_mw: float = 0.0


class HourlyBidCreate(BaseModel):
    """Schema for creating an hourly bid"""
    hour: int = Field(..., ge=1, le=24, description="Hour (1-24)")
    segments: List[BidSegmentCreate] = Field(
        ...,
        max_length=10,
        description="Bid segments (max 10)"
    )


class HourlyBidResponse(BaseModel):
    """Schema for hourly bid response"""
    hour: int
    segments: List[BidSegmentResponse]
    total_capacity_mw: float
    weighted_avg_price: float


class DailyBidCreate(BaseModel):
    """Schema for creating a daily bid"""
    resource_id: str = Field(..., description="Resource ID")
    market_type: str = Field(default="day_ahead", description="day_ahead or real_time")
    trading_date: date = Field(..., description="Trading date")
    hourly_bids: List[HourlyBidCreate] = Field(..., description="24 hourly bids")
    risk_level: str = Field(default="moderate", description="conservative/moderate/aggressive")

    @field_validator('market_type')
    @classmethod
    def validate_market_type(cls, v):
        if v not in ['day_ahead', 'real_time']:
            raise ValueError('market_type must be day_ahead or real_time')
        return v

    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        if v not in ['conservative', 'moderate', 'aggressive']:
            raise ValueError('risk_level must be conservative, moderate, or aggressive')
        return v


class DailyBidResponse(BaseModel):
    """Schema for daily bid response"""
    bid_id: str
    resource_id: str
    market_type: str
    trading_date: str
    hourly_bids: List[HourlyBidResponse]
    status: str
    created_at: str
    updated_at: Optional[str] = None
    submitted_at: Optional[str] = None
    kpx_reference_id: Optional[str] = None
    risk_level: str
    smp_forecast_used: bool
    total_daily_mwh: float
    average_price: float


class BidOptimizeRequest(BaseModel):
    """Schema for bid optimization request"""
    use_smp_forecast: bool = Field(default=True, description="Use AI SMP forecast")
    risk_level: str = Field(default="moderate", description="Risk level")


class BidSubmitResponse(BaseModel):
    """Schema for bid submission response"""
    status: str
    bid_id: str
    kpx_reference_id: Optional[str] = None
    submitted_at: str


# ============================================================================
# Resource Schemas
# ============================================================================

class ResourceCreate(BaseModel):
    """Schema for creating a resource"""
    resource_id: str
    name: str
    resource_type: str = Field(..., description="solar, wind, solar_ess, or wind_ess")
    installed_capacity_mw: float = Field(..., gt=0)
    latitude: float = Field(default=33.489)
    longitude: float = Field(default=126.498)
    region: str = Field(default="jeju")

    @field_validator('resource_type')
    @classmethod
    def validate_resource_type(cls, v):
        if v not in ['solar', 'wind', 'solar_ess', 'wind_ess']:
            raise ValueError('Invalid resource type')
        return v


class ResourceResponse(BaseModel):
    """Schema for resource response"""
    resource_id: str
    name: str
    resource_type: str
    installed_capacity_mw: float
    current_output_mw: float
    availability_percent: float
    effective_capacity_mw: float
    utilization_rate: float
    latitude: float
    longitude: float
    region: str
    connection_status: str
    capacity_factor: float
    curtailment_rate: float


class PortfolioResponse(BaseModel):
    """Schema for portfolio response"""
    portfolio_id: str
    name: str
    operator_id: str
    resources: List[ResourceResponse]
    total_capacity_mw: float
    current_output_mw: float
    utilization_rate: float
    resource_count: int


# ============================================================================
# Settlement Schemas
# ============================================================================

class HourlySettlementResponse(BaseModel):
    """Schema for hourly settlement response"""
    hour: int
    bid_quantity_mw: float
    actual_generation_mw: float
    cleared_quantity_mw: float
    imbalance_mw: float
    deviation_percent: float
    clearing_price_krw: float
    generation_revenue_krw: float
    imbalance_charge_krw: float
    net_revenue_krw: float
    penalty_type: str


class DailySettlementResponse(BaseModel):
    """Schema for daily settlement response"""
    settlement_id: str
    resource_id: str
    trading_date: str
    bid_id: str
    hourly_settlements: List[HourlySettlementResponse]
    total_generation_revenue: float
    total_imbalance_charges: float
    net_daily_revenue: float
    total_bid_mwh: float
    total_actual_mwh: float
    accuracy_percent: float
    average_clearing_price: float
    hours_over_generation: int
    hours_under_generation: int
    hours_no_penalty: int
    is_finalized: bool


# ============================================================================
# Market Schemas
# ============================================================================

class MarketStatusResponse(BaseModel):
    """Schema for market status response"""
    current_time: str
    dam: Dict[str, Any]
    rtm: Dict[str, Any]


class SMPForecastResponse(BaseModel):
    """Schema for SMP forecast response"""
    q10: List[float]
    q50: List[float]
    q90: List[float]
    model_used: str
    confidence: float
    created_at: str


# ============================================================================
# Validation Schemas
# ============================================================================

class ValidationErrorDetail(BaseModel):
    """Schema for validation error detail"""
    type: str
    message: str
    hour: Optional[int] = None


class ValidationResponse(BaseModel):
    """Schema for validation response"""
    is_valid: bool
    total_errors: int
    errors: List[ValidationErrorDetail]
    bid_id: str


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Schema for error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    version: str
    uptime: float
    active_connections: int
