"""
REST API 모듈 (Task 13)
======================
전력 수요 예측 서비스를 위한 REST API를 제공합니다.

주요 엔드포인트:
- /predict: 수요 예측
- /models: 모델 관리
- /data: 데이터 관리
- /health: 상태 확인
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import logging
import json

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionHorizon(str, Enum):
    """예측 시간대"""
    H1 = "1h"
    H6 = "6h"
    H12 = "12h"
    H24 = "24h"
    H48 = "48h"


class ModelType(str, Enum):
    """모델 타입"""
    LSTM = "lstm"
    TFT = "tft"
    ENSEMBLE = "ensemble"


class PredictionRequest(BaseModel):
    """예측 요청"""
    location: str = Field(default="jeju", description="예측 위치")
    horizons: List[str] = Field(default=["1h", "6h", "24h"], description="예측 시간대")
    model_type: Optional[str] = Field(default="ensemble", description="모델 타입")
    include_confidence: bool = Field(default=True, description="신뢰 구간 포함")
    features: Optional[Dict[str, float]] = Field(default=None, description="추가 피처")


class SinglePrediction(BaseModel):
    """단일 예측 결과"""
    timestamp: str
    horizon: str
    prediction: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    """예측 응답"""
    request_id: str
    location: str
    model_type: str
    created_at: str
    predictions: List[SinglePrediction]
    metadata: Dict[str, Any] = {}


class HistoricalDataRequest(BaseModel):
    """과거 데이터 요청"""
    location: str = "jeju"
    start_date: str
    end_date: str
    resolution: str = "hourly"


class HistoricalDataResponse(BaseModel):
    """과거 데이터 응답"""
    location: str
    start_date: str
    end_date: str
    data: List[Dict[str, Any]]
    count: int


class ModelInfo(BaseModel):
    """모델 정보"""
    name: str
    version: str
    type: str
    created_at: str
    metrics: Dict[str, float]
    status: str


class ModelListResponse(BaseModel):
    """모델 목록 응답"""
    models: List[ModelInfo]
    count: int


class HealthResponse(BaseModel):
    """상태 응답"""
    status: str
    version: str
    uptime: float
    models_loaded: int
    last_prediction: Optional[str] = None


class ForecastRequest(BaseModel):
    """예보 요청"""
    location: str = "jeju"
    start_time: Optional[str] = None
    hours_ahead: int = Field(default=24, ge=1, le=168)
    include_weather: bool = True


class ForecastResponse(BaseModel):
    """예보 응답"""
    location: str
    forecast_start: str
    forecast_end: str
    forecasts: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    code: str


# ============================================================================
# API Application
# ============================================================================

def create_app(
    title: str = "Jeju Power Demand Forecast API",
    version: str = "1.0.0",
    debug: bool = False
) -> FastAPI:
    """
    FastAPI 앱 생성

    Args:
        title: API 제목
        version: API 버전
        debug: 디버그 모드

    Returns:
        FastAPI 인스턴스
    """
    app = FastAPI(
        title=title,
        version=version,
        description="제주도 전력 수요 예측 API",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 상태 저장
    app.state.start_time = datetime.now()
    app.state.prediction_count = 0
    app.state.last_prediction = None
    app.state.models = {}

    return app


# 기본 앱 인스턴스
app = create_app()


# ============================================================================
# Dependencies
# ============================================================================

class PredictionService:
    """예측 서비스"""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}

    def get_model(self, model_type: str):
        """모델 조회"""
        return self._models.get(model_type)

    def predict(
        self,
        location: str,
        horizons: List[str],
        model_type: str,
        features: Optional[Dict[str, float]] = None
    ) -> List[SinglePrediction]:
        """예측 수행"""
        predictions = []
        now = datetime.now()

        for horizon in horizons:
            # 시간대 파싱
            hours = int(horizon.replace('h', ''))
            target_time = now + timedelta(hours=hours)

            # 모의 예측 (실제로는 모델 호출)
            base_demand = 1000.0  # MW
            variation = np.random.randn() * 50
            pred_value = base_demand + variation

            # 신뢰 구간
            confidence = 0.95 - hours * 0.01
            std = 30 + hours * 2
            lower = pred_value - 1.96 * std
            upper = pred_value + 1.96 * std

            predictions.append(SinglePrediction(
                timestamp=target_time.isoformat(),
                horizon=horizon,
                prediction=round(pred_value, 2),
                lower_bound=round(lower, 2),
                upper_bound=round(upper, 2),
                confidence=round(confidence, 3)
            ))

        return predictions


class DataService:
    """데이터 서비스"""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_historical_data(
        self,
        location: str,
        start_date: date,
        end_date: date,
        resolution: str = "hourly"
    ) -> List[Dict[str, Any]]:
        """과거 데이터 조회"""
        data = []
        current = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())

        delta = timedelta(hours=1) if resolution == "hourly" else timedelta(days=1)

        while current <= end:
            data.append({
                "timestamp": current.isoformat(),
                "demand": round(900 + np.random.randn() * 100, 2),
                "temperature": round(20 + np.random.randn() * 5, 1),
                "humidity": round(60 + np.random.randn() * 10, 1),
            })
            current += delta

        return data

    def get_weather_forecast(
        self,
        location: str,
        hours_ahead: int
    ) -> List[Dict[str, Any]]:
        """기상 예보 조회"""
        forecasts = []
        now = datetime.now()

        for h in range(hours_ahead):
            forecasts.append({
                "timestamp": (now + timedelta(hours=h)).isoformat(),
                "temperature": round(20 + np.sin(h * np.pi / 12) * 5, 1),
                "humidity": round(60 + np.random.randn() * 5, 1),
                "wind_speed": round(3 + np.random.randn(), 1),
            })

        return forecasts


# 서비스 인스턴스
prediction_service = PredictionService()
data_service = DataService()


def get_prediction_service() -> PredictionService:
    """예측 서비스 의존성"""
    return prediction_service


def get_data_service() -> DataService:
    """데이터 서비스 의존성"""
    return data_service


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Jeju Power Demand Forecast API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """상태 확인"""
    uptime = (datetime.now() - app.state.start_time).total_seconds()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        models_loaded=len(app.state.models),
        last_prediction=app.state.last_prediction
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    전력 수요 예측

    다중 시간대에 대한 전력 수요를 예측합니다.
    """
    import uuid

    try:
        predictions = service.predict(
            location=request.location,
            horizons=request.horizons,
            model_type=request.model_type or "ensemble",
            features=request.features
        )

        # 상태 업데이트
        app.state.prediction_count += 1
        app.state.last_prediction = datetime.now().isoformat()

        return PredictionResponse(
            request_id=str(uuid.uuid4())[:8],
            location=request.location,
            model_type=request.model_type or "ensemble",
            created_at=datetime.now().isoformat(),
            predictions=predictions,
            metadata={
                "total_predictions": app.state.prediction_count
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{horizon}", response_model=SinglePrediction, tags=["Prediction"])
async def predict_single_horizon(
    horizon: str,
    location: str = Query(default="jeju"),
    service: PredictionService = Depends(get_prediction_service)
):
    """단일 시간대 예측"""
    predictions = service.predict(
        location=location,
        horizons=[horizon],
        model_type="ensemble"
    )

    if predictions:
        return predictions[0]
    else:
        raise HTTPException(status_code=404, detail="No prediction available")


@app.get("/data/historical", response_model=HistoricalDataResponse, tags=["Data"])
async def get_historical_data(
    location: str = Query(default="jeju"),
    start_date: str = Query(..., description="시작 날짜 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="종료 날짜 (YYYY-MM-DD)"),
    resolution: str = Query(default="hourly", description="해상도 (hourly/daily)"),
    service: DataService = Depends(get_data_service)
):
    """과거 데이터 조회"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        data = service.get_historical_data(location, start, end, resolution)

        return HistoricalDataResponse(
            location=location,
            start_date=start_date,
            end_date=end_date,
            data=data,
            count=len(data)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(
    request: ForecastRequest,
    pred_service: PredictionService = Depends(get_prediction_service),
    data_service: DataService = Depends(get_data_service)
):
    """
    통합 예보

    전력 수요 예측과 기상 예보를 함께 제공합니다.
    """
    now = datetime.now()

    # 예측
    horizons = [f"{h}h" for h in [1, 6, 12, 24] if h <= request.hours_ahead]
    predictions = pred_service.predict(
        location=request.location,
        horizons=horizons,
        model_type="ensemble"
    )

    # 기상 예보
    weather = data_service.get_weather_forecast(
        request.location,
        request.hours_ahead
    )

    # 결합
    forecasts = []
    for pred in predictions:
        forecast_entry = {
            "timestamp": pred.timestamp,
            "horizon": pred.horizon,
            "demand_prediction": pred.prediction,
            "demand_lower": pred.lower_bound,
            "demand_upper": pred.upper_bound,
        }

        # 해당 시간대의 기상 데이터 추가
        horizon_hours = int(pred.horizon.replace('h', ''))
        if horizon_hours < len(weather):
            weather_data = weather[horizon_hours]
            forecast_entry.update({
                "temperature": weather_data["temperature"],
                "humidity": weather_data["humidity"],
            })

        forecasts.append(forecast_entry)

    return ForecastResponse(
        location=request.location,
        forecast_start=now.isoformat(),
        forecast_end=(now + timedelta(hours=request.hours_ahead)).isoformat(),
        forecasts=forecasts
    )


@app.get("/models", response_model=ModelListResponse, tags=["Models"])
async def list_models():
    """모델 목록 조회"""
    models = [
        ModelInfo(
            name="lstm_v1",
            version="1.0.0",
            type="lstm",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 45.2, "mape": 3.5},
            status="active"
        ),
        ModelInfo(
            name="tft_v1",
            version="1.0.0",
            type="tft",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 42.1, "mape": 3.2},
            status="active"
        ),
        ModelInfo(
            name="ensemble_v1",
            version="1.0.0",
            type="ensemble",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 40.5, "mape": 3.0},
            status="active"
        ),
    ]

    return ModelListResponse(models=models, count=len(models))


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str):
    """모델 상세 정보 조회"""
    # 모의 데이터
    models = {
        "lstm_v1": ModelInfo(
            name="lstm_v1",
            version="1.0.0",
            type="lstm",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 45.2, "mape": 3.5, "mae": 35.1},
            status="active"
        ),
        "tft_v1": ModelInfo(
            name="tft_v1",
            version="1.0.0",
            type="tft",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 42.1, "mape": 3.2, "mae": 32.8},
            status="active"
        ),
    }

    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return models[model_name]


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """서비스 메트릭 조회"""
    uptime = (datetime.now() - app.state.start_time).total_seconds()

    return {
        "uptime_seconds": uptime,
        "total_predictions": app.state.prediction_count,
        "models_loaded": len(app.state.models),
        "last_prediction": app.state.last_prediction,
        "memory_usage_mb": 0,  # 실제 구현 필요
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code="INTERNAL_ERROR"
        ).model_dump()
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """시작 이벤트"""
    logger.info("Starting Jeju Power Demand Forecast API...")
    app.state.start_time = datetime.now()


@app.on_event("shutdown")
async def shutdown_event():
    """종료 이벤트"""
    logger.info("Shutting down API...")


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """서버 실행"""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server(reload=True)
