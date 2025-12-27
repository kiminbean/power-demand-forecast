"""
Power Demand Forecast API
==========================

FastAPI 기반 전력 수요 예측 REST API

Usage:
------
# 개발 모드
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

Author: Power Demand Forecast Team
Date: 2024-12
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .schemas import (
    PredictionRequest,
    ConditionalPredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    ConditionalPredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    ModelType,
    DailyPredictionRequest,
    DailyPredictionResponse,
)
from .service import get_prediction_service, initialize_service, PredictionService
from .daily_predictor import get_daily_predictor, initialize_daily_predictor, DailyPredictor

# SMP/Bidding 라우터 (v2.0)
try:
    from .smp_routes import router as smp_router
    from .bidding_routes import router as bidding_router
    SMP_ROUTES_AVAILABLE = True
except ImportError as e:
    SMP_ROUTES_AVAILABLE = False
    smp_router = None
    bidding_router = None

# v6 Dashboard 라우터
try:
    from .v6_routes import router as v6_router
    V6_ROUTES_AVAILABLE = True
except ImportError as e:
    V6_ROUTES_AVAILABLE = False
    v6_router = None

# Power Plant 라우터 (v6.2.0)
try:
    from .power_plant_routes import router as power_plant_router
    POWER_PLANT_ROUTES_AVAILABLE = True
except ImportError as e:
    POWER_PLANT_ROUTES_AVAILABLE = False
    power_plant_router = None

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan (Startup/Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    try:
        service = initialize_service()
        logger.info(f"Hourly models loaded successfully")
        logger.info(f"Device: {service.get_device()}")
    except Exception as e:
        logger.error(f"Failed to initialize hourly service: {e}")
        # 서비스 초기화 실패해도 앱은 시작 (healthcheck에서 상태 확인 가능)

    # Daily BiLSTM model initialization
    try:
        daily_predictor = initialize_daily_predictor()
        if daily_predictor.is_ready():
            logger.info(f"Daily BiLSTM model loaded successfully")
        else:
            logger.warning("Daily BiLSTM model not available")
    except Exception as e:
        logger.error(f"Failed to initialize daily predictor: {e}")

    logger.info(f"API server ready at http://{settings.HOST}:{settings.PORT}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down API server...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS.split(",") if settings.CORS_ALLOW_METHODS != "*" else ["*"],
    allow_headers=settings.CORS_ALLOW_HEADERS.split(",") if settings.CORS_ALLOW_HEADERS != "*" else ["*"],
)

# SMP/Bidding 라우터 등록 (v2.0)
if SMP_ROUTES_AVAILABLE:
    app.include_router(smp_router)
    app.include_router(bidding_router)

# v6 Dashboard 라우터 등록
if V6_ROUTES_AVAILABLE:
    app.include_router(v6_router)
    logger.info("v6 Dashboard routes registered")

# Power Plant 라우터 등록 (v6.2.0)
if POWER_PLANT_ROUTES_AVAILABLE:
    app.include_router(power_plant_router)
    logger.info("Power Plant routes registered")


# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.perf_counter()

    # 요청 처리
    response = await call_next(request)

    # 처리 시간 계산
    process_time = (time.perf_counter() - start_time) * 1000

    # 로깅
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.2f}ms"
    )

    # 헤더에 처리 시간 추가
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    return response


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """ValueError 처리"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="INVALID_VALUE",
            error_message=str(exc),
            detail=None
        ).model_dump()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """RuntimeError 처리"""
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error_code="SERVICE_UNAVAILABLE",
            error_message=str(exc),
            detail="Service may not be fully initialized"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="내부 서버 오류가 발생했습니다",
            detail=str(exc) if settings.DEBUG else None
        ).model_dump()
    )


# ============================================================
# Dependencies
# ============================================================

def get_service() -> PredictionService:
    """예측 서비스 의존성 (시간별)"""
    service = get_prediction_service()
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again later."
        )
    return service


def get_daily_service() -> DailyPredictor:
    """일간 예측 서비스 의존성"""
    predictor = get_daily_predictor()
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Daily prediction model not available. Please check model file."
        )
    return predictor


# ============================================================
# Routes - Health & Info
# ============================================================

@app.get(
    "/",
    summary="API 루트",
    description="API 정보 반환"
)
async def root():
    """API 루트 엔드포인트"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="헬스체크",
    description="서비스 상태 확인"
)
async def health_check():
    """서비스 상태 확인"""
    service = get_prediction_service()

    return HealthResponse(
        status="healthy" if service.is_ready() else "unhealthy",
        version=__version__,
        models_loaded=service.is_ready(),
        device=service.get_device(),
        uptime_seconds=round(service.get_uptime(), 2)
    )


@app.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="모델 정보",
    description="로드된 모델 정보 조회"
)
async def get_models(service: PredictionService = Depends(get_service)):
    """모델 정보 조회"""
    return ModelInfoResponse(
        models=service.get_model_info(),
        default_model=settings.DEFAULT_MODEL
    )


# ============================================================
# Routes - Prediction
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        503: {"model": ErrorResponse, "description": "서비스 불가"}
    },
    summary="단일 예측",
    description="시계열 데이터를 기반으로 다음 시간의 전력 수요를 예측합니다."
)
async def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_service)
):
    """
    단일 예측 엔드포인트

    - **data**: 시계열 데이터 (최소 168개 = 7일)
    - **model_type**: 사용할 모델 (demand_only, weather_full, conditional)
    """
    try:
        result = service.predict(
            data=request.data,
            model_type=request.model_type
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/conditional",
    response_model=ConditionalPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        503: {"model": ErrorResponse, "description": "서비스 불가"}
    },
    summary="조건부 예측",
    description="겨울철에는 기상 데이터를 활용한 최적화된 예측을 수행합니다."
)
async def predict_conditional(
    request: ConditionalPredictionRequest,
    service: PredictionService = Depends(get_service)
):
    """
    조건부 예측 엔드포인트

    겨울철(12월, 1월, 2월)에는 기상 데이터를 활용하여
    더 정확한 예측을 제공합니다.

    - **data**: 시계열 데이터 (기상 데이터 포함 권장)
    - **mode**: soft (확률적 블렌딩) 또는 hard (이진 선택)
    """
    try:
        result = service.predict_conditional(
            data=request.data,
            mode=request.mode
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Conditional prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        503: {"model": ErrorResponse, "description": "서비스 불가"}
    },
    summary="배치 예측",
    description="슬라이딩 윈도우를 사용한 배치 예측을 수행합니다."
)
async def predict_batch(
    request: BatchPredictionRequest,
    service: PredictionService = Depends(get_service)
):
    """
    배치 예측 엔드포인트

    슬라이딩 윈도우 방식으로 여러 시점의 예측을 한 번에 수행합니다.

    - **data**: 시계열 데이터
    - **model_type**: 사용할 모델
    - **step**: 슬라이딩 윈도우 스텝 (1-24)
    """
    try:
        result = service.predict_batch(
            data=request.data,
            model_type=request.model_type,
            step=request.step
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Routes - Daily Prediction (BiLSTM v18)
# ============================================================

@app.post(
    "/predict/daily",
    response_model=DailyPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        503: {"model": ErrorResponse, "description": "서비스 불가"}
    },
    summary="일간 예측",
    description="BiLSTM 모델을 사용한 일간 전력 수요 예측 (7일 시퀀스 기반)"
)
async def predict_daily(
    request: DailyPredictionRequest,
    predictor: DailyPredictor = Depends(get_daily_service)
):
    """
    일간 전력 수요 예측 엔드포인트

    BiLSTM v18 모델을 사용하여 다음 날의 전력 수요를 예측합니다.
    - MAPE: 6.17%
    - R²: 0.726

    - **data**: 일간 시계열 데이터 (최소 7일)
    """
    import time
    import pandas as pd
    from datetime import timedelta

    start_time = time.perf_counter()

    try:
        # Convert request data to DataFrame
        records = []
        for item in request.data:
            record = {
                'power_mwh': item.power_mwh,
            }
            if item.avg_temp is not None:
                record['avg_temp'] = item.avg_temp
            if item.min_temp is not None:
                record['min_temp'] = item.min_temp
            if item.max_temp is not None:
                record['max_temp'] = item.max_temp
            if item.humidity is not None:
                record['humidity'] = item.humidity
            if item.sunlight is not None:
                record['sunlight'] = item.sunlight
            if item.dew_point is not None:
                record['dew_point'] = item.dew_point
            records.append(record)

        # Create DataFrame with date index
        dates = [item.date for item in request.data]
        df = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
        df.sort_index(inplace=True)

        # Run prediction
        prediction, metadata = predictor.predict(df)

        # Calculate prediction date (next day after last data point)
        last_date = df.index[-1]
        prediction_date = last_date + timedelta(days=1)

        processing_time = (time.perf_counter() - start_time) * 1000

        return DailyPredictionResponse(
            success=True,
            prediction=round(prediction, 2),
            prediction_date=prediction_date,
            model_used=metadata.get('model', 'BiLSTM (v18)'),
            model_info=metadata,
            processing_time_ms=round(processing_time, 2)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Daily prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI 진입점"""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
