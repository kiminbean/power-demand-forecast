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
)
from .service import get_prediction_service, initialize_service, PredictionService

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
        logger.info(f"Models loaded successfully")
        logger.info(f"Device: {service.get_device()}")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        # 서비스 초기화 실패해도 앱은 시작 (healthcheck에서 상태 확인 가능)

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
    """예측 서비스 의존성"""
    service = get_prediction_service()
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again later."
        )
    return service


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
