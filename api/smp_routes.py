"""
SMP API Routes
===============

SMP 예측 및 분석 API 엔드포인트
SMP quantile 모델 (v4.1, 2026-11-05 재학습) 연동

v4.1 모델 성능(스케일러 누수 수정 후 재평가):
- MAPE: walk-forward 평균 8.76% (구버전 7%대 수치는 누수로 무효)
- R²: 0.463 (구버전 0.7대 수치는 누수로 무효)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .smp_schemas import (
    SMPRegion,

    SMPCurrentResponse,
    SMPPredictionRequest,
    SMPPredictionPoint,
    SMPPredictionResponse,
    SMPHistoricalRequest,
    SMPHistoricalResponse,
    SMPComparisonResponse,
    SMPErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/smp",
    tags=["SMP"],
    responses={
        500: {"model": SMPErrorResponse, "description": "Internal Server Error"}
    }
)


# ============================================================
# 실제 SMP 예측 모델 로드
# ============================================================

_smp_predictor = None
_model_load_attempted = False


def get_smp_predictor():
    """SMP 예측기 싱글톤 인스턴스 반환"""
    global _smp_predictor, _model_load_attempted

    if _smp_predictor is None and not _model_load_attempted:
        _model_load_attempted = True
        try:
            # 공개 API와 같은 공유 서빙 예측기 (v4.1 우선, v3.2 대체)
            from src.smp.models.serving import get_serving_smp_predictor
            _smp_predictor = get_serving_smp_predictor()
            if _smp_predictor.is_ready():
                logger.info("SMP 공유 서빙 예측기 준비 완료")
            else:
                logger.warning("SMP 모델 준비 안됨, 폴백 모드 사용")
                _smp_predictor = None
        except Exception as e:
            logger.error(f"SMP 모델 로드 실패: {e}")
            _smp_predictor = None

    return _smp_predictor


# ============================================================
# 데모 데이터 생성기 (폴백용)
# ============================================================

def generate_demo_smp(hours: int = 24) -> dict:
    """데모용 SMP 데이터 생성 (모델 실패 시 폴백)"""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    # 시간대별 SMP 패턴
    hour_factors = np.array([
        0.75, 0.72, 0.70, 0.68, 0.70, 0.75,
        0.85, 0.95, 1.05, 1.10, 1.12, 1.15,
        1.18, 1.15, 1.10, 1.05, 1.00, 1.05,
        1.10, 1.05, 0.95, 0.88, 0.82, 0.78
    ])

    start_hour = base_time.hour
    hour_factors_shifted = np.roll(hour_factors, -start_hour)[:hours]

    base_smp = 150
    noise = np.random.normal(0, 5, hours)

    q50 = base_smp * hour_factors_shifted + noise
    q10 = q50 * 0.85
    q90 = q50 * 1.15

    return {
        'times': [base_time + timedelta(hours=i) for i in range(hours)],
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'model_used': 'fallback'
    }


def get_real_smp_prediction(hours: int = 24) -> dict:
    """실제 모델 예측 또는 폴백"""
    predictor = get_smp_predictor()

    if predictor is not None and predictor.is_ready():
        try:
            result = predictor.predict_24h()
            # 요청된 시간만큼만 반환
            return {
                'times': result['times'][:hours],
                'q10': result['q10'][:hours],
                'q50': result['q50'][:hours],
                'q90': result['q90'][:hours],
                'model_used': result.get('model_used'),
                'model_version': result.get('model_version'),
                'mape': result.get('mape'),  # v3.2는 검증 지표가 없어 None
                'coverage': result.get('coverage'),
            }
        except Exception as e:
            logger.error(f"모델 예측 실패, 폴백 사용: {e}")

    return generate_demo_smp(hours)


# ============================================================
# Endpoints
# ============================================================

@router.get(
    "/current",
    response_model=SMPCurrentResponse,
    summary="현재 SMP 조회",
    description="검증된 현재 SMP 관측값을 사용할 수 없어 일시적으로 제공되지 않습니다."
)
async def get_current_smp(
    region: SMPRegion = Query(SMPRegion.JEJU, description="지역")
):
    """Reject predictions when verified current observations are unavailable."""
    raise HTTPException(status_code=503, detail="Verified current SMP data unavailable")


@router.post(
    "/predict",
    response_model=SMPPredictionResponse,
    summary="SMP 예측",
    description="지정된 시간 동안의 SMP를 예측합니다. (v3.2 BiLSTM+Attention Optuna 모델)"
)
async def predict_smp(request: SMPPredictionRequest):
    """SMP 예측 - 실제 학습된 모델 사용"""
    try:
        # 최대 24시간까지 예측 가능
        hours = min(request.hours, 24)

        # 실제 모델 예측
        prediction = get_real_smp_prediction(hours)
        model_used = prediction.get('model_used', 'fallback')

        predictions = []
        for i in range(hours):
            smp = float(prediction['q50'][i])
            q10 = float(prediction['q10'][i])
            q90 = float(prediction['q90'][i])

            if request.region == SMPRegion.JEJU:
                smp *= 0.95
                q10 *= 0.95
                q90 *= 0.95

            point = SMPPredictionPoint(
                hour=(prediction['times'][i].hour % 24) + 1,
                timestamp=prediction['times'][i],
                predicted_smp=round(smp, 2),
                lower_bound=round(q10, 2) if request.include_confidence else None,
                upper_bound=round(q90, 2) if request.include_confidence else None,
                confidence=0.8 if request.include_confidence else None,  # 80% coverage
            )
            predictions.append(point)

        smp_values = [p.predicted_smp for p in predictions]

        # 모델 정보 (Dict[str, str] 형식으로 모든 값을 문자열로 변환)
        model_info = {
            "model_type": "BiLSTM+Attention" if model_used != 'fallback' else "fallback",
            "version": str(prediction.get('model_version') or model_used) if model_used != 'fallback' else "demo",
            "mode": "production" if model_used != 'fallback' else "demo",
        }

        # 실제 모델 사용 시 추가 정보 (문자열로 변환). 검증 지표가 없으면 넣지 않는다.
        if model_used != 'fallback':
            if prediction.get('mape') is not None:
                model_info["mape"] = str(round(prediction['mape'], 2))
            if prediction.get('coverage') is not None:
                model_info["coverage_80"] = str(round(prediction['coverage'], 1))
            model_info["quantiles"] = "0.1, 0.5, 0.9"

        return SMPPredictionResponse(
            region=request.region,
            predictions=predictions,
            summary={
                "mean": round(np.mean(smp_values), 2),
                "max": round(np.max(smp_values), 2),
                "min": round(np.min(smp_values), 2),
                "std": round(np.std(smp_values), 2),
            },
            model_info=model_info,
            generated_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error predicting SMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/compare",
    response_model=SMPComparisonResponse,
    summary="육지/제주 SMP 비교",
    description="검증된 SMP 관측값이 없어 비교 결과를 제공하지 않습니다."
)
async def compare_smp():
    """Reject comparisons without verified current and historical data."""
    raise HTTPException(status_code=503, detail="Verified SMP comparison data unavailable")


@router.get(
    "/historical",
    response_model=SMPHistoricalResponse,
    summary="과거 SMP 조회",
    description="검증된 과거 SMP 데이터가 없어 일시적으로 제공되지 않습니다."
)
async def get_historical_smp(
    start_date: datetime = Query(..., description="시작일"),
    end_date: datetime = Query(..., description="종료일"),
    region: Optional[SMPRegion] = Query(None, description="지역")
):
    """과거 SMP 조회"""
    try:
        # 날짜 범위 검증
        if end_date < start_date:
            raise HTTPException(
                status_code=400,
                detail="end_date must be after start_date"
            )

        # 최대 30일
        max_days = 30
        if (end_date - start_date).days > max_days:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum date range is {max_days} days"
            )

        # 검증된 관측 자료가 없으면 임의 데이터를 반환하지 않는다.
        raise HTTPException(
            status_code=503, detail="Verified historical SMP data unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical SMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/model/info",
    summary="SMP 모델 정보",
    description="현재 사용 중인 SMP 예측 모델 정보를 반환합니다."
)
async def get_model_info():
    """SMP 모델 정보"""
    predictor = get_smp_predictor()

    if predictor is not None and predictor.is_ready():
        info = predictor.model_info()
        return {
            "status": "ready",
            "model": info.get("label"),
            "version": info.get("version"),
            "mape": info.get("mape"),
            "coverage_80": info.get("coverage"),
            "quantiles": [0.1, 0.5, 0.9],
            "details": info,
            "timestamp": datetime.now().isoformat(),
        }
    else:
        return {
            "status": "fallback",
            "model": "Demo Pattern Generator",
            "message": "실제 모델을 로드할 수 없어 폴백 모드로 동작 중",
            "timestamp": datetime.now().isoformat(),
        }


@router.get(
    "/health",
    summary="SMP API 상태 확인"
)
async def smp_health():
    """SMP API 상태 확인"""
    predictor = get_smp_predictor()
    model_ready = predictor is not None and predictor.is_ready()

    return {
        "status": "healthy",
        "service": "smp",
        "mode": "production" if model_ready else "demo",
        "model_ready": model_ready,
        "model_version": "3.2.0" if model_ready else "fallback",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# RTM (Real-Time Market) Endpoints - CatBoost Model
# ============================================================

_rtm_predictor = None
_rtm_model_load_attempted = False


def get_rtm_predictor():
    """RTM용 CatBoost 예측기 싱글톤 인스턴스 반환"""
    global _rtm_predictor, _rtm_model_load_attempted

    if _rtm_predictor is None and not _rtm_model_load_attempted:
        _rtm_model_load_attempted = True
        try:
            from src.smp.models.smp_catboost_predictor import SMPCatBoostPredictor
            _rtm_predictor = SMPCatBoostPredictor()
            if _rtm_predictor.is_ready():
                logger.info("RTM CatBoost 모델 로드 완료 (MAPE: 5.28%, R²: 0.827)")
            else:
                logger.warning("RTM CatBoost 모델 준비 안됨, 폴백 모드 사용")
                _rtm_predictor = None
        except Exception as e:
            logger.error(f"RTM CatBoost 모델 로드 실패: {e}")
            _rtm_predictor = None

    return _rtm_predictor


@router.get(
    "/rtm/predict",
    summary="RTM SMP 예측 (다음 1시간)",
    description="""
실시간시장(RTM)용 SMP 예측 - CatBoost 모델 사용

**모델 성능:**
- MAPE: 5.28% (BiLSTM 7.17%보다 우수)
- R²: 0.827

**용도:** 실시간 의사결정, 단기 예측
"""
)
async def predict_rtm_smp():
    """RTM SMP 예측 (다음 1시간)"""
    predictor = get_rtm_predictor()

    if predictor is not None and predictor.is_ready():
        try:
            result = predictor.predict_next_hour()
            return {
                "status": "success",
                "prediction": {
                    "time": result['time'].isoformat(),
                    "smp": round(result['smp'], 2),
                    "confidence_low": round(result['confidence_low'], 2),
                    "confidence_high": round(result['confidence_high'], 2),
                },
                "model": {
                    "name": result['model_used'],
                    "mape": round(result['mape'], 2),
                    "r2": round(result.get('r2', 0.83), 4),
                    "type": "single-step",
                    "purpose": "RTM",
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"RTM 예측 실패: {e}")

    # 폴백
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    next_hour = base_time + timedelta(hours=1)

    return {
        "status": "fallback",
        "prediction": {
            "time": next_hour.isoformat(),
            "smp": 100.0,
            "confidence_low": 85.0,
            "confidence_high": 115.0,
        },
        "model": {
            "name": "fallback",
            "mape": 15.0,
            "type": "pattern-based",
            "purpose": "RTM",
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.get(
    "/rtm/predict/{hours}",
    summary="RTM SMP 다중 시간 예측",
    description="""
RTM용 다중 시간 SMP 예측 - CatBoost 재귀 예측

**주의:** 재귀 예측이므로 시간이 멀어질수록 오차 누적 가능
**권장:** 1~6시간

**모델 성능:**
- 기본 MAPE: 5.28%
- 재귀 예측 시 오차 증가 가능
"""
)
async def predict_rtm_smp_multi(hours: int):
    """RTM SMP 다중 시간 예측"""
    # 유효성 검사
    if hours < 1 or hours > 12:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 12")

    predictor = get_rtm_predictor()

    if predictor is not None and predictor.is_ready():
        try:
            result = predictor.predict_hours(hours)
            return {
                "status": "success",
                "predictions": [
                    {
                        "hour": p['hour'],
                        "time": p['time'].isoformat(),
                        "smp": round(p['smp'], 2),
                        "confidence_low": round(p['confidence_low'], 2),
                        "confidence_high": round(p['confidence_high'], 2),
                    }
                    for p in result['predictions']
                ],
                "summary": {
                    "mean": round(np.mean(result['smp_values']), 2),
                    "min": round(min(result['smp_values']), 2),
                    "max": round(max(result['smp_values']), 2),
                },
                "model": {
                    "name": result['model_used'],
                    "mape": round(result['mape'], 2),
                    "type": "recursive-multi-step",
                    "purpose": "RTM",
                    "warning": result.get('warning', ''),
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"RTM 다중 예측 실패: {e}")

    # 폴백
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    predictions = []
    for h in range(1, hours + 1):
        predictions.append({
            "hour": h,
            "time": (base_time + timedelta(hours=h)).isoformat(),
            "smp": 100.0,
            "confidence_low": 85.0,
            "confidence_high": 115.0,
        })

    return {
        "status": "fallback",
        "predictions": predictions,
        "model": {"name": "fallback", "mape": 15.0},
        "timestamp": datetime.now().isoformat(),
    }


@router.get(
    "/rtm/model/info",
    summary="RTM 모델 정보",
    description="RTM용 CatBoost 모델 정보를 반환합니다."
)
async def get_rtm_model_info():
    """RTM 모델 정보"""
    predictor = get_rtm_predictor()

    if predictor is not None and predictor.is_ready():
        info = predictor.get_model_info()
        return {
            "status": "ready",
            "model": "CatBoost v3.19",
            "purpose": "RTM (Real-Time Market)",
            "prediction_type": "single-step (1 hour)",
            "mape": info.get('mape', 5.28),
            "r2": info.get('r2', 0.827),
            "features": info.get('features', 68),
            "comparison": {
                "vs_bilstm": {
                    "mape_improvement": "1.89%p better (5.28% vs 7.17%)",
                    "r2_improvement": "0.057 better (0.827 vs 0.77)",
                },
            },
            "timestamp": datetime.now().isoformat(),
        }
    else:
        return {
            "status": "fallback",
            "model": "Pattern Generator",
            "message": "CatBoost 모델을 로드할 수 없어 폴백 모드로 동작 중",
            "timestamp": datetime.now().isoformat(),
        }
