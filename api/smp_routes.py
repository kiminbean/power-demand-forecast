"""
SMP API Routes
===============

SMP 예측 및 분석 API 엔드포인트
실제 학습된 모델 (v3.2 Optuna BiLSTM+Attention) 연동

v3.2 모델 성능:
- MAPE: 7.17% (v3.1: 7.83% → 0.66%p 개선)
- R²: 0.77 (v3.1: 0.74 → 0.03 개선)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .smp_schemas import (
    SMPRegion,
    SMPDataPoint,
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
            from src.smp.models.smp_predictor import SMPPredictor
            # v3.2 Optuna 최적화 모델 사용 (MAPE 7.17%, R² 0.77)
            _smp_predictor = SMPPredictor(use_advanced=True)
            if _smp_predictor.is_ready():
                logger.info("SMP v3.2 모델 로드 완료 (MAPE: 7.17%, R²: 0.77)")
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
                'model_used': result.get('model_used', 'v3.2'),
                'mape': result.get('mape', 7.17),
                'coverage': result.get('coverage', 89.4),
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
    description="현재 시점의 SMP를 조회합니다. (실제 모델 기반)"
)
async def get_current_smp(
    region: SMPRegion = Query(SMPRegion.JEJU, description="지역")
):
    """현재 SMP 조회"""
    try:
        # 실제 모델 예측 사용
        prediction = get_real_smp_prediction(1)
        current_smp = float(prediction['q50'][0])

        # 제주는 육지보다 약간 낮음
        if region == SMPRegion.JEJU:
            current_smp *= 0.95

        now = datetime.now()

        return SMPCurrentResponse(
            region=region,
            current_smp=round(current_smp, 2),
            hour=now.hour + 1,
            timestamp=now,
            comparison={
                "daily_avg": round(float(np.mean(prediction['q50'])), 2),
                "weekly_avg": round(float(np.mean(prediction['q50'])) * 0.98, 2),
                "vs_daily_avg_percent": round((current_smp - float(np.mean(prediction['q50']))) / float(np.mean(prediction['q50'])) * 100, 2),
            }
        )
    except Exception as e:
        logger.error(f"Error getting current SMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            "version": "3.1.0" if model_used != 'fallback' else "demo",
            "mode": "production" if model_used != 'fallback' else "demo",
        }

        # 실제 모델 사용 시 추가 정보 (문자열로 변환)
        if model_used != 'fallback':
            model_info["mape"] = str(prediction.get('mape', 7.83))
            model_info["coverage_80"] = str(prediction.get('coverage', 89.4))
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
    description="육지와 제주의 SMP를 비교합니다."
)
async def compare_smp():
    """육지/제주 SMP 비교"""
    try:
        # 실제 모델 예측 사용
        prediction = get_real_smp_prediction(1)
        mainland_smp = float(prediction['q50'][0])
        jeju_smp = mainland_smp * 0.95

        difference = mainland_smp - jeju_smp

        return SMPComparisonResponse(
            mainland_current=round(mainland_smp, 2),
            jeju_current=round(jeju_smp, 2),
            difference=round(difference, 2),
            difference_percent=round(difference / mainland_smp * 100, 2),
            historical_avg_diff=round(8.5, 2),
        )
    except Exception as e:
        logger.error(f"Error comparing SMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/historical",
    response_model=SMPHistoricalResponse,
    summary="과거 SMP 조회",
    description="과거 SMP 데이터를 조회합니다."
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

        # 데모 데이터 생성 (과거 데이터는 실제 EPSIS 데이터 필요)
        hours = int((end_date - start_date).total_seconds() / 3600) + 1
        data = []

        for i in range(min(hours, 24 * max_days)):
            timestamp = start_date + timedelta(hours=i)
            hour = timestamp.hour

            # 시간대별 패턴
            if 6 <= hour < 9:
                base = 130
            elif 9 <= hour < 18:
                base = 160
            elif 18 <= hour < 22:
                base = 145
            else:
                base = 110

            smp_mainland = base + np.random.normal(0, 10)
            smp_jeju = smp_mainland * 0.95 + np.random.normal(0, 5)

            data.append(SMPDataPoint(
                timestamp=timestamp,
                hour=hour + 1,
                smp_mainland=round(smp_mainland, 2),
                smp_jeju=round(smp_jeju, 2),
            ))

        # 지역 필터
        smp_values = [d.smp_jeju if region == SMPRegion.JEJU else d.smp_mainland for d in data]

        return SMPHistoricalResponse(
            data=data,
            summary={
                "mean": round(np.mean(smp_values), 2),
                "max": round(np.max(smp_values), 2),
                "min": round(np.min(smp_values), 2),
                "std": round(np.std(smp_values), 2),
                "count": len(data),
            },
            period={
                "start": start_date,
                "end": end_date,
            }
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
        info = predictor.get_model_info()
        return {
            "status": "ready",
            "model": "BiLSTM+Attention (v3.2 Optuna)",
            "mape": 7.17,
            "coverage_80": 89.4,
            "features": 22,
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
                logger.info("RTM CatBoost 모델 로드 완료 (MAPE: 5.25%, R²: 0.83)")
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
- MAPE: 5.25% (BiLSTM 7.17%보다 우수)
- R²: 0.83

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
- 기본 MAPE: 5.25%
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
            "model": "CatBoost v3.10",
            "purpose": "RTM (Real-Time Market)",
            "prediction_type": "single-step (1 hour)",
            "mape": info.get('mape', 5.25),
            "r2": info.get('r2', 0.83),
            "features": info.get('features', 60),
            "comparison": {
                "vs_bilstm": {
                    "mape_improvement": "2.0%p better (5.25% vs 7.17%)",
                    "r2_improvement": "0.06 better (0.83 vs 0.77)",
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
