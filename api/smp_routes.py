"""
SMP API Routes
===============

SMP 예측 및 분석 API 엔드포인트
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
# 데모 데이터 생성기
# ============================================================

def generate_demo_smp(hours: int = 24) -> dict:
    """데모용 SMP 데이터 생성"""
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
    }


# ============================================================
# Endpoints
# ============================================================

@router.get(
    "/current",
    response_model=SMPCurrentResponse,
    summary="현재 SMP 조회",
    description="현재 시점의 SMP를 조회합니다."
)
async def get_current_smp(
    region: SMPRegion = Query(SMPRegion.JEJU, description="지역")
):
    """현재 SMP 조회"""
    try:
        # 데모 데이터 사용
        demo_data = generate_demo_smp(1)
        current_smp = demo_data['q50'][0]

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
                "daily_avg": round(150.0, 2),
                "weekly_avg": round(148.5, 2),
                "vs_daily_avg_percent": round((current_smp - 150) / 150 * 100, 2),
            }
        )
    except Exception as e:
        logger.error(f"Error getting current SMP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict",
    response_model=SMPPredictionResponse,
    summary="SMP 예측",
    description="지정된 시간 동안의 SMP를 예측합니다."
)
async def predict_smp(request: SMPPredictionRequest):
    """SMP 예측"""
    try:
        demo_data = generate_demo_smp(request.hours)

        predictions = []
        for i in range(request.hours):
            smp = demo_data['q50'][i]
            if request.region == SMPRegion.JEJU:
                smp *= 0.95

            point = SMPPredictionPoint(
                hour=(demo_data['times'][i].hour % 24) + 1,
                timestamp=demo_data['times'][i],
                predicted_smp=round(smp, 2),
                lower_bound=round(demo_data['q10'][i], 2) if request.include_confidence else None,
                upper_bound=round(demo_data['q90'][i], 2) if request.include_confidence else None,
                confidence=0.9 if request.include_confidence else None,
            )
            predictions.append(point)

        smp_values = [p.predicted_smp for p in predictions]

        return SMPPredictionResponse(
            region=request.region,
            predictions=predictions,
            summary={
                "mean": round(np.mean(smp_values), 2),
                "max": round(np.max(smp_values), 2),
                "min": round(np.min(smp_values), 2),
                "std": round(np.std(smp_values), 2),
            },
            model_info={
                "model_type": request.model_type or "ensemble",
                "version": "2.0.0",
                "mode": "demo",
            },
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
        demo_data = generate_demo_smp(1)
        mainland_smp = demo_data['q50'][0]
        jeju_smp = mainland_smp * 0.95

        difference = mainland_smp - jeju_smp

        return SMPComparisonResponse(
            mainland_current=round(mainland_smp, 2),
            jeju_current=round(jeju_smp, 2),
            difference=round(difference, 2),
            difference_percent=round(difference / mainland_smp * 100, 2),
            historical_avg_diff=round(8.5, 2),  # 데모용 과거 평균 차이
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

        # 데모 데이터 생성
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
    "/health",
    summary="SMP API 상태 확인"
)
async def smp_health():
    """SMP API 상태 확인"""
    return {
        "status": "healthy",
        "service": "smp",
        "mode": "demo",
        "timestamp": datetime.now().isoformat(),
    }
