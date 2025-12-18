"""
Bidding API Routes
===================

입찰 전략 및 수익 시뮬레이션 API 엔드포인트
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from .bidding_schemas import (
    RiskLevel,
    EnergyType,
    BiddingStrategyRequest,
    BiddingHourDetail,
    BiddingStrategyResponse,
    RevenueSimulationRequest,
    ScenarioResult,
    RevenueSimulationResponse,
    GenerationPredictionRequest,
    GenerationPredictionPoint,
    GenerationPredictionResponse,
    FullBiddingAnalysisRequest,
    FullBiddingAnalysisResponse,
    BiddingErrorResponse,
)

# SMP 모듈 import (optional)
try:
    from src.smp.bidding import (
        BiddingStrategyOptimizer,
        RevenueCalculator,
        RiskAnalyzer,
    )
    SMP_BIDDING_AVAILABLE = True
except ImportError:
    SMP_BIDDING_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/bidding",
    tags=["Bidding"],
    responses={
        500: {"model": BiddingErrorResponse, "description": "Internal Server Error"}
    }
)


# ============================================================
# 데모 데이터 생성기
# ============================================================

def generate_demo_generation(
    capacity_kw: float,
    energy_type: str,
    hours: int = 24
) -> np.ndarray:
    """데모용 발전량 데이터 생성"""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_hour = base_time.hour

    if energy_type == 'solar':
        pattern = np.array([
            0, 0, 0, 0, 0.05, 0.15,
            0.35, 0.55, 0.75, 0.85, 0.92, 0.95,
            0.95, 0.90, 0.80, 0.65, 0.45, 0.20,
            0.05, 0, 0, 0, 0, 0
        ])
    else:  # wind
        pattern = np.array([
            0.45, 0.48, 0.52, 0.55, 0.58, 0.55,
            0.50, 0.45, 0.42, 0.40, 0.38, 0.35,
            0.32, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.58, 0.55, 0.52, 0.50, 0.48
        ])

    pattern_shifted = np.roll(pattern, -start_hour)[:hours]
    noise = np.random.normal(0, 0.05, hours)

    generation = capacity_kw * np.clip(pattern_shifted + noise, 0, 1)
    return generation


def generate_demo_smp(hours: int = 24) -> dict:
    """데모용 SMP 데이터 생성"""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

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

@router.post(
    "/strategy",
    response_model=BiddingStrategyResponse,
    summary="입찰 전략 추천",
    description="SMP 예측과 발전량 예측을 기반으로 최적 입찰 전략을 추천합니다."
)
async def get_bidding_strategy(request: BiddingStrategyRequest):
    """입찰 전략 추천"""
    try:
        hours = request.prediction_hours

        # 데이터 생성
        smp_data = generate_demo_smp(hours)
        energy = 'solar' if request.energy_type == EnergyType.SOLAR else 'wind'
        if request.energy_type == EnergyType.HYBRID:
            solar_gen = generate_demo_generation(request.capacity_kw * 0.6, 'solar', hours)
            wind_gen = generate_demo_generation(request.capacity_kw * 0.4, 'wind', hours)
            generation = solar_gen + wind_gen
        else:
            generation = generate_demo_generation(request.capacity_kw, energy, hours)

        # 리스크 수준에 따른 SMP 선택
        risk_map = {
            RiskLevel.CONSERVATIVE: smp_data['q10'],
            RiskLevel.MODERATE: smp_data['q50'],
            RiskLevel.AGGRESSIVE: smp_data['q90'],
        }
        smp_values = risk_map.get(request.risk_level, smp_data['q50'])

        # 시간별 분석
        hourly_details = []
        for i in range(hours):
            gen_kw = generation[i]
            smp = smp_values[i]
            revenue = gen_kw * smp

            hourly_details.append({
                'hour': i + 1,
                'smp_predicted': round(smp_data['q50'][i], 2),
                'smp_lower': round(smp_data['q10'][i], 2),
                'smp_upper': round(smp_data['q90'][i], 2),
                'generation_kw': round(gen_kw, 2),
                'expected_revenue': round(revenue, 0),
            })

        # 수익 순위
        sorted_hours = sorted(
            enumerate(hourly_details),
            key=lambda x: x[1]['expected_revenue'],
            reverse=True
        )

        for rank, (idx, _) in enumerate(sorted_hours, 1):
            hourly_details[idx]['rank'] = rank

        # 추천 시간대 (상위 50%)
        top_count = max(1, hours // 2)
        recommended_indices = {idx for idx, _ in sorted_hours[:top_count]}

        for idx, detail in enumerate(hourly_details):
            detail['recommended'] = idx in recommended_indices

        recommended_hours = [i + 1 for i in recommended_indices]
        recommended_hours.sort()

        # 통계 계산
        total_generation = sum(d['generation_kw'] for d in hourly_details if d['recommended'])
        total_revenue = sum(d['expected_revenue'] for d in hourly_details if d['recommended'])
        avg_smp = np.mean([smp_values[i] for i in recommended_indices])
        revenue_per_kwh = total_revenue / total_generation if total_generation > 0 else 0

        return BiddingStrategyResponse(
            risk_level=request.risk_level,
            recommended_hours=recommended_hours,
            total_hours=hours,
            total_generation_kwh=round(total_generation, 2),
            total_revenue=round(total_revenue, 0),
            average_smp=round(avg_smp, 2),
            revenue_per_kwh=round(revenue_per_kwh, 2),
            hourly_details=[BiddingHourDetail(**d) for d in hourly_details],
            generated_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error getting bidding strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/simulate",
    response_model=RevenueSimulationResponse,
    summary="수익 시뮬레이션",
    description="다양한 시나리오에서 예상 수익을 시뮬레이션합니다."
)
async def simulate_revenue(request: RevenueSimulationRequest):
    """수익 시뮬레이션"""
    try:
        hours = request.hours

        # 데이터 생성
        smp_data = generate_demo_smp(hours)
        energy = 'solar' if request.energy_type == EnergyType.SOLAR else 'wind'
        generation = generate_demo_generation(request.capacity_kw, energy, hours)

        # 시나리오별 수익 계산
        scenarios_data = {
            'q10': smp_data['q10'],
            'q50': smp_data['q50'],
            'q90': smp_data['q90'],
        }

        scenario_results = []
        for name, smp_values in scenarios_data.items():
            hourly_revenue = [generation[i] * smp_values[i] for i in range(hours)]
            total = sum(hourly_revenue)
            avg = np.mean(hourly_revenue)
            best_hour = np.argmax(hourly_revenue) + 1
            worst_hour = np.argmin(hourly_revenue) + 1

            scenario_results.append(ScenarioResult(
                scenario_name=name,
                total_revenue=round(total, 0),
                average_hourly=round(avg, 2),
                best_hour=best_hour,
                worst_hour=worst_hour,
            ))

        # 요약 통계
        revenues = [s.total_revenue for s in scenario_results]
        expected = scenario_results[1].total_revenue  # q50

        return RevenueSimulationResponse(
            expected_revenue=round(expected, 0),
            best_case=round(max(revenues), 0),
            worst_case=round(min(revenues), 0),
            risk_adjusted=round(expected * 0.9, 0),
            revenue_range=round(max(revenues) - min(revenues), 0),
            scenarios=scenario_results,
            generated_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error simulating revenue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generation/predict",
    response_model=GenerationPredictionResponse,
    summary="발전량 예측",
    description="태양광/풍력 발전량을 예측합니다."
)
async def predict_generation(request: GenerationPredictionRequest):
    """발전량 예측"""
    try:
        hours = request.hours
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        energy = 'solar' if request.energy_type == EnergyType.SOLAR else 'wind'
        generation = generate_demo_generation(request.capacity_kw, energy, hours)

        predictions = []
        for i in range(hours):
            timestamp = base_time + timedelta(hours=i)
            gen_kw = generation[i]
            capacity_factor = gen_kw / request.capacity_kw * 100 if request.capacity_kw > 0 else 0

            predictions.append(GenerationPredictionPoint(
                hour=(timestamp.hour % 24) + 1,
                timestamp=timestamp,
                generation_kw=round(gen_kw, 2),
                capacity_factor=round(capacity_factor, 2),
                uncertainty=round(5 + np.random.uniform(0, 5), 2),
            ))

        total_gen = sum(p.generation_kw for p in predictions)
        avg_cf = np.mean([p.capacity_factor for p in predictions])

        return GenerationPredictionResponse(
            energy_type=request.energy_type,
            capacity_kw=request.capacity_kw,
            predictions=predictions,
            summary={
                "total_generation_kwh": round(total_gen, 2),
                "average_capacity_factor": round(avg_cf, 2),
                "peak_hour": max(predictions, key=lambda x: x.generation_kw).hour,
                "peak_generation_kw": round(max(p.generation_kw for p in predictions), 2),
            },
            generated_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error predicting generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze",
    response_model=FullBiddingAnalysisResponse,
    summary="종합 입찰 분석",
    description="SMP 예측, 발전량 예측, 입찰 전략을 종합적으로 분석합니다."
)
async def full_bidding_analysis(request: FullBiddingAnalysisRequest):
    """종합 입찰 분석"""
    try:
        # 입찰 전략
        strategy_request = BiddingStrategyRequest(
            capacity_kw=request.capacity_kw,
            energy_type=request.energy_type,
            risk_level=request.risk_level,
            location=request.location,
        )
        strategy = await get_bidding_strategy(strategy_request)

        # 수익 시뮬레이션 (선택)
        simulation = None
        if request.include_simulation:
            sim_request = RevenueSimulationRequest(
                capacity_kw=request.capacity_kw,
                energy_type=request.energy_type,
            )
            simulation = await simulate_revenue(sim_request)

        # 발전량 예측 (선택)
        generation = None
        if request.include_generation:
            gen_request = GenerationPredictionRequest(
                capacity_kw=request.capacity_kw,
                energy_type=request.energy_type,
                weather=request.weather,
                location=request.location,
            )
            generation = await predict_generation(gen_request)

        return FullBiddingAnalysisResponse(
            strategy=strategy,
            simulation=simulation,
            generation=generation,
            generated_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error in full bidding analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="Bidding API 상태 확인"
)
async def bidding_health():
    """Bidding API 상태 확인"""
    return {
        "status": "healthy",
        "service": "bidding",
        "smp_module_available": SMP_BIDDING_AVAILABLE,
        "mode": "demo" if not SMP_BIDDING_AVAILABLE else "production",
        "timestamp": datetime.now().isoformat(),
    }
