"""
Settlement Calculator Service (KPX Integration)
================================================

실제 SMP 데이터와 제주 파일럿 정산 규정을 기반으로 정산 금액을 계산하는 서비스

KPX 정산 규정 (제주 파일럿):
- 허용 오차: ±12% (허용 범위 내 패널티 없음)
- 과발전: SMP의 80% (20% 페널티)
- 부족발전: SMP의 120% (20% 페널티)

계산 공식:
- 발전수익 = Σ (실제발전량 × SMP)
- 불균형정산 = Σ (허용오차 초과분 × 페널티요율 × SMP)
- 순수익 = 발전수익 - 불균형정산

Author: Power Demand Forecast Team
Date: 2024-12
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# KPX 제주 시범사업 정산 규정 (2024-2025 Gemini 검증)
# ============================================================
#
# [이중 정산 구조 (Double Settlement)]
# - DA (Day-Ahead): 하루전 시장 정산
# - RT (Real-Time): 실시간 시장 정산
# - 불균형 = |입찰량 - 실적량| × (RT_SMP - DA_SMP) + 페널티
#
# [허용 오차 및 페널티]
# - Tier 1: ±8% 이내 → 페널티 없음 (용량 정산금 100%)
# - Tier 2: ±8~15% → 경미한 페널티 (용량 정산금 50%)
# - Tier 3: ±15% 초과 → 강한 페널티 (용량 정산금 0%)
#
# [특이사항]
# - RT-SMP 0원 리스크: 재생에너지 집중 시 출력제어로 RT-SMP=0
# - 용량 정산금(CP): SMP 페널티보다 수익 영향 큼
# ============================================================

# Tier 1: 페널티 없음 구간 (기존 12% → 8%로 보수적 조정)
JEJU_TOLERANCE_TIER1_PERCENT = 8.0   # ±8% 허용 오차 (페널티 없음)
JEJU_TOLERANCE_TIER2_PERCENT = 15.0  # ±15% (경미한 페널티)

# 과발전/부족발전 페널티율 (DA-RT 차액 기반)
OVER_GEN_PENALTY_RATE = 0.80   # 과발전: RT-SMP 적용 (DA-SMP의 80% 수준 가정)
UNDER_GEN_PENALTY_RATE = 1.25  # 부족발전: RT-SMP + 페널티 (125%)

# 용량 정산금(CP) 계수 (오차율 구간별)
CP_COEFFICIENT_TIER1 = 1.0   # ±8% 이내: 100% 지급
CP_COEFFICIENT_TIER2 = 0.5   # ±8~15%: 50% 지급
CP_COEFFICIENT_TIER3 = 0.0   # ±15% 초과: 0% 지급

# 용량 정산금 단가 (원/MW, 시간당)
CP_PRICE_PER_MW = 5000.0  # 예시: 5,000원/MW/시간

# RT-SMP 변동성 (0원 리스크 반영)
RT_SMP_ZERO_RISK_HOURS = [11, 12, 13, 14]  # 태양광 피크 시간대
RT_SMP_DISCOUNT_FACTOR = 0.3  # 0원 리스크 시간대 할인율

# 레거시 호환성 (기존 코드 지원)
JEJU_TOLERANCE_PERCENT = JEJU_TOLERANCE_TIER1_PERCENT

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PenaltyType:
    """불균형 페널티 유형"""
    NO_PENALTY = "no_penalty"           # Tier 1: ±8% 이내 (페널티 없음)
    MILD_OVER = "mild_over"             # Tier 2: 과발전 8~15% (경미)
    MILD_UNDER = "mild_under"           # Tier 2: 부족발전 8~15% (경미)
    SEVERE_OVER = "severe_over"         # Tier 3: 과발전 15% 초과 (강함)
    SEVERE_UNDER = "severe_under"       # Tier 3: 부족발전 15% 초과 (강함)
    # 레거시 호환
    OVER_GENERATION = "over_generation"
    UNDER_GENERATION = "under_generation"


class PenaltyTier:
    """페널티 등급"""
    TIER1 = "tier1"  # ±8% 이내: 페널티 없음, CP 100%
    TIER2 = "tier2"  # ±8~15%: 경미한 페널티, CP 50%
    TIER3 = "tier3"  # ±15% 초과: 강한 페널티, CP 0%


@dataclass
class HourlySettlement:
    """시간별 정산 데이터 (KPX 제주 시범사업 이중 정산 규정 적용)"""
    timestamp: datetime
    hour: int                           # 1-24
    # 발전량
    cleared_mw: float                   # DA 낙찰량 (MW)
    actual_generation_mw: float         # 실제 발전량 (MW)
    imbalance_mw: float                 # 불균형량 (MW)
    deviation_percent: float            # 편차율 (%)
    # 가격
    da_smp_krw: float                   # DA 시장 정산가 (원/MWh)
    rt_smp_krw: float                   # RT 시장 정산가 (원/MWh)
    # 정산금
    generation_revenue_krw: float       # 발전 수익 (원)
    imbalance_charge_krw: float         # 불균형 정산금 (원)
    capacity_payment_krw: float         # 용량 정산금 (원)
    net_revenue_krw: float              # 순수익 (원)
    # 페널티 정보
    penalty_tier: str                   # 페널티 등급 (tier1/tier2/tier3)
    penalty_type: str                   # 페널티 유형
    penalty_rate: float                 # 페널티 요율
    cp_coefficient: float               # 용량 정산금 계수 (1.0/0.5/0.0)
    # RT-SMP 0원 리스크
    is_zero_risk_hour: bool             # RT-SMP 0원 리스크 시간대 여부
    # 레거시 호환
    smp_krw: float = 0.0                # DA-SMP (레거시)


@dataclass
class DailySettlement:
    """일별 정산 데이터 (KPX 제주 시범사업 이중 정산 규정 적용)"""
    date: str
    # 발전량 (MWh)
    cleared_mwh: float                  # DA 총 낙찰량
    actual_generation_mwh: float        # 총 실제 발전량
    imbalance_mwh: float                # 총 불균형량
    # 금액 (백만원)
    revenue_million: float              # 발전 수익
    imbalance_million: float            # 불균형 정산금
    capacity_payment_million: float     # 용량 정산금 (신규)
    net_revenue_million: float          # 순수익
    # 성능 지표
    accuracy_pct: float                 # 예측 정확도 (%)
    avg_da_smp: float                   # 평균 DA-SMP (원/MWh)
    avg_rt_smp: float                   # 평균 RT-SMP (원/MWh)
    avg_deviation: float                # 평균 편차 (%)
    # 페널티 Tier 통계
    hours_tier1: int                    # Tier 1 시간 수 (±8% 이내)
    hours_tier2: int                    # Tier 2 시간 수 (±8~15%)
    hours_tier3: int                    # Tier 3 시간 수 (±15% 초과)
    # RT-SMP 0원 리스크
    hours_zero_risk: int                # 0원 리스크 시간 수
    # 레거시 호환
    avg_smp: float = 0.0                # DA-SMP (레거시)
    hours_no_penalty: int = 0           # Tier 1 (레거시)
    hours_over_generation: int = 0      # 과발전 (레거시)
    hours_under_generation: int = 0     # 부족발전 (레거시)


@dataclass
class SettlementSummary:
    """정산 요약 통계 (KPX 제주 시범사업 이중 정산 규정)"""
    generation_revenue_million: float
    generation_change_pct: float
    imbalance_charges_million: float
    imbalance_change_pct: float
    capacity_payment_million: float     # 용량 정산금 (신규)
    net_revenue_million: float
    net_change_pct: float
    forecast_accuracy_pct: float
    accuracy_change_pct: float
    # 추가 통계 (KPX 규정)
    total_cleared_mwh: float = 0.0
    total_actual_mwh: float = 0.0
    avg_da_smp: float = 0.0             # 평균 DA-SMP (신규)
    avg_rt_smp: float = 0.0             # 평균 RT-SMP (신규)
    avg_deviation_pct: float = 0.0
    # Tier별 통계 (Gemini 권장)
    total_hours_tier1: int = 0          # ±8% 이내
    total_hours_tier2: int = 0          # ±8~15%
    total_hours_tier3: int = 0          # ±15% 초과
    total_hours_zero_risk: int = 0      # RT-SMP 0원 리스크
    # 레거시 호환
    total_hours_no_penalty: int = 0
    total_hours_over_gen: int = 0
    total_hours_under_gen: int = 0


class SettlementCalculator:
    """
    정산 계산 서비스

    실제 SMP 데이터와 발전량을 기반으로 정산 금액을 계산합니다.
    """

    def __init__(self):
        self._smp_data: Optional[Dict] = None
        self._portfolio_capacity_mw: float = 224.9  # 제주 포트폴리오 총 용량
        self._load_smp_data()

    def _load_smp_data(self) -> None:
        """SMP 데이터 로드"""
        try:
            import pandas as pd

            smp_file = os.path.join(PROJECT_ROOT, "data", "smp", "smp_5years_epsis.csv")

            if os.path.exists(smp_file):
                df = pd.read_csv(smp_file, encoding='utf-8-sig')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)

                # 최근 30일 데이터만 메모리에 유지
                cutoff = datetime.now() - timedelta(days=30)
                df = df[df['timestamp'] >= cutoff]

                self._smp_data = df.set_index('timestamp').to_dict('index')
                logger.info(f"SMP data loaded: {len(self._smp_data)} records")
            else:
                logger.warning(f"SMP file not found: {smp_file}")
                self._smp_data = {}

        except Exception as e:
            logger.error(f"Failed to load SMP data: {e}")
            self._smp_data = {}

    def _get_smp_for_date(self, date: datetime) -> List[float]:
        """특정 날짜의 24시간 SMP 데이터 반환"""
        import pandas as pd

        smp_values = []

        for hour in range(24):
            ts = date.replace(hour=hour, minute=0, second=0, microsecond=0)

            if self._smp_data and ts in self._smp_data:
                # 제주 SMP 우선, 없으면 육지 SMP
                smp = self._smp_data[ts].get('smp_jeju', 0)
                if smp == 0 or pd.isna(smp):
                    smp = self._smp_data[ts].get('smp_mainland', 95.0)
                smp_values.append(float(smp))
            else:
                # 데이터 없으면 시뮬레이션 (시간대별 패턴)
                base_smp = 95.0
                hour_factor = np.sin((hour - 6) * np.pi / 12) * 25
                smp_values.append(base_smp + hour_factor)

        return smp_values

    def _estimate_hourly_generation(self, date: datetime) -> List[float]:
        """
        시간별 발전량 추정 (MW)

        태양광/풍력 패턴 기반 시뮬레이션:
        - 태양광: 일출~일몰 패턴 + 날씨 영향
        - 풍력: 야간 강함 + 계절/날씨 변동
        - 일별 변동성 추가 (날씨, 계절, 요일)
        """
        generation = []

        # 포트폴리오 구성 (제주 기준)
        solar_capacity = 80.0   # MW (태양광)
        wind_capacity = 100.0   # MW (풍력)
        other_capacity = 44.9   # MW (기타)

        # 일별 변동 요소 (날짜 기반 시드로 일관성 유지)
        day_seed = date.toordinal()
        np.random.seed(day_seed)

        # 날씨 조건 시뮬레이션 (맑음/흐림/비)
        weather_condition = np.random.choice(['sunny', 'cloudy', 'rainy'], p=[0.5, 0.35, 0.15])
        weather_factors = {
            'sunny': {'solar': 1.0, 'wind': 0.7},
            'cloudy': {'solar': 0.4, 'wind': 0.9},
            'rainy': {'solar': 0.15, 'wind': 1.2}
        }
        weather = weather_factors[weather_condition]

        # 요일 효과 (주말 수요 감소 → 출력 제한 가능)
        is_weekend = date.weekday() >= 5
        weekend_factor = 0.85 if is_weekend else 1.0

        # 일별 랜덤 변동 (±30%)
        daily_variation = 0.7 + np.random.random() * 0.6

        for hour in range(24):
            # 태양광 발전 패턴 (6시~18시)
            if 6 <= hour <= 18:
                solar_factor = np.sin((hour - 6) * np.pi / 12)
                solar_output = solar_capacity * solar_factor * 0.7 * weather['solar']
            else:
                solar_output = 0

            # 풍력 발전 패턴 (야간 강함) + 날씨 영향
            wind_base = 0.3 + 0.4 * np.sin((hour + 6) * np.pi / 12)
            wind_output = wind_capacity * wind_base * weather['wind']

            # 기타 (일정)
            other_output = other_capacity * 0.6

            # 총 발전량 + 시간별 변동
            total = solar_output + wind_output + other_output
            hourly_noise = 0.85 + np.random.random() * 0.3  # ±15% 시간별 변동
            total *= daily_variation * weekend_factor * hourly_noise

            generation.append(max(0, total))

        # 시드 리셋
        np.random.seed(None)

        return generation

    def _estimate_forecast_accuracy(self) -> float:
        """예측 정확도 추정 (90~98%)"""
        return 90 + np.random.random() * 8

    def _estimate_rt_smp(self, hour: int, da_smp: float) -> float:
        """
        RT-SMP 추정 (DA-SMP 기반)

        RT-SMP는 실시간 수급 상황에 따라 DA-SMP와 다르게 형성됨.
        - 태양광 피크 시간대: 출력제어로 0원 리스크
        - 야간: DA-SMP와 유사

        Args:
            hour: 시간 (0-23)
            da_smp: DA 시장 정산가

        Returns:
            float: 추정 RT-SMP
        """
        # 0원 리스크 시간대 (태양광 피크)
        if hour in RT_SMP_ZERO_RISK_HOURS:
            # 30% 확률로 0원, 나머지는 DA-SMP의 50~80%
            np.random.seed(hour * 100)
            if np.random.random() < 0.3:
                return 0.0  # 출력제어로 0원
            else:
                return da_smp * (0.5 + np.random.random() * 0.3)

        # 일반 시간대: DA-SMP ± 10% 변동
        return da_smp * (0.9 + np.random.random() * 0.2)

    def _calculate_hourly_settlement(
        self,
        hour: int,
        timestamp: datetime,
        cleared_mw: float,
        actual_mw: float,
        da_smp: float
    ) -> HourlySettlement:
        """
        시간별 정산 계산 (KPX 제주 시범사업 이중 정산 규정)

        Gemini 토론 결과 반영:
        - Tier 1 (±8%): 페널티 없음, CP 100%
        - Tier 2 (±8~15%): 경미한 페널티, CP 50%
        - Tier 3 (±15% 초과): 강한 페널티, CP 0%
        - DA-RT 가격 차액 기반 불균형 정산
        - RT-SMP 0원 리스크 반영

        Args:
            hour: 시간 (1-24)
            timestamp: 타임스탬프
            cleared_mw: DA 낙찰량 (MW)
            actual_mw: 실제 발전량 (MW)
            da_smp: DA 시장 정산가 (원/MWh)

        Returns:
            HourlySettlement: 시간별 정산 결과
        """
        # RT-SMP 추정 (0원 리스크 포함)
        rt_smp = self._estimate_rt_smp(hour - 1, da_smp)
        is_zero_risk = hour - 1 in RT_SMP_ZERO_RISK_HOURS

        # 불균형량 계산
        imbalance_mw = actual_mw - cleared_mw

        # 편차율 계산
        if cleared_mw > 0:
            deviation_percent = abs(imbalance_mw / cleared_mw) * 100
        else:
            deviation_percent = 0.0

        # 페널티 Tier 및 유형 결정
        if deviation_percent <= JEJU_TOLERANCE_TIER1_PERCENT:
            # Tier 1: ±8% 이내 - 페널티 없음
            penalty_tier = PenaltyTier.TIER1
            penalty_type = PenaltyType.NO_PENALTY
            penalty_rate = 0.0
            cp_coefficient = CP_COEFFICIENT_TIER1

        elif deviation_percent <= JEJU_TOLERANCE_TIER2_PERCENT:
            # Tier 2: ±8~15% - 경미한 페널티
            penalty_tier = PenaltyTier.TIER2
            if imbalance_mw > 0:
                penalty_type = PenaltyType.MILD_OVER
                penalty_rate = OVER_GEN_PENALTY_RATE
            else:
                penalty_type = PenaltyType.MILD_UNDER
                penalty_rate = UNDER_GEN_PENALTY_RATE
            cp_coefficient = CP_COEFFICIENT_TIER2

        else:
            # Tier 3: ±15% 초과 - 강한 페널티
            penalty_tier = PenaltyTier.TIER3
            if imbalance_mw > 0:
                penalty_type = PenaltyType.SEVERE_OVER
                penalty_rate = OVER_GEN_PENALTY_RATE * 0.7  # 더 낮은 비율
            else:
                penalty_type = PenaltyType.SEVERE_UNDER
                penalty_rate = UNDER_GEN_PENALTY_RATE * 1.3  # 더 높은 비율
            cp_coefficient = CP_COEFFICIENT_TIER3

        # 발전 수익 계산 (DA 정산분)
        generation_revenue = Decimal(str(actual_mw * da_smp * 1000))

        # 불균형 정산금 계산 (DA-RT 차액 기반)
        if penalty_tier == PenaltyTier.TIER1:
            # Tier 1: 불균형 정산금 없음
            imbalance_charge = Decimal('0')
        else:
            # Tier 2, 3: 허용 오차 초과분에 대해 DA-RT 차액 정산
            if imbalance_mw > 0:
                # 과발전: 초과분은 RT-SMP로 정산 (DA-SMP보다 낮을 수 있음)
                excess_mw = abs(imbalance_mw) - (cleared_mw * JEJU_TOLERANCE_TIER1_PERCENT / 100)
                if excess_mw > 0:
                    # 손실 = 초과분 × (DA-SMP - RT-SMP)
                    price_diff = max(0, da_smp - rt_smp)
                    penalty_amount = excess_mw * price_diff * 1000
                    imbalance_charge = Decimal(str(penalty_amount))
                else:
                    imbalance_charge = Decimal('0')
            else:
                # 부족발전: 부족분은 RT-SMP로 Buy-back + 페널티
                shortfall_mw = abs(imbalance_mw) - (cleared_mw * JEJU_TOLERANCE_TIER1_PERCENT / 100)
                if shortfall_mw > 0:
                    # 비용 = 부족분 × RT-SMP × 페널티율
                    penalty_amount = shortfall_mw * rt_smp * (penalty_rate - 1) * 1000
                    imbalance_charge = Decimal(str(max(0, penalty_amount)))
                else:
                    imbalance_charge = Decimal('0')

        # 용량 정산금 계산
        capacity_payment = Decimal(str(cleared_mw * CP_PRICE_PER_MW * cp_coefficient))

        # 순수익 계산
        net_revenue = generation_revenue + capacity_payment - imbalance_charge

        return HourlySettlement(
            timestamp=timestamp,
            hour=hour,
            cleared_mw=cleared_mw,
            actual_generation_mw=actual_mw,
            imbalance_mw=imbalance_mw,
            deviation_percent=round(deviation_percent, 2),
            da_smp_krw=da_smp,
            rt_smp_krw=rt_smp,
            generation_revenue_krw=float(generation_revenue.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            imbalance_charge_krw=float(imbalance_charge.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            capacity_payment_krw=float(capacity_payment.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            net_revenue_krw=float(net_revenue.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            penalty_tier=penalty_tier,
            penalty_type=penalty_type,
            penalty_rate=penalty_rate,
            cp_coefficient=cp_coefficient,
            is_zero_risk_hour=is_zero_risk,
            smp_krw=da_smp  # 레거시 호환
        )

    def calculate_daily_settlement(self, date: datetime) -> DailySettlement:
        """
        일별 정산 계산 (KPX 제주 시범사업 이중 정산 규정)

        Gemini 토론 결과 반영:
        - 3-Tier 페널티 시스템
        - DA-RT 이중 정산 구조
        - 용량 정산금(CP) 포함
        - RT-SMP 0원 리스크 반영

        Args:
            date: 정산 대상 날짜

        Returns:
            DailySettlement: 일별 정산 결과
        """
        # 시간별 DA-SMP (하루전 시장)
        hourly_da_smp = self._get_smp_for_date(date)

        # 시간별 실제 발전량
        hourly_actual = self._estimate_hourly_generation(date)

        # 시간별 낙찰량 (예측량 기반)
        hourly_cleared = self._estimate_hourly_cleared(date, hourly_actual)

        # 시간별 정산 계산
        hourly_settlements = []
        for hour in range(24):
            ts = date.replace(hour=hour, minute=0, second=0, microsecond=0)
            hs = self._calculate_hourly_settlement(
                hour=hour + 1,
                timestamp=ts,
                cleared_mw=hourly_cleared[hour],
                actual_mw=hourly_actual[hour],
                da_smp=hourly_da_smp[hour]
            )
            hourly_settlements.append(hs)

        # 일별 합계 - 발전량
        total_cleared = sum(hourly_cleared)  # MWh
        total_actual = sum(hourly_actual)  # MWh
        total_imbalance = sum(abs(hs.imbalance_mw) for hs in hourly_settlements)  # MWh

        # 일별 합계 - 금액
        total_revenue = sum(hs.generation_revenue_krw for hs in hourly_settlements)  # 원
        total_imbalance_charge = sum(hs.imbalance_charge_krw for hs in hourly_settlements)  # 원
        total_cp = sum(hs.capacity_payment_krw for hs in hourly_settlements)  # 원 (신규)
        total_net = sum(hs.net_revenue_krw for hs in hourly_settlements)  # 원

        # 예측 정확도 (낙찰량 대비 실제 발전량)
        if total_cleared > 0:
            accuracy = 100 * (1 - abs(total_actual - total_cleared) / total_cleared)
        else:
            accuracy = 0.0

        # 평균 가격
        avg_da_smp = sum(hourly_da_smp) / 24
        avg_rt_smp = sum(hs.rt_smp_krw for hs in hourly_settlements) / 24

        # 평균 편차
        avg_deviation = sum(hs.deviation_percent for hs in hourly_settlements) / 24

        # Tier별 통계
        hours_tier1 = sum(1 for hs in hourly_settlements if hs.penalty_tier == PenaltyTier.TIER1)
        hours_tier2 = sum(1 for hs in hourly_settlements if hs.penalty_tier == PenaltyTier.TIER2)
        hours_tier3 = sum(1 for hs in hourly_settlements if hs.penalty_tier == PenaltyTier.TIER3)

        # 0원 리스크 시간
        hours_zero_risk = sum(1 for hs in hourly_settlements if hs.is_zero_risk_hour)

        # 레거시 호환 - 과발전/부족발전 통계
        hours_over = sum(1 for hs in hourly_settlements
                        if hs.penalty_type in [PenaltyType.MILD_OVER, PenaltyType.SEVERE_OVER])
        hours_under = sum(1 for hs in hourly_settlements
                         if hs.penalty_type in [PenaltyType.MILD_UNDER, PenaltyType.SEVERE_UNDER])

        return DailySettlement(
            date=date.strftime("%Y-%m-%d"),
            # 발전량
            cleared_mwh=round(total_cleared, 1),
            actual_generation_mwh=round(total_actual, 1),
            imbalance_mwh=round(total_imbalance, 1),
            # 금액
            revenue_million=round(total_revenue / 1_000_000, 2),
            imbalance_million=round(-total_imbalance_charge / 1_000_000, 2),
            capacity_payment_million=round(total_cp / 1_000_000, 2),
            net_revenue_million=round(total_net / 1_000_000, 2),
            # 성능 지표
            accuracy_pct=round(max(0, accuracy), 1),
            avg_da_smp=round(avg_da_smp, 2),
            avg_rt_smp=round(avg_rt_smp, 2),
            avg_deviation=round(avg_deviation, 2),
            # Tier 통계
            hours_tier1=hours_tier1,
            hours_tier2=hours_tier2,
            hours_tier3=hours_tier3,
            hours_zero_risk=hours_zero_risk,
            # 레거시 호환
            avg_smp=round(avg_da_smp, 2),
            hours_no_penalty=hours_tier1,
            hours_over_generation=hours_over,
            hours_under_generation=hours_under
        )

    def _estimate_hourly_cleared(self, date: datetime, actual_generation: List[float]) -> List[float]:
        """
        시간별 낙찰량 추정 (예측 기반)

        실제 시스템에서는 입찰/낙찰 데이터를 사용하지만,
        시뮬레이션에서는 실제 발전량에 오차를 더해 예측량으로 사용

        Args:
            date: 대상 날짜
            actual_generation: 실제 발전량 리스트 (24시간)

        Returns:
            List[float]: 낙찰량 리스트 (24시간)
        """
        # 일별 시드로 일관된 결과 생성
        day_seed = date.toordinal() + 1000
        np.random.seed(day_seed)

        cleared = []
        for actual in actual_generation:
            # 예측 오차: ±15% 범위 (평균 5% 오차)
            error_factor = 1 + np.random.normal(0, 0.08)  # 표준편차 8%
            error_factor = np.clip(error_factor, 0.85, 1.15)  # ±15% 제한

            # 낙찰량 = 예측량 (실제 발전량 기반 역산)
            cleared_mw = actual / error_factor
            cleared.append(max(0, cleared_mw))

        np.random.seed(None)  # 시드 리셋
        return cleared

    def calculate_recent_settlements(self, days: int = 7) -> List[DailySettlement]:
        """
        최근 N일 정산 계산

        Args:
            days: 조회 일수

        Returns:
            List[DailySettlement]: 일별 정산 목록
        """
        np.random.seed(42)  # 일관된 결과를 위해

        settlements = []
        base_date = datetime.now()

        for i in range(days):
            date = base_date - timedelta(days=i+1)  # 어제부터
            settlement = self.calculate_daily_settlement(date)
            settlements.append(settlement)

        return settlements

    def calculate_summary(self, days: int = 7) -> SettlementSummary:
        """
        정산 요약 통계 계산 (KPX 제주 시범사업 이중 정산 규정)

        Gemini 토론 결과 반영:
        - 3-Tier 페널티 시스템
        - 용량 정산금(CP) 포함
        - DA-RT 이중 정산 구조
        - RT-SMP 0원 리스크 반영

        Args:
            days: 기준 일수

        Returns:
            SettlementSummary: 정산 요약
        """
        # 현재 기간 정산
        current_settlements = self.calculate_recent_settlements(days)

        # 이전 기간 정산 (비교용)
        prev_settlements = []
        base_date = datetime.now() - timedelta(days=days)
        for i in range(days):
            date = base_date - timedelta(days=i+1)
            settlement = self.calculate_daily_settlement(date)
            prev_settlements.append(settlement)

        # 현재 기간 합계
        current_revenue = sum(s.revenue_million for s in current_settlements)
        current_imbalance = sum(s.imbalance_million for s in current_settlements)
        current_cp = sum(s.capacity_payment_million for s in current_settlements)
        current_net = sum(s.net_revenue_million for s in current_settlements)
        current_accuracy = sum(s.accuracy_pct for s in current_settlements) / len(current_settlements)
        current_cleared = sum(s.cleared_mwh for s in current_settlements)
        current_actual = sum(s.actual_generation_mwh for s in current_settlements)
        current_da_smp = sum(s.avg_da_smp for s in current_settlements) / len(current_settlements)
        current_rt_smp = sum(s.avg_rt_smp for s in current_settlements) / len(current_settlements)
        current_deviation = sum(s.avg_deviation for s in current_settlements) / len(current_settlements)
        # Tier 통계
        current_tier1 = sum(s.hours_tier1 for s in current_settlements)
        current_tier2 = sum(s.hours_tier2 for s in current_settlements)
        current_tier3 = sum(s.hours_tier3 for s in current_settlements)
        current_zero_risk = sum(s.hours_zero_risk for s in current_settlements)
        # 레거시 호환
        current_no_penalty = sum(s.hours_no_penalty for s in current_settlements)
        current_over_gen = sum(s.hours_over_generation for s in current_settlements)
        current_under_gen = sum(s.hours_under_generation for s in current_settlements)

        # 이전 기간 합계
        prev_revenue = sum(s.revenue_million for s in prev_settlements)
        prev_imbalance = sum(s.imbalance_million for s in prev_settlements)
        prev_net = sum(s.net_revenue_million for s in prev_settlements)
        prev_accuracy = sum(s.accuracy_pct for s in prev_settlements) / len(prev_settlements)

        # 변화율 계산
        def calc_change(current: float, prev: float) -> float:
            if prev == 0:
                return 0.0
            return ((current - prev) / abs(prev)) * 100

        return SettlementSummary(
            generation_revenue_million=round(current_revenue, 1),
            generation_change_pct=round(calc_change(current_revenue, prev_revenue), 1),
            imbalance_charges_million=round(current_imbalance, 1),
            imbalance_change_pct=round(calc_change(abs(current_imbalance), abs(prev_imbalance)), 1),
            capacity_payment_million=round(current_cp, 1),
            net_revenue_million=round(current_net, 1),
            net_change_pct=round(calc_change(current_net, prev_net), 1),
            forecast_accuracy_pct=round(current_accuracy, 1),
            accuracy_change_pct=round(current_accuracy - prev_accuracy, 1),
            # 추가 통계
            total_cleared_mwh=round(current_cleared, 1),
            total_actual_mwh=round(current_actual, 1),
            avg_da_smp=round(current_da_smp, 2),
            avg_rt_smp=round(current_rt_smp, 2),
            avg_deviation_pct=round(current_deviation, 2),
            # Tier 통계
            total_hours_tier1=current_tier1,
            total_hours_tier2=current_tier2,
            total_hours_tier3=current_tier3,
            total_hours_zero_risk=current_zero_risk,
            # 레거시 호환
            total_hours_no_penalty=current_no_penalty,
            total_hours_over_gen=current_over_gen,
            total_hours_under_gen=current_under_gen
        )


# 싱글톤 인스턴스
_calculator: Optional[SettlementCalculator] = None


def get_settlement_calculator() -> SettlementCalculator:
    """정산 계산기 인스턴스 반환"""
    global _calculator
    if _calculator is None:
        _calculator = SettlementCalculator()
    return _calculator


def initialize_settlement_calculator() -> None:
    """정산 계산기 초기화"""
    global _calculator
    _calculator = SettlementCalculator()
    logger.info("Settlement calculator initialized")
