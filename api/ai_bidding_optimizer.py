"""
AI Bidding Optimizer
====================

실제 SMP 예측 모델을 사용한 10-segment 입찰 최적화

- SMP 예측: BiLSTM+Attention v3.2 Optuna (MAPE 7.17%, R² 0.77)
- 입찰 최적화: Quantile 기반 확률적 청산 모델
- 10-segment 입찰 곡선 최적화

Author: Claude Code
Date: 2025-12
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """리스크 수준"""
    CONSERVATIVE = "conservative"  # 보수적: 청산 확률 우선
    MODERATE = "moderate"          # 중립: 균형
    AGGRESSIVE = "aggressive"      # 공격적: 수익 우선


@dataclass
class BidSegment:
    """입찰 구간"""
    segment_id: int
    quantity_mw: float
    price_krw_mwh: float
    clearing_probability: float = 0.0  # 청산 확률
    expected_revenue: float = 0.0       # 기대 수익


@dataclass
class HourlyOptimizedBid:
    """시간별 최적화 입찰"""
    hour: int
    segments: List[BidSegment]
    total_mw: float
    avg_price: float
    smp_forecast: Dict[str, float]  # q10, q50, q90
    expected_revenue: float
    clearing_probability: float


@dataclass
class OptimizationResult:
    """최적화 결과"""
    trading_date: str
    capacity_mw: float
    risk_level: str
    hourly_bids: List[HourlyOptimizedBid]
    total_daily_mwh: float
    expected_daily_revenue: float
    model_used: str
    optimization_method: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'trading_date': self.trading_date,
            'capacity_mw': self.capacity_mw,
            'risk_level': self.risk_level,
            'hourly_bids': [],
            'total_daily_mwh': self.total_daily_mwh,
            'expected_daily_revenue': self.expected_daily_revenue,
            'model_used': self.model_used,
            'optimization_method': self.optimization_method,
            'created_at': self.created_at,
        }

        for hb in self.hourly_bids:
            result['hourly_bids'].append({
                'hour': hb.hour,
                'segments': [asdict(s) for s in hb.segments],
                'total_mw': hb.total_mw,
                'avg_price': hb.avg_price,
                'smp_forecast': hb.smp_forecast,
                'expected_revenue': hb.expected_revenue,
                'clearing_probability': hb.clearing_probability,
            })

        return result


class AIBiddingOptimizer:
    """AI 기반 10-segment 입찰 최적화기

    실제 SMP 예측 모델을 사용하여 최적 입찰 곡선을 생성합니다.

    최적화 전략:
    1. SMP 예측의 quantile 분포 활용 (q10, q50, q90)
    2. 청산 확률 기반 세그먼트 가격 결정
    3. 리스크 수준에 따른 가격 조정
    4. 기대 수익 최대화
    """

    def __init__(self):
        self._smp_predictor = None
        self._model_loaded = False

    def _load_smp_model(self):
        """SMP 예측 모델 로드 (lazy loading)"""
        if self._model_loaded:
            return self._smp_predictor

        try:
            from src.smp.models.smp_predictor import SMPPredictor
            self._smp_predictor = SMPPredictor(use_advanced=True)
            if self._smp_predictor.is_ready():
                logger.info("AI Bidding: SMP v3.2 model loaded (MAPE: 7.17%, R²: 0.77)")
                self._model_loaded = True
            else:
                logger.warning("AI Bidding: SMP model not ready, using fallback")
                self._smp_predictor = None
        except Exception as e:
            logger.error(f"AI Bidding: Failed to load SMP model: {e}")
            self._smp_predictor = None

        self._model_loaded = True
        return self._smp_predictor

    def get_smp_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """실제 SMP 예측 가져오기"""
        predictor = self._load_smp_model()

        if predictor is not None:
            try:
                result = predictor.predict_24h()
                return {
                    'q10': result['q10'][:hours],
                    'q50': result['q50'][:hours],
                    'q90': result['q90'][:hours],
                    'times': result['times'][:hours],
                    'model_used': 'BiLSTM+Attention v3.2 Optuna',
                    'mape': 7.17,
                }
            except Exception as e:
                logger.error(f"SMP prediction failed: {e}")

        # Fallback: 패턴 기반 예측
        return self._generate_fallback_forecast(hours)

    def _generate_fallback_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """폴백 SMP 예측 (패턴 기반)"""
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        # 시간대별 SMP 패턴 (실제 제주 SMP 기반)
        hourly_pattern = np.array([
            85, 82, 80, 78, 80, 85,   # 0-5시: 심야
            95, 110, 125, 135, 140, 145,  # 6-11시: 오전 피크
            148, 145, 140, 135, 128, 122,  # 12-17시: 오후
            115, 108, 100, 95, 90, 87     # 18-23시: 저녁
        ])

        start_hour = base_time.hour
        pattern = np.roll(hourly_pattern, -start_hour)[:hours]

        # 노이즈 추가 (일관된 시드 사용)
        seed = int(base_time.strftime("%Y%m%d%H"))
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(0, 5, hours)

        q50 = pattern + noise
        q10 = q50 * 0.85
        q90 = q50 * 1.15

        times = [base_time + timedelta(hours=i) for i in range(hours)]

        return {
            'q10': q10.tolist(),
            'q50': q50.tolist(),
            'q90': q90.tolist(),
            'times': times,
            'model_used': 'Pattern-based Fallback',
            'mape': 15.0,  # 폴백은 정확도 낮음
        }

    def calculate_clearing_probability(
        self,
        bid_price: float,
        smp_q10: float,
        smp_q50: float,
        smp_q90: float
    ) -> float:
        """입찰 가격의 청산 확률 계산

        가정: SMP 분포가 정규분포를 따름
        - 입찰가 <= q10: 90% 확률
        - 입찰가 = q50: 50% 확률
        - 입찰가 >= q90: 10% 확률

        Args:
            bid_price: 입찰 가격
            smp_q10: SMP 10% 분위
            smp_q50: SMP 50% 분위 (중앙값)
            smp_q90: SMP 90% 분위

        Returns:
            청산 확률 (0-1)
        """
        if bid_price <= smp_q10:
            return 0.95  # 거의 확실히 청산
        elif bid_price >= smp_q90:
            return 0.05  # 거의 청산 안됨
        elif bid_price <= smp_q50:
            # q10 ~ q50 구간: 선형 보간
            ratio = (bid_price - smp_q10) / (smp_q50 - smp_q10 + 1e-6)
            return 0.95 - 0.45 * ratio  # 95% -> 50%
        else:
            # q50 ~ q90 구간: 선형 보간
            ratio = (bid_price - smp_q50) / (smp_q90 - smp_q50 + 1e-6)
            return 0.50 - 0.45 * ratio  # 50% -> 5%

    def optimize_segments(
        self,
        capacity_mw: float,
        smp_q10: float,
        smp_q50: float,
        smp_q90: float,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        num_segments: int = 10
    ) -> List[BidSegment]:
        """10-segment 입찰 곡선 최적화

        최적화 목표:
        - 기대 수익 최대화: E[Revenue] = Σ(price × quantity × clearing_prob)
        - 리스크 조정: 리스크 수준에 따라 가격 범위 조정

        Args:
            capacity_mw: 총 입찰 용량 (MW)
            smp_q10: SMP 10% 분위
            smp_q50: SMP 50% 분위
            smp_q90: SMP 90% 분위
            risk_level: 리스크 수준
            num_segments: 세그먼트 수

        Returns:
            최적화된 입찰 세그먼트 리스트
        """
        # 리스크 수준별 가격 범위 설정
        if risk_level == RiskLevel.CONSERVATIVE:
            # 보수적: q10 ~ q50 범위, 청산 확률 높음
            price_min = smp_q10 * 0.95
            price_max = smp_q50 * 1.0
            base_offset = -5  # 가격 낮춤
        elif risk_level == RiskLevel.AGGRESSIVE:
            # 공격적: q50 ~ q90 범위, 수익 우선
            price_min = smp_q50 * 0.95
            price_max = smp_q90 * 1.05
            base_offset = 5  # 가격 높임
        else:  # MODERATE
            # 중립: q10 ~ q90 전체 범위
            price_min = smp_q10 * 0.98
            price_max = smp_q90 * 1.02
            base_offset = 0

        # 세그먼트별 용량 (균등 분배)
        segment_qty = capacity_mw / num_segments

        # 가격 분포 생성 (지수적 증가)
        # 낮은 가격에 더 많은 세그먼트 배치 (청산 확률 높음)
        price_range = price_max - price_min
        segments = []

        for i in range(num_segments):
            # 비선형 가격 분포 (로그 스케일)
            # 초기 세그먼트는 낮은 가격, 후반 세그먼트는 높은 가격
            t = i / (num_segments - 1) if num_segments > 1 else 0

            # 수정된 로지스틱 함수로 S-curve 생성
            # 리스크 수준에 따라 곡선 모양 조정
            if risk_level == RiskLevel.CONSERVATIVE:
                # 보수적: 낮은 가격에 집중
                adjusted_t = t ** 1.5
            elif risk_level == RiskLevel.AGGRESSIVE:
                # 공격적: 높은 가격에 집중
                adjusted_t = t ** 0.7
            else:
                # 중립: 선형
                adjusted_t = t

            price = price_min + price_range * adjusted_t + base_offset
            price = round(max(price, 1), 1)  # 최소 1원

            # 청산 확률 계산
            clearing_prob = self.calculate_clearing_probability(
                price, smp_q10, smp_q50, smp_q90
            )

            # 기대 수익 = 가격 × 용량 × 청산확률
            expected_revenue = price * segment_qty * clearing_prob

            segments.append(BidSegment(
                segment_id=i + 1,
                quantity_mw=round(segment_qty, 2),
                price_krw_mwh=round(price, 1),
                clearing_probability=round(clearing_prob, 3),
                expected_revenue=round(expected_revenue, 2),
            ))

        return segments

    def optimize_hourly_bids(
        self,
        capacity_mw: float = 50.0,
        risk_level: str = "moderate",
        hours: int = 24
    ) -> OptimizationResult:
        """시간별 최적 입찰 생성

        Args:
            capacity_mw: 입찰 용량 (MW)
            risk_level: 리스크 수준 (conservative/moderate/aggressive)
            hours: 예측 시간

        Returns:
            OptimizationResult: 시간별 최적화된 입찰
        """
        # 리스크 레벨 변환
        try:
            risk = RiskLevel(risk_level.lower())
        except ValueError:
            risk = RiskLevel.MODERATE

        # SMP 예측 가져오기
        forecast = self.get_smp_forecast(hours)
        q10 = forecast['q10']
        q50 = forecast['q50']
        q90 = forecast['q90']
        times = forecast['times']
        model_used = forecast['model_used']

        # 시간별 최적화
        hourly_bids = []
        total_expected_revenue = 0

        for h in range(min(hours, len(q50))):
            # 해당 시간의 SMP 예측
            smp_q10 = q10[h]
            smp_q50 = q50[h]
            smp_q90 = q90[h]

            # 세그먼트 최적화
            segments = self.optimize_segments(
                capacity_mw=capacity_mw,
                smp_q10=smp_q10,
                smp_q50=smp_q50,
                smp_q90=smp_q90,
                risk_level=risk
            )

            # 시간별 통계
            total_mw = sum(s.quantity_mw for s in segments)
            avg_price = sum(s.price_krw_mwh * s.quantity_mw for s in segments) / total_mw
            hour_revenue = sum(s.expected_revenue for s in segments)
            avg_clearing = sum(s.clearing_probability * s.quantity_mw for s in segments) / total_mw

            total_expected_revenue += hour_revenue

            # 시간 추출
            if hasattr(times[h], 'hour'):
                hour_val = times[h].hour
            else:
                hour_val = h

            hourly_bids.append(HourlyOptimizedBid(
                hour=hour_val,
                segments=segments,
                total_mw=round(total_mw, 2),
                avg_price=round(avg_price, 1),
                smp_forecast={
                    'q10': round(smp_q10, 1),
                    'q50': round(smp_q50, 1),
                    'q90': round(smp_q90, 1),
                },
                expected_revenue=round(hour_revenue, 2),
                clearing_probability=round(avg_clearing, 3),
            ))

        # 거래일 (익일)
        trading_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        return OptimizationResult(
            trading_date=trading_date,
            capacity_mw=capacity_mw,
            risk_level=risk.value,
            hourly_bids=hourly_bids,
            total_daily_mwh=round(capacity_mw * hours * 0.7, 2),  # 70% 이용률 가정
            expected_daily_revenue=round(total_expected_revenue, 2),
            model_used=model_used,
            optimization_method="Quantile-based Probabilistic Clearing Model",
        )


# Singleton instance
_optimizer_instance = None


def get_ai_bidding_optimizer() -> AIBiddingOptimizer:
    """AI 입찰 최적화기 싱글톤 인스턴스"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = AIBiddingOptimizer()
    return _optimizer_instance


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("AI Bidding Optimizer Test")
    print("=" * 60)

    optimizer = get_ai_bidding_optimizer()

    # SMP 예측 테스트
    print("\n1. SMP Forecast:")
    forecast = optimizer.get_smp_forecast(6)
    print(f"   Model: {forecast['model_used']}")
    print(f"   Q10: {[round(v, 1) for v in forecast['q10'][:6]]}")
    print(f"   Q50: {[round(v, 1) for v in forecast['q50'][:6]]}")
    print(f"   Q90: {[round(v, 1) for v in forecast['q90'][:6]]}")

    # 세그먼트 최적화 테스트
    print("\n2. Segment Optimization (50MW, Moderate):")
    segments = optimizer.optimize_segments(
        capacity_mw=50,
        smp_q10=80,
        smp_q50=100,
        smp_q90=125,
        risk_level=RiskLevel.MODERATE
    )

    print(f"   {'Seg':>3} {'Qty':>6} {'Price':>7} {'Prob':>6} {'E[Rev]':>10}")
    print("   " + "-" * 40)
    for s in segments:
        print(f"   {s.segment_id:>3} {s.quantity_mw:>5.1f}MW {s.price_krw_mwh:>6.1f}원 "
              f"{s.clearing_probability:>5.1%} {s.expected_revenue:>9.0f}원")

    # 시간별 최적화 테스트
    print("\n3. Hourly Optimization (50MW, 6 hours):")
    result = optimizer.optimize_hourly_bids(
        capacity_mw=50,
        risk_level="moderate",
        hours=6
    )

    print(f"   Trading Date: {result.trading_date}")
    print(f"   Model: {result.model_used}")
    print(f"   Method: {result.optimization_method}")
    print(f"   Expected Daily Revenue: {result.expected_daily_revenue:,.0f}원")

    print(f"\n   {'Hour':>4} {'AvgPrice':>8} {'E[Rev]':>10} {'ClearProb':>9}")
    print("   " + "-" * 40)
    for hb in result.hourly_bids:
        print(f"   {hb.hour:>4}시 {hb.avg_price:>7.1f}원 "
              f"{hb.expected_revenue:>9.0f}원 {hb.clearing_probability:>8.1%}")

    print("\n" + "=" * 60)
    print("Test Complete!")
