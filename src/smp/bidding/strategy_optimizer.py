"""
입찰 전략 최적화
================

민간 발전사업자를 위한 최적 입찰 전략 계산

주요 기능:
1. 최적 입찰 시간대 추천
2. 리스크 조정 수익 계산
3. 시나리오 분석
4. 입찰 가격 최적화

입찰 전략:
- 시간대별 SMP 예측값과 발전량 예측값을 결합
- 리스크 허용도에 따른 보수적/공격적 전략
- 불확실성 구간을 고려한 의사결정

Author: Claude Code
Date: 2025-12
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """리스크 수준"""
    CONSERVATIVE = "conservative"  # 보수적 (하위 10% 기준)
    MODERATE = "moderate"          # 중립 (50% 기준)
    AGGRESSIVE = "aggressive"      # 공격적 (상위 90% 기준)


@dataclass
class BiddingHour:
    """시간별 입찰 정보

    Attributes:
        hour: 시간 (1-24)
        smp_predicted: 예측 SMP (원/kWh)
        smp_low: 하위 예측 (10%)
        smp_high: 상위 예측 (90%)
        generation_kw: 예측 발전량 (kW)
        expected_revenue: 예상 수익 (원)
        rank: 수익 순위 (1이 가장 높음)
        is_recommended: 입찰 추천 여부
    """
    hour: int
    smp_predicted: float
    smp_low: float = 0.0
    smp_high: float = 0.0
    generation_kw: float = 0.0
    expected_revenue: float = 0.0
    rank: int = 0
    is_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiddingStrategy:
    """입찰 전략 결과

    Attributes:
        total_hours: 총 입찰 시간
        recommended_hours: 추천 입찰 시간대 리스트
        total_generation: 총 예상 발전량 (kWh)
        total_revenue: 총 예상 수익 (원)
        average_smp: 평균 SMP (원/kWh)
        revenue_per_kwh: kWh당 수익 (원)
        risk_level: 리스크 수준
        confidence_interval: 신뢰 구간 (%)
        hourly_details: 시간별 상세 정보
    """
    total_hours: int = 24
    recommended_hours: List[int] = field(default_factory=list)
    total_generation: float = 0.0
    total_revenue: float = 0.0
    average_smp: float = 0.0
    revenue_per_kwh: float = 0.0
    risk_level: str = "moderate"
    confidence_interval: float = 80.0
    hourly_details: List[BiddingHour] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['hourly_details'] = [h.to_dict() if hasattr(h, 'to_dict') else h
                                    for h in self.hourly_details]
        return result


class BiddingStrategyOptimizer:
    """입찰 전략 최적화기

    SMP 예측과 발전량 예측을 기반으로 최적 입찰 전략을 계산합니다.

    Example:
        >>> optimizer = BiddingStrategyOptimizer()
        >>> strategy = optimizer.optimize(
        ...     smp_predictions={'q50': [100, 110, 120, ...], 'q10': [...], 'q90': [...]},
        ...     generation_predictions=[50, 100, 200, ...],
        ...     risk_tolerance=0.5
        ... )
        >>> print(f"추천 시간: {strategy.recommended_hours}")
    """

    def __init__(
        self,
        min_generation_threshold: float = 0.1,
        top_hours_ratio: float = 0.5
    ):
        """
        Args:
            min_generation_threshold: 최소 발전량 임계값 (이용률 기준, 0~1)
            top_hours_ratio: 상위 추천 시간대 비율 (0~1)
        """
        self.min_generation_threshold = min_generation_threshold
        self.top_hours_ratio = top_hours_ratio

    def optimize(
        self,
        smp_predictions: Dict[str, List[float]],
        generation_predictions: List[float],
        capacity_kw: float = 1000.0,
        risk_tolerance: float = 0.5
    ) -> BiddingStrategy:
        """최적 입찰 전략 계산

        Args:
            smp_predictions: SMP 예측값 딕셔너리
                - 'q50': 중앙값 예측 (필수)
                - 'q10': 하위 10% (선택)
                - 'q90': 상위 90% (선택)
            generation_predictions: 시간별 발전량 예측 (kW)
            capacity_kw: 설비용량 (kW)
            risk_tolerance: 리스크 허용도 (0=보수적, 1=공격적)

        Returns:
            BiddingStrategy: 최적화된 입찰 전략
        """
        # 입력 검증
        q50 = smp_predictions.get('q50', smp_predictions.get('q_50', []))
        q10 = smp_predictions.get('q10', smp_predictions.get('q_10', q50))
        q90 = smp_predictions.get('q90', smp_predictions.get('q_90', q50))

        hours = len(q50)
        if hours == 0:
            raise ValueError("SMP 예측값이 비어있습니다")

        # 발전량 조정
        if len(generation_predictions) < hours:
            generation_predictions = list(generation_predictions) + [0] * (hours - len(generation_predictions))

        # 리스크 수준 결정
        if risk_tolerance < 0.33:
            risk_level = RiskLevel.CONSERVATIVE
            smp_values = q10  # 보수적: 하위 예측 사용
        elif risk_tolerance > 0.66:
            risk_level = RiskLevel.AGGRESSIVE
            smp_values = q90  # 공격적: 상위 예측 사용
        else:
            risk_level = RiskLevel.MODERATE
            smp_values = q50  # 중립: 중앙값 사용

        # 시간별 분석
        hourly_data = []
        for h in range(hours):
            gen_kw = generation_predictions[h]
            smp = smp_values[h]

            # 예상 수익 = 발전량 × SMP
            revenue = gen_kw * smp

            hourly_data.append({
                'hour': h + 1,
                'smp_predicted': q50[h],
                'smp_low': q10[h],
                'smp_high': q90[h],
                'generation_kw': gen_kw,
                'expected_revenue': revenue,
            })

        # 수익 순위 계산
        sorted_hours = sorted(
            hourly_data,
            key=lambda x: x['expected_revenue'],
            reverse=True
        )

        for rank, item in enumerate(sorted_hours, 1):
            item['rank'] = rank

        # 상위 시간대 추천
        min_gen = capacity_kw * self.min_generation_threshold
        top_n = max(1, int(hours * self.top_hours_ratio))

        recommended = []
        for item in sorted_hours:
            if item['generation_kw'] >= min_gen and len(recommended) < top_n:
                item['is_recommended'] = True
                recommended.append(item['hour'])
            else:
                item['is_recommended'] = False

        # BiddingHour 객체 생성
        hourly_details = []
        for h in range(hours):
            data = next(d for d in hourly_data if d['hour'] == h + 1)
            hourly_details.append(BiddingHour(**data))

        # 전략 요약
        rec_data = [d for d in hourly_data if d['is_recommended']]
        total_gen = sum(d['generation_kw'] for d in rec_data)
        total_rev = sum(d['expected_revenue'] for d in rec_data)
        avg_smp = np.mean([d['smp_predicted'] for d in rec_data]) if rec_data else 0

        strategy = BiddingStrategy(
            total_hours=hours,
            recommended_hours=sorted(recommended),
            total_generation=total_gen,
            total_revenue=total_rev,
            average_smp=avg_smp,
            revenue_per_kwh=total_rev / total_gen if total_gen > 0 else 0,
            risk_level=risk_level.value,
            confidence_interval=80.0,
            hourly_details=hourly_details,
        )

        return strategy

    def compare_strategies(
        self,
        smp_predictions: Dict[str, List[float]],
        generation_predictions: List[float],
        capacity_kw: float = 1000.0
    ) -> Dict[str, BiddingStrategy]:
        """다양한 리스크 수준의 전략 비교

        Args:
            smp_predictions: SMP 예측값
            generation_predictions: 발전량 예측값
            capacity_kw: 설비용량

        Returns:
            리스크 수준별 전략 딕셔너리
        """
        strategies = {}

        for risk_tolerance, name in [(0.1, 'conservative'), (0.5, 'moderate'), (0.9, 'aggressive')]:
            strategy = self.optimize(
                smp_predictions=smp_predictions,
                generation_predictions=generation_predictions,
                capacity_kw=capacity_kw,
                risk_tolerance=risk_tolerance
            )
            strategies[name] = strategy

        return strategies


class RevenueCalculator:
    """수익 시뮬레이션

    다양한 시나리오에서 예상 수익을 계산합니다.
    """

    def simulate(
        self,
        smp_scenarios: Dict[str, List[float]],
        generation: List[float],
        hours: int = 24
    ) -> Dict[str, Any]:
        """수익 시뮬레이션

        Args:
            smp_scenarios: SMP 시나리오 (q10, q50, q90)
            generation: 발전량 예측 (kW)
            hours: 시간 수

        Returns:
            시나리오별 수익 정보
        """
        results = {}

        for scenario_name, smp_values in smp_scenarios.items():
            hourly_revenue = []
            for h in range(min(hours, len(smp_values), len(generation))):
                rev = generation[h] * smp_values[h]
                hourly_revenue.append(rev)

            total = sum(hourly_revenue)
            avg = np.mean(hourly_revenue) if hourly_revenue else 0

            results[scenario_name] = {
                'total_revenue': total,
                'average_hourly': avg,
                'max_hour': max(range(len(hourly_revenue)), key=lambda i: hourly_revenue[i]) + 1 if hourly_revenue else 0,
                'min_hour': min(range(len(hourly_revenue)), key=lambda i: hourly_revenue[i]) + 1 if hourly_revenue else 0,
                'hourly_revenue': hourly_revenue,
            }

        # 기대값과 범위 계산
        if 'q50' in results:
            expected = results['q50']['total_revenue']
        else:
            expected = np.mean([r['total_revenue'] for r in results.values()])

        revenues = [r['total_revenue'] for r in results.values()]
        best_case = max(revenues)
        worst_case = min(revenues)

        results['summary'] = {
            'expected_revenue': expected,
            'best_case': best_case,
            'worst_case': worst_case,
            'revenue_range': best_case - worst_case,
            'risk_adjusted': expected * 0.9,  # 10% 리스크 할인
        }

        return results


class RiskAnalyzer:
    """리스크 분석

    입찰 전략의 리스크를 분석합니다.
    """

    def analyze(
        self,
        strategy: BiddingStrategy,
        historical_smp: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """리스크 분석

        Args:
            strategy: 입찰 전략
            historical_smp: 과거 SMP 데이터 (선택)

        Returns:
            리스크 분석 결과
        """
        hourly = strategy.hourly_details

        # SMP 변동성 분석
        smp_values = [h.smp_predicted for h in hourly]
        smp_ranges = [h.smp_high - h.smp_low for h in hourly if h.smp_high > 0]

        volatility = np.std(smp_values) if smp_values else 0
        avg_uncertainty = np.mean(smp_ranges) if smp_ranges else 0

        # 집중도 리스크 (상위 시간에 집중된 수익)
        recommended = [h for h in hourly if h.is_recommended]
        if recommended:
            top_revenue = max(h.expected_revenue for h in recommended)
            total_revenue = sum(h.expected_revenue for h in recommended)
            concentration = top_revenue / total_revenue if total_revenue > 0 else 0
        else:
            concentration = 0

        # 리스크 점수 (0-100)
        # 변동성, 불확실성, 집중도를 고려
        risk_score = min(100, (volatility / 10) + (avg_uncertainty / 5) + (concentration * 30))

        # 리스크 등급
        if risk_score < 30:
            risk_grade = 'LOW'
            risk_message = '리스크가 낮습니다. 안정적인 수익이 예상됩니다.'
        elif risk_score < 60:
            risk_grade = 'MEDIUM'
            risk_message = '중간 수준의 리스크입니다. SMP 변동에 주의하세요.'
        else:
            risk_grade = 'HIGH'
            risk_message = '리스크가 높습니다. 보수적인 입찰을 권장합니다.'

        return {
            'risk_score': round(risk_score, 1),
            'risk_grade': risk_grade,
            'risk_message': risk_message,
            'smp_volatility': round(volatility, 2),
            'average_uncertainty': round(avg_uncertainty, 2),
            'concentration_risk': round(concentration * 100, 1),
            'recommendation': '분산 입찰' if concentration > 0.3 else '집중 입찰 가능',
        }


def create_bidding_optimizer(**kwargs) -> BiddingStrategyOptimizer:
    """입찰 전략 최적화기 팩토리 함수"""
    return BiddingStrategyOptimizer(**kwargs)


if __name__ == "__main__":
    # 테스트
    print("입찰 전략 최적화 테스트")
    print("=" * 60)

    # 샘플 SMP 예측 (24시간)
    np.random.seed(42)
    base_smp = [80, 85, 82, 80, 85, 95, 110, 130,
                145, 155, 160, 165, 168, 165, 158, 145,
                135, 125, 115, 105, 95, 88, 82, 78]

    smp_predictions = {
        'q50': base_smp,
        'q10': [s * 0.85 for s in base_smp],  # 하위 15%
        'q90': [s * 1.15 for s in base_smp],  # 상위 15%
    }

    # 샘플 발전량 예측 (1MW 태양광 기준)
    generation = [0, 0, 0, 0, 50, 150, 400, 600,
                  750, 850, 900, 920, 900, 850, 750, 600,
                  400, 150, 50, 0, 0, 0, 0, 0]  # kW

    # 최적화기 생성
    optimizer = BiddingStrategyOptimizer()

    # 전략 최적화
    strategy = optimizer.optimize(
        smp_predictions=smp_predictions,
        generation_predictions=generation,
        capacity_kw=1000,
        risk_tolerance=0.5  # 중립
    )

    print(f"\n리스크 수준: {strategy.risk_level}")
    print(f"추천 입찰 시간: {strategy.recommended_hours}")
    print(f"총 예상 발전량: {strategy.total_generation:,.0f} kWh")
    print(f"총 예상 수익: {strategy.total_revenue:,.0f} 원")
    print(f"평균 SMP: {strategy.average_smp:.1f} 원/kWh")
    print(f"kWh당 수익: {strategy.revenue_per_kwh:.1f} 원")

    # 시간별 상세
    print("\n시간별 상세:")
    print("-" * 60)
    print(f"{'시간':>4} {'SMP':>8} {'발전량':>8} {'수익':>10} {'순위':>4} {'추천':>4}")
    print("-" * 60)
    for h in strategy.hourly_details:
        rec = "✓" if h.is_recommended else ""
        print(f"{h.hour:>4}시 {h.smp_predicted:>7.1f} {h.generation_kw:>7.0f} kW "
              f"{h.expected_revenue:>9,.0f} 원 {h.rank:>4} {rec:>4}")

    # 전략 비교
    print("\n" + "=" * 60)
    print("리스크 수준별 전략 비교")
    print("-" * 60)

    strategies = optimizer.compare_strategies(
        smp_predictions=smp_predictions,
        generation_predictions=generation,
        capacity_kw=1000
    )

    for name, strat in strategies.items():
        print(f"{name:15} 예상 수익: {strat.total_revenue:>12,.0f} 원")

    # 수익 시뮬레이션
    print("\n" + "=" * 60)
    print("수익 시뮬레이션")
    print("-" * 60)

    calculator = RevenueCalculator()
    simulation = calculator.simulate(smp_predictions, generation)

    print(f"기대 수익: {simulation['summary']['expected_revenue']:,.0f} 원")
    print(f"최선 시나리오: {simulation['summary']['best_case']:,.0f} 원")
    print(f"최악 시나리오: {simulation['summary']['worst_case']:,.0f} 원")
    print(f"리스크 조정 수익: {simulation['summary']['risk_adjusted']:,.0f} 원")

    # 리스크 분석
    print("\n" + "=" * 60)
    print("리스크 분석")
    print("-" * 60)

    analyzer = RiskAnalyzer()
    risk = analyzer.analyze(strategy)

    print(f"리스크 점수: {risk['risk_score']}/100")
    print(f"리스크 등급: {risk['risk_grade']}")
    print(f"메시지: {risk['risk_message']}")
    print(f"SMP 변동성: {risk['smp_volatility']}")
    print(f"집중도 리스크: {risk['concentration_risk']}%")

    print("\n모든 테스트 완료!")
