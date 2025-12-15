#!/usr/bin/env python3
"""
제주 여객선 승객 추정기 (v2.0)
===============================

기존 jeju_ferry_crawler.py를 대체하는 새로운 구현.

WHY THIS APPROACH:
-----------------
기존 크롤러의 문제점:
1. 공공데이터포털 API가 "연간/항로별" 데이터만 제공 (일별 불가)
2. 4개 수집 함수가 return None으로 미구현
3. 기본 추정값 1,500명/일 = 실제(5,500명)의 27% 수준으로 부정확

새로운 접근법:
- 항공 입도객 데이터 × 해운 분담률(5.5%)로 추정
- 성수기/비수기 변동이 자연스럽게 반영
- KOMSA(한국해양교통안전공단) 통계 기반 검증된 비율

VALIDATION:
----------
- 연간 항공 입도객: ~1,500만 명
- 연간 해운 입도객: ~100만 명 (해운 분담률 ~6.5%)
- 본토↔제주 항로만 고려 시: ~80만 명 (분담률 ~5.5%)
  (마라도, 가파도 등 제주 근해 항로 제외)

Usage:
    from tools.crawlers.jeju_ferry_estimator import JejuFerryEstimator
    
    estimator = JejuFerryEstimator()
    
    # 항공 데이터로부터 해운 승객 추정
    air_data = {'arrival': 45000, 'departure': 44000}
    ferry_data = estimator.estimate_from_air(air_data)
    
    # 기상 조건 반영 (선택적)
    weather = {'wave_height': 3.5, 'wind_speed': 15.0}
    ferry_data = estimator.estimate_from_air(air_data, weather_data=weather)
    
    # DataFrame 변환
    df = estimator.estimate_range(air_df, start_date, end_date)
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Union
from pathlib import Path

import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 상수 정의 (통계 기반)
# =============================================================================

# 해운 분담률: KOMSA 통계 기반
# 연간 항공 ~1,500만 vs 해운 ~80만 (본토↔제주만) ≈ 5.3%
# 계절 변동 고려하여 5.5% 사용
FERRY_RATIO_DEFAULT = 0.055

# 계절별 해운 분담률 조정
# 여름(성수기): 항공이 더 선호됨 → 해운 비율 약간 감소
# 겨울(비수기): 해운 선호 증가 (차량 이동 등)
SEASONAL_FERRY_RATIO = {
    1: 0.060,  # 1월: 겨울 비수기, 해운 비율 높음
    2: 0.055,  # 2월: 설 연휴
    3: 0.055,  # 3월
    4: 0.055,  # 4월
    5: 0.050,  # 5월: 성수기 시작
    6: 0.050,  # 6월
    7: 0.045,  # 7월: 성수기, 항공 선호
    8: 0.045,  # 8월: 성수기, 항공 선호
    9: 0.050,  # 9월: 추석
    10: 0.055, # 10월
    11: 0.058, # 11월
    12: 0.060, # 12월: 겨울 비수기
}

# 기상 조건 임계값 (여객선 운항 제한)
WEATHER_THRESHOLDS = {
    'wave_height': 3.0,    # 파고 3m 이상 시 결항 가능
    'wind_speed': 14.0,    # 풍속 14m/s 이상 시 결항 가능
    'visibility': 1.0,     # 시정 1km 미만 시 결항 가능
}

# 결항 시 감소 비율
CANCELLATION_REDUCTION = {
    'full': 0.0,      # 전면 결항
    'partial': 0.3,   # 일부 운항 (30%)
    'delayed': 0.7,   # 지연 운항 (70%)
}


@dataclass
class FerryEstimate:
    """여객선 승객 추정 결과"""
    date: str                    # YYYY-MM-DD
    arrival: int                 # 입도 승객 추정
    departure: int               # 출도 승객 추정
    net_flow: int               # 순 유입 (입도 - 출도)
    ferry_ratio: float          # 적용된 해운 분담률
    weather_factor: float       # 기상 조건 보정 계수 (0.0 ~ 1.0)
    source: str                 # 데이터 출처
    confidence: float           # 추정 신뢰도 (0.0 ~ 1.0)
    note: Optional[str] = None  # 추가 정보
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FerryEstimate':
        return cls(**data)


class JejuFerryEstimator:
    """
    제주 여객선 승객 추정기
    
    항공 입도객 데이터를 기반으로 해운 승객을 추정합니다.
    기상 조건(파고, 풍속)에 따른 결항 가능성도 반영합니다.
    """
    
    def __init__(
        self,
        ferry_ratio: float = FERRY_RATIO_DEFAULT,
        use_seasonal_ratio: bool = True,
        apply_weather_factor: bool = True,
    ):
        """
        Args:
            ferry_ratio: 기본 해운 분담률 (default: 0.055)
            use_seasonal_ratio: 계절별 분담률 사용 여부
            apply_weather_factor: 기상 조건 보정 적용 여부
        """
        self.ferry_ratio = ferry_ratio
        self.use_seasonal_ratio = use_seasonal_ratio
        self.apply_weather_factor = apply_weather_factor
        
        logger.info(f"JejuFerryEstimator 초기화: ratio={ferry_ratio}, "
                   f"seasonal={use_seasonal_ratio}, weather={apply_weather_factor}")
    
    def estimate_from_air(
        self,
        air_data: Dict[str, int],
        date: Optional[str] = None,
        weather_data: Optional[Dict[str, float]] = None,
    ) -> FerryEstimate:
        """
        항공 승객 데이터로부터 해운 승객 추정
        
        Args:
            air_data: {'arrival': int, 'departure': int} 항공 승객 수
            date: 날짜 (YYYY-MM-DD), 계절별 분담률 적용용
            weather_data: {'wave_height': float, 'wind_speed': float} 기상 조건
            
        Returns:
            FerryEstimate 객체
        """
        if not air_data or 'arrival' not in air_data:
            raise ValueError("air_data must contain 'arrival' key")
        
        # 날짜 파싱
        if date:
            dt = datetime.strptime(date, "%Y-%m-%d")
            month = dt.month
        else:
            dt = datetime.now()
            date = dt.strftime("%Y-%m-%d")
            month = dt.month
        
        # 해운 분담률 결정
        if self.use_seasonal_ratio:
            ratio = SEASONAL_FERRY_RATIO.get(month, self.ferry_ratio)
        else:
            ratio = self.ferry_ratio
        
        # 기상 조건 보정
        weather_factor = 1.0
        weather_note = None
        
        if self.apply_weather_factor and weather_data:
            weather_factor, weather_note = self._calculate_weather_factor(weather_data)
        
        # 승객 수 추정
        air_arrival = air_data.get('arrival', 0)
        air_departure = air_data.get('departure', air_arrival)
        
        ferry_arrival = int(air_arrival * ratio * weather_factor)
        ferry_departure = int(air_departure * ratio * weather_factor)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(air_data, weather_factor)
        
        return FerryEstimate(
            date=date,
            arrival=ferry_arrival,
            departure=ferry_departure,
            net_flow=ferry_arrival - ferry_departure,
            ferry_ratio=ratio,
            weather_factor=weather_factor,
            source="air_traffic_ratio",
            confidence=confidence,
            note=weather_note,
        )
    
    def _calculate_weather_factor(
        self,
        weather_data: Dict[str, float]
    ) -> tuple[float, Optional[str]]:
        """
        기상 조건에 따른 보정 계수 계산
        
        Returns:
            (factor, note) 튜플
        """
        wave_height = weather_data.get('wave_height', 0)
        wind_speed = weather_data.get('wind_speed', 0)
        visibility = weather_data.get('visibility', 10)  # km
        
        # 결항 조건 체크
        if wave_height >= 4.0 or wind_speed >= 18.0:
            return CANCELLATION_REDUCTION['full'], "전면 결항 (악천후)"
        
        if wave_height >= 3.5 or wind_speed >= 15.0:
            return CANCELLATION_REDUCTION['partial'], "일부 결항 (기상 악화)"
        
        if wave_height >= 3.0 or wind_speed >= 14.0 or visibility < 1.0:
            return CANCELLATION_REDUCTION['delayed'], "지연 운항 (기상 주의)"
        
        # 약간의 영향
        if wave_height >= 2.0 or wind_speed >= 10.0:
            return 0.9, "기상 영향 (경미)"
        
        return 1.0, None
    
    def _calculate_confidence(
        self,
        air_data: Dict[str, int],
        weather_factor: float
    ) -> float:
        """추정 신뢰도 계산"""
        base_confidence = 0.85  # 기본 신뢰도
        
        # 항공 데이터 품질에 따른 조정
        air_arrival = air_data.get('arrival', 0)
        if air_arrival < 10000:
            base_confidence -= 0.1  # 데이터가 적으면 신뢰도 감소
        elif air_arrival > 50000:
            base_confidence += 0.05  # 데이터가 많으면 신뢰도 증가
        
        # 기상 보정 시 신뢰도 감소
        if weather_factor < 1.0:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.5), 0.95)
    
    def estimate_range(
        self,
        air_df: pd.DataFrame,
        date_column: str = 'date',
        arrival_column: str = 'arrival',
        departure_column: str = 'departure',
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        기간별 해운 승객 일괄 추정
        
        Args:
            air_df: 항공 승객 DataFrame
            date_column: 날짜 컬럼명
            arrival_column: 입도 컬럼명
            departure_column: 출도 컬럼명
            weather_df: 기상 데이터 DataFrame (선택)
            
        Returns:
            해운 승객 추정 DataFrame
        """
        results = []
        
        for idx, row in air_df.iterrows():
            date_val = row[date_column]
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)
            
            air_data = {
                'arrival': int(row[arrival_column]),
                'departure': int(row.get(departure_column, row[arrival_column])),
            }
            
            # 기상 데이터 매칭 (있는 경우)
            weather_data = None
            if weather_df is not None:
                weather_row = weather_df[weather_df[date_column] == date_val]
                if not weather_row.empty:
                    weather_data = {
                        'wave_height': weather_row.iloc[0].get('wave_height', 0),
                        'wind_speed': weather_row.iloc[0].get('wind_speed', 0),
                    }
            
            try:
                estimate = self.estimate_from_air(air_data, date_str, weather_data)
                results.append(estimate.to_dict())
            except Exception as e:
                logger.warning(f"추정 실패 ({date_str}): {e}")
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"해운 승객 추정 완료: {len(df)}일")
        
        return df
    
    def estimate_daily_default(self, date: str) -> FerryEstimate:
        """
        항공 데이터 없이 기본값으로 추정 (fallback)
        
        KOMSA 통계 기반:
        - 연간 해운 입도객 ~80만 명 (본토↔제주)
        - 일 평균 ~2,200명
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        month = dt.month
        
        # 기본 일 평균 (연간 80만 / 365일)
        base_daily = 2200
        
        # 계절 가중치 (성수기/비수기)
        seasonal_weights = {
            1: 0.8, 2: 0.9, 3: 0.95, 4: 1.0,
            5: 1.1, 6: 1.0, 7: 1.2, 8: 1.3,
            9: 1.0, 10: 1.1, 11: 0.9, 12: 0.85,
        }
        
        weight = seasonal_weights.get(month, 1.0)
        daily_estimate = int(base_daily * weight)
        
        return FerryEstimate(
            date=date,
            arrival=daily_estimate,
            departure=daily_estimate,
            net_flow=0,
            ferry_ratio=0.055,
            weather_factor=1.0,
            source="komsa_statistics_fallback",
            confidence=0.6,  # 기본값이므로 신뢰도 낮음
            note="항공 데이터 없이 KOMSA 통계 기반 추정",
        )
    
    def validate_estimate(
        self,
        estimate: FerryEstimate,
        actual: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        추정값 검증 (실제값 비교 가능 시)
        """
        validation = {
            'estimated_arrival': estimate.arrival,
            'estimated_departure': estimate.departure,
            'confidence': estimate.confidence,
        }
        
        if actual:
            actual_arrival = actual.get('arrival', 0)
            actual_departure = actual.get('departure', 0)
            
            if actual_arrival > 0:
                arrival_error = abs(estimate.arrival - actual_arrival) / actual_arrival
                validation['arrival_error_rate'] = arrival_error
            
            if actual_departure > 0:
                departure_error = abs(estimate.departure - actual_departure) / actual_departure
                validation['departure_error_rate'] = departure_error
        
        return validation


# =============================================================================
# CLI 인터페이스
# =============================================================================

def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="제주 여객선 승객 추정기 (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 추정 (항공 데이터 입력)
  python jeju_ferry_estimator.py --air-arrival 45000 --air-departure 44000
  
  # 날짜 지정 (계절별 분담률 적용)
  python jeju_ferry_estimator.py --air-arrival 45000 --date 2024-08-15
  
  # 기상 조건 반영
  python jeju_ferry_estimator.py --air-arrival 45000 --wave-height 3.5 --wind-speed 15.0
  
  # 항공 데이터 없이 기본 추정
  python jeju_ferry_estimator.py --date 2024-08-15 --fallback
        """
    )
    
    parser.add_argument('--air-arrival', type=int, help='항공 입도 승객 수')
    parser.add_argument('--air-departure', type=int, help='항공 출도 승객 수')
    parser.add_argument('--date', type=str, help='날짜 (YYYY-MM-DD)')
    parser.add_argument('--wave-height', type=float, help='파고 (m)')
    parser.add_argument('--wind-speed', type=float, help='풍속 (m/s)')
    parser.add_argument('--fallback', action='store_true', help='기본값 추정 모드')
    parser.add_argument('--no-seasonal', action='store_true', help='계절별 분담률 비활성화')
    
    args = parser.parse_args()
    
    # 추정기 초기화
    estimator = JejuFerryEstimator(
        use_seasonal_ratio=not args.no_seasonal,
    )
    
    # 날짜 설정
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    if args.fallback:
        # 기본값 추정
        result = estimator.estimate_daily_default(date)
    elif args.air_arrival:
        # 항공 데이터 기반 추정
        air_data = {
            'arrival': args.air_arrival,
            'departure': args.air_departure or args.air_arrival,
        }
        
        weather_data = None
        if args.wave_height or args.wind_speed:
            weather_data = {
                'wave_height': args.wave_height or 0,
                'wind_speed': args.wind_speed or 0,
            }
        
        result = estimator.estimate_from_air(air_data, date, weather_data)
    else:
        parser.print_help()
        return
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"🚢 제주 여객선 승객 추정 결과")
    print(f"{'='*60}")
    print(f"날짜: {result.date}")
    print(f"입도 (도착): {result.arrival:,}명")
    print(f"출도 (출발): {result.departure:,}명")
    print(f"순 유입: {result.net_flow:+,}명")
    print(f"{'='*60}")
    print(f"해운 분담률: {result.ferry_ratio:.1%}")
    print(f"기상 보정: {result.weather_factor:.1%}")
    print(f"추정 신뢰도: {result.confidence:.1%}")
    print(f"데이터 출처: {result.source}")
    if result.note:
        print(f"비고: {result.note}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
