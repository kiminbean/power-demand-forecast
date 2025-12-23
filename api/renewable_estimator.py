"""
제주 재생에너지 발전량 추정기
================================

데이터 소스:
1. 태양광: 한국동서발전 제주 태양광 발전 실적 데이터 (2018-2024)
2. 풍력: 풍속 기반 물리적 모델 추정

추정 방법:
- 태양광: 시간/월별 평균 패턴 + 현재 기상 조건 보정
- 풍력: 풍력 발전 출력 곡선 (Power Curve) 적용
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 제주 재생에너지 설비 용량 (2024년 기준, MW)
JEJU_SOLAR_CAPACITY_MW = 446.0  # 태양광 총 설비용량
JEJU_WIND_CAPACITY_MW = 296.0   # 풍력 총 설비용량 (가시리, 김녕, 탐라해상 등)


@dataclass
class RenewableGeneration:
    """재생에너지 발전량 데이터"""
    timestamp: datetime
    solar_mw: float  # 태양광 발전량 (MW)
    wind_mw: float   # 풍력 발전량 (MW)
    total_mw: float  # 총 재생에너지 발전량 (MW)
    solar_capacity_mw: float  # 태양광 설비용량
    wind_capacity_mw: float   # 풍력 설비용량
    solar_cf: float  # 태양광 이용률 (%)
    wind_cf: float   # 풍력 이용률 (%)
    data_source: str = "Estimation based on historical data + weather"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "solar_mw": round(self.solar_mw, 1),
            "wind_mw": round(self.wind_mw, 1),
            "total_mw": round(self.total_mw, 1),
            "solar_capacity_mw": self.solar_capacity_mw,
            "wind_capacity_mw": self.wind_capacity_mw,
            "solar_cf": round(self.solar_cf, 1),
            "wind_cf": round(self.wind_cf, 1),
            "data_source": self.data_source,
        }


class RenewableEnergyEstimator:
    """제주 재생에너지 발전량 추정기"""

    # 태양광 발전 데이터 파일
    SOLAR_DATA_FILE = "data/raw/한국동서발전_제주_기상관측_태양광발전.csv"

    def __init__(self):
        self.solar_patterns: Optional[pd.DataFrame] = None
        self.solar_hourly_avg: Optional[pd.DataFrame] = None
        self._load_solar_data()

    def _load_solar_data(self):
        """태양광 발전 실적 데이터 로드 및 패턴 분석"""
        try:
            # 프로젝트 루트 기준 경로
            project_root = Path(__file__).parent.parent
            solar_path = project_root / self.SOLAR_DATA_FILE

            if not solar_path.exists():
                logger.warning(f"Solar data file not found: {solar_path}")
                return

            # 데이터 로드
            df = pd.read_csv(solar_path)
            df['일시'] = pd.to_datetime(df['일시'])
            df = df.rename(columns={
                '태양광 발전량(MWh)': 'solar_mwh',
                '태양광 설비용량(MW)': 'capacity_mw',
                '기온': 'temperature',
                '습도': 'humidity',
                '일사량': 'solar_radiation'
            })

            # 결측치 처리
            df['solar_mwh'] = pd.to_numeric(df['solar_mwh'], errors='coerce').fillna(0)

            # 시간/월별 추출
            df['hour'] = df['일시'].dt.hour
            df['month'] = df['일시'].dt.month

            # 시간/월별 평균 발전량 계산
            self.solar_hourly_avg = df.groupby(['month', 'hour']).agg({
                'solar_mwh': 'mean',
                'capacity_mw': 'last'
            }).reset_index()

            # 최신 설비 용량으로 스케일 조정 (2024년 기준)
            scale_factor = JEJU_SOLAR_CAPACITY_MW / self.solar_hourly_avg['capacity_mw'].iloc[-1]
            self.solar_hourly_avg['solar_mw_scaled'] = self.solar_hourly_avg['solar_mwh'] * scale_factor

            logger.info(f"Loaded solar patterns: {len(self.solar_hourly_avg)} hour/month combinations")
            logger.info(f"Scale factor for current capacity: {scale_factor:.2f}x")

        except Exception as e:
            logger.error(f"Failed to load solar data: {e}")

    def estimate_solar_generation(
        self,
        timestamp: Optional[datetime] = None,
        cloud_factor: float = 1.0  # 0.0 (구름 많음) ~ 1.0 (맑음)
    ) -> float:
        """
        태양광 발전량 추정 (MW)

        Args:
            timestamp: 추정 시점 (기본: 현재)
            cloud_factor: 구름 보정 계수 (0.0~1.0)

        Returns:
            추정 발전량 (MW)
        """
        if timestamp is None:
            timestamp = datetime.now()

        hour = timestamp.hour
        month = timestamp.month

        # 패턴 기반 추정
        if self.solar_hourly_avg is not None:
            mask = (self.solar_hourly_avg['month'] == month) & (self.solar_hourly_avg['hour'] == hour)
            pattern_data = self.solar_hourly_avg[mask]

            if not pattern_data.empty:
                base_mw = pattern_data['solar_mw_scaled'].iloc[0]
                return max(0, base_mw * cloud_factor)

        # 폴백: 단순 일사량 모델
        if 6 <= hour <= 18:
            # 일출~일몰 시간의 태양광 발전
            solar_angle = np.sin((hour - 6) * np.pi / 12)
            base_mw = JEJU_SOLAR_CAPACITY_MW * solar_angle * 0.2  # 평균 이용률 20%
            return max(0, base_mw * cloud_factor)

        return 0.0

    def estimate_wind_generation(
        self,
        wind_speed: float,  # m/s
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        풍력 발전량 추정 (MW) - Power Curve 기반

        풍력 터빈 출력 곡선:
        - Cut-in speed: 3 m/s
        - Rated speed: 12-14 m/s
        - Cut-out speed: 25 m/s

        Args:
            wind_speed: 풍속 (m/s)
            timestamp: 추정 시점 (미사용, 확장용)

        Returns:
            추정 발전량 (MW)
        """
        # 풍력 발전기 특성
        cut_in = 3.0    # 발전 시작 풍속 (m/s)
        rated = 12.0    # 정격 풍속 (m/s)
        cut_out = 25.0  # 발전 정지 풍속 (m/s)

        if wind_speed < cut_in:
            # 풍속 미달 - 발전 없음
            capacity_factor = 0.0
        elif wind_speed < rated:
            # 부분 출력 영역 (3차 곡선)
            # P = k * v^3 (풍력 에너지는 풍속의 세제곱에 비례)
            capacity_factor = ((wind_speed - cut_in) / (rated - cut_in)) ** 3
            # 최대 정격의 85% 정도로 제한 (실제 터빈 효율)
            capacity_factor = min(capacity_factor, 0.85)
        elif wind_speed <= cut_out:
            # 정격 출력 영역
            capacity_factor = 0.85  # 정격 출력의 85%
        else:
            # 강풍으로 인한 발전 정지
            capacity_factor = 0.0

        # 제주 풍력 발전량 추정
        wind_mw = JEJU_WIND_CAPACITY_MW * capacity_factor

        return wind_mw

    def estimate_current(
        self,
        wind_speed: float = 5.0,
        humidity: float = 60.0,
        timestamp: Optional[datetime] = None
    ) -> RenewableGeneration:
        """
        현재 재생에너지 발전량 추정

        Args:
            wind_speed: 풍속 (m/s)
            humidity: 습도 (%) - 구름양 추정용
            timestamp: 추정 시점

        Returns:
            RenewableGeneration 객체
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 습도 기반 구름 보정 (습도 높으면 구름 많음)
        # 습도 60% 이하: 맑음 (1.0)
        # 습도 80% 이상: 흐림 (0.3)
        if humidity <= 60:
            cloud_factor = 1.0
        elif humidity >= 80:
            cloud_factor = 0.3
        else:
            cloud_factor = 1.0 - (humidity - 60) / 20 * 0.7

        # 태양광 추정
        solar_mw = self.estimate_solar_generation(timestamp, cloud_factor)

        # 풍력 추정
        wind_mw = self.estimate_wind_generation(wind_speed, timestamp)

        # 이용률 계산
        solar_cf = (solar_mw / JEJU_SOLAR_CAPACITY_MW * 100) if JEJU_SOLAR_CAPACITY_MW > 0 else 0
        wind_cf = (wind_mw / JEJU_WIND_CAPACITY_MW * 100) if JEJU_WIND_CAPACITY_MW > 0 else 0

        return RenewableGeneration(
            timestamp=timestamp,
            solar_mw=solar_mw,
            wind_mw=wind_mw,
            total_mw=solar_mw + wind_mw,
            solar_capacity_mw=JEJU_SOLAR_CAPACITY_MW,
            wind_capacity_mw=JEJU_WIND_CAPACITY_MW,
            solar_cf=solar_cf,
            wind_cf=wind_cf,
            data_source=f"Historical pattern (solar) + Power curve (wind, {wind_speed:.1f}m/s)"
        )

    def estimate_hourly(
        self,
        wind_speed_forecast: Optional[list] = None,
        humidity_forecast: Optional[list] = None,
        base_date: Optional[datetime] = None
    ) -> list[RenewableGeneration]:
        """
        24시간 재생에너지 발전량 추정

        Args:
            wind_speed_forecast: 24시간 풍속 예보 (m/s)
            humidity_forecast: 24시간 습도 예보 (%)
            base_date: 기준 날짜

        Returns:
            24개의 RenewableGeneration 객체 리스트
        """
        if base_date is None:
            base_date = datetime.now().replace(minute=0, second=0, microsecond=0)

        # 기본값: 일정한 풍속/습도
        if wind_speed_forecast is None:
            wind_speed_forecast = [5.0] * 24
        if humidity_forecast is None:
            humidity_forecast = [60.0] * 24

        results = []
        for hour in range(24):
            ts = base_date.replace(hour=hour)
            gen = self.estimate_current(
                wind_speed=wind_speed_forecast[hour],
                humidity=humidity_forecast[hour],
                timestamp=ts
            )
            results.append(gen)

        return results


# 싱글톤 인스턴스
_estimator = None

def get_estimator() -> RenewableEnergyEstimator:
    """싱글톤 추정기 인스턴스 반환"""
    global _estimator
    if _estimator is None:
        _estimator = RenewableEnergyEstimator()
    return _estimator


def estimate_current_renewable(
    wind_speed: float = 5.0,
    humidity: float = 60.0
) -> RenewableGeneration:
    """현재 재생에너지 발전량 추정 (편의 함수)"""
    return get_estimator().estimate_current(wind_speed=wind_speed, humidity=humidity)


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    estimator = RenewableEnergyEstimator()

    print("=" * 60)
    print("제주 재생에너지 발전량 추정 테스트")
    print("=" * 60)

    # 현재 시간 추정
    current = estimator.estimate_current(wind_speed=5.0, humidity=50.0)
    print(f"\n현재 시간 ({current.timestamp.strftime('%H:%M')}):")
    print(f"  태양광: {current.solar_mw:.1f} MW (이용률 {current.solar_cf:.1f}%)")
    print(f"  풍력:   {current.wind_mw:.1f} MW (이용률 {current.wind_cf:.1f}%)")
    print(f"  합계:   {current.total_mw:.1f} MW")

    # 24시간 추정
    print("\n24시간 발전량 추정:")
    hourly = estimator.estimate_hourly()
    for gen in hourly:
        print(f"  {gen.timestamp.hour:02d}:00 - 태양광: {gen.solar_mw:6.1f} MW, 풍력: {gen.wind_mw:6.1f} MW, 합계: {gen.total_mw:6.1f} MW")

    print("=" * 60)
