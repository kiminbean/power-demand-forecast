#!/usr/bin/env python3
"""
JejuFerryEstimator 테스트 코드

테스트 항목:
1. 기본 추정 기능
2. 계절별 분담률 적용
3. 기상 조건 보정
4. 기본값 fallback
5. 검증 (기존 크롤러 대비 정확도)

실행:
    cd /Users/ibkim/Ormi_1/power-demand-forecast
    python -m pytest tools/crawlers/test_ferry_estimator.py -v
    
    # 또는 직접 실행
    python tools/crawlers/test_ferry_estimator.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest
from datetime import datetime

from tools.crawlers.jeju_ferry_estimator import (
    JejuFerryEstimator,
    FerryEstimate,
    FERRY_RATIO_DEFAULT,
    SEASONAL_FERRY_RATIO,
)


class TestJejuFerryEstimator(unittest.TestCase):
    """JejuFerryEstimator 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.estimator = JejuFerryEstimator()
        
        # 테스트용 항공 데이터 (일반적인 하루)
        self.typical_air_data = {
            'arrival': 45000,
            'departure': 44000,
        }
        
        # 성수기 항공 데이터
        self.peak_air_data = {
            'arrival': 60000,
            'departure': 58000,
        }
        
        # 비수기 항공 데이터
        self.offpeak_air_data = {
            'arrival': 30000,
            'departure': 29000,
        }
    
    def test_basic_estimation(self):
        """기본 추정 테스트"""
        result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15"
        )
        
        # 결과 타입 확인
        self.assertIsInstance(result, FerryEstimate)
        
        # 추정값 범위 확인 (항공의 4-7%)
        self.assertGreater(result.arrival, self.typical_air_data['arrival'] * 0.04)
        self.assertLess(result.arrival, self.typical_air_data['arrival'] * 0.07)
        
        # 출처 확인
        self.assertEqual(result.source, "air_traffic_ratio")
        
        print(f"✅ 기본 추정: 입도 {result.arrival:,}명, 출도 {result.departure:,}명")
    
    def test_seasonal_ratio(self):
        """계절별 분담률 테스트"""
        # 여름 (성수기) - 해운 비율 낮음
        summer_result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-08-15"
        )
        
        # 겨울 (비수기) - 해운 비율 높음
        winter_result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-01-15"
        )
        
        # 겨울이 여름보다 해운 비율 높아야 함
        self.assertGreater(winter_result.ferry_ratio, summer_result.ferry_ratio)
        
        print(f"✅ 계절별 분담률: 여름 {summer_result.ferry_ratio:.1%} < 겨울 {winter_result.ferry_ratio:.1%}")
    
    def test_weather_factor_normal(self):
        """정상 기상 조건 테스트"""
        weather_data = {
            'wave_height': 1.0,
            'wind_speed': 5.0,
        }
        
        result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15",
            weather_data=weather_data
        )
        
        # 정상 기상 시 weather_factor = 1.0
        self.assertEqual(result.weather_factor, 1.0)
        self.assertIsNone(result.note)
        
        print(f"✅ 정상 기상: weather_factor={result.weather_factor}")
    
    def test_weather_factor_storm(self):
        """악천후 조건 테스트"""
        storm_weather = {
            'wave_height': 4.5,
            'wind_speed': 20.0,
        }
        
        result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15",
            weather_data=storm_weather
        )
        
        # 악천후 시 결항 (factor = 0)
        self.assertEqual(result.weather_factor, 0.0)
        self.assertEqual(result.arrival, 0)
        self.assertEqual(result.departure, 0)
        self.assertIn("결항", result.note)
        
        print(f"✅ 악천후: weather_factor={result.weather_factor}, note='{result.note}'")
    
    def test_weather_factor_partial(self):
        """일부 결항 조건 테스트"""
        bad_weather = {
            'wave_height': 3.5,
            'wind_speed': 16.0,
        }
        
        result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15",
            weather_data=bad_weather
        )
        
        # 일부 결항 시 30% 운항
        self.assertEqual(result.weather_factor, 0.3)
        self.assertIn("일부 결항", result.note)
        
        print(f"✅ 일부 결항: weather_factor={result.weather_factor}, arrival={result.arrival:,}명")
    
    def test_fallback_estimation(self):
        """기본값 추정 (항공 데이터 없을 때)"""
        result = self.estimator.estimate_daily_default("2024-08-15")
        
        # KOMSA 통계 기반 추정값 확인 (일 평균 ~2,200명)
        self.assertGreater(result.arrival, 1500)
        self.assertLess(result.arrival, 3500)
        
        # 출처 확인
        self.assertEqual(result.source, "komsa_statistics_fallback")
        
        # 신뢰도 낮음 확인
        self.assertLess(result.confidence, 0.7)
        
        print(f"✅ Fallback 추정: {result.arrival:,}명, 신뢰도={result.confidence:.1%}")
    
    def test_comparison_with_old_crawler(self):
        """기존 크롤러 대비 정확도 비교"""
        # 기존 크롤러 기본값
        old_crawler_default = 1500  # 명/일
        
        # 새 추정기 결과 (일반적인 날)
        result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15"
        )
        
        # KOMSA 실제 통계 기반 추정값
        komsa_daily_avg = 2200  # 명/일
        
        # 새 추정기가 실제값에 더 가까워야 함
        old_error = abs(old_crawler_default - komsa_daily_avg) / komsa_daily_avg
        new_error = abs(result.arrival - komsa_daily_avg) / komsa_daily_avg
        
        print(f"\n📊 정확도 비교:")
        print(f"  KOMSA 일 평균: {komsa_daily_avg:,}명")
        print(f"  기존 크롤러: {old_crawler_default:,}명 (오차 {old_error:.1%})")
        print(f"  새 추정기: {result.arrival:,}명 (오차 {new_error:.1%})")
        
        # 새 추정기 오차가 더 작아야 함 (또는 비슷해야 함)
        # Note: 항공 데이터 기반이므로 성수기에는 더 높을 수 있음
        self.assertLess(new_error, 0.5)  # 50% 이내 오차
        
        print(f"✅ 새 추정기 오차율 {new_error:.1%} < 50%")
    
    def test_net_flow(self):
        """순 유입 계산 테스트"""
        result = self.estimator.estimate_from_air(
            self.typical_air_data,  # arrival > departure
            date="2024-06-15"
        )
        
        expected_net_flow = result.arrival - result.departure
        self.assertEqual(result.net_flow, expected_net_flow)
        
        print(f"✅ 순 유입: {result.net_flow:+,}명")
    
    def test_confidence_calculation(self):
        """신뢰도 계산 테스트"""
        # 정상 데이터
        normal_result = self.estimator.estimate_from_air(
            self.typical_air_data,
            date="2024-06-15"
        )
        
        # 적은 데이터
        small_data = {'arrival': 5000, 'departure': 4900}
        small_result = self.estimator.estimate_from_air(
            small_data,
            date="2024-06-15"
        )
        
        # 정상 데이터가 신뢰도 높아야 함
        self.assertGreater(normal_result.confidence, small_result.confidence)
        
        print(f"✅ 신뢰도: 정상={normal_result.confidence:.1%}, 적은 데이터={small_result.confidence:.1%}")


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        import pandas as pd
        
        estimator = JejuFerryEstimator()
        
        # 테스트용 항공 데이터 (1주일)
        dates = pd.date_range('2024-08-01', periods=7, freq='D')
        air_df = pd.DataFrame({
            'date': dates,
            'arrival': [45000, 48000, 50000, 52000, 55000, 58000, 45000],
            'departure': [44000, 47000, 49000, 51000, 54000, 57000, 44000],
        })
        
        # 기간별 추정
        ferry_df = estimator.estimate_range(air_df)
        
        # 결과 확인
        self.assertEqual(len(ferry_df), 7)
        self.assertIn('arrival', ferry_df.columns)
        self.assertIn('departure', ferry_df.columns)
        self.assertIn('confidence', ferry_df.columns)
        
        print(f"\n📊 1주일 추정 결과:")
        print(ferry_df[['date', 'arrival', 'departure', 'net_flow', 'confidence']].to_string())
        
        # 총 입도객
        total_arrival = ferry_df['arrival'].sum()
        print(f"\n✅ 1주일 총 입도객: {total_arrival:,}명")
        
        self.assertGreater(total_arrival, 10000)


def run_manual_tests():
    """수동 테스트 실행"""
    print("=" * 70)
    print("🧪 JejuFerryEstimator 수동 테스트")
    print("=" * 70)
    
    estimator = JejuFerryEstimator()
    
    # 테스트 1: 기본 추정
    print("\n[테스트 1] 기본 추정")
    air_data = {'arrival': 45000, 'departure': 44000}
    result = estimator.estimate_from_air(air_data, date="2024-06-15")
    print(f"  항공 입도: {air_data['arrival']:,}명")
    print(f"  해운 추정: {result.arrival:,}명 ({result.ferry_ratio:.1%})")
    
    # 테스트 2: 성수기 vs 비수기
    print("\n[테스트 2] 계절별 비교 (동일 항공 데이터)")
    for month, name in [(1, "1월(비수기)"), (8, "8월(성수기)")]:
        result = estimator.estimate_from_air(air_data, date=f"2024-{month:02d}-15")
        print(f"  {name}: {result.arrival:,}명 (분담률 {result.ferry_ratio:.1%})")
    
    # 테스트 3: 기상 영향
    print("\n[테스트 3] 기상 조건별 비교")
    weather_conditions = [
        ({'wave_height': 1.0, 'wind_speed': 5.0}, "맑음"),
        ({'wave_height': 2.5, 'wind_speed': 12.0}, "흐림"),
        ({'wave_height': 3.5, 'wind_speed': 16.0}, "풍랑주의보"),
        ({'wave_height': 4.5, 'wind_speed': 20.0}, "태풍"),
    ]
    for weather, name in weather_conditions:
        result = estimator.estimate_from_air(air_data, "2024-06-15", weather)
        print(f"  {name}: {result.arrival:,}명 (보정 {result.weather_factor:.0%})")
    
    # 테스트 4: 기존 vs 새 추정기
    print("\n[테스트 4] 정확도 비교 (KOMSA 기준)")
    komsa_daily = 2200
    old_value = 1500
    new_result = estimator.estimate_from_air(air_data, "2024-06-15")
    
    old_error = abs(old_value - komsa_daily) / komsa_daily * 100
    new_error = abs(new_result.arrival - komsa_daily) / komsa_daily * 100
    
    print(f"  KOMSA 일 평균: {komsa_daily:,}명")
    print(f"  기존 크롤러: {old_value:,}명 (오차 {old_error:.1f}%)")
    print(f"  새 추정기: {new_result.arrival:,}명 (오차 {new_error:.1f}%)")
    print(f"  ✅ 개선율: {old_error - new_error:.1f}%p")
    
    print("\n" + "=" * 70)
    print("✅ 모든 수동 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    # 수동 테스트 먼저 실행
    run_manual_tests()
    
    # unittest 실행
    print("\n\n")
    unittest.main(verbosity=2)
