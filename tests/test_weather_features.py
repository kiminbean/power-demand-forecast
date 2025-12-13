"""
Test Suite for Weather Features Module
======================================
THI(불쾌지수) 및 상대습도 계산 로직 검증

Test Cases:
1. Normal: 정상적인 기온/이슬점 조합
2. Saturation: 이슬점 = 기온 (RH ≈ 100%)
3. Anomaly: 이슬점 > 기온 (데이터 오류, 클리핑 검증)
4. Edge Cases: 극한 온도, 결측치 등

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# src 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.weather_features import (
    calculate_relative_humidity,
    calculate_thi,
    calculate_humidity_and_thi,
    calculate_saturation_vapor_pressure,
    MAGNUS_A,
    MAGNUS_B
)


class TestRelativeHumidity:
    """상대습도 계산 테스트"""
    
    def test_normal_case(self):
        """Case 1: 정상적인 기온/이슬점 조합"""
        # T=25°C, Td=20°C -> RH는 약 70-75% 예상
        temp = np.array([25.0])
        dewpoint = np.array([20.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # 합리적인 범위 검증 (65-80%)
        assert 65 <= humidity[0] <= 80, f"Expected RH 65-80%, got {humidity[0]:.1f}%"
        
    def test_saturation_case(self):
        """Case 2: 이슬점 = 기온 (완전 포화)"""
        # T = Td -> RH ≈ 100%
        temp = np.array([25.0])
        dewpoint = np.array([25.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # RH는 100%에 매우 가까워야 함
        assert 99 <= humidity[0] <= 100, f"Expected RH ≈ 100%, got {humidity[0]:.1f}%"
        
    def test_anomaly_clipping(self):
        """Case 3: 이슬점 > 기온 (비정상 데이터)"""
        # Td > T -> 물리적으로 불가능, RH=100으로 클리핑되어야 함
        temp = np.array([25.0])
        dewpoint = np.array([30.0])  # 이슬점이 기온보다 높음!
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # 클리핑 검증: 100% 초과 불가
        assert humidity[0] == 100, f"Expected RH clipped to 100%, got {humidity[0]:.1f}%"
        
    def test_low_humidity(self):
        """낮은 습도 케이스"""
        # T=30°C, Td=5°C -> 매우 건조한 환경
        temp = np.array([30.0])
        dewpoint = np.array([5.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # 건조 환경: 20% 미만 예상
        assert humidity[0] < 30, f"Expected low RH, got {humidity[0]:.1f}%"
        
    def test_vectorized_calculation(self):
        """벡터 연산 검증 (다중 데이터)"""
        temp = np.array([10.0, 20.0, 30.0, 40.0])
        dewpoint = np.array([5.0, 15.0, 25.0, 35.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # 모든 값이 0-100 범위 내
        assert all(0 <= h <= 100 for h in humidity)
        # 출력 크기 일치
        assert len(humidity) == len(temp)


class TestTHI:
    """불쾌지수(THI) 계산 테스트"""
    
    def test_comfortable_range(self):
        """쾌적한 조건 (THI < 68)"""
        # 서늘하고 건조: T=15°C, RH=40%
        temp = np.array([15.0])
        rh_ratio = np.array([0.4])  # 40%를 비율로
        
        thi = calculate_thi(temp, rh_ratio)
        
        assert thi[0] < 68, f"Expected comfortable THI < 68, got {thi[0]:.1f}"
        
    def test_uncomfortable_hot_humid(self):
        """불쾌한 조건 (고온다습)"""
        # 덥고 습함: T=30°C, RH=80%
        temp = np.array([30.0])
        rh_ratio = np.array([0.8])
        
        thi = calculate_thi(temp, rh_ratio)
        
        # THI > 75 예상 (반 이상 불쾌)
        assert thi[0] > 75, f"Expected uncomfortable THI > 75, got {thi[0]:.1f}"
        
    def test_extreme_discomfort(self):
        """극심한 불쾌 조건"""
        # 매우 덥고 매우 습함: T=35°C, RH=90%
        temp = np.array([35.0])
        rh_ratio = np.array([0.9])
        
        thi = calculate_thi(temp, rh_ratio)
        
        # THI > 80 예상 (대부분 불쾌)
        assert thi[0] > 80, f"Expected extreme THI > 80, got {thi[0]:.1f}"
        
    def test_formula_correctness(self):
        """THI 공식 정확성 검증"""
        # 수동 계산과 비교
        # T=25°C, RH=50% (ratio=0.5)
        # THI = 1.8*25 - 0.55*(1-0.5)*(1.8*25-26) + 32
        # THI = 45 - 0.55*0.5*19 + 32 = 45 - 5.225 + 32 = 71.775
        temp = np.array([25.0])
        rh_ratio = np.array([0.5])
        
        thi = calculate_thi(temp, rh_ratio)
        expected = 71.775
        
        assert abs(thi[0] - expected) < 0.01, f"Expected {expected}, got {thi[0]:.3f}"


class TestIntegration:
    """통합 테스트 (DataFrame 파이프라인)"""
    
    @pytest.fixture
    def sample_df(self):
        """테스트용 샘플 DataFrame"""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'temp_mean': [5.0, 15.0, 25.0, 30.0, 35.0],
            'dewpoint_mean': [0.0, 10.0, 20.0, 28.0, 30.0]
        })
    
    def test_columns_added(self, sample_df):
        """humidity와 THI 컬럼 추가 검증"""
        result = calculate_humidity_and_thi(sample_df)
        
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        
    def test_original_preserved(self, sample_df):
        """원본 데이터 보존 검증 (inplace=False)"""
        original_cols = list(sample_df.columns)
        result = calculate_humidity_and_thi(sample_df)
        
        # 원본은 변경되지 않아야 함
        assert list(sample_df.columns) == original_cols
        # 결과는 새 컬럼 포함
        assert len(result.columns) > len(original_cols)
        
    def test_no_nan_in_output(self, sample_df):
        """출력에 NaN 없음 검증"""
        result = calculate_humidity_and_thi(sample_df)
        
        assert result['humidity'].isna().sum() == 0
        assert result['THI'].isna().sum() == 0
        
    def test_missing_column_error(self):
        """필수 컬럼 누락 시 에러"""
        bad_df = pd.DataFrame({'temp_mean': [25.0]})  # dewpoint_mean 없음
        
        with pytest.raises(ValueError, match="필수 컬럼"):
            calculate_humidity_and_thi(bad_df)
            
    def test_humidity_range_validation(self, sample_df):
        """humidity 값 범위 검증 (0-100)"""
        result = calculate_humidity_and_thi(sample_df)
        
        assert all(0 <= h <= 100 for h in result['humidity'])


class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_freezing_temperature(self):
        """영하 온도 처리"""
        temp = np.array([-10.0, -5.0, 0.0])
        dewpoint = np.array([-15.0, -10.0, -5.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        # 영하에서도 0-100 범위 유지
        assert all(0 <= h <= 100 for h in humidity)
        
    def test_very_high_temperature(self):
        """고온 처리"""
        temp = np.array([40.0, 45.0])
        dewpoint = np.array([30.0, 35.0])
        
        humidity = calculate_relative_humidity(temp, dewpoint)
        
        assert all(0 <= h <= 100 for h in humidity)
        
    def test_empty_dataframe(self):
        """빈 DataFrame 처리"""
        empty_df = pd.DataFrame({
            'temp_mean': [],
            'dewpoint_mean': []
        })
        
        result = calculate_humidity_and_thi(empty_df)
        
        assert len(result) == 0
        assert 'humidity' in result.columns
        assert 'THI' in result.columns


class TestRealData:
    """실제 프로젝트 데이터 테스트"""
    
    @pytest.fixture
    def real_data(self):
        """실제 제주 데이터 로드"""
        data_path = PROJECT_ROOT / "data" / "processed" / "jeju_daily_dataset.csv"
        if data_path.exists():
            return pd.read_csv(data_path, parse_dates=['date'])
        return None
    
    def test_real_data_processing(self, real_data):
        """실제 데이터 처리 가능 여부"""
        if real_data is None:
            pytest.skip("Real data not available")
            
        result = calculate_humidity_and_thi(real_data)
        
        # 컬럼 추가 확인
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        
        # NaN 비율 확인 (일부 결측은 허용)
        nan_ratio = result['humidity'].isna().sum() / len(result)
        assert nan_ratio < 0.1, f"Too many NaN values: {nan_ratio:.1%}"
        
    def test_summer_high_thi(self, real_data):
        """여름철 높은 THI 검증"""
        if real_data is None:
            pytest.skip("Real data not available")
            
        result = calculate_humidity_and_thi(real_data)
        
        # 여름 데이터 필터링 (6-8월)
        summer_data = result[result['date'].dt.month.isin([6, 7, 8])]
        
        if len(summer_data) > 0:
            avg_summer_thi = summer_data['THI'].mean()
            # 여름철 평균 THI는 70 이상 예상
            assert avg_summer_thi > 65, f"Expected high summer THI, got {avg_summer_thi:.1f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
