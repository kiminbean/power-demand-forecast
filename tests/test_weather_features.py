"""
Test Suite for Weather Features Module
======================================
THI(불쾌지수), 상대습도, Wind Chill(체감온도) 계산 로직 검증

Test Cases:
- FEAT-001: THI 및 RH 계산
- FEAT-002: Wind Chill 계산

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
    # FEAT-001: THI
    calculate_relative_humidity,
    calculate_thi,
    calculate_humidity_and_thi,
    calculate_saturation_vapor_pressure,
    MAGNUS_A,
    MAGNUS_B,
    # FEAT-002: Wind Chill
    calculate_wind_chill,
    calculate_wind_chill_from_df,
    convert_ms_to_kmh,
    add_weather_features,
    WIND_CHILL_TEMP_THRESHOLD,
    WIND_CHILL_WIND_THRESHOLD,
    MS_TO_KMH
)


# ============================================================================
# FEAT-001: THI Tests
# ============================================================================

class TestRelativeHumidity:
    """상대습도 계산 테스트"""
    
    def test_normal_case(self):
        """Case 1: 정상적인 기온/이슬점 조합"""
        temp = np.array([25.0])
        dewpoint = np.array([20.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert 65 <= humidity[0] <= 80, f"Expected RH 65-80%, got {humidity[0]:.1f}%"
        
    def test_saturation_case(self):
        """Case 2: 이슬점 = 기온 (완전 포화)"""
        temp = np.array([25.0])
        dewpoint = np.array([25.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert 99 <= humidity[0] <= 100, f"Expected RH ≈ 100%, got {humidity[0]:.1f}%"
        
    def test_anomaly_clipping(self):
        """Case 3: 이슬점 > 기온 (비정상 데이터)"""
        temp = np.array([25.0])
        dewpoint = np.array([30.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert humidity[0] == 100, f"Expected RH clipped to 100%, got {humidity[0]:.1f}%"
        
    def test_low_humidity(self):
        """낮은 습도 케이스"""
        temp = np.array([30.0])
        dewpoint = np.array([5.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert humidity[0] < 30, f"Expected low RH, got {humidity[0]:.1f}%"
        
    def test_vectorized_calculation(self):
        """벡터 연산 검증"""
        temp = np.array([10.0, 20.0, 30.0, 40.0])
        dewpoint = np.array([5.0, 15.0, 25.0, 35.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert all(0 <= h <= 100 for h in humidity)
        assert len(humidity) == len(temp)


class TestTHI:
    """불쾌지수(THI) 계산 테스트"""
    
    def test_comfortable_range(self):
        """쾌적한 조건 (THI < 68)"""
        temp = np.array([15.0])
        rh_ratio = np.array([0.4])
        thi = calculate_thi(temp, rh_ratio)
        assert thi[0] < 68, f"Expected comfortable THI < 68, got {thi[0]:.1f}"
        
    def test_uncomfortable_hot_humid(self):
        """불쾌한 조건 (고온다습)"""
        temp = np.array([30.0])
        rh_ratio = np.array([0.8])
        thi = calculate_thi(temp, rh_ratio)
        assert thi[0] > 75, f"Expected uncomfortable THI > 75, got {thi[0]:.1f}"
        
    def test_extreme_discomfort(self):
        """극심한 불쾌 조건"""
        temp = np.array([35.0])
        rh_ratio = np.array([0.9])
        thi = calculate_thi(temp, rh_ratio)
        assert thi[0] > 80, f"Expected extreme THI > 80, got {thi[0]:.1f}"
        
    def test_formula_correctness(self):
        """THI 공식 정확성 검증"""
        temp = np.array([25.0])
        rh_ratio = np.array([0.5])
        thi = calculate_thi(temp, rh_ratio)
        expected = 71.775
        assert abs(thi[0] - expected) < 0.01, f"Expected {expected}, got {thi[0]:.3f}"


class TestTHIIntegration:
    """THI 통합 테스트"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'temp_mean': [5.0, 15.0, 25.0, 30.0, 35.0],
            'dewpoint_mean': [0.0, 10.0, 20.0, 28.0, 30.0]
        })
    
    def test_columns_added(self, sample_df):
        result = calculate_humidity_and_thi(sample_df)
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        
    def test_original_preserved(self, sample_df):
        original_cols = list(sample_df.columns)
        result = calculate_humidity_and_thi(sample_df)
        assert list(sample_df.columns) == original_cols
        
    def test_no_nan_in_output(self, sample_df):
        result = calculate_humidity_and_thi(sample_df)
        assert result['humidity'].isna().sum() == 0
        assert result['THI'].isna().sum() == 0
        
    def test_missing_column_error(self):
        bad_df = pd.DataFrame({'temp_mean': [25.0]})
        with pytest.raises(ValueError, match="필수 컬럼"):
            calculate_humidity_and_thi(bad_df)


# ============================================================================
# FEAT-002: Wind Chill Tests
# ============================================================================

class TestUnitConversion:
    """단위 변환 테스트"""
    
    def test_ms_to_kmh_conversion(self):
        """m/s -> km/h 변환 정확성"""
        wind_ms = np.array([1.0, 5.0, 10.0])
        wind_kmh = convert_ms_to_kmh(wind_ms)
        
        expected = np.array([3.6, 18.0, 36.0])
        np.testing.assert_array_almost_equal(wind_kmh, expected)
        
    def test_ms_to_kmh_zero(self):
        """풍속 0 처리"""
        wind_ms = np.array([0.0])
        wind_kmh = convert_ms_to_kmh(wind_ms)
        assert wind_kmh[0] == 0.0


class TestWindChill:
    """체감온도 계산 테스트"""
    
    def test_normal_winter_case(self):
        """Case 1: 정상적인 동절기 조건"""
        temp = np.array([0.0])
        wind_kmh = np.array([20.0])
        wc = calculate_wind_chill(temp, wind_kmh)
        assert wc[0] < temp[0], f"Wind chill should be lower than temp, got {wc[0]:.1f}"
        
    def test_formula_correctness(self):
        """JAG/Siple 공식 정확성 검증"""
        temp = np.array([-10.0])
        wind_kmh = np.array([30.0])
        v_power = 30.0 ** 0.16
        expected = 13.12 + 0.6215 * (-10) - 11.37 * v_power + 0.3965 * (-10) * v_power
        wc = calculate_wind_chill(temp, wind_kmh)
        assert abs(wc[0] - expected) < 0.1, f"Expected {expected:.2f}, got {wc[0]:.2f}"
        
    def test_high_wind_extreme_cold(self):
        """극한 조건: 강풍 + 혹한"""
        temp = np.array([-20.0])
        wind_kmh = np.array([50.0])
        wc = calculate_wind_chill(temp, wind_kmh)
        assert wc[0] < -30, f"Expected extreme wind chill < -30, got {wc[0]:.1f}"
        
    def test_zero_wind(self):
        """풍속 0일 때"""
        temp = np.array([5.0])
        wind_kmh = np.array([0.0])
        wc = calculate_wind_chill(temp, wind_kmh)
        assert abs(wc[0] - 16.23) < 0.1
        
    def test_validity_mask_applied(self):
        """유효 범위 마스크 적용 테스트"""
        temp = np.array([15.0])
        wind_kmh = np.array([20.0])
        wc_no_mask = calculate_wind_chill(temp, wind_kmh, apply_validity_mask=False)
        wc_with_mask = calculate_wind_chill(temp, wind_kmh, apply_validity_mask=True)
        assert wc_with_mask[0] == temp[0], "Should return original temp when out of valid range"
        assert wc_no_mask[0] != temp[0], "Should apply formula when mask is disabled"
        
    def test_low_wind_validity_mask(self):
        """낮은 풍속에서 유효 범위 마스크"""
        temp = np.array([0.0])
        wind_kmh = np.array([3.0])
        wc_with_mask = calculate_wind_chill(temp, wind_kmh, apply_validity_mask=True)
        assert wc_with_mask[0] == temp[0], "Should return original temp when wind too low"
        
    def test_vectorized_calculation(self):
        """벡터 연산 검증"""
        temp = np.array([5.0, 0.0, -5.0, -10.0])
        wind_kmh = np.array([10.0, 20.0, 30.0, 40.0])
        wc = calculate_wind_chill(temp, wind_kmh)
        assert len(wc) == len(temp)
        assert all(wc[i] <= temp[i] for i in range(len(temp)))


class TestWindChillDataFrame:
    """Wind Chill DataFrame 통합 테스트"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'temp_mean': [10.0, 5.0, 0.0, -5.0, -10.0],
            'wind_speed_mean': [2.0, 5.0, 8.0, 10.0, 12.0]
        })
    
    def test_column_added(self, sample_df):
        result = calculate_wind_chill_from_df(sample_df)
        assert 'wind_chill' in result.columns
        
    def test_unit_conversion_applied(self, sample_df):
        result = calculate_wind_chill_from_df(sample_df, wind_unit='ms')
        wc_at_idx1 = result.loc[1, 'wind_chill']
        v_power = 18.0 ** 0.16
        expected = 13.12 + 0.6215 * 5 - 11.37 * v_power + 0.3965 * 5 * v_power
        assert abs(wc_at_idx1 - expected) < 0.1
        
    def test_kmh_unit_direct(self):
        df = pd.DataFrame({'temp_mean': [0.0], 'wind_speed_mean': [20.0]})
        result = calculate_wind_chill_from_df(df, wind_unit='kmh')
        v_power = 20.0 ** 0.16
        expected = 13.12 + 0.6215 * 0 - 11.37 * v_power + 0.3965 * 0 * v_power
        assert abs(result['wind_chill'].iloc[0] - expected) < 0.1
        
    def test_invalid_unit_error(self):
        df = pd.DataFrame({'temp_mean': [0.0], 'wind_speed_mean': [5.0]})
        with pytest.raises(ValueError, match="잘못된 풍속 단위"):
            calculate_wind_chill_from_df(df, wind_unit='mph')
            
    def test_missing_column_error(self):
        df = pd.DataFrame({'temp_mean': [0.0]})
        with pytest.raises(ValueError, match="필수 컬럼"):
            calculate_wind_chill_from_df(df)
            
    def test_nan_handling(self):
        df = pd.DataFrame({
            'temp_mean': [0.0, np.nan, -5.0],
            'wind_speed_mean': [5.0, 10.0, np.nan]
        })
        result = calculate_wind_chill_from_df(df)
        assert not np.isnan(result['wind_chill'].iloc[0])
        assert np.isnan(result['wind_chill'].iloc[1])
        assert np.isnan(result['wind_chill'].iloc[2])


# ============================================================================
# Unified Feature Tests
# ============================================================================

class TestAddWeatherFeatures:
    """통합 피처 생성 테스트"""
    
    @pytest.fixture
    def full_df(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'temp_mean': [30.0, 20.0, 5.0, -5.0],
            'dewpoint_mean': [25.0, 15.0, 0.0, -10.0],
            'wind_speed_mean': [2.0, 3.0, 8.0, 12.0]
        })
    
    def test_all_features_added(self, full_df):
        result = add_weather_features(full_df)
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        assert 'wind_chill' in result.columns
        
    def test_selective_features(self, full_df):
        result_thi = add_weather_features(full_df, include_thi=True, include_wind_chill=False)
        assert 'THI' in result_thi.columns
        assert 'wind_chill' not in result_thi.columns
        
        result_wc = add_weather_features(full_df, include_thi=False, include_wind_chill=True)
        assert 'THI' not in result_wc.columns
        assert 'wind_chill' in result_wc.columns
        
    def test_missing_dewpoint_warning(self):
        df = pd.DataFrame({'temp_mean': [5.0], 'wind_speed_mean': [8.0]})
        with pytest.warns(UserWarning, match="THI를 계산할 수 없습니다"):
            result = add_weather_features(df, include_thi=True)
        assert 'wind_chill' in result.columns


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_freezing_temperature_thi(self):
        temp = np.array([-10.0, -5.0, 0.0])
        dewpoint = np.array([-15.0, -10.0, -5.0])
        humidity = calculate_relative_humidity(temp, dewpoint)
        assert all(0 <= h <= 100 for h in humidity)
        
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame({
            'temp_mean': [],
            'dewpoint_mean': [],
            'wind_speed_mean': []
        })
        result = add_weather_features(empty_df)
        assert len(result) == 0
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        assert 'wind_chill' in result.columns


# ============================================================================
# Real Data Tests
# ============================================================================

class TestRealData:
    """실제 프로젝트 데이터 테스트"""
    
    @pytest.fixture
    def real_data(self):
        data_path = PROJECT_ROOT / "data" / "processed" / "jeju_daily_dataset.csv"
        if data_path.exists():
            return pd.read_csv(data_path, parse_dates=['date'])
        return None
    
    def test_real_data_thi_features(self, real_data):
        """실제 데이터 THI 피처 생성 (wind_speed_mean 불필요)"""
        if real_data is None:
            pytest.skip("Real data not available")
            
        result = add_weather_features(real_data, include_wind_chill=False)
        
        assert 'humidity' in result.columns
        assert 'THI' in result.columns
        
    def test_real_data_wind_chill_conditional(self, real_data):
        """실제 데이터 wind_chill (컬럼 있을 때만)"""
        if real_data is None:
            pytest.skip("Real data not available")
        
        # wind_speed_mean 컬럼이 없으면 스킵
        if 'wind_speed_mean' not in real_data.columns:
            pytest.skip("wind_speed_mean column not available")
            
        result = add_weather_features(real_data)
        assert 'wind_chill' in result.columns
        
    def test_summer_high_thi(self, real_data):
        """여름철 높은 THI 검증"""
        if real_data is None:
            pytest.skip("Real data not available")
            
        result = add_weather_features(real_data, include_wind_chill=False)
        summer_data = result[result['date'].dt.month.isin([6, 7, 8])]
        
        if len(summer_data) > 0:
            avg_summer_thi = summer_data['THI'].mean()
            assert avg_summer_thi > 65, f"Expected high summer THI, got {avg_summer_thi:.1f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
