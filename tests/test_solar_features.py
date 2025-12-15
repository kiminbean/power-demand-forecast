"""
FEAT-004: 태양광 관련 파생 변수 단위 테스트

테스트 범위:
1. 일출/일몰 계산 (태양 고도, 적위, 시간각)
2. 일사량 관련 계산 (이론 일사량, 맑은 하늘 지수)
3. 구름 감쇠 계수
4. 태양광 발전량 추정
5. BTM 태양광 효과
6. 통합 함수 (add_solar_features)
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.solar_features import (
    # 상수
    JEJU_LATITUDE,
    JEJU_LONGITUDE,
    # 태양 위치 계산
    calculate_day_of_year,
    calculate_solar_declination,
    calculate_hour_angle,
    calculate_solar_elevation,
    is_daylight,
    calculate_daylight_hours,
    # 일사량 계산
    calculate_theoretical_irradiance,
    calculate_clear_sky_index,
    calculate_cloud_attenuation,
    estimate_solar_generation,
    # BTM
    estimate_btm_solar,
    calculate_btm_effect,
    # 통합 함수
    add_solar_features,
    get_solar_feature_names,
)


# ============================================================
# 태양 위치 계산 테스트
# ============================================================

class TestDayOfYear:
    """calculate_day_of_year 함수 테스트"""

    def test_jan_1(self):
        """1월 1일 = 1일"""
        dates = pd.DatetimeIndex(['2024-01-01'])
        doy = calculate_day_of_year(dates)
        assert doy[0] == 1

    def test_dec_31(self):
        """12월 31일 = 365/366일"""
        dates = pd.DatetimeIndex(['2024-12-31'])  # 2024는 윤년
        doy = calculate_day_of_year(dates)
        assert doy[0] == 366

    def test_summer_solstice(self):
        """하지 (6월 21일) = 172/173일"""
        dates = pd.DatetimeIndex(['2024-06-21'])
        doy = calculate_day_of_year(dates)
        assert doy[0] == 173  # 윤년


class TestSolarDeclination:
    """calculate_solar_declination 함수 테스트"""

    def test_summer_solstice(self):
        """하지에 최대 양수 적위 (~+23.5°)"""
        doy = np.array([172])  # 6월 21일경
        dec = calculate_solar_declination(doy)
        dec_deg = dec[0] * 180 / np.pi
        assert 20 < dec_deg < 25  # 약 23.5도

    def test_winter_solstice(self):
        """동지에 최대 음수 적위 (~-23.5°)"""
        doy = np.array([356])  # 12월 21일경
        dec = calculate_solar_declination(doy)
        dec_deg = dec[0] * 180 / np.pi
        assert -25 < dec_deg < -20  # 약 -23.5도

    def test_equinox(self):
        """춘분/추분에 적위 ~0°"""
        doy = np.array([80])  # 3월 21일경
        dec = calculate_solar_declination(doy)
        dec_deg = dec[0] * 180 / np.pi
        assert -5 < dec_deg < 5  # 약 0도


class TestHourAngle:
    """calculate_hour_angle 함수 테스트"""

    def test_noon(self):
        """정오에 시간각 ~0"""
        # 제주 경도에서 태양시 정오
        hour = np.array([12])
        ha = calculate_hour_angle(hour, JEJU_LONGITUDE)
        # 태양시 보정으로 정확히 0은 아님
        assert abs(ha[0]) < 1  # 1 라디안 이내

    def test_morning_negative(self):
        """오전에 시간각 음수"""
        hour = np.array([6])
        ha = calculate_hour_angle(hour, JEJU_LONGITUDE)
        assert ha[0] < 0

    def test_afternoon_positive(self):
        """오후에 시간각 양수"""
        hour = np.array([18])
        ha = calculate_hour_angle(hour, JEJU_LONGITUDE)
        assert ha[0] > 0


class TestSolarElevation:
    """calculate_solar_elevation 함수 테스트"""

    def test_noon_positive(self):
        """정오에 태양 고도 양수"""
        doy = np.array([172])  # 하지
        hour = np.array([12])
        elev = calculate_solar_elevation(doy, hour)
        assert elev[0] > 0

    def test_midnight_negative(self):
        """자정에 태양 고도 음수"""
        doy = np.array([172])
        hour = np.array([0])
        elev = calculate_solar_elevation(doy, hour)
        assert elev[0] < 0

    def test_summer_higher_than_winter(self):
        """여름 정오 고도 > 겨울 정오 고도"""
        summer_doy = np.array([172])  # 하지
        winter_doy = np.array([356])  # 동지
        hour = np.array([12])

        summer_elev = calculate_solar_elevation(summer_doy, hour)
        winter_elev = calculate_solar_elevation(winter_doy, hour)

        assert summer_elev[0] > winter_elev[0]

    def test_jeju_summer_solstice_noon(self):
        """제주 하지 정오 태양 고도 (~80°)"""
        doy = np.array([172])
        hour = np.array([12])
        elev = calculate_solar_elevation(doy, hour, JEJU_LATITUDE, JEJU_LONGITUDE)
        # 제주 위도(33.5°)에서 하지 정오: 90 - 33.5 + 23.5 ≈ 80°
        assert 75 < elev[0] < 85


class TestIsDaylight:
    """is_daylight 함수 테스트"""

    def test_noon_is_daylight(self):
        """정오는 주간"""
        dates = pd.DatetimeIndex(['2024-06-21 12:00:00'])
        daylight = is_daylight(dates)
        assert daylight[0] == 1

    def test_midnight_is_not_daylight(self):
        """자정은 야간"""
        dates = pd.DatetimeIndex(['2024-06-21 00:00:00'])
        daylight = is_daylight(dates)
        assert daylight[0] == 0

    def test_sunrise_boundary(self):
        """일출 시간 경계"""
        # 제주 하지 일출은 대략 5:30 AM
        dates = pd.DatetimeIndex(['2024-06-21 05:00:00', '2024-06-21 06:00:00'])
        daylight = is_daylight(dates)
        # 5시는 애매, 6시는 확실히 주간
        assert daylight[1] == 1


class TestDaylightHours:
    """calculate_daylight_hours 함수 테스트"""

    def test_summer_longer_than_winter(self):
        """여름 일조 시간 > 겨울 일조 시간"""
        summer = pd.DatetimeIndex(['2024-06-21'])
        winter = pd.DatetimeIndex(['2024-12-21'])

        summer_hours = calculate_daylight_hours(summer)
        winter_hours = calculate_daylight_hours(winter)

        assert summer_hours[0] > winter_hours[0]

    def test_jeju_summer_solstice(self):
        """제주 하지 일조 시간 (~14시간)"""
        dates = pd.DatetimeIndex(['2024-06-21'])
        hours = calculate_daylight_hours(dates, JEJU_LATITUDE)
        assert 13 < hours[0] < 15  # 약 14시간


# ============================================================
# 일사량 계산 테스트
# ============================================================

class TestTheoreticalIrradiance:
    """calculate_theoretical_irradiance 함수 테스트"""

    def test_noon_positive(self):
        """정오에 이론 일사량 양수"""
        dates = pd.DatetimeIndex(['2024-06-21 12:00:00'])
        irr = calculate_theoretical_irradiance(dates)
        assert irr[0] > 0

    def test_midnight_zero(self):
        """자정에 이론 일사량 0"""
        dates = pd.DatetimeIndex(['2024-06-21 00:00:00'])
        irr = calculate_theoretical_irradiance(dates)
        assert irr[0] == 0

    def test_summer_higher_than_winter(self):
        """여름 정오 > 겨울 정오"""
        summer = pd.DatetimeIndex(['2024-06-21 12:00:00'])
        winter = pd.DatetimeIndex(['2024-12-21 12:00:00'])

        summer_irr = calculate_theoretical_irradiance(summer)
        winter_irr = calculate_theoretical_irradiance(winter)

        assert summer_irr[0] > winter_irr[0]


class TestClearSkyIndex:
    """calculate_clear_sky_index 함수 테스트"""

    def test_clear_day(self):
        """맑은 날: 실제 ≈ 이론, CSI ≈ 1"""
        actual = np.array([2.5])
        theoretical = np.array([2.5])
        csi = calculate_clear_sky_index(actual, theoretical)
        assert abs(csi[0] - 1.0) < 0.1

    def test_cloudy_day(self):
        """흐린 날: 실제 < 이론, CSI < 1"""
        actual = np.array([1.0])
        theoretical = np.array([2.5])
        csi = calculate_clear_sky_index(actual, theoretical)
        assert csi[0] < 1.0

    def test_night_zero(self):
        """야간: 이론 0이면 CSI 0"""
        actual = np.array([0.0])
        theoretical = np.array([0.0])
        csi = calculate_clear_sky_index(actual, theoretical)
        assert csi[0] == 0

    def test_clipping(self):
        """CSI 상한 클리핑"""
        actual = np.array([5.0])  # 이론보다 높음 (측정 오차)
        theoretical = np.array([2.5])
        csi = calculate_clear_sky_index(actual, theoretical)
        assert csi[0] <= 1.5


class TestCloudAttenuation:
    """calculate_cloud_attenuation 함수 테스트"""

    def test_clear_sky(self):
        """전운량 0: 감쇠 없음 (1.0)"""
        cloud = np.array([0])
        att = calculate_cloud_attenuation(cloud)
        assert att[0] == 1.0

    def test_overcast(self):
        """전운량 10: 최대 감쇠 (0.0)"""
        cloud = np.array([10])
        att = calculate_cloud_attenuation(cloud)
        assert att[0] == 0.0

    def test_partial_cloud(self):
        """전운량 5: 중간 감쇠 (0.5)"""
        cloud = np.array([5])
        att = calculate_cloud_attenuation(cloud)
        assert att[0] == 0.5

    def test_range(self):
        """모든 값이 0~1 범위"""
        cloud = np.array([0, 3, 5, 7, 10])
        att = calculate_cloud_attenuation(cloud)
        assert all(0 <= a <= 1 for a in att)


# ============================================================
# 발전량 추정 테스트
# ============================================================

class TestEstimateSolarGeneration:
    """estimate_solar_generation 함수 테스트"""

    def test_zero_irradiance(self):
        """일사량 0이면 발전량 0"""
        irr = np.array([0.0])
        gen = estimate_solar_generation(irr)
        assert gen[0] == 0

    def test_positive_generation(self):
        """양수 일사량이면 양수 발전량"""
        irr = np.array([2.5])  # MJ/m²
        gen = estimate_solar_generation(irr)
        assert gen[0] > 0

    def test_capacity_scaling(self):
        """설비용량에 비례"""
        irr = np.array([2.5])
        gen_300 = estimate_solar_generation(irr, capacity_mw=300)
        gen_600 = estimate_solar_generation(irr, capacity_mw=600)
        assert abs(gen_600[0] / gen_300[0] - 2.0) < 0.1

    def test_stc_condition(self):
        """STC 조건 (3.6 MJ/m²)에서 ~80% 발전"""
        irr = np.array([3.6])  # STC 일사량
        gen = estimate_solar_generation(irr, capacity_mw=100, performance_ratio=0.80)
        # 100MW × 0.8 = 80MWh 예상
        assert 70 < gen[0] < 90


class TestEstimateBtmSolar:
    """estimate_btm_solar 함수 테스트"""

    def test_btm_positive(self):
        """계측 발전량에 비례한 BTM"""
        estimated = np.array([100.0])
        metered = np.array([50.0])
        btm = estimate_btm_solar(estimated, metered, btm_ratio=0.5)
        assert btm[0] == 25.0  # 50 × 0.5

    def test_btm_non_negative(self):
        """BTM은 음수 불가"""
        estimated = np.array([30.0])
        metered = np.array([50.0])
        btm = estimate_btm_solar(estimated, metered, btm_ratio=0.3)
        assert btm[0] >= 0


# ============================================================
# 통합 함수 테스트
# ============================================================

class TestAddSolarFeatures:
    """add_solar_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        dates = pd.date_range('2024-06-21', periods=24, freq='h')
        return pd.DataFrame({
            '일사': [0, 0, 0, 0, 0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                    3.2, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1, 0, 0, 0, 0],
            '전운량': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            '일조': [0, 0, 0, 0, 0, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0, 0, 0, 0, 0, 0],
            'solar_capacity_mw': 300.0,
            'solar_generation_mwh': [0, 0, 0, 0, 0, 1, 10, 30, 50, 70, 90, 100,
                                     110, 100, 90, 70, 50, 30, 10, 1, 0, 0, 0, 0]
        }, index=dates)

    def test_add_all_features(self, sample_df):
        """모든 특성 추가"""
        result = add_solar_features(sample_df)

        expected_cols = get_solar_feature_names()
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_solar_elevation_range(self, sample_df):
        """태양 고도 범위 확인"""
        result = add_solar_features(sample_df)
        # 하지 기준, 최대 고도 ~80°
        assert result['solar_elevation'].max() > 70
        # 자정 기준, 최소 고도 < 0
        assert result['solar_elevation'].min() < 0

    def test_daylight_pattern(self, sample_df):
        """주간 패턴 확인"""
        result = add_solar_features(sample_df)
        # 정오(12시)는 주간
        assert result.loc['2024-06-21 12:00:00', 'is_daylight'] == 1
        # 자정(0시)은 야간
        assert result.loc['2024-06-21 00:00:00', 'is_daylight'] == 0

    def test_cloud_attenuation_value(self, sample_df):
        """구름 감쇠 값 확인 (전운량 5 → 0.5)"""
        result = add_solar_features(sample_df)
        assert all(result['cloud_attenuation'] == 0.5)

    def test_inplace_modification(self, sample_df):
        """inplace 수정"""
        original_id = id(sample_df)
        result = add_solar_features(sample_df, inplace=True)
        assert id(result) == original_id
        assert 'solar_elevation' in sample_df.columns

    def test_copy_modification(self, sample_df):
        """복사본 수정"""
        original_cols = sample_df.columns.tolist()
        result = add_solar_features(sample_df, inplace=False)
        assert sample_df.columns.tolist() == original_cols
        assert 'solar_elevation' in result.columns

    def test_non_datetime_index_error(self):
        """datetime 인덱스 없을 때 에러"""
        df = pd.DataFrame({'일사': [1, 2, 3]})
        with pytest.raises(ValueError, match="datetime 인덱스"):
            add_solar_features(df)

    def test_selective_features(self, sample_df):
        """선택적 특성 추가"""
        result = add_solar_features(
            sample_df,
            include_theoretical=False,
            include_btm=False
        )
        assert 'theoretical_irradiance' not in result.columns
        assert 'btm_effect' not in result.columns
        assert 'solar_elevation' in result.columns


class TestGetSolarFeatureNames:
    """get_solar_feature_names 함수 테스트"""

    def test_default_features(self):
        """기본 특성 이름"""
        names = get_solar_feature_names()
        expected = [
            'solar_elevation', 'is_daylight', 'theoretical_irradiance',
            'clear_sky_index', 'cloud_attenuation', 'solar_estimated', 'btm_effect'
        ]
        assert names == expected

    def test_selective_features(self):
        """선택적 특성 이름"""
        names = get_solar_feature_names(
            include_theoretical=False,
            include_btm=False
        )
        assert 'theoretical_irradiance' not in names
        assert 'btm_effect' not in names
        assert 'solar_elevation' in names


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_row(self):
        """단일 행 DataFrame"""
        df = pd.DataFrame({
            '일사': [2.5],
            '전운량': [3],
        }, index=pd.DatetimeIndex(['2024-06-21 12:00:00']))
        result = add_solar_features(df)
        assert len(result) == 1
        assert 'solar_elevation' in result.columns

    def test_leap_year(self):
        """윤년 처리"""
        df = pd.DataFrame({
            '일사': [2.5],
            '전운량': [3],
        }, index=pd.DatetimeIndex(['2024-02-29 12:00:00']))
        result = add_solar_features(df)
        assert len(result) == 1

    def test_year_boundary(self):
        """연말-연초 경계"""
        dates = pd.date_range('2023-12-31 22:00:00', periods=4, freq='h')
        df = pd.DataFrame({
            '일사': [0, 0, 0, 0],
            '전운량': [5, 5, 5, 5],
        }, index=dates)
        result = add_solar_features(df)
        assert len(result) == 4

    def test_missing_optional_columns(self):
        """선택적 컬럼 없을 때"""
        df = pd.DataFrame({
            '일사': [2.5],
            '전운량': [3],
            # solar_capacity_mw, solar_generation_mwh 없음
        }, index=pd.DatetimeIndex(['2024-06-21 12:00:00']))

        result = add_solar_features(df)
        assert 'solar_elevation' in result.columns


# ============================================================
# 성능 및 데이터 타입 테스트
# ============================================================

class TestPerformanceAndTypes:
    """성능 및 데이터 타입 테스트"""

    def test_large_dataframe(self):
        """대용량 DataFrame 처리"""
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        df = pd.DataFrame({
            '일사': np.random.uniform(0, 3, len(dates)),
            '전운량': np.random.randint(0, 11, len(dates)),
        }, index=dates)

        result = add_solar_features(df)
        assert len(result) == 8760

    def test_output_types(self):
        """출력 데이터 타입 확인"""
        dates = pd.date_range('2024-06-21', periods=24, freq='h')
        df = pd.DataFrame({
            '일사': np.random.uniform(0, 3, len(dates)),
            '전운량': np.random.randint(0, 11, len(dates)),
        }, index=dates)

        result = add_solar_features(df)

        # 연속형은 float
        assert result['solar_elevation'].dtype in [np.float64, np.float32]
        assert result['theoretical_irradiance'].dtype in [np.float64, np.float32]

        # 플래그는 int8
        assert result['is_daylight'].dtype == np.int8

    def test_value_ranges(self):
        """값 범위 확인"""
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        df = pd.DataFrame({
            '일사': np.random.uniform(0, 3, len(dates)),
            '전운량': np.random.randint(0, 11, len(dates)),
        }, index=dates)

        result = add_solar_features(df)

        # 태양 고도: -90 ~ 90
        assert result['solar_elevation'].min() >= -90
        assert result['solar_elevation'].max() <= 90

        # 주간 플래그: 0 또는 1
        assert result['is_daylight'].isin([0, 1]).all()

        # 구름 감쇠: 0 ~ 1
        assert result['cloud_attenuation'].min() >= 0
        assert result['cloud_attenuation'].max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
