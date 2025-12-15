"""
FEAT-003: 시간 특성 생성 단위 테스트

테스트 범위:
1. 주기적 인코딩 (cyclical encoding)
2. 시간/요일/월 인코딩
3. 주말/공휴일 플래그
4. 통합 함수 (add_time_features)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.time_features import (
    cyclical_encode,
    encode_hour,
    encode_dayofweek,
    encode_month,
    encode_dayofyear,
    is_weekend,
    is_holiday,
    is_workday,
    add_time_features,
    get_time_feature_names,
    get_korean_holidays,
    get_all_korean_holidays,
)


# ============================================================
# 주기적 인코딩 테스트
# ============================================================

class TestCyclicalEncode:
    """cyclical_encode 함수 테스트"""

    def test_encode_basic(self):
        """기본 인코딩 테스트"""
        values = np.array([0, 1, 2, 3])
        period = 4
        sin_enc, cos_enc = cyclical_encode(values, period)

        # 0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)
        np.testing.assert_array_almost_equal(sin_enc, [0, 1, 0, -1])
        np.testing.assert_array_almost_equal(cos_enc, [1, 0, -1, 0])

    def test_encode_cyclic_property(self):
        """순환 속성 테스트: 시작점과 끝점이 유사해야 함"""
        values = np.array([0, 23])
        period = 24
        sin_enc, cos_enc = cyclical_encode(values, period)

        # 0과 23은 유사해야 함
        assert abs(sin_enc[0] - sin_enc[1]) < 0.3  # sin(0) ≈ sin(23/24 * 2π)
        assert abs(cos_enc[0] - cos_enc[1]) < 0.1  # cos(0) ≈ cos(23/24 * 2π)

    def test_encode_unit_circle(self):
        """단위원 속성: sin² + cos² = 1"""
        values = np.array([0, 3, 7, 11, 15, 19, 23])
        period = 24
        sin_enc, cos_enc = cyclical_encode(values, period)

        # sin² + cos² = 1
        sum_squares = sin_enc**2 + cos_enc**2
        np.testing.assert_array_almost_equal(sum_squares, np.ones(len(values)))


class TestEncodeHour:
    """encode_hour 함수 테스트"""

    def test_hour_encoding_values(self):
        """특정 시간의 인코딩 값 테스트"""
        hours = np.array([0, 6, 12, 18])
        sin_h, cos_h = encode_hour(hours)

        # 0시: sin=0, cos=1
        assert abs(sin_h[0]) < 1e-10
        assert abs(cos_h[0] - 1) < 1e-10

        # 6시: sin=1, cos≈0
        assert abs(sin_h[1] - 1) < 1e-10
        assert abs(cos_h[1]) < 1e-10

        # 12시: sin=0, cos=-1
        assert abs(sin_h[2]) < 1e-10
        assert abs(cos_h[2] + 1) < 1e-10

        # 18시: sin=-1, cos≈0
        assert abs(sin_h[3] + 1) < 1e-10
        assert abs(cos_h[3]) < 1e-10

    def test_hour_full_day(self):
        """24시간 전체 인코딩"""
        hours = np.arange(24)
        sin_h, cos_h = encode_hour(hours)

        assert len(sin_h) == 24
        assert len(cos_h) == 24

        # 범위 확인
        assert sin_h.min() >= -1
        assert sin_h.max() <= 1
        assert cos_h.min() >= -1
        assert cos_h.max() <= 1


class TestEncodeDayOfWeek:
    """encode_dayofweek 함수 테스트"""

    def test_dayofweek_encoding(self):
        """요일 인코딩 테스트"""
        days = np.array([0, 1, 2, 3, 4, 5, 6])  # 월~일
        sin_d, cos_d = encode_dayofweek(days)

        # 7개 값
        assert len(sin_d) == 7
        assert len(cos_d) == 7

        # 단위원 속성
        sum_squares = sin_d**2 + cos_d**2
        np.testing.assert_array_almost_equal(sum_squares, np.ones(7))

    def test_monday_encoding(self):
        """월요일 인코딩 (0)"""
        sin_d, cos_d = encode_dayofweek(np.array([0]))

        # 0: sin=0, cos=1
        assert abs(sin_d[0]) < 1e-10
        assert abs(cos_d[0] - 1) < 1e-10


class TestEncodeMonth:
    """encode_month 함수 테스트"""

    def test_month_encoding(self):
        """월 인코딩 테스트 (1-12)"""
        months = np.array([1, 4, 7, 10])  # 1월, 4월, 7월, 10월
        sin_m, cos_m = encode_month(months)

        # 1월 (0번째): sin≈0, cos≈1
        assert abs(sin_m[0]) < 1e-10
        assert abs(cos_m[0] - 1) < 1e-10

    def test_month_december_january_similarity(self):
        """12월과 1월의 유사성"""
        months = np.array([1, 12])
        sin_m, cos_m = encode_month(months)

        # 12월(11번째)과 1월(0번째)은 유사해야 함
        assert abs(sin_m[0] - sin_m[1]) < 0.6  # 거리가 가까움
        assert abs(cos_m[0] - cos_m[1]) < 0.3

    def test_month_full_year(self):
        """12개월 전체 인코딩"""
        months = np.arange(1, 13)
        sin_m, cos_m = encode_month(months)

        assert len(sin_m) == 12
        assert len(cos_m) == 12


class TestEncodeDayOfYear:
    """encode_dayofyear 함수 테스트"""

    def test_dayofyear_encoding(self):
        """연중 일수 인코딩"""
        days = np.array([1, 91, 182, 274])  # 1/1, 4/1, 7/1, 10/1 근처
        sin_d, cos_d = encode_dayofyear(days)

        assert len(sin_d) == 4
        assert len(cos_d) == 4

    def test_leap_year(self):
        """윤년 처리"""
        days = np.array([366])
        is_leap = np.array([True])
        sin_d, cos_d = encode_dayofyear(days, is_leap)

        # 366일째는 거의 0번째와 같아야 함
        sin_0, cos_0 = encode_dayofyear(np.array([1]), np.array([True]))
        assert abs(sin_d[0] - sin_0[0]) < 0.02


# ============================================================
# 이진 플래그 테스트
# ============================================================

class TestIsWeekend:
    """is_weekend 함수 테스트"""

    def test_weekend_detection(self):
        """주말 감지"""
        days = np.array([0, 1, 2, 3, 4, 5, 6])  # 월~일
        weekend = is_weekend(days)

        expected = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(weekend, expected)

    def test_all_weekdays(self):
        """평일만"""
        days = np.array([0, 1, 2, 3, 4])
        weekend = is_weekend(days)
        assert weekend.sum() == 0

    def test_all_weekend(self):
        """주말만"""
        days = np.array([5, 6])
        weekend = is_weekend(days)
        assert weekend.sum() == 2


class TestIsHoliday:
    """is_holiday 함수 테스트"""

    def test_holiday_detection(self):
        """공휴일 감지"""
        # 2024년 1월 1일 (신정), 1월 2일 (평일)
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
        holidays = is_holiday(dates)

        assert holidays[0] == 1  # 신정
        assert holidays[1] == 0  # 평일

    def test_chuseok_detection(self):
        """추석 감지 (2024년)"""
        # 2024년 추석: 9/16, 9/17, 9/18
        dates = pd.DatetimeIndex(['2024-09-16', '2024-09-17', '2024-09-18', '2024-09-19'])
        holidays = is_holiday(dates)

        assert holidays[0] == 1
        assert holidays[1] == 1
        assert holidays[2] == 1
        assert holidays[3] == 0

    def test_custom_holidays(self):
        """사용자 정의 공휴일"""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
        custom_holidays = {date(2024, 1, 2)}  # 1월 2일만 공휴일
        holidays = is_holiday(dates, custom_holidays)

        assert holidays[0] == 0  # 사용자 정의에 없음
        assert holidays[1] == 1  # 사용자 정의에 있음


class TestIsWorkday:
    """is_workday 함수 테스트"""

    def test_workday_detection(self):
        """근무일 감지"""
        # 월요일(평일), 토요일(주말)
        days = np.array([0, 5])
        workday = is_workday(days)

        assert workday[0] == 1  # 월요일
        assert workday[1] == 0  # 토요일

    def test_workday_with_holiday(self):
        """공휴일 고려한 근무일"""
        # 2024-01-01 (월요일, 신정)
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
        days = dates.dayofweek.values
        workday = is_workday(days, dates)

        assert workday[0] == 0  # 신정 (공휴일)
        assert workday[1] == 1  # 평일


# ============================================================
# 한국 공휴일 테스트
# ============================================================

class TestKoreanHolidays:
    """한국 공휴일 함수 테스트"""

    def test_fixed_holidays_2024(self):
        """2024년 고정 공휴일"""
        holidays = get_korean_holidays(2024)

        assert date(2024, 1, 1) in holidays   # 신정
        assert date(2024, 3, 1) in holidays   # 삼일절
        assert date(2024, 5, 5) in holidays   # 어린이날
        assert date(2024, 6, 6) in holidays   # 현충일
        assert date(2024, 8, 15) in holidays  # 광복절
        assert date(2024, 10, 3) in holidays  # 개천절
        assert date(2024, 10, 9) in holidays  # 한글날
        assert date(2024, 12, 25) in holidays # 성탄절

    def test_lunar_holidays_2024(self):
        """2024년 음력 공휴일"""
        holidays = get_korean_holidays(2024)

        # 설날: 2/9, 2/10, 2/11
        assert date(2024, 2, 9) in holidays
        assert date(2024, 2, 10) in holidays
        assert date(2024, 2, 11) in holidays

        # 추석: 9/16, 9/17, 9/18
        assert date(2024, 9, 16) in holidays
        assert date(2024, 9, 17) in holidays
        assert date(2024, 9, 18) in holidays

    def test_all_holidays_range(self):
        """여러 연도 공휴일"""
        holidays = get_all_korean_holidays(2013, 2025)

        # 13년간의 공휴일이 포함되어야 함
        assert len(holidays) > 100  # 대략 연간 15개 이상

        # 다양한 연도의 신정 포함
        assert date(2013, 1, 1) in holidays
        assert date(2020, 1, 1) in holidays
        assert date(2025, 1, 1) in holidays


# ============================================================
# 통합 함수 테스트
# ============================================================

class TestAddTimeFeatures:
    """add_time_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        dates = pd.date_range('2024-01-01', periods=48, freq='H')
        return pd.DataFrame({'value': np.random.randn(48)}, index=dates)

    def test_add_all_features(self, sample_df):
        """모든 특성 추가"""
        result = add_time_features(sample_df)

        # 기본 특성 확인
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'dayofweek_sin' in result.columns
        assert 'dayofweek_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_holiday' in result.columns

    def test_inplace_modification(self, sample_df):
        """inplace 수정"""
        original_id = id(sample_df)
        result = add_time_features(sample_df, inplace=True)

        assert id(result) == original_id
        assert 'hour_sin' in sample_df.columns

    def test_copy_modification(self, sample_df):
        """복사본 수정"""
        original_cols = sample_df.columns.tolist()
        result = add_time_features(sample_df, inplace=False)

        assert sample_df.columns.tolist() == original_cols
        assert 'hour_sin' in result.columns

    def test_selective_features(self, sample_df):
        """선택적 특성 추가"""
        result = add_time_features(
            sample_df,
            include_hour=True,
            include_dayofweek=False,
            include_month=False,
            include_weekend=True,
            include_holiday=False
        )

        assert 'hour_sin' in result.columns
        assert 'dayofweek_sin' not in result.columns
        assert 'month_sin' not in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_holiday' not in result.columns

    def test_include_dayofyear(self, sample_df):
        """연중 일수 인코딩 포함"""
        result = add_time_features(sample_df, include_dayofyear=True)

        assert 'dayofyear_sin' in result.columns
        assert 'dayofyear_cos' in result.columns

    def test_include_workday(self, sample_df):
        """근무일 플래그 포함"""
        result = add_time_features(sample_df, include_workday=True)

        assert 'is_workday' in result.columns

    def test_non_datetime_index_error(self):
        """datetime 인덱스 없을 때 에러"""
        df = pd.DataFrame({'value': [1, 2, 3]})

        with pytest.raises(ValueError, match="datetime 인덱스"):
            add_time_features(df)

    def test_new_year_holiday(self, sample_df):
        """신정 공휴일 감지"""
        result = add_time_features(sample_df)

        # 2024-01-01의 is_holiday는 1이어야 함
        jan1_rows = result.loc['2024-01-01', 'is_holiday']
        assert (jan1_rows == 1).all()


class TestGetTimeFeatureNames:
    """get_time_feature_names 함수 테스트"""

    def test_default_features(self):
        """기본 특성 이름"""
        names = get_time_feature_names()

        expected = [
            'hour_sin', 'hour_cos',
            'dayofweek_sin', 'dayofweek_cos',
            'month_sin', 'month_cos',
            'is_weekend', 'is_holiday'
        ]
        assert names == expected

    def test_selective_features(self):
        """선택적 특성 이름"""
        names = get_time_feature_names(
            include_hour=True,
            include_dayofweek=False,
            include_month=False,
            include_weekend=True,
            include_holiday=False
        )

        assert names == ['hour_sin', 'hour_cos', 'is_weekend']

    def test_all_features(self):
        """모든 특성 이름"""
        names = get_time_feature_names(
            include_hour=True,
            include_dayofweek=True,
            include_month=True,
            include_dayofyear=True,
            include_weekend=True,
            include_holiday=True,
            include_workday=True
        )

        assert len(names) == 11  # 2+2+2+2+1+1+1


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_row(self):
        """단일 행 DataFrame"""
        df = pd.DataFrame(
            {'value': [1]},
            index=pd.DatetimeIndex(['2024-01-01 12:00:00'])
        )
        result = add_time_features(df)

        assert len(result) == 1
        assert 'hour_sin' in result.columns

    def test_leap_year_february(self):
        """윤년 2월 29일"""
        df = pd.DataFrame(
            {'value': [1]},
            index=pd.DatetimeIndex(['2024-02-29 12:00:00'])
        )
        result = add_time_features(df, include_dayofyear=True)

        # 60번째 날 (윤년)
        assert result.index.dayofyear[0] == 60

    def test_year_boundary(self):
        """연말-연초 경계"""
        dates = pd.date_range('2023-12-31 22:00:00', periods=4, freq='H')
        df = pd.DataFrame({'value': np.random.randn(4)}, index=dates)
        result = add_time_features(df)

        # 2023년 12월 31일과 2024년 1월 1일 모두 처리
        assert len(result) == 4


# ============================================================
# 성능 및 데이터 타입 테스트
# ============================================================

class TestPerformanceAndTypes:
    """성능 및 데이터 타입 테스트"""

    def test_large_dataframe(self):
        """대용량 DataFrame 처리"""
        # 1년치 시간별 데이터
        dates = pd.date_range('2024-01-01', periods=8760, freq='H')
        df = pd.DataFrame({'value': np.random.randn(8760)}, index=dates)

        result = add_time_features(df)

        assert len(result) == 8760
        assert 'hour_sin' in result.columns

    def test_output_types(self):
        """출력 데이터 타입 확인"""
        dates = pd.date_range('2024-01-01', periods=24, freq='H')
        df = pd.DataFrame({'value': np.random.randn(24)}, index=dates)
        result = add_time_features(df)

        # sin/cos는 float
        assert result['hour_sin'].dtype in [np.float64, np.float32]
        assert result['hour_cos'].dtype in [np.float64, np.float32]

        # 플래그는 int8
        assert result['is_weekend'].dtype == np.int8
        assert result['is_holiday'].dtype == np.int8

    def test_value_ranges(self):
        """값 범위 확인"""
        dates = pd.date_range('2024-01-01', periods=8760, freq='H')
        df = pd.DataFrame({'value': np.random.randn(8760)}, index=dates)
        result = add_time_features(df)

        # sin/cos 범위: [-1, 1]
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1

        # 플래그 범위: [0, 1]
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['is_holiday'].isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
