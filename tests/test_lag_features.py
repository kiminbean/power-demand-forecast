"""
FEAT-005: 지연 변수 및 이동평균 단위 테스트

테스트 범위:
1. 기본 지연 변수 (lag features)
2. 이동평균 (moving averages)
3. 롤링 통계 (rolling statistics)
4. 차분 및 변화율 (difference, pct change)
5. 통합 함수 (add_lag_features)
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.lag_features import (
    # 기본 지연 변수
    create_lag_feature,
    create_lag_features,
    # 이동평균
    create_moving_average,
    create_moving_averages,
    create_exponential_moving_average,
    # 롤링 통계
    create_rolling_std,
    create_rolling_min_max,
    # 차분
    create_difference,
    create_pct_change,
    # 특화 함수
    create_demand_lag_features,
    create_weather_lag_features,
    # 통합 함수
    add_lag_features,
    get_lag_feature_names,
)


# ============================================================
# 기본 지연 변수 테스트
# ============================================================

class TestCreateLagFeature:
    """create_lag_feature 함수 테스트"""

    def test_lag_1(self):
        """단일 지연 (t-1)"""
        s = pd.Series([1, 2, 3, 4, 5])
        lagged = create_lag_feature(s, 1)

        assert pd.isna(lagged.iloc[0])
        assert lagged.iloc[1] == 1
        assert lagged.iloc[4] == 4

    def test_lag_2(self):
        """이중 지연 (t-2)"""
        s = pd.Series([1, 2, 3, 4, 5])
        lagged = create_lag_feature(s, 2)

        assert pd.isna(lagged.iloc[0])
        assert pd.isna(lagged.iloc[1])
        assert lagged.iloc[2] == 1
        assert lagged.iloc[4] == 3

    def test_lag_naming(self):
        """컬럼명 생성"""
        s = pd.Series([1, 2, 3], name='value')
        lagged = create_lag_feature(s, 1)
        assert lagged.name == 'value_lag_1'

    def test_lag_with_prefix(self):
        """접두사 지정"""
        s = pd.Series([1, 2, 3], name='value')
        lagged = create_lag_feature(s, 1, prefix='demand')
        assert lagged.name == 'demand_lag_1'


class TestCreateLagFeatures:
    """create_lag_features 함수 테스트"""

    def test_multiple_lags(self):
        """여러 지연 변수"""
        s = pd.Series([1, 2, 3, 4, 5], name='value')
        df = create_lag_features(s, [1, 2, 3])

        assert len(df.columns) == 3
        assert 'value_lag_1' in df.columns
        assert 'value_lag_2' in df.columns
        assert 'value_lag_3' in df.columns

    def test_lag_values(self):
        """지연 값 검증"""
        s = pd.Series([10, 20, 30, 40, 50], name='value')
        df = create_lag_features(s, [1, 2])

        assert df.iloc[2, 0] == 20  # lag_1 at index 2
        assert df.iloc[2, 1] == 10  # lag_2 at index 2


# ============================================================
# 이동평균 테스트
# ============================================================

class TestCreateMovingAverage:
    """create_moving_average 함수 테스트"""

    def test_ma_basic(self):
        """기본 이동평균"""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma = create_moving_average(s, window=3)

        # shift(1) 적용으로 인덱스 3에서 (1+2+3)/3 = 2
        assert ma.iloc[3] == 2.0

    def test_ma_excludes_current(self):
        """현재 시점 제외 확인 (데이터 누수 방지)"""
        s = pd.Series([1, 2, 3, 4, 5])
        ma = create_moving_average(s, window=2)

        # 인덱스 3: shift(1) 후 (2+3)/2 = 2.5
        # 현재 값 4가 포함되지 않음
        assert ma.iloc[3] == 2.5

    def test_ma_naming(self):
        """컬럼명 생성"""
        s = pd.Series([1, 2, 3], name='temp')
        ma = create_moving_average(s, window=3)
        assert ma.name == 'temp_ma_3h'


class TestCreateMovingAverages:
    """create_moving_averages 함수 테스트"""

    def test_multiple_windows(self):
        """여러 윈도우 이동평균"""
        s = pd.Series(range(100), name='value')
        df = create_moving_averages(s, [6, 12, 24])

        assert len(df.columns) == 3
        assert 'value_ma_6h' in df.columns
        assert 'value_ma_12h' in df.columns
        assert 'value_ma_24h' in df.columns


class TestExponentialMovingAverage:
    """create_exponential_moving_average 함수 테스트"""

    def test_ema_basic(self):
        """기본 EMA"""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema = create_exponential_moving_average(s, span=3)

        # EMA는 최근 값에 더 큰 가중치
        assert len(ema) == 10
        assert not pd.isna(ema.iloc[3])

    def test_ema_naming(self):
        """컬럼명 생성"""
        s = pd.Series([1, 2, 3], name='value')
        ema = create_exponential_moving_average(s, span=3)
        assert ema.name == 'value_ema_3h'


# ============================================================
# 롤링 통계 테스트
# ============================================================

class TestCreateRollingStd:
    """create_rolling_std 함수 테스트"""

    def test_std_basic(self):
        """기본 롤링 표준편차"""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        std = create_rolling_std(s, window=3)

        # 일정한 증가 패턴이므로 std는 일정
        assert not pd.isna(std.iloc[4])
        assert std.iloc[4] > 0

    def test_std_constant_series(self):
        """상수 시리즈의 표준편차 = 0"""
        s = pd.Series([5, 5, 5, 5, 5, 5, 5, 5])
        std = create_rolling_std(s, window=3)

        # 상수이므로 std = 0
        assert std.iloc[5] == 0

    def test_std_naming(self):
        """컬럼명 생성"""
        s = pd.Series([1, 2, 3], name='value')
        std = create_rolling_std(s, window=3)
        assert std.name == 'value_std_3h'


class TestCreateRollingMinMax:
    """create_rolling_min_max 함수 테스트"""

    def test_minmax_basic(self):
        """기본 롤링 최소/최대"""
        s = pd.Series([1, 5, 2, 8, 3, 9, 4, 10])
        df = create_rolling_min_max(s, window=3, prefix='value')

        assert 'value_min_3h' in df.columns
        assert 'value_max_3h' in df.columns
        assert 'value_range_3h' in df.columns

    def test_range_calculation(self):
        """범위 계산 검증"""
        s = pd.Series([1, 10, 5, 2, 8], name='value')
        df = create_rolling_min_max(s, window=3, prefix='value')

        # 인덱스 4: shift(1) 후 [10, 5, 2] → min=2, max=10, range=8
        assert df['value_min_3h'].iloc[4] == 2
        assert df['value_max_3h'].iloc[4] == 10
        assert df['value_range_3h'].iloc[4] == 8


# ============================================================
# 차분 및 변화율 테스트
# ============================================================

class TestCreateDifference:
    """create_difference 함수 테스트"""

    def test_diff_basic(self):
        """기본 차분"""
        s = pd.Series([10, 12, 15, 13, 18])
        diff = create_difference(s, periods=1)

        assert pd.isna(diff.iloc[0])
        assert diff.iloc[1] == 2   # 12 - 10
        assert diff.iloc[2] == 3   # 15 - 12
        assert diff.iloc[3] == -2  # 13 - 15

    def test_diff_periods(self):
        """다중 기간 차분"""
        s = pd.Series([10, 12, 15, 13, 18])
        diff = create_difference(s, periods=2)

        assert pd.isna(diff.iloc[0])
        assert pd.isna(diff.iloc[1])
        assert diff.iloc[2] == 5  # 15 - 10


class TestCreatePctChange:
    """create_pct_change 함수 테스트"""

    def test_pct_basic(self):
        """기본 변화율"""
        s = pd.Series([100, 110, 99, 110])
        pct = create_pct_change(s, periods=1)

        assert pd.isna(pct.iloc[0])
        assert abs(pct.iloc[1] - 0.10) < 0.001  # 10% 증가
        assert abs(pct.iloc[2] - (-0.10)) < 0.001  # 10% 감소


# ============================================================
# 전력 수요 지연 변수 테스트
# ============================================================

class TestCreateDemandLagFeatures:
    """create_demand_lag_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성 (1주일)"""
        dates = pd.date_range('2024-01-01', periods=24*7, freq='h')
        return pd.DataFrame({
            'power_demand': np.random.randn(len(dates)) * 50 + 500
        }, index=dates)

    def test_default_lags(self, sample_df):
        """기본 지연 변수"""
        result = create_demand_lag_features(sample_df)

        assert 'demand_lag_1' in result.columns
        assert 'demand_lag_24' in result.columns
        assert 'demand_lag_48' in result.columns
        assert 'demand_lag_168' in result.columns

    def test_with_ma(self, sample_df):
        """이동평균 포함"""
        result = create_demand_lag_features(sample_df, include_ma=True)

        assert 'demand_ma_6h' in result.columns
        assert 'demand_ma_24h' in result.columns

    def test_with_std(self, sample_df):
        """표준편차 포함"""
        result = create_demand_lag_features(sample_df, include_std=True)

        assert 'demand_std_24h' in result.columns

    def test_with_diff(self, sample_df):
        """차분 포함"""
        result = create_demand_lag_features(sample_df, include_diff=True)

        assert 'demand_diff_1h' in result.columns
        assert 'demand_diff_24h' in result.columns

    def test_missing_column_error(self):
        """존재하지 않는 컬럼 에러"""
        df = pd.DataFrame({'value': [1, 2, 3]})

        with pytest.raises(ValueError, match="컬럼을 찾을 수 없습니다"):
            create_demand_lag_features(df, demand_col='power_demand')


# ============================================================
# 기상 지연 변수 테스트
# ============================================================

class TestCreateWeatherLagFeatures:
    """create_weather_lag_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        dates = pd.date_range('2024-01-01', periods=48, freq='h')
        return pd.DataFrame({
            '기온': np.random.randn(len(dates)) * 5 + 10,
            '일사': np.maximum(0, np.random.randn(len(dates)) * 0.5 + 1),
            '습도': np.random.randn(len(dates)) * 10 + 60
        }, index=dates)

    def test_temp_features(self, sample_df):
        """기온 특성"""
        result = create_weather_lag_features(sample_df, include_temp=True)

        assert 'temp_ma_6h' in result.columns
        assert 'temp_lag_1' in result.columns
        assert 'temp_range_24h' in result.columns

    def test_irradiance_features(self, sample_df):
        """일사량 특성"""
        result = create_weather_lag_features(sample_df, include_irradiance=True)

        assert 'irradiance_ma_6h' in result.columns
        assert 'irradiance_sum_24h' in result.columns

    def test_humidity_features(self, sample_df):
        """습도 특성"""
        result = create_weather_lag_features(sample_df, include_humidity=True)

        assert 'humidity_ma_6h' in result.columns


# ============================================================
# 통합 함수 테스트
# ============================================================

class TestAddLagFeatures:
    """add_lag_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        dates = pd.date_range('2024-01-01', periods=24*7, freq='h')
        return pd.DataFrame({
            'power_demand': np.random.randn(len(dates)) * 50 + 500,
            '기온': np.random.randn(len(dates)) * 5 + 10,
            '일사': np.maximum(0, np.random.randn(len(dates)) * 0.5 + 1),
            '습도': np.random.randn(len(dates)) * 10 + 60
        }, index=dates)

    def test_add_all_features(self, sample_df):
        """모든 특성 추가"""
        result = add_lag_features(sample_df)

        # 원본 컬럼 + 새 컬럼
        assert len(result.columns) > 4
        assert 'demand_lag_1' in result.columns
        assert 'temp_ma_6h' in result.columns

    def test_inplace_modification(self, sample_df):
        """inplace 수정"""
        original_id = id(sample_df)
        result = add_lag_features(sample_df, inplace=True)

        assert id(result) == original_id
        assert 'demand_lag_1' in sample_df.columns

    def test_copy_modification(self, sample_df):
        """복사본 수정"""
        original_cols = sample_df.columns.tolist()
        result = add_lag_features(sample_df, inplace=False)

        assert sample_df.columns.tolist() == original_cols
        assert 'demand_lag_1' in result.columns

    def test_non_datetime_index_error(self):
        """datetime 인덱스 없을 때 에러"""
        df = pd.DataFrame({'power_demand': [1, 2, 3]})

        with pytest.raises(ValueError, match="datetime 인덱스"):
            add_lag_features(df)

    def test_selective_features(self, sample_df):
        """선택적 특성 추가"""
        result = add_lag_features(
            sample_df,
            include_demand_features=True,
            include_weather_features=False
        )

        assert 'demand_lag_1' in result.columns
        assert 'temp_ma_6h' not in result.columns

    def test_fill_na_ffill(self, sample_df):
        """NaN forward fill"""
        result = add_lag_features(sample_df, fill_na_method='ffill')

        # 첫 번째 값 이후에는 NaN이 없어야 함 (ffill 후)
        # (하지만 첫 몇 행은 여전히 NaN일 수 있음)
        assert result['demand_lag_1'].iloc[10:].isna().sum() == 0


class TestGetLagFeatureNames:
    """get_lag_feature_names 함수 테스트"""

    def test_default_names(self):
        """기본 특성 이름"""
        names = get_lag_feature_names()

        assert 'demand_lag_1' in names
        assert 'demand_lag_24' in names
        assert 'demand_ma_6h' in names
        assert 'temp_ma_6h' in names

    def test_selective_names(self):
        """선택적 특성 이름"""
        names = get_lag_feature_names(
            include_demand=True,
            include_weather=False
        )

        assert 'demand_lag_1' in names
        assert 'temp_ma_6h' not in names


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_short_series(self):
        """짧은 시리즈"""
        s = pd.Series([1, 2, 3])
        lagged = create_lag_feature(s, 1)

        assert len(lagged) == 3
        assert lagged.iloc[0] != lagged.iloc[0]  # NaN

    def test_all_nan_lag(self):
        """모두 NaN인 경우"""
        s = pd.Series([1, 2])
        lagged = create_lag_feature(s, 5)  # 시리즈보다 큰 lag

        assert lagged.isna().all()

    def test_negative_lag(self):
        """음수 지연 (미래 값)"""
        s = pd.Series([1, 2, 3, 4, 5])
        lagged = create_lag_feature(s, -1)

        assert lagged.iloc[0] == 2  # 미래 값
        assert pd.isna(lagged.iloc[4])


# ============================================================
# 성능 및 데이터 타입 테스트
# ============================================================

class TestPerformanceAndTypes:
    """성능 및 데이터 타입 테스트"""

    def test_large_dataframe(self):
        """대용량 DataFrame 처리"""
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        df = pd.DataFrame({
            'power_demand': np.random.randn(len(dates)) * 50 + 500,
            '기온': np.random.randn(len(dates)) * 5 + 10,
            '일사': np.maximum(0, np.random.randn(len(dates))),
            '습도': np.random.randn(len(dates)) * 10 + 60
        }, index=dates)

        result = add_lag_features(df)
        assert len(result) == 8760

    def test_output_types(self):
        """출력 데이터 타입 확인"""
        dates = pd.date_range('2024-01-01', periods=48, freq='h')
        df = pd.DataFrame({
            'power_demand': np.random.randn(len(dates)) * 50 + 500,
            '기온': np.random.randn(len(dates)) * 5 + 10,
            '일사': np.maximum(0, np.random.randn(len(dates))),
            '습도': np.random.randn(len(dates)) * 10 + 60
        }, index=dates)

        result = add_lag_features(df)

        # 모든 새 컬럼은 숫자형
        for col in result.columns:
            if col not in df.columns:
                assert result[col].dtype in [np.float64, np.float32]


# ============================================================
# 데이터 누수 방지 테스트
# ============================================================

class TestDataLeakagePrevention:
    """데이터 누수 방지 테스트"""

    def test_lag_no_leakage(self):
        """지연 변수 데이터 누수 방지"""
        s = pd.Series([1, 2, 3, 4, 5])
        lagged = create_lag_feature(s, 1)

        # 인덱스 4의 lag_1은 인덱스 3의 값이어야 함
        assert lagged.iloc[4] == s.iloc[3]

    def test_ma_no_leakage(self):
        """이동평균 데이터 누수 방지"""
        s = pd.Series([1, 2, 3, 4, 5])
        ma = create_moving_average(s, window=2)

        # 인덱스 4의 MA는 인덱스 2, 3의 평균이어야 함 (현재 값 제외)
        # shift(1) 후 [3, 4]의 평균 = 3.5
        assert ma.iloc[4] == 3.5

    def test_std_no_leakage(self):
        """롤링 표준편차 데이터 누수 방지"""
        s = pd.Series([1, 2, 3, 4, 5])
        std = create_rolling_std(s, window=2, min_periods=2)

        # 인덱스 4의 std는 인덱스 2, 3의 std이어야 함 (현재 값 제외)
        expected_std = np.std([3, 4], ddof=1)  # pandas default ddof=1
        assert abs(std.iloc[4] - expected_std) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
