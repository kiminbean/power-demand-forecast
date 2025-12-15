"""
External Features 모듈 단위 테스트

테스트 범위:
1. 인구 데이터 로드 및 전처리
2. 전기차 데이터 로드 및 전처리
3. 인구 피처 추가
4. 전기차 피처 추가
5. 통합 함수 테스트
6. 피처 이름 목록
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.external_features import (
    load_population_data,
    load_ev_data,
    add_population_features,
    add_ev_features,
    add_external_features,
    get_external_feature_names,
    POPULATION_FEATURES,
    POPULATION_DERIVED,
    EV_FEATURES,
    EV_DERIVED,
    EXTERNAL_FEATURES_ALL,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_main_df():
    """메인 데이터프레임 생성"""
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'power_demand': np.random.uniform(400, 600, 72),
        'temp_mean': np.random.uniform(10, 25, 72)
    }, index=dates)


@pytest.fixture
def sample_population_df():
    """인구 데이터프레임 생성 (시간별)"""
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'base_population': np.full(72, 680000),
        'tourist_stock': np.random.uniform(50000, 100000, 72),
        'net_flow': np.random.uniform(-5000, 5000, 72),
        'estimated_population': np.random.uniform(700000, 800000, 72)
    }, index=dates)


@pytest.fixture
def sample_ev_df():
    """전기차 데이터프레임 생성 (시간별)"""
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'ev_cumulative': np.linspace(50000, 50100, 72)
    }, index=dates)


@pytest.fixture
def temp_population_csv():
    """임시 인구 데이터 CSV 파일 생성"""
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'base_population': [680000, 680000, 680000],
        'total_arrival': [15000, 16000, 14000],
        'total_departure': [14000, 15000, 13000],
        'net_flow': [1000, 1000, 1000],
        'tourist_stock': [50000, 51000, 52000],
        'estimated_population': [730000, 731000, 732000]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_ev_csv():
    """임시 전기차 데이터 CSV 파일 생성"""
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'Cumulative EV Count': [50000, 50050, 50100],
        'ev_daily_new': [50, 50, 50]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# ============================================================
# load_population_data 테스트
# ============================================================

class TestLoadPopulationData:
    """load_population_data 함수 테스트"""

    def test_load_basic(self, temp_population_csv):
        """기본 로드 테스트"""
        df = load_population_data(temp_population_csv, resample_to_hourly=False)

        assert len(df) == 3
        assert 'base_population' in df.columns
        assert 'estimated_population' in df.columns

    def test_resample_to_hourly(self, temp_population_csv):
        """시간별 리샘플링 테스트"""
        df = load_population_data(temp_population_csv, resample_to_hourly=True)

        # 3일 = 최대 48시간 (첫째날 00:00 ~ 셋째날 00:00)
        # ffill이므로 마지막 시간까지 채워짐
        assert len(df) >= 48
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_forward_fill(self, temp_population_csv):
        """Forward fill 확인"""
        df = load_population_data(temp_population_csv, resample_to_hourly=True)

        # 첫째 날의 값이 시간별로 유지되어야 함
        first_day_values = df.loc['2024-01-01', 'base_population']
        assert (first_day_values == 680000).all()

    def test_columns_selected(self, temp_population_csv):
        """필요한 컬럼만 선택"""
        df = load_population_data(temp_population_csv, resample_to_hourly=False)

        expected_cols = ['base_population', 'total_arrival', 'total_departure',
                        'net_flow', 'tourist_stock', 'estimated_population']
        for col in expected_cols:
            assert col in df.columns

    def test_duplicate_dates_handled(self):
        """중복 날짜 처리 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
                'base_population': [680000, 680001, 680002],
                'estimated_population': [730000, 730001, 730002]
            })
            df.to_csv(f.name, index=False)

            result = load_population_data(f.name, resample_to_hourly=False)

            # 마지막 값이 유지되어야 함
            assert len(result) == 2
            assert result.loc['2024-01-01', 'base_population'] == 680001

        os.unlink(f.name)


# ============================================================
# load_ev_data 테스트
# ============================================================

class TestLoadEvData:
    """load_ev_data 함수 테스트"""

    def test_load_basic(self, temp_ev_csv):
        """기본 로드 테스트"""
        df = load_ev_data(temp_ev_csv, resample_to_hourly=False)

        assert len(df) == 3
        assert 'ev_cumulative' in df.columns

    def test_resample_to_hourly(self, temp_ev_csv):
        """시간별 리샘플링 테스트"""
        df = load_ev_data(temp_ev_csv, resample_to_hourly=True)

        assert len(df) >= 48
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_column_rename(self, temp_ev_csv):
        """컬럼명 표준화 테스트"""
        df = load_ev_data(temp_ev_csv, resample_to_hourly=False)

        # 'Cumulative EV Count' -> 'ev_cumulative'
        assert 'ev_cumulative' in df.columns

    def test_forward_fill(self, temp_ev_csv):
        """Forward fill 확인"""
        df = load_ev_data(temp_ev_csv, resample_to_hourly=True)

        # 첫째 날의 값이 시간별로 유지되어야 함
        first_day_values = df.loc['2024-01-01', 'ev_cumulative']
        assert (first_day_values == 50000).all()


# ============================================================
# add_population_features 테스트
# ============================================================

class TestAddPopulationFeatures:
    """add_population_features 함수 테스트"""

    def test_basic_features_added(self, sample_main_df, sample_population_df):
        """기본 피처 추가"""
        result = add_population_features(
            sample_main_df, sample_population_df, include_derived=False
        )

        assert 'estimated_population' in result.columns
        assert 'tourist_stock' in result.columns
        assert 'net_flow' in result.columns

    def test_derived_features_added(self, sample_main_df, sample_population_df):
        """파생 피처 추가"""
        result = add_population_features(
            sample_main_df, sample_population_df, include_derived=True
        )

        assert 'population_change' in result.columns
        assert 'population_change_pct' in result.columns
        assert 'tourist_ratio' in result.columns
        assert 'net_flow_ma7d' in result.columns
        assert 'demand_per_capita' in result.columns

    def test_original_preserved(self, sample_main_df, sample_population_df):
        """원본 데이터 유지"""
        original_cols = sample_main_df.columns.tolist()
        result = add_population_features(sample_main_df, sample_population_df)

        for col in original_cols:
            assert col in result.columns

    def test_copy_not_modify_original(self, sample_main_df, sample_population_df):
        """원본 데이터프레임 수정 안함"""
        original_cols = sample_main_df.columns.tolist()
        _ = add_population_features(sample_main_df, sample_population_df)

        assert sample_main_df.columns.tolist() == original_cols

    def test_population_change_calculation(self, sample_main_df, sample_population_df):
        """인구 변화량 계산 확인"""
        result = add_population_features(sample_main_df, sample_population_df)

        # population_change는 24시간 전 대비
        # 처음 24시간은 NaN이어야 함
        assert pd.isna(result['population_change'].iloc[:24]).all()

    def test_tourist_ratio_calculation(self, sample_main_df, sample_population_df):
        """관광객 비율 계산 확인"""
        result = add_population_features(sample_main_df, sample_population_df)

        # tourist_ratio = tourist_stock / base_population * 100
        expected = result['tourist_stock'] / result['base_population'] * 100
        np.testing.assert_array_almost_equal(
            result['tourist_ratio'].values,
            expected.values
        )

    def test_demand_per_capita(self, sample_main_df, sample_population_df):
        """1인당 수요 계산 확인"""
        result = add_population_features(sample_main_df, sample_population_df)

        # demand_per_capita = power_demand / estimated_population * 1000
        expected = result['power_demand'] / result['estimated_population'] * 1000
        np.testing.assert_array_almost_equal(
            result['demand_per_capita'].values,
            expected.values
        )

    def test_index_alignment(self, sample_main_df):
        """인덱스 정렬 테스트"""
        # 다른 인덱스를 가진 population_df
        pop_dates = pd.date_range('2024-01-01', periods=48, freq='h')
        population_df = pd.DataFrame({
            'estimated_population': np.full(48, 750000)
        }, index=pop_dates)

        result = add_population_features(sample_main_df, population_df)

        # 결과는 main_df의 인덱스를 따라야 함
        assert len(result) == len(sample_main_df)
        pd.testing.assert_index_equal(result.index, sample_main_df.index)


# ============================================================
# add_ev_features 테스트
# ============================================================

class TestAddEvFeatures:
    """add_ev_features 함수 테스트"""

    def test_basic_features_added(self, sample_main_df, sample_ev_df):
        """기본 피처 추가"""
        result = add_ev_features(sample_main_df, sample_ev_df, include_derived=False)

        assert 'ev_cumulative' in result.columns

    def test_derived_features_added(self, sample_main_df, sample_ev_df):
        """파생 피처 추가"""
        # 더 긴 데이터 필요 (30일 이상)
        dates = pd.date_range('2024-01-01', periods=24*35, freq='h')
        main_df = pd.DataFrame({
            'power_demand': np.random.uniform(400, 600, len(dates))
        }, index=dates)
        ev_df = pd.DataFrame({
            'ev_cumulative': np.linspace(50000, 51000, len(dates))
        }, index=dates)

        result = add_ev_features(main_df, ev_df, include_derived=True)

        assert 'ev_growth_monthly' in result.columns
        assert 'ev_growth_rate' in result.columns
        assert 'ev_cumulative_log' in result.columns
        assert 'ev_cumulative_norm' in result.columns

    def test_log_transform(self, sample_main_df, sample_ev_df):
        """로그 변환 확인"""
        result = add_ev_features(sample_main_df, sample_ev_df, include_derived=True)

        # log1p 변환 확인
        expected = np.log1p(result['ev_cumulative'])
        np.testing.assert_array_almost_equal(
            result['ev_cumulative_log'].values,
            expected.values
        )

    def test_normalization(self, sample_main_df, sample_ev_df):
        """정규화 확인"""
        result = add_ev_features(sample_main_df, sample_ev_df, include_derived=True)

        # 0-1 범위 확인
        assert result['ev_cumulative_norm'].min() >= 0
        assert result['ev_cumulative_norm'].max() <= 1

    def test_ev_penetration_with_population(self, sample_main_df, sample_ev_df):
        """전기차 보급률 계산 (인구 데이터 있을 때)"""
        # estimated_population 추가
        sample_main_df['estimated_population'] = 750000

        result = add_ev_features(sample_main_df, sample_ev_df, include_derived=True)

        assert 'ev_penetration' in result.columns
        # ev_penetration = ev_cumulative / estimated_population * 1000
        expected = result['ev_cumulative'] / 750000 * 1000
        np.testing.assert_array_almost_equal(
            result['ev_penetration'].values,
            expected.values
        )

    def test_copy_not_modify_original(self, sample_main_df, sample_ev_df):
        """원본 데이터프레임 수정 안함"""
        original_cols = sample_main_df.columns.tolist()
        _ = add_ev_features(sample_main_df, sample_ev_df)

        assert sample_main_df.columns.tolist() == original_cols


# ============================================================
# add_external_features 테스트
# ============================================================

class TestAddExternalFeatures:
    """add_external_features 함수 테스트"""

    def test_with_population_only(self, sample_main_df, temp_population_csv):
        """인구 피처만 추가"""
        result = add_external_features(
            sample_main_df,
            population_path=temp_population_csv,
            include_population=True,
            include_ev=False
        )

        assert 'estimated_population' in result.columns
        assert 'ev_cumulative' not in result.columns

    def test_with_ev_only(self, sample_main_df, temp_ev_csv):
        """전기차 피처만 추가"""
        result = add_external_features(
            sample_main_df,
            ev_path=temp_ev_csv,
            include_population=False,
            include_ev=True
        )

        assert 'ev_cumulative' in result.columns
        assert 'estimated_population' not in result.columns

    def test_with_both(self, sample_main_df, temp_population_csv, temp_ev_csv):
        """인구 + 전기차 피처 모두 추가"""
        result = add_external_features(
            sample_main_df,
            population_path=temp_population_csv,
            ev_path=temp_ev_csv,
            include_population=True,
            include_ev=True
        )

        assert 'estimated_population' in result.columns
        assert 'ev_cumulative' in result.columns

    def test_missing_file_warning(self, sample_main_df):
        """파일 없을 때 경고"""
        with pytest.warns(UserWarning, match="not found"):
            result = add_external_features(
                sample_main_df,
                population_path='/nonexistent/path.csv',
                include_population=True,
                include_ev=False
            )

        # 파일 없어도 원본 데이터는 유지
        assert 'power_demand' in result.columns

    def test_copy_not_modify_original(self, sample_main_df, temp_population_csv):
        """원본 데이터프레임 수정 안함"""
        original_cols = sample_main_df.columns.tolist()
        _ = add_external_features(
            sample_main_df,
            population_path=temp_population_csv,
            include_population=True,
            include_ev=False
        )

        assert sample_main_df.columns.tolist() == original_cols

    def test_derived_features_flag(self, sample_main_df, temp_population_csv):
        """파생 피처 플래그 테스트"""
        result_with = add_external_features(
            sample_main_df,
            population_path=temp_population_csv,
            include_population=True,
            include_ev=False,
            include_derived=True
        )

        result_without = add_external_features(
            sample_main_df,
            population_path=temp_population_csv,
            include_population=True,
            include_ev=False,
            include_derived=False
        )

        # 파생 피처 포함 시 더 많은 컬럼
        assert len(result_with.columns) > len(result_without.columns)


# ============================================================
# get_external_feature_names 테스트
# ============================================================

class TestGetExternalFeatureNames:
    """get_external_feature_names 함수 테스트"""

    def test_all_features(self):
        """모든 피처 이름"""
        names = get_external_feature_names(
            include_population=True,
            include_ev=True,
            include_derived=True
        )

        # 인구 기본 + 파생
        assert 'estimated_population' in names
        assert 'tourist_stock' in names
        assert 'population_change' in names
        assert 'tourist_ratio' in names

        # 전기차 기본 + 파생
        assert 'ev_cumulative' in names
        assert 'ev_growth_rate' in names
        assert 'ev_cumulative_log' in names

    def test_population_only(self):
        """인구 피처만"""
        names = get_external_feature_names(
            include_population=True,
            include_ev=False,
            include_derived=True
        )

        assert 'estimated_population' in names
        assert 'ev_cumulative' not in names

    def test_ev_only(self):
        """전기차 피처만"""
        names = get_external_feature_names(
            include_population=False,
            include_ev=True,
            include_derived=True
        )

        assert 'ev_cumulative' in names
        assert 'estimated_population' not in names

    def test_no_derived(self):
        """파생 피처 제외"""
        names = get_external_feature_names(
            include_population=True,
            include_ev=True,
            include_derived=False
        )

        assert 'estimated_population' in names
        assert 'ev_cumulative' in names
        assert 'population_change' not in names
        assert 'ev_growth_rate' not in names

    def test_empty(self):
        """피처 없음"""
        names = get_external_feature_names(
            include_population=False,
            include_ev=False
        )

        assert names == []


# ============================================================
# Constants 테스트
# ============================================================

class TestConstants:
    """상수 테스트"""

    def test_population_features(self):
        """POPULATION_FEATURES 확인"""
        assert 'estimated_population' in POPULATION_FEATURES
        assert 'tourist_stock' in POPULATION_FEATURES
        assert 'net_flow' in POPULATION_FEATURES

    def test_population_derived(self):
        """POPULATION_DERIVED 확인"""
        assert 'population_change' in POPULATION_DERIVED
        assert 'tourist_ratio' in POPULATION_DERIVED
        assert 'net_flow_ma7d' in POPULATION_DERIVED

    def test_ev_features(self):
        """EV_FEATURES 확인"""
        assert 'ev_cumulative' in EV_FEATURES

    def test_ev_derived(self):
        """EV_DERIVED 확인"""
        assert 'ev_growth_rate' in EV_DERIVED
        assert 'ev_penetration' in EV_DERIVED
        assert 'ev_cumulative_log' in EV_DERIVED

    def test_all_features_combined(self):
        """EXTERNAL_FEATURES_ALL 확인"""
        expected = POPULATION_FEATURES + POPULATION_DERIVED + EV_FEATURES + EV_DERIVED
        assert EXTERNAL_FEATURES_ALL == expected


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_row(self, sample_population_df, sample_ev_df):
        """단일 행 DataFrame"""
        single_df = pd.DataFrame(
            {'power_demand': [500]},
            index=pd.DatetimeIndex(['2024-01-01 12:00:00'])
        )

        result = add_population_features(single_df, sample_population_df)
        assert len(result) == 1

    def test_empty_df(self, sample_population_df):
        """빈 DataFrame"""
        empty_df = pd.DataFrame(
            {'power_demand': []},
            index=pd.DatetimeIndex([])
        )

        result = add_population_features(empty_df, sample_population_df)
        assert len(result) == 0

    def test_mismatched_index(self):
        """인덱스 불일치"""
        main_dates = pd.date_range('2024-01-01', periods=24, freq='h')
        pop_dates = pd.date_range('2024-02-01', periods=24, freq='h')

        main_df = pd.DataFrame({'power_demand': np.random.rand(24)}, index=main_dates)
        pop_df = pd.DataFrame({'estimated_population': np.full(24, 750000)}, index=pop_dates)

        # ffill로 채워지므로 에러 없이 처리됨 (NaN이 있을 수 있음)
        result = add_population_features(main_df, pop_df)
        assert len(result) == 24

    def test_partial_columns(self):
        """일부 컬럼만 있는 경우"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'date': ['2024-01-01'],
                'estimated_population': [750000]
                # 다른 컬럼 없음
            })
            df.to_csv(f.name, index=False)

            result = load_population_data(f.name, resample_to_hourly=False)

            assert 'estimated_population' in result.columns
            assert len(result) == 1

        os.unlink(f.name)


# ============================================================
# 데이터 타입 테스트
# ============================================================

class TestDataTypes:
    """데이터 타입 테스트"""

    def test_numeric_types(self, sample_main_df, sample_population_df):
        """수치형 타입 확인"""
        result = add_population_features(sample_main_df, sample_population_df)

        assert result['estimated_population'].dtype in [np.float64, np.int64]
        assert result['tourist_ratio'].dtype == np.float64

    def test_index_preserved(self, sample_main_df, sample_population_df):
        """인덱스 타입 유지"""
        result = add_population_features(sample_main_df, sample_population_df)

        assert isinstance(result.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
