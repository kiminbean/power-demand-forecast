"""
DATA-003: 데이터셋 병합 모듈 단위 테스트

테스트 범위:
1. 데이터 로드 함수 (load_cleaned_data, load_visitors_data, etc.)
2. 일별→시간별 확장 (expand_daily_to_hourly)
3. 데이터셋 병합 (merge_all_datasets)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.merge_datasets import (
    load_cleaned_data,
    load_visitors_data,
    load_ev_data,
    load_solar_data,
    expand_daily_to_hourly,
    merge_all_datasets,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_cleaned_csv():
    """임시 전처리된 데이터 CSV"""
    dates = pd.date_range('2024-01-01', periods=48, freq='h')
    df = pd.DataFrame({
        'power_demand': np.random.uniform(400, 600, 48),
        'temp_mean': np.random.uniform(5, 15, 48)
    }, index=dates)
    df.index.name = 'datetime'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, encoding='utf-8-sig')
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_visitors_csv():
    """임시 관광객 데이터 CSV"""
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        '날짜': dates.strftime('%Y-%m-%d'),
        '일별_입도객수': [15000, 16000, 14000]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False, encoding='utf-8-sig')
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_ev_csv():
    """임시 전기차 데이터 CSV"""
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'Cumulative EV Count': [50000, 50050, 50100],
        'ev_daily_new': [50, 50, 50]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False, encoding='utf-8-sig')
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def temp_solar_csv():
    """임시 태양광 데이터 CSV"""
    dates = pd.date_range('2024-01-01', periods=48, freq='h')
    df = pd.DataFrame({
        '일시': dates,
        '태양광 설비용량(MW)': np.full(48, 500.0),
        '태양광 발전량(MWh)': np.random.uniform(0, 100, 48)
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False, encoding='utf-8-sig')
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def sample_cleaned_df():
    """샘플 전처리된 DataFrame"""
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'power_demand': np.random.uniform(400, 600, 72),
        'temp_mean': np.random.uniform(5, 15, 72)
    }, index=dates)


@pytest.fixture
def sample_visitors_df():
    """샘플 관광객 DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D').date,
        'visitors': [15000, 16000, 14000]
    })


@pytest.fixture
def sample_ev_df():
    """샘플 전기차 DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D').date,
        'ev_cumulative': [50000, 50050, 50100],
        'ev_daily_new': [50, 50, 50]
    })


@pytest.fixture
def sample_solar_df():
    """샘플 태양광 DataFrame"""
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'solar_capacity_mw': np.full(72, 500.0),
        'solar_generation_mwh': np.random.uniform(0, 100, 72)
    }, index=dates)


# ============================================================
# load_cleaned_data 테스트
# ============================================================

class TestLoadCleanedData:
    """load_cleaned_data 함수 테스트"""

    def test_basic_load(self, temp_cleaned_csv):
        """기본 로드 테스트"""
        df = load_cleaned_data(Path(temp_cleaned_csv))

        assert len(df) == 48
        assert 'power_demand' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_name(self, temp_cleaned_csv):
        """인덱스 이름 확인"""
        df = load_cleaned_data(Path(temp_cleaned_csv))

        assert df.index.name == 'datetime'

    def test_datetime_parsed(self, temp_cleaned_csv):
        """datetime 파싱 확인"""
        df = load_cleaned_data(Path(temp_cleaned_csv))

        assert pd.api.types.is_datetime64_any_dtype(df.index)


# ============================================================
# load_visitors_data 테스트
# ============================================================

class TestLoadVisitorsData:
    """load_visitors_data 함수 테스트"""

    def test_basic_load(self, temp_visitors_csv):
        """기본 로드 테스트"""
        df = load_visitors_data(Path(temp_visitors_csv))

        assert len(df) == 3
        assert 'date' in df.columns
        assert 'visitors' in df.columns

    def test_column_rename(self, temp_visitors_csv):
        """컬럼명 변경 확인"""
        df = load_visitors_data(Path(temp_visitors_csv))

        assert 'visitors' in df.columns
        assert '일별_입도객수' not in df.columns

    def test_na_dropped(self):
        """NaN 제거 확인"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                '날짜': ['2024-01-01', '2024-01-02', '2024-01-03'],
                '일별_입도객수': [15000, np.nan, 14000]
            })
            data.to_csv(f.name, index=False, encoding='utf-8-sig')

            df = load_visitors_data(Path(f.name))

            assert len(df) == 2  # NaN 행 제거됨

        Path(f.name).unlink()


# ============================================================
# load_ev_data 테스트
# ============================================================

class TestLoadEvData:
    """load_ev_data 함수 테스트"""

    def test_basic_load(self, temp_ev_csv):
        """기본 로드 테스트"""
        df = load_ev_data(Path(temp_ev_csv))

        assert len(df) == 3
        assert 'date' in df.columns
        assert 'ev_cumulative' in df.columns
        assert 'ev_daily_new' in df.columns

    def test_column_rename(self, temp_ev_csv):
        """컬럼명 변경 확인"""
        df = load_ev_data(Path(temp_ev_csv))

        assert 'ev_cumulative' in df.columns
        assert 'Cumulative EV Count' not in df.columns


# ============================================================
# load_solar_data 테스트
# ============================================================

class TestLoadSolarData:
    """load_solar_data 함수 테스트"""

    def test_basic_load(self, temp_solar_csv):
        """기본 로드 테스트"""
        df = load_solar_data(Path(temp_solar_csv))

        assert len(df) == 48
        assert 'solar_capacity_mw' in df.columns
        assert 'solar_generation_mwh' in df.columns

    def test_datetime_index(self, temp_solar_csv):
        """datetime 인덱스 확인"""
        df = load_solar_data(Path(temp_solar_csv))

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_column_rename(self, temp_solar_csv):
        """컬럼명 변경 확인"""
        df = load_solar_data(Path(temp_solar_csv))

        assert 'solar_capacity_mw' in df.columns
        assert '태양광 설비용량(MW)' not in df.columns


# ============================================================
# expand_daily_to_hourly 테스트
# ============================================================

class TestExpandDailyToHourly:
    """expand_daily_to_hourly 함수 테스트"""

    def test_basic_expansion(self):
        """기본 확장 테스트"""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='D'),
            'value': [100, 200]
        })
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert len(result) == 48
        assert 'value' in result.columns

    def test_values_broadcast(self):
        """값이 올바르게 브로드캐스트되는지 확인"""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='D'),
            'value': [100, 200]
        })
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        # 첫째 날 모든 시간은 100
        first_day = result.loc['2024-01-01', 'value']
        assert (first_day == 100).all()

        # 둘째 날 모든 시간은 200
        second_day = result.loc['2024-01-02', 'value']
        assert (second_day == 200).all()

    def test_missing_dates_nan(self):
        """데이터 없는 날짜는 NaN"""
        daily_df = pd.DataFrame({
            'date': ['2024-01-01'],
            'value': [100]
        })
        # 3일치 시간별 인덱스 (1일만 데이터 있음)
        hourly_index = pd.date_range('2024-01-01', periods=72, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        # 첫째 날은 값 있음
        assert result.loc['2024-01-01', 'value'].notna().all()

        # 셋째 날은 NaN
        assert result.loc['2024-01-03', 'value'].isna().all()

    def test_duplicate_dates_first(self):
        """중복 날짜는 첫 번째 값 사용"""
        daily_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01'],
            'value': [100, 999]
        })
        hourly_index = pd.date_range('2024-01-01', periods=24, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        # 첫 번째 값(100) 사용
        assert (result['value'] == 100).all()

    def test_multiple_columns(self):
        """여러 컬럼 확장"""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='D'),
            'value1': [100, 200],
            'value2': [10, 20]
        })
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert 'value1' in result.columns
        assert 'value2' in result.columns
        assert (result.loc['2024-01-01', 'value1'] == 100).all()
        assert (result.loc['2024-01-01', 'value2'] == 10).all()

    def test_date_column_excluded(self):
        """date 컬럼은 결과에서 제외"""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='D'),
            'value': [100, 200]
        })
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert 'date' not in result.columns

    def test_string_date_handled(self):
        """문자열 날짜 처리"""
        daily_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'value': [100, 200]
        })
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert len(result) == 48

    def test_copy_not_modify_original(self):
        """원본 수정 안함"""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2, freq='D'),
            'value': [100, 200]
        })
        original_len = len(daily_df)
        hourly_index = pd.date_range('2024-01-01', periods=48, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert len(daily_df) == original_len


# ============================================================
# merge_all_datasets 테스트
# ============================================================

class TestMergeAllDatasets:
    """merge_all_datasets 함수 테스트"""

    def test_basic_merge(self, sample_cleaned_df, sample_visitors_df,
                         sample_ev_df, sample_solar_df):
        """기본 병합 테스트"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        # 원본 컬럼 유지
        assert 'power_demand' in result.columns
        assert 'temp_mean' in result.columns

        # 새 컬럼 추가
        assert 'visitors' in result.columns
        assert 'ev_cumulative' in result.columns
        assert 'solar_generation_mwh' in result.columns

    def test_stats_returned(self, sample_cleaned_df, sample_visitors_df,
                            sample_ev_df, sample_solar_df):
        """통계 반환 확인"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert 'steps' in stats
        assert len(stats['steps']) == 3  # visitors, ev, solar
        assert 'final_shape' in stats

    def test_visitors_merge_stats(self, sample_cleaned_df, sample_visitors_df,
                                  sample_ev_df, sample_solar_df):
        """관광객 병합 통계"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        visitors_step = next(
            (s for s in stats['steps'] if s['step'] == 'visitors_merge'),
            None
        )
        assert visitors_step is not None
        assert 'coverage_pct' in visitors_step

    def test_ev_merge_stats(self, sample_cleaned_df, sample_visitors_df,
                            sample_ev_df, sample_solar_df):
        """전기차 병합 통계"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        ev_step = next(
            (s for s in stats['steps'] if s['step'] == 'ev_merge'),
            None
        )
        assert ev_step is not None
        assert 'coverage_pct' in ev_step

    def test_solar_merge_stats(self, sample_cleaned_df, sample_visitors_df,
                               sample_ev_df, sample_solar_df):
        """태양광 병합 통계"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        solar_step = next(
            (s for s in stats['steps'] if s['step'] == 'solar_merge'),
            None
        )
        assert solar_step is not None
        assert 'coverage_pct' in solar_step

    def test_missing_interpolated(self, sample_cleaned_df, sample_visitors_df,
                                  sample_ev_df, sample_solar_df):
        """결측치 보간 확인"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        # visitors와 ev_cumulative는 보간됨
        assert result['visitors'].isna().sum() == 0
        assert result['ev_cumulative'].isna().sum() == 0

    def test_ev_daily_new_fillna_zero(self, sample_cleaned_df, sample_visitors_df,
                                      sample_ev_df, sample_solar_df):
        """ev_daily_new NaN은 0으로 채움"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert result['ev_daily_new'].isna().sum() == 0

    def test_index_preserved(self, sample_cleaned_df, sample_visitors_df,
                             sample_ev_df, sample_solar_df):
        """인덱스 유지"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert len(result) == len(sample_cleaned_df)
        pd.testing.assert_index_equal(result.index, sample_cleaned_df.index)

    def test_copy_not_modify_original(self, sample_cleaned_df, sample_visitors_df,
                                      sample_ev_df, sample_solar_df):
        """원본 수정 안함"""
        original_cols = sample_cleaned_df.columns.tolist()

        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert sample_cleaned_df.columns.tolist() == original_cols

    def test_final_missing_tracked(self, sample_cleaned_df, sample_visitors_df,
                                   sample_ev_df, sample_solar_df):
        """최종 결측치 추적"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert 'final_missing' in stats


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_merge_pipeline(self):
        """전체 병합 파이프라인"""
        # 긴 기간의 데이터 생성
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # 7일

        cleaned_df = pd.DataFrame({
            'power_demand': np.random.uniform(400, 600, 168),
            'temp_mean': np.random.uniform(5, 15, 168)
        }, index=dates)

        visitors_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7, freq='D').date,
            'visitors': np.random.uniform(10000, 20000, 7)
        })

        ev_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7, freq='D').date,
            'ev_cumulative': np.linspace(50000, 50350, 7),
            'ev_daily_new': np.full(7, 50)
        })

        solar_df = pd.DataFrame({
            'solar_capacity_mw': np.full(168, 500.0),
            'solar_generation_mwh': np.random.uniform(0, 100, 168)
        }, index=dates)

        result, stats = merge_all_datasets(
            cleaned_df, visitors_df, ev_df, solar_df
        )

        # 모든 데이터 병합됨
        assert len(result) == 168
        assert 'visitors' in result.columns
        assert 'ev_cumulative' in result.columns
        assert 'solar_generation_mwh' in result.columns

        # 결측치 처리됨
        assert result['visitors'].isna().sum() == 0
        assert result['ev_cumulative'].isna().sum() == 0

    def test_partial_overlap(self):
        """부분 겹침 데이터"""
        # cleaned: 1/1 ~ 1/3
        dates = pd.date_range('2024-01-01', periods=72, freq='h')
        cleaned_df = pd.DataFrame({
            'power_demand': np.random.uniform(400, 600, 72)
        }, index=dates)

        # visitors: 1/2 ~ 1/4 (1/1 데이터 없음)
        visitors_df = pd.DataFrame({
            'date': pd.date_range('2024-01-02', periods=3, freq='D').date,
            'visitors': [15000, 16000, 17000]
        })

        ev_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='D').date,
            'ev_cumulative': [50000, 50050, 50100],
            'ev_daily_new': [50, 50, 50]
        })

        solar_df = pd.DataFrame({
            'solar_capacity_mw': np.full(72, 500.0),
            'solar_generation_mwh': np.random.uniform(0, 100, 72)
        }, index=dates)

        result, stats = merge_all_datasets(
            cleaned_df, visitors_df, ev_df, solar_df
        )

        # 보간으로 1/1 visitors도 채워짐
        assert result['visitors'].isna().sum() == 0


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_daily_data(self):
        """빈 일별 데이터"""
        daily_df = pd.DataFrame({
            'date': [],
            'value': []
        })
        hourly_index = pd.date_range('2024-01-01', periods=24, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert len(result) == 24
        assert result['value'].isna().all()

    def test_single_day(self):
        """단일 일 데이터"""
        daily_df = pd.DataFrame({
            'date': ['2024-01-01'],
            'value': [100]
        })
        hourly_index = pd.date_range('2024-01-01', periods=24, freq='h')

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        assert len(result) == 24
        assert (result['value'] == 100).all()

    def test_non_contiguous_dates(self):
        """비연속 날짜"""
        daily_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-05'],  # 2-4일 없음
            'value': [100, 500]
        })
        hourly_index = pd.date_range('2024-01-01', periods=120, freq='h')  # 5일

        result = expand_daily_to_hourly(daily_df, 'date', hourly_index)

        # 1일, 5일만 값 있음
        assert result.loc['2024-01-01', 'value'].notna().all()
        assert result.loc['2024-01-05', 'value'].notna().all()
        assert result.loc['2024-01-03', 'value'].isna().all()

    def test_solar_pre_2018(self):
        """2018년 이전 태양광 데이터"""
        dates = pd.date_range('2017-01-01', periods=72, freq='h')

        cleaned_df = pd.DataFrame({
            'power_demand': np.random.uniform(400, 600, 72)
        }, index=dates)

        visitors_df = pd.DataFrame({
            'date': pd.date_range('2017-01-01', periods=3, freq='D').date,
            'visitors': [15000, 16000, 17000]
        })

        ev_df = pd.DataFrame({
            'date': pd.date_range('2017-01-01', periods=3, freq='D').date,
            'ev_cumulative': [50000, 50050, 50100],
            'ev_daily_new': [50, 50, 50]
        })

        # 태양광 데이터 없음 (빈 DataFrame)
        solar_df = pd.DataFrame(
            columns=['solar_capacity_mw', 'solar_generation_mwh'],
            index=pd.DatetimeIndex([])
        )

        result, stats = merge_all_datasets(
            cleaned_df, visitors_df, ev_df, solar_df
        )

        # 2018년 이전이므로 태양광 NaN 유지
        assert result['solar_generation_mwh'].isna().all()


# ============================================================
# 데이터 타입 테스트
# ============================================================

class TestDataTypes:
    """데이터 타입 테스트"""

    def test_numeric_types(self, sample_cleaned_df, sample_visitors_df,
                           sample_ev_df, sample_solar_df):
        """수치형 타입 확인"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert result['visitors'].dtype in [np.float64, np.int64]
        assert result['ev_cumulative'].dtype in [np.float64, np.int64]

    def test_index_type(self, sample_cleaned_df, sample_visitors_df,
                        sample_ev_df, sample_solar_df):
        """인덱스 타입 확인"""
        result, stats = merge_all_datasets(
            sample_cleaned_df,
            sample_visitors_df,
            sample_ev_df,
            sample_solar_df
        )

        assert isinstance(result.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
