"""
DATA-002: 전처리 모듈 단위 테스트

테스트 범위:
1. 결측치 보간 (interpolate_missing_values)
2. 이상치 처리 (handle_outliers)
3. 기상 데이터 전처리 (preprocess_weather_data)
4. 전력 데이터 전처리 (preprocess_power_data)
5. 전처리 리포트 생성 (create_preprocessing_report)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import (
    interpolate_missing_values,
    handle_outliers,
    preprocess_weather_data,
    preprocess_power_data,
    create_preprocessing_report,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df_with_missing():
    """결측치가 있는 샘플 DataFrame"""
    return pd.DataFrame({
        'A': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
        'B': [10.0, 20.0, np.nan, np.nan, 50.0, 60.0],
        'C': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    })


@pytest.fixture
def sample_df_with_outliers():
    """이상치가 있는 샘플 DataFrame"""
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 100)
    # 이상치 추가
    normal_data[5] = 200  # 상위 이상치
    normal_data[10] = 20  # 하위 이상치
    normal_data[15] = 250  # 극단 이상치

    return pd.DataFrame({
        'value': normal_data,
        'other': np.random.rand(100) * 50
    })


@pytest.fixture
def sample_weather_df():
    """샘플 기상 데이터 DataFrame"""
    dates = pd.date_range('2024-01-01', periods=48, freq='h')
    np.random.seed(42)

    return pd.DataFrame({
        'datetime': dates,
        '기온': np.random.uniform(0, 20, 48),
        '습도': np.random.uniform(40, 80, 48),
        '풍속': np.random.uniform(0, 10, 48),
        '강수량': [np.nan if i % 10 != 0 else 5.0 for i in range(48)],
        '일조': [0.0 if (h.hour >= 19 or h.hour <= 5) else np.random.uniform(0, 1, 1)[0]
                for h in dates],
        '일사': [0.0 if (h.hour >= 19 or h.hour <= 5) else np.random.uniform(0, 0.5, 1)[0]
                for h in dates]
    })


@pytest.fixture
def sample_power_df():
    """샘플 전력 데이터 DataFrame"""
    dates = pd.date_range('2024-01-01', periods=48, freq='h')

    return pd.DataFrame({
        '거래일자': [d.strftime('%Y-%m-%d') for d in dates],
        '시간': [(d.hour + 1) for d in dates],  # 1-24 형식
        '전력거래량(MWh)': np.random.uniform(400, 600, 48)
    })


# ============================================================
# interpolate_missing_values 테스트
# ============================================================

class TestInterpolateMissingValues:
    """interpolate_missing_values 함수 테스트"""

    def test_basic_interpolation(self, sample_df_with_missing):
        """기본 선형 보간"""
        result, stats = interpolate_missing_values(sample_df_with_missing)

        # 결측치가 없어야 함
        assert result['A'].isnull().sum() == 0
        assert result['B'].isnull().sum() == 0

    def test_linear_interpolation_values(self):
        """선형 보간 값 확인"""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0]
        })
        result, stats = interpolate_missing_values(df)

        # 1과 3 사이의 선형 보간 = 2
        assert result['A'].iloc[1] == 2.0

    def test_stats_returned(self, sample_df_with_missing):
        """통계 반환 확인"""
        result, stats = interpolate_missing_values(sample_df_with_missing)

        assert 'A' in stats
        assert 'B' in stats
        assert stats['A']['missing_before'] == 2
        assert stats['B']['missing_before'] == 2

    def test_specific_columns(self, sample_df_with_missing):
        """특정 컬럼만 보간"""
        result, stats = interpolate_missing_values(
            sample_df_with_missing,
            columns=['A']
        )

        assert result['A'].isnull().sum() == 0
        # B는 처리되지 않음
        assert result['B'].isnull().sum() == 2

    def test_no_missing_values(self):
        """결측치 없는 경우"""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        result, stats = interpolate_missing_values(df)

        # 통계가 비어있어야 함
        assert len(stats) == 0

    def test_edge_missing_handled(self):
        """시작/끝 결측치 처리 (ffill/bfill)"""
        df = pd.DataFrame({
            'A': [np.nan, 2.0, 3.0, np.nan]
        })
        result, stats = interpolate_missing_values(df)

        # ffill/bfill로 처리됨
        assert result['A'].isnull().sum() == 0
        assert result['A'].iloc[0] == 2.0  # bfill
        assert result['A'].iloc[3] == 3.0  # ffill

    def test_all_nan_column(self):
        """전체 NaN 컬럼"""
        df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan]
        })
        result, stats = interpolate_missing_values(df)

        # 전부 NaN이면 보간 불가능
        assert result['A'].isnull().all()

    def test_method_time(self):
        """시간 기반 보간"""
        dates = pd.date_range('2024-01-01', periods=5, freq='h')
        df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, np.nan, 5.0]
        }, index=dates)

        result, stats = interpolate_missing_values(df, method='time')

        # 시간 기반 선형 보간
        assert result['A'].isnull().sum() == 0

    def test_limit_parameter(self):
        """연속 보간 제한"""
        df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, np.nan, 5.0]
        })
        result, stats = interpolate_missing_values(df, limit=1)

        # limit=1이므로 중간 값은 여전히 NaN (ffill/bfill로 처리됨)
        # ffill/bfill이 마지막에 적용되므로 결측치 없음
        assert result['A'].isnull().sum() == 0

    def test_copy_not_modify_original(self, sample_df_with_missing):
        """원본 수정 안함"""
        original_nulls = sample_df_with_missing['A'].isnull().sum()
        result, stats = interpolate_missing_values(sample_df_with_missing)

        assert sample_df_with_missing['A'].isnull().sum() == original_nulls

    def test_missing_column_skipped(self, sample_df_with_missing):
        """존재하지 않는 컬럼 스킵"""
        result, stats = interpolate_missing_values(
            sample_df_with_missing,
            columns=['A', 'NonExistent']
        )

        # 에러 없이 처리
        assert result['A'].isnull().sum() == 0


# ============================================================
# handle_outliers 테스트
# ============================================================

class TestHandleOutliers:
    """handle_outliers 함수 테스트"""

    def test_clip_method(self, sample_df_with_outliers):
        """클리핑 방법"""
        result, stats = handle_outliers(
            sample_df_with_outliers,
            columns=['value'],
            method='clip'
        )

        # 이상치가 경계값으로 클리핑됨
        assert 'value' in stats
        assert stats['value']['outlier_count'] > 0

    def test_interpolate_method(self, sample_df_with_outliers):
        """보간 방법"""
        result, stats = handle_outliers(
            sample_df_with_outliers,
            columns=['value'],
            method='interpolate'
        )

        # 이상치가 보간으로 대체됨
        assert 'value' in stats
        assert result['value'].isnull().sum() == 0

    def test_iqr_bounds(self):
        """IQR 경계 계산 확인"""
        # 알려진 분포로 테스트
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100은 이상치
        })
        result, stats = handle_outliers(df, columns=['value'], k=1.5)

        assert 'value' in stats
        assert stats['value']['outlier_count'] >= 1
        # 100이 클리핑되어야 함
        assert result['value'].max() < 100

    def test_k_parameter(self):
        """k 파라미터 영향"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]
        })

        # k=1.5: 더 민감
        result_strict, stats_strict = handle_outliers(df, k=1.5)

        # k=3.0: 더 관대
        result_loose, stats_loose = handle_outliers(df, k=3.0)

        # 더 작은 k일수록 더 많은 이상치 감지
        strict_count = stats_strict.get('value', {}).get('outlier_count', 0)
        loose_count = stats_loose.get('value', {}).get('outlier_count', 0)
        assert strict_count >= loose_count

    def test_stats_content(self, sample_df_with_outliers):
        """통계 내용 확인"""
        result, stats = handle_outliers(sample_df_with_outliers, columns=['value'])

        if 'value' in stats:
            assert 'outlier_count' in stats['value']
            assert 'outlier_pct' in stats['value']
            assert 'lower_bound' in stats['value']
            assert 'upper_bound' in stats['value']
            assert 'method' in stats['value']

    def test_no_outliers(self):
        """이상치 없는 경우"""
        df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        result, stats = handle_outliers(df)

        # 이상치 없으면 통계 비어있음
        assert len(stats) == 0 or stats.get('value', {}).get('outlier_count', 0) == 0

    def test_all_same_values(self):
        """모든 값이 동일한 경우"""
        df = pd.DataFrame({
            'value': [5.0, 5.0, 5.0, 5.0, 5.0]
        })
        result, stats = handle_outliers(df)

        # IQR = 0이므로 이상치 없음
        assert len(stats) == 0

    def test_empty_series_after_dropna(self):
        """dropna 후 빈 시리즈"""
        df = pd.DataFrame({
            'value': [np.nan, np.nan, np.nan]
        })
        result, stats = handle_outliers(df)

        # 에러 없이 처리
        assert len(stats) == 0

    def test_copy_not_modify_original(self, sample_df_with_outliers):
        """원본 수정 안함"""
        original_max = sample_df_with_outliers['value'].max()
        result, stats = handle_outliers(sample_df_with_outliers)

        assert sample_df_with_outliers['value'].max() == original_max

    def test_specific_columns(self, sample_df_with_outliers):
        """특정 컬럼만 처리"""
        result, stats = handle_outliers(
            sample_df_with_outliers,
            columns=['value']
        )

        # 'other' 컬럼은 처리되지 않음
        assert 'other' not in stats or stats.get('other', {}).get('outlier_count', 0) == 0


# ============================================================
# preprocess_weather_data 테스트
# ============================================================

class TestPreprocessWeatherData:
    """preprocess_weather_data 함수 테스트"""

    def test_basic_preprocessing(self, sample_weather_df):
        """기본 전처리"""
        result, stats = preprocess_weather_data(sample_weather_df)

        assert 'datetime' in result.columns
        assert '기온' in result.columns
        assert len(stats['steps']) > 0

    def test_precipitation_fill_zero(self, sample_weather_df):
        """강수량 NaN → 0 처리"""
        result, stats = preprocess_weather_data(sample_weather_df)

        # 강수량에 NaN이 없어야 함
        assert result['강수량'].isnull().sum() == 0

        # 특수 처리 통계 확인
        special_step = next(
            (s for s in stats['steps'] if s['step'] == 'special_columns'),
            None
        )
        assert special_step is not None
        assert '강수량' in special_step['details']

    def test_night_solar_fill_zero(self):
        """야간 일조/일사 NaN → 0 처리"""
        # 야간 시간에 NaN이 있는 데이터
        dates = pd.date_range('2024-01-01 20:00:00', periods=6, freq='h')
        df = pd.DataFrame({
            'datetime': dates,
            '일조': [np.nan] * 6,
            '일사': [np.nan] * 6,
            '기온': [10.0] * 6
        })

        result, stats = preprocess_weather_data(df)

        # 야간 NaN은 0으로 처리되어야 함
        # 모든 시간이 야간(20:00~01:00)이므로
        assert (result['일조'] == 0).all() or result['일조'].isnull().sum() == 0

    def test_outlier_handling(self, sample_weather_df):
        """이상치 처리 단계"""
        result, stats = preprocess_weather_data(sample_weather_df)

        outlier_step = next(
            (s for s in stats['steps'] if s['step'] == 'outlier_handling'),
            None
        )
        assert outlier_step is not None
        assert 'config' in outlier_step

    def test_interpolation_step(self, sample_weather_df):
        """보간 단계"""
        result, stats = preprocess_weather_data(sample_weather_df)

        interp_step = next(
            (s for s in stats['steps'] if s['step'] == 'interpolation'),
            None
        )
        assert interp_step is not None

    def test_custom_config(self, sample_weather_df):
        """사용자 정의 설정"""
        result, stats = preprocess_weather_data(
            sample_weather_df,
            interpolation_config={'method': 'linear', 'limit': 5},
            outlier_config={'method': 'interpolate', 'k': 2.0}
        )

        # 설정이 적용되었는지 확인
        outlier_step = next(
            (s for s in stats['steps'] if s['step'] == 'outlier_handling'),
            None
        )
        assert outlier_step['config']['k'] == 2.0

    def test_datetime_column_handling(self):
        """datetime 컬럼 처리"""
        # '일시' 컬럼 사용
        df = pd.DataFrame({
            '일시': pd.date_range('2024-01-01', periods=5, freq='h'),
            '기온': [10.0, 11.0, 12.0, 13.0, 14.0]
        })

        result, stats = preprocess_weather_data(df)

        assert 'datetime' in result.columns

    def test_final_missing_tracked(self, sample_weather_df):
        """최종 결측치 추적"""
        result, stats = preprocess_weather_data(sample_weather_df)

        assert 'final_missing' in stats

    def test_copy_not_modify_original(self, sample_weather_df):
        """원본 수정 안함"""
        original_shape = sample_weather_df.shape
        result, stats = preprocess_weather_data(sample_weather_df)

        assert sample_weather_df.shape == original_shape


# ============================================================
# preprocess_power_data 테스트
# ============================================================

class TestPreprocessPowerData:
    """preprocess_power_data 함수 테스트"""

    def test_basic_preprocessing(self, sample_power_df):
        """기본 전처리"""
        result, stats = preprocess_power_data(sample_power_df)

        assert 'datetime' in result.columns
        assert 'power_demand' in result.columns

    def test_column_rename(self, sample_power_df):
        """컬럼명 변경"""
        result, stats = preprocess_power_data(sample_power_df)

        # '전력거래량(MWh)' → 'power_demand'
        assert 'power_demand' in result.columns
        assert '전력거래량(MWh)' not in result.columns

    def test_datetime_creation(self, sample_power_df):
        """datetime 컬럼 생성"""
        result, stats = preprocess_power_data(sample_power_df)

        assert 'datetime' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['datetime'])

    def test_hour_conversion(self, sample_power_df):
        """시간 변환 (1-24 → 0-23)"""
        result, stats = preprocess_power_data(sample_power_df)

        # datetime의 시간 확인
        hours = pd.to_datetime(result['datetime']).dt.hour
        assert hours.min() >= 0
        assert hours.max() <= 23

    def test_outlier_handling(self, sample_power_df):
        """이상치 처리"""
        result, stats = preprocess_power_data(sample_power_df)

        outlier_step = next(
            (s for s in stats['steps'] if s['step'] == 'outlier_handling'),
            None
        )
        assert outlier_step is not None

    def test_missing_value_interpolation(self):
        """결측치 보간"""
        df = pd.DataFrame({
            '거래일자': ['2024-01-01'] * 5,
            '시간': [1, 2, 3, 4, 5],
            '전력거래량(MWh)': [500.0, np.nan, 520.0, np.nan, 540.0]
        })

        result, stats = preprocess_power_data(df)

        # 결측치가 보간됨
        assert result['power_demand'].isnull().sum() == 0

        # 보간 단계 통계
        interp_step = next(
            (s for s in stats['steps'] if s['step'] == 'interpolation'),
            None
        )
        assert interp_step is not None
        assert interp_step['interpolated_count'] == 2

    def test_custom_outlier_config(self, sample_power_df):
        """사용자 정의 이상치 설정"""
        result, stats = preprocess_power_data(
            sample_power_df,
            outlier_config={'method': 'interpolate', 'k': 2.0}
        )

        outlier_step = next(
            (s for s in stats['steps'] if s['step'] == 'outlier_handling'),
            None
        )
        assert outlier_step['config']['k'] == 2.0

    def test_only_required_columns(self, sample_power_df):
        """필요한 컬럼만 유지"""
        # 추가 컬럼이 있는 데이터
        sample_power_df['extra_col'] = 999

        result, stats = preprocess_power_data(sample_power_df)

        assert 'datetime' in result.columns
        assert 'power_demand' in result.columns
        assert 'extra_col' not in result.columns

    def test_sorted_by_datetime(self, sample_power_df):
        """datetime으로 정렬"""
        # 역순 데이터
        reversed_df = sample_power_df.iloc[::-1].reset_index(drop=True)

        result, stats = preprocess_power_data(reversed_df)

        # 정렬되어 있어야 함
        datetimes = pd.to_datetime(result['datetime'])
        assert (datetimes.diff().dropna() >= pd.Timedelta(0)).all()

    def test_copy_not_modify_original(self, sample_power_df):
        """원본 수정 안함"""
        original_cols = sample_power_df.columns.tolist()
        result, stats = preprocess_power_data(sample_power_df)

        assert sample_power_df.columns.tolist() == original_cols


# ============================================================
# create_preprocessing_report 테스트
# ============================================================

class TestCreatePreprocessingReport:
    """create_preprocessing_report 함수 테스트"""

    def test_basic_report(self):
        """기본 리포트 생성"""
        weather_stats = {
            'steps': [{'step': 'test', 'details': {}}],
            'final_missing': {}
        }
        power_stats = {
            'steps': [{'step': 'test2', 'details': {}}],
            'final_missing': {}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'reports' / 'test_report.json'
            report = create_preprocessing_report(weather_stats, power_stats, output_path)

            assert 'generated_at' in report
            assert 'task_id' in report
            assert 'weather_preprocessing' in report
            assert 'power_preprocessing' in report
            assert 'summary' in report

    def test_report_saved(self):
        """리포트 파일 저장"""
        weather_stats = {'steps': [], 'final_missing': {}}
        power_stats = {'steps': [], 'final_missing': {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'reports' / 'test_report.json'
            report = create_preprocessing_report(weather_stats, power_stats, output_path)

            assert output_path.exists()

            # JSON 파일 내용 확인
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_report = json.load(f)

            assert saved_report['task_id'] == 'DATA-002'

    def test_summary_content(self):
        """요약 내용 확인"""
        weather_stats = {
            'steps': [{'step': '1'}, {'step': '2'}],
            'final_missing': {'col1': 5}
        }
        power_stats = {
            'steps': [{'step': '1'}],
            'final_missing': {}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'reports' / 'test_report.json'
            report = create_preprocessing_report(weather_stats, power_stats, output_path)

            assert report['summary']['weather_steps'] == 2
            assert report['summary']['power_steps'] == 1
            assert report['summary']['weather_final_missing'] == {'col1': 5}

    def test_directory_creation(self):
        """디렉토리 자동 생성"""
        weather_stats = {'steps': [], 'final_missing': {}}
        power_stats = {'steps': [], 'final_missing': {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            # 존재하지 않는 중첩 디렉토리
            output_path = Path(tmpdir) / 'a' / 'b' / 'c' / 'report.json'
            report = create_preprocessing_report(weather_stats, power_stats, output_path)

            assert output_path.exists()


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_weather_pipeline(self):
        """전체 기상 데이터 파이프라인"""
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # 7일
        np.random.seed(42)

        df = pd.DataFrame({
            'datetime': dates,
            '기온': np.random.uniform(0, 20, 168),
            '습도': np.random.uniform(40, 80, 168),
            '풍속': np.random.uniform(0, 10, 168),
            '강수량': [np.nan if i % 12 != 0 else 5.0 for i in range(168)],
            '일조': [0.0 if (h.hour >= 19 or h.hour <= 5) else np.random.uniform(0, 1, 1)[0]
                    for h in dates],
        })

        # 일부 결측치 추가
        df.loc[10:15, '기온'] = np.nan
        df.loc[50, '습도'] = np.nan

        # 이상치 추가
        df.loc[100, '기온'] = 100.0  # 이상치

        result, stats = preprocess_weather_data(df)

        # 결측치 처리됨
        assert result['기온'].isnull().sum() == 0
        assert result['습도'].isnull().sum() == 0
        assert result['강수량'].isnull().sum() == 0

        # 이상치 처리됨
        assert result['기온'].max() < 100.0

    def test_full_power_pipeline(self):
        """전체 전력 데이터 파이프라인"""
        dates = pd.date_range('2024-01-01', periods=168, freq='h')

        df = pd.DataFrame({
            '거래일자': [d.strftime('%Y-%m-%d') for d in dates],
            '시간': [(d.hour + 1) for d in dates],
            '전력거래량(MWh)': np.random.uniform(400, 600, 168)
        })

        # 결측치 추가
        df.loc[50, '전력거래량(MWh)'] = np.nan
        df.loc[100, '전력거래량(MWh)'] = np.nan

        # 이상치 추가
        df.loc[75, '전력거래량(MWh)'] = 2000.0

        result, stats = preprocess_power_data(df)

        # 결측치 처리됨
        assert result['power_demand'].isnull().sum() == 0

        # 이상치 처리됨
        assert result['power_demand'].max() < 2000.0


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_dataframe(self):
        """빈 DataFrame"""
        df = pd.DataFrame({'A': []})
        result, stats = interpolate_missing_values(df)

        assert len(result) == 0

    def test_single_row(self):
        """단일 행"""
        df = pd.DataFrame({'A': [1.0]})
        result, stats = interpolate_missing_values(df)

        assert len(result) == 1

    def test_all_missing(self):
        """전체 결측치"""
        df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        result, stats = interpolate_missing_values(df)

        # 전부 NaN이면 보간 불가
        assert result['A'].isnull().all()

    def test_non_numeric_columns_ignored(self):
        """비수치형 컬럼 무시"""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': ['a', 'b', 'c']
        })
        result, stats = interpolate_missing_values(df)

        # 수치형만 처리
        assert 'A' in stats
        assert 'B' not in stats

    def test_large_dataset(self):
        """대용량 데이터"""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'A': np.random.randn(n)
        })
        # 10% 결측치
        mask = np.random.random(n) < 0.1
        df.loc[mask, 'A'] = np.nan

        result, stats = interpolate_missing_values(df)

        assert result['A'].isnull().sum() == 0


# ============================================================
# 데이터 타입 테스트
# ============================================================

class TestDataTypes:
    """데이터 타입 테스트"""

    def test_numeric_preserved(self, sample_df_with_missing):
        """수치형 타입 유지"""
        result, stats = interpolate_missing_values(sample_df_with_missing)

        assert result['A'].dtype in [np.float64, np.int64]

    def test_stats_types(self, sample_df_with_missing):
        """통계 값 타입"""
        result, stats = interpolate_missing_values(sample_df_with_missing)

        for col, col_stats in stats.items():
            assert isinstance(col_stats['missing_before'], int)
            assert isinstance(col_stats['missing_after'], int)
            assert isinstance(col_stats['interpolated'], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
