"""
DATA-002: 결측치 및 이상치 전처리 (선형 보간법)

전처리 파이프라인:
1. 결측치 처리: pandas 선형 보간법 (interpolate)
2. 이상치 처리: IQR 기반 클리핑 또는 보간 대체
3. 단위 변환: 필요시 kWh → MWh

Notes:
- 판다스 보간법 사용 필수
- 임의 데이터 가공 금지
- 모든 변환은 추적 가능해야 함

Reference: JPD_RNN_Weather 논문 - 선형 보간법 적용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


def interpolate_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'linear',
    limit: Optional[int] = None,
    limit_direction: str = 'both'
) -> Tuple[pd.DataFrame, Dict]:
    """
    결측치 선형 보간 처리

    Args:
        df: 입력 데이터프레임
        columns: 보간할 컬럼 목록 (None이면 모든 수치형 컬럼)
        method: 보간 방법 ('linear', 'time', 'index', 'pad', etc.)
        limit: 연속 보간 최대 횟수
        limit_direction: 보간 방향 ('forward', 'backward', 'both')

    Returns:
        Tuple[pd.DataFrame, Dict]: 보간된 데이터프레임, 보간 통계
    """
    df_result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in columns:
        if col not in df_result.columns:
            continue

        missing_before = df_result[col].isnull().sum()

        if missing_before > 0:
            # pandas 선형 보간법 적용
            df_result[col] = df_result[col].interpolate(
                method=method,
                limit=limit,
                limit_direction=limit_direction
            )

            # 경계값 처리 (시작/끝 NaN)
            df_result[col] = df_result[col].ffill().bfill()

            missing_after = df_result[col].isnull().sum()

            stats[col] = {
                'missing_before': int(missing_before),
                'missing_after': int(missing_after),
                'interpolated': int(missing_before - missing_after),
                'method': method
            }

    return df_result, stats


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'clip',
    k: float = 1.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    이상치 처리 (IQR 기반)

    Args:
        df: 입력 데이터프레임
        columns: 처리할 컬럼 목록
        method: 처리 방법 ('clip': 경계값으로 대체, 'interpolate': NaN 후 보간)
        k: IQR 배수 (기본 1.5)

    Returns:
        Tuple[pd.DataFrame, Dict]: 처리된 데이터프레임, 이상치 통계
    """
    df_result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in columns:
        if col not in df_result.columns:
            continue

        series = df_result[col].dropna()
        if len(series) == 0:
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        # 이상치 개수
        outliers_mask = (df_result[col] < lower) | (df_result[col] > upper)
        outlier_count = outliers_mask.sum()

        if outlier_count > 0:
            if method == 'clip':
                # 경계값으로 클리핑
                df_result[col] = df_result[col].clip(lower=lower, upper=upper)
            elif method == 'interpolate':
                # NaN으로 대체 후 보간
                df_result.loc[outliers_mask, col] = np.nan
                df_result[col] = df_result[col].interpolate(method='linear')
                df_result[col] = df_result[col].ffill().bfill()

            stats[col] = {
                'outlier_count': int(outlier_count),
                'outlier_pct': float(outlier_count / len(df_result) * 100),
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'method': method
            }

    return df_result, stats


def preprocess_weather_data(
    weather_df: pd.DataFrame,
    interpolation_config: Optional[Dict] = None,
    outlier_config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    기상 데이터 전처리

    처리 순서:
    1. 특수 컬럼 처리 (강수량: 0 대체, 일조/일사: 야간 0 대체)
    2. 이상치 처리 (IQR 클리핑)
    3. 결측치 보간 (선형 보간)

    Args:
        weather_df: 기상 데이터프레임
        interpolation_config: 보간 설정
        outlier_config: 이상치 처리 설정

    Returns:
        Tuple[pd.DataFrame, Dict]: 전처리된 데이터프레임, 전처리 통계
    """
    df = weather_df.copy()
    stats = {'steps': []}

    # datetime 컬럼 확인/생성
    if 'datetime' not in df.columns and '일시' in df.columns:
        df['datetime'] = pd.to_datetime(df['일시'])

    if 'datetime' in df.columns:
        df = df.set_index('datetime')

    # Step 1: 특수 컬럼 처리
    special_cols = {}

    # 강수량: NaN → 0 (비가 오지 않은 시간)
    if '강수량' in df.columns:
        precip_na = df['강수량'].isnull().sum()
        df['강수량'] = df['강수량'].fillna(0)
        special_cols['강수량'] = {
            'treatment': 'fill_zero',
            'filled_count': int(precip_na)
        }

    # 일조/일사: 야간(일몰~일출)은 0으로 처리
    for col in ['일조', '일사']:
        if col in df.columns:
            # 시간 추출
            if hasattr(df.index, 'hour'):
                hour = df.index.hour
                # 야간 시간대 (19:00 ~ 05:00) NaN을 0으로
                night_mask = (hour >= 19) | (hour <= 5)
                night_na_mask = df[col].isnull() & night_mask
                night_na_count = night_na_mask.sum()
                df.loc[night_na_mask, col] = 0
                special_cols[col] = {
                    'treatment': 'night_fill_zero',
                    'night_filled': int(night_na_count)
                }

    stats['steps'].append({
        'step': 'special_columns',
        'details': special_cols
    })

    # Step 2: 이상치 처리
    outlier_cols = ['기온', '습도', '풍속', '지면온도', 'm005Te', 'm01Te', 'm02Te', 'm03Te']
    outlier_cols = [c for c in outlier_cols if c in df.columns]

    config = outlier_config or {'method': 'clip', 'k': 1.5}
    df, outlier_stats = handle_outliers(df, columns=outlier_cols, **config)

    stats['steps'].append({
        'step': 'outlier_handling',
        'config': config,
        'details': outlier_stats
    })

    # Step 3: 결측치 보간
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    config = interpolation_config or {'method': 'linear', 'limit_direction': 'both'}
    df, interp_stats = interpolate_missing_values(df, columns=numeric_cols, **config)

    stats['steps'].append({
        'step': 'interpolation',
        'config': config,
        'details': interp_stats
    })

    # 최종 결측치 확인
    final_missing = df.isnull().sum()
    stats['final_missing'] = {col: int(val) for col, val in final_missing.items() if val > 0}

    return df.reset_index(), stats


def preprocess_power_data(
    power_df: pd.DataFrame,
    outlier_config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    전력 데이터 전처리

    Args:
        power_df: 전력 데이터프레임
        outlier_config: 이상치 처리 설정

    Returns:
        Tuple[pd.DataFrame, Dict]: 전처리된 데이터프레임, 전처리 통계
    """
    df = power_df.copy()
    stats = {'steps': []}

    # datetime 컬럼 생성
    if 'datetime' not in df.columns:
        df['hour'] = df['시간'] - 1  # 1-24 → 0-23
        df['datetime'] = pd.to_datetime(df['거래일자']) + pd.to_timedelta(df['hour'], unit='h')

    # 컬럼명 정리
    if '전력거래량(MWh)' in df.columns:
        df = df.rename(columns={'전력거래량(MWh)': 'power_demand'})

    # 필요한 컬럼만 선택
    df = df[['datetime', 'power_demand']].copy()
    df = df.set_index('datetime').sort_index()

    # 이상치 처리
    config = outlier_config or {'method': 'clip', 'k': 1.5}
    df, outlier_stats = handle_outliers(df, columns=['power_demand'], **config)

    stats['steps'].append({
        'step': 'outlier_handling',
        'config': config,
        'details': outlier_stats
    })

    # 결측치 확인 (있다면 보간)
    missing_count = df['power_demand'].isnull().sum()
    if missing_count > 0:
        df['power_demand'] = df['power_demand'].interpolate(method='linear')
        df['power_demand'] = df['power_demand'].ffill().bfill()
        stats['steps'].append({
            'step': 'interpolation',
            'interpolated_count': int(missing_count)
        })

    return df.reset_index(), stats


def create_preprocessing_report(
    weather_stats: Dict,
    power_stats: Dict,
    output_path: Path
) -> Dict:
    """
    전처리 리포트 생성

    Args:
        weather_stats: 기상 데이터 전처리 통계
        power_stats: 전력 데이터 전처리 통계
        output_path: 리포트 저장 경로

    Returns:
        Dict: 전체 전처리 리포트
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'task_id': 'DATA-002',
        'description': '결측치 및 이상치 전처리 (선형 보간법)',
        'weather_preprocessing': weather_stats,
        'power_preprocessing': power_stats,
        'summary': {
            'weather_steps': len(weather_stats.get('steps', [])),
            'power_steps': len(power_stats.get('steps', [])),
            'weather_final_missing': weather_stats.get('final_missing', {}),
            'power_final_missing': power_stats.get('final_missing', {})
        }
    }

    # JSON 리포트 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Saved] Preprocessing report: {output_path}")

    return report


def run_preprocessing_pipeline(
    weather_path: Path,
    power_path: Path,
    output_dir: Path,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    전처리 파이프라인 실행

    Args:
        weather_path: 기상 데이터 경로
        power_path: 전력 데이터 경로
        output_dir: 출력 디렉토리
        config: 전처리 설정

    Returns:
        Tuple[pd.DataFrame, Dict]: 전처리된 병합 데이터, 리포트
    """
    config = config or {}

    print("=" * 60)
    print("DATA-002: Preprocessing Pipeline")
    print("=" * 60)

    # 1. 기상 데이터 로드 및 전처리
    print("\n[Step 1] Loading and preprocessing weather data...")
    weather_df = pd.read_csv(weather_path, encoding='utf-8-sig')
    weather_processed, weather_stats = preprocess_weather_data(
        weather_df,
        interpolation_config=config.get('interpolation'),
        outlier_config=config.get('outlier')
    )
    print(f"  - Weather data shape: {weather_processed.shape}")

    # 2. 전력 데이터 로드 및 전처리
    print("\n[Step 2] Loading and preprocessing power data...")
    power_df = pd.read_csv(power_path, encoding='utf-8-sig')
    power_processed, power_stats = preprocess_power_data(
        power_df,
        outlier_config=config.get('outlier')
    )
    print(f"  - Power data shape: {power_processed.shape}")

    # 3. 데이터 병합
    print("\n[Step 3] Merging datasets...")
    merged_df = pd.merge(
        power_processed,
        weather_processed,
        on='datetime',
        how='inner'
    )
    merged_df = merged_df.set_index('datetime').sort_index()
    print(f"  - Merged data shape: {merged_df.shape}")

    # 4. 최종 품질 검사
    print("\n[Step 4] Final quality check...")
    final_missing = merged_df.isnull().sum()
    missing_cols = final_missing[final_missing > 0]
    if len(missing_cols) > 0:
        print(f"  - Warning: Remaining missing values: {missing_cols.to_dict()}")
    else:
        print("  - All missing values handled successfully!")

    # 5. 결과 저장
    print("\n[Step 5] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전처리된 데이터 저장
    output_data_path = output_dir / 'jeju_hourly_cleaned.csv'
    merged_df.to_csv(output_data_path, encoding='utf-8-sig')
    print(f"  - Saved cleaned data: {output_data_path}")

    # 리포트 저장
    report_path = output_dir.parent / 'reports' / 'preprocessing_report.json'
    report = create_preprocessing_report(weather_stats, power_stats, report_path)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)

    return merged_df, report


def main():
    """메인 실행"""
    base_path = Path(__file__).parent.parent.parent

    # 경로 설정
    weather_path = base_path / "data/processed/jeju_weather_hourly_merged.csv"
    power_path = base_path / "data/raw/jeju_hourly_power_2013_2024.csv"
    output_dir = base_path / "data/processed"

    # 설정
    config = {
        'interpolation': {
            'method': 'linear',
            'limit_direction': 'both'
        },
        'outlier': {
            'method': 'clip',
            'k': 1.5
        }
    }

    # 파이프라인 실행
    merged_df, report = run_preprocessing_pipeline(
        weather_path=weather_path,
        power_path=power_path,
        output_dir=output_dir,
        config=config
    )

    # 요약 출력
    print("\n=== Data Summary ===")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Date range: {merged_df.index.min()} ~ {merged_df.index.max()}")
    print(f"\nColumn dtypes:\n{merged_df.dtypes}")

    return merged_df, report


if __name__ == "__main__":
    main()
