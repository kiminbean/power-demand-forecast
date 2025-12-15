"""
DATA-003: 데이터셋 병합 (전력 + 기상 + 외부 변수)

병합 대상:
1. 전력-기상 데이터 (hourly): jeju_hourly_cleaned.csv
2. 관광객 데이터 (daily → hourly): jejudo_daily_visitors_2013_2025.csv
3. 전기차 데이터 (daily → hourly): jeju_CAR_daily_2013_2024.csv
4. 태양광 발전 데이터 (hourly, 2018~2024): 한국동서발전_제주_기상관측_태양광발전.csv

처리 전략:
- 일별 데이터는 해당 일의 모든 시간에 동일 값 적용 (broadcast)
- 태양광 데이터는 2018년 이전 데이터에 NaN 유지
- 모든 병합은 datetime 기준 left join
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import json


def load_cleaned_data(data_path: Path) -> pd.DataFrame:
    """
    전처리된 전력-기상 데이터 로드

    Args:
        data_path: jeju_hourly_cleaned.csv 경로

    Returns:
        pd.DataFrame: datetime 인덱스 데이터프레임
    """
    df = pd.read_csv(data_path, encoding='utf-8-sig', index_col=0, parse_dates=True)
    df.index.name = 'datetime'
    print(f"[Load] Cleaned data: {df.shape} ({df.index.min()} ~ {df.index.max()})")
    return df


def load_visitors_data(data_path: Path) -> pd.DataFrame:
    """
    관광객 데이터 로드 및 정리

    Args:
        data_path: 관광객 데이터 경로

    Returns:
        pd.DataFrame: date, visitors 컬럼
    """
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['날짜']).dt.date
    df = df.rename(columns={'일별_입도객수': 'visitors'})
    df = df[['date', 'visitors']].dropna()
    print(f"[Load] Visitors data: {len(df)} days ({df['date'].min()} ~ {df['date'].max()})")
    return df


def load_ev_data(data_path: Path) -> pd.DataFrame:
    """
    전기차 데이터 로드 및 정리

    Args:
        data_path: 전기차 데이터 경로

    Returns:
        pd.DataFrame: date, ev_cumulative, ev_daily_new 컬럼
    """
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.rename(columns={
        'Cumulative EV Count': 'ev_cumulative',
        'ev_daily_new': 'ev_daily_new'
    })
    df = df[['date', 'ev_cumulative', 'ev_daily_new']].dropna()
    print(f"[Load] EV data: {len(df)} days ({df['date'].min()} ~ {df['date'].max()})")
    return df


def load_solar_data(data_path: Path) -> pd.DataFrame:
    """
    태양광 발전 데이터 로드 및 정리

    Args:
        data_path: 태양광 데이터 경로

    Returns:
        pd.DataFrame: datetime 인덱스, solar_capacity, solar_generation 컬럼
    """
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['일시'])
    df = df.rename(columns={
        '태양광 설비용량(MW)': 'solar_capacity_mw',
        '태양광 발전량(MWh)': 'solar_generation_mwh'
    })
    df = df[['datetime', 'solar_capacity_mw', 'solar_generation_mwh']]
    df = df.set_index('datetime')
    print(f"[Load] Solar data: {len(df)} hours ({df.index.min()} ~ {df.index.max()})")
    return df


def expand_daily_to_hourly(
    daily_df: pd.DataFrame,
    date_col: str,
    hourly_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    일별 데이터를 시간별로 확장 (broadcast)

    Args:
        daily_df: 일별 데이터프레임
        date_col: 날짜 컬럼명
        hourly_index: 시간별 인덱스

    Returns:
        pd.DataFrame: 시간별로 확장된 데이터프레임
    """
    df = daily_df.copy()

    # 날짜 컬럼 정리
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # 중복 제거 (첫 번째 값 유지)
    df = df.drop_duplicates(subset=[date_col], keep='first')

    # 시간별 인덱스에서 날짜 추출
    hourly_dates = pd.Series(hourly_index.date, index=hourly_index, name='date')

    # 일별 데이터를 딕셔너리로 변환
    daily_dict = df.set_index(date_col).to_dict('index')

    # 시간별 데이터 생성
    result = pd.DataFrame(index=hourly_index)
    result['date'] = hourly_dates

    for col in df.columns:
        if col != date_col:
            result[col] = result['date'].map(
                lambda d, c=col: daily_dict.get(d, {}).get(c, np.nan)
            )

    result = result.drop(columns=['date'])
    return result


def merge_all_datasets(
    cleaned_df: pd.DataFrame,
    visitors_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    solar_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    모든 데이터셋 병합

    Args:
        cleaned_df: 전처리된 전력-기상 데이터
        visitors_df: 관광객 데이터
        ev_df: 전기차 데이터
        solar_df: 태양광 데이터

    Returns:
        Tuple[pd.DataFrame, Dict]: 병합된 데이터프레임, 병합 통계
    """
    stats = {'steps': []}
    result_df = cleaned_df.copy()
    hourly_index = result_df.index

    print("\n[Merge] Starting dataset merge...")

    # 1. 관광객 데이터 병합 (daily → hourly)
    print("  - Expanding visitors data to hourly...")
    visitors_hourly = expand_daily_to_hourly(visitors_df, 'date', hourly_index)
    result_df = result_df.join(visitors_hourly, how='left')
    visitors_coverage = result_df['visitors'].notna().sum() / len(result_df) * 100
    stats['steps'].append({
        'step': 'visitors_merge',
        'coverage_pct': round(visitors_coverage, 2),
        'missing_count': int(result_df['visitors'].isna().sum())
    })
    print(f"    Coverage: {visitors_coverage:.2f}%")

    # 2. 전기차 데이터 병합 (daily → hourly)
    print("  - Expanding EV data to hourly...")
    ev_hourly = expand_daily_to_hourly(ev_df, 'date', hourly_index)
    result_df = result_df.join(ev_hourly, how='left')
    ev_coverage = result_df['ev_cumulative'].notna().sum() / len(result_df) * 100
    stats['steps'].append({
        'step': 'ev_merge',
        'coverage_pct': round(ev_coverage, 2),
        'missing_count': int(result_df['ev_cumulative'].isna().sum())
    })
    print(f"    Coverage: {ev_coverage:.2f}%")

    # 3. 태양광 데이터 병합 (hourly)
    print("  - Merging solar data...")
    result_df = result_df.join(solar_df, how='left')
    solar_coverage = result_df['solar_generation_mwh'].notna().sum() / len(result_df) * 100
    stats['steps'].append({
        'step': 'solar_merge',
        'coverage_pct': round(solar_coverage, 2),
        'missing_count': int(result_df['solar_generation_mwh'].isna().sum()),
        'note': 'Solar data available from 2018-01 to 2024-05'
    })
    print(f"    Coverage: {solar_coverage:.2f}% (2018-01 ~ 2024-05 only)")

    # 4. 결측치 처리 전략
    print("\n  - Handling missing values...")

    # 관광객: 결측치는 전후 값으로 보간
    if result_df['visitors'].isna().sum() > 0:
        result_df['visitors'] = result_df['visitors'].interpolate(method='linear')
        result_df['visitors'] = result_df['visitors'].ffill().bfill()

    # 전기차: 결측치는 전후 값으로 보간
    if result_df['ev_cumulative'].isna().sum() > 0:
        result_df['ev_cumulative'] = result_df['ev_cumulative'].interpolate(method='linear')
        result_df['ev_cumulative'] = result_df['ev_cumulative'].ffill().bfill()

    if result_df['ev_daily_new'].isna().sum() > 0:
        result_df['ev_daily_new'] = result_df['ev_daily_new'].fillna(0)

    # 태양광: 2018년 이전은 NaN 유지 (데이터 없음), 2018년 이후 결측은 0으로 처리
    solar_start = pd.Timestamp('2018-01-01')
    solar_mask = result_df.index >= solar_start
    result_df.loc[solar_mask & result_df['solar_generation_mwh'].isna(), 'solar_generation_mwh'] = 0
    result_df.loc[solar_mask & result_df['solar_capacity_mw'].isna(), 'solar_capacity_mw'] = \
        result_df.loc[solar_mask, 'solar_capacity_mw'].ffill().bfill()

    stats['final_shape'] = result_df.shape
    stats['final_missing'] = {
        col: int(val) for col, val in result_df.isnull().sum().items() if val > 0
    }

    return result_df, stats


def run_merge_pipeline(
    base_path: Path,
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    데이터 병합 파이프라인 실행

    Args:
        base_path: 프로젝트 루트 경로
        output_path: 출력 파일 경로 (None이면 기본 경로)

    Returns:
        Tuple[pd.DataFrame, Dict]: 병합된 데이터프레임, 통계
    """
    print("=" * 60)
    print("DATA-003: Dataset Merge Pipeline")
    print("=" * 60)

    # 경로 설정
    cleaned_path = base_path / "data/processed/jeju_hourly_cleaned.csv"
    visitors_path = base_path / "data/raw/jejudo_daily_visitors_2013_2025.csv"
    ev_path = base_path / "data/raw/jeju_CAR_daily_2013_2024.csv"
    solar_path = base_path / "data/raw/한국동서발전_제주_기상관측_태양광발전.csv"

    if output_path is None:
        output_path = base_path / "data/processed/jeju_hourly_merged.csv"

    # 1. 데이터 로드
    print("\n[Step 1] Loading datasets...")
    cleaned_df = load_cleaned_data(cleaned_path)
    visitors_df = load_visitors_data(visitors_path)
    ev_df = load_ev_data(ev_path)
    solar_df = load_solar_data(solar_path)

    # 2. 데이터 병합
    print("\n[Step 2] Merging datasets...")
    merged_df, stats = merge_all_datasets(cleaned_df, visitors_df, ev_df, solar_df)

    # 3. 결과 저장
    print("\n[Step 3] Saving merged data...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, encoding='utf-8-sig')
    print(f"  - Saved to: {output_path}")

    # 4. 통계 리포트 저장
    report_path = base_path / "data/reports/merge_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'generated_at': datetime.now().isoformat(),
        'task_id': 'DATA-003',
        'description': '데이터셋 병합 (전력 + 기상 + 외부 변수)',
        'input_files': {
            'cleaned': str(cleaned_path),
            'visitors': str(visitors_path),
            'ev': str(ev_path),
            'solar': str(solar_path)
        },
        'output_file': str(output_path),
        'statistics': stats
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  - Report saved to: {report_path}")

    # 5. 요약 출력
    print("\n" + "=" * 60)
    print("Merge Complete!")
    print("=" * 60)
    print(f"Final shape: {merged_df.shape}")
    print(f"Date range: {merged_df.index.min()} ~ {merged_df.index.max()}")
    print(f"\nNew columns added:")
    new_cols = ['visitors', 'ev_cumulative', 'ev_daily_new', 'solar_capacity_mw', 'solar_generation_mwh']
    for col in new_cols:
        if col in merged_df.columns:
            missing = merged_df[col].isna().sum()
            print(f"  - {col}: {missing} missing ({missing/len(merged_df)*100:.2f}%)")

    return merged_df, report


def main():
    """메인 실행"""
    base_path = Path(__file__).parent.parent.parent
    merged_df, report = run_merge_pipeline(base_path)

    # 추가 요약
    print("\n=== Column Summary ===")
    print(f"Total columns: {len(merged_df.columns)}")
    print("\nColumn list:")
    for i, col in enumerate(merged_df.columns, 1):
        dtype = merged_df[col].dtype
        missing = merged_df[col].isna().sum()
        print(f"  {i:2d}. {col}: {dtype} (missing: {missing})")

    return merged_df


if __name__ == "__main__":
    main()
