"""
External Features Module
========================

외부 데이터 피처 엔지니어링:
- 제주도 실제 인구 (거주자 + 관광객)
- 전기차 누적 등록대수

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings


def load_population_data(
    file_path: str,
    resample_to_hourly: bool = True
) -> pd.DataFrame:
    """
    제주도 인구 데이터 로드 및 전처리

    Parameters
    ----------
    file_path : str
        인구 데이터 파일 경로
    resample_to_hourly : bool
        시간별 데이터로 리샘플링 여부

    Returns
    -------
    pd.DataFrame
        전처리된 인구 데이터
    """
    df = pd.read_csv(file_path, parse_dates=['date'])

    # 주요 컬럼 선택
    columns_to_keep = [
        'date',
        'base_population',      # 기본 거주 인구
        'total_arrival',        # 일일 입도객
        'total_departure',      # 일일 출도객
        'net_flow',             # 순 유입 (입도 - 출도)
        'tourist_stock',        # 관광객 재고 (누적)
        'estimated_population'  # 추정 실제 인구
    ]

    available_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[available_cols].copy()

    # 중복 날짜 처리: 마지막 값 사용
    df = df.drop_duplicates(subset='date', keep='last')
    df.set_index('date', inplace=True)
    df = df.sort_index()

    if resample_to_hourly:
        # 일별 → 시간별 리샘플링 (forward fill)
        df = df.resample('h').ffill()

    return df


def load_ev_data(
    file_path: str,
    resample_to_hourly: bool = True
) -> pd.DataFrame:
    """
    전기차 누적대수 데이터 로드 및 전처리

    Parameters
    ----------
    file_path : str
        전기차 데이터 파일 경로
    resample_to_hourly : bool
        시간별 데이터로 리샘플링 여부

    Returns
    -------
    pd.DataFrame
        전처리된 전기차 데이터
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)

    # 컬럼명 정리
    df.columns = df.columns.str.strip()

    # 컬럼명 표준화
    rename_map = {
        'Cumulative EV Count': 'ev_cumulative',
        'ev_daily_new': 'ev_daily_new'
    }
    df.rename(columns=rename_map, inplace=True)

    if resample_to_hourly:
        # 일별 → 시간별 리샘플링 (forward fill)
        df = df.resample('h').ffill()

    return df


def add_population_features(
    df: pd.DataFrame,
    population_df: pd.DataFrame,
    include_derived: bool = True
) -> pd.DataFrame:
    """
    인구 관련 피처 추가

    Parameters
    ----------
    df : pd.DataFrame
        메인 데이터프레임 (datetime index)
    population_df : pd.DataFrame
        인구 데이터프레임
    include_derived : bool
        파생 피처 포함 여부

    Returns
    -------
    pd.DataFrame
        인구 피처가 추가된 데이터프레임
    """
    df = df.copy()

    # 인덱스 정렬 후 병합
    population_df = population_df.reindex(df.index, method='ffill')

    # 기본 피처 추가
    for col in population_df.columns:
        if col not in df.columns:
            df[col] = population_df[col]

    if include_derived and 'estimated_population' in df.columns:
        # 파생 피처 생성

        # 1. 인구 변화율 (일별)
        df['population_change'] = df['estimated_population'].diff(24)  # 24시간 전 대비
        df['population_change_pct'] = df['population_change'] / df['estimated_population'].shift(24) * 100

        # 2. 관광객 비율
        if 'tourist_stock' in df.columns and 'base_population' in df.columns:
            df['tourist_ratio'] = df['tourist_stock'] / df['base_population'] * 100

        # 3. 순유입 이동평균 (7일)
        if 'net_flow' in df.columns:
            df['net_flow_ma7d'] = df['net_flow'].rolling(window=24*7, min_periods=1).mean()

        # 4. 인구 대비 전력수요 (per capita)
        if 'power_demand' in df.columns:
            df['demand_per_capita'] = df['power_demand'] / df['estimated_population'] * 1000  # kW per 1000 people

    return df


def add_ev_features(
    df: pd.DataFrame,
    ev_df: pd.DataFrame,
    include_derived: bool = True
) -> pd.DataFrame:
    """
    전기차 관련 피처 추가

    Parameters
    ----------
    df : pd.DataFrame
        메인 데이터프레임 (datetime index)
    ev_df : pd.DataFrame
        전기차 데이터프레임
    include_derived : bool
        파생 피처 포함 여부

    Returns
    -------
    pd.DataFrame
        전기차 피처가 추가된 데이터프레임
    """
    df = df.copy()

    # 인덱스 정렬 후 병합
    ev_df = ev_df.reindex(df.index, method='ffill')

    # 기본 피처 추가
    for col in ev_df.columns:
        if col not in df.columns:
            df[col] = ev_df[col]

    if include_derived and 'ev_cumulative' in df.columns:
        # 파생 피처 생성

        # 1. 전기차 증가율 (월별)
        df['ev_growth_monthly'] = df['ev_cumulative'].diff(24*30)  # 30일 전 대비
        df['ev_growth_rate'] = df['ev_growth_monthly'] / df['ev_cumulative'].shift(24*30) * 100

        # 2. 전기차 보급률 (인구 대비)
        if 'estimated_population' in df.columns:
            df['ev_penetration'] = df['ev_cumulative'] / df['estimated_population'] * 1000  # per 1000 people

        # 3. 전기차 로그 변환 (스케일 조정)
        df['ev_cumulative_log'] = np.log1p(df['ev_cumulative'])

        # 4. 전기차 정규화 (0-1 스케일)
        ev_min = df['ev_cumulative'].min()
        ev_max = df['ev_cumulative'].max()
        if ev_max > ev_min:
            df['ev_cumulative_norm'] = (df['ev_cumulative'] - ev_min) / (ev_max - ev_min)

    return df


def add_external_features(
    df: pd.DataFrame,
    population_path: Optional[str] = None,
    ev_path: Optional[str] = None,
    include_population: bool = True,
    include_ev: bool = True,
    include_derived: bool = True
) -> pd.DataFrame:
    """
    모든 외부 피처 추가 (통합 함수)

    Parameters
    ----------
    df : pd.DataFrame
        메인 데이터프레임 (datetime index)
    population_path : str, optional
        인구 데이터 파일 경로
    ev_path : str, optional
        전기차 데이터 파일 경로
    include_population : bool
        인구 피처 포함 여부
    include_ev : bool
        전기차 피처 포함 여부
    include_derived : bool
        파생 피처 포함 여부

    Returns
    -------
    pd.DataFrame
        외부 피처가 추가된 데이터프레임
    """
    df = df.copy()

    # 기본 경로 설정
    project_root = Path(__file__).parent.parent.parent

    if include_population:
        if population_path is None:
            population_path = project_root / 'data' / 'processed' / 'jeju_daily_population_2013_2024_v2.csv'

        if Path(population_path).exists():
            population_df = load_population_data(str(population_path))
            df = add_population_features(df, population_df, include_derived)
        else:
            warnings.warn(f"Population data not found: {population_path}")

    if include_ev:
        if ev_path is None:
            ev_path = project_root / 'data' / 'raw' / 'jeju_CAR_daily_2013_2024.csv'

        if Path(ev_path).exists():
            ev_df = load_ev_data(str(ev_path))
            df = add_ev_features(df, ev_df, include_derived)
        else:
            warnings.warn(f"EV data not found: {ev_path}")

    return df


def get_external_feature_names(
    include_population: bool = True,
    include_ev: bool = True,
    include_derived: bool = True
) -> List[str]:
    """
    외부 피처 이름 목록 반환

    Returns
    -------
    List[str]
        피처 이름 목록
    """
    features = []

    if include_population:
        features.extend([
            'estimated_population',
            'tourist_stock',
            'net_flow'
        ])

        if include_derived:
            features.extend([
                'population_change',
                'population_change_pct',
                'tourist_ratio',
                'net_flow_ma7d',
                'demand_per_capita'
            ])

    if include_ev:
        features.extend([
            'ev_cumulative'
        ])

        if include_derived:
            features.extend([
                'ev_growth_monthly',
                'ev_growth_rate',
                'ev_penetration',
                'ev_cumulative_log',
                'ev_cumulative_norm'
            ])

    return features


# 간편 사용을 위한 피처 그룹 정의
POPULATION_FEATURES = [
    'estimated_population',
    'tourist_stock',
    'net_flow'
]

POPULATION_DERIVED = [
    'population_change',
    'tourist_ratio',
    'net_flow_ma7d'
]

EV_FEATURES = [
    'ev_cumulative'
]

EV_DERIVED = [
    'ev_growth_rate',
    'ev_penetration',
    'ev_cumulative_log'
]

EXTERNAL_FEATURES_ALL = POPULATION_FEATURES + POPULATION_DERIVED + EV_FEATURES + EV_DERIVED
