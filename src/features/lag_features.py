"""
FEAT-005: 지연 변수 및 이동평균 생성

시계열 예측을 위한 지연 변수(Lag Features) 및 이동평균 모듈

생성되는 특성:
1. Lag Features: 과거 시점의 값 (t-1, t-24, t-48, t-168)
2. Moving Averages: 이동평균 (6h, 12h, 24h, 168h)
3. Rolling Statistics: 롤링 표준편차, 최대, 최소
4. Difference Features: 전 시점 대비 변화량

주의사항:
- 데이터 누수(Data Leakage) 방지: 예측 시점 이전 데이터만 사용
- NaN 처리: 시작 부분의 결측치는 forward fill 또는 제거
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict


# ============================================================
# 기본 지연 변수 함수
# ============================================================

def create_lag_feature(
    series: pd.Series,
    lag: int,
    prefix: str = None
) -> pd.Series:
    """
    단일 지연 변수를 생성합니다.

    Args:
        series: 입력 시리즈
        lag: 지연 시점 (양수: 과거, 음수: 미래)
        prefix: 컬럼명 접두사 (None이면 원본 이름 사용)

    Returns:
        pd.Series: 지연된 시리즈

    Examples:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> create_lag_feature(s, 1)  # t-1
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        4    4.0
    """
    lagged = series.shift(lag)

    if prefix is not None:
        lagged.name = f"{prefix}_lag_{lag}"
    elif series.name is not None:
        lagged.name = f"{series.name}_lag_{lag}"
    else:
        lagged.name = f"lag_{lag}"

    return lagged


def create_lag_features(
    series: pd.Series,
    lags: List[int],
    prefix: str = None
) -> pd.DataFrame:
    """
    여러 지연 변수를 한 번에 생성합니다.

    Args:
        series: 입력 시리즈
        lags: 지연 시점 리스트
        prefix: 컬럼명 접두사

    Returns:
        pd.DataFrame: 지연 변수 데이터프레임

    Examples:
        >>> s = pd.Series([1, 2, 3, 4, 5], name='value')
        >>> create_lag_features(s, [1, 2, 3])
           value_lag_1  value_lag_2  value_lag_3
        0          NaN          NaN          NaN
        1          1.0          NaN          NaN
        2          2.0          1.0          NaN
        3          3.0          2.0          1.0
        4          4.0          3.0          2.0
    """
    lag_dict = {}
    for lag in lags:
        lagged = create_lag_feature(series, lag, prefix)
        lag_dict[lagged.name] = lagged

    return pd.DataFrame(lag_dict)


# ============================================================
# 이동평균 함수
# ============================================================

def create_moving_average(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
    prefix: str = None
) -> pd.Series:
    """
    이동평균을 계산합니다.

    Args:
        series: 입력 시리즈
        window: 윈도우 크기 (시간)
        min_periods: 최소 데이터 포인트 수
        prefix: 컬럼명 접두사

    Returns:
        pd.Series: 이동평균 시리즈

    Notes:
        - 현재 시점은 포함하지 않고, 과거 window 시간의 평균 계산
        - 데이터 누수 방지를 위해 shift(1) 적용
    """
    # 현재 시점 제외 (데이터 누수 방지)
    ma = series.shift(1).rolling(window=window, min_periods=min_periods).mean()

    if prefix is not None:
        ma.name = f"{prefix}_ma_{window}h"
    elif series.name is not None:
        ma.name = f"{series.name}_ma_{window}h"
    else:
        ma.name = f"ma_{window}h"

    return ma


def create_moving_averages(
    series: pd.Series,
    windows: List[int],
    min_periods: int = 1,
    prefix: str = None
) -> pd.DataFrame:
    """
    여러 윈도우의 이동평균을 한 번에 생성합니다.

    Args:
        series: 입력 시리즈
        windows: 윈도우 크기 리스트
        min_periods: 최소 데이터 포인트 수
        prefix: 컬럼명 접두사

    Returns:
        pd.DataFrame: 이동평균 데이터프레임
    """
    ma_dict = {}
    for window in windows:
        ma = create_moving_average(series, window, min_periods, prefix)
        ma_dict[ma.name] = ma

    return pd.DataFrame(ma_dict)


def create_exponential_moving_average(
    series: pd.Series,
    span: int,
    prefix: str = None
) -> pd.Series:
    """
    지수이동평균(EMA)을 계산합니다.

    Args:
        series: 입력 시리즈
        span: EMA 스팬 (반감기 관련)
        prefix: 컬럼명 접두사

    Returns:
        pd.Series: EMA 시리즈
    """
    # 현재 시점 제외 (데이터 누수 방지)
    ema = series.shift(1).ewm(span=span, adjust=False).mean()

    if prefix is not None:
        ema.name = f"{prefix}_ema_{span}h"
    elif series.name is not None:
        ema.name = f"{series.name}_ema_{span}h"
    else:
        ema.name = f"ema_{span}h"

    return ema


# ============================================================
# 롤링 통계 함수
# ============================================================

def create_rolling_std(
    series: pd.Series,
    window: int,
    min_periods: int = 2,
    prefix: str = None
) -> pd.Series:
    """
    롤링 표준편차(변동성)를 계산합니다.

    Args:
        series: 입력 시리즈
        window: 윈도우 크기
        min_periods: 최소 데이터 포인트 수
        prefix: 컬럼명 접두사

    Returns:
        pd.Series: 롤링 표준편차 시리즈
    """
    # 현재 시점 제외 (데이터 누수 방지)
    std = series.shift(1).rolling(window=window, min_periods=min_periods).std()

    if prefix is not None:
        std.name = f"{prefix}_std_{window}h"
    elif series.name is not None:
        std.name = f"{series.name}_std_{window}h"
    else:
        std.name = f"std_{window}h"

    return std


def create_rolling_min_max(
    series: pd.Series,
    window: int,
    min_periods: int = 1,
    prefix: str = None
) -> pd.DataFrame:
    """
    롤링 최소/최대값을 계산합니다.

    Args:
        series: 입력 시리즈
        window: 윈도우 크기
        min_periods: 최소 데이터 포인트 수
        prefix: 컬럼명 접두사

    Returns:
        pd.DataFrame: min, max, range 컬럼
    """
    shifted = series.shift(1)
    rolling = shifted.rolling(window=window, min_periods=min_periods)

    base_name = prefix if prefix is not None else (series.name if series.name else "value")

    return pd.DataFrame({
        f"{base_name}_min_{window}h": rolling.min(),
        f"{base_name}_max_{window}h": rolling.max(),
        f"{base_name}_range_{window}h": rolling.max() - rolling.min()
    })


# ============================================================
# 차분 및 변화율 함수
# ============================================================

def create_difference(
    series: pd.Series,
    periods: int = 1,
    prefix: str = None
) -> pd.Series:
    """
    차분(변화량)을 계산합니다.

    Args:
        series: 입력 시리즈
        periods: 차분 간격
        prefix: 컬럼명 접두사

    Returns:
        pd.Series: 차분 시리즈
    """
    diff = series.diff(periods)

    if prefix is not None:
        diff.name = f"{prefix}_diff_{periods}h"
    elif series.name is not None:
        diff.name = f"{series.name}_diff_{periods}h"
    else:
        diff.name = f"diff_{periods}h"

    return diff


def create_pct_change(
    series: pd.Series,
    periods: int = 1,
    prefix: str = None
) -> pd.Series:
    """
    변화율(%)을 계산합니다.

    Args:
        series: 입력 시리즈
        periods: 변화율 간격
        prefix: 컬럼명 접두사

    Returns:
        pd.Series: 변화율 시리즈 (소수점, 예: 0.05 = 5%)
    """
    pct = series.pct_change(periods)

    if prefix is not None:
        pct.name = f"{prefix}_pct_{periods}h"
    elif series.name is not None:
        pct.name = f"{series.name}_pct_{periods}h"
    else:
        pct.name = f"pct_{periods}h"

    return pct


# ============================================================
# 특화된 지연 변수 함수 (전력 수요 예측용)
# ============================================================

def create_demand_lag_features(
    df: pd.DataFrame,
    demand_col: str = 'power_demand',
    lags: List[int] = None,
    include_ma: bool = True,
    include_std: bool = True,
    include_diff: bool = True
) -> pd.DataFrame:
    """
    전력 수요 예측을 위한 지연 변수 세트를 생성합니다.

    기본 지연 시점:
    - t-1: 직전 시간
    - t-24: 어제 같은 시간
    - t-48: 그저께 같은 시간
    - t-168: 지난주 같은 시간 (7일 × 24시간)

    Args:
        df: 입력 DataFrame
        demand_col: 전력 수요 컬럼명
        lags: 지연 시점 리스트 (None이면 기본값)
        include_ma: 이동평균 포함 여부
        include_std: 롤링 표준편차 포함 여부
        include_diff: 차분 포함 여부

    Returns:
        pd.DataFrame: 지연 변수 데이터프레임
    """
    if lags is None:
        lags = [1, 24, 48, 168]  # 1시간, 1일, 2일, 1주

    if demand_col not in df.columns:
        raise ValueError(f"컬럼을 찾을 수 없습니다: {demand_col}")

    demand = df[demand_col]
    result_dfs = []

    # 1. 기본 지연 변수
    lag_df = create_lag_features(demand, lags, prefix='demand')
    result_dfs.append(lag_df)

    # 2. 이동평균
    if include_ma:
        ma_windows = [6, 12, 24, 168]  # 6시간, 12시간, 1일, 1주
        ma_df = create_moving_averages(demand, ma_windows, prefix='demand')
        result_dfs.append(ma_df)

    # 3. 롤링 표준편차 (변동성)
    if include_std:
        std_windows = [24, 168]  # 1일, 1주
        std_dfs = []
        for window in std_windows:
            std_series = create_rolling_std(demand, window, prefix='demand')
            std_dfs.append(std_series)
        result_dfs.append(pd.concat(std_dfs, axis=1))

    # 4. 차분 (변화량)
    if include_diff:
        diff_periods = [1, 24]  # 1시간 전 대비, 24시간 전 대비
        diff_dfs = []
        for period in diff_periods:
            diff_series = create_difference(demand, period, prefix='demand')
            diff_dfs.append(diff_series)
        result_dfs.append(pd.concat(diff_dfs, axis=1))

    return pd.concat(result_dfs, axis=1)


def create_weather_lag_features(
    df: pd.DataFrame,
    temp_col: str = '기온',
    irradiance_col: str = '일사',
    humidity_col: str = '습도',
    include_temp: bool = True,
    include_irradiance: bool = True,
    include_humidity: bool = True,
    ma_windows: List[int] = None
) -> pd.DataFrame:
    """
    기상 변수의 지연/이동평균 특성을 생성합니다.

    Args:
        df: 입력 DataFrame
        temp_col: 기온 컬럼명
        irradiance_col: 일사량 컬럼명
        humidity_col: 습도 컬럼명
        include_temp: 기온 특성 포함 여부
        include_irradiance: 일사량 특성 포함 여부
        include_humidity: 습도 특성 포함 여부
        ma_windows: 이동평균 윈도우 리스트

    Returns:
        pd.DataFrame: 기상 지연 변수 데이터프레임
    """
    if ma_windows is None:
        ma_windows = [6, 12, 24]

    result_dfs = []

    # 기온 특성
    if include_temp and temp_col in df.columns:
        temp = df[temp_col]

        # 이동평균
        temp_ma = create_moving_averages(temp, ma_windows, prefix='temp')
        result_dfs.append(temp_ma)

        # 지연 변수
        temp_lags = create_lag_features(temp, [1, 24], prefix='temp')
        result_dfs.append(temp_lags)

        # 롤링 통계 (일교차 파악)
        temp_minmax = create_rolling_min_max(temp, 24, prefix='temp')
        result_dfs.append(temp_minmax)

    # 일사량 특성
    if include_irradiance and irradiance_col in df.columns:
        irr = df[irradiance_col]

        # 이동평균
        irr_ma = create_moving_averages(irr, ma_windows, prefix='irradiance')
        result_dfs.append(irr_ma)

        # 일간 누적
        irr_sum = irr.shift(1).rolling(window=24, min_periods=1).sum()
        irr_sum.name = 'irradiance_sum_24h'
        result_dfs.append(irr_sum.to_frame())

    # 습도 특성
    if include_humidity and humidity_col in df.columns:
        hum = df[humidity_col]

        # 이동평균
        hum_ma = create_moving_averages(hum, [6, 24], prefix='humidity')
        result_dfs.append(hum_ma)

    if not result_dfs:
        return pd.DataFrame(index=df.index)

    return pd.concat(result_dfs, axis=1)


# ============================================================
# 통합 함수
# ============================================================

def add_lag_features(
    df: pd.DataFrame,
    demand_col: str = 'power_demand',
    temp_col: str = '기온',
    irradiance_col: str = '일사',
    humidity_col: str = '습도',
    demand_lags: List[int] = None,
    ma_windows: List[int] = None,
    include_demand_features: bool = True,
    include_weather_features: bool = True,
    include_diff: bool = True,
    include_std: bool = True,
    fill_na_method: str = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 지연 변수 및 이동평균 특성을 추가합니다.

    추가되는 특성:
    - demand_lag_*: 전력 수요 지연 변수
    - demand_ma_*: 전력 수요 이동평균
    - demand_std_*: 전력 수요 롤링 표준편차
    - demand_diff_*: 전력 수요 차분
    - temp_ma_*, temp_lag_*: 기온 특성
    - irradiance_ma_*, irradiance_sum_*: 일사량 특성
    - humidity_ma_*: 습도 특성

    Args:
        df: 입력 DataFrame (datetime 인덱스 필요)
        demand_col: 전력 수요 컬럼명
        temp_col: 기온 컬럼명
        irradiance_col: 일사량 컬럼명
        humidity_col: 습도 컬럼명
        demand_lags: 전력 수요 지연 시점 리스트
        ma_windows: 이동평균 윈도우 리스트
        include_demand_features: 전력 수요 특성 포함 여부
        include_weather_features: 기상 특성 포함 여부
        include_diff: 차분 특성 포함 여부
        include_std: 롤링 표준편차 포함 여부
        fill_na_method: NaN 처리 방법 ('ffill', 'bfill', None)
        inplace: True면 원본 수정

    Returns:
        pd.DataFrame: 지연 변수가 추가된 DataFrame

    Raises:
        ValueError: datetime 인덱스가 없는 경우
    """
    if not inplace:
        df = df.copy()

    # datetime 인덱스 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame은 datetime 인덱스가 필요합니다.")

    feature_dfs = []

    # 전력 수요 지연 변수
    if include_demand_features and demand_col in df.columns:
        demand_features = create_demand_lag_features(
            df,
            demand_col=demand_col,
            lags=demand_lags,
            include_ma=True,
            include_std=include_std,
            include_diff=include_diff
        )
        feature_dfs.append(demand_features)

    # 기상 지연 변수
    if include_weather_features:
        weather_features = create_weather_lag_features(
            df,
            temp_col=temp_col,
            irradiance_col=irradiance_col,
            humidity_col=humidity_col,
            ma_windows=ma_windows
        )
        feature_dfs.append(weather_features)

    # 특성 병합
    if feature_dfs:
        all_features = pd.concat(feature_dfs, axis=1)

        # NaN 처리
        if fill_na_method == 'ffill':
            all_features = all_features.ffill()
        elif fill_na_method == 'bfill':
            all_features = all_features.bfill()

        # DataFrame에 추가
        for col in all_features.columns:
            df[col] = all_features[col]

    return df


def get_lag_feature_names(
    include_demand: bool = True,
    include_weather: bool = True,
    demand_lags: List[int] = None,
    ma_windows: List[int] = None,
    include_diff: bool = True,
    include_std: bool = True
) -> List[str]:
    """
    생성될 지연 변수 컬럼명 목록을 반환합니다.

    Args:
        include_demand: 전력 수요 특성 포함 여부
        include_weather: 기상 특성 포함 여부
        demand_lags: 전력 수요 지연 시점 리스트
        ma_windows: 이동평균 윈도우 리스트
        include_diff: 차분 특성 포함 여부
        include_std: 롤링 표준편차 포함 여부

    Returns:
        List[str]: 컬럼명 목록
    """
    if demand_lags is None:
        demand_lags = [1, 24, 48, 168]
    if ma_windows is None:
        ma_windows = [6, 12, 24]

    names = []

    if include_demand:
        # 지연 변수
        names.extend([f"demand_lag_{lag}" for lag in demand_lags])

        # 이동평균
        demand_ma_windows = [6, 12, 24, 168]
        names.extend([f"demand_ma_{w}h" for w in demand_ma_windows])

        # 표준편차
        if include_std:
            names.extend(['demand_std_24h', 'demand_std_168h'])

        # 차분
        if include_diff:
            names.extend(['demand_diff_1h', 'demand_diff_24h'])

    if include_weather:
        # 기온
        names.extend([f"temp_ma_{w}h" for w in ma_windows])
        names.extend(['temp_lag_1', 'temp_lag_24'])
        names.extend(['temp_min_24h', 'temp_max_24h', 'temp_range_24h'])

        # 일사량
        names.extend([f"irradiance_ma_{w}h" for w in ma_windows])
        names.append('irradiance_sum_24h')

        # 습도
        names.extend(['humidity_ma_6h', 'humidity_ma_24h'])

    return names


# ============================================================
# 데모 및 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEAT-005: Lag Features Demo")
    print("=" * 60)

    # 샘플 데이터 생성 (1주일, 시간별)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=24*7, freq='h')

    # 시뮬레이션된 데이터 (일간 패턴 포함)
    hour_pattern = np.sin(np.pi * np.arange(24*7) / 12) * 50  # 일간 패턴
    df = pd.DataFrame({
        'power_demand': 500 + hour_pattern + np.random.randn(len(dates)) * 20,
        '기온': 10 + 5 * np.sin(np.pi * np.arange(len(dates)) / 12) + np.random.randn(len(dates)),
        '일사': np.maximum(0, np.sin(np.pi * (np.arange(len(dates)) % 24 - 6) / 12)) * 2,
        '습도': 60 + np.random.randn(len(dates)) * 10
    }, index=dates)

    print(f"\n[Input] Sample data: {len(df)} hours")
    print(f"Date range: {df.index.min()} ~ {df.index.max()}")

    # 지연 변수 추가
    df_with_lag = add_lag_features(df)

    print(f"\n[Output] Added {len(df_with_lag.columns) - len(df.columns)} columns")
    print(f"Feature names: {get_lag_feature_names()}")

    # 결과 샘플 출력
    print("\n=== Sample Output (day 2) ===")
    display_cols = ['power_demand', 'demand_lag_1', 'demand_lag_24',
                    'demand_ma_6h', 'demand_ma_24h', 'demand_diff_1h']
    print(df_with_lag.loc['2024-01-02 08:00:00':'2024-01-02 12:00:00', display_cols].to_string())

    # NaN 통계
    print("\n=== NaN Statistics ===")
    nan_counts = df_with_lag.isnull().sum()
    for col, count in nan_counts.items():
        if count > 0:
            print(f"{col}: {count} NaN ({count/len(df_with_lag)*100:.1f}%)")

    # 상관관계
    print("\n=== Correlation with Power Demand ===")
    lag_cols = [c for c in df_with_lag.columns if 'demand_' in c and c != 'power_demand']
    for col in lag_cols[:6]:  # 상위 6개만
        corr = df_with_lag['power_demand'].corr(df_with_lag[col])
        if not np.isnan(corr):
            print(f"{col}: r = {corr:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
