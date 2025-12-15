"""
Weather Feature Engineering Module
==================================
기상 데이터에서 파생 변수를 생성하는 모듈

Features:
- humidity: 상대습도 (August-Roche-Magnus 공식으로 이슬점에서 역산)
- THI: 불쾌지수 (Temperature-Humidity Index) - 여름철 냉방 수요 지표
- wind_chill: 체감온도 (JAG/Siple 공식) - 동절기 난방 수요 지표
- HDD: 난방도일 (Heating Degree Days) - max(18-T, 0)
- CDD: 냉방도일 (Cooling Degree Days) - max(T-18, 0)

Tasks:
- FEAT-001: THI 및 상대습도 계산
- FEAT-002: HDD/CDD (난방/냉방 도일) 생성

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# August-Roche-Magnus 상수 (습도 계산용)
# Reference: Alduchov & Eskridge (1996)
MAGNUS_A = 17.625
MAGNUS_B = 243.04

# Wind Chill 유효 범위 상수
# Reference: JAG/Siple Wind Chill Temperature Index
WIND_CHILL_TEMP_THRESHOLD = 10.0  # °C 이하에서만 유효
WIND_CHILL_WIND_THRESHOLD = 4.8   # km/h 이상에서만 유효

# 단위 변환 상수
MS_TO_KMH = 3.6  # 1 m/s = 3.6 km/h


# ============================================================================
# Humidity & THI Functions (FEAT-001)
# ============================================================================

def calculate_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """
    온도에서 포화 수증기압을 계산합니다.
    
    August-Roche-Magnus 근사식:
    Es = exp((a * T) / (b + T))
    
    Args:
        temperature: 섭씨 온도 배열
        
    Returns:
        포화 수증기압 (상대값)
    """
    return np.exp((MAGNUS_A * temperature) / (MAGNUS_B + temperature))


def calculate_relative_humidity(
    temp_mean: np.ndarray, 
    dewpoint_mean: np.ndarray
) -> np.ndarray:
    """
    기온과 이슬점에서 상대습도를 계산합니다.
    
    August-Roche-Magnus 공식:
    RH = 100 * exp((a * Td) / (b + Td)) / exp((a * T) / (b + T))
    
    Args:
        temp_mean: 평균 기온 (섭씨)
        dewpoint_mean: 평균 이슬점 온도 (섭씨)
        
    Returns:
        상대습도 (0-100%)
        
    Note:
        - 이슬점이 기온보다 높은 비정상 데이터는 RH=100으로 클리핑
        - 물리적 한계 적용: 0 <= RH <= 100
    """
    # 포화 수증기압 계산
    e_saturation = calculate_saturation_vapor_pressure(temp_mean)
    e_actual = calculate_saturation_vapor_pressure(dewpoint_mean)
    
    # 상대습도 계산 (0으로 나누기 방지)
    with np.errstate(divide='ignore', invalid='ignore'):
        humidity = 100 * (e_actual / e_saturation)
    
    # 물리적 한계 클리핑 (0-100%)
    humidity = np.clip(humidity, 0, 100)
    
    return humidity


def calculate_thi(
    temp_mean: np.ndarray, 
    humidity_ratio: np.ndarray
) -> np.ndarray:
    """
    불쾌지수(Temperature-Humidity Index)를 계산합니다.
    
    THI 공식:
    THI = 1.8 * T - 0.55 * (1 - RH) * (1.8 * T - 26) + 32
    
    Args:
        temp_mean: 평균 기온 (섭씨)
        humidity_ratio: 상대습도 비율 (0.0 ~ 1.0)
        
    Returns:
        불쾌지수 (THI)
        
    Note:
        THI 해석:
        - < 68: 쾌적
        - 68-74: 일부 불쾌감
        - 75-79: 반 이상 불쾌
        - 80+: 대부분 불쾌
    """
    t_fahrenheit_component = 1.8 * temp_mean
    
    thi = (
        t_fahrenheit_component 
        - 0.55 * (1 - humidity_ratio) * (t_fahrenheit_component - 26) 
        + 32
    )
    
    return thi


def calculate_humidity_and_thi(
    df: pd.DataFrame,
    temp_col: str = 'temp_mean',
    dewpoint_col: str = 'dewpoint_mean',
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 상대습도(humidity)와 불쾌지수(THI) 컬럼을 추가합니다.
    
    Args:
        df: 입력 DataFrame (temp_mean, dewpoint_mean 컬럼 필수)
        temp_col: 기온 컬럼명 (기본값: 'temp_mean')
        dewpoint_col: 이슬점 컬럼명 (기본값: 'dewpoint_mean')
        inplace: True면 원본 수정, False면 복사본 반환
        
    Returns:
        humidity, THI 컬럼이 추가된 DataFrame
        
    Raises:
        ValueError: 필수 컬럼이 없는 경우
    """
    # 필수 컬럼 검증
    required_cols = [temp_col, dewpoint_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    # 복사본 생성 (inplace=False인 경우)
    result = df if inplace else df.copy()
    
    # numpy 배열로 변환 (벡터 연산 최적화)
    temp = result[temp_col].values
    dewpoint = result[dewpoint_col].values
    
    # 1단계: 상대습도 계산 (%)
    humidity = calculate_relative_humidity(temp, dewpoint)
    result['humidity'] = humidity
    
    # 2단계: THI 계산 (humidity를 비율로 변환: % -> ratio)
    humidity_ratio = humidity / 100.0
    thi = calculate_thi(temp, humidity_ratio)
    result['THI'] = thi
    
    return result


# ============================================================================
# Wind Chill Functions (FEAT-002)
# ============================================================================

def convert_ms_to_kmh(wind_speed_ms: np.ndarray) -> np.ndarray:
    """
    풍속을 m/s에서 km/h로 변환합니다.
    
    Args:
        wind_speed_ms: 풍속 (m/s)
        
    Returns:
        풍속 (km/h)
    """
    return wind_speed_ms * MS_TO_KMH


def calculate_wind_chill(
    temp_celsius: np.ndarray,
    wind_speed_kmh: np.ndarray,
    apply_validity_mask: bool = False
) -> np.ndarray:
    """
    체감온도(Wind Chill Temperature)를 계산합니다.
    
    JAG/Siple 공식 (기상청/Environment Canada 표준):
    Twc = 13.12 + 0.6215*Ta - 11.37*(V^0.16) + 0.3965*Ta*(V^0.16)
    
    Args:
        temp_celsius: 기온 (섭씨)
        wind_speed_kmh: 풍속 (km/h) - 반드시 km/h 단위!
        apply_validity_mask: True면 유효 범위 외 값을 원래 기온으로 대체
        
    Returns:
        체감온도 (섭씨)
        
    Note:
        - 공식 유효 범위: T <= 10°C, V >= 4.8 km/h
        - 유효 범위 외에서는 체감온도 ≈ 실제 기온
        - apply_validity_mask=True: 범위 외 값을 기온으로 대체
        - apply_validity_mask=False: 전체 구간에 공식 적용 (ML Feature용)
        
    Reference:
        Environment Canada Wind Chill Index
        https://www.canada.ca/en/environment-climate-change/services/weather-health/wind-chill-cold-weather.html
    """
    # 풍속의 0.16 거듭제곱 (재사용을 위해 미리 계산)
    # 풍속이 0일 때 0^0.16 = 0 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        v_power = np.where(wind_speed_kmh > 0, np.power(wind_speed_kmh, 0.16), 0)
    
    # JAG/Siple Wind Chill 공식
    wind_chill = (
        13.12 
        + 0.6215 * temp_celsius 
        - 11.37 * v_power 
        + 0.3965 * temp_celsius * v_power
    )
    
    if apply_validity_mask:
        # 유효 범위 마스크: T <= 10°C AND V >= 4.8 km/h
        valid_mask = (temp_celsius <= WIND_CHILL_TEMP_THRESHOLD) & \
                     (wind_speed_kmh >= WIND_CHILL_WIND_THRESHOLD)
        # 유효 범위 외에서는 실제 기온 사용
        wind_chill = np.where(valid_mask, wind_chill, temp_celsius)
    
    return wind_chill


def calculate_wind_chill_from_df(
    df: pd.DataFrame,
    temp_col: str = 'temp_mean',
    wind_col: str = 'wind_speed_mean',
    wind_unit: str = 'ms',
    apply_validity_mask: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 체감온도(wind_chill) 컬럼을 추가합니다.
    
    Args:
        df: 입력 DataFrame
        temp_col: 기온 컬럼명 (기본값: 'temp_mean')
        wind_col: 풍속 컬럼명 (기본값: 'wind_speed_mean')
        wind_unit: 풍속 단위 ('ms' = m/s, 'kmh' = km/h)
        apply_validity_mask: True면 유효 범위 외 값을 기온으로 대체
        inplace: True면 원본 수정, False면 복사본 반환
        
    Returns:
        wind_chill 컬럼이 추가된 DataFrame
        
    Raises:
        ValueError: 필수 컬럼이 없거나 잘못된 단위인 경우
        
    Example:
        >>> df = pd.DataFrame({
        ...     'temp_mean': [5.0, -5.0, 0.0],
        ...     'wind_speed_mean': [5.0, 10.0, 3.0]  # m/s
        ... })
        >>> result = calculate_wind_chill_from_df(df)
        >>> 'wind_chill' in result.columns
        True
    """
    # 필수 컬럼 검증
    required_cols = [temp_col, wind_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    # 단위 검증
    valid_units = ['ms', 'kmh']
    if wind_unit not in valid_units:
        raise ValueError(f"잘못된 풍속 단위입니다. 유효한 값: {valid_units}")
    
    # 복사본 생성
    result = df if inplace else df.copy()
    
    # numpy 배열로 변환
    temp = result[temp_col].values.astype(float)
    wind = result[wind_col].values.astype(float)
    
    # 단위 변환 (m/s -> km/h)
    if wind_unit == 'ms':
        wind_kmh = convert_ms_to_kmh(wind)
    else:
        wind_kmh = wind
    
    # 체감온도 계산
    wind_chill = calculate_wind_chill(temp, wind_kmh, apply_validity_mask)
    
    # NaN 처리: 입력에 NaN이 있으면 출력도 NaN
    nan_mask = np.isnan(temp) | np.isnan(wind)
    wind_chill = np.where(nan_mask, np.nan, wind_chill)
    
    result['wind_chill'] = wind_chill
    
    return result


# ============================================================================
# HDD/CDD Functions (FEAT-002)
# ============================================================================

# 기준 온도 상수
BASE_TEMPERATURE = 18.0  # 난방/냉방 기준온도 (°C)


def calculate_hdd(
    temperature: np.ndarray,
    base_temp: float = BASE_TEMPERATURE
) -> np.ndarray:
    """
    난방도일(Heating Degree Days)을 계산합니다.

    HDD = max(기준온도 - 기온, 0)

    Args:
        temperature: 기온 배열 (섭씨)
        base_temp: 기준온도 (기본값: 18°C)

    Returns:
        난방도일 (HDD)

    Note:
        - HDD는 난방 수요의 대리 지표
        - 기온이 기준온도보다 낮을수록 HDD 증가 → 난방 수요 증가
        - 시간별 데이터의 경우 일별로 합산하여 사용 권장
    """
    return np.maximum(base_temp - temperature, 0)


def calculate_cdd(
    temperature: np.ndarray,
    base_temp: float = BASE_TEMPERATURE
) -> np.ndarray:
    """
    냉방도일(Cooling Degree Days)을 계산합니다.

    CDD = max(기온 - 기준온도, 0)

    Args:
        temperature: 기온 배열 (섭씨)
        base_temp: 기준온도 (기본값: 18°C)

    Returns:
        냉방도일 (CDD)

    Note:
        - CDD는 냉방 수요의 대리 지표
        - 기온이 기준온도보다 높을수록 CDD 증가 → 냉방 수요 증가
        - 시간별 데이터의 경우 일별로 합산하여 사용 권장
    """
    return np.maximum(temperature - base_temp, 0)


def calculate_cumulative_degree_days(
    degree_days: np.ndarray,
    reset_period: str = 'yearly'
) -> np.ndarray:
    """
    누적 도일(Cumulative Degree Days)을 계산합니다.

    Args:
        degree_days: 도일 배열 (HDD 또는 CDD)
        reset_period: 리셋 주기 ('yearly', 'monthly', 'none')

    Returns:
        누적 도일

    Note:
        - 'none': 전체 기간 누적
        - 실제 사용시 datetime index와 함께 groupby 필요
    """
    if reset_period == 'none':
        return np.cumsum(degree_days)
    else:
        # DataFrame에서 처리해야 하므로 단순 누적만 반환
        return np.cumsum(degree_days)


def calculate_hdd_cdd(
    df: pd.DataFrame,
    temp_col: str = 'temp_mean',
    base_temp: float = BASE_TEMPERATURE,
    include_cumulative: bool = True,
    cumulative_period: str = 'yearly',
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 HDD/CDD 및 누적 도일 컬럼을 추가합니다.

    Args:
        df: 입력 DataFrame (datetime index 권장)
        temp_col: 기온 컬럼명 (기본값: 'temp_mean')
        base_temp: 기준온도 (기본값: 18°C)
        include_cumulative: 연간 누적 도일 추가 여부
        cumulative_period: 누적 리셋 주기 ('yearly', 'monthly')
        inplace: True면 원본 수정, False면 복사본 반환

    Returns:
        HDD, CDD (및 누적) 컬럼이 추가된 DataFrame

    Raises:
        ValueError: 필수 컬럼이 없는 경우

    Example:
        >>> df = pd.DataFrame({
        ...     'temp_mean': [30.0, 15.0, 5.0, -5.0],
        ... }, index=pd.date_range('2024-01-01', periods=4))
        >>> result = calculate_hdd_cdd(df)
        >>> result[['HDD', 'CDD']].values
        array([[ 0., 12.],
               [ 3.,  0.],
               [13.,  0.],
               [23.,  0.]])
    """
    # 필수 컬럼 검증
    if temp_col not in df.columns:
        raise ValueError(f"필수 컬럼이 없습니다: {temp_col}")

    # 복사본 생성
    result = df if inplace else df.copy()

    # numpy 배열로 변환
    temp = result[temp_col].values.astype(float)

    # HDD/CDD 계산
    hdd = calculate_hdd(temp, base_temp)
    cdd = calculate_cdd(temp, base_temp)

    # NaN 처리
    nan_mask = np.isnan(temp)
    hdd = np.where(nan_mask, np.nan, hdd)
    cdd = np.where(nan_mask, np.nan, cdd)

    result['HDD'] = hdd
    result['CDD'] = cdd

    # 누적 도일 계산
    if include_cumulative and hasattr(result.index, 'year'):
        if cumulative_period == 'yearly':
            # 연간 누적 (매년 1월 1일 리셋)
            result['HDD_cumsum'] = result.groupby(result.index.year)['HDD'].cumsum()
            result['CDD_cumsum'] = result.groupby(result.index.year)['CDD'].cumsum()
        elif cumulative_period == 'monthly':
            # 월간 누적 (매월 1일 리셋)
            year_month = result.index.to_period('M')
            result['HDD_cumsum'] = result.groupby(year_month)['HDD'].cumsum()
            result['CDD_cumsum'] = result.groupby(year_month)['CDD'].cumsum()
    elif include_cumulative:
        # datetime index가 없으면 단순 누적
        result['HDD_cumsum'] = result['HDD'].cumsum()
        result['CDD_cumsum'] = result['CDD'].cumsum()

    return result


# ============================================================================
# Unified Feature Functions
# ============================================================================

def add_weather_features(
    df: pd.DataFrame,
    include_thi: bool = True,
    include_wind_chill: bool = True,
    include_hdd_cdd: bool = True,
    temp_col: str = 'temp_mean',
    dewpoint_col: str = 'dewpoint_mean',
    wind_col: str = 'wind_speed_mean',
    wind_unit: str = 'ms',
    base_temp: float = BASE_TEMPERATURE,
    include_cumulative: bool = True
) -> pd.DataFrame:
    """
    기상 관련 파생 변수를 일괄 추가하는 편의 함수입니다.

    Args:
        df: 입력 DataFrame
        include_thi: THI(불쾌지수) 및 humidity 추가 여부
        include_wind_chill: wind_chill(체감온도) 추가 여부
        include_hdd_cdd: HDD/CDD(난방/냉방 도일) 추가 여부
        temp_col: 기온 컬럼명
        dewpoint_col: 이슬점 컬럼명 (THI 계산용)
        wind_col: 풍속 컬럼명 (wind_chill 계산용)
        wind_unit: 풍속 단위 ('ms' = m/s, 'kmh' = km/h)
        base_temp: HDD/CDD 기준온도 (기본값: 18°C)
        include_cumulative: 누적 HDD/CDD 추가 여부

    Returns:
        파생 변수가 추가된 DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'temp_mean': [25.0, 5.0],
        ...     'dewpoint_mean': [20.0, 0.0],
        ...     'wind_speed_mean': [2.0, 8.0]
        ... }, index=pd.date_range('2024-01-01', periods=2))
        >>> result = add_weather_features(df)
        >>> 'HDD' in result.columns and 'CDD' in result.columns
        True
    """
    result = df.copy()

    # THI 및 humidity 추가 (여름철 냉방 수요 지표)
    if include_thi:
        if dewpoint_col in result.columns:
            result = calculate_humidity_and_thi(
                result,
                temp_col=temp_col,
                dewpoint_col=dewpoint_col
            )
        else:
            # dewpoint가 없으면 건너뜀
            import warnings
            warnings.warn(
                f"'{dewpoint_col}' 컬럼이 없어 THI를 계산할 수 없습니다.",
                UserWarning
            )

    # Wind Chill 추가 (동절기 난방 수요 지표)
    if include_wind_chill:
        if wind_col in result.columns:
            result = calculate_wind_chill_from_df(
                result,
                temp_col=temp_col,
                wind_col=wind_col,
                wind_unit=wind_unit,
                apply_validity_mask=False  # ML용: 전체 구간 계산
            )
        else:
            import warnings
            warnings.warn(
                f"'{wind_col}' 컬럼이 없어 wind_chill을 계산할 수 없습니다.",
                UserWarning
            )

    # HDD/CDD 추가 (난방/냉방 도일)
    if include_hdd_cdd:
        if temp_col in result.columns:
            result = calculate_hdd_cdd(
                result,
                temp_col=temp_col,
                base_temp=base_temp,
                include_cumulative=include_cumulative
            )
        else:
            import warnings
            warnings.warn(
                f"'{temp_col}' 컬럼이 없어 HDD/CDD를 계산할 수 없습니다.",
                UserWarning
            )

    return result


# ============================================================================
# Main (Demo)
# ============================================================================

if __name__ == "__main__":
    # 샘플 데이터: 여름/겨울 시나리오
    sample_data = pd.DataFrame({
        'temp_mean': [30.0, 25.0, 15.0, 5.0, 0.0, -5.0],
        'dewpoint_mean': [25.0, 20.0, 10.0, 0.0, -5.0, -10.0],
        'wind_speed_mean': [2.0, 3.0, 4.0, 5.0, 8.0, 10.0]  # m/s
    }, index=pd.date_range('2024-01-01', periods=6, name='datetime'))

    result = add_weather_features(sample_data)

    print("=" * 70)
    print("Weather Features Demo (FEAT-001 + FEAT-002)")
    print("=" * 70)

    print("\n[Input]")
    print(sample_data)

    print("\n[Output - THI & Wind Chill (FEAT-001)]")
    print(result[['temp_mean', 'humidity', 'THI', 'wind_chill']].round(2))

    print("\n[Output - HDD & CDD (FEAT-002)]")
    print(result[['temp_mean', 'HDD', 'CDD', 'HDD_cumsum', 'CDD_cumsum']].round(2))

    print("\n[해석]")
    print("- THI > 75: 여름철 냉방 수요 증가 예상")
    print("- wind_chill < 0: 동절기 난방 수요 증가 예상")
    print("- HDD > 0: 난방 필요 (기온 < 18°C)")
    print("- CDD > 0: 냉방 필요 (기온 > 18°C)")
    print(f"- 누적 HDD: {result['HDD_cumsum'].iloc[-1]:.1f}")
    print(f"- 누적 CDD: {result['CDD_cumsum'].iloc[-1]:.1f}")
