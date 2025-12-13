"""
Weather Feature Engineering Module
==================================
기상 데이터에서 파생 변수를 생성하는 모듈

Features:
- humidity: 상대습도 (August-Roche-Magnus 공식으로 이슬점에서 역산)
- THI: 불쾌지수 (Temperature-Humidity Index)

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
from typing import Optional


# August-Roche-Magnus 상수
MAGNUS_A = 17.625
MAGNUS_B = 243.04


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
        
    Example:
        >>> df = pd.DataFrame({
        ...     'temp_mean': [25.0, 30.0, 15.0],
        ...     'dewpoint_mean': [20.0, 25.0, 10.0]
        ... })
        >>> result = calculate_humidity_and_thi(df)
        >>> 'humidity' in result.columns
        True
        >>> 'THI' in result.columns
        True
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


def add_weather_features(
    df: pd.DataFrame,
    include_thi: bool = True,
    include_apparent_temp: bool = False
) -> pd.DataFrame:
    """
    기상 관련 파생 변수를 일괄 추가하는 편의 함수입니다.
    
    Args:
        df: 입력 DataFrame
        include_thi: THI 추가 여부
        include_apparent_temp: 체감온도 추가 여부 (미구현)
        
    Returns:
        파생 변수가 추가된 DataFrame
    """
    result = df.copy()
    
    if include_thi:
        result = calculate_humidity_and_thi(result)
    
    # TODO: 체감온도 추가 (향후 구현)
    if include_apparent_temp:
        raise NotImplementedError("체감온도 계산은 아직 구현되지 않았습니다.")
    
    return result


if __name__ == "__main__":
    # 간단한 사용 예시
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'temp_mean': [5.0, 15.0, 25.0, 30.0, 35.0],
        'dewpoint_mean': [0.0, 10.0, 20.0, 28.0, 30.0]
    })
    
    result = calculate_humidity_and_thi(sample_data)
    print("Sample Output:")
    print(result[['date', 'temp_mean', 'dewpoint_mean', 'humidity', 'THI']])
