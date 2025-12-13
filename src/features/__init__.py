"""
Features Module
===============
기상 데이터 파생 변수 생성 모듈

FEAT-001: THI (불쾌지수) 및 습도 - 여름철 냉방 수요 지표
FEAT-002: Wind Chill (체감온도) - 동절기 난방 수요 지표
"""

from .weather_features import (
    # FEAT-001: THI
    calculate_humidity_and_thi,
    calculate_relative_humidity,
    calculate_thi,
    # FEAT-002: Wind Chill
    calculate_wind_chill,
    calculate_wind_chill_from_df,
    convert_ms_to_kmh,
    # Unified
    add_weather_features
)

__all__ = [
    # FEAT-001
    'calculate_humidity_and_thi',
    'calculate_relative_humidity',
    'calculate_thi',
    # FEAT-002
    'calculate_wind_chill',
    'calculate_wind_chill_from_df',
    'convert_ms_to_kmh',
    # Unified
    'add_weather_features'
]
