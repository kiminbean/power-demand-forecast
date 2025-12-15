"""
Features Module
===============
기상 데이터 파생 변수 생성 모듈

FEAT-001: THI (불쾌지수) 및 습도 - 여름철 냉방 수요 지표
FEAT-002: HDD/CDD (난방/냉방 도일) - 에너지 수요 누적 지표
FEAT-003: 시간 특성 (주기적 인코딩) - 시계열 패턴 학습
FEAT-004: 태양광 특성 - BTM 태양광 가설 검증
FEAT-006: 외부 데이터 (인구, 전기차) - 실제 전력 소비자 반영
"""

from .weather_features import (
    # FEAT-001: THI
    calculate_humidity_and_thi,
    calculate_relative_humidity,
    calculate_thi,
    # FEAT-002: HDD/CDD
    calculate_hdd,
    calculate_cdd,
    calculate_hdd_cdd,
    # Wind Chill
    calculate_wind_chill,
    calculate_wind_chill_from_df,
    convert_ms_to_kmh,
    # Unified
    add_weather_features
)

from .time_features import (
    # FEAT-003: 시간 특성
    add_time_features,
    get_time_feature_names,
    # 주기적 인코딩
    cyclical_encode,
    encode_hour,
    encode_dayofweek,
    encode_month,
    encode_dayofyear,
    # 이진 플래그
    is_weekend,
    is_holiday,
    is_workday,
    # 공휴일
    get_korean_holidays,
    get_all_korean_holidays,
)

from .solar_features import (
    # FEAT-004: 태양광 특성
    add_solar_features,
    get_solar_feature_names,
    # 태양 위치 계산
    calculate_solar_elevation,
    calculate_solar_declination,
    is_daylight,
    calculate_daylight_hours,
    # 일사량 및 발전량
    calculate_theoretical_irradiance,
    calculate_clear_sky_index,
    calculate_cloud_attenuation,
    estimate_solar_generation,
    # BTM 효과
    estimate_btm_solar,
    calculate_btm_effect,
)

from .lag_features import (
    # FEAT-005: 지연 변수
    add_lag_features,
    get_lag_feature_names,
    # 기본 지연 변수
    create_lag_feature,
    create_lag_features,
    # 이동평균
    create_moving_average,
    create_moving_averages,
    create_exponential_moving_average,
    # 롤링 통계
    create_rolling_std,
    create_rolling_min_max,
    # 차분
    create_difference,
    create_pct_change,
    # 특화 함수
    create_demand_lag_features,
    create_weather_lag_features,
)

from .external_features import (
    # FEAT-006: 외부 데이터 (인구, 전기차)
    add_external_features,
    load_population_data,
    load_ev_data,
    add_population_features,
    add_ev_features,
    get_external_feature_names,
    # 피처 그룹
    POPULATION_FEATURES,
    POPULATION_DERIVED,
    EV_FEATURES,
    EV_DERIVED,
    EXTERNAL_FEATURES_ALL,
)

__all__ = [
    # FEAT-001: THI
    'calculate_humidity_and_thi',
    'calculate_relative_humidity',
    'calculate_thi',
    # FEAT-002: HDD/CDD
    'calculate_hdd',
    'calculate_cdd',
    'calculate_hdd_cdd',
    # Wind Chill
    'calculate_wind_chill',
    'calculate_wind_chill_from_df',
    'convert_ms_to_kmh',
    # Unified weather
    'add_weather_features',
    # FEAT-003: 시간 특성
    'add_time_features',
    'get_time_feature_names',
    'cyclical_encode',
    'encode_hour',
    'encode_dayofweek',
    'encode_month',
    'encode_dayofyear',
    'is_weekend',
    'is_holiday',
    'is_workday',
    'get_korean_holidays',
    'get_all_korean_holidays',
    # FEAT-004: 태양광 특성
    'add_solar_features',
    'get_solar_feature_names',
    'calculate_solar_elevation',
    'calculate_solar_declination',
    'is_daylight',
    'calculate_daylight_hours',
    'calculate_theoretical_irradiance',
    'calculate_clear_sky_index',
    'calculate_cloud_attenuation',
    'estimate_solar_generation',
    'estimate_btm_solar',
    'calculate_btm_effect',
    # FEAT-005: 지연 변수
    'add_lag_features',
    'get_lag_feature_names',
    'create_lag_feature',
    'create_lag_features',
    'create_moving_average',
    'create_moving_averages',
    'create_exponential_moving_average',
    'create_rolling_std',
    'create_rolling_min_max',
    'create_difference',
    'create_pct_change',
    'create_demand_lag_features',
    'create_weather_lag_features',
    # FEAT-006: 외부 데이터
    'add_external_features',
    'load_population_data',
    'load_ev_data',
    'add_population_features',
    'add_ev_features',
    'get_external_feature_names',
    'POPULATION_FEATURES',
    'POPULATION_DERIVED',
    'EV_FEATURES',
    'EV_DERIVED',
    'EXTERNAL_FEATURES_ALL',
]
