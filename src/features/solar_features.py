"""
FEAT-004: 태양광 관련 파생 변수 생성

태양광 발전 및 BTM(Behind The Meter) 효과 분석을 위한 특성 엔지니어링 모듈

핵심 가설:
- 제주도의 숨겨진 태양광 발전량(BTM)이 한전 전력 수요에 주요 영향
- 실제 전력 수요 = 표시 수요 + 숨겨진 태양광 발전량

생성되는 특성:
1. solar_estimated: 일사량 기반 예상 발전량 (MWh)
2. cloud_attenuation: 전운량 기반 일사 감쇠 계수 (0~1)
3. is_daylight: 일출/일몰 시간 플래그 (0/1)
4. clear_sky_index: 맑은 하늘 대비 실제 일사량 비율
5. btm_effect: BTM 태양광 추정 효과 (MWh)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime
import math


# ============================================================
# 제주도 지리 정보
# ============================================================

# 제주시 좌표 (기상관측소 기준)
JEJU_LATITUDE = 33.5142  # 북위 33.5142°
JEJU_LONGITUDE = 126.5298  # 동경 126.5298°
JEJU_TIMEZONE = 9  # UTC+9 (KST)

# 태양광 발전 관련 상수
SOLAR_CONSTANT = 1361  # 태양상수 (W/m²)
ATMOSPHERE_TRANSMITTANCE = 0.75  # 대기 투과율 (맑은 날)


# ============================================================
# 일출/일몰 계산 함수
# ============================================================

def calculate_day_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """연중 일수 계산 (1-365/366)"""
    return dates.dayofyear.values


def calculate_solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    """
    태양 적위(Solar Declination) 계산

    태양 적위는 태양이 천구의 적도면에서 얼마나 떨어져 있는지를 나타냄
    북반구 여름에는 양수, 겨울에는 음수

    Args:
        day_of_year: 연중 일수 (1-365)

    Returns:
        np.ndarray: 태양 적위 (라디안)
    """
    # 간략화된 공식 (Spencer, 1971)
    gamma = 2 * np.pi * (day_of_year - 1) / 365
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    return declination


def calculate_hour_angle(hour: np.ndarray, longitude: float = JEJU_LONGITUDE) -> np.ndarray:
    """
    시간각(Hour Angle) 계산

    태양 정오를 기준으로 한 각도. 정오에 0, 오전에 음수, 오후에 양수

    Args:
        hour: 시간 (0-23, KST - 한국표준시)
        longitude: 경도

    Returns:
        np.ndarray: 시간각 (라디안)
    """
    # 태양시 보정 (경도에 따른 시간 보정)
    # 표준 자오선: 135°E (한국표준시 기준)
    # 제주(126.53°E)는 표준 자오선보다 서쪽이므로 태양시가 더 늦음
    # 시간 보정 = (실제 경도 - 표준 자오선) * 4분/도 / 60 = 시간 단위
    standard_meridian = 135.0
    time_correction = (longitude - standard_meridian) * 4 / 60  # 시간 단위

    # 지역 태양시 (Local Solar Time)
    # KST를 그대로 사용하고 경도 보정만 적용
    local_solar_time = hour + time_correction

    # 시간각 (라디안)
    # 정오(12시)에 0, 1시간당 15도 회전
    hour_angle = (local_solar_time - 12) * 15 * np.pi / 180
    return hour_angle


def calculate_solar_elevation(
    day_of_year: np.ndarray,
    hour: np.ndarray,
    latitude: float = JEJU_LATITUDE,
    longitude: float = JEJU_LONGITUDE
) -> np.ndarray:
    """
    태양 고도각(Solar Elevation) 계산

    수평선으로부터 태양까지의 각도. 일출/일몰 시 0°, 정오에 최대

    Args:
        day_of_year: 연중 일수
        hour: 시간 (KST)
        latitude: 위도
        longitude: 경도

    Returns:
        np.ndarray: 태양 고도각 (도)
    """
    lat_rad = latitude * np.pi / 180
    declination = calculate_solar_declination(day_of_year)
    hour_angle = calculate_hour_angle(hour, longitude)

    # 태양 고도각 공식
    sin_elevation = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )

    # -1 ~ 1 범위 제한
    sin_elevation = np.clip(sin_elevation, -1, 1)

    elevation = np.arcsin(sin_elevation) * 180 / np.pi
    return elevation


def is_daylight(
    dates: pd.DatetimeIndex,
    latitude: float = JEJU_LATITUDE,
    longitude: float = JEJU_LONGITUDE,
    min_elevation: float = 0.0
) -> np.ndarray:
    """
    일출/일몰 시간 플래그 계산

    태양 고도각이 임계값 이상이면 주간(1), 아니면 야간(0)

    Args:
        dates: datetime 인덱스
        latitude: 위도
        longitude: 경도
        min_elevation: 최소 태양 고도 (기본 0도 = 수평선)

    Returns:
        np.ndarray: 주간 플래그 (0/1)
    """
    day_of_year = calculate_day_of_year(dates)
    hour = dates.hour.values

    elevation = calculate_solar_elevation(day_of_year, hour, latitude, longitude)

    return (elevation > min_elevation).astype(np.int8)


def calculate_daylight_hours(
    dates: pd.DatetimeIndex,
    latitude: float = JEJU_LATITUDE
) -> np.ndarray:
    """
    일조 가능 시간 계산 (해당 날짜의 총 일조 시간)

    Args:
        dates: datetime 인덱스
        latitude: 위도

    Returns:
        np.ndarray: 일조 가능 시간 (시간)
    """
    day_of_year = calculate_day_of_year(dates)
    lat_rad = latitude * np.pi / 180
    declination = calculate_solar_declination(day_of_year)

    # 일몰 시간각 계산
    cos_hour_angle = -np.tan(lat_rad) * np.tan(declination)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)

    # 일조 시간 (시간)
    daylight_hours = 2 * np.arccos(cos_hour_angle) * 180 / np.pi / 15

    return daylight_hours


# ============================================================
# 일사량 및 발전량 추정 함수
# ============================================================

def calculate_theoretical_irradiance(
    dates: pd.DatetimeIndex,
    latitude: float = JEJU_LATITUDE,
    longitude: float = JEJU_LONGITUDE
) -> np.ndarray:
    """
    이론적 맑은 하늘 일사량 계산 (MJ/m²/hour)

    대기권 밖 태양 복사량에 대기 투과율을 적용한 값

    Args:
        dates: datetime 인덱스
        latitude: 위도
        longitude: 경도

    Returns:
        np.ndarray: 이론적 일사량 (MJ/m²)
    """
    day_of_year = calculate_day_of_year(dates)
    hour = dates.hour.values

    # 태양 고도각
    elevation = calculate_solar_elevation(day_of_year, hour, latitude, longitude)
    elevation_rad = np.maximum(elevation, 0) * np.pi / 180  # 음수는 0으로

    # 대기권 밖 복사량 (일-지구 거리 보정)
    distance_factor = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)

    # 대기질량 (Air Mass) - Kasten & Young (1989)
    # AM = 1 / (sin(elevation) + 0.50572 * (elevation + 6.07995)^-1.6364)
    elevation_deg = np.maximum(elevation, 0.5)  # 0.5도 이하는 0.5로
    air_mass = 1 / (
        np.sin(elevation_deg * np.pi / 180)
        + 0.50572 * (elevation_deg + 6.07995) ** -1.6364
    )
    air_mass = np.where(elevation <= 0, np.inf, air_mass)

    # 대기 투과 후 복사량 (W/m²)
    transmittance = ATMOSPHERE_TRANSMITTANCE ** air_mass
    irradiance_wm2 = SOLAR_CONSTANT * distance_factor * np.sin(elevation_rad) * transmittance

    # W/m² -> MJ/m²/hour (1시간 적산)
    # 1 W = 1 J/s, 1시간 = 3600초, 1 MJ = 10^6 J
    irradiance_mj = irradiance_wm2 * 3600 / 1e6

    # 야간은 0
    irradiance_mj = np.where(elevation <= 0, 0, irradiance_mj)

    return irradiance_mj


def calculate_clear_sky_index(
    actual_irradiance: np.ndarray,
    theoretical_irradiance: np.ndarray,
    min_theoretical: float = 0.01
) -> np.ndarray:
    """
    맑은 하늘 지수(Clear Sky Index) 계산

    실제 일사량 / 이론적 일사량 비율
    1에 가까우면 맑음, 0에 가까우면 흐림

    Args:
        actual_irradiance: 실제 측정 일사량
        theoretical_irradiance: 이론적 맑은 하늘 일사량
        min_theoretical: 최소 이론 일사량 (0 나누기 방지)

    Returns:
        np.ndarray: 맑은 하늘 지수 (0~1+)
    """
    # 이론 일사량이 매우 작은 경우 (야간 등) NaN 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = actual_irradiance / np.maximum(theoretical_irradiance, min_theoretical)

    # 야간이나 이론값이 매우 작은 경우 0으로
    csi = np.where(theoretical_irradiance < min_theoretical, 0, csi)

    # 1 초과 값 클리핑 (측정 오차 등으로 이론값 초과 가능)
    csi = np.clip(csi, 0, 1.5)

    return csi


def calculate_cloud_attenuation(
    cloud_cover: np.ndarray,
    max_cloud: float = 10.0
) -> np.ndarray:
    """
    전운량 기반 일사 감쇠 계수 계산

    전운량이 0이면 감쇠 없음(1.0), 10이면 최대 감쇠(0.0)

    Args:
        cloud_cover: 전운량 (0-10)
        max_cloud: 최대 전운량 값

    Returns:
        np.ndarray: 감쇠 계수 (0~1, 1=감쇠 없음)
    """
    attenuation = 1.0 - (cloud_cover / max_cloud)
    return np.clip(attenuation, 0, 1)


def estimate_solar_generation(
    irradiance: np.ndarray,
    capacity_mw: float = 300.0,
    efficiency: float = 0.15,
    performance_ratio: float = 0.80
) -> np.ndarray:
    """
    일사량 기반 태양광 발전량 추정 (MWh)

    공식: Generation = Irradiance × Area × Efficiency × PR
    간략화: Generation = Irradiance × Capacity × (Efficiency/STC_Irradiance) × PR

    표준 테스트 조건(STC): 1000 W/m² = 3.6 MJ/m²/hour

    Args:
        irradiance: 일사량 (MJ/m²)
        capacity_mw: 설비용량 (MW)
        efficiency: 모듈 효율 (기본 15%)
        performance_ratio: 성능비 (기본 80%)

    Returns:
        np.ndarray: 예상 발전량 (MWh)
    """
    # STC 조건 일사량 (MJ/m²/hour)
    STC_IRRADIANCE = 3.6  # 1000 W/m² × 3600s / 10^6 = 3.6 MJ/m²

    # 발전량 추정
    # irradiance/STC_IRRADIANCE = 현재 일사량이 STC 대비 몇 %인지
    generation = capacity_mw * (irradiance / STC_IRRADIANCE) * performance_ratio

    return np.maximum(generation, 0)


# ============================================================
# BTM (Behind The Meter) 태양광 추정
# ============================================================

def estimate_btm_solar(
    estimated_total: np.ndarray,
    metered_solar: np.ndarray,
    btm_ratio: float = 0.3
) -> np.ndarray:
    """
    BTM(Behind The Meter) 태양광 발전량 추정

    BTM = 총 예상 발전량 - 계측 발전량 (양수 부분만)
    또는 예상 발전량의 일정 비율로 추정

    제주도 BTM 태양광 비율 (2023년 기준 추정):
    - 총 태양광 설비: ~1,000 MW
    - 계측 태양광: ~400 MW
    - BTM 비율: ~60% (가정용, 상업용 자가소비)

    Args:
        estimated_total: 총 예상 발전량
        metered_solar: 계측된 태양광 발전량
        btm_ratio: BTM 비율 (계측 발전량 대비)

    Returns:
        np.ndarray: BTM 발전량 추정치
    """
    # 방법 1: 차이 기반 (계측 데이터가 있을 때)
    # btm = estimated_total - metered_solar

    # 방법 2: 비율 기반 (더 안정적)
    # 계측 발전량의 btm_ratio 배가 BTM으로 추정
    btm = metered_solar * btm_ratio

    return np.maximum(btm, 0)


def calculate_btm_effect(
    df: pd.DataFrame,
    irradiance_col: str = '일사',
    cloud_col: str = '전운량',
    metered_col: str = 'solar_generation_mwh',
    capacity_col: str = 'solar_capacity_mw',
    btm_capacity_ratio: float = 1.5
) -> np.ndarray:
    """
    BTM 태양광 효과 계산 (전력 수요에 미치는 영향)

    가설: 실제 수요 = 표시 수요 + BTM 발전량
    BTM 발전량이 클수록 표시 수요는 감소

    Args:
        df: 입력 DataFrame
        irradiance_col: 일사량 컬럼명
        cloud_col: 전운량 컬럼명
        metered_col: 계측 발전량 컬럼명
        capacity_col: 설비용량 컬럼명
        btm_capacity_ratio: BTM 설비가 계측 설비의 몇 배인지

    Returns:
        np.ndarray: BTM 효과 추정치 (MWh)
    """
    irradiance = df[irradiance_col].values
    cloud_cover = df[cloud_col].values

    # 구름 감쇠 적용
    effective_irradiance = irradiance * calculate_cloud_attenuation(cloud_cover)

    # 계측 설비 용량 (없으면 평균값 사용)
    if capacity_col in df.columns:
        capacity = df[capacity_col].fillna(df[capacity_col].mean()).values
    else:
        capacity = 300.0  # 기본값

    # BTM 설비 용량 추정
    btm_capacity = capacity * btm_capacity_ratio

    # BTM 발전량 추정
    btm_generation = estimate_solar_generation(
        effective_irradiance,
        capacity_mw=btm_capacity,
        performance_ratio=0.75  # BTM은 성능비 약간 낮게
    )

    return btm_generation


# ============================================================
# 통합 함수
# ============================================================

def add_solar_features(
    df: pd.DataFrame,
    irradiance_col: str = '일사',
    sunshine_col: str = '일조',
    cloud_col: str = '전운량',
    metered_gen_col: str = 'solar_generation_mwh',
    metered_cap_col: str = 'solar_capacity_mw',
    include_theoretical: bool = True,
    include_clear_sky_index: bool = True,
    include_cloud_attenuation: bool = True,
    include_daylight: bool = True,
    include_estimated_gen: bool = True,
    include_btm: bool = True,
    btm_capacity_ratio: float = 1.5,
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 태양광 관련 특성을 추가합니다.

    추가되는 컬럼:
    - theoretical_irradiance: 이론적 맑은 하늘 일사량 (MJ/m²)
    - clear_sky_index: 맑은 하늘 지수 (0~1.5)
    - cloud_attenuation: 구름 감쇠 계수 (0~1)
    - is_daylight: 주간 플래그 (0/1)
    - solar_elevation: 태양 고도각 (도)
    - solar_estimated: 예상 발전량 (MWh)
    - btm_effect: BTM 태양광 효과 (MWh)

    Args:
        df: 입력 DataFrame (datetime 인덱스 필요)
        irradiance_col: 일사량 컬럼명
        sunshine_col: 일조 컬럼명
        cloud_col: 전운량 컬럼명
        metered_gen_col: 계측 발전량 컬럼명
        metered_cap_col: 설비용량 컬럼명
        include_*: 각 특성 포함 여부
        btm_capacity_ratio: BTM 설비 비율
        inplace: True면 원본 수정

    Returns:
        pd.DataFrame: 태양광 특성이 추가된 DataFrame

    Raises:
        ValueError: datetime 인덱스가 없는 경우
    """
    if not inplace:
        df = df.copy()

    # datetime 인덱스 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame은 datetime 인덱스가 필요합니다.")

    idx = df.index
    day_of_year = calculate_day_of_year(idx)
    hour = idx.hour.values

    # 태양 고도각 (기본 계산)
    elevation = calculate_solar_elevation(day_of_year, hour)
    df['solar_elevation'] = elevation

    # 주간 플래그
    if include_daylight:
        df['is_daylight'] = (elevation > 0).astype(np.int8)

    # 이론적 일사량
    if include_theoretical:
        theoretical = calculate_theoretical_irradiance(idx)
        df['theoretical_irradiance'] = theoretical

        # 맑은 하늘 지수
        if include_clear_sky_index and irradiance_col in df.columns:
            actual_irr = df[irradiance_col].values
            df['clear_sky_index'] = calculate_clear_sky_index(actual_irr, theoretical)

    # 구름 감쇠 계수
    if include_cloud_attenuation and cloud_col in df.columns:
        df['cloud_attenuation'] = calculate_cloud_attenuation(df[cloud_col].values)

    # 예상 발전량
    if include_estimated_gen and irradiance_col in df.columns:
        irradiance = df[irradiance_col].values

        # 구름 감쇠 적용
        if cloud_col in df.columns:
            attenuation = calculate_cloud_attenuation(df[cloud_col].values)
        else:
            attenuation = 1.0

        # 설비용량
        if metered_cap_col in df.columns:
            capacity = df[metered_cap_col].fillna(300.0).values
        else:
            capacity = 300.0

        # 총 설비 용량 (계측 + BTM)
        total_capacity = capacity * (1 + btm_capacity_ratio) if include_btm else capacity

        # 발전량 추정
        df['solar_estimated'] = estimate_solar_generation(
            irradiance * attenuation,
            capacity_mw=total_capacity if isinstance(total_capacity, (int, float)) else total_capacity.mean()
        )

    # BTM 효과
    if include_btm and irradiance_col in df.columns and cloud_col in df.columns:
        df['btm_effect'] = calculate_btm_effect(
            df,
            irradiance_col=irradiance_col,
            cloud_col=cloud_col,
            metered_col=metered_gen_col,
            capacity_col=metered_cap_col,
            btm_capacity_ratio=btm_capacity_ratio
        )

    return df


def get_solar_feature_names(
    include_theoretical: bool = True,
    include_clear_sky_index: bool = True,
    include_cloud_attenuation: bool = True,
    include_daylight: bool = True,
    include_estimated_gen: bool = True,
    include_btm: bool = True
) -> list:
    """
    생성될 태양광 특성 컬럼명 목록을 반환합니다.

    Args:
        include_*: 각 특성 포함 여부

    Returns:
        list: 컬럼명 목록
    """
    names = ['solar_elevation']  # 기본 포함

    if include_daylight:
        names.append('is_daylight')
    if include_theoretical:
        names.append('theoretical_irradiance')
        if include_clear_sky_index:
            names.append('clear_sky_index')
    if include_cloud_attenuation:
        names.append('cloud_attenuation')
    if include_estimated_gen:
        names.append('solar_estimated')
    if include_btm:
        names.append('btm_effect')

    return names


# ============================================================
# 데모 및 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEAT-004: Solar Features Demo")
    print("=" * 60)

    # 샘플 데이터 생성 (1년, 시간별)
    dates = pd.date_range('2024-01-01', periods=24*365, freq='h')
    np.random.seed(42)

    # 시뮬레이션된 데이터
    df = pd.DataFrame({
        '일사': np.maximum(0, np.random.normal(0.5, 0.3, len(dates))),  # MJ/m²
        '전운량': np.random.randint(0, 11, len(dates)),  # 0-10
        '일조': np.random.uniform(0, 1, len(dates)),  # 0-1
        'solar_capacity_mw': 300.0,
        'solar_generation_mwh': np.maximum(0, np.random.normal(30, 20, len(dates)))
    }, index=dates)

    # 야간 일사량 0으로
    df.loc[df.index.hour.isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]), '일사'] = 0

    print(f"\n[Input] Sample data: {len(df)} hours")
    print(f"Date range: {df.index.min()} ~ {df.index.max()}")

    # 태양광 특성 추가
    df_with_solar = add_solar_features(df)

    print(f"\n[Output] Added columns: {get_solar_feature_names()}")

    # 결과 샘플 출력
    print("\n=== Sample Output (2024-06-21 Summer Solstice) ===")
    summer_solstice = df_with_solar.loc['2024-06-21']
    display_cols = ['solar_elevation', 'is_daylight', 'theoretical_irradiance',
                    'clear_sky_index', 'cloud_attenuation', 'solar_estimated', 'btm_effect']
    print(summer_solstice[display_cols].to_string())

    # 통계
    print("\n=== Statistics ===")
    for col in get_solar_feature_names():
        if col in df_with_solar.columns:
            print(f"{col}: min={df_with_solar[col].min():.3f}, max={df_with_solar[col].max():.3f}, mean={df_with_solar[col].mean():.3f}")

    # 일출/일몰 시간 확인
    print("\n=== Daylight Hours by Month ===")
    monthly_daylight = df_with_solar.groupby(df_with_solar.index.month)['is_daylight'].sum() / 24
    print("Daylight hours per day by month:")
    for month in range(1, 13):
        days_in_month = monthly_daylight.loc[month] / (30 if month != 2 else 29)
        print(f"  Month {month:2d}: {days_in_month:.1f} hours")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
