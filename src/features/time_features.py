"""
FEAT-003: 시간 특성 생성 (주기적 인코딩)

시간 관련 특성 엔지니어링 모듈
- 주기적 인코딩: sin/cos 변환으로 시간의 순환적 특성 표현
- 이진 플래그: 주말, 공휴일 여부

주기적 인코딩을 사용하는 이유:
- 시간은 순환적 특성을 가짐 (23시 → 0시, 12월 → 1월)
- 단순 정수 인코딩은 이 순환성을 표현하지 못함
- sin/cos 쌍을 사용하면 유사한 시간대가 유사한 벡터를 가짐
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
from datetime import date


# ============================================================
# 한국 공휴일 정의 (2013-2025)
# ============================================================

def get_korean_holidays(year: int) -> List[date]:
    """
    특정 연도의 한국 공휴일 목록을 반환합니다.

    고정 공휴일:
    - 1월 1일: 신정
    - 3월 1일: 삼일절
    - 5월 5일: 어린이날
    - 6월 6일: 현충일
    - 8월 15일: 광복절
    - 10월 3일: 개천절
    - 10월 9일: 한글날
    - 12월 25일: 성탄절

    음력 공휴일 (연도별 양력 날짜):
    - 설날 (음력 1월 1일 전후 3일)
    - 석가탄신일 (음력 4월 8일)
    - 추석 (음력 8월 15일 전후 3일)

    Args:
        year: 연도

    Returns:
        List[date]: 공휴일 날짜 목록
    """
    holidays = []

    # 고정 공휴일
    fixed_holidays = [
        (1, 1),   # 신정
        (3, 1),   # 삼일절
        (5, 5),   # 어린이날
        (6, 6),   # 현충일
        (8, 15),  # 광복절
        (10, 3),  # 개천절
        (10, 9),  # 한글날
        (12, 25), # 성탄절
    ]

    for month, day in fixed_holidays:
        holidays.append(date(year, month, day))

    # 음력 공휴일 (양력 변환 - 주요 연도별 데이터)
    # 설날, 석가탄신일, 추석
    lunar_holidays = {
        2013: [(2, 9), (2, 10), (2, 11), (5, 17), (9, 18), (9, 19), (9, 20)],
        2014: [(1, 30), (1, 31), (2, 1), (5, 6), (9, 7), (9, 8), (9, 9)],
        2015: [(2, 18), (2, 19), (2, 20), (5, 25), (9, 26), (9, 27), (9, 28)],
        2016: [(2, 7), (2, 8), (2, 9), (5, 14), (9, 14), (9, 15), (9, 16)],
        2017: [(1, 27), (1, 28), (1, 29), (5, 3), (10, 3), (10, 4), (10, 5)],
        2018: [(2, 15), (2, 16), (2, 17), (5, 22), (9, 23), (9, 24), (9, 25)],
        2019: [(2, 4), (2, 5), (2, 6), (5, 12), (9, 12), (9, 13), (9, 14)],
        2020: [(1, 24), (1, 25), (1, 26), (4, 30), (9, 30), (10, 1), (10, 2)],
        2021: [(2, 11), (2, 12), (2, 13), (5, 19), (9, 20), (9, 21), (9, 22)],
        2022: [(1, 31), (2, 1), (2, 2), (5, 8), (9, 9), (9, 10), (9, 11)],
        2023: [(1, 21), (1, 22), (1, 23), (5, 27), (9, 28), (9, 29), (9, 30)],
        2024: [(2, 9), (2, 10), (2, 11), (5, 15), (9, 16), (9, 17), (9, 18)],
        2025: [(1, 28), (1, 29), (1, 30), (5, 5), (10, 5), (10, 6), (10, 7)],
    }

    if year in lunar_holidays:
        for month, day in lunar_holidays[year]:
            holidays.append(date(year, month, day))

    return sorted(set(holidays))


def get_all_korean_holidays(start_year: int = 2013, end_year: int = 2025) -> set:
    """
    여러 연도의 한국 공휴일을 set으로 반환합니다.

    Args:
        start_year: 시작 연도
        end_year: 종료 연도

    Returns:
        set: 모든 공휴일 날짜 set
    """
    all_holidays = set()
    for year in range(start_year, end_year + 1):
        all_holidays.update(get_korean_holidays(year))
    return all_holidays


# ============================================================
# 주기적 인코딩 함수
# ============================================================

def cyclical_encode(
    values: np.ndarray,
    period: int
) -> tuple:
    """
    값을 주기적으로 인코딩합니다 (sin/cos 변환).

    수식:
    - sin_encoded = sin(2π × value / period)
    - cos_encoded = cos(2π × value / period)

    Args:
        values: 인코딩할 값 배열 (0부터 시작)
        period: 주기 (예: 시간=24, 요일=7, 월=12)

    Returns:
        tuple: (sin_encoded, cos_encoded)

    Examples:
        >>> hour = np.array([0, 6, 12, 18])
        >>> sin_h, cos_h = cyclical_encode(hour, 24)
        >>> # 0시: (0, 1), 6시: (1, 0), 12시: (0, -1), 18시: (-1, 0)
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def encode_hour(hour: np.ndarray) -> tuple:
    """
    시간을 주기적으로 인코딩합니다.

    - 주기: 24시간
    - 입력 범위: 0-23
    - 유사한 시간대가 유사한 벡터를 가짐 (23시 ≈ 0시)

    Args:
        hour: 시간 배열 (0-23)

    Returns:
        tuple: (hour_sin, hour_cos)
    """
    return cyclical_encode(hour, 24)


def encode_dayofweek(dayofweek: np.ndarray) -> tuple:
    """
    요일을 주기적으로 인코딩합니다.

    - 주기: 7일
    - 입력 범위: 0(월요일)-6(일요일)
    - 주말(토,일)과 주중이 구분됨

    Args:
        dayofweek: 요일 배열 (0=월요일, 6=일요일)

    Returns:
        tuple: (dayofweek_sin, dayofweek_cos)
    """
    return cyclical_encode(dayofweek, 7)


def encode_month(month: np.ndarray) -> tuple:
    """
    월을 주기적으로 인코딩합니다.

    - 주기: 12개월
    - 입력 범위: 1-12 (내부적으로 0-11로 변환)
    - 12월 ≈ 1월 (연말과 연초가 유사)

    Args:
        month: 월 배열 (1-12)

    Returns:
        tuple: (month_sin, month_cos)
    """
    # 1-12를 0-11로 변환하여 주기적 인코딩
    return cyclical_encode(month - 1, 12)


def encode_dayofyear(dayofyear: np.ndarray, is_leap_year: np.ndarray = None) -> tuple:
    """
    연중 일수를 주기적으로 인코딩합니다.

    - 주기: 365일 (또는 366일)
    - 입력 범위: 1-365/366
    - 계절 패턴을 더 세밀하게 표현

    Args:
        dayofyear: 연중 일수 배열 (1-365/366)
        is_leap_year: 윤년 여부 배열 (None이면 365일 기준)

    Returns:
        tuple: (dayofyear_sin, dayofyear_cos)
    """
    if is_leap_year is not None:
        period = np.where(is_leap_year, 366, 365)
    else:
        period = 365

    # 1-365를 0-364로 변환
    angle = 2 * np.pi * (dayofyear - 1) / period
    return np.sin(angle), np.cos(angle)


# ============================================================
# 이진 플래그 함수
# ============================================================

def is_weekend(dayofweek: np.ndarray) -> np.ndarray:
    """
    주말 여부를 반환합니다.

    Args:
        dayofweek: 요일 배열 (0=월요일, 6=일요일)

    Returns:
        np.ndarray: 주말이면 1, 아니면 0
    """
    return np.isin(dayofweek, [5, 6]).astype(np.int8)


def is_holiday(
    dates: Union[pd.DatetimeIndex, pd.Series],
    holidays: set = None
) -> np.ndarray:
    """
    공휴일 여부를 반환합니다.

    Args:
        dates: 날짜 인덱스 또는 시리즈
        holidays: 공휴일 set (None이면 한국 공휴일 사용)

    Returns:
        np.ndarray: 공휴일이면 1, 아니면 0
    """
    if holidays is None:
        # 데이터 범위에 맞는 공휴일 로드
        if hasattr(dates, 'year'):
            min_year = dates.year.min()
            max_year = dates.year.max()
        else:
            min_year = pd.to_datetime(dates).year.min()
            max_year = pd.to_datetime(dates).year.max()
        holidays = get_all_korean_holidays(min_year, max_year)

    # datetime을 date로 변환하여 비교
    if hasattr(dates, 'date'):
        date_only = dates.date
    else:
        date_only = pd.to_datetime(dates).date

    return np.array([d in holidays for d in date_only], dtype=np.int8)


def is_workday(
    dayofweek: np.ndarray,
    dates: Union[pd.DatetimeIndex, pd.Series] = None,
    holidays: set = None
) -> np.ndarray:
    """
    근무일 여부를 반환합니다 (주말도 아니고 공휴일도 아닌 날).

    Args:
        dayofweek: 요일 배열
        dates: 날짜 (공휴일 체크용, None이면 주말만 체크)
        holidays: 공휴일 set

    Returns:
        np.ndarray: 근무일이면 1, 아니면 0
    """
    weekend = is_weekend(dayofweek)

    if dates is not None:
        holiday = is_holiday(dates, holidays)
        return ((weekend == 0) & (holiday == 0)).astype(np.int8)
    else:
        return (weekend == 0).astype(np.int8)


# ============================================================
# 통합 함수
# ============================================================

def add_time_features(
    df: pd.DataFrame,
    include_hour: bool = True,
    include_dayofweek: bool = True,
    include_month: bool = True,
    include_dayofyear: bool = False,
    include_weekend: bool = True,
    include_holiday: bool = True,
    include_workday: bool = False,
    holidays: set = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    DataFrame에 시간 특성을 추가합니다.

    추가되는 컬럼:
    - hour_sin, hour_cos: 시간의 주기적 인코딩
    - dayofweek_sin, dayofweek_cos: 요일의 주기적 인코딩
    - month_sin, month_cos: 월의 주기적 인코딩
    - dayofyear_sin, dayofyear_cos: 연중 일수의 주기적 인코딩 (옵션)
    - is_weekend: 주말 플래그 (0/1)
    - is_holiday: 공휴일 플래그 (0/1)
    - is_workday: 근무일 플래그 (0/1)

    Args:
        df: 입력 DataFrame (datetime 인덱스 필요)
        include_hour: 시간 인코딩 포함 여부
        include_dayofweek: 요일 인코딩 포함 여부
        include_month: 월 인코딩 포함 여부
        include_dayofyear: 연중 일수 인코딩 포함 여부
        include_weekend: 주말 플래그 포함 여부
        include_holiday: 공휴일 플래그 포함 여부
        include_workday: 근무일 플래그 포함 여부
        holidays: 사용자 정의 공휴일 set
        inplace: True면 원본 수정, False면 복사본 반환

    Returns:
        pd.DataFrame: 시간 특성이 추가된 DataFrame

    Raises:
        ValueError: datetime 인덱스가 없는 경우

    Examples:
        >>> df = pd.DataFrame({'value': [1, 2, 3]},
        ...                   index=pd.date_range('2024-01-01', periods=3, freq='h'))
        >>> df_with_time = add_time_features(df)
        >>> df_with_time.columns.tolist()
        ['value', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
         'month_sin', 'month_cos', 'is_weekend', 'is_holiday']
    """
    if not inplace:
        df = df.copy()

    # datetime 인덱스 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame은 datetime 인덱스가 필요합니다.")

    idx = df.index

    # 시간 인코딩
    if include_hour:
        hour_sin, hour_cos = encode_hour(idx.hour.values)
        df['hour_sin'] = hour_sin
        df['hour_cos'] = hour_cos

    # 요일 인코딩
    if include_dayofweek:
        dow_sin, dow_cos = encode_dayofweek(idx.dayofweek.values)
        df['dayofweek_sin'] = dow_sin
        df['dayofweek_cos'] = dow_cos

    # 월 인코딩
    if include_month:
        month_sin, month_cos = encode_month(idx.month.values)
        df['month_sin'] = month_sin
        df['month_cos'] = month_cos

    # 연중 일수 인코딩
    if include_dayofyear:
        # is_leap_year는 이미 numpy array
        is_leap = np.array(idx.is_leap_year)
        doy_sin, doy_cos = encode_dayofyear(
            idx.dayofyear.values,
            is_leap
        )
        df['dayofyear_sin'] = doy_sin
        df['dayofyear_cos'] = doy_cos

    # 주말 플래그
    if include_weekend:
        df['is_weekend'] = is_weekend(idx.dayofweek.values)

    # 공휴일 플래그
    if include_holiday:
        df['is_holiday'] = is_holiday(idx, holidays)

    # 근무일 플래그
    if include_workday:
        df['is_workday'] = is_workday(idx.dayofweek.values, idx, holidays)

    return df


def get_time_feature_names(
    include_hour: bool = True,
    include_dayofweek: bool = True,
    include_month: bool = True,
    include_dayofyear: bool = False,
    include_weekend: bool = True,
    include_holiday: bool = True,
    include_workday: bool = False
) -> List[str]:
    """
    생성될 시간 특성 컬럼명 목록을 반환합니다.

    Args:
        include_*: 각 특성 포함 여부

    Returns:
        List[str]: 컬럼명 목록
    """
    names = []

    if include_hour:
        names.extend(['hour_sin', 'hour_cos'])
    if include_dayofweek:
        names.extend(['dayofweek_sin', 'dayofweek_cos'])
    if include_month:
        names.extend(['month_sin', 'month_cos'])
    if include_dayofyear:
        names.extend(['dayofyear_sin', 'dayofyear_cos'])
    if include_weekend:
        names.append('is_weekend')
    if include_holiday:
        names.append('is_holiday')
    if include_workday:
        names.append('is_workday')

    return names


# ============================================================
# 데모 및 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEAT-003: Time Features Demo")
    print("=" * 60)

    # 샘플 데이터 생성 (1주일, 시간별)
    dates = pd.date_range('2024-01-01', periods=168, freq='h')  # 7일 * 24시간
    df = pd.DataFrame({
        'value': np.random.randn(len(dates))
    }, index=dates)

    print(f"\n[Input] Sample data: {len(df)} hours")
    print(f"Date range: {df.index.min()} ~ {df.index.max()}")

    # 시간 특성 추가
    df_with_time = add_time_features(df)

    print(f"\n[Output] Added columns: {get_time_feature_names()}")
    print(f"Total columns: {len(df_with_time.columns)}")

    # 결과 샘플 출력
    print("\n=== Sample Output (first 24 hours) ===")
    display_cols = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                    'is_weekend', 'is_holiday']
    print(df_with_time[display_cols].head(24).to_string())

    # 주기적 인코딩 검증
    print("\n=== Cyclical Encoding Verification ===")
    print("Hour encoding (0, 6, 12, 18, 23):")
    for h in [0, 6, 12, 18, 23]:
        sin_val = df_with_time.loc[df_with_time.index.hour == h, 'hour_sin'].iloc[0]
        cos_val = df_with_time.loc[df_with_time.index.hour == h, 'hour_cos'].iloc[0]
        print(f"  Hour {h:2d}: sin={sin_val:+.4f}, cos={cos_val:+.4f}")

    # 주말/공휴일 카운트
    print("\n=== Weekend/Holiday Counts ===")
    print(f"Weekend hours: {df_with_time['is_weekend'].sum()} / {len(df_with_time)}")
    print(f"Holiday hours: {df_with_time['is_holiday'].sum()} / {len(df_with_time)}")

    # 공휴일 목록 출력
    print("\n=== Korean Holidays (2024) ===")
    holidays_2024 = get_korean_holidays(2024)
    for h in holidays_2024[:10]:
        print(f"  - {h}")
    print(f"  ... and {len(holidays_2024) - 10} more")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
