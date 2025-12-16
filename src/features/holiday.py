"""
특별일/공휴일 처리 모듈 (Task 11)
=================================
공휴일, 특별일, 이벤트 등의 영향을 모델링합니다.

주요 컴포넌트:
- HolidayCalendar: 공휴일 캘린더
- SpecialDayEncoder: 특별일 인코딩
- EventImpactModeler: 이벤트 영향 모델링
- SeasonalAdjuster: 계절별 조정
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from enum import Enum
import json
import logging
import calendar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DayType(Enum):
    """요일 타입"""
    WEEKDAY = 'weekday'
    SATURDAY = 'saturday'
    SUNDAY = 'sunday'
    HOLIDAY = 'holiday'
    BRIDGE_DAY = 'bridge_day'      # 징검다리 휴일
    SPECIAL_EVENT = 'special_event'


class HolidayType(Enum):
    """공휴일 타입"""
    NATIONAL = 'national'           # 국경일
    PUBLIC = 'public'               # 공휴일
    LUNAR = 'lunar'                 # 음력 명절
    SUBSTITUTE = 'substitute'       # 대체공휴일
    REGIONAL = 'regional'           # 지역 공휴일
    CUSTOM = 'custom'               # 사용자 정의


class Season(Enum):
    """계절"""
    SPRING = 'spring'
    SUMMER = 'summer'
    FALL = 'fall'
    WINTER = 'winter'


@dataclass
class Holiday:
    """공휴일"""
    name: str
    date: date
    holiday_type: HolidayType
    observed: bool = True           # 실제 휴무 여부
    impact_factor: float = 1.0      # 영향 계수
    description: str = ''


@dataclass
class SpecialEvent:
    """특별 이벤트"""
    name: str
    start_date: date
    end_date: date
    impact_factor: float = 1.0
    category: str = 'event'
    recurring: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DayInfo:
    """날짜 정보"""
    date: date
    day_type: DayType
    is_holiday: bool
    holiday_name: Optional[str] = None
    is_weekend: bool = False
    is_bridge_day: bool = False
    events: List[str] = field(default_factory=list)
    season: Optional[Season] = None
    impact_factor: float = 1.0


class KoreanHolidayCalendar:
    """
    한국 공휴일 캘린더

    한국의 법정 공휴일 및 대체공휴일을 관리합니다.
    """

    # 고정 공휴일 (월, 일)
    FIXED_HOLIDAYS = {
        (1, 1): ('신정', HolidayType.PUBLIC),
        (3, 1): ('삼일절', HolidayType.NATIONAL),
        (5, 5): ('어린이날', HolidayType.PUBLIC),
        (6, 6): ('현충일', HolidayType.NATIONAL),
        (8, 15): ('광복절', HolidayType.NATIONAL),
        (10, 3): ('개천절', HolidayType.NATIONAL),
        (10, 9): ('한글날', HolidayType.NATIONAL),
        (12, 25): ('크리스마스', HolidayType.PUBLIC),
    }

    # 음력 명절 (양력 변환 필요)
    LUNAR_HOLIDAYS = {
        'seollal': ('설날', [(1, -1), (1, 1), (1, 2)]),  # 음력 1월 1일 전후
        'chuseok': ('추석', [(8, 14), (8, 15), (8, 16)]),  # 음력 8월 15일 전후
    }

    def __init__(self):
        """캘린더 초기화"""
        self._holidays: Dict[date, Holiday] = {}
        self._custom_holidays: Dict[date, Holiday] = {}

    def add_fixed_holidays(self, year: int) -> None:
        """고정 공휴일 추가"""
        for (month, day), (name, h_type) in self.FIXED_HOLIDAYS.items():
            holiday_date = date(year, month, day)
            self._holidays[holiday_date] = Holiday(
                name=name,
                date=holiday_date,
                holiday_type=h_type
            )

    def add_lunar_holidays(self, year: int, lunar_dates: Dict[str, List[date]]) -> None:
        """
        음력 공휴일 추가

        Args:
            year: 연도
            lunar_dates: 음력 공휴일 양력 변환 날짜
        """
        for key, dates in lunar_dates.items():
            name = self.LUNAR_HOLIDAYS[key][0] if key in self.LUNAR_HOLIDAYS else key
            for d in dates:
                self._holidays[d] = Holiday(
                    name=name,
                    date=d,
                    holiday_type=HolidayType.LUNAR
                )

    def add_custom_holiday(
        self,
        name: str,
        holiday_date: date,
        holiday_type: HolidayType = HolidayType.CUSTOM,
        impact_factor: float = 1.0
    ) -> None:
        """사용자 정의 공휴일 추가"""
        holiday = Holiday(
            name=name,
            date=holiday_date,
            holiday_type=holiday_type,
            impact_factor=impact_factor
        )
        self._custom_holidays[holiday_date] = holiday
        self._holidays[holiday_date] = holiday

    def get_holidays(
        self,
        year: int,
        month: Optional[int] = None
    ) -> List[Holiday]:
        """공휴일 목록 조회"""
        holidays = []
        for d, h in self._holidays.items():
            if d.year == year:
                if month is None or d.month == month:
                    holidays.append(h)
        return sorted(holidays, key=lambda x: x.date)

    def is_holiday(self, d: date) -> bool:
        """공휴일 여부"""
        return d in self._holidays

    def get_holiday(self, d: date) -> Optional[Holiday]:
        """공휴일 정보 조회"""
        return self._holidays.get(d)

    def get_bridge_days(self, year: int) -> List[date]:
        """
        징검다리 휴일 조회

        공휴일과 주말 사이의 평일을 찾습니다.
        """
        bridge_days = []

        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)

            for day in range(1, last_day + 1):
                d = date(year, month, day)

                # 이미 공휴일이거나 주말이면 건너뜀
                if self.is_holiday(d) or d.weekday() >= 5:
                    continue

                # 전날과 다음날 확인
                prev_day = d - timedelta(days=1)
                next_day = d + timedelta(days=1)

                prev_is_off = self.is_holiday(prev_day) or prev_day.weekday() >= 5
                next_is_off = self.is_holiday(next_day) or next_day.weekday() >= 5

                if prev_is_off and next_is_off:
                    bridge_days.append(d)

        return bridge_days


class HolidayCalendar:
    """
    통합 공휴일 캘린더

    여러 국가의 공휴일을 지원하고, 사용자 정의 이벤트를 관리합니다.
    """

    def __init__(
        self,
        country: str = 'KR',
        include_regional: bool = False
    ):
        """
        Args:
            country: 국가 코드
            include_regional: 지역 공휴일 포함 여부
        """
        self.country = country
        self.include_regional = include_regional

        self._korean_calendar = KoreanHolidayCalendar()
        self._events: Dict[date, List[SpecialEvent]] = {}
        self._initialized_years: Set[int] = set()

    def initialize_year(self, year: int) -> None:
        """연도별 공휴일 초기화"""
        if year in self._initialized_years:
            return

        if self.country == 'KR':
            self._korean_calendar.add_fixed_holidays(year)
            # 음력 공휴일은 별도 설정 필요
            self._add_default_lunar_holidays(year)

        self._initialized_years.add(year)

    def _add_default_lunar_holidays(self, year: int) -> None:
        """기본 음력 공휴일 추가 (근사치)"""
        # 실제로는 음력 변환 라이브러리 필요
        # 여기서는 예시 날짜 사용
        lunar_dates = {
            'seollal': [
                date(year, 1, 21),  # 예시
                date(year, 1, 22),
                date(year, 1, 23),
            ],
            'chuseok': [
                date(year, 9, 16),  # 예시
                date(year, 9, 17),
                date(year, 9, 18),
            ],
        }
        self._korean_calendar.add_lunar_holidays(year, lunar_dates)

    def add_event(self, event: SpecialEvent) -> None:
        """이벤트 추가"""
        current = event.start_date
        while current <= event.end_date:
            if current not in self._events:
                self._events[current] = []
            self._events[current].append(event)
            current += timedelta(days=1)

    def get_day_info(self, d: date) -> DayInfo:
        """날짜 정보 조회"""
        # 연도 초기화
        self.initialize_year(d.year)

        # 기본 정보
        is_weekend = d.weekday() >= 5
        is_holiday = self._korean_calendar.is_holiday(d)
        holiday = self._korean_calendar.get_holiday(d)

        # 요일 타입 결정
        if is_holiday:
            day_type = DayType.HOLIDAY
        elif d.weekday() == 5:
            day_type = DayType.SATURDAY
        elif d.weekday() == 6:
            day_type = DayType.SUNDAY
        else:
            # 징검다리 휴일 확인
            bridge_days = self._korean_calendar.get_bridge_days(d.year)
            if d in bridge_days:
                day_type = DayType.BRIDGE_DAY
            else:
                day_type = DayType.WEEKDAY

        # 이벤트
        events = self._events.get(d, [])
        event_names = [e.name for e in events]

        # 계절
        season = self._get_season(d)

        # 영향 계수
        impact_factor = self._calculate_impact_factor(
            day_type, is_holiday, events
        )

        return DayInfo(
            date=d,
            day_type=day_type,
            is_holiday=is_holiday,
            holiday_name=holiday.name if holiday else None,
            is_weekend=is_weekend,
            is_bridge_day=(day_type == DayType.BRIDGE_DAY),
            events=event_names,
            season=season,
            impact_factor=impact_factor
        )

    def _get_season(self, d: date) -> Season:
        """계절 결정"""
        month = d.month
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER

    def _calculate_impact_factor(
        self,
        day_type: DayType,
        is_holiday: bool,
        events: List[SpecialEvent]
    ) -> float:
        """영향 계수 계산"""
        factor = 1.0

        # 요일 타입별 기본 계수
        type_factors = {
            DayType.WEEKDAY: 1.0,
            DayType.SATURDAY: 0.85,
            DayType.SUNDAY: 0.80,
            DayType.HOLIDAY: 0.75,
            DayType.BRIDGE_DAY: 0.90,
            DayType.SPECIAL_EVENT: 1.0,
        }
        factor *= type_factors.get(day_type, 1.0)

        # 이벤트 영향
        for event in events:
            factor *= event.impact_factor

        return factor


class SpecialDayEncoder:
    """
    특별일 인코더

    날짜 정보를 피처로 인코딩합니다.
    """

    def __init__(
        self,
        calendar: Optional[HolidayCalendar] = None,
        encoding_type: str = 'onehot'
    ):
        """
        Args:
            calendar: 공휴일 캘린더
            encoding_type: 인코딩 방식 ('onehot', 'ordinal', 'cyclical')
        """
        self.calendar = calendar or HolidayCalendar()
        self.encoding_type = encoding_type

    def encode(self, dates: List[date]) -> pd.DataFrame:
        """
        날짜 인코딩

        Args:
            dates: 날짜 리스트

        Returns:
            인코딩된 피처 DataFrame
        """
        features = []

        for d in dates:
            day_info = self.calendar.get_day_info(d)
            feature = self._encode_day(d, day_info)
            features.append(feature)

        return pd.DataFrame(features)

    def _encode_day(self, d: date, day_info: DayInfo) -> Dict[str, Any]:
        """단일 날짜 인코딩"""
        feature = {
            'date': d,
            'year': d.year,
            'month': d.month,
            'day': d.day,
            'day_of_week': d.weekday(),
            'day_of_year': d.timetuple().tm_yday,
            'week_of_year': d.isocalendar()[1],
            'is_weekend': int(day_info.is_weekend),
            'is_holiday': int(day_info.is_holiday),
            'is_bridge_day': int(day_info.is_bridge_day),
            'impact_factor': day_info.impact_factor,
        }

        # 요일 타입 원핫 인코딩
        for dtype in DayType:
            feature[f'day_type_{dtype.value}'] = int(day_info.day_type == dtype)

        # 계절 원핫 인코딩
        for season in Season:
            feature[f'season_{season.value}'] = int(day_info.season == season)

        # 주기적 인코딩 (선택적)
        if self.encoding_type == 'cyclical':
            # 요일 주기
            feature['day_sin'] = np.sin(2 * np.pi * d.weekday() / 7)
            feature['day_cos'] = np.cos(2 * np.pi * d.weekday() / 7)
            # 월 주기
            feature['month_sin'] = np.sin(2 * np.pi * d.month / 12)
            feature['month_cos'] = np.cos(2 * np.pi * d.month / 12)
            # 연중 주기
            feature['doy_sin'] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365)
            feature['doy_cos'] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365)

        return feature

    def get_feature_names(self) -> List[str]:
        """피처 이름 목록"""
        sample_date = date(2025, 1, 1)
        day_info = self.calendar.get_day_info(sample_date)
        feature = self._encode_day(sample_date, day_info)
        return list(feature.keys())


class EventImpactModeler:
    """
    이벤트 영향 모델러

    특별 이벤트가 전력 수요에 미치는 영향을 모델링합니다.
    """

    def __init__(self):
        """모델러 초기화"""
        self._impact_history: Dict[str, List[float]] = {}
        self._learned_impacts: Dict[str, float] = {}

    def record_impact(
        self,
        event_name: str,
        actual_demand: float,
        expected_demand: float
    ) -> None:
        """
        이벤트 영향 기록

        Args:
            event_name: 이벤트 이름
            actual_demand: 실제 수요
            expected_demand: 예상 수요
        """
        impact = actual_demand / (expected_demand + 1e-8)

        if event_name not in self._impact_history:
            self._impact_history[event_name] = []

        self._impact_history[event_name].append(impact)
        self._update_learned_impact(event_name)

    def _update_learned_impact(self, event_name: str) -> None:
        """학습된 영향 계수 업데이트"""
        if event_name in self._impact_history:
            impacts = self._impact_history[event_name]
            # 지수 이동 평균
            if len(impacts) == 1:
                self._learned_impacts[event_name] = impacts[0]
            else:
                alpha = 0.3
                prev = self._learned_impacts.get(event_name, 1.0)
                self._learned_impacts[event_name] = alpha * impacts[-1] + (1 - alpha) * prev

    def get_impact(self, event_name: str) -> float:
        """이벤트 영향 계수 조회"""
        return self._learned_impacts.get(event_name, 1.0)

    def get_all_impacts(self) -> Dict[str, float]:
        """모든 이벤트 영향 계수 조회"""
        return self._learned_impacts.copy()

    def predict_demand_adjustment(
        self,
        base_demand: float,
        events: List[str]
    ) -> float:
        """
        수요 조정 예측

        Args:
            base_demand: 기본 수요
            events: 이벤트 목록

        Returns:
            조정된 수요
        """
        adjustment = 1.0
        for event in events:
            adjustment *= self.get_impact(event)

        return base_demand * adjustment


class SeasonalAdjuster:
    """
    계절별 조정기

    계절에 따른 수요 패턴을 조정합니다.
    """

    # 기본 계절별 영향 계수 (제주 기준)
    DEFAULT_SEASONAL_FACTORS = {
        Season.SPRING: 0.95,   # 봄: 냉난방 수요 감소
        Season.SUMMER: 1.15,   # 여름: 냉방 수요 증가
        Season.FALL: 0.90,     # 가을: 냉난방 수요 최저
        Season.WINTER: 1.10,   # 겨울: 난방 수요 증가
    }

    def __init__(
        self,
        seasonal_factors: Optional[Dict[Season, float]] = None
    ):
        """
        Args:
            seasonal_factors: 계절별 영향 계수
        """
        self.seasonal_factors = seasonal_factors or self.DEFAULT_SEASONAL_FACTORS.copy()

    def get_factor(self, season: Season) -> float:
        """계절별 영향 계수 조회"""
        return self.seasonal_factors.get(season, 1.0)

    def adjust(
        self,
        demand: float,
        d: date,
        calendar: Optional[HolidayCalendar] = None
    ) -> float:
        """
        수요 조정

        Args:
            demand: 기본 수요
            d: 날짜
            calendar: 공휴일 캘린더

        Returns:
            조정된 수요
        """
        if calendar:
            day_info = calendar.get_day_info(d)
            season = day_info.season
        else:
            season = self._get_season(d)

        factor = self.get_factor(season)
        return demand * factor

    def _get_season(self, d: date) -> Season:
        """계절 결정"""
        month = d.month
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.FALL
        else:
            return Season.WINTER

    def update_factor(self, season: Season, factor: float) -> None:
        """영향 계수 업데이트"""
        self.seasonal_factors[season] = factor


class HolidayFeatureGenerator:
    """
    공휴일 피처 생성기

    모델 학습을 위한 공휴일 관련 피처를 생성합니다.
    """

    def __init__(
        self,
        calendar: Optional[HolidayCalendar] = None,
        encoder: Optional[SpecialDayEncoder] = None,
        seasonal_adjuster: Optional[SeasonalAdjuster] = None
    ):
        """
        Args:
            calendar: 공휴일 캘린더
            encoder: 특별일 인코더
            seasonal_adjuster: 계절별 조정기
        """
        self.calendar = calendar or HolidayCalendar()
        self.encoder = encoder or SpecialDayEncoder(self.calendar)
        self.seasonal_adjuster = seasonal_adjuster or SeasonalAdjuster()

    def generate(
        self,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        피처 생성

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            피처 DataFrame
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        return self.encoder.encode(dates)

    def add_to_dataframe(
        self,
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        DataFrame에 피처 추가

        Args:
            df: 입력 DataFrame
            date_column: 날짜 컬럼명

        Returns:
            피처가 추가된 DataFrame
        """
        df = df.copy()

        # 날짜 파싱
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column]).dt.date.tolist()
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date.tolist()
        else:
            raise ValueError(f"Cannot find date column: {date_column}")

        # 피처 생성
        features_df = self.encoder.encode(dates)

        # 날짜 컬럼 제거 (중복 방지)
        features_df = features_df.drop(columns=['date'], errors='ignore')

        # 병합
        for col in features_df.columns:
            df[col] = features_df[col].values

        return df


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_korean_holiday_calendar(year: int) -> HolidayCalendar:
    """한국 공휴일 캘린더 생성"""
    calendar = HolidayCalendar(country='KR')
    calendar.initialize_year(year)
    return calendar


def create_holiday_encoder(
    encoding_type: str = 'onehot',
    calendar: Optional[HolidayCalendar] = None
) -> SpecialDayEncoder:
    """공휴일 인코더 생성"""
    return SpecialDayEncoder(calendar, encoding_type)


def create_feature_generator(
    country: str = 'KR'
) -> HolidayFeatureGenerator:
    """피처 생성기 생성"""
    calendar = HolidayCalendar(country=country)
    return HolidayFeatureGenerator(calendar)


def add_holiday_features(
    df: pd.DataFrame,
    date_column: str = 'date',
    encoding_type: str = 'onehot'
) -> pd.DataFrame:
    """
    DataFrame에 공휴일 피처 추가

    Args:
        df: 입력 DataFrame
        date_column: 날짜 컬럼명
        encoding_type: 인코딩 방식

    Returns:
        피처가 추가된 DataFrame
    """
    generator = create_feature_generator()
    generator.encoder.encoding_type = encoding_type
    return generator.add_to_dataframe(df, date_column)
