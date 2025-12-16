"""
특별일/공휴일 처리 테스트
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta


class TestDayType:
    """DayType 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.features.holiday import DayType

        assert DayType.WEEKDAY.value == 'weekday'
        assert DayType.HOLIDAY.value == 'holiday'
        assert DayType.BRIDGE_DAY.value == 'bridge_day'


class TestHolidayType:
    """HolidayType 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.features.holiday import HolidayType

        assert HolidayType.NATIONAL.value == 'national'
        assert HolidayType.LUNAR.value == 'lunar'


class TestSeason:
    """Season 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.features.holiday import Season

        assert Season.SPRING.value == 'spring'
        assert Season.SUMMER.value == 'summer'
        assert Season.FALL.value == 'fall'
        assert Season.WINTER.value == 'winter'


class TestHoliday:
    """Holiday 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import Holiday, HolidayType

        holiday = Holiday(
            name='신정',
            date=date(2025, 1, 1),
            holiday_type=HolidayType.PUBLIC
        )

        assert holiday.name == '신정'
        assert holiday.date == date(2025, 1, 1)
        assert holiday.impact_factor == 1.0


class TestSpecialEvent:
    """SpecialEvent 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import SpecialEvent

        event = SpecialEvent(
            name='여름 축제',
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 7),
            impact_factor=1.2
        )

        assert event.name == '여름 축제'
        assert event.impact_factor == 1.2


class TestKoreanHolidayCalendar:
    """KoreanHolidayCalendar 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()

        assert calendar is not None

    def test_add_fixed_holidays(self):
        """고정 공휴일 추가"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        # 신정 확인
        assert calendar.is_holiday(date(2025, 1, 1))
        # 삼일절 확인
        assert calendar.is_holiday(date(2025, 3, 1))
        # 광복절 확인
        assert calendar.is_holiday(date(2025, 8, 15))

    def test_is_holiday(self):
        """공휴일 여부"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        assert calendar.is_holiday(date(2025, 5, 5))  # 어린이날
        assert not calendar.is_holiday(date(2025, 5, 6))  # 평일

    def test_get_holiday(self):
        """공휴일 정보 조회"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        holiday = calendar.get_holiday(date(2025, 10, 3))

        assert holiday is not None
        assert holiday.name == '개천절'

    def test_add_custom_holiday(self):
        """사용자 정의 공휴일"""
        from src.features.holiday import KoreanHolidayCalendar, HolidayType

        calendar = KoreanHolidayCalendar()

        calendar.add_custom_holiday(
            name='회사 창립일',
            holiday_date=date(2025, 4, 1),
            holiday_type=HolidayType.CUSTOM
        )

        assert calendar.is_holiday(date(2025, 4, 1))

    def test_get_holidays(self):
        """공휴일 목록"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        holidays = calendar.get_holidays(2025)

        assert len(holidays) == 8  # 고정 공휴일 8개

    def test_get_holidays_by_month(self):
        """월별 공휴일 목록"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        # 10월 공휴일
        october_holidays = calendar.get_holidays(2025, month=10)

        assert len(october_holidays) == 2  # 개천절, 한글날

    def test_get_bridge_days(self):
        """징검다리 휴일"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        bridge_days = calendar.get_bridge_days(2025)

        # 징검다리 휴일이 있을 수 있음
        assert isinstance(bridge_days, list)


class TestHolidayCalendar:
    """HolidayCalendar 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import HolidayCalendar

        calendar = HolidayCalendar(country='KR')

        assert calendar.country == 'KR'

    def test_initialize_year(self):
        """연도 초기화"""
        from src.features.holiday import HolidayCalendar

        calendar = HolidayCalendar()
        calendar.initialize_year(2025)

        assert 2025 in calendar._initialized_years

    def test_add_event(self):
        """이벤트 추가"""
        from src.features.holiday import HolidayCalendar, SpecialEvent

        calendar = HolidayCalendar()

        event = SpecialEvent(
            name='축제',
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 3),
            impact_factor=1.2
        )
        calendar.add_event(event)

        day_info = calendar.get_day_info(date(2025, 7, 2))

        assert '축제' in day_info.events

    def test_get_day_info_weekday(self):
        """평일 정보"""
        from src.features.holiday import HolidayCalendar, DayType

        calendar = HolidayCalendar()

        # 2025-01-06은 월요일
        day_info = calendar.get_day_info(date(2025, 1, 6))

        assert day_info.day_type == DayType.WEEKDAY
        assert not day_info.is_weekend
        assert not day_info.is_holiday

    def test_get_day_info_weekend(self):
        """주말 정보"""
        from src.features.holiday import HolidayCalendar, DayType

        calendar = HolidayCalendar()

        # 2025-01-04는 토요일
        day_info = calendar.get_day_info(date(2025, 1, 4))

        assert day_info.day_type == DayType.SATURDAY
        assert day_info.is_weekend

    def test_get_day_info_holiday(self):
        """공휴일 정보"""
        from src.features.holiday import HolidayCalendar, DayType

        calendar = HolidayCalendar()
        calendar.initialize_year(2025)

        day_info = calendar.get_day_info(date(2025, 1, 1))

        assert day_info.day_type == DayType.HOLIDAY
        assert day_info.is_holiday
        assert day_info.holiday_name == '신정'

    def test_season_detection(self):
        """계절 감지"""
        from src.features.holiday import HolidayCalendar, Season

        calendar = HolidayCalendar()

        # 봄
        spring_info = calendar.get_day_info(date(2025, 4, 15))
        assert spring_info.season == Season.SPRING

        # 여름
        summer_info = calendar.get_day_info(date(2025, 7, 15))
        assert summer_info.season == Season.SUMMER

        # 가을
        fall_info = calendar.get_day_info(date(2025, 10, 15))
        assert fall_info.season == Season.FALL

        # 겨울
        winter_info = calendar.get_day_info(date(2025, 1, 15))
        assert winter_info.season == Season.WINTER


class TestSpecialDayEncoder:
    """SpecialDayEncoder 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import SpecialDayEncoder

        encoder = SpecialDayEncoder()

        assert encoder is not None

    def test_encode(self):
        """인코딩"""
        from src.features.holiday import SpecialDayEncoder

        encoder = SpecialDayEncoder()

        dates = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
        features = encoder.encode(dates)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 3
        assert 'is_holiday' in features.columns

    def test_encode_onehot(self):
        """원핫 인코딩"""
        from src.features.holiday import SpecialDayEncoder

        encoder = SpecialDayEncoder(encoding_type='onehot')

        dates = [date(2025, 1, 1)]
        features = encoder.encode(dates)

        # 요일 타입 원핫 인코딩 확인
        assert 'day_type_weekday' in features.columns
        assert 'day_type_holiday' in features.columns

    def test_encode_cyclical(self):
        """주기적 인코딩"""
        from src.features.holiday import SpecialDayEncoder

        encoder = SpecialDayEncoder(encoding_type='cyclical')

        dates = [date(2025, 1, 1)]
        features = encoder.encode(dates)

        assert 'day_sin' in features.columns
        assert 'day_cos' in features.columns
        assert 'month_sin' in features.columns
        assert 'month_cos' in features.columns

    def test_get_feature_names(self):
        """피처 이름 목록"""
        from src.features.holiday import SpecialDayEncoder

        encoder = SpecialDayEncoder()
        names = encoder.get_feature_names()

        assert 'is_holiday' in names
        assert 'day_of_week' in names


class TestEventImpactModeler:
    """EventImpactModeler 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import EventImpactModeler

        modeler = EventImpactModeler()

        assert modeler is not None

    def test_record_impact(self):
        """영향 기록"""
        from src.features.holiday import EventImpactModeler

        modeler = EventImpactModeler()

        modeler.record_impact('축제', actual_demand=1200, expected_demand=1000)

        impact = modeler.get_impact('축제')

        assert impact == pytest.approx(1.2, rel=0.1)

    def test_get_impact_unknown(self):
        """알려지지 않은 이벤트"""
        from src.features.holiday import EventImpactModeler

        modeler = EventImpactModeler()

        impact = modeler.get_impact('unknown_event')

        assert impact == 1.0

    def test_predict_demand_adjustment(self):
        """수요 조정 예측"""
        from src.features.holiday import EventImpactModeler

        modeler = EventImpactModeler()

        modeler.record_impact('축제A', 1100, 1000)
        modeler.record_impact('축제B', 1050, 1000)

        adjusted = modeler.predict_demand_adjustment(1000, ['축제A', '축제B'])

        assert adjusted > 1000  # 두 축제 영향으로 증가


class TestSeasonalAdjuster:
    """SeasonalAdjuster 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import SeasonalAdjuster

        adjuster = SeasonalAdjuster()

        assert adjuster is not None

    def test_get_factor(self):
        """계절별 영향 계수"""
        from src.features.holiday import SeasonalAdjuster, Season

        adjuster = SeasonalAdjuster()

        summer_factor = adjuster.get_factor(Season.SUMMER)
        winter_factor = adjuster.get_factor(Season.WINTER)

        assert summer_factor > 1.0  # 여름은 냉방 수요 증가
        assert winter_factor > 1.0  # 겨울은 난방 수요 증가

    def test_adjust(self):
        """수요 조정"""
        from src.features.holiday import SeasonalAdjuster

        adjuster = SeasonalAdjuster()

        # 여름 수요 조정
        adjusted = adjuster.adjust(1000, date(2025, 7, 15))

        assert adjusted > 1000  # 여름은 증가

    def test_update_factor(self):
        """영향 계수 업데이트"""
        from src.features.holiday import SeasonalAdjuster, Season

        adjuster = SeasonalAdjuster()

        adjuster.update_factor(Season.SUMMER, 1.3)

        assert adjuster.get_factor(Season.SUMMER) == 1.3


class TestHolidayFeatureGenerator:
    """HolidayFeatureGenerator 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.holiday import HolidayFeatureGenerator

        generator = HolidayFeatureGenerator()

        assert generator is not None

    def test_generate(self):
        """피처 생성"""
        from src.features.holiday import HolidayFeatureGenerator

        generator = HolidayFeatureGenerator()

        features = generator.generate(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7)
        )

        assert len(features) == 7
        assert 'is_holiday' in features.columns

    def test_add_to_dataframe(self):
        """DataFrame에 피처 추가"""
        from src.features.holiday import HolidayFeatureGenerator

        generator = HolidayFeatureGenerator()

        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })

        result = generator.add_to_dataframe(df, date_column='date')

        assert 'is_holiday' in result.columns
        assert 'day_of_week' in result.columns
        assert len(result) == 5


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_korean_holiday_calendar(self):
        """한국 공휴일 캘린더 생성"""
        from src.features.holiday import create_korean_holiday_calendar

        calendar = create_korean_holiday_calendar(2025)

        assert calendar is not None
        assert 2025 in calendar._initialized_years

    def test_create_holiday_encoder(self):
        """공휴일 인코더 생성"""
        from src.features.holiday import create_holiday_encoder

        encoder = create_holiday_encoder(encoding_type='cyclical')

        assert encoder is not None
        assert encoder.encoding_type == 'cyclical'

    def test_create_feature_generator(self):
        """피처 생성기 생성"""
        from src.features.holiday import create_feature_generator

        generator = create_feature_generator(country='KR')

        assert generator is not None

    def test_add_holiday_features(self):
        """간편 피처 추가"""
        from src.features.holiday import add_holiday_features

        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'demand': [100, 110, 105, 108, 102]
        })

        result = add_holiday_features(df, date_column='date')

        assert 'is_holiday' in result.columns
        assert len(result) == 5


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self):
        """전체 워크플로우"""
        from src.features.holiday import (
            HolidayCalendar, SpecialEvent, SpecialDayEncoder,
            EventImpactModeler, SeasonalAdjuster
        )

        # 1. 캘린더 설정
        calendar = HolidayCalendar(country='KR')
        calendar.initialize_year(2025)

        # 2. 이벤트 추가
        event = SpecialEvent(
            name='제주 마라톤',
            start_date=date(2025, 4, 20),
            end_date=date(2025, 4, 20),
            impact_factor=0.95
        )
        calendar.add_event(event)

        # 3. 날짜 정보 조회
        marathon_day = calendar.get_day_info(date(2025, 4, 20))
        assert '제주 마라톤' in marathon_day.events

        # 4. 인코딩
        encoder = SpecialDayEncoder(calendar)
        features = encoder.encode([date(2025, 4, 20)])
        assert len(features) == 1

        # 5. 영향 모델링
        impact_modeler = EventImpactModeler()
        impact_modeler.record_impact('제주 마라톤', 950, 1000)

        # 6. 계절 조정
        adjuster = SeasonalAdjuster()
        adjusted = adjuster.adjust(1000, date(2025, 4, 20), calendar)

        assert adjusted < 1000  # 봄은 수요 감소

    def test_feature_pipeline(self):
        """피처 파이프라인"""
        from src.features.holiday import (
            HolidayFeatureGenerator, add_holiday_features
        )

        # 수요 데이터 생성
        dates = pd.date_range('2025-01-01', periods=365)
        df = pd.DataFrame({
            'date': dates,
            'demand': np.random.randn(365) * 100 + 1000
        })

        # 피처 추가
        result = add_holiday_features(df, date_column='date')

        # 검증
        assert 'is_holiday' in result.columns
        assert 'season_spring' in result.columns
        assert 'day_type_weekday' in result.columns

        # 공휴일 피처 확인
        holiday_rows = result[result['is_holiday'] == 1]
        assert len(holiday_rows) > 0  # 공휴일이 있어야 함

