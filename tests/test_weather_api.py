"""
기상 API 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class TestWeatherVariable:
    """WeatherVariable 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.data.weather_api import WeatherVariable

        assert WeatherVariable.TEMPERATURE.value == 'temperature'
        assert WeatherVariable.HUMIDITY.value == 'humidity'
        assert WeatherVariable.WIND_SPEED.value == 'wind_speed'


class TestWeatherObservation:
    """WeatherObservation 테스트"""

    def test_creation(self):
        """관측 데이터 생성"""
        from src.data.weather_api import WeatherObservation

        obs = WeatherObservation(
            timestamp=datetime.now(),
            location='jeju',
            variables={'temperature': 22.5, 'humidity': 65.0},
            source='test'
        )

        assert obs.location == 'jeju'
        assert obs.variables['temperature'] == 22.5

    def test_to_dict(self):
        """딕셔너리 변환"""
        from src.data.weather_api import WeatherObservation

        obs = WeatherObservation(
            timestamp=datetime.now(),
            location='jeju',
            variables={'temperature': 22.5},
            source='test'
        )

        d = obs.to_dict()

        assert isinstance(d, dict)
        assert d['location'] == 'jeju'
        assert 'timestamp' in d

    def test_from_dict(self):
        """딕셔너리에서 생성"""
        from src.data.weather_api import WeatherObservation

        data = {
            'timestamp': datetime.now().isoformat(),
            'location': 'jeju',
            'variables': {'temperature': 22.5},
            'source': 'test'
        }

        obs = WeatherObservation.from_dict(data)

        assert obs.location == 'jeju'


class TestWeatherForecast:
    """WeatherForecast 테스트"""

    def test_creation(self):
        """예보 데이터 생성"""
        from src.data.weather_api import WeatherForecast

        now = datetime.now()
        forecast = WeatherForecast(
            issue_time=now,
            valid_time=now + timedelta(hours=1),
            location='jeju',
            variables={'temperature': 23.0},
            source='test'
        )

        assert forecast.location == 'jeju'
        assert forecast.confidence == 1.0

    def test_to_dict(self):
        """딕셔너리 변환"""
        from src.data.weather_api import WeatherForecast

        now = datetime.now()
        forecast = WeatherForecast(
            issue_time=now,
            valid_time=now + timedelta(hours=1),
            location='jeju',
            variables={'temperature': 23.0},
            source='test'
        )

        d = forecast.to_dict()

        assert 'issue_time' in d
        assert 'valid_time' in d


class TestRateLimiter:
    """RateLimiter 테스트"""

    def test_creation(self):
        """생성"""
        from src.data.weather_api import RateLimiter

        limiter = RateLimiter(max_calls=10, period=1.0)

        assert limiter.max_calls == 10
        assert limiter.period == 1.0

    def test_acquire(self):
        """호출 권한 획득"""
        from src.data.weather_api import RateLimiter

        limiter = RateLimiter(max_calls=5, period=1.0)

        for _ in range(3):
            limiter.acquire()

        assert len(limiter.calls) == 3

    def test_reset(self):
        """초기화"""
        from src.data.weather_api import RateLimiter

        limiter = RateLimiter(max_calls=5, period=1.0)
        limiter.acquire()
        limiter.acquire()
        limiter.reset()

        assert len(limiter.calls) == 0


class TestWeatherCache:
    """WeatherCache 테스트"""

    def test_creation(self, tmp_path):
        """생성"""
        from src.data.weather_api import WeatherCache

        cache = WeatherCache(cache_dir=tmp_path, ttl=3600)

        assert cache.ttl == 3600

    def test_set_get_memory(self):
        """메모리 캐시"""
        from src.data.weather_api import WeatherCache

        cache = WeatherCache(ttl=3600)
        params = {'location': 'jeju', 'type': 'current'}
        data = {'temperature': 22.5}

        cache.set(params, data)
        result = cache.get(params)

        assert result == data

    def test_set_get_file(self, tmp_path):
        """파일 캐시"""
        from src.data.weather_api import WeatherCache

        cache = WeatherCache(cache_dir=tmp_path, ttl=3600)
        params = {'location': 'jeju'}
        data = {'temperature': 22.5}

        cache.set(params, data)

        # 새 캐시 인스턴스 (메모리 캐시 없음)
        cache2 = WeatherCache(cache_dir=tmp_path, ttl=3600)
        result = cache2.get(params)

        assert result == data

    def test_cache_miss(self):
        """캐시 미스"""
        from src.data.weather_api import WeatherCache

        cache = WeatherCache(ttl=3600)
        result = cache.get({'unknown': 'params'})

        assert result is None

    def test_clear(self, tmp_path):
        """캐시 초기화"""
        from src.data.weather_api import WeatherCache

        cache = WeatherCache(cache_dir=tmp_path, ttl=3600)
        cache.set({'key': 'value'}, {'data': 1})
        cache.clear()

        assert cache.get({'key': 'value'}) is None


class TestKMAClient:
    """KMAClient 테스트"""

    def test_creation(self):
        """생성"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')

        assert client.config.api_key == 'test_key'

    def test_jeju_stations(self):
        """제주 관측소 정보"""
        from src.data.weather_api import KMAClient

        assert 'jeju' in KMAClient.JEJU_STATIONS
        assert 'seogwipo' in KMAClient.JEJU_STATIONS

    def test_get_current_observation(self):
        """현재 관측 데이터"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')
        obs = client.get_current_observation('jeju')

        assert obs.location == 'jeju'
        assert 'temperature' in obs.variables
        assert obs.source == 'KMA_ASOS'

    def test_get_historical_observations(self):
        """과거 관측 데이터"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')
        end = datetime.now()
        start = end - timedelta(hours=6)

        observations = client.get_historical_observations('jeju', start, end)

        assert len(observations) > 0
        assert all(obs.source == 'KMA_ASOS' for obs in observations)

    def test_get_forecast(self):
        """예보 데이터"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')
        forecasts = client.get_forecast('jeju', hours_ahead=12)

        assert len(forecasts) == 12
        assert all(fc.source == 'KMA_VillageForcast' for fc in forecasts)

    def test_get_asos_data_hourly(self):
        """ASOS 시간별 데이터"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')
        df = client.get_asos_data('184', datetime.now(), 'hourly')

        assert isinstance(df, pd.DataFrame)
        assert 'temperature' in df.columns
        assert len(df) == 24

    def test_get_asos_data_daily(self):
        """ASOS 일별 데이터"""
        from src.data.weather_api import KMAClient

        client = KMAClient(api_key='test_key')
        df = client.get_asos_data('184', datetime.now(), 'daily')

        assert isinstance(df, pd.DataFrame)
        assert 'temperature_max' in df.columns


class TestOpenWeatherClient:
    """OpenWeatherClient 테스트"""

    def test_creation(self):
        """생성"""
        from src.data.weather_api import OpenWeatherClient

        client = OpenWeatherClient(api_key='test_key')

        assert client.config.api_key == 'test_key'

    def test_jeju_coords(self):
        """제주 좌표"""
        from src.data.weather_api import OpenWeatherClient

        coords = OpenWeatherClient.JEJU_COORDS

        assert 'lat' in coords
        assert 'lon' in coords

    def test_get_current_observation(self):
        """현재 관측 데이터"""
        from src.data.weather_api import OpenWeatherClient

        client = OpenWeatherClient(api_key='test_key')
        obs = client.get_current_observation()

        assert obs.source == 'OpenWeather'
        assert 'temperature' in obs.variables

    def test_get_historical_observations(self):
        """과거 관측 데이터"""
        from src.data.weather_api import OpenWeatherClient

        client = OpenWeatherClient(api_key='test_key')
        end = datetime.now()
        start = end - timedelta(hours=3)

        observations = client.get_historical_observations('jeju', start, end)

        assert len(observations) > 0

    def test_get_forecast(self):
        """예보 데이터"""
        from src.data.weather_api import OpenWeatherClient

        client = OpenWeatherClient(api_key='test_key')
        forecasts = client.get_forecast(hours_ahead=24)

        assert len(forecasts) == 24
        assert all(fc.source == 'OpenWeather_Forecast' for fc in forecasts)


class TestWeatherDataFetcher:
    """WeatherDataFetcher 테스트"""

    def test_creation(self):
        """생성"""
        from src.data.weather_api import WeatherDataFetcher, KMAClient

        clients = {'kma': KMAClient('test_key')}
        fetcher = WeatherDataFetcher(clients)

        assert fetcher._primary_source == 'kma'

    def test_set_primary_source(self):
        """기본 소스 설정"""
        from src.data.weather_api import (
            WeatherDataFetcher, KMAClient, OpenWeatherClient
        )

        clients = {
            'kma': KMAClient('test'),
            'openweather': OpenWeatherClient('test')
        }
        fetcher = WeatherDataFetcher(clients)

        fetcher.set_primary_source('openweather')

        assert fetcher._primary_source == 'openweather'

    def test_fetch_current(self):
        """현재 데이터 수집"""
        from src.data.weather_api import WeatherDataFetcher, KMAClient

        clients = {'kma': KMAClient('test_key')}
        fetcher = WeatherDataFetcher(clients)

        results = fetcher.fetch_current('jeju')

        assert 'kma' in results
        assert results['kma'].location == 'jeju'

    def test_fetch_historical(self):
        """과거 데이터 수집"""
        from src.data.weather_api import WeatherDataFetcher, KMAClient

        clients = {'kma': KMAClient('test_key')}
        fetcher = WeatherDataFetcher(clients)

        end = datetime.now()
        start = end - timedelta(hours=3)
        observations = fetcher.fetch_historical('jeju', start, end)

        assert len(observations) > 0

    def test_fetch_forecast(self):
        """예보 수집"""
        from src.data.weather_api import WeatherDataFetcher, KMAClient

        clients = {'kma': KMAClient('test_key')}
        fetcher = WeatherDataFetcher(clients)

        results = fetcher.fetch_forecast('jeju', hours_ahead=12)

        assert 'kma' in results
        assert len(results['kma']) == 12

    def test_to_dataframe(self):
        """DataFrame 변환"""
        from src.data.weather_api import WeatherDataFetcher, KMAClient

        clients = {'kma': KMAClient('test_key')}
        fetcher = WeatherDataFetcher(clients)

        observations = fetcher.fetch_historical()
        df = fetcher.to_dataframe(observations)

        assert isinstance(df, pd.DataFrame)
        assert 'temperature' in df.columns

    def test_merge_sources_average(self):
        """소스 병합 (평균)"""
        from src.data.weather_api import (
            WeatherDataFetcher, KMAClient, OpenWeatherClient
        )

        clients = {
            'kma': KMAClient('test'),
            'openweather': OpenWeatherClient('test')
        }
        fetcher = WeatherDataFetcher(clients)

        observations = fetcher.fetch_current('jeju')
        merged = fetcher.merge_sources(observations, strategy='average')

        assert merged.source == 'merged'
        assert 'temperature' in merged.variables

    def test_merge_sources_primary(self):
        """소스 병합 (기본 소스)"""
        from src.data.weather_api import (
            WeatherDataFetcher, KMAClient, OpenWeatherClient
        )

        clients = {
            'kma': KMAClient('test'),
            'openweather': OpenWeatherClient('test')
        }
        fetcher = WeatherDataFetcher(clients)

        observations = fetcher.fetch_current('jeju')
        merged = fetcher.merge_sources(observations, strategy='primary')

        assert merged.source == 'KMA_ASOS'


class TestWeatherDataPipeline:
    """WeatherDataPipeline 테스트"""

    def test_creation(self, tmp_path):
        """생성"""
        from src.data.weather_api import (
            WeatherDataPipeline, WeatherDataFetcher, KMAClient
        )

        clients = {'kma': KMAClient('test')}
        fetcher = WeatherDataFetcher(clients)
        pipeline = WeatherDataPipeline(fetcher, tmp_path)

        assert pipeline.output_dir.exists()

    def test_add_transformer(self):
        """변환 함수 추가"""
        from src.data.weather_api import (
            WeatherDataPipeline, WeatherDataFetcher, KMAClient
        )

        clients = {'kma': KMAClient('test')}
        fetcher = WeatherDataFetcher(clients)
        pipeline = WeatherDataPipeline(fetcher)

        def sample_transformer(df):
            df['transformed'] = True
            return df

        pipeline.add_transformer(sample_transformer)

        assert len(pipeline._transformers) == 1

    def test_run(self, tmp_path):
        """파이프라인 실행"""
        from src.data.weather_api import (
            WeatherDataPipeline, WeatherDataFetcher, KMAClient
        )

        clients = {'kma': KMAClient('test')}
        fetcher = WeatherDataFetcher(clients)
        pipeline = WeatherDataPipeline(fetcher, tmp_path)

        end = datetime.now()
        start = end - timedelta(hours=3)
        df = pipeline.run('jeju', start, end, include_forecast=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_run_with_transformer(self):
        """변환 함수와 함께 실행"""
        from src.data.weather_api import (
            WeatherDataPipeline, WeatherDataFetcher, KMAClient
        )

        clients = {'kma': KMAClient('test')}
        fetcher = WeatherDataFetcher(clients)
        pipeline = WeatherDataPipeline(fetcher)

        def add_column(df):
            df['new_col'] = 1
            return df

        pipeline.add_transformer(add_column)
        df = pipeline.run(include_forecast=False)

        assert 'new_col' in df.columns


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_kma_client(self):
        """KMA 클라이언트 생성"""
        from src.data.weather_api import create_kma_client

        client = create_kma_client('test_key')

        assert client is not None
        assert client.config.api_key == 'test_key'

    def test_create_openweather_client(self):
        """OpenWeather 클라이언트 생성"""
        from src.data.weather_api import create_openweather_client

        client = create_openweather_client('test_key')

        assert client is not None

    def test_create_weather_fetcher(self):
        """데이터 수집기 생성"""
        from src.data.weather_api import create_weather_fetcher

        fetcher = create_weather_fetcher(kma_api_key='test')

        assert fetcher is not None
        assert 'kma' in fetcher.clients

    def test_create_weather_fetcher_both(self):
        """양쪽 API로 데이터 수집기 생성"""
        from src.data.weather_api import create_weather_fetcher

        fetcher = create_weather_fetcher(
            kma_api_key='test1',
            openweather_api_key='test2'
        )

        assert 'kma' in fetcher.clients
        assert 'openweather' in fetcher.clients

    def test_create_weather_fetcher_default(self):
        """기본 데이터 수집기 생성"""
        from src.data.weather_api import create_weather_fetcher

        fetcher = create_weather_fetcher()

        assert fetcher is not None
        assert len(fetcher.clients) > 0

    def test_create_weather_pipeline(self):
        """파이프라인 생성"""
        from src.data.weather_api import create_weather_pipeline

        pipeline = create_weather_pipeline()

        assert pipeline is not None
        assert pipeline.fetcher is not None


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self, tmp_path):
        """전체 워크플로우"""
        from src.data.weather_api import (
            create_weather_fetcher, create_weather_pipeline
        )

        # 1. 데이터 수집기 생성
        fetcher = create_weather_fetcher(kma_api_key='test')

        # 2. 현재 데이터 수집
        current = fetcher.fetch_current('jeju')
        assert 'kma' in current

        # 3. 과거 데이터 수집
        end = datetime.now()
        start = end - timedelta(hours=6)
        historical = fetcher.fetch_historical('jeju', start, end)
        assert len(historical) > 0

        # 4. 예보 수집
        forecast = fetcher.fetch_forecast('jeju', hours_ahead=24)
        assert 'kma' in forecast

        # 5. DataFrame 변환
        df = fetcher.to_dataframe(historical)
        assert isinstance(df, pd.DataFrame)

        # 6. 파이프라인 실행
        pipeline = create_weather_pipeline(fetcher, tmp_path)
        result_df = pipeline.run('jeju', start, end)
        assert len(result_df) > 0

    def test_multi_source_workflow(self):
        """다중 소스 워크플로우"""
        from src.data.weather_api import create_weather_fetcher

        fetcher = create_weather_fetcher(
            kma_api_key='test1',
            openweather_api_key='test2'
        )

        # 모든 소스에서 현재 데이터 수집
        current_all = fetcher.fetch_current('jeju')
        assert len(current_all) == 2

        # 소스 병합
        merged = fetcher.merge_sources(current_all, strategy='average')
        assert merged.source == 'merged'
        assert 'kma' in merged.metadata['sources']
        assert 'openweather' in merged.metadata['sources']

