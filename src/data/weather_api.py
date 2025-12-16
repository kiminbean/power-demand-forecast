"""
실시간 기상 API 연동 모듈 (Task 7)
==================================
다양한 기상 API를 통합하여 실시간 기상 데이터를 수집합니다.

지원 API:
- 기상청 공공데이터 API (KMA)
- OpenWeather API
- 기상청 ASOS/AWS

주요 컴포넌트:
- WeatherAPIClient: API 클라이언트 베이스 클래스
- KMAClient: 기상청 API 클라이언트
- OpenWeatherClient: OpenWeather API 클라이언트
- WeatherDataFetcher: 통합 데이터 수집기
- WeatherCache: 캐싱 시스템
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import json
import time
import hashlib
import logging
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class WeatherVariable(Enum):
    """기상 변수 타입"""
    TEMPERATURE = 'temperature'           # 기온 (°C)
    HUMIDITY = 'humidity'                  # 습도 (%)
    WIND_SPEED = 'wind_speed'             # 풍속 (m/s)
    WIND_DIRECTION = 'wind_direction'     # 풍향 (°)
    PRECIPITATION = 'precipitation'        # 강수량 (mm)
    PRESSURE = 'pressure'                  # 기압 (hPa)
    CLOUD_COVER = 'cloud_cover'           # 운량 (%)
    SOLAR_RADIATION = 'solar_radiation'   # 일사량 (MJ/m²)
    DEWPOINT = 'dewpoint'                  # 이슬점 (°C)
    VISIBILITY = 'visibility'              # 시정 (m)


@dataclass
class WeatherObservation:
    """기상 관측 데이터"""
    timestamp: datetime
    location: str
    variables: Dict[str, float]
    source: str
    quality_flag: str = 'good'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'variables': self.variables,
            'source': self.source,
            'quality_flag': self.quality_flag,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeatherObservation':
        """딕셔너리에서 생성"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            location=data['location'],
            variables=data['variables'],
            source=data['source'],
            quality_flag=data.get('quality_flag', 'good'),
            metadata=data.get('metadata', {})
        )


@dataclass
class WeatherForecast:
    """기상 예보 데이터"""
    issue_time: datetime
    valid_time: datetime
    location: str
    variables: Dict[str, float]
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'issue_time': self.issue_time.isoformat(),
            'valid_time': self.valid_time.isoformat(),
            'location': self.location,
            'variables': self.variables,
            'source': self.source,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeatherForecast':
        """딕셔너리에서 생성"""
        return cls(
            issue_time=datetime.fromisoformat(data['issue_time']),
            valid_time=datetime.fromisoformat(data['valid_time']),
            location=data['location'],
            variables=data['variables'],
            source=data['source'],
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class APIConfig:
    """API 설정"""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 60  # requests per minute
    headers: Dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """API 호출 속도 제한"""

    def __init__(self, max_calls: int, period: float = 60.0):
        """
        Args:
            max_calls: 기간 내 최대 호출 수
            period: 기간 (초)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []

    def acquire(self) -> None:
        """호출 권한 획득 (필요시 대기)"""
        now = time.time()

        # 오래된 호출 기록 제거
        self.calls = [t for t in self.calls if now - t < self.period]

        if len(self.calls) >= self.max_calls:
            # 대기 시간 계산
            wait_time = self.period - (now - self.calls[0])
            if wait_time > 0:
                logger.debug(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        self.calls.append(time.time())

    def reset(self) -> None:
        """호출 기록 초기화"""
        self.calls = []


class WeatherCache:
    """기상 데이터 캐시"""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: int = 3600,  # 1시간
        max_size: int = 1000
    ):
        """
        Args:
            cache_dir: 캐시 디렉토리
            ttl: 캐시 유효 시간 (초)
            max_size: 최대 캐시 항목 수
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl = ttl
        self.max_size = max_size
        self._memory_cache: Dict[str, tuple] = {}  # key: (data, timestamp)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def get(self, params: Dict[str, Any]) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        key = self._generate_key(params)

        # 메모리 캐시 확인
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache hit (memory): {key[:8]}...")
                return data
            else:
                del self._memory_cache[key]

        # 파일 캐시 확인
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    if time.time() - cached['timestamp'] < self.ttl:
                        logger.debug(f"Cache hit (file): {key[:8]}...")
                        return cached['data']
                    else:
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")

        return None

    def set(self, params: Dict[str, Any], data: Any) -> None:
        """캐시에 데이터 저장"""
        key = self._generate_key(params)
        timestamp = time.time()

        # 메모리 캐시 저장
        self._memory_cache[key] = (data, timestamp)

        # 캐시 크기 제한
        if len(self._memory_cache) > self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self._memory_cache.keys(),
                           key=lambda k: self._memory_cache[k][1])
            del self._memory_cache[oldest_key]

        # 파일 캐시 저장
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'data': data, 'timestamp': timestamp}, f)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

    def clear(self) -> None:
        """캐시 초기화"""
        self._memory_cache.clear()
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


class WeatherAPIClient(ABC):
    """기상 API 클라이언트 베이스 클래스"""

    def __init__(self, config: APIConfig):
        """
        Args:
            config: API 설정
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self._session = None

    @abstractmethod
    def get_current_observation(
        self,
        location: str,
        variables: Optional[List[WeatherVariable]] = None
    ) -> WeatherObservation:
        """현재 관측 데이터 조회"""
        pass

    @abstractmethod
    def get_historical_observations(
        self,
        location: str,
        start_time: datetime,
        end_time: datetime,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherObservation]:
        """과거 관측 데이터 조회"""
        pass

    @abstractmethod
    def get_forecast(
        self,
        location: str,
        hours_ahead: int = 24,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherForecast]:
        """예보 데이터 조회"""
        pass

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        method: str = 'GET'
    ) -> Dict[str, Any]:
        """API 요청 수행"""
        self.rate_limiter.acquire()

        # 실제 구현에서는 requests 라이브러리 사용
        # 여기서는 시뮬레이션용 더미 응답 반환
        logger.debug(f"API request: {endpoint} with params {params}")

        return {'status': 'success', 'data': {}}


class KMAClient(WeatherAPIClient):
    """
    기상청 API 클라이언트

    기상청 공공데이터 API를 통해 기상 데이터를 조회합니다.
    - ASOS (종관기상관측)
    - AWS (자동기상관측)
    - 동네예보
    - 중기예보
    """

    # 제주 관측소 정보
    JEJU_STATIONS = {
        'jeju': {'stn_id': '184', 'name': '제주'},
        'seogwipo': {'stn_id': '189', 'name': '서귀포'},
        'seongsan': {'stn_id': '188', 'name': '성산'},
        'gosan': {'stn_id': '185', 'name': '고산'},
    }

    # 동네예보 격자 좌표 (제주)
    JEJU_GRID = {'nx': 53, 'ny': 38}

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        """
        Args:
            api_key: 기상청 API 키
            cache_dir: 캐시 디렉토리
        """
        config = APIConfig(
            base_url='http://apis.data.go.kr/1360000',
            api_key=api_key,
            rate_limit=100
        )
        super().__init__(config)
        self.cache = WeatherCache(cache_dir) if cache_dir else None

    def get_current_observation(
        self,
        location: str = 'jeju',
        variables: Optional[List[WeatherVariable]] = None
    ) -> WeatherObservation:
        """현재 관측 데이터 조회 (ASOS)"""
        station = self.JEJU_STATIONS.get(location, self.JEJU_STATIONS['jeju'])
        now = datetime.now()

        # 캐시 확인
        cache_params = {'type': 'current', 'location': location}
        if self.cache:
            cached = self.cache.get(cache_params)
            if cached:
                return WeatherObservation.from_dict(cached)

        # API 호출 시뮬레이션 (실제 구현에서는 requests 사용)
        observation = WeatherObservation(
            timestamp=now,
            location=location,
            variables={
                'temperature': 22.5,
                'humidity': 65.0,
                'wind_speed': 3.2,
                'wind_direction': 180.0,
                'precipitation': 0.0,
                'pressure': 1013.25,
            },
            source='KMA_ASOS',
            metadata={'station_id': station['stn_id']}
        )

        # 캐시 저장
        if self.cache:
            self.cache.set(cache_params, observation.to_dict())

        return observation

    def get_historical_observations(
        self,
        location: str = 'jeju',
        start_time: datetime = None,
        end_time: datetime = None,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherObservation]:
        """과거 관측 데이터 조회"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()

        observations = []
        current = start_time

        while current <= end_time:
            obs = WeatherObservation(
                timestamp=current,
                location=location,
                variables={
                    'temperature': 20.0 + np.random.randn() * 3,
                    'humidity': 60.0 + np.random.randn() * 10,
                    'wind_speed': 3.0 + np.random.randn() * 1,
                },
                source='KMA_ASOS'
            )
            observations.append(obs)
            current += timedelta(hours=1)

        return observations

    def get_forecast(
        self,
        location: str = 'jeju',
        hours_ahead: int = 24,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherForecast]:
        """동네예보 조회"""
        now = datetime.now()
        forecasts = []

        for h in range(hours_ahead):
            valid_time = now + timedelta(hours=h)
            forecast = WeatherForecast(
                issue_time=now,
                valid_time=valid_time,
                location=location,
                variables={
                    'temperature': 22.0 + np.sin(h * np.pi / 12) * 5,
                    'humidity': 65.0 + np.random.randn() * 5,
                    'precipitation_probability': max(0, min(100, 30 + np.random.randn() * 20)),
                },
                source='KMA_VillageForcast',
                confidence=1.0 - h * 0.02  # 시간이 지날수록 신뢰도 감소
            )
            forecasts.append(forecast)

        return forecasts

    def get_asos_data(
        self,
        station_id: str,
        date: datetime,
        data_type: str = 'hourly'
    ) -> pd.DataFrame:
        """
        ASOS 데이터 조회

        Args:
            station_id: 관측소 ID
            date: 조회 날짜
            data_type: 'hourly' 또는 'daily'

        Returns:
            DataFrame with weather observations
        """
        # 시뮬레이션 데이터 생성
        if data_type == 'hourly':
            hours = range(24)
            data = {
                'datetime': [datetime(date.year, date.month, date.day, h) for h in hours],
                'temperature': [20.0 + np.sin(h * np.pi / 12) * 5 + np.random.randn() for h in hours],
                'humidity': [60.0 + np.random.randn() * 10 for _ in hours],
                'wind_speed': [3.0 + np.random.randn() * 1 for _ in hours],
                'precipitation': [max(0, np.random.randn() * 2) for _ in hours],
            }
        else:
            data = {
                'date': [date],
                'temperature_max': [25.0 + np.random.randn() * 2],
                'temperature_min': [18.0 + np.random.randn() * 2],
                'temperature_avg': [21.5 + np.random.randn()],
                'humidity_avg': [65.0 + np.random.randn() * 5],
                'precipitation': [max(0, np.random.randn() * 5)],
            }

        return pd.DataFrame(data)


class OpenWeatherClient(WeatherAPIClient):
    """
    OpenWeather API 클라이언트

    OpenWeather API를 통해 전 세계 기상 데이터를 조회합니다.
    - Current Weather Data
    - Hourly Forecast (48시간)
    - Daily Forecast (7일)
    - Historical Data
    """

    # 제주 좌표
    JEJU_COORDS = {'lat': 33.5, 'lon': 126.5}

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        """
        Args:
            api_key: OpenWeather API 키
            cache_dir: 캐시 디렉토리
        """
        config = APIConfig(
            base_url='https://api.openweathermap.org/data/2.5',
            api_key=api_key,
            rate_limit=60
        )
        super().__init__(config)
        self.cache = WeatherCache(cache_dir) if cache_dir else None

    def get_current_observation(
        self,
        location: str = 'jeju',
        variables: Optional[List[WeatherVariable]] = None
    ) -> WeatherObservation:
        """현재 기상 데이터 조회"""
        now = datetime.now()

        # 캐시 확인
        cache_params = {'type': 'current', 'location': location}
        if self.cache:
            cached = self.cache.get(cache_params)
            if cached:
                return WeatherObservation.from_dict(cached)

        # API 응답 시뮬레이션
        observation = WeatherObservation(
            timestamp=now,
            location=location,
            variables={
                'temperature': 23.0,
                'humidity': 62.0,
                'wind_speed': 3.5,
                'wind_direction': 225.0,
                'pressure': 1012.0,
                'cloud_cover': 40.0,
            },
            source='OpenWeather',
            metadata={'coords': self.JEJU_COORDS}
        )

        if self.cache:
            self.cache.set(cache_params, observation.to_dict())

        return observation

    def get_historical_observations(
        self,
        location: str = 'jeju',
        start_time: datetime = None,
        end_time: datetime = None,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherObservation]:
        """과거 데이터 조회 (One Call API Historical)"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()

        observations = []
        current = start_time

        while current <= end_time:
            obs = WeatherObservation(
                timestamp=current,
                location=location,
                variables={
                    'temperature': 21.0 + np.random.randn() * 3,
                    'humidity': 58.0 + np.random.randn() * 10,
                    'wind_speed': 3.5 + np.random.randn() * 1,
                    'cloud_cover': 40.0 + np.random.randn() * 20,
                },
                source='OpenWeather_Historical'
            )
            observations.append(obs)
            current += timedelta(hours=1)

        return observations

    def get_forecast(
        self,
        location: str = 'jeju',
        hours_ahead: int = 48,
        variables: Optional[List[WeatherVariable]] = None
    ) -> List[WeatherForecast]:
        """48시간 시간별 예보 조회"""
        now = datetime.now()
        forecasts = []

        for h in range(min(hours_ahead, 48)):
            valid_time = now + timedelta(hours=h)
            forecast = WeatherForecast(
                issue_time=now,
                valid_time=valid_time,
                location=location,
                variables={
                    'temperature': 23.0 + np.sin(h * np.pi / 12) * 4,
                    'humidity': 60.0 + np.random.randn() * 5,
                    'wind_speed': 3.5 + np.random.randn() * 0.5,
                    'precipitation_probability': max(0, min(100, 25 + np.random.randn() * 15)),
                },
                source='OpenWeather_Forecast',
                confidence=0.95 - h * 0.01
            )
            forecasts.append(forecast)

        return forecasts


class WeatherDataFetcher:
    """
    통합 기상 데이터 수집기

    여러 API 소스에서 기상 데이터를 수집하고 통합합니다.
    """

    def __init__(
        self,
        clients: Dict[str, WeatherAPIClient],
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            clients: API 클라이언트 딕셔너리
            cache_dir: 캐시 디렉토리
        """
        self.clients = clients
        self.cache = WeatherCache(cache_dir) if cache_dir else None
        self._primary_source = list(clients.keys())[0] if clients else None

    def set_primary_source(self, source: str) -> None:
        """기본 데이터 소스 설정"""
        if source in self.clients:
            self._primary_source = source
        else:
            raise ValueError(f"Unknown source: {source}")

    def fetch_current(
        self,
        location: str = 'jeju',
        sources: Optional[List[str]] = None
    ) -> Dict[str, WeatherObservation]:
        """
        현재 기상 데이터 수집

        Args:
            location: 위치
            sources: 데이터 소스 리스트 (None이면 모든 소스)

        Returns:
            소스별 관측 데이터
        """
        if sources is None:
            sources = list(self.clients.keys())

        results = {}
        for source in sources:
            if source in self.clients:
                try:
                    results[source] = self.clients[source].get_current_observation(location)
                except Exception as e:
                    logger.error(f"Failed to fetch from {source}: {e}")

        return results

    def fetch_historical(
        self,
        location: str = 'jeju',
        start_time: datetime = None,
        end_time: datetime = None,
        source: Optional[str] = None
    ) -> List[WeatherObservation]:
        """
        과거 기상 데이터 수집

        Args:
            location: 위치
            start_time: 시작 시간
            end_time: 종료 시간
            source: 데이터 소스 (None이면 기본 소스)

        Returns:
            관측 데이터 리스트
        """
        if source is None:
            source = self._primary_source

        if source not in self.clients:
            raise ValueError(f"Unknown source: {source}")

        return self.clients[source].get_historical_observations(
            location, start_time, end_time
        )

    def fetch_forecast(
        self,
        location: str = 'jeju',
        hours_ahead: int = 24,
        sources: Optional[List[str]] = None
    ) -> Dict[str, List[WeatherForecast]]:
        """
        예보 데이터 수집

        Args:
            location: 위치
            hours_ahead: 예보 시간
            sources: 데이터 소스 리스트

        Returns:
            소스별 예보 데이터
        """
        if sources is None:
            sources = list(self.clients.keys())

        results = {}
        for source in sources:
            if source in self.clients:
                try:
                    results[source] = self.clients[source].get_forecast(
                        location, hours_ahead
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch forecast from {source}: {e}")

        return results

    def to_dataframe(
        self,
        observations: List[WeatherObservation]
    ) -> pd.DataFrame:
        """관측 데이터를 DataFrame으로 변환"""
        records = []
        for obs in observations:
            record = {
                'timestamp': obs.timestamp,
                'location': obs.location,
                'source': obs.source,
                **obs.variables
            }
            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index('timestamp')
        return df

    def merge_sources(
        self,
        observations: Dict[str, WeatherObservation],
        strategy: str = 'average'
    ) -> WeatherObservation:
        """
        여러 소스의 데이터 병합

        Args:
            observations: 소스별 관측 데이터
            strategy: 병합 전략 ('average', 'median', 'primary')

        Returns:
            병합된 관측 데이터
        """
        if not observations:
            raise ValueError("No observations to merge")

        if strategy == 'primary' and self._primary_source in observations:
            return observations[self._primary_source]

        # 변수별 값 수집
        all_variables: Dict[str, List[float]] = {}
        for obs in observations.values():
            for var, value in obs.variables.items():
                if var not in all_variables:
                    all_variables[var] = []
                all_variables[var].append(value)

        # 병합
        merged_variables = {}
        for var, values in all_variables.items():
            if strategy == 'average':
                merged_variables[var] = np.mean(values)
            elif strategy == 'median':
                merged_variables[var] = np.median(values)
            else:
                merged_variables[var] = values[0]

        first_obs = list(observations.values())[0]
        return WeatherObservation(
            timestamp=first_obs.timestamp,
            location=first_obs.location,
            variables=merged_variables,
            source='merged',
            metadata={'sources': list(observations.keys()), 'strategy': strategy}
        )


class WeatherDataPipeline:
    """
    기상 데이터 파이프라인

    데이터 수집, 정제, 변환을 자동화합니다.
    """

    def __init__(
        self,
        fetcher: WeatherDataFetcher,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            fetcher: 데이터 수집기
            output_dir: 출력 디렉토리
        """
        self.fetcher = fetcher
        self.output_dir = Path(output_dir) if output_dir else None
        self._transformers: List[Callable] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_transformer(self, transformer: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """변환 함수 추가"""
        self._transformers.append(transformer)

    def run(
        self,
        location: str = 'jeju',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_forecast: bool = True
    ) -> pd.DataFrame:
        """
        파이프라인 실행

        Args:
            location: 위치
            start_time: 시작 시간
            end_time: 종료 시간
            include_forecast: 예보 포함 여부

        Returns:
            처리된 데이터
        """
        # 과거 데이터 수집
        observations = self.fetcher.fetch_historical(location, start_time, end_time)
        df = self.fetcher.to_dataframe(observations)

        # 예보 데이터 추가
        if include_forecast:
            forecasts_dict = self.fetcher.fetch_forecast(location)
            for source, forecasts in forecasts_dict.items():
                forecast_records = []
                for fc in forecasts:
                    record = {
                        'timestamp': fc.valid_time,
                        'location': fc.location,
                        'source': fc.source,
                        'is_forecast': True,
                        **fc.variables
                    }
                    forecast_records.append(record)

                if forecast_records:
                    fc_df = pd.DataFrame(forecast_records).set_index('timestamp')
                    df = pd.concat([df, fc_df])

        # 변환 적용
        for transformer in self._transformers:
            df = transformer(df)

        # 저장
        if self.output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f'weather_data_{timestamp}.parquet'
            df.to_parquet(output_file)
            logger.info(f"Saved weather data to {output_file}")

        return df


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_kma_client(
    api_key: str,
    cache_dir: Optional[Path] = None
) -> KMAClient:
    """기상청 API 클라이언트 생성"""
    return KMAClient(api_key, cache_dir)


def create_openweather_client(
    api_key: str,
    cache_dir: Optional[Path] = None
) -> OpenWeatherClient:
    """OpenWeather API 클라이언트 생성"""
    return OpenWeatherClient(api_key, cache_dir)


def create_weather_fetcher(
    kma_api_key: Optional[str] = None,
    openweather_api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> WeatherDataFetcher:
    """
    통합 기상 데이터 수집기 생성

    Args:
        kma_api_key: 기상청 API 키
        openweather_api_key: OpenWeather API 키
        cache_dir: 캐시 디렉토리

    Returns:
        WeatherDataFetcher 인스턴스
    """
    clients = {}

    if kma_api_key:
        clients['kma'] = create_kma_client(kma_api_key, cache_dir)

    if openweather_api_key:
        clients['openweather'] = create_openweather_client(openweather_api_key, cache_dir)

    # API 키 없이도 테스트용 클라이언트 생성
    if not clients:
        clients['kma'] = KMAClient('test_key', cache_dir)

    return WeatherDataFetcher(clients, cache_dir)


def create_weather_pipeline(
    fetcher: Optional[WeatherDataFetcher] = None,
    output_dir: Optional[Path] = None
) -> WeatherDataPipeline:
    """기상 데이터 파이프라인 생성"""
    if fetcher is None:
        fetcher = create_weather_fetcher()

    return WeatherDataPipeline(fetcher, output_dir)
