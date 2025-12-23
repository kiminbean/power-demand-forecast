"""
Real-time Data Client for External Sources

Primary: Web Crawlers (no API key required)
1. JejuRealtimeCrawler - 제주 실시간 전력수급 (KPX 웹페이지 크롤링)
2. SMPCrawler - 실시간 SMP 조회 (KPX 웹페이지 크롤링)
3. KMAWeatherCrawler - 실시간 기상 데이터 (기상청 날씨누리 크롤링)

Fallback: Public Data APIs (API key required)
4. KPX 계통한계가격(SMP) API
5. KPX 현재전력수급현황 API
6. 기상청 초단기실황 API

Sources:
- https://www.kpx.or.kr/powerinfoJeju.es (Jeju realtime - crawl)
- https://new.kpx.or.kr/bidSmpLfdDataRt.es (SMP - crawl)
- https://www.weather.go.kr/w/obs-climate/land/city-obs.do (Weather - crawl)
- https://www.data.go.kr APIs (fallback)
"""

import os
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import crawlers (primary data source)
try:
    from tools.crawlers.jeju_realtime_crawler import JejuRealtimeCrawler
    from src.smp.crawlers.smp_crawler import SMPCrawler
    from tools.crawlers.kma_weather_crawler import KMAWeatherCrawler
    CRAWLERS_AVAILABLE = True
    WEATHER_CRAWLER_AVAILABLE = True
    logger.info("Crawlers loaded successfully - using web scraping for real-time data")
except ImportError as e:
    CRAWLERS_AVAILABLE = False
    WEATHER_CRAWLER_AVAILABLE = False
    logger.warning(f"Crawlers not available, using API fallback: {e}")

# API Configuration
DATA_GO_KR_API_KEY = os.getenv("DATA_GO_KR_API_KEY", "")

# API Endpoints
KPX_SMP_API = "https://openapi.kpx.or.kr/openapi/smp1hToday/getSmp1hToday"
KPX_POWER_SUPPLY_API = "https://openapi.kpx.or.kr/openapi/sukub5mMaxDatetime/getSukub5mMaxDatetime"
KMA_WEATHER_API = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# Jeju Grid Coordinates for KMA API (제주시 기준)
JEJU_NX = 52
JEJU_NY = 38

# Cache TTL (seconds)
SMP_CACHE_TTL = 300  # 5 minutes
POWER_SUPPLY_CACHE_TTL = 60  # 1 minute
WEATHER_CACHE_TTL = 600  # 10 minutes


@dataclass
class SMPData:
    """SMP (System Marginal Price) Data"""
    trade_day: str  # YYYYMMDD
    trade_hour: int  # 1-24
    smp_land: float  # 육지 SMP (원/kWh)
    smp_jeju: float  # 제주 SMP (원/kWh)
    timestamp: datetime
    data_source: str = "KPX Real-time API"


@dataclass
class PowerSupplyData:
    """Real-time Power Supply Data"""
    base_datetime: datetime
    supply_capacity_mw: float  # 공급능력
    current_demand_mw: float  # 현재수요
    forecast_demand_mw: float  # 최대예측수요
    supply_reserve_mw: float  # 공급예비력
    supply_reserve_rate: float  # 공급예비율 (%)
    oper_reserve_mw: float  # 운영예비력
    oper_reserve_rate: float  # 운영예비율 (%)
    data_source: str = "KPX Real-time API"


@dataclass
class WeatherData:
    """Real-time Weather Data from KMA"""
    temperature: float  # 기온 (°C)
    humidity: float  # 습도 (%)
    wind_speed: float  # 풍속 (m/s)
    wind_direction: float  # 풍향 (deg)
    precipitation: float  # 강수량 (mm)
    precipitation_type: int  # 강수형태 (0=없음, 1=비, 2=비/눈, 3=눈)
    base_datetime: datetime
    data_source: str = "KMA Real-time API"
    _weather_condition: Optional[str] = None  # 크롤러에서 가져온 실제 날씨 상태

    @property
    def condition(self) -> str:
        """Get weather condition string"""
        # 크롤러에서 가져온 실제 날씨 상태가 있으면 사용
        if self._weather_condition:
            return self._weather_condition

        # 없으면 강수/습도 기반 추정
        if self.precipitation_type == 1:
            return "비"
        elif self.precipitation_type == 2:
            return "비/눈"
        elif self.precipitation_type == 3:
            return "눈"
        elif self.humidity > 80:
            return "흐림"
        elif self.humidity > 60:
            return "구름많음"
        else:
            return "맑음"


class RealtimeAPIClient:
    """Client for fetching real-time data from external sources

    Primary: Web crawlers (JejuRealtimeCrawler, SMPCrawler)
    Fallback: Public Data APIs (requires API key approval)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DATA_GO_KR_API_KEY
        self.use_crawlers = CRAWLERS_AVAILABLE

        if self.use_crawlers:
            logger.info("Using web crawlers for real-time data (no API key required)")
        elif not self.api_key:
            logger.warning("No API key and crawlers not available. Real-time data will fail.")

        # Cache storage with timestamps
        self._smp_cache: Dict[str, tuple[Any, datetime]] = {}
        self._power_cache: Optional[tuple[PowerSupplyData, datetime]] = None
        self._weather_cache: Optional[tuple[WeatherData, datetime]] = None
        self._jeju_power_cache: Optional[tuple[Any, datetime]] = None  # Jeju specific

    def _is_cache_valid(self, cache_time: Optional[datetime], ttl_seconds: int) -> bool:
        """Check if cache is still valid"""
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < ttl_seconds

    def _parse_xml(self, xml_text: str) -> Optional[ET.Element]:
        """Parse XML response"""
        try:
            return ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return None

    def _get_smp_via_crawler(self) -> Optional[SMPData]:
        """Get SMP data using web crawler (synchronous)"""
        if not self.use_crawlers:
            return None

        try:
            with SMPCrawler() as crawler:
                smp_data = crawler.get_current_smp()
                if smp_data:
                    result = SMPData(
                        trade_day=smp_data.date.replace("-", ""),
                        trade_hour=smp_data.hour,
                        smp_land=smp_data.smp_mainland,
                        smp_jeju=smp_data.smp_jeju,
                        timestamp=datetime.now(),
                        data_source="KPX Web Crawler (실시간)",
                    )
                    logger.info(f"SMP via crawler: Jeju={result.smp_jeju:.2f}, Mainland={result.smp_land:.2f}")
                    return result
        except Exception as e:
            logger.warning(f"SMP crawler failed: {e}")
        return None

    def _get_jeju_power_via_crawler(self) -> Optional[PowerSupplyData]:
        """Get Jeju power supply data using web crawler (synchronous)"""
        if not self.use_crawlers:
            return None

        # Check cache
        if self._jeju_power_cache:
            data, cache_time = self._jeju_power_cache
            if self._is_cache_valid(cache_time, POWER_SUPPLY_CACHE_TTL):
                logger.debug("Using cached Jeju power data from crawler")
                return data

        try:
            with JejuRealtimeCrawler() as crawler:
                jeju_data = crawler.fetch_realtime()
                if jeju_data:
                    # Parse timestamp
                    try:
                        base_dt = datetime.strptime(jeju_data.timestamp, "%Y-%m-%d %H:%M")
                    except ValueError:
                        base_dt = datetime.now()

                    result = PowerSupplyData(
                        base_datetime=base_dt,
                        supply_capacity_mw=jeju_data.supply_capacity,
                        current_demand_mw=jeju_data.current_demand,
                        forecast_demand_mw=jeju_data.current_demand * 1.05,  # 예상
                        supply_reserve_mw=jeju_data.supply_reserve,
                        supply_reserve_rate=jeju_data.reserve_rate,
                        oper_reserve_mw=jeju_data.operation_reserve,
                        oper_reserve_rate=(jeju_data.operation_reserve / jeju_data.current_demand * 100) if jeju_data.current_demand > 0 else 0,
                        data_source="KPX Jeju Crawler (실시간)",
                    )

                    # Update cache
                    self._jeju_power_cache = (result, datetime.now())
                    logger.info(f"Jeju power via crawler: {result.current_demand_mw:.0f} MW demand, {result.supply_reserve_rate:.1f}% reserve")
                    return result
        except Exception as e:
            logger.warning(f"Jeju power crawler failed: {e}")
        return None

    def _get_weather_via_crawler(self) -> Optional[WeatherData]:
        """Get weather data using KMA web crawler (synchronous)"""
        if not WEATHER_CRAWLER_AVAILABLE:
            return None

        # Check cache
        if self._weather_cache:
            data, cache_time = self._weather_cache
            if self._is_cache_valid(cache_time, WEATHER_CACHE_TTL):
                logger.debug("Using cached weather data from crawler")
                return data

        try:
            with KMAWeatherCrawler() as crawler:
                kma_data = crawler.fetch_jeju_weather()
                if kma_data:
                    # Parse timestamp
                    try:
                        base_dt = datetime.strptime(kma_data.timestamp, "%Y-%m-%d %H:%M")
                    except ValueError:
                        base_dt = datetime.now()

                    # Convert wind direction from Korean to degrees
                    wind_deg = self._convert_wind_direction(kma_data.wind_direction)

                    result = WeatherData(
                        temperature=kma_data.temperature,
                        humidity=kma_data.humidity or 60.0,
                        wind_speed=kma_data.wind_speed or 3.0,
                        wind_direction=wind_deg,
                        precipitation=kma_data.precipitation or 0.0,
                        precipitation_type=0,  # Not available from crawler
                        base_datetime=base_dt,
                        data_source="KMA Weather Crawler (실시간)",
                        _weather_condition=kma_data.weather_condition,  # 실제 날씨 상태
                    )

                    # Update cache
                    self._weather_cache = (result, datetime.now())
                    logger.info(f"Weather via crawler: {result.temperature}°C, {result.humidity}% humidity, {result.condition}")
                    return result
        except Exception as e:
            logger.warning(f"Weather crawler failed: {e}")
        return None

    def _convert_wind_direction(self, direction: Optional[str]) -> float:
        """Convert Korean wind direction to degrees"""
        if not direction:
            return 0.0

        direction_map = {
            "북": 0.0, "N": 0.0,
            "북북동": 22.5, "NNE": 22.5,
            "북동": 45.0, "NE": 45.0,
            "동북동": 67.5, "ENE": 67.5,
            "동": 90.0, "E": 90.0,
            "동남동": 112.5, "ESE": 112.5,
            "남동": 135.0, "SE": 135.0,
            "남남동": 157.5, "SSE": 157.5,
            "남": 180.0, "S": 180.0,
            "남남서": 202.5, "SSW": 202.5,
            "남서": 225.0, "SW": 225.0,
            "서남서": 247.5, "WSW": 247.5,
            "서": 270.0, "W": 270.0,
            "서북서": 292.5, "WNW": 292.5,
            "북서": 315.0, "NW": 315.0,
            "북북서": 337.5, "NNW": 337.5,
        }
        return direction_map.get(direction, 0.0)

    async def get_smp_realtime(self, area_code: str = "9") -> Optional[SMPData]:
        """
        Get real-time SMP data

        Primary: Web crawler (no API key required)
        Fallback: KPX API (requires approval)

        Args:
            area_code: "1" for 육지(mainland), "9" for 제주(Jeju)

        Returns:
            SMPData object or None if failed
        """
        cache_key = f"smp_{area_code}"

        # Check cache
        if cache_key in self._smp_cache:
            data, cache_time = self._smp_cache[cache_key]
            if self._is_cache_valid(cache_time, SMP_CACHE_TTL):
                logger.debug(f"Using cached SMP data for area {area_code}")
                return data

        # Try crawler first (primary source)
        if self.use_crawlers:
            smp_data = self._get_smp_via_crawler()
            if smp_data:
                self._smp_cache[cache_key] = (smp_data, datetime.now())
                return smp_data

        # Fallback to API
        try:
            params = {
                "ServiceKey": self.api_key,
                "areaCd": area_code,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(KPX_SMP_API, params=params)
                response.raise_for_status()

            root = self._parse_xml(response.text)
            if root is None:
                return None

            # Check result code
            result_code = root.findtext(".//resultCode")
            if result_code != "00":
                result_msg = root.findtext(".//resultMsg", "Unknown error")
                logger.error(f"SMP API error: {result_code} - {result_msg}")
                return None

            # Parse items (get the latest one)
            items = root.findall(".//item")
            if not items:
                logger.warning("No SMP data items found")
                return None

            # Get the latest item (usually sorted by hour)
            latest_item = items[-1]

            trade_day = latest_item.findtext("tradeDay", "")
            trade_hour = int(latest_item.findtext("tradHour", "0"))
            smp_value = float(latest_item.findtext("smp", "0"))

            # For complete data, we need both land and Jeju
            smp_land = smp_value if area_code == "1" else 0.0
            smp_jeju = smp_value if area_code == "9" else 0.0

            smp_data = SMPData(
                trade_day=trade_day,
                trade_hour=trade_hour,
                smp_land=smp_land,
                smp_jeju=smp_jeju,
                timestamp=datetime.now(),
            )

            # Update cache
            self._smp_cache[cache_key] = (smp_data, datetime.now())
            logger.info(f"Fetched real-time SMP for area {area_code}: {smp_value} 원/kWh")

            return smp_data

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching SMP: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching SMP: {e}")
            return None

    async def get_power_supply_realtime(self, prefer_jeju: bool = True) -> Optional[PowerSupplyData]:
        """
        Get real-time power supply status

        Primary: Jeju crawler (제주 실시간 데이터, no API key required)
        Fallback: KPX nationwide API (requires approval)

        Args:
            prefer_jeju: If True, use Jeju-specific crawler (default)

        Returns:
            PowerSupplyData object or None if failed
        """
        # Try Jeju crawler first (primary for Jeju dashboard)
        if prefer_jeju and self.use_crawlers:
            jeju_data = self._get_jeju_power_via_crawler()
            if jeju_data:
                return jeju_data

        # Check cache for nationwide data
        if self._power_cache:
            data, cache_time = self._power_cache
            if self._is_cache_valid(cache_time, POWER_SUPPLY_CACHE_TTL):
                logger.debug("Using cached power supply data")
                return data

        # Fallback to nationwide API
        try:
            params = {
                "ServiceKey": self.api_key,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(KPX_POWER_SUPPLY_API, params=params)
                response.raise_for_status()

            root = self._parse_xml(response.text)
            if root is None:
                return None

            # Check result code
            result_code = root.findtext(".//resultCode")
            if result_code != "00":
                result_msg = root.findtext(".//resultMsg", "Unknown error")
                logger.error(f"Power supply API error: {result_code} - {result_msg}")
                return None

            # Parse data
            item = root.find(".//item")
            if item is None:
                logger.warning("No power supply data item found")
                return None

            base_datetime_str = item.findtext("baseDatetime", "")
            try:
                base_datetime = datetime.strptime(base_datetime_str, "%Y%m%d%H%M%S")
            except ValueError:
                base_datetime = datetime.now()

            power_data = PowerSupplyData(
                base_datetime=base_datetime,
                supply_capacity_mw=float(item.findtext("suppAbility", "0")),
                current_demand_mw=float(item.findtext("currPwrTot", "0")),
                forecast_demand_mw=float(item.findtext("forecastLoad", "0")),
                supply_reserve_mw=float(item.findtext("suppReservePwr", "0")),
                supply_reserve_rate=float(item.findtext("suppReserveRate", "0")),
                oper_reserve_mw=float(item.findtext("operReservePwr", "0")),
                oper_reserve_rate=float(item.findtext("operReserveRate", "0")),
            )

            # Update cache
            self._power_cache = (power_data, datetime.now())
            logger.info(f"Fetched real-time power supply: {power_data.current_demand_mw} MW demand")

            return power_data

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching power supply: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching power supply: {e}")
            return None

    async def get_weather_realtime(self, nx: int = JEJU_NX, ny: int = JEJU_NY) -> Optional[WeatherData]:
        """
        Get real-time weather data

        Primary: KMA Weather Crawler (no API key required)
        Fallback: KMA API (requires approval)

        Args:
            nx: Grid X coordinate (default: Jeju)
            ny: Grid Y coordinate (default: Jeju)

        Returns:
            WeatherData object or None if failed
        """
        # Check cache first
        if self._weather_cache:
            data, cache_time = self._weather_cache
            if self._is_cache_valid(cache_time, WEATHER_CACHE_TTL):
                logger.debug("Using cached weather data")
                return data

        # Try crawler first (primary source - no API key needed)
        if WEATHER_CRAWLER_AVAILABLE:
            weather_data = self._get_weather_via_crawler()
            if weather_data:
                return weather_data
            logger.debug("Weather crawler failed, trying API fallback")

        # Fallback to API
        try:
            # 초단기실황 API requires base_time to be the most recent hour
            now = datetime.now()
            # Use previous hour's data (API updates at 40 minutes past each hour)
            if now.minute < 40:
                base_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            else:
                base_time = now.replace(minute=0, second=0, microsecond=0)

            params = {
                "ServiceKey": self.api_key,
                "base_date": base_time.strftime("%Y%m%d"),
                "base_time": base_time.strftime("%H00"),
                "nx": nx,
                "ny": ny,
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "XML",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(KMA_WEATHER_API, params=params)
                response.raise_for_status()

            root = self._parse_xml(response.text)
            if root is None:
                return None

            # Check result code
            result_code = root.findtext(".//resultCode")
            if result_code != "00":
                result_msg = root.findtext(".//resultMsg", "Unknown error")
                logger.error(f"Weather API error: {result_code} - {result_msg}")
                return None

            # Parse items
            items = root.findall(".//item")
            if not items:
                logger.warning("No weather data items found")
                return None

            # Initialize with defaults
            temperature = 10.0
            humidity = 60.0
            wind_speed = 3.0
            wind_direction = 0.0
            precipitation = 0.0
            precipitation_type = 0

            # Parse each category
            for item in items:
                category = item.findtext("category", "")
                obs_value = item.findtext("obsrValue", "0")

                try:
                    value = float(obs_value)
                except ValueError:
                    continue

                if category == "T1H":  # 기온
                    temperature = value
                elif category == "REH":  # 습도
                    humidity = value
                elif category == "WSD":  # 풍속
                    wind_speed = value
                elif category == "VEC":  # 풍향
                    wind_direction = value
                elif category == "RN1":  # 1시간 강수량
                    precipitation = value
                elif category == "PTY":  # 강수형태
                    precipitation_type = int(value)

            weather_data = WeatherData(
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                precipitation=precipitation,
                precipitation_type=precipitation_type,
                base_datetime=base_time,
            )

            # Update cache
            self._weather_cache = (weather_data, datetime.now())
            logger.info(f"Fetched real-time weather: {temperature}°C, {humidity}% humidity")

            return weather_data

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None

    async def get_all_realtime_data(self) -> Dict[str, Any]:
        """
        Get all real-time data in a single call

        Returns:
            Dictionary with all available real-time data
        """
        import asyncio

        # Fetch all data concurrently
        smp_jeju, power_supply, weather = await asyncio.gather(
            self.get_smp_realtime(area_code="9"),  # Jeju SMP
            self.get_power_supply_realtime(),
            self.get_weather_realtime(),
            return_exceptions=True
        )

        result = {
            "timestamp": datetime.now().isoformat(),
            "smp_jeju": None,
            "power_supply": None,
            "weather": None,
            "errors": [],
        }

        # Process results
        if isinstance(smp_jeju, SMPData):
            result["smp_jeju"] = {
                "value": smp_jeju.smp_jeju,
                "trade_hour": smp_jeju.trade_hour,
                "data_source": smp_jeju.data_source,
            }
        elif isinstance(smp_jeju, Exception):
            result["errors"].append(f"SMP: {str(smp_jeju)}")

        if isinstance(power_supply, PowerSupplyData):
            result["power_supply"] = {
                "current_demand_mw": power_supply.current_demand_mw,
                "supply_capacity_mw": power_supply.supply_capacity_mw,
                "supply_reserve_rate": power_supply.supply_reserve_rate,
                "base_datetime": power_supply.base_datetime.isoformat(),
                "data_source": power_supply.data_source,
            }
        elif isinstance(power_supply, Exception):
            result["errors"].append(f"Power supply: {str(power_supply)}")

        if isinstance(weather, WeatherData):
            result["weather"] = {
                "temperature": weather.temperature,
                "humidity": weather.humidity,
                "wind_speed": weather.wind_speed,
                "condition": weather.condition,
                "base_datetime": weather.base_datetime.isoformat(),
                "data_source": weather.data_source,
            }
        elif isinstance(weather, Exception):
            result["errors"].append(f"Weather: {str(weather)}")

        return result


# Singleton instance
realtime_client = RealtimeAPIClient()


# Helper functions for direct use
async def get_realtime_smp_jeju() -> Optional[float]:
    """Get current Jeju SMP value"""
    data = await realtime_client.get_smp_realtime(area_code="9")
    return data.smp_jeju if data else None


async def get_realtime_demand() -> Optional[float]:
    """Get current nationwide demand (MW)"""
    data = await realtime_client.get_power_supply_realtime()
    return data.current_demand_mw if data else None


async def get_realtime_weather() -> Optional[Dict[str, Any]]:
    """Get current Jeju weather"""
    data = await realtime_client.get_weather_realtime()
    if data:
        return {
            "temperature": data.temperature,
            "humidity": data.humidity,
            "wind_speed": data.wind_speed,
            "condition": data.condition,
        }
    return None
