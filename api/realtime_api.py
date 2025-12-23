"""
Real-time API Client for External Data Sources

APIs:
1. KPX 계통한계가격(SMP) API - 제주/육지 SMP 실시간 조회
2. KPX 현재전력수급현황 API - 전국 실시간 수급 현황
3. 기상청 초단기실황 API - 실시간 기상 관측 데이터

Sources:
- https://www.data.go.kr/data/15076302/openapi.do (SMP)
- https://www.data.go.kr/data/15056640/openapi.do (전력수급)
- https://www.data.go.kr/data/15084084/openapi.do (기상청)
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

    @property
    def condition(self) -> str:
        """Get weather condition string"""
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
    """Client for fetching real-time data from external APIs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DATA_GO_KR_API_KEY
        if not self.api_key:
            logger.warning("No API key found. Real-time API calls will fail.")

        # Cache storage with timestamps
        self._smp_cache: Dict[str, tuple[Any, datetime]] = {}
        self._power_cache: Optional[tuple[PowerSupplyData, datetime]] = None
        self._weather_cache: Optional[tuple[WeatherData, datetime]] = None

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

    async def get_smp_realtime(self, area_code: str = "9") -> Optional[SMPData]:
        """
        Get real-time SMP data from KPX API

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

    async def get_power_supply_realtime(self) -> Optional[PowerSupplyData]:
        """
        Get real-time power supply status from KPX API

        Note: This is nationwide data, not Jeju-specific

        Returns:
            PowerSupplyData object or None if failed
        """
        # Check cache
        if self._power_cache:
            data, cache_time = self._power_cache
            if self._is_cache_valid(cache_time, POWER_SUPPLY_CACHE_TTL):
                logger.debug("Using cached power supply data")
                return data

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
        Get real-time weather data from KMA API (초단기실황)

        Args:
            nx: Grid X coordinate (default: Jeju)
            ny: Grid Y coordinate (default: Jeju)

        Returns:
            WeatherData object or None if failed
        """
        # Check cache
        if self._weather_cache:
            data, cache_time = self._weather_cache
            if self._is_cache_valid(cache_time, WEATHER_CACHE_TTL):
                logger.debug("Using cached weather data")
                return data

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
