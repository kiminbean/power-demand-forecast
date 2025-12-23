"""
기상청 날씨누리 실시간 기상 크롤러
===================================

기상청 날씨누리(weather.go.kr) 종관기상관측 페이지에서 실시간 기상 데이터 크롤링

데이터 출처: https://www.weather.go.kr/w/obs-climate/land/city-obs.do
업데이트 주기: 1시간 (정시)
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
from bs4 import BeautifulSoup

# 로깅 설정
logger = logging.getLogger(__name__)


# 제주 지역 관측소 코드
JEJU_STATIONS = {
    "제주": 184,
    "서귀포": 189,
    "성산": 188,
    "고산": 185,
}

# 기본 관측소 (제주시)
DEFAULT_STATION = "제주"


@dataclass
class WeatherData:
    """기상 관측 데이터"""

    station_name: str  # 관측소명
    station_code: int  # 관측소 코드
    timestamp: str  # 관측 시간
    temperature: float  # 기온 (°C)
    humidity: Optional[float] = None  # 습도 (%)
    wind_direction: Optional[str] = None  # 풍향
    wind_speed: Optional[float] = None  # 풍속 (m/s)
    precipitation: Optional[float] = None  # 강수량 (mm)
    pressure: Optional[float] = None  # 기압 (hPa)
    fetched_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    source: str = "weather.go.kr"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "station_name": self.station_name,
            "station_code": self.station_code,
            "timestamp": self.timestamp,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "wind_direction": self.wind_direction,
            "wind_speed": self.wind_speed,
            "precipitation": self.precipitation,
            "pressure": self.pressure,
            "fetched_at": self.fetched_at,
            "source": self.source,
        }


class KMAWeatherCrawler:
    """기상청 날씨누리 실시간 기상 크롤러"""

    # 종관기상관측 (주요 도시) 페이지
    BASE_URL = "https://www.weather.go.kr/w/obs-climate/land/city-obs.do"

    def __init__(self, timeout: int = 30):
        """
        초기화

        Args:
            timeout: HTTP 요청 타임아웃 (초)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.weather.go.kr/",
            }
        )

    def close(self):
        """세션 종료"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _parse_float(self, value: str) -> Optional[float]:
        """문자열을 float로 변환 (결측치 처리)"""
        if not value or value.strip() in ["", "-", "－", "−"]:
            return None
        try:
            # 숫자 외 문자 제거
            clean = re.sub(r"[^\d.\-]", "", value.strip())
            return float(clean) if clean else None
        except (ValueError, TypeError):
            return None

    def fetch_weather(
        self, station: str = DEFAULT_STATION
    ) -> Optional[WeatherData]:
        """
        실시간 기상 데이터 조회

        Args:
            station: 관측소명 (제주, 서귀포, 성산, 고산)

        Returns:
            WeatherData 또는 None (실패 시)
        """
        try:
            logger.info(f"기상청 날씨누리 데이터 요청: {station}")

            resp = self.session.get(self.BASE_URL, timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # 관측 시간 추출
            timestamp = self._extract_timestamp(soup, resp.text)

            # 테이블에서 데이터 추출
            table = soup.find("table", class_="table-col")
            if not table:
                logger.error("기상 데이터 테이블을 찾을 수 없습니다")
                return None

            tbody = table.find("tbody")
            if not tbody:
                logger.error("테이블 본문(tbody)을 찾을 수 없습니다")
                return None

            # 지역명으로 행 찾기
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue

                # 첫 번째 셀에서 지역명 확인
                location_cell = cells[0]
                location_link = location_cell.find("a")
                location_text = (
                    location_link.get_text(strip=True)
                    if location_link
                    else location_cell.get_text(strip=True)
                )

                if station in location_text or location_text in station:
                    return self._parse_row(
                        cells, station, timestamp
                    )

            logger.warning(f"{station} 관측소 데이터를 찾을 수 없습니다")
            return None

        except requests.RequestException as e:
            logger.error(f"HTTP 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"데이터 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_row(
        self, cells: List, station: str, timestamp: str
    ) -> Optional[WeatherData]:
        """
        테이블 행 파싱

        테이블 컬럼 순서 (city-obs.do) - 2025년 기준:
        0: 지점명
        1: 날씨 (아이콘)
        2: 시정 (km)
        3: 운량 (10분위)
        4: 중하운량
        5: 현재기온 (°C)
        6: 이슬점온도 (°C)
        7: 체감온도 (°C)
        8: 일강수 (mm)
        9: 적설 (cm)
        10: 습도 (%)
        11: 풍향
        12: 풍속 (m/s) - JS로 로드되어 빈 값일 수 있음
        13: 해면기압 (hPa)
        """
        try:
            if len(cells) < 14:
                logger.warning(f"컬럼 수 부족: {len(cells)}개 (최소 14개 필요)")
                return None

            # 각 셀에서 텍스트 추출 (올바른 인덱스)
            temperature = self._parse_float(cells[5].get_text(strip=True))
            precipitation = self._parse_float(cells[8].get_text(strip=True))
            humidity = self._parse_float(cells[10].get_text(strip=True))
            wind_direction = cells[11].get_text(strip=True) if len(cells) > 11 else None
            wind_speed = self._parse_float(cells[12].get_text(strip=True)) if len(cells) > 12 else None
            pressure = self._parse_float(cells[13].get_text(strip=True)) if len(cells) > 13 else None

            if temperature is None:
                logger.warning(f"{station}: 기온 데이터 없음")
                return None

            station_code = JEJU_STATIONS.get(station, 184)

            data = WeatherData(
                station_name=station,
                station_code=station_code,
                timestamp=timestamp,
                temperature=temperature,
                humidity=humidity,
                wind_direction=wind_direction if wind_direction and wind_direction != "-" else None,
                wind_speed=wind_speed,
                precipitation=precipitation,
                pressure=pressure,
            )

            logger.info(
                f"기상 데이터 수집 완료: {station} {temperature}°C, "
                f"습도 {humidity}%, 풍속 {wind_speed}m/s"
            )
            return data

        except Exception as e:
            logger.error(f"행 파싱 실패: {e}")
            return None

    def _extract_timestamp(self, soup: BeautifulSoup, html: str) -> str:
        """
        HTML에서 관측 시간 추출

        Args:
            soup: BeautifulSoup 객체
            html: HTML 텍스트

        Returns:
            타임스탬프 문자열 (YYYY-MM-DD HH:MM)
        """
        # 패턴 1: 2025.12.23.11:00 형식
        pattern1 = r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2}):(\d{2})"
        match = re.search(pattern1, html)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}"

        # 패턴 2: tm= 파라미터에서 추출
        pattern2 = r"tm=(\d{4})\.(\d{2})\.(\d{2})\.(\d{2}):(\d{2})"
        match = re.search(pattern2, html)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}"

        # 대체: 현재 시간 (정시로 반올림)
        now = datetime.now()
        return now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

    def fetch_jeju_weather(self) -> Optional[WeatherData]:
        """
        제주 기상 데이터 조회 (편의 메서드)

        Returns:
            WeatherData 또는 None
        """
        return self.fetch_weather("제주")

    def fetch_all_jeju_stations(self) -> Dict[str, WeatherData]:
        """
        제주 지역 전체 관측소 데이터 조회

        Returns:
            관측소명: WeatherData 딕셔너리
        """
        results = {}
        for station in JEJU_STATIONS.keys():
            data = self.fetch_weather(station)
            if data:
                results[station] = data
        return results

    def get_status(self) -> Dict[str, Any]:
        """
        현재 기상 상태 조회 (대시보드용)

        Returns:
            상태 정보 딕셔너리
        """
        data = self.fetch_jeju_weather()

        if not data:
            return {
                "status": "error",
                "message": "기상 데이터 조회 실패",
            }

        # 상태 판단 (기온 기반)
        if data.temperature >= 30:
            status = "hot"
            status_text = "무더위"
        elif data.temperature >= 25:
            status = "warm"
            status_text = "더움"
        elif data.temperature >= 15:
            status = "normal"
            status_text = "쾌적"
        elif data.temperature >= 5:
            status = "cool"
            status_text = "서늘"
        else:
            status = "cold"
            status_text = "추움"

        return {
            "status": status,
            "status_text": status_text,
            "data": data.to_dict(),
            "message": f"제주 {data.temperature}°C ({status_text})",
        }


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="기상청 날씨누리 실시간 기상 크롤러")
    parser.add_argument("--station", "-s", default="제주", help="관측소명")
    parser.add_argument("--all", "-a", action="store_true", help="제주 전체 관측소 조회")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("기상청 날씨누리 실시간 기상 현황")
    print("=" * 60)

    with KMAWeatherCrawler() as crawler:
        if args.all:
            # 전체 제주 관측소
            print("\n[제주 지역 전체 관측소]")
            results = crawler.fetch_all_jeju_stations()
            for station, data in results.items():
                print(f"\n📍 {station}")
                print(f"   🌡️  기온: {data.temperature}°C")
                print(f"   💧 습도: {data.humidity}%" if data.humidity else "   💧 습도: -")
                print(f"   💨 풍속: {data.wind_speed}m/s ({data.wind_direction})" if data.wind_speed else "   💨 풍속: -")
        else:
            # 단일 관측소
            data = crawler.fetch_weather(args.station)

            if data:
                print(f"\n📍 관측소: {data.station_name}")
                print(f"📅 관측시간: {data.timestamp}")
                print(f"🌡️  현재기온: {data.temperature}°C")
                print(f"💧 습도: {data.humidity}%" if data.humidity else "💧 습도: -")
                print(f"💨 풍향: {data.wind_direction}" if data.wind_direction else "💨 풍향: -")
                print(f"💨 풍속: {data.wind_speed} m/s" if data.wind_speed else "💨 풍속: -")
                print(f"🌧️  강수량: {data.precipitation} mm" if data.precipitation else "🌧️  강수량: 0 mm")
                print(f"🔵 기압: {data.pressure} hPa" if data.pressure else "🔵 기압: -")
                print()

                # 상태 확인
                status = crawler.get_status()
                print(f"상태: {status['status_text']} ({status['status']})")
            else:
                print("❌ 데이터 조회 실패")

    print("=" * 60)


if __name__ == "__main__":
    main()
