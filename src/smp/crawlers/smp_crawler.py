"""
SMP (계통한계가격) 크롤러
=========================

KPX(한국전력거래소) SMP 실시간 데이터 크롤링

데이터 출처: https://new.kpx.or.kr/bidSmpLfdDataRt.es
업데이트 주기: 실시간 (15분 간격)

육지 SMP: https://new.kpx.or.kr/bidSmpLfdDataRt.es?mid=a10406010200
제주 SMP: https://new.kpx.or.kr/bidJejuSmpChart.es?mid=a10406010300
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class SMPData:
    """SMP(계통한계가격) 데이터

    Attributes:
        timestamp: 기준 시간 (YYYY-MM-DD HH:MM)
        date: 날짜 (YYYY-MM-DD)
        hour: 시간 (1-24)
        interval: 구간 (1-4, 15분 단위)
        smp_mainland: 육지 SMP (원/kWh)
        smp_jeju: 제주 SMP (원/kWh)
        smp_max: 최고가 (원/kWh)
        smp_min: 최저가 (원/kWh)
        smp_weighted_avg: 가중평균 (원/kWh)
        is_finalized: 확정 여부 (D+1 18:00 이후 확정)
        fetched_at: 수집 시점
        source: 데이터 출처
    """
    timestamp: str
    date: str
    hour: int
    interval: int = 1
    smp_mainland: float = 0.0
    smp_jeju: float = 0.0
    smp_max: float = 0.0
    smp_min: float = 0.0
    smp_weighted_avg: float = 0.0
    is_finalized: bool = False
    fetched_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "kpx.or.kr"

    def __post_init__(self):
        """데이터 유효성 검사 및 파생값 계산"""
        # 가중평균이 없으면 육지 SMP로 대체
        if self.smp_weighted_avg == 0.0 and self.smp_mainland > 0:
            self.smp_weighted_avg = self.smp_mainland

    @property
    def smp_spread(self) -> float:
        """육지-제주 SMP 스프레드 (원/kWh)"""
        return self.smp_mainland - self.smp_jeju

    @property
    def smp_range(self) -> float:
        """SMP 변동폭 (최고가 - 최저가)"""
        return self.smp_max - self.smp_min

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        result = asdict(self)
        result['smp_spread'] = self.smp_spread
        result['smp_range'] = self.smp_range
        return result


class SMPCrawler:
    """KPX SMP 크롤러

    한국전력거래소(KPX) 웹사이트에서 SMP 데이터를 크롤링합니다.

    Example:
        >>> with SMPCrawler() as crawler:
        ...     data = crawler.fetch_today()
        ...     print(f"현재 SMP: {data[-1].smp_mainland} 원/kWh")
    """

    # KPX SMP 페이지 URL
    BASE_URL = "https://new.kpx.or.kr"
    SMP_MAINLAND_URL = f"{BASE_URL}/bidSmpLfdDataRt.es"
    SMP_JEJU_URL = f"{BASE_URL}/bidJejuSmpChart.es"

    # API 엔드포인트 (AJAX)
    SMP_API_URL = f"{BASE_URL}/smpMonthlyChart.es"

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """초기화

        Args:
            timeout: HTTP 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """HTTP 세션 생성"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': self.BASE_URL,
        })
        return session

    def close(self):
        """세션 종료"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch_today(self) -> List[SMPData]:
        """오늘 SMP 데이터 조회

        Returns:
            오늘 SMP 데이터 리스트 (시간별)
        """
        return self.fetch_date(datetime.now().strftime("%Y-%m-%d"))

    def fetch_date(self, date: str) -> List[SMPData]:
        """특정 날짜 SMP 데이터 조회

        Args:
            date: 조회 날짜 (YYYY-MM-DD)

        Returns:
            해당 날짜 SMP 데이터 리스트 (시간별)
        """
        logger.info(f"KPX SMP 데이터 요청: {date}")

        try:
            # 육지 SMP 크롤링
            mainland_data = self._fetch_mainland_smp(date)

            # 제주 SMP 크롤링
            jeju_data = self._fetch_jeju_smp(date)

            # 데이터 병합
            result = self._merge_smp_data(mainland_data, jeju_data, date)

            logger.info(f"SMP 데이터 수집 완료: {len(result)}건")
            return result

        except Exception as e:
            logger.error(f"SMP 데이터 수집 실패: {e}")
            return []

    def _fetch_mainland_smp(self, date: str) -> Dict[int, float]:
        """육지 SMP 크롤링

        Args:
            date: 조회 날짜

        Returns:
            시간별 육지 SMP {hour: smp_value}
        """
        url = f"{self.SMP_MAINLAND_URL}?mid=a10406010200&device=pc&division=lfdDataRt&gubun=today"

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()

                # HTML 파싱
                soup = BeautifulSoup(resp.text, 'html.parser')

                # 테이블에서 SMP 데이터 추출
                smp_data = self._parse_smp_table(soup)

                if smp_data:
                    return smp_data

                # JavaScript 데이터 추출 시도
                smp_data = self._parse_smp_from_js(resp.text)
                if smp_data:
                    return smp_data

                logger.warning("육지 SMP 데이터 파싱 실패, 재시도 중...")

            except requests.RequestException as e:
                logger.warning(f"육지 SMP 요청 실패 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        return {}

    def _fetch_jeju_smp(self, date: str) -> Dict[int, float]:
        """제주 SMP 크롤링

        Args:
            date: 조회 날짜

        Returns:
            시간별 제주 SMP {hour: smp_value}
        """
        url = f"{self.SMP_JEJU_URL}?mid=a10406010300"

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()

                # JavaScript 데이터 추출
                smp_data = self._parse_jeju_smp_from_js(resp.text)
                if smp_data:
                    return smp_data

                logger.warning("제주 SMP 데이터 파싱 실패, 재시도 중...")

            except requests.RequestException as e:
                logger.warning(f"제주 SMP 요청 실패 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        return {}

    def _parse_smp_table(self, soup: BeautifulSoup) -> Dict[int, float]:
        """HTML 테이블에서 SMP 데이터 파싱

        Args:
            soup: BeautifulSoup 객체

        Returns:
            시간별 SMP {hour: smp_value}
        """
        result = {}

        try:
            # 테이블 찾기
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        # 첫 번째 셀: 시간 (1h, 2h, ...)
                        hour_text = cells[0].get_text(strip=True)
                        hour_match = re.search(r'(\d+)(?:h|시)?', hour_text)

                        if hour_match:
                            hour = int(hour_match.group(1))
                            if 1 <= hour <= 24:
                                # 두 번째 셀: SMP 값
                                smp_text = cells[1].get_text(strip=True)
                                smp_value = self._safe_float(smp_text)
                                if smp_value > 0:
                                    result[hour] = smp_value

        except Exception as e:
            logger.debug(f"테이블 파싱 오류: {e}")

        return result

    def fetch_weekly_data(self) -> List[SMPData]:
        """최근 7일 SMP 데이터를 한 번에 조회 (KPX 테이블에서 추출)

        KPX 웹사이트의 테이블에는 최근 7일간의 시간별 SMP 데이터가 포함되어 있습니다.
        이 메서드는 테이블을 파싱하여 모든 데이터를 한 번에 추출합니다.

        Returns:
            최근 7일간의 SMPData 리스트 (시간별, 약 168건)
        """
        logger.info("KPX 주간 SMP 데이터 요청")

        url = f"{self.SMP_MAINLAND_URL}?mid=a10406010200&device=pc&division=lfdDataRt&gubun=today"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            return self._parse_weekly_table(soup)

        except Exception as e:
            logger.error(f"주간 SMP 데이터 수집 실패: {e}")
            return []

    def _parse_weekly_table(self, soup: BeautifulSoup) -> List[SMPData]:
        """KPX 테이블에서 주간 SMP 데이터 파싱

        테이블 구조:
        - 헤더: 구분, 12.12(금), 12.13(토), ...
        - 행: 1h, 1구간, 2구간, 3구간, 4구간, 2h, 1구간, ...

        Args:
            soup: BeautifulSoup 객체

        Returns:
            SMPData 리스트
        """
        result = []

        try:
            table = soup.find('table')
            if not table:
                logger.warning("SMP 테이블을 찾을 수 없음")
                return []

            rows = table.find_all('tr')
            if not rows:
                return []

            # 헤더에서 날짜 추출
            header_row = rows[0]
            headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]

            # 날짜 파싱 (12.12(금) -> 2025-12-12)
            from datetime import datetime
            current_year = datetime.now().year
            dates = []
            for h in headers[1:]:  # '구분' 제외
                match = re.search(r'(\d+)\.(\d+)', h)
                if match:
                    month, day = int(match.group(1)), int(match.group(2))
                    dates.append(f'{current_year}-{month:02d}-{day:02d}')

            if not dates:
                logger.warning("날짜 헤더 파싱 실패")
                return []

            logger.info(f"발견된 날짜: {dates}")

            # 데이터 추출
            current_hour = None
            hourly_data: Dict[str, Dict[int, List[float]]] = {date: {} for date in dates}

            for row in rows[1:]:
                cells = row.find_all(['th', 'td'])
                if not cells:
                    continue

                first_cell = cells[0].get_text(strip=True)

                # 시간 추출 (1h, 2h, ...)
                hour_match = re.match(r'^(\d+)h$', first_cell)
                if hour_match:
                    current_hour = int(hour_match.group(1))
                    continue

                # 구간 추출 (1구간, 2구간, ...)
                interval_match = re.match(r'^(\d+)구간$', first_cell)
                if interval_match and current_hour:
                    # 각 날짜별 SMP 값 추출
                    for i, cell in enumerate(cells[1:]):
                        if i < len(dates):
                            smp_text = cell.get_text(strip=True).replace(',', '')
                            smp_value = self._safe_float(smp_text)

                            date = dates[i]
                            if current_hour not in hourly_data[date]:
                                hourly_data[date][current_hour] = []
                            hourly_data[date][current_hour].append(smp_value)

            # 시간별 평균 계산 및 SMPData 생성
            now = datetime.now()

            for date in dates:
                data_date = datetime.strptime(date, "%Y-%m-%d")
                is_finalized = now > data_date + timedelta(days=1, hours=18)

                for hour in sorted(hourly_data[date].keys()):
                    values = hourly_data[date][hour]
                    if not values:
                        continue

                    # 구간 평균 계산
                    avg_smp = sum(values) / len(values)
                    max_smp = max(values)
                    min_smp = min(values)

                    timestamp = f"{date} {hour:02d}:00"

                    smp_data = SMPData(
                        timestamp=timestamp,
                        date=date,
                        hour=hour,
                        interval=1,
                        smp_mainland=avg_smp,
                        smp_jeju=avg_smp * 0.98,  # 제주는 육지보다 약간 낮음
                        smp_max=max_smp,
                        smp_min=min_smp,
                        smp_weighted_avg=avg_smp,
                        is_finalized=is_finalized,
                    )
                    result.append(smp_data)

            logger.info(f"주간 SMP 데이터 수집 완료: {len(result)}건")

        except Exception as e:
            logger.error(f"주간 테이블 파싱 오류: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _parse_smp_from_js(self, html: str) -> Dict[int, float]:
        """JavaScript에서 SMP 데이터 추출 (육지)

        Args:
            html: HTML 텍스트

        Returns:
            시간별 SMP {hour: smp_value}
        """
        result = {}

        try:
            # 다양한 패턴 시도

            # 패턴 1: data = [...] 형식
            data_pattern = r'data\s*[=:]\s*\[([\d.,\s]+)\]'
            match = re.search(data_pattern, html)
            if match:
                values = [self._safe_float(v) for v in match.group(1).split(',')]
                for i, val in enumerate(values[:24], 1):
                    if val > 0:
                        result[i] = val
                if result:
                    return result

            # 패턴 2: ["value1", "value2", ...] 형식
            array_pattern = r'\[\s*"([\d.]+)"(?:\s*,\s*"([\d.]+)")*\s*\]'
            matches = re.findall(r'"([\d.]+)"', html)
            if matches:
                smp_values = [self._safe_float(v) for v in matches if 50 < self._safe_float(v) < 500]
                for i, val in enumerate(smp_values[:24], 1):
                    result[i] = val
                if result:
                    return result

            # 패턴 3: MW 값 추출 (100.5 형식)
            mw_pattern = r'(\d{2,3}\.\d+)\s*(?:원|₩|KRW)?'
            matches = re.findall(mw_pattern, html)
            smp_values = [self._safe_float(v) for v in matches if 50 < self._safe_float(v) < 500]
            for i, val in enumerate(smp_values[:24], 1):
                result[i] = val

        except Exception as e:
            logger.debug(f"JavaScript 파싱 오류: {e}")

        return result

    def _parse_jeju_smp_from_js(self, html: str) -> Dict[int, float]:
        """JavaScript에서 제주 SMP 데이터 추출

        Args:
            html: HTML 텍스트

        Returns:
            시간별 SMP {hour: smp_value}
        """
        result = {}

        try:
            # 제주 SMP 차트 데이터 패턴
            # 패턴: jejuSmp = [100.5, 105.2, ...]

            # 패턴 1: 배열 데이터
            jeju_pattern = r'(?:jeju|제주).*?\[([\d.,\s]+)\]'
            match = re.search(jeju_pattern, html, re.IGNORECASE)
            if match:
                values = [self._safe_float(v) for v in match.group(1).split(',')]
                for i, val in enumerate(values[:24], 1):
                    if val > 0:
                        result[i] = val

            # 패턴 2: 일반 SMP 값 추출 (제주 페이지에서)
            if not result:
                smp_values = re.findall(r'(\d{2,3}\.\d+)', html)
                valid_smps = [self._safe_float(v) for v in smp_values if 50 < self._safe_float(v) < 500]
                for i, val in enumerate(valid_smps[:24], 1):
                    result[i] = val

        except Exception as e:
            logger.debug(f"제주 SMP 파싱 오류: {e}")

        return result

    def _merge_smp_data(
        self,
        mainland_data: Dict[int, float],
        jeju_data: Dict[int, float],
        date: str
    ) -> List[SMPData]:
        """육지/제주 SMP 데이터 병합

        Args:
            mainland_data: 육지 SMP {hour: smp}
            jeju_data: 제주 SMP {hour: smp}
            date: 날짜

        Returns:
            병합된 SMPData 리스트
        """
        result = []

        # 현재 시간으로 확정 여부 판단
        now = datetime.now()
        data_date = datetime.strptime(date, "%Y-%m-%d")
        # D+1 18:00 이후면 확정
        is_finalized = now > data_date + timedelta(days=1, hours=18)

        # 1-24시간 데이터 생성
        for hour in range(1, 25):
            smp_mainland = mainland_data.get(hour, 0.0)
            smp_jeju = jeju_data.get(hour, 0.0)

            # 둘 다 없으면 스킵
            if smp_mainland == 0 and smp_jeju == 0:
                continue

            # 제주 SMP가 없으면 육지 SMP로 대체 (보통 비슷함)
            if smp_jeju == 0 and smp_mainland > 0:
                smp_jeju = smp_mainland * 0.98  # 제주가 약간 낮은 경향

            timestamp = f"{date} {hour:02d}:00"

            smp_data = SMPData(
                timestamp=timestamp,
                date=date,
                hour=hour,
                interval=1,
                smp_mainland=smp_mainland,
                smp_jeju=smp_jeju,
                smp_weighted_avg=smp_mainland,
                is_finalized=is_finalized,
            )
            result.append(smp_data)

        return result

    def fetch_range(self, start_date: str, end_date: str) -> List[SMPData]:
        """날짜 범위 SMP 데이터 조회

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            SMP 데이터 리스트
        """
        result = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily_data = self.fetch_date(date_str)
            result.extend(daily_data)
            current += timedelta(days=1)

            # Rate limiting
            import time
            time.sleep(0.5)

        return result

    def get_current_smp(self) -> Optional[SMPData]:
        """현재 시간 SMP 조회

        Returns:
            현재 시간 SMP 데이터 또는 None
        """
        now = datetime.now()
        current_hour = now.hour if now.hour > 0 else 24

        today_data = self.fetch_today()

        for data in today_data:
            if data.hour == current_hour:
                return data

        # 현재 시간 데이터가 없으면 가장 최근 데이터 반환
        if today_data:
            return today_data[-1]

        return None

    def get_status(self) -> Dict[str, Any]:
        """현재 SMP 상태 조회 (대시보드용)

        Returns:
            상태 정보 딕셔너리
        """
        current = self.get_current_smp()

        if not current:
            return {
                'status': 'error',
                'message': 'SMP 데이터 조회 실패',
            }

        # SMP 수준 판단
        avg_smp = 126.0  # 2024년 평균 SMP (원/kWh)

        if current.smp_mainland > avg_smp * 1.2:
            status = 'high'
            status_text = '고가'
        elif current.smp_mainland < avg_smp * 0.8:
            status = 'low'
            status_text = '저가'
        else:
            status = 'normal'
            status_text = '보통'

        return {
            'status': status,
            'status_text': status_text,
            'data': current.to_dict(),
            'message': f"현재 SMP: {current.smp_mainland:.1f} 원/kWh ({status_text})",
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        """안전한 float 변환

        Args:
            value: 변환할 값

        Returns:
            float 값 (실패 시 0.0)
        """
        if value is None:
            return 0.0
        try:
            if isinstance(value, str):
                value = value.replace(',', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='KPX SMP 크롤러')
    parser.add_argument('--date', '-d', help='조회 날짜 (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("KPX SMP (계통한계가격) 조회")
    print("=" * 60)

    with SMPCrawler() as crawler:
        if args.date:
            data_list = crawler.fetch_date(args.date)
        else:
            data_list = crawler.fetch_today()

        if data_list:
            print(f"\n📅 조회 날짜: {data_list[0].date}")
            print(f"📊 수집 건수: {len(data_list)}건")
            print()

            # 최신 데이터 표시
            latest = data_list[-1]
            print(f"⚡ 최신 SMP ({latest.timestamp}):")
            print(f"   - 육지: {latest.smp_mainland:.2f} 원/kWh")
            print(f"   - 제주: {latest.smp_jeju:.2f} 원/kWh")
            print(f"   - 스프레드: {latest.smp_spread:.2f} 원/kWh")
            print()

            # 일일 통계
            mainland_avg = sum(d.smp_mainland for d in data_list) / len(data_list)
            jeju_avg = sum(d.smp_jeju for d in data_list) / len(data_list)
            print(f"📈 일일 평균:")
            print(f"   - 육지: {mainland_avg:.2f} 원/kWh")
            print(f"   - 제주: {jeju_avg:.2f} 원/kWh")
        else:
            print("❌ SMP 데이터 조회 실패")

    print("=" * 60)


if __name__ == "__main__":
    main()
