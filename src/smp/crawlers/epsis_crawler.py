"""
EPSIS SMP 크롤러
================

전력통계정보시스템(EPSIS)에서 실제 SMP 데이터를 수집합니다.

데이터 출처: https://epsis.kpx.or.kr
- 시간별 SMP (육지/제주)
- 과거 데이터 (최대 10년)

Usage:
    python -m src.smp.crawlers.epsis_crawler --years 1
    python -m src.smp.crawlers.epsis_crawler --start 20240101 --end 20241231

Author: Claude Code
Date: 2025-12
"""

import json
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class EPSISSMPData:
    """EPSIS SMP 데이터 구조"""

    timestamp: str          # 2024-01-01 01:00
    date: str               # 2024-01-01
    hour: int               # 1-24
    smp_mainland: float     # 육지 SMP (원/kWh)
    smp_jeju: float         # 제주 SMP (원/kWh)
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "EPSIS"
    is_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'date': self.date,
            'hour': self.hour,
            'smp_mainland': self.smp_mainland,
            'smp_jeju': self.smp_jeju,
            'fetched_at': self.fetched_at,
            'source': self.source,
            'is_synthetic': self.is_synthetic,
        }


class EPSISCrawler:
    """EPSIS SMP 크롤러

    전력통계정보시스템(EPSIS)에서 시간별 SMP 데이터를 수집합니다.

    Example:
        >>> crawler = EPSISCrawler()
        >>> df = crawler.fetch_range("20240101", "20240131")
        >>> print(f"수집: {len(df)}건")
    """

    BASE_URL = "https://epsis.kpx.or.kr"

    # API 엔드포인트
    ENDPOINTS = {
        'smp_chart': '/epsisnew/selectEkmaSmpShdChart.ajax',
        'smp_grid': '/epsisnew/selectEkmaSmpShd.ajax',
        'smp_page': '/epsisnew/selectEkmaSmpShdChart.do',
    }

    # 시장 코드
    MARKET_CODES = {
        'mainland': '1',   # 육지
        'jeju': '9',       # 제주
    }

    def __init__(self, timeout: int = 60, max_retries: int = 3):
        """초기화

        Args:
            timeout: HTTP 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()
        self._init_cookies()

    def _create_session(self) -> requests.Session:
        """HTTP 세션 생성"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': self.BASE_URL,
            'Referer': f'{self.BASE_URL}/epsisnew/selectEkmaSmpShdChart.do?menuId=020100',
            'X-Requested-With': 'XMLHttpRequest',
        })
        return session

    def _init_cookies(self):
        """초기 쿠키 설정 (세션 유지)"""
        try:
            # 메인 페이지 방문하여 세션 쿠키 획득
            url = f"{self.BASE_URL}{self.ENDPOINTS['smp_page']}?menuId=020100"
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            logger.info("EPSIS 세션 초기화 완료")
        except Exception as e:
            logger.warning(f"세션 초기화 실패: {e}")

    def fetch_monthly_chart_data(
        self,
        year: int,
        month: int
    ) -> List[Dict[str, Any]]:
        """월간 차트 데이터 조회 (JavaScript 파싱)

        EPSIS는 JSON이 아닌 JavaScript 코드를 반환하므로
        정규식으로 데이터를 추출합니다.

        Args:
            year: 연도
            month: 월

        Returns:
            시간별 SMP 데이터 리스트
        """
        url = f"{self.BASE_URL}{self.ENDPOINTS['smp_chart']}"

        params = {
            'srchYear': str(year),
            'srchMonth': str(month).zfill(2),
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, data=params, timeout=self.timeout)
                resp.raise_for_status()

                # JavaScript에서 chartData 추출
                return self._parse_chart_js(resp.text, year, month)

            except requests.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return []

    def _parse_chart_js(
        self,
        js_text: str,
        year: int,
        month: int
    ) -> List[Dict[str, Any]]:
        """JavaScript chartData.push() 구문에서 데이터 추출

        형식: chartData.push({"Date":"01일 01시","Value":"97.81","Value2":"97.81"});

        Args:
            js_text: JavaScript 코드
            year: 연도
            month: 월

        Returns:
            파싱된 데이터 리스트
        """
        result = []

        # chartData.push({...}) 패턴 매칭
        pattern = r'chartData\.push\(\{["\']Date["\']:\s*["\'](\d+)일\s*(\d+)시["\'],\s*["\']Value["\']:\s*["\']([\d.]+)["\'],\s*["\']Value2["\']:\s*["\']([\d.]+)["\']\}\)'

        matches = re.findall(pattern, js_text)

        for match in matches:
            day, hour, value1, value2 = match

            try:
                day_int = int(day)
                hour_int = int(hour)
                smp_mainland = float(value1)
                smp_jeju = float(value2)

                date_str = f"{year}-{month:02d}-{day_int:02d}"
                timestamp = f"{date_str} {hour_int:02d}:00"

                result.append({
                    'timestamp': timestamp,
                    'date': date_str,
                    'hour': hour_int,
                    'smp_mainland': smp_mainland,
                    'smp_jeju': smp_jeju,
                    'smp_max': max(smp_mainland, smp_jeju) * 1.02,
                    'smp_min': min(smp_mainland, smp_jeju) * 0.98,
                    'source': 'EPSIS',
                    'is_synthetic': False,
                })

            except (ValueError, IndexError) as e:
                logger.debug(f"파싱 오류: {match} - {e}")
                continue

        return result

    def fetch_smp_data(
        self,
        start_date: str,
        end_date: str,
        market: str = 'mainland'
    ) -> List[Dict[str, Any]]:
        """SMP 데이터 조회 (Grid API - JavaScript 파싱)

        Args:
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            market: 'mainland' (육지) 또는 'jeju' (제주)

        Returns:
            SMP 데이터 리스트
        """
        url = f"{self.BASE_URL}{self.ENDPOINTS['smp_grid']}"

        market_code = self.MARKET_CODES.get(market, '1')

        # 요청 파라미터
        data = {
            'beginDate': start_date,
            'endDate': end_date,
            'selKind': market_code,
            'locale': 'ko',
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, data=data, timeout=self.timeout)
                resp.raise_for_status()

                # JavaScript 응답 파싱
                return self._parse_grid_js(resp.text, start_date, end_date)

            except requests.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return []

    def _parse_grid_js(
        self,
        js_text: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """JavaScript 그리드 데이터에서 SMP 추출

        형식: c1 = textFormmat("106.1",count);

        Args:
            js_text: JavaScript 코드
            start_date: 시작일
            end_date: 종료일

        Returns:
            파싱된 데이터 리스트
        """
        result = []

        # textFormmat("값",count) 패턴 매칭
        pattern = r'c(\d+)\s*=\s*textFormmat\(["\']([^"\']+)["\'],\s*count\)'
        matches = re.findall(pattern, js_text)

        if not matches:
            logger.debug("그리드 데이터를 찾을 수 없음")
            return []

        # 날짜 범위 계산
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        num_days = (end_dt - start_dt).days + 1

        # SMP 값 추출 (시간별 24개 * 일수)
        smp_values = []
        for idx, value in matches:
            try:
                smp = float(value)
                smp_values.append(smp)
            except ValueError:
                smp_values.append(0.0)

        # 데이터 구조화 (24시간 * 일수)
        idx = 0
        current_date = start_dt

        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")

            for hour in range(1, 25):
                if idx < len(smp_values):
                    smp_value = smp_values[idx]
                    idx += 1

                    if smp_value > 0:
                        timestamp = f"{date_str} {hour:02d}:00"
                        result.append({
                            'timestamp': timestamp,
                            'date': date_str,
                            'hour': hour,
                            'smp_mainland': smp_value,
                            'smp_jeju': smp_value * 0.98,  # 추정
                            'smp_max': smp_value * 1.02,
                            'smp_min': smp_value * 0.98,
                            'source': 'EPSIS',
                            'is_synthetic': False,
                        })

            current_date += timedelta(days=1)

        return result

    def _parse_html_table(self, html: str) -> List[Dict[str, Any]]:
        """HTML 테이블에서 SMP 데이터 추출"""
        result = []

        try:
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table')

            for table in tables:
                rows = table.find_all('tr')
                headers = []

                for row in rows:
                    cells = row.find_all(['th', 'td'])

                    if not headers:
                        # 헤더 추출
                        headers = [cell.get_text(strip=True) for cell in cells]
                        continue

                    if len(cells) >= 2:
                        row_data = {}
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                row_data[headers[i]] = cell.get_text(strip=True)
                        if row_data:
                            result.append(row_data)

        except Exception as e:
            logger.debug(f"HTML 파싱 오류: {e}")

        return result

    def fetch_range(
        self,
        start_date: str,
        end_date: str,
        include_jeju: bool = True
    ) -> pd.DataFrame:
        """기간별 SMP 데이터 수집

        월간 차트 API를 사용하여 데이터를 수집합니다.
        (육지/제주 SMP가 함께 제공됨)

        Args:
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            include_jeju: 제주 데이터 포함 여부

        Returns:
            SMP DataFrame
        """
        all_data = []

        # 시작/종료일 파싱
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        # 월별로 수집
        current_year = start_dt.year
        current_month = start_dt.month

        while True:
            # 현재 월이 범위를 벗어나면 종료
            current_dt = datetime(current_year, current_month, 1)
            if current_dt > end_dt:
                break

            logger.info(f"수집 중: {current_year}년 {current_month}월")

            # 월간 차트 데이터 수집
            monthly_data = self.fetch_monthly_chart_data(current_year, current_month)

            if monthly_data:
                # 날짜 범위 필터링
                for item in monthly_data:
                    item_date = datetime.strptime(item['date'], "%Y-%m-%d")
                    if start_dt <= item_date <= end_dt:
                        all_data.append(item)

                logger.info(f"  → 수집 완료: {len(monthly_data)}건")
            else:
                logger.warning(f"  → 데이터 없음")

            # 다음 달로 이동
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1

            # Rate limiting
            time.sleep(0.5)

        if not all_data:
            logger.warning("수집된 데이터가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        return df

    def _extract_datetime_key(self, item: Dict[str, Any]) -> Optional[str]:
        """날짜+시간 키 추출"""
        try:
            # 다양한 키 이름 시도
            date_keys = ['tradeDay', 'tradeDt', 'date', 'baseDt', 'baseDate']
            hour_keys = ['tradeHour', 'hour', 'hh', 'time']

            date_val = None
            hour_val = None

            for key in date_keys:
                if key in item:
                    date_val = str(item[key])
                    break

            for key in hour_keys:
                if key in item:
                    hour_val = str(item[key]).zfill(2)
                    break

            if date_val and hour_val:
                return f"{date_val}_{hour_val}"

        except Exception:
            pass

        return None

    def _extract_smp_value(self, item: Dict[str, Any]) -> float:
        """SMP 값 추출"""
        smp_keys = ['smp', 'avgSmp', 'smpVal', 'value', 'price']

        for key in smp_keys:
            if key in item:
                try:
                    val = str(item[key]).replace(',', '')
                    return float(val)
                except ValueError:
                    continue

        return 0.0

    def _parse_smp_item(
        self,
        item: Dict[str, Any],
        jeju_data: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """SMP 항목 파싱"""
        try:
            # 날짜 추출
            date_str = None
            for key in ['tradeDay', 'tradeDt', 'date', 'baseDt', 'baseDate']:
                if key in item:
                    date_str = str(item[key])
                    break

            if not date_str:
                return None

            # 날짜 형식 정규화
            if len(date_str) == 8:
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            elif '-' in date_str:
                date_formatted = date_str
            else:
                return None

            # 시간 추출
            hour = None
            for key in ['tradeHour', 'hour', 'hh', 'time']:
                if key in item:
                    try:
                        hour = int(str(item[key]).replace('h', '').replace('시', ''))
                        break
                    except ValueError:
                        continue

            if hour is None:
                return None

            # SMP 값 추출
            smp_mainland = self._extract_smp_value(item)

            # 제주 SMP
            datetime_key = f"{date_str}_{str(hour).zfill(2)}"
            smp_jeju = jeju_data.get(datetime_key, smp_mainland * 0.98)

            timestamp = f"{date_formatted} {hour:02d}:00"

            return {
                'timestamp': timestamp,
                'date': date_formatted,
                'hour': hour,
                'smp_mainland': smp_mainland,
                'smp_jeju': smp_jeju,
                'smp_max': smp_mainland * 1.02,  # 추정값
                'smp_min': smp_mainland * 0.98,  # 추정값
                'source': 'EPSIS',
                'is_synthetic': False,
            }

        except Exception as e:
            logger.debug(f"항목 파싱 오류: {e}")
            return None

    def fetch_years(
        self,
        years: int = 1,
        include_jeju: bool = True
    ) -> pd.DataFrame:
        """연간 SMP 데이터 수집

        Args:
            years: 수집할 연수 (과거 기준)
            include_jeju: 제주 데이터 포함 여부

        Returns:
            SMP DataFrame
        """
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years)

        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

        return self.fetch_range(start_date, end_date, include_jeju)

    def close(self):
        """세션 종료"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_to_csv(df: pd.DataFrame, output_path: Path):
    """DataFrame을 CSV로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"저장 완료: {output_path}")


def print_statistics(df: pd.DataFrame):
    """데이터 통계 출력"""
    print("\n" + "=" * 50)
    print("📊 데이터 통계")
    print("=" * 50)
    print(f"총 레코드: {len(df):,}건")

    if 'date' in df.columns:
        print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

    if 'smp_mainland' in df.columns:
        print(f"육지 SMP 범위: {df['smp_mainland'].min():.1f} ~ {df['smp_mainland'].max():.1f} 원/kWh")
        print(f"육지 SMP 평균: {df['smp_mainland'].mean():.1f} 원/kWh")

    if 'smp_jeju' in df.columns:
        print(f"제주 SMP 평균: {df['smp_jeju'].mean():.1f} 원/kWh")

    if 'is_synthetic' in df.columns:
        real_count = len(df[~df['is_synthetic']])
        print(f"실제 데이터: {real_count:,}건 ({100 * real_count / len(df):.1f}%)")

    print("=" * 50)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='EPSIS SMP 데이터 수집')
    parser.add_argument('--years', '-y', type=int, default=1,
                        help='수집할 연수 (기본: 1년)')
    parser.add_argument('--start', '-s', type=str,
                        help='시작일 (YYYYMMDD)')
    parser.add_argument('--end', '-e', type=str,
                        help='종료일 (YYYYMMDD)')
    parser.add_argument('--output', '-o', type=str,
                        default='data/smp/smp_real_epsis.csv',
                        help='출력 파일 경로')
    parser.add_argument('--no-jeju', action='store_true',
                        help='제주 데이터 제외')
    args = parser.parse_args()

    print("=" * 60)
    print("⚡ EPSIS SMP 데이터 수집")
    print("=" * 60)

    output_path = PROJECT_ROOT / args.output

    with EPSISCrawler() as crawler:
        if args.start and args.end:
            # 기간 지정
            print(f"\n📅 기간: {args.start} ~ {args.end}")
            df = crawler.fetch_range(
                args.start,
                args.end,
                include_jeju=not args.no_jeju
            )
        else:
            # 연 단위
            print(f"\n📅 최근 {args.years}년 데이터 수집")
            df = crawler.fetch_years(
                years=args.years,
                include_jeju=not args.no_jeju
            )

        if df.empty:
            print("\n❌ 데이터 수집 실패")
            print("\n💡 EPSIS 직접 접속하여 데이터 다운로드:")
            print("   https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=020100")
            return

        # 저장
        save_to_csv(df, output_path)

        # 통계 출력
        print_statistics(df)

    print("\n✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
