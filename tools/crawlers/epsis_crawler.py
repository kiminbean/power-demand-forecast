#!/usr/bin/env python3
"""
EPSIS (전력통계정보시스템) 실시간 전력 수급 데이터 크롤러

전력거래소(KPX)의 EPSIS에서 실시간 전력 수급 현황 데이터를 수집합니다.
- 공급능력 (MW)
- 현재수요/현재부하 (MW)
- 공급예비력 (MW)
- 공급예비율 (%)

⚠️ 주의: EPSIS는 **전국 데이터**만 제공합니다.
   제주 전용 데이터는 별도 API 또는 추정치 사용이 필요합니다.
   (제주 수요 ≈ 전국의 약 1.3~2.0%, 시즌별 변동)

Usage:
    # 현재 시점 데이터 조회 (전국)
    python epsis_crawler.py

    # 제주 추정치 포함 조회
    python epsis_crawler.py --jeju

    # 특정 날짜 범위 조회
    python epsis_crawler.py --start 2024-12-01 --end 2024-12-17

    # 주기적 크롤링 (5분 간격)
    python epsis_crawler.py --periodic --interval 300

    # JSON 파일로 저장
    python epsis_crawler.py --output data/epsis_realtime.json

    # CSV 파일로 저장
    python epsis_crawler.py --output data/epsis_realtime.csv --format csv

Author: Claude Code
Created: 2024-12-17
"""

import requests
import json
import re
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
import csv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PowerSupplyData:
    """전력 수급 데이터 구조"""
    timestamp: str          # 데이터 시점 (YYYY-MM-DD HH:MM)
    supply_capacity: float  # 공급능력 (MW)
    current_demand: float   # 현재수요/현재부하 (MW)
    reserve_power: float    # 공급예비력 (MW)
    reserve_rate: float     # 공급예비율 (%)
    temperature: Optional[float] = None  # 기온 (°C)
    fetched_at: str = ""    # 크롤링 시점
    region: str = "전국"    # 지역 (전국/제주)
    is_estimated: bool = False  # 추정치 여부

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def utilization_rate(self) -> float:
        """설비 이용률 (%)"""
        if self.supply_capacity > 0:
            return round(self.current_demand / self.supply_capacity * 100, 2)
        return 0.0

    @property
    def status(self) -> str:
        """수급 상태 판단"""
        if self.reserve_rate >= 10:
            return "안정"
        elif self.reserve_rate >= 5:
            return "주의"
        elif self.reserve_rate >= 3:
            return "경계"
        else:
            return "위기"


class JejuEstimator:
    """
    제주 전력 수요 추정기

    전국 데이터를 기반으로 제주 전력 수요를 추정합니다.
    추정 비율은 계절 및 시간대에 따라 조정됩니다.

    참고 데이터 (2024년 기준):
    - 제주 최대 수요: 약 1,100~1,200 MW
    - 전국 최대 수요: 약 95,000~100,000 MW
    - 기본 비율: 약 1.2~1.5%

    조정 요인:
    - 여름(7-8월): 관광 성수기로 수요 증가 (+20%)
    - 겨울(12-2월): 난방 수요 증가 (+10%)
    - 주간(9-21시): 상업/관광 활동으로 +5%
    """

    # 제주/전국 수요 비율 (계절별)
    SEASONAL_RATIOS = {
        'summer': 0.018,    # 7-8월: 1.8% (관광 성수기)
        'winter': 0.016,    # 12-2월: 1.6% (난방 수요)
        'spring': 0.014,    # 3-5월: 1.4%
        'autumn': 0.015,    # 9-11월: 1.5%
        'default': 0.015,   # 기본값: 1.5%
    }

    # 제주 공급능력 기준 (MW)
    JEJU_SUPPLY_CAPACITY = 1500  # 제주 전력 계통 공급능력

    def __init__(self):
        self.base_ratio = self.SEASONAL_RATIOS['default']

    def get_season(self, timestamp: str) -> str:
        """시점의 계절 반환"""
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            month = dt.month
        except ValueError:
            return 'default'

        if month in [7, 8]:
            return 'summer'
        elif month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [9, 10, 11]:
            return 'autumn'
        return 'default'

    def get_time_adjustment(self, timestamp: str) -> float:
        """시간대별 조정 계수"""
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            hour = dt.hour
        except ValueError:
            return 1.0

        # 주간 시간대 (9-21시): 관광/상업 활동 증가
        if 9 <= hour <= 21:
            return 1.05
        return 1.0

    def estimate_jeju_demand(self, national_data: PowerSupplyData) -> PowerSupplyData:
        """
        전국 데이터로부터 제주 수요 추정

        Args:
            national_data: 전국 전력 수급 데이터

        Returns:
            제주 추정 데이터
        """
        season = self.get_season(national_data.timestamp)
        base_ratio = self.SEASONAL_RATIOS.get(season, self.base_ratio)
        time_adj = self.get_time_adjustment(national_data.timestamp)

        # 최종 비율
        ratio = base_ratio * time_adj

        # 제주 수요 추정
        jeju_demand = national_data.current_demand * ratio

        # 제주 공급능력은 고정값 사용 (약 1,500 MW)
        jeju_supply = self.JEJU_SUPPLY_CAPACITY

        # 예비력 및 예비율 계산
        jeju_reserve = jeju_supply - jeju_demand
        jeju_reserve_rate = (jeju_reserve / jeju_supply) * 100 if jeju_supply > 0 else 0

        return PowerSupplyData(
            timestamp=national_data.timestamp,
            supply_capacity=jeju_supply,
            current_demand=round(jeju_demand, 1),
            reserve_power=round(jeju_reserve, 1),
            reserve_rate=round(jeju_reserve_rate, 1),
            temperature=national_data.temperature,
            fetched_at=national_data.fetched_at,
            region="제주",
            is_estimated=True,
        )


class EPSISCrawler:
    """EPSIS 실시간 전력 수급 크롤러"""

    BASE_URL = "https://epsis.kpx.or.kr/epsisnew"
    REALTIME_ENDPOINT = "/selectEkgeEpsMepRealChartAjax.ajax"

    # 데이터 필드 매핑
    FIELD_MAPPING = {
        'c1': 'supply_capacity',    # 공급능력 (MW)
        'c2': 'current_demand',     # 현재부하 (MW)
        'c5': 'reserve_power',      # 공급예비력 (MW)
        'c6': 'reserve_rate',       # 공급예비율 (%)
        'temperature': 'temperature',  # 기온
        'baseDatetime': 'timestamp',   # 기준시간
    }

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Args:
            timeout: 요청 타임아웃 (초)
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
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://epsis.kpx.or.kr',
            'Referer': 'https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMepRealChart.do?menuId=030300',
            'X-Requested-With': 'XMLHttpRequest',
        })
        return session

    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        EPSIS 응답 파싱

        EPSIS는 JavaScript 코드 형식으로 응답합니다.
        정규식을 사용하여 gridData를 추출합니다.
        """
        # 방법 1: gridData에서 직접 값 추출 (c1, c2, c5, c6 변수 할당 패턴)
        data_list = self._extract_grid_data_from_js(response_text)

        if data_list:
            logger.debug(f"gridData에서 {len(data_list)}건 추출")
            return data_list

        # 방법 2: chartData 추출 (폴백)
        chart_data = self._extract_chart_data(response_text)
        if chart_data:
            logger.debug(f"chartData에서 {len(chart_data)}건 추출")
            return chart_data

        logger.warning("데이터 추출 실패")
        return []

    def _extract_grid_data_from_js(self, response_text: str) -> List[Dict[str, Any]]:
        """
        JavaScript 코드에서 gridData 추출

        EPSIS 응답 형식:
            c1 = textFormmat("101768.0",0);
            c2 = textFormmat("72947.0",0);
            c5 = textFormmat("28821.0",0);
            c6 = textFormmat("39.5",0);
            temperature = textFormmat("",0);
            gridData.push({"year":"2025-12-17 20:25", ...});
        """
        data_list = []

        # gridData.push 블록 찾기
        # 각 블록 앞에 c1, c2, c5, c6 값이 정의됨
        pattern = r'c1\s*=\s*textFormmat\("([^"]*)".*?c2\s*=\s*textFormmat\("([^"]*)".*?c5\s*=\s*textFormmat\("([^"]*)".*?c6\s*=\s*textFormmat\("([^"]*)".*?temperature\s*=\s*textFormmat\("([^"]*)".*?gridData\.push\(\{"year":"([^"]+)"'

        matches = re.findall(pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                data_list.append({
                    'c1': match[0],  # 공급능력
                    'c2': match[1],  # 현재수요
                    'c5': match[2],  # 예비력
                    'c6': match[3],  # 예비율
                    'temperature': match[4] if match[4] else None,
                    'baseDatetime': match[5],  # 시간
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"gridData 파싱 실패: {e}")
                continue

        return data_list

    def _extract_chart_data(self, response_text: str) -> List[Dict[str, Any]]:
        """chartData 추출 (5분 간격 수요 데이터)"""
        data_list = []

        pattern = r'chartData\.push\(\{"Date":"([^"]+)",\s*"Value":"([^"]+)",\s*"Value2":"([^"]+)",\s*"Value3":"([^"]*)"\s*\}\)'

        matches = re.findall(pattern, response_text)

        for match in matches:
            try:
                data_list.append({
                    'time': match[0],
                    'value': match[1],
                    'value2': match[2],
                    'value3': match[3] if match[3] else None,
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"chartData 파싱 실패: {e}")
                continue

        return data_list

    def _extract_data_regex(self, response_text: str) -> List[Dict[str, Any]]:
        """정규식을 사용한 데이터 추출 (폴백 방식)"""
        data_list = []

        # 패턴: c1:"1350", c2:"1050" 형식
        pattern = r'c1["\']?\s*:\s*["\']?(\d+\.?\d*)["\']?\s*,\s*c2["\']?\s*:\s*["\']?(\d+\.?\d*)["\']?.*?c5["\']?\s*:\s*["\']?(\d+\.?\d*)["\']?\s*,\s*c6["\']?\s*:\s*["\']?(\d+\.?\d*)["\']?'

        matches = re.findall(pattern, response_text, re.IGNORECASE)

        for match in matches:
            try:
                data_list.append({
                    'c1': float(match[0]),
                    'c2': float(match[1]),
                    'c5': float(match[2]),
                    'c6': float(match[3]),
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"데이터 추출 실패: {e}")
                continue

        return data_list

    def fetch_realtime_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[PowerSupplyData]:
        """
        실시간 전력 수급 데이터 조회

        Args:
            start_date: 시작 날짜 (YYYYMMDD 또는 YYYY-MM-DD)
            end_date: 종료 날짜 (YYYYMMDD 또는 YYYY-MM-DD)

        Returns:
            PowerSupplyData 객체 리스트
        """
        # 날짜 기본값 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = end_date

        # 날짜 형식 정규화 (YYYYMMDD)
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        url = f"{self.BASE_URL}{self.REALTIME_ENDPOINT}"

        payload = {
            'beginDate': start_date,
            'endDate': end_date,
        }

        logger.info(f"EPSIS 실시간 데이터 요청: {start_date} ~ {end_date}")

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    data=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                logger.debug(f"응답 상태: {response.status_code}")
                logger.debug(f"응답 길이: {len(response.text)} bytes")

                # 응답 파싱
                raw_data = self._parse_response(response.text)

                if not raw_data:
                    logger.warning("응답에 데이터가 없습니다")
                    return []

                # PowerSupplyData 객체로 변환
                result = []
                for item in raw_data:
                    try:
                        data = PowerSupplyData(
                            timestamp=self._format_timestamp(item.get('baseDatetime', '')),
                            supply_capacity=self._safe_float(item.get('c1', 0)),
                            current_demand=self._safe_float(item.get('c2', 0)),
                            reserve_power=self._safe_float(item.get('c5', 0)),
                            reserve_rate=self._safe_float(item.get('c6', 0)),
                            temperature=self._safe_float(item.get('temperature')) if item.get('temperature') else None,
                        )
                        result.append(data)
                    except Exception as e:
                        logger.warning(f"데이터 변환 실패: {e}, item: {item}")
                        continue

                logger.info(f"총 {len(result)}건 데이터 수집 완료")
                return result

            except requests.RequestException as e:
                logger.error(f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 지수 백오프
                continue

        logger.error("최대 재시도 횟수 초과")
        return []

    def fetch_latest(self) -> Optional[PowerSupplyData]:
        """최신 실시간 데이터 1건 조회"""
        data = self.fetch_realtime_data()
        if data:
            # 가장 최근 데이터 반환
            return sorted(data, key=lambda x: x.timestamp, reverse=True)[0]
        return None

    @staticmethod
    def _safe_float(value: Any) -> float:
        """안전한 float 변환"""
        if value is None:
            return 0.0
        try:
            # 쉼표 제거 (천단위 구분자)
            if isinstance(value, str):
                value = value.replace(",", "")
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _format_timestamp(timestamp: Any) -> str:
        """타임스탬프 형식 정규화"""
        if not timestamp:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

        timestamp = str(timestamp)

        # 다양한 형식 처리
        formats = [
            "%Y%m%d%H%M",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d%H",
            "%Y%m%d",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp, fmt)
                return dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue

        return timestamp

    def close(self):
        """세션 종료"""
        self.session.close()


class EPSISDataStore:
    """EPSIS 데이터 저장소"""

    def __init__(self, output_path: Union[str, Path]):
        """
        Args:
            output_path: 저장 경로 (JSON 또는 CSV)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: List[PowerSupplyData], append: bool = True):
        """
        데이터 저장

        Args:
            data: PowerSupplyData 리스트
            append: 기존 파일에 추가 여부
        """
        if not data:
            logger.warning("저장할 데이터가 없습니다")
            return

        suffix = self.output_path.suffix.lower()

        if suffix == '.json':
            self._save_json(data, append)
        elif suffix == '.csv':
            self._save_csv(data, append)
        else:
            logger.error(f"지원하지 않는 파일 형식: {suffix}")

    def _save_json(self, data: List[PowerSupplyData], append: bool):
        """JSON 형식으로 저장"""
        existing_data = []

        if append and self.output_path.exists():
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # 새 데이터 추가
        new_data = [d.to_dict() for d in data]
        combined = existing_data + new_data

        # 중복 제거 (timestamp + region 기준)
        seen = set()
        unique_data = []
        for item in combined:
            key = (item['timestamp'], item.get('region', '전국'))
            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        # 시간순 정렬
        unique_data.sort(key=lambda x: x['timestamp'])

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON 저장 완료: {self.output_path} ({len(unique_data)}건)")

    def _save_csv(self, data: List[PowerSupplyData], append: bool):
        """CSV 형식으로 저장"""
        fieldnames = [
            'timestamp', 'supply_capacity', 'current_demand',
            'reserve_power', 'reserve_rate', 'temperature',
            'fetched_at', 'region', 'is_estimated'
        ]

        mode = 'a' if append and self.output_path.exists() else 'w'
        write_header = not (append and self.output_path.exists())

        with open(self.output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for item in data:
                writer.writerow(item.to_dict())

        logger.info(f"CSV 저장 완료: {self.output_path} ({len(data)}건 추가)")

    def load(self) -> List[Dict[str, Any]]:
        """저장된 데이터 로드"""
        if not self.output_path.exists():
            return []

        suffix = self.output_path.suffix.lower()

        if suffix == '.json':
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif suffix == '.csv':
            with open(self.output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)

        return []


def run_periodic_crawling(
    crawler: EPSISCrawler,
    store: Optional[EPSISDataStore],
    interval: int = 300,
    max_runs: Optional[int] = None
):
    """
    주기적 크롤링 실행

    Args:
        crawler: EPSISCrawler 인스턴스
        store: 데이터 저장소 (None이면 콘솔 출력만)
        interval: 크롤링 간격 (초)
        max_runs: 최대 실행 횟수 (None이면 무한)
    """
    logger.info(f"주기적 크롤링 시작 (간격: {interval}초)")

    run_count = 0

    try:
        while True:
            run_count += 1
            logger.info(f"=== 크롤링 실행 #{run_count} ===")

            data = crawler.fetch_realtime_data()

            if data:
                latest = data[-1]
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 최신 데이터:")
                print(f"  시점: {latest.timestamp}")
                print(f"  공급능력: {latest.supply_capacity:,.0f} MW")
                print(f"  현재수요: {latest.current_demand:,.0f} MW")
                print(f"  예비력: {latest.reserve_power:,.0f} MW")
                print(f"  예비율: {latest.reserve_rate:.1f}%")
                print(f"  상태: {latest.status}")

                if store:
                    store.save(data)

            if max_runs and run_count >= max_runs:
                logger.info(f"최대 실행 횟수({max_runs}) 도달")
                break

            logger.info(f"다음 크롤링까지 {interval}초 대기...")
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(
        description='EPSIS 실시간 전력 수급 데이터 크롤러',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 현재 시점 데이터 조회
  python epsis_crawler.py

  # 특정 날짜 범위 조회
  python epsis_crawler.py --start 2024-12-01 --end 2024-12-17

  # 주기적 크롤링 (5분 간격, 10회 실행)
  python epsis_crawler.py --periodic --interval 300 --max-runs 10

  # JSON 파일로 저장
  python epsis_crawler.py --output data/epsis_realtime.json
        """
    )

    parser.add_argument(
        '--start', '-s',
        help='시작 날짜 (YYYY-MM-DD 또는 YYYYMMDD)'
    )
    parser.add_argument(
        '--end', '-e',
        help='종료 날짜 (YYYY-MM-DD 또는 YYYYMMDD)'
    )
    parser.add_argument(
        '--output', '-o',
        help='출력 파일 경로 (JSON 또는 CSV)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='출력 형식 (기본: json)'
    )
    parser.add_argument(
        '--periodic', '-p',
        action='store_true',
        help='주기적 크롤링 모드 활성화'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=300,
        help='크롤링 간격 (초, 기본: 300)'
    )
    parser.add_argument(
        '--max-runs', '-m',
        type=int,
        help='최대 실행 횟수 (주기적 모드에서만)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    parser.add_argument(
        '--jeju', '-j',
        action='store_true',
        help='제주 추정치 포함 (전국 데이터 기반 추정)'
    )

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 크롤러 초기화
    crawler = EPSISCrawler()

    # 저장소 설정
    store = None
    if args.output:
        output_path = args.output
        if not output_path.endswith(('.json', '.csv')):
            output_path += f'.{args.format}'
        store = EPSISDataStore(output_path)

    try:
        if args.periodic:
            # 주기적 크롤링 모드
            run_periodic_crawling(
                crawler,
                store,
                interval=args.interval,
                max_runs=args.max_runs
            )
        else:
            # 단일 조회 모드
            data = crawler.fetch_realtime_data(args.start, args.end)

            if data:
                # 제주 추정기 초기화
                jeju_estimator = JejuEstimator() if args.jeju else None

                print("\n" + "="*60)
                print(" EPSIS 실시간 전력 수급 데이터 (전국)")
                print("="*60)

                for item in data[-5:]:  # 최근 5건만 출력
                    print(f"\n[{item.timestamp}] 전국")
                    print(f"  공급능력: {item.supply_capacity:>8,.0f} MW")
                    print(f"  현재수요: {item.current_demand:>8,.0f} MW")
                    print(f"  예비력:   {item.reserve_power:>8,.0f} MW")
                    print(f"  예비율:   {item.reserve_rate:>8.1f} %")
                    print(f"  이용률:   {item.utilization_rate:>8.1f} %")
                    print(f"  상태:     {item.status}")

                    # 제주 추정치 출력
                    if jeju_estimator:
                        jeju = jeju_estimator.estimate_jeju_demand(item)
                        print(f"\n[{item.timestamp}] 제주 (추정)")
                        print(f"  공급능력: {jeju.supply_capacity:>8,.0f} MW")
                        print(f"  현재수요: {jeju.current_demand:>8,.0f} MW")
                        print(f"  예비력:   {jeju.reserve_power:>8,.0f} MW")
                        print(f"  예비율:   {jeju.reserve_rate:>8.1f} %")
                        print(f"  이용률:   {jeju.utilization_rate:>8.1f} %")
                        print(f"  상태:     {jeju.status}")

                print(f"\n총 {len(data)}건 조회됨")

                if store:
                    # 제주 추정치도 함께 저장
                    if args.jeju:
                        jeju_data = [jeju_estimator.estimate_jeju_demand(d) for d in data]
                        store.save(data + jeju_data)
                    else:
                        store.save(data)
            else:
                print("데이터를 가져오지 못했습니다.")

    finally:
        crawler.close()


if __name__ == '__main__':
    main()
