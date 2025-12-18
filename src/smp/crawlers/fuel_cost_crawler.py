"""
연료비 단가 크롤러
==================

EPSIS(전력통계정보시스템)에서 연료원별 정산단가 데이터를 수집합니다.

데이터 출처: https://epsis.kpx.or.kr/epsisnew/selectEkmaUpsBftChart.do?menuId=040701
업데이트 주기: 월간 (M+1월 초)

연료 유형:
- 원자력 (nuclear)
- 유연탄 (bituminous_coal)
- 무연탄 (anthracite)
- 유류 (oil)
- LNG
- 양수 (pumped_storage)
- 신재생 (renewable)
- 기타 (other)
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import requests
import pandas as pd

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class FuelCostData:
    """연료비 단가 데이터

    Attributes:
        date: 날짜 (YYYY-MM-DD, 월 단위 데이터는 해당 월 첫째날)
        nuclear: 원자력 단가 (원/kWh)
        bituminous_coal: 유연탄 단가 (원/kWh)
        anthracite: 무연탄 단가 (원/kWh)
        oil: 유류 단가 (원/kWh)
        lng: LNG 단가 (원/kWh)
        pumped_storage: 양수 단가 (원/kWh)
        renewable: 신재생 단가 (원/kWh)
        other: 기타 단가 (원/kWh)
        fetched_at: 수집 시점
        source: 데이터 출처
    """
    date: str
    nuclear: float = 0.0
    bituminous_coal: float = 0.0
    anthracite: float = 0.0
    oil: float = 0.0
    lng: float = 0.0
    pumped_storage: float = 0.0
    renewable: float = 0.0
    other: float = 0.0
    fetched_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "epsis.kpx.or.kr"

    @property
    def avg_thermal(self) -> float:
        """화력발전 평균 단가 (유연탄, LNG, 유류 평균)"""
        costs = [c for c in [self.bituminous_coal, self.lng, self.oil] if c > 0]
        return sum(costs) / len(costs) if costs else 0.0

    @property
    def lng_to_coal_ratio(self) -> float:
        """LNG/유연탄 단가 비율 (연료비 스위칭 지표)"""
        if self.bituminous_coal > 0:
            return self.lng / self.bituminous_coal
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        result = asdict(self)
        result['avg_thermal'] = self.avg_thermal
        result['lng_to_coal_ratio'] = self.lng_to_coal_ratio
        return result


class FuelCostCrawler:
    """EPSIS 연료비 단가 크롤러

    전력통계정보시스템(EPSIS)에서 연료원별 정산단가를 크롤링합니다.

    Example:
        >>> with FuelCostCrawler() as crawler:
        ...     data = crawler.fetch_monthly(2024)
        ...     print(f"2024년 LNG 평균: {sum(d.lng for d in data)/len(data):.2f} 원/kWh")
    """

    # EPSIS URL
    BASE_URL = "https://epsis.kpx.or.kr"
    FUEL_COST_URL = f"{BASE_URL}/epsisnew/selectEkmaUpsBftChart.do"
    FUEL_COST_API = f"{BASE_URL}/epsisnew/selectEkmaUpsBftChart.ajax"

    # 연료 유형 매핑
    FUEL_MAPPING = {
        'Value': 'nuclear',
        'Value2': 'bituminous_coal',
        'Value3': 'anthracite',
        'Value4': 'oil',
        'Value5': 'lng',
        'Value6': 'pumped_storage',
        'Value7': 'renewable',
        'Value8': 'other',
    }

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

    def fetch_monthly(self, year: int, month: Optional[int] = None) -> List[FuelCostData]:
        """월별 연료비 단가 조회

        Args:
            year: 연도 (예: 2024)
            month: 월 (1-12, None이면 전체 연도)

        Returns:
            FuelCostData 리스트
        """
        logger.info(f"EPSIS 연료비 단가 요청: {year}년 {month if month else '전체'}월")

        try:
            # 페이지 접속하여 세션 초기화
            self.session.get(self.FUEL_COST_URL, timeout=self.timeout)

            # AJAX 요청으로 데이터 가져오기
            params = {
                'selDate': f"{year}01" if not month else f"{year}{month:02d}",
                'selRegion': '',
            }

            resp = self.session.post(
                self.FUEL_COST_API,
                data=params,
                timeout=self.timeout
            )
            resp.raise_for_status()

            # 데이터 파싱
            result = self._parse_fuel_data(resp.text, year, month)

            logger.info(f"연료비 단가 수집 완료: {len(result)}건")
            return result

        except Exception as e:
            logger.error(f"연료비 단가 수집 실패: {e}")
            return []

    def fetch_range(
        self,
        start_year: int,
        end_year: int,
        start_month: int = 1,
        end_month: int = 12
    ) -> List[FuelCostData]:
        """기간별 연료비 단가 조회

        Args:
            start_year: 시작 연도
            end_year: 종료 연도
            start_month: 시작 월 (기본: 1)
            end_month: 종료 월 (기본: 12)

        Returns:
            FuelCostData 리스트
        """
        result = []

        for year in range(start_year, end_year + 1):
            sm = start_month if year == start_year else 1
            em = end_month if year == end_year else 12

            for month in range(sm, em + 1):
                monthly_data = self.fetch_monthly(year, month)
                result.extend(monthly_data)

                # Rate limiting
                import time
                time.sleep(0.5)

        return result

    def _parse_fuel_data(
        self,
        text: str,
        year: int,
        month: Optional[int] = None
    ) -> List[FuelCostData]:
        """응답 텍스트에서 연료비 데이터 파싱

        Args:
            text: 응답 텍스트
            year: 연도
            month: 월

        Returns:
            FuelCostData 리스트
        """
        result = []
        seen_dates = set()

        try:
            # chartData.push({...}) 형식 추출
            push_pattern = r'chartData\.push\(\{([^}]+)\}\)'
            matches = re.findall(push_pattern, text)

            if not matches:
                # 대안: chartData = [...] 형식
                array_pattern = r'chartData\s*=\s*\[([\s\S]*?)\];'
                match = re.search(array_pattern, text)
                if match:
                    matches = re.findall(r'\{([^}]+)\}', match.group(1))

            if not matches:
                logger.warning("chartData를 찾을 수 없습니다")
                return result

            for item_content in matches:
                try:
                    # JavaScript 객체를 JSON으로 변환
                    item_str = '{' + item_content + '}'
                    json_str = self._js_to_json(item_str)
                    item = json.loads(json_str)

                    # 날짜 형식 확인 (YYYY/MM/DD 형식만 사용, 연간 데이터 제외)
                    date_str = item.get('Date', '')
                    if not date_str or len(date_str) < 10:
                        continue

                    fuel_data = self._parse_item(item)
                    if fuel_data:
                        # 중복 제거 (같은 날짜는 한 번만)
                        if fuel_data.date in seen_dates:
                            continue
                        seen_dates.add(fuel_data.date)

                        # 연도/월 필터링
                        item_year = int(fuel_data.date[:4])
                        if item_year != year:
                            continue

                        if month:
                            item_month = int(fuel_data.date[5:7])
                            if item_month != month:
                                continue

                        result.append(fuel_data)

                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"항목 파싱 실패: {e}")
                    continue

        except Exception as e:
            logger.error(f"연료비 데이터 파싱 오류: {e}")

        # 날짜순 정렬
        result.sort(key=lambda x: x.date)
        return result

    def _parse_item(self, item: Dict[str, Any]) -> Optional[FuelCostData]:
        """단일 데이터 항목 파싱

        Args:
            item: 데이터 항목 딕셔너리

        Returns:
            FuelCostData 또는 None
        """
        try:
            # 날짜 파싱
            date_str = item.get('Date', '')
            if not date_str:
                return None

            # YYYY/MM/DD 또는 YYYY-MM-DD 형식 처리
            date_str = date_str.replace('/', '-')
            if len(date_str) == 7:  # YYYY-MM
                date_str = f"{date_str}-01"

            # 연료비 값 추출
            values = {}
            for js_key, py_key in self.FUEL_MAPPING.items():
                val = item.get(js_key)
                if val is not None:
                    values[py_key] = self._safe_float(val)
                else:
                    values[py_key] = 0.0

            return FuelCostData(
                date=date_str,
                **values
            )

        except Exception as e:
            logger.debug(f"항목 변환 실패: {e}")
            return None

    @staticmethod
    def _js_to_json(js_str: str) -> str:
        """JavaScript 객체 문자열을 JSON으로 변환

        Args:
            js_str: JavaScript 객체 문자열

        Returns:
            JSON 문자열
        """
        # Number() 래퍼 제거
        result = re.sub(r'Number\("([^"]+)"\)', r'\1', js_str)
        result = re.sub(r'Number\((\d+\.?\d*)\)', r'\1', result)

        # 키에 따옴표 추가
        result = re.sub(r'(\w+)\s*:', r'"\1":', result)

        # 작은따옴표를 큰따옴표로 변환
        result = result.replace("'", '"')

        return result

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

    def get_latest(self) -> Optional[FuelCostData]:
        """최신 연료비 단가 조회

        Returns:
            최신 FuelCostData 또는 None
        """
        now = datetime.now()
        # 전월 데이터 조회 (M+1월 초 업데이트)
        if now.day < 10:
            # 이번 달 초면 전전월 데이터
            target = now - timedelta(days=40)
        else:
            target = now - timedelta(days=10)

        data = self.fetch_monthly(target.year, target.month)
        if data:
            return data[-1]
        return None

    def get_historical_average(
        self,
        months: int = 12
    ) -> Dict[str, float]:
        """과거 평균 연료비 단가 조회

        Args:
            months: 조회 기간 (월)

        Returns:
            연료별 평균 단가 딕셔너리
        """
        end = datetime.now()
        start = end - timedelta(days=months * 30)

        data = self.fetch_range(
            start.year, end.year,
            start.month, end.month
        )

        if not data:
            return {}

        df = pd.DataFrame([d.to_dict() for d in data])
        fuel_cols = ['nuclear', 'bituminous_coal', 'anthracite', 'oil', 'lng', 'pumped_storage', 'renewable', 'other']

        return {col: float(df[col].mean()) for col in fuel_cols if col in df.columns}


class FuelCostDataStore:
    """연료비 데이터 저장소"""

    def __init__(self, output_path: Union[str, Path]):
        """초기화

        Args:
            output_path: 저장 파일 경로
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: List[FuelCostData], append: bool = True) -> int:
        """데이터 저장

        Args:
            data: FuelCostData 리스트
            append: 기존 데이터에 추가 여부

        Returns:
            저장된 레코드 수
        """
        if not data:
            logger.warning("저장할 데이터 없음")
            return 0

        df = pd.DataFrame([d.to_dict() for d in data])

        if append and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['date'], keep='last')

        df = df.sort_values('date').reset_index(drop=True)
        df.to_csv(self.output_path, index=False, encoding='utf-8-sig')

        logger.info(f"연료비 데이터 저장 완료: {len(df)}건 → {self.output_path}")
        return len(df)

    def load(self) -> List[Dict[str, Any]]:
        """데이터 로드

        Returns:
            연료비 데이터 딕셔너리 리스트
        """
        if not self.output_path.exists():
            return []

        try:
            df = pd.read_csv(self.output_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []

    def load_as_dataframe(self) -> Optional[pd.DataFrame]:
        """데이터 로드 (DataFrame)

        Returns:
            DataFrame 또는 None
        """
        if not self.output_path.exists():
            return None
        return pd.read_csv(self.output_path)


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='EPSIS 연료비 단가 크롤러')
    parser.add_argument('--year', '-y', type=int, default=2024, help='조회 연도')
    parser.add_argument('--month', '-m', type=int, help='조회 월 (1-12)')
    parser.add_argument('--output', '-o', type=str, help='저장 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("EPSIS 연료비 단가 조회")
    print("=" * 60)

    with FuelCostCrawler() as crawler:
        if args.month:
            data_list = crawler.fetch_monthly(args.year, args.month)
        else:
            data_list = crawler.fetch_monthly(args.year)

        if data_list:
            print(f"\n📅 조회 기간: {args.year}년 {args.month if args.month else '전체'}월")
            print(f"📊 수집 건수: {len(data_list)}건")
            print()

            # 최신 데이터 표시
            latest = data_list[-1]
            print(f"⚡ 최신 연료비 단가 ({latest.date}):")
            print(f"   - 원자력: {latest.nuclear:.2f} 원/kWh")
            print(f"   - 유연탄: {latest.bituminous_coal:.2f} 원/kWh")
            print(f"   - 무연탄: {latest.anthracite:.2f} 원/kWh")
            print(f"   - 유류:   {latest.oil:.2f} 원/kWh")
            print(f"   - LNG:    {latest.lng:.2f} 원/kWh")
            print(f"   - 신재생: {latest.renewable:.2f} 원/kWh")
            print()
            print(f"📈 파생 지표:")
            print(f"   - 화력 평균: {latest.avg_thermal:.2f} 원/kWh")
            print(f"   - LNG/석탄 비율: {latest.lng_to_coal_ratio:.2f}")

            # 저장
            if args.output:
                store = FuelCostDataStore(args.output)
                store.save(data_list, append=True)

        else:
            print("❌ 연료비 단가 조회 실패")
            print("\n참고: EPSIS 데이터는 2002년부터 제공됩니다.")
            print("최근 연도 데이터는 M+1월 초에 업데이트됩니다.")

    print("=" * 60)


if __name__ == "__main__":
    main()
