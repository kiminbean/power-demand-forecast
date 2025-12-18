"""
제주 실시간 전력수급 크롤러
============================

KPX(한국전력거래소) 제주 실시간 전력수급현황 페이지 크롤링

데이터 출처: https://www.kpx.or.kr/powerinfoJeju.es?mid=a10404040000
업데이트 주기: 5분
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class JejuRealtimeData:
    """제주 실시간 전력수급 데이터"""

    timestamp: str  # 기준 시간
    supply_capacity: float  # 공급능력 (MW)
    current_demand: float  # 현재부하/수요 (MW)
    supply_reserve: float  # 공급예비력 (MW)
    operation_reserve: float  # 운영예비력 (MW)
    fetched_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "kpx.or.kr"

    @property
    def reserve_rate(self) -> float:
        """예비율 계산 (%)"""
        if self.current_demand > 0:
            return (self.supply_reserve / self.current_demand) * 100
        return 0.0

    @property
    def utilization_rate(self) -> float:
        """이용률 계산 (%)"""
        if self.supply_capacity > 0:
            return (self.current_demand / self.supply_capacity) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp,
            'supply_capacity': self.supply_capacity,
            'current_demand': self.current_demand,
            'supply_reserve': self.supply_reserve,
            'operation_reserve': self.operation_reserve,
            'reserve_rate': self.reserve_rate,
            'utilization_rate': self.utilization_rate,
            'fetched_at': self.fetched_at,
            'source': self.source,
        }


class JejuRealtimeCrawler:
    """제주 실시간 전력수급 크롤러"""

    KPX_JEJU_URL = "https://www.kpx.or.kr/powerinfoJeju.es?mid=a10404040000"

    def __init__(self, timeout: int = 30):
        """
        초기화

        Args:
            timeout: HTTP 요청 타임아웃 (초)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        })

    def close(self):
        """세션 종료"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch_realtime(self) -> Optional[JejuRealtimeData]:
        """
        제주 실시간 전력수급 데이터 조회

        Returns:
            JejuRealtimeData 또는 None (실패 시)
        """
        try:
            logger.info(f"KPX 제주 실시간 데이터 요청: {self.KPX_JEJU_URL}")

            resp = self.session.get(self.KPX_JEJU_URL, timeout=self.timeout)
            resp.raise_for_status()

            # MW 패턴으로 데이터 추출
            # 순서: 공급능력, 현재부하, 공급예비력, 운영예비력
            mw_pattern = r'([\d,]+)\s*MW'
            matches = re.findall(mw_pattern, resp.text)

            if len(matches) < 4:
                logger.error(f"데이터 추출 실패: {len(matches)}개만 발견 (최소 4개 필요)")
                return None

            # 숫자 변환 (쉼표 제거)
            values = [float(m.replace(',', '')) for m in matches[:4]]

            # 타임스탬프 추출
            timestamp = self._extract_timestamp(resp.text)

            data = JejuRealtimeData(
                timestamp=timestamp,
                supply_capacity=values[0],
                current_demand=values[1],
                supply_reserve=values[2],
                operation_reserve=values[3],
            )

            logger.info(f"제주 실시간 데이터 수집 완료: {data.current_demand:.0f} MW (예비율: {data.reserve_rate:.1f}%)")
            return data

        except requests.RequestException as e:
            logger.error(f"HTTP 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"데이터 처리 실패: {e}")
            return None

    def _extract_timestamp(self, html: str) -> str:
        """
        HTML에서 타임스탬프 추출

        Args:
            html: HTML 텍스트

        Returns:
            타임스탬프 문자열
        """
        # 패턴: 2025.12.18(목) 10:20 형식
        pattern = r'(\d{4}\.\d{2}\.\d{2})\s*\([^)]+\)\s*(\d{2}:\d{2})'
        match = re.search(pattern, html)

        if match:
            date_str = match.group(1).replace('.', '-')
            time_str = match.group(2)
            return f"{date_str} {time_str}"

        # 대체: 현재 시간 (5분 단위로 반올림)
        now = datetime.now()
        minute = (now.minute // 5) * 5
        return now.replace(minute=minute, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

    def get_status(self) -> Dict[str, Any]:
        """
        현재 전력 수급 상태 조회 (대시보드용)

        Returns:
            상태 정보 딕셔너리
        """
        data = self.fetch_realtime()

        if not data:
            return {
                'status': 'error',
                'message': '데이터 조회 실패',
            }

        # 상태 판단
        if data.reserve_rate >= 15:
            status = 'safe'
            status_text = '정상'
        elif data.reserve_rate >= 10:
            status = 'normal'
            status_text = '관심'
        elif data.reserve_rate >= 5:
            status = 'warning'
            status_text = '주의'
        else:
            status = 'danger'
            status_text = '위험'

        return {
            'status': status,
            'status_text': status_text,
            'data': data.to_dict(),
            'message': f"제주 전력수급 {status_text} (예비율: {data.reserve_rate:.1f}%)",
        }


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='제주 실시간 전력수급 크롤러')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("제주 실시간 전력수급 현황 (KPX)")
    print("=" * 60)

    with JejuRealtimeCrawler() as crawler:
        data = crawler.fetch_realtime()

        if data:
            print(f"\n📅 기준시간: {data.timestamp}")
            print(f"⚡ 공급능력: {data.supply_capacity:,.0f} MW")
            print(f"📊 현재부하: {data.current_demand:,.0f} MW")
            print(f"🔋 공급예비력: {data.supply_reserve:,.0f} MW")
            print(f"🛡️ 운영예비력: {data.operation_reserve:,.0f} MW")
            print(f"📈 예비율: {data.reserve_rate:.1f}%")
            print(f"📉 이용률: {data.utilization_rate:.1f}%")
            print()

            # 상태 확인
            status = crawler.get_status()
            print(f"상태: {status['status_text']} ({status['status']})")
        else:
            print("❌ 데이터 조회 실패")

    print("=" * 60)


if __name__ == "__main__":
    main()
