#!/usr/bin/env python3
"""
제주공항 항공 승객 데이터 크롤러
- 일별 도착(Arrival) / 출발(Departure) 승객 수 수집
- 공공데이터포털 API + 한국공항공사 웹 크롤링 지원

Why: 제주도 체류인구 = 기존인구 + (입도객 - 출도객)
     전력 수요 예측 모델의 핵심 피처

Usage:
    from tools.crawlers.jeju_air_crawler import JejuAirCrawler
    
    crawler = JejuAirCrawler()
    
    # 특정 날짜 수집
    data = crawler.get_daily_passengers("2024-01-15")
    
    # 기간 수집
    df = crawler.get_passengers_range("2024-01-01", "2024-12-31")
    
    # CSV 저장
    crawler.save_to_csv(df, "jeju_air_passengers.csv")
"""

import os
import re
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from .jeju_transport_config import (
    AIRPORT_CODES,
    DATA_GO_KR_CONFIG,
    AIR_PORTAL_CONFIG,
    KAC_CONFIG,
    CRAWL_CONFIG,
    validate_passenger_count,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DailyAirPassengers:
    """일별 항공 승객 데이터"""
    date: str                    # YYYY-MM-DD
    arrival_domestic: int        # 국내선 도착
    arrival_international: int   # 국제선 도착
    departure_domestic: int      # 국내선 출발
    departure_international: int # 국제선 출발
    arrival_total: int           # 총 도착
    departure_total: int         # 총 출발
    net_flow: int               # 순 유입 (도착 - 출발)
    source: str                  # 데이터 출처
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DailyAirPassengers':
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)


class JejuAirCrawler:
    """
    제주공항 항공 승객 데이터 크롤러
    
    데이터 수집 전략:
    1. 공공데이터포털 API (우선) - API 키 필요
    2. 한국공항공사 웹 크롤링 (대안)
    3. 항공정보포털 웹 크롤링 (백업)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            api_key: 공공데이터포털 API 키 (없으면 환경변수에서 읽음)
            cache_dir: 캐시 디렉토리 경로
            use_cache: 캐시 사용 여부
        """
        self.api_key = api_key or os.environ.get("DATA_GO_KR_API_KEY")
        self.use_cache = use_cache
        
        # 캐시 디렉토리 설정
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache" / "air"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(
            timeout=CRAWL_CONFIG["timeout"],
            headers=CRAWL_CONFIG["headers"],
            follow_redirects=True,
        )
        
        # 제주공항 코드
        self.jeju_code = AIRPORT_CODES["제주"]  # CJU
        
        logger.info(f"JejuAirCrawler 초기화 완료 (API 키: {'있음' if self.api_key else '없음'})")
    
    def __del__(self):
        """클라이언트 정리"""
        if hasattr(self, 'client'):
            self.client.close()
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_daily_passengers(
        self,
        date: str,
        force_refresh: bool = False,
    ) -> Optional[DailyAirPassengers]:
        """
        특정 날짜의 항공 승객 수 조회
        
        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            force_refresh: 캐시 무시하고 새로 수집
            
        Returns:
            DailyAirPassengers 또는 None
        """
        # 캐시 확인
        if self.use_cache and not force_refresh:
            cached = self._load_from_cache(date)
            if cached:
                logger.debug(f"캐시에서 로드: {date}")
                return cached
        
        # 데이터 수집 시도
        data = None
        
        # 1차: 공공데이터포털 API
        if self.api_key:
            data = self._fetch_from_data_go_kr(date)
        
        # 2차: 한국공항공사 웹 크롤링
        if not data:
            data = self._fetch_from_kac_web(date)
        
        # 3차: 항공정보포털 웹 크롤링
        if not data:
            data = self._fetch_from_airportal(date)
        
        # 캐시 저장
        if data and self.use_cache:
            self._save_to_cache(date, data)
        
        return data
    
    def get_passengers_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        기간별 항공 승객 수 조회
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            DataFrame with columns:
            - date, arrival_domestic, arrival_international,
            - departure_domestic, departure_international,
            - arrival_total, departure_total, net_flow, source
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = []
        current = start
        total_days = (end - start).days + 1
        
        logger.info(f"항공 승객 데이터 수집 시작: {start_date} ~ {end_date} ({total_days}일)")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            try:
                data = self.get_daily_passengers(date_str)
                if data:
                    results.append(data.to_dict())
                else:
                    logger.warning(f"데이터 없음: {date_str}")
            except Exception as e:
                logger.error(f"수집 실패 ({date_str}): {e}")
            
            # 진행 상황 콜백
            if progress_callback:
                progress = (current - start).days + 1
                progress_callback(progress, total_days, date_str)
            
            # 요청 간격
            time.sleep(CRAWL_CONFIG["request_delay"])
            current += timedelta(days=1)
        
        logger.info(f"수집 완료: {len(results)}일 데이터")
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        DataFrame을 CSV로 저장
        
        Args:
            df: 저장할 DataFrame
            filename: 파일명
            output_dir: 출력 디렉토리 (기본: data/raw/)
            
        Returns:
            저장된 파일 경로
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(__file__).parent.parent.parent / "data" / "raw"
        
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"저장 완료: {filepath}")
        
        return str(filepath)
    
    # =========================================================================
    # 공공데이터포털 API
    # =========================================================================
    
    def _fetch_from_data_go_kr(self, date: str) -> Optional[DailyAirPassengers]:
        """
        공공데이터포털 API로 데이터 수집
        
        API: 한국공항공사_공항별 여객실적
        """
        if not self.api_key:
            logger.debug("API 키 없음, 스킵")
            return None
        
        try:
            # API 엔드포인트
            url = "https://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR"
            
            # 파라미터
            params = {
                "serviceKey": self.api_key,
                "from_time": date.replace("-", ""),  # YYYYMMDD
                "to_time": date.replace("-", ""),
                "airport": self.jeju_code,
                "type": "json",
            }
            
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 파싱
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if not items:
                logger.debug(f"공공데이터포털 API: 데이터 없음 ({date})")
                return None
            
            # 집계
            arrival_dom, arrival_int = 0, 0
            departure_dom, departure_int = 0, 0
            
            for item in items:
                if isinstance(item, dict):
                    # 국내선/국제선, 출발/도착 구분
                    line_type = item.get("line", "")  # D: 국내선, I: 국제선
                    io_type = item.get("io", "")      # I: 도착, O: 출발
                    pax = int(item.get("sumPax", 0))
                    
                    if io_type == "I":  # 도착
                        if line_type == "D":
                            arrival_dom += pax
                        else:
                            arrival_int += pax
                    else:  # 출발
                        if line_type == "D":
                            departure_dom += pax
                        else:
                            departure_int += pax
            
            arrival_total = arrival_dom + arrival_int
            departure_total = departure_dom + departure_int
            
            return DailyAirPassengers(
                date=date,
                arrival_domestic=arrival_dom,
                arrival_international=arrival_int,
                departure_domestic=departure_dom,
                departure_international=departure_int,
                arrival_total=arrival_total,
                departure_total=departure_total,
                net_flow=arrival_total - departure_total,
                source="data.go.kr",
            )
            
        except Exception as e:
            logger.warning(f"공공데이터포털 API 실패: {e}")
            return None
    
    # =========================================================================
    # 한국공항공사 웹 크롤링
    # =========================================================================
    
    def _fetch_from_kac_web(self, date: str) -> Optional[DailyAirPassengers]:
        """
        한국공항공사 웹사이트에서 데이터 수집
        
        URL: https://www.airport.co.kr/www/cms/frFlightStatsCon/
        """
        try:
            # 통계 페이지 URL
            url = "https://www.airport.co.kr/www/cms/frFlightStatsCon/passengerStats.do"
            
            # POST 파라미터
            dt = datetime.strptime(date, "%Y-%m-%d")
            params = {
                "MENU_ID": "1240",
                "sYyyy": dt.strftime("%Y"),
                "sMm": dt.strftime("%m"),
                "sDd": dt.strftime("%d"),
                "eYyyy": dt.strftime("%Y"),
                "eMm": dt.strftime("%m"),
                "eDd": dt.strftime("%d"),
                "airportCode": self.jeju_code,
            }
            
            response = self.client.post(url, data=params)
            response.raise_for_status()
            
            # HTML 파싱
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 테이블에서 데이터 추출
            table = soup.find('table', class_='tbl_type1')
            if not table:
                logger.debug(f"KAC 웹: 테이블 없음 ({date})")
                return None
            
            # 숫자 추출 (간단한 패턴 매칭)
            # Note: 실제 HTML 구조에 따라 조정 필요
            numbers = re.findall(r'[\d,]+', table.get_text())
            numbers = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
            
            if len(numbers) < 4:
                logger.debug(f"KAC 웹: 데이터 부족 ({date})")
                return None
            
            # 추정 매핑 (페이지 구조에 따라 조정)
            arrival_total = numbers[0] if len(numbers) > 0 else 0
            departure_total = numbers[1] if len(numbers) > 1 else 0
            
            return DailyAirPassengers(
                date=date,
                arrival_domestic=arrival_total,  # 상세 구분 불가
                arrival_international=0,
                departure_domestic=departure_total,
                departure_international=0,
                arrival_total=arrival_total,
                departure_total=departure_total,
                net_flow=arrival_total - departure_total,
                source="kac_web",
            )
            
        except Exception as e:
            logger.warning(f"KAC 웹 크롤링 실패: {e}")
            return None
    
    # =========================================================================
    # 항공정보포털 웹 크롤링
    # =========================================================================
    
    def _fetch_from_airportal(self, date: str) -> Optional[DailyAirPassengers]:
        """
        항공정보포털(airportal.go.kr)에서 데이터 수집
        
        Note: 실시간 데이터 위주로, 과거 데이터는 제한적
        """
        try:
            # 통계 페이지
            url = "https://www.airportal.go.kr/knowledge/statsnew/airport/AirportD.jsp"
            
            dt = datetime.strptime(date, "%Y-%m-%d")
            params = {
                "mode": "list",
                "iArport": self.jeju_code,
                "startDt": date.replace("-", ""),
                "endDt": date.replace("-", ""),
            }
            
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 테이블 파싱
            # Note: 실제 HTML 구조에 따라 조정 필요
            table = soup.find('table')
            if not table:
                logger.debug(f"AirPortal: 테이블 없음 ({date})")
                return None
            
            numbers = re.findall(r'[\d,]+', table.get_text())
            numbers = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
            
            if len(numbers) < 2:
                return None
            
            arrival_total = numbers[0] if len(numbers) > 0 else 0
            departure_total = numbers[1] if len(numbers) > 1 else 0
            
            return DailyAirPassengers(
                date=date,
                arrival_domestic=arrival_total,
                arrival_international=0,
                departure_domestic=departure_total,
                departure_international=0,
                arrival_total=arrival_total,
                departure_total=departure_total,
                net_flow=arrival_total - departure_total,
                source="airportal",
            )
            
        except Exception as e:
            logger.warning(f"항공정보포털 크롤링 실패: {e}")
            return None
    
    # =========================================================================
    # 캐시 관리
    # =========================================================================
    
    def _get_cache_path(self, date: str) -> Path:
        """캐시 파일 경로 반환"""
        year_month = date[:7]  # YYYY-MM
        return self.cache_dir / year_month / f"{date}.json"
    
    def _load_from_cache(self, date: str) -> Optional[DailyAirPassengers]:
        """캐시에서 데이터 로드"""
        cache_path = self._get_cache_path(date)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return DailyAirPassengers.from_dict(data)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            return None
    
    def _save_to_cache(self, date: str, data: DailyAirPassengers):
        """데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def clear_cache(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """캐시 삭제"""
        import shutil
        
        if start_date is None and end_date is None:
            # 전체 캐시 삭제
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("전체 캐시 삭제 완료")
        else:
            # 기간별 캐시 삭제
            start = datetime.strptime(start_date or "2013-01-01", "%Y-%m-%d")
            end = datetime.strptime(end_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
            
            current = start
            deleted = 0
            while current <= end:
                cache_path = self._get_cache_path(current.strftime("%Y-%m-%d"))
                if cache_path.exists():
                    cache_path.unlink()
                    deleted += 1
                current += timedelta(days=1)
            
            logger.info(f"캐시 삭제 완료: {deleted}개 파일")


# =============================================================================
# CLI 인터페이스
# =============================================================================

def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="제주공항 항공 승객 데이터 크롤러",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 특정 날짜 수집
  python jeju_air_crawler.py --date 2024-01-15
  
  # 기간 수집
  python jeju_air_crawler.py --start 2024-01-01 --end 2024-12-31
  
  # CSV 저장
  python jeju_air_crawler.py --start 2024-01-01 --end 2024-12-31 --output jeju_air.csv
        """
    )
    
    parser.add_argument('--date', type=str, help='특정 날짜 (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='출력 CSV 파일명')
    parser.add_argument('--api-key', type=str, help='공공데이터포털 API 키')
    parser.add_argument('--no-cache', action='store_true', help='캐시 사용 안함')
    parser.add_argument('--clear-cache', action='store_true', help='캐시 삭제')
    
    args = parser.parse_args()
    
    # 크롤러 초기화
    crawler = JejuAirCrawler(
        api_key=args.api_key,
        use_cache=not args.no_cache,
    )
    
    # 캐시 삭제
    if args.clear_cache:
        crawler.clear_cache(args.start, args.end)
        return
    
    # 데이터 수집
    if args.date:
        # 단일 날짜
        data = crawler.get_daily_passengers(args.date)
        if data:
            print(f"\n=== {args.date} 제주공항 승객 현황 ===")
            print(f"도착: {data.arrival_total:,}명 (국내 {data.arrival_domestic:,} / 국제 {data.arrival_international:,})")
            print(f"출발: {data.departure_total:,}명 (국내 {data.departure_domestic:,} / 국제 {data.departure_international:,})")
            print(f"순 유입: {data.net_flow:+,}명")
            print(f"데이터 출처: {data.source}")
        else:
            print(f"데이터를 수집할 수 없습니다: {args.date}")
    
    elif args.start and args.end:
        # 기간 수집
        def progress(current, total, date):
            pct = current / total * 100
            print(f"\r진행: {current}/{total} ({pct:.1f}%) - {date}", end='', flush=True)
        
        df = crawler.get_passengers_range(args.start, args.end, progress_callback=progress)
        print()  # 줄바꿈
        
        if not df.empty:
            print(f"\n=== 수집 결과 ===")
            print(f"기간: {args.start} ~ {args.end}")
            print(f"데이터: {len(df)}일")
            print(f"총 도착: {df['arrival_total'].sum():,}명")
            print(f"총 출발: {df['departure_total'].sum():,}명")
            print(f"순 유입 합계: {df['net_flow'].sum():+,}명")
            
            if args.output:
                filepath = crawler.save_to_csv(df, args.output)
                print(f"\n저장 완료: {filepath}")
        else:
            print("수집된 데이터가 없습니다.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
