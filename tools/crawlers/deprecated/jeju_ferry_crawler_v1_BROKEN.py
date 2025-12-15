#!/usr/bin/env python3
"""
제주항 여객선 승객 데이터 크롤러
- 본토 ↔ 제주 여객선 입출항 승객 수 수집
- 공공데이터포털 파일 데이터 + 웹 크롤링 지원

Why: 항공 외에 여객선을 통한 입출도객도 체류인구에 영향
     전체 입도객의 약 5-10%를 차지하지만 무시할 수 없음

주요 항로:
- 제주 ↔ 완도 (주력)
- 제주 ↔ 우수영(해남)
- 제주 ↔ 녹동

Usage:
    from tools.crawlers.jeju_ferry_crawler import JejuFerryCrawler
    
    crawler = JejuFerryCrawler()
    
    # 특정 날짜 수집
    data = crawler.get_daily_passengers("2024-01-15")
    
    # 기간 수집
    df = crawler.get_passengers_range("2024-01-01", "2024-12-31")
    
    # CSV 저장
    crawler.save_to_csv(df, "jeju_ferry_passengers.csv")
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
from io import StringIO

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from .jeju_transport_config import (
    FERRY_ROUTES,
    MAINLAND_FERRY_ROUTES,
    DATA_GO_KR_CONFIG,
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
class DailyFerryPassengers:
    """일별 여객선 승객 데이터"""
    date: str                # YYYY-MM-DD
    arrival_total: int       # 제주 도착 (입항)
    departure_total: int     # 제주 출발 (출항)
    net_flow: int           # 순 유입 (도착 - 출발)
    routes: Dict[str, Dict] # 항로별 상세 (optional)
    source: str             # 데이터 출처
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DailyFerryPassengers':
        return cls(**data)
    
    def to_dict(self) -> dict:
        return asdict(self)


class JejuFerryCrawler:
    """
    제주 여객선 승객 데이터 크롤러
    
    데이터 수집 전략:
    1. 공공데이터포털 파일 데이터 (월별/연별)
    2. 한국해양교통안전공단 데이터
    3. 제주특별자치도 통계
    
    Note: 여객선 데이터는 일별 공개가 제한적이므로
          월별 데이터를 일별로 분배(Distribution)하는 전략 사용
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            api_key: 공공데이터포털 API 키
            cache_dir: 캐시 디렉토리 경로
            use_cache: 캐시 사용 여부
        """
        self.api_key = api_key or os.environ.get("DATA_GO_KR_API_KEY")
        self.use_cache = use_cache
        
        # 캐시 디렉토리 설정
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache" / "ferry"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(
            timeout=CRAWL_CONFIG["timeout"],
            headers=CRAWL_CONFIG["headers"],
            follow_redirects=True,
        )
        
        # 월별 데이터 캐시
        self._monthly_data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("JejuFerryCrawler 초기화 완료")
    
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
    ) -> Optional[DailyFerryPassengers]:
        """
        특정 날짜의 여객선 승객 수 조회
        
        Note: 여객선은 일별 데이터가 제한적이므로
              월별 데이터에서 추정하거나, 평균값 사용
        
        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            force_refresh: 캐시 무시하고 새로 수집
            
        Returns:
            DailyFerryPassengers 또는 None
        """
        # 캐시 확인
        if self.use_cache and not force_refresh:
            cached = self._load_from_cache(date)
            if cached:
                logger.debug(f"캐시에서 로드: {date}")
                return cached
        
        # 데이터 수집 시도
        data = None
        
        # 1차: 월별 데이터에서 일별 추정
        data = self._estimate_from_monthly(date)
        
        # 2차: 공공데이터포털 파일에서 직접 검색
        if not data:
            data = self._fetch_from_data_go_kr(date)
        
        # 3차: 제주도청 통계에서 크롤링
        if not data:
            data = self._fetch_from_jeju_gov(date)
        
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
        기간별 여객선 승객 수 조회
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            DataFrame with columns:
            - date, arrival_total, departure_total, net_flow, source
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = []
        current = start
        total_days = (end - start).days + 1
        
        logger.info(f"여객선 승객 데이터 수집 시작: {start_date} ~ {end_date} ({total_days}일)")
        
        # 월별 데이터 미리 로드
        self._preload_monthly_data(start_date, end_date)
        
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
            
            current += timedelta(days=1)
        
        logger.info(f"수집 완료: {len(results)}일 데이터")
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        
        # routes 컬럼은 dict이므로 제거하거나 문자열로 변환
        if 'routes' in df.columns:
            df['routes'] = df['routes'].apply(lambda x: json.dumps(x) if x else None)
        
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_monthly_data(
        self,
        year: int,
        month: int,
    ) -> Optional[pd.DataFrame]:
        """
        월별 여객선 통계 조회
        
        Args:
            year: 연도
            month: 월
            
        Returns:
            월별 통계 DataFrame
        """
        cache_key = f"{year:04d}-{month:02d}"
        
        if cache_key in self._monthly_data_cache:
            return self._monthly_data_cache[cache_key]
        
        # 공공데이터포털에서 연간 파일 다운로드 후 필터링
        df = self._download_yearly_data(year)
        
        if df is not None and not df.empty:
            # 월 필터링 (컬럼명은 데이터에 따라 조정)
            if '연월' in df.columns:
                monthly = df[df['연월'].str.contains(f"{year}{month:02d}")]
            elif '월' in df.columns:
                monthly = df[df['월'] == month]
            else:
                monthly = df
            
            self._monthly_data_cache[cache_key] = monthly
            return monthly
        
        return None
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        DataFrame을 CSV로 저장
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
    # 내부 메서드: 데이터 수집
    # =========================================================================
    
    def _preload_monthly_data(self, start_date: str, end_date: str):
        """기간 내 월별 데이터 미리 로드"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        years = set()
        current = start
        while current <= end:
            years.add(current.year)
            current += timedelta(days=32)  # 대략 한 달씩
        
        for year in years:
            self._download_yearly_data(year)
    
    def _download_yearly_data(self, year: int) -> Optional[pd.DataFrame]:
        """
        연간 여객선 수송실적 데이터 다운로드
        
        소스: 공공데이터포털 - 제주특별자치도_여객선수송실적현황
        """
        cache_key = f"yearly_{year}"
        
        if cache_key in self._monthly_data_cache:
            return self._monthly_data_cache[cache_key]
        
        try:
            # 공공데이터포털 파일 다운로드 URL
            # Note: 실제 URL은 파일이 업데이트될 때마다 변경될 수 있음
            url = f"https://www.data.go.kr/data/15056326/fileData.do"
            
            response = self.client.get(url)
            
            # 페이지에서 실제 파일 URL 추출
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # CSV/Excel 파일 링크 찾기
            file_links = soup.find_all('a', href=re.compile(r'\.(csv|xlsx|xls)'))
            
            if not file_links:
                logger.warning(f"여객선 데이터 파일 없음: {year}")
                return None
            
            # 최신 파일 다운로드
            file_url = file_links[0].get('href')
            if not file_url.startswith('http'):
                file_url = 'https://www.data.go.kr' + file_url
            
            file_response = self.client.get(file_url)
            
            # 파일 형식에 따라 파싱
            if file_url.endswith('.csv'):
                df = pd.read_csv(StringIO(file_response.text), encoding='utf-8-sig')
            else:
                # Excel 파일
                df = pd.read_excel(file_response.content)
            
            self._monthly_data_cache[cache_key] = df
            logger.info(f"여객선 연간 데이터 로드: {year} ({len(df)}행)")
            
            return df
            
        except Exception as e:
            logger.warning(f"연간 데이터 다운로드 실패 ({year}): {e}")
            return None
    
    def _estimate_from_monthly(self, date: str) -> Optional[DailyFerryPassengers]:
        """
        월별 데이터에서 일별 승객 수 추정
        
        전략:
        1. 해당 월의 총 승객 수를 일수로 나눔 (균등 분배)
        2. 주말/공휴일 가중치 적용 (선택적)
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        year, month = dt.year, dt.month
        
        # 월별 데이터 조회
        monthly_df = self.get_monthly_data(year, month)
        
        if monthly_df is None or monthly_df.empty:
            return self._get_default_estimate(date)
        
        try:
            # 제주 관련 항로만 필터링
            jeju_df = monthly_df[
                monthly_df['항로'].str.contains('제주', na=False) |
                monthly_df['항로명'].str.contains('제주', na=False) if '항로명' in monthly_df.columns else True
            ]
            
            if jeju_df.empty:
                return self._get_default_estimate(date)
            
            # 입항(도착) / 출항(출발) 집계
            # 컬럼명은 데이터에 따라 조정 필요
            arrival_col = None
            departure_col = None
            
            for col in jeju_df.columns:
                if '입항' in col or '도착' in col:
                    arrival_col = col
                elif '출항' in col or '출발' in col:
                    departure_col = col
            
            if arrival_col and departure_col:
                monthly_arrival = jeju_df[arrival_col].sum()
                monthly_departure = jeju_df[departure_col].sum()
            else:
                # 합계 컬럼 사용
                total_col = [c for c in jeju_df.columns if '합계' in c or '여객' in c]
                if total_col:
                    total = jeju_df[total_col[0]].sum()
                    monthly_arrival = total // 2
                    monthly_departure = total // 2
                else:
                    return self._get_default_estimate(date)
            
            # 해당 월의 일수
            import calendar
            days_in_month = calendar.monthrange(year, month)[1]
            
            # 일별 균등 분배
            daily_arrival = int(monthly_arrival / days_in_month)
            daily_departure = int(monthly_departure / days_in_month)
            
            return DailyFerryPassengers(
                date=date,
                arrival_total=daily_arrival,
                departure_total=daily_departure,
                net_flow=daily_arrival - daily_departure,
                routes={},
                source="monthly_estimate",
            )
            
        except Exception as e:
            logger.warning(f"월별 데이터 추정 실패: {e}")
            return self._get_default_estimate(date)
    
    def _get_default_estimate(self, date: str) -> DailyFerryPassengers:
        """
        기본 추정값 반환
        
        제주 여객선 평균 일 수송량 기반 추정:
        - 일 평균 약 1,500~2,500명 (입도 + 출도)
        - 계절 변동 고려
        """
        dt = datetime.strptime(date, "%Y-%m-%d")
        month = dt.month
        
        # 계절별 가중치 (성수기 vs 비수기)
        seasonal_weights = {
            1: 0.8, 2: 0.7, 3: 0.9, 4: 1.0,
            5: 1.1, 6: 1.0, 7: 1.3, 8: 1.4,
            9: 1.0, 10: 1.1, 11: 0.9, 12: 0.8,
        }
        
        base_passengers = 1500  # 기본 일 승객 수 (편도)
        weight = seasonal_weights.get(month, 1.0)
        
        arrival = int(base_passengers * weight)
        departure = int(base_passengers * weight)
        
        return DailyFerryPassengers(
            date=date,
            arrival_total=arrival,
            departure_total=departure,
            net_flow=0,  # 장기적으로 입출도 균형
            routes={},
            source="default_estimate",
        )
    
    def _fetch_from_data_go_kr(self, date: str) -> Optional[DailyFerryPassengers]:
        """공공데이터포털에서 직접 검색"""
        # 월별 데이터 기반이므로 _estimate_from_monthly로 대체
        return None
    
    def _fetch_from_jeju_gov(self, date: str) -> Optional[DailyFerryPassengers]:
        """제주도청 통계에서 크롤링"""
        try:
            url = "https://www.jeju.go.kr/open/open/iopenboard.htm"
            params = {"category": "1822"}  # 여객선 수송실적 카테고리
            
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            # HTML 파싱 및 데이터 추출
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 데이터 테이블 찾기
            # Note: 실제 구조에 따라 조정 필요
            
            return None  # 구현 필요
            
        except Exception as e:
            logger.warning(f"제주도청 크롤링 실패: {e}")
            return None
    
    # =========================================================================
    # 캐시 관리
    # =========================================================================
    
    def _get_cache_path(self, date: str) -> Path:
        """캐시 파일 경로 반환"""
        year_month = date[:7]  # YYYY-MM
        return self.cache_dir / year_month / f"{date}.json"
    
    def _load_from_cache(self, date: str) -> Optional[DailyFerryPassengers]:
        """캐시에서 데이터 로드"""
        cache_path = self._get_cache_path(date)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return DailyFerryPassengers.from_dict(data)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            return None
    
    def _save_to_cache(self, date: str, data: DailyFerryPassengers):
        """데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def clear_cache(self):
        """캐시 삭제"""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("캐시 삭제 완료")


# =============================================================================
# CLI 인터페이스
# =============================================================================

def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="제주 여객선 승객 데이터 크롤러",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 특정 날짜 수집
  python jeju_ferry_crawler.py --date 2024-01-15
  
  # 기간 수집
  python jeju_ferry_crawler.py --start 2024-01-01 --end 2024-12-31
  
  # CSV 저장
  python jeju_ferry_crawler.py --start 2024-01-01 --end 2024-12-31 --output jeju_ferry.csv
        """
    )
    
    parser.add_argument('--date', type=str, help='특정 날짜 (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='출력 CSV 파일명')
    parser.add_argument('--no-cache', action='store_true', help='캐시 사용 안함')
    parser.add_argument('--clear-cache', action='store_true', help='캐시 삭제')
    
    args = parser.parse_args()
    
    # 크롤러 초기화
    crawler = JejuFerryCrawler(use_cache=not args.no_cache)
    
    # 캐시 삭제
    if args.clear_cache:
        crawler.clear_cache()
        return
    
    # 데이터 수집
    if args.date:
        # 단일 날짜
        data = crawler.get_daily_passengers(args.date)
        if data:
            print(f"\n=== {args.date} 제주 여객선 승객 현황 ===")
            print(f"도착 (입항): {data.arrival_total:,}명")
            print(f"출발 (출항): {data.departure_total:,}명")
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
