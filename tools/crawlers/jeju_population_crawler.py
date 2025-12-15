#!/usr/bin/env python3
"""
제주도 체류인구 추정 크롤러 v2.0
Convolution (Survival Function) 방식으로 체류 관광객 계산

핵심 공식:
    체류_관광객(t) = Σ 입도객(t-k) × P(k일차 잔존)
    체류_인구(t) = 주민등록인구(t) + 체류_관광객(t)

Survival Rates (평균 체류 2.95일):
    [1.0, 0.90, 0.65, 0.30, 0.10, 0.0]
    - Day 0: 100% (오늘 입도)
    - Day 1: 90% (1일 체류자 10% 출발)
    - Day 2: 65% (2일 체류자 25% 출발)
    - Day 3: 30% (3일 체류자 35% 출발)
    - Day 4: 10% (4일 체류자 20% 출발)
    - Day 5: 0% (5일+ 체류자 모두 출발)

Usage:
    from tools.crawlers.jeju_population_crawler import JejuPopulationCrawler
    
    crawler = JejuPopulationCrawler()
    df = crawler.calculate_from_visitors_data("data/raw/jeju_daily_visitors_v10.csv")
    crawler.save_to_csv(df, "jeju_daily_population.csv")

Gemini 검증 결과: ✅ PASS (2024-12-15)
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 상수 정의
# =============================================================================

# 제주도 주민등록인구 (연도별)
JEJU_BASE_POPULATION = {
    2013: 583_284,
    2014: 592_131,
    2015: 604_670,
    2016: 620_817,
    2017: 641_597,
    2018: 661_190,
    2019: 671_811,
    2020: 674_635,
    2021: 679_896,
    2022: 682_458,
    2023: 680_398,
    2024: 680_000,
    2025: 680_000,
}

# Survival Function (잔존율)
# 체류일수 확률: P(1일)=0.10, P(2일)=0.25, P(3일)=0.35, P(4일)=0.20, P(5일+)=0.10
# → 잔존율 = 아직 제주에 남아있을 확률
DEFAULT_SURVIVAL_RATES = np.array([1.0, 0.90, 0.65, 0.30, 0.10, 0.0])

# 교통수단 비율 (제주도 공식 통계 기준)
TRANSPORT_RATIO = {
    'air': 0.94,    # 항공 94%
    'ferry': 0.06,  # 여객선 6%
}


@dataclass
class DailyPopulation:
    """일별 체류인구 데이터"""
    date: str
    base_population: int
    air_arrival: int
    air_departure: int
    ferry_arrival: int
    ferry_departure: int
    total_arrival: int
    total_departure: int
    net_flow: int
    tourist_stock: int  # 체류 관광객
    estimated_population: int
    data_source: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class JejuPopulationCrawler:
    """
    제주도 일별 체류인구 추정 크롤러 v2.0
    
    Convolution (Survival Function) 방식으로 체류 관광객을 계산합니다.
    누적 오차(Drift) 문제가 없는 안정적인 계산 방식입니다.
    """
    
    def __init__(
        self,
        survival_rates: Optional[np.ndarray] = None,
        transport_ratio: Optional[Dict[str, float]] = None,
        api_key: Optional[str] = None,  # 호환성 유지
        use_cache: bool = True,         # 호환성 유지
        ferry_weight: float = 1.0,      # 호환성 유지
    ):
        """
        Args:
            survival_rates: 잔존율 배열 (기본: [1.0, 0.9, 0.65, 0.3, 0.1, 0.0])
            transport_ratio: 교통수단 비율 (air, ferry)
        """
        self.survival_rates = survival_rates if survival_rates is not None else DEFAULT_SURVIVAL_RATES
        self.transport_ratio = transport_ratio or TRANSPORT_RATIO
        self.max_stay = len(self.survival_rates)
        
        # 평균 체류일수 = Σ(survival_rates)
        self.avg_stay_days = float(self.survival_rates.sum())
        
        # 출력 디렉토리
        self.output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"JejuPopulationCrawler v2.0 초기화")
        logger.info(f"  - Survival Rates: {self.survival_rates}")
        logger.info(f"  - 평균 체류일수: {self.avg_stay_days:.2f}일")
    
    def calculate_from_visitors_data(
        self,
        visitors_csv_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        기존 입도객 CSV 파일로부터 체류인구 계산
        
        Args:
            visitors_csv_path: 입도객 CSV 파일 경로
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            체류인구 DataFrame
        """
        logger.info(f"입도객 데이터 로드: {visitors_csv_path}")
        
        # 데이터 로드
        df = pd.read_csv(visitors_csv_path, encoding='utf-8-sig')
        
        # 컬럼명 정규화
        col_mapping = {
            '날짜': 'date', '일별_입도객수': 'arrival',
            '데이터소스': 'source', '비고': 'note',
        }
        df.columns = [col_mapping.get(c, c) for c in df.columns]
        
        if 'date' not in df.columns or 'arrival' not in df.columns:
            raise ValueError("CSV에 'date'와 'arrival' 컬럼이 필요합니다.")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"  로드 완료: {len(df):,}일 ({df['date'].min()} ~ {df['date'].max()})")
        
        # 기간 필터링
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        df = df.reset_index(drop=True)
        logger.info(f"  필터링 후: {len(df):,}일")
        
        # 주민등록인구 매핑
        df['year'] = df['date'].dt.year
        df['base_population'] = df['year'].map(JEJU_BASE_POPULATION)
        df['base_population'] = df['base_population'].fillna(680000).astype(int)
        
        # =====================================================
        # 핵심: Convolution 방식 체류 관광객 계산
        # =====================================================
        logger.info("체류 관광객 계산 중 (Convolution 방식)...")
        
        arrivals = df['arrival'].values
        tourist_stock = np.zeros(len(arrivals))
        
        for i in range(len(arrivals)):
            stock = 0
            for k in range(min(self.max_stay, i + 1)):
                stock += arrivals[i - k] * self.survival_rates[k]
            tourist_stock[i] = stock
        
        df['tourist_stock'] = tourist_stock.astype(int)
        
        # 총 체류인구 = 주민등록인구 + 체류 관광객
        df['estimated_population'] = df['base_population'] + df['tourist_stock']
        
        # 출도객 추정 (참고용)
        df = self._estimate_departure(df)
        
        # 교통수단별 분리
        df = self._split_by_transport(df)
        
        # 출력 포맷
        df = self._format_output(df)
        
        logger.info(f"체류인구 계산 완료: {len(df):,}일")
        
        return df
    
    def _estimate_departure(self, df: pd.DataFrame) -> pd.DataFrame:
        """출도객 추정 (체류일수 확률 분포 기반)"""
        stay_prob = {1: 0.10, 2: 0.25, 3: 0.35, 4: 0.20, 5: 0.10}
        
        df['departure'] = 0.0
        for lag, prob in stay_prob.items():
            df['departure'] += df['arrival'].shift(lag).fillna(0) * prob
        
        df['departure'] = df['departure'].round().astype(int)
        df['net_flow'] = df['arrival'] - df['departure']
        
        return df
    
    def _split_by_transport(self, df: pd.DataFrame) -> pd.DataFrame:
        """교통수단별 분리 (항공/여객선)"""
        air_ratio = self.transport_ratio['air']
        ferry_ratio = self.transport_ratio['ferry']
        
        df['air_arrival'] = (df['arrival'] * air_ratio).round().astype(int)
        df['air_departure'] = (df['departure'] * air_ratio).round().astype(int)
        df['ferry_arrival'] = (df['arrival'] * ferry_ratio).round().astype(int)
        df['ferry_departure'] = (df['departure'] * ferry_ratio).round().astype(int)
        
        return df
    
    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """출력 포맷 정리"""
        output_cols = [
            'date', 'base_population',
            'air_arrival', 'air_departure',
            'ferry_arrival', 'ferry_departure',
            'arrival', 'departure', 'net_flow',
            'tourist_stock', 'estimated_population'
        ]
        
        if 'source' in df.columns:
            output_cols.append('source')
        
        output_df = df[output_cols].copy()
        
        output_df = output_df.rename(columns={
            'arrival': 'total_arrival',
            'departure': 'total_departure',
            'source': 'data_source',
        })
        
        return output_df
    
    def get_daily_population(
        self,
        start_date: str,
        end_date: str,
        visitors_csv_path: Optional[str] = None,
        initial_population: Optional[int] = None,  # 미사용
        progress_callback: Optional[Callable] = None,  # 미사용
    ) -> pd.DataFrame:
        """
        기간별 일별 체류인구 계산 (호환성 메서드)
        """
        if visitors_csv_path is None:
            visitors_csv_path = str(
                Path(__file__).parent.parent.parent / 
                "data" / "raw" / "jeju_daily_visitors_v10.csv"
            )
        
        return self.calculate_from_visitors_data(
            visitors_csv_path=visitors_csv_path,
            start_date=start_date,
            end_date=end_date,
        )
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """DataFrame을 CSV로 저장"""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir
        
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"저장 완료: {filepath}")
        
        return str(filepath)
    
    def print_summary(self, df: pd.DataFrame):
        """결과 요약 출력"""
        print("\n" + "=" * 70)
        print("제주도 체류인구 계산 결과 (Convolution v2.0)")
        print("=" * 70)
        
        print(f"\n📊 기간: {df['date'].min()} ~ {df['date'].max()}")
        print(f"   데이터: {len(df):,}일")
        
        print(f"\n📈 체류인구:")
        print(f"   평균: {df['estimated_population'].mean():,.0f}명")
        print(f"   최소: {df['estimated_population'].min():,}명")
        print(f"   최대: {df['estimated_population'].max():,}명")
        
        print(f"\n👥 체류 관광객:")
        print(f"   평균: {df['tourist_stock'].mean():,.0f}명/일")
        print(f"   최소: {df['tourist_stock'].min():,}명")
        print(f"   최대: {df['tourist_stock'].max():,}명")
        
        print(f"\n✈️ 입도객: {df['total_arrival'].sum():,}명 (일평균 {df['total_arrival'].mean():,.0f}명)")
        print(f"🚢 출도객: {df['total_departure'].sum():,}명 (일평균 {df['total_departure'].mean():,.0f}명)")


# =============================================================================
# CLI 인터페이스
# =============================================================================

def main():
    """CLI 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="제주도 일별 체류인구 계산 (Convolution v2.0)",
    )
    
    parser.add_argument('--input', '-i', type=str, 
                        default='data/raw/jeju_daily_visitors_v10.csv',
                        help='입도객 CSV 파일 경로')
    parser.add_argument('--output', '-o', type=str, 
                        default='jeju_daily_population.csv',
                        help='출력 CSV 파일명')
    parser.add_argument('--start', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='종료 날짜 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    crawler = JejuPopulationCrawler()
    
    df = crawler.calculate_from_visitors_data(
        visitors_csv_path=args.input,
        start_date=args.start,
        end_date=args.end,
    )
    
    crawler.print_summary(df)
    filepath = crawler.save_to_csv(df, args.output)
    print(f"\n📁 저장 완료: {filepath}")


if __name__ == "__main__":
    main()
