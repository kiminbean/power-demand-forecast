# KMA Weather Data Crawlers
from .kma_crawler import KMACrawler
from .kma_api import KMAAPI
from .config import STATION_CODES, ELEMENT_CODES

# Jeju Transport & Population Config
from .jeju_transport_config import (
    AIRPORT_CODES,
    FERRY_ROUTES,
    MAINLAND_FERRY_ROUTES,
    get_airport_code,
    get_mainland_routes,
    validate_passenger_count,
)

# Jeju Population Crawler (still functional)
from .jeju_population_crawler import JejuPopulationCrawler, DailyPopulation

# NEW: Ferry Estimator (replaces broken JejuFerryCrawler)
from .jeju_ferry_estimator import JejuFerryEstimator, FerryEstimate

# NEW: Visitor Crawler (제주관광협회 공식 데이터)
from .jeju_visitor_crawler import JejuVisitorCrawler, DailyVisitors

# =============================================================================
# WORKING CRAWLERS
# =============================================================================
# - JejuVisitorCrawler: 제주관광협회 입도객 데이터 (visitjeju.or.kr)
#   * 2013년~현재 일별 데이터
#   * HTML 파싱 + EasyOCR (GPU 가속)
#   * jeju_daily_visitors_v10.csv 생성
#
# - JejuFerryEstimator: 해운 승객 추정기
#   * 항공 데이터 × 5.5% 비율
#   * 계절별/기상 조건 반영
#
# - JejuPopulationCrawler: 체류인구 계산기
#   * 입도객-출도객 기반 추정
# =============================================================================

# =============================================================================
# DEPRECATED MODULES (DO NOT USE)
# =============================================================================
# - JejuAirCrawler: 삭제됨 - 3개 데이터 소스 모두 동작 불가
#   대안: JejuVisitorCrawler 사용 (제주관광협회 공식)
#
# - JejuFerryCrawler: deprecated/ 폴더 - 4개 함수 미구현
#   대안: JejuFerryEstimator 사용
# =============================================================================

__all__ = [
    # KMA Crawlers
    'KMACrawler', 
    'KMAAPI', 
    'STATION_CODES', 
    'ELEMENT_CODES',
    
    # Jeju Transport Config
    'AIRPORT_CODES',
    'FERRY_ROUTES',
    'MAINLAND_FERRY_ROUTES',
    'get_airport_code',
    'get_mainland_routes',
    'validate_passenger_count',
    
    # Jeju Crawlers & Estimators (WORKING)
    'JejuVisitorCrawler',   # 제주관광협회 입도객 (PRIMARY)
    'JejuFerryEstimator',   # 해운 승객 추정
    'JejuPopulationCrawler', # 체류인구 계산
    
    # Data Classes
    'DailyVisitors',  # 입도객 데이터
    'FerryEstimate',  # 해운 추정 데이터
    'DailyPopulation', # 체류인구 데이터
]
