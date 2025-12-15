"""
제주도 입출도객 데이터 수집 설정
항공(Air Portal) 및 해운(Ferry) 데이터 소스 정의

Why: 전력 수요 예측 모델의 핵심 피처인 '체류인구'를 계산하기 위해
     입도객 - 출도객 데이터를 일별로 수집해야 함
"""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# =============================================================================
# .env 파일 자동 로드
# =============================================================================

try:
    from dotenv import load_dotenv
    # 프로젝트 루트의 .env 파일 로드
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드됨: {env_path}")
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수에서 직접 읽음


# =============================================================================
# 공항 코드 정의
# =============================================================================

AIRPORT_CODES = {
    # 제주
    "제주": "CJU",
    "JEJU": "CJU",
    
    # 주요 출발지 (제주행)
    "김포": "GMP",
    "김해": "PUS",
    "대구": "TAE",
    "청주": "CJJ",
    "광주": "KWJ",
    "여수": "RSU",
    "울산": "USN",
    "무안": "MWX",
    "사천": "HIN",
    "원주": "WJU",
    "양양": "YNY",
    "포항": "KPO",
    "군산": "KUV",
    
    # 국제선
    "인천": "ICN",
}

# 역매핑 (코드 → 공항명)
AIRPORT_NAMES = {v: k for k, v in AIRPORT_CODES.items()}


# =============================================================================
# 여객선 항로 정의
# =============================================================================

FERRY_ROUTES = {
    # 본토 ↔ 제주 주요 항로
    "제주-완도": {"departure": "완도", "arrival": "제주"},
    "완도-제주": {"departure": "제주", "arrival": "완도"},
    "제주-우수영": {"departure": "우수영", "arrival": "제주"},
    "우수영-제주": {"departure": "제주", "arrival": "우수영"},
    "제주-녹동": {"departure": "녹동", "arrival": "제주"},
    "녹동-제주": {"departure": "제주", "arrival": "녹동"},
    
    # 부속섬 항로 (체류인구에는 미포함 - 제주도 내 이동)
    "모슬포-가파도": {"departure": "모슬포", "arrival": "가파도", "internal": True},
    "모슬포-마라도": {"departure": "모슬포", "arrival": "마라도", "internal": True},
    "산이수동-마라도": {"departure": "산이수동", "arrival": "마라도", "internal": True},
    "성산-우도": {"departure": "성산", "arrival": "우도", "internal": True},
}

# 본토 ↔ 제주 항로만 필터링 (체류인구 계산용)
MAINLAND_FERRY_ROUTES = {
    k: v for k, v in FERRY_ROUTES.items() 
    if not v.get("internal", False)
}


# =============================================================================
# API 설정
# =============================================================================

@dataclass
class APIConfig:
    """API 접속 설정"""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    wait_between_requests: float = 1.0


# 공공데이터포털 API 설정
DATA_GO_KR_CONFIG = APIConfig(
    base_url="https://apis.data.go.kr",
    api_key=os.environ.get("DATA_GO_KR_API_KEY"),
    timeout=30,
    max_retries=3,
    wait_between_requests=0.5,
)

# 항공정보포털 설정
AIR_PORTAL_CONFIG = APIConfig(
    base_url="https://www.airportal.go.kr",
    timeout=30,
    max_retries=3,
    wait_between_requests=1.0,
)

# 한국공항공사 설정
KAC_CONFIG = APIConfig(
    base_url="https://www.airport.co.kr",
    timeout=30,
    max_retries=3,
    wait_between_requests=1.0,
)


# =============================================================================
# 데이터 소스 URL (공공데이터포털)
# =============================================================================

DATA_SOURCES = {
    # 한국공항공사_공항별 여객실적
    "airport_passenger": {
        "name": "한국공항공사_공항별 여객실적",
        "url": "https://www.data.go.kr/data/15002610/fileData.do",
        "api_endpoint": "/B551177/PassengerNoticeKR/getfPassengerNoticeIKR",
        "format": "json",
        "description": "공항별 국내선/국제선 여객 출발/도착 수",
    },
    
    # 국토교통부_항공기 출도착현황
    "flight_arrival_departure": {
        "name": "국토교통부_항공기 출도착현황",
        "url": "https://www.data.go.kr/data/15092485/fileData.do",
        "format": "csv",
        "description": "공항별, 일자별, 시간대별 항공기 출도착 현황",
    },
    
    # 제주특별자치도_여객선수송실적현황
    "ferry_passenger": {
        "name": "제주특별자치도_여객선수송실적현황",
        "url": "https://www.data.go.kr/data/15056326/fileData.do",
        "format": "csv",
        "description": "제주↔본토 여객선 항로별 수송실적",
    },
    
    # 제주관광공사_제주방문관광객_통계
    "jeju_visitor_stats": {
        "name": "제주관광공사_제주방문관광객_통계",
        "url": "https://www.data.go.kr/data/15007317/fileData.do",
        "format": "csv",
        "description": "제주 방문 관광객 통계 (내/외국인)",
    },
}


# =============================================================================
# 크롤링 설정
# =============================================================================

CRAWL_CONFIG = {
    # HTTP 요청 설정
    "headers": {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
    },
    
    # 재시도 설정
    "max_retries": 3,
    "retry_delay": 2.0,
    
    # 요청 간격 (서버 부하 방지)
    "request_delay": 1.0,
    
    # 타임아웃
    "timeout": 30,
    
    # 동시 요청 수
    "max_concurrent": 5,
}


# =============================================================================
# 데이터 검증 기준
# =============================================================================

VALIDATION_RULES = {
    # 일별 항공 승객 수 범위 (제주공항 기준)
    "air_passenger_daily": {
        "min": 5000,      # 최소 5,000명/일 (비수기, 악천후)
        "max": 150000,    # 최대 150,000명/일 (성수기)
        "typical_range": (30000, 80000),  # 일반적 범위
    },
    
    # 일별 여객선 승객 수 범위
    "ferry_passenger_daily": {
        "min": 100,       # 최소 100명/일
        "max": 10000,     # 최대 10,000명/일
        "typical_range": (500, 3000),
    },
    
    # 제주도 총 체류인구 범위
    "total_population": {
        "min": 650000,    # 주민등록인구 기준 최소
        "max": 1000000,   # 성수기 최대
    },
}


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_api_key() -> Optional[str]:
    """공공데이터포털 API 키 반환"""
    return os.environ.get("DATA_GO_KR_API_KEY")


def get_airport_code(airport_name: str) -> str:
    """공항명으로 코드 조회"""
    airport_upper = airport_name.upper()
    
    # 정확한 매칭
    if airport_name in AIRPORT_CODES:
        return AIRPORT_CODES[airport_name]
    if airport_upper in AIRPORT_CODES:
        return AIRPORT_CODES[airport_upper]
    
    # 부분 매칭
    for name, code in AIRPORT_CODES.items():
        if airport_name in name or name in airport_name:
            return code
    
    raise ValueError(f"알 수 없는 공항: {airport_name}")


def get_mainland_routes() -> list:
    """본토 ↔ 제주 여객선 항로 목록 반환"""
    return list(MAINLAND_FERRY_ROUTES.keys())


def validate_passenger_count(count: int, transport_type: str = "air") -> bool:
    """승객 수 유효성 검증"""
    if transport_type == "air":
        rules = VALIDATION_RULES["air_passenger_daily"]
    elif transport_type == "ferry":
        rules = VALIDATION_RULES["ferry_passenger_daily"]
    else:
        return True
    
    return rules["min"] <= count <= rules["max"]


def is_typical_count(count: int, transport_type: str = "air") -> bool:
    """일반적인 범위 내 승객 수인지 확인"""
    if transport_type == "air":
        rules = VALIDATION_RULES["air_passenger_daily"]
    elif transport_type == "ferry":
        rules = VALIDATION_RULES["ferry_passenger_daily"]
    else:
        return True
    
    min_typical, max_typical = rules["typical_range"]
    return min_typical <= count <= max_typical


def list_airports():
    """사용 가능한 공항 목록 출력"""
    print("=== 공항 코드 목록 ===")
    for name, code in sorted(AIRPORT_CODES.items(), key=lambda x: x[1]):
        print(f"  {name}: {code}")


def list_ferry_routes():
    """여객선 항로 목록 출력"""
    print("=== 여객선 항로 목록 ===")
    print("\n[본토 ↔ 제주]")
    for route, info in MAINLAND_FERRY_ROUTES.items():
        print(f"  {route}: {info['departure']} → {info['arrival']}")
    
    print("\n[제주 내부 (부속섬)]")
    internal_routes = {k: v for k, v in FERRY_ROUTES.items() if v.get("internal")}
    for route, info in internal_routes.items():
        print(f"  {route}: {info['departure']} → {info['arrival']}")


def check_api_key():
    """API 키 상태 확인"""
    api_key = get_api_key()
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:]
        print(f"✅ API 키 설정됨: {masked}")
        return True
    else:
        print("⚠️  API 키 미설정 (웹 크롤링 모드로 동작)")
        return False


if __name__ == "__main__":
    print("제주도 입출도객 데이터 수집 설정\n")
    check_api_key()
    print()
    list_airports()
    print()
    list_ferry_routes()
