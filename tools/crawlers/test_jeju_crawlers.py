#!/usr/bin/env python3
"""
제주도 입출도객 크롤러 테스트 스크립트
Usage: python test_jeju_crawlers.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """모듈 임포트 테스트"""
    print("=== 1. 모듈 임포트 테스트 ===")
    
    try:
        from tools.crawlers import (
            JejuAirCrawler, 
            JejuFerryCrawler, 
            JejuPopulationCrawler,
            AIRPORT_CODES,
            FERRY_ROUTES,
            MAINLAND_FERRY_ROUTES,
        )
        print("✅ 모든 모듈 임포트 성공")
        return True
    except ImportError as e:
        print(f"❌ 임포트 실패: {e}")
        return False


def test_config():
    """설정 테스트"""
    print("\n=== 2. 설정 테스트 ===")
    
    from tools.crawlers import AIRPORT_CODES, FERRY_ROUTES, MAINLAND_FERRY_ROUTES
    
    print(f"공항 코드 수: {len(AIRPORT_CODES)}")
    print(f"  - 제주: {AIRPORT_CODES.get('제주', 'N/A')}")
    print(f"  - 김포: {AIRPORT_CODES.get('김포', 'N/A')}")
    
    print(f"여객선 항로 수: {len(FERRY_ROUTES)}")
    print(f"본토↔제주 항로 수: {len(MAINLAND_FERRY_ROUTES)}")
    
    return True


def test_air_crawler():
    """항공 크롤러 테스트"""
    print("\n=== 3. 항공 크롤러 테스트 ===")
    
    from tools.crawlers import JejuAirCrawler
    
    try:
        crawler = JejuAirCrawler(use_cache=False)
        print("✅ JejuAirCrawler 초기화 성공")
        
        # 기본 추정값 테스트 (실제 수집 없이)
        print("  - 캐시 디렉토리:", crawler.cache_dir)
        print("  - 제주공항 코드:", crawler.jeju_code)
        
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_ferry_crawler():
    """여객선 크롤러 테스트"""
    print("\n=== 4. 여객선 크롤러 테스트 ===")
    
    from tools.crawlers import JejuFerryCrawler
    
    try:
        crawler = JejuFerryCrawler(use_cache=False)
        print("✅ JejuFerryCrawler 초기화 성공")
        
        # 기본 추정값 테스트
        data = crawler._get_default_estimate("2024-08-15")
        print(f"  - 8월 기본 추정 도착: {data.arrival_total:,}명")
        print(f"  - 8월 기본 추정 출발: {data.departure_total:,}명")
        
        data = crawler._get_default_estimate("2024-01-15")
        print(f"  - 1월 기본 추정 도착: {data.arrival_total:,}명")
        
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_population_crawler():
    """체류인구 크롤러 테스트"""
    print("\n=== 5. 체류인구 크롤러 테스트 ===")
    
    from tools.crawlers import JejuPopulationCrawler
    from tools.crawlers.jeju_population_crawler import JEJU_BASE_POPULATION
    
    try:
        crawler = JejuPopulationCrawler(use_cache=False, ferry_weight=1.0)
        print("✅ JejuPopulationCrawler 초기화 성공")
        
        print("  - 연도별 기준 인구:")
        for year in [2020, 2022, 2024]:
            pop = JEJU_BASE_POPULATION.get(year, "N/A")
            print(f"    {year}년: {pop:,}명" if isinstance(pop, int) else f"    {year}년: {pop}")
        
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def test_data_collection_single():
    """단일 날짜 데이터 수집 테스트 (실제 네트워크 요청)"""
    print("\n=== 6. 데이터 수집 테스트 (단일 날짜) ===")
    print("⚠️  이 테스트는 네트워크 요청을 수행합니다.")
    
    user_input = input("실행하시겠습니까? (y/N): ").strip().lower()
    if user_input != 'y':
        print("건너뜀")
        return True
    
    from tools.crawlers import JejuAirCrawler
    
    try:
        crawler = JejuAirCrawler(use_cache=True)
        
        # 어제 날짜로 테스트
        from datetime import datetime, timedelta
        test_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        print(f"테스트 날짜: {test_date}")
        data = crawler.get_daily_passengers(test_date)
        
        if data:
            print(f"✅ 데이터 수집 성공!")
            print(f"  - 도착: {data.arrival_total:,}명")
            print(f"  - 출발: {data.departure_total:,}명")
            print(f"  - 순유입: {data.net_flow:+,}명")
            print(f"  - 출처: {data.source}")
        else:
            print("⚠️  데이터를 수집하지 못했습니다 (API 키 없음 또는 데이터 미공개)")
        
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("제주도 입출도객 크롤러 테스트")
    print("=" * 60)
    
    results = []
    
    results.append(("임포트", test_imports()))
    results.append(("설정", test_config()))
    results.append(("항공 크롤러", test_air_crawler()))
    results.append(("여객선 크롤러", test_ferry_crawler()))
    results.append(("체류인구 크롤러", test_population_crawler()))
    results.append(("데이터 수집", test_data_collection_single()))
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
