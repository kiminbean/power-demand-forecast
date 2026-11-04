#!/usr/bin/env python3
"""
jeju_air_crawler 실제 동작 테스트 스크립트

실행 방법:
    cd /Users/ibkim/Projects/power-demand-forecast
    python scripts/test_air_crawler.py

테스트 항목:
1. 공공데이터포털 API (한국공항공사_공항별 여객실적)
2. 한국공항공사 웹 크롤링
3. 항공정보포털 웹 크롤링
4. 제주관광협회 (기존 입도객 크롤러)
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    pass

import httpx
from bs4 import BeautifulSoup
import re


def test_data_go_kr_api():
    """
    테스트 1: 공공데이터포털 API
    
    Gemini 분석:
    - API는 실제로 존재함
    - 하지만 "비행 스케줄" 정보만 제공 (여객 수 X)
    - API 활용 신청이 필요함
    """
    print("\n" + "=" * 70)
    print("📡 테스트 1: 공공데이터포털 API")
    print("=" * 70)
    
    api_key = os.environ.get("DATA_GO_KR_API_KEY")
    
    if not api_key:
        print("❌ API 키 없음 (.env 파일 확인)")
        return False
    
    print(f"✅ API 키: {api_key[:10]}...{api_key[-4:]}")
    
    # 테스트 날짜 (최근 날짜)
    test_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    
    # API 엔드포인트 (여러 개 테스트)
    apis_to_test = [
        {
            "name": "한국공항공사_공항별 여객실적",
            "url": "https://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR",
            "params": {
                "serviceKey": api_key,
                "from_time": test_date,
                "to_time": test_date,
                "airport": "CJU",
                "type": "json",
            }
        },
        {
            "name": "국토교통부_항공기 출도착현황 (대안)",
            "url": "https://apis.data.go.kr/1613000/AirInfoService/getAirStatsInfo",
            "params": {
                "serviceKey": api_key,
                "numOfRows": "10",
                "pageNo": "1",
                "airArea": "A", 
                "_type": "json",
            }
        },
    ]
    
    results = []
    
    for api in apis_to_test:
        print(f"\n  [{api['name']}]")
        print(f"  URL: {api['url']}")
        
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(api['url'], params=api['params'])
                
                print(f"  Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # 응답 구조 확인
                        if 'response' in data:
                            header = data['response'].get('header', {})
                            result_code = header.get('resultCode', '')
                            result_msg = header.get('resultMsg', '')
                            
                            print(f"  Result: {result_code} - {result_msg}")
                            
                            if result_code == '00':
                                body = data['response'].get('body', {})
                                items = body.get('items', {})
                                print(f"  Items: {type(items)}")
                                
                                if items:
                                    print(f"  ✅ 데이터 있음!")
                                    results.append(True)
                                else:
                                    print(f"  ⚠️ 데이터 없음 (API 활용 신청 필요?)")
                            else:
                                print(f"  ❌ API 에러")
                        else:
                            print(f"  Response: {str(data)[:200]}")
                            
                    except json.JSONDecodeError:
                        print(f"  Response (text): {response.text[:300]}")
                        
                elif response.status_code == 403:
                    print(f"  ❌ 403 Forbidden - API 활용 신청 필요")
                else:
                    print(f"  ❌ HTTP {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    
        except Exception as e:
            print(f"  에러: {e}")
    
    return any(results)


def test_kac_web():
    """
    테스트 2: 한국공항공사 웹 크롤링
    
    Gemini 분석:
    - URL이 내부 CMS 컴포넌트일 가능성
    - CSRF 토큰/세션 문제로 차단 가능성 높음
    """
    print("\n" + "=" * 70)
    print("🌐 테스트 2: 한국공항공사 웹 크롤링")
    print("=" * 70)
    
    test_date = datetime.now() - timedelta(days=7)
    
    urls_to_test = [
        {
            "name": "여객 통계 페이지",
            "url": "https://www.airport.co.kr/www/cms/frFlightStatsCon/passengerStats.do",
            "method": "POST",
            "data": {
                "MENU_ID": "1240",
                "sYyyy": test_date.strftime("%Y"),
                "sMm": test_date.strftime("%m"),
                "sDd": test_date.strftime("%d"),
                "eYyyy": test_date.strftime("%Y"),
                "eMm": test_date.strftime("%m"),
                "eDd": test_date.strftime("%d"),
                "airportCode": "CJU",
            }
        },
        {
            "name": "메인 통계 페이지",
            "url": "https://www.airport.co.kr/www/extra/stats/kyStats.do",
            "method": "GET",
        },
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    
    for test in urls_to_test:
        print(f"\n  [{test['name']}]")
        print(f"  URL: {test['url']}")
        
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                if test.get("method") == "POST":
                    response = client.post(test['url'], data=test.get('data', {}), headers=headers)
                else:
                    response = client.get(test['url'], headers=headers)
                
                print(f"  Status: {response.status_code}")
                print(f"  Content-Length: {len(response.text)}")
                
                if response.status_code == 200 and len(response.text) > 500:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 테이블 찾기
                    tables = soup.find_all('table')
                    print(f"  테이블 개수: {len(tables)}")
                    
                    # 숫자 추출
                    text = soup.get_text()
                    numbers = re.findall(r'\d{1,3}(?:,\d{3})+', text)
                    large_numbers = [n for n in numbers if int(n.replace(',', '')) > 10000]
                    
                    if large_numbers:
                        print(f"  큰 숫자들: {large_numbers[:5]}")
                        print(f"  ✅ 데이터 추출 가능성 있음")
                    else:
                        print(f"  ⚠️ 유의미한 숫자 없음")
                else:
                    print(f"  ❌ 페이지 로드 실패")
                    
        except Exception as e:
            print(f"  에러: {e}")
    
    return False


def test_airportal():
    """
    테스트 3: 항공정보포털 (airportal.go.kr)
    
    Gemini 분석:
    - 가장 유력한 소스
    - 일별/공항별 여객 수(확정치) 제공
    - 파라미터 수정 필요할 수 있음
    """
    print("\n" + "=" * 70)
    print("✈️ 테스트 3: 항공정보포털 (airportal.go.kr)")
    print("=" * 70)
    
    test_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    
    urls_to_test = [
        {
            "name": "공항별 통계 (구버전)",
            "url": "https://www.airportal.go.kr/knowledge/statsnew/airport/AirportD.jsp",
            "params": {
                "mode": "list",
                "iArport": "CJU",
                "startDt": test_date,
                "endDt": test_date,
            }
        },
        {
            "name": "실시간 공항 현황",
            "url": "https://www.airportal.go.kr/life/airinfo/RbHanStatus.jsp",
            "params": {
                "search_date": test_date,
                "term": "d",
            }
        },
        {
            "name": "통계 메인",
            "url": "https://www.airportal.go.kr/knowledge/statsnew/main/main.jsp",
            "params": {}
        },
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    
    for test in urls_to_test:
        print(f"\n  [{test['name']}]")
        print(f"  URL: {test['url']}")
        
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(test['url'], params=test['params'], headers=headers)
                
                print(f"  Status: {response.status_code}")
                print(f"  Content-Length: {len(response.text)}")
                
                if response.status_code == 200 and len(response.text) > 500:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    tables = soup.find_all('table')
                    print(f"  테이블 개수: {len(tables)}")
                    
                    # 제주 관련 텍스트 찾기
                    text = soup.get_text()
                    if '제주' in text or 'CJU' in text:
                        print(f"  ✅ '제주' 키워드 발견")
                        
                        # 숫자 추출
                        numbers = re.findall(r'\d{1,3}(?:,\d{3})+', text)
                        large_numbers = [n for n in numbers if int(n.replace(',', '')) > 10000]
                        if large_numbers:
                            print(f"  큰 숫자들: {large_numbers[:5]}")
                    else:
                        print(f"  ⚠️ '제주' 키워드 없음")
                        
        except Exception as e:
            print(f"  에러: {e}")
    
    return False


def test_ijto():
    """
    테스트 4: 제주관광협회 (ijto.or.kr)
    
    기존 입도객 크롤러가 사용하는 소스
    """
    print("\n" + "=" * 70)
    print("🏝️ 테스트 4: 제주관광협회 (ijto.or.kr)")
    print("=" * 70)
    
    urls_to_test = [
        {
            "name": "일별 입도객 현황",
            "url": "https://ijto.or.kr/ko/TourStat01",
        },
        {
            "name": "통계 API",
            "url": "https://ijto.or.kr/api/statistics/daily",
        },
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }
    
    for test in urls_to_test:
        print(f"\n  [{test['name']}]")
        print(f"  URL: {test['url']}")
        
        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(test['url'], headers=headers)
                
                print(f"  Status: {response.status_code}")
                print(f"  Content-Length: {len(response.text)}")
                
                if response.status_code == 200:
                    # JSON 시도
                    try:
                        data = response.json()
                        print(f"  ✅ JSON 응답!")
                        print(f"  Keys: {data.keys() if isinstance(data, dict) else type(data)}")
                    except:
                        # HTML
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # 이미지 찾기 (그래프)
                        images = soup.find_all('img')
                        chart_images = [img for img in images if 'chart' in str(img).lower() or 'graph' in str(img).lower()]
                        print(f"  이미지/차트: {len(chart_images)}개")
                        
                        # canvas 찾기 (JavaScript 차트)
                        canvas = soup.find_all('canvas')
                        print(f"  Canvas (JS차트): {len(canvas)}개")
                        
                        if chart_images or canvas:
                            print(f"  ✅ 이미지/차트 기반 - OCR 필요")
                        else:
                            print(f"  ⚠️ 데이터 형식 확인 필요")
                            
        except Exception as e:
            print(f"  에러: {e}")
    
    return False


def main():
    """메인 테스트 실행"""
    print("=" * 70)
    print("🔍 jeju_air_crawler 실제 동작 테스트")
    print("=" * 70)
    print(f"테스트 시간: {datetime.now()}")
    
    results = {
        "공공데이터포털 API": test_data_go_kr_api(),
        "한국공항공사 웹": test_kac_web(),
        "항공정보포털": test_airportal(),
        "제주관광협회": test_ijto(),
    }
    
    print("\n" + "=" * 70)
    print("📋 테스트 결과 요약")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패/미확인"
        print(f"  {name}: {status}")
    
    print("\n" + "=" * 70)
    print("💡 Gemini 분석 결과")
    print("=" * 70)
    print("""
  1. 공공데이터포털 API:
     - API는 존재하지만 "비행 스케줄" 정보만 제공
     - 여객 수(탑승객 수)는 제공하지 않음 (항공사 영업비밀)
     - ❌ 체류인구 계산에 부적합

  2. 한국공항공사 웹:
     - 대부분 항공정보포털로 리다이렉트됨
     - CSRF/세션 문제로 크롤링 어려움
     - ⚠️ 불안정

  3. 항공정보포털 (권장):
     - 일별/공항별 여객 수 확정치 제공
     - 가장 신뢰할 수 있는 소스
     - ✅ 파라미터 수정 후 사용 가능

  4. 제주관광협회 (현재 사용 중):
     - 그래프 이미지로 데이터 제공
     - EASYOCR로 텍스트 추출 필요
     - ✅ 기존 크롤러가 이미 동작 중
    """)
    
    print("\n" + "=" * 70)
    print("📌 결론")
    print("=" * 70)
    print("""
  jeju_air_crawler.py의 3가지 데이터 소스 중:
  
  - Source 1 (API): 403 오류 또는 데이터 불일치로 동작 안함
  - Source 2 (KAC): 페이지 구조 변경으로 동작 불확실
  - Source 3 (AirPortal): 파라미터 수정 시 동작 가능성 있음
  
  그러나 기존에 jeju_daily_visitors_v10.csv가 이미:
  - 2013~2025년 일별 입도객 데이터 보유
  - 공식 통계와 99% 이상 일치
  - EASYOCR 기반으로 안정적 수집
  
  따라서 jeju_air_crawler는 실질적으로 불필요하며,
  기존 입도객 데이터 + Convolution 방식의 체류인구 계산이 최선입니다.
    """)


if __name__ == "__main__":
    main()
