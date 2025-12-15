#!/usr/bin/env python3
"""
기상 데이터 다운로드 CLI 스크립트

사용법:
    # 명령줄 인자 사용
    python scripts/download_weather.py --station 제주 --start 2013-01-01 --end 2024-12-31 --elements 기온

    # 자연어 명령
    python scripts/download_weather.py "제주도 2013~2024년 기온 데이터를 다운로드해줘"

    # 일괄 다운로드 (연도별)
    python scripts/download_weather.py --station 제주 --start-year 2013 --end-year 2024 --elements 기온 --batch

지원하는 지점:
    서울, 인천, 대전, 대구, 부산, 광주, 울산, 제주, 서귀포, 고산 등

지원하는 요소:
    기온, 습도, 강수량, 풍속, 일사량, 일조시간, 기압, 지면온도 등
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.crawlers import KMACrawler, STATION_CODES, ELEMENT_CODES
from tools.crawlers.config import list_stations, list_elements


def parse_natural_language(query: str) -> dict:
    """
    자연어 명령을 파싱하여 다운로드 파라미터 추출

    예시:
        "제주도 2013~2024년 기온 데이터를 csv파일로 다운로드해줘"
        "서울 2020년 1월부터 2023년 12월까지 습도와 강수량 데이터"
        "부산 지역의 최근 5년간 일사량 자료"
    """
    result = {
        "station": None,
        "start_date": None,
        "end_date": None,
        "elements": [],
        "data_type": "일",
    }

    query_lower = query.lower()

    # 1. 지점 파싱
    for station_name in STATION_CODES.keys():
        # 지역명 변형 처리 (제주도 → 제주, 서울시 → 서울 등)
        variations = [
            station_name,
            station_name + "도",
            station_name + "시",
            station_name + "지역",
            station_name + " 지역",
        ]
        for var in variations:
            if var in query:
                result["station"] = station_name
                break
        if result["station"]:
            break

    # 2. 연도 범위 파싱
    # 패턴: 2013~2024년, 2013-2024년, 2013년~2024년, 2013년부터 2024년
    year_range_patterns = [
        r"(\d{4})\s*[~\-부터]\s*(\d{4})",  # 2013~2024, 2013-2024
        r"(\d{4})년\s*[~\-부터]\s*(\d{4})년",  # 2013년~2024년
        r"(\d{4})년부터\s*(\d{4})년",  # 2013년부터 2024년
    ]

    for pattern in year_range_patterns:
        match = re.search(pattern, query)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            result["start_date"] = f"{start_year}-01-01"
            result["end_date"] = f"{end_year}-12-31"
            break

    # 단일 연도 파싱
    if not result["start_date"]:
        single_year = re.search(r"(\d{4})년", query)
        if single_year:
            year = int(single_year.group(1))
            result["start_date"] = f"{year}-01-01"
            result["end_date"] = f"{year}-12-31"

    # 상대적 기간 파싱 (최근 N년)
    if not result["start_date"]:
        recent_match = re.search(r"최근\s*(\d+)\s*년", query)
        if recent_match:
            years = int(recent_match.group(1))
            current_year = datetime.now().year
            result["start_date"] = f"{current_year - years}-01-01"
            result["end_date"] = f"{current_year}-12-31"

    # 3. 요소 파싱
    element_keywords = {
        "기온": ["기온", "온도", "temperature", "temp"],
        "습도": ["습도", "humidity"],
        "강수량": ["강수", "비", "rain", "precipitation"],
        "풍속": ["풍속", "바람", "wind"],
        "일사량": ["일사", "일사량", "sunshine", "solar"],
        "일조시간": ["일조", "일조시간"],
        "기압": ["기압", "pressure"],
        "지면온도": ["지면", "지면온도", "지중"],
        "이슬점온도": ["이슬점", "dewpoint"],
        "구름": ["구름", "운량", "cloud"],
        "적설": ["눈", "적설", "snow"],
    }

    for elem_name, keywords in element_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if elem_name not in result["elements"]:
                    result["elements"].append(elem_name)
                break

    # 4. 자료형태 파싱
    if "시간" in query or "hourly" in query_lower:
        result["data_type"] = "시간"
    elif "분" in query:
        result["data_type"] = "분"

    return result


def validate_params(params: dict) -> bool:
    """파라미터 유효성 검사"""
    if not params.get("station"):
        print("오류: 지점을 지정해주세요.")
        print("지원하는 지점 목록:")
        for name in sorted(STATION_CODES.keys()):
            print(f"  - {name}")
        return False

    if not params.get("start_date") or not params.get("end_date"):
        print("오류: 기간을 지정해주세요.")
        print("예: --start 2013-01-01 --end 2024-12-31")
        print("또는: '2013~2024년'")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="기상청 기상자료개방포털에서 기상 데이터를 다운로드합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 명령줄 인자
    parser.add_argument(
        "query",
        nargs="?",
        help="자연어 명령 (예: '제주도 2013~2024년 기온 데이터를 다운로드해줘')",
    )
    parser.add_argument(
        "--station", "-s",
        help="지점명 또는 지점코드 (예: 제주, 184)",
    )
    parser.add_argument(
        "--start", "-S",
        help="시작일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", "-E",
        help="종료일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="시작 연도 (batch 모드용)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="종료 연도 (batch 모드용)",
    )
    parser.add_argument(
        "--elements", "-e",
        nargs="+",
        help="요소 목록 (예: 기온 습도 강수량)",
    )
    parser.add_argument(
        "--data-type", "-t",
        choices=["일", "시간", "분"],
        default="일",
        help="자료형태 (기본: 일)",
    )
    parser.add_argument(
        "--output", "-o",
        help="출력 파일명",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default="data/raw",
        help="출력 디렉토리 (기본: data/raw)",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="연도별 일괄 다운로드 모드",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="헤드리스 모드 (기본: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="브라우저 창 표시 (디버깅용)",
    )
    parser.add_argument(
        "--list-stations",
        action="store_true",
        help="지점 목록 출력",
    )
    parser.add_argument(
        "--list-elements",
        action="store_true",
        help="요소 목록 출력",
    )

    args = parser.parse_args()

    # 목록 출력
    if args.list_stations:
        list_stations()
        return 0

    if args.list_elements:
        list_elements()
        return 0

    # 파라미터 결정
    params = {}

    if args.query:
        # 자연어 파싱
        print(f"자연어 명령 파싱: {args.query}")
        params = parse_natural_language(args.query)
        print(f"파싱 결과: {params}")

    # 명령줄 인자로 덮어쓰기
    if args.station:
        params["station"] = args.station
    if args.start:
        params["start_date"] = args.start
    if args.end:
        params["end_date"] = args.end
    if args.elements:
        params["elements"] = args.elements
    if args.data_type:
        params["data_type"] = args.data_type

    # Batch 모드
    if args.batch and args.start_year and args.end_year:
        params["start_year"] = args.start_year
        params["end_year"] = args.end_year

    # 유효성 검사
    if not args.batch:
        if not validate_params(params):
            return 1

    # 크롤러 실행
    headless = not args.no_headless
    crawler = KMACrawler(headless=headless, download_dir=args.output_dir)

    try:
        if args.batch:
            # 일괄 다운로드
            if not params.get("station"):
                print("오류: --station 옵션이 필요합니다.")
                return 1

            start_year = params.get("start_year") or args.start_year
            end_year = params.get("end_year") or args.end_year

            if not start_year or not end_year:
                print("오류: --start-year와 --end-year 옵션이 필요합니다.")
                return 1

            downloaded = crawler.download_batch(
                station=params["station"],
                start_year=start_year,
                end_year=end_year,
                elements=params.get("elements"),
                data_type=params.get("data_type", "일"),
            )

            print(f"\n다운로드 완료: {len(downloaded)}개 파일")
            for f in downloaded:
                print(f"  - {f}")

        else:
            # 단일 다운로드
            result = crawler.download(
                station=params["station"],
                start_date=params["start_date"],
                end_date=params["end_date"],
                elements=params.get("elements"),
                data_type=params.get("data_type", "일"),
                output_filename=args.output,
            )

            if result:
                print(f"\n다운로드 완료: {result}")
                return 0
            else:
                print("\n다운로드 실패")
                return 1

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
