#!/usr/bin/env python3
"""
제주도 일별 체류인구 데이터 수집 스크립트 (간소화 버전)
기존 입도객 데이터를 활용하여 체류인구 계산

실행 방법:
    cd /Users/ibkim/Ormi_1/power-demand-forecast
    python scripts/collect_jeju_population.py

입력 파일:
    data/raw/jeju_daily_visitors_v10.csv (기존 크롤러로 수집된 입도객 데이터)

출력 파일:
    data/processed/jeju_daily_population_2013_2024.csv
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("제주도 일별 체류인구 데이터 계산")
    print("=" * 70)
    
    # 크롤러 임포트
    try:
        from tools.crawlers import JejuPopulationCrawler
        print("✅ 크롤러 모듈 로드 완료")
    except ImportError as e:
        print(f"❌ 크롤러 임포트 실패: {e}")
        sys.exit(1)
    
    # 입력/출력 경로
    input_path = project_root / "data" / "raw" / "jeju_daily_visitors_v10.csv"
    output_dir = project_root / "data" / "processed"
    output_file = "jeju_daily_population_2013_2024.csv"
    
    if not input_path.exists():
        print(f"❌ 입도객 데이터 파일 없음: {input_path}")
        print("   먼저 jeju_tourism_crawler로 데이터를 수집해주세요.")
        sys.exit(1)
    
    print(f"\n📁 입력: {input_path}")
    print(f"📁 출력: {output_dir / output_file}")
    
    # 크롤러 초기화
    crawler = JejuPopulationCrawler()
    
    # 체류인구 계산 (2013-2024)
    print("\n🔄 체류인구 계산 중...")
    df = crawler.calculate_from_visitors_data(
        visitors_csv_path=str(input_path),
        start_date="2013-01-01",
        end_date="2024-12-31",
    )
    
    # 요약 출력
    crawler.print_summary(df)
    
    # CSV 저장
    filepath = crawler.save_to_csv(df, output_file, str(output_dir))
    
    print("\n" + "=" * 70)
    print("✅ 완료!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
