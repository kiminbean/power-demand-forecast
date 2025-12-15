# 제주도 입출도객 & 체류인구 데이터

제주도 전력 수요 예측 모델을 위한 **입도객/출도객 데이터** 관리 도구입니다.

## 크롤러 상태 (2024-12-15 업데이트)

| 크롤러 | 상태 | 설명 |
|--------|:----:|------|
| **JejuVisitorCrawler** | ✅ 동작 | 제주관광협회 공식 데이터 (PRIMARY) |
| JejuFerryEstimator | ✅ 동작 | 해운 승객 추정기 |
| JejuPopulationCrawler | ⚠️ 확인 필요 | 체류인구 계산 |

## 사용법

### 1. 제주 입도객 크롤러 (JejuVisitorCrawler)

**데이터 소스**: 제주관광협회 (visitjeju.or.kr) 공식 통계

#### Python API

```python
from tools.crawlers import JejuVisitorCrawler

crawler = JejuVisitorCrawler()

# 특정 날짜 크롤링
data = crawler.crawl_date("2025-12-14")
print(f"입도객: {data.daily_visitors:,}명")

# CSV 업데이트
crawler.update_csv(date="2025-12-14")

# 최신 업데이트
crawler.update_to_latest()
```

#### CLI

```bash
python tools/crawlers/jeju_visitor_crawler.py --date 2025-12-14
python tools/crawlers/jeju_visitor_crawler.py --update
```

### 2. 여객선 추정기 (JejuFerryEstimator)

```python
from tools.crawlers import JejuFerryEstimator

estimator = JejuFerryEstimator()
ferry = estimator.estimate_from_air({'arrival': 45000}, "2024-08-15")
```

## 데이터 파일

| 파일 | 기간 | 용도 |
|------|------|------|
| `jeju_daily_visitors_v10.csv` | 2013-2025 | 일별 입도객 |
| `jeju_daily_population_2013_2024_v2.csv` | 2013-2024 | 체류인구 |
