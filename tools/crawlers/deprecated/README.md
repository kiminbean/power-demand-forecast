# Deprecated Crawlers

이 폴더에는 더 이상 사용되지 않는 크롤러 코드가 보관됩니다.

---

## 삭제된 파일

### ~~jeju_air_crawler.py~~ (완전 삭제됨)

**삭제 일자**: 2024-12-15

**삭제 사유** (Claude + Gemini 크로스체크):
- 3개 데이터 소스 모두 동작 불가
- "환각 코드" - 실제 API에 없는 `sumPax`, `line`, `io` 필드 파싱 시도
- 재사용 가치 없음 (완전히 새로 작성 필요)

**대안**: `jeju_daily_visitors_v10.csv` (2013-2025, 4,378일) 사용

---

## 보관 중인 파일

### jeju_ferry_crawler_v1_BROKEN.py

**폐기 일자**: 2024-12-15

**폐기 사유**:
1. 4개 수집 함수가 `return None`으로 미구현
2. 기본 추정값 1,500명/일 = 실제(~2,200명)의 68% 수준으로 부정확
3. 공공데이터포털 15056326 데이터가 "연간/항로별"이라 일별 추정 불가

**대체 모듈**: `jeju_ferry_estimator.py` (항공 × 5.5% 비율 추정)

---

## 사용 금지

```python
# ❌ 이 폴더의 코드는 import하지 마세요

# ✅ 올바른 사용
# 항공 데이터
import pandas as pd
air_df = pd.read_csv("data/processed/jeju_daily_visitors_v10.csv")

# 해운 데이터
from tools.crawlers import JejuFerryEstimator
estimator = JejuFerryEstimator()
```
