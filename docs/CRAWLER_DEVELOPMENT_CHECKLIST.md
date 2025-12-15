# 크롤러 개발 프로세스 체크리스트
# Crawler Development Process Checklist
# =====================================

## 1단계: 데이터 소스 검증 (BEFORE CODING)

### 1.1 데이터 존재 확인
- [ ] API/웹페이지가 실제로 존재하는가?
- [ ] 로그인 없이 접근 가능한가?
- [ ] 데이터가 원하는 형식(JSON/CSV/HTML)인가?

### 1.2 데이터 품질 확인  
- [ ] 데이터 갱신 주기는? (실시간/일별/월별/연간)
- [ ] 데이터 행 수는 충분한가?
- [ ] 필요한 컬럼(여객 수 등)이 존재하는가?

### 1.3 수동 테스트
- [ ] 브라우저에서 직접 접근 테스트
- [ ] curl/httpie로 API 호출 테스트
- [ ] 실제 응답 데이터 샘플 확보

---

## 2단계: 설계 검증

### 2.1 데이터 매핑
- [ ] 소스 데이터 → 목표 데이터 변환 로직 정의
- [ ] 연간 데이터를 일별로 분배하는 것이 타당한가?
- [ ] 추정값 사용 시 근거 통계 확보

### 2.2 대안 전략
- [ ] 1순위 소스 실패 시 2순위 소스 정의
- [ ] 모든 소스 실패 시 기본값 정의 (통계 기반)
- [ ] 기본값의 정확도 검증

---

## 3단계: 구현 체크

### 3.1 함수 완성도
- [ ] `return None` 또는 `pass`로 끝나는 함수 없음
- [ ] 모든 분기에 대한 처리 로직 존재
- [ ] 에러 핸들링 및 로깅

### 3.2 테스트 코드
- [ ] 각 데이터 소스별 단위 테스트
- [ ] 통합 테스트 (전체 파이프라인)
- [ ] 실제 데이터로 검증

---

## 4단계: 검증 자동화

### 4.1 CI/CD 통합
```yaml
# .github/workflows/crawler-validation.yml
name: Crawler Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Test data sources
        run: python scripts/test_data_sources.py
      - name: Validate estimates
        run: python scripts/validate_estimates.py
```

### 4.2 주기적 검증
- [ ] 주 1회 데이터 소스 접근 가능 여부 확인
- [ ] 월 1회 추정값 vs 실제값 비교
- [ ] 분기 1회 전체 크롤러 리뷰

---

## jeju_ferry_crawler 적용 사례

### 문제점
1. ❌ 1단계 미수행: API 존재 여부만 확인, 데이터 구조 미확인
2. ❌ 2단계 미수행: 연간 데이터 → 일별 분배 타당성 미검증
3. ❌ 3단계 위반: return None 함수 4개 존재
4. ❌ 4단계 미수행: 테스트 코드 없음

### 교훈
```
"API가 존재한다" ≠ "필요한 데이터가 있다"
"코드가 실행된다" ≠ "올바른 결과를 반환한다"
```

### 수정된 접근법

```python
# AS-IS: 하드코딩된 부정확한 추정
def _get_default_estimate(self, date: str):
    base_passengers = 1500  # ❌ 근거 없음
    ...

# TO-BE: 항공 데이터 기반 비율 추정
class JejuFerryEstimator:
    """검증된 통계 기반 해운 승객 추정"""
    
    # 근거: KOMSA 통계, 해운 분담률 ~5.5%
    FERRY_RATIO = 0.055
    
    def estimate(self, air_passengers: dict) -> dict:
        """항공 승객 수로부터 해운 승객 추정"""
        return {
            'arrival': int(air_passengers['arrival'] * self.FERRY_RATIO),
            'departure': int(air_passengers['departure'] * self.FERRY_RATIO),
            'source': 'air_traffic_ratio',
            'confidence': 0.85,  # 추정 신뢰도
        }
```

---

## 체크리스트 템플릿 (새 크롤러 개발 시)

```markdown
# [크롤러명] 개발 체크리스트

## 데이터 소스
- 소스 URL: 
- 접근 방식: API / 웹크롤링
- 인증 필요: 예 / 아니오
- 테스트 완료: [ ]

## 데이터 검증
- 데이터 형식: JSON / CSV / HTML
- 갱신 주기: 
- 전체 행 수: 
- 필요 컬럼 존재: [ ]

## 구현 검증
- 미구현 함수 없음: [ ]
- 에러 핸들링 완료: [ ]
- 테스트 코드 작성: [ ]

## 승인
- 코드 리뷰: [ ]
- 실데이터 테스트: [ ]
- 배포 승인: [ ]
```
