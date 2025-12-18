# Project Status Backup
> Last Updated: 2025-12-18 12:45 KST

## Project Overview
- **Project**: 제주도 전력 수요 예측 시스템
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v2.0.0

---

## v2.0.0 SMP 예측 및 입찰 지원 시스템 (2025-12-18)

### 새로운 기능 (NEW!)

#### SMP (계통한계가격) 모듈
- [x] **SMP 크롤러** (`src/smp/crawlers/smp_crawler.py`)
  - KPX SMP 데이터 크롤링
  - 육지/제주 SMP 수집

- [x] **SMP 데이터 저장소** (`src/smp/crawlers/smp_data_store.py`)
  - CSV/JSON/Parquet 지원
  - 학습 데이터 추출 기능

- [x] **연료비 크롤러** (`src/smp/crawlers/fuel_cost_crawler.py`)
  - EPSIS 연료비 데이터 크롤링
  - 8가지 연료 유형 지원

#### 예측 모델
- [x] **SMP LSTM 모델** (`src/smp/models/smp_lstm.py`)
  - BiLSTM + Temporal Attention
  - Quantile 예측 (10%, 50%, 90%)
  - 48시간 입력 → 24시간 출력

- [x] **SMP TFT 모델** (`src/smp/models/smp_tft.py`)
  - Temporal Fusion Transformer
  - Variable Selection Network
  - Interpretable Multi-Head Attention

- [x] **발전량 예측기** (`src/smp/models/generation_predictor.py`)
  - 물리 기반 태양광 발전량 계산
  - 물리 기반 풍력 발전량 계산
  - 불확실성 추정

#### 입찰 전략 엔진
- [x] **입찰 전략 최적화** (`src/smp/bidding/strategy_optimizer.py`)
  - 리스크 수준별 전략 (보수적/중립/공격적)
  - 시간별 수익 분석
  - 최적 입찰 시간대 추천

- [x] **수익 시뮬레이션** (`src/smp/bidding/strategy_optimizer.py`)
  - 시나리오별 수익 계산 (Q10, Q50, Q90)
  - 리스크 조정 수익

- [x] **리스크 분석** (`src/smp/bidding/strategy_optimizer.py`)
  - VaR 기반 리스크 점수
  - 리스크 등급 판정

#### Dashboard v2.0
- [x] **입찰 지원 대시보드** (`src/dashboard/app_v2.py`)
  - 📊 입찰 지원 탭 (메인)
  - 📈 SMP 분석 탭
  - ☀️ 발전량 예측 탭
  - ⚡ 수급 현황 탭
  - ⚙️ 설정 탭

#### API 확장
- [x] **SMP API** (`api/smp_routes.py`, `api/smp_schemas.py`)
  - GET /smp/current - 현재 SMP 조회
  - POST /smp/predict - SMP 예측
  - GET /smp/compare - 육지/제주 비교
  - GET /smp/historical - 과거 데이터

- [x] **Bidding API** (`api/bidding_routes.py`, `api/bidding_schemas.py`)
  - POST /bidding/strategy - 입찰 전략 추천
  - POST /bidding/simulate - 수익 시뮬레이션
  - POST /bidding/generation/predict - 발전량 예측
  - POST /bidding/analyze - 종합 분석

### 테스트
- [x] SMP 모듈 테스트 (`tests/test_smp.py`) - 17개 테스트 통과
- [x] 전체 테스트: **1,488+ passed**

### 커밋
```
f9844ff feat: Add generation predictor and bidding strategy optimizer (Phase 3-4)
84a1b27 feat: Add Dashboard v2.0 for SMP prediction and bidding support
711ea37 feat: Add SMP and Bidding API endpoints (Phase 6)
a06c198 test: Add comprehensive SMP module tests (Phase 7)
f57e4bf fix: Fix BiddingHour attribute names and deprecated use_container_width
```

### Dashboard v2.0 상태
- **실행 중**: http://localhost:8502
- **상태**: 정상 동작 (HTTP 200)

---

## v1.x 기능 (이전 버전)

### 크롤러
- [x] EPSIS 전국 실시간 크롤러
- [x] 제주 전력수급현황 크롤러 (자동 다운로드)
- [x] 제주 실시간 크롤러 (KPX 5분 간격)

### 대시보드
- [x] Dashboard v1.0 (EPSIS 실시간)
- [x] 전국/제주 수급 현황
- [x] 예측 차트 (24시간)
- [x] 시나리오 분석

### 모델 성능
| Metric | Value |
|--------|-------|
| MAPE | 6.32% |
| R² | 0.852 |
| Best Model | conditional_soft |

---

## How to Run

### 1. Dashboard v2.0 (SMP 예측 및 입찰)
```bash
streamlit run src/dashboard/app_v2.py
```

### 2. Dashboard v1.0 (EPSIS)
```bash
streamlit run src/dashboard/app_v1.py
```

### 3. API 서버
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. 테스트
```bash
python -m pytest tests/test_smp.py -v
```

---

## Key Files (v2.0)

### SMP 모듈
```
/src/smp/                           - SMP 모듈 루트
/src/smp/crawlers/smp_crawler.py    - SMP 크롤러
/src/smp/crawlers/smp_data_store.py - 데이터 저장소
/src/smp/crawlers/fuel_cost_crawler.py - 연료비 크롤러
/src/smp/models/smp_lstm.py         - LSTM 모델
/src/smp/models/smp_tft.py          - TFT 모델
/src/smp/models/generation_predictor.py - 발전량 예측기
/src/smp/bidding/strategy_optimizer.py  - 입찰 전략
```

### Dashboard
```
/src/dashboard/app_v2.py   - SMP 예측 및 입찰 대시보드 (NEW!)
/src/dashboard/app_v1.py   - EPSIS 실시간 대시보드
```

### API
```
/api/smp_routes.py         - SMP API 라우터
/api/smp_schemas.py        - SMP Pydantic 스키마
/api/bidding_routes.py     - Bidding API 라우터
/api/bidding_schemas.py    - Bidding Pydantic 스키마
```

### Tests
```
/tests/test_smp.py         - SMP 모듈 테스트 (17 tests)
```

---

## Notes
- Python 3.11+, PyTorch 2.0+, MPS (Apple Silicon)
- v2.0은 민간 태양광/풍력 발전사업자를 위한 입찰 지원 기능 추가
- Dashboard v2.0은 Demo 모드로 동작 (실제 모델 학습 필요)
