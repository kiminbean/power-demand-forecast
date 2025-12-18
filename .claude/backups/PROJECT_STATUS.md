# Project Status Backup
> Last Updated: 2025-12-18 10:35 KST

## Project Overview
- **Project**: 제주도 전력 수요 예측 시스템
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v1.1.2

---

## Recent Changes (2025-12-18)

### New Features
- [x] **제주 실시간 크롤러** (`tools/crawlers/jeju_realtime_crawler.py`) - NEW!
  - KPX 제주 실시간 전력수급 페이지 크롤링
  - URL: https://www.kpx.or.kr/powerinfoJeju.es?mid=a10404040000
  - 5분 간격 업데이트 (60초 캐시 TTL)
  - 데이터: 공급능력, 현재부하, 공급예비력, 운영예비력, 예비율
  - 상태 판단: 정상(≥15%) / 관심(≥10%) / 주의(≥5%) / 위험(<5%)

- [x] **대시보드 실시간 데이터 연동** (`src/dashboard/app_v1.py`)
  - 제주 실측 탭 상단에 실시간 데이터 표시
  - 4개 게이지 (공급능력, 현재부하, 공급예비력, 예비율)
  - 상태 색상 표시 (녹색/노랑/주황/빨강)

- [x] **제주 크롤러 자동 다운로드 기능** (`tools/crawlers/jeju_power_crawler.py`)
  - `auto_download()` 메서드 추가
  - 캐시 관리 (7일 TTL)
  - CLI 옵션 추가: `--auto-download`, `--force`, `--zip`
  - 다운로드 링크 추출 패턴 확장 (3가지)
  - 스트리밍 다운로드 (메모리 효율)
  - ZIP 파일 유효성 검사

- [x] **대시보드 자동 다운로드 연동** (`src/dashboard/app_v1.py`)
  - 제주 실측 탭에서 `auto_download()` 사용
  - 수동 ZIP 경로 지정 불필요

### Commits
```
43cbef3 feat: Add Jeju realtime power crawler with KPX integration
73e5889 docs: Update CHANGELOG.md for v1.1.1 release
32c4810 refactor: Remove redundant Jeju estimation tab from dashboard
fa971d5 test: Add comprehensive tests for auto_download functionality
31d086e feat: Add auto-download functionality to Jeju power crawler
```

### Tests
- [x] 제주 실시간 크롤러 테스트 완료 (`tests/test_jeju_realtime_crawler.py`, 23개)
- [x] 자동 다운로드 기능 테스트 완료
- [x] 전체 테스트: **1,471 passed**

---

## Previous Changes (2025-12-17)

### New Features
- [x] **제주 전력수급현황 크롤러** (`tools/crawlers/jeju_power_crawler.py`)
  - 공공데이터포털 실측 데이터 로드 (data.go.kr)
  - ZIP 파일 처리 (5개 CSV: 계통수요, 공급능력, 공급예비력, 예측수요, 운영예비력)
  - 14,592건 데이터 (2023-09-01 ~ 2025-04-30)

- [x] **대시보드 제주 실측 탭** (`src/dashboard/app_v1.py`)
  - "📊 제주 실측" 탭 추가 (전국 현황과 함께)
  - 4개 게이지 (공급능력, 계통수요, 공급예비력, 예비율)
  - 7일간 수급 추이 차트
  - 상세 데이터 테이블

### Bug Fixes
- [x] EPSIS 실시간 데이터 `AttributeError` 수정 (dict 접근 방식)
- [x] 예비력/예비율 그래프 표시 문제 해결 (fill 제거, line width 증가)
- [x] 전국 탭 게이지 표시 문제 해결 (metrics → gauge charts)
- [x] Dashboard test import errors 수정
- [x] Streamlit deprecation warnings 수정 (`use_container_width` → `width`)

### Tests
- [x] 제주 크롤러 테스트 추가 (`tests/test_jeju_crawler.py`, 33개)
- [x] 전체 테스트: **1,448 passed**, 3 skipped

---

## Completed Tasks

### Backend (100% Complete)
- [x] Task 1-22: Core ML Pipeline
- [x] Task 23: XAI (Explainability)
- [x] Task 24: Scenario Analysis
- [x] Task 25: Integrated Pipeline
- [x] API Server (FastAPI)
- [x] Monitoring System
- [x] EPSIS 크롤러 (전국 실시간 데이터)
- [x] 제주 전력수급 크롤러 (공공데이터포털 + 자동 다운로드)
- [x] 제주 실시간 크롤러 (KPX 5분 간격)
- [x] All tests passing (1,471 tests)

### Frontend (100% Complete)
- [x] Streamlit Dashboard (app.py - API 연동)
- [x] Streamlit Dashboard v1.0 (app_v1.py - EPSIS 실시간)
- [x] EPSIS 실시간 수급 현황 (전국/제주 실측)
- [x] 실시간 예측 차트 (24시간)
- [x] 시나리오 분석 (폭염/한파)
- [x] 과거 데이터 비교
- [x] 모델 성능 지표 표시

### Model Performance
| Metric | Value |
|--------|-------|
| MAPE | 6.32% |
| R² | 0.852 |
| Best Model | conditional_soft |

---

## Data Sources

### EPSIS (전력통계정보시스템)
- **URL**: epsis.kpx.or.kr
- **Data**: 전국 실시간 전력수급 (5분 간격)
- **Fields**: 공급능력, 현재수요, 예비력, 예비율

### KPX 제주 실시간 (NEW!)
- **URL**: kpx.or.kr/powerinfoJeju.es
- **Data**: 제주 실시간 전력수급 (5분 간격)
- **Fields**: 공급능력, 현재부하, 공급예비력, 운영예비력, 예비율
- **Cache TTL**: 60초

### 공공데이터포털 (제주)
- **URL**: data.go.kr/data/15125113
- **Data**: 제주 전력수급현황 (시간별)
- **Period**: 2023-09-01 ~ 2025-04-30
- **Records**: 14,592건
- **Fields**: 계통수요, 공급능력, 공급예비력, 예측수요, 운영예비력
- **Auto-download**: 7일 캐시 TTL

### 기상청/한전
- **Data**: 시간별 기상 데이터, 전력 수요
- **Period**: 2013-2024
- **Records**: 105,190건

---

## Running Services

### API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Status
- URL: http://localhost:8000
- Device: MPS (Apple Silicon)
- Models: demand_only, weather_full
```

### Dashboard (v1.0 - EPSIS)
```bash
streamlit run src/dashboard/app_v1.py

# Status
- URL: http://localhost:8501
- Features: EPSIS 실시간, 제주 실측 (자동 다운로드)
```

### Dashboard (API 연동)
```bash
streamlit run src/dashboard/app.py

# Status
- URL: http://localhost:8501
- API 연동: 완료
```

---

## Key Files

### Crawlers
```
/tools/crawlers/epsis_crawler.py          - EPSIS 전국 실시간 크롤러
/tools/crawlers/jeju_power_crawler.py     - 제주 전력수급 크롤러 (자동 다운로드)
/tools/crawlers/jeju_realtime_crawler.py  - 제주 실시간 크롤러 (KPX 5분 간격) - NEW!
```

### Dashboard
```
/src/dashboard/app.py      - API 연동 대시보드
/src/dashboard/app_v1.py   - EPSIS 실시간 대시보드 (1,800+ lines)
/.streamlit/config.toml    - Streamlit 설정
```

### Data
```
/data/processed/jeju_hourly_merged.csv  - 과거 데이터 (105,190 records)
/data/jeju_power_supply.zip             - 제주 실측 데이터 (14,592 records)
```

### Tests
```
/tests/test_jeju_crawler.py          - 제주 크롤러 테스트 (45 tests)
/tests/test_jeju_realtime_crawler.py - 제주 실시간 크롤러 테스트 (23 tests) - NEW!
/tests/test_dashboard.py             - 대시보드 테스트 (23 tests)
```

---

## How to Run

### 1. API 서버 실행
```bash
cd /Users/ibkim/Ormi_1/power-demand-forecast
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. 대시보드 실행 (EPSIS 버전)
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run src/dashboard/app_v1.py
```

### 3. 제주 데이터 자동 다운로드
```bash
python tools/crawlers/jeju_power_crawler.py --auto-download
python tools/crawlers/jeju_power_crawler.py --auto-download --force  # 강제 다운로드
```

### 4. 테스트 실행
```bash
python -m pytest tests/ -v
```

### 5. 브라우저 접속
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## Notes
- Python 3.13, PyTorch 2.0+, MPS (Apple Silicon)
- Protobuf 환경변수 필요: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- 제주 실측 데이터: 자동 다운로드 지원 (7일 캐시)
- EPSIS 크롤러는 실시간 웹 크롤링
