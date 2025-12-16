# Project Status Backup
> Last Updated: 2025-12-16 19:25 KST

## Project Overview
- **Project**: 제주도 전력 수요 예측 시스템
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v1.0.0 (Released)

---

## Completed Tasks

### Backend (100% Complete)
- [x] Task 1-22: Core ML Pipeline
- [x] Task 23: XAI (Explainability)
- [x] Task 24: Scenario Analysis
- [x] Task 25: Integrated Pipeline
- [x] API Server (FastAPI)
- [x] Monitoring System
- [x] All tests passing (1,423 tests)

### Frontend (100% Complete)
- [x] Streamlit Dashboard 생성
- [x] **API 연동 완료** (FastAPI 실시간 연동)
- [x] 실시간 예측 차트 (24시간)
- [x] 기상 조건 입력 인터페이스 (온도, 습도, 풍속)
- [x] 시나리오 분석 (폭염/한파) - API 배치 예측
- [x] 과거 데이터 비교
- [x] 모델 성능 지표 표시
- [x] Streamlit Cloud 배포 설정

### Model Performance
| Metric | Value |
|--------|-------|
| MAPE | 6.32% |
| R² | 0.852 |
| Best Model | conditional_soft |

### API Endpoints (Running on http://localhost:8000)
- `GET /health` - Health check
- `GET /models` - Model info (demand_only: 17 features, weather_full: 18 features)
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction
- `POST /predict/conditional` - Conditional prediction (soft/hard mode)

---

## Running Services

### API Server
```bash
# 실행 명령
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 상태
- URL: http://localhost:8000
- Device: MPS (Apple Silicon)
- Models: demand_only, weather_full
```

### Dashboard
```bash
# 실행 명령
streamlit run src/dashboard/app.py

# 상태
- URL: http://localhost:8501
- API 연동: 완료
```

---

## Dashboard Features (API 연동 버전)

### Tab 1: 실시간 예측
- "예측 실행" 버튼 → API `/predict/conditional` 호출
- 실제 BiLSTM 모델로 예측 수행
- 최근 168시간 데이터 기반
- 예측 결과: 수요값, 모델명, 처리시간 표시

### Tab 2: 시나리오 분석
- API `/predict/batch` 호출
- 5가지 시나리오 프리셋:
  - 평년
  - 약한 폭염 (+3°C)
  - 심한 폭염 (+7°C)
  - 약한 한파 (-5°C)
  - 심한 한파 (-10°C)
- 다중 시나리오 비교 차트
- 통계 테이블 (평균/피크/최소 수요)

### Tab 3: 과거 데이터
- 기간별 필터링
- 시계열 차트
- 시간대별 패턴 분석
- 상세 데이터 테이블

### Tab 4: 모델 정보
- API `/models`에서 실시간 조회
- 모델별 피처 수, Hidden Size, 레이어 수 표시
- 모델 비교 차트

### Tab 5: 시스템 정보
- API 상태 JSON
- 데이터 정보
- API 엔드포인트 목록
- 사용 가이드

---

## Key Files

### Dashboard
```
/src/dashboard/app.py           - API 연동 Streamlit 대시보드 (950 lines)
/.streamlit/config.toml         - Streamlit 설정
/requirements-streamlit.txt     - Dashboard 의존성
```

### Backend
```
/api/main.py              - FastAPI 서버
/api/service.py           - 예측 서비스 로직
/api/schemas.py           - Pydantic 스키마
/src/pipeline.py          - Main pipeline
/src/models/              - LSTM, BiLSTM, TFT models
/src/analysis/            - XAI, Anomaly, Scenario
```

### Data
```
/data/processed/jeju_hourly_merged.csv  - 105,190 records (2013-2024)
```

---

## How to Run

### 1. API 서버 실행
```bash
cd /Users/ibkim/Ormi_1/power-demand-forecast
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. 대시보드 실행
```bash
streamlit run src/dashboard/app.py
```

### 3. 브라우저 접속
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## Notes
- Python 3.13, PyTorch 2.0+, MPS (Apple Silicon)
- 대시보드는 API 서버가 실행 중일 때 실시간 예측 가능
- API 오프라인 시 경고 메시지 표시
- Data: 2013-2024 hourly data (105,190 records)
