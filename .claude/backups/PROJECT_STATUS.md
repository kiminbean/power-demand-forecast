# Project Status Backup
> Last Updated: 2025-12-18 13:50 KST

## Project Overview
- **Project**: 제주도 전력 수요 예측 시스템
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v2.0.0

---

## v2.0.0 SMP 예측 및 입찰 지원 시스템 (2025-12-18)

### 최신 세션 (2025-12-18 13:50)

#### 실제 SMP 데이터 크롤링 완료 ✅
- **KPX 크롤러 개선**: `fetch_weekly_data()` 메서드 추가
- **수집 데이터**: 168건 (7일 × 24시간)
- **데이터 기간**: 2025-12-12 ~ 2025-12-18
- **SMP 범위**: 560~840 원/kWh
- **저장 위치**: `data/smp/smp_history_real.csv`

```
📅 날짜별 SMP 통계:
  2025-12-12: 24건 | 평균  695.05 | 범위  596.26 ~  796.37
  2025-12-13: 24건 | 평균  664.26 | 범위  566.62 ~  773.21
  2025-12-14: 24건 | 평균  683.78 | 범위  574.77 ~  839.76
  2025-12-15: 24건 | 평균  711.14 | 범위  603.00 ~  836.41
  2025-12-16: 24건 | 평균  694.23 | 범위  600.69 ~  778.75
  2025-12-17: 24건 | 평균  702.79 | 범위  567.43 ~  832.73
  2025-12-18: 24건 | 평균  441.38 | 범위    0.00 ~  760.64 (진행 중)
```

### 완료된 작업 ✅

#### Phase 1-7: 전체 구현 완료
- [x] **SMP 크롤러** (`src/smp/crawlers/smp_crawler.py`)
- [x] **SMP 데이터 저장소** (`src/smp/crawlers/smp_data_store.py`)
- [x] **연료비 크롤러** (`src/smp/crawlers/fuel_cost_crawler.py`)
- [x] **SMP LSTM 모델** (`src/smp/models/smp_lstm.py`)
- [x] **SMP TFT 모델** (`src/smp/models/smp_tft.py`)
- [x] **발전량 예측기** (`src/smp/models/generation_predictor.py`)
- [x] **입찰 전략 최적화** (`src/smp/bidding/strategy_optimizer.py`)
- [x] **Dashboard v2.0** (`src/dashboard/app_v2.py`)
- [x] **SMP API** (`api/smp_routes.py`, `api/smp_schemas.py`)
- [x] **Bidding API** (`api/bidding_routes.py`, `api/bidding_schemas.py`)
- [x] **테스트** (`tests/test_smp.py`) - 17개 테스트 통과

### 오늘 세션에서 수정된 버그
1. **BiddingHour 속성명 오류 수정** (`app_v2.py`)
   - `h.smp` → `h.smp_predicted`
   - `h.generation` → `h.generation_kw`
   - `h.revenue` → `h.expected_revenue`
   - `h.recommended` → `h.is_recommended`

2. **Deprecated API 수정**
   - `use_container_width=True` → `width="stretch"`

### 커밋 히스토리
```
dbeace5 feat: Add SMP prediction module (v2.0 Phase 1-2)
f9844ff feat: Add generation predictor and bidding strategy optimizer (Phase 3-4)
84a1b27 feat: Add Dashboard v2.0 for SMP prediction and bidding support
711ea37 feat: Add SMP and Bidding API endpoints (Phase 6)
a06c198 test: Add comprehensive SMP module tests (Phase 7)
f57e4bf fix: Fix BiddingHour attribute names and deprecated use_container_width
ba9b6ee chore: Update PROJECT_STATUS.md with latest fix
```

### Dashboard v2.0 상태
- **URL**: http://localhost:8502
- **실행 명령**: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v2.py --server.port 8502`
- **상태**: 정상 동작 (HTTP 200)

---

## MCP 서버 설정

### 설치된 MCP 서버 (`~/.claude/settings.json`)
- sequential-thinking
- task-master-ai
- gemini
- context7
- gemini-web
- gemini-code-reviewer
- **figma** (NEW - 재시작 후 `/mcp`에서 인증 필요)

### Figma MCP 인증 방법
1. Claude Code 재시작
2. `/mcp` 입력
3. "figma" 선택 → "Authenticate"
4. 브라우저에서 "Allow Access" 클릭

---

## How to Run

### 1. Dashboard v2.0 (SMP 예측 및 입찰)
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v2.py --server.port 8502
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

---

## Session Recovery

다음 세션에서 복구하려면:
1. `.claude/backups/PROJECT_STATUS.md` 읽기
2. `git log --oneline -10` 확인
3. Dashboard v2.0 실행하여 동작 확인

---

## Notes
- Python 3.11+, PyTorch 2.0+, MPS (Apple Silicon)
- v2.0은 민간 태양광/풍력 발전사업자를 위한 입찰 지원 기능 추가
- Dashboard v2.0은 Demo 모드로 동작 (실제 모델 학습 필요)
- 모든 변경사항 origin/main에 푸시 완료
