# Project Status Backup
> Last Updated: 2026-11-06 (Demand retrain on raw target + AR combination experiments)

## Project Overview
- **Project**: Jeju Power Demand Forecast System (RE-BMS)
- **Repository**: https://github.com/kiminbean/power-demand-forecast (PRIVATE) ✅ Confirmed
- **Current Version**: v8.0.0 (Real-time API Edition) ✅ Complete
- **Previous Version**: v7.0.0 (React Desktop Web Application)
- **License**: Proprietary (All Rights Reserved)

---

## Current Session (2026-11-06)

### Task: Demand model improvement (raw target, AR/ramp7 gate, AR + correction) ✅ EXPERIMENTS COMPLETE · production unchanged

- 보고서: `docs/2026-11-06-demand-retrain-raw-target.md` · 스크립트: `docs/benchmark-2026-11-06/{compare_demand_retrain,diagnose_ar_residual,report_tables,probe_ar_anchor_collapse}.py`
- 커밋: 5665fac0(클리핑 제거·CSV 919시간 복원), e120d68e(게이트: persistence·ramp7·시간대별 AR, DM p<0.05, MAE 1% 개선, `--anchor ar`)
- 결과(demand_only, 원본 목표, 고정 test 2023-07~2024-12 / walk-forward 2021~2024 평균 MAPE):
  - 운영 LSTM(서빙 경로 재현, 원본 입력) 4.16% · 직전값 앵커 원본 재학습 4.17% / 3.66% 불합격 · AR 앵커 LSTM 3.78% / 3.45% 불합격(학습 붕괴, AR + 상수)
  - AR + GBM 잔차 보정(B) 3.63% / 3.35% **합격**(기준선 3종, 4/4 연도, AR 대비 MAE −4.0% / −2.7%, 서빙 모델 대비 −13%)
  - 시간대별 AR 3.78% / 3.45%
- 정정: 경쟁 비교 보고서의 운영 LSTM 4.31%·"최대 수요 53 MWh 과소예측"은 2026-11-05에 저장된(잘린 입력) 예측을 채점한 착오 → 서빙 경로 4.16% / R² 0.939, 상한 위 631시간 bias −5.9 MWh. 같은 날 보고서 정정
- AR 앵커 붕괴: 원래 체크포인트는 마지막 은닉 ReLU가 모든 test 입력에서 0 → 보정량 = 출력 편향(상수). 잔차 표준화·LeakyReLU(2~3에폭, 시드 1개)로도 학습 손실 불변. GBM은 비슷한 정보로 잔차를 학습했지만 LSTM 실패 원인은 미확정
- 산출물(git 미추적): 예측 CSV(models/demand_retrain_2026-11-06_raw_{last,ar}/, models/demand_ar_gbm_2026-11-06/). 두 실행의 oot_report.json은 커밋

### Task: AR + GBM formalization (training pipeline, serving path, opt-in API) ✅ COMPLETE · default unchanged

- 사용자 결정: 서빙 산출물은 최신 분할 절차(학습 ~2024-06, 검증 2024H2로 트리 수 선택), API는 선택형 `model_type="ar_gbm"`
- 코드: src/models/ar_gbm.py, src/training/train_demand_ar_gbm.py, src/inference/ar_gbm_predictor.py, api/{service,main,schemas,config}.py, requirements(scikit-learn==1.7.2)
- 산출물: models/demand_ar_gbm/ (375그루, 검증 MAE 18.60 vs AR 19.50)
- 검증: 파이프라인 = 진단 스크립트(5개 분할 예측 차이 0), 서빙 경로 = 학습 경로(실데이터 5개 시각 차이 0, 최소 337시간), tests/ 2248 passed
- 독립 리뷰 반영: 게이트 통과 산출물만 서빙, sha256·원자적 쓰기, 버전·상수 검사, 503 응답에서 경로 숨김, sklearn 지연 import, 휴일 표 범위 밖 학습 거부
- 한계: 음력 공휴일 표 2025년까지(2026년 설·추석 누락, LSTM도 동일), Docker 미빌드·compose 마운트 없음, 운영 API(launchd) 미재시작

### Next Steps (demand)
- 사용자 결정 대기: 운영 기본값을 ar_gbm으로 바꿀지, 운영 API 재시작 시점
- 음력 공휴일 표 2026년 이후 추가(공식 자료로 날짜 확인 필요)
- 운영 교체는 반드시 사용자 확인 후

### Task: Competitor benchmark (MAPE/R² vs operators, companies, papers) ✅ COMPLETE

- 보고서: `docs/2026-11-06-competitor-benchmark.md` · 재현 스크립트: `docs/benchmark-2026-11-06/verify_{ar,smp,solar}.py`
- 수요 LSTM: 같은 목표(KPX 제주 전력거래량) 경쟁자 없음 → 판정 불가. 단 원본 목표 기준 4.31%/R² 0.926으로 시간대별 선형 AR(3.75%/0.951)에 고정 test·walk-forward 4/4 연도 모두 짐 (DM +10.65) — 2026-11-06 정정: 서빙 경로 4.16%/0.939, AR 대비 DM +14.21 (위 Demand model improvement 참고)
- 수요 목표가 694.658 MWh에서 잘려 있음(test 포함 전체 기간 Q3+1.5·IQR, `src/data/preprocessing.py` handle_outliers) → 기존 4.03%/0.937은 외부 보고 금지
- SMP: 서빙 naive(MAE 11.97, 2025H2 walk-forward)는 vpp-capstone LightGBM(naive 대비 −17%, 2025)·VOLTA(−10~27%, 2024)보다 나쁨. 0.5×어제 + 0.5×7일 중앙값 혼합이 MAE −8.8~9.6%
- 태양광(RE-BMS): KPX 오차율 17.0%로 월×시간 기후값 12.5%보다 나쁨. "R²=0.95"는 무작위 분할·전체 fit 스케일러·측정 일사량 평가 수치
- CLAUDE.md 정정: SMP v4.1 8.76%는 육지 SMP 목표 수치, 현재 생산은 naive
- 운영: 2026-11-06 01:04부터 제주 SMP API 429(조사 작업의 대량 수집 직후 일일 한도 소진) → 00:58 마지막 검증 가격으로 폴백 중. 실패 시 backoff 없음(약 30초마다 재호출)

### Next Steps (competitor benchmark)
- 수요: 클리핑 제거 후 원본 목표 재학습, 배포 게이트에 AR·7일 ramp 추가
- SMP: 혼합 규칙 서빙 검토 → LightGBM(D-1 11:00 정보만)
- 태양광: "R²=0.95" 라벨 제거, 시간순 재학습 + 기상청 예보 입력, KPX 규칙 평가
- SMP API 캐시(일일 한도 보호)

### Task: Demand LSTM retrain + out-of-time validation ✅ COMPLETE

- 기존 production 수요 LSTM은 리포트 MAPE 6.06%를 재현 못 함(재현 10.05%), persistence(t-1) 5.12%보다 나쁨
- 직접 타깃 재학습(7.74%/8.46%)도 persistence 미달 → 중단
- 잔차 타깃 LSTM(`LastValueResidual`)으로 재학습: demand_only test 4.03%/R² 0.937, weather_full 4.14%/0.933, walk-forward 2021~2024 평균 3.6% (persistence 4.49%), 4/4 fold 우위
- models/production 교체(이전 버전 백업: models/_pre_retrain_backup_2026-11-05/demand_lstm/), API(launchd com.powerdemand.api, :8001) 재시작 후 /predict가 학습 예측과 일치 확인
- 코드: src/training/retrain_demand_oot.py, src/models/lstm.py(LastValueResidual), src/inference/predict.py(residual_target 래핑), train_production.train_model(model_wrapper 인자), tests/test_retrain_demand_oot.py

### Next Steps
- 다중 시점(24h) 예측에도 잔차 구조 적용 검토
- tools/generate_model_figures.py 그림 재생성 (새 training_report.json 스키마 호환)

---

## Current Session (2025-01-10)

### Task: SMP Model Version History & Documentation ✅ COMPLETE

#### Completed Work

1. **SMP 모델 버전 이력 이미지 생성**
   - 파일: `results/smp_model_version_history.png`
   - 24개 SMP 모델 버전 (v1.0 ~ v5.0) 정리
   - MAPE, R², 학습일자, 모델 아키텍처, 설명 포함

2. **SMP 모델 학습 코드 위치 정리**
   - 디렉토리: `src/smp/models/`
   - 주요 파일: `train_smp_v3_12_stacking.py` (BEST), `train_smp_v3_19_recent.py` (Production)

3. **GitHub 저장소 Private 변경**
   - `gh repo edit --visibility private` 실행
   - 상태: ✅ PRIVATE

4. **세션 요약 저장**
   - 파일: `.claude/backups/SESSION_SUMMARY_2025-01-10.md`

#### SMP Model Summary

| 버전 | MAPE | R² | 상태 |
|------|------|-----|------|
| v3.12 Stacking | 5.25% | 0.834 | ⭐ BEST |
| v3.19 CatBoost | 5.28% | 0.827 | ⭐ Production |

---

## Previous Session (2025-12-28)

### Task: CatBoost v3.19 Documentation & Volatility Analysis ✅ COMPLETE

#### Completed Work

1. **CatBoost v3.19 Deployment**
   - Updated `smp_catboost_predictor.py` with v3.19 version
   - Updated `api/smp_routes.py` with v3.19 metrics
   - Model: MAPE 5.28%, R² 0.827

2. **Colab Notebooks Created**
   - `notebooks/CatBoost_v3_19_Validation_Colab.ipynb` - Model verification
   - `notebooks/CatBoost_v3_19_Full_Pipeline_Colab.ipynb` - Train/Val/Test pipeline

3. **Volatility-Based Performance Analysis**
   | Volatility | Samples | MAPE |
   |------------|---------|------|
   | Normal (<10%) | 97.6% | 2.21% |
   | High (10-20%) | 1.9% | 8.17% |
   | Very High (20-30%) | 0.3% | 13.69% |
   | Extreme (>30%) | 0.2% | 40.68% |

4. **Colab vs Original Metrics Difference Explained**
   - Original: 2024-12-16 ~ (367 days, 8,741 samples, 6.36% volatility)
   - Colab: 2025-03-18 ~ (275 days, 6,556 samples, 6.04% volatility)
   - Original includes volatile winter period (Dec-Feb)
   - `smp_lag1` dominance (61.59%) → high accuracy in stable periods

5. **Technical Documentation Updated**
   - Added Section 9: SMP 예측 모델 (CatBoost v3.19)
   - Version updated to 6.0.0
   - Committed: `c6b8d5d7`

6. **Unit Tests Added**
   - 7 new CatBoost-specific tests in `tests/test_smp.py`
   - All 24 tests passed

#### Key Findings

- Model relies heavily on `smp_lag1` (61.59% importance) → "Naive Forecast" behavior
- Works excellently in stable markets (MAPE ~2%), fails in volatile periods (MAPE ~40%)
- Different test periods explain Colab (0.99%) vs Original (5.28%) MAPE difference

---

## Previous Session (2025-12-26)

### Task: Mobile App v7.0.0 Cross-Platform Edition ✅ COMPLETE

#### Completed Work

1. **ExecoDashboardScreen** - Main command center
   - Real-time KPIs (Demand, SMP, Renewable %, Frequency, Weather)
   - API status indicators (SMP, Power, Weather)
   - Power supply visualization
   - Plant statistics (Solar, Wind, ESS)

2. **BiddingScreen** - Enhanced with web-v7 features
   - Hour selection (00:00 - 23:00)
   - Risk level selection (Conservative/Moderate/Aggressive)
   - 10-segment bidding curve
   - Bid status workflow (Draft → Approved)
   - DAM/RTM simulation navigation

3. **SettlementScreen** - Enhanced with web-v7 features
   - Period selector (1 week/1 month/quarter)
   - DA/RT SMP info cards
   - Current SMP card with high/low
   - Revenue chart and transaction history

4. **App.tsx** - Updated navigation
   - 5-tab structure: Dashboard, SMP, Bidding, Portfolio, Settlement
   - Error boundary for stability
   - Alan AI floating button
   - DAM/RTM simulation screens

5. **Package.json** - Updated to v7.0.0
   - Cross-platform support (iOS, Android, Web)
   - 100% feature parity with web-v7

#### Build Status
- ✅ Web build successful (1.09 MB bundle)
- ✅ TypeScript compilation passed
- ✅ All screens render correctly

---

### Previous Task: Real-time API Integration ✅ COMPLETE

#### Completed Work

1. **제주시범사업 SMP API Integration**
   - Endpoint: `https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2`
   - Returns: Real-time Jeju SMP (원/kWh) + 수요예측량 (MW)
   - Status: ✅ Connected (84.3원/kWh working)

2. **KMA Weather API Integration**
   - Endpoint: `https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst`
   - Returns: Temperature, Humidity, Wind Speed
   - Status: ✅ Connected (2°C working)

3. **Docker Deployment Fixed**
   - Fixed `.dockerignore` (tools directory now included)
   - Fixed `Dockerfile.api` (added `COPY tools/ ./tools/`)
   - Fixed crawler imports (individual try/except)
   - Status: ✅ All containers running

4. **Data Source Priority Configuration**
   | Data Type | Priority 1 | Priority 2 |
   |-----------|------------|------------|
   | SMP | 제주시범사업 API | Crawler |
   | Power Supply | Crawler | - |
   | Weather | KMA API | Crawler |

5. **Git Commit & Push**
   - Commit: "feat: Add Jeju SMP API and improve data source priorities"
   - Commit: "docs: Update CLAUDE.md with Docker deployment and API integration docs"
   - Status: ✅ All pushed to GitHub

6. **Session Cleanup**
   - All background tasks killed (uvicorn, streamlit, vite, expo)
   - Docker containers stopped (rebms-api, rebms-web-v7, rebms-mobile)
   - Status: ✅ Clean state

#### Current Dashboard Status
```
✅ SMP: 84.3원/kWh (제주시범사업 API)
✅ Power Supply: 865MW, 73.4% reserve (Crawler)
✅ Weather: 2°C (KMA API)
✅ Frequency: 59.99Hz (Simulated)
```

---

## Previous Session: SMP Model Experiments

### Goal: Reach R² 0.9+ (Best achieved: v3.12 CatBoost CV R² 0.834)

#### Experiments Completed

| Model | Strategy | Test R² | Test MAPE | Result |
|-------|----------|---------|-----------|--------|
| v3.12 CatBoost CV | 5-fold CV baseline | 0.834 | 5.25% | **BEST** ✅ |
| v3.14 +Fuel | Fuel price features | 0.534 | 12.56% | ❌ FAILED |
| v3.15 +Supply | Supply margin features | 0.107 | 12.09% | ❌ FAILED |
| v3.16 +Weather | Weather features | 0.788 | 5.73% | ❌ FAILED |
| v3.18 Transform | Box-Cox target transform | 0.793 | 5.69% | ❌ FAILED |
| v3.19 +EMA | EMA + recent data | 0.827 | 5.28% | ❌ FAILED |
| v3.20 Recent | 2022+ data only | 0.779 | 7.75% | ❌ FAILED |

#### Key Findings

1. **External data doesn't help**:
   - Fuel prices: 0% feature importance, data mismatch (2022+ only)
   - Weather: <0.5% importance for all weather features
   - Supply margin: Not in top 20 features, only 33% data overlap

2. **smp_lag1 dominates everything**:
   - 40-80% feature importance in ALL models
   - Recent SMP history is the only reliable predictor

3. **Concept drift is severe**:
   - Fold 1 (2020-2021 COVID era) consistently has R² ~ -2.5
   - This drags down CV performance to ~0.1 for most models
   - v3.12's R² 0.834 benefits from Folds 3-5 having similar train/test periods

4. **Target transformation doesn't help**:
   - Box-Cox, log, sqrt all tested - none improved performance
   - The distribution is not the issue

5. **More recent data = worse performance**:
   - Using only 2022+ data: R² 0.779
   - Using only 2023+ data: R² 0.760
   - Less training data hurts more than concept drift

6. **The R² 0.9+ goal may be unreachable**:
   - SMP is inherently noisy and unpredictable
   - External factors not captured in available data
   - v3.12's 0.834 is likely near the ceiling for this feature set

#### Files Created This Session

```
src/smp/models/train_smp_v3_14_fuel.py       - Fuel price features (FAILED)
src/smp/models/train_smp_v3_15_supply.py     - Supply margin features (FAILED)
src/smp/models/train_smp_v3_16_weather.py    - Weather features (FAILED)
src/smp/models/train_smp_v3_18_transform.py  - Target transformation (FAILED)
src/smp/models/train_smp_v3_19_recent.py     - EMA + recent data (FAILED)
src/smp/models/train_smp_v3_20_recent_only.py - Recent data only (FAILED)

models/smp_v3_14_fuel/
models/smp_v3_15_supply/
models/smp_v3_16_weather/
models/smp_v3_18_transform/
models/smp_v3_19_recent/
models/smp_v3_20_recent/
```

#### Recommendations for Future Work

1. **Accept v3.12 as production model**:
   - R² 0.834, MAPE 5.25% is a solid result
   - Significant external data doesn't improve it

2. **Focus on other improvements**:
   - Confidence intervals / uncertainty quantification
   - Hourly prediction for different time horizons
   - Model monitoring and drift detection

3. **Alternative approaches (if R² 0.9+ still needed)**:
   - Real-time market data (if available)
   - Transmission constraints and grid topology
   - Bidding behavior patterns from market participants

---

## Model Comparison (Full History)

| Model | MAPE | R² | Features | Strategy |
|-------|------|-----|----------|----------|
| v3.2 (Optuna) | 7.42% | 0.760 | 22 | BiLSTM+Attention |
| v3.3-v3.6 | - | < 0.6 | - | Generation data (failed) |
| v3.7 | 9.21% | 0.402 | 26 | Long-term MA (failed) |
| v3.8 LightGBM | 5.46% | 0.815 | 36 | LightGBM default |
| v3.9 Optuna LGB | 5.49% | 0.821 | 60 | Optuna-tuned LightGBM |
| v3.10 CatBoost | 5.38% | 0.826 | 60 | Optuna-tuned CatBoost |
| v3.11 XGBoost | 5.39% | 0.822 | 60 | Optuna-tuned XGBoost |
| **v3.12 CatBoost CV** | **5.25%** | **0.834** | 60 | **5-fold CV best** ✅ |
| v3.14 +Fuel | 12.56% | 0.534 | 70 | Fuel prices (FAILED) |
| v3.15 +Supply | 12.09% | 0.107 | 77 | Supply margin (FAILED) |
| v3.16 +Weather | 5.73% | 0.788 | 75 | Weather (FAILED) |
| v3.18 Transform | 5.69% | 0.793 | 60 | Box-Cox (FAILED) |
| v3.19 +EMA | 5.28% | 0.827 | 67 | EMA features (FAILED) |
| v3.20 Recent | 7.75% | 0.779 | 60 | 2022+ data (FAILED) |

---

## Key Files

### SMP Models
```
models/smp_v3_12_stacking/      - v3.12 CatBoost CV (BEST: R² 0.834, MAPE 5.25%)
models/smp_v3_10_catboost/      - v3.10 CatBoost (R² 0.826, MAPE 5.38%)
models/smp_v3_9_optuna_lgb/     - v3.9 Optuna LGB (R² 0.821)
models/smp_v3_8_ensemble/       - v3.8 LightGBM (R² 0.815)
src/smp/models/train_smp_v3_12_stacking.py  - v3.12 source code
```

### Crawlers
```
src/crawlers/public_data_crawler.py    - data.go.kr search
src/crawlers/download_public_data.py   - data.go.kr download
src/crawlers/epsis_crawler.py          - EPSIS SMP data
```

### web-v7 (Current Production)
```
web-v7/src/pages/ExecoDashboard.tsx  - Main dashboard
web-v7/src/hooks/useApi.ts          - API hooks
```

---

## Environment
- Python 3.13, PyTorch 2.0+
- Node.js 20, React 18, TypeScript
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- Docker Desktop

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Run `git log --oneline -10`
3. **DOCKER**: `docker compose -f docker/docker-compose.v7.yml up -d --build`
4. **DASHBOARDS**:
   - ExecO Dashboard: http://localhost:8700
   - Bidding Dashboard: http://localhost:8600
   - Mobile App: http://localhost:3001
5. **API KEY**: `DATA_GO_KR_API_KEY=7d42f7c08...` (set in .env or Docker environment)
6. **BEST MODEL**: v3.12 CatBoost CV (R² 0.834, MAPE 5.25%)
7. **ALL APIS CONNECTED**: SMP (제주시범사업), Weather (KMA), Power (Crawler)
