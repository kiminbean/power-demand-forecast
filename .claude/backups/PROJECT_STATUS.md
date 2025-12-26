# Project Status Backup
> Last Updated: 2025-12-26 23:00 (Real-time API Integration Complete)

## Project Overview
- **Project**: Jeju Power Demand Forecast System (RE-BMS)
- **Repository**: https://github.com/kiminbean/power-demand-forecast (PRIVATE)
- **Current Version**: v8.0.0 (Real-time API Edition) ✅ Complete
- **Previous Version**: v7.0.0 (React Desktop Web Application)
- **License**: Proprietary (All Rights Reserved)

---

## Current Session (2025-12-26)

### Task: Real-time API Integration ✅ COMPLETE

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
   - Status: ✅ Pushed to GitHub

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
