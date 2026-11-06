# Claude Code Instructions

## Backup Protocol

**IMPORTANT**: Update `.claude/backups/PROJECT_STATUS.md` after completing significant tasks:

1. **When to backup**:
   - After completing a major feature
   - After fixing critical bugs
   - Before ending a session
   - Every 30 minutes during long sessions

2. **What to backup**:
   - Current task status
   - Completed items
   - Next steps
   - Recent commits
   - Any blockers or issues

3. **Backup command**:
   ```bash
   # Update PROJECT_STATUS.md with current progress
   ```

---

## Project Context

### 제주도 전력 수요 예측 시스템

**Tech Stack**:
- Backend: Python 3.13, PyTorch, FastAPI
- ML Models: LSTM, BiLSTM, TFT, Ensemble
- Frontend: Streamlit (in progress)
- Database: File-based (CSV, Parquet)

**Key Directories**:
```
/src/           - Source code
/api/           - FastAPI server
/tests/         - Test suite (1,423 tests)
/models/        - Trained models
/data/          - Datasets
/results/       - Analysis results
/.claude/       - Claude backups
```

**Model Performance (2026-11-06 경쟁 비교 기준 · 상세 `docs/2026-11-06-competitor-benchmark.md`)**:
- 전력수요 LSTM(1h, 전력거래량): MAPE 4.16% / R² 0.939 (원본 목표, 서빙 경로 재현, 고정 test). 시간대별 선형 AR보다 나쁨(2019~2022 적합 3.75% / 0.951, 2013~ 적합한 게이트 AR 3.78% / 0.950). 처음 보고한 4.31% / 0.926은 잘린 입력으로 만든 예측을 채점한 착오
- 전력수요 후보(2026-11-06 원본 목표 재학습): AR + GBM 잔차 보정만 배포 게이트 통과(고정 test 3.63% / 0.954, walk-forward 평균 3.35%) → `docs/2026-11-06-demand-retrain-raw-target.md`
  - 정식화 완료: `src/models/ar_gbm.py`, `src/training/train_demand_ar_gbm.py`, `src/inference/ar_gbm_predictor.py`, 산출물 `models/demand_ar_gbm/`(학습 ~2024-06, 검증 2024년 하반기)
  - **API 기본값은 ar_gbm이다**(2026-11-06 사용자 승인). 2025년 실측 시간외 비교에서 MAPE 4.67% / R² 0.929로 서빙 LSTM(5.22% / 0.910)보다 정확했다(12개월 모두 우위, 같은 보고서 12절)
  - `model_type`을 생략하면 ar_gbm을 쓴다(최소 337시간). 산출물이 없거나 이력이 짧으면 demand_only로 대체하고 `model_used`에 표시한다. demand_only도 336시간이 필요해 336시간 미만은 400이다
  - 되돌리기: 루트 `.env`에 `DEFAULT_MODEL=demand_only` → launchd는 `launchctl kickstart -k gui/$(id -u)/com.powerdemand.api`, Docker는 `up -d --force-recreate --wait api`(`docker restart`는 env_file을 다시 읽지 않음)
  - 2025년 실측: `tools/fetch_kpx_jeju_trade.py` → `data/raw/jeju_hourly_power_2025.csv`(data.go.kr 15133498). 2026년 실측은 2027-07 공개 예정
  - 음력 공휴일 표는 2027년까지다(우주항공청 월력요항). 대체공휴일·선거일·노동절·제헌절은 학습 기준과 같게 넣지 않았다
  - GBM은 pickle이라 `scikit-learn==1.7.2`에서만 불러온다(requirements 고정, 로더가 버전 검사)
- SMP: 서빙 = 어제 같은 시각 naive (MAE 11.97원/kWh, MAPE 11.75%, R² 0.20; walk-forward 2025-07-22~12-18)
- 태양광(RE-BMS): KPX 오차율 17.0% (월×시간 기후값 12.5%보다 나쁨). 화면의 "R²=0.95"는 누수 평가
- v3.2 Optuna(7.42% / 0.760)는 스케일러 누수로 무효

---

## Session Recovery

If conversation is lost, read:
1. `.claude/backups/PROJECT_STATUS.md` - Current status
2. `CHANGELOG.md` - Version history
3. `README.md` - Project overview
4. `git log --oneline -20` - Recent commits

---

## Coding Standards

- Korean comments for domain logic
- English for technical documentation
- Follow existing code patterns
- Run tests after major changes
- Commit frequently with descriptive messages

---

## CRITICAL: UTF-8 Crash Prevention (Claude Code v2.0.72+)

**FATAL BUG**: Claude Code CLI crashes when Korean text appears in UI elements.

### Technical Cause
```
Rust panic: byte index N is not a char boundary; it is inside '한글' (bytes X..Y)
```
- Korean characters = 3 bytes in UTF-8
- Rust slices by byte index, not character boundary
- UI truncation cuts mid-character → **IMMEDIATE CRASH**

### Real Crash Example (2024-12-18)
```
byte index 5 is not a char boundary; it is inside '화' (bytes 3..6) of `완화)`
fatal runtime error: failed to initiate panic, error 5, aborting
```
- String: "완화)" = 완[0-2] + 화[3-5] + )[6]
- Rust tried to slice at byte 5 (middle of '화') → PANIC

### Crash Triggers
1. **TodoWrite content/activeForm** with Korean
2. **Session history** containing Korean in API responses
3. **Code output** with Korean that gets truncated in status bar
4. **Error messages** with Korean in stack traces

### MANDATORY RULES

1. **TodoWrite tool - ENGLISH ONLY**
   ```json
   // ❌ CRASH: {"content": "모델 학습", "activeForm": "학습 중"}
   // ✅ SAFE:  {"content": "Train model", "activeForm": "Training"}
   ```

2. **All status/progress messages**: English only

3. **Avoid Korean in console output** that may appear in status bar

### Recovery Commands

```bash
# 1. Quick recovery (move todo files)
mkdir -p ~/.claude/todos_backup && mv ~/.claude/todos/*.json ~/.claude/todos_backup/

# 2. Full cleanup (if crashes persist)
rm -rf ~/.claude/todos/*.json
rm -rf ~/.claude/todos_backup/

# 3. Nuclear option (clear all session data)
rm -rf ~/.claude/projects/-Users-ibkim-Project-power-demand-forecast/
```

### Preventive Checks

```bash
# Check for Korean in todo files
grep -l '[가-힣]' ~/.claude/todos/*.json 2>/dev/null && echo "WARNING: Korean found!"

# Check all claude files for Korean
find ~/.claude -name "*.json" -exec grep -l '[가-힣]' {} \; 2>/dev/null
```

**Bug report**: https://github.com/anthropics/claude-code/issues

---

## Auto Commit Protocol

**IMPORTANT**: Automatically commit changes after completing each task:

1. **When to commit**:
   - After completing a feature or significant code change
   - After fixing a bug
   - After refactoring code
   - Before starting a new, unrelated task

2. **Commit message format**:
   ```
   <type>: <short description>

   <detailed description if needed>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   ```

3. **Commit types**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `refactor`: Code refactoring
   - `docs`: Documentation changes
   - `test`: Test additions/changes
   - `chore`: Maintenance tasks

4. **Do NOT auto-push**: Only commit locally, let user decide when to push

---

## SMP Model Improvement Roadmap (R² 0.9+ Target)

**Last Updated**: 2026-11-06 (v4.1 8.76%가 육지 SMP 목표 수치였음을 정정, 현재 생산 = naive)

### Current Status

| Model Version | MAPE | R² | Status |
|---------------|------|-----|--------|
| v1.0~v8.1 (구버전 전체) | 무효 | 무효 | 스케일러 누수(시간 분할 전 fit)로 평가 자체가 무효 — 아래 2025-12-26 표는 역사 기록 |
| **v4.0 (2026-11-05 재학습)** | 17.17% (test) | 0.124 | 시간외 평가·신뢰 가능하나 성능 미달 |
| v4.1 (2026-11-05 재학습) | 8.76% (walk-forward 평균) | 0.463 | **육지 SMP 목표**로 학습된 수치(제주 아님). 제주 하루전 재학습은 16.82% / R² 0.136, MAE 15.34로 naive보다 나빠 게이트 불합격 (`models/smp_v41_serving/`) |
| **seasonal naive 24h (현재 생산)** | **11.75% (walk-forward 분할 평균)** | **0.203** | 어제 같은 시각 가격. MAE 11.97원/kWh. 경쟁 모델은 이 naive 대비 MAE −17% (`docs/2026-11-06-competitor-benchmark.md`) |
| v3.4 (2026-11-05 재학습) | 21.28% (test) | -0.537 | 사용 보류(일반화 실패) |
| **Target** | <5% | **0.9+** | Requires power generation data |

### API Keys Status

| API Source | Key Status | Usage |
|------------|------------|-------|
| data.go.kr | ✅ Available | Jeju SMP, Weather |
| 제주시범사업 API | ✅ Connected | Real-time Jeju SMP + 수요예측 |
| KMA (기상청) | ✅ Connected | 단기예보 API |
| EIA | ❌ Not set | US energy prices |
| KRX | ❌ Not set | K-ETS carbon prices |

### R² 0.9+ Roadmap

#### Phase 1: Real Data Acquisition ⏳
1. **Power Generation Data** (data.go.kr API)
   - 한국전력거래소_제주 시간별 전력 수급 현황
   - 신재생에너지 발전량 (태양광, 풍력)
   - HVDC 송전량 (제주-육지 연결)

2. **Fuel Price Data** (Yahoo Finance)
   - ✅ WTI, Brent crude oil
   - ✅ Natural gas futures
   - ✅ Heating oil

3. **Carbon Price Data**
   - K-ETS (한국탄소배출권거래소)
   - EU-ETS (참조용)

#### Phase 2: Data Preprocessing ⏳
- Remove synthetic/sample data
- Proper time alignment (hourly)
- Handle missing values with domain knowledge
- Validate data quality (no constant features)

#### Phase 3: Feature Engineering ⏳
- Net Load calculation (demand - renewable generation)
- Fuel price time-lag effects (7-day, 30-day MA)
- HVDC transmission patterns
- Marginal unit identification signals

#### Phase 4: Model Architecture ⏳
- BiLSTM + Multi-Head Attention (proven best)
- Quantile regression for uncertainty
- Walk-forward validation (5 splits)

#### Phase 5: Hyperparameter Tuning ⏳
- Optuna-based optimization
- Learning rate, hidden size, attention heads
- Dropout and regularization

### Data Sources

```
data/
├── smp/                    # SMP historical data
│   └── smp_5years_epsis.csv
├── processed/              # Preprocessed datasets
│   ├── jeju_weather_hourly_merged.csv
│   └── smp_enhanced_dataset.csv
└── external/               # Crawler outputs
    ├── epsis/              # Power generation
    ├── fuel/               # Fuel prices
    └── carbon/             # Carbon prices
```

### Crawlers

```
src/crawlers/
├── __init__.py
├── epsis_crawler.py        # EPSIS power generation
├── fuel_crawler.py         # Yahoo Finance fuel prices
├── carbon_crawler.py       # K-ETS carbon prices
└── data_collector.py       # Master orchestrator
```

### Key Insights from Gemini Discussion

1. **Net Load (순부하)** is critical for Jeju SMP prediction
   - Net Load = Demand - Renewable Generation
   - High renewable penetration causes SMP volatility

2. **HVDC Interconnection** affects Jeju SMP
   - Jeju connected to mainland via HVDC
   - Import/export dynamics impact local prices

3. **Fuel Price Time Lag**
   - LNG contracts have 3-6 month lag
   - Moving averages capture delayed effect

4. **Marginal Unit Identification**
   - SMP set by highest-cost generator in operation
   - Usually LNG or oil-fired plants during peak

---

## Model Training Results

### 2026-11-05 재학습 결과 (누수 수정 후 · 현재 기준)

| Model | 평가 | MAPE | R² | 비고 |
|-------|------|------|-----|------|
| v4.0 | test(시간외) | 17.17% | 0.124 | 80% 구간 커버리지 77.5% |
| v4.1 | walk-forward 평균 | 8.76% | 0.463 | 육지 SMP 목표 수치(제주 아님). 제주 재학습 16.82% / 0.136 → 게이트 불합격 |
| **seasonal naive 24h** | **walk-forward 분할 평균** | **11.75%** | **0.203** | MAE 11.97원/kWh — 현재 생산 |
| v3.4 | test(시간외) | 21.28% | -0.537 | 사용 보류(일반화 실패) |

원본: `models/smp_v4/smp_v4_metrics.json`, `models/smp_v4/smp_v41_metrics.json`, `models/smp_v3_4_kpx/metrics.json`. 경위·누수 버전 백업(`models/_pre_retrain_backup_2026-11-05/`): `docs/2026-11-02-forecast-upgrade-status.md`.

### 2026-11-06 전력수요 LSTM 재학습 (models/production · 현재 생산)

1시간 앞 수요 예측. 기존 체크포인트는 리포트 MAPE 6.06%를 재현하지 못했고(재현 10.05%), persistence(t-1, MAPE 5.12%)보다도 나빴다. 직접 타깃 재학습도 persistence를 못 넘어(7.74%/8.46%) **잔차 타깃**(예측 = 현재 수요 + LSTM 보정량, `LastValueResidual`)으로 교체했다.

| Model | 고정 test 2023-07~2024-12 MAPE / R² | walk-forward 2021~2024 평균 MAPE / R² | persistence (test / wf) |
|-------|------|------|------|
| **demand_only** (17 피처) | **4.03% / 0.937** | **3.61% / 0.935** | 5.12% / 4.49% |
| **weather_full** (18 피처, 기상은 기온 하나) | 4.14% / 0.933 | 3.60% / 0.935 | 5.12% / 4.49% |

- **주의 (2026-11-06 경쟁 비교):** 위 수치는 694.658 MWh에서 잘린 목표 기준이다. 상한이 test를 포함한 전체 기간으로 계산돼 누수가 있다. 원본 목표·서빙 경로(원본 입력) 기준 고정 test는 demand_only 4.16% / R² 0.939, weather_full 4.26% / 0.936, persistence 5.28%다(처음 보고한 4.31%·4.41%는 잘린 입력 착오). 시간대별 선형 AR이 3.75% / 0.951로 LSTM보다 우수하고, walk-forward 4/4 연도 모두 우위다 → `docs/2026-11-06-competitor-benchmark.md`
- **원본 목표 재학습 (2026-11-06 · `docs/2026-11-06-demand-retrain-raw-target.md`):**
  - 클리핑을 없앤 데이터(5665fac0)로 다시 학습했다. 게이트는 persistence·ramp7·시간대별 AR 세 기준선 모두 대비 MAPE·RMSE 우위, DM p<0.05, MAE 1% 이상 개선이다(e120d68e).
  - 직전값 앵커 LSTM: 4.17% / 0.938로 불합격(AR·ramp7보다 나쁨). 서빙 중인 모델(4.16%)과 같은 수준이다.
  - AR 앵커 LSTM: 10개 모델(피처 2종 × 분할 5개) 모두 학습이 붕괴해 "AR + 상수"가 돼 불합격. 원인은 미확정.
  - 합격은 AR + GBM 잔차 보정(`docs/benchmark-2026-11-06/diagnose_ar_residual.py`)뿐이다: 고정 3.63%, walk-forward 평균 3.35%.
  - 운영 모델은 그대로다.
- 합격 기준(2026-11-05 당시): 고정 test와 walk-forward 평균 모두 MAPE·RMSE가 persistence보다 낮을 것 → 두 모델 합격, walk-forward 4/4 fold 모두 우위
- 체크포인트 `model_config.residual_target: true` → `src/inference/predict.py`가 자동으로 래핑. 이 플래그 없이 기본 모델만 쓰면 보정량(≈0)을 수요로 내보내므로 주의
- 재현: `python src/training/retrain_demand_oot.py --residual` · 리포트 `models/production/training_report.json`, `models/demand_retrain_2026-11-05_residual_v2/*/oot_report.json` · 이전 버전 백업 `models/_pre_retrain_backup_2026-11-05/demand_lstm/`

### 2025-12-26 실험 기록 (스케일러 누수 시대 — 수치 무효, 역사 참고용)

| Model | Features | MAPE | R² | Notes |
|-------|----------|------|-----|-------|
| v3.1 (baseline) | 21 | 7.83% | 0.736 | Simple BiLSTM+Attention |
| **v3.2 (Optuna)** | 22 | **7.42%** | **0.760** | **Optuna-optimized (best)** |
| v5.0 (Transformer) | 21 | 8.25% | 0.537 | Rejected (unsuitable) |
| v6.0 (BiLSTM+Weather) | 45 | 7.83% | 0.707 | Added weather features |
| v7.0 (Enhanced) | 45 | 9.62% | 0.57 | Added fuel prices (synthetic data issue) |
| v8.0 (Full Feature) | 109 | 11.84% | -1.48 | Too many features, overfitting |
| v8.1 (SMP+Weather) | 64 | 14.31% | -0.11 | Data loss from feature engineering |

### v3.2 Optuna Hyperparameters (Best Configuration)

```json
{
  "input_hours": 96,
  "hidden_size": 64,
  "num_layers": 1,
  "dropout": 0.198,
  "n_heads": 4,
  "learning_rate": 0.000165,
  "weight_decay": 0.000476,
  "batch_size": 32,
  "noise_std": 0.0099
}
```

**Optuna Tuning Results:**
- 30 trials completed (3h 38m)
- Best trial MAPE: 7.10% (validation)
- Full training MAPE: 7.42%, R²: 0.760
- Improvement: 0.41%p MAPE reduction from v3.1

### Key Findings

1. **Simpler is Better**: v3.1/v3.2 with ~22 features outperforms complex models
2. **Optimal Sequence Length**: 96 hours (4 days) lookback found by Optuna
3. **Model Size**: Hidden size 64 with 1 layer is optimal (not 2)
4. **Noise Injection**: Small noise (0.01) helps regularization
5. **Critical Missing Data**: Power generation data (unavailable via API) is essential

### Bottleneck Analysis

**Why R² 0.9+ is Difficult:**
1. SMP is determined by marginal generator (requires generation data)
2. Renewable output (solar, wind) causes SMP volatility
3. HVDC interconnection affects Jeju-mainland price spread
4. data.go.kr API requires specific subscription for power data

**Best Achievable with Current Data:**
- R² ~0.76 achieved with v3.2 Optuna-optimized model
- MAPE ~7.4% is the current best (validated)
- Further improvement requires power generation data

### Recommended Next Steps

1. **Acquire Power Generation Data** (for R² 0.9+)
   - Apply for data.go.kr API subscription
   - Download from EPSIS manually if needed
   - Focus on: Solar, Wind, Thermal generation

2. **Deploy v3.2 Model** ✅
   - Model saved at `models/smp_v3_optuna/`
   - Hyperparameters documented above
   - Ready for production use

3. **Future Improvements**
   - Ensemble with different architectures
   - Add seasonal/holiday features
   - Real-time adaptation

---

## Docker Deployment (2025-12-26)

### Container Architecture

```
docker-compose.v7.yml
├── api (FastAPI)          → Port 8506  (container rebms-api, 메모리 상한 1GB)
├── web-v6 (Bidding)       → Port 8600
├── web-v7 (ExecO)         → Port 8700
└── mobile (React Native)  → Port 8601
```

- 2026-11-06: api만 배포해 운영 중이다(Colima VM 2 vCPU / 2GB). 시간별 수요 모델(`models/production`, `models/demand_ar_gbm`)과 일일 모델은 읽기 전용 마운트다. 재학습 뒤에는 재빌드 없이 재시작만 하면 된다
- `Dockerfile.api`는 `requirements-docker.txt`(CPU torch, 호스트 검증 버전 고정)를 설치한다. pandas 3은 ar_gbm 예측을 400으로 만든다
- 컨테이너 상태 검사(`/api/v1/health`)는 모델 로드와 무관하게 healthy다. 모델 상태는 `/health`의 `models_loaded`와 `/models`로 확인한다

### Quick Start

```bash
# API만 시작 (현재 운영 방식)
docker compose -f docker/docker-compose.v7.yml up -d --build --wait api

# Start all services
docker compose -f docker/docker-compose.v7.yml up -d --build

# Check status
docker compose -f docker/docker-compose.v7.yml ps

# View logs
docker compose -f docker/docker-compose.v7.yml logs -f api

# Stop services
docker compose -f docker/docker-compose.v7.yml down
```

### Important Docker Notes

1. **tools directory**: Must be included in build context
   - `.dockerignore` only excludes `tools/deprecated/` and `tools/__pycache__/`
   - `Dockerfile.api` includes `COPY tools/ ./tools/`

2. **Rebuild with no-cache** if crawlers fail:
   ```bash
   docker compose -f docker/docker-compose.v7.yml build --no-cache api
   ```

---

## Real-time API Integration (2025-12-26)

### Data Source Priorities

| Data Type | Priority 1 (Primary) | Priority 2 (Backup) |
|-----------|---------------------|---------------------|
| **SMP** | 제주시범사업 API | Web Crawler |
| **Power Supply** | Web Crawler | - |
| **Weather** | KMA API (기상청) | Web Crawler |

### API Endpoints

#### 1. 제주시범사업 SMP API
```
URL: https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2
Method: GET
Parameters:
  - serviceKey: API key (required)
  - pageNo: 1
  - numOfRows: 100
  - dataType: JSON
Response: Real-time Jeju SMP (원/kWh) + 수요예측량 (MW)
```

#### 2. KMA Weather API (기상청 단기예보)
```
URL: https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst
Method: GET
Parameters:
  - serviceKey: API key (required)
  - base_date: YYYYMMDD
  - base_time: HHMM
  - nx: 52 (Jeju)
  - ny: 38 (Jeju)
Response: Temperature, Humidity, Wind Speed, etc.
```

### Crawler Fallbacks

| Crawler | Location | Purpose |
|---------|----------|---------|
| JejuRealtimeCrawler | tools/jeju_realtime_crawler.py | Power supply data |
| KMAWeatherCrawler | tools/kma_weather_crawler.py | Weather backup |

### Current Status (All Connected)

```
✅ SMP: 84.3원/kWh (제주시범사업 API)
✅ Power Supply: 865MW, 73.4% reserve (Crawler)
✅ Weather: 2°C (KMA API)
```

### API Key Configuration

Set in environment or `.env` file:
```bash
DATA_GO_KR_API_KEY=7d42f7c08ba4abd4354d07567d3f6cb0d7478d66cb861e890e6c77a0e3c4d362
```

### Troubleshooting

1. **SMP API Error 830 (KOSPO)**: KOSPO portal discontinued, use 제주시범사업 API instead
2. **Crawler import error**: Rebuild Docker with `--no-cache`
3. **Weather API timeout**: Fallback to crawler automatically
