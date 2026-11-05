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

**Model Performance (v3.2 Optuna - Best SMP Model)**:
- MAPE: 7.42%
- R²: 0.760

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

**Last Updated**: 2026-11-05 (SMP 스케일러 누수 수정·재학습 이후 재평가)

### Current Status

| Model Version | MAPE | R² | Status |
|---------------|------|-----|--------|
| v1.0~v8.1 (구버전 전체) | 무효 | 무효 | 스케일러 누수(시간 분할 전 fit)로 평가 자체가 무효 — 아래 2025-12-26 표는 역사 기록 |
| **v4.0 (2026-11-05 재학습)** | 17.17% (test) | 0.124 | 시간외 평가·신뢰 가능하나 성능 미달 |
| **v4.1 (2026-11-05 재학습, 현재 생산)** | **8.76% (walk-forward 평균)** | **0.463** | 구버전 수치와 직접 비교 불가(평가 방식 상이). MAE 10.53원/kWh·커버리지 68.6% |
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
| **v4.1** | **walk-forward 평균** | **8.76%** | **0.463** | MAE 10.53원/kWh·커버리지 68.6% — 현재 생산 |
| v3.4 | test(시간외) | 21.28% | -0.537 | 사용 보류(일반화 실패) |

원본: `models/smp_v4/smp_v4_metrics.json`, `models/smp_v4/smp_v41_metrics.json`, `models/smp_v3_4_kpx/metrics.json`. 경위·누수 버전 백업(`models/_pre_retrain_backup_2026-11-05/`): `docs/2026-11-02-forecast-upgrade-status.md`.

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
├── api (FastAPI)          → Port 8000
├── web-v6 (Bidding)       → Port 8600
├── web-v7 (ExecO)         → Port 8700
└── mobile (React Native)  → Port 3001
```

### Quick Start

```bash
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
