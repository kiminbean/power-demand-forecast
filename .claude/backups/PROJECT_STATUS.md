# Project Status Backup
> Last Updated: 2025-12-26 07:25 (Current Session - v3.5/v3.6 Generation Data Experiments)

## Project Overview
- **Project**: Jeju Power Demand Forecast System (RE-BMS)
- **Repository**: https://github.com/kiminbean/power-demand-forecast (PRIVATE)
- **Current Version**: v8.0.0 (Weather Map Edition) ✅ Complete
- **Previous Version**: v7.0.0 (React Desktop Web Application)
- **License**: Proprietary (All Rights Reserved)

---

## Current Session (2025-12-26)

### Task: SMP Model R² 0.9+ Improvement with Generation Data

#### Request
- Use crawlers to search and download generation data from 공공데이터포털
- Improve R² beyond 0.760 (v3.2 baseline)
- Incorporate power generation data to improve SMP prediction

#### Progress
| Task | Status |
|------|--------|
| Check existing crawler code | ✅ Complete |
| Search public data portal | ✅ Complete |
| Identify available datasets | ✅ Complete |
| Train v3.3 (Jeju LNG/Oil generation) | ✅ Complete (R² 0.158) - FAILED |
| Train v3.4 (KPX nationwide generation) | ✅ Complete (R² -0.105) - FAILED |
| Train v3.5 (Jeju power trading/demand) | ✅ Complete (R² 0.506) - UNDERPERFORMED |
| Train v3.6 (Solar generation) | ✅ Complete (R² 0.250) - FAILED |

#### Model Comparison

| Model | MAPE | R² | Features | Data Period | Notes |
|-------|------|-----|----------|-------------|-------|
| v3.2 (Optuna) | 7.42% | **0.760** | 22 | 5 years SMP only | **BASELINE** |
| v3.3 (LNG/Oil) | 17.34% | 0.158 | 37 | ~11 months overlap | Data period too short |
| v3.4 (KPX gen) | 17.58% | -0.105 | 53 | ~1 year overlap | Data period too short |
| v3.5 (Power demand) | 11.09% | 0.506 | 60 | ~4 years overlap | Demand ≠ Generation |
| v3.6 (Solar gen) | 23.82% | 0.250 | 73 | ~3.5 years overlap | Solar-SMP corr=-0.106 |

#### Key Findings

1. **Data Period Matters**: Models with short overlap periods (v3.3, v3.4) performed poorly
2. **Demand ≠ Generation**: Power trading volume (수요) is NOT the same as generation (발전량)
   - SMP is determined by marginal generator costs, not directly by demand
3. **Zero SMP Problem**: 141 records have smp_jeju = 0 (during high solar generation)
   - Fixed MAPE calculation to exclude SMP < 10 won/kWh
4. **Solar Generation is Key**: High solar → Low SMP (can be 0)
   - v3.6 uses solar generation data which should be more relevant

#### Available Data Files in data/raw/

```
# Power Demand (NOT generation)
jeju_hourly_power_2013_2024.csv        - 전력거래량 (2013-2024, 105K records)

# Generation Data
제주 시간대별 발전량(LNG)_240331.csv    - LNG generation (2023.05-2024.03)
제주 시간대별 발전량(유류)_240331.csv   - Oil generation (2023.05-2024.03)
한국전력거래소_시간별 발전량_20211231.csv - Nationwide hourly (2017-2021)

# Solar Generation + Weather (RECOMMENDED for v3.6)
한국동서발전_제주_기상관측_태양광발전.csv - Solar gen with weather (2018-2024, 56K records)
  - Columns: 일시, 기온, 습도, 일사량, 태양광 발전량(MWh) 등
  - Overlap with SMP: ~3.5 years (2020-12 ~ 2024-05)
```

#### Files Created This Session

```
src/crawlers/public_data_crawler.py    - Search data.go.kr
src/crawlers/download_public_data.py   - Download from data.go.kr
src/smp/models/train_smp_v3_3_generation.py  - LNG/Oil generation model
src/smp/models/train_smp_v3_4_kpx_generation.py - KPX nationwide model
src/smp/models/train_smp_v3_5_jeju_power.py  - Power demand model
src/smp/models/train_smp_v3_6_solar.py       - Solar generation model (READY)
models/smp_v3_5_jeju_power/metrics.json      - v3.5 results
```

#### Code Fixes Applied

1. **MAPE Zero Division Fix** (in v3.5, v3.6):
```python
# Filter out near-zero actuals for MAPE
mask = actuals > 10  # Only consider SMP > 10 won/kWh for MAPE
mape = mean_absolute_percentage_error(actuals[mask], preds[mask]) * 100
```

2. **SMP 24:00 Timestamp Fix**:
```python
df['timestamp'] = df['timestamp'].str.replace(' 24:00', ' 00:00')
mask = df['hour'] == 24
df.loc[mask, 'datetime'] = df.loc[mask, 'datetime'] + pd.Timedelta(days=1)
```

#### Next Steps (Pending)

1. **Run v3.6** (Solar generation model):
```bash
source .venv/bin/activate && python src/smp/models/train_smp_v3_6_solar.py
```

2. **If v3.6 still underperforms**:
   - Try combining solar + power demand features
   - Use Optuna tuning on v3.6 model
   - Download additional generation data from 공공데이터포털

3. **For R² 0.9+**:
   - Need actual Jeju generation data (not just demand)
   - Consider real-time fuel prices (LNG, oil)
   - Apply for data.go.kr API subscription

---

## Previous Session (2025-12-26 06:00)

### Task: SMP Model v3.2 Optuna Tuning ✅ Complete

#### v3.2 Optuna Results (CURRENT BEST)
| Metric | v3.1 Baseline | v3.2 Optuna | Improvement |
|--------|---------------|-------------|-------------|
| MAPE | 7.83% | **7.42%** | -0.41%p |
| R² | 0.736 | **0.760** | +0.024 |

**Best Hyperparameters:**
```json
{
  "input_hours": 96,
  "hidden_size": 64,
  "num_layers": 1,
  "dropout": 0.198,
  "n_heads": 4,
  "learning_rate": 0.000165,
  "batch_size": 32,
  "noise_std": 0.0099
}
```

**Model saved at:** `models/smp_v3_optuna/`

---

## Previous Sessions

### 2025-12-23: Create web-v8 Dashboard ✅ Complete
### 2025-12-22: System Architecture Documentation ✅ Complete
### 2025-12-20: RE-BMS v6.0.0 Release ✅ Complete

---

## Key Files

### SMP Models
```
models/smp_v3_optuna/           - v3.2 Optuna (BEST: R² 0.760)
models/smp_v3_5_jeju_power/     - v3.5 Power demand (R² 0.506)
src/smp/models/train_smp_v3_6_solar.py  - v3.6 Solar (READY)
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
3. To continue v3.6 solar model:
```bash
source .venv/bin/activate && python src/smp/models/train_smp_v3_6_solar.py
```
4. v3.2 Optuna (R² 0.760) remains the best model until v3.6 is tested
