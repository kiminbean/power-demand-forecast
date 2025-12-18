# Project Status Backup
> Last Updated: 2025-12-18 22:50 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.0

---

## v4.0.0 SMP Model v3.1 + Dashboard v4 (2025-12-18)

### Session Summary (2025-12-18 22:50)

#### v3.1 SMP Model - Training Success

**v3.0 Failure Analysis:**
- MAPE: 100%, R²: -28.5 (complete failure)
- Cause: `torch.clamp(pred, min=-10, max=10)` disrupted gradient flow
- Cause: Complex loss function with disabled components

**v3.1 Fix Applied:**
1. Removed incorrect clamping logic
2. Simplified loss function (Huber + Quantile)
3. Used StandardScaler instead of custom normalization
4. Stable Attention mechanism

**v3.1 Final Results:**
| Metric | v2.1 (Previous) | v3.1 (Current) | Target | Status |
|--------|-----------------|----------------|--------|--------|
| MAPE | 10.68% | **7.83%** | <10% | ✅ |
| R² | 0.59 | **0.74** | >0.65 | ✅ |
| Coverage | 82.5% | **89.4%** | >85% | ✅ |
| MAE | 11.27 | **8.93** | - | ✅ |
| RMSE | 14.67 | **12.02** | - | ✅ |

**Model Architecture:**
- BiLSTM + Stable Attention (4 heads)
- 22 features, 249,952 parameters
- Noise Injection (std=0.02, prob=0.5)
- Quantile outputs: Q10, Q50, Q90

---

### Dashboard v4.0

**Features:**
1. 60hz.io style dark theme
2. Interactive Jeju map with power plants
3. Real-time EPSIS data integration
4. SMP prediction with v3.1 model
5. 24-hour forecast with confidence intervals
6. XAI analysis tab

**Run Command:**
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

---

### Key Files (v4.0.0)

```
# v3.1 SMP Model
src/smp/models/train_smp_v3_fixed.py  - Training pipeline
models/smp_v3/smp_v3_model.pt         - Trained model (250K params)
models/smp_v3/smp_v3_metrics.json     - Performance metrics
models/smp_v3/smp_v3_scaler.npy       - Feature scaler

# Dashboard v4
src/dashboard/app_v4.py              - Main dashboard
src/smp/models/smp_predictor.py      - Prediction interface (v3.1 support)

# Data
data/smp/smp_5years_epsis.csv        - 5 years EPSIS data (26,240 records)
data/jeju_plants/jeju_power_plants.json  - Plant locations
```

---

### Model Comparison

| Version | MAPE | R² | Coverage | Parameters | Status |
|---------|------|-----|----------|------------|--------|
| v1.0 (Basic LSTM) | 6.32% | 0.85 | N/A | ~500K | Synthetic data |
| v2.0 (Synthetic) | 2.89% | 0.82 | N/A | 1M | Overfitted |
| v2.1 (Advanced) | 10.68% | 0.59 | 82.5% | 172K | Real data |
| **v3.1 (Current)** | **7.83%** | **0.74** | **89.4%** | **250K** | **Production** |

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Check `models/smp_v3/smp_v3_metrics.json`
3. Run `git log --oneline -10`

---

## Notes
- Python 3.11+, PyTorch 2.0+, MPS (Apple Silicon)
- v3.1 achieves all target metrics
- Dashboard v4 integrates v3.1 model
- EPSIS real data: 2022-01-01 ~ 2024-12-31
