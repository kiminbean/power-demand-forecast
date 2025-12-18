# Project Status Backup
> Last Updated: 2025-12-18 23:30 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.0 (Released + Bug Fixes)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.0

---

## v4.0.0 Release Status (2025-12-18)

### Test Results
```
1488 passed, 3 skipped, 16 warnings in 25.30s
```

| Status | Count |
|--------|-------|
| Passed | 1,488 |
| Skipped | 3 |
| Warnings | 16 (deprecation only) |

### Recent Commits
```
2da33d4 fix: Replace deprecated use_container_width with width="stretch"
69d2484 fix: Handle 24:00 timestamp format in SMP data loading
0906704 fix: Replace float.clip() with min/max for utilization calc
c3f0f82 docs: Update PROJECT_STATUS with v4.0.0 release info
7efa70d docs: Add v4.0.0 release notes to CHANGELOG
```

### Bug Fixes Applied
1. **float.clip() Error** - Changed to `min(max(...))` for utilization calculation
2. **24:00 Timestamp** - Added `_fix_timestamp_24h()` helper for SMP data loading
3. **Streamlit Deprecation** - Replaced `use_container_width=True` with `width="stretch"`

---

## SMP Model v3.1

### Performance Results
| Metric | v2.1 (Previous) | v3.1 (Current) | Target | Status |
|--------|-----------------|----------------|--------|--------|
| MAPE | 10.68% | **7.83%** | <10% | ✅ |
| R² | 0.59 | **0.74** | >0.65 | ✅ |
| Coverage | 82.5% | **89.4%** | >85% | ✅ |
| MAE | 11.27 | **8.93** | - | ✅ |
| RMSE | 14.67 | **12.02** | - | ✅ |

### Architecture
- BiLSTM + Stable Attention (4 heads)
- 22 features, 249,952 parameters
- Noise Injection (std=0.02, prob=0.5)
- Quantile outputs: Q10, Q50, Q90
- Huber + Quantile Loss

### Prediction Test Results
```
Average SMP: 135.07 won/kWh
Min SMP: 103.08 won/kWh (01:00)
Max SMP: 161.40 won/kWh (08:00)
Confidence Interval: ±17.9 won/kWh
```

---

## Dashboard v4.0

### Features
1. 60hz.io style dark theme
2. Interactive Jeju map with power plants (Folium)
3. Real-time EPSIS data integration
4. SMP prediction with v3.1 model
5. 24-hour forecast with confidence intervals
6. XAI analysis tab (attention weights)

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

### Verified Tabs
- ✅ Main Dashboard (Jeju map, power plants)
- ✅ SMP Prediction (24h forecast, confidence bands)
- ✅ XAI Analysis (attention heatmap)

---

## Key Files (v4.0.0)

```
# SMP Model v3.1
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

# Documentation
docs/screenshots/                     - Dashboard screenshots
CHANGELOG.md                          - Version history
README.md                             - Project overview (with screenshots)
```

---

## Model Version History

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
4. View release: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.0

---

## Environment
- Python 3.13, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- EPSIS real data: 2020-12-19 ~ 2025-12-18

## Notes
- v4.0.0 release includes all bug fixes
- All 1,488 tests passing
- Dashboard fully functional (no errors/warnings)
- Tag updated to include bug fix commits
