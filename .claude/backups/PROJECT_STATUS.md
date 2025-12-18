# Project Status Backup
> Last Updated: 2025-12-18 23:30 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.0 (Released)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.0

---

## v4.0.0 Release Complete (2025-12-18)

### Release Summary

**Completed Tasks:**
- ✅ SMP Model v3.1 training (MAPE 7.83%, R² 0.74)
- ✅ Dashboard v4 with 60hz.io dark theme
- ✅ Screenshots added to README
- ✅ CHANGELOG updated with v4.0.0 release notes
- ✅ GitHub release tag v4.0.0 created

### Recent Commits
```
7efa70d docs: Add v4.0.0 release notes to CHANGELOG
a3300e8 docs: Add screenshots to README
7c15dbf feat: Add SMP v3.1 model and Dashboard v4.0
f7f9e9c docs: Add Korean text bug workaround for Claude Code v2.0.72
f42affc fix: Convert attention list to numpy array in XAI page
```

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

### v3.0 Failure Root Cause
- `torch.clamp(pred, min=-10, max=10)` disrupted gradient flow
- Complex loss function with disabled components
- Fixed in v3.1 with simplified architecture

---

## Dashboard v4.0

### Features
1. 60hz.io style dark theme
2. Interactive Jeju map with power plants (Folium)
3. Real-time EPSIS data integration
4. SMP prediction with v3.1 model
5. 24-hour forecast with confidence intervals
6. XAI analysis tab

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

### Screenshots
- `docs/screenshots/01_main_dashboard.png`
- `docs/screenshots/02_smp_prediction.png`
- `docs/screenshots/05_system_architecture.png`

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
- Python 3.11+, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- EPSIS real data: 2022-01-01 ~ 2024-12-31

## Notes
- v4.0.0 is a major release with all target metrics achieved
- Dashboard v4 integrates v3.1 model for production use
- All documentation updated (README, CHANGELOG, screenshots)
