# Project Status Backup
> Last Updated: 2025-12-19 10:00 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.2 (Reserve Rate Alert System)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.2

---

## v4.0.2 Release (2025-12-19)

### New Feature: Reserve Rate Alert System
Dashboard now displays visual alerts based on KPX standard reserve rate thresholds.

### KPX Standard Thresholds
| Reserve Rate | Status | Alert Level |
|--------------|--------|-------------|
| ≥15% | Normal | None |
| 10-15% | 관심 (Caution) | 🟡 Yellow banner |
| 5-10% | 주의 (Warning) | 🟠 Orange banner |
| <5% | 위험 (Critical) | 🔴 Red pulsing banner |

### Features Added
- **Alert Banners**: Full-width banners with icons and severity colors
- **CSS Animations**: Pulsing effect for critical alerts
- **Reserve Badge**: Dynamic color-coded badge in metrics card
- **Test Mode**: Sidebar toggle to simulate low reserve rates
- **Reserve Slider**: Adjust test reserve rate (0-30%)

### Screenshots
- `docs/screenshots/07_alert_caution.png` - Caution at 12%
- `docs/screenshots/08_alert_warning.png` - Warning at 7%
- `docs/screenshots/09_alert_critical.png` - Critical at 3%

### Recent Commits
```
05565a0 docs: Add v4.0.2 release notes to CHANGELOG
9efa658 docs: Add reserve rate alert screenshots to README
ad5ebcc docs: Add alert level screenshots for reserve rate system
33b7b07 feat: Add sidebar test mode for reserve rate alerts
bff880e feat: Add reserve rate alert system with KPX thresholds
```

---

## Test Results (2025-12-19)

```
1488 passed, 3 skipped, 16 warnings in 24.92s
```

| Status | Count |
|--------|-------|
| Passed | 1,488 |
| Skipped | 3 |
| Warnings | 16 (deprecation only) |

---

## KPX Realtime Integration (v4.0.1)

### Live Power Data
Dashboard shows **real-time power supply/demand** from KPX (한국전력거래소):

| Data Item | Source | Update Interval |
|-----------|--------|-----------------|
| Current Demand | KPX 실시간 | 5 minutes |
| Supply Capacity | KPX 실시간 | 5 minutes |
| Reserve Rate | KPX 실시간 | 5 minutes |
| Plant Generation | Distributed from KPX total | 60 seconds cache |

### Data Priority
1. **KPX 실시간** (Primary) - Live from https://www.kpx.or.kr
2. **EPSIS 파일** (Secondary) - Historical file data
3. **시뮬레이션** (Fallback) - Simulated values

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

---

## Dashboard v4.0.2

### Features
1. 60hz.io style dark theme
2. Interactive Jeju map with power plants (Folium)
3. 🔴 KPX realtime data integration
4. 🚨 Reserve rate alert system (NEW)
5. 🧪 Test mode for alert simulation (NEW)
6. SMP prediction with v3.1 model
7. 24-hour forecast with confidence intervals
8. XAI analysis tab (attention weights)

### Data Source Indicators
- 🔴 **KPX 실시간 연동** - Live KPX data active
- 📊 **EPSIS 데이터 연동** - Using EPSIS files
- ⚠️ **시뮬레이션 모드** - Fallback simulation

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

### Verified Tabs
- ✅ Main Dashboard (Jeju map, KPX realtime, alerts)
- ✅ SMP Prediction (24h forecast, confidence bands)
- ✅ XAI Analysis (attention heatmap)

---

## Key Files (v4.0.2)

```
# SMP Model v3.1
src/smp/models/train_smp_v3_fixed.py  - Training pipeline
models/smp_v3/smp_v3_model.pt         - Trained model (250K params)
models/smp_v3/smp_v3_metrics.json     - Performance metrics
models/smp_v3/smp_v3_scaler.npy       - Feature scaler

# Dashboard v4.0.2
src/dashboard/app_v4.py              - Main dashboard (alerts + KPX)
src/smp/models/smp_predictor.py      - Prediction interface (v3.1 support)

# Crawlers
tools/crawlers/jeju_realtime_crawler.py  - KPX realtime data crawler
src/smp/crawlers/epsis_crawler.py        - EPSIS SMP crawler

# Data
data/smp/smp_5years_epsis.csv        - 5 years EPSIS data (26,240 records)
data/jeju_plants/jeju_power_plants.json  - Plant locations (Dec 2025)

# Screenshots
docs/screenshots/01_main_dashboard.png   - Main dashboard
docs/screenshots/02_smp_prediction.png   - SMP prediction
docs/screenshots/04_kpx_realtime.png     - KPX realtime
docs/screenshots/07_alert_caution.png    - Alert caution (12%)
docs/screenshots/08_alert_warning.png    - Alert warning (7%)
docs/screenshots/09_alert_critical.png   - Alert critical (3%)

# Documentation
CHANGELOG.md                          - Version history
README.md                             - Project overview (with screenshots)
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v4.0.2 | 2025-12-19 | Reserve rate alert system |
| v4.0.1 | 2025-12-19 | KPX realtime integration |
| v4.0.0 | 2025-12-18 | SMP v3.1 + Dashboard v4 |
| v3.1 | 2025-12-17 | Improved SMP model |

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Check `models/smp_v3/smp_v3_metrics.json`
3. Run `git log --oneline -10`
4. View release: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.2

---

## Environment
- Python 3.13, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- EPSIS real data: 2020-12-19 ~ 2025-12-18

## Notes
- v4.0.2 includes reserve rate alert system
- v4.0.1 includes KPX realtime integration
- v4.0.0 release includes all bug fixes
- All 1,488 tests passing
- Dashboard fully functional with live data and alerts
- Power plant data updated to Dec 2025
