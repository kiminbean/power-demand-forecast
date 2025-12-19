# Project Status Backup
> Last Updated: 2025-12-19 10:15 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.2 (Reserve Rate Alert System + Alert History)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.2

---

## Latest Updates (2025-12-19)

### Alert History Feature (NEW)
- **AlertHistory class** with JSON persistence
- Stores up to 100 recent alerts
- Duplicate prevention (same status within 1 minute)
- Sidebar display with statistics and recent alerts

### Alert History Sidebar
| Feature | Description |
|---------|-------------|
| Statistics | Count of critical/warning/caution alerts |
| Recent Alerts | Last 10 alerts with timestamps |
| Persistence | JSON file at `data/alerts/alert_history.json` |

### Recent Commits
```
812311a docs: Add alert history sidebar screenshot
ec16077 feat: Add alert history log feature
c8fd811 test: Add reserve rate alert threshold tests
32a27ab docs: Update PROJECT_STATUS backup for v4.0.2
05565a0 docs: Add v4.0.2 release notes to CHANGELOG
9efa658 docs: Add reserve rate alert screenshots to README
ad5ebcc docs: Add alert level screenshots for reserve rate system
33b7b07 feat: Add sidebar test mode for reserve rate alerts
bff880e feat: Add reserve rate alert system with KPX thresholds
```

---

## Test Results (2025-12-19)

```
1528 passed, 3 skipped, 16 warnings
```

| Status | Count |
|--------|-------|
| Passed | 1,528 |
| Skipped | 3 |
| Warnings | 16 (deprecation only) |

### Test Breakdown
- Alert threshold tests: 28
- Alert history tests: 12
- Other tests: 1,488

---

## v4.0.2 Release (2025-12-19)

### Reserve Rate Alert System
Dashboard displays visual alerts based on KPX standard reserve rate thresholds.

### KPX Standard Thresholds
| Reserve Rate | Status | Alert Level |
|--------------|--------|-------------|
| >=15% | Normal | None |
| 10-15% | Caution | Yellow banner |
| 5-10% | Warning | Orange banner |
| <5% | Critical | Red pulsing banner |

### Features
- **Alert Banners**: Full-width banners with icons and severity colors
- **CSS Animations**: Pulsing effect for critical alerts
- **Reserve Badge**: Dynamic color-coded badge in metrics card
- **Test Mode**: Sidebar toggle to simulate low reserve rates
- **Reserve Slider**: Adjust test reserve rate (0-30%)
- **Alert History**: Sidebar with statistics and recent alerts (NEW)

---

## SMP Model v3.1

### Performance Results
| Metric | v2.1 (Previous) | v3.1 (Current) | Target | Status |
|--------|-----------------|----------------|--------|--------|
| MAPE | 10.68% | **7.83%** | <10% | Pass |
| R2 | 0.59 | **0.74** | >0.65 | Pass |
| Coverage | 82.5% | **89.4%** | >85% | Pass |
| MAE | 11.27 | **8.93** | - | Pass |
| RMSE | 14.67 | **12.02** | - | Pass |

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
3. KPX realtime data integration
4. Reserve rate alert system
5. Test mode for alert simulation
6. Alert history sidebar (NEW)
7. SMP prediction with v3.1 model
8. 24-hour forecast with confidence intervals
9. XAI analysis tab (attention weights)

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

### Verified Tabs
- Main Dashboard (Jeju map, KPX realtime, alerts)
- SMP Prediction (24h forecast, confidence bands)
- XAI Analysis (attention heatmap)

---

## Key Files

```
# SMP Model v3.1
src/smp/models/train_smp_v3_fixed.py  - Training pipeline
models/smp_v3/smp_v3_model.pt         - Trained model (250K params)
models/smp_v3/smp_v3_metrics.json     - Performance metrics
models/smp_v3/smp_v3_scaler.npy       - Feature scaler

# Dashboard v4.0.2
src/dashboard/app_v4.py              - Main dashboard (alerts + history)
src/smp/models/smp_predictor.py      - Prediction interface (v3.1 support)

# Alert History
data/alerts/alert_history.json       - Alert history data (NEW)

# Screenshots
docs/screenshots/01_main_dashboard.png   - Main dashboard
docs/screenshots/02_smp_prediction.png   - SMP prediction
docs/screenshots/04_kpx_realtime.png     - KPX realtime
docs/screenshots/07_alert_caution.png    - Alert caution (12%)
docs/screenshots/08_alert_warning.png    - Alert warning (7%)
docs/screenshots/09_alert_critical.png   - Alert critical (3%)
docs/screenshots/10_alert_history_sidebar.png - Alert history sidebar (NEW)

# Documentation
CHANGELOG.md                          - Version history
README.md                             - Project overview (with screenshots)
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v4.0.2 | 2025-12-19 | Reserve rate alert system + Alert history |
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
- v4.0.2 includes reserve rate alert system + alert history
- All 1,528 tests passing
- Dashboard fully functional with live data and alerts
- Alert history persists to JSON file
