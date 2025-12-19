# Project Status Backup
> Last Updated: 2025-12-19 10:30 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.3 (Email Notification for Critical Alerts)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.3

---

## v4.0.3 Release (2025-12-19)

### New Feature: Email Notification System
Dashboard sends email alerts when reserve rate drops below 5% (critical level).

### Email Configuration
```bash
EMAIL_ALERTS_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_RECIPIENT_EMAILS=admin1@example.com,admin2@example.com
```

### Features
| Feature | Description |
|---------|-------------|
| SMTP Support | Gmail with TLS encryption |
| HTML Emails | Formatted alert with power data table |
| Rate Limiting | Max 1 email per 5 minutes (prevents spam) |
| Email Logging | Audit trail in `data/alerts/email_log.json` |
| Alert History | Sidebar with statistics and recent alerts |

### Recent Commits
```
cf5bf9b feat: Add email notification for critical alerts
e3623bb docs: Update PROJECT_STATUS with alert history feature
812311a docs: Add alert history sidebar screenshot
ec16077 feat: Add alert history log feature
c8fd811 test: Add reserve rate alert threshold tests
32a27ab docs: Update PROJECT_STATUS backup for v4.0.2
05565a0 docs: Add v4.0.2 release notes to CHANGELOG
9efa658 docs: Add reserve rate alert screenshots to README
ad5ebcc docs: Add alert level screenshots for reserve rate system
33b7b07 feat: Add sidebar test mode for reserve rate alerts
```

---

## Test Results (2025-12-19)

```
1547 passed, 3 skipped, 19 warnings
```

| Status | Count |
|--------|-------|
| Passed | 1,547 |
| Skipped | 3 |
| Warnings | 19 (deprecation only) |

### Test Breakdown
- Email notifier tests: 19 (NEW)
- Alert threshold tests: 28
- Alert history tests: 12
- Other tests: 1,488

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

## Dashboard v4.0.3

### Features
1. 60hz.io style dark theme
2. Interactive Jeju map with power plants (Folium)
3. KPX realtime data integration
4. Reserve rate alert system (KPX thresholds)
5. Test mode for alert simulation
6. Alert history sidebar
7. **Email notification for critical alerts (NEW)**
8. SMP prediction with v3.1 model
9. 24-hour forecast with confidence intervals
10. XAI analysis tab (attention weights)

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

---

## Key Files

```
# SMP Model v3.1
src/smp/models/train_smp_v3_fixed.py  - Training pipeline
models/smp_v3/smp_v3_model.pt         - Trained model (250K params)
models/smp_v3/smp_v3_metrics.json     - Performance metrics

# Dashboard v4.0.3
src/dashboard/app_v4.py              - Main dashboard (email + alerts)
src/smp/models/smp_predictor.py      - Prediction interface

# Alert System
data/alerts/alert_history.json       - Alert history data
data/alerts/email_log.json           - Email send log (NEW)

# Configuration
.env.example                         - Environment config (email settings)

# Screenshots
docs/screenshots/01_main_dashboard.png
docs/screenshots/02_smp_prediction.png
docs/screenshots/07_alert_caution.png
docs/screenshots/08_alert_warning.png
docs/screenshots/09_alert_critical.png
docs/screenshots/10_alert_history_sidebar.png
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v4.0.3 | 2025-12-19 | Email notification for critical alerts |
| v4.0.2 | 2025-12-19 | Reserve rate alert system + Alert history |
| v4.0.1 | 2025-12-19 | KPX realtime integration |
| v4.0.0 | 2025-12-18 | SMP v3.1 + Dashboard v4 |

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Check `models/smp_v3/smp_v3_metrics.json`
3. Run `git log --oneline -10`
4. View release: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.3

---

## Environment
- Python 3.13, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- EPSIS real data: 2020-12-19 ~ 2025-12-18

## Notes
- v4.0.3 adds email notification for critical alerts
- Email requires Gmail App Password (2FA enabled)
- Rate limiting prevents email spam (5-min cooldown)
- All 1,547 tests passing
