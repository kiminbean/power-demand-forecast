# Project Status Backup
> Last Updated: 2025-12-19 13:20 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.7 (Chart Pattern Bug Fix)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.4

---

## Latest Changes (2025-12-19)

### Bug Fix: Chart Spike at 1 PM (v4.0.7)
Fixed sudden spike in power chart around 1 PM caused by model divergence.

| Issue | Resolution |
|-------|------------|
| Root Cause | Two different pattern formulas (Sine vs Piecewise Linear) |
| 13:00 Discrepancy | Sine=1.24 vs Linear=0.88 (41% difference) |
| Result | Sudden vertical spike at current time |

**Fix**: Created unified `get_load_pattern(hour)` function (Single Source of Truth)
- Past data generation: uses `get_load_pattern()`
- Current pattern calc: uses `get_load_pattern()`
- Future forecast: uses `get_load_pattern()`

**Verified by**: Gemini cross-check analysis confirmed the bug diagnosis

### Bug Fix: Reserve Rate Display (v4.0.6)
Fixed reserve rate showing 911% instead of correct ~132-152%.

| Issue | Resolution |
|-------|------------|
| Wrong field | Used `supply_reserve` (MW) instead of `reserve_rate` (%) |
| Value shown | 911% (MW value) |
| Correct value | ~132-152% (calculated %) |

### Dashboard Layout (GE Inertia Style)
New layout with real-time power supply chart and forecast comparison.

### Layout Structure
| Position | Content |
|----------|---------|
| Left (3/4) | Real-time power chart (actual vs forecast) |
| Right Top | Gauge charts (demand, reserve rate) |
| Right Bottom | Jeju map (compact) |

### Chart Features
- 12h past (actual data) + 12h future (forecast)
- Confidence band for forecast uncertainty
- Red dashed line marks current time (center)
- Color coding: Blue (actual), Yellow (forecast)

### Recent Commits
```
4e29b71 fix: Unify load pattern function to eliminate chart spike bug
188184b docs: Update PROJECT_STATUS for v4.0.6 reserve rate fix
94f5afb fix: Correct reserve rate display using percentage instead of MW value
afd6d85 feat: Redesign dashboard layout with real-time power chart
```

---

## v4.0.4 Release (2025-12-19)

### New Feature: Slack Webhook Integration
Dashboard sends Slack notifications for all alert levels (critical, danger, warning).

### Slack Configuration
```bash
SLACK_ALERTS_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#alerts
```

### Features
| Feature | Description |
|---------|-------------|
| All Alert Levels | Critical, Danger, Warning |
| Rich Formatting | Slack Block Kit with colors |
| Color Coding | Red/Orange/Yellow by severity |
| Rate Limiting | 5-min cooldown per alert type |
| Message Logging | `data/alerts/slack_log.json` |

### Notification Comparison
| Channel | Alert Levels | Trigger |
|---------|--------------|---------|
| Email | Critical only | < 5% |
| Slack | All levels | < 15% |

### Recent Commits
```
bd19513 feat: Add Slack webhook notifications for alerts
f29270c docs: Update PROJECT_STATUS for v4.0.3 release
cf5bf9b feat: Add email notification for critical alerts
e3623bb docs: Update PROJECT_STATUS with alert history feature
812311a docs: Add alert history sidebar screenshot
```

---

## Test Results (2025-12-19)

```
1564 passed, 3 skipped, 19 warnings
```

| Status | Count |
|--------|-------|
| Passed | 1,564 |
| Skipped | 3 |
| Warnings | 19 |

### Test Breakdown
- Slack notifier tests: 17 (NEW)
- Email notifier tests: 19
- Alert threshold tests: 28
- Alert history tests: 12
- Other tests: 1,488

---

## SMP Model v3.1

### Performance Results
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAPE | 7.83% | <10% | Pass |
| R2 | 0.74 | >0.65 | Pass |
| Coverage | 89.4% | >85% | Pass |

---

## Dashboard v4.0.5

### Features
1. 60hz.io style dark theme
2. **GE Inertia style layout - NEW**
3. **Real-time power chart (actual vs forecast) - NEW**
4. **Gauge charts (demand, reserve rate) - NEW**
5. Interactive Jeju map with power plants
6. KPX realtime data integration
7. Reserve rate alert system (KPX thresholds)
8. Test mode for alert simulation
9. Alert history sidebar
10. Email notification (critical alerts)
11. Slack webhook (all alerts)
12. SMP prediction with v3.1 model
13. 24-hour forecast with confidence intervals
14. XAI analysis tab

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

---

## Key Files

```
# Dashboard v4.0.4
src/dashboard/app_v4.py              - Main dashboard (Slack + Email + Alerts)

# Alert System
data/alerts/alert_history.json       - Alert history
data/alerts/email_log.json           - Email send log
data/alerts/slack_log.json           - Slack send log (NEW)

# Configuration
.env.example                         - All settings (Email + Slack)

# SMP Model
models/smp_v3/smp_v3_model.pt        - Trained model
models/smp_v3/smp_v3_metrics.json    - Performance metrics
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v4.0.7 | 2025-12-19 | Chart pattern spike fix (Gemini verified) |
| v4.0.6 | 2025-12-19 | Reserve rate bug fix (911% → 132%) |
| v4.0.5 | 2025-12-19 | GE Inertia layout, real-time chart |
| v4.0.4 | 2025-12-19 | Slack webhook notifications |
| v4.0.3 | 2025-12-19 | Email notification (critical) |
| v4.0.2 | 2025-12-19 | Reserve rate alert system |
| v4.0.1 | 2025-12-19 | KPX realtime integration |
| v4.0.0 | 2025-12-18 | SMP v3.1 + Dashboard v4 |

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Run `git log --oneline -10`
3. View release: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.4

---

## Environment
- Python 3.13, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)

## Notes
- v4.0.4 adds Slack webhook for all alert levels
- Email only sends for critical alerts (< 5%)
- Slack sends for all alerts (< 15%)
- All 1,564 tests passing
