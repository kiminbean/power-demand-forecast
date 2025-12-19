# Project Status Backup
> Last Updated: 2025-12-19 14:45 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System
- **Repository**: https://github.com/kiminbean/power-demand-forecast
- **Version**: v4.0.7 (Enhanced Chart Visualization)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.7

---

## Latest Changes (2025-12-19)

### New Feature: Enhanced Power Chart (v4.0.7)

#### 예비전력 시각화
| 항목 | 설명 |
|------|------|
| 표시 방식 | 공급능력-수요 사이 초록색 음영 |
| 실측 | 진한 초록색 (rgba 0.25) |
| 예측 | 연한 초록색 (rgba 0.15) |
| 호버 | 예비전력 MW 값 표시 |

#### 태양광 + 풍력 발전량 표시
| 발전원 | 색상 | 패턴 |
|--------|------|------|
| 태양광 | 노란색 | 6시~18시 사인파, 피크 150 MW |
| 풍력 | 파란색 | 100~160 MW 변동 |

#### 차트 구조
```
공급능력 ━━━━━━━━━━━━━━━━━━━ (초록선)
         ░░░░░░░░░░░░░░░░░░░  ← 예비전력 (초록 음영)
전력수요 ━━━━━━━━━━━━━━━━━━━ (파란/노란선)

         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 풍력 (파란 음영)
         ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ← 태양광 (노란 음영)
```

### New: Technical Documentation (v4.0.7)
- `docs/TECHNICAL_DOCUMENTATION.md` - 558줄 상세 기술 설명서
- 모델 아키텍처, 피처 엔지니어링, 성능 평가 포함
- THI, HDD, CDD 상세 설명

### Documentation: README Update (v4.0.7)
Updated README.md with v4.0.7 bug fixes and documentation.

| Update | Details |
|--------|---------|
| Test Badge | 1436 → 1564 passed |
| Dashboard Section | Added v4.0.7 features table |
| Run Command | Added v4 dashboard command |
| Project Structure | Added `app_v4.py` reference |
| New Section | v4.0.7 버그 수정 with code examples |
| Version | Updated to v4.0.7 |

### Bug Fix: SMP Zero Value (v4.0.7)
Fixed SMP showing 0.0원 instead of actual price (~141.5원).

| Issue | Resolution |
|-------|------------|
| Root Cause | `smp_jeju=0.0` in EPSIS data (2024-12-18 13:00) |
| Fallback | Use `smp_mainland` when `smp_jeju` is 0 |
| Result | 0.0원 → 141.5원 |

### Streamlit Deprecation Fix (v4.0.7)
Fixed `use_container_width` deprecation warnings.

| Before | After |
|--------|-------|
| `use_container_width=True` | `width="stretch"` |

### Bug Fix: Chart Spike at 1 PM (v4.0.7)
Fixed sudden spike caused by model divergence (Sine vs Piecewise Linear).

**Fix**: Created unified `get_load_pattern(hour)` function

### Bug Fix: Reserve Rate Display (v4.0.6)
Fixed reserve rate showing 911% instead of ~121%.

| Before | After |
|--------|-------|
| `supply_reserve` (911 MW) | `reserve_rate` (121.8%) |

### Dashboard Status (v4.0.7)
| Metric | Value | Status |
|--------|-------|--------|
| 현재 수요 | 651 MW | ✅ |
| 예비율 | 121.8% | ✅ |
| SMP | 141.5원 | ✅ |

### Recent Commits
```
d75161c feat: Add solar and wind power generation to power chart
b223c38 feat: Add reserve power visualization to power chart
4e4397d docs: Add technical documentation for power demand forecast model
d85f5f1 docs: Update README with v4.0.7 bug fixes
1b25595 fix: Handle zero SMP values by falling back to mainland price
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

## Dashboard v4.0.7

### Features
1. 60hz.io style dark theme
2. GE Inertia style layout
3. Real-time power chart (actual vs forecast)
4. Gauge charts (demand, reserve rate)
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
15. **예비전력 시각화 (초록 음영) - NEW**
16. **태양광 발전량 표시 (노란 음영) - NEW**
17. **풍력 발전량 표시 (파란 음영) - NEW**

### Run Command
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

---

## Key Files

```
# Dashboard v4.0.7
src/dashboard/app_v4.py              - Main dashboard (Enhanced Chart)

# Documentation
docs/TECHNICAL_DOCUMENTATION.md      - 기술 설명서 (558줄) - NEW

# Alert System
data/alerts/alert_history.json       - Alert history
data/alerts/email_log.json           - Email send log
data/alerts/slack_log.json           - Slack send log

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
| v4.0.7 | 2025-12-19 | Enhanced chart (예비전력, 태양광, 풍력 시각화) |
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
3. View release: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.7

---

## Environment
- Python 3.13, PyTorch 2.0+
- Apple Silicon MPS (M1 MacBook Pro 32GB)

## Notes
- v4.0.7 adds enhanced power chart with reserve/solar/wind visualization
- Technical documentation added: `docs/TECHNICAL_DOCUMENTATION.md`
- Chart now shows: 예비전력(초록), 태양광(노란), 풍력(파란) 음영
- Email only sends for critical alerts (< 5%)
- Slack sends for all alerts (< 15%)
- All 1,564 tests passing
