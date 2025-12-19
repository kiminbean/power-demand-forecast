# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.4] - 2025-12-19

### Highlights
- 💬 **Slack Webhook Integration**: Real-time Slack notifications for all alert levels
- 🎨 **Rich Message Formatting**: Slack Block Kit with color-coded attachments

### Added

#### Slack Notification System (`src/dashboard/app_v4.py`)
- **SlackNotifier Class**: Webhook-based notifications for all alert levels
  - Supports critical, danger, and warning alerts
  - Color-coded messages (red/orange/yellow)
  - Rich formatting with Slack Block Kit
  - Rate limiting (5-minute cooldown per alert type)
  - Message log persistence

- **Environment Configuration** (`.env`):
  ```
  SLACK_ALERTS_ENABLED=true
  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
  SLACK_CHANNEL=#alerts
  ```

#### Tests (`tests/test_dashboard.py`)
- 17 new tests for SlackNotifier:
  - Configuration tests (4)
  - Rate limiting tests (4)
  - Logging tests (5)
  - Alert level tests (4)
- Total tests: 1,564 passed

### Changed
- Dashboard sends Slack notifications for all alert levels (not just critical)
- Added toast notification on successful Slack send

---

## [4.0.3] - 2025-12-19

### Highlights
- 📧 **Email Notification System**: Automatic email alerts for critical reserve rate conditions
- 🔒 **Rate Limiting**: Prevents email spam (max 1 email per 5 minutes for same alert type)
- 📜 **Alert History**: Sidebar display of recent alerts with statistics

### Added

#### Email Notification System (`src/dashboard/app_v4.py`)
- **EmailNotifier Class**: SMTP-based email notification for critical alerts
  - Gmail SMTP support with TLS encryption
  - HTML-formatted alert emails with power data
  - Configurable sender and multiple recipients
  - Rate limiting to prevent spam (5-minute cooldown)
  - Email log persistence for audit trail

- **Environment Configuration** (`.env.example`):
  ```
  EMAIL_ALERTS_ENABLED=false
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your-email@gmail.com
  SMTP_PASSWORD=your-app-password
  ALERT_SENDER_EMAIL=your-email@gmail.com
  ALERT_RECIPIENT_EMAILS=admin1@example.com,admin2@example.com
  ```

#### Alert History Feature
- **AlertHistory Class**: JSON-persistent alert history tracking
- **Sidebar Display**: Recent alerts list with timestamps and statistics
- **Duplicate Prevention**: Same status alerts within 1 minute are skipped

#### Tests (`tests/test_dashboard.py`)
- 19 new tests for EmailNotifier:
  - Configuration tests (5)
  - Rate limiting tests (4)
  - Logging tests (5)
  - Recipient parsing tests (5)
- Total tests: 1,547 passed

### Changed
- Dashboard now sends email on critical alerts (reserve rate < 5%)
- Added toast notification on successful email send

---

## [4.0.2] - 2025-12-19

### Highlights
- 🚨 **Reserve Rate Alert System**: Visual alerts based on KPX standard thresholds
- 🧪 **Test Mode**: Sidebar toggle to simulate low reserve rate scenarios
- 📸 **Alert Screenshots**: Documentation for all alert levels

### Added

#### Reserve Rate Alert System (`src/dashboard/app_v4.py`)
- **KPX Standard Thresholds**:
  | Reserve Rate | Status | Alert |
  |--------------|--------|-------|
  | ≥15% | Normal | None |
  | 10-15% | 관심 (Caution) | 🟡 Yellow banner |
  | 5-10% | 주의 (Warning) | 🟠 Orange banner |
  | <5% | 위험 (Critical) | 🔴 Red pulsing banner |

- **CSS Animations**: Pulsing effect for critical alerts
- **Alert Banners**: Full-width banners with icons and severity colors
- **Reserve Badge**: Dynamic color-coded badge in metrics card

#### Test Mode
- **Sidebar Toggle**: Enable/disable alert test mode
- **Reserve Rate Slider**: Adjust test reserve rate (0-30%)
- **Data Source Indicator**: Shows "테스트 모드" when active

#### Screenshots (`docs/screenshots/`)
- `07_alert_caution.png`: Caution alert at 12% reserve
- `08_alert_warning.png`: Warning alert at 7% reserve
- `09_alert_critical.png`: Critical alert at 3% reserve

### Changed
- **README.md**: Added alert system screenshots in table format
- **Dashboard Header**: Reserve rate status now integrated with alert system

---

## [4.0.1] - 2025-12-19

### Highlights
- 🔴 **KPX Realtime Integration**: Live power supply/demand data from 한국전력거래소
- 🗺️ **Map Enhancement**: Power plant generation distributed based on actual KPX totals
- 📊 **Updated Power Plants**: December 2025 data with latest Jeju installations

### Added

#### KPX Realtime Data Integration (`src/dashboard/app_v4.py`)
- **Live data fetching** from KPX (https://www.kpx.or.kr) every 60 seconds
- **Data priority system**:
  1. KPX 실시간 (Primary) - Live power data
  2. EPSIS 파일 (Secondary) - Historical file data
  3. 시뮬레이션 (Fallback) - Simulated values
- **Header status indicator** showing current data source:
  - 🔴 KPX 실시간 연동
  - 📊 EPSIS 데이터 연동
  - ⚠️ 시뮬레이션 모드

#### Power Plant Generation Display
- **Realtime distribution**: Plant-level generation calculated from KPX total demand
- **Proportional allocation**: Generation distributed by plant capacity ratio
- **Type-based estimation**: Solar, wind, thermal, ESS with time-of-day factors

#### Updated Power Plant Data (`data/jeju_plants/jeju_power_plants.json`)
- **Wind**: 417.8 MW total (17 farms including 동복-북촌, 대정, 한동-평대)
- **Solar**: 562.6 MW (1,620+ distributed sites)
- **Thermal**: 598.8 MW (남제주 + LNG 복합)
- **ESS**: 460 MWh (including new long-duration 260 MWh)
- **HVDC**: 900 MW (3 submarine links)

### Changed
- `get_current_power_status()`: Now prioritizes KPX realtime data
- `get_jeju_power_plants()`: Distributes generation based on actual demand
- Dashboard header: Shows realtime data source status

### Verified
```
KPX 제주 실시간 데이터 수집 완료: 724 MW (예비율: 66.0%)
```

---

## [4.0.0] - 2025-12-18

### Highlights
- 🎯 **SMP Model v3.1**: Significantly improved prediction accuracy (MAPE 7.83%)
- 🎨 **Dashboard v4**: New 60hz.io-style dark theme with interactive Jeju map
- 📡 **Real-time Integration**: EPSIS data integration for live power market data

### Added

#### SMP Model v3.1 (`src/smp/models/train_smp_v3_fixed.py`)
- **BiLSTM + Stable Attention** architecture (4 heads)
- **Quantile Regression** outputs (Q10, Q50, Q90) for uncertainty estimation
- **22 engineered features** including temporal, price lags, and technical indicators
- **Noise Injection** for robustness (std=0.02, prob=0.5)
- **Walk-forward Validation** for realistic evaluation
- Model: 249,952 parameters

#### Dashboard v4 (`src/dashboard/app_v4.py`)
- **60hz.io-style dark theme** with professional UI
- **Interactive Jeju map** showing power plant locations (Folium)
- **Real-time EPSIS data** integration
- **24-hour SMP forecast** with confidence intervals
- **XAI analysis tab** for model interpretability

#### SMP Predictor Updates (`src/smp/models/smp_predictor.py`)
- Multi-version model support (v2.1, v3.1)
- `use_advanced=True` for v3.1 model
- `use_v2=True` for legacy v2.1 support
- Automatic feature engineering for each version

#### Documentation
- **Screenshots** added to README (`docs/screenshots/`)
- Main dashboard, SMP prediction, system architecture images

### Model Performance

| Metric | v2.1 (Previous) | v3.1 (Current) | Target | Status |
|--------|-----------------|----------------|--------|--------|
| MAPE | 10.68% | **7.83%** | <10% | ✅ |
| R² | 0.59 | **0.74** | >0.65 | ✅ |
| Coverage | 82.5% | **89.4%** | >85% | ✅ |
| MAE | 11.27 | **8.93** | - | ✅ |
| RMSE | 14.67 | **12.02** | - | ✅ |

### Fixed
- **v3.0 Training Failure**: Removed incorrect `torch.clamp()` that disrupted gradient flow
- **Loss Function**: Simplified from complex multi-component to Huber + Quantile
- **Normalization**: Changed from custom normalization to StandardScaler

### Key Files
```
src/smp/models/train_smp_v3_fixed.py  - Training pipeline
src/smp/models/smp_predictor.py       - Prediction interface
src/dashboard/app_v4.py               - Dashboard v4
models/smp_v3/smp_v3_model.pt         - Trained model
models/smp_v3/smp_v3_scaler.npy       - Feature scaler
models/smp_v3/smp_v3_metrics.json     - Performance metrics
data/smp/smp_5years_epsis.csv         - 5 years EPSIS data (26,240 records)
```

### Run Dashboard
```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504
```

---

## [1.1.2] - 2025-12-18

### Added

#### Jeju Realtime Crawler
- **Jeju Realtime Crawler** (`tools/crawlers/jeju_realtime_crawler.py`):
  - KPX 제주 실시간 전력수급 페이지 크롤링 (5분 간격 업데이트)
  - `JejuRealtimeData`: 실시간 전력수급 데이터 클래스
  - `JejuRealtimeCrawler`: KPX 웹페이지 크롤러
  - 데이터: 공급능력, 현재부하, 공급예비력, 운영예비력, 예비율
  - 상태 판단: 정상(≥15%), 관심(≥10%), 주의(≥5%), 위험(<5%)

#### Dashboard Integration
- **실시간 데이터 표시** (`src/dashboard/app_v1.py`):
  - 제주 실측 탭 상단에 KPX 실시간 데이터 표시
  - 60초 캐시 TTL (5분 업데이트 주기에 맞춤)
  - 4개 게이지 차트 (공급능력, 현재부하, 공급예비력, 예비율)
  - 상태 색상 표시 (녹색/노랑/주황/빨강)

#### Tests
- **Realtime Crawler Tests** (`tests/test_jeju_realtime_crawler.py`):
  - 23 comprehensive tests for realtime crawler
  - JejuRealtimeData tests (reserve_rate, utilization_rate, to_dict)
  - JejuRealtimeCrawler tests (fetch, timestamp extraction, status)
  - Total tests: 1,448 → 1,471

---

## [1.1.1] - 2025-12-18

### Added

#### Auto Download Feature
- **Jeju Crawler Auto Download** (`tools/crawlers/jeju_power_crawler.py`):
  - `auto_download()` method with cache management (7-day TTL)
  - `_check_cached_zip()` helper for cache validation
  - `_get_data_dir()` helper for data directory resolution
  - CLI options: `--auto-download`, `--force`, `--zip`
  - Streaming download for memory efficiency
  - ZIP file validation (minimum size check)

#### Tests
- **Auto Download Tests** (`tests/test_jeju_crawler.py`):
  - 12 new tests for auto_download functionality
  - Cache reuse/invalidation tests
  - Force download behavior tests
  - CLI option parsing tests
  - Total tests: 1,436 → 1,448

### Changed
- Dashboard uses `auto_download()` for Jeju actual data loading
- No manual ZIP path configuration required

### Removed
- **제주 추정 Tab**: Removed redundant EPSIS-based Jeju estimation tab
  - Actual measured data from data.go.kr is more accurate
  - Dashboard now has 2 tabs: "전국 현황", "제주 실측"
  - Removed ~125 lines of estimation code

---

## [1.1.0] - 2025-12-17

### Added

#### Data Crawlers
- **Jeju Power Crawler** (`tools/crawlers/jeju_power_crawler.py`):
  - `JejuPowerData`: Dataclass for Jeju power supply data
  - `JejuPowerCrawler`: Crawler for 공공데이터포털 (data.go.kr)
  - `JejuPowerDataStore`: CSV/JSON data persistence
  - ZIP file processing (5 CSVs: 계통수요, 공급능력, 공급예비력, 예측수요, 운영예비력)
  - 14,592 hourly records (2023-09-01 ~ 2025-04-30)

#### Dashboard Enhancements
- **제주 실측 Tab** (`src/dashboard/app_v1.py`):
  - Real Jeju power supply data visualization
  - 4 gauge charts (공급능력, 계통수요, 공급예비력, 예비율)
  - 7-day trend chart with secondary Y-axis for reserve rate
  - Detailed data table (last 48 hours)
  - Data source info and refresh functionality

#### Tests
- **Jeju Crawler Tests** (`tests/test_jeju_crawler.py`):
  - 33 comprehensive tests
  - JejuPowerData dataclass tests
  - ZIP processing tests
  - Data parsing tests
  - Integration tests with real data
  - Edge case tests

### Changed
- Updated test count: 1,423 → 1,436 tests
- Updated README.md with crawler and dashboard documentation
- Updated PROJECT_STATUS.md with v1.1.0 progress

### Fixed
- **EPSIS AttributeError**: Fixed dict access syntax (`d['timestamp']` instead of `d.timestamp`)
- **Chart Visibility**: Removed `fill='tozeroy'` and increased line width for reserve power/rate charts
- **National Tab Gauges**: Changed from `st.metric()` to gauge charts for reserve power and rate
- **Streamlit Deprecation**: Replaced `use_container_width=True` with `width="stretch"` (53 occurrences)

---

## [1.0.0] - 2025-12-16

### Added

#### Deep Learning Models
- **LSTM Model** (`src/models/lstm.py`): Long Short-Term Memory network for time series forecasting
- **BiLSTM Model** (`src/models/bilstm.py`): Bidirectional LSTM with enhanced temporal learning
- **Temporal Fusion Transformer** (`src/models/transformer.py`): State-of-the-art attention-based model
- **Ensemble Models** (`src/models/ensemble.py`): Weighted average and stacking ensemble methods

#### Feature Engineering
- **Weather Features** (`src/features/weather_features.py`):
  - THI (Temperature-Humidity Index) using August-Roche-Magnus formula
  - Wind Chill calculation using JAG/Siple formula
  - HDD/CDD (Heating/Cooling Degree Days)
- **Time Features** (`src/features/time_features.py`): Cyclical encoding for hour, day, month
- **Solar Features** (`src/features/solar_features.py`): Solar radiation derived features

#### Training & AutoML (Task 19)
- **Model Selection** (`src/training/model_selection.py`):
  - `AutoMLPipeline`: End-to-end automated model selection
  - `HyperparameterTuner`: Optuna-based Bayesian optimization
  - `ModelComparator`: Multi-model comparison framework
  - `ModelFactory`: Unified model creation interface

#### API & Documentation (Task 20)
- **FastAPI Server** (`src/api/main.py`): REST API for predictions
- **API Documentation** (`src/api/docs.py`):
  - OpenAPI custom documentation
  - Model Cards with JSON/Markdown export
  - Comprehensive error codes
- **Pydantic Schemas** (`src/api/schemas.py`): Request/response validation

#### Load Testing (Task 21)
- **Locust Tests** (`tests/load_testing.py`):
  - `PowerDemandAPIUser`: Standard user simulation
  - `HeavyUser`: High-frequency request simulation
  - `LightUser`: Occasional request simulation
  - `LoadTestAnalyzer`: Result analysis and reporting
  - `PerformanceCriteria`: SLA validation

#### Anomaly Detection (Task 22)
- **Detection Methods** (`src/analysis/anomaly_detection.py`):
  - `ZScoreDetector`: Statistical outlier detection
  - `IQRDetector`: Interquartile range based detection
  - `IsolationForestDetector`: Tree-based anomaly detection
  - `AutoencoderDetector`: Deep learning based detection
  - `RealtimeAnomalyDetector`: Streaming anomaly detection
  - `EnsembleAnomalyDetector`: Multi-method voting ensemble

#### Explainable AI (Task 23)
- **Explainability Methods** (`src/analysis/explainability.py`):
  - `GradientExplainer`: Gradient-based feature attribution
  - `IntegratedGradientsExplainer`: Path integral attribution
  - `PerturbationExplainer`: Sensitivity-based importance
  - `SHAPExplainer`: Shapley value explanations
  - `AttentionExplainer`: Attention weight visualization
  - `ExplanationReport`: Comprehensive report generation

#### Scenario Analysis (Task 24)
- **What-if Analysis** (`src/analysis/scenario_analysis.py`):
  - `ScenarioGenerator`: Predefined and custom scenario creation
  - `ScenarioRunner`: Model-based scenario simulation
  - `SensitivityAnalyzer`: Feature sensitivity analysis
  - `ScenarioComparator`: Multi-scenario comparison
  - `ScenarioReport`: JSON/Markdown report generation

#### Integrated Pipeline (Task 25)
- **Pipeline** (`src/pipeline.py`):
  - `PowerDemandPipeline`: End-to-end workflow orchestration
  - `PipelineConfig`: Centralized configuration
  - `PipelineResult`: Structured result tracking
  - Stages: Data load, preprocess, feature engineering, training, prediction, analysis, reporting

#### Monitoring System (Task 18)
- **Metrics** (`src/monitoring/metrics.py`):
  - `MetricsCollector`: Prometheus-compatible metrics
  - `ModelMetrics`: Model-specific performance tracking
  - `SystemMetrics`: Resource utilization monitoring
- **Alerting** (`src/monitoring/alerting.py`):
  - `AlertManager`: Centralized alert management
  - `ThresholdRule`: Threshold-based alerts
  - `AnomalyRule`: Anomaly-based alerts
  - `TrendRule`: Trend detection alerts
- **Health Checks** (`src/monitoring/health_checks.py`):
  - `HealthChecker`: System health monitoring
  - `SystemHealthCheck`: CPU, memory, disk checks
  - `ModelHealthCheck`: Model availability checks
  - `DependencyHealthCheck`: External service checks
- **Logging** (`src/monitoring/logging_config.py`):
  - Structured JSON logging
  - Thread-local context management
  - Log rotation and retention

#### CI/CD
- **GitHub Actions Workflows**:
  - `ci.yml`: Continuous Integration (lint, test, coverage)
  - `cd.yml`: Continuous Deployment (Docker, staging, production)
  - `release.yml`: Automated release workflow

#### Testing
- **Comprehensive Test Suite** (1423 tests):
  - Unit tests for all modules
  - Integration tests (`tests/test_integration.py`)
  - API tests (`tests/test_api.py`)
  - Load tests (`tests/load_testing.py`)

#### Documentation
- **README.md**: Comprehensive project documentation
- **API Docs**: Swagger UI and ReDoc integration
- **Model Cards**: ML model documentation

### Infrastructure

- **Docker Support**:
  - `Dockerfile`: Multi-stage build for production
  - `docker-compose.yml`: Full stack deployment
- **Requirements**:
  - `requirements.txt`: Core dependencies
  - `requirements-api.txt`: API server dependencies
  - `requirements-dev.txt`: Development dependencies

---

## [Unreleased]

### Planned
- Real-time prediction streaming
- A/B testing framework
- Model versioning and registry
- Distributed training support
- Advanced visualization dashboard

---

[4.0.0]: https://github.com/kiminbean/power-demand-forecast/releases/tag/v4.0.0
[1.1.2]: https://github.com/kiminbean/power-demand-forecast/releases/tag/v1.1.2
[1.1.1]: https://github.com/kiminbean/power-demand-forecast/releases/tag/v1.1.1
[1.1.0]: https://github.com/kiminbean/power-demand-forecast/releases/tag/v1.1.0
[1.0.0]: https://github.com/kiminbean/power-demand-forecast/releases/tag/v1.0.0
[Unreleased]: https://github.com/kiminbean/power-demand-forecast/compare/v4.0.0...HEAD
