# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
