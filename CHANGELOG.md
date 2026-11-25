# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-26

### Changed (2026-11-23~26) — Streamlit 대시보드(app_v4) UI 개편
- **화면 전면 개편**(`src/dashboard/app_v4.py`, `.streamlit/config.toml`, 2026-11-25 운영 반영): 라이트·다크 테마(시스템 설정), 앱 바·상태 문장·KPI 4장, 값마다 실측·예측·추정·예시 칩과 출처·기준 시각, 발전소 지도와 지역 인스펙터, 수급 추이를 '추정 패턴'으로 표기, 360~1920px 대응. 데이터 로직은 바꾸지 않았다(`logic_guard` changed_logic_REVIEW [] · removed []).
- **SMP 머리 숫자 기준**: 오늘 오차·예측 평균선·시간대별 요약은 기준선 입력이 0.00인('예측 아님') 시간을 뺀 예측 시간으로 계산한다(예: 40.8 → 21.4 원/kWh). 24시간 전체 값은 같은 줄에 회색으로 남긴다.
- **평가와 판정**: 경쟁사 UI 기준(60Hertz 8.7 / Electricity Maps 9.2) 루브릭으로 10차까지 평가했다. 10차 평균 8.57(R1~R8 모두 8.0 이상)·무결성 27/27이고, 화면 배치는 마무리 수정(`778a3974`) 뒤 독립 재평가(10b) 22/22다. 합격선(평균 8.5 이상과 관문)을 충족했다. 공개 주소 :8443에서 두 테마 × 1440·390px로 확인했다(2026-11-26).
- **설명 페이지**(`docs/mentor-tech-explainer.html`, 2026-11-26): '한계' 칸을 개편된 메인 대시보드에 맞게 고쳤다. 02장 배포 구조(루트 / → execo 웹사이트 :3311, /jref → :8501)와 그림 1, 웹 v6 종료(2026-11-07, 공개 웹 버전 4개), API 허용 목록 근거 줄도 지금 설정에 맞게 바로잡았다. 이전부터 커밋하지 않았던 11-10 수정(J-REF 태양광 링크, 자료 행 수, 검증 MAPE 문장)도 사용자 승인으로 함께 커밋했다.
- **문서 검토**(2026-11-26): 독립 검토 2명(사실·형식)이 지적한 19건을 반영했다. 관문 명령을 그대로 실행되는 형태로 고쳤고(35 passed · 29/29 · changed_logic_REVIEW [] · removed [] 확인), 개편 보고서에서 GitHub 취소선이 되던 '~' 44줄을 이스케이프했다.

### Added (2026-11-23~26)
- 설계 명세 `docs/2026-11-23-dashboard-ui-top1-design.md`(1~14장), 전후 보고서 `docs/2026-11-23-dashboard-ui-top1-report.md`, 검증 도구 `docs/2026-11-23-dashboard-ui-top1/`(`ui_guard.py`·`logic_guard.py`·`shoot.py`·대비·지도 시점·터치 탐침), 개편 전후 스크린샷.

### Known issues (2026-11-26)
- 대시보드 남은 결함(보고서 5.3절, 높음 0건): 사이드바를 연 1181\~1320px 넘침, SMP 표 CSV 내보내기의 0.00 입력 시간 모형 원값, 게이지 설명 '아님)' 줄 끊김 등. 승인 대기 P1\~P11은 5.1절에 있다.
- SMP 수집기가 `data/smp/smp_real_epsis.csv`를 원자적으로 쓰지 않아, 쓰는 도중 읽으면 부분 표가 1시간 캐시될 수 있다('1023일 전 가격'). 데이터 로직이라 승인 뒤 고친다.

### Added (2026-11-06~07)
- **AR + GBM 수요 모델(`ar_gbm`)**: 시간대별 선형 AR에 HistGradientBoosting 잔차 보정을 더한 모델이다. 학습(`src/training/train_demand_ar_gbm.py`), 추론(`src/inference/ar_gbm_predictor.py`), 산출물(`models/demand_ar_gbm/`, 학습 ~2024-06)을 갖췄다. 고정 test(2023-07~2024-12)에서 MAPE 3.63%·R² 0.954로 시간대별 AR(3.78%)과 LSTM(4.16%)을 이겼다. walk-forward 평균은 3.35%다.
- **수요 배포 게이트 기준선**: ramp7·시간대별 AR 기준선, Diebold-Mariano 검정, MAE 1% 개선 조건을 추가했다(e120d68e).
- **2025년 실측 수집과 시간외 비교**: `tools/fetch_kpx_jeju_trade.py`(data.go.kr 15133498)로 실측을 받고 `docs/benchmark-2026-11-06/compare_default_2025.py`로 비교한다.
- **경쟁 비교 보고서** `docs/2026-11-06-competitor-benchmark.md`: 공개 수치 128건을 원문과 대조했다. 1.1절에는 경쟁사 이름과 실제 수치를, 기간을 맞춘 ar_gbm 수치(`competitor_same_basis.py`)와 나란히 둔다.
- **설명 페이지 05장 "경쟁 비교"**(`docs/mentor-tech-explainer.html`)를 추가하고, 용어 사전에 계통수요/전력거래량과 MAE를 넣었다.

### Changed (2026-11-05~07)
- **API 기본 수요 모델을 ar_gbm으로 교체**(2026-11-06, 사용자 승인): 2025년 실측에서 MAPE 4.67%·R² 0.929로 LSTM(5.22%·0.910)보다 정확했고, 12개월 모두 앞섰다.
  - `model_type`을 생략하면 ar_gbm을 쓴다. 이력이 337시간 미만이거나 산출물이 없으면 demand_only로 대체하고 `model_used`에 표시한다.
  - 되돌리기: `DEFAULT_MODEL=demand_only`
- **수요 LSTM 잔차 타깃 재학습**(2026-11-05~06): 2025-12 체크포인트를 교체했다. 이 체크포인트는 6.06%로 보고됐지만 같은 경로로 다시 채점하면 9.78%였다. 새 모델의 고정 test는 4.16%·R² 0.939다(원본 목표, 서빙 경로).
- **SMP 서빙을 어제 같은 시각 가격(naive)으로 교체**(2026-11-05): 제주 재학습 v4.1(16.82%)이 naive(11.75%)보다 나빠 게이트를 통과하지 못했다. 성능을 다시 검증하지 않은 v3.2는 어제 가격이 없을 때만 대체로 쓴다(3ccbf736, 417f5b41). 제주 SMP는 시범사업 API가 실패하면 마지막 검증 가격을 쓰고, 재시작해도 그 값을 유지한다(fb61e8e7, 19570fbb).
- **음력 공휴일 표 2026·2027년 추가**: 우주항공청 월력요항 기준이다. 학습 기준과 같게 설·추석 3일과 부처님오신날만 넣었다.
- **Docker API**: `requirements-docker.txt`를 설치한다(CPU torch, 호스트 검증 버전 고정, pandas<3). 수요 모델을 읽기 전용으로 마운트하고 메모리 상한 1GB를 걸었다. API 컨테이너만 :8506에서 운영한다.
- **앱 간 수치 일치**(2026-11-05): 발전소 목록, SMP 시각, KPI, 모델 메타데이터가 웹 v5~v7·Streamlit·모바일에서 같은 출처를 쓰게 했다(70b199cf 외).
- **문서 정정**:
  - 설명 페이지의 낡은 수치 51곳을 정정하고, 폭을 880px에서 1560px로 넓히고, 검토 지적 14건을 반영했다(2026-11-06).
  - RE-BMS v5/v6 기술문서와 `docs/TECHNICAL_DOCUMENTATION.md`의 옛 수요 LSTM 수치(6.06%)를 현재 수치로 바꿨다. README에는 현재 운영 모델 성능 표를 추가했다(2026-11-07).

### Fixed (2026-11-06)
- **수요 목표 누수**: test를 포함한 전 기간으로 IQR 클리핑(694.658 MWh 상한)을 하던 것을 없앴다. 처리된 CSV의 919시간을 원본 값으로 복원했다(5665fac0).
- **수요 수치 착오**: 경쟁 비교 초판의 LSTM 4.31%는 잘린 입력으로 만든 예측을 채점한 값이었다. 서빙 경로로 다시 재서 4.16%로 고쳤다.
- **실패 테스트**: 삭제된 크롤러를 부르던 수동 점검 스크립트 `tools/crawlers/test_jeju_crawlers.py`를 제거했다(a39e9bb3). 루트 pytest는 2,271개 통과, 3개 건너뜀이다.

### Fixed
- **SMP 스케일러 데이터 누수**: 시간 분할 이전에 스케일러를 fit하던 누수 수정 — SMP 학습 스크립트 3종이 공통 헬퍼(`src/smp/models/time_split_scaling.py`)로 분할 이후 train 슬라이스에만 fit. `np.save`(dict) 예외는 `np.savez`로 수정.
- **SMP 3개 모델 재학습(누수 수정 코드)**: v4.0 test MAPE 17.17%·R² 0.124·커버리지 77.5%, v4.1 walk-forward 평균 MAPE 8.76%·R² 0.463·MAE 10.53원/kWh·커버리지 68.6%, v3.4 test MAPE 21.28%·R² -0.537(사용 보류). 재학습 전 수치(MAPE 7.83%·R² 0.736 등)는 누수로 부풀려져 무효. 누수 버전 기준치는 `models/_pre_retrain_backup_2026-11-05/`에 보관.
- **평가 지표 통일**: 구간 MAPE를 zero-denominator 정책을 갖는 WMAPE로 통일.

### Changed
- **문서 정정**: 멘토링 문서 2종(`docs/mentor-tech-explainer.html`·`docs/mentoring_study_hub.html`)의 SMP 수치를 원본 메트릭 기준으로 정정하고 hub 표면 정리(강조색 축소·표 행 구분선만·오탈자 수정). 구 수치가 남아 있던 기술문서(RE-BMS v5/v6, BiLSTM v3.1, system_architecture)와 CLAUDE.md, CHANGELOG에 무효 명시+현행값 병기.
- **정산 결과 구분**: 모의계산 결과에 `is_simulated`/`data_source` 배지 추가로 실계산·모의계산 구분.
- 웹 v7 라우팅·CI 복구, rolling-origin 평가 도구 추가.

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
