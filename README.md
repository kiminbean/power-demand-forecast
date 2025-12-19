# 제주도 기후 변화에 따른 전력수요량 예측

> **Jeju Island Power Demand Forecasting using Weather Variables and Deep Learning**

<!-- CI/CD Badges -->
[![CI](https://github.com/kiminbean/power-demand-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/kiminbean/power-demand-forecast/actions/workflows/ci.yml)
[![CD](https://github.com/kiminbean/power-demand-forecast/actions/workflows/cd.yml/badge.svg)](https://github.com/kiminbean/power-demand-forecast/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/kiminbean/power-demand-forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/kiminbean/power-demand-forecast)
[![Release](https://img.shields.io/github/v/release/kiminbean/power-demand-forecast)](https://github.com/kiminbean/power-demand-forecast/releases)

<!-- Tech Stack Badges -->
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-1436%20passed-brightgreen.svg)](#테스트)

## 프로젝트 개요

제주시 기상 데이터(기온, 일사량, 지중온도 등)를 활용하여 제주도 전력 사용량을 예측하는 딥러닝 모델을 개발합니다. 1시간 후(+1h)부터 24시간 후(+24h)까지의 전력 수요를 예측할 수 있는 다중 시간대 예측 모델을 구현합니다.

### 핵심 가설

> **제주도에 설치된 태양광 발전량(Behind-the-Meter, BTM)이 한전 전력 수요에 숨겨진 주요 특성**

태양광 발전이 활발한 낮 시간대에는 자가소비로 인해 계통 전력 수요가 감소하고, 일몰 후에는 급격히 증가하는 "덕 커브(Duck Curve)" 현상이 발생합니다. 이러한 숨겨진 태양광 발전량을 기상 변수(일사량, 전운량 등)를 통해 간접적으로 모델링합니다.

---

## 주요 기능

### 🧠 딥러닝 모델
| 모델 | 설명 |
|------|------|
| **LSTM** | 장단기 메모리 순환 신경망 |
| **BiLSTM** | 양방향 LSTM |
| **TFT** | Temporal Fusion Transformer |
| **Ensemble** | 가중 평균 및 스태킹 앙상블 |

### 🌡️ 기상 피처 엔지니어링
- **THI (Temperature-Humidity Index)**: 온습도 지수 (August-Roche-Magnus 공식)
- **Wind Chill**: 체감 온도 (JAG/Siple 공식)
- **HDD/CDD**: 냉난방 도일 (Heating/Cooling Degree Days)

### 🤖 AutoML (Task 19)
- Optuna 기반 하이퍼파라미터 자동 최적화
- 모델 비교 및 자동 선택
- 베이지안 최적화 탐색

### 🔍 이상 탐지 (Task 22)
| 방법 | 설명 |
|------|------|
| **Z-Score** | 통계적 이상치 탐지 |
| **IQR** | 사분위수 기반 탐지 |
| **Isolation Forest** | 트리 기반 이상 탐지 |
| **Autoencoder** | 딥러닝 기반 이상 탐지 |
| **Ensemble** | 다중 방법 앙상블 |

### 🔮 설명 가능한 AI (Task 23)
| 방법 | 설명 |
|------|------|
| **Gradient** | 그래디언트 기반 피처 기여도 |
| **Integrated Gradients** | 적분 그래디언트 |
| **Perturbation** | 섭동 기반 민감도 분석 |
| **SHAP** | Shapley 값 기반 설명 |
| **Attention** | 어텐션 가중치 시각화 |

### 📊 시나리오 분석 (Task 24)
- **What-if 분석**: 다양한 기상 시나리오 시뮬레이션
- **민감도 분석**: 피처 변화에 따른 예측 민감도
- **탄력성 분석**: 피처 변화율 대비 예측 변화율

### 🚀 API & 인프라
| 기능 | 설명 |
|------|------|
| **FastAPI** | REST API 서버 |
| **Prometheus** | 메트릭 수집 및 모니터링 |
| **Alerting** | 임계값 기반 알림 시스템 |
| **Locust** | 부하 테스트 |
| **Docker** | 컨테이너화 배포 |

### 📡 실시간 데이터 크롤러
| 크롤러 | 데이터 소스 | 설명 |
|--------|-------------|------|
| **EPSIS Crawler** | 전력통계정보시스템 | 전국 실시간 전력수급 (5분 간격) |
| **Jeju Power Crawler** | 공공데이터포털 | 제주 전력수급현황 (시간별 실측) |

### 📊 Streamlit 대시보드
| 기능 | 설명 |
|------|------|
| **EPSIS 실시간** | 전국/제주 전력수급 현황 |
| **제주 실측 데이터** | 공공데이터포털 실측 데이터 시각화 |
| **예측 시각화** | 24시간 전력 수요 예측 |
| **시나리오 분석** | 폭염/한파 시나리오 시뮬레이션 |

---

## 📸 스크린샷

### 메인 대시보드
![Main Dashboard](docs/screenshots/01_main_dashboard.png)
*60hz.io 스타일 다크 테마 대시보드 - 제주도 전력 발전소 위치 및 실시간 전력수급 현황*

### SMP 예측
![SMP Prediction](docs/screenshots/02_smp_prediction.png)
*BiLSTM + Attention 모델 기반 24시간 SMP(계통한계가격) 예측 및 신뢰구간*

### KPX 실시간 연동
![KPX Realtime](docs/screenshots/04_kpx_realtime.png)
*한국전력거래소(KPX) 실시간 전력수급 데이터 연동 - 현재수요, 예비율, 발전소별 출력*

### 시스템 아키텍처
![System Architecture](docs/screenshots/05_system_architecture.png)
*전체 시스템 구성도 - 데이터 수집, 모델 학습, 예측, 대시보드*

### 예비율 경보 시스템
| 관심 단계 (10-15%) | 주의 경보 (5-10%) | 위험 경보 (<5%) |
|:---:|:---:|:---:|
| ![Caution](docs/screenshots/07_alert_caution.png) | ![Warning](docs/screenshots/08_alert_warning.png) | ![Critical](docs/screenshots/09_alert_critical.png) |

*KPX 기준 예비율 단계별 경보 시스템 - 15% 미만 시 관심, 10% 미만 시 주의, 5% 미만 시 위험 경보 발령*

---

## 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/kiminbean/power-demand-forecast.git
cd power-demand-forecast

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 통합 파이프라인 실행

```python
from src.pipeline import PowerDemandPipeline, PipelineConfig

# 설정
config = PipelineConfig(
    sequence_length=168,      # 7일 입력
    prediction_horizon=24,    # 24시간 예측
    model_type="lstm",
    hidden_size=128,
    epochs=100
)

# 파이프라인 실행
pipeline = PowerDemandPipeline(config)
result = pipeline.run()

print(f"Success: {result['success']}")
print(f"Stages completed: {result['stages_completed']}")
```

### API 서버 실행

```bash
# API 서버 시작
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 또는 Docker로 실행
docker-compose up -d
```

### 예측 API 호출

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "location": "jeju",
    "horizons": ["1h", "6h", "24h"],
    "model_type": "lstm"
})

predictions = response.json()["predictions"]
```

### 대시보드 실행

```bash
# EPSIS 실시간 대시보드 (권장)
streamlit run src/dashboard/app_v1.py

# API 연동 대시보드
streamlit run src/dashboard/app.py
```

### 제주 전력수급 크롤러 사용

```python
from tools.crawlers.jeju_power_crawler import JejuPowerCrawler

# 크롤러 초기화
crawler = JejuPowerCrawler()

# ZIP 파일에서 데이터 로드 (공공데이터포털)
data = crawler.load_from_zip("data/jeju_power_supply.zip")

# 최신 데이터 확인
latest = data[-1]
print(f"계통수요: {latest.system_demand:.1f} MW")
print(f"공급능력: {latest.supply_capacity:.1f} MW")
print(f"예비율: {latest.reserve_rate:.1f}%")

crawler.close()
```

---

## 고급 기능

### 이상 탐지

```python
from src.analysis.anomaly_detection import (
    ZScoreDetector,
    EnsembleAnomalyDetector,
    detect_anomalies
)

# 간편 함수
result = detect_anomalies(demand_data, method='zscore')
print(f"Detected {len(result.anomalies)} anomalies")

# 앙상블 탐지
ensemble = EnsembleAnomalyDetector(voting='majority')
result = ensemble.detect(demand_data, timestamps)
```

### 설명 가능한 AI (XAI)

```python
from src.analysis.explainability import (
    GradientExplainer,
    explain_prediction,
    ExplanationReport
)

# 예측 설명
explanation = explain_prediction(
    model=trained_model,
    inputs=input_tensor,
    feature_names=feature_names,
    method='gradient'
)

# 상위 기여 피처
top_features = explanation.top_contributors(n=5)
for feat in top_features:
    print(f"{feat.feature_name}: {feat.contribution:.4f}")
```

### 시나리오 분석

```python
from src.analysis.scenario_analysis import (
    ScenarioGenerator,
    ScenarioRunner,
    run_what_if_analysis
)

# What-if 분석
report = run_what_if_analysis(
    model=trained_model,
    input_data=current_data,
    feature_names=feature_names
)

# 시나리오 비교
summary = report.to_dict()['summary']
print(f"Highest demand scenario: {summary['highest_demand_scenario']}")
```

### AutoML 모델 선택

```python
from src.training.model_selection import AutoMLPipeline

# AutoML 파이프라인
automl = AutoMLPipeline(
    input_size=n_features,
    output_size=24,
    device=device
)

# 최적 모델 탐색
best_model, results = automl.run(
    train_loader=train_loader,
    val_loader=val_loader,
    n_trials=50
)
```

---

## 목표 성능

| 지표 | 1순위 목표 | 2순위 목표 |
|------|-----------|-----------|
| **MAPE** | 5~6% | - |
| **R²** | - | 75% 이상 |

### 참조 논문 성능 (JPD_RNN_Weather, 2025)

| 모델 | MAE | MSE | R² |
|------|-----|-----|-----|
| RNN + Temp | 5.946 | 76.937 | 0.985 |
| LSTM + Temp | 4.701 | 57.032 | 0.989 |
| BiLSTM + Temp | 4.69 | 55.685 | 0.989 |

---

## 데이터셋

### 기간
- **학습 데이터**: 2013년 1월 ~ 2022년 12월 (10년)
- **검증 데이터**: 2023년 1월 ~ 2023년 6월 (6개월)
- **테스트 데이터**: 2023년 7월 ~ 2024년 12월 (18개월)

### 데이터 소스

| 파일명 | 설명 | 기간 | 출처 |
|--------|------|------|------|
| `jeju_hourly_power_*.csv` | 시간별 전력거래량 (MWh) | 2013-2024 | 한국전력거래소 |
| `jeju_temp_hourly_*.csv` | 시간별 기상관측 데이터 | 2013-2024 | 기상청 ASOS |
| `jeju_CAR_daily_*.csv` | 일별 전기차 등록대수 | 2013-2024 | 제주도 |
| `jejudo_daily_visitors_*.csv` | 일별 입도객 수 | 2013-2025 | 제주관광공사 |
| `jeju_power_supply.zip` | 제주 전력수급현황 (시간별) | 2023-2025 | 공공데이터포털 |

### 실시간 데이터 소스

| 소스 | URL | 데이터 | 주기 |
|------|-----|--------|------|
| **EPSIS** | epsis.kpx.or.kr | 전국 전력수급 | 5분 |
| **공공데이터포털** | data.go.kr/data/15125113 | 제주 전력수급 | 시간별 |

---

## 프로젝트 구조

```
power-demand-forecast/
│
├── src/
│   ├── api/                    # FastAPI 서버
│   │   ├── main.py             # API 엔드포인트
│   │   ├── docs.py             # API 문서 및 모델 카드
│   │   └── schemas.py          # Pydantic 스키마
│   │
│   ├── dashboard/              # Streamlit 대시보드
│   │   ├── app.py              # API 연동 대시보드
│   │   └── app_v1.py           # EPSIS 실시간 대시보드
│   │
│   ├── models/                 # 딥러닝 모델
│   │   ├── lstm.py             # LSTM 모델
│   │   ├── transformer.py      # TFT 모델
│   │   └── ensemble.py         # 앙상블 모델
│   │
│   ├── training/               # 학습 모듈
│   │   ├── trainer.py          # 학습 루프
│   │   └── model_selection.py  # AutoML (Task 19)
│   │
│   ├── analysis/               # 분석 도구
│   │   ├── anomaly_detection.py    # 이상 탐지 (Task 22)
│   │   ├── explainability.py       # XAI (Task 23)
│   │   └── scenario_analysis.py    # 시나리오 분석 (Task 24)
│   │
│   ├── monitoring/             # 모니터링 (Task 18)
│   │   ├── metrics.py          # Prometheus 메트릭
│   │   ├── alerting.py         # 알림 시스템
│   │   └── health_checks.py    # 헬스 체크
│   │
│   ├── features/               # 피처 엔지니어링
│   │   └── weather_features.py # THI, Wind Chill, HDD/CDD
│   │
│   └── pipeline.py             # 통합 파이프라인 (Task 25)
│
├── tools/
│   └── crawlers/               # 데이터 크롤러
│       ├── epsis_crawler.py    # EPSIS 전국 실시간 크롤러
│       └── jeju_power_crawler.py  # 제주 전력수급 크롤러
│
├── tests/                      # 테스트 (1436 tests)
│   ├── test_pipeline.py        # 파이프라인 테스트
│   ├── test_jeju_crawler.py    # 제주 크롤러 테스트
│   ├── test_dashboard.py       # 대시보드 테스트
│   ├── test_anomaly_detection.py
│   ├── test_explainability.py
│   ├── test_scenario_analysis.py
│   ├── load_testing.py         # Locust 부하 테스트 (Task 21)
│   └── ...
│
├── .github/workflows/          # CI/CD
│   ├── ci.yml                  # 지속적 통합
│   ├── cd.yml                  # 지속적 배포
│   └── release.yml             # 릴리스 자동화
│
├── docker-compose.yml          # Docker 구성
├── Dockerfile                  # 컨테이너 이미지
└── requirements.txt            # 의존성
```

---

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 모듈 테스트
pytest tests/test_pipeline.py -v
pytest tests/test_anomaly_detection.py -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

**테스트 현황**: ✅ 1436 passed, 3 skipped

---

## API 문서

API 서버 실행 후 다음 URL에서 문서 확인:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 주요 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/health` | 헬스 체크 |
| `POST` | `/predict` | 전력 수요 예측 |
| `GET` | `/models` | 모델 목록 조회 |
| `GET` | `/metrics` | Prometheus 메트릭 |

---

## 모니터링

### Prometheus 메트릭

```python
from src.monitoring.metrics import MetricsCollector

collector = MetricsCollector()

# 예측 메트릭 기록
collector.record_prediction(
    model_name="lstm",
    latency_ms=45.2,
    prediction_value=850.5
)

# 메트릭 내보내기
metrics = collector.export_prometheus()
```

### 알림 설정

```python
from src.monitoring.alerting import AlertManager, ThresholdRule

manager = AlertManager()

# 임계값 규칙 추가
manager.add_rule(ThresholdRule(
    name="high_demand",
    metric="demand",
    threshold=1000,
    condition="above",
    severity="warning"
))

# 알림 확인
alerts = manager.check_all(current_metrics)
```

---

## Docker 배포

```bash
# 이미지 빌드
docker build -t power-demand-forecast .

# 컨테이너 실행
docker run -p 8000:8000 power-demand-forecast

# Docker Compose로 전체 스택 실행
docker-compose up -d
```

---

## 하드웨어 환경

```
Device: Apple M1 MacBook Pro
Memory: 32GB
GPU: Apple Silicon MPS (Metal Performance Shaders)
```

### MPS 사용 코드

```python
import torch

def get_device():
    """최적 디바이스 자동 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
model = model.to(device)
```

---

## 참고 자료

### 논문
- 정현수, 길준민. (2025). "기상 변수 통합 순환 신경망을 활용한 제주시 전력 수요 예측". 컴퓨터교육학회 논문지, 28(7), 99-105.

### 데이터 출처
- [한국전력거래소](https://www.kpx.or.kr/) - 전력거래량 데이터
- [기상청 기상자료개방포털](https://data.kma.go.kr/) - ASOS 기상관측 데이터
- [공공데이터포털](https://www.data.go.kr/) - 기상청 ASOS API, 제주 전력수급현황
- [전력통계정보시스템 (EPSIS)](https://epsis.kpx.or.kr/) - 실시간 전력수급 데이터

---

## 라이선스

MIT License

---

## 기여자

- **프로젝트 담당**: kiminbean
- **참조 논문 저자**: 정현수, 길준민 (제주대학교)
- **AI Assistant**: Claude Code (Anthropic)

---

*Last Updated: 2025-12-19 | v4.0.2*
