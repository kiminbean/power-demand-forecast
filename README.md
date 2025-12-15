# 제주도 기후 변화에 따른 전력수요량 예측

> **Jeju Island Power Demand Forecasting using Weather Variables and Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 프로젝트 개요

제주시 기상 데이터(기온, 일사량, 지중온도 등)를 활용하여 제주도 전력 사용량을 예측하는 딥러닝 모델을 개발합니다. 1시간 후(+1h)부터 24시간 후(+24h)까지의 전력 수요를 예측할 수 있는 다중 시간대 예측 모델을 구현합니다.

### 핵심 가설

> **제주도에 설치된 태양광 발전량(Behind-the-Meter, BTM)이 한전 전력 수요에 숨겨진 주요 특성**

태양광 발전이 활발한 낮 시간대에는 자가소비로 인해 계통 전력 수요가 감소하고, 일몰 후에는 급격히 증가하는 "덕 커브(Duck Curve)" 현상이 발생합니다. 이러한 숨겨진 태양광 발전량을 기상 변수(일사량, 전운량 등)를 통해 간접적으로 모델링합니다.

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
| `jeju_hourly_power_2013_2024.csv` | 시간별 전력거래량 (MWh) | 2013-2024 | 한국전력거래소 |
| `jeju_temp_hourly_*.csv` | 시간별 기상관측 데이터 | 2013-2024 | 기상청 ASOS |
| `한국동서발전_제주_기상관측_태양광발전.csv` | 태양광 발전량 및 기상 | 2018-2024.05 | 한국동서발전 |
| `jeju_CAR_daily_2013_2024.csv` | 일별 전기차 등록대수 | 2013-2024 | 제주도 |
| `jejudo_daily_visitors_2013_2025.csv` | 일별 입도객 수 | 2013-2025 | 제주관광공사 |

### 주요 변수

#### Target (예측 대상)
- `전력거래량(MWh)`: 시간별 제주도 전력 수요

#### Features (입력 변수)

**기상 변수 (높은 상관관계)**
| 변수명 | 설명 | 상관계수 |
|--------|------|----------|
| 기온 | 대기 온도 (°C) | 0.68~0.83 |
| 지면온도 | 지면 온도 (°C) | 0.66~0.77 |
| 지중온도 (5/10/20/30cm) | 토양 온도 (°C) | 0.57~0.69 |
| 일사량 | 태양 복사 에너지 (MJ/m²) | -0.26 (음의 상관) |

**기상 변수 (보조)**
- 습도 (%)
- 강수량 (mm)
- 풍속 (m/s)
- 전운량 (1/10)
- 이슬점온도 (°C)
- 현지기압/해면기압 (hPa)

**파생 변수**
- THI (불쾌지수): 여름철 냉방 수요 지표
- HDD/CDD (난방/냉방 도일): 계절별 에너지 수요 지표
- 시간 특성: hour_sin, hour_cos, dayofweek_sin, dayofweek_cos
- 휴일 여부: is_weekend, is_holiday

**외부 변수**
- 전기차 누적 등록대수 (충전 수요)
- 일별 관광객 수 (관광 전력 수요)
- 태양광 설비용량 / 발전량 (BTM 효과 추정)

---

## 모델 아키텍처

### 1. LSTM (Long Short-Term Memory)
```
Input(48h) → LSTM(hidden=64, layers=2) → Dropout(0.2) → Linear → Output(1~24h)
```

### 2. BiLSTM (Bidirectional LSTM)
```
Input(48h) → BiLSTM(hidden=64, layers=2) → Dropout(0.2) → Linear → Output(1~24h)
```

### 3. Vanilla RNN (Baseline)
```
Input(48h) → RNN(hidden=50, layers=2) → Linear → Output(1~24h)
```

### 4. Transformer (추가 실험)
```
Input(48h) → Positional Encoding → Multi-Head Attention → FFN → Output(1~24h)
```

### 하이퍼파라미터 (논문 참조)

| 파라미터 | 값 |
|----------|-----|
| Sequence Length | 48 (시간) |
| Hidden Size | 50~64 |
| Num Layers | 2 |
| Dropout | 0.2 |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100 |
| Optimizer | Adam |
| Early Stopping | patience=15 |

---

## 프로젝트 구조

```
power-demand-forecast/
│
├── README.md                    # 프로젝트 설명서 (본 문서)
├── feature_list.json            # 작업 명세서
├── requirements.txt             # Python 패키지 목록
│
├── data/
│   ├── raw/                     # 원본 데이터
│   │   ├── jeju_hourly_power_2013_2024.csv
│   │   ├── jeju_temp_hourly_*.csv
│   │   ├── 한국동서발전_제주_기상관측_태양광발전.csv
│   │   ├── jeju_CAR_daily_2013_2024.csv
│   │   ├── jejudo_daily_visitors_2013_2025.csv
│   │   └── JPD_RNN_Weather.pdf  # 참조 논문
│   │
│   ├── processed/               # 전처리된 데이터
│   │   └── jeju_hourly_merged.csv
│   │
│   └── features/                # 피처 엔지니어링 결과
│       └── jeju_hourly_features.csv
│
├── src/
│   ├── data/                    # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # 전처리 (결측치, 이상치)
│   │   ├── merge_datasets.py    # 데이터셋 병합
│   │   └── dataset.py           # PyTorch Dataset 클래스
│   │
│   ├── features/                # 피처 엔지니어링
│   │   ├── __init__.py
│   │   ├── weather_features.py  # THI, HDD, CDD 등
│   │   ├── time_features.py     # 시간 특성
│   │   └── solar_features.py    # 태양광 관련 특성
│   │
│   ├── models/                  # 모델 정의
│   │   ├── __init__.py
│   │   ├── lstm.py              # LSTM 모델
│   │   ├── bilstm.py            # BiLSTM 모델
│   │   ├── rnn.py               # Vanilla RNN
│   │   └── transformer.py       # Transformer 모델
│   │
│   ├── training/                # 학습 모듈
│   │   ├── __init__.py
│   │   ├── trainer.py           # 학습 루프
│   │   ├── train_lstm.py        # LSTM 학습 스크립트
│   │   └── train_multi_horizon.py  # 다중 시간대 학습
│   │
│   ├── evaluation/              # 평가 모듈
│   │   ├── __init__.py
│   │   ├── metrics.py           # MAPE, R², MAE, MSE
│   │   └── visualize.py         # 시각화 함수
│   │
│   └── utils/                   # 유틸리티
│       ├── __init__.py
│       ├── device.py            # MPS/CUDA/CPU 선택
│       └── config.py            # 설정 관리
│
├── notebooks/                   # Jupyter/Colab 노트북
│   ├── 01_EDA.ipynb             # 탐색적 데이터 분석
│   ├── 02_Preprocessing.ipynb   # 전처리 과정
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_LSTM_Training.ipynb   # LSTM 학습
│   ├── 05_Model_Comparison.ipynb
│   └── 06_Results_Analysis.ipynb
│
├── tools/
│   └── crawlers/                # 데이터 수집 도구
│       ├── __init__.py
│       ├── config.py
│       ├── kma_crawler.py
│       ├── kma_api.py
│       └── download_weather.py
│
├── models/                      # 학습된 모델 저장
│   ├── lstm_best.pt
│   ├── bilstm_best.pt
│   └── checkpoints/
│
├── results/                     # 실험 결과
│   ├── figures/                 # 시각화 이미지
│   ├── metrics/                 # 성능 지표 CSV
│   └── reports/                 # 분석 보고서
│
├── logs/                        # 학습 로그
│   └── tensorboard/
│
└── tests/                       # 테스트 코드
    ├── __init__.py
    └── test_weather_features.py
```

---

## 설치 및 실행

### 환경 요구사항

- **OS**: macOS (M1/M2/M3), Linux, Windows
- **Python**: 3.10+
- **GPU**: Apple Silicon MPS (권장), NVIDIA CUDA (선택)
- **RAM**: 16GB 이상 권장

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/power-demand-forecast.git
cd power-demand-forecast

# 2. 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비

```bash
# 기상 데이터 다운로드 (API 키 필요)
export KMA_API_KEY='your_api_key'
python tools/crawlers/download_weather.py "제주도 2013~2024년 기온 데이터를 시간 단위로 다운로드해줘"
```

### 전처리 실행

```bash
# 데이터 전처리 및 병합
python src/data/preprocessing.py
python src/data/merge_datasets.py

# 피처 엔지니어링
python src/features/weather_features.py
```

### 모델 학습

```bash
# LSTM 모델 학습
python src/training/train_lstm.py

# 다중 시간대 예측 학습 (1h~24h)
python src/training/train_multi_horizon.py --horizons 1 6 12 24

# Production 모델 학습 (권장)
python src/training/train_production.py
```

### Production 모델 추론

```bash
# CLI로 예측
python src/inference/predict.py --data data.csv --model conditional

# 배치 예측
python src/inference/predict.py --data data.csv --model demand_only --batch --output predictions.csv
```

```python
# Python에서 사용
from inference import predict, ProductionPredictor

# 방법 1: 간편 함수
result = predict(df, model='conditional', mode='soft')
print(f"Predicted: {result.predicted_demand:.2f} MW")

# 방법 2: ProductionPredictor 클래스
predictor = ProductionPredictor()
predictor.load_models()

# 단일 예측
pred = predictor.predict_demand_only(df)

# Conditional 예측 (겨울철 자동 최적화)
result = predictor.predict_conditional(df, mode='soft')

# 배치 예측
batch = predictor.predict_batch(df, model='demand_only', step=24)
```

### Google Colab에서 실행

```python
# Colab 노트북에서 실행
!git clone https://github.com/your-repo/power-demand-forecast.git
%cd power-demand-forecast
!pip install -r requirements.txt

# 노트북 실행
# notebooks/04_LSTM_Training.ipynb
```

---

## 데이터 전처리 방법론

### 결측치 처리

논문과 동일하게 **선형 보간법(Linear Interpolation)** 사용:

```python
# 시간 연속성을 유지하면서 결측치 보간
df['기온'] = df['기온'].interpolate(method='linear')

# 시작/끝 결측치는 forward/backward fill
df['기온'] = df['기온'].fillna(method='ffill').fillna(method='bfill')
```

### 이상치 처리

1. **IQR 방법**: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR 범위 밖 값 탐지
2. **도메인 기반**: 물리적으로 불가능한 값 제거 (예: 음수 일사량)
3. **보간 대체**: 이상치를 NaN으로 변환 후 선형 보간

```python
def detect_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return (series < lower) | (series > upper)
```

### 정규화

**Min-Max Scaling** (논문 동일):

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## 평가 지표

### 1. MAPE (Mean Absolute Percentage Error)

$$MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### 2. R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

### 3. MAE (Mean Absolute Error)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 4. MSE (Mean Squared Error)

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

---

## 실험 계획

### Phase 1: 데이터 전처리 및 EDA
- [ ] 원본 데이터 품질 검사
- [ ] 결측치/이상치 분석 및 처리
- [ ] 피어슨 상관계수 분석
- [ ] 데이터셋 병합

### Phase 2: 피처 엔지니어링
- [ ] THI (불쾌지수) 생성
- [ ] HDD/CDD (난방/냉방 도일) 생성
- [ ] 시간 특성 생성
- [ ] 태양광 발전 추정 변수 생성

### Phase 3: 모델 개발
- [ ] Baseline RNN 구현
- [ ] LSTM 구현
- [ ] BiLSTM 구현
- [ ] Multi-horizon 예측 구현 (1h~24h)

### Phase 4: 실험 및 평가
- [ ] 기상변수 포함/미포함 비교
- [ ] 시퀀스 길이 실험 (24h, 48h, 72h)
- [ ] 예측 시간대별 성능 비교

### Phase 5: 결과 분석
- [ ] 시계열 예측 시각화
- [ ] 오차 분석
- [ ] 계절별/시간대별 성능 분석

---

## 참고 자료

### 논문
- 정현수, 길준민. (2025). "기상 변수 통합 순환 신경망을 활용한 제주시 전력 수요 예측". 컴퓨터교육학회 논문지, 28(7), 99-105.

### 데이터 출처
- [한국전력거래소](https://www.kpx.or.kr/) - 전력거래량 데이터
- [기상청 기상자료개방포털](https://data.kma.go.kr/) - ASOS 기상관측 데이터
- [공공데이터포털](https://www.data.go.kr/) - 기상청 ASOS API

### 관련 프로젝트
- [Carbon Free Island Jeju 2030](https://oecd-opsi.org/innovations/carbon-free-island-jeju-by-2030/)

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

## 라이선스

MIT License

---

## 기여자

- **프로젝트 담당**: [Your Name]
- **참조 논문 저자**: 정현수, 길준민 (제주대학교)

---

*Last Updated: 2024-12-14*
