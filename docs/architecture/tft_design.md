# Temporal Fusion Transformer (TFT) Architecture Design

## Task 1.1: TFT 아키텍처 연구 및 설계

**Status:** In Progress
**Created:** 2025-12-16
**Author:** Claude Code

---

## 1. TFT 논문 분석 (Google, 2020)

### 1.1 논문 정보
- **Title:** Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **Authors:** Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, Tomas Pfister (Google Cloud AI)
- **Published:** International Journal of Forecasting, 2021
- **arXiv:** https://arxiv.org/abs/1912.09363

### 1.2 핵심 아이디어
TFT는 다음을 결합한 하이브리드 아키텍처:
1. **LSTM Encoder-Decoder**: 지역적(local) 시계열 패턴 학습
2. **Self-Attention**: 장기(long-term) 의존성 학습
3. **Variable Selection Network**: 피처 중요도 자동 학습
4. **Gated Residual Network (GRN)**: 정보 흐름 제어

### 1.3 TFT의 장점
| 특성 | 설명 |
|------|------|
| 해석 가능성 | Attention weight로 시간/피처 중요도 분석 |
| 다중 시간대 예측 | 여러 horizon 동시 예측 |
| 피처 유형 지원 | Static, Known Future, Unknown 피처 구분 |
| 불확실성 정량화 | Quantile 출력으로 예측 구간 제공 |

---

## 2. 기존 LSTM 아키텍처와 비교

### 2.1 현재 프로젝트 LSTM 구조 (`src/models/lstm.py`)

```
LSTMModel:
    Input (batch, seq_len=48, features=38)
        ↓
    LSTM (hidden=64, layers=2, dropout=0.2)
        ↓
    FC Layer 1 (64 → 64, ReLU, Dropout)
        ↓
    FC Layer 2 (64 → 32, ReLU, Dropout)
        ↓
    FC Layer 3 (32 → 1)
        ↓
    Output (batch, 1)
```

### 2.2 LSTM vs TFT 비교

| 특성 | LSTM (현재) | TFT (목표) |
|------|-------------|------------|
| 피처 처리 | 모든 피처 동일 처리 | Static/Known/Unknown 구분 |
| 어텐션 | 없음 | Interpretable Multi-Head Attention |
| 피처 선택 | 없음 (수동) | Variable Selection Network (자동) |
| 출력 유형 | Point Estimate | Quantile (불확실성) |
| 해석 가능성 | 낮음 | 높음 (Attention 시각화) |
| 파라미터 수 | ~50K | ~200-500K |
| 학습 시간 | 빠름 | 느림 (2-3배) |

### 2.3 성능 기대치
- 논문 기준 LSTM 대비 **10-20% RMSE 개선** 예상
- 특히 장기 예측 (12h, 24h)에서 더 큰 개선

---

## 3. 피처 구조 설계 (Static, Known, Unknown)

### 3.1 제주도 전력수요 예측 피처 분류

#### Static Covariates (시간에 따라 변하지 않는 특성)
```python
STATIC_CATEGORICALS = []  # 현재 프로젝트에서는 없음 (단일 지역)

STATIC_REALS = []  # 현재 프로젝트에서는 없음
```

#### Time-Varying Known (미래 값을 알 수 있는 특성)
```python
TIME_VARYING_KNOWN_CATEGORICALS = [
    'hour',           # 시간 (0-23)
    'dayofweek',      # 요일 (0-6)
    'month',          # 월 (1-12)
    'is_weekend',     # 주말 여부
    'is_holiday',     # 공휴일 여부
]

TIME_VARYING_KNOWN_REALS = [
    'hour_sin', 'hour_cos',           # 시간 주기 인코딩
    'dayofweek_sin', 'dayofweek_cos', # 요일 주기 인코딩
    'month_sin', 'month_cos',         # 월 주기 인코딩
    'day_of_year_sin', 'day_of_year_cos',
]
```

#### Time-Varying Unknown (현재까지만 알 수 있는 특성)
```python
TIME_VARYING_UNKNOWN_REALS = [
    # Target
    '전력거래량(MWh)',

    # 기상 변수
    'temp_mean',              # 기온
    'ground_temp_mean',       # 지면온도
    'soil_temp_5cm_mean',     # 지중온도
    'humidity_mean',          # 습도
    'wind_speed_mean',        # 풍속
    'irradiance_mean',        # 일사량
    'total_cloud_cover_mean', # 전운량
    'dewpoint_mean',          # 이슬점온도

    # 파생 변수
    'THI',                    # 불쾌지수
    'HDD', 'CDD',             # 난방/냉방 도일
    'wind_chill',             # 체감온도

    # Lag 피처
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
    'demand_ma_24h', 'demand_ma_168h',
    'temp_lag_1h', 'temp_lag_24h',

    # 외부 변수
    'ev_cumsum',              # 전기차 누적 등록
    'visitors_daily',         # 일별 관광객
]
```

### 3.2 피처 그룹 요약

| 그룹 | 개수 | 설명 |
|------|------|------|
| Static Categorical | 0 | - |
| Static Real | 0 | - |
| Known Categorical | 5 | 시간/요일/월/주말/공휴일 |
| Known Real | 8 | 주기 인코딩 |
| Unknown Real | ~25 | 기상, 파생, Lag, 외부 |
| **Total** | ~38 | 기존 LSTM과 동일 |

---

## 4. TFT 아키텍처 설계

### 4.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Processing                              │
├─────────────────────────────────────────────────────────────────┤
│  Static Encoder    │  Known Encoder    │  Unknown Encoder        │
│  (Embedding)       │  (Embedding+FC)   │  (FC)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Variable Selection Networks (VSN)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Static VSN   │  │ Encoder VSN  │  │ Decoder VSN  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM Encoder-Decoder                          │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   LSTM Encoder       │ →  │   LSTM Decoder       │          │
│  │   (Past Sequence)    │    │   (Future Sequence)  │          │
│  └──────────────────────┘    └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Static Enrichment Layer                             │
│         (Static context → LSTM outputs)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           Interpretable Multi-Head Attention                     │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Self-Attention over temporal dimension          │          │
│  │  (Masked for future in encoder)                  │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Position-wise Feed-Forward                       │
│                      + Layer Norm                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Quantile Outputs                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │ q=0.1  │  │ q=0.5  │  │ q=0.9  │  │  ...   │               │
│  └────────┘  └────────┘  └────────┘  └────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            Output: (batch, prediction_length, num_quantiles)
```

### 4.2 핵심 컴포넌트

#### 4.2.1 Gated Residual Network (GRN)
```python
class GatedResidualNetwork(nn.Module):
    """
    GRN: 정보 흐름을 게이트로 제어하는 네트워크

    GRN(a, c) = LayerNorm(a + GLU(η₁))
    where η₁ = W₁η₂ + b₁
          η₂ = ELU(W₂a + W₃c + b₂)
          GLU(x) = sigmoid(x[:, :d]) ⊙ x[:, d:]
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dropout=0.1, context_size=None):
        ...
```

#### 4.2.2 Variable Selection Network (VSN)
```python
class VariableSelectionNetwork(nn.Module):
    """
    VSN: 각 변수의 중요도를 학습하여 가중치 부여

    - 각 변수별 GRN 적용
    - Softmax로 변수 가중치 계산
    - 가중 합으로 최종 표현 생성
    """
    def __init__(self, input_sizes, hidden_size, dropout=0.1,
                 context_size=None):
        ...
```

#### 4.2.3 Interpretable Multi-Head Attention
```python
class InterpretableMultiHeadAttention(nn.Module):
    """
    해석 가능한 Multi-Head Attention

    - 일반 MHA와 다르게 모든 헤드가 Value를 공유
    - Attention weight가 직접 해석 가능
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        ...
```

### 4.3 하이퍼파라미터 설계

```python
TFT_CONFIG = {
    # 모델 크기
    'hidden_size': 64,           # LSTM/GRN hidden dimension
    'num_attention_heads': 4,    # Attention heads
    'dropout': 0.1,              # Dropout rate

    # LSTM 설정
    'lstm_layers': 2,            # LSTM layer 수

    # 시퀀스 길이
    'encoder_length': 48,        # 과거 48시간 (2일)
    'decoder_length': 24,        # 미래 24시간 (1일)

    # Quantile 출력
    'quantiles': [0.1, 0.5, 0.9],  # 10%, 50%, 90% 분위수

    # 학습 설정
    'learning_rate': 0.001,
    'batch_size': 64,
    'max_epochs': 100,
    'early_stopping_patience': 10,
}
```

---

## 5. 출력 레이어 설계 (Multi-horizon + Quantile)

### 5.1 출력 구조

```
Output Shape: (batch_size, prediction_length, num_quantiles)
             = (64, 24, 3)  # 24시간 예측, 3개 분위수

Horizons: [1, 2, 3, ..., 24] hours
Quantiles: [0.1, 0.5, 0.9]
```

### 5.2 Quantile Loss

```python
def quantile_loss(y_pred, y_true, quantiles):
    """
    Quantile Loss (Pinball Loss)

    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors))
    return torch.mean(torch.stack(losses))
```

### 5.3 기존 호환성
- Point estimate가 필요한 경우: `q=0.5` (median) 사용
- 기존 RMSE/MAE 메트릭과 호환

---

## 6. 구현 계획

### 6.1 파일 구조
```
src/models/
├── lstm.py              # 기존 LSTM (유지)
├── transformer.py       # TFT 구현 (신규)
│   ├── GatedResidualNetwork
│   ├── VariableSelectionNetwork
│   ├── InterpretableMultiHeadAttention
│   ├── TemporalFusionTransformer
│   └── QuantileLoss
└── __init__.py          # 모델 팩토리 업데이트
```

### 6.2 개발 순서
1. **1.2**: Variable Selection Network 구현
2. **1.3**: Temporal Self-Attention 구현
3. **1.4**: Encoder-Decoder 통합
4. **1.5**: Quantile Output Layer
5. **1.6**: 학습 파이프라인
6. **1.7**: 성능 비교 실험

---

## 7. 참고 자료

### 논문
- [TFT Paper (arXiv)](https://arxiv.org/abs/1912.09363)
- [Google Research](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/)

### 라이브러리
- [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting) - TFT 참조 구현
- [darts](https://github.com/unit8co/darts) - 시계열 예측 라이브러리

### 코드 참조
- 기존 LSTM: `src/models/lstm.py`
- 피처 정의: `src/features/time_features.py`, `src/features/weather_features.py`
