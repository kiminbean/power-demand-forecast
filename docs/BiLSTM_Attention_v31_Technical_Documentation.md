# BiLSTM+Attention v3.1 Technical Documentation

> SMP (System Marginal Price) Prediction Model for RE-BMS
>
> Version: 3.1.0 | Last Updated: 2024-12-24

---

## 1. Executive Summary

BiLSTM+Attention v3.1은 제주도 전력시장의 **계통한계가격(SMP)**을 예측하는 딥러닝 모델입니다. 양방향 LSTM과 Multi-Head Self-Attention 메커니즘을 결합하여 시계열 패턴을 학습하고, Quantile Regression을 통해 예측 불확실성을 정량화합니다.

### Key Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **MAPE** | 7.83% | Mean Absolute Percentage Error |
| **RMSE** | 12.02원 | Root Mean Square Error |
| **R²** | 0.736 | Coefficient of Determination |
| **Coverage (80%)** | 89.4% | Prediction Interval Coverage |
| **Parameters** | 249,952 | Total Trainable Parameters |
| **Features** | 22 | Input Feature Dimensions |

---

## 2. Model Architecture

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Layer (48 hours × 22 features)         │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BiLSTM Encoder (2 layers)                    │
│                    hidden_size=64, bidirectional=True           │
│                    Output: (batch, 48, 128)                     │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Multi-Head Self-Attention (4 heads)             │
│                 d_model=128, d_k=32                             │
│                 Output: (batch, 48, 128) + Attention Weights    │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Residual Connection + LayerNorm              │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Last Timestep Selection                      │
│                    (batch, 128)                                 │
└─────────────────────────────────────────────────────────────────┘
                          ↓                    ↓
        ┌─────────────────────────┐  ┌─────────────────────────┐
        │    Point Prediction     │  │   Quantile Predictions  │
        │    FC → ReLU → FC       │  │   q10, q50, q90 heads   │
        │    Output: (batch, 24)  │  │   Output: (batch, 24)×3 │
        └─────────────────────────┘  └─────────────────────────┘
```

### 2.2 Component Details

#### BiLSTM Encoder
```python
nn.LSTM(
    input_size=22,        # Feature dimensions
    hidden_size=64,       # Hidden state size
    num_layers=2,         # Stacked LSTM layers
    dropout=0.3,          # Dropout between layers
    bidirectional=True,   # Forward + Backward
    batch_first=True
)
# Output shape: (batch, seq_len, 128)  # 64×2 for bidirectional
```

#### Multi-Head Self-Attention
```python
class StableAttention:
    d_model = 128        # Input dimension (BiLSTM output)
    n_heads = 4          # Number of attention heads
    d_k = 32             # Key/Query dimension per head
    scale = sqrt(d_k)    # Scaling factor

    # Q, K, V projections with Xavier initialization (gain=0.1)
    # Softmax attention with dropout
```

#### Output Heads
- **Point Prediction**: `FC(128→64) → ReLU → Dropout(0.3) → FC(64→24)`
- **Quantile Heads**: 3 parallel heads for q10, q50, q90

---

## 3. Feature Engineering

### 3.1 Feature Categories (22 total)

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| **Base Price** | smp_mainland, smp_jeju, smp_max, smp_min | 4 | Raw SMP values |
| **Time Cyclical** | hour_sin/cos, dow_sin/cos, is_weekend, month_sin/cos | 7 | Cyclic time encoding |
| **Season/Peak** | is_summer, is_winter, peak_morning, peak_evening, off_peak | 5 | Temporal patterns |
| **Statistical** | smp_ma24, smp_std24, smp_diff, smp_range | 4 | Rolling statistics |
| **Lag Features** | smp_lag_24h, smp_lag_168h | 2 | Historical references |

### 3.2 Feature Transformation

```python
# Cyclical encoding (prevents discontinuity)
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

# Rolling statistics
smp_ma24 = SMP.rolling(24).mean()    # 24-hour moving average
smp_std24 = SMP.rolling(24).std()    # 24-hour volatility

# Lag features
smp_lag_24h = SMP.shift(24)          # Same hour yesterday
smp_lag_168h = SMP.shift(168)        # Same hour last week
```

### 3.3 Normalization

- **Features**: StandardScaler (z-score normalization)
- **Target**: StandardScaler (stored separately for denormalization)

---

## 4. Training Pipeline

### 4.1 Data Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Training Period | 2022-01-01 ~ 2024-12-31 | 3 years of hourly data |
| Input Sequence | 48 hours | Look-back window |
| Output Sequence | 24 hours | Prediction horizon |
| Batch Size | 64 | Training batch size |

### 4.2 Loss Function

```python
# Combined Loss = MSE Weight × Huber Loss + Quantile Weight × Quantile Loss

class CombinedLossV31:
    mse_weight = 0.5
    quantile_weight = 0.5

    # Huber Loss (Smooth L1) - robust to outliers
    mse_loss = SmoothL1Loss(predictions['point'], targets)

    # Quantile Loss - uncertainty estimation
    quantile_loss = Σ max(q × error, (q-1) × error) for q in [0.1, 0.5, 0.9]

    total_loss = 0.5 × mse_loss + 0.5 × quantile_loss
```

### 4.3 Training Techniques

| Technique | Configuration | Purpose |
|-----------|---------------|---------|
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate |
| **Scheduler** | ReduceLROnPlateau (patience=10, factor=0.5) | Learning rate decay |
| **Early Stopping** | patience=25 | Prevent overfitting |
| **Gradient Clipping** | max_norm=1.0 | Training stability |
| **Noise Injection** | std=0.02, prob=0.5 | Regularization |
| **Dropout** | 0.3 | Regularization |

### 4.4 Training Progress

```
Epoch 001/150 | Loss: 0.8234 | Val MAPE: 15.23%
Epoch 050/150 | Loss: 0.2156 | Val MAPE: 9.45%
Epoch 100/150 | Loss: 0.1423 | Val MAPE: 8.12%
Epoch 127/150 | Loss: 0.1198 | Val MAPE: 7.83% ← Best Model Saved
```

---

## 5. Prediction Output

### 5.1 Output Format

```python
{
    'times': [datetime, ...],      # 24 prediction timestamps
    'q10': [float, ...],           # Lower bound (10th percentile)
    'q50': [float, ...],           # Point prediction (median)
    'q90': [float, ...],           # Upper bound (90th percentile)
    'model_used': 'v3.1',
    'interval_width': float,       # Average q90-q10
    'coverage': 89.4,              # 80% interval coverage
    'mape': 7.83,
    'attention': [[float, ...]]    # Optional: attention weights
}
```

### 5.2 Example Prediction

```
Hour    Q10 (원)   Q50 (원)   Q90 (원)   Interval Width
────────────────────────────────────────────────────
00:00    85.2      95.3      108.4      23.2
06:00    78.5      88.1       99.7      21.2
12:00   102.3     115.8      132.5      30.2
18:00   118.7     135.2      156.8      38.1
```

---

## 6. Model Files

### 6.1 File Structure

```
models/smp_v3/
├── smp_v3_model.pt       # Model checkpoint (1.0MB)
│   ├── model_state_dict  # Trained weights
│   ├── model_kwargs      # Architecture parameters
│   ├── config            # Training configuration
│   ├── metrics           # Evaluation metrics
│   ├── target_mean       # Target denormalization mean
│   └── target_std        # Target denormalization std
├── smp_v3_scaler.npy     # Feature scaler (1.3KB)
│   ├── feature_scaler_mean
│   ├── feature_scaler_scale
│   └── feature_names
└── smp_v3_metrics.json   # Metrics summary
```

### 6.2 Model Loading

```python
from src.smp.models.smp_predictor import SMPPredictor

# Initialize with v3.1 model
predictor = SMPPredictor(use_advanced=True)

# Check readiness
if predictor.is_ready():
    result = predictor.predict_24h()
    print(f"Q50 Mean: {np.mean(result['q50']):.2f} 원/kWh")
```

---

## 7. Integration Points

### 7.1 API Endpoint

```
GET /api/v1/smp-forecast

Response:
{
    "q10": [85.2, 82.1, ...],
    "q50": [95.3, 92.4, ...],
    "q90": [108.4, 105.6, ...],
    "model_used": "BiLSTM+Attention v3.1",
    "confidence": 0.894,
    "created_at": "2024-12-24T10:00:00Z"
}
```

### 7.2 AI Bidding Optimizer Integration

```python
# SMP predictions are used for bid optimization
from api.ai_bidding_optimizer import AIBiddingOptimizer

optimizer = AIBiddingOptimizer()
# Uses SMP q10, q50, q90 for clearing probability calculation
result = optimizer.optimize_hourly_bids(capacity_mw=50, risk_level='moderate')
```

---

## 8. Performance Comparison

| Model Version | MAPE | R² | Coverage | Parameters |
|---------------|------|-----|----------|------------|
| v1.0 LSTM | 12.5% | 0.52 | - | 156K |
| v2.1 Advanced | 10.7% | 0.59 | 82.5% | 201K |
| **v3.1 BiLSTM+Att** | **7.83%** | **0.74** | **89.4%** | **250K** |

### Improvements in v3.1
- BiLSTM + Stable Attention mechanism
- Huber Loss + Quantile Loss combination
- Noise Injection for regularization
- Gradient Clipping for stability
- Xavier initialization for attention weights

---

## 9. Limitations & Future Work

### Current Limitations
1. **External Factors**: Does not incorporate real-time fuel prices or demand forecasts
2. **Extreme Events**: May underperform during unusual market conditions
3. **Lag Time**: Uses historical data, no real-time market signals

### Future Improvements
- [ ] Incorporate weather forecasts (temperature, solar irradiance)
- [ ] Add fuel cost features from crawlers
- [ ] Implement Transformer architecture for longer sequences
- [ ] Online learning for model adaptation

---

## 10. References

- **Data Source**: EPSIS (전력통계정보시스템) - 5 years hourly SMP data
- **Framework**: PyTorch 2.x
- **Hardware**: Apple M1 MPS / NVIDIA CUDA

---

*Document generated for RE-BMS v6.1*
*BiLSTM+Attention v3.1 SMP Prediction Model*
