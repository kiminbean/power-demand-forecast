# RE-BMS v6.0 기술 문서 (간략 버전)

## 1. 시스템 개요

**RE-BMS (Renewable Energy Bidding Management System)**는 제주도 재생에너지 전력 거래를 위한 입찰관리 시스템입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                        RE-BMS v6.0                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Desktop Web   │   Mobile Web    │      FastAPI Backend    │
│   (port 8600)   │   (port 8601)   │      (port 8506)        │
│   React + Vite  │   React + Vite  │   Python + PyTorch      │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 2. 핵심 기능 (3개)

| 기능 | 설명 | 사용자 가치 |
|------|------|------------|
| **SMP 예측** | AI가 24시간 전력가격 예측 | 최적 입찰 시점 파악 |
| **입찰 관리** | 10구간 입찰가격 설정 | KPX 시장 수익 최대화 |
| **정산** | 수익/불균형 정산 현황 | 실적 모니터링 |

---

## 3. 기술 스택

### 프론트엔드 (web-v6)
```
React 18 + TypeScript + Vite + Tailwind CSS + Recharts
```

### 백엔드 (api/)
```
FastAPI + Python 3.13 + PyTorch (LSTM-Attention 모델)
```

### 배포
```
Docker + Nginx (Basic Auth 인증)
```

---

## 4. 폴더 구조 (핵심만)

```
web-v6/
├── src/
│   ├── pages/           ← 7개 페이지
│   │   ├── Dashboard.tsx      # 대시보드
│   │   ├── SMPPrediction.tsx  # SMP 예측 ⭐
│   │   ├── Bidding.tsx        # 입찰 관리 ⭐
│   │   ├── Settlement.tsx     # 정산 ⭐
│   │   ├── Portfolio.tsx      # 포트폴리오
│   │   ├── Map.tsx            # 제주 지도
│   │   └── Analysis.tsx       # 분석
│   │
│   ├── components/      ← 재사용 컴포넌트
│   │   ├── Layout/      # Header, Sidebar
│   │   ├── Charts/      # SMPChart, PowerSupplyChart
│   │   └── Cards/       # KPICard
│   │
│   ├── services/api.ts  ← API 통신
│   ├── hooks/useApi.ts  ← 데이터 훅
│   └── types/index.ts   ← TypeScript 타입
```

---

## 5. API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /api/v1/smp-forecast` | SMP 24시간 예측 (q10/q50/q90) |
| `GET /api/v1/market-status` | DAM/RTM 시장 상태 |
| `GET /api/v1/resources` | 제주 발전소 20개 목록 |
| `GET /api/v1/health` | API 상태 확인 |

**응답 예시 (SMP 예측):**
```json
{
  "q10": [62.67, 75.49, ...],   // 하한 (10%)
  "q50": [77.95, 71.16, ...],   // 중앙값 (예측)
  "q90": [93.23, 86.83, ...],   // 상한 (90%)
  "confidence": 0.92
}
```

---

## 6. 데이터 흐름

```
EPSIS (KPX) ─→ 크롤러 ─→ CSV 저장 ─→ LSTM 모델 ─→ API ─→ React UI
   │              │           │            │          │
   │          매시간       data/smp/    models/smp/   │
   │          자동수집                               실시간 표시
```

---

## 7. 딥러닝 모델 (SMP 예측)

### 모델 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                  LSTM-Attention v3.1                        │
├─────────────────────────────────────────────────────────────┤
│  Input (24h × 22 features)                                  │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │  BiLSTM Layer 1 (128 hidden × 2)        │               │
│  │  BiLSTM Layer 2 (128 hidden × 2)        │               │
│  │  BiLSTM Layer 3 (128 hidden × 2)        │               │
│  └─────────────────────────────────────────┘               │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │  Stable Attention Mechanism             │               │
│  └─────────────────────────────────────────┘               │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │  Quantile Output (q10, q50, q90)        │               │
│  └─────────────────────────────────────────┘               │
│         ↓                                                   │
│  Output (24h SMP 예측)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 모델 정보
| 항목 | 값 |
|------|-----|
| **모델명** | LSTM-Attention v3.1 |
| **아키텍처** | BiLSTM + Stable Attention |
| **파라미터 수** | 249,952개 |
| **입력 피처** | 22개 |
| **학습 장치** | MPS (Apple Silicon) |
| **데이터 소스** | EPSIS (2024-01 ~ 2024-12) |

### 성능 지표
| 지표 | 값 | 의미 |
|------|-----|------|
| **MAPE** | 7.83% | 평균 절대 백분율 오차 (낮을수록 좋음) |
| **MAE** | 8.93 원 | 평균 절대 오차 |
| **RMSE** | 12.02 원 | 평균 제곱근 오차 |
| **R²** | 0.736 | 결정 계수 (1에 가까울수록 좋음) |
| **Coverage** | 89.4% | 80% 신뢰구간 커버리지 |

### 입력 피처 (22개)
```
시간적 피처:    hour, dayofweek, month, is_weekend, is_holiday
SMP 피처:      smp_lag_1h ~ smp_lag_24h (24개 래그)
기상 피처:     temperature, wind_speed, solar_radiation
수급 피처:     demand, supply, renewable_ratio
```

### 모델 개선 기법
| 기법 | 효과 |
|------|------|
| BiLSTM | 양방향 시계열 패턴 학습 |
| Stable Attention | 중요 시점 가중치 부여 |
| Huber Loss | 이상치에 강건한 학습 |
| Quantile Loss | 신뢰구간 (q10/q50/q90) 출력 |
| Noise Injection | 과적합 방지 |
| Gradient Clipping | 학습 안정화 |

### 학습 설정
```python
{
    "input_hours": 24,       # 입력: 과거 24시간
    "output_hours": 24,      # 출력: 미래 24시간
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "bidirectional": True,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.001,
    "patience": 25           # Early Stopping
}
```

---

## 8. 딥러닝 모델 (전력 수요 예측)

### 모델 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│              Power Demand LSTM Model                        │
├─────────────────────────────────────────────────────────────┤
│  Input (168h × 17~18 features)                              │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │  LSTM Layer 1 (128 hidden)              │               │
│  │  LSTM Layer 2 (128 hidden)              │               │
│  └─────────────────────────────────────────┘               │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │  Fully Connected Layer                   │               │
│  └─────────────────────────────────────────┘               │
│         ↓                                                   │
│  Output (1h 수요 예측)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 모델 비교 (2종)
| 항목 | Demand Only | Weather Full |
|------|-------------|--------------|
| **입력 피처** | 17개 | 18개 (기상 포함) |
| **MAPE** | 6.47% | **6.06%** ⭐ |
| **R²** | 0.846 | **0.865** ⭐ |
| **RMSE** | 40.63 MW | **38.02 MW** ⭐ |
| **MAE** | 30.41 MW | **27.81 MW** ⭐ |

### 주요 성능 지표 (Weather Full 모델)
| 지표 | 값 | 의미 |
|------|-----|------|
| **MAPE** | 6.06% | 평균 절대 백분율 오차 |
| **R²** | 0.865 | 결정 계수 (설명력 86.5%) |
| **RMSE** | 38.02 MW | 평균 제곱근 오차 |
| **MAE** | 27.81 MW | 평균 절대 오차 |
| **sMAPE** | 5.89% | 대칭 평균 백분율 오차 |
| **CV-RMSE** | 7.66% | 변동계수 RMSE |

### 입력 피처 (17개 + 기상 1개)
```
수요 피처:     power_demand, demand_lag_1/24/168h
시간 피처:     hour_sin/cos, dayofweek_sin/cos, month_sin/cos
특수일 피처:   is_weekend, is_holiday
통계 피처:     demand_ma_6h/24h, demand_std_24h
차분 피처:     demand_diff_1h/24h
기상 피처:     temperature (Weather Full만)
```

### 학습 설정
```python
{
    "seq_length": 168,       # 입력: 과거 7일 (168시간)
    "horizon": 1,            # 출력: 1시간 후 예측
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "patience": 15           # Early Stopping
}
```

---

## 9. 실행 방법

### 개발 모드
```bash
cd web-v6
npm install
npm run dev        # http://localhost:8508
```

### Docker 배포
```bash
docker compose -f docker/docker-compose.v6.yml up -d

# Desktop: http://localhost:8600
# Mobile:  http://localhost:8601
# 로그인:  admin / rebms2025
```

---

## 10. 주요 컴포넌트 설명

### SMP 예측 차트
```tsx
// SMPPrediction.tsx 핵심 로직
const { data: forecast } = useSMPForecast();  // API 호출

<AreaChart data={chartData}>
  <Area dataKey="q90" />  {/* 상한 */}
  <Area dataKey="q50" />  {/* 예측값 (노란색) */}
  <Area dataKey="q10" />  {/* 하한 */}
</AreaChart>
```

### 입찰 관리
```tsx
// Bidding.tsx 핵심 로직
const [segments, setSegments] = useState<BidSegment[]>();  // 10구간

// AI 최적화 버튼
const handleOptimize = () => {
  // SMP 예측 기반으로 구간별 가격 자동 설정
};
```

### 테마 (라이트/다크)
```tsx
// ThemeContext.tsx
const { isDark, toggleTheme } = useTheme();

// 로고 자동 전환
<img src={isDark ? '/logo-light.png' : '/logo-dark.png'} />
```

---

## 11. 배포 아키텍처

```
                    ┌─────────────────┐
                    │   사용자 브라우저   │
                    └────────┬────────┘
                             │ HTTP
                    ┌────────▼────────┐
                    │     Nginx       │
                    │  (Basic Auth)   │
                    │   port 8600     │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │  정적파일   │     │ /api/*    │     │  /health  │
    │  React앱   │     │  프록시    │     │  상태체크  │
    └───────────┘     └─────┬─────┘     └───────────┘
                            │
                    ┌───────▼───────┐
                    │   FastAPI     │
                    │   port 8506   │
                    └───────────────┘
```

---

## 12. 브랜딩

- **로고**: eXeco (진한 파란색 #04265E)
- **테마**: 라이트/다크 모드 지원
- **폰트**: Pretendard

---

## 버전 정보

| 버전 | 날짜 | 주요 변경 |
|-----|------|----------|
| v6.0.0 | 2025-12-20 | React 데스크톱 웹앱 출시 |
| v5.0.0 | 2025-12-20 | 모바일 최적화 버전 추가 |

---

*© 2025 eXeco - RE-BMS v6.0*
