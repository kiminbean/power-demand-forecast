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

> **⚠️ 무효 수치 — 2026-11-05 재평가**: 아래 수치는 스케일러 누수(시간 분할 전 스케일러 fit)로 부풀려져 신뢰할 수 없다. 현재 생산 모델은 v4.1이며 시간외(walk-forward) 재평가 지표는 **MAPE 8.76% · MAE 10.53원/kWh · R² 0.463 · 80% 구간 커버리지 68.6%**다. 원본: `models/smp_v4/smp_v41_metrics.json`, 경위: `docs/2026-11-02-forecast-upgrade-status.md`.

| 지표 | 값(구버전·무효) | 의미 |
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

> **2026-11-07 갱신:**
> - 2026-11-06부터 API 기본 수요 모델은 AR + GBM 잔차 보정(`ar_gbm`)이다. 아래 LSTM은 선택형이다(`model_type`: `demand_only`·`weather_full`).
> - v6 웹은 수요 예측 API를 호출하지 않는다. 공개 프록시 허용 목록에도 없다.
> - 이전 판의 6.06%·6.47%는 2025-12 학습 리포트 값이다. 같은 기간을 API와 같은 경로로 다시 채점하면 9.78%·12.44%여서 재현되지 않았다. 그래서 2026-11-05 재학습 모델로 교체했다(`docs/benchmark-2026-11-06/rescore_old_lstm_out.json`).
> - 재학습 모델의 출력은 "직전 1시간 수요 + LSTM 보정량"(잔차 타깃)이다.

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

### 모델 비교 (원본 목표, API와 같은 추론 경로)
| 항목 | ar_gbm (API 기본) | LSTM Demand Only | LSTM Weather Full | 기준선: 직전 1시간 값 |
|------|------|------|------|------|
| **입력** | 과거 수요 시차·요일(AR), 그 잔차를 GBM이 보정 | 17개 피처 | 18개 (기온 포함) | – |
| **2025년 실측 MAPE / R²** | **4.67% / 0.929** ⭐ | 5.22% / 0.910 | – (2025년 기온 자료 없음) | 6.16% / 0.876 |
| **고정 test 2023-07~2024-12 MAPE / R²** | **3.63% / 0.954** ⭐ | 4.16% / 0.939 | 4.26% / 0.936 | 5.28% / 0.902 |
| **MAE (고정 test)** | 17.32 MWh | 19.96 MWh | 20.46 MWh | 25.38 MWh |
| **RMSE (고정 test)** | 23.43 MWh | 26.90 MWh | 27.54 MWh | 34.12 MWh |

### 주요 성능 지표 (ar_gbm, 2025년 실측 8,760시간)
| 지표 | 값 | 의미 |
|------|-----|------|
| **MAPE** | 4.67% | 평균 절대 백분율 오차 |
| **R²** | 0.929 | 결정 계수 (설명력 92.9%) |
| **RMSE** | 29.14 MWh | 평균 제곱근 오차 |
| **MAE** | 21.85 MWh | 평균 절대 오차 |

근거: `docs/benchmark-2026-11-06/compare_default_2025_out.json` · `docs/2026-11-06-demand-retrain-raw-target.md` 3·12절

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
| v6.1.0 | 2025-12-22 | 입찰 워크플로우 및 KPX 시뮬레이션 추가 |
| v6.0.0 | 2025-12-20 | React 데스크톱 웹앱 출시 |
| v5.0.0 | 2025-12-20 | 모바일 최적화 버전 추가 |

---

## 13. v6.1.0 신규 기능: 입찰 워크플로우

### 입찰 제출 프로세스

```
┌──────────────────────────────────────────────────────────────────┐
│                    입찰 워크플로우 (v6.1.0)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1. 입찰 작성]                                                   │
│       │                                                          │
│       ▼                                                          │
│  [2. 입찰 제출] ───→ BidReviewModal (내부 검토)                   │
│       │                    │                                     │
│       │              ┌─────┴─────┐                               │
│       │              │           │                               │
│       │         [승인]       [반려]                              │
│       │              │           │                               │
│       ▼              ▼           ▼                               │
│  [3. KPX 제출] ◄── 승인됨     작성 중으로                         │
│       │                       되돌아감                           │
│       ▼                                                          │
│  [4. KPX 시뮬레이션]                                              │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────┐                        │
│  │  Merit Order 매칭 시뮬레이션         │                        │
│  │  - 공급/수요 곡선 시각화             │                        │
│  │  - 청산가격(SMP) 결정               │                        │
│  │  - 낙찰 결과 확인                   │                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 주요 컴포넌트

| 컴포넌트 | 파일 경로 | 설명 |
|---------|----------|------|
| BidReviewModal | `components/Modals/BidReviewModal.tsx` | 내부 검토 및 승인 모달 |
| KPXSimulation | `pages/KPXSimulation.tsx` | KPX 시장 시뮬레이션 페이지 |

### 입찰 상태 (BidStatus)

| 상태 | 설명 |
|------|------|
| `draft` | 작성 중 |
| `pending_review` | 검토 대기 |
| `approved` | 승인됨 (KPX 제출 가능) |
| `submitted_kpx` | KPX 제출됨 |
| `matched` | 매칭 완료 |
| `rejected` | 반려됨 |

### KPX 시장 시뮬레이션

Merit Order 방식의 전력시장 매칭 과정을 시각적으로 시뮬레이션:

1. **입찰 수집**: 공급자/수요자 입찰 데이터 수집
2. **가격 정렬**: 공급은 가격 오름차순, 수요는 가격 내림차순
3. **수급 매칭**: 공급 곡선과 수요 곡선의 교차점 탐색
4. **청산가격 결정**: 균형가격(SMP) 결정
5. **결과 표시**: 낙찰/탈락 결과 및 예상 수익 표시

---

*© 2025 eXeco - RE-BMS v6.1*
