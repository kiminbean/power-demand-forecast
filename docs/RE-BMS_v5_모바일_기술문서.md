# RE-BMS v5.0 모바일 기술 문서

## 1. 시스템 개요

**RE-BMS Mobile**은 v6 데스크톱 버전을 모바일에 최적화한 웹앱입니다.

```
┌─────────────────────────────────────────────────┐
│              RE-BMS Mobile v5.0                 │
├─────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   SMP   │  │   입찰   │  │   정산   │        │
│  │   예측   │  │   관리   │  │         │        │
│  └─────────┘  └─────────┘  └─────────┘        │
│         [ 하단 네비게이션 바 ]                   │
└─────────────────────────────────────────────────┘
```

---

## 2. 핵심 기능 (3개만)

| 기능 | 설명 | 모바일 최적화 |
|------|------|-------------|
| **SMP 예측** | 24시간 전력가격 예측 | 접이식 상세 테이블 |
| **입찰 관리** | 10구간 가격 설정 | 터치 친화적 입력 |
| **정산** | 수익/불균형 현황 | 카드형 요약 UI |

---

## 3. 기술 스택

```
프론트엔드: React 18 + TypeScript + Vite + Tailwind CSS
차트:       Recharts (모바일 최적화)
배포:       Docker + Nginx
인증:       Basic Auth
```

---

## 4. 폴더 구조

```
web-v5/
├── src/
│   ├── pages/                 ← 3개 페이지만
│   │   ├── SMPPrediction.tsx  # SMP 예측
│   │   ├── Bidding.tsx        # 입찰 관리
│   │   └── Settlement.tsx     # 정산
│   │
│   ├── components/
│   │   └── MobileLayout.tsx   # 하단 네비게이션
│   │
│   ├── services/api.ts        # API 통신
│   ├── hooks/useApi.ts        # 데이터 훅
│   └── contexts/ThemeContext.tsx
│
├── public/
│   ├── manifest.json          # PWA 설정
│   ├── logo-dark.png
│   └── logo-light.png
│
├── Dockerfile
└── nginx.conf
```

---

## 5. 모바일 UI 특징

### 하단 네비게이션
```tsx
// MobileLayout.tsx
const navItems = [
  { path: '/smp', label: 'SMP 예측', icon: TrendingUp },
  { path: '/bidding', label: '입찰', icon: Gavel },
  { path: '/settlement', label: '정산', icon: Receipt },
];

<nav className="fixed bottom-0 left-0 right-0">
  {navItems.map(...)}
</nav>
```

### 접이식 패널
```tsx
// 상세 정보를 접었다 펼 수 있음
const [showDetails, setShowDetails] = useState(false);

<button onClick={() => setShowDetails(!showDetails)}>
  {showDetails ? <ChevronUp /> : <ChevronDown />}
</button>
```

### 터치 최적화
```css
/* 큰 터치 영역 */
.btn-primary {
  padding: 12px 16px;  /* py-3 px-4 */
  min-height: 48px;
}

/* iOS 줌 방지 */
input, select {
  font-size: 16px !important;
}
```

---

## 6. PWA 지원

### manifest.json
```json
{
  "name": "RE-BMS Mobile",
  "short_name": "RE-BMS",
  "display": "standalone",
  "orientation": "portrait",
  "theme_color": "#04265E"
}
```

### 홈 화면 추가
- iOS: Safari → 공유 → 홈 화면에 추가
- Android: Chrome → 메뉴 → 앱 설치

---

## 7. API 연동

v5 모바일은 v6와 동일한 API 사용:

| 엔드포인트 | 용도 |
|-----------|------|
| `/api/v1/smp-forecast` | SMP 예측 데이터 |
| `/api/v1/market-status` | 시장 상태 |
| `/api/v1/settlements/recent` | 정산 내역 |
| `/api/v1/health` | API 상태 |

---

## 8. 딥러닝 모델 성능

### 모델 정보
| 항목 | 값 |
|------|-----|
| **모델명** | LSTM-Attention v3.1 |
| **아키텍처** | BiLSTM + Stable Attention |
| **파라미터 수** | 249,952개 |
| **입력 피처** | 22개 |
| **학습 장치** | MPS (Apple Silicon) |

### 성능 지표

> **⚠️ 무효 수치 — 2026-11-05 재평가**: 아래 수치는 스케일러 누수(시간 분할 전 스케일러 fit)로 부풀려져 신뢰할 수 없다. 현재 생산 모델은 v4.1이며 시간외(walk-forward) 재평가 지표는 **MAPE 8.76% · MAE 10.53원/kWh · R² 0.463 · 80% 구간 커버리지 68.6%**다. 원본: `models/smp_v4/smp_v41_metrics.json`, 경위: `docs/2026-11-02-forecast-upgrade-status.md`.

| 지표 | 값(구버전·무효) | 설명 |
|------|-----|------|
| **MAPE** | 7.83% | 평균 절대 백분율 오차 |
| **MAE** | 8.93 원 | 평균 절대 오차 |
| **RMSE** | 12.02 원 | 평균 제곱근 오차 |
| **R²** | 0.736 | 결정 계수 |
| **Coverage 80%** | 89.4% | 신뢰구간 커버리지 |

### 모델 개선 기법
```
✓ BiLSTM + Stable Attention
✓ Huber Loss + Quantile Loss
✓ Noise Injection (과적합 방지)
✓ Gradient Clipping (학습 안정화)
```

### 학습 설정
```python
config = {
    "input_hours": 24,      # 입력: 24시간
    "output_hours": 24,     # 출력: 24시간 예측
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "bidirectional": True,
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.001,
    "patience": 25          # Early Stopping
}
```

---

## 9. 전력 수요 예측 모델 (API)

> **2026-11-07 갱신:** 이 앱(v5)은 수요 예측 API를 호출하지 않는다. 아래는 같은 저장소의 API(`api/`, :8001·:8506)가 쓰는 모델이다.
>
> 이전 판의 6.06%·6.47%는 2025-12 학습 리포트 값이다. 같은 기간(2023-07~2024-12)을 API와 같은 경로로 다시 채점하면 9.78%·12.44%여서 재현되지 않았다. 그래서 2026-11-05 재학습 모델로 교체했다(`docs/benchmark-2026-11-06/rescore_old_lstm_out.json`).

### 모델 정보
| 항목 | 값 |
|------|-----|
| **기본 모델** | AR + GBM 잔차 보정 (`ar_gbm`, 2026-11-06부터 API 기본값). 시간대별 선형 AR이 틀린 양을 HistGradientBoosting이 보정한다. 최근 337시간 이상 필요 |
| **선택형 모델** | Power Demand LSTM (`demand_only` 17 피처, `weather_full` 18 피처). 2층 LSTM(은닉 128), 입력 168시간, 출력 = 직전 1시간 수요 + 보정량(잔차 타깃) |
| **예측 대상** | KPX 제주 전력거래량, 1시간 앞 |
| **학습 장치** | LSTM은 MPS(Apple Silicon), ar_gbm은 CPU |

### 모델 비교 (MAPE / R²)
| 모델 | 2025년 실측 (학습에 안 씀) | 고정 test 2023-07~2024-12 |
|------|------|------|
| **ar_gbm (기본)** ⭐ | **4.67% / 0.929** | **3.63% / 0.954** |
| LSTM demand_only | 5.22% / 0.910 | 4.16% / 0.939 |
| LSTM weather_full | – (2025년 기온 자료 없음) | 4.26% / 0.936 |
| 기준선: 직전 1시간 값 | 6.16% / 0.876 | 5.28% / 0.902 |

### 성능 지표 (ar_gbm, 2025년 실측 8,760시간)
| 지표 | 값 |
|------|-----|
| **MAPE** | 4.67% |
| **R²** | 0.929 |
| **RMSE** | 29.14 MWh |
| **MAE** | 21.85 MWh |

근거: `docs/benchmark-2026-11-06/compare_default_2025_out.json` · `docs/2026-11-06-demand-retrain-raw-target.md` 3·12절

---

## 10. 실행 방법

### 개발 모드
```bash
cd web-v5
npm install
npm run dev    # http://localhost:8507
```

### Docker 배포
```bash
docker compose -f docker/docker-compose.v6.yml up -d mobile

# 접속: http://localhost:8601
# 로그인: admin / rebms2025
```

---

## 11. v6 데스크톱과 차이점

| 항목 | v6 데스크톱 | v5 모바일 |
|------|-----------|----------|
| **페이지 수** | 7개 | 3개 |
| **네비게이션** | 좌측 사이드바 | 하단 탭바 |
| **레이아웃** | 가로 그리드 | 세로 스크롤 |
| **차트 크기** | 400px | 180~200px |
| **상세정보** | 항상 표시 | 접이식 |
| **포트** | 8600 | 8601 |

---

## 배포 URL

| 환경 | URL |
|------|-----|
| 개발 | http://localhost:8507 |
| 운영 | http://localhost:8601 |

---

*© 2025 eXeco - RE-BMS Mobile v5.0*
