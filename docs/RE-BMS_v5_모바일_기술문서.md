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
| 지표 | 값 | 설명 |
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

## 9. 실행 방법

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

## 10. v6 데스크톱과 차이점

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
