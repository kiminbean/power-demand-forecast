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

## 7. 실행 방법

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

## 8. 주요 컴포넌트 설명

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

## 9. 배포 아키텍처

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

## 10. 브랜딩

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
