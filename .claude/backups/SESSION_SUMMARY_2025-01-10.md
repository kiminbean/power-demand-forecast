# 세션 요약 (2025-01-10)

## 세션 개요
- **날짜**: 2025-01-10
- **주요 작업**: 모델 버전 이력 정리, 이미지 생성, GitHub 설정

---

## 1. 완료된 작업

### 1.1 Power Demand Forecast LSTM Colab 노트북 확인
- **파일**: `notebooks/Power_Demand_Forecast_LSTM_Production_Colab.ipynb`
- **모델**: weather_full LSTM (MAPE 6.06%, R² 0.865)
- **상태**: 이전 세션에서 완료, 커밋됨

### 1.2 SMP 예측 모델 버전 이력 이미지 생성
- **요청**: "smp 예측 모델 버전별 log 확인해서 학습 일자 MAPE, R2, 학습내용. 이미지 형태로 만들어줘"
- **결과 파일**: `results/smp_model_version_history.png`
- **내용**: 24개 SMP 모델 버전 (v1.0 ~ v5.0) 정리

#### SMP 모델 버전 요약

| Phase | 버전 | MAPE | R² | 결과 |
|-------|------|------|-----|------|
| 초기 개발 | v1.0 | 19.69% | 0.275 | Baseline |
| 초기 개발 | v2.0 Advanced | 10.68% | 0.592 | 5-fold CV |
| 초기 개발 | v3.1 | 7.83% | 0.736 | BiLSTM+Attention |
| 초기 개발 | v3.2 Optuna | 7.17% | 0.771 | Optuna 튜닝 |
| 외부 데이터 | v3.3~v3.7 | 11~24% | <0.5 | ❌ FAILED |
| **Gradient Boosting** | **v3.12 Stacking** | **5.25%** | **0.834** | **⭐ BEST** |
| Production | v3.19 +EMA | 5.28% | 0.827 | ⭐ 배포됨 |
| 아키텍처 | v4.0~v5.0 | 8~23% | <0.6 | ❌ FAILED |

### 1.3 SMP 모델 학습 코드 위치 안내
- **디렉토리**: `src/smp/models/`
- **주요 파일**:
  - `train_smp_v3_12_stacking.py` - 최고 성능 모델
  - `train_smp_v3_19_recent.py` - Production 모델
  - `smp_catboost_predictor.py` - 실시간 추론 클래스

### 1.4 GitHub 저장소 Private 변경
- **요청**: "power-demand-forecast 깃허브에 프라이빗을 변경해주세요"
- **명령어**: `gh repo edit kiminbean/power-demand-forecast --visibility private --accept-visibility-change-consequences`
- **결과**: ✅ PRIVATE으로 변경 완료

---

## 2. 프로젝트 현황

### 2.1 모델 현황

| 모델 | 용도 | 성능 | 상태 |
|------|------|------|------|
| SMP v3.19 CatBoost | RTM 실시간 예측 | MAPE 5.28%, R² 0.827 | ✅ Production |
| SMP v3.12 Stacking | 최고 성능 | MAPE 5.25%, R² 0.834 | ✅ Available |
| Power Demand LSTM | 전력 수요 예측 | MAPE 6.06%, R² 0.865 | ✅ Production |

### 2.2 프론트엔드 현황

| 앱 | 버전 | 포트 | 상태 |
|----|------|------|------|
| web-v7 (ExecO) | 7.0.0 | 8700 | ✅ Production |
| web-v6 (Bidding) | 6.0.0 | 8600 | ✅ Available |
| mobile | 7.0.0 | 3001 | ✅ Production |

### 2.3 API 현황

| API | 소스 | 상태 |
|-----|------|------|
| SMP | 제주시범사업 API | ✅ Connected |
| Weather | KMA API | ✅ Connected |
| Power Supply | Crawler | ✅ Connected |

---

## 3. 주요 파일 위치

```
/results/smp_model_version_history.png  # SMP 모델 버전 이력 이미지
/notebooks/Power_Demand_Forecast_LSTM_Production_Colab.ipynb  # LSTM Colab 노트북
/src/smp/models/                        # SMP 모델 학습 코드
/models/smp_v3_12_stacking/             # 최고 성능 모델
/models/smp_v3_19_recent/               # Production 모델
/.claude/backups/PROJECT_STATUS.md      # 프로젝트 상태 백업
```

---

## 4. 사용된 프롬프트

### 프롬프트 1: SMP 모델 버전 이력 이미지 생성
```
smp 예측 모델 버전별 log 확인해서 학습 일자 MAPE, R2, 학습내용. 이미지 형태로 만들어줘.
```

### 프롬프트 2: 버전 구분 질문
```
v3 프론트가 있습니까? 아니면 빽단의 예측 모델의 버전입니까?
```
→ 답변: 백엔드 예측 모델 버전 (프론트엔드와 별개)

### 프롬프트 3: 코드 위치 질문
```
smp 예측 모델 (백엔드) 사용했던 코드는 어디에 있습니까?
```
→ 답변: `src/smp/models/` 디렉토리

### 프롬프트 4: GitHub Private 변경
```
power-demand-forecast 깃허브에 프라이빗을 변경해주세요.
```
→ 결과: ✅ 완료

---

## 5. Key Findings (SMP 모델 실험 결론)

1. **smp_lag1 의존도**: Feature Importance 61.59% → 이전 시간 SMP에 크게 의존
2. **외부 데이터 무효**: 연료가격, 기상, 발전량 추가해도 성능 향상 없음
3. **Gradient Boosting 우위**: BiLSTM/Transformer보다 CatBoost/LightGBM이 더 효과적
4. **변동성 한계**:
   - 안정 시장 (변동성 <10%): MAPE ~2%
   - 변동 시장 (변동성 >30%): MAPE ~40%

---

## 6. 다음 세션 시작 방법

```bash
# 1. 프로젝트 상태 확인
cat .claude/backups/PROJECT_STATUS.md

# 2. 최근 커밋 확인
git log --oneline -10

# 3. Docker 시작 (필요시)
docker compose -f docker/docker-compose.v7.yml up -d --build

# 4. 대시보드 접속
# - ExecO: http://localhost:8700
# - Bidding: http://localhost:8600
# - Mobile: http://localhost:3001
```

---

*Last Updated: 2025-01-10*
