# 전력 수요 예측 프로젝트 - 프롬프트 히스토리

> **작성일**: 2025-12-17
> **프로젝트**: 제주도 전력 수요 예측 시스템
> **Repository**: https://github.com/kiminbean/power-demand-forecast

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Phase 1: 초기 설정 및 기반 구축](#2-phase-1-초기-설정-및-기반-구축)
3. [Phase 2: 모델 개발](#3-phase-2-모델-개발)
4. [Phase 3: 피처 엔지니어링](#4-phase-3-피처-엔지니어링)
5. [Phase 4: API 개발](#5-phase-4-api-개발)
6. [Phase 5: MLOps 및 인프라](#6-phase-5-mlops-및-인프라)
7. [Phase 6: 프론트엔드 개발](#7-phase-6-프론트엔드-개발)
8. [Phase 7: 신재생에너지 연동](#8-phase-7-신재생에너지-연동)
9. [유용한 프롬프트 템플릿](#9-유용한-프롬프트-템플릿)

---

## 1. 프로젝트 개요

### 1.1 시스템 구성
- **Backend**: Python 3.13, PyTorch, FastAPI
- **ML Models**: LSTM, BiLSTM, TFT (Temporal Fusion Transformer), Ensemble
- **Frontend**: Streamlit Dashboard
- **Database**: CSV/Parquet 파일 기반
- **CI/CD**: GitHub Actions, Docker, GHCR

### 1.2 최종 성능
| 메트릭 | 값 |
|--------|------|
| MAPE | 6.32% |
| R² | 0.852 |
| 테스트 케이스 | 1,423개 |

---

## 2. Phase 1: 초기 설정 및 기반 구축

### PRD 생성 프롬프트
```
제주도 전력수요 예측 시스템의 지속적인 개선을 위한 PRD를 작성해줘.
현재 LSTM 기반 모델과 FastAPI 서비스가 구축되어 있으며, CI/CD 파이프라인이 완성되었습니다.

포함할 내용:
1. 모델 개선 (Transformer, Ensemble, Attention)
2. 피처 엔지니어링 (AutoML, 피처 스토어)
3. MLOps 인프라 (모델 레지스트리, 자동 재학습)
4. API 서비스 개선 (실시간 스트리밍, 인증)
5. 모니터링 (Prometheus, Grafana)
6. 데이터 파이프라인 (Airflow)
```

### Task Master 초기화
```
Task Master를 사용해서 PRD 기반으로 태스크를 생성해줘.
각 태스크는 구체적인 구현 단계와 의존성을 포함해야 해.
```

---

## 3. Phase 2: 모델 개발

### Task 1: Temporal Fusion Transformer (TFT) 구현

#### 서브태스크 1.1: 아키텍처 설계
```
Temporal Fusion Transformer (TFT) 논문을 분석하고 프로젝트에 맞는 아키텍처를 설계해줘.

포함 내용:
1. TFT 논문 (Google, 2020) 핵심 분석
2. 기존 LSTM 아키텍처와 비교
3. 입력 피처 구조 설계 (static, known, unknown)
4. 출력 레이어 설계 (다중 시간대 예측)
5. 아키텍처 다이어그램 작성
```

#### 서브태스크 1.2: Variable Selection Network 구현
```
피처별 중요도를 학습하는 Variable Selection Network를 구현해줘.

구현 항목:
1. GRN (Gated Residual Network)
2. Variable Selection Network
3. 피처 그룹별 선택 로직 (static, temporal)
4. 단위 테스트
```

#### 서브태스크 1.3: Temporal Self-Attention 구현
```
시계열 패턴을 학습하는 Interpretable Multi-Head Attention을 구현해줘.

구현 항목:
1. Multi-Head Attention 레이어
2. Positional Encoding
3. Masked Attention (미래 정보 차단)
4. Attention weight 추출 기능
```

#### 서브태스크 1.4-1.5: Encoder-Decoder 통합
```
LSTM Encoder와 Temporal Attention을 결합한 전체 TFT 모델을 구현해줘.

구현 항목:
1. LSTM Encoder (과거 시퀀스 인코딩)
2. LSTM Decoder (미래 시퀀스 디코딩)
3. Static Covariate Encoder
4. Quantile Output Layer (불확실성 추정)
5. forward() 메서드
```

#### 서브태스크 1.6: 학습 파이프라인
```
TFT 모델 학습을 위한 데이터 로더와 학습 루프를 구현해줘.

구현 항목:
1. TFT용 Dataset 클래스 (Static/Known/Unknown 피처 분리)
2. 학습/검증/테스트 데이터 분할
3. Early Stopping, LR Scheduler
4. 체크포인트 저장/로드
```

#### 서브태스크 1.7: LSTM vs TFT 비교 실험
```
동일 데이터셋에서 LSTM과 TFT 모델의 성능을 비교하는 실험을 수행해줘.

비교 항목:
1. 다중 시간대 (1h, 6h, 12h, 24h) 예측 비교
2. RMSE, MAE, MAPE, R² 메트릭 비교
3. 학습 시간 및 추론 시간 비교
4. 실험 결과 문서화
```

#### 서브태스크 1.8: 하이퍼파라미터 튜닝
```
Optuna를 활용한 TFT 하이퍼파라미터 최적화를 수행해줘.

탐색 공간:
- hidden_size: [32, 64, 128, 256]
- num_heads: [1, 2, 4, 8]
- dropout: [0.1, 0.2, 0.3]
- learning_rate: [1e-4, 1e-3, 1e-2]
- batch_size: [32, 64, 128]
```

#### 서브태스크 1.9: Attention 시각화
```
학습된 Attention weight를 시각화하는 도구를 구현해줘.

구현 항목:
1. Attention heatmap 생성
2. 시간대별 중요도 시각화
3. 피처별 기여도 시각화
4. Plotly 인터랙티브 플롯
```

### Task 2: Ensemble 모델 구현
```
LSTM, BiLSTM, Transformer 모델의 예측을 결합하는 앙상블 기법을 구현해줘.

구현 항목:
1. Stacking 앙상블
2. Blending 앙상블
3. 모델 가중치 자동 최적화
4. 성능 비교 및 최적 조합 선정
```

### Task 4: Probabilistic Forecasting 구현
```
불확실성을 정량화하는 확률적 예측 모델을 구현해줘.

구현 항목:
1. Quantile Regression (0.1, 0.5, 0.9 분위수)
2. Prediction Interval 계산
3. 불확실성 시각화
4. 리스크 관리 활용 가이드
```

---

## 4. Phase 3: 피처 엔지니어링

### Task 5: AutoML 피처 선택
```
SHAP, Permutation Importance 기반 자동 피처 선택 시스템을 구현해줘.

구현 항목:
1. SHAP 기반 피처 중요도 분석
2. Permutation Importance 구현
3. 자동 피처 제거 로직 (중요도 낮은 피처)
4. 피처 중요도 리포트 생성
```

### Task 6: 피처 스토어 구축
```
재사용 가능한 피처 저장소를 구축해줘.

구현 항목:
1. 피처 스토어 아키텍처 설계
2. 피처 버전 관리
3. 학습/추론 시 동일 피처 보장
4. 피처 계보(Lineage) 추적
```

### Task 7: 실시간 기상 API 연동
```
기상청 API를 통한 실시간 기상 데이터 수집 시스템을 구축해줘.

구현 항목:
1. 기상청 단기예보 API 연동
2. 예보 데이터 파싱 및 저장
3. 데이터 갱신 스케줄러
4. 에러 처리 및 재시도 로직
```

### Task 8: 태양광 발전량 추정
```
일사량/전운량으로 BTM 태양광 발전량을 추정하는 모델을 구현해줘.

구현 항목:
1. 태양광 발전량 추정 모델 설계
2. Duck Curve 효과 모델링
3. 일몰 전후 수요 급변 예측 개선
4. 수요 예측 모델에 통합
```

---

## 5. Phase 4: API 개발

### Task 13: REST API 개발
```
FastAPI를 사용해서 전력 수요 예측 REST API를 개발해줘.

엔드포인트:
1. GET /health - 상태 확인
2. GET /models - 모델 정보
3. POST /predict - 단일 예측
4. POST /predict/batch - 배치 예측
5. POST /predict/conditional - 조건부 예측 (계절별 기상 가중치)

요구사항:
- Pydantic 스키마 검증
- 에러 핸들링
- CORS 설정
- Swagger/ReDoc 문서
```

### Task 12: 다중 시간대 예측
```
1시간, 6시간, 12시간, 24시간 다중 시간대 예측 기능을 구현해줘.

구현 항목:
1. Multi-horizon 예측 모델
2. 각 horizon별 최적화된 학습
3. API 엔드포인트 추가 (step 파라미터)
4. 성능 비교
```

---

## 6. Phase 5: MLOps 및 인프라

### Task 9: MLflow 실험 추적
```
MLflow를 활용한 모델 버전 관리 및 실험 추적 시스템을 구축해줘.

구현 항목:
1. MLflow 서버 설정
2. 모델 메타데이터 저장
3. 실험 추적 및 비교
4. 모델 스테이징 (dev/staging/prod)
```

### Task 10: 자동 재학습 파이프라인
```
데이터 드리프트 감지 시 자동으로 모델을 재학습하는 파이프라인을 구축해줘.

구현 항목:
1. 드리프트 감지 알고리즘
2. 재학습 트리거 조건 정의
3. 자동 검증 및 배포 로직
4. 알림 시스템 연동
```

### Task 15: Docker 컨테이너화
```
API 서버를 Docker로 컨테이너화하고 GHCR에 배포해줘.

구현 항목:
1. Dockerfile 작성 (멀티스테이지 빌드)
2. docker-compose.yml
3. GitHub Actions CI/CD 파이프라인
4. GHCR 푸시 자동화
```

### Task 16: CI/CD 파이프라인
```
GitHub Actions 기반 CI/CD 파이프라인을 구축해줘.

구현 항목:
1. 테스트 자동화 (pytest)
2. 린트 검사 (ruff)
3. Docker 빌드 및 푸시
4. 자동 배포 (staging/production)
```

### Task 17: 통합 테스트
```
API 및 모델의 통합 테스트를 작성해줘.

테스트 항목:
1. API 엔드포인트 테스트
2. 모델 예측 정확성 테스트
3. 성능 테스트 (응답 시간)
4. 에러 처리 테스트
```

### Task 18: 모니터링 시스템
```
Prometheus와 Grafana 기반 모니터링 시스템을 구축해줘.

구현 항목:
1. Prometheus 메트릭 수집
2. 예측 정확도 실시간 추적
3. API 응답 시간 모니터링
4. Grafana 대시보드
```

### Task 19: AutoML 모델 선택
```
자동으로 최적의 모델을 선택하는 AutoML 시스템을 구현해줘.

구현 항목:
1. 여러 모델 후보 학습
2. 검증 성능 기반 자동 선택
3. 하이퍼파라미터 최적화
4. 결과 리포트 생성
```

### Task 20: API 문서 및 모델 카드
```
API 문서와 모델 카드를 작성해줘.

포함 내용:
1. API 사용 가이드 (요청/응답 예시)
2. 에러 코드 문서
3. 모델 성능 벤치마크
4. 모델 한계점 및 편향 분석
```

### Task 21: 부하 테스트
```
Locust 기반 API 부하 테스트를 구현해줘.

테스트 시나리오:
1. 동시 사용자 100명 시뮬레이션
2. 최대 처리량 측정
3. 응답 시간 분포 분석
4. 병목 지점 식별
```

### Task 22: 이상치 탐지
```
비정상 전력 수요 패턴을 탐지하는 시스템을 구현해줘.

구현 항목:
1. Isolation Forest 기반 탐지
2. Autoencoder 기반 탐지
3. 실시간 이상치 알림
4. 원인 분석 지원
```

### Task 23: 설명 가능한 AI (XAI)
```
SHAP/LIME 기반 예측 설명 기능을 구현해줘.

구현 항목:
1. SHAP 설명 생성
2. LIME 설명 생성
3. 피처 기여도 시각화
4. API 엔드포인트 (/explain)
```

### Task 24: 시나리오 분석
```
폭염/한파 등 극단적 상황에서의 수요 예측 시나리오 분석 기능을 구현해줘.

시나리오:
1. 평년
2. 약한 폭염 (+3°C)
3. 심한 폭염 (+7°C)
4. 약한 한파 (-5°C)
5. 심한 한파 (-10°C)
```

### Task 25: 통합 파이프라인
```
데이터 수집부터 예측까지 전체 파이프라인을 통합해줘.

포함 기능:
1. 데이터 로드 및 전처리
2. 모델 선택 및 학습
3. 평가 및 시각화
4. 결과 리포트 생성
```

---

## 7. Phase 6: 프론트엔드 개발

### Task 14: Streamlit 대시보드 (기본)
```
Streamlit을 사용해서 전력 수요 예측 대시보드를 개발해줘.

기능:
1. 실시간 예측 차트 (24시간)
2. 기상 조건 입력 인터페이스 (온도, 습도, 풍속)
3. 시나리오 분석 (폭염/한파)
4. 과거 데이터 비교
5. 모델 성능 지표 표시
```

### API 연동 대시보드
```
Streamlit 대시보드를 FastAPI 서버와 연동해줘.

변경 내용:
1. API 클라이언트 클래스 추가
2. 실시간 예측 시 API 호출
3. 시나리오 분석 시 배치 예측 API 사용
4. API 연결 상태 표시 (헤더)
5. 모델 정보 API에서 실시간 조회
```

---

## 8. Phase 7: 신재생에너지 연동

### kpx-demand-forecast API 연동
```
Ormi/kpx-demand-forecast/ 폴더에 있는 태양광과 풍력발전기의
발전량을 예측하는 API를 Streamlit 대시보드에 연동해줘.

구현 항목:
1. RenewableAPIClient 클래스 추가
2. 신재생에너지 발전량 예측 탭 추가
3. 통합 에너지 현황 탭 추가
4. 사이드바에 API 상태 표시
5. 시스템 정보 탭에 두 API 모두 표시
```

---

## 9. 유용한 프롬프트 템플릿

### 버그 수정
```
[파일명]에서 [에러 메시지] 에러가 발생합니다.
에러 원인을 분석하고 수정해줘.
```

### 기능 추가
```
[기존 기능]에 [새로운 기능]을 추가해줘.

요구사항:
1. [구체적인 요구사항 1]
2. [구체적인 요구사항 2]
3. 기존 코드 패턴 유지
4. 테스트 케이스 추가
```

### 리팩토링
```
[파일명]의 [함수/클래스명]을 리팩토링해줘.

목표:
1. 가독성 향상
2. 성능 최적화
3. 중복 코드 제거
4. 타입 힌트 추가
```

### 테스트 작성
```
[파일명]의 [함수/클래스명]에 대한 테스트를 작성해줘.

테스트 항목:
1. 정상 케이스
2. 경계값
3. 예외 처리
4. 엣지 케이스
```

### 문서화
```
[기능명]에 대한 사용자 가이드를 작성해줘.

포함 내용:
1. 개요 및 목적
2. 설치 방법
3. 사용 예시 (코드 포함)
4. API 레퍼런스
5. FAQ
```

### 성능 최적화
```
[파일명]의 [함수/클래스명] 성능을 최적화해줘.

현재 문제:
- [성능 이슈 설명]

목표:
- 실행 시간 50% 감소
- 메모리 사용량 감소
```

---

## Git 커밋 히스토리 (주요 커밋)

```
e899b2c docs: Add auto commit protocol to CLAUDE.md
06e8b66 Integrate renewable energy API with Streamlit dashboard
f17983c Add API-connected Streamlit dashboard
2ac63f7 Add conversation backup system
ce9eac5 Add missing dependencies to requirements.txt
cdd9431 Docs: Update README with v1.0.0 features
303a61f Feat(TASK-025): Implement integrated pipeline
b35d39a Feat(TASK-024): Implement Scenario Analysis module
ca95ce4 Feat(TASK-023): Implement XAI (Explainable AI) module
18c767d Feat(Task-22): Add anomaly detection system
f6c605c Feat(Task-21): Add Locust-based load testing
bced33f Feat(Task-20): Add API documentation and model cards
110bd1f Feat(Task-19): Add AutoML model selection system
bf9f175 Feat(Task-18): Add comprehensive monitoring system
0d1c762 Feat(Task-17): Add comprehensive integration tests
a6acdca Feat(CICD-001): CI/CD 파이프라인 테스트 추가 (Task 16)
7fc5b75 Feat(DOCK-001): Docker 컨테이너화 (Task 15)
85fc29a Feat(DASH-001): Dashboard UI 개발 (Task 14)
93a9b45 Feat(API-001): REST API 개발 (Task 13)
065dc0b Feat(MODEL-014): Implement multi-horizon forecasting (Task 12)
cdc4e42 Feat(FEAT-012): Implement holiday and special day processing (Task 11)
ccc1e45 Feat(TRAIN-003): Implement online learning and model updates (Task 10)
6f4ec9c Feat(TRAIN-002): Implement MLflow-style experiment tracking (Task 9)
a686f3d Feat(MODEL-013): Implement solar power generation estimation (Task 8)
48f7b15 Feat(DATA-003): Implement real-time weather API integration (Task 7)
fb6c448 Feat(FEAT-011): Implement Feature Store for ML pipelines (Task 6)
5c96b8a Feat(FEAT-010): Implement AutoML Feature Selection (Task 5)
07adf23 Feat(MODEL-012): Implement Probabilistic Forecasting (Task 4)
681c9e5 Feat(MODEL-011): Implement Ensemble models (Task 2)
2a8410b Feat(MODEL-010): Complete TFT production integration (Subtask 1.10)
```

---

## 참고 자료

1. **PRD 파일**: `.taskmaster/docs/prd.txt`
2. **Task 목록**: `.taskmaster/tasks/tasks.json`
3. **프로젝트 상태**: `.claude/backups/PROJECT_STATUS.md`
4. **API 문서**: `http://localhost:8000/docs`

---

> **Note**: 이 문서는 프로젝트 히스토리와 Task 정의를 기반으로 재구성한 프롬프트 가이드입니다.
> 실제 사용된 프롬프트는 대화 세션에 저장되어 있으며, 각 세션은 독립적으로 관리됩니다.
