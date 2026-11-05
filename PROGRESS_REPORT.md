# Hybrid Agent Pipeline 진행 보고서

## 프로젝트 개요
- **프로젝트**: 제주도 전력 수요 예측 (Power Demand Forecasting)
- **위치**: `/Users/ibkim/Ormi_1/power-demand-forecast`
- **파이프라인**: Anthropic State Persistence + DeepMind IMO 2025 Verification Loop
- **최종 업데이트**: 2026-11-05

---

## 2026-11 업데이트 (최신)

이 문서의 아래 섹션은 2024-12 하이브리드 에이전트 파이프라인 초기 구축 당시 기록이며 현재 상태를 반영하지 않는다. 현재 작업 트리의 구현·확인 범위는 `docs/2026-11-02-forecast-upgrade-status.md`가 단일 기준이다.

### 2026-11-04: SMP 학습 스케일러 누수 수정 (우선 후속 과제 1번, 코드 수정 완료)

- 신규: `src/smp/models/time_split_scaling.py` — `fit_transform_time_split`(train 슬라이스만 fit), `temporal_split_bounds`, `carve_validation_from_train`
- 수정: `train_smp_v4.py`(0.8/0.1), `train_smp_v3_4_kpx_generation.py`(0.7/0.15), `train_smp_v4_smp_only.py`(val carve 방식, split별 스케일러 리셋) — 시간 분할 이후 train 구간에만 스케일러 fit
- 버그 수정: `train_smp_v4.py`·`train_smp_v4_enhanced.py`의 `np.save(dict)` 예외 → `np.savez('smp_v4_scaler.npz')`
- 회귀 테스트: `tests/test_smp_time_split_scaling.py` 16/16 통과 (`.venv/bin/python -m pytest` 직접 호출)

### 2026-11-05: SMP 3개 모델 재학습·시간외 평가 완료 (우선 후속 과제 1번 완결)

- 수정 코드(a2dcaf75)로 v4.0·v4.1 SMP 전용·v3.4 KPX 결합 모델 재학습(로그: `logs/retrain_2026-11-05_{v4,v41,v34}.log`, 각 exit 0)
- 새 시간외 성능 vs 누수 버전(백업: `models/_pre_retrain_backup_2026-11-05/`):
  - v4.0: test MAPE 22.67%→17.17%, R² -0.886→0.124, 80% 커버리지 51.7%→77.5%
  - v4.1: walk-forward 평균 MAPE 8.83%→8.76%, R² 0.426→0.463, 커버리지 71.7%→68.6%
  - v3.4: test MAPE 17.58%→21.28%, R² -0.105→-0.537 — 재학습 후에도 일반화 실패, 추가 데이터 결합 전까지 사용 보류
- 재학습 후 발견·수정: `train_smp_v4_smp_only.py`가 v41 스케일러를 `np.save`(dict) npy로 저장(a2dcaf75 수정 누락) → `np.savez`로 전환, 아티팩트 `smp_v41_scaler.npz` 복원, 누수 버전 스케일러 npy 2건 삭제
- 재검증: `pytest tests/test_smp_time_split_scaling.py` 16/16 통과

---

## 하이브리드 에이전트 파이프라인 현황

### 아키텍처 구성

| 역할 | 담당 | 상태 | 설명 |
|------|------|------|------|
| **Controller** | `agent_harness.py` | ✅ 구현완료 | 상태 관리, 루프 제어 |
| **Worker** | Claude Code CLI | ✅ 사용가능 | 코드 생성 |
| **Verifier** | Gemini CLI | ✅ 통합됨 | L2 검증 (코드 리뷰) |
| **State File** | `feature_list.json` | ✅ 구현완료 | 20개 태스크 정의 |
| **Progress Log** | `results/claude-progress.txt` | ✅ 구현완료 | 작업 히스토리 |
| **Git Versioning** | `.git/hooks/auto-commit.sh` | ✅ 구현완료 | 자동 커밋 |

### 검증 체계 (IMO 2025 Style)

```
┌─────────────────────────────────────────────────────────────┐
│                    Verification Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│  L1 (Deterministic): pytest 실행 - 기능 정확성               │
│  L2 (Probabilistic): Gemini 리뷰 - 아키텍처, 보안, 엣지케이스 │
└─────────────────────────────────────────────────────────────┘
```

### 사용법

```bash
# 작업 상태 확인
python agent_harness.py --status

# 다음 작업 자동 실행
python agent_harness.py

# 특정 작업 실행
python agent_harness.py --task DATA-001

# 특정 작업 검증만 수행
python agent_harness.py --verify DATA-001

# 작업 초기화
python agent_harness.py --reset DATA-001
```

---

---

## 아키텍처 설계

### 역할 분담
| 역할 | 담당 | 설명 |
|------|------|------|
| **Controller** | agent_harness.py | 상태 관리, 루프 제어 |
| **Worker** | Claude (수동 전환) | 코드 생성 |
| **Verifier** | Gemini CLI | L2 검증 (코드 리뷰) |

### 검증 체계
- **L1 (Deterministic)**: pytest 실행 - 기능 정확성
- **L2 (Probabilistic)**: Gemini 리뷰 - 아키텍처, 보안, 엣지 케이스

---

## 완료된 작업

### FEAT-001: THI 및 상대습도 계산 ✅
**상태**: `done`

**생성된 파일**:
```
src/features/
├── __init__.py
└── weather_features.py    # RH 및 THI 계산 로직

tests/
├── __init__.py
└── test_weather_features.py  # 19개 테스트 케이스
```

**핵심 구현**:
1. August-Roche-Magnus 공식으로 이슬점 → 상대습도 역산
   - 상수: a=17.625, b=243.04
   - RH = 100 × exp(a×Td/(b+Td)) / exp(a×T/(b+T))
2. 클리핑 로직: np.clip(humidity, 0, 100)
3. THI 공식: 1.8×T - 0.55×(1-RH_ratio)×(1.8×T-26) + 32
4. 벡터 연산(numpy)으로 성능 최적화

**검증 결과**:
- L1: 19/19 테스트 통과 (pytest, 0.51s)
- L2: PASS (Gemini 코드 리뷰 승인)

**커밋**: `8a016e1 Feat(FEAT-001): Implement THI and RH calculation using Dewpoint (August-Roche-Magnus)`

---

### FEAT-002: 체감온도 (Wind Chill) ❌
**상태**: `cancelled`

**사유**: 데이터셋 내 wind_speed 컬럼 부재로 구현 불가

---

### MODEL-001: LSTM 모델에 THI 통합 🔄
**상태**: `in_progress`

**생성된 파일**:
```
src/training/
├── __init__.py
└── train_lstm_thi_comparison.py  # THI 포함/미포함 비교 실험
```

**구현 내용**:
1. LSTM 기반 전력 수요 예측 모델
2. MPS(Apple Silicon GPU) 지원
3. THI 포함/미포함 A/B 테스트
4. 자동 성능 비교 리포트 생성

**실험 설정**:
```python
CONFIG = {
    'sequence_length': 14,  # 14일 시퀀스
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,  # Early stopping
    'train_ratio': 0.8,
    'val_ratio': 0.1,
}
```

**Feature 구성**:
- BASE_FEATURES (18개): temp_mean, temp_max, temp_min, temp_range, dewpoint_mean, sunshine_hours, solar_radiation, soil_temp_5cm, soil_temp_10cm, soil_temp_20cm, CDD, HDD, month_sin, month_cos, dayofweek_sin, dayofweek_cos, is_weekend, is_holiday
- THI_FEATURES (20개): BASE_FEATURES + humidity, THI

**실행 명령**:
```bash
cd /Users/ibkim/Ormi_1/power-demand-forecast
source .venv/bin/activate
python src/training/train_lstm_thi_comparison.py
```

---

## 현재 상태 (feature_list.json)

```json
[
  {
    "id": "FEAT-001",
    "description": "기상 데이터를 활용한 불쾌지수(THI) 파생 변수 생성",
    "status": "done",
    "files_changed": ["src/features/weather_features.py", "tests/test_weather_features.py"],
    "retry_count": 0
  },
  {
    "id": "FEAT-002",
    "description": "동절기 전력 수요 예측을 위한 체감온도(Wind Chill) 생성",
    "status": "cancelled",
    "note": "데이터셋 내 wind_speed 컬럼 부재로 구현 불가",
    "retry_count": 0
  },
  {
    "id": "MODEL-001",
    "description": "LSTM 모델 파이프라인에 THI 변수 통합 및 성능 검증",
    "status": "in_progress",
    "files_changed": ["src/training/train_lstm_thi_comparison.py"],
    "retry_count": 0
  }
]
```

---

## 다음 단계

1. **MODEL-001 완료**: 학습 결과 확인 및 R² 비교
2. **FEAT-003 (예정)**: 지중온도 기반 계절 지연 효과 변수 생성
3. **커밋 및 푸시**: MODEL-001 완료 후 main 브랜치에 병합

---

## 파일 구조

```
power-demand-forecast/
├── data/
│   └── processed/
│       └── jeju_daily_dataset.csv
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── weather_features.py      # ✅ FEAT-001
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_lstm_thi_comparison.py  # 🔄 MODEL-001
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/
│   ├── __init__.py
│   └── test_weather_features.py     # ✅ FEAT-001
├── feature_list.json
├── agent_harness.py
└── .venv/
```

---

## 기술적 결정 사항

### 1. Claude Code CLI 크레딧 문제
- **문제**: Claude Code CLI 크레딧 부족으로 자동화 불가
- **해결**: Human-in-the-loop 모드로 전환, Claude(채팅)가 Worker 역할 수행

### 2. 데이터 제약
- **문제**: wind_speed 컬럼 부재로 체감온도 계산 불가
- **해결**: FEAT-002 취소, MODEL-001으로 우선 진행

### 3. MPS 활용
- **설정**: Apple Silicon MPS를 최우선 디바이스로 사용
- **구현**: `get_device()` 함수로 자동 선택 (MPS > CUDA > CPU)

---

## 참고 문서

- [Transcript 전체 기록]: `/mnt/transcripts/2025-12-13-09-21-57-hybrid-agent-pipeline-implementation.txt`
- [August-Roche-Magnus 공식]: Alduchov & Eskridge (1996)
- [THI 공식]: 기상청 표준

---

*Last Updated: 2024-12-13*
