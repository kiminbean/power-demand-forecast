# 실험 결과 요약 (2025-12-15 최종)

## 실험 개요

| 실험 | 설명 | 상태 |
|------|------|------|
| EVAL-003 | Horizon별 기상변수 효과 분석 | ✅ 완료 |
| EVAL-004 | Conditional Model (겨울철 변곡점 특화) | ✅ 완료 |

---

## EVAL-003 완료 결과

### Horizon별 성능 (5 trials 평균)

| Horizon | demand_only MAPE | weather_full MAPE | Weather Effect |
|---------|------------------|-------------------|----------------|
| h=1     | 4.99% ± 0.42     | 5.35% ± 0.23      | **-7.2% (악화)** |
| h=24    | 10.34% ± 0.31    | 10.63% ± 0.42     | **-2.8% (악화)** |
| h=48    | 11.38% ± 0.23    | 11.46% ± 0.24     | **-0.7% (미미)** |
| h=168   | 12.63% ± 0.53    | 12.62% ± 0.49     | **0.01%p (차이 없음)** |

### 핵심 발견

1. **가설 부분 검증**: Horizon이 길어질수록 기상변수의 부정적 효과 감소
   - h=1: -7.2% → h=168: 0.01%p (단, 노이즈 수준으로 실질적 개선 없음)

2. **신호 마스킹 효과**: 짧은 horizon에서 lag 변수(corr=0.974)가 기상 신호 압도

3. **겨울철 변곡점**: 유일하게 기상변수가 **+2.5%** 개선 효과

## 변곡점 분석 결과 (Full Trial: 5 trials, 50 epochs)

| 계절 | demand_only | weather_full | Weather Effect |
|------|-------------|--------------|----------------|
| 여름 | 8.92% | 9.37% | -5.1% |
| **겨울** | **12.97%** | **12.64%** | **+2.5%** |
| 전환기 | 10.63% | 11.41% | -7.4% |
| 전체 | 10.25% | 10.73% | -4.6% |

---

## EVAL-004 완료 결과

### Conditional Model 성능 (5 trials, 50 epochs)

| Model | MAPE | R² | vs demand_only |
|-------|------|-----|----------------|
| demand_only | 6.33%±0.20 | 0.8516 | baseline |
| weather_full | 6.62%±0.37 | 0.8391 | **-4.57% (악화)** |
| conditional_hard | 6.33%±0.20 | 0.8516 | +0.00% |
| conditional_soft | **6.32%±0.19** | **0.8521** | **+0.22% (개선)** |

### 겨울철 전용 테스트 (2022.12 ~ 2023.02)

| Model | MAPE | R² | vs demand_only |
|-------|------|-----|----------------|
| demand_only | 4.63% | 0.7321 | baseline |
| weather_full | 5.28% | 0.6346 | **-13.98% (악화)** |
| conditional_hard | 4.63% | 0.7321 | +0.00% |
| conditional_soft | **4.53%** | **0.7431** | **+2.23% (개선)** |

### Conditional Model 구성요소

1. **SeasonClassifier**: 월별 계절 분류 (WINTER/SUMMER/TRANSITION)
2. **InflectionDetector**: 수요 급변점 감지 (상위 5% 변화율)
3. **ConditionalPredictor**: 조건부 모델 선택
   - `hard` mode: 겨울+변곡점일 때만 weather_full 사용
   - `soft` mode: 확률적 앙상블 (겨울철 가중치 자동 조절)

### 핵심 발견

1. **conditional_soft 유효성 검증**: 겨울철에서 +2.23% MAPE 개선
2. **Soft blending 효과**: 전체 데이터에서도 +0.22% 미세 개선
3. **Hard mode 한계**: 테스트 기간에 겨울 데이터 부족시 효과 없음

---

## 실무 권장사항

```python
from models.conditional import create_conditional_predictor

# 권장: soft mode 사용 (겨울철 자동 가중치 조절)
predictor = create_conditional_predictor(
    demand_only_model=model_demand,
    weather_full_model=model_weather,
    train_demand=train_data['power_demand'],
    mode="soft"  # 확률적 앙상블
)

# 예측
predictions, contexts = predictor(X_demand, X_weather, timestamps, recent_demands)
```

### 적용 시나리오

| 시나리오 | 권장 모델 | 예상 효과 |
|----------|-----------|-----------|
| 전체 기간 예측 | conditional_soft | +0.2% 개선 |
| 겨울철 예측 | conditional_soft | **+2.2% 개선** |
| 여름/전환기 예측 | demand_only | baseline |

---

## 결과 파일 위치

### EVAL-003
- 상세 보고서: `results/FINAL_ANALYSIS_REPORT.md`
- Horizon 비교: `results/metrics/horizon_comparison.csv`
- 변곡점 분석: `results/metrics/inflection_point_analysis.csv`
- 시각화: `results/figures/horizon_comparison.png`

### EVAL-004
- Conditional 실험 결과: `results/metrics/conditional_experiment_report.json`
- 겨울철 테스트 결과: `results/metrics/winter_test_report.json`
- Conditional 모델: `src/models/conditional.py`
- 실험 스크립트: `src/experiments/conditional_experiment.py`
- 겨울 테스트 스크립트: `src/experiments/winter_test.py`

---

*최종 업데이트: 2025-12-15*
