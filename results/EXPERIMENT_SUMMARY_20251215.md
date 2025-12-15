# 실험 결과 요약 (2025-12-15 최종)

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

## 실무 권장사항

```python
if season == "winter" and is_inflection_point:
    model = "weather_full"  # 겨울철 급변구간만 기상변수 유효
else:
    model = "demand_only"   # 모든 다른 경우 기상변수 제외
```

## 결과 파일 위치

- 상세 보고서: `results/FINAL_ANALYSIS_REPORT.md`
- Horizon 비교: `results/metrics/horizon_comparison.csv`
- 변곡점 분석: `results/metrics/inflection_point_analysis.csv`
- 시각화: `results/figures/horizon_comparison.png`

*실험 완료: 2025-12-15*
