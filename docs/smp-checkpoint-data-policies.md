# SMP 체크포인트별 학습 데이터 정책 (시간대·결측)

> 2026-11-05 작성. 후속 과제 3번(체크포인트별 학습 시간대·결측 정책 명시)의 문서화 부분.
> 각 체크포인트가 **학습 시점에 실제로 적용한** 전처리를 학습 스크립트 기준으로 기록한다.
> 운영(추론) 전처리와의 불일치 지점은 각 절 말미에 표기한다.

## 공통 사항

- **시간대**: 모든 체크포인트 tz-naive. EPSIS/KPX가 제공하는 한국 표준시(KST) 시각을 그대로 사용하며 UTC 변환은 없다.
- **원천 데이터**: `data/smp/smp_5years_epsis.csv` (43,725행, 2020-12-19~2026-11-04, 결측 0건 — 2026-11-05 검증). EPSIS 기준 SMP.
  - `smp_jeju`는 검증된 SMP가 아니라 EPSIS 제주 시세로, 운영 노출 시 "제주 day-ahead"로 구분(커밋 2a3df5fe 정책).
- **24:00 표기**: EPSIS/KPX의 `24:00`은 pandas가 파싱 불가 → 체크포인트별로 아래처럼 처리(결과는 모두 "다음날 00:00"과 동일).
- **공통 파이프라인**(`src/data/` 쪽): 분할(train/val/test) 이후 각 분할 내에서 `ffill().bfill()` —
  분할 **선행 구간**의 결측이 분할 **이후 값**으로 채워질 수 있다(알려진 한계, 문서 기재 완료).

## v4.0 — `train_smp_v4.py` → `models/smp_v4/`

- 입력 168h → 출력 24h. 피처 32개(SMP 4 + 시간 8 + lag 4 + rolling 통계 등).
- 24:00: `fix_hour_24()` — `hour==24` 행을 다음날 00:00 datetime으로 변환 후 정렬.
- 시간 피처: hour/dow/month sin·cos, is_weekend, doy_sin — datetime에서 파생.
- 결측: 원천 결측 0건이므로 실질 미동작. 파생 시 `fillna(0)`(rolling std, min_periods=1), 초기 lag(24/48/168/336h)는 **backward fill** —
  학습 구간 맨 앞 lag 미존재분이 미래 값으로 채워짐(한계 문서화됨).
- 스케일러: train 슬라이스만 fit(`time_split_scaling.py`, 커밋 a2dcaf75), npz 저장(mean/scale/target_mean/target_std/feature_names).

## v4.1 — `train_smp_v4_smp_only.py` → `models/smp_v4/` (walk-forward)

- 입력 48h → 출력 24h. SMP 단독 파생 피처(lag, rolling std, pct_change, 동시간 전일/전주).
- 24:00: `timestamp` 문자열 ` 24:00`→` 00:00` 치환 후, `hour==24` 행은 `date + clip(hour, 23)h`로 보정.
- 결측: `dropna(subset=['smp_mainland'])` 후 진행. 초기 lag는 backward fill, `pct_change()` 첫 행은 `fillna(0)`.
- 스케일러: v4.0과 동일 train-only fit + npz.

## v3.4 — `train_smp_v3_4_kpx_generation.py` → `models/smp_v3_4_kpx/`

- KPX 시간별 발전량 wide 테이블을 melt → `date + hour h`로 datetime 생성(hour 24는 자연히 다음날 00:00).
- SMP는 위 공통 24:00 치환+보정 후 발전량과 merge.
- 결측: `pd.to_numeric(errors='coerce')` 후 **행 단위 dropna()** — v4 계열과 달리 결측 시점 자체를 제거.
- 스케일러 미저장(알려진 결함, 문서 기재 완료).
- 2026-11-05 재평가: test R² −0.537 → **운영 사용 보류**.

## v3.19 CatBoost — `train_smp_v3_19_recent.py` / `smp_catboost_predictor.py`

- 동일 `smp_5years_epsis.csv`를 2022-01-01 이후만 사용(최근 구간 특화), `smp_mainland > 0` 필터.
- 24:00: v4.0과 동일한 `fix_hour_24()`(다음날 00:00).
- 결측: 행 제거 없이 파생 단계에서 처리 — rolling std/diff는 `shift(1).fillna(0)`, lag 대비 비율은 `fillna(1)`(학습 시점에만 과거 참조 → 누수 없음).
- 예측기(`smp_catboost_predictor.py`)는 학습과 동일 CSV 전제로 로드.
- CI 재현을 위해 catboost가 requirements.txt에 선언됨(커밋 "fix: declare catboost dependency").

## 운영(추론) 측 불일치 메모

- `SMPPredictor`(구버전 v3.2/advanced 경로)는 pkl/npy 스케일러를 사용 — 신규 npz 규격과 별개 계열.
- v4 계열 추론 시 lag 생성 로직이 학습과 다르면(bfill 대신 0 채움 등) 성능이 달라질 수 있음 →
  후속 과제 3번의 비교 부분("신규 전처리와 운영 데이터 기준 성능 비교")은 운영 데이터 확보 후 수행 예정.
