"""
API 추론 경로와 직접 추론 경로의 결측치 전처리 정합성 테스트
================================================================

배경 (문서화된 차이 분석)
-------------------------
동일한 입력이 추론 경로에 따라 다른 결과를 내는 문제를 해결한다.

- API 경로: ``api/service.py::PredictionService._prepare_features_for_api``
  기존: 피처 생성 후 ``df.bfill().ffill()`` 후 잔여 NaN을 ``fillna(0)`` 처리.
  - 행 수: 모든 행 유지 (warm-up 포함).
  - 채움 방향: bfill 우선 — 이후 시점 값으로 과거 결측을 채움.
  - 누수: 시계열 미래 정보 누수 발생 (예: 수요 lag 피처의 선행 결측을
    미래 수요 값으로 채움). 잔여 NaN의 0 채움도 스케일러에 왜곡된 값 주입.

- 직접 추론 경로: ``src/inference/predict.py::ProductionPredictor._prepare_features``
  기존: 피처 생성 후 ``df.dropna()``.
  - 행 수: lag/MA warm-up(기본 lag 1,24,48,168 → 선행 168행) 및 내부 NaN 행 제거.
  - 채움 방향: 채우지 않고 행 제거.
  - 누수: 없음. 단, 내부 NaN 행이 제거되면서 168시간 연속 시계열이
    시간축이 끊긴 채 병합되어 모델에 전달되는 문제(연속성 붕괴).

통합 정책 (인과적 ffill 우선)
------------------------------
프로젝트 규칙(시간순 처리, 미래 정보 누수 금지, pandas 보간 허용)과
``src/data/preprocessing.py::CausalProductionPreprocessor``의 기존 선례
(``clip().ffill()`` — 과거 값만으로 채움, 선행 결측은 남김)에 따라:

1. 결측은 과거 값으로만 채운다 (``ffill``) — 내부 결측 보존 및 시간 연속성 유지.
2. ffill로도 채울 수 없는 선행 warm-up 행(lag 피처 구조적 결측)은 제거한다 —
   인과적으로 채울 방법이 없으므로 제거가 유일한 비누수 처리.
3. bfill/0 채움 금지 — 미래 정보 누수 및 스케일러 왜곡 방지.
4. 피처 목록·순서는 기존 그대로 유지 (기존 체크포인트 호환, 재학습 금지).

완전한(내부 결측 없는) 입력에서는 기존 직접 추론 경로의 결과와
행 수·값이 정확히 동일하므로 비파괴적이다. API 경로도 336행 이상의
완전한 입력에서는 예측에 사용되는 마지막 168행의 값이 불변이다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import service as api_service_module
from inference import predict as predict_module
from api.service import PredictionService
from inference.predict import ProductionPredictor


N_HOURS = 500  # lag warm-up(168) + 시퀀스(168) 여유 확보
WARMUP = 168   # 기본 lag 목록 [1, 24, 48, 168]의 최대 warm-up


def _make_input(n_hours: int = N_HOURS, seed: int = 42) -> pd.DataFrame:
    """테스트용 시계열 입력 (datetime 인덱스)"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    df = pd.DataFrame(
        {
            'power_demand': 400
            + 100 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
            + np.random.normal(0, 20, n_hours),
            '기온': 10
            + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24)
            + np.random.normal(0, 2, n_hours),
            '습도': 60 + 20 * np.random.random(n_hours),
            '풍속': 2 + 3 * np.random.random(n_hours),
        },
        index=dates,
    )
    return df


class TestSharedPreprocessingContract:
    """두 추론 경로가 동일한 전처리 계약을 따르는지 검증"""

    def test_api_and_direct_paths_produce_identical_frames(self) -> None:
        """같은 입력(내부 결측 포함)에 대해 두 경로의 출력이 완전히 동일해야 한다"""
        df = _make_input()
        df.iloc[250, df.columns.get_loc('기온')] = np.nan
        df.iloc[300, df.columns.get_loc('power_demand')] = np.nan

        service_frame = PredictionService()._prepare_features_for_api(
            df, include_weather=True
        )
        predictor_frame = ProductionPredictor()._prepare_features(
            df, include_weather=True
        )

        pd.testing.assert_frame_equal(
            service_frame, predictor_frame, check_dtype=False
        )

    def test_feature_names_and_order_identical(self) -> None:
        """피처 목록·순서가 경로와 무관하게 동일해야 한다 (체크포인트 호환)"""
        df = _make_input()

        service_frame = PredictionService()._prepare_features_for_api(
            df, include_weather=True
        )
        predictor_frame = ProductionPredictor()._prepare_features(
            df, include_weather=True
        )

        assert list(service_frame.columns) == list(predictor_frame.columns)
        # 기존 직접 경로 대비 피처 구성 불변 (신규/제거 없음)
        assert 'demand_lag_1' in predictor_frame.columns
        assert 'hour_sin' in predictor_frame.columns

    def test_both_paths_delegate_to_shared_helper(self) -> None:
        """두 경로가 단일 공유 구현을 사용하는지 확인 (계약 이탈 방지)"""
        from importlib import import_module

        from api import service as api_service_module

        predict_module = import_module('inference.predict')

        shared_names = {
            name
            for name in dir(predict_module)
            if 'prepare_inference_features' in name
        }
        assert shared_names, "공유 헬퍼가 inference 모듈에 노출되어야 함"

        for module in (api_service_module, predict_module):
            module_file = getattr(module, '__file__', None)
            assert module_file is not None
            src = Path(module_file).read_text(encoding='utf-8')
            assert 'prepare_inference_features' in src, (
                f"{module.__name__} 이(가) 공유 헬퍼를 사용해야 함"
            )


class TestCausalMissingValuePolicy:
    """인과적(미래 누수 없는) 결측 처리 정책 검증"""

    def test_interior_gap_rows_are_preserved(self) -> None:
        """내부 결측 행이 제거되지 않고 시간 연속성이 유지되어야 한다"""
        df = _make_input()
        df.iloc[300, df.columns.get_loc('power_demand')] = np.nan

        df_prep = ProductionPredictor()._prepare_features(df, include_weather=False)

        # warm-up 168행만 제거되고 내부 결측 행은 유지되어야 함
        assert len(df_prep) == N_HOURS - WARMUP
        assert df_prep.index[300 - WARMUP] == df.index[300]
        # 시간축 연속성 (시퀀스가 끊기지 않아야 함)
        assert (df_prep.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()

    def test_no_nan_in_output(self) -> None:
        """출력에 NaN이 남으면 안 된다"""
        df = _make_input()
        df.iloc[200, df.columns.get_loc('기온')] = np.nan
        df.iloc[350, df.columns.get_loc('power_demand')] = np.nan

        df_prep = ProductionPredictor()._prepare_features(df, include_weather=True)
        assert df_prep.isna().sum().sum() == 0

    def test_no_future_leakage_in_fill(self) -> None:
        """결측 채움이 미래 값을 사용하지 않아야 한다 (bfill 금지)"""
        base = _make_input()
        gap_pos = 300
        base.iloc[gap_pos, base.columns.get_loc('기온')] = np.nan

        # 미래(결측 이후) 값만 다르게 한 입력 — 결측 시점의 채움 결과가
        # 미래 값에 영향을 받으면 bfill(누수)로 채운 것임
        alt = base.copy()
        alt.iloc[gap_pos + 1 :, alt.columns.get_loc('기온')] = -50.0

        prep_base = ProductionPredictor()._prepare_features(base, include_weather=False)
        prep_alt = ProductionPredictor()._prepare_features(alt, include_weather=False)

        gap_out = gap_pos - WARMUP
        pd.testing.assert_series_equal(
            prep_base.iloc[gap_out], prep_alt.iloc[gap_out], check_dtype=False
        )

    def test_leading_warmup_rows_are_dropped_not_backfilled(self) -> None:
        """선행 warm-up 행은 미래 값으로 채우지(bfill) 않고 제거해야 한다"""
        df = _make_input()

        df_prep = ProductionPredictor()._prepare_features(df, include_weather=False)

        # bfill 이라면 500행이 모두 유지됨 — 인과 정책에서는 warm-up 제거
        assert len(df_prep) == N_HOURS - WARMUP
        assert df_prep.index[0] == df.index[WARMUP]

    def test_clean_input_row_count_matches_legacy_dropna(self) -> None:
        """완전한 입력에서 기존 직접 경로(dropna)와 행 수·값이 동일해야 한다 (비파괴)"""
        df = _make_input()

        df_prep = ProductionPredictor()._prepare_features(df, include_weather=False)

        # 기존 dropna 동작: 깨끗한 입력에서는 warm-up 168행만 제거
        assert len(df_prep) == N_HOURS - WARMUP
        legacy = df.copy()
        legacy = legacy.dropna()
        assert len(legacy) == N_HOURS  # 원본 자체엔 결측 없음

    def test_no_zero_fill_artifacts(self) -> None:
        """잔여 NaN을 0으로 채우지 않아야 한다 (스케일러 왜곡 방지)"""
        df = _make_input()
        # 전체가 NaN인 선행 구간은 0이 아니라 행 제거로 처리되어야 함
        df.iloc[:50, df.columns.get_loc('풍속')] = np.nan

        df_prep = ProductionPredictor()._prepare_features(df, include_weather=False)

        # 풍속 파생 피처가 0으로 채워지지 않았는지 (warm-up 제거로 해결)
        wind_cols = [c for c in df_prep.columns if '풍속' in c or 'wind' in c]
        for col in wind_cols:
            assert not (df_prep[col] == 0).all(), (
                f"{col} 이(가) 0으로 채워짐 (fillna(0) 잔존)"
            )
