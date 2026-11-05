"""구간 집계 지표(WMAPE) 정의 통일 회귀 테스트.

src.evaluation.metrics.wmape가 구간 집계 백분율 지표의 단일 정의이며,
rolling_origin segment_metrics가 동일 정의(WMAPE 컬럼)를 따르는지 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import mape, wmape
from src.evaluation.rolling_origin import (
    OFF_PEAK_HOURS_END,
    PEAK_HOURS_START,
    rolling_origin_error_frame,
    segment_metrics,
)


def _daily_cycle_series(n: int, start: str = "2024-01-01") -> pd.Series:
    idx = pd.DatetimeIndex(pd.date_range(start=start, periods=n, freq="h"))
    hours = np.asarray([t.hour for t in idx])
    values = 50 + 30 * np.sin(np.pi * (hours - 6) / 15.0).clip(min=0)
    return pd.Series(values, index=idx)


def _persistence_fn(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    return np.full(len(test_index), float(train.iloc[-1]))


class TestWmape:
    """metrics.wmape — 구간 집계 표준 정의."""

    def test_definition_matches_manual_formula(self):
        y_true = np.array([100.0, 200.0, 50.0])
        y_pred = np.array([90.0, 220.0, 55.0])
        expected = (10.0 + 20.0 + 5.0) / (100.0 + 200.0 + 50.0) * 100
        assert wmape(y_true, y_pred) == pytest.approx(expected, rel=1e-12)

    def test_equals_mean_abs_error_over_mean_abs_true(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(10, 300, size=500)
        y_pred = y_true + rng.normal(0, 15, size=500)
        manual = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true)) * 100
        assert wmape(y_true, y_pred) == pytest.approx(float(manual), rel=1e-12)

    def test_all_zero_true_returns_nan(self):
        """실측 절대값 합이 0이면 NaN — 0 나눗셈 폭발 정책 통일."""
        assert np.isnan(wmape(np.zeros(10), np.ones(10)))

    def test_near_zero_true_does_not_explode_unlike_mape(self):
        """0 근처 실측 하나가 있어도 점별 MAPE처럼 폭발하지 않는다."""
        y_true = np.array([1e-9, 100.0, 100.0])
        y_pred = np.array([1.0, 100.0, 100.0])
        point_mape = mape(y_true, y_pred)
        segment_wmape = wmape(y_true, y_pred)
        assert point_mape > 1e5  # 점별 MAPE는 0 근처에서 폭발
        assert segment_wmape < 1.0  # WMAPE는 안정적

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            wmape(np.array([1.0, 2.0]), np.array([1.0]))


class TestSegmentWmape:
    """segment_metrics가 wmape 정의를 WMAPE 컬럼으로 따르는지 검증."""

    def _frame(self) -> pd.DataFrame:
        series = _daily_cycle_series(400)
        return rolling_origin_error_frame(
            series, _persistence_fn, model_name="m", n_origins=2, horizon=24, min_train=168
        )

    def test_segments_use_wmape_column(self):
        seg = segment_metrics(self._frame())
        assert "WMAPE" in seg.columns
        assert "MAPE" not in seg.columns

    def test_wmape_matches_manual_computation(self):
        frame = self._frame()
        model_frame = frame[frame["model"] == "m"]
        hours = pd.DatetimeIndex(model_frame["timestamp"]).hour
        peak_mask = (hours >= PEAK_HOURS_START) & (hours < OFF_PEAK_HOURS_END)
        sub = model_frame.loc[peak_mask]
        manual = float(sub["abs_error"].mean()) / float(sub["y_true"].abs().mean()) * 100
        seg = segment_metrics(frame)
        row = seg[
            (seg["model"] == "m")
            & (seg["segment_type"] == "peak_hour")
            & (seg["segment"] == "peak")
        ].iloc[0]
        assert row["WMAPE"] == pytest.approx(manual, rel=1e-12)

    def test_wmape_finite_for_positive_series(self):
        seg = segment_metrics(self._frame())
        assert seg["WMAPE"].notna().all()
        assert (seg["WMAPE"] >= 0).all()
