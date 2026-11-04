"""rolling-origin 세부 구간(segment) 지표 회귀 테스트."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.evaluation.rolling_origin import (
    BASELINE_PERSISTENCE,
    OFF_PEAK_HOURS_END,
    PEAK_HOURS_START,
    rolling_origin_error_frame,
    segment_metrics,
)


def _daily_cycle_series(n: int, start: str = "2024-01-01") -> pd.Series:
    """피크 시간(09-21시)에 수요가 높은 결정론적 일주기 시계열."""
    idx = pd.DatetimeIndex(pd.date_range(start=start, periods=n, freq="h"))
    hours = np.asarray([t.hour for t in idx])
    values = 50 + 30 * np.sin(np.pi * (hours - 6) / 15.0).clip(min=0)
    return pd.Series(values, index=idx)


def _persistence_fn(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    return np.full(len(test_index), float(train.iloc[-1]))


class TestErrorFrame:
    def test_error_frame_has_per_timestamp_rows_for_all_models(self):
        series = _daily_cycle_series(400)
        frame = rolling_origin_error_frame(
            series, _persistence_fn, model_name="m", n_origins=2, horizon=24, min_train=168
        )
        assert {"model", "origin", "timestamp", "y_true", "y_pred"} <= set(frame.columns)
        # 모델 + 표준 기준 3종 = 4개 모델 × 2 기점 × 24시간
        assert len(frame) == 4 * 2 * 24
        assert frame["abs_error"].notna().all()
        assert frame["sq_error"].notna().all()

    def test_error_frame_is_causal(self):
        series = _daily_cycle_series(400)
        frame = rolling_origin_error_frame(
            series, _persistence_fn, model_name="m", n_origins=1, horizon=24, min_train=168
        )
        for origin in frame["origin"].unique():
            assert (pd.DatetimeIndex(frame["timestamp"]) > origin).all()


class TestSegmentMetrics:
    def _frame(self, n_origins: int = 2) -> pd.DataFrame:
        series = _daily_cycle_series(400)
        return rolling_origin_error_frame(
            series, _persistence_fn, model_name="m", n_origins=n_origins, horizon=24, min_train=168
        )

    def test_segments_cover_all_timestamps_exactly_once(self):
        seg = segment_metrics(self._frame())
        for seg_type in ("peak_hour", "weekday_weekend"):
            subset = seg[seg["segment_type"] == seg_type]
            # 모델별로 전체 48 타임스탬프를 세그먼트가 정확히 한 번씩 커버
            counts = subset.groupby("model")["n"].sum()
            assert (counts == 48).all(), f"{seg_type} 커버 불일치: {counts.to_dict()}"

    def test_peak_segment_hours_match_constant(self):
        seg = segment_metrics(self._frame())
        peak = seg[(seg["segment_type"] == "peak_hour") & (seg["segment"] == "peak")]
        assert (peak["n"] > 0).all()
        frame = self._frame()
        peak_hours = {
            t.hour for t in frame["timestamp"] if PEAK_HOURS_START <= t.hour < OFF_PEAK_HOURS_END
        }
        assert peak_hours  # 피크 정의가 테스트 데이터와 교차함

    def test_holiday_segment_uses_provided_dates(self):
        frame = self._frame()
        # 기점이 14시에 착지하므로 평가창은 1/15~1/17 — 1/16을 휴일로 지정
        holiday = pd.Timestamp("2024-01-16").date()
        assert holiday in set(pd.to_datetime(frame["timestamp"]).dt.normalize().dt.date)
        seg = segment_metrics(frame, holidays={holiday})
        holiday_rows = seg[seg["segment_type"] == "holiday"]
        assert len(holiday_rows) > 0
        assert set(holiday_rows["segment"]) == {"holiday", "non_holiday"}

    def test_segment_mae_matches_manual_computation(self):
        frame = self._frame()
        model_frame = frame[frame["model"] == "m"]
        # 기점이 14시(피크)에 착지 → 피크 세그먼트로 수동 검증 (항상 존재)
        hours = pd.DatetimeIndex(model_frame["timestamp"]).hour
        peak_mask = (hours >= PEAK_HOURS_START) & (hours < OFF_PEAK_HOURS_END)
        manual_mae = float(model_frame.loc[peak_mask, "abs_error"].mean())
        seg = segment_metrics(frame)
        row = seg[
            (seg["model"] == "m")
            & (seg["segment_type"] == "peak_hour")
            & (seg["segment"] == "peak")
        ].iloc[0]
        assert row["MAE"] == pytest.approx(manual_mae, rel=1e-12)

    def test_persistence_segmentation_discriminates(self):
        """기점이 피크(14시)에 착지하면 persistence는 비피크(야간)에서 훨씬 나쁘다.

        세그먼트 분해가 오차의 시간대 편중을 실제로 구분해내는지 검증한다.
        """
        seg = segment_metrics(self._frame())
        piv = seg[seg["model"] == BASELINE_PERSISTENCE].pivot_table(
            index="segment_type", columns="segment", values="MAE"
        )
        peak_mae = piv.loc["peak_hour", "peak"]
        offpeak_mae = piv.loc["peak_hour", "off_peak"]
        assert offpeak_mae > 2 * peak_mae
