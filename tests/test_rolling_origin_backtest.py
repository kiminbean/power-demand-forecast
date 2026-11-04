"""rolling-origin 백테스트 회귀 테스트."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.evaluation.rolling_origin import (
    BASELINE_PERSISTENCE,
    BASELINE_SEASONAL_NAIVE_168,
    BASELINE_SEASONAL_NAIVE_24,
    persistence_forecast,
    rolling_origin_splits,
    run_rolling_origin_backtest,
    seasonal_naive_forecast,
)


def _hourly_series(n: int, start: str = "2024-01-01") -> pd.Series:
    idx = pd.DatetimeIndex(pd.date_range(start=start, periods=n, freq="h"))
    rng = np.random.default_rng(42)
    daily = 100 + 20 * np.sin(2 * np.pi * (np.asarray([t.hour for t in idx]) / 24.0))
    return pd.Series(daily + rng.normal(0, 1, n), index=idx)


class TestRollingOriginSplits:
    def test_splits_are_strictly_temporal(self):
        series = _hourly_series(400)
        splits = rolling_origin_splits(series, n_origins=3, horizon=24, min_train=168)
        assert len(splits) == 3
        for train_index, test_index in splits:
            # 학습 종료 시점이 평가 시작보다 항상 이전이어야 한다.
            assert train_index[-1] < test_index[0]
            assert len(test_index) == 24

    def test_origins_advance_and_do_not_overlap_test_windows(self):
        series = _hourly_series(400)
        splits = rolling_origin_splits(series, n_origins=3, horizon=24, min_train=168)
        origins = [train_index[-1] for train_index, _ in splits]
        assert origins == sorted(origins)
        test_windows = [test_index for _, test_index in splits]
        for earlier, later in zip(test_windows, test_windows[1:]):
            assert earlier[-1] < later[0]

    def test_insufficient_data_raises_with_context(self):
        series = _hourly_series(100)
        with pytest.raises(ValueError, match="at least"):
            rolling_origin_splits(series, n_origins=5, horizon=24, min_train=168)


class TestBaselines:
    def test_persistence_repeats_last_observation(self):
        train = _hourly_series(200)
        pred = persistence_forecast(train, horizon=6)
        assert np.allclose(pred, train.iloc[-1])

    def test_seasonal_naive_24_uses_previous_day_same_hour(self):
        train = _hourly_series(200)
        test_index = pd.date_range(
            cast(pd.Timestamp, train.index[-1]) + pd.Timedelta(hours=1), periods=24, freq="h"
        )
        pred = seasonal_naive_forecast(train, test_index, seasonality=24)
        expected = np.array([train.loc[ts - pd.Timedelta(hours=24)] for ts in test_index])
        assert np.allclose(pred, expected)

    def test_seasonal_naive_insufficient_history_raises(self):
        train = _hourly_series(10)
        test_index = pd.date_range(
            cast(pd.Timestamp, train.index[-1]) + pd.Timedelta(hours=1), periods=3, freq="h"
        )
        with pytest.raises(ValueError, match="seasonal naive"):
            seasonal_naive_forecast(train, test_index, seasonality=24)


class TestBacktestCausality:
    def test_forecast_fn_receives_only_past_data(self):
        """기점 이후 실측을 바꿔도 예측 함수에 전달되는 학습 데이터는 불변해야 한다."""
        series = _hourly_series(400)
        seen_lengths = []

        def forecast_fn(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
            seen_lengths.append(len(train))
            return persistence_forecast(train, len(test_index))

        run_rolling_origin_backtest(
            series, forecast_fn, model_name="probe", n_origins=3, horizon=24, min_train=168
        )
        # 모든 기점에서 학습 길이는 min_train 이상·전체 이하이고 단조 증가한다.
        assert seen_lengths == sorted(seen_lengths)
        assert all(168 <= n < len(series) for n in seen_lengths)

    def test_future_mutation_does_not_change_baseline_predictions(self):
        series = _hourly_series(400)
        splits = rolling_origin_splits(series, n_origins=1, horizon=24, min_train=168)
        train_index, test_index = splits[0]
        train = series.loc[train_index]
        before = seasonal_naive_forecast(train, test_index, seasonality=24)
        # 미래 구간을 임의로 바꿔도 기준 예측은 동일해야 한다.
        poisoned = series.copy()
        poisoned.loc[test_index] = 9999.0
        after = seasonal_naive_forecast(poisoned.loc[train_index], test_index, seasonality=24)
        assert np.allclose(before, after)


class TestBacktestResults:
    def test_perfect_model_scores_zero_error(self):
        series = _hourly_series(400)

        def oracle(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
            return series.loc[test_index].to_numpy(dtype=float)

        per_origin, summary = run_rolling_origin_backtest(
            series, oracle, model_name="oracle", n_origins=2, horizon=24, min_train=168
        )
        oracle_rows = per_origin[per_origin["model"] == "oracle"]
        assert len(oracle_rows) == 2
        assert np.allclose(oracle_rows["MAE"], 0.0, atol=1e-9)
        assert np.allclose(oracle_rows["RMSE"], 0.0, atol=1e-9)

    def test_summary_contains_model_and_all_baselines(self):
        series = _hourly_series(400)

        def dummy(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
            return np.zeros(len(test_index))

        _, summary = run_rolling_origin_backtest(
            series, dummy, model_name="dummy", n_origins=2, horizon=24, min_train=168
        )
        names = set(summary["model"])
        assert names == {
            "dummy",
            BASELINE_PERSISTENCE,
            BASELINE_SEASONAL_NAIVE_24,
            BASELINE_SEASONAL_NAIVE_168,
        }
        assert (summary["n_origins"] == 2).all()
        assert (summary["horizon"] == 24).all()

    def test_baseline_summary_matches_persistence_definition(self):
        """persistence 기준 요약은 직접 계산한 MAE와 일치해야 한다."""
        series = _hourly_series(400)

        def identity_persistence(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
            return np.full(len(test_index), float(train.iloc[-1]))

        _, summary = run_rolling_origin_backtest(
            series,
            identity_persistence,
            model_name="copy_of_persistence",
            n_origins=2,
            horizon=24,
            min_train=168,
        )
        pivoted = summary.set_index("model")["MAE"]
        assert np.isclose(
            pivoted["copy_of_persistence"], pivoted[BASELINE_PERSISTENCE], rtol=1e-12
        )
