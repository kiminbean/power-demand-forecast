"""tools/run_smp_rolling_origin.py 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from tools.run_smp_rolling_origin import (
    INTERVAL_TARGET,
    Z_80,
    build_ar_features,
    fix_hour_24,
    fit_ridge_ar,
    interval_coverage,
    load_smp_series,
    make_ridge_ar_forecaster,
)


class TestFixHour24:
    """24:00 표기 변환 테스트."""

    def test_converts_24_to_next_midnight(self):
        assert fix_hour_24("2025-01-01 24:00") == pd.Timestamp("2025-01-02 00:00")

    def test_normal_timestamp_unchanged(self):
        assert fix_hour_24("2025-01-01 13:00") == pd.Timestamp("2025-01-01 13:00")


class TestBuildARFeatures:
    """AR 피처 생성 테스트."""

    def test_feature_shape_and_lags(self):
        rng = pd.date_range("2025-01-01", periods=400, freq="h")
        values = np.arange(400, dtype=float)
        X = build_ar_features(values, rng)
        max_lag = 336
        assert X.shape == (400 - max_lag, 10)
        # 첫 행은 시각 336을 예측: lag1=values[335]
        assert X[0, 0] == 335.0
        assert X[0, 3] == values[336 - 336]
        # 마지막 행: lag1=values[398]
        assert X[-1, 0] == 398.0
        # hour 피처는 [-1, 1] 범위의 sin/cos
        assert np.all((X[:, 6] >= -1) & (X[:, 6] <= 1))
        # 대상 시각만큼의 hour sin/cos 반영: 마지막 행 = 399시각(=17일 15시)
        ts_last = rng[399]
        assert X[-1, 6] == np.sin(2 * np.pi * ts_last.hour / 24)

    def test_requires_more_than_max_lag(self):
        with pytest.raises(ValueError):
            build_ar_features(
                np.arange(336.0), pd.date_range("2025-01-01", periods=336, freq="h")
            )


class TestRidgeARForecaster:
    """Ridge AR 예측기 테스트."""

    def _synthetic_series(self) -> pd.Series:
        rng = pd.date_range("2025-01-01", periods=600, freq="h")
        # 주기성이 있는 결정론적 시계열
        values = 100 + 10 * np.sin(2 * np.pi * rng.hour / 24) + 0.01 * np.arange(600)
        return pd.Series(values, index=rng, name="smp_mainland")

    def test_forecast_length_and_state(self):
        series = self._synthetic_series()
        fn = make_ridge_ar_forecaster(window_days=10)
        train = series.iloc[:500]
        test_index = series.index[500:524]
        preds = fn(train, test_index)
        assert preds.shape == (24,)
        assert fn.state["sigma"] is not None and fn.state["sigma"] > 0
        # 결정론성: 동일 입력 -> 동일 출력
        fn2 = make_ridge_ar_forecaster(window_days=10)
        np.testing.assert_allclose(fn2(train, test_index), preds)

    def test_forecast_reasonable_scale(self):
        series = self._synthetic_series()
        fn = make_ridge_ar_forecaster(window_days=10)
        preds = fn(series.iloc[:500], series.index[500:524])
        assert np.all((preds > 80) & (preds < 120))


class TestIntervalCoverage:
    """구간 커버리지 계산 테스트."""

    def test_perfect_coverage_at_zero_sigma(self):
        rows = [
            {"origin": "o1", "y_true": 10.0, "y_pred": 10.0, "sigma": 1.0},
            {"origin": "o1", "y_true": 10.5, "y_pred": 10.0, "sigma": 1.0},
        ]
        result = interval_coverage(rows)
        assert result["coverage_pct"] == 100.0
        assert result["target_pct"] == INTERVAL_TARGET
        assert result["mean_width"] == pytest.approx(2 * Z_80, abs=1e-3)

    def test_zero_coverage_outside_interval(self):
        rows = [
            {"origin": "o1", "y_true": 50.0, "y_pred": 10.0, "sigma": 1.0},
        ]
        assert interval_coverage(rows)["coverage_pct"] == 0.0

    def test_per_origin_keys(self):
        rows = [
            {"origin": "o1", "y_true": 10.0, "y_pred": 10.0, "sigma": 1.0},
            {"origin": "o2", "y_true": 10.0, "y_pred": 10.0, "sigma": 1.0},
        ]
        result = interval_coverage(rows)
        assert set(result["per_origin"]) == {"o1", "o2"}


class TestLoadSmpSeries:
    """데이터 적재 테스트 (실제 CSV 존재 시)."""

    def test_loads_hourly_sorted(self):
        from pathlib import Path

        data_path = Path(__file__).resolve().parents[1] / "data" / "smp" / "smp_5years_epsis.csv"
        if not data_path.exists():
            pytest.skip("실데이터 부재")
        series = load_smp_series(data_path)
        assert series.index.is_monotonic_increasing
        assert series.index.freq is not None or len(series) > 0
        assert series.notna().all()
