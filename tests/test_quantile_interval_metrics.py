"""
분위수·구간 예측 평가 지표 회귀 테스트
=====================================

EVAL: pinball loss, interval (Winkler) score, 실측 커버리지.
SMP q10/q50/q90 등 분위수 예측의 품질을 평가하기 위한 신규 지표 계약.

기존 compute_prediction_intervals (잔차 정규분포 가정)과 달리
명시적 하한/상한 분위수 예측에 대해 계산한다.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    coverage,
    evaluate_quantile_forecast,
    interval_score,
    pinball_loss,
)


class TestPinballLoss:
    """pinball loss (quantile loss) 테스트."""

    def test_perfect_prediction_zero_loss(self):
        # 완벽한 예측은 손실 0
        y = np.array([10.0, 20.0, 30.0])
        q = y.copy()
        assert pinball_loss(y, q, quantile=0.5) == 0.0

    def test_known_value_over_prediction(self):
        # y=10, q=8, τ=0.9: y-q=2>=0 → 2*τ = 1.8
        y = np.array([10.0])
        q = np.array([8.0])
        assert pinball_loss(y, q, quantile=0.9) == pytest.approx(1.8)

    def test_known_value_under_prediction(self):
        # y=10, q=8, τ=0.1: y-q=2>=0 → 2*(1-τ)? 아니오 — 표준 pinball:
        # ρ_τ(u) = u*(τ - 1{u<0}), u>=0 → u*τ
        assert pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.1) == pytest.approx(0.2)

    def test_under_prediction_penalized_more_at_high_quantile(self):
        # y=10, q=12 (과대예측), u=-2<0 → |u|*(1-τ)
        y = np.array([10.0])
        q = np.array([12.0])
        # τ=0.9 → 2*0.1 = 0.2, τ=0.1 → 2*0.9 = 1.8
        assert pinball_loss(y, q, quantile=0.9) == pytest.approx(0.2)
        assert pinball_loss(y, q, quantile=0.1) == pytest.approx(1.8)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0, 2.0]), np.array([1.0]), quantile=0.5)

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0]), np.array([1.0]), quantile=1.5)


class TestCoverage:
    """실측 커버리지 테스트."""

    def test_full_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lower = np.zeros(3)
        upper = np.array([4.0, 4.0, 4.0])
        assert coverage(y, lower, upper) == pytest.approx(100.0)

    def test_partial_coverage(self):
        # 3개 중 2개만 구간 내 → 66.67%
        y = np.array([1.0, 5.0, 3.0])
        lower = np.zeros(3)
        upper = np.array([4.0, 4.0, 4.0])
        assert coverage(y, lower, upper) == pytest.approx(200.0 / 3.0, rel=1e-6)

    def test_boundary_inclusive(self):
        # 경계값은 구간에 포함 (폐구간)
        y = np.array([0.0, 4.0])
        lower = np.zeros(2)
        upper = np.array([4.0, 4.0])
        assert coverage(y, lower, upper) == pytest.approx(100.0)


class TestIntervalScore:
    """Winkler interval score 테스트 (중앙 (1-α) 구간)."""

    def test_inside_interval_equals_width(self):
        # 실측이 구간 내 → 점수 = 폭
        y = np.array([10.0])
        lower = np.array([8.0])
        upper = np.array([12.0])
        # α=0.2 (80% 구간)
        assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(4.0)

    def test_below_lower_adds_penalty(self):
        # y=6, [8,12], α=0.2 → 4 + (2/0.2)*(8-6) = 4 + 10*2 = 24
        y = np.array([6.0])
        lower = np.array([8.0])
        upper = np.array([12.0])
        assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(24.0)

    def test_above_upper_adds_penalty(self):
        # y=15, [8,12], α=0.2 → 4 + 10*(15-12) = 34
        y = np.array([15.0])
        lower = np.array([8.0])
        upper = np.array([12.0])
        assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(34.0)

    def test_mean_over_multiple_samples(self):
        # 두 표본: 내부(4) + 하단(24) → 평균 14
        y = np.array([10.0, 6.0])
        lower = np.array([8.0, 8.0])
        upper = np.array([12.0, 12.0])
        assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(14.0)

    def test_zero_width_containing_y_zero_score(self):
        y = np.array([10.0])
        lower = np.array([10.0])
        upper = np.array([10.0])
        assert interval_score(y, lower, upper, alpha=0.2) == pytest.approx(0.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            interval_score(np.array([1.0]), np.array([0.0]), np.array([2.0]), alpha=0.0)


class TestEvaluateQuantileForecast:
    """q10/q50/q90 통합 평가 테스트."""

    def test_returns_expected_keys(self):
        n = 50
        rng = np.random.default_rng(42)
        y = rng.normal(100, 10, n)
        q10 = y - 20.0
        q50 = y.copy()
        q90 = y + 20.0
        result = evaluate_quantile_forecast(y, q10=q10, q50=q50, q90=q90)
        for key in (
            "pinball_q10",
            "pinball_q50",
            "pinball_q90",
            "coverage_80",
            "interval_score_80",
            "mean_interval_width",
        ):
            assert key in result, f"키 누락: {key}"

    def test_perfect_median_zero_pinball(self):
        y = np.array([10.0, 20.0])
        result = evaluate_quantile_forecast(
            y, q10=y - 1.0, q50=y.copy(), q90=y + 1.0
        )
        assert result["pinball_q50"] == pytest.approx(0.0)

    def test_wide_interval_full_coverage(self):
        y = np.array([10.0, 20.0, 30.0])
        result = evaluate_quantile_forecast(
            y, q10=y - 100.0, q50=y.copy(), q90=y + 100.0
        )
        assert result["coverage_80"] == pytest.approx(100.0)
        assert result["mean_interval_width"] == pytest.approx(200.0)

    def test_q50_close_to_median_mae(self):
        # q50 pinball τ=0.5 → 절반 계수(0.5*|err|)이므로 MAE/2
        y = np.array([10.0, 20.0])
        q50 = y + 4.0
        result = evaluate_quantile_forecast(y, q10=y - 5.0, q50=q50, q90=y + 5.0)
        assert result["pinball_q50"] == pytest.approx(4.0 * 0.5)
