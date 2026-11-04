"""Diagnostics for quantile forecasts with possible crossing."""

import numpy as np
import pytest


def test_crossings_reported_without_discarding_individual_losses() -> None:
    from src.evaluation import diagnose_quantile_forecast

    y = np.array([1.0, 2.0, 3.0, 4.0])
    q10 = np.array([0.0, 3.0, 2.0, 5.0])
    q50 = np.array([1.0, 2.0, 5.0, 4.0])
    q90 = np.array([2.0, 4.0, 4.0, 3.0])
    result = diagnose_quantile_forecast(y, q10, q50, q90)

    assert result["sample_count"] == 4
    assert result["crossing_count"] == 3
    assert result["crossing_rate_pct"] == pytest.approx(75.0)
    assert result["q10_gt_q50_count"] == 2
    assert result["q50_gt_q90_count"] == 2
    assert result["pinball_q10"] == pytest.approx(0.5)
    assert result["pinball_q50"] == pytest.approx(0.25)
    assert result["pinball_q90"] == pytest.approx(0.325)


def test_equal_quantiles_are_not_crossed() -> None:
    from src.evaluation import diagnose_quantile_forecast

    values = np.array([1.0, 2.0])
    result = diagnose_quantile_forecast(values, values, values, values)
    assert result["crossing_count"] == 0
    assert result["crossing_rate_pct"] == 0.0


def test_diagnostics_validate_inputs() -> None:
    from src.evaluation import diagnose_quantile_forecast

    with pytest.raises(ValueError):
        diagnose_quantile_forecast(np.array([np.nan]), *[np.ones(1)] * 3)
    with pytest.raises(ValueError):
        diagnose_quantile_forecast(np.ones((1, 2)), *[np.ones((1, 2))] * 3)
    with pytest.raises(ValueError):
        diagnose_quantile_forecast(np.ones(2), *[np.ones(1)] * 3)


def test_public_quantile_exports() -> None:
    import src.evaluation as evaluation
    from src.evaluation import metrics

    for name in ("pinball_loss", "coverage", "interval_score", "evaluate_quantile_forecast"):
        assert name in evaluation.__all__
        assert getattr(evaluation, name) is getattr(metrics, name)
