"""Validation of explicit quantile forecast metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    coverage,
    evaluate_quantile_forecast,
    interval_score,
    pinball_loss,
)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_observations_rejected_by_all_metrics(bad: float) -> None:
    """Missing actuals cannot silently count as outside the interval."""
    y = np.array([1.0, bad])
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 2.0])
    for call in (
        lambda: pinball_loss(y, lower, 0.1),
        lambda: coverage(y, lower, upper),
        lambda: interval_score(y, lower, upper, 0.2),
        lambda: evaluate_quantile_forecast(y, lower, np.ones(2), upper),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_rejected(bad: float) -> None:
    y = np.array([1.0])
    for call in (
        lambda: pinball_loss(y, np.array([bad]), 0.5),
        lambda: coverage(y, np.array([0.0]), np.array([bad])),
        lambda: interval_score(y, np.array([bad]), np.array([2.0]), 0.2),
        lambda: evaluate_quantile_forecast(y, np.array([0.0]), np.array([bad]), np.array([2.0])),
    ):
        with pytest.raises(ValueError):
            call()


def test_empty_arrays_rejected() -> None:
    empty = np.array([])
    for call in (
        lambda: pinball_loss(empty, empty, 0.5),
        lambda: coverage(empty, empty, empty),
        lambda: interval_score(empty, empty, empty, 0.2),
        lambda: evaluate_quantile_forecast(empty, empty, empty, empty),
    ):
        with pytest.raises(ValueError):
            call()


def test_inverted_intervals_rejected() -> None:
    y, lower, upper = np.array([1.0]), np.array([2.0]), np.array([0.0])
    for call in (
        lambda: coverage(y, lower, upper),
        lambda: interval_score(y, lower, upper, 0.2),
        lambda: evaluate_quantile_forecast(y, lower, y, upper),
    ):
        with pytest.raises(ValueError):
            call()


def test_crossed_median_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_quantile_forecast(np.array([1.0]), np.array([0.0]), np.array([3.0]), np.array([2.0]))


def test_multidimensional_arrays_rejected() -> None:
    matrix = np.array([[1.0, 2.0]])
    for call in (
        lambda: pinball_loss(matrix, matrix, 0.5),
        lambda: coverage(matrix, matrix, matrix),
        lambda: interval_score(matrix, matrix, matrix, 0.2),
        lambda: evaluate_quantile_forecast(matrix, matrix, matrix, matrix),
    ):
        with pytest.raises(ValueError):
            call()
