"""Regression tests for degenerate ensemble and explanation inputs."""

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from src.analysis.explainability import PerturbationExplainer
from src.models.ensemble import WeightedAverageEnsemble


@pytest.mark.parametrize("n_models", [1, 2])
def test_grid_fit_skips_zero_weight_candidates(n_models: int) -> None:
    ensemble = WeightedAverageEnsemble([nn.Identity() for _ in range(n_models)])
    predictions = [torch.tensor([float(i), float(i + 1)]) for i in range(n_models)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = ensemble.fit(predictions, predictions[0], method="grid")
    assert np.isfinite(result["final_mse"])
    assert np.sum(result["optimal_weights"]) > 0


def test_singleton_perturbation_produces_finite_explanation() -> None:
    model = nn.Linear(2, 1)
    explainer = PerturbationExplainer(model, n_samples=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        explanation = explainer.explain(torch.tensor([[1.0, 2.0]]), ["a", "b"])
    assert np.isfinite(explanation.base_value)
    assert all(np.isfinite(c.contribution) for c in explanation.contributions)
