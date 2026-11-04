"""Focused regression tests for the v4 dashboard gauge."""

import ast
from pathlib import Path

import plotly.graph_objects as go
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"


@pytest.fixture
def dashboard_gauges():
    """Load the real chart function and caller statements without starting Streamlit."""
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_gauge_chart")
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = {
        node.targets[0].id: node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"reserve_color", "reserve_gauge", "demand_gauge"}
    }
    namespace = {"go": go}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE), "exec"), namespace)

    def render(rate, demand=800):
        namespace["power_status"] = {"reserve_rate": rate, "demand": demand}
        for name in ("reserve_color", "reserve_gauge", "demand_gauge"):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[assignments[name]], type_ignores=[])), str(SOURCE), "exec"), namespace)
        return namespace["reserve_gauge"].data[0], namespace["demand_gauge"].data[0]

    return render


@pytest.mark.parametrize("rate", [-5, 0, 9.9, 10, 15, 30, 100.6])
def test_reserve_scale_contains_actual_value(dashboard_gauges, rate):
    reserve, _ = dashboard_gauges(rate)
    low, high = reserve.gauge.axis.range
    assert low <= rate <= high
    assert reserve.value == rate


def test_reserve_warning_is_low_not_high(dashboard_gauges):
    reserve, _ = dashboard_gauges(100.6)
    assert reserve.gauge.threshold.value == 10
    steps = reserve.gauge.steps
    assert list(steps[0].range)[1] == 10
    assert list(steps[1].range) == [10, 15]
    assert list(steps[-1].range)[1] >= 100.6
    assert steps[0].color != steps[-1].color


def test_demand_gauge_retains_existing_threshold(dashboard_gauges):
    _, demand = dashboard_gauges(100.6)
    assert list(demand.gauge.axis.range) == [0, 1200]
    assert demand.gauge.threshold.value == 1080
