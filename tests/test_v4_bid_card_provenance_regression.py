"""Regression coverage for the illustrative bid card's provenance."""

import ast
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"


def test_bid_card_discloses_fixed_heuristic_and_example_probability() -> None:
    """The displayed bid is calculated, but its probability is not a forecast."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignment = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "optimal_bid" for target in node.targets)
    )
    card = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and any(isinstance(part, ast.Constant) and "입찰 추천" in str(part.value) for part in node.args[0].values)
    )
    for current in (100.0, 200.0):
        optimal_bid = eval(compile(ast.Expression(assignment.value), str(DASHBOARD), "eval"), {"smp_data": {"current": current}})
        html = eval(compile(ast.Expression(card.args[0]), str(DASHBOARD), "eval"), {"optimal_bid": optimal_bid})
        assert f"{current * 0.95:.1f} 원/kWh" in html
        assert "85%" in html
        assert "고정 할인율" in html
        assert "예시" in html
        assert "실제 낙찰 확률 아님" in html
