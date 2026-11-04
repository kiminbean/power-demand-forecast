"""Provenance of the frequency card displayed by the v4 dashboard."""

import ast
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"


def test_frequency_card_labels_simulated_value_at_display() -> None:
    """The frequency card must not present its random value as a measurement."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    card = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and any(isinstance(part, ast.Constant) and "계통 주파수" in str(part.value) for part in node.args[0].values)
    )
    html = eval(compile(ast.Expression(card.args[0]), str(DASHBOARD), "eval"), {"power_status": {"frequency": 60.0}})
    assert "시뮬레이션" in html
    assert "실측 아님" in html
    assert "60.00" in html
