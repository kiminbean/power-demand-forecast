"""Ensure static model metrics and daily totals disclose their provenance."""

import ast
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"


def card_html(title: str) -> str:
    """Extract a static card's HTML without running Streamlit."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "markdown" or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            html = arg.value
        elif isinstance(arg, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in arg.values):
            html = "".join(v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
        else:
            continue
        if title in html:
            return html
    raise AssertionError(f"Card not found: {title}")


def test_model_metrics_disclose_fixed_reference_not_current_accuracy() -> None:
    html = card_html("모델 성능")
    assert "고정 참고값" in html
    assert "현재 예측 성능 아님" in html
    assert "평가 조건 미확인" in html
    for value in ("10.68%", "11.27 원/kWh", "82.5%", "0.59"):
        assert value in html


def test_daily_summary_discloses_static_examples_not_actual_results() -> None:
    html = card_html("일간 요약")
    assert "예시" in html
    assert "실제 일간 집계 아님" in html
    for value in ("12,450 MWh", "5,890 톤", "42.3%", "₩1.2B"):
        assert value in html
