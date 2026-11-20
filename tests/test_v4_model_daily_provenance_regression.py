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
    # b25eef55에서 카드 본문이 검증된 벤치마크 값으로 교체됨에 따라 기대값 갱신.
    # 회귀 검증 취지는 유지: 고정 참조 기준 공시 + 현재 예측 성능이 아님을 명시.
    assert "벤치마크 검증값" in html
    assert "README 운영 모델 성능 표 기준" in html
    assert "모델 성능이 아님" in html
    assert "학습에 안 쓴 기간" in html
    for value in ("4.67%", "17.0%", "11.75%", "0.929", "0.41", "0.203"):
        assert value in html


def test_daily_summary_discloses_static_examples_not_actual_results() -> None:
    html = card_html("일간 요약")
    assert "예시" in html
    assert "실제 일간 집계 아님" in html
    for value in ("12,450 MWh", "5,890 톤", "42.3%", "₩1.2B"):
        assert value in html
