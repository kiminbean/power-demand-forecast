"""Provenance of the static analysis cards shown on the dashboard."""

import ast
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"


def card_html(title: str) -> str:
    """Extract the literal HTML passed to the matching Streamlit card."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    return next(
        node.args[0].value
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and title in node.args[0].value
    )


def test_xai_card_identifies_examples_as_not_model_analysis() -> None:
    html = card_html("XAI 분석 요약")
    assert "예시" in html
    assert "실제 모델 분석 결과 아님" in html
    assert "Attention 가중치 기반 분석 결과" not in html
    assert "수요 증가 영향 15%" in html


def test_risk_card_identifies_examples_as_not_operational_alerts() -> None:
    html = card_html("리스크 알림")
    assert "예시" in html
    assert "실제 운영 경보 또는 입찰 권고 아님" in html
    assert "덕커브 주의" in html
    assert "입찰 기회" in html
