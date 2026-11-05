"""Provenance/correctness regressions for still-active v4 dashboard cards.

Covers issues not addressed by the earlier labeled-card passes
(frequency/bid/model-performance/risk/daily-summary/analysis cards):

1. SMP 통계 카드의 일평균/일최고/일최저가 전체 기간 통계가 아닌
   최근 1일 통계여야 한다 (라벨은 '일'을 주장).
2. 기상 현황 카드는 난수 시뮬레이션 값임을 표시해야 한다.
3. 재생에너지 비율 카드는 추정치임을 표시해야 한다.
4. 푸터는 사용하지 않는 기상청 AMOS 출처를 주장하지 않는다.
5. _fix_timestamp_24h: 24:00은 다음 날 00:00으로 변환되어야 한다.
"""

import ast
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd


DASHBOARD = Path(__file__).resolve().parents[1] / "src/dashboard/app_v4.py"
SOURCE = DASHBOARD.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _extract_function(name: str) -> str:
    """Return the (decorator-free) source of a top-level function."""
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(SOURCE, node)
            assert seg is not None
            return seg
    raise AssertionError(f"function {name} not found in app_v4.py")


def _eval_card_html(title: str, ctx: dict) -> str:
    """Evaluate the st.markdown f-string in main() for the card titled `title`."""
    main = next(
        node for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    for node in ast.walk(main):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "markdown" or not node.args:
            continue
        arg = node.args[0]
        texts = (
            [part.value for part in arg.values if isinstance(part, ast.Constant)]
            if isinstance(arg, ast.JoinedStr)
            else ([arg.value] if isinstance(arg, ast.Constant) else [])
        )
        if any(title in str(t) for t in texts):
            return eval(compile(ast.Expression(arg), str(DASHBOARD), "eval"), dict(ctx))
    raise AssertionError(f"card '{title}' not found in main()")


def _make_smp_history() -> pd.DataFrame:
    """Two days whose full-period and latest-day statistics clearly differ."""
    rows = []
    for day, values in {
        "2026-10-31": [100.0] * 24,
        "2026-11-01": [190.0, 200.0, 210.0] * 8,
    }.items():
        for hour in range(1, 25):
            rows.append({
                "date": pd.Timestamp(day),
                "timestamp": f"{day} {hour:02d}:00",
                "hour": hour,
                "smp_jeju": values[hour - 1],
                "smp_mainland": values[hour - 1],
            })
    return pd.DataFrame(rows)


def test_smp_daily_stats_computed_from_latest_day_not_full_history() -> None:
    """'일평균/일최고/일최저' must describe the latest day, not the whole file."""
    namespace: dict = {
        "pd": pd,
        "datetime": datetime,
        "timedelta": timedelta,
        "random": random,
        "Dict": Dict,
        "load_smp_history": _make_smp_history,
    }
    exec(compile(_extract_function("get_smp_data"), str(DASHBOARD), "exec"), namespace)
    result = namespace["get_smp_data"]()

    # 최근 1일(2026-11-01) 통계: 평균 200, 최고 210, 최저 190
    assert result["daily_avg"] == 200.0
    assert result["daily_max"] == 210.0
    assert result["daily_min"] == 190.0
    # 전체 기간 통계(150/210/100)가 아니어야 한다
    assert result["daily_avg"] != 150.0
    assert result["daily_min"] != 100.0


def test_fix_timestamp_24h_maps_hour_24_to_next_day() -> None:
    """'2026-11-02 24:00' must become 2026-11-03 00:00, not same-day 00:00."""
    namespace = {"pd": pd}
    exec(compile(_extract_function("_fix_timestamp_24h"), str(DASHBOARD), "exec"), namespace)
    df = pd.DataFrame({"timestamp": ["2026-11-01 23:00", "2026-11-02 24:00", "not-a-date"]})
    fixed = namespace["_fix_timestamp_24h"](df)

    assert fixed["timestamp"].iloc[0] == pd.Timestamp("2026-11-01 23:00:00")
    assert fixed["timestamp"].iloc[1] == pd.Timestamp("2026-11-03 00:00:00")
    assert len(fixed) == 2  # 파싱 불가능한 행은 제거


def test_weather_card_labels_simulated_value_at_display() -> None:
    """The weather card must not present random values as measurements."""
    html = _eval_card_html(
        "기상 현황", {"weather": {"temperature": 12.3, "wind_speed": 4.5}}
    )
    assert "시뮬레이션" in html
    assert "실측 아님" in html


def test_weather_card_names_kma_only_when_api_weather_is_used() -> None:
    """API 서버 기상청 값을 쓸 때만 실황으로 표시하고, 시뮬레이션 고지는 빠진다."""
    html = _eval_card_html(
        "기상 현황", {"weather": {"temperature": 18.0, "wind_speed": 4.7, "source": "kma"}}
    )
    assert "기상청 초단기실황" in html
    assert "시뮬레이션" not in html
    assert "18" in html and "4.7" in html


def test_renewable_ratio_card_marks_estimated_provenance() -> None:
    """The renewable-ratio card must disclose that the ratio is estimated."""
    html = _eval_card_html("재생에너지 비율", {"power_status": {"renewable_ratio": 25.0}})
    assert "추정" in html


def test_footer_does_not_claim_unused_kma_amos_source() -> None:
    """The footer must not attribute data to 기상청 AMOS (never fetched here)."""
    html = _eval_card_html("데이터 출처", {})
    assert "AMOS" not in html
    assert "기상청" not in html
