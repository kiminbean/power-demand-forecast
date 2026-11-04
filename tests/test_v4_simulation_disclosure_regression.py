"""Disclosure regressions for still-unlabeled simulated values in v4 dashboard.

감사에서 보고만 되고 수정되지 않은 4건에 대한 회귀 테스트:

1. 발전소 지도 팝업의 무작위 지터 기반 '현재 발전량/이용률'이
   등록부(registry) 실측값처럼 보이지 않도록 추정 표시가 필요하다.
2. 공급 도넛/시간대별 발전량 차트는 시뮬레이션임을 표시해야 한다.
3. SMP 이전 시간값 부재 시 current*0.98 가짜값으로 변화율을
   만들지 않는다 — 이전값이 없으면 화살표/변화율을 숨긴다.
4. '현재 수요' 카드는 관측 시각과 경과 시간을 함께 표시해야 한다
   (EPSIS 파일 데이터는 며칠 묵은 값일 수 있다).

streamlit이 .venv에 없으므로 AST/소스세그먼트 기반으로 검증한다.
"""

import ast
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pytest


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


def _extract_main() -> ast.FunctionDef:
    """Return the AST node of main()."""
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("function main not found in app_v4.py")


def _eval_card_html(title: str, ctx: dict) -> str:
    """Evaluate the st.markdown f-string in main() for the card titled `title`."""
    main = _extract_main()
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


def _make_smp_history_for_hours(hour_values: Dict[int, float]) -> pd.DataFrame:
    """지정한 시간대에만 유효한 SMP 행을 담은 히스토리 DataFrame 생성."""
    rows = []
    for day in ("2026-10-31", "2026-11-01"):
        for hour, value in hour_values.items():
            rows.append({
                "date": pd.Timestamp(day),
                "timestamp": f"{day} {hour:02d}:00",
                "hour": hour,
                "smp_jeju": value,
                "smp_mainland": value,
            })
    return pd.DataFrame(rows)


def _run_get_smp_data(smp_df: pd.DataFrame) -> Dict:
    """get_smp_data를 히스토리 주입으로 실행한 결과 반환."""
    namespace: dict = {
        "pd": pd,
        "datetime": datetime,
        "timedelta": timedelta,
        "random": random,
        "Dict": Dict,
        "load_smp_history": lambda: smp_df,
    }
    exec(compile(_extract_function("get_smp_data"), str(DASHBOARD), "exec"), namespace)
    return namespace["get_smp_data"]()


# ============================================================================
# 1. 발전소 지도 팝업: 지터 발전량/이용률의 출처 오도
# ============================================================================


def test_map_popup_marks_generation_and_utilization_as_estimated() -> None:
    """팝업의 발전량/이용률은 시뮬레이션 추정치임을 밝혀야 한다."""
    seg = _extract_function("create_jeju_map")

    # 행 라벨 자체에 추정 표시
    assert "현재 발전량 (추정)" in seg
    assert "이용률 (추정)" in seg
    # 등록부 출처 라인과 별개로 시뮬레이션 고지
    assert "시뮬레이션 추정치" in seg
    assert "실측 아님" in seg


# ============================================================================
# 2. 공급 도넛/시간대별 발전량 차트: 시뮬레이션 라벨 부재
# ============================================================================


def test_supply_donut_discloses_simulation() -> None:
    """공급 도넛은 시뮬레이션 추정임을 차트에 표기해야 한다."""
    seg = _extract_function("create_supply_donut")
    assert "시뮬레이션" in seg
    assert "추정" in seg


def test_generation_timeline_discloses_simulation() -> None:
    """시간대별 발전량 차트(시뮬레이션 값)는 실측 아님을 표기해야 한다."""
    seg = _extract_function("create_generation_timeline")
    assert "시뮬레이션" in seg
    assert "실측 아님" in seg


def test_main_section_headers_disclose_simulation() -> None:
    """공급 구성/시간대별 발전량 섹션 제목에 시뮬레이션 표시가 있어야 한다."""
    main_seg = ast.get_source_segment(SOURCE, _extract_main())
    assert main_seg is not None
    assert "전력 공급 구성 (시뮬레이션" in main_seg
    assert "시간대별 발전량 (시뮬레이션)" in main_seg


# ============================================================================
# 3. SMP 변화 화살표: current*0.98 가짜 이전값 폐지
# ============================================================================


def test_smp_source_no_longer_fabricates_previous_value() -> None:
    """get_smp_data에는 current*0.98 가짜 폴백이 남아 있지 않아야 한다."""
    seg = _extract_function("get_smp_data")
    assert "0.98" not in seg


def test_smp_change_hidden_when_previous_hour_missing() -> None:
    """이전 시간 실측값이 없으면 변화율을 계산하지 않는다 (None + 플래그)."""
    current_hour = datetime.now().hour
    cur = current_hour if current_hour > 0 else 24
    prev = current_hour - 1 if current_hour > 1 else 24
    # 현재/이전 시간이 아닌 임의 시간대에만 데이터 존재
    extra = next(h for h in range(1, 25) if h not in (cur, prev))

    result = _run_get_smp_data(_make_smp_history_for_hours({extra: 150.0}))

    assert result["prev_available"] is False
    assert result["change"] is None
    assert result["change_pct"] is None


def test_smp_change_uses_real_previous_value_when_present() -> None:
    """이전 시간 실측값이 있으면 실제 차이를 사용한다 (가짜 2%가 아님)."""
    current_hour = datetime.now().hour
    cur = current_hour if current_hour > 0 else 24
    prev = current_hour - 1 if current_hour > 1 else 24

    cur_value = 200.0
    prev_value = 100.0 if cur != prev else cur_value
    result = _run_get_smp_data(_make_smp_history_for_hours({cur: cur_value, prev: prev_value}))

    assert result["prev_available"] is True
    assert result["change"] == pytest.approx(cur_value - prev_value)
    assert result["change_pct"] == pytest.approx((cur_value - prev_value) / prev_value * 100)


def test_smp_card_hides_arrow_when_previous_missing() -> None:
    """SMP 카드의 화살표는 prev_available 조건 안에 있고, 미만족 시 안내 문구."""
    main = _extract_main()
    guarded = False
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        seg = ast.get_source_segment(SOURCE, node) or ""
        if "prev_available" in seg and "smp_arrow" in seg and "이전 시간 데이터 없음" in seg:
            guarded = True
    assert guarded, "SMP 변화 화살표는 prev_available 조건으로 보호되어야 한다"

    # 카드 f-string이 조건부 문구 변수를 실제로 렌더에 사용하는지 확인
    html = _eval_card_html(
        "최근 SMP (제주)",
        {"smp_data": {"current": 200.0}, "smp_change_class": "", "smp_change_html": "이전 시간 데이터 없음"},
    )
    assert "이전 시간 데이터 없음" in html


# ============================================================================
# 4. 현재 수요 카드: 관측 시각·경과시간 표시
# ============================================================================


def _run_format_observation_age(data_date: object, now: datetime) -> Tuple[str, str]:
    """_format_observation_age 헬퍼를 격리 실행한다."""
    namespace: dict = {
        "pd": pd,
        "datetime": datetime,
        "timedelta": timedelta,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(
        compile(_extract_function("_format_observation_age"), str(DASHBOARD), "exec"),
        namespace,
    )
    return namespace["_format_observation_age"](data_date, now=now)


def test_observation_age_formats_date_only_input() -> None:
    """날짜 문자열(EPSIS)은 'N일 N시간 경과' 형태로 계산된다."""
    observed, elapsed = _run_format_observation_age(
        "2026-11-01", now=datetime(2026, 11, 2, 3, 30)
    )
    assert observed == "2026-11-01"
    assert elapsed == "1일 3시간 경과"


def test_observation_age_formats_datetime_input_and_garbage() -> None:
    """시각 문자열(KPX)은 'N시간 경과', 파싱 불가값은 'N/A'."""
    observed, elapsed = _run_format_observation_age(
        "2026-11-02 05:00", now=datetime(2026, 11, 2, 7, 0)
    )
    assert observed == "2026-11-02 05:00"
    assert elapsed == "2시간 경과"

    assert _run_format_observation_age("N/A", now=datetime(2026, 11, 2)) == ("N/A", "N/A")
    assert _run_format_observation_age("최신", now=datetime(2026, 11, 2)) == ("최신", "N/A")


def test_demand_card_shows_observation_time_and_age() -> None:
    """'현재 수요' 카드에 관측 시각·경과시간이 렌더되어야 한다."""
    html = _eval_card_html(
        "현재 수요",
        {
            "power_status": {"demand": 850.0},
            "reserve_class": "status-online",
            "reserve_rate": 15.0,
            "reserve_text": "정상",
            "demand_observed_text": "2026-11-01",
            "demand_elapsed_text": "1일 3시간 경과",
        },
    )
    assert "관측" in html
    assert "2026-11-01" in html
    assert "1일 3시간 경과" in html
