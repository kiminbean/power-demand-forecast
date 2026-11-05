"""Streamlit 발전소 지도가 다른 공개 앱(API·v7 지도)과 같은 발전소·발전량을 쓰는지 검증한다.

2026-11-05: Streamlit은 태양광·ESS에 운영중 필터가 없어 분산형 태양광 합산값(557.3MW)·건설중 ESS를
포함했고(총 29개), 발전소별 발전량도 자체 난수 배분이라 v6·v7(/api/v1/resources)과 달랐다.
"""

import ast
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src/dashboard/app_v4.py"
PLANTS_JSON = ROOT / "data/jeju_plants/jeju_power_plants.json"
SOURCE = DASHBOARD.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _extract_function(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node)
    raise AssertionError(f"function {name} not found")


def _run(api_resources):
    namespace: dict = {
        "pd": pd, "np": np, "random": random, "datetime": datetime, "Dict": Dict,
        "load_power_plants_data": lambda: json.loads(PLANTS_JSON.read_text(encoding="utf-8")),
        "fetch_jeju_realtime": lambda: None,
        "fetch_backend_json": lambda path: api_resources if path == "/api/v1/resources" else None,
    }
    exec(compile(_extract_function("_finite_number"), str(DASHBOARD), "exec"), namespace)
    exec(compile(_extract_function("get_jeju_power_plants"), str(DASHBOARD), "exec"), namespace)
    return namespace["get_jeju_power_plants"]()


def test_renewable_and_ess_plants_match_the_shared_registry():
    df = _run(None)
    counts = df["type"].value_counts().to_dict()
    assert {t: counts.get(t, 0) for t in ("wind", "solar", "ess")} == {"wind": 16, "solar": 4, "ess": 4}
    assert "solar_distributed" not in set(df["id"])
    assert "ess_005" not in set(df["id"])  # 건설중
    assert set(df.loc[df["type"] == "thermal", "id"]) <= {"thermal_001", "thermal_002", "thermal_003"}


def test_plant_generation_uses_api_resource_outputs_by_id():
    api = [{"id": "wind_003", "current_output": 22.4}, {"id": "solar_001", "current_output": 0.3},
           {"id": "ess_002", "current_output": float("nan")}]
    df = _run(api).set_index("id")
    assert df.loc["wind_003", "generation"] == 22.4
    assert df.loc["solar_001", "generation"] == 0.3
    assert df.loc["wind_003", "data_source"] == "API 서버 시뮬레이션"
    # 유효하지 않은 값은 쓰지 않고 기존 추정을 유지한다
    assert df.loc["ess_002", "data_source"] != "API 서버 시뮬레이션"
