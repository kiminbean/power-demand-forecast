"""공개 웹 5종(v5·v6·v7·모바일·Streamlit)이 같은 원천 값을 보도록 백엔드 계약을 고정한다.

2026-11-05 교차 점검에서 발견한 불일치:
- 발전소 목록이 백엔드(20개)·v7 지도(21개)·데이터 파일마다 달랐다 → data/jeju_plants JSON 운영중 시설로 통일
- /dashboard/kpis 발전량(섬 전체 ML+고정 ESS)과 /resources 합계(포트폴리오 시뮬레이션)가 달랐다
- /model/info·/smp-forecast가 스케일러 누수 시절 MAPE 7.17%·근거 없는 coverage 94.5%·신뢰도 92%를 표시했다
- SMP 예측 입력이 2025-12-18에서 멈춘 파일의 마지막 96시간이었다
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import v6_routes
from api.main import app
from src.smp.models.smp_predictor import SMPPredictor

PLANTS_JSON = Path(__file__).resolve().parents[1] / "data/jeju_plants/jeju_power_plants.json"


def _operating_plants():
    data = json.loads(PLANTS_JSON.read_text(encoding="utf-8"))
    plants = data["wind_farms"] + data["solar_farms"] + data["ess_facilities"]
    return [p for p in plants if p["status"] == "운영중" and p.get("subtype") != "distributed"]


def test_plant_registry_matches_canonical_json():
    expected = _operating_plants()
    assert [p["id"] for p in v6_routes.JEJU_PLANTS] == [p["id"] for p in expected]
    assert [p["capacity"] for p in v6_routes.JEJU_PLANTS] == [float(p["capacity_mw"]) for p in expected]
    counts = {t: sum(p["type"] == t for p in v6_routes.JEJU_PLANTS) for t in ("wind", "solar", "ess")}
    assert counts == {"wind": 16, "solar": 4, "ess": 4}
    names = {p["name"] for p in v6_routes.JEJU_PLANTS}
    assert "분산형 태양광 (주택/상업)" not in names  # 1,620개소 합산값은 개별 발전소가 아니다
    assert "한동-평대해상풍력" not in names  # 건설중


@pytest.fixture
def offline_realtime(monkeypatch):
    async def none(*args, **kwargs):
        return None
    monkeypatch.setattr(v6_routes.realtime_client, "get_power_supply_realtime", none)
    monkeypatch.setattr(v6_routes.realtime_client, "get_weather_realtime", none)


def test_kpi_output_and_utilization_equal_resources_sum(offline_realtime):
    with TestClient(app) as client:
        resources = client.get("/api/v1/resources").json()
        kpis = client.get("/api/v1/dashboard/kpis").json()
    capacity = sum(r["capacity"] for r in resources)
    output = sum(r["current_output"] for r in resources)
    assert kpis["resource_count"] == len(resources) == 24
    assert kpis["total_capacity_mw"] == round(capacity, 1)
    assert kpis["current_output_mw"] == pytest.approx(round(output, 1), abs=0.051)
    assert kpis["utilization_pct"] == pytest.approx(round(output / capacity * 100, 1), abs=0.051)


def test_model_info_has_no_invalid_metrics():
    with TestClient(app) as client:
        info = client.get("/api/v1/model/info").json()
    assert info["mape"] is None
    assert info["coverage"] is None
    assert "재검증" in info["status"]


def test_forecast_reports_target_date_without_invalid_metrics():
    with TestClient(app) as client:
        forecast = client.get("/api/v1/smp-forecast").json()
    assert forecast["confidence"] is None
    assert "7.17" not in forecast["model_used"] and "7.165" not in forecast["model_used"]
    assert len(forecast["q50"]) == 24
    datetime.strptime(forecast["forecast_date"], "%Y-%m-%d")


def test_predictor_fallback_is_not_labelled_as_model_forecast(monkeypatch):
    class FallbackPredictor:
        def is_ready(self):
            return True

        def predict_24h(self):
            return {"q10": [90.0] * 24, "q50": [100.0] * 24, "q90": [110.0] * 24, "model_used": "fallback"}

    monkeypatch.setattr(v6_routes, "get_or_create_smp_predictor", lambda: FallbackPredictor())
    with TestClient(app) as client:
        forecast = client.get("/api/v1/smp-forecast").json()
    assert "대체값" in forecast["model_used"] and "v3.2" not in forecast["model_used"]
    assert forecast["forecast_date"] is None


def _write_smp_csv(path: Path, end_day: str, days: int) -> None:
    end = datetime.strptime(end_day, "%Y-%m-%d")
    rows = []
    for d in range(days, -1, -1):
        day = end - timedelta(days=d)
        for h in range(1, 25):
            rows.append({
                "timestamp": f"{day:%Y-%m-%d} {h:02d}:00", "date": f"{day:%Y-%m-%d}", "hour": h,
                "smp_mainland": 100.0 + h, "smp_jeju": 100.0 + h, "smp_max": 110.0, "smp_min": 90.0,
                "source": "EPSIS", "is_synthetic": False,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_predictor_input_ends_at_start_of_today(tmp_path):
    # 최신 파일에는 오늘 24시까지의 하루전 가격이 이미 들어 있다. 오늘 1~24시를 예측하려면
    # 입력은 어제 24시 구간 끝(= 오늘 0시)에서 끊어야 한다.
    today = datetime(2026, 11, 5, 16, 14)
    csv = tmp_path / "smp_real_epsis.csv"
    _write_smp_csv(csv, "2026-11-05", days=6)
    predictor = SMPPredictor.__new__(SMPPredictor)
    predictor.data_path = csv
    predictor._cached_data = None
    predictor._cache_timestamp = None
    predictor._cache_ttl_minutes = 5

    window = predictor._load_recent_data(96, end_at=datetime(2026, 11, 5))

    assert window["datetime"].iloc[-1] == pd.Timestamp("2026-11-05 00:00")
    assert len(window) == 96
    assert SMPPredictor.forecast_start(window) == datetime(2026, 11, 5)
    assert today.date() == SMPPredictor.forecast_start(window).date()
