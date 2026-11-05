"""SMP v4.1 서빙 경로(데이터 로더·정규화·추론기·서빙 묶음·API) 회귀 테스트."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from api import v6_routes
from api.main import app
from src.smp.models import smp_v41_features as F
from src.smp.models.serving import BASELINE_LABEL, V32_LABEL, V41_LABEL, ServingSMPPredictor
from src.smp.models.smp_v41_predictor import (
    InsufficientHistoryError, JejuDAHistoryProvider, SeasonalNaivePredictor, SMPV41Predictor,
)
from src.smp.models.train_smp_v41_serving import window_starts
from src.smp.models.train_smp_v4_smp_only import SMPModelV41


def _frame(start: str, hours: int, skip_days=()):
    rows = []
    t = pd.Timestamp(start)
    for i in range(hours):
        end = t + pd.Timedelta(hours=i + 1)
        trade_day = (end - pd.Timedelta(hours=1)).normalize()
        if trade_day.strftime("%Y-%m-%d") in skip_days:
            continue
        hour = (end - trade_day) // pd.Timedelta(hours=1)
        price = 100 + 20 * np.sin(i / 24 * 2 * np.pi)
        rows.append({"timestamp": f"{trade_day:%Y-%m-%d} {hour:02d}:00", "date": f"{trade_day:%Y-%m-%d}", "hour": hour,
                     "smp_mainland": price, "smp_jeju": price, "smp_max": price * 1.02, "smp_min": price * 0.98,
                     "source": "test", "is_synthetic": False})
    return pd.DataFrame(rows)


def test_hour_24_is_end_of_day_and_sorted_last():
    df = F.prepare_smp_frame(_frame("2026-01-01", 48))
    day1 = df[df["date"] == "2026-01-01"]
    assert day1["datetime"].iloc[-1] == pd.Timestamp("2026-01-02 00:00")
    assert df["datetime"].is_monotonic_increasing


def test_windows_never_span_a_gap_including_feature_warmup():
    df = F.prepare_smp_frame(_frame("2026-01-01", 24 * 40, skip_days=("2026-01-20",)))
    starts = window_starts(df["datetime"], 0, len(df))
    need = F.FEATURE_WARMUP_HOURS + F.INPUT_HOURS + F.OUTPUT_HOURS
    for s in starts[:: max(1, len(starts) // 50)]:
        span = df["datetime"].iloc[s - F.FEATURE_WARMUP_HOURS:s + F.INPUT_HOURS + F.OUTPUT_HOURS]
        assert len(span) == need and (span.diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    gap_end = pd.Timestamp("2026-01-21 01:00")
    assert all(df["datetime"].iloc[s - F.FEATURE_WARMUP_HOURS] >= gap_end or
               df["datetime"].iloc[s + F.INPUT_HOURS + F.OUTPUT_HOURS - 1] < pd.Timestamp("2026-01-20 01:00")
               for s in starts)


def test_normalization_matches_standard_scaler_after_clipping():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5)) * [1, 10, 100, 0, 3]
    y = rng.normal(size=500) * 20 + 120
    stats = F.fit_normalization(x, y)
    clipped = np.clip(x, stats["clip_min"], stats["clip_max"])
    np.testing.assert_allclose(F.apply_normalization(x, stats), StandardScaler().fit_transform(clipped), atol=1e-9)
    assert stats["target_mean"] == pytest.approx(y.mean())


def _artifact(tmp_path, gate=True):
    torch.manual_seed(0)
    kwargs = {"input_size": 40, "hidden_size": 8, "num_layers": 1, "n_heads": 2, "dropout": 0.0, "output_hours": 24}
    model = SMPModelV41(**kwargs)
    stats = {"clip_min": [-1e9] * 40, "clip_max": [1e9] * 40, "mean": [0.0] * 40, "scale": [100.0] * 40,
             "target_mean": 120.0, "target_std": 20.0}
    path = tmp_path / "smp_v41_serving.pt"
    torch.save({"version": "4.1-serving", "model_state_dict": model.state_dict(), "model_kwargs": kwargs,
                "feature_names": [f"f{i}" for i in range(40)], "normalization": stats, "input_hours": 48,
                "train_range": ["2025-01-01", "2026-01-01"], "data_path": "x", "eval_end": "2025-12-19",
                "walk_forward": {"v41": {"mape": 9.5, "coverage_80": 76.0}}, "splits": [], "gate_passed": gate,
                "trained_at": "2026-11-05T18:00:00", "seed": 42}, path)
    return path


def _history_csv(tmp_path, end: pd.Timestamp, hours: int, skip_days=()):
    path = tmp_path / "history.csv"
    _frame(str(end - pd.Timedelta(hours=hours)), hours, skip_days).to_csv(path, index=False)
    return path


def _provider(history_path, tmp_path):
    return JejuDAHistoryProvider(history_path, tmp_path / "missing.csv", api_topup=False)


def test_v41_predicts_today_trade_hours_with_ordered_quantiles(tmp_path):
    now = datetime(2026, 11, 5, 16, 14)
    history = _history_csv(tmp_path, pd.Timestamp("2026-11-05"), 24 * 30)
    p = SMPV41Predictor(_artifact(tmp_path), _provider(history, tmp_path))
    result = p.predict_24h(now=now)
    assert result["forecast_date"] == "2026-11-05"
    assert result["times"][0] == datetime(2026, 11, 5, 0, 0) and len(result["times"]) == 24
    assert np.all(np.isfinite(result["q50"])) and len(result["q50"]) == 24
    assert np.all(result["q10"] <= result["q50"]) and np.all(result["q50"] <= result["q90"])
    assert (result["mape"], result["coverage"]) == (9.5, 76.0)


def test_v41_refuses_when_recent_history_has_a_gap(tmp_path):
    now = datetime(2026, 11, 5, 16, 14)
    history = _history_csv(tmp_path, pd.Timestamp("2026-11-05"), 24 * 30, skip_days=("2026-10-28",))
    p = SMPV41Predictor(_artifact(tmp_path), _provider(history, tmp_path))
    with pytest.raises(InsufficientHistoryError):
        p.predict_24h(now=now)


def test_v41_not_ready_when_gate_failed(tmp_path):
    p = SMPV41Predictor(_artifact(tmp_path, gate=False), _provider(tmp_path / "h.csv", tmp_path))
    assert p.model is not None and not p.is_ready()


def test_baseline_forecasts_yesterday_same_trade_hours_and_backtests_mape(tmp_path):
    now = datetime(2026, 11, 5, 16, 14)
    path = _history_csv(tmp_path, pd.Timestamp("2026-11-05"), 24 * 10)
    df = F.prepare_smp_frame(pd.read_csv(path))
    yesterday = df[df["date"] == "2026-11-04"].sort_values("hour")["smp_jeju"].to_numpy()

    p = SeasonalNaivePredictor(_provider(path, tmp_path))
    result = p.predict_24h(now=now)

    np.testing.assert_allclose(result["q50"], yesterday)
    assert result["forecast_date"] == "2026-11-05" and result["coverage"] is None
    assert np.all(result["q10"] <= result["q50"]) and np.all(result["q50"] <= result["q90"])
    # 사인 패턴이 정확히 하루 주기라 '전날 값'은 오차가 0이다 (재현 MAPE 0, 9일치 평가).
    assert result["mape"] == pytest.approx(0.0, abs=1e-9) and p.backtest["days"] == 9


def test_baseline_mape_matches_manual_calculation(tmp_path):
    path = _history_csv(tmp_path, pd.Timestamp("2026-11-05"), 24 * 3)
    raw = pd.read_csv(path)
    raw.loc[raw["date"] == "2026-11-04", ["smp_jeju", "smp_mainland"]] *= 1.1  # 11-04만 10% 높게
    raw.to_csv(path, index=False)
    p = SeasonalNaivePredictor(_provider(path, tmp_path))
    p.predict_24h(now=datetime(2026, 11, 5, 9))
    df = F.prepare_smp_frame(raw).set_index("datetime")["smp_jeju"]
    d2, d3, d4 = (df.iloc[i:i + 24].to_numpy() for i in (0, 24, 48))  # 11-02, 11-03, 11-04 거래일
    errors = np.concatenate([d3 - d2, d4 - d3])
    actuals = np.concatenate([d3, d4])
    assert p.backtest["days"] == 2
    assert p.mape == pytest.approx(np.mean(np.abs(errors) / np.abs(actuals)) * 100)  # 0%와 9.09%의 평균


def test_serving_prefers_baseline_over_v32_when_v41_gate_failed():
    serving = ServingSMPPredictor(_Fake(ready=False),
                                  _Fake({**FORECAST, "model_used": "baseline-24h", "mape": 11.75, "coverage": None},
                                        mape=11.75, coverage=None),
                                  _Fake({**FORECAST, "model_used": "v3.2"}))
    serving.baseline.backtest = {"days": 150}
    result = serving.predict_24h()
    assert (result["model_version"], result["model_label"], result["mape"]) == ("baseline-24h", BASELINE_LABEL, 11.75)
    info = serving.model_info()
    assert (info["version"], info["mape"], info["coverage"]) == ("baseline-24h", 11.75, None)


class _Fake:
    def __init__(self, result=None, error=None, ready=True, mape=9.5, coverage=76.0):
        self.result, self.error, self.ready = result, error, ready
        self.mape, self.coverage = mape, coverage
        self.meta = {"trained_at": "t", "train_range": ["a", "b"], "eval_end": "2025-12-19"}

    def is_ready(self):
        return self.ready

    def predict_24h(self):
        if self.error:
            raise self.error
        return dict(self.result)


FORECAST = {"q10": [90.0] * 24, "q50": [100.0] * 24, "q90": [110.0] * 24, "forecast_date": "2026-11-05",
            "times": [datetime(2026, 11, 5) + timedelta(hours=i) for i in range(24)]}


def test_serving_falls_back_to_v32_without_metrics_and_reports_it():
    serving = ServingSMPPredictor(_Fake(error=InsufficientHistoryError("gap")),
                                  _Fake(error=InsufficientHistoryError("no yesterday")),
                                  _Fake({**FORECAST, "model_used": "v3.2", "mape": 7.17, "coverage": 89.4}))
    result = serving.predict_24h()
    assert (result["model_version"], result["model_label"]) == ("v3.2", V32_LABEL)
    assert result["mape"] is None and result["coverage"] is None and "gap" in result["fallback_reason"]
    assert serving.model_info()["version"] == "v3.2"


def test_serving_uses_v41_and_reports_its_walk_forward_metrics():
    serving = ServingSMPPredictor(_Fake({**FORECAST, "model_used": "v4.1", "mape": 9.5, "coverage": 76.0}), None, None)
    assert serving.predict_24h()["model_label"] == V41_LABEL
    info = serving.model_info()
    assert (info["version"], info["mape"], info["coverage"]) == ("v4.1", 9.5, 76.0)


def test_api_reports_the_baseline_when_it_serves(monkeypatch):
    baseline = _Fake({**FORECAST, "model_used": "baseline-24h", "mape": 11.75, "coverage": None},
                     mape=11.75, coverage=None)
    baseline.backtest = {"days": 150}
    serving = ServingSMPPredictor(_Fake(ready=False), baseline, None)
    monkeypatch.setattr(v6_routes, "get_or_create_smp_predictor", lambda: serving)
    with TestClient(app) as client:
        forecast = client.get("/api/v1/smp-forecast").json()
        info = client.get("/api/v1/model/info").json()
    assert forecast["model_used"] == f"{BASELINE_LABEL} · 검증 MAPE 11.75%"
    assert (info["version"], info["mape"], info["coverage"], info["status"]) == ("baseline-24h", 11.75, None, "active")
    assert "150일" in info["data_source"]


def test_model_info_rounds_metrics_for_display(monkeypatch):
    baseline = _Fake({**FORECAST, "model_used": "baseline-24h", "mape": 12.476786533138402, "coverage": None},
                     mape=12.476786533138402, coverage=None)
    baseline.backtest = {"days": 150}
    monkeypatch.setattr(v6_routes, "get_or_create_smp_predictor",
                        lambda: ServingSMPPredictor(_Fake(ready=False), baseline, None))
    with TestClient(app) as client:
        info = client.get("/api/v1/model/info").json()
    assert info["mape"] == 12.48


def test_api_reports_the_model_that_actually_served(monkeypatch):
    serving = ServingSMPPredictor(_Fake({**FORECAST, "model_used": "v4.1", "mape": 9.5, "coverage": 76.0}), None, None)
    monkeypatch.setattr(v6_routes, "get_or_create_smp_predictor", lambda: serving)
    with TestClient(app) as client:
        forecast = client.get("/api/v1/smp-forecast").json()
        info = client.get("/api/v1/model/info").json()
    assert forecast["model_used"] == f"{V41_LABEL} · 검증 MAPE 9.50%"
    assert forecast["forecast_date"] == "2026-11-05" and forecast["confidence"] is None
    assert (info["version"], info["mape"], info["coverage"], info["status"]) == ("v4.1", 9.5, 76.0, "active")
