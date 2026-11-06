"""Out-of-time check before switching the API default model: AR+GBM vs the served LSTM on 2025.

Both serving artifacts are scored the way they are served, on 2025 hours neither model saw:
- ar_gbm: models/demand_ar_gbm (AR and GBM fitted through 2024-06, trees chosen on 2024-H2),
  run through ArGbmPredictor.forecast, the API's path.
- demand_only: models/production (trained through 2022-12, validated on 2023-H1), run through
  ProductionPredictor.predict_batch, the API's path. /predict ran it when model_type was omitted.
  weather_full and 'conditional' need 2025 기온, which the repo does not have.
Baselines on the same target hours: persistence (t-1), ramp7, the hourly AR refitted on every
hour through 2024-12-31 (what was knowable before 2025), and the AR inside the ar_gbm artifact
(shows what the GBM correction adds).

Actuals: data/raw/jeju_hourly_power_2025.csv written by tools/fetch_kpx_jeju_trade.py
(KPX 지역별 시간대별 전력거래량, data.go.kr 15133498), appended to the 2013-2024 raw file.

Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/compare_default_2025.py [out.json]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import diebold_mariano  # noqa: E402
from features.time_features import is_holiday  # noqa: E402
from inference.ar_gbm_predictor import ArGbmPredictor  # noqa: E402
from inference.predict import ProductionPredictor  # noqa: E402
from models.hourly_ar import HourlyLinearAR, ramp7_forecast  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("compare_default_2025_out.json"))
RAW_FILES = (ROOT / "data/raw/jeju_hourly_power_2013_2024.csv", ROOT / "data/raw/jeju_hourly_power_2025.csv")
TEST_START, TEST_END = pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-12-31 23:00")
# 2025년 직전까지 알 수 있던 자료로 적합한 시간대별 AR (배포 게이트의 AR 기준선과 같은 방식)
AR_FIT_END = "2024-12-31 23:00"
# 시차 피처(168h)·LSTM 입력 창(168h)·AR+GBM 최소 이력(337h)을 채우고도 남는 앞 구간
WARMUP = pd.Timedelta(days=30)
# 기본값 교체 조건: 서빙 중인 LSTM보다 MAPE·RMSE가 낮고, 절대오차 차이가 우연이 아니어야 한다
DM_ALPHA = 0.05
MIN_MAE_GAIN = 0.01
GATE_BASELINES = ("persistence", "ramp7", "hourly_ar")
NAMES = ("ar_gbm", "demand_only", "persistence", "ramp7", "hourly_ar", "artifact_ar")


def load_demand() -> pd.Series:
    """원본 raw 파일(시간 h = (h-1):00에 시작하는 구간)을 이어 붙인 시간별 전력거래량."""
    parts = []
    for path in RAW_FILES:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; produce 2025 with tools/fetch_kpx_jeju_trade.py")
        df = pd.read_csv(path, encoding="utf-8-sig")
        ts = pd.to_datetime(df["거래일자"]) + pd.to_timedelta(df["시간"] - 1, unit="h")
        parts.append(pd.Series(df["전력거래량(MWh)"].to_numpy(dtype=float), index=ts))
    demand = pd.concat(parts).sort_index().rename("power_demand")
    demand.index.name = "datetime"
    if demand.index.duplicated().any() or len(demand) != len(pd.date_range(demand.index[0], demand.index[-1], freq="h")):
        raise ValueError("Raw demand must be strictly hourly without gaps or duplicates")
    return demand


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    e = pred - actual
    return {"n": int(len(e)), "MAPE": round(float(np.mean(np.abs(e) / np.abs(actual)) * 100), 4),
            "MAE": round(float(np.mean(np.abs(e))), 3), "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "R2": round(float(1 - (e ** 2).sum() / ((actual - actual.mean()) ** 2).sum()), 4),
            "bias": round(float(e.mean()), 3)}


def predictions(demand: pd.Series, targets: pd.DatetimeIndex) -> dict:
    """targets 각 시각의 1시간 앞 예측. 모두 t-1까지의 관측만 쓴다."""
    history = demand.loc[targets[0] - WARMUP:targets[-1] - pd.Timedelta(hours=1)].reset_index()
    ar_gbm = ArGbmPredictor(ROOT / "models/demand_ar_gbm")
    ar_gbm.load()
    lstm = ProductionPredictor(model_dir=str(ROOT / "models/production"), device=torch.device("cpu"))
    lstm.load_models()
    res = lstm.predict_batch(history, model="demand_only")
    preds = {
        "ar_gbm": ar_gbm.forecast(history),
        # predict_batch는 입력 창의 마지막 시각을 붙인다 → 예측 대상은 그다음 시각
        "demand_only": pd.Series(res.predictions, index=pd.DatetimeIndex(res.timestamps) + pd.Timedelta(hours=1)),
        "persistence": demand.shift(1),
        "ramp7": ramp7_forecast(demand),
        "hourly_ar": HourlyLinearAR().fit(demand, train_end=AR_FIT_END).predict(demand),
        "artifact_ar": ar_gbm.model.ar.predict(demand),
    }
    out = {}
    for name, pred in preds.items():
        aligned = pred.reindex(targets)
        if aligned.isna().any():
            raise ValueError(f"{name} does not cover every test hour")
        out[name] = aligned.to_numpy(dtype=float)
    return out


def main() -> None:
    demand = load_demand()
    targets = pd.date_range(TEST_START, TEST_END, freq="h")
    actual = demand.reindex(targets).to_numpy()
    preds = predictions(demand, targets)

    def dm(x: str, y: str) -> dict:
        return {k: round(v, 6) for k, v in diebold_mariano(preds[x] - actual, preds[y] - actual).items()}

    scores = {name: metrics(actual, preds[name]) for name in NAMES}
    tests = {f"{x}_vs_{y}": dm(x, y) for x in ("ar_gbm", "demand_only")
             for y in NAMES if y != x and not (x == "demand_only" and y == "ar_gbm")}
    holiday = is_holiday(targets).astype(bool)
    months = targets.month
    gain = 1 - scores["ar_gbm"]["MAE"] / scores["demand_only"]["MAE"]
    switch = (scores["ar_gbm"]["MAPE"] < scores["demand_only"]["MAPE"]
              and scores["ar_gbm"]["RMSE"] < scores["demand_only"]["RMSE"]
              and tests["ar_gbm_vs_demand_only"]["p_a_better"] < DM_ALPHA)
    result = {
        "test": [str(targets[0]), str(targets[-1]), len(targets)],
        "data": {"files": [str(p.relative_to(ROOT)) for p in RAW_FILES],
                 "sha256_2025": hashlib.sha256(RAW_FILES[1].read_bytes()).hexdigest(),
                 "source": "data.go.kr 15133498 한국전력거래소_지역별 시간대별 전력거래량_20251231 (제주도)"},
        "metrics": scores,
        "dm": tests,
        "holiday_hours": {"n": int(holiday.sum()),
                          **{name: metrics(actual[holiday], preds[name][holiday]) for name in NAMES}},
        "monthly_mape": {f"{m:02d}": {name: metrics(actual[months == m], preds[name][months == m])["MAPE"]
                                      for name in NAMES} for m in range(1, 13)},
        "ar_gbm_vs_baselines": {b: {"beats_mape_rmse": scores["ar_gbm"]["MAPE"] < scores[b]["MAPE"]
                                    and scores["ar_gbm"]["RMSE"] < scores[b]["RMSE"],
                                    "dm_p": tests[f"ar_gbm_vs_{b}"]["p_a_better"],
                                    "mae_gain": round(1 - scores["ar_gbm"]["MAE"] / scores[b]["MAE"], 4)}
                                for b in GATE_BASELINES},
        "decision": {"switch_default_to_ar_gbm": bool(switch),
                     "mae_gain_vs_demand_only": round(gain, 4),
                     "material": bool(gain >= MIN_MAE_GAIN),
                     "rule": f"ar_gbm MAPE and RMSE below demand_only, DM one-sided p < {DM_ALPHA}"},
    }
    Path(OUT).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"test {targets[0]} .. {targets[-1]} ({len(targets)} h)")
    for name in NAMES:
        s = scores[name]
        print(f"  {name:12s} MAPE {s['MAPE']:6.3f}%  MAE {s['MAE']:7.2f}  RMSE {s['RMSE']:7.2f}  "
              f"R2 {s['R2']:.4f}  bias {s['bias']:+.2f}")
    d = tests["ar_gbm_vs_demand_only"]
    print(f"  DM ar_gbm vs demand_only: stat {d['stat']:+.2f}, p {d['p_a_better']:.2e}; MAE gain {gain:+.2%}")
    print(f"  switch_default_to_ar_gbm = {switch}  ->  {OUT}")


if __name__ == "__main__":
    main()
