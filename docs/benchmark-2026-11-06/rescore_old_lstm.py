"""Re-score the replaced 2025-12 demand LSTM (reported MAPE 6.06% / 6.47%) through the serving path.

The 2025-12-16 training report (models/_pre_retrain_backup_2026-11-05/demand_lstm/training_report.json)
lists weather_full MAPE 6.06% and demand_only 6.47% on the test split 2023-07-01~2024-12-31
(src/training/train_production.py at commit 995b6161, split_data_by_time defaults). This script runs
that checkpoint and the current LSTM (models/production) through ProductionPredictor.predict_batch,
the API's path, on the same hours with raw actuals, and on 2025 (data/raw/jeju_hourly_power_2025.csv,
tools/fetch_kpx_jeju_trade.py). It also scores the old checkpoint against the old clipped target.

Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/rescore_old_lstm.py [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from inference.predict import ProductionPredictor  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("rescore_old_lstm_out.json"))
OLD_DIR = ROOT / "models/_pre_retrain_backup_2026-11-05/demand_lstm"
NEW_DIR = ROOT / "models/production"
# 시차 피처(168h)와 입력 창(168h)을 채우고도 남는 앞 구간
WARMUP = pd.Timedelta(days=30)
OLD_CAP = 694.65805125  # 예전 전처리가 쓰던 목표 상한 (전 기간 Q3+1.5·IQR)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    e = pred - actual
    return {"n": int(len(e)), "MAPE": round(float(np.mean(np.abs(e) / np.abs(actual)) * 100), 4),
            "R2": round(float(1 - (e ** 2).sum() / ((actual - actual.mean()) ** 2).sum()), 4)}


def served(predictor: ProductionPredictor, model: str, frame: pd.DataFrame, targets: pd.DatetimeIndex) -> np.ndarray:
    """targets 각 시각의 1시간 앞 예측 (predict_batch는 입력 창의 마지막 시각을 붙인다)."""
    window = frame.loc[targets[0] - WARMUP:targets[-1] - pd.Timedelta(hours=1)].reset_index()
    res = predictor.predict_batch(window, model=model)
    pred = pd.Series(res.predictions, index=pd.DatetimeIndex(res.timestamps) + pd.Timedelta(hours=1)).reindex(targets)
    if pred.isna().any():
        raise ValueError(f"{model} does not cover every target hour")
    return pred.to_numpy()


def raw_demand() -> pd.DataFrame:
    parts = []
    for name in ("jeju_hourly_power_2013_2024.csv", "jeju_hourly_power_2025.csv"):
        df = pd.read_csv(ROOT / "data/raw" / name, encoding="utf-8-sig")
        ts = pd.to_datetime(df["거래일자"]) + pd.to_timedelta(df["시간"] - 1, unit="h")
        parts.append(pd.Series(df["전력거래량(MWh)"].to_numpy(dtype=float), index=ts))
    out = pd.concat(parts).sort_index().rename("power_demand").to_frame()
    out.index.name = "datetime"
    return out


def main() -> None:
    merged = pd.read_csv(ROOT / "data/processed/jeju_hourly_merged.csv", parse_dates=["datetime"],
                         usecols=["datetime", "power_demand", "기온"]).set_index("datetime").sort_index()
    raw = raw_demand()
    fixed = merged.index[(merged.index >= "2023-07-01") & (merged.index <= "2024-12-31 23:00")]
    y2025 = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    old = ProductionPredictor(model_dir=str(OLD_DIR), device=torch.device("cpu"))
    old.load_models()
    new = ProductionPredictor(model_dir=str(NEW_DIR), device=torch.device("cpu"))
    new.load_models()

    a_fixed = merged["power_demand"].reindex(fixed).to_numpy()
    clipped = merged.assign(power_demand=merged["power_demand"].clip(upper=OLD_CAP))
    a25 = raw["power_demand"].reindex(y2025).to_numpy()
    result = {
        "reported_2025_12": json.loads((OLD_DIR / "training_report.json").read_text())["results"],
        "fixed_test_2023-07_2024-12": {
            "old_weather_full": metrics(a_fixed, served(old, "weather_full", merged, fixed)),
            "old_demand_only": metrics(a_fixed, served(old, "demand_only", merged, fixed)),
            "old_weather_full_clipped_target": metrics(clipped["power_demand"].reindex(fixed).to_numpy(),
                                                       served(old, "weather_full", clipped, fixed)),
            "current_demand_only": metrics(a_fixed, served(new, "demand_only", merged, fixed)),
            "persistence": metrics(a_fixed, merged["power_demand"].shift(1).reindex(fixed).to_numpy()),
        },
        "year_2025": {
            "old_demand_only": metrics(a25, served(old, "demand_only", raw, y2025)),
            "current_demand_only": metrics(a25, served(new, "demand_only", raw, y2025)),
            "persistence": metrics(a25, raw["power_demand"].shift(1).reindex(y2025).to_numpy()),
        },
        "note": "old weather_full needs 기온, which the repo does not have for 2025",
    }
    result["reported_2025_12"] = {k: {m: round(v["metrics"][m], 4) for m in ("MAPE", "R2")}
                                  for k, v in result["reported_2025_12"].items()}
    Path(OUT).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for period in ("fixed_test_2023-07_2024-12", "year_2025"):
        for name, m in result[period].items():
            print(f"{period:28s} {name:32s} MAPE {m['MAPE']:7.4f}%  R2 {m['R2']:.4f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
