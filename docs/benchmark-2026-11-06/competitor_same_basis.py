"""Score ar_gbm on the periods the competitor table (benchmark report section 1.1) lines up against.

Reads the AR+GBM (B) predictions that diagnose_ar_residual.py saves to models/demand_ar_gbm_2026-11-06/
(local only, about 10 MB): the fixed split trained through 2022-12, and the four walk-forward years,
each retrained through the previous June. The 2025 row is copied from compare_default_2025_out.json
(the served artifact, trained through 2024-06).

- 2024H2: the half-year the KPX day-ahead (jlfd) and real-time (jlfdRt) forecasts were scored on
  (the exact start hour of that scoring was not kept; starting on 07-02 moves MAPE by 0.005 points).
- 2022 and 2023: the test years compared with the Park thesis and with Hwang & Hwang.
- ratio_to_ar: MAPE divided by the hourly AR's MAPE on the same hours (report section 3.4).

Usage (repo root, after diagnose_ar_residual.py):
  .venv/bin/python docs/benchmark-2026-11-06/competitor_same_basis.py [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PRED_DIR = ROOT / "models/demand_ar_gbm_2026-11-06"
SERVED_2025 = HERE / "compare_default_2025_out.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "competitor_same_basis_out.json"
FEATURES = "demand_only"
YEARS = (2021, 2022, 2023, 2024)
# KPX jlfd·jlfdRt를 채점한 2024년 하반기 (경쟁 비교 보고서 1.1절·3.3절)
H2_START, H2_END = "2024-07-01", "2025-01-01"
# 시작 시각 민감도용: KPX 시범사업 원자료의 첫날
H2_ALT_START = "2024-07-02"
# 시간 수가 다르면 저장된 예측이 다른 실행의 것이므로 멈춘다
EXPECTED_HOURS = {"fixed": 13200, "2024H2": 4416, "2024H2_from_07-02": 4392,
                  "2021": 8760, "2022": 8760, "2023": 8760, "2024": 8784}


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    return {"MAPE": round(float(np.mean(np.abs(err) / actual) * 100), 4),
            "MAE": round(float(np.mean(np.abs(err))), 3),
            "R2": round(float(1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2)), 4)}


def score(frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy(float)
    out = {"n": int(len(frame)),
           "ar_gbm": metrics(actual, frame["pred"].to_numpy(float)),
           "hourly_ar": metrics(actual, frame["hourly_ar"].to_numpy(float))}
    out["ratio_to_ar"] = round(out["ar_gbm"]["MAPE"] / out["hourly_ar"]["MAPE"], 4)
    return out


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}: run diagnose_ar_residual.py first")
    frame = pd.read_csv(path, parse_dates=["datetime"])
    # 실측이 0 이하이면 MAPE가 정의되지 않는다
    if frame[["actual", "pred", "hourly_ar"]].isna().any().any() or (frame["actual"] <= 0).any():
        raise SystemExit(f"{path.name}: missing values or non-positive actuals")
    return frame


fixed = load(PRED_DIR / f"{FEATURES}_fixed_test_predictions.csv")
periods = {"fixed": fixed,
           "2024H2": fixed[(fixed["datetime"] >= H2_START) & (fixed["datetime"] < H2_END)],
           "2024H2_from_07-02": fixed[(fixed["datetime"] >= H2_ALT_START) & (fixed["datetime"] < H2_END)]}
for year in YEARS:
    periods[str(year)] = load(PRED_DIR / f"{FEATURES}_wf_{year}_predictions.csv")

result = {}
for key, frame in periods.items():
    if len(frame) != EXPECTED_HOURS[key]:
        raise SystemExit(f"{key}: {len(frame)} hours, expected {EXPECTED_HOURS[key]}")
    result[key] = score(frame)

served = json.loads(SERVED_2025.read_text())["metrics"]
result["2025_served"] = {"n": served["ar_gbm"]["n"],
                         "ar_gbm": {k: served["ar_gbm"][k] for k in ("MAPE", "MAE", "R2")},
                         "hourly_ar": {k: served["hourly_ar"][k] for k in ("MAPE", "MAE", "R2")}}
result["2025_served"]["ratio_to_ar"] = round(served["ar_gbm"]["MAPE"] / served["hourly_ar"]["MAPE"], 4)

OUT.write_text(json.dumps({"features": FEATURES, "periods": result}, ensure_ascii=False, indent=1) + "\n")
for key, row in result.items():
    g = row["ar_gbm"]
    print(f"{key:12s} n={row['n']:5d}  MAPE {g['MAPE']:.4f}  MAE {g['MAE']:.3f}  R2 {g['R2']:.4f}"
          f"  AR {row['hourly_ar']['MAPE']:.4f}  ratio {row['ratio_to_ar']:.4f}")
print(f"wrote {OUT}")
