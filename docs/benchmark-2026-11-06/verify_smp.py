"""Independent check of the SMP diagnostic: does 0.5*naive24 + 0.5*7-day same-hour median beat the served naive?

Uses the raw KPX Jeju pilot API rows (96 per day) and rebuilds the hourly day-ahead jsmp series itself.
Every input of both rules (jsmp of D-1..D-7) is published before the D-1 11:00 bid gate.
Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/verify_smp.py [raw.csv] [out.json]
The default raw file is the API dump saved during the 2026-11-06 benchmark (data/raw is gitignored, local only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
RAW = sys.argv[1] if len(sys.argv) > 1 else str(
    _HERE.parents[2] / "data/raw/kpx_jeju_pilot_raw_2024-07-02_2026-11-04.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(_HERE.with_name("verify_smp_out.json"))
MAPE_FLOOR = 10.0  # project rule: score MAPE only where |actual| > 10 KRW/kWh
WINDOWS = {
    "project_walk_forward_2025-07-22_to_2025-12-18": ("2025-07-22 00:00", "2025-12-18 23:00"),
    "calendar_2025": ("2025-01-01 00:00", "2025-12-31 23:00"),
    "full_2024-07-09_to_2026-11-04": ("2024-07-09 00:00", "2026-11-04 23:00"),
}


def metrics(y, p):
    ok = y.notna() & p.notna()
    y, p = y[ok].to_numpy(float), p[ok].to_numpy(float)
    e = p - y
    m = np.abs(y) > MAPE_FLOOR
    return {"n": int(len(y)), "MAE": round(float(np.mean(np.abs(e))), 3),
            "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "MAPE_abs_gt_10": round(float(np.mean(np.abs(e[m]) / np.abs(y[m])) * 100), 3),
            "R2_pooled": round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)}


def main():
    raw = pd.read_csv(RAW, dtype={"date": str})
    per_hour = raw.groupby(["date", "hour"])["jsmp"].agg(["first", "nunique"])
    assert (per_hour["nunique"] == 1).all(), "jsmp should be constant within an hour"
    ts = pd.to_datetime(per_hour.index.get_level_values(0), format="%Y%m%d") + \
        pd.to_timedelta(per_hour.index.get_level_values(1) - 1, unit="h")
    s = pd.Series(per_hour["first"].to_numpy(float), index=ts).sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    naive = s.shift(24)
    med7 = pd.concat([s.shift(24 * k) for k in range(1, 8)], axis=1).median(axis=1)
    blend = 0.5 * naive + 0.5 * med7
    out = {"series": {"start": str(s.index.min()), "end": str(s.index.max()), "missing_hours": int(s.isna().sum())}}
    for name, (a, b) in WINDOWS.items():
        y = s[a:b]
        r_naive, r_blend = metrics(y, naive[a:b]), metrics(y, blend[a:b])
        out[name] = {"naive24": r_naive, "blend": r_blend, "median7": metrics(y, med7[a:b]),
                     "blend_MAE_gain_vs_naive_pct": round((1 - r_blend["MAE"] / r_naive["MAE"]) * 100, 2)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
