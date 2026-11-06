"""Independent third check of the workflow's two key demand claims (read-only on the repo).

1. data/processed/jeju_hourly_merged.csv clips power_demand at Q3 + 1.5*IQR of the full series.
2. An hour-of-day linear AR beats the deployed residual LSTM on the same test hours.

Built from data/raw (not the agents' merged CSV) with numpy lstsq, so it shares no code with them.
Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/verify_ar.py [out.json]

Note (2026-11-06): the "LSTM" scores here re-score the stored 2026-11-05 predictions, which were made
from clipped inputs. The model as served (raw inputs) is scored by compare_demand_retrain.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = str(Path(__file__).resolve().parents[2])
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("verify_ar_out.json"))
LAGS = [1, 2, 24, 25, 48, 168, 169]
FIXED_TEST = ("2023-07-01 00:00", "2024-12-31 23:00")
FIXED_TRAIN_END = "2022-12-31 23:00"
WF_YEARS = [2021, 2022, 2023, 2024]
# LSTM walk-forward MAPEs from models/demand_retrain_2026-11-05_residual_v2/demand_only/oot_report.json (capped target)
LSTM_WF_MAPE = {2021: 3.49, 2022: 3.31, 2023: 3.45, 2024: 4.20}
NW_LAG = 24


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = p - y
    return {"n": int(ok.sum()),
            "MAPE": round(float(np.mean(np.abs(e) / np.abs(y)) * 100), 3),
            "MAE": round(float(np.mean(np.abs(e))), 2),
            "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 2),
            "R2": round(float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)}


def load_raw():
    df = pd.read_csv(f"{PROJ}/data/raw/jeju_hourly_power_2013_2024.csv", encoding="utf-8-sig")
    assert set(df["지역"].unique()) == {"제주도"}, df["지역"].unique()
    ts = pd.to_datetime(df["거래일자"]) + pd.to_timedelta(df["시간"] - 1, unit="h")
    s = pd.Series(df["전력거래량(MWh)"].astype(float).values, index=ts).sort_index()
    assert not s.index.duplicated().any()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def load_processed():
    df = pd.read_csv(f"{PROJ}/data/processed/jeju_hourly_merged.csv",
                     usecols=["datetime", "power_demand"], parse_dates=["datetime"], encoding="utf-8-sig")
    s = df.set_index("datetime")["power_demand"].astype(float).sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def design(s):
    X = pd.DataFrame({f"l{k}": s.shift(k) for k in LAGS}, index=s.index)
    for d in range(1, 7):
        X[f"dow{d}"] = (s.index.dayofweek == d).astype(float)
    X.insert(0, "const", 1.0)
    return X


def ar_forecast(s, train_start, train_end, test_start, test_end):
    """Per-hour-of-day OLS; coefficients use only rows inside the training window."""
    X, hours = design(s), s.index.hour
    pred = pd.Series(np.nan, index=s.index)
    tr = (s.index >= train_start) & (s.index <= train_end)
    te = (s.index >= test_start) & (s.index <= test_end)
    for h in range(24):
        m_tr = tr & (hours == h)
        A = X[m_tr]
        ok = A.notna().all(axis=1) & s[m_tr].notna()
        beta, *_ = np.linalg.lstsq(A[ok].to_numpy(), s[m_tr][ok].to_numpy(), rcond=None)
        m_te = te & (hours == h)
        B = X[m_te]
        okb = B.notna().all(axis=1)
        pred[B.index[okb]] = B[okb].to_numpy() @ beta
    return pred[te]


def ramp7(s):
    return s.shift(1) + sum(s.shift(24 * k) - s.shift(24 * k + 1) for k in range(1, 8)) / 7


def diebold_mariano(abs_e1, abs_e2, lag=NW_LAG):
    """DM statistic on d = |e1| - |e2| with a Newey-West long-run variance; negative => model 1 better."""
    d = np.asarray(abs_e1, float) - np.asarray(abs_e2, float)
    d = d[np.isfinite(d)]
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    return {"mean_diff_MAE": round(float(d.mean()), 3), "DM_stat": round(float(d.mean() / np.sqrt(lrv / n)), 2), "n": n}


def main():
    out = {}
    raw, proc = load_raw(), load_processed()
    common = raw.index.intersection(proc.index)
    diff = (proc[common] - raw[common]).dropna()
    clipped = diff[diff.abs() > 1e-6]
    q1, q3 = raw.quantile(0.25), raw.quantile(0.75)
    out["cap_check"] = {
        "rows_differing": int(len(clipped)),
        "all_differences_negative": bool((clipped < 0).all()),
        "processed_max": round(float(proc.max()), 5),
        "raw_max": round(float(raw.max()), 3),
        "raw_full_series_Q3_plus_1p5IQR": round(float(q3 + 1.5 * (q3 - q1)), 5),
        "clipped_by_year": {int(k): int(v) for k, v in clipped.groupby(clipped.index.year).size().items()},
        "raw_missing_hours": int(raw.isna().sum()),
    }

    lstm = pd.read_csv(f"{PROJ}/models/demand_retrain_2026-11-05_residual_v2/demand_only/"
                       "demand_only_fixed_test_predictions.csv", parse_dates=["datetime"]).set_index("datetime")
    idx = lstm.index
    raw_t = raw.reindex(idx)
    out["lstm_csv_actual_equals_processed"] = bool(np.allclose(lstm["actual"], proc.reindex(idx), atol=1e-3))

    fixed = {}
    for name, series in (("raw", raw), ("capped", proc)):
        ar = ar_forecast(series, "2019-01-01", FIXED_TRAIN_END, *FIXED_TEST).reindex(idx)
        ar_long = ar_forecast(series, "2013-01-01", FIXED_TRAIN_END, *FIXED_TEST).reindex(idx)
        actual = raw_t if name == "raw" else lstm["actual"]
        fixed[name] = {
            "LSTM": metrics(actual, lstm["pred"]),
            "AR_train2019_2022": metrics(actual, ar),
            "AR_train2013_2022": metrics(actual, ar_long),
            "ramp7": metrics(actual, ramp7(series).reindex(idx)),
            "persistence": metrics(actual, series.shift(1).reindex(idx)),
            "DM_LSTM_vs_AR": diebold_mariano((lstm["pred"] - actual).abs(), (ar - actual).abs()),
            "DM_LSTM_vs_ramp7": diebold_mariano((lstm["pred"] - actual).abs(),
                                                (ramp7(series).reindex(idx) - actual).abs()),
        }
    out["fixed_test"] = fixed

    wf = {}
    for y in WF_YEARS:
        tr_end = f"{y - 1}-06-30 23:00"
        te = (f"{y}-01-01 00:00", f"{y}-12-31 23:00")
        row = {"LSTM_reported_capped_MAPE": LSTM_WF_MAPE[y]}
        for name, series in (("capped", proc), ("raw", raw)):
            ar = ar_forecast(series, "2013-01-01", tr_end, *te)
            row[f"AR_{name}"] = metrics(series[ar.index], ar)
            r7 = ramp7(series)[ar.index]
            row[f"ramp7_{name}"] = metrics(series[ar.index], r7)
            row[f"persistence_{name}"] = metrics(series[ar.index], series.shift(1)[ar.index])
        wf[y] = row
    out["walk_forward"] = wf
    out["walk_forward_mean_MAPE_capped"] = {
        "LSTM_reported": round(float(np.mean(list(LSTM_WF_MAPE.values()))), 3),
        "AR": round(float(np.mean([wf[y]["AR_capped"]["MAPE"] for y in WF_YEARS])), 3),
        "ramp7": round(float(np.mean([wf[y]["ramp7_capped"]["MAPE"] for y in WF_YEARS])), 3),
        "persistence": round(float(np.mean([wf[y]["persistence_capped"]["MAPE"] for y in WF_YEARS])), 3),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
