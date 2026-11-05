"""Independent check of the workflow's solar-forecaster claims (read-only on the repo; written without its scripts).

Claims under test:
1. The served artifact's scaler/initial constant were fit on more than the chronological training split (leak).
2. Served as in api/renewable_ml_predictor.py (current temp/humidity only, humidity->cloud, sine irradiance,
   fixed 446 MW), the forecast is worse than a month-by-hour climatology under KPX's error rule.

KPX rule: error_h = |forecast - actual| / installed capacity * 100, averaged over hours with capacity factor >= 10%.
Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/verify_solar.py [out.json]
"""
import json
import os
import pickle
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJ = str(Path(__file__).resolve().parents[2])
OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("verify_solar_out.json"))
FEATURES = ["hour", "month", "day_of_year", "hour_sin", "hour_cos",
            "temperature", "humidity", "cloud_cover", "radiation", "capacity_mw"]
SERVED_CAPACITY = 446.0
CF_MIN = 0.10          # KPX: only hours with capacity factor >= 10% are scored
REGISTRATION_BAR = 10.0  # KPX registration test: monthly mean error <= 10%


def load():
    d = f"{PROJ}/data/raw"
    name = next(f for f in os.listdir(d)
                if unicodedata.normalize("NFC", f) == "한국동서발전_제주_기상관측_태양광발전.csv")
    df = pd.read_csv(os.path.join(d, name), encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["일시"], errors="raise")
    df = df.set_index("datetime").sort_index()
    assert not df.index.duplicated().any()
    return df


def feature_frame(df):
    X = pd.DataFrame(index=df.index)
    X["hour"], X["month"], X["day_of_year"] = df.index.hour, df.index.month, df.index.dayofyear
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * X["hour"] / 24), np.cos(2 * np.pi * X["hour"] / 24)
    X["temperature"], X["humidity"] = df["기온"], df["습도"]
    X["cloud_cover"], X["radiation"], X["capacity_mw"] = df["전운량(10분위)"], df["일사량"], df["태양광 설비용량(MW)"]
    return X[FEATURES].fillna(0)


def served_features(issue_ts, cur_temp, cur_hum, capacity):
    """Mirror RenewableMLPredictor.predict_24h -> predict_hour -> SolarMLPredictor.predict with no forecasts passed."""
    rows = []
    for h in range(24):
        temp = cur_temp + (4 * np.sin((h - 6) * np.pi / 12) if 6 <= h <= 18 else -2)
        cc = 8.0 if cur_hum >= 80 else (5.0 if cur_hum >= 60 else 2.0)
        rad = 0.0 if (h < 6 or h > 19) else 3.5 * np.sin((h - 6) * np.pi / 13) * (1.0 - cc / 10 * 0.8)
        rows.append({"hour": h, "month": issue_ts.month, "day_of_year": issue_ts.dayofyear,  # datetime.now() at issue
                     "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                     "temperature": temp, "humidity": cur_hum, "cloud_cover": cc, "radiation": rad,
                     "capacity_mw": capacity})
    return pd.DataFrame(rows)[FEATURES]


def served_forecast(df, days, model, scaler, issue_offset, capacity_mode):
    preds = {}
    for day in days:
        issue = day + issue_offset
        if issue not in df.index:
            continue
        cap_served = SERVED_CAPACITY if capacity_mode == "fixed" else float(df.loc[day + pd.Timedelta(hours=12), "태양광 설비용량(MW)"])
        X = served_features(issue, float(df.loc[issue, "기온"]), float(df.loc[issue, "습도"]), cap_served)
        p = np.clip(model.predict(scaler.transform(X)), 0, cap_served)
        p[(np.arange(24) < 6) | (np.arange(24) > 19)] = 0.0
        for h in range(24):
            preds[day + pd.Timedelta(hours=h)] = p[h]
    return pd.Series(preds)


def kpx_score(actual, cap, fc):
    ok = actual.notna() & fc.reindex(actual.index).notna()
    a, c, f = actual[ok], cap[ok], fc.reindex(actual.index)[ok]
    q = (a / c) >= CF_MIN
    err = (f[q] - a[q]).abs() / c[q] * 100
    monthly = err.groupby(err.index.to_period("M")).mean()
    e_all = f - a
    return {"kpx_mean_error_pct": round(float(err.mean()), 2), "scored_hours": int(q.sum()),
            "months_passing_10pct": f"{int((monthly <= REGISTRATION_BAR).sum())}/{len(monthly)}",
            "R2_all_hours": round(float(1 - (e_all ** 2).sum() / ((a - a.mean()) ** 2).sum()), 3)}


def main():
    df = load()
    tgt = df["태양광 발전량(MWh)"]
    data = df[tgt.notna()]
    X_all, y_all = feature_frame(data), data["태양광 발전량(MWh)"].to_numpy(float)
    split_time = data.index[int(len(data) * 0.8)]
    tr = data.index < split_time
    out = {"rows_with_target": int(len(data)), "chronological_split_time": str(split_time),
           "holdout_range": [str(data.index[~tr].min()), str(data.index[~tr].max())]}

    # 1) Provenance of the served artifact
    model = pickle.load(open(f"{PROJ}/models/renewable/solar_lgbm_model.pkl", "rb"))
    scaler = pickle.load(open(f"{PROJ}/models/renewable/solar_scaler.pkl", "rb"))
    _, _, y_rs_tr, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    out["provenance"] = {
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
        "scaler_mean_max_abs_diff_vs_all_rows": float(np.abs(scaler.mean_ - X_all.mean().to_numpy()).max()),
        "scaler_mean_max_abs_diff_vs_chrono_train": float(np.abs(scaler.mean_ - X_all[tr].mean().to_numpy()).max()),
        "init_constant": float(np.ravel(model.init_.constant_)[0]),
        "y_mean_all_rows": float(y_all.mean()),
        "y_mean_chrono_train": float(y_all[tr].mean()),
        "y_mean_random_split_train_rs42": float(y_rs_tr.mean()),
        "max_capacity_in_chrono_train": float(data.loc[tr, "태양광 설비용량(MW)"].max()),
    }

    # 2) Holdout evaluation on full target days
    hold = data[~tr]
    days = pd.DatetimeIndex(sorted({ts.normalize() for ts in hold.index}))
    days = days[[(hold.index.normalize() == d).sum() == 24 for d in days]]
    idx = hold.index[hold.index.normalize().isin(days)]
    actual, cap = tgt.reindex(idx), df["태양광 설비용량(MW)"].reindex(idx)
    res = {}
    for label, off, capmode in (("served_issue_D-1_10h_fixed446", pd.Timedelta(hours=-14), "fixed"),
                                ("served_issue_D-1_10h_true_capacity", pd.Timedelta(hours=-14), "true"),
                                ("served_issue_D_00h_fixed446", pd.Timedelta(0), "fixed")):
        res[label] = kpx_score(actual, cap, served_forecast(df, days, model, scaler, off, capmode))

    # Climatology: month-by-hour capacity factor from the training period only, times the target hour's capacity
    cf = (data["태양광 발전량(MWh)"] / data["태양광 설비용량(MW)"])[tr]
    clim = cf.groupby([cf.index.month, cf.index.hour]).mean()
    fc_clim = pd.Series([clim.get((t.month, t.hour), 0.0) for t in idx], index=idx) * cap
    res["climatology_month_hour"] = kpx_score(actual, cap, fc_clim)

    # Timing-fair persistence for a 10:00 D-1 issue: same hour on D-2, rescaled to the target hour's capacity
    cf_full = tgt / df["태양광 설비용량(MW)"]
    res["persistence_D-2_same_hour"] = kpx_score(actual, cap, cf_full.shift(48).reindex(idx) * cap)

    # Clean retrain: chronological split, scaler on train only, served with the same heuristics
    sc = StandardScaler().fit(X_all[tr])
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    gbr.fit(sc.transform(X_all[tr]), y_all[tr])
    res["clean_retrain_served_D-1_10h_true_capacity"] = kpx_score(
        actual, cap, served_forecast(df, days, gbr, sc, pd.Timedelta(hours=-14), "true"))
    res["clean_retrain_measured_irradiance_same_hour_UNAVAILABLE_AT_ISSUE"] = kpx_score(
        actual, cap, pd.Series(np.clip(gbr.predict(sc.transform(X_all.loc[idx])), 0, None), index=idx))
    out["holdout_days"] = int(len(days))
    out["results"] = res
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
