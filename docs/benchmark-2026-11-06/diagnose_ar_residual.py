"""Does the hourly AR's residual hold signal that a richer model could learn?

The AR-anchored LSTM (models/demand_retrain_2026-11-06_raw_ar) collapsed to "AR + constant": its last
hidden ReLUs output 0 for every input, so the correction it adds is the output bias alone. This script
separates "training failure" from "empty residual" by fitting gradient boosting to the AR residual and
running AR + correction through the same folds and deployment gate as src/training/retrain_demand_oot.py
(fixed test + walk-forward 2021-2024, AR refit per fold at the fold's train end).

  A (lstm_info): observed features at t-1, calendar at t, the AR forecast for t
  B (plus_resid): A + past AR residuals and yesterday/last-week holiday flags (the LSTM could derive
                  these from its 168-hour window, which holds y and the AR forecast for every hour)

B's predictions are written in the training pipeline's CSV format to models/demand_ar_gbm_2026-11-06/ so
compare_demand_retrain.py scores them next to the LSTM runs.

Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/diagnose_ar_residual.py [out.json]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import compute_all_metrics, diebold_mariano  # noqa: E402
from models.hourly_ar import HourlyLinearAR  # noqa: E402
from training.retrain_demand_oot import (  # noqa: E402
    FEATURE_SETS, FIXED_SPLIT, GATE_BASELINES, TARGET, WF_TEST_YEARS, baseline_predictions, beats,
    fill_short_gaps, mae_gain, material, passes_gate, significant,
)
from training.train_production import TIME_FEATURES, load_data  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("diagnose_ar_residual_out.json"))
DATA = ROOT / "data/processed/jeju_hourly_merged.csv"
PRED_DIR = ROOT / "models/demand_ar_gbm_2026-11-06"
SAVED_VARIANT = "B_plus_resid"
RESID_LAGS = (1, 2, 3, 24, 168)
SEED = 42
# 트리 수는 검증 MAE로 고른다 (sklearn 내장 조기 종료는 무작위 분할이라 시계열에 누수)
STEP, MAX_ITER, PATIENCE = 25, 2000, 8
GBM_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=SEED)
METRIC_KEYS = ("MAPE", "RMSE", "MAE", "R2")


def load_frame(name: str) -> pd.DataFrame:
    """재학습 스크립트와 같은 경로(결측 보간 → load_data → 피처 선택)로 데이터를 만든다."""
    with tempfile.TemporaryDirectory() as tmp:
        filled = Path(tmp) / "filled.csv"
        fill_short_gaps(DATA, filled)
        df_all, _ = load_data(str(filled), include_weather=name == "weather_full")
    return df_all[FEATURE_SETS[name]]


def design(df: pd.DataFrame, train_end: str):
    """(A 피처, B 피처, 수요, AR 예측)을 같은 행으로 맞춰 반환한다. 관측값은 모두 t-1 이전 값만 쓴다."""
    y = df[TARGET]
    ar_pred = HourlyLinearAR().fit(y, train_end=train_end).predict(y)
    resid = y - ar_pred
    a = df.drop(columns=TIME_FEATURES).shift(1).add_suffix("_prev").join(df[TIME_FEATURES])
    a["ar_forecast"] = ar_pred
    b = a.copy()
    for k in RESID_LAGS:
        b[f"resid_lag_{k}"] = resid.shift(k)
    # 어제·지난주 같은 시각이 휴일이면 AR의 24·168시간 시차가 낮게 끌려간다
    b["is_holiday_lag_24"] = df["is_holiday"].shift(24)
    b["is_holiday_lag_168"] = df["is_holiday"].shift(168)
    ok = b.notna().all(axis=1)
    return {"A_lstm_info": a[ok], "B_plus_resid": b[ok]}, y[ok], ar_pred[ok]


def fit_correction(x: pd.DataFrame, r: pd.Series, tr, va, te) -> dict:
    """AR 잔차를 학습하고 검증 MAE가 가장 낮은 트리 수의 test 보정량을 돌려준다 (0그루 = 보정 없음)."""
    gbm = HistGradientBoostingRegressor(max_iter=STEP, early_stopping=False, warm_start=True, **GBM_PARAMS)
    best = {"iter": 0, "val_mae": float(np.mean(np.abs(r[va]))), "test_corr": np.zeros(int(te.sum()))}
    stale = 0
    for it in range(STEP, MAX_ITER + 1, STEP):
        gbm.set_params(max_iter=it)
        gbm.fit(x[tr], r[tr])
        val_mae = float(np.mean(np.abs(r[va] - gbm.predict(x[va]))))
        if val_mae < best["val_mae"]:
            best, stale = {"iter": it, "val_mae": val_mae, "test_corr": gbm.predict(x[te])}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best


def score_fold(df: pd.DataFrame, bounds):
    """한 분할에서 두 변형을 학습하고, 게이트와 같은 기준선·DM 검정으로 채점한다. (결과, 저장할 예측 프레임)"""
    train_end, val_end, test_end = map(pd.Timestamp, bounds)
    xs, y, ar_pred = design(df, bounds[0])
    idx = y.index
    tr, va, te = idx <= train_end, (idx > train_end) & (idx <= val_end), (idx > val_end) & (idx <= test_end)
    actual = y[te].to_numpy()
    base_preds = baseline_predictions(df[TARGET], idx[te], train_end=bounds[0])
    if not np.allclose(base_preds["hourly_ar"], ar_pred[te].to_numpy()):
        raise AssertionError("AR inside the design differs from the gate's hourly_ar baseline")
    out = {"test": [str(idx[te][0]), str(idx[te][-1]), int(te.sum())],
           "baselines": {n: {k: float(v) for k, v in compute_all_metrics(actual, p).items() if k in METRIC_KEYS}
                         for n, p in base_preds.items()}}
    frame = None
    for label, x in xs.items():
        best = fit_correction(x, y - ar_pred, tr, va, te)
        pred = ar_pred[te].to_numpy() + best["test_corr"]
        out[label] = {"best_iter": best["iter"], "val_mae": round(best["val_mae"], 3),
                      "correction_std": round(float(np.std(best["test_corr"])), 3),
                      "metrics": {k: float(v) for k, v in compute_all_metrics(actual, pred).items()
                                  if k in METRIC_KEYS},
                      "dm": {b: diebold_mariano(pred - actual, base_preds[b] - actual) for b in GATE_BASELINES}}
        if label == SAVED_VARIANT:
            # 학습 파이프라인의 예측 CSV와 같은 열 구성
            frame = pd.DataFrame({"datetime": idx[te], "actual": actual, "pred": pred,
                                  **{"persistence" if n == "persistence_t-1" else n: p
                                     for n, p in base_preds.items()}})
    return out, frame


def gate(fixed: dict, wf: dict, label: str) -> dict:
    """retrain_demand_oot.py의 배포 게이트를 그대로 적용한다."""
    fm, fb = fixed[label]["metrics"], fixed["baselines"]
    wf_mean = {k: float(np.mean([f[label]["metrics"][k] for f in wf.values()])) for k in METRIC_KEYS}
    wf_base = {b: {k: float(np.mean([f["baselines"][b][k] for f in wf.values()])) for k in METRIC_KEYS}
               for b in GATE_BASELINES}
    fixed_ok = passes_gate(fm, fb) and significant(fixed[label]["dm"]) and material(fm, fb)
    wf_ok = passes_gate(wf_mean, wf_base) and material(wf_mean, wf_base)
    return {"wf_mean": wf_mean, "wf_mean_baselines": wf_base,
            "detail": {b: {"fixed": beats(fm, fb[b]), "fixed_mae_gain": mae_gain(fm, fb[b]),
                           "fixed_dm_p": fixed[label]["dm"][b]["p_a_better"],
                           "wf_mean": beats(wf_mean, wf_base[b]), "wf_mean_mae_gain": mae_gain(wf_mean, wf_base[b]),
                           "wf_folds_beating": sum(beats(f[label]["metrics"], f["baselines"][b]) for f in wf.values())}
                       for b in GATE_BASELINES},
            "passed": fixed_ok and wf_ok}


def main() -> None:
    result = {"split": {"fixed": FIXED_SPLIT, "wf_years": WF_TEST_YEARS},
              "gbm": {**GBM_PARAMS, "step": STEP, "max_iter": MAX_ITER, "patience_steps": PATIENCE}}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_SETS:
        df = load_frame(name)
        fixed, frame = score_fold(df, FIXED_SPLIT)
        frame.to_csv(PRED_DIR / f"{name}_fixed_test_predictions.csv", index=False)
        wf = {}
        for y in WF_TEST_YEARS:
            # walk-forward 경계는 재학습 스크립트와 같다: 전년 6월 말까지 학습, 하반기 검증, 해당 연도 test
            wf[str(y)], frame = score_fold(df, (f"{y - 1}-06-30 23:00", f"{y - 1}-12-31 23:00",
                                                f"{y}-12-31 23:00"))
            frame.to_csv(PRED_DIR / f"{name}_wf_{y}_predictions.csv", index=False)
        result[name] = {"fixed": fixed, "walk_forward": wf,
                        "gate": {label: gate(fixed, wf, label) for label in ("A_lstm_info", "B_plus_resid")}}
        for label, g in result[name]["gate"].items():
            fm = fixed[label]["metrics"]
            print(f"{name} {label}: fixed MAPE {fm['MAPE']:.3f} MAE {fm['MAE']:.2f} "
                  f"(AR {fixed['baselines']['hourly_ar']['MAE']:.2f}) | wf MAPE {g['wf_mean']['MAPE']:.3f} "
                  f"MAE {g['wf_mean']['MAE']:.2f} (AR {g['wf_mean_baselines']['hourly_ar']['MAE']:.2f}) "
                  f"| passed {g['passed']}", flush=True)
    Path(OUT).write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
