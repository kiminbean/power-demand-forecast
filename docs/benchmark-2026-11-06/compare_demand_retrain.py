"""Compare the 2026-11-06 raw-target demand retrains with the served model and the classical baselines.

Reads the prediction CSVs written by src/training/retrain_demand_oot.py and diagnose_ar_residual.py:
  models/demand_retrain_2026-11-06_raw_{last,ar}/{model}_{fixed_test,wf_<year>}_predictions.csv
  models/demand_ar_gbm_2026-11-06/{model}_{fixed_test,wf_<year>}_predictions.csv
The served model (models/production) is re-run through the API's own inference path
(ProductionPredictor.predict_batch) on raw demand, so it is scored as it is served. Its stored
2026-11-05 predictions were made from clipped inputs and must not stand in for it.

Every score uses the raw (unclipped) actuals. Also scores the hours above the old clip
(actual > 694.658 MWh).

Usage (repo root): .venv/bin/python docs/benchmark-2026-11-06/compare_demand_retrain.py [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import diebold_mariano  # noqa: E402
from inference.predict import ProductionPredictor  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("compare_demand_retrain_out.json"))
RUNS = {"lstm_last_raw": "models/demand_retrain_2026-11-06_raw_last",
        "lstm_ar_raw": "models/demand_retrain_2026-11-06_raw_ar",
        # AR + gradient boosting 잔차 보정 (diagnose_ar_residual.py의 B 변형, 게이트 결과는 그 출력 JSON)
        "ar_gbm_raw": "models/demand_ar_gbm_2026-11-06"}
# 예측 CSV가 없을 때 무엇을 먼저 돌려야 하는지 알려 준다 (빈 결과를 조용히 쓰지 않는다)
PRODUCERS = {"lstm_last_raw": "src/training/retrain_demand_oot.py --anchor last",
             "lstm_ar_raw": "src/training/retrain_demand_oot.py --anchor ar",
             "ar_gbm_raw": "docs/benchmark-2026-11-06/diagnose_ar_residual.py"}
DATA = ROOT / "data/processed/jeju_hourly_merged.csv"
SERVED_DIR = ROOT / "models/production"
# 시차 피처(168h)와 입력 창(168h)을 채우고도 남는 앞 구간
WARMUP = pd.Timedelta(days=30)
MODELS = ("demand_only", "weather_full")
WF_YEARS = (2021, 2022, 2023, 2024)
OLD_CAP = 694.65805125  # 예전 전처리의 상한 (전 기간 Q3+1.5·IQR)
BASELINES = ("persistence", "ramp7", "hourly_ar")


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    e = pred - actual
    return {"n": int(len(e)), "MAPE": round(float(np.mean(np.abs(e) / np.abs(actual)) * 100), 3),
            "MAE": round(float(np.mean(np.abs(e))), 2), "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 2),
            "R2": round(float(1 - (e ** 2).sum() / ((actual - actual.mean()) ** 2).sum()), 4),
            "bias": round(float(e.mean()), 2)}


def read_predictions(path: Path, run: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; produce it with {PRODUCERS[run]}")
    return pd.read_csv(path, parse_dates=["datetime"])


def served_predictions(predictor: ProductionPredictor, model: str, targets: pd.DatetimeIndex) -> np.ndarray:
    """운영 체크포인트를 API와 같은 추론 경로로 원본 수요에 돌려 targets 시각의 1시간 앞 예측을 낸다."""
    raw = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime").sort_index()
    window = raw.loc[targets[0] - WARMUP:targets[-1] - pd.Timedelta(hours=1)]
    res = predictor.predict_batch(window.reset_index(), model=model)
    # predict_batch는 입력 창의 마지막 시각을 붙인다 → 예측 대상은 그다음 시각
    pred = pd.Series(res.predictions, index=pd.DatetimeIndex(res.timestamps) + pd.Timedelta(hours=1))
    aligned = pred.reindex(targets)
    if aligned.isna().any():
        raise ValueError(f"Served {model} predictions do not cover the test hours")
    return aligned.to_numpy()


def score_frame(df: pd.DataFrame, extra: dict) -> dict:
    """후보(pred)·기준선 열을 모두 채점하고, 후보 대 AR의 DM 검정과 예전 상한 위 시간 성능을 더한다."""
    a = df["actual"].to_numpy()
    cols = {"model": df["pred"].to_numpy(), **{b: df[b].to_numpy() for b in BASELINES}, **extra}
    out = {name: metrics(a, p) for name, p in cols.items()}

    def dm(x: str, y: str) -> dict:
        return {k: round(v, 4) for k, v in diebold_mariano(cols[x] - a, cols[y] - a).items()}

    for name in ("model", *extra):
        out[f"dm_{name}_vs_ar"] = dm(name, "hourly_ar")
    for name in extra:
        out[f"dm_model_vs_{name}"] = dm("model", name)
        # 서빙 모델은 경쟁 비교 보고서 정정에도 쓰므로 나머지 기준선 대비 검정도 남긴다
        out[f"dm_{name}_vs_ramp7"] = dm(name, "ramp7")
        out[f"dm_{name}_vs_persistence"] = dm(name, "persistence")
    peak = a > OLD_CAP
    out["above_old_cap"] = {name: metrics(a[peak], p[peak]) for name, p in cols.items()} if peak.any() else {}
    return out


def main() -> None:
    result = {}
    predictor = ProductionPredictor(model_dir=str(SERVED_DIR), device=torch.device("cpu"))
    predictor.load_models()
    for model in MODELS:
        result[model] = {}
        served = None
        for run, rel in RUNS.items():
            run_dir = ROOT / rel
            fixed = read_predictions(run_dir / f"{model}_fixed_test_predictions.csv", run)
            targets = pd.DatetimeIndex(fixed["datetime"])
            if served is None:
                served = pd.Series(served_predictions(predictor, model, targets), index=targets)
            if not targets.equals(served.index):
                raise ValueError(f"{run} fixed test hours differ from the other runs")
            entry = {"fixed": score_frame(fixed, {"deployed_served": served.to_numpy()}), "walk_forward": {}}
            folds = [read_predictions(run_dir / f"{model}_wf_{y}_predictions.csv", run) for y in WF_YEARS]
            for y, f in zip(WF_YEARS, folds):
                entry["walk_forward"][str(y)] = score_frame(f, {})
            pooled = pd.concat(folds, ignore_index=True)
            keys = ("MAPE", "MAE", "RMSE", "R2")
            entry["wf_mean"] = {name: {k: round(float(np.mean([entry["walk_forward"][str(y)][name][k]
                                                                for y in WF_YEARS])), 4) for k in keys}
                                for name in ("model",) + BASELINES}
            entry["wf_pooled_dm_model_vs_ar"] = {k: round(v, 4) for k, v in diebold_mariano(
                pooled["pred"] - pooled["actual"], pooled["hourly_ar"] - pooled["actual"]).items()}
            # GBM 후보는 oot_report가 없고 게이트 결과가 diagnose_ar_residual_out.json에 있다
            report = run_dir / "oot_report.json"
            if report.exists():
                rep = json.loads(report.read_text())["models"].get(model, {})
                entry["gate"] = {"passed": rep.get("passed"), "detail": rep.get("gate")}
            result[model][run] = entry
    Path(OUT).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
