"""Print the Markdown tables of docs/2026-11-06-demand-retrain-raw-target.md from the saved outputs.

Unrounded sources first, so the report never rounds twice:
  LSTM runs   models/demand_retrain_2026-11-06_raw_{last,ar}/oot_report.json
  AR + GBM    docs/benchmark-2026-11-06/diagnose_ar_residual_out.json
  bias, DM, peak hours, served model   docs/benchmark-2026-11-06/compare_demand_retrain_out.json

Usage (repo root, after compare_demand_retrain.py):
  .venv/bin/python docs/benchmark-2026-11-06/report_tables.py [compare_out.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
COMPARE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "compare_demand_retrain_out.json"
LSTM = {"last": ROOT / "models/demand_retrain_2026-11-06_raw_last/oot_report.json",
        "ar": ROOT / "models/demand_retrain_2026-11-06_raw_ar/oot_report.json"}
GBM_LABEL = "B_plus_resid"
YEARS = ("2021", "2022", "2023", "2024")
BASE_NAMES = {"hourly_ar": "시간대별 AR", "ramp7": "ramp7", "persistence_t-1": "persistence t-1"}
SERVED = "deployed_served"
SERVED_LABEL = "운영 LSTM (서빙 경로 재현, 원본 입력)"
# 24시차 HAC 분산의 정규 근사는 먼 꼬리에서 의미가 없으므로 이보다 작은 p는 자른다
P_FLOOR = 1e-10


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def pval(p: float) -> str:
    return f"<{P_FLOOR:.0e}" if p < P_FLOOR else f"{p:.1e}"


def fmt(m: dict) -> list:
    return [f"{m['MAPE']:.2f}", f"{m['MAE']:.2f}", f"{m['RMSE']:.2f}", f"{m['R2']:.3f}"]


def main() -> None:
    lstm = {k: load(p)["models"] for k, p in LSTM.items()}
    gbm = load(HERE / "diagnose_ar_residual_out.json")
    cmp_ = load(COMPARE)
    for model in ("demand_only", "weather_full"):
        fx_c = cmp_[model]["lstm_ar_raw"]["fixed"]
        print(f"\n### 고정 test — {model}\n")
        print(row(["모델", "MAPE %", "MAE", "RMSE", "R²", "bias", "DM vs AR", "DM vs 운영"]))
        print(row(["---"] + ["---:"] * 7))
        dep = fx_c[SERVED]
        dm_served_ar = fx_c[f"dm_{SERVED}_vs_ar"]["stat"]
        print(row([SERVED_LABEL] + fmt(dep) + [f"{dep['bias']:+.2f}", f"{dm_served_ar:+.2f}", "–"]))
        for key, run, label in (("last", "lstm_last_raw", "LSTM 직전값 앵커 (원본 재학습)"),
                                ("ar", "lstm_ar_raw", "LSTM AR 앵커 (원본 재학습)")):
            m = lstm[key][model]["fixed"]["metrics"]
            c = cmp_[model][run]["fixed"]
            print(row([label] + fmt(m) + [f"{c['model']['bias']:+.2f}", f"{c['dm_model_vs_ar']['stat']:+.2f}",
                                          f"{c[f'dm_model_vs_{SERVED}']['stat']:+.2f}"]))
        g = gbm[model]["fixed"][GBM_LABEL]
        c = cmp_[model]["ar_gbm_raw"]["fixed"]
        print(row(["**AR + GBM 잔차 보정 (B)**"] + fmt(g["metrics"]) +
                  [f"{c['model']['bias']:+.2f}", f"{c['dm_model_vs_ar']['stat']:+.2f}",
                   f"{c[f'dm_model_vs_{SERVED}']['stat']:+.2f}"]))
        for b, label in BASE_NAMES.items():
            m = lstm["ar"][model]["fixed"]["baselines"][b]
            bias = fx_c["persistence" if b == "persistence_t-1" else b]["bias"]
            # DM 통계량은 두 모델을 바꾸면 부호만 바뀐다 → AR 대 운영은 운영 대 AR의 음수
            vs_served = f"{-dm_served_ar:+.2f}" if b == "hourly_ar" else "–"
            print(row([label] + fmt(m) + [f"{bias:+.2f}", "–", vs_served]))

    print("\n### walk-forward — demand_only (MAPE % / MAE)\n")
    heads = ["연도", "LSTM 직전값", "LSTM AR 앵커", "AR + GBM (B)"] + list(BASE_NAMES.values())
    print(row(heads))
    print(row(["---"] + ["---:"] * (len(heads) - 1)))
    wf_last, wf_ar = lstm["last"]["demand_only"]["walk_forward"], lstm["ar"]["demand_only"]["walk_forward"]
    wf_gbm = gbm["demand_only"]["walk_forward"]
    for y in YEARS:
        cells = [f"{wf_last[y]['metrics']['MAPE']:.2f} / {wf_last[y]['metrics']['MAE']:.2f}",
                 f"{wf_ar[y]['metrics']['MAPE']:.2f} / {wf_ar[y]['metrics']['MAE']:.2f}",
                 f"{wf_gbm[y][GBM_LABEL]['metrics']['MAPE']:.2f} / {wf_gbm[y][GBM_LABEL]['metrics']['MAE']:.2f}"]
        cells += [f"{wf_ar[y]['baselines'][b]['MAPE']:.2f} / {wf_ar[y]['baselines'][b]['MAE']:.2f}" for b in BASE_NAMES]
        print(row([y] + cells))
    means = [lstm["last"]["demand_only"]["wf_mean"], lstm["ar"]["demand_only"]["wf_mean"],
             gbm["demand_only"]["gate"][GBM_LABEL]["wf_mean"]]
    means += [lstm["ar"]["demand_only"]["wf_mean_baselines"][b] for b in BASE_NAMES]
    print(row(["**평균**"] + [f"{m['MAPE']:.2f} / {m['MAE']:.2f}" for m in means]))

    print("\n### 예전 상한(694.66 MWh) 초과 시간 — 고정 test (demand_only)\n")
    cap = {run: cmp_["demand_only"][run]["fixed"]["above_old_cap"]
           for run in ("lstm_last_raw", "lstm_ar_raw", "ar_gbm_raw")}
    print(row(["모델", "시간 수", "MAPE %", "MAE", "bias"]))
    print(row(["---", "---:", "---:", "---:", "---:"]))
    peak_rows = ((SERVED_LABEL, cap["lstm_ar_raw"][SERVED]),
                 ("LSTM 직전값 앵커 (원본 재학습)", cap["lstm_last_raw"]["model"]),
                 ("LSTM AR 앵커 (원본 재학습)", cap["lstm_ar_raw"]["model"]),
                 ("AR + GBM 잔차 보정 (B)", cap["ar_gbm_raw"]["model"]),
                 ("시간대별 AR", cap["ar_gbm_raw"]["hourly_ar"]),
                 ("ramp7", cap["ar_gbm_raw"]["ramp7"]),
                 ("persistence t-1", cap["ar_gbm_raw"]["persistence"]))
    for label, m in peak_rows:
        print(row([label, str(m["n"]), f"{m['MAPE']:.2f}", f"{m['MAE']:.2f}", f"{m['bias']:+.2f}"]))

    print("\n### 게이트 상세 — AR + GBM (B)\n")
    print(row(["모델", "기준선", "고정 MAE 개선 (+ = 개선)", "고정 DM p", "WF 평균 MAE 개선", "WF 연도별 우위"]))
    print(row(["---", "---", "---:", "---:", "---:", "---:"]))
    for model in ("demand_only", "weather_full"):
        for label in ("A_lstm_info", GBM_LABEL):
            g = gbm[model]["gate"][label]
            for b, d in g["detail"].items():
                print(row([f"{model} {label[0]} ({'통과' if g['passed'] else '불합격'})", BASE_NAMES[b],
                           f"{d['fixed_mae_gain'] * 100:.1f}%", pval(d['fixed_dm_p']),
                           f"{d['wf_mean_mae_gain'] * 100:.1f}%", f"{d['wf_folds_beating']}/4"]))

    print("\n### 게이트 상세 — LSTM\n")
    print(row(["모델", "기준선", "고정 MAPE·RMSE 우위", "고정 MAE 개선 (+ = 개선)", "고정 DM p",
               "WF 평균 MAE 개선", "WF 연도별 우위"]))
    print(row(["---", "---", "---", "---:", "---:", "---:", "---:"]))
    for key, label in (("last", "직전값 앵커"), ("ar", "AR 앵커")):
        for model in ("demand_only", "weather_full"):
            e = lstm[key].get(model)
            if not e or "wf_mean" not in e:
                continue
            for b, d in e["gate"].items():
                print(row([f"{label} {model} ({'통과' if e['passed'] else '불합격'})", BASE_NAMES[b],
                           "예" if d["fixed"] else "아니오", f"{d['fixed_mae_gain'] * 100:.1f}%",
                           pval(d['fixed_dm_p']), f"{d['wf_mean_mae_gain'] * 100:.1f}%",
                           f"{e['wf_folds_beating'][b]}/4"]))


if __name__ == "__main__":
    main()
