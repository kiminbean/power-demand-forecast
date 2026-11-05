"""
SMP rolling-origin 평가 실행 도구
==================================

`src/evaluation/rolling_origin.py`의 rolling-origin 백테스트를 실제 SMP 데이터에
적용하는 재현 가능한 실행 경로(CLI)를 제공한다.

- 데이터: EPSIS 시간별 SMP(`data/smp/smp_5years_epsis.csv`)의 smp_mainland.
- 모델: 기점별로 학습 슬라이스 "최근 N일"만 사용하는 Ridge 자기회귀(AR) 예측.
  재귀(recursive) 방식으로 horizon을 예측하며, 학습은 항상 기점 이전 관측만 사용한다.
- 기준선: persistence, seasonal naive 24/168 (라이브러리 기본 제공).
- 구간 보정: 학습 창 잔차 표준편차로 80% 예측구간을 만들어 실측 커버리지를 기록.

결과는 JSON으로 저장한다(기점별 지표 + 요약 + 구간 보정 + 실행 설정).

사용 예:
    python tools/run_smp_rolling_origin.py --n-origins 5 --horizon 24

Author: Hermes Agent
Date: 2026-11-05
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import mae, mape, rmse  # noqa: E402
from src.evaluation.rolling_origin import (  # noqa: E402
    BASELINE_PERSISTENCE,
    BASELINE_SEASONAL_NAIVE_168,
    BASELINE_SEASONAL_NAIVE_24,
    default_baselines,
    rolling_origin_splits,
)

# Ridge AR 하이퍼파라미터 (결정론적 평가를 위한 고정값)
RIDGE_ALPHA = 1.0
AR_LAGS = (1, 24, 168, 336)
ROLLING_WINDOWS = (24, 168)
# 80% 정규 근사 z값 (양측)
Z_80 = 1.2816
INTERVAL_TARGET = 80.0
# 기본 학습 창(일): 기점별 재학습에 사용하는 최근 구간
DEFAULT_WINDOW_DAYS = 56
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs"


def fix_hour_24(ts: str) -> pd.Timestamp:
    """EPSIS의 'YYYY-MM-DD 24:00' 표기를 다음날 00:00으로 변환한다(학습과 동일)."""
    ts = str(ts)
    if " 24:00" in ts:
        return pd.Timestamp(ts.replace(" 24:00", " 00:00")) + pd.Timedelta(days=1)
    return pd.Timestamp(ts)


def load_smp_series(data_path: Path = DEFAULT_DATA_PATH) -> pd.Series:
    """
    EPSIS CSV를 시간별 smp_mainland Series로 적재한다.

    결측 시점은 ffill(인과적, 과거값만 사용)로 채워 rolling-origin 분할 전후로
    미래 정보가 유입되지 않게 한다. 선행 결측(채울 과거가 없는 구간)은 제거한다.
    """
    df = pd.read_csv(data_path)
    df["datetime"] = df["timestamp"].apply(fix_hour_24)
    series = (
        df.set_index("datetime")["smp_mainland"]
        .sort_index()
        .asfreq("h")
        .ffill()
        .dropna()
    )
    series.name = "smp_mainland"
    return series


def build_ar_features(history: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    재귀 예측용 AR 피처 행렬을 만든다.

    Args:
        history: 과거 관측값(시간순). 최소 max(AR_LAGS)+1 길이 필요.
        timestamps: 각 행의 예측 대상 시각 (history 길이와 동일).

    Returns:
        (len(history) - max(AR_LAGS), n_features) 배열. 각 행 i는 시각 i를 예측하며,
        [lag1, lag24, lag168, lag336, rolling_mean24, rolling_mean168,
        hour_sin, hour_cos, dow_sin, dow_cos]. 음수 인덱스 역참조(누수)를 피하기 위해
        첫 max(AR_LAGS)행은 생성하지 않는다.
    """
    max_lag = max(AR_LAGS)
    if len(history) <= max_lag:
        raise ValueError(
            f"history needs > {max_lag} observations, got {len(history)}"
        )
    rows = []
    for i in range(max_lag, len(history)):
        past = history[:i]
        ts = timestamps[i]
        feats = [history[i - lag] for lag in AR_LAGS]
        feats += [float(past[-w:].mean()) for w in ROLLING_WINDOWS]
        feats += [
            np.sin(2 * np.pi * ts.hour / 24),
            np.cos(2 * np.pi * ts.hour / 24),
            np.sin(2 * np.pi * ts.dayofweek / 7),
            np.cos(2 * np.pi * ts.dayofweek / 7),
        ]
        rows.append(feats)
    return np.asarray(rows, dtype=float)


def fit_ridge_ar(train_window: np.ndarray, train_times: pd.DatetimeIndex):
    """Ridge AR 모델을 학습창에 fit하고 (모델, 학습 잔차 표준편차)를 돌려준다."""
    from sklearn.linear_model import Ridge

    X = build_ar_features(train_window, train_times)
    max_lag = max(AR_LAGS)
    y = train_window[max_lag:]
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y)
    residual_sigma = float(np.std(y - model.predict(X), ddof=1))
    # 완전 적합(잔차 0) 등으로 nan/0이 나면 구간 계산이 무의미해지므로 하한을 둔다.
    if not np.isfinite(residual_sigma) or residual_sigma <= 0:
        residual_sigma = 1e-6
    return model, residual_sigma


def make_ridge_ar_forecaster(window_days: int = DEFAULT_WINDOW_DAYS):
    """
    (학습 Series, 평가 인덱스) -> 예측 배열 시그니처의 Ridge AR 예측 함수를 만든다.

    학습 Series의 마지막 window_days*24 관측만 사용해 재학습하며, 재귀 방식으로
    평가 인덱스 길이만큼 예측한다. (모델, 잔차 표준편차)도 함께 반환하기 위해
    상태를 outer 스코프에 기록한다.
    """
    state = {"sigma": None}

    def forecast(train: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
        values = train.sort_index().to_numpy(dtype=float)
        times = pd.DatetimeIndex(train.sort_index().index)
        window = int(window_days * 24)
        # 피처 행이 최소 1개 생기려면 창이 max_lag+1 이상이어야 한다.
        window = min(max(window, max(AR_LAGS) + 1), len(values))
        train_window = values[-window:]
        train_times = times[-window:]
        model, sigma = fit_ridge_ar(train_window, train_times)
        state["sigma"] = sigma

        history = list(values)
        # 재귀 예측: 평가 시각을 하나씩 확장하며 피처를 다시 계산한다.
        # 피처 생성에 필요한 것은 최근 max(AR_LAGS)+1 구간뿐이므로 잘라낸다.
        keep = max(AR_LAGS) + 1
        history_times = list(times[-keep:])
        history = list(values[-keep:])
        preds = []
        for ts in test_index:
            ext_history = np.asarray(history, dtype=float)
            ext_times = pd.DatetimeIndex(history_times + [ts])
            x = build_ar_features(ext_history, ext_times)[-1:].reshape(1, -1)
            y_hat = float(model.predict(x)[0])
            preds.append(y_hat)
            history.append(y_hat)
            history_times.append(ts)
        return np.asarray(preds, dtype=float)

    forecast.state = state  # type: ignore[attr-defined]
    return forecast


def interval_coverage(
    per_origin_rows: List[dict], z: float = Z_80
) -> dict:
    """
    (기점, 예측) 기록으로 80% 예측구간 커버리지를 계산한다.

    per_origin_rows의 각 항목: {"origin", "y_true", "y_pred", "sigma"}.
    """
    covered = 0
    total = 0
    widths = []
    per_origin = {}
    for row in per_origin_rows:
        lower = row["y_pred"] - z * row["sigma"]
        upper = row["y_pred"] + z * row["sigma"]
        width = upper - lower
        widths.append(width)
        hit = lower <= row["y_true"] <= upper
        covered += int(hit)
        total += 1
        agg = per_origin.setdefault(
            row["origin"], {"n": 0, "covered": 0, "width_sum": 0.0}
        )
        agg["n"] += 1
        agg["covered"] += int(hit)
        agg["width_sum"] += float(width)
    return {
        "target_pct": INTERVAL_TARGET,
        "z": z,
        "coverage_pct": round(100.0 * covered / total, 2) if total else None,
        "mean_width": round(float(np.mean(widths)), 3) if widths else None,
        "per_origin": {
            str(origin): {
                "coverage_pct": round(100.0 * agg["covered"] / agg["n"], 2),
                "mean_width": round(agg["width_sum"] / agg["n"], 3),
            }
            for origin, agg in per_origin.items()
        },
    }


def run_backtest(
    series: pd.Series,
    n_origins: int,
    horizon: int,
    window_days: int,
) -> dict:
    """rolling-origin 백테스트를 실행하고 결과 dict를 반환한다."""
    splits = rolling_origin_splits(series, n_origins, horizon)
    forecast_fn = make_ridge_ar_forecaster(window_days)

    per_origin_rows: List[dict] = []
    summary_rows: List[dict] = []

    for train_index, test_index in splits:
        train = series.loc[train_index]
        y_true = series.loc[test_index].to_numpy(dtype=float)
        origin = pd.Timestamp(train_index[-1])

        y_pred = forecast_fn(train, test_index)
        sigma = forecast_fn.state["sigma"]  # type: ignore[union-attr]
        for ts, actual, pred in zip(test_index, y_true, y_pred):
            per_origin_rows.append(
                {
                    "origin": str(origin),
                    "timestamp": str(ts),
                    "y_true": float(actual),
                    "y_pred": float(pred),
                    "sigma": float(sigma),
                }
            )
        summary_rows.append(
            {
                "model": "ridge_ar",
                "origin": str(origin),
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
            }
        )

        for name, y_base in default_baselines(train, test_index).items():
            summary_rows.append(
                {
                    "model": name,
                    "origin": str(origin),
                    "MAE": mae(y_true, y_base),
                    "RMSE": rmse(y_true, y_base),
                    "MAPE": mape(y_true, y_base),
                }
            )

    per_origin_df = pd.DataFrame(summary_rows)
    summary = (
        per_origin_df.groupby("model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "config": {
            "n_origins": n_origins,
            "horizon": horizon,
            "ridge_window_days": window_days,
            "ridge_alpha": RIDGE_ALPHA,
            "ar_lags": list(AR_LAGS),
            "min_train": 168,
            "gap_fill": "ffill (인과적)",
            "series_length": int(len(series)),
            "series_range": [str(series.index[0]), str(series.index[-1])],
        },
        "origins": [r["origin"] for r in summary_rows if r["model"] == "ridge_ar"],
        "summary": summary,
        "per_origin": summary_rows,
        "interval_calibration": interval_coverage(per_origin_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SMP rolling-origin 백테스트 실행")
    parser.add_argument("--n-origins", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    series = load_smp_series(args.data)
    result = run_backtest(series, args.n_origins, args.horizon, args.window_days)
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")

    output = args.output or (
        DEFAULT_OUTPUT_DIR
        / f"rolling_origin_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"saved: {output}")
    print(pd.DataFrame(result["summary"]).to_string(index=False))
    calib = result["interval_calibration"]
    print(
        f"interval {calib['target_pct']:.0f}%: "
        f"coverage {calib['coverage_pct']}%, mean width {calib['mean_width']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
