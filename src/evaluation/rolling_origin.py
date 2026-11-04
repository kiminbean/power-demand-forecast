"""
Rolling-origin 백테스트 평가
=============================

시계열 예측 모델을 여러 예측 기점(origin)에서 시간순으로만 평가한다.

핵심 보장:
- 각 origin의 학습 데이터는 origin 이전 관측만 포함한다 (미래 누수 없음).
- 기준 모델(persistence, seasonal naive)도 동일한 제약을 받는다.
- 예측 함수는 학습 Series만 받으므로 미래 값을 참조할 수 없다.

Author: Hermes Agent
Date: 2026-11
"""

from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, mape, rmse

# 계절성 주기 (시간별 데이터)
SEASONALITY_HOURLY = 24
SEASONALITY_WEEKLY = 168

# 기준 모델 이름 상수
BASELINE_PERSISTENCE = "persistence"
BASELINE_SEASONAL_NAIVE_24 = "seasonal_naive_24"
BASELINE_SEASONAL_NAIVE_168 = "seasonal_naive_168"

# 세부 구간(segment) 정의: 피크 시간대 (09:00 이상 21:00 미만)
PEAK_HOURS_START = 9
OFF_PEAK_HOURS_END = 21


def rolling_origin_splits(
    series: pd.Series,
    n_origins: int,
    horizon: int,
    min_train: int = SEASONALITY_WEEKLY,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    시간순 rolling-origin 분할을 생성한다.

    Args:
        series: 시간별 인덱스를 가진 대상 Series (정렬 필수)
        n_origins: 평가할 기점 수
        horizon: 각 기점에서 예측할 시간 수
        min_train: 첫 기점까지 확보할 최소 학습 관측 수

    Returns:
        (train_index, test_index) 목록. 각 기점은 동일한 간격으로 배치되며
        학습 인덱스는 기점 이하, 평가 인덱스는 기점 초과 미래 구간이다.

    Raises:
        ValueError: 데이터가 분할 생성에 부족한 경우
    """
    if n_origins < 1:
        raise ValueError(f"n_origins must be >= 1, got {n_origins}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    series = series.sort_index()
    index = pd.DatetimeIndex(series.index)
    n = len(index)
    # 마지막 기점은 데이터 끝에서 horizon 만큼 앞이어야 평가 구간이 존재한다.
    required = min_train + (n_origins - 1) * horizon + horizon
    if n < required:
        raise ValueError(
            f"Need at least {required} observations for {n_origins} origins "
            f"with horizon {horizon} and min_train {min_train}, got {n}"
        )

    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    last_origin_pos = n - horizon - 1  # 0-based 위치
    for k in range(n_origins):
        # 기점을 뒤에서부터 동일 간격(horizon)으로 배치한다.
        origin_pos = last_origin_pos - (n_origins - 1 - k) * horizon
        train_index = pd.DatetimeIndex(index[: origin_pos + 1])
        test_index = pd.DatetimeIndex(index[origin_pos + 1 : origin_pos + 1 + horizon])
        splits.append((train_index, test_index))
    return splits


def persistence_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """마지막 관측값을 horizon 동안 유지하는 기준 예측."""
    if len(train) == 0:
        raise ValueError("persistence forecast requires non-empty train series")
    return np.full(horizon, float(train.iloc[-1]))


def seasonal_naive_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    seasonality: int,
) -> np.ndarray:
    """직전 계절 동시간 값으로 예측하는 기준 예측 (t - seasonality)."""
    if len(train) < seasonality:
        raise ValueError(
            f"seasonal naive requires >= {seasonality} observations, got {len(train)}"
        )
    values = train.sort_index()
    predictions = []
    for ts in test_index:
        lagged = ts - pd.Timedelta(hours=seasonality)
        # 계절 lag: 관측이 없으면 계절 주기 배수만큼 더 과거로 이동한다.
        steps = 1
        while lagged not in values.index and steps * seasonality <= len(values):
            lagged = ts - pd.Timedelta(hours=seasonality * (steps + 1))
            steps += 1
        if lagged not in values.index:
            raise ValueError(f"No seasonal observation available for {ts}")
        predictions.append(float(values.loc[lagged]))
    return np.asarray(predictions)


def default_baselines(
    train: pd.Series, test_index: pd.DatetimeIndex
) -> Dict[str, np.ndarray]:
    """표준 기준 모델 예측 묶음을 반환한다."""
    horizon = len(test_index)
    return {
        BASELINE_PERSISTENCE: persistence_forecast(train, horizon),
        BASELINE_SEASONAL_NAIVE_24: seasonal_naive_forecast(
            train, test_index, SEASONALITY_HOURLY
        ),
        BASELINE_SEASONAL_NAIVE_168: seasonal_naive_forecast(
            train, test_index, SEASONALITY_WEEKLY
        ),
    }


def run_rolling_origin_backtest(
    series: pd.Series,
    forecast_fn: Callable[[pd.Series, pd.DatetimeIndex], np.ndarray],
    model_name: str = "model",
    n_origins: int = 5,
    horizon: int = 24,
    min_train: int = SEASONALITY_WEEKLY,
    include_baselines: bool = True,
    extra_baselines: Optional[
        Dict[str, Callable[[pd.Series, pd.DatetimeIndex], np.ndarray]]
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    rolling-origin 백테스트를 실행하고 모델·기준별 지표를 반환한다.

    Args:
        series: 시간별 인덱스 대상 Series
        forecast_fn: (학습 Series, 평가 인덱스) -> 예측 배열. 학습 데이터는
            기점 이전 관측만 포함되므로 미래 정보를 받을 수 없다.
        model_name: 평가 모델 이름
        n_origins: 기점 수
        horizon: 기점별 예측 시간 수
        min_train: 첫 기점까지의 최소 학습 관측 수
        include_baselines: 표준 기준 모델 포함 여부
        extra_baselines: 추가 기준 모델 {이름: 예측 함수}

    Returns:
        DataFrame: (모델, 기점)별 MAE/RMSE/MAPE와 전체 평균 요약.
    """
    series = series.sort_index()
    splits = rolling_origin_splits(series, n_origins, horizon, min_train)

    rows = []

    def _record(name: str, origin: pd.Timestamp, y_true: np.ndarray, y_pred: np.ndarray):
        rows.append(
            {
                "model": name,
                "origin": origin,
                "n_test": len(y_true),
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
            }
        )

    for train_index, test_index in splits:
        train = series.loc[train_index]
        y_true = series.loc[test_index].to_numpy(dtype=float)
        origin = cast(pd.Timestamp, train_index[-1])

        _record(model_name, origin, y_true, np.asarray(forecast_fn(train, test_index), dtype=float))

        baselines: Dict[str, np.ndarray] = {}
        if include_baselines:
            baselines.update(default_baselines(train, test_index))
        if extra_baselines:
            for name, fn in extra_baselines.items():
                baselines[name] = np.asarray(fn(train, test_index), dtype=float)
        for name, y_pred in baselines.items():
            _record(name, origin, y_true, y_pred)

    per_origin = pd.DataFrame(rows)
    summary = (
        per_origin.groupby("model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .reset_index()
    )
    summary["n_origins"] = n_origins
    summary["horizon"] = horizon
    return per_origin, summary


def rolling_origin_error_frame(
    series: pd.Series,
    forecast_fn: Callable[[pd.Series, pd.DatetimeIndex], np.ndarray],
    model_name: str = "model",
    n_origins: int = 5,
    horizon: int = 24,
    min_train: int = SEASONALITY_WEEKLY,
    include_baselines: bool = True,
    extra_baselines: Optional[
        Dict[str, Callable[[pd.Series, pd.DatetimeIndex], np.ndarray]]
    ] = None,
) -> pd.DataFrame:
    """
    rolling-origin 백테스트의 타임스탬프별 오류를 long 형태로 반환한다.

    run_rolling_origin_backtest와 동일한 분할·기준 모델을 사용하지만
    기점별 집계 대신 (모델, 기점, 타임스탬프) 단위의 y_true/y_pred/오차를
    남겨 segment_metrics 등 후속 분해에 쓸 수 있게 한다.

    Args:
        series: 시간별 인덱스 대상 Series
        forecast_fn: (학습 Series, 평가 인덱스) -> 예측 배열
        model_name: 평가 모델 이름
        n_origins: 기점 수
        horizon: 기점별 예측 시간 수
        min_train: 첫 기점까지의 최소 학습 관측 수
        include_baselines: 표준 기준 모델 포함 여부
        extra_baselines: 추가 기준 모델 {이름: 예측 함수}

    Returns:
        DataFrame: model, origin, timestamp, y_true, y_pred, abs_error, sq_error
    """
    series = series.sort_index()
    splits = rolling_origin_splits(series, n_origins, horizon, min_train)

    rows = []
    for train_index, test_index in splits:
        train = series.loc[train_index]
        y_true = series.loc[test_index].to_numpy(dtype=float)
        origin = cast(pd.Timestamp, train_index[-1])

        candidates: Dict[str, np.ndarray] = {
            model_name: np.asarray(forecast_fn(train, test_index), dtype=float)
        }
        if include_baselines:
            candidates.update(default_baselines(train, test_index))
        if extra_baselines:
            for name, fn in extra_baselines.items():
                candidates[name] = np.asarray(fn(train, test_index), dtype=float)

        for name, y_pred in candidates.items():
            errors = np.abs(y_true - y_pred)
            for ts, actual, pred, abs_err, sq_err in zip(
                test_index, y_true, y_pred, errors, errors**2
            ):
                rows.append(
                    {
                        "model": name,
                        "origin": origin,
                        "timestamp": ts,
                        "y_true": float(actual),
                        "y_pred": float(pred),
                        "abs_error": float(abs_err),
                        "sq_error": float(sq_err),
                    }
                )
    return pd.DataFrame(rows)


def segment_metrics(
    error_frame: pd.DataFrame,
    holidays: Optional[set] = None,
) -> pd.DataFrame:
    """
    타임스탬프별 오류 프레임을 구간별 지표로 분해한다.

    구간 정의:
    - peak_hour: PEAK_HOURS_START 이상 OFF_PEAK_HOURS_END 미만 (피크/비피크)
    - weekday_weekend: 주중/주말 (요일 5 이상)
    - holiday: 호출자가 전달한 휴일 날짜 집합 기준 (휴일/비휴일)

    Args:
        error_frame: rolling_origin_error_frame 출력. timestamp 컬럼 필수
        holidays: 휴일 datetime.date 집합 (선택). 미제공 시 holiday 구간 미생성

    Returns:
        DataFrame: model, segment_type, segment, n, MAE, RMSE, MAPE.
        각 segment_type은 모델별 전체 타임스탬프를 정확히 한 번씩 커버한다.
    """
    if "timestamp" not in error_frame.columns:
        raise ValueError("error_frame must contain a 'timestamp' column")
    frame = error_frame.copy()
    ts = pd.to_datetime(frame["timestamp"])
    frame["timestamp"] = ts
    hours = ts.dt.hour
    weekdays = ts.dt.dayofweek

    frame["peak_flag"] = (hours >= PEAK_HOURS_START) & (hours < OFF_PEAK_HOURS_END)
    frame["weekend_flag"] = weekdays >= 5
    if holidays is not None:
        frame["holiday_flag"] = ts.dt.normalize().dt.date.isin(holidays)

    def _aggregate(labels: pd.Series, seg_type: str) -> pd.DataFrame:
        grouped = frame.groupby(["model", labels], sort=False)
        result = grouped.agg(
            n=("abs_error", "size"),
            mae_val=("abs_error", "mean"),
            rmse_val=("sq_error", lambda s: float(np.sqrt(s.mean()))),
        ).reset_index()
        true_means = grouped["y_true"].mean().reset_index()["y_true"]
        # MAPE: 구간별 실측 평균 대비 절대 오차 비율 (0 나눗셈 방지)
        denom = true_means.to_numpy(dtype=float)
        result["MAPE"] = np.where(
            denom > 0, result["mae_val"] / denom * 100.0, np.nan
        )
        result = result.rename(
            columns={result.columns[1]: "segment", "mae_val": "MAE", "rmse_val": "RMSE"}
        )
        result.insert(1, "segment_type", seg_type)
        return result[["model", "segment_type", "segment", "n", "MAE", "RMSE", "MAPE"]]

    parts = [
        _aggregate(frame["peak_flag"].map({True: "peak", False: "off_peak"}), "peak_hour"),
        _aggregate(
            frame["weekend_flag"].map({True: "weekend", False: "weekday"}),
            "weekday_weekend",
        ),
    ]
    if holidays is not None:
        parts.append(
            _aggregate(
                frame["holiday_flag"].map({True: "holiday", False: "non_holiday"}),
                "holiday",
            )
        )
    return pd.concat(parts, ignore_index=True)
