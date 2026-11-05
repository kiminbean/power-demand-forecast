"""
시간대별 선형 자기회귀(AR): 1시간 앞 제주 전력수요의 기준선이자 LSTM 보정의 앵커
==========================================================================

24개 시간대마다 최소제곱 회귀를 따로 적합한다. 입력은 시차 1, 2, 24, 25, 48, 168, 169시간의
수요와 요일 더미(월요일 기준 6개)다. 2026-11-06 경쟁 비교에서 이 단순 모델이 잔차 LSTM보다
정확했다(고정 test MAPE 3.75% vs 4.31%, docs/2026-11-06-competitor-benchmark.md).

계수는 학습 구간 행만으로 적합하고, 시각 t의 예측은 t-1까지의 관측만 쓴다.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

AR_LAGS = (1, 2, 24, 25, 48, 168, 169)
N_TERMS = 1 + len(AR_LAGS) + 6  # 상수 + 시차 + 요일 더미
# 시간대별 회귀 항마다 최소 이만큼의 학습 행을 요구한다 (적으면 계수가 불안정하다)
MIN_ROWS_PER_TERM = 5


def _check_hourly(series: pd.Series) -> None:
    index = pd.DatetimeIndex(series.index)
    if len(index) > 1 and (np.diff(index.asi8) != pd.Timedelta(hours=1).value).any():
        raise ValueError("Series index must be strictly hourly and sorted")


def _design(series: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame({f"lag_{k}": series.shift(k) for k in AR_LAGS}, index=series.index)
    dow = series.index.dayofweek
    for d in range(1, 7):
        X[f"dow_{d}"] = (dow == d).astype(float)
    X.insert(0, "const", 1.0)
    return X


class HourlyLinearAR:
    """예측 시각의 시간대(0~23)별로 계수가 다른 선형 AR."""

    def __init__(self) -> None:
        self.coef: Optional[np.ndarray] = None  # (24, N_TERMS)

    def fit(self, series: pd.Series, train_end, train_start=None) -> "HourlyLinearAR":
        """train_start ≤ t ≤ train_end인 예측 시각 t의 행만으로 계수를 적합한다."""
        _check_hourly(series)
        X, hours = _design(series), series.index.hour
        mask = series.index <= pd.Timestamp(train_end)
        if train_start is not None:
            mask &= series.index >= pd.Timestamp(train_start)
        coef = np.empty((24, N_TERMS))
        for h in range(24):
            rows = mask & (hours == h)
            A, b = X[rows], series[rows]
            ok = (A.notna().all(axis=1) & b.notna()).to_numpy()
            if ok.sum() < MIN_ROWS_PER_TERM * N_TERMS:
                raise ValueError(f"Too few training rows for hour {h}: {int(ok.sum())}")
            coef[h] = np.linalg.lstsq(A.to_numpy()[ok], b.to_numpy()[ok], rcond=None)[0]
        self.coef = coef
        return self

    def predict(self, series: pd.Series) -> pd.Series:
        """각 시각 t의 1시간 앞 예측. 시차가 모자란 앞부분은 NaN이다."""
        if self.coef is None:
            raise RuntimeError("HourlyLinearAR must be fit before predict")
        _check_hourly(series)
        X = _design(series).to_numpy()
        values = np.einsum("ij,ij->i", X, self.coef[series.index.hour])
        return pd.Series(values, index=series.index, name="hourly_ar")

    def predict_next(self, series: pd.Series) -> pd.Series:
        """행 t에 t+1 시각의 예측을 둔다(t까지의 관측만 사용). LSTM 입력 창의 앵커 열로 쓴다."""
        _check_hourly(series)
        nxt = pd.DatetimeIndex([series.index[-1] + pd.Timedelta(hours=1)])
        extended = pd.concat([series, pd.Series([np.nan], index=nxt)])
        return pd.Series(self.predict(extended).to_numpy()[1:], index=series.index, name="ar_forecast")

    def to_dict(self) -> Dict:
        if self.coef is None:
            raise RuntimeError("HourlyLinearAR must be fit before to_dict")
        return {"lags": list(AR_LAGS), "coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, data: Dict) -> "HourlyLinearAR":
        if tuple(data.get("lags", ())) != AR_LAGS:
            raise ValueError(f"Unsupported AR lags: {data.get('lags')}")
        coef = np.asarray(data["coef"], dtype=float)
        if coef.shape != (24, N_TERMS) or not np.isfinite(coef).all():
            raise ValueError(f"AR coefficients must be finite with shape (24, {N_TERMS})")
        model = cls()
        model.coef = coef
        return model


def ramp7_forecast(series: pd.Series) -> pd.Series:
    """y(t-1) + 지난 7일 같은 시각의 1시간 변화량 평균. 학습이 필요 없는 기준선."""
    _check_hourly(series)
    ramp = sum(series.shift(24 * k) - series.shift(24 * k + 1) for k in range(1, 8)) / 7
    return (series.shift(1) + ramp).rename("ramp7")
