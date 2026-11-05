"""Tests for the hour-of-day linear AR baseline/anchor (src/models/hourly_ar.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.models.hourly_ar import HourlyLinearAR, ramp7_forecast

NOISE_SD = 5.0


def _ar_process(n: int = 168 * 40, seed: int = 0) -> pd.Series:
    """y(t) = 50 + 0.5 y(t-1) + 0.25 y(t-24) + 0.15 y(t-168) + e,  e ~ N(0, 5²) (평균 500)."""
    rng = np.random.default_rng(seed)
    y = np.full(n, 500.0)
    for t in range(169, n):
        y[t] = 50 + 0.5 * y[t - 1] + 0.25 * y[t - 24] + 0.15 * y[t - 168] + rng.normal(0, NOISE_SD)
    return pd.Series(y, index=pd.date_range("2023-01-02", periods=n, freq="h"))


def test_out_of_sample_error_matches_noise_level():
    s = _ar_process()
    train_end = s.index[168 * 30]
    pred = HourlyLinearAR().fit(s, train_end=train_end).predict(s)
    err = (pred - s)[s.index > train_end]
    rmse = float(np.sqrt((err ** 2).mean()))
    # 시차가 어긋나면 오차가 커지고, 미래 값을 쓰면 잡음보다 작아진다
    assert 0.9 * NOISE_SD < rmse < 1.15 * NOISE_SD


def test_prediction_for_t_does_not_use_y_t():
    s = _ar_process()
    model = HourlyLinearAR().fit(s, train_end=s.index[3000])
    t, t_next = s.index[5000], s.index[5001]
    bumped = s.copy()
    bumped[t] += 1000.0
    before, after = model.predict(s), model.predict(bumped)
    assert after[t] == pytest.approx(before[t])
    assert abs(after[t_next] - before[t_next]) > 100


def test_fit_ignores_rows_after_train_end():
    s = _ar_process()
    train_end = s.index[3000]
    changed = s.copy()
    changed[s.index > train_end] *= 3
    a = HourlyLinearAR().fit(s, train_end=train_end)
    b = HourlyLinearAR().fit(changed, train_end=train_end)
    np.testing.assert_allclose(a.coef, b.coef)


def test_predict_next_places_forecast_for_following_hour_on_current_row():
    s = _ar_process()
    model = HourlyLinearAR().fit(s, train_end=s.index[3000])
    nxt, pred = model.predict_next(s), model.predict(s)
    # 행 t에는 t+1 시각의 예측이 있어야 LSTM 입력 창의 마지막 행에서 바로 앵커로 쓸 수 있다
    np.testing.assert_allclose(nxt.to_numpy()[169:-1], pred.to_numpy()[170:])
    assert np.isfinite(nxt.iloc[-1])


def test_dict_roundtrip_keeps_predictions():
    s = _ar_process()
    model = HourlyLinearAR().fit(s, train_end=s.index[3000])
    restored = HourlyLinearAR.from_dict(json.loads(json.dumps(model.to_dict())))
    pd.testing.assert_series_equal(model.predict(s), restored.predict(s))


def test_fit_requires_enough_rows_per_hour():
    s = _ar_process(n=24 * 10)
    with pytest.raises(ValueError, match="Too few"):
        HourlyLinearAR().fit(s, train_end=s.index[-1])


def test_rejects_irregular_index():
    s = _ar_process(n=24 * 30)
    with pytest.raises(ValueError, match="hourly"):
        HourlyLinearAR().fit(s.drop(s.index[300]), train_end=s.index[-1])


def test_ramp7_is_exact_on_linear_trend():
    s = pd.Series(np.arange(400, dtype=float), index=pd.date_range("2023-01-01", periods=400, freq="h"))
    r = ramp7_forecast(s)
    # y(t-1) + 지난 7일 같은 시각 변화량 평균(=1) = t, y(t-169)가 필요하므로 앞 169시간은 NaN
    np.testing.assert_allclose(r.iloc[169:], s.iloc[169:])
    assert r.iloc[:169].isna().all()
