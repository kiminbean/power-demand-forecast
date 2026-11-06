"""AR + GBM 잔차 보정 수요 모델과 그 서빙 경로 테스트."""

import copy
import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.inference import ar_gbm_predictor
from src.inference.ar_gbm_predictor import ArGbmPredictor
from src.models.ar_gbm import MIN_HISTORY_HOURS, ArGbmModel
from src.training.train_demand_ar_gbm import FEATURES, load_frame
from src.training.train_production import TIME_FEATURES

TRAIN_END, VAL_END = "2023-04-20 23:00", "2023-05-20 23:00"
GATE_PASSED = {"gate": {"passed": True}}


def _raw(n_hours: int = 24 * 150, seed: int = 0) -> pd.Series:
    """일·주 패턴 위에, 시차 선형 AR로는 못 잡는 비선형 효과를 심은 시간별 수요."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_hours, freq="h", name="datetime")
    base = (500 + 80 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 8) / 24)
            - 30 * (idx.dayofweek.to_numpy() >= 5))
    y = np.empty(n_hours)
    y[0] = base[0]
    for t in range(1, n_hours):
        # 직전 수요가 높으면 다음 시각에 계단처럼 더 오른다 (GBM이 배울 잔차)
        y[t] = base[t] + 0.7 * (y[t - 1] - base[t - 1]) + (25.0 if y[t - 1] > 560 else 0.0) + rng.normal(0, 8)
    return pd.Series(y, index=idx, name="power_demand")


def _training_frame(raw: pd.Series, directory) -> pd.DataFrame:
    """실제 학습 경로(train_demand_ar_gbm.load_frame: 결측 보간 → load_data → 피처 선택)로 만든다."""
    path = directory / f"raw_{id(raw)}.csv"
    raw.reset_index().to_csv(path, index=False)
    return load_frame(path)


@pytest.fixture(scope="module")
def raw() -> pd.Series:
    return _raw()


@pytest.fixture(scope="module")
def frame(raw, tmp_path_factory) -> pd.DataFrame:
    return _training_frame(raw, tmp_path_factory.mktemp("frame"))


@pytest.fixture(scope="module")
def fitted(frame) -> ArGbmModel:
    return ArGbmModel(FEATURES, TIME_FEATURES).fit(frame, TRAIN_END, VAL_END)


def test_fit_learns_residual_the_linear_ar_misses(fitted):
    # 시드 0~2에서 AR 대비 검증 MAE 개선은 3.6~5.4%다. 신호가 없으면 트리 0그루(보정 없음)를 고른다
    assert fitted.info["best_iter"] > 0
    assert fitted.info["val_mae"] < 0.98 * fitted.info["ar_val_mae"]


def test_fit_ignores_data_after_validation(frame, fitted):
    changed = frame.copy()
    after = changed.index > pd.Timestamp(VAL_END)
    changed.loc[after, "power_demand"] *= 1.7
    refit = ArGbmModel(FEATURES, TIME_FEATURES).fit(changed, TRAIN_END, VAL_END)
    # 검증 구간 뒤의 값은 AR 계수, 트리 수, 학습 구간 예측에 영향을 주지 않아야 한다
    np.testing.assert_array_equal(refit.ar.coef, fitted.ar.coef)
    assert refit.info["best_iter"] == fitted.info["best_iter"]
    before = frame.index <= pd.Timestamp(VAL_END)
    np.testing.assert_array_equal(refit.predict(frame)[before].to_numpy(), fitted.predict(frame)[before].to_numpy())


def test_inputs_and_prediction_use_only_past_observations(raw, frame, fitted, tmp_path):
    t = raw.index[3000]
    changed = raw.copy()
    changed[changed.index >= t] *= 2
    changed_frame = _training_frame(changed, tmp_path)
    # t 이후 수요를 바꿔도 t의 입력과 예측은 같아야 한다 (t-1까지의 관측과 t 달력만 쓴다)
    pd.testing.assert_series_equal(fitted.design(frame).loc[t], fitted.design(changed_frame).loc[t])
    assert fitted.predict(frame)[t] == fitted.predict(changed_frame)[t]


def test_save_load_roundtrip_and_version_guards(tmp_path, frame, fitted):
    fitted.save(tmp_path, extra=GATE_PASSED)
    np.testing.assert_array_equal(ArGbmModel.load(tmp_path).predict(frame).to_numpy(),
                                  fitted.predict(frame).to_numpy())
    meta = json.loads((tmp_path / "metadata.json").read_text())
    for key, value, error in (("sklearn_version", "0.0.1", "scikit-learn"),
                              ("numpy_version", "0.9.0", "numpy"),
                              ("resid_lags", [1, 24], "resid_lags")):
        (tmp_path / "metadata.json").write_text(json.dumps({**meta, key: value}))
        # pickle과 입력 정의가 학습 때와 다르면 요청 때가 아니라 불러올 때 실패해야 한다
        with pytest.raises((RuntimeError, ValueError), match=error):
            ArGbmModel.load(tmp_path)


def test_load_rejects_mixed_artifact_files(tmp_path, fitted):
    fitted.save(tmp_path, extra=GATE_PASSED)
    ar = json.loads((tmp_path / "ar_coefficients.json").read_text())
    ar["coef"][0][0] += 5.0
    # AR만 다른 학습의 것으로 바뀌면 GBM이 엉뚱한 잔차 위에 얹힌다 → 해시로 거부
    (tmp_path / "ar_coefficients.json").write_text(json.dumps(ar))
    with pytest.raises(ValueError, match="does not match metadata"):
        ArGbmModel.load(tmp_path)


def test_model_without_trees_predicts_ar(tmp_path, frame, fitted):
    model = copy.deepcopy(fitted)
    model.gbm = None
    model.info = {**model.info, "best_iter": 0}
    model.save(tmp_path)
    assert not (tmp_path / "gbm.joblib").exists()
    loaded = ArGbmModel.load(tmp_path)
    pred = loaded.predict(frame).dropna()
    np.testing.assert_allclose(pred, loaded.design(frame)["ar_forecast"].reindex(pred.index))


@pytest.fixture()
def predictor(tmp_path, fitted) -> ArGbmPredictor:
    fitted.save(tmp_path, extra=GATE_PASSED)
    p = ArGbmPredictor(tmp_path)
    p.load()
    return p


def test_serving_refuses_artifact_that_did_not_pass_gate(tmp_path, fitted):
    for gate in ({}, {"gate": {"passed": False}}, {"gate": {"passed": None, "evaluated": False}}):
        fitted.save(tmp_path, extra=gate)
        with pytest.raises(ValueError, match="deployment gate"):
            ArGbmPredictor(tmp_path).load()


def test_serving_path_matches_training_path(raw, frame, fitted, predictor):
    expected = fitted.predict(frame)
    for t in (raw.index[2000], raw.index[-1]):
        # 서빙은 t 직전 337시간의 원본 수요만 받는다. 학습 경로는 전체 이력으로 피처를 만든다
        history = raw[raw.index < t].tail(MIN_HISTORY_HOURS).reset_index()
        assert predictor.forecast_next(history) == pytest.approx(expected[t], abs=1e-9)
        utc = history.assign(datetime=history["datetime"].dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC"))
        assert predictor.forecast_next(utc) == predictor.forecast_next(history)


def test_serving_batch_covers_every_hour_with_full_lags(raw, frame, fitted, predictor):
    history = raw.iloc[1000:1500]
    out = predictor.forecast(history.reset_index())
    # 마지막 값은 관측 다음 시각, 첫 값은 잔차 168시간 시차와 AR 169시간 시차가 모두 찬 시각이다
    assert out.index[-1] == history.index[-1] + pd.Timedelta(hours=1)
    assert out.index[0] == history.index[MIN_HISTORY_HOURS]
    np.testing.assert_allclose(out.iloc[:-1], fitted.predict(frame).reindex(out.index[:-1]), atol=1e-9)


def test_serving_rejects_short_gapped_or_empty_history(raw, predictor):
    history = raw.iloc[:2000].reset_index()
    predictor.forecast_next(history.tail(MIN_HISTORY_HOURS))
    with pytest.raises(ValueError, match="Insufficient data"):
        predictor.forecast_next(history.tail(MIN_HISTORY_HOURS - 1))
    gapped = history.tail(MIN_HISTORY_HOURS + 5).drop(index=history.index[-10])
    with pytest.raises(ValueError, match="hourly"):
        predictor.forecast_next(gapped)
    blank = history.tail(MIN_HISTORY_HOURS).assign(power_demand=np.nan)
    with pytest.raises(ValueError, match="Insufficient data"):
        predictor.forecast(blank)


def test_predictor_without_artifact_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="AR\\+GBM model not found"):
        ArGbmPredictor(tmp_path / "missing").load()


def test_holiday_table_warning_once_per_year(raw, predictor, monkeypatch, caplog):
    # 합성 데이터(2023년)를 표가 끝난 해로 취급해 경고 경로를 시험한다
    monkeypatch.setattr(ar_gbm_predictor, "LUNAR_HOLIDAY_LAST_YEAR", 2022)
    history = raw.iloc[:2000].tail(MIN_HISTORY_HOURS).reset_index()
    with caplog.at_level(logging.WARNING, logger=ar_gbm_predictor.__name__):
        predictor.forecast_next(history)
        predictor.forecast_next(history)
    warnings = [r for r in caplog.records if "Lunar holidays for 2023" in r.getMessage()]
    assert len(warnings) == 1 and "Buddha" in warnings[0].getMessage()
