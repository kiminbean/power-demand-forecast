"""Regression tests for the demand LSTM out-of-time retraining helpers."""

import numpy as np
import pandas as pd
import pytest

import torch

from src.models.lstm import LastValueResidual, create_model
from src.training.retrain_demand_oot import baselines, fill_short_gaps, split


def test_residual_wrapper_starts_at_persistence():
    torch.manual_seed(0)
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=2, dropout=0.0)
    model = LastValueResidual(base).eval()
    x = torch.rand(4, 10, 3)
    # 출력층이 0으로 초기화되므로 예측은 마지막 시점의 타깃 값과 같아야 한다
    assert torch.allclose(model(x), x[:, -1, 0:1])
    # 저장 시 base만 쓰므로 state_dict 키가 기존 LSTMModel과 같아야 API가 그대로 로드할 수 있다
    assert set(model.base.state_dict()) == set(create_model(
        model_type="lstm", input_size=3, hidden_size=8, num_layers=2, dropout=0.0).state_dict())


def test_residual_wrapper_maps_anchor_column_into_target_scale():
    torch.manual_seed(0)
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
    model = LastValueResidual(base, target_idx=2, anchor_scale=2.0, anchor_shift=-0.5).eval()
    x = torch.rand(4, 10, 3)
    # 앵커 열(AR 예측)은 자기 min-max로 스케일되므로 타깃 스케일로 옮긴 뒤 더해야 한다
    assert torch.allclose(model(x), x[:, -1, 2:3] * 2.0 - 0.5)


def test_inference_refuses_ar_anchored_checkpoint():
    from src.inference.predict import _wrap_if_residual
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
    checkpoint = _checkpoint(True)
    checkpoint["model_config"]["residual_anchor"] = "ar_forecast"
    # 서빙 경로는 AR 앵커 열을 아직 만들지 못하므로 잘못된 앵커로 조용히 예측하면 안 된다
    with pytest.raises(ValueError, match="ar_forecast"):
        _wrap_if_residual(base, checkpoint)


def _hourly(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n, freq="h"),
        "power_demand": np.arange(n, dtype=float) + 1000,
        "기온": 10.0,
    })


def test_fill_short_gaps_interpolates_missing_hour(tmp_path):
    src, out = tmp_path / "src.csv", tmp_path / "out.csv"
    _hourly().drop(index=[100]).to_csv(src, index=False)
    filled = fill_short_gaps(src, out)
    result = pd.read_csv(out, parse_dates=["datetime"])
    assert filled == ["2023-01-05 04:00:00"]
    assert len(result) == 400
    assert result.loc[100, "power_demand"] == pytest.approx(1100.0)


def test_fill_short_gaps_rejects_long_gap(tmp_path):
    src = tmp_path / "src.csv"
    _hourly().drop(index=[100, 101, 102]).to_csv(src, index=False)
    with pytest.raises(ValueError, match="Gap longer"):
        fill_short_gaps(src, tmp_path / "out.csv")


def test_split_is_strictly_chronological():
    df = _hourly(1000).set_index("datetime")
    tr, va, te = split(df, "2023-01-15 23:00", "2023-01-30 23:00", "2023-02-28 23:00")
    assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()


def test_persistence_baseline_uses_previous_hour():
    s = pd.Series(np.arange(300, dtype=float) + 100,
                  index=pd.date_range("2023-01-01", periods=300, freq="h"))
    idx = s.index[200:]
    result = baselines(s, idx, s.loc[idx].values)
    # 매시간 1씩 증가하므로 t-1 예측의 절대오차는 정확히 1
    assert result["persistence_t-1"]["MAE"] == pytest.approx(1.0)


def _ar_series(n: int = 168 * 20, seed: int = 0) -> pd.Series:
    """선형 AR이 참 모형인 시계열 (평균 500)."""
    rng = np.random.default_rng(seed)
    y = np.full(n, 500.0)
    for t in range(169, n):
        y[t] = 50 + 0.5 * y[t - 1] + 0.25 * y[t - 24] + 0.15 * y[t - 168] + rng.normal(0, 5)
    return pd.Series(y, index=pd.date_range("2023-01-02", periods=n, freq="h"))


def test_baselines_include_ramp7_and_hourly_ar():
    s = _ar_series()
    train_end = s.index[168 * 15]
    idx = s.index[s.index > train_end]
    result = baselines(s, idx, s.loc[idx].values, train_end=train_end)
    assert {"persistence_t-1", "ramp7", "hourly_ar"} <= set(result)
    assert result["hourly_ar"]["MAE"] < result["persistence_t-1"]["MAE"]


def test_baselines_reject_missing_history():
    s = _ar_series(n=400)
    # t=168에서 t-168 naive는 가능하지만 ramp7은 y(t-169)가 없다 → 조용히 NaN 지표를 내면 안 된다
    with pytest.raises(ValueError, match="ramp7"):
        baselines(s, s.index[168:], s.iloc[168:].values)


def test_gate_requires_beating_every_gate_baseline():
    from src.training.retrain_demand_oot import passes_gate
    base = {"persistence_t-1": {"MAPE": 5.0, "RMSE": 30.0},
            "ramp7": {"MAPE": 4.0, "RMSE": 25.0},
            "hourly_ar": {"MAPE": 3.8, "RMSE": 24.0}}
    assert passes_gate({"MAPE": 3.7, "RMSE": 23.0}, base)
    assert not passes_gate({"MAPE": 4.3, "RMSE": 29.0}, base)  # persistence만 이김
    assert not passes_gate({"MAPE": 3.7, "RMSE": 24.5}, base)  # MAPE만 이김


def test_gate_requires_significant_win_over_every_gate_baseline():
    from src.training.retrain_demand_oot import significant
    dm = {"persistence_t-1": {"p_a_better": 0.001}, "ramp7": {"p_a_better": 0.01},
          "hourly_ar": {"p_a_better": 0.2}}
    # AR 앵커 모델은 학습이 거의 안 돼도 AR을 아주 작은 차이로 이길 수 있다 → 우연이면 탈락
    assert not significant(dm)
    dm["hourly_ar"]["p_a_better"] = 0.01
    assert significant(dm)


def test_gate_requires_material_mae_gain_over_every_gate_baseline():
    from src.training.retrain_demand_oot import material
    base = {"persistence_t-1": {"MAE": 25.0}, "ramp7": {"MAE": 19.0}, "hourly_ar": {"MAE": 18.0}}
    # 1에폭 스모크에서 AR 앵커 모델은 MAE 0.0014 차이로도 DM이 유의했다 → 효과 크기도 요구한다
    assert not material({"MAE": 17.9}, base)  # AR 대비 0.6% 개선
    assert material({"MAE": 17.8}, base)      # AR 대비 1.1% 개선


def test_ar_anchor_column_holds_next_hour_forecast_from_train_fit():
    from src.training.retrain_demand_oot import add_ar_anchor
    s = _ar_series()
    df = pd.DataFrame({"power_demand": s, "x": 1.0})
    train_end = s.index[168 * 14]  # 시간대별 회귀에 필요한 최소 행 수(14항 × 5)를 넘기는 길이
    out, ar = add_ar_anchor(df, train_end)
    assert out["ar_forecast"].notna().all()
    assert (out.index[1:] - out.index[:-1] == pd.Timedelta(hours=1)).all()
    # 행 t의 값은 t+1 시각의 AR 예측이어야 입력 창 마지막 행에서 타깃의 앵커가 된다
    t = out.index[500]
    assert out.loc[t, "ar_forecast"] == pytest.approx(ar.predict(s)[t + pd.Timedelta(hours=1)])
    # 학습 구간 뒤 값을 바꿔도 계수가 같아야 한다 (test 정보 누수 없음)
    changed = df.copy()
    changed.loc[changed.index > train_end, "power_demand"] *= 2
    np.testing.assert_allclose(add_ar_anchor(changed, train_end)[1].coef, ar.coef)


def test_anchor_affine_matches_min_max_scaler():
    from src.data.dataset import TimeSeriesScaler
    from src.training.retrain_demand_oot import anchor_affine
    rng = np.random.default_rng(1)
    train = np.column_stack([rng.uniform(300, 700, 200), rng.uniform(0, 1, 200),
                             rng.uniform(320, 680, 200)])
    scale, shift = anchor_affine(train, anchor_idx=2)
    scaler = TimeSeriesScaler().fit(train)
    target_scaled = (train[:, 2] - scaler.min_vals[0]) / scaler.range_vals[0]
    np.testing.assert_allclose(scaler.transform(train)[:, 2] * scale + shift, target_scaled)


def _checkpoint(residual: bool, first_feature: str = "power_demand") -> dict:
    return {"model_config": {"model_type": "lstm", "residual_target": residual},
            "features": [first_feature, "hour_sin", "hour_cos"]}


def test_inference_wraps_residual_checkpoint_without_reinitializing():
    from src.inference.predict import _wrap_if_residual
    torch.manual_seed(0)
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=1, dropout=0.0).eval()
    x = torch.rand(2, 10, 3)
    raw = base(x).detach()
    wrapped = _wrap_if_residual(base, _checkpoint(True)).eval()
    # predict.py는 models.lstm, 테스트는 src.models.lstm으로 임포트해 클래스 객체가 다르므로 이름으로 확인
    assert type(wrapped).__name__ == "LastValueResidual"
    # 불러온 가중치를 0으로 덮어쓰지 않아야 한다
    assert torch.allclose(wrapped(x), x[:, -1, 0:1] + raw, atol=1e-6)


def test_inference_leaves_plain_checkpoint_untouched():
    from src.inference.predict import _wrap_if_residual
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
    assert _wrap_if_residual(base, _checkpoint(False)) is base


def test_inference_rejects_residual_without_demand_first():
    from src.inference.predict import _wrap_if_residual
    base = create_model(model_type="lstm", input_size=3, hidden_size=8, num_layers=1, dropout=0.0)
    with pytest.raises(ValueError, match="power_demand"):
        _wrap_if_residual(base, _checkpoint(True, first_feature="기온"))


def test_predict_period_stops_at_test_end():
    from src.data.dataset import TimeSeriesScaler
    from src.training.retrain_demand_oot import predict_period
    df = pd.DataFrame({"power_demand": np.arange(600, dtype=float) + 100, "x": 1.0},
                      index=pd.date_range("2023-01-01", periods=600, freq="h"))
    scaler = TimeSeriesScaler()
    scaler.fit(df.values)
    model = LastValueResidual(create_model(model_type="lstm", input_size=2, hidden_size=4,
                                           num_layers=1, dropout=0.0)).eval()
    start, end = df.index[300], df.index[399]
    idx, _, _ = predict_period(model, scaler, df, start, torch.device("cpu"), test_end=end)
    # walk-forward fold는 해당 테스트 구간만 평가해야 한다 (데이터 끝까지 넘어가면 안 됨)
    assert idx[0] == start and idx[-1] == end and len(idx) == 100
