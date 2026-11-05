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
