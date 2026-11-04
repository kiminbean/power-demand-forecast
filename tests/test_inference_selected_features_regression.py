"""Selected model features determine usable inference history."""

import numpy as np
import pandas as pd
import pytest

from api.schemas import ModelType, TimeSeriesData
from api.service import PredictionService
from inference.feature_preprocessing import prepare_inference_features
from inference.predict import ProductionPredictor
from tests.test_batch_insufficient_history_regression import configured_predictor, frame_for


@pytest.mark.parametrize("start", [None, 300])
def test_unused_optional_missing_values_do_not_drop_selected_history(start: int | None) -> None:
    frame = frame_for(400)
    frame["습도"] = np.nan
    if start is not None:
        frame.iloc[start:, frame.columns.get_loc("습도")] = 60.0
    result = prepare_inference_features(frame, selected_features=["power_demand", "demand_lag_168"])
    assert len(result) == 232
    assert result.index[0] == frame.index[168]


def test_selected_values_are_causally_filled_and_order_preserved() -> None:
    frame = frame_for(400)
    frame["기온"] = np.nan
    frame.iloc[200:, frame.columns.get_loc("기온")] = 3.0
    frame.iloc[250, frame.columns.get_loc("기온")] = np.nan
    altered = frame.copy()
    altered.iloc[251:, altered.columns.get_loc("기온")] = 99.0
    features = ["기온", "power_demand"]
    result = prepare_inference_features(frame, selected_features=features)
    other = prepare_inference_features(altered, selected_features=features)
    assert result.index[0] == frame.index[200]
    assert result.loc[frame.index[250], "기온"] == 3.0
    pd.testing.assert_series_equal(result.loc[frame.index[250], features], other.loc[frame.index[250], features])
    assert (result.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_demand_features_use_causally_filled_raw_history() -> None:
    frame = frame_for(400)
    frame.iloc[300, frame.columns.get_loc("power_demand")] = np.nan
    features = ["power_demand", "demand_lag_1", "demand_ma_6h", "demand_diff_1h"]
    result = prepare_inference_features(frame, selected_features=features)
    filled = frame["power_demand"].ffill()
    for position in (300, 301):
        timestamp = frame.index[position]
        assert result.loc[timestamp, "demand_lag_1"] == filled.iloc[position - 1]
        assert result.loc[timestamp, "demand_ma_6h"] == pytest.approx(
            filled.iloc[position - 6:position].mean()
        )
        assert result.loc[timestamp, "demand_diff_1h"] == (
            filled.iloc[position] - filled.iloc[position - 1]
        )


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_selected_nonfinite_values_are_rejected(value: float) -> None:
    frame = frame_for(400)
    frame.iloc[-1, frame.columns.get_loc("power_demand")] = value
    with pytest.raises(ValueError, match="(?i)(finite|infinite)"):
        prepare_inference_features(frame, selected_features=["power_demand", "demand_lag_168"])


def test_unused_infinite_values_do_not_block_selected_features() -> None:
    frame = frame_for(400)
    frame["unrelated"] = np.inf
    result = prepare_inference_features(frame, selected_features=["power_demand", "demand_lag_168"])
    assert len(result) == 232


def test_missing_configured_feature_is_explicit() -> None:
    with pytest.raises(ValueError, match="missing_feature"):
        prepare_inference_features(frame_for(400), selected_features=["missing_feature"])


def test_all_missing_selected_feature_yields_no_usable_history() -> None:
    frame = frame_for(400)
    frame["기온"] = np.nan
    result = prepare_inference_features(frame, selected_features=["power_demand", "기온"])
    assert result.empty


def test_unused_dewpoint_does_not_break_weather_with_no_temp_mean() -> None:
    frame = frame_for(400)
    frame["기온"] = 12.0
    frame["dewpoint_mean"] = 5.0
    result = prepare_inference_features(
        frame, include_weather=True, selected_features=["power_demand", "기온"]
    )
    assert len(result) == 400


def test_production_checkpoints_match_clean_legacy_features_and_predictions() -> None:
    import torch
    from features import add_lag_features, add_time_features, add_weather_features

    predictor = ProductionPredictor(device=torch.device("cpu"))
    predictor.load_models()
    frame = frame_for(400)
    frame["power_demand"] = 500 + 40 * np.sin(np.arange(400) / 24)
    frame["기온"] = 12 + np.cos(np.arange(400) / 24)
    for weather, config, predict, scaler, model in (
        (False, predictor.config_demand, predictor.predict_demand_only, predictor.scaler_demand, predictor.model_demand),
        (True, predictor.config_weather, predictor.predict_weather_full, predictor.scaler_weather, predictor.model_weather),
    ):
        features = config["features"]
        # 이전 직접 추론 경로의 생성 순서와 dropna를 독립적으로 재현한다.
        old = add_time_features(frame.copy(), include_holiday=True)
        old = add_lag_features(old, demand_col="power_demand")
        if weather and "기온" in old.columns:
            old = add_weather_features(old, include_thi=True, include_hdd_cdd=True)
        old = old.dropna()
        new = prepare_inference_features(frame, include_weather=weather, selected_features=features)
        pd.testing.assert_frame_equal(old[features], new.loc[old.index, features])
        sequence = old[features].to_numpy()[-config["training_config"]["seq_length"]:]
        with torch.no_grad():
            output = model(torch.tensor(scaler.transform(sequence), dtype=torch.float32).unsqueeze(0))
        expected = scaler.inverse_transform_target(output.numpy().flatten())[0]
        assert predict(frame) == pytest.approx(float(expected), abs=1e-5)


def test_direct_and_api_single_and_batch_ignore_unused_missing_columns(configured_predictor) -> None:
    frame = frame_for(400)
    frame["습도"] = np.nan
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value,
                              humidity=60.0 if i >= 300 else None)
               for i, (ts, value) in enumerate(frame["power_demand"].items())]
    assert configured_predictor.predict_demand_only(frame) == service._run_prediction(frame, "demand_only")
    direct = configured_predictor.predict_batch(frame, step=24)
    api = service.predict_batch(records, step=24)
    assert direct.predictions.tolist() == [item["prediction"] for item in api.predictions]


def test_winter_conditional_falls_back_when_weather_features_missing(configured_predictor) -> None:
    frame = frame_for(400)
    configured_predictor.model_weather = configured_predictor.model_demand
    configured_predictor.scaler_weather = configured_predictor.scaler_demand
    configured_predictor.config_weather = {
        "features": ["power_demand", "기온"], "training_config": {"seq_length": 168}
    }
    service = PredictionService()
    service.predictor = configured_predictor
    service._is_initialized = True
    records = [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=value)
               for ts, value in frame["power_demand"].items()]
    direct = configured_predictor.predict_conditional(frame)
    api = service.predict(records, ModelType.CONDITIONAL)
    assert direct.model_used == "demand_only (fallback)"
    assert api.model_used == "demand_only (fallback)"
    assert api.prediction == pytest.approx(direct.predicted_demand)



