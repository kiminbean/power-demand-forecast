"""Regression tests for checkpoint-selected weather features."""

import numpy as np
import pytest
import torch

from src.features.weather_features import MAGNUS_B, add_weather_features
from src.inference.feature_preprocessing import prepare_inference_features
from src.inference.predict import ProductionPredictor
from tests.test_batch_insufficient_history_regression import frame_for


@pytest.mark.parametrize("feature", ["humidity", "THI", "wind_chill", "HDD", "CDD", "HDD_cumsum", "CDD_cumsum"])
def test_selected_derived_weather_features_need_no_unrelated_korean_temperature(feature: str) -> None:
    frame = frame_for(400)
    frame["temp_mean"] = 10.0
    frame["dewpoint_mean"] = 5.0
    frame["wind_speed_mean"] = 3.0
    expected = add_weather_features(frame)
    actual = prepare_inference_features(frame, include_weather=True, selected_features=[feature])
    np.testing.assert_allclose(actual[feature], expected[feature])


def test_real_weather_checkpoint_ignores_malformed_unused_weather_columns() -> None:
    predictor = ProductionPredictor(device=torch.device("cpu"))
    predictor.load_models()
    frame = frame_for(400)
    frame["power_demand"] = 500 + 40 * np.sin(np.arange(400) / 24)
    frame["기온"] = 12.0
    frame["temp_mean"] = 12.0
    baseline = predictor.predict_weather_full(frame)
    frame["dewpoint_mean"] = "bad"
    frame["wind_speed_mean"] = "bad"
    assert predictor.predict_weather_full(frame) == pytest.approx(baseline)
    result = predictor.predict_conditional(frame, mode="hard")
    assert result.model_used == "weather_full"
    assert result.predicted_demand == pytest.approx(baseline)


@pytest.mark.parametrize("column", ["기온", "습도", "일사"])
def test_unused_weather_lag_inputs_do_not_break_demand_inference(column: str) -> None:
    frame = frame_for(400)
    frame[column] = "bad"
    result = prepare_inference_features(frame, selected_features=["power_demand"])
    assert len(result) == 400


def test_selected_weather_lags_ignore_other_bad_weather_sources() -> None:
    frame = frame_for(400)
    frame["기온"] = 12.0
    frame["습도"] = "bad"
    result = prepare_inference_features(frame, selected_features=["temp_ma_6h"])
    assert result["temp_ma_6h"].iloc[-1] == pytest.approx(12.0)


def test_precomputed_weather_feature_is_preserved_without_korean_temperature() -> None:
    frame = frame_for(400)
    frame["temp_mean"] = 10.0
    frame["HDD"] = 999.0
    result = prepare_inference_features(frame, include_weather=True, selected_features=["HDD"])
    assert (result["HDD"] == 999.0).all()


@pytest.mark.parametrize("feature", ["HDD", "THI"])
def test_nonfinite_raw_temperature_cannot_be_hidden_by_weather_formula(feature: str) -> None:
    frame = frame_for(400)
    frame["temp_mean"] = 10.0
    frame["dewpoint_mean"] = 5.0
    frame.iloc[-1, frame.columns.get_loc("temp_mean")] = np.inf
    with pytest.raises(ValueError, match="finite"):
        prepare_inference_features(frame, include_weather=True, selected_features=[feature])


@pytest.mark.parametrize("column", ["기온", "습도", "일사", "temp_mean", "dewpoint_mean", "wind_speed_mean"])
def test_unselected_path_rejects_nonfinite_weather_derivation_input(column: str) -> None:
    frame = frame_for(400)
    for name in ["기온", "습도", "일사", "temp_mean", "dewpoint_mean", "wind_speed_mean"]:
        frame[name] = 10.0
    frame.iloc[-1, frame.columns.get_loc(column)] = np.inf
    with pytest.raises(ValueError, match="finite"):
        prepare_inference_features(frame, include_weather=True)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_unselected_path_rejects_nonfinite_uncomputed_weather_column(value: float) -> None:
    frame = frame_for(400)
    frame["temp_mean"] = 10.0
    frame.iloc[-1, frame.columns.get_loc("temp_mean")] = value
    with pytest.raises(ValueError, match="finite"):
        prepare_inference_features(frame, include_weather=True)


@pytest.mark.parametrize("features", [None, ["humidity"], ["THI"]])
def test_invalid_humidity_calculation_is_not_forward_filled(features: list[str] | None) -> None:
    frame = frame_for(400)
    frame["기온"] = 10.0
    frame["temp_mean"] = 10.0
    frame["dewpoint_mean"] = 5.0
    frame.loc[frame.index[300], ["temp_mean", "dewpoint_mean"]] = -MAGNUS_B

    with np.errstate(all="ignore"), pytest.raises(ValueError, match="humidity.*finite"):
        prepare_inference_features(frame, include_weather=True, selected_features=features)


def test_selected_humidity_ignores_nonfinite_unselected_thi() -> None:
    frame = frame_for(400)
    frame["기온"] = 10.0
    frame["temp_mean"] = 1e308
    frame["dewpoint_mean"] = 5.0

    with np.errstate(all="ignore"):
        result = prepare_inference_features(frame, include_weather=True, selected_features=["humidity"])
    assert np.isfinite(result["humidity"]).all()
