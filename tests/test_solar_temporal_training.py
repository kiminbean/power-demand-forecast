import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api import renewable_ml_predictor as solar


class RecordingRegressor:
    def __init__(self, **kwargs: object) -> None:
        self.inputs = []
        self.targets = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.inputs.append(np.asarray(X))
        self.targets.append(np.asarray(y))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self.inputs.append(np.asarray(X))
        self.targets.append(np.asarray(y))
        return 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.inputs.append(np.asarray(X))
        return np.full(len(X), 5.0)


def test_solar_training_uses_temporal_holdout_and_train_only_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sklearn import ensemble

    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    rows = pd.DataFrame({
        "일시": dates,
        "기온": np.arange(10, dtype=float),
        "습도": 50.0,
        "전운량(10분위)": 2.0,
        "일사량": 1.0,
        "태양광 설비용량(MW)": 100.0,
        "태양광 발전량(MWh)": np.arange(10, dtype=float),
    }).sample(frac=1, random_state=42)
    data_path = tmp_path / "solar.csv"
    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    rows.to_csv(data_path, index=False)
    monkeypatch.setattr(solar.SolarMLPredictor, "SOLAR_DATA_PATH", data_path)
    monkeypatch.setattr(solar, "SOLAR_MODEL_PATH", model_path)
    monkeypatch.setattr(solar, "SOLAR_SCALER_PATH", scaler_path)
    monkeypatch.setattr(ensemble, "GradientBoostingRegressor", RecordingRegressor)

    with caplog.at_level(logging.INFO, logger=solar.__name__):
        predictor = solar.SolarMLPredictor()

    np.testing.assert_array_equal(predictor.model.targets[0], np.arange(8))
    np.testing.assert_array_equal(predictor.model.targets[2], np.arange(8, 10))
    temperature_idx = predictor.feature_names.index("temperature")
    assert predictor.scaler.mean_[temperature_idx] == pytest.approx(3.5)
    assert predictor.model.inputs[2][:, temperature_idx].min() > 1.0
    with model_path.open("rb") as file:
        assert isinstance(pickle.load(file), RecordingRegressor)
    with scaler_path.open("rb") as file:
        assert pickle.load(file).mean_[temperature_idx] == pytest.approx(3.5)
    assert predictor.predict(12, 10.0, 50.0, month=1, day_of_year=1)[0] == 5.0
    radiation_idx = predictor.feature_names.index("radiation")
    inferred_radiation = 3.5 * np.sin(6 * np.pi / 13) * (1 - 5 / 10 * 0.8)
    assert predictor.model.inputs[-1][0, radiation_idx] == pytest.approx(
        (inferred_radiation - predictor.scaler.mean_[radiation_idx])
        / predictor.scaler.scale_[radiation_idx]
    )
    assert "measured irradiance" in caplog.text
    assert "default inference" in caplog.text


def test_solar_training_splits_after_large_first_timestamp_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from sklearn import ensemble

    rows = pd.DataFrame({
        "일시": ["2024-01-01 00:00:00"] * 9 + ["2024-01-01 01:00:00"],
        "기온": np.arange(10, dtype=float),
        "습도": 50.0,
        "전운량(10분위)": 2.0,
        "일사량": 1.0,
        "태양광 설비용량(MW)": 100.0,
        "태양광 발전량(MWh)": np.arange(10, dtype=float),
    }).sample(frac=1, random_state=42)
    data_path = tmp_path / "solar.csv"
    rows.to_csv(data_path, index=False)
    monkeypatch.setattr(solar.SolarMLPredictor, "SOLAR_DATA_PATH", data_path)
    monkeypatch.setattr(solar, "SOLAR_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(solar, "SOLAR_SCALER_PATH", tmp_path / "scaler.pkl")
    monkeypatch.setattr(ensemble, "GradientBoostingRegressor", RecordingRegressor)

    with caplog.at_level(logging.INFO, logger=solar.__name__):
        predictor = solar.SolarMLPredictor()

    np.testing.assert_array_equal(np.sort(predictor.model.targets[0]), np.arange(9))
    assert len(predictor.model.targets) == 2  # No R² score for one holdout sample.
    assert "holdout R² unavailable" in caplog.text
    temperature_idx = predictor.feature_names.index("temperature")
    assert predictor.scaler.mean_[temperature_idx] == pytest.approx(4.0)


@pytest.mark.parametrize("invalid_time", ["not-a-date", ""])
def test_solar_training_rejects_malformed_or_missing_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    invalid_time: str,
) -> None:
    rows = pd.DataFrame({
        "일시": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", invalid_time],
        "기온": [1.0, 2.0, 3.0],
        "습도": 50.0,
        "전운량(10분위)": 2.0,
        "일사량": 1.0,
        "태양광 설비용량(MW)": 100.0,
        "태양광 발전량(MWh)": [1.0, 2.0, 3.0],
    })
    data_path = tmp_path / "solar.csv"
    rows.to_csv(data_path, index=False)
    monkeypatch.setattr(solar.SolarMLPredictor, "SOLAR_DATA_PATH", data_path)
    monkeypatch.setattr(solar, "SOLAR_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(solar, "SOLAR_SCALER_PATH", tmp_path / "scaler.pkl")

    predictor = solar.SolarMLPredictor()

    assert predictor.model is None
    assert "Failed to train model" in caplog.text
    assert not (tmp_path / "model.pkl").exists()
    assert not (tmp_path / "scaler.pkl").exists()
