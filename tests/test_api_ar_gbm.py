"""API의 선택형 AR + GBM 모델(model_type="ar_gbm") 테스트."""

import json
import shutil
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config import settings
from api.schemas import ModelType, TimeSeriesData
from api.service import ModelNotLoadedError, PredictionService
from src.models.ar_gbm import MIN_HISTORY_HOURS, ArGbmModel
from src.training.train_production import TIME_FEATURES
from tests.test_ar_gbm import FEATURES, GATE_PASSED, TRAIN_END, VAL_END, _raw, _training_frame


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    raw = _raw()
    frame = _training_frame(raw, tmp_path_factory.mktemp("frame"))
    model = ArGbmModel(FEATURES, TIME_FEATURES).fit(frame, TRAIN_END, VAL_END)
    path = tmp_path_factory.mktemp("ar_gbm")
    model.save(path, extra=GATE_PASSED)
    return raw, path, model.predict(frame)


def _service(monkeypatch, model_dir) -> PredictionService:
    monkeypatch.setattr(settings, "AR_GBM_MODEL_DIR", str(model_dir))
    service = PredictionService()
    service.predictor = Mock()  # LSTM 경로는 이 테스트에서 쓰지 않는다
    service._is_initialized = True
    service._load_ar_gbm()
    return service


def _data(series: pd.Series):
    return [TimeSeriesData(datetime=ts.to_pydatetime(), power_demand=float(v)) for ts, v in series.items()]


def test_initialize_loads_lstm_and_optional_ar_gbm(monkeypatch, artifact, tmp_path):
    _, path, _ = artifact
    for model_dir, loaded in ((path, True), (tmp_path / "missing", False)):
        monkeypatch.setattr(settings, "AR_GBM_MODEL_DIR", str(model_dir))
        with patch("api.service.ProductionPredictor"):
            service = PredictionService()
            service.initialize()
        # 선택형 모델이 없어도 기본 서비스는 준비 상태가 되어야 한다
        assert service.is_ready()
        assert (service.ar_gbm is not None) is loaded


def test_service_predicts_next_hour_like_training_path(monkeypatch, artifact):
    raw, path, expected = artifact
    service = _service(monkeypatch, path)
    t = raw.index[2500]
    history = raw[raw.index < t].tail(MIN_HISTORY_HOURS)
    result = service.predict(_data(history), ModelType.AR_GBM)
    # API 계약: timestamp는 마지막 입력 시각, 예측값은 그다음 시각
    assert result.model_used == "ar_gbm"
    assert pd.Timestamp(result.timestamp) == history.index[-1]
    assert result.prediction == pytest.approx(round(expected[t], 2))
    entry = next(m for m in service.get_model_info() if m["name"] == "ar_gbm")
    assert entry["min_history_hours"] == MIN_HISTORY_HOURS and entry["status"] == "loaded"
    assert (entry["train_end"], entry["val_end"]) == (str(pd.Timestamp(TRAIN_END)), str(pd.Timestamp(VAL_END)))
    assert entry["holiday_table_through"] >= 2025


def test_service_batch_follows_lstm_label_contract(monkeypatch, artifact):
    raw, path, expected = artifact
    service = _service(monkeypatch, path)
    history = raw.iloc[1000:1400]
    result = service.predict_batch(_data(history), ModelType.AR_GBM, step=4)
    labels = pd.DatetimeIndex([p["timestamp"] for p in result.predictions])
    # 첫 예측 대상은 337번째 이후 시각, 마지막은 관측 다음 시각 → 표시는 각각 1시간 앞
    full = pd.date_range(history.index[MIN_HISTORY_HOURS - 1], history.index[-1], freq="h")
    assert labels.equals(full[::4])
    values = [p["prediction"] for p in result.predictions]
    targets = labels + pd.Timedelta(hours=1)
    np.testing.assert_allclose(values[:-1], expected.reindex(targets[:-1]).round(2), atol=1e-9)
    assert result.model_used == "ar_gbm" and result.total_predictions == len(full[::4])


def test_service_rejects_short_history(monkeypatch, artifact):
    raw, path, _ = artifact
    service = _service(monkeypatch, path)
    with pytest.raises(ValueError, match="Insufficient data"):
        service.predict(_data(raw.iloc[:MIN_HISTORY_HOURS - 1]), ModelType.AR_GBM)


def test_missing_or_broken_artifact_disables_only_ar_gbm(monkeypatch, tmp_path, artifact):
    raw, path, _ = artifact
    history = _data(raw.iloc[:MIN_HISTORY_HOURS])
    missing = _service(monkeypatch, tmp_path / "missing")
    assert missing.ar_gbm is None and "not found" in missing._ar_gbm_error
    # 응답에는 서버 경로를 싣지 않는다 (DEBUG일 때만 원인을 붙인다)
    with pytest.raises(ModelNotLoadedError) as excinfo:
        missing.predict(history, ModelType.AR_GBM)
    assert str(excinfo.value) == "Model 'ar_gbm' is not available"
    monkeypatch.setattr(settings, "DEBUG", True)
    with pytest.raises(ModelNotLoadedError, match="not found"):
        missing.predict(history, ModelType.AR_GBM)

    broken = tmp_path / "broken"
    shutil.copytree(path, broken)
    meta = json.loads((broken / "metadata.json").read_text())
    del meta["design_columns"]
    (broken / "metadata.json").write_text(json.dumps(meta))
    # 형식이 깨진 산출물(KeyError)도 기본 서비스 초기화를 막지 않는다
    service = _service(monkeypatch, broken)
    assert service.ar_gbm is None and service._ar_gbm_error.startswith("KeyError")
    with pytest.raises(ModelNotLoadedError):
        service.predict_batch(history, ModelType.AR_GBM)


@pytest.fixture
def mock_service():
    with patch('api.main.get_prediction_service') as get_service:
        service = Mock()
        service.is_ready.return_value = True
        get_service.return_value = service
        yield service


@pytest.fixture
def client(mock_service):
    with patch('api.main.initialize_service'):
        from api.main import app
        with TestClient(app) as c:
            yield c


def _payload(model_type: str, n: int = 168):
    times = pd.date_range("2024-01-08", periods=n, freq="h")
    return {"data": [{"datetime": ts.isoformat(), "power_demand": 600.0} for ts in times], "model_type": model_type}


@pytest.mark.parametrize("endpoint", ["/predict", "/predict/batch"])
def test_endpoints_map_unloaded_ar_gbm_to_503(client, mock_service, endpoint):
    error = ModelNotLoadedError("Model 'ar_gbm' is not available")
    mock_service.predict.side_effect = error
    mock_service.predict_batch.side_effect = error
    response = client.post(endpoint, json=_payload("ar_gbm"))
    assert response.status_code == 503
    assert response.json()["detail"] == "Model 'ar_gbm' is not available"


def test_predict_endpoint_passes_ar_gbm_model_type(client, mock_service):
    mock_service.predict.return_value = {"success": True, "prediction": 612.3, "model_used": "ar_gbm",
                                         "timestamp": "2024-01-14T23:00:00", "processing_time_ms": 1.0}
    response = client.post("/predict", json=_payload("ar_gbm"))
    assert response.status_code == 200
    assert mock_service.predict.call_args.kwargs["model_type"] == ModelType.AR_GBM
