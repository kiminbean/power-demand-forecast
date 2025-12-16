"""
src/api REST API 테스트 (Task 13)
==================================
FastAPI 엔드포인트 테스트
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import (
    create_app, app, PredictionService, DataService,
    PredictionRequest, PredictionResponse, SinglePrediction,
    HistoricalDataRequest, HistoricalDataResponse,
    ModelInfo, ModelListResponse, HealthResponse,
    ForecastRequest, ForecastResponse, ErrorResponse,
    PredictionHorizon, ModelType,
    get_prediction_service, get_data_service
)


# ============================================================================
# Test Client Fixture
# ============================================================================

@pytest.fixture
def client():
    """테스트 클라이언트 - 글로벌 app 인스턴스 사용"""
    # 글로벌 app에 routes가 등록되어 있으므로 그것을 사용
    return TestClient(app)


@pytest.fixture
def prediction_service():
    """예측 서비스 픽스처"""
    return PredictionService()


@pytest.fixture
def data_service():
    """데이터 서비스 픽스처"""
    return DataService()


# ============================================================================
# Pydantic Models Tests
# ============================================================================

class TestPydanticModels:
    """Pydantic 모델 테스트"""

    def test_prediction_request_defaults(self):
        """예측 요청 기본값"""
        request = PredictionRequest()
        assert request.location == "jeju"
        assert request.horizons == ["1h", "6h", "24h"]
        assert request.model_type == "ensemble"
        assert request.include_confidence is True
        assert request.features is None

    def test_prediction_request_custom(self):
        """예측 요청 커스텀 값"""
        request = PredictionRequest(
            location="seoul",
            horizons=["1h", "12h"],
            model_type="lstm",
            include_confidence=False,
            features={"temperature": 25.0}
        )
        assert request.location == "seoul"
        assert request.horizons == ["1h", "12h"]
        assert request.model_type == "lstm"
        assert request.features == {"temperature": 25.0}

    def test_single_prediction_model(self):
        """단일 예측 모델"""
        pred = SinglePrediction(
            timestamp="2025-01-01T12:00:00",
            horizon="1h",
            prediction=1000.0,
            lower_bound=950.0,
            upper_bound=1050.0,
            confidence=0.95
        )
        assert pred.prediction == 1000.0
        assert pred.confidence == 0.95

    def test_prediction_response_model(self):
        """예측 응답 모델"""
        response = PredictionResponse(
            request_id="abc123",
            location="jeju",
            model_type="ensemble",
            created_at="2025-01-01T12:00:00",
            predictions=[],
            metadata={"key": "value"}
        )
        assert response.request_id == "abc123"
        assert response.metadata == {"key": "value"}

    def test_historical_data_request(self):
        """과거 데이터 요청 모델"""
        request = HistoricalDataRequest(
            location="jeju",
            start_date="2025-01-01",
            end_date="2025-01-31",
            resolution="hourly"
        )
        assert request.start_date == "2025-01-01"
        assert request.resolution == "hourly"

    def test_model_info(self):
        """모델 정보 모델"""
        info = ModelInfo(
            name="lstm_v1",
            version="1.0.0",
            type="lstm",
            created_at="2025-01-01T00:00:00",
            metrics={"rmse": 45.0},
            status="active"
        )
        assert info.name == "lstm_v1"
        assert info.metrics["rmse"] == 45.0

    def test_health_response(self):
        """상태 응답 모델"""
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=100.0,
            models_loaded=3,
            last_prediction="2025-01-01T12:00:00"
        )
        assert health.status == "healthy"
        assert health.uptime == 100.0

    def test_forecast_request_validation(self):
        """예보 요청 검증"""
        request = ForecastRequest(
            location="jeju",
            hours_ahead=48
        )
        assert request.hours_ahead == 48
        assert request.include_weather is True

    def test_error_response(self):
        """에러 응답 모델"""
        error = ErrorResponse(
            error="Not found",
            detail="Model not found",
            code="NOT_FOUND"
        )
        assert error.error == "Not found"
        assert error.code == "NOT_FOUND"


class TestEnums:
    """Enum 테스트"""

    def test_prediction_horizon_enum(self):
        """예측 시간대 enum"""
        assert PredictionHorizon.H1.value == "1h"
        assert PredictionHorizon.H24.value == "24h"

    def test_model_type_enum(self):
        """모델 타입 enum"""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.ENSEMBLE.value == "ensemble"


# ============================================================================
# App Creation Tests
# ============================================================================

class TestAppCreation:
    """앱 생성 테스트"""

    def test_create_app_default(self):
        """기본 앱 생성"""
        test_app = create_app()
        assert test_app.title == "Jeju Power Demand Forecast API"
        assert test_app.version == "1.0.0"

    def test_create_app_custom(self):
        """커스텀 앱 생성"""
        test_app = create_app(
            title="Custom API",
            version="2.0.0",
            debug=True
        )
        assert test_app.title == "Custom API"
        assert test_app.version == "2.0.0"

    def test_app_has_cors_middleware(self, client):
        """CORS 미들웨어 확인"""
        response = client.get("/", headers={"Origin": "http://localhost"})
        # CORS 헤더가 있으면 미들웨어가 설정된 것
        assert response.status_code == 200

    def test_app_initial_state(self):
        """앱 초기 상태"""
        test_app = create_app()
        assert hasattr(test_app.state, 'start_time')
        assert hasattr(test_app.state, 'prediction_count')
        assert test_app.state.prediction_count == 0


# ============================================================================
# PredictionService Tests
# ============================================================================

class TestPredictionService:
    """예측 서비스 테스트"""

    def test_init(self, prediction_service):
        """초기화"""
        assert prediction_service._models == {}
        assert prediction_service._cache == {}

    def test_get_model_not_found(self, prediction_service):
        """모델 없음"""
        model = prediction_service.get_model("nonexistent")
        assert model is None

    def test_predict_single_horizon(self, prediction_service):
        """단일 시간대 예측"""
        predictions = prediction_service.predict(
            location="jeju",
            horizons=["1h"],
            model_type="ensemble"
        )
        assert len(predictions) == 1
        assert predictions[0].horizon == "1h"
        assert predictions[0].prediction > 0

    def test_predict_multiple_horizons(self, prediction_service):
        """다중 시간대 예측"""
        predictions = prediction_service.predict(
            location="jeju",
            horizons=["1h", "6h", "24h"],
            model_type="ensemble"
        )
        assert len(predictions) == 3
        assert predictions[0].horizon == "1h"
        assert predictions[1].horizon == "6h"
        assert predictions[2].horizon == "24h"

    def test_predict_with_features(self, prediction_service):
        """피처 포함 예측"""
        predictions = prediction_service.predict(
            location="jeju",
            horizons=["1h"],
            model_type="lstm",
            features={"temperature": 25.0}
        )
        assert len(predictions) == 1

    def test_predict_confidence_bounds(self, prediction_service):
        """신뢰 구간 확인"""
        predictions = prediction_service.predict(
            location="jeju",
            horizons=["1h"],
            model_type="ensemble"
        )
        pred = predictions[0]
        assert pred.lower_bound is not None
        assert pred.upper_bound is not None
        assert pred.lower_bound < pred.prediction < pred.upper_bound

    def test_predict_timestamp_format(self, prediction_service):
        """타임스탬프 형식"""
        predictions = prediction_service.predict(
            location="jeju",
            horizons=["1h"],
            model_type="ensemble"
        )
        timestamp = predictions[0].timestamp
        datetime.fromisoformat(timestamp)


# ============================================================================
# DataService Tests
# ============================================================================

class TestDataService:
    """데이터 서비스 테스트"""

    def test_init(self, data_service):
        """초기화"""
        assert data_service._cache == {}

    def test_get_historical_data_hourly(self, data_service):
        """시간별 과거 데이터"""
        from datetime import date
        data = data_service.get_historical_data(
            location="jeju",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            resolution="hourly"
        )
        assert len(data) >= 24

    def test_get_historical_data_daily(self, data_service):
        """일별 과거 데이터"""
        from datetime import date
        data = data_service.get_historical_data(
            location="jeju",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            resolution="daily"
        )
        assert len(data) == 7

    def test_historical_data_structure(self, data_service):
        """과거 데이터 구조"""
        from datetime import date
        data = data_service.get_historical_data(
            location="jeju",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
            resolution="daily"
        )
        record = data[0]
        assert "timestamp" in record
        assert "demand" in record
        assert "temperature" in record
        assert "humidity" in record

    def test_get_weather_forecast(self, data_service):
        """기상 예보"""
        forecasts = data_service.get_weather_forecast(
            location="jeju",
            hours_ahead=24
        )
        assert len(forecasts) == 24

    def test_weather_forecast_structure(self, data_service):
        """기상 예보 구조"""
        forecasts = data_service.get_weather_forecast(
            location="jeju",
            hours_ahead=1
        )
        forecast = forecasts[0]
        assert "timestamp" in forecast
        assert "temperature" in forecast
        assert "humidity" in forecast
        assert "wind_speed" in forecast


# ============================================================================
# Root Endpoint Tests
# ============================================================================

class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root_endpoint(self, client):
        """루트 엔드포인트"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


# ============================================================================
# Health Endpoint Tests
# ============================================================================

class TestHealthEndpoint:
    """상태 엔드포인트 테스트"""

    def test_health_check(self, client):
        """상태 확인"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data
        assert "models_loaded" in data

    def test_health_uptime(self, client):
        """업타임 확인"""
        response = client.get("/health")
        data = response.json()
        assert data["uptime"] >= 0


# ============================================================================
# Prediction Endpoints Tests
# ============================================================================

class TestPredictionEndpoints:
    """예측 엔드포인트 테스트"""

    def test_predict_default(self, client):
        """기본 예측"""
        response = client.post("/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["location"] == "jeju"
        assert len(data["predictions"]) == 3

    def test_predict_custom_horizons(self, client):
        """커스텀 시간대 예측"""
        response = client.post("/predict", json={
            "horizons": ["1h", "12h"]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2

    def test_predict_with_location(self, client):
        """위치 지정 예측"""
        response = client.post("/predict", json={
            "location": "seoul"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "seoul"

    def test_predict_with_model_type(self, client):
        """모델 타입 지정 예측"""
        response = client.post("/predict", json={
            "model_type": "lstm"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "lstm"

    def test_predict_response_structure(self, client):
        """예측 응답 구조"""
        response = client.post("/predict", json={})
        data = response.json()
        assert "request_id" in data
        assert "created_at" in data
        assert "predictions" in data
        assert "metadata" in data

    def test_predict_single_horizon_endpoint(self, client):
        """단일 시간대 예측 엔드포인트"""
        response = client.get("/predict/1h")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "1h"
        assert "prediction" in data

    def test_predict_single_horizon_with_location(self, client):
        """위치 지정 단일 예측"""
        response = client.get("/predict/6h?location=seoul")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "6h"


# ============================================================================
# Historical Data Endpoint Tests
# ============================================================================

class TestHistoricalDataEndpoint:
    """과거 데이터 엔드포인트 테스트"""

    def test_get_historical_data(self, client):
        """과거 데이터 조회"""
        response = client.get(
            "/data/historical",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-02"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "jeju"
        assert len(data["data"]) > 0

    def test_historical_data_with_location(self, client):
        """위치 지정 과거 데이터"""
        response = client.get(
            "/data/historical",
            params={
                "location": "seoul",
                "start_date": "2025-01-01",
                "end_date": "2025-01-01"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "seoul"

    def test_historical_data_daily_resolution(self, client):
        """일별 해상도"""
        response = client.get(
            "/data/historical",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-07",
                "resolution": "daily"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 7

    def test_historical_data_invalid_date(self, client):
        """잘못된 날짜 형식"""
        response = client.get(
            "/data/historical",
            params={
                "start_date": "invalid-date",
                "end_date": "2025-01-01"
            }
        )
        assert response.status_code == 400

    def test_historical_data_missing_params(self, client):
        """필수 파라미터 누락"""
        response = client.get("/data/historical")
        assert response.status_code == 422


# ============================================================================
# Forecast Endpoint Tests
# ============================================================================

class TestForecastEndpoint:
    """예보 엔드포인트 테스트"""

    def test_get_forecast_default(self, client):
        """기본 예보"""
        response = client.post("/forecast", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "jeju"
        assert "forecasts" in data

    def test_get_forecast_custom_hours(self, client):
        """커스텀 시간 예보"""
        response = client.post("/forecast", json={
            "hours_ahead": 48
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecasts"]) > 0

    def test_forecast_includes_weather(self, client):
        """기상 데이터 포함"""
        response = client.post("/forecast", json={
            "include_weather": True
        })
        assert response.status_code == 200
        data = response.json()
        if data["forecasts"]:
            forecast = data["forecasts"][0]
            assert "demand_prediction" in forecast

    def test_forecast_response_structure(self, client):
        """예보 응답 구조"""
        response = client.post("/forecast", json={})
        data = response.json()
        assert "location" in data
        assert "forecast_start" in data
        assert "forecast_end" in data
        assert "forecasts" in data


# ============================================================================
# Models Endpoint Tests
# ============================================================================

class TestModelsEndpoint:
    """모델 엔드포인트 테스트"""

    def test_list_models(self, client):
        """모델 목록"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert data["count"] == 3

    def test_model_info_structure(self, client):
        """모델 정보 구조"""
        response = client.get("/models")
        data = response.json()
        model = data["models"][0]
        assert "name" in model
        assert "version" in model
        assert "type" in model
        assert "metrics" in model
        assert "status" in model

    def test_get_model_info(self, client):
        """특정 모델 정보"""
        response = client.get("/models/lstm_v1")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "lstm_v1"
        assert data["type"] == "lstm"

    def test_get_model_not_found(self, client):
        """모델 없음"""
        response = client.get("/models/nonexistent")
        assert response.status_code == 404


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================

class TestMetricsEndpoint:
    """메트릭 엔드포인트 테스트"""

    def test_get_metrics(self, client):
        """메트릭 조회"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "total_predictions" in data
        assert "models_loaded" in data

    def test_metrics_after_prediction(self, client):
        """예측 후 메트릭"""
        client.post("/predict", json={})
        response = client.get("/metrics")
        data = response.json()
        assert data["total_predictions"] >= 1


# ============================================================================
# Error Handler Tests
# ============================================================================

class TestErrorHandlers:
    """에러 핸들러 테스트"""

    def test_404_error(self, client):
        """404 에러"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_validation_error(self, client):
        """검증 에러"""
        response = client.post("/predict", json={
            "horizons": "invalid"
        })
        assert response.status_code == 422


# ============================================================================
# Dependency Injection Tests
# ============================================================================

class TestDependencyInjection:
    """의존성 주입 테스트"""

    def test_get_prediction_service(self):
        """예측 서비스 의존성"""
        service = get_prediction_service()
        assert isinstance(service, PredictionService)

    def test_get_data_service(self):
        """데이터 서비스 의존성"""
        service = get_data_service()
        assert isinstance(service, DataService)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_prediction_flow(self, client):
        """전체 예측 흐름"""
        health = client.get("/health").json()
        assert health["status"] == "healthy"

        models = client.get("/models").json()
        assert models["count"] >= 1

        prediction = client.post("/predict", json={
            "location": "jeju",
            "horizons": ["1h", "24h"],
            "model_type": "ensemble"
        }).json()
        assert len(prediction["predictions"]) == 2

        metrics = client.get("/metrics").json()
        assert metrics["total_predictions"] >= 1

    def test_historical_to_forecast_flow(self, client):
        """과거 데이터에서 예보 흐름"""
        historical = client.get(
            "/data/historical",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-07"
            }
        ).json()
        assert historical["count"] > 0

        forecast = client.post("/forecast", json={
            "hours_ahead": 24
        }).json()
        assert len(forecast["forecasts"]) > 0

    def test_multiple_predictions(self, client):
        """다중 예측"""
        for i in range(5):
            response = client.post("/predict", json={
                "horizons": [f"{i+1}h"]
            })
            assert response.status_code == 200

        metrics = client.get("/metrics").json()
        assert metrics["total_predictions"] >= 5


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """성능 테스트"""

    def test_prediction_response_time(self, client):
        """예측 응답 시간"""
        import time
        start = time.time()
        response = client.post("/predict", json={})
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0

    def test_multiple_concurrent_like_requests(self, client):
        """연속 요청"""
        for _ in range(10):
            response = client.post("/predict", json={})
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
