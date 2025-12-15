"""
API 테스트
==========

FastAPI 엔드포인트 테스트
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_predictor():
    """Mock ProductionPredictor"""
    with patch('api.service.ProductionPredictor') as MockPredictor:
        mock = MockPredictor.return_value
        mock.load_models.return_value = None
        mock.predict_demand_only.return_value = 850.5
        mock.predict_weather_full.return_value = 865.3
        mock.predict_conditional.return_value = Mock(
            timestamp=datetime.now(),
            predicted_demand=855.0,
            model_used="conditional_soft (w=0.3)",
            context={
                "is_winter": True,
                "month": 1,
                "weather_weight": 0.3,
                "pred_demand_only": 850.5
            }
        )
        mock.predict_batch.return_value = Mock(
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(10)],
            predictions=[800 + i * 10 for i in range(10)],
            model_used="demand_only"
        )
        mock.model_demand = Mock()
        mock.model_weather = Mock()
        mock.config_demand = {
            'model_config': {'model_type': 'LSTM', 'hidden_size': 128, 'num_layers': 2},
            'n_features': 25,
            'training_config': {'seq_length': 168}
        }
        mock.config_weather = {
            'model_config': {'model_type': 'LSTM', 'hidden_size': 128, 'num_layers': 2},
            'n_features': 35,
            'training_config': {'seq_length': 168}
        }
        mock.device = "cpu"

        yield mock


@pytest.fixture
def mock_service(mock_predictor):
    """Mock PredictionService"""
    with patch('api.main.get_prediction_service') as mock_get_service:
        service = Mock()
        service.is_ready.return_value = True
        service.get_uptime.return_value = 3600.0
        service.get_device.return_value = "cpu"
        service.predictor = mock_predictor
        service.get_model_info.return_value = [
            {
                "name": "demand_only",
                "type": "LSTM",
                "n_features": 25,
                "seq_length": 168,
                "hidden_size": 128,
                "num_layers": 2,
                "status": "loaded"
            }
        ]

        mock_get_service.return_value = service
        yield service


@pytest.fixture
def client(mock_service):
    """Test client with mocked service"""
    # Mock the lifespan to prevent actual model loading
    with patch('api.main.initialize_service'):
        from api.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sample_data():
    """샘플 시계열 데이터 (168개)"""
    base_time = datetime(2024, 1, 8, 0, 0, 0)
    return [
        {
            "datetime": (base_time + timedelta(hours=i)).isoformat(),
            "power_demand": 750.0 + (i % 24) * 10,
            "기온": 5.0 + (i % 24) * 0.5,
            "습도": 60.0 + (i % 24),
            "풍속": 3.0,
            "강수량": 0.0
        }
        for i in range(168)
    ]


# ============================================================
# Root & Health Check Tests
# ============================================================

class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root_returns_api_info(self, client):
        """루트 엔드포인트가 API 정보를 반환하는지 확인"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestHealthCheck:
    """헬스체크 테스트"""

    def test_health_check_healthy(self, client, mock_service):
        """정상 상태 헬스체크"""
        mock_service.is_ready.return_value = True

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert "version" in data
        assert "device" in data
        assert "uptime_seconds" in data

    def test_health_check_unhealthy(self, client, mock_service):
        """비정상 상태 헬스체크"""
        mock_service.is_ready.return_value = False

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["models_loaded"] is False


class TestModelsEndpoint:
    """모델 정보 엔드포인트 테스트"""

    def test_get_models_info(self, client, mock_service):
        """모델 정보 조회"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default_model" in data
        assert len(data["models"]) > 0

    def test_get_models_service_unavailable(self, client, mock_service):
        """서비스 불가 시 모델 정보 조회"""
        mock_service.is_ready.return_value = False

        response = client.get("/models")

        assert response.status_code == 503


# ============================================================
# Prediction Tests
# ============================================================

class TestPredictEndpoint:
    """단일 예측 엔드포인트 테스트"""

    def test_predict_demand_only(self, client, mock_service, sample_data):
        """demand_only 모델 예측"""
        mock_service.predict.return_value = Mock(
            success=True,
            prediction=850.5,
            model_used="demand_only",
            timestamp=datetime.now(),
            processing_time_ms=45.2
        )

        response = client.post(
            "/predict",
            json={
                "data": sample_data,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert data["model_used"] == "demand_only"
        assert "processing_time_ms" in data

    def test_predict_insufficient_data(self, client, mock_service):
        """데이터 부족 시 에러"""
        short_data = [
            {
                "datetime": datetime.now().isoformat(),
                "power_demand": 750.0
            }
            for _ in range(50)  # 168 미만
        ]

        response = client.post(
            "/predict",
            json={
                "data": short_data,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_model_type(self, client, mock_service, sample_data):
        """잘못된 모델 타입"""
        response = client.post(
            "/predict",
            json={
                "data": sample_data,
                "model_type": "invalid_model"
            }
        )

        assert response.status_code == 422


class TestConditionalPredictEndpoint:
    """조건부 예측 엔드포인트 테스트"""

    def test_conditional_predict_soft(self, client, mock_service, sample_data):
        """soft 모드 조건부 예측"""
        mock_service.predict_conditional.return_value = Mock(
            success=True,
            prediction=855.0,
            model_used="conditional_soft (w=0.3)",
            timestamp=datetime.now(),
            context={
                "is_winter": True,
                "month": 1,
                "weather_weight": 0.3,
                "pred_demand_only": 850.5
            },
            processing_time_ms=78.5
        )

        response = client.post(
            "/predict/conditional",
            json={
                "data": sample_data,
                "mode": "soft"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "context" in data
        assert "is_winter" in data["context"]

    def test_conditional_predict_hard(self, client, mock_service, sample_data):
        """hard 모드 조건부 예측"""
        mock_service.predict_conditional.return_value = Mock(
            success=True,
            prediction=865.0,
            model_used="weather_full",
            timestamp=datetime.now(),
            context={
                "is_winter": True,
                "month": 1,
                "weather_weight": 1.0,
                "pred_demand_only": 850.5
            },
            processing_time_ms=85.0
        )

        response = client.post(
            "/predict/conditional",
            json={
                "data": sample_data,
                "mode": "hard"
            }
        )

        assert response.status_code == 200


class TestBatchPredictEndpoint:
    """배치 예측 엔드포인트 테스트"""

    def test_batch_predict(self, client, mock_service, sample_data):
        """배치 예측"""
        mock_service.predict_batch.return_value = Mock(
            success=True,
            predictions=[
                {"timestamp": "2024-01-15T00:00:00", "prediction": 720.5},
                {"timestamp": "2024-01-15T01:00:00", "prediction": 715.2}
            ],
            model_used="demand_only",
            total_predictions=2,
            statistics={
                "mean": 717.85,
                "std": 3.75,
                "min": 715.2,
                "max": 720.5
            },
            processing_time_ms=150.0
        )

        response = client.post(
            "/predict/batch",
            json={
                "data": sample_data,
                "model_type": "demand_only",
                "step": 1
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data
        assert "statistics" in data
        assert "total_predictions" in data

    def test_batch_predict_with_step(self, client, mock_service, sample_data):
        """스텝이 있는 배치 예측"""
        mock_service.predict_batch.return_value = Mock(
            success=True,
            predictions=[
                {"timestamp": "2024-01-15T00:00:00", "prediction": 720.5}
            ],
            model_used="demand_only",
            total_predictions=1,
            statistics={"mean": 720.5, "std": 0.0, "min": 720.5, "max": 720.5},
            processing_time_ms=50.0
        )

        response = client.post(
            "/predict/batch",
            json={
                "data": sample_data,
                "model_type": "demand_only",
                "step": 24  # 24시간 간격
            }
        )

        assert response.status_code == 200

    def test_batch_predict_invalid_step(self, client, mock_service, sample_data):
        """잘못된 스텝 값"""
        response = client.post(
            "/predict/batch",
            json={
                "data": sample_data,
                "model_type": "demand_only",
                "step": 100  # 너무 큰 값
            }
        )

        assert response.status_code == 422


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """에러 처리 테스트"""

    def test_service_unavailable(self, client, mock_service):
        """서비스 불가"""
        mock_service.is_ready.return_value = False

        response = client.post(
            "/predict",
            json={
                "data": [{"datetime": "2024-01-01T00:00:00", "power_demand": 750.0}] * 168,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 503

    def test_prediction_error(self, client, mock_service, sample_data):
        """예측 중 에러"""
        mock_service.predict.side_effect = ValueError("Invalid input data")

        response = client.post(
            "/predict",
            json={
                "data": sample_data,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 400

    def test_internal_error(self, client, mock_service, sample_data):
        """내부 서버 에러"""
        mock_service.predict.side_effect = RuntimeError("Unexpected error")

        response = client.post(
            "/predict",
            json={
                "data": sample_data,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 500


# ============================================================
# Schema Validation Tests
# ============================================================

class TestSchemaValidation:
    """스키마 검증 테스트"""

    def test_missing_required_field(self, client, mock_service):
        """필수 필드 누락"""
        response = client.post(
            "/predict",
            json={
                "model_type": "demand_only"
                # data 필드 누락
            }
        )

        assert response.status_code == 422

    def test_invalid_datetime_format(self, client, mock_service):
        """잘못된 datetime 형식"""
        response = client.post(
            "/predict",
            json={
                "data": [
                    {"datetime": "invalid-date", "power_demand": 750.0}
                ] * 168,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 422

    def test_negative_power_demand(self, client, mock_service):
        """음수 전력 수요 (유효한 입력으로 처리)"""
        # 음수도 유효한 입력으로 처리 (모델이 처리)
        data = [
            {
                "datetime": (datetime.now() + timedelta(hours=i)).isoformat(),
                "power_demand": -100.0  # 음수
            }
            for i in range(168)
        ]

        mock_service.predict.return_value = Mock(
            success=True,
            prediction=0.0,
            model_used="demand_only",
            timestamp=datetime.now(),
            processing_time_ms=50.0
        )

        response = client.post(
            "/predict",
            json={
                "data": data,
                "model_type": "demand_only"
            }
        )

        # 음수도 허용 (검증은 모델 레벨에서)
        assert response.status_code == 200


# ============================================================
# CORS Tests
# ============================================================

class TestCORS:
    """CORS 테스트"""

    def test_cors_headers(self, client):
        """CORS 헤더 확인"""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # FastAPI의 기본 CORS 처리
        assert response.status_code in [200, 400]


# ============================================================
# Performance Tests
# ============================================================

class TestPerformance:
    """성능 테스트"""

    def test_response_includes_processing_time(self, client, mock_service, sample_data):
        """응답에 처리 시간 포함"""
        mock_service.predict.return_value = Mock(
            success=True,
            prediction=850.5,
            model_used="demand_only",
            timestamp=datetime.now(),
            processing_time_ms=45.2
        )

        response = client.post(
            "/predict",
            json={
                "data": sample_data,
                "model_type": "demand_only"
            }
        )

        assert response.status_code == 200
        # X-Process-Time 헤더 확인
        assert "X-Process-Time" in response.headers
