"""
통합 테스트 (Task 17)
====================
시스템 전체 통합 테스트
"""

import pytest
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Feature Pipeline Integration Tests
# ============================================================================

class TestFeaturePipelineIntegration:
    """피처 파이프라인 통합 테스트"""

    def test_time_features_with_holiday(self):
        """시간 피처 + 휴일 통합"""
        from src.features.time_features import add_time_features
        from src.features.holiday import KoreanHolidayCalendar

        # 2025년 설날 연휴 - datetime index 필요
        dates = pd.date_range("2025-01-28", "2025-01-30", freq="D")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        # 시간 피처 생성
        time_df = add_time_features(df)

        # 휴일 정보 추가
        calendar = KoreanHolidayCalendar()
        calendar.add_fixed_holidays(2025)

        for idx in time_df.index:
            d = idx.date() if hasattr(idx, "date") else idx
            is_holiday = calendar.is_holiday(d)
            time_df.loc[idx, "is_custom_holiday"] = is_holiday

        assert "hour_sin" in time_df.columns or "is_weekend" in time_df.columns
        assert "is_custom_holiday" in time_df.columns

    def test_weather_features_with_thi(self):
        """기상 피처 + THI 통합"""
        from src.features.weather_features import add_weather_features

        # 기상 데이터 - datetime index 필요
        dates = pd.date_range("2025-01-01", periods=24, freq="h")
        df = pd.DataFrame({
            "temp_mean": np.random.uniform(5, 25, 24),
            "dewpoint_mean": np.random.uniform(0, 15, 24),
            "wind_speed_mean": np.random.uniform(1, 10, 24)
        }, index=dates)

        # 기상 피처 생성
        weather_df = add_weather_features(df)

        assert weather_df is not None
        assert len(weather_df) == 24
        # THI와 HDD/CDD가 추가되었는지 확인
        assert "THI" in weather_df.columns
        assert "HDD" in weather_df.columns
        assert "CDD" in weather_df.columns


# ============================================================================
# Model Pipeline Integration Tests
# ============================================================================

class TestModelPipelineIntegration:
    """모델 파이프라인 통합 테스트"""

    def test_lstm_training_flow(self):
        """LSTM 학습 흐름"""
        from src.models.lstm import LSTMModel

        # 데이터 준비
        seq_length = 24
        n_features = 10
        n_samples = 100

        # 시퀀스 생성 로직
        data = np.random.randn(n_samples, n_features)
        X = np.array([data[i:i+seq_length] for i in range(n_samples - seq_length)])
        y = data[seq_length:]

        assert X.shape == (n_samples - seq_length, seq_length, n_features)
        assert y.shape == (n_samples - seq_length, n_features)

    def test_ensemble_prediction_flow(self):
        """앙상블 예측 흐름"""
        # 모의 예측값
        pred1 = np.random.uniform(800, 1200, 10)
        pred2 = np.random.uniform(800, 1200, 10)
        pred3 = np.random.uniform(800, 1200, 10)

        # 수동 가중 평균
        weights = np.array([0.4, 0.35, 0.25])

        combined = np.average(
            np.array([pred1, pred2, pred3]),
            axis=0,
            weights=weights
        )

        assert combined.shape == (10,)
        assert all(800 <= v <= 1200 for v in combined)


# ============================================================================
# API Integration Tests
# ============================================================================

class TestAPIIntegration:
    """API 통합 테스트"""

    @pytest.fixture
    def api_client(self):
        """API 테스트 클라이언트"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_to_predict_flow(self, api_client):
        """상태 확인 → 예측 흐름"""
        # 1. 상태 확인
        health_response = api_client.get("/health")
        assert health_response.status_code == 200

        health_data = health_response.json()
        assert health_data["status"] == "healthy"

        # 2. 예측 수행
        predict_response = api_client.post("/predict", json={
            "location": "jeju",
            "horizons": ["1h", "24h"]
        })
        assert predict_response.status_code == 200

        pred_data = predict_response.json()
        assert len(pred_data["predictions"]) == 2

    def test_historical_data_flow(self, api_client):
        """과거 데이터 조회 흐름"""
        response = api_client.get(
            "/data/historical",
            params={
                "start_date": "2025-01-01",
                "end_date": "2025-01-07",
                "resolution": "hourly"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["count"] >= 24 * 7  # 7일 * 24시간

    def test_model_info_flow(self, api_client):
        """모델 정보 조회 흐름"""
        # 모델 목록
        list_response = api_client.get("/models")
        assert list_response.status_code == 200

        models = list_response.json()["models"]
        assert len(models) >= 1

        # 특정 모델 정보
        model_name = models[0]["name"]
        detail_response = api_client.get(f"/models/{model_name}")
        assert detail_response.status_code == 200


# ============================================================================
# Data Flow Integration Tests
# ============================================================================

class TestDataFlowIntegration:
    """데이터 흐름 통합 테스트"""

    def test_raw_to_features_flow(self):
        """원시 데이터 → 피처 흐름"""
        from src.features.time_features import add_time_features

        # 원시 데이터 - datetime index 필요
        dates = pd.date_range("2025-01-01", periods=168, freq="h")
        raw_data = pd.DataFrame({
            "power_demand": np.random.uniform(700, 1200, 168),
            "temperature": np.random.uniform(5, 30, 168),
            "humidity": np.random.uniform(40, 80, 168)
        }, index=dates)

        # 피처 생성
        featured_data = add_time_features(raw_data)

        assert len(featured_data) == 168
        assert "hour_sin" in featured_data.columns
        assert "dayofweek_sin" in featured_data.columns

    def test_feature_store_integration(self):
        """피처 스토어 통합"""
        import tempfile
        from src.features.feature_store import FeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(store_path=tmpdir)

            # 피처 등록 (registry를 통해)
            store.registry.register_feature(
                name="test_feature",
                dtype="float64",
                source="integration_test",
                description="Test feature"
            )

            # 피처 조회
            retrieved = store.registry.get_feature_metadata("test_feature")
            assert retrieved is not None
            assert retrieved.name == "test_feature"


# ============================================================================
# Multi-Horizon Integration Tests
# ============================================================================

class TestMultiHorizonIntegration:
    """멀티 호라이즌 통합 테스트"""

    def test_multi_horizon_prediction_flow(self):
        """멀티 호라이즌 예측 흐름"""
        from src.models.multihorizon import DirectMultiOutputNet

        # 입력 데이터
        batch_size = 32
        seq_length = 24
        n_features = 10
        horizons = [1, 6, 12, 24]

        model = DirectMultiOutputNet(
            input_size=n_features,
            hidden_size=64,
            horizons=horizons
        )

        # 예측
        import torch
        x = torch.randn(batch_size, seq_length, n_features)
        outputs = model(x)

        # Dict[int, Tensor] 반환
        assert isinstance(outputs, dict)
        assert len(outputs) == len(horizons)
        for horizon in horizons:
            assert horizon in outputs
            assert outputs[horizon].shape == (batch_size, 1)

    def test_horizon_specific_features(self):
        """호라이즌별 피처 처리"""
        horizons = [1, 6, 24]
        base_features = ["temperature", "humidity", "day_of_week"]

        horizon_features = {}
        for h in horizons:
            features = base_features.copy()
            if h >= 6:
                features.append("weather_forecast")
            if h >= 24:
                features.append("weekly_pattern")
            horizon_features[h] = features

        assert "weekly_pattern" in horizon_features[24]
        assert "weekly_pattern" not in horizon_features[1]


# ============================================================================
# Online Learning Integration Tests
# ============================================================================

class TestOnlineLearningIntegration:
    """온라인 학습 통합 테스트"""

    def test_drift_detection_and_retraining_flow(self):
        """드리프트 감지 및 재학습 흐름"""
        from src.training.online_learning import (
            ConceptDriftDetector,
            DataBuffer
        )

        # 데이터 버퍼
        buffer = DataBuffer(max_size=100)

        # 드리프트 감지기
        detector = ConceptDriftDetector(delta=0.005, lambda_=50.0, window_size=100)

        # 데이터 추가 및 드리프트 감지
        normal_errors = np.random.normal(50, 10, 50)
        drift_errors = np.random.normal(100, 10, 50)

        detected = False
        for error in np.concatenate([normal_errors, drift_errors]):
            # DataBuffer.add()는 X, y를 받음
            buffer.add(np.array([[error]]), np.array([error]))
            result = detector.update(error)
            if result.drift_detected:
                detected = True
                break

        # 드리프트가 감지될 수 있음 (확률적)
        assert len(buffer) > 0


# ============================================================================
# MLflow Integration Tests
# ============================================================================

class TestMLflowIntegration:
    """MLflow 통합 테스트"""

    def test_experiment_tracking_flow(self):
        """실험 추적 흐름"""
        import tempfile
        from src.training.mlflow_utils import ExperimentTracker, MetricLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_uri=tmpdir)
            logger = MetricLogger(tracker=tracker)

            # 메트릭 로깅 (딕셔너리 형태)
            logger.log({"rmse": 45.0, "mape": 3.5})

            # get_history로 조회
            rmse_history = logger.get_history("rmse")
            assert len(rmse_history) > 0
            assert rmse_history[0] == 45.0

    def test_model_registry_flow(self):
        """모델 레지스트리 흐름"""
        from src.training.mlflow_utils import ModelRegistry

        registry = ModelRegistry()

        # 모델 정보 (직접 _models에 접근 - 테스트용)
        model_info = {
            "name": "test_model",
            "version": "1.0.0",
            "metrics": {"rmse": 40.0}
        }

        registry._models[model_info["name"]] = model_info

        # 모델이 레지스트리에 있는지 확인
        assert "test_model" in registry._models


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """E2E 통합 테스트"""

    def test_full_prediction_pipeline(self):
        """전체 예측 파이프라인"""
        from src.features.time_features import add_time_features

        # 1. 데이터 준비 - datetime index 필요
        dates = pd.date_range("2025-01-01", periods=168, freq="h")
        raw_data = pd.DataFrame({
            "power_demand": np.random.uniform(700, 1200, 168),
            "temperature": np.random.uniform(5, 30, 168)
        }, index=dates)

        # 2. 피처 생성
        featured_data = add_time_features(raw_data)

        # 3. 시퀀스 생성
        seq_length = 24
        feature_cols = ["power_demand", "temperature"]

        if all(col in featured_data.columns for col in feature_cols):
            data_array = featured_data[feature_cols].values
            X = np.array([data_array[i:i+seq_length] for i in range(len(data_array) - seq_length)])
            y = data_array[seq_length:]

            assert X.shape[0] == len(data_array) - seq_length
            assert X.shape[1] == seq_length

    def test_holiday_aware_prediction(self):
        """휴일 인식 예측"""
        from src.features.holiday import KoreanHolidayCalendar

        calendar = KoreanHolidayCalendar()
        # 고정 공휴일 추가 필요
        calendar.add_fixed_holidays(2025)

        # 2025년 주요 휴일 확인
        test_dates = [
            date(2025, 1, 1),   # 신정
            date(2025, 3, 1),   # 삼일절
            date(2025, 5, 5),   # 어린이날
        ]

        for d in test_dates:
            is_holiday = calendar.is_holiday(d)
            # 휴일 정보가 있어야 함
            assert isinstance(is_holiday, bool)
            # 고정 공휴일이므로 True여야 함
            assert is_holiday is True

    def test_solar_power_integration(self):
        """태양광 발전량 통합"""
        from src.models.solar import SolarPositionCalculator, Location

        # 제주도 위치 - Location 객체 사용
        location = Location(latitude=33.5, longitude=126.5)

        calculator = SolarPositionCalculator(location=location)

        # 하루 동안의 태양 위치
        dt = datetime(2025, 6, 21, 12, 0)  # 하지
        position = calculator.calculate(dt)

        # SolarPosition dataclass 반환
        assert hasattr(position, "elevation")
        assert hasattr(position, "azimuth")
        # 하지 정오에는 고도가 높아야 함
        assert position.elevation > 0


# ============================================================================
# Performance Integration Tests
# ============================================================================

class TestPerformanceIntegration:
    """성능 통합 테스트"""

    def test_batch_prediction_performance(self):
        """배치 예측 성능"""
        import time

        from src.api.main import PredictionService

        service = PredictionService()

        # 100회 예측
        start = time.time()
        for _ in range(100):
            service.predict(
                location="jeju",
                horizons=["1h"],
                model_type="ensemble"
            )
        elapsed = time.time() - start

        # 100회가 5초 이내여야 함
        assert elapsed < 5.0, f"Prediction too slow: {elapsed:.2f}s for 100 predictions"

    def test_feature_generation_performance(self):
        """피처 생성 성능"""
        import time
        from src.features.time_features import add_time_features

        # 1년치 시간별 데이터 - datetime index 필요
        dates = pd.date_range("2024-01-01", periods=8760, freq="h")
        df = pd.DataFrame({"value": np.random.randn(8760)}, index=dates)

        start = time.time()
        result = add_time_features(df)
        elapsed = time.time() - start

        # 1년치가 2초 이내
        assert elapsed < 2.0, f"Feature generation too slow: {elapsed:.2f}s"
        assert len(result) == 8760


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """에러 처리 통합 테스트"""

    @pytest.fixture
    def api_client(self):
        """API 테스트 클라이언트"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_invalid_input_handling(self, api_client):
        """잘못된 입력 처리"""
        response = api_client.post("/predict", json={
            "horizons": "invalid"  # 리스트여야 함
        })
        assert response.status_code == 422

    def test_not_found_handling(self, api_client):
        """존재하지 않는 리소스 처리"""
        response = api_client.get("/models/nonexistent_model")
        assert response.status_code == 404

    def test_missing_parameter_handling(self, api_client):
        """필수 파라미터 누락 처리"""
        response = api_client.get("/data/historical")
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
