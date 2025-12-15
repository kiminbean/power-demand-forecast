"""
Tests for Inference Module
==========================

Production 모델 추론 모듈 테스트

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predict import (
    ProductionPredictor,
    PredictionResult,
    BatchPredictionResult,
    predict,
    predict_batch,
    get_predictor,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_dataframe():
    """테스트용 데이터프레임 생성"""
    n_hours = 500  # Enough for lag features + sequence
    dates = pd.date_range('2024-01-01', periods=n_hours, freq='h')

    np.random.seed(42)

    df = pd.DataFrame({
        'datetime': dates,
        'power_demand': 400 + 100 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 20, n_hours),
        '기온': 10 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 2, n_hours),
        '습도': 60 + 20 * np.random.random(n_hours),
        '풍속': 2 + 3 * np.random.random(n_hours),
    })

    return df


@pytest.fixture
def winter_dataframe():
    """겨울철 테스트용 데이터프레임"""
    n_hours = 500
    dates = pd.date_range('2024-01-15', periods=n_hours, freq='h')

    np.random.seed(42)

    df = pd.DataFrame({
        'datetime': dates,
        'power_demand': 450 + 80 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 15, n_hours),
        '기온': -5 + 5 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 2, n_hours),
        '습도': 50 + 20 * np.random.random(n_hours),
        '풍속': 3 + 4 * np.random.random(n_hours),
    })

    return df


@pytest.fixture
def summer_dataframe():
    """여름철 테스트용 데이터프레임"""
    n_hours = 500
    dates = pd.date_range('2024-07-15', periods=n_hours, freq='h')

    np.random.seed(42)

    df = pd.DataFrame({
        'datetime': dates,
        'power_demand': 500 + 100 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 25, n_hours),
        '기온': 28 + 5 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.normal(0, 2, n_hours),
        '습도': 70 + 15 * np.random.random(n_hours),
        '풍속': 2 + 2 * np.random.random(n_hours),
    })

    return df


@pytest.fixture
def mock_predictor():
    """모의 예측기"""
    predictor = ProductionPredictor()

    # Mock models
    predictor.model_demand = Mock()
    predictor.model_demand.eval = Mock()
    predictor.model_demand.return_value = torch.tensor([[450.0]])

    predictor.model_weather = Mock()
    predictor.model_weather.eval = Mock()
    predictor.model_weather.return_value = torch.tensor([[460.0]])

    # Mock scalers
    predictor.scaler_demand = Mock()
    predictor.scaler_demand.transform = Mock(return_value=np.random.randn(168, 17))
    predictor.scaler_demand.inverse_transform_target = Mock(return_value=np.array([450.0]))

    predictor.scaler_weather = Mock()
    predictor.scaler_weather.transform = Mock(return_value=np.random.randn(168, 18))
    predictor.scaler_weather.inverse_transform_target = Mock(return_value=np.array([460.0]))

    # Mock configs
    predictor.config_demand = {
        'features': ['power_demand'] + [f'feat_{i}' for i in range(16)],
        'n_features': 17,
        'training_config': {'seq_length': 168},
        'model_config': {'model_type': 'lstm', 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}
    }

    predictor.config_weather = {
        'features': ['power_demand'] + [f'feat_{i}' for i in range(17)],
        'n_features': 18,
        'training_config': {'seq_length': 168},
        'model_config': {'model_type': 'lstm', 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2}
    }

    predictor.is_loaded = True

    return predictor


# ============================================================
# Test PredictionResult
# ============================================================

class TestPredictionResult:
    """PredictionResult 테스트"""

    def test_creation(self):
        """기본 생성 테스트"""
        result = PredictionResult(
            timestamp=datetime.now(),
            predicted_demand=450.0,
            model_used="demand_only"
        )

        assert result.predicted_demand == 450.0
        assert result.model_used == "demand_only"
        assert result.confidence is None
        assert result.context is None

    def test_with_context(self):
        """컨텍스트 포함 테스트"""
        context = {'is_winter': True, 'weather_weight': 0.2}
        result = PredictionResult(
            timestamp=datetime.now(),
            predicted_demand=450.0,
            model_used="conditional_soft",
            context=context
        )

        assert result.context['is_winter'] is True
        assert result.context['weather_weight'] == 0.2


class TestBatchPredictionResult:
    """BatchPredictionResult 테스트"""

    def test_creation(self):
        """기본 생성 테스트"""
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(10)]
        predictions = np.array([450.0 + i * 10 for i in range(10)])

        result = BatchPredictionResult(
            timestamps=timestamps,
            predictions=predictions,
            model_used="demand_only"
        )

        assert len(result.timestamps) == 10
        assert len(result.predictions) == 10
        assert result.model_used == "demand_only"


# ============================================================
# Test ProductionPredictor Initialization
# ============================================================

class TestProductionPredictorInit:
    """ProductionPredictor 초기화 테스트"""

    def test_default_init(self):
        """기본 초기화"""
        predictor = ProductionPredictor()

        assert predictor.model_dir == PROJECT_ROOT / "models" / "production"
        assert predictor.is_loaded is False
        assert predictor.model_demand is None
        assert predictor.model_weather is None

    def test_custom_model_dir(self):
        """커스텀 모델 디렉토리"""
        predictor = ProductionPredictor(model_dir="/custom/path")

        assert predictor.model_dir == Path("/custom/path")

    def test_check_loaded_raises_error(self):
        """로드 전 사용시 에러"""
        predictor = ProductionPredictor()

        with pytest.raises(RuntimeError, match="Models not loaded"):
            predictor._check_loaded()


# ============================================================
# Test Feature Preparation
# ============================================================

class TestFeaturePreparation:
    """피처 준비 테스트"""

    def test_prepare_features_basic(self, sample_dataframe):
        """기본 피처 준비"""
        predictor = ProductionPredictor()

        df_prep = predictor._prepare_features(sample_dataframe, include_weather=False)

        # Time features should be added
        assert 'hour_sin' in df_prep.columns
        assert 'hour_cos' in df_prep.columns
        assert 'dayofweek_sin' in df_prep.columns
        assert 'is_weekend' in df_prep.columns

        # Lag features should be added
        assert 'demand_lag_1' in df_prep.columns
        assert 'demand_lag_24' in df_prep.columns

    def test_prepare_features_with_weather(self, sample_dataframe):
        """기상 피처 포함"""
        predictor = ProductionPredictor()

        df_prep = predictor._prepare_features(sample_dataframe, include_weather=True)

        # Weather features should be added
        assert 'THI' in df_prep.columns or '기온' in df_prep.columns

    def test_prepare_features_drops_na(self, sample_dataframe):
        """NaN 제거 확인"""
        predictor = ProductionPredictor()

        df_prep = predictor._prepare_features(sample_dataframe, include_weather=False)

        assert df_prep.isna().sum().sum() == 0


class TestGetSequence:
    """시퀀스 추출 테스트"""

    def test_get_sequence_shape(self, sample_dataframe):
        """시퀀스 형태 확인"""
        predictor = ProductionPredictor()

        df_prep = predictor._prepare_features(sample_dataframe, include_weather=False)
        features = [c for c in df_prep.columns]

        sequence = predictor._get_sequence(df_prep, features, seq_length=168)

        assert sequence.shape[0] == 168
        assert sequence.shape[1] == len(features)

    def test_get_sequence_insufficient_data(self, sample_dataframe):
        """데이터 부족시 에러"""
        predictor = ProductionPredictor()

        small_df = sample_dataframe.head(100)
        df_prep = predictor._prepare_features(small_df, include_weather=False)
        features = [c for c in df_prep.columns]

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor._get_sequence(df_prep, features, seq_length=168)


# ============================================================
# Test Predictions (with mocked models)
# ============================================================

class TestPredictDemandOnly:
    """demand_only 예측 테스트"""

    def test_predict_returns_float(self, mock_predictor, sample_dataframe):
        """float 반환 확인"""
        # Mock _prepare_features to return valid data
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=pd.date_range('2024-01-01', periods=200, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_demand_only(sample_dataframe)

            assert isinstance(result, float)
            assert result == 450.0

    def test_predict_tensor_return(self, mock_predictor, sample_dataframe):
        """텐서 반환 옵션"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=pd.date_range('2024-01-01', periods=200, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_demand_only(sample_dataframe, return_tensor=True)

            assert isinstance(result, torch.Tensor)


class TestPredictConditional:
    """Conditional 예측 테스트"""

    def test_winter_uses_weather_weight(self, mock_predictor, winter_dataframe):
        """겨울철 기상 가중치 적용"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 18),
                columns=['power_demand'] + [f'feat_{i}' for i in range(17)],
                index=pd.date_range('2024-01-15', periods=200, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_conditional(winter_dataframe, mode='soft')

            assert result.context['is_winter'] is True
            assert result.context['weather_weight'] > 0
            assert 'conditional_soft' in result.model_used

    def test_summer_uses_demand_only(self, mock_predictor, summer_dataframe):
        """여름철 demand_only 사용"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=pd.date_range('2024-07-15', periods=200, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_conditional(summer_dataframe, mode='soft')

            assert result.context['is_winter'] is False
            assert result.model_used == 'demand_only'

    def test_hard_mode_winter(self, mock_predictor, winter_dataframe):
        """Hard 모드 겨울철"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 18),
                columns=['power_demand'] + [f'feat_{i}' for i in range(17)],
                index=pd.date_range('2024-01-15', periods=200, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_conditional(winter_dataframe, mode='hard')

            assert result.model_used == 'weather_full'


class TestPredictBatch:
    """배치 예측 테스트"""

    def test_batch_returns_correct_type(self, mock_predictor, sample_dataframe):
        """올바른 타입 반환"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(500, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=pd.date_range('2024-01-01', periods=500, freq='h')
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_batch(sample_dataframe, model='demand_only', step=24)

            assert isinstance(result, BatchPredictionResult)
            assert isinstance(result.predictions, np.ndarray)
            assert len(result.timestamps) == len(result.predictions)

    def test_batch_step_affects_count(self, mock_predictor, sample_dataframe):
        """스텝이 결과 수에 영향"""
        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(500, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=pd.date_range('2024-01-01', periods=500, freq='h')
            )
            mock_prep.return_value = mock_df

            result_step1 = mock_predictor.predict_batch(sample_dataframe, step=1)
            result_step24 = mock_predictor.predict_batch(sample_dataframe, step=24)

            assert len(result_step1.predictions) > len(result_step24.predictions)


# ============================================================
# Test Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_get_predictor_singleton(self):
        """싱글톤 패턴 확인"""
        # Reset singleton
        import inference.predict as predict_module
        predict_module._predictor = None

        with patch.object(ProductionPredictor, 'load_models'):
            p1 = get_predictor()
            p2 = get_predictor()

            assert p1 is p2

    def test_predict_function_calls_predictor(self):
        """predict 함수가 예측기 호출"""
        # Import the actual module (not the function)
        from importlib import import_module
        predict_module = import_module('inference.predict')

        mock_pred = Mock()
        mock_pred.predict_conditional.return_value = PredictionResult(
            timestamp=datetime.now(),
            predicted_demand=450.0,
            model_used="conditional_soft"
        )

        with patch.object(predict_module, 'get_predictor', return_value=mock_pred):
            result = predict(pd.DataFrame(), model='conditional')

            mock_pred.predict_conditional.assert_called_once()


# ============================================================
# Test Season Detection
# ============================================================

class TestSeasonDetection:
    """계절 감지 테스트"""

    @pytest.mark.parametrize("month,is_winter", [
        (1, True),
        (2, True),
        (3, False),
        (6, False),
        (7, False),
        (12, True),
    ])
    def test_winter_detection(self, mock_predictor, month, is_winter):
        """월별 겨울 감지"""
        dates = pd.date_range(f'2024-{month:02d}-15', periods=200, freq='h')
        df = pd.DataFrame({
            'datetime': dates,
            'power_demand': np.random.randn(200) + 450
        })

        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 17),
                columns=['power_demand'] + [f'feat_{i}' for i in range(16)],
                index=dates
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_conditional(df, mode='soft')

            assert result.context['is_winter'] == is_winter


# ============================================================
# Test Weather Weight by Month
# ============================================================

class TestWeatherWeight:
    """기상 가중치 테스트"""

    @pytest.mark.parametrize("month,expected_weight", [
        (1, 0.3),   # Peak winter
        (2, 0.2),   # Late winter
        (12, 0.2),  # Early winter
        (7, 0.0),   # Summer
    ])
    def test_weather_weight_by_month(self, mock_predictor, month, expected_weight):
        """월별 기상 가중치"""
        dates = pd.date_range(f'2024-{month:02d}-15', periods=200, freq='h')
        df = pd.DataFrame({
            'datetime': dates,
            'power_demand': np.random.randn(200) + 450
        })

        with patch.object(mock_predictor, '_prepare_features') as mock_prep:
            mock_df = pd.DataFrame(
                np.random.randn(200, 18),
                columns=['power_demand'] + [f'feat_{i}' for i in range(17)],
                index=dates
            )
            mock_prep.return_value = mock_df

            result = mock_predictor.predict_conditional(df, mode='soft')

            assert result.context['weather_weight'] == expected_weight


# ============================================================
# Test Error Handling
# ============================================================

class TestErrorHandling:
    """에러 처리 테스트"""

    def test_predict_without_loading(self, sample_dataframe):
        """로드 없이 예측 시도"""
        predictor = ProductionPredictor()

        with pytest.raises(RuntimeError, match="Models not loaded"):
            predictor.predict_demand_only(sample_dataframe)

    def test_weather_model_not_loaded(self, sample_dataframe):
        """weather 모델 미로드 시"""
        predictor = ProductionPredictor()
        predictor.is_loaded = True
        predictor.model_demand = Mock()
        predictor.model_weather = None  # Not loaded

        with pytest.raises(RuntimeError, match="weather_full model not loaded"):
            predictor.predict_weather_full(sample_dataframe)


# ============================================================
# Integration Tests (requires actual models)
# ============================================================

@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "production" / "demand_only.pt").exists(),
    reason="Production models not available"
)
class TestIntegration:
    """통합 테스트 (실제 모델 필요)"""

    def test_full_prediction_pipeline(self, sample_dataframe):
        """전체 예측 파이프라인"""
        predictor = ProductionPredictor()
        predictor.load_models()

        result = predictor.predict_demand_only(sample_dataframe)

        assert isinstance(result, float)
        assert result > 0

    def test_conditional_prediction_pipeline(self, winter_dataframe):
        """Conditional 예측 파이프라인"""
        predictor = ProductionPredictor()
        predictor.load_models()

        result = predictor.predict_conditional(winter_dataframe, mode='soft')

        assert isinstance(result, PredictionResult)
        assert result.predicted_demand > 0
        assert result.context['is_winter'] is True

    def test_batch_prediction_pipeline(self, sample_dataframe):
        """배치 예측 파이프라인"""
        predictor = ProductionPredictor()
        predictor.load_models()

        result = predictor.predict_batch(sample_dataframe, model='demand_only', step=24)

        assert len(result.predictions) > 0
        assert all(p > 0 for p in result.predictions)
