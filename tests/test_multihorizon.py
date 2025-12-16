"""
Multi-horizon 예측 테스트
"""

import pytest
import torch
import numpy as np
import pandas as pd


class TestPredictionStrategy:
    """PredictionStrategy 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.models.multihorizon import PredictionStrategy

        assert PredictionStrategy.RECURSIVE.value == 'recursive'
        assert PredictionStrategy.DIRECT.value == 'direct'
        assert PredictionStrategy.MULTI_OUTPUT.value == 'multi_output'


class TestHorizonConfig:
    """HorizonConfig 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import HorizonConfig, PredictionStrategy

        config = HorizonConfig(
            horizons=[1, 6, 12, 24],
            strategy=PredictionStrategy.MULTI_OUTPUT
        )

        assert len(config.horizons) == 4
        assert config.strategy == PredictionStrategy.MULTI_OUTPUT


class TestHorizonPrediction:
    """HorizonPrediction 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import HorizonPrediction

        pred = HorizonPrediction(
            horizon=6,
            prediction=np.array([100, 105, 110]),
            confidence=0.95
        )

        assert pred.horizon == 6
        assert len(pred.prediction) == 3


class TestMultiHorizonResult:
    """MultiHorizonResult 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import MultiHorizonResult, HorizonPrediction

        predictions = {
            1: HorizonPrediction(1, np.array([100])),
            6: HorizonPrediction(6, np.array([105])),
        }

        result = MultiHorizonResult(predictions=predictions)

        assert 1 in result.predictions
        assert 6 in result.predictions

    def test_to_dataframe(self):
        """DataFrame 변환"""
        from src.models.multihorizon import MultiHorizonResult, HorizonPrediction

        predictions = {
            1: HorizonPrediction(1, np.array([100, 101, 102])),
            6: HorizonPrediction(6, np.array([105, 106, 107])),
        }

        result = MultiHorizonResult(predictions=predictions)
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'pred_h1' in df.columns
        assert 'pred_h6' in df.columns


class TestDirectMultiOutputNet:
    """DirectMultiOutputNet 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import DirectMultiOutputNet

        model = DirectMultiOutputNet(
            input_size=10,
            hidden_size=64,
            horizons=[1, 6, 12, 24]
        )

        assert model.input_size == 10
        assert model.num_horizons == 4

    def test_forward(self):
        """순전파"""
        from src.models.multihorizon import DirectMultiOutputNet

        model = DirectMultiOutputNet(
            input_size=10,
            hidden_size=64,
            horizons=[1, 6, 12]
        )

        x = torch.randn(16, 24, 10)
        output = model(x)

        assert isinstance(output, dict)
        assert 1 in output
        assert 6 in output
        assert 12 in output
        assert output[1].shape == (16, 1)


class TestRecursiveLSTM:
    """RecursiveLSTM 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import RecursiveLSTM

        model = RecursiveLSTM(input_size=10, hidden_size=64)

        assert model.input_size == 10

    def test_forward(self):
        """단일 스텝 예측"""
        from src.models.multihorizon import RecursiveLSTM

        model = RecursiveLSTM(input_size=10)
        x = torch.randn(16, 24, 10)

        pred = model(x)

        assert pred.shape == (16, 1)

    def test_predict_multi_step(self):
        """다중 스텝 예측"""
        from src.models.multihorizon import RecursiveLSTM

        model = RecursiveLSTM(input_size=10)
        x = torch.randn(16, 24, 10)

        predictions = model.predict_multi_step(x, horizons=[1, 3, 6])

        assert 1 in predictions
        assert 3 in predictions
        assert 6 in predictions


class TestHorizonSpecificModel:
    """HorizonSpecificModel 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import HorizonSpecificModel

        model = HorizonSpecificModel(
            input_size=20,
            hidden_size=64,
            horizons=[1, 6, 12]
        )

        assert len(model.horizons) == 3

    def test_forward_single(self):
        """단일 시간대 예측"""
        from src.models.multihorizon import HorizonSpecificModel

        model = HorizonSpecificModel(input_size=20, horizons=[1, 6])
        x = torch.randn(16, 20)

        pred = model(x, horizon=6)

        assert pred.shape == (16, 1)

    def test_forward_all(self):
        """모든 시간대 예측"""
        from src.models.multihorizon import HorizonSpecificModel

        model = HorizonSpecificModel(input_size=20, horizons=[1, 6, 12])
        x = torch.randn(16, 20)

        predictions = model(x)

        assert isinstance(predictions, dict)
        assert len(predictions) == 3


class TestAttentionMultiHorizon:
    """AttentionMultiHorizon 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import AttentionMultiHorizon

        model = AttentionMultiHorizon(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            horizons=[1, 6, 12]
        )

        assert model.input_size == 10
        assert len(model.horizons) == 3

    def test_forward(self):
        """순전파"""
        from src.models.multihorizon import AttentionMultiHorizon

        model = AttentionMultiHorizon(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            horizons=[1, 6]
        )

        x = torch.randn(16, 24, 10)
        output = model(x)

        assert 1 in output
        assert 6 in output
        assert output[1].shape == (16, 1)


class TestMultiHorizonPredictor:
    """MultiHorizonPredictor 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import (
            MultiHorizonPredictor, DirectMultiOutputNet, HorizonConfig
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6])
        config = HorizonConfig(horizons=[1, 6])

        predictor = MultiHorizonPredictor(model, config)

        assert predictor.model is model

    def test_predict(self):
        """예측"""
        from src.models.multihorizon import (
            MultiHorizonPredictor, DirectMultiOutputNet, HorizonConfig
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6, 12])
        config = HorizonConfig(horizons=[1, 6, 12])

        predictor = MultiHorizonPredictor(model, config)

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X)

        assert 1 in result.predictions
        assert 6 in result.predictions
        assert 12 in result.predictions

    def test_predict_with_intervals(self):
        """신뢰 구간 포함 예측"""
        from src.models.multihorizon import (
            MultiHorizonPredictor, DirectMultiOutputNet, HorizonConfig
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6])
        config = HorizonConfig(horizons=[1, 6])

        predictor = MultiHorizonPredictor(model, config)

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X, return_intervals=True)

        assert result.predictions[1].lower_bound is not None
        assert result.predictions[1].upper_bound is not None

    def test_predict_with_uncertainty(self):
        """불확실성 포함 예측"""
        from src.models.multihorizon import (
            MultiHorizonPredictor, DirectMultiOutputNet, HorizonConfig
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6], dropout=0.1)
        config = HorizonConfig(horizons=[1, 6])

        predictor = MultiHorizonPredictor(model, config)

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict_with_uncertainty(X, n_samples=20)

        assert result.predictions[1].lower_bound is not None


class TestHybridMultiHorizonPredictor:
    """HybridMultiHorizonPredictor 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import (
            HybridMultiHorizonPredictor, DirectMultiOutputNet
        )

        short_model = DirectMultiOutputNet(input_size=10, horizons=[1, 3, 6])
        long_model = DirectMultiOutputNet(input_size=10, horizons=[12, 24, 48])

        predictor = HybridMultiHorizonPredictor(
            short_term_model=short_model,
            long_term_model=long_model,
            short_term_horizons=[1, 3, 6],
            long_term_horizons=[12, 24, 48]
        )

        assert predictor is not None

    def test_predict(self):
        """예측"""
        from src.models.multihorizon import (
            HybridMultiHorizonPredictor, DirectMultiOutputNet
        )

        short_model = DirectMultiOutputNet(input_size=10, horizons=[1, 6])
        long_model = DirectMultiOutputNet(input_size=10, horizons=[12, 24])

        predictor = HybridMultiHorizonPredictor(
            short_term_model=short_model,
            long_term_model=long_model,
            short_term_horizons=[1, 6],
            long_term_horizons=[12, 24]
        )

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X)

        assert 1 in result.predictions
        assert 6 in result.predictions
        assert 12 in result.predictions
        assert 24 in result.predictions


class TestEnsembleMultiHorizonPredictor:
    """EnsembleMultiHorizonPredictor 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import (
            EnsembleMultiHorizonPredictor, DirectMultiOutputNet
        )

        models = [
            DirectMultiOutputNet(input_size=10, horizons=[1, 6]),
            DirectMultiOutputNet(input_size=10, horizons=[1, 6]),
        ]

        predictor = EnsembleMultiHorizonPredictor(
            models=models,
            horizons=[1, 6]
        )

        assert len(predictor.models) == 2

    def test_predict(self):
        """앙상블 예측"""
        from src.models.multihorizon import (
            EnsembleMultiHorizonPredictor, DirectMultiOutputNet
        )

        models = [
            DirectMultiOutputNet(input_size=10, horizons=[1, 6]),
            DirectMultiOutputNet(input_size=10, horizons=[1, 6]),
            DirectMultiOutputNet(input_size=10, horizons=[1, 6]),
        ]

        predictor = EnsembleMultiHorizonPredictor(
            models=models,
            horizons=[1, 6]
        )

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X)

        assert result.model_name == 'Ensemble'
        assert result.predictions[1].lower_bound is not None


class TestMultiHorizonTrainer:
    """MultiHorizonTrainer 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.multihorizon import (
            MultiHorizonTrainer, DirectMultiOutputNet
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6])
        trainer = MultiHorizonTrainer(model, horizons=[1, 6])

        assert trainer.model is model

    def test_train(self):
        """학습"""
        from src.models.multihorizon import (
            MultiHorizonTrainer, DirectMultiOutputNet
        )

        model = DirectMultiOutputNet(input_size=10, horizons=[1, 6])
        trainer = MultiHorizonTrainer(model, horizons=[1, 6])

        # 데이터 생성
        X = np.random.randn(100, 24, 10).astype(np.float32)
        y = {
            1: np.random.randn(100).astype(np.float32),
            6: np.random.randn(100).astype(np.float32)
        }

        history = trainer.train(
            X, y,
            epochs=5,
            batch_size=16
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_direct_multi_output_model(self):
        """직접 다중 출력 모델 생성"""
        from src.models.multihorizon import create_direct_multi_output_model

        model = create_direct_multi_output_model(
            input_size=10,
            horizons=[1, 6, 12, 24]
        )

        assert model is not None
        assert len(model.horizons) == 4

    def test_create_attention_multi_horizon_model(self):
        """Attention 모델 생성"""
        from src.models.multihorizon import create_attention_multi_horizon_model

        model = create_attention_multi_horizon_model(
            input_size=10,
            hidden_size=64,
            num_heads=4
        )

        assert model is not None

    def test_create_multi_horizon_predictor(self):
        """예측기 생성"""
        from src.models.multihorizon import (
            create_multi_horizon_predictor, DirectMultiOutputNet
        )

        model = DirectMultiOutputNet(input_size=10)
        predictor = create_multi_horizon_predictor(
            model,
            horizons=[1, 6, 12],
            strategy='multi_output'
        )

        assert predictor is not None


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self):
        """전체 워크플로우"""
        from src.models.multihorizon import (
            DirectMultiOutputNet, MultiHorizonTrainer,
            MultiHorizonPredictor, HorizonConfig
        )

        horizons = [1, 6, 12, 24]

        # 1. 모델 생성
        model = DirectMultiOutputNet(
            input_size=10,
            hidden_size=64,
            horizons=horizons
        )

        # 2. 학습 데이터 생성
        n_samples = 200
        X_train = np.random.randn(n_samples, 24, 10).astype(np.float32)
        y_train = {h: np.random.randn(n_samples).astype(np.float32) for h in horizons}

        # 3. 학습
        trainer = MultiHorizonTrainer(model, horizons)
        history = trainer.train(X_train, y_train, epochs=10, batch_size=32)

        assert history['train_loss'][-1] < history['train_loss'][0]

        # 4. 예측
        config = HorizonConfig(horizons=horizons)
        predictor = MultiHorizonPredictor(model, config)

        X_test = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X_test, return_intervals=True)

        # 검증
        assert len(result.predictions) == 4
        for h in horizons:
            assert h in result.predictions
            assert len(result.predictions[h].prediction) == 16

    def test_ensemble_workflow(self):
        """앙상블 워크플로우"""
        from src.models.multihorizon import (
            DirectMultiOutputNet, EnsembleMultiHorizonPredictor
        )

        horizons = [1, 6]

        # 여러 모델 생성
        models = [
            DirectMultiOutputNet(input_size=10, hidden_size=32, horizons=horizons),
            DirectMultiOutputNet(input_size=10, hidden_size=64, horizons=horizons),
            DirectMultiOutputNet(input_size=10, hidden_size=128, horizons=horizons),
        ]

        # 앙상블 예측기
        predictor = EnsembleMultiHorizonPredictor(
            models=models,
            horizons=horizons
        )

        X = np.random.randn(16, 24, 10).astype(np.float32)
        result = predictor.predict(X)

        # 검증
        assert result.model_name == 'Ensemble'
        for h in horizons:
            assert result.predictions[h].lower_bound is not None
            assert result.predictions[h].upper_bound is not None

