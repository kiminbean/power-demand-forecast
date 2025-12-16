"""
AutoML 모델 선택 시스템 테스트 (Task 19)
=========================================
모델 비교, 하이퍼파라미터 튜닝 테스트
"""

import pytest
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@pytest.fixture
def sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    n_samples = 200
    seq_length = 24
    n_features = 10

    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y = np.random.randn(n_samples, 1).astype(np.float32)

    return X, y


@pytest.fixture
def data_loaders(sample_data):
    """Create train/val data loaders"""
    X, y = sample_data

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# ============================================================================
# Model Type Tests
# ============================================================================

class TestModelType:
    """모델 타입 열거형 테스트"""

    def test_model_types_exist(self):
        """모델 타입 정의 확인"""
        from src.training.model_selection import ModelType

        assert ModelType.LSTM.value == 'lstm'
        assert ModelType.BILSTM.value == 'bilstm'
        assert ModelType.TFT.value == 'tft'
        assert ModelType.ENSEMBLE.value == 'ensemble'


class TestModelConfig:
    """모델 설정 테스트"""

    def test_model_config_creation(self):
        """ModelConfig 생성"""
        from src.training.model_selection import ModelConfig, ModelType

        config = ModelConfig(
            model_type=ModelType.LSTM,
            name='test_lstm',
            hyperparameters={'hidden_size': 64, 'num_layers': 2}
        )

        assert config.model_type == ModelType.LSTM
        assert config.name == 'test_lstm'
        assert config.hyperparameters['hidden_size'] == 64

    def test_model_config_to_dict(self):
        """ModelConfig 딕셔너리 변환"""
        from src.training.model_selection import ModelConfig, ModelType

        config = ModelConfig(
            model_type=ModelType.LSTM,
            name='test_lstm',
            hyperparameters={'hidden_size': 64}
        )

        result = config.to_dict()
        assert result['model_type'] == 'lstm'
        assert result['name'] == 'test_lstm'


class TestModelResult:
    """모델 결과 테스트"""

    def test_model_result_creation(self):
        """ModelResult 생성"""
        from src.training.model_selection import ModelResult, ModelConfig, ModelType

        config = ModelConfig(
            model_type=ModelType.LSTM,
            name='test',
            hyperparameters={}
        )

        result = ModelResult(
            config=config,
            metrics={'val_loss': 0.5, 'val_mse': 0.25},
            training_time=10.5,
            best_epoch=15
        )

        assert result.metrics['val_loss'] == 0.5
        assert result.training_time == 10.5
        assert result.best_epoch == 15

    def test_model_result_to_dict(self):
        """ModelResult 딕셔너리 변환"""
        from src.training.model_selection import ModelResult, ModelConfig, ModelType

        config = ModelConfig(
            model_type=ModelType.LSTM,
            name='test',
            hyperparameters={}
        )

        result = ModelResult(
            config=config,
            metrics={'val_loss': 0.5},
            training_time=10.0
        )

        result_dict = result.to_dict()
        assert 'config' in result_dict
        assert 'metrics' in result_dict
        assert 'training_time' in result_dict


# ============================================================================
# Search Space Tests
# ============================================================================

class TestSearchSpace:
    """탐색 공간 테스트"""

    def test_search_space_presets(self):
        """탐색 공간 프리셋 확인"""
        from src.training.model_selection import (
            LSTM_SEARCH_SPACE,
            TFT_SEARCH_SPACE,
            ENSEMBLE_SEARCH_SPACE
        )

        assert len(LSTM_SEARCH_SPACE) > 0
        assert len(TFT_SEARCH_SPACE) > 0
        assert len(ENSEMBLE_SEARCH_SPACE) > 0

    def test_get_search_space(self):
        """모델별 탐색 공간 가져오기"""
        from src.training.model_selection import get_search_space, ModelType

        lstm_space = get_search_space(ModelType.LSTM)
        tft_space = get_search_space(ModelType.TFT)

        assert len(lstm_space) > 0
        assert len(tft_space) > 0

    def test_search_space_types(self):
        """SearchSpace 타입별 동작"""
        from src.training.model_selection import SearchSpace

        # Categorical
        cat_space = SearchSpace('test', 'categorical', choices=[1, 2, 3])
        assert cat_space.choices == [1, 2, 3]

        # Int
        int_space = SearchSpace('test', 'int', low=1, high=10)
        assert int_space.low == 1
        assert int_space.high == 10

        # Float
        float_space = SearchSpace('test', 'float', low=0.0, high=1.0)
        assert float_space.low == 0.0
        assert float_space.high == 1.0


# ============================================================================
# Model Factory Tests
# ============================================================================

class TestModelFactory:
    """모델 팩토리 테스트"""

    def test_factory_creation(self, device):
        """팩토리 생성"""
        from src.training.model_selection import ModelFactory

        factory = ModelFactory(input_size=10, output_size=1, device=device)

        assert factory.input_size == 10
        assert factory.output_size == 1
        assert factory.device == device

    def test_create_lstm(self, device):
        """LSTM 모델 생성"""
        from src.training.model_selection import ModelFactory, ModelType

        factory = ModelFactory(input_size=10, output_size=1, device='cpu')

        model = factory.create(
            ModelType.LSTM,
            hidden_size=32,
            num_layers=2,
            dropout=0.1
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_create_bilstm(self, device):
        """BiLSTM 모델 생성"""
        from src.training.model_selection import ModelFactory, ModelType

        factory = ModelFactory(input_size=10, output_size=1, device='cpu')

        model = factory.create(
            ModelType.BILSTM,
            hidden_size=32,
            num_layers=1
        )

        assert model is not None


# ============================================================================
# Model Comparator Tests
# ============================================================================

class TestModelComparator:
    """모델 비교기 테스트"""

    def test_comparator_creation(self):
        """비교기 생성"""
        from src.training.model_selection import ModelComparator, ModelFactory, simple_train_fn

        factory = ModelFactory(input_size=10)
        comparator = ModelComparator(
            factory,
            simple_train_fn,
            primary_metric='val_loss',
            direction='minimize'
        )

        assert comparator.primary_metric == 'val_loss'
        assert comparator.direction == 'minimize'

    def test_add_models(self):
        """모델 추가"""
        from src.training.model_selection import (
            ModelComparator, ModelFactory, ModelConfig, ModelType, simple_train_fn
        )

        factory = ModelFactory(input_size=10)
        comparator = ModelComparator(factory, simple_train_fn)

        config1 = ModelConfig(ModelType.LSTM, 'lstm1', {'hidden_size': 32})
        config2 = ModelConfig(ModelType.LSTM, 'lstm2', {'hidden_size': 64})

        comparator.add_model(config1)
        comparator.add_models([config2])

        assert len(comparator._configs) == 2

    def test_compare_models(self, data_loaders):
        """모델 비교 실행"""
        from src.training.model_selection import (
            ModelComparator, ModelFactory, ModelConfig, ModelType, simple_train_fn
        )

        train_loader, val_loader = data_loaders
        factory = ModelFactory(input_size=10, device='cpu')
        comparator = ModelComparator(factory, simple_train_fn)

        comparator.add_model(
            ModelConfig(ModelType.LSTM, 'lstm_small', {'hidden_size': 16, 'num_layers': 1})
        )
        comparator.add_model(
            ModelConfig(ModelType.LSTM, 'lstm_medium', {'hidden_size': 32, 'num_layers': 1})
        )

        results = comparator.compare(
            train_loader, val_loader,
            epochs=2, patience=2, verbose=False
        )

        assert len(results) == 2
        assert all('val_loss' in r.metrics for r in results)

    def test_get_best_model(self, data_loaders):
        """최적 모델 가져오기"""
        from src.training.model_selection import (
            ModelComparator, ModelFactory, ModelConfig, ModelType, simple_train_fn
        )

        train_loader, val_loader = data_loaders
        factory = ModelFactory(input_size=10, device='cpu')
        comparator = ModelComparator(factory, simple_train_fn)

        comparator.add_model(
            ModelConfig(ModelType.LSTM, 'lstm1', {'hidden_size': 16, 'num_layers': 1})
        )

        comparator.compare(train_loader, val_loader, epochs=2, patience=2, verbose=False)

        best = comparator.get_best_model()
        assert best is not None
        assert best.config.name == 'lstm1'

    def test_comparison_table(self, data_loaders):
        """비교 테이블 생성"""
        from src.training.model_selection import (
            ModelComparator, ModelFactory, ModelConfig, ModelType, simple_train_fn
        )
        import pandas as pd

        train_loader, val_loader = data_loaders
        factory = ModelFactory(input_size=10, device='cpu')
        comparator = ModelComparator(factory, simple_train_fn)

        comparator.add_model(
            ModelConfig(ModelType.LSTM, 'lstm1', {'hidden_size': 16, 'num_layers': 1})
        )

        comparator.compare(train_loader, val_loader, epochs=2, patience=2, verbose=False)

        table = comparator.get_comparison_table()
        assert isinstance(table, pd.DataFrame)
        assert 'model_name' in table.columns
        assert 'val_loss' in table.columns


# ============================================================================
# AutoML Pipeline Tests
# ============================================================================

class TestAutoMLPipeline:
    """AutoML 파이프라인 테스트"""

    def test_pipeline_creation(self):
        """파이프라인 생성"""
        from src.training.model_selection import AutoMLPipeline, simple_train_fn

        pipeline = AutoMLPipeline(
            input_size=10,
            output_size=1,
            train_fn=simple_train_fn,
            device='cpu'
        )

        assert pipeline.input_size == 10
        assert pipeline.output_size == 1

    def test_create_automl_pipeline_factory(self):
        """팩토리 함수로 파이프라인 생성"""
        from src.training.model_selection import create_automl_pipeline

        pipeline = create_automl_pipeline(
            input_size=10,
            output_size=1,
            device='cpu'
        )

        assert pipeline is not None

    def test_pipeline_run(self, data_loaders):
        """파이프라인 실행"""
        from src.training.model_selection import (
            AutoMLPipeline, simple_train_fn, ModelConfig, ModelType
        )

        train_loader, val_loader = data_loaders

        pipeline = AutoMLPipeline(
            input_size=10,
            output_size=1,
            train_fn=simple_train_fn,
            device='cpu'
        )

        # 간단한 설정으로 빠르게 테스트
        configs = [
            ModelConfig(ModelType.LSTM, 'lstm_test', {'hidden_size': 16, 'num_layers': 1})
        ]

        best_model, best_params = pipeline.run(
            train_loader, val_loader,
            model_configs=configs,
            tune_best=False,  # 튜닝 스킵
            epochs=2, patience=2
        )

        assert best_model is not None
        assert isinstance(best_params, dict)

    def test_pipeline_comparison_table(self, data_loaders):
        """파이프라인 비교 테이블"""
        from src.training.model_selection import (
            AutoMLPipeline, simple_train_fn, ModelConfig, ModelType
        )
        import pandas as pd

        train_loader, val_loader = data_loaders

        pipeline = AutoMLPipeline(
            input_size=10,
            train_fn=simple_train_fn,
            device='cpu'
        )

        configs = [
            ModelConfig(ModelType.LSTM, 'lstm_test', {'hidden_size': 16, 'num_layers': 1})
        ]

        pipeline.run(
            train_loader, val_loader,
            model_configs=configs,
            tune_best=False,
            epochs=2, patience=2
        )

        table = pipeline.get_comparison_table()
        assert isinstance(table, pd.DataFrame)


# ============================================================================
# Simple Train Function Tests
# ============================================================================

class TestSimpleTrainFn:
    """간단한 학습 함수 테스트"""

    def test_simple_train_fn(self, data_loaders):
        """학습 함수 동작"""
        from src.training.model_selection import simple_train_fn
        from src.models.lstm import LSTMModel

        train_loader, val_loader = data_loaders

        model = LSTMModel(
            input_size=10,
            hidden_size=16,
            num_layers=1
        )

        metrics = simple_train_fn(
            model,
            train_loader,
            val_loader,
            epochs=3,
            patience=2,
            device='cpu'
        )

        assert 'val_loss' in metrics
        assert 'val_mse' in metrics
        assert 'val_mae' in metrics
        assert 'best_epoch' in metrics

    def test_simple_train_fn_early_stopping(self, data_loaders):
        """조기 종료 동작"""
        from src.training.model_selection import simple_train_fn
        from src.models.lstm import LSTMModel

        train_loader, val_loader = data_loaders

        model = LSTMModel(
            input_size=10,
            hidden_size=16,
            num_layers=1
        )

        metrics = simple_train_fn(
            model,
            train_loader,
            val_loader,
            epochs=100,  # 많은 에폭
            patience=3,  # 작은 patience
            device='cpu'
        )

        # 조기 종료되어야 함
        assert metrics['best_epoch'] < 100


# ============================================================================
# Save Results Tests
# ============================================================================

class TestSaveResults:
    """결과 저장 테스트"""

    def test_comparator_save_results(self, data_loaders):
        """비교기 결과 저장"""
        from src.training.model_selection import (
            ModelComparator, ModelFactory, ModelConfig, ModelType, simple_train_fn
        )

        train_loader, val_loader = data_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = ModelFactory(input_size=10, device='cpu')
            comparator = ModelComparator(factory, simple_train_fn)

            comparator.add_model(
                ModelConfig(ModelType.LSTM, 'lstm1', {'hidden_size': 16, 'num_layers': 1})
            )

            comparator.compare(
                train_loader, val_loader,
                epochs=2, patience=2, verbose=False
            )

            comparator.save_results(tmpdir)

            # 파일 생성 확인
            assert (Path(tmpdir) / 'comparison_results.json').exists()
            assert (Path(tmpdir) / 'comparison_table.csv').exists()


# ============================================================================
# Integration Tests
# ============================================================================

class TestAutoMLIntegration:
    """AutoML 통합 테스트"""

    def test_full_pipeline(self, data_loaders):
        """전체 파이프라인 통합"""
        from src.training.model_selection import (
            AutoMLPipeline, simple_train_fn, ModelConfig, ModelType
        )

        train_loader, val_loader = data_loaders

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AutoMLPipeline(
                input_size=10,
                output_size=1,
                train_fn=simple_train_fn,
                device='cpu'
            )

            configs = [
                ModelConfig(ModelType.LSTM, 'lstm_small', {'hidden_size': 16, 'num_layers': 1}),
                ModelConfig(ModelType.LSTM, 'lstm_large', {'hidden_size': 32, 'num_layers': 2}),
            ]

            best_model, best_params = pipeline.run(
                train_loader, val_loader,
                model_configs=configs,
                tune_best=False,
                output_dir=tmpdir,
                epochs=3, patience=2
            )

            # 결과 확인
            assert best_model is not None
            assert (Path(tmpdir) / 'automl_results.json').exists()
            assert (Path(tmpdir) / 'comparison' / 'comparison_results.json').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
