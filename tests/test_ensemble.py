"""
Ensemble 모델 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""
    def __init__(self, input_size: int = 10, output_size: int = 1, bias: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.bias = bias

    def forward(self, x):
        # x: (batch, seq, features)
        out = self.linear(x[:, -1, :])  # 마지막 시점만 사용
        return out + self.bias


class TestWeightedAverageEnsemble:
    """WeightedAverageEnsemble 테스트"""

    def test_creation(self):
        """앙상블 생성 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel(bias=i) for i in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        assert ensemble.n_models == 3
        assert len(ensemble.models) == 3

    def test_forward(self):
        """순전파 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        x = torch.randn(8, 24, 10)  # (batch, seq, features)
        output = ensemble(x)

        assert output.shape == (8, 1)

    def test_forward_with_individual(self):
        """개별 예측 반환 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        x = torch.randn(8, 24, 10)
        output, individual = ensemble(x, return_individual=True)

        assert output.shape == (8, 1)
        assert len(individual) == 3
        for pred in individual:
            assert pred.shape == (8, 1)

    def test_initial_weights(self):
        """초기 가중치 테스트 (균등 분포)"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        weights = ensemble.get_weights()
        expected = np.array([1/3, 1/3, 1/3])

        np.testing.assert_array_almost_equal(weights, expected, decimal=5)

    def test_set_weights(self):
        """가중치 설정 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        new_weights = [0.5, 0.3, 0.2]
        ensemble.set_weights(new_weights)

        # softmax 정규화 후 비교
        weights = torch.softmax(torch.tensor(new_weights), dim=0).numpy()
        result = ensemble.get_weights()

        np.testing.assert_array_almost_equal(result, weights, decimal=5)

    def test_fit(self):
        """가중치 학습 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        # 모의 예측 및 타겟
        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        result = ensemble.fit(val_preds, val_targets, method='scipy')

        assert 'optimal_weights' in result
        assert 'final_mse' in result
        assert ensemble._is_fitted

    def test_fit_grid(self):
        """그리드 서치 학습 테스트"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(2)]  # 2개로 테스트 (속도)
        ensemble = WeightedAverageEnsemble(models)

        val_preds = [torch.randn(50, 1) for _ in range(2)]
        val_targets = torch.randn(50, 1)

        result = ensemble.fit(val_preds, val_targets, method='grid')

        assert 'optimal_weights' in result


class TestStackingEnsemble:
    """StackingEnsemble 테스트"""

    def test_creation(self):
        """Stacking 앙상블 생성"""
        from src.models.ensemble import StackingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = StackingEnsemble(models, meta_learner_type='ridge')

        assert ensemble.n_models == 3
        assert ensemble.meta_learner is not None

    def test_forward(self):
        """Stacking 순전파"""
        from src.models.ensemble import StackingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = StackingEnsemble(models)

        x = torch.randn(8, 24, 10)
        output = ensemble(x)

        assert output.shape == (8,) or output.shape == (8, 1)

    def test_fit(self):
        """Meta-learner 학습"""
        from src.models.ensemble import StackingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = StackingEnsemble(models, meta_learner_type='mlp')

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        result = ensemble.fit(val_preds, val_targets, epochs=10)

        assert 'final_loss' in result
        assert ensemble._is_fitted

    def test_meta_learner_types(self):
        """다양한 Meta-learner 타입"""
        from src.models.ensemble import StackingEnsemble

        models = [SimpleModel() for _ in range(2)]

        for learner_type in ['ridge', 'mlp']:
            ensemble = StackingEnsemble(models, meta_learner_type=learner_type)
            assert ensemble.meta_learner is not None


class TestBlendingEnsemble:
    """BlendingEnsemble 테스트"""

    def test_creation(self):
        """Blending 앙상블 생성"""
        from src.models.ensemble import BlendingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = BlendingEnsemble(models, blend_ratio=0.3)

        assert ensemble.blend_ratio == 0.3

    def test_forward(self):
        """Blending 순전파"""
        from src.models.ensemble import BlendingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = BlendingEnsemble(models)

        x = torch.randn(8, 24, 10)
        output = ensemble(x)

        assert output.shape == (8,) or output.shape == (8, 1)

    def test_fit(self):
        """Blending 학습"""
        from src.models.ensemble import BlendingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = BlendingEnsemble(models)

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        result = ensemble.fit(val_preds, val_targets, epochs=10)

        assert 'final_loss' in result


class TestUncertaintyEnsemble:
    """UncertaintyEnsemble 테스트"""

    def test_creation(self):
        """불확실성 앙상블 생성"""
        from src.models.ensemble import UncertaintyEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = UncertaintyEnsemble(models)

        assert ensemble.n_models == 3

    def test_forward_without_uncertainty(self):
        """불확실성 없이 예측"""
        from src.models.ensemble import UncertaintyEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = UncertaintyEnsemble(models)

        x = torch.randn(8, 24, 10)
        output = ensemble(x, return_uncertainty=False)

        assert output.shape == (8, 1)

    def test_forward_with_uncertainty(self):
        """불확실성 포함 예측"""
        from src.models.ensemble import UncertaintyEnsemble

        models = [SimpleModel(bias=i) for i in range(3)]  # 다른 bias로 분산 생성
        ensemble = UncertaintyEnsemble(models)

        x = torch.randn(8, 24, 10)
        mean, std, lower, upper = ensemble(x, return_uncertainty=True)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)
        assert lower.shape == (8, 1)
        assert upper.shape == (8, 1)

        # 상한 >= 평균 >= 하한
        assert torch.all(upper >= mean)
        assert torch.all(mean >= lower)

    def test_fit(self):
        """불확실성 앙상블 학습"""
        from src.models.ensemble import UncertaintyEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = UncertaintyEnsemble(models)

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        result = ensemble.fit(val_preds, val_targets)

        assert 'optimal_weights' in result


class TestEnsembleOptimizer:
    """EnsembleOptimizer 테스트"""

    def test_optuna_optimization(self):
        """Optuna 최적화 테스트"""
        pytest.importorskip('optuna')

        from src.models.ensemble import WeightedAverageEnsemble, EnsembleOptimizer

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        optimizer = EnsembleOptimizer(ensemble)
        result = optimizer.optimize_with_optuna(val_preds, val_targets, n_trials=5)

        assert 'best_weights' in result
        assert len(result['best_weights']) == 3
        assert ensemble._is_fitted


class TestCreateEnsemble:
    """create_ensemble 팩토리 함수 테스트"""

    def test_weighted_average(self):
        """가중 평균 앙상블 생성"""
        from src.models.ensemble import create_ensemble, WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = create_ensemble(models, 'weighted_average')

        assert isinstance(ensemble, WeightedAverageEnsemble)

    def test_stacking(self):
        """Stacking 앙상블 생성"""
        from src.models.ensemble import create_ensemble, StackingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = create_ensemble(models, 'stacking', meta_learner_type='mlp')

        assert isinstance(ensemble, StackingEnsemble)

    def test_blending(self):
        """Blending 앙상블 생성"""
        from src.models.ensemble import create_ensemble, BlendingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = create_ensemble(models, 'blending', blend_ratio=0.25)

        assert isinstance(ensemble, BlendingEnsemble)

    def test_uncertainty(self):
        """불확실성 앙상블 생성"""
        from src.models.ensemble import create_ensemble, UncertaintyEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = create_ensemble(models, 'uncertainty')

        assert isinstance(ensemble, UncertaintyEnsemble)

    def test_invalid_type(self):
        """잘못된 타입 에러"""
        from src.models.ensemble import create_ensemble

        models = [SimpleModel() for _ in range(3)]

        with pytest.raises(ValueError):
            create_ensemble(models, 'invalid_type')


class TestEvaluateEnsemble:
    """evaluate_ensemble 함수 테스트"""

    def test_evaluation(self):
        """앙상블 평가"""
        from src.models.ensemble import WeightedAverageEnsemble, evaluate_ensemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        metrics = evaluate_ensemble(ensemble, val_preds, val_targets)

        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics

        # 값이 합리적인지 확인
        assert metrics['RMSE'] >= 0
        assert metrics['MAE'] >= 0
        assert metrics['MAPE'] >= 0


class TestCompareWithIndividual:
    """compare_with_individual 함수 테스트"""

    def test_comparison(self):
        """개별 모델 대비 비교"""
        from src.models.ensemble import WeightedAverageEnsemble, compare_with_individual

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)

        val_preds = [torch.randn(100, 1) for _ in range(3)]
        val_targets = torch.randn(100, 1)

        result = compare_with_individual(
            ensemble, val_preds, val_targets,
            model_names=['LSTM', 'BiLSTM', 'TFT']
        )

        assert 'individual' in result
        assert 'ensemble' in result
        assert 'improvement' in result

        assert 'LSTM' in result['individual']
        assert 'vs_best_individual' in result['improvement']


class TestEnsembleSaveLoad:
    """앙상블 저장/로드 테스트"""

    def test_save(self, tmp_path):
        """앙상블 저장"""
        from src.models.ensemble import WeightedAverageEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = WeightedAverageEnsemble(models)
        ensemble.set_weights([0.5, 0.3, 0.2])
        ensemble._is_fitted = True

        save_path = tmp_path / 'ensemble'
        ensemble.save(save_path)

        assert (save_path / 'ensemble_metadata.json').exists()
        assert (save_path / 'model_0.pt').exists()
        assert (save_path / 'model_1.pt').exists()
        assert (save_path / 'model_2.pt').exists()


class TestEnsembleIntegration:
    """앙상블 통합 테스트"""

    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        from src.models.ensemble import (
            WeightedAverageEnsemble,
            evaluate_ensemble,
            compare_with_individual
        )

        # 1. 모델 생성
        models = [SimpleModel(bias=i * 0.1) for i in range(3)]

        # 2. 앙상블 생성
        ensemble = WeightedAverageEnsemble(models)

        # 3. 모의 데이터
        val_preds = [torch.randn(100, 1) + i * 0.1 for i in range(3)]
        val_targets = torch.randn(100, 1)

        # 4. 학습
        fit_result = ensemble.fit(val_preds, val_targets)
        assert ensemble._is_fitted

        # 5. 평가
        metrics = evaluate_ensemble(ensemble, val_preds, val_targets)
        assert metrics['RMSE'] >= 0

        # 6. 비교
        comparison = compare_with_individual(
            ensemble, val_preds, val_targets,
            model_names=['M1', 'M2', 'M3']
        )
        assert 'improvement' in comparison

    def test_with_lstm_models(self):
        """LSTM 모델과의 통합"""
        from src.models.lstm import LSTMModel
        from src.models.ensemble import WeightedAverageEnsemble

        # LSTM 모델 생성
        lstm = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
        bilstm = LSTMModel(input_size=10, hidden_size=32, num_layers=1, bidirectional=True)

        ensemble = WeightedAverageEnsemble([lstm, bilstm])

        x = torch.randn(4, 24, 10)
        output = ensemble(x)

        assert output.shape == (4, 1)

    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        from src.models.ensemble import StackingEnsemble

        models = [SimpleModel() for _ in range(3)]
        ensemble = StackingEnsemble(models)

        x = torch.randn(8, 24, 10, requires_grad=True)
        output = ensemble(x)

        loss = output.sum()
        loss.backward()

        # meta_learner 그래디언트 확인
        for param in ensemble.meta_learner.parameters():
            assert param.grad is not None
