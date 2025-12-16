"""
Probabilistic Forecasting 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class SimpleDropoutModel(nn.Module):
    """테스트용 드롭아웃 모델"""
    def __init__(self, input_size: int = 10, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        out = self.fc1(x[:, -1, :])
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TestPredictionInterval:
    """PredictionInterval 테스트"""

    def test_creation(self):
        """예측 구간 생성"""
        from src.models.probabilistic import PredictionInterval

        interval = PredictionInterval(
            point_estimate=np.array([1.0, 2.0, 3.0]),
            lower=np.array([0.5, 1.5, 2.5]),
            upper=np.array([1.5, 2.5, 3.5]),
            confidence=0.9
        )

        assert interval.confidence == 0.9
        assert len(interval.point_estimate) == 3

    def test_with_std(self):
        """표준편차 포함"""
        from src.models.probabilistic import PredictionInterval

        interval = PredictionInterval(
            point_estimate=np.array([1.0]),
            lower=np.array([0.5]),
            upper=np.array([1.5]),
            confidence=0.9,
            std=np.array([0.25])
        )

        assert interval.std is not None


class TestMCDropout:
    """Monte Carlo Dropout 테스트"""

    def test_creation(self):
        """MC Dropout 생성"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel()
        mc = MCDropout(model, n_samples=10)

        assert mc.n_samples == 10

    def test_forward(self):
        """Forward pass"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel()
        mc = MCDropout(model, n_samples=10)

        x = torch.randn(4, 24, 10)
        output = mc.forward(x)

        assert output.shape == (4, 1)

    def test_predict_samples(self):
        """샘플 예측"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel()
        mc = MCDropout(model, n_samples=20)

        x = torch.randn(4, 24, 10)
        samples = mc.predict_samples(x)

        assert samples.shape == (20, 4, 1)

    def test_predict_with_uncertainty(self):
        """불확실성 포함 예측"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel(dropout=0.3)
        mc = MCDropout(model, n_samples=50)

        x = torch.randn(8, 24, 10)
        mean, std = mc.predict_with_uncertainty(x)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)
        # 드롭아웃으로 인한 분산이 있어야 함
        assert torch.any(std > 0)

    def test_predict_with_samples(self):
        """샘플과 함께 예측"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel()
        mc = MCDropout(model, n_samples=30)

        x = torch.randn(4, 24, 10)
        mean, std, samples = mc.predict_with_uncertainty(x, return_samples=True)

        assert samples.shape == (30, 4, 1)

    def test_predict_interval(self):
        """예측 구간"""
        from src.models.probabilistic import MCDropout

        model = SimpleDropoutModel()
        mc = MCDropout(model, n_samples=50)

        x = torch.randn(8, 24, 10)
        interval = mc.predict_interval(x, confidence=0.9)

        assert interval.confidence == 0.9
        assert len(interval.point_estimate) == 8
        assert len(interval.lower) == 8
        assert len(interval.upper) == 8
        # 상한 >= 하한
        assert np.all(interval.upper >= interval.lower)


class TestDeepEnsembleUncertainty:
    """Deep Ensemble 테스트"""

    def test_creation(self):
        """앙상블 생성"""
        from src.models.probabilistic import DeepEnsembleUncertainty

        models = [SimpleDropoutModel() for _ in range(5)]
        ensemble = DeepEnsembleUncertainty(models)

        assert ensemble.n_models == 5

    def test_predict_samples(self):
        """모델별 예측"""
        from src.models.probabilistic import DeepEnsembleUncertainty

        models = [SimpleDropoutModel() for _ in range(3)]
        ensemble = DeepEnsembleUncertainty(models)

        x = torch.randn(4, 24, 10)
        samples = ensemble.predict_samples(x)

        assert samples.shape == (3, 4, 1)

    def test_predict_with_uncertainty(self):
        """불확실성 예측"""
        from src.models.probabilistic import DeepEnsembleUncertainty

        # 다른 초기화로 다양성 확보
        models = [SimpleDropoutModel() for _ in range(5)]
        ensemble = DeepEnsembleUncertainty(models)

        x = torch.randn(8, 24, 10)
        mean, std = ensemble.predict_with_uncertainty(x)

        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)

    def test_aggregation_methods(self):
        """집계 방법"""
        from src.models.probabilistic import DeepEnsembleUncertainty

        models = [SimpleDropoutModel() for _ in range(3)]

        ensemble_mean = DeepEnsembleUncertainty(models, aggregation='mean')
        ensemble_median = DeepEnsembleUncertainty(models, aggregation='median')

        x = torch.randn(4, 24, 10)

        mean_pred, _ = ensemble_mean.predict_with_uncertainty(x)
        median_pred, _ = ensemble_median.predict_with_uncertainty(x)

        # 둘 다 유효한 예측
        assert mean_pred.shape == median_pred.shape

    def test_predict_interval(self):
        """예측 구간"""
        from src.models.probabilistic import DeepEnsembleUncertainty

        models = [SimpleDropoutModel() for _ in range(5)]
        ensemble = DeepEnsembleUncertainty(models)

        x = torch.randn(8, 24, 10)
        interval = ensemble.predict_interval(x, confidence=0.8)

        assert interval.confidence == 0.8
        assert np.all(interval.upper >= interval.lower)


class TestPinballLoss:
    """Pinball Loss 테스트"""

    def test_creation(self):
        """손실 함수 생성"""
        from src.models.probabilistic import PinballLoss

        loss = PinballLoss([0.1, 0.5, 0.9])
        assert len(loss.quantiles) == 3

    def test_forward_2d(self):
        """2D 입력 손실 계산"""
        from src.models.probabilistic import PinballLoss

        loss = PinballLoss([0.1, 0.5, 0.9])

        predictions = torch.randn(8, 3)
        targets = torch.randn(8)

        loss_value = loss(predictions, targets)

        assert loss_value.dim() == 0  # 스칼라
        assert loss_value >= 0

    def test_forward_3d(self):
        """3D 입력 손실 계산"""
        from src.models.probabilistic import PinballLoss

        loss = PinballLoss([0.1, 0.5, 0.9])

        predictions = torch.randn(8, 24, 3)
        targets = torch.randn(8, 24)

        loss_value = loss(predictions, targets)

        assert loss_value.dim() == 0

    def test_gradient_flow(self):
        """그래디언트 흐름"""
        from src.models.probabilistic import PinballLoss

        loss_fn = PinballLoss([0.1, 0.5, 0.9])

        predictions = torch.randn(8, 3, requires_grad=True)
        targets = torch.randn(8)

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None


class TestCalibrationMetrics:
    """Calibration 메트릭 테스트"""

    def test_coverage(self):
        """Coverage 계산"""
        from src.models.probabilistic import CalibrationMetrics

        lower = np.array([0, 1, 2, 3, 4])
        upper = np.array([2, 3, 4, 5, 6])
        actual = np.array([1, 2, 3, 4, 5])  # 모두 구간 내

        coverage = CalibrationMetrics.coverage(lower, upper, actual)
        assert coverage == 1.0

    def test_coverage_partial(self):
        """부분 Coverage"""
        from src.models.probabilistic import CalibrationMetrics

        lower = np.array([0, 1, 2, 3, 4])
        upper = np.array([1, 2, 3, 4, 5])
        actual = np.array([0.5, 1.5, 2.5, 10, 10])  # 3개만 구간 내

        coverage = CalibrationMetrics.coverage(lower, upper, actual)
        assert coverage == 0.6

    def test_interval_width(self):
        """구간 너비"""
        from src.models.probabilistic import CalibrationMetrics

        lower = np.array([0, 1, 2])
        upper = np.array([2, 3, 4])

        width = CalibrationMetrics.interval_width(lower, upper)
        assert width == 2.0

    def test_interval_score(self):
        """Interval Score"""
        from src.models.probabilistic import CalibrationMetrics

        lower = np.array([0, 1, 2])
        upper = np.array([2, 3, 4])
        actual = np.array([1, 2, 3])

        score = CalibrationMetrics.interval_score(lower, upper, actual, alpha=0.1)

        # 모두 구간 내이므로 패널티 없음
        assert score == 2.0  # 평균 구간 너비

    def test_evaluate(self):
        """종합 평가"""
        from src.models.probabilistic import CalibrationMetrics, PredictionInterval

        interval = PredictionInterval(
            point_estimate=np.array([1.0, 2.0, 3.0]),
            lower=np.array([0.5, 1.5, 2.5]),
            upper=np.array([1.5, 2.5, 3.5]),
            confidence=0.9
        )
        actual = np.array([1.0, 2.0, 3.0])

        metrics = CalibrationMetrics().evaluate(interval, actual)

        assert 'coverage' in metrics
        assert 'interval_width' in metrics
        assert 'interval_score' in metrics


class TestProbabilisticWrapper:
    """ProbabilisticWrapper 테스트"""

    def test_mc_dropout_wrapper(self):
        """MC Dropout 래퍼"""
        from src.models.probabilistic import ProbabilisticWrapper

        model = SimpleDropoutModel()
        wrapper = ProbabilisticWrapper(model, method='mc_dropout', n_samples=20)

        x = torch.randn(4, 24, 10)
        interval = wrapper.predict_interval(x, confidence=0.9)

        assert interval.confidence == 0.9

    def test_ensemble_wrapper(self):
        """앙상블 래퍼"""
        from src.models.probabilistic import ProbabilisticWrapper

        models = [SimpleDropoutModel() for _ in range(3)]
        wrapper = ProbabilisticWrapper(models)

        x = torch.randn(4, 24, 10)
        interval = wrapper.predict_interval(x, confidence=0.8)

        assert interval.confidence == 0.8

    def test_evaluate_calibration(self):
        """보정 평가"""
        from src.models.probabilistic import ProbabilisticWrapper

        model = SimpleDropoutModel()
        wrapper = ProbabilisticWrapper(model, method='mc_dropout', n_samples=30)

        x = torch.randn(8, 24, 10)
        actual = np.random.randn(8)

        metrics = wrapper.evaluate_calibration(x, actual, confidence_levels=[0.5, 0.9])

        assert 'coverage_50' in metrics
        assert 'coverage_90' in metrics
        assert 'ACE' in metrics


class TestCreateProbabilisticModel:
    """create_probabilistic_model 테스트"""

    def test_mc_dropout(self):
        """MC Dropout 생성"""
        from src.models.probabilistic import create_probabilistic_model, MCDropout

        model = SimpleDropoutModel()
        prob_model = create_probabilistic_model(model, 'mc_dropout', n_samples=50)

        assert isinstance(prob_model, MCDropout)
        assert prob_model.n_samples == 50

    def test_invalid_method(self):
        """잘못된 방법"""
        from src.models.probabilistic import create_probabilistic_model

        model = SimpleDropoutModel()

        with pytest.raises(ValueError):
            create_probabilistic_model(model, 'invalid_method')


class TestCalculatePredictionIntervals:
    """calculate_prediction_intervals 테스트"""

    def test_normal_method(self):
        """정규분포 방법"""
        from src.models.probabilistic import calculate_prediction_intervals

        predictions = np.random.randn(100, 10)  # 100 샘플, 10 포인트
        lower, upper = calculate_prediction_intervals(predictions, confidence=0.95, method='normal')

        assert lower.shape == (10,)
        assert upper.shape == (10,)
        assert np.all(upper >= lower)

    def test_percentile_method(self):
        """백분위수 방법"""
        from src.models.probabilistic import calculate_prediction_intervals

        predictions = np.random.randn(100, 10)
        lower, upper = calculate_prediction_intervals(predictions, confidence=0.9, method='percentile')

        assert lower.shape == (10,)
        assert upper.shape == (10,)

    def test_bootstrap_method(self):
        """부트스트랩 방법"""
        from src.models.probabilistic import calculate_prediction_intervals

        predictions = np.random.randn(50, 10)
        lower, upper = calculate_prediction_intervals(predictions, confidence=0.8, method='bootstrap')

        assert lower.shape == (10,)
        assert upper.shape == (10,)

    def test_single_prediction_error(self):
        """단일 예측 에러"""
        from src.models.probabilistic import calculate_prediction_intervals

        predictions = np.random.randn(10)  # 1D array

        with pytest.raises(ValueError):
            calculate_prediction_intervals(predictions, confidence=0.9)


class TestIntegration:
    """통합 테스트"""

    def test_full_mc_dropout_pipeline(self):
        """MC Dropout 전체 파이프라인"""
        from src.models.probabilistic import (
            MCDropout,
            CalibrationMetrics,
        )

        # 모델 생성
        model = SimpleDropoutModel(dropout=0.3)
        mc = MCDropout(model, n_samples=50)

        # 데이터
        x = torch.randn(32, 24, 10)
        actual = torch.randn(32, 1).numpy().squeeze()

        # 예측 구간
        interval = mc.predict_interval(x, confidence=0.9)

        # 보정 평가
        metrics = CalibrationMetrics().evaluate(interval, actual)

        assert 'coverage' in metrics
        assert 0 <= metrics['coverage'] <= 1

    def test_full_ensemble_pipeline(self):
        """앙상블 전체 파이프라인"""
        from src.models.probabilistic import (
            DeepEnsembleUncertainty,
            CalibrationMetrics,
        )

        # 모델 생성
        models = [SimpleDropoutModel() for _ in range(5)]
        ensemble = DeepEnsembleUncertainty(models)

        # 데이터
        x = torch.randn(32, 24, 10)
        actual = torch.randn(32, 1).numpy().squeeze()

        # 예측
        mean, std = ensemble.predict_with_uncertainty(x)
        interval = ensemble.predict_interval(x, confidence=0.9)

        # 평가
        metrics = CalibrationMetrics().evaluate(interval, actual)

        assert mean.shape == (32, 1)
        assert 'coverage' in metrics

    def test_with_lstm_model(self):
        """LSTM 모델과 통합"""
        from src.models.lstm import LSTMModel
        from src.models.probabilistic import MCDropout

        lstm = LSTMModel(input_size=10, hidden_size=32, num_layers=1, dropout=0.2)
        mc = MCDropout(lstm, n_samples=20)

        x = torch.randn(4, 24, 10)
        mean, std = mc.predict_with_uncertainty(x)

        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
