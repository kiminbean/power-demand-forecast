"""
MODEL-012: Probabilistic Forecasting 구현
==========================================

불확실성을 정량화하는 확률적 예측 모델

주요 기능:
1. Quantile Regression - 분위수 기반 예측 구간
2. Monte Carlo Dropout - 드롭아웃 기반 불확실성
3. Deep Ensemble - 앙상블 기반 불확실성
4. Prediction Interval - 신뢰구간 계산
5. Calibration - 예측 구간 보정

Author: Claude Code
Date: 2025-12
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class PredictionInterval:
    """예측 구간 결과"""
    point_estimate: np.ndarray  # 점 추정치 (median or mean)
    lower: np.ndarray  # 하한
    upper: np.ndarray  # 상한
    confidence: float  # 신뢰수준 (e.g., 0.9)
    std: Optional[np.ndarray] = None  # 표준편차
    quantiles: Optional[Dict[float, np.ndarray]] = None  # 분위수별 예측


class QuantileRegressor(nn.Module):
    """
    Quantile Regression 모델

    여러 분위수에 대한 동시 예측으로 예측 분포를 추정합니다.

    Args:
        base_model: 기본 모델 (LSTM 등)
        quantiles: 예측할 분위수 리스트 (default: [0.1, 0.5, 0.9])

    Example:
        >>> base = LSTMModel(input_size=38)
        >>> model = QuantileRegressor(base, quantiles=[0.05, 0.5, 0.95])
        >>> output = model(x)  # (batch, n_quantiles)
    """

    def __init__(
        self,
        base_model: nn.Module,
        quantiles: List[float] = None,
        hidden_size: int = 64
    ):
        super().__init__()
        self.base_model = base_model
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_quantiles = len(self.quantiles)

        # Quantile 출력 레이어
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(self.n_quantiles)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantile 예측

        Args:
            x: 입력 텐서 (batch, seq, features)

        Returns:
            quantile_predictions: (batch, n_quantiles)
        """
        # 기본 모델에서 특성 추출
        if hasattr(self.base_model, 'get_features'):
            features = self.base_model.get_features(x)
        else:
            # LSTM의 경우 마지막 hidden state 사용
            features = self.base_model(x)
            if features.dim() > 2:
                features = features[:, -1, :]

        # 각 분위수에 대한 예측
        quantile_preds = []
        for head in self.quantile_heads:
            pred = head(features)
            quantile_preds.append(pred)

        return torch.cat(quantile_preds, dim=-1)

    def predict_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.9
    ) -> PredictionInterval:
        """
        예측 구간 계산

        Args:
            x: 입력 텐서
            confidence: 신뢰수준

        Returns:
            PredictionInterval 객체
        """
        self.eval()
        with torch.no_grad():
            preds = self(x)

        preds_np = preds.cpu().numpy()

        # 분위수를 딕셔너리로 변환
        quantile_dict = {q: preds_np[:, i] for i, q in enumerate(self.quantiles)}

        # 신뢰구간에 해당하는 분위수 찾기
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        # 가장 가까운 분위수 사용
        lower_idx = np.argmin(np.abs(np.array(self.quantiles) - lower_q))
        upper_idx = np.argmin(np.abs(np.array(self.quantiles) - upper_q))
        median_idx = np.argmin(np.abs(np.array(self.quantiles) - 0.5))

        return PredictionInterval(
            point_estimate=preds_np[:, median_idx],
            lower=preds_np[:, lower_idx],
            upper=preds_np[:, upper_idx],
            confidence=confidence,
            quantiles=quantile_dict
        )


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout 불확실성 추정

    학습된 모델의 드롭아웃을 추론 시에도 활성화하여
    여러 번 샘플링으로 불확실성을 추정합니다.

    Args:
        model: 드롭아웃이 포함된 기본 모델
        n_samples: Monte Carlo 샘플 수 (default: 100)
        dropout_rate: 드롭아웃 비율 (모델의 기본값 사용 시 None)

    Example:
        >>> mc_model = MCDropout(lstm_model, n_samples=50)
        >>> mean, std = mc_model.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        dropout_rate: Optional[float] = None
    ):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # 드롭아웃 레이어 활성화 함수
        self._enable_dropout = self._create_dropout_enabler()

    def _create_dropout_enabler(self):
        """드롭아웃 활성화 함수 생성"""
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        return enable_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """단일 forward pass (드롭아웃 활성화 상태)"""
        self.model.apply(self._enable_dropout)
        return self.model(x)

    def predict_samples(self, x: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo 샘플링

        Args:
            x: 입력 텐서

        Returns:
            samples: (n_samples, batch, output_size)
        """
        samples = []

        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.forward(x)
            samples.append(pred)

        return torch.stack(samples, dim=0)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        불확실성 포함 예측

        Args:
            x: 입력 텐서
            return_samples: 샘플 반환 여부

        Returns:
            mean: 평균 예측
            std: 표준편차 (epistemic uncertainty)
            samples: (optional) 모든 샘플
        """
        samples = self.predict_samples(x)  # (n_samples, batch, output)

        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        if return_samples:
            return mean, std, samples
        return mean, std

    def predict_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.9
    ) -> PredictionInterval:
        """
        예측 구간 계산

        Args:
            x: 입력 텐서
            confidence: 신뢰수준

        Returns:
            PredictionInterval 객체
        """
        samples = self.predict_samples(x)  # (n_samples, batch, output)
        samples_np = samples.cpu().numpy()

        mean = np.mean(samples_np, axis=0)
        std = np.std(samples_np, axis=0)

        # 분위수 기반 구간
        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        lower = np.percentile(samples_np, lower_q * 100, axis=0)
        upper = np.percentile(samples_np, upper_q * 100, axis=0)

        return PredictionInterval(
            point_estimate=mean.squeeze(),
            lower=lower.squeeze(),
            upper=upper.squeeze(),
            confidence=confidence,
            std=std.squeeze()
        )


class DeepEnsembleUncertainty:
    """
    Deep Ensemble 불확실성 추정

    여러 독립적으로 학습된 모델의 예측을 결합하여
    epistemic uncertainty를 추정합니다.

    Args:
        models: 학습된 모델 리스트
        aggregation: 집계 방법 ('mean', 'median')

    Example:
        >>> models = [train_model() for _ in range(5)]
        >>> ensemble = DeepEnsembleUncertainty(models)
        >>> mean, std = ensemble.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = 'mean'
    ):
        self.models = models
        self.n_models = len(models)
        self.aggregation = aggregation

    def predict_samples(self, x: torch.Tensor) -> torch.Tensor:
        """
        각 모델의 예측 수집

        Returns:
            predictions: (n_models, batch, output_size)
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                # TFT 등 tuple 반환 처리
                if isinstance(pred, tuple):
                    pred = pred[0]
                    if pred.dim() == 3:  # quantile output
                        pred = pred[:, :, 1]  # median
            predictions.append(pred)

        return torch.stack(predictions, dim=0)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        불확실성 포함 예측

        Returns:
            mean/median: 집계된 예측
            std: 모델 간 표준편차 (epistemic uncertainty)
        """
        samples = self.predict_samples(x)  # (n_models, batch, output)

        if self.aggregation == 'mean':
            point_estimate = samples.mean(dim=0)
        else:  # median
            point_estimate = samples.median(dim=0)[0]

        std = samples.std(dim=0)

        return point_estimate, std

    def predict_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.9
    ) -> PredictionInterval:
        """예측 구간 계산"""
        samples = self.predict_samples(x)
        samples_np = samples.cpu().numpy()

        mean = np.mean(samples_np, axis=0)
        std = np.std(samples_np, axis=0)

        lower_q = (1 - confidence) / 2
        upper_q = 1 - lower_q

        lower = np.percentile(samples_np, lower_q * 100, axis=0)
        upper = np.percentile(samples_np, upper_q * 100, axis=0)

        return PredictionInterval(
            point_estimate=mean.squeeze(),
            lower=lower.squeeze(),
            upper=upper.squeeze(),
            confidence=confidence,
            std=std.squeeze()
        )


class PinballLoss(nn.Module):
    """
    Pinball Loss (Quantile Loss)

    분위수 회귀를 위한 손실 함수

    Args:
        quantiles: 분위수 리스트

    Example:
        >>> loss_fn = PinballLoss([0.1, 0.5, 0.9])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.register_buffer(
            'quantile_tensor',
            torch.tensor(self.quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Pinball Loss 계산

        Args:
            predictions: (batch, n_quantiles) or (batch, seq, n_quantiles)
            targets: (batch,) or (batch, seq)

        Returns:
            loss: 스칼라 손실값
        """
        # 타겟 shape 맞추기
        if targets.dim() < predictions.dim():
            targets = targets.unsqueeze(-1)

        # 분위수를 predictions 디바이스로 이동
        quantiles = self.quantile_tensor.to(predictions.device)

        # 분위수 차원 확장
        if predictions.dim() == 2:
            quantiles = quantiles.view(1, -1)
        elif predictions.dim() == 3:
            quantiles = quantiles.view(1, 1, -1)

        # 에러 계산
        errors = targets - predictions

        # Pinball loss: q * max(e, 0) + (1-q) * max(-e, 0)
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)

        return loss.mean()


class CalibrationMetrics:
    """
    예측 구간 보정 메트릭

    예측된 신뢰구간이 실제 데이터를 얼마나 잘 포함하는지 평가합니다.

    Example:
        >>> calibrator = CalibrationMetrics()
        >>> metrics = calibrator.evaluate(intervals, actuals)
    """

    @staticmethod
    def coverage(
        lower: np.ndarray,
        upper: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        Coverage 계산 (실제값이 구간 내에 있는 비율)

        Args:
            lower: 하한
            upper: 상한
            actual: 실제값

        Returns:
            coverage: 0~1 사이 값
        """
        in_interval = (actual >= lower) & (actual <= upper)
        return float(np.mean(in_interval))

    @staticmethod
    def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        """평균 구간 너비"""
        return float(np.mean(upper - lower))

    @staticmethod
    def interval_score(
        lower: np.ndarray,
        upper: np.ndarray,
        actual: np.ndarray,
        alpha: float = 0.1
    ) -> float:
        """
        Interval Score (Winkler Score)

        낮을수록 좋음 (좁은 구간 + 높은 coverage)

        Args:
            lower: 하한
            upper: 상한
            actual: 실제값
            alpha: 신뢰수준의 오류 (1 - confidence)

        Returns:
            score: 낮을수록 좋음
        """
        width = upper - lower
        below = actual < lower
        above = actual > upper

        penalty_below = (2 / alpha) * (lower - actual) * below
        penalty_above = (2 / alpha) * (actual - upper) * above

        score = width + penalty_below + penalty_above
        return float(np.mean(score))

    @staticmethod
    def calibration_error(
        intervals: Dict[float, PredictionInterval],
        actual: np.ndarray
    ) -> Dict[str, float]:
        """
        Calibration Error 계산

        각 신뢰수준에서 실제 coverage와 기대 coverage의 차이

        Args:
            intervals: {confidence: PredictionInterval} 딕셔너리
            actual: 실제값

        Returns:
            에러 메트릭 딕셔너리
        """
        errors = {}

        for confidence, interval in intervals.items():
            actual_coverage = CalibrationMetrics.coverage(
                interval.lower, interval.upper, actual
            )
            expected_coverage = confidence

            errors[f'coverage_{int(confidence*100)}'] = actual_coverage
            errors[f'error_{int(confidence*100)}'] = abs(actual_coverage - expected_coverage)

        # 평균 절대 보정 오차 (ACE)
        ace = np.mean([v for k, v in errors.items() if k.startswith('error_')])
        errors['ACE'] = ace

        return errors

    def evaluate(
        self,
        interval: PredictionInterval,
        actual: np.ndarray
    ) -> Dict[str, float]:
        """
        종합 평가

        Args:
            interval: 예측 구간
            actual: 실제값

        Returns:
            평가 메트릭 딕셔너리
        """
        return {
            'coverage': self.coverage(interval.lower, interval.upper, actual),
            'expected_coverage': interval.confidence,
            'interval_width': self.interval_width(interval.lower, interval.upper),
            'interval_score': self.interval_score(
                interval.lower, interval.upper, actual,
                alpha=1 - interval.confidence
            )
        }


class ProbabilisticWrapper:
    """
    확률적 예측 래퍼

    기존 모델을 확률적 예측 모델로 변환합니다.

    Args:
        model: 기본 모델
        method: 불확실성 추정 방법
            - 'mc_dropout': Monte Carlo Dropout
            - 'quantile': Quantile Regression
            - 'ensemble': Deep Ensemble (models가 리스트인 경우)
        n_samples: MC Dropout 샘플 수
        quantiles: Quantile Regression 분위수

    Example:
        >>> wrapper = ProbabilisticWrapper(lstm_model, method='mc_dropout')
        >>> interval = wrapper.predict_interval(x, confidence=0.9)
    """

    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        method: str = 'mc_dropout',
        n_samples: int = 100,
        quantiles: List[float] = None
    ):
        self.method = method

        if isinstance(model, list):
            self.predictor = DeepEnsembleUncertainty(model)
        elif method == 'mc_dropout':
            self.predictor = MCDropout(model, n_samples=n_samples)
        elif method == 'quantile':
            warnings.warn("Quantile method requires trained QuantileRegressor")
            self.predictor = model
        else:
            raise ValueError(f"Unknown method: {method}")

        self.calibrator = CalibrationMetrics()

    def predict_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.9
    ) -> PredictionInterval:
        """예측 구간 계산"""
        return self.predictor.predict_interval(x, confidence=confidence)

    def evaluate_calibration(
        self,
        x: torch.Tensor,
        actual: np.ndarray,
        confidence_levels: List[float] = None
    ) -> Dict[str, float]:
        """
        다양한 신뢰수준에서 보정 평가

        Args:
            x: 입력 텐서
            actual: 실제값
            confidence_levels: 평가할 신뢰수준 리스트

        Returns:
            보정 메트릭 딕셔너리
        """
        confidence_levels = confidence_levels or [0.5, 0.8, 0.9, 0.95]
        intervals = {}

        for conf in confidence_levels:
            intervals[conf] = self.predictor.predict_interval(x, confidence=conf)

        return CalibrationMetrics.calibration_error(intervals, actual)


def create_probabilistic_model(
    base_model: nn.Module,
    method: str = 'mc_dropout',
    **kwargs
) -> Union[MCDropout, QuantileRegressor]:
    """
    확률적 모델 팩토리 함수

    Args:
        base_model: 기본 모델
        method: 불확실성 추정 방법
        **kwargs: 추가 설정

    Returns:
        확률적 예측 모델

    Example:
        >>> prob_model = create_probabilistic_model(lstm, 'mc_dropout', n_samples=50)
    """
    if method == 'mc_dropout':
        n_samples = kwargs.get('n_samples', 100)
        return MCDropout(base_model, n_samples=n_samples)
    elif method == 'quantile':
        quantiles = kwargs.get('quantiles', [0.1, 0.5, 0.9])
        hidden_size = kwargs.get('hidden_size', 64)
        return QuantileRegressor(base_model, quantiles=quantiles, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_prediction_intervals(
    predictions: np.ndarray,
    confidence: float = 0.9,
    method: str = 'normal'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    점 추정치에서 예측 구간 계산 (잔차 기반)

    Args:
        predictions: 예측값 배열 (n_samples, n_points) 또는 (n_points,)
        confidence: 신뢰수준
        method: 구간 계산 방법 ('normal', 'bootstrap', 'percentile')

    Returns:
        lower: 하한
        upper: 상한
    """
    if predictions.ndim == 1:
        # 단일 예측 - 정규분포 가정 불가
        raise ValueError("Multiple predictions required for interval calculation")

    if method == 'normal':
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        z = stats.norm.ppf((1 + confidence) / 2)
        lower = mean - z * std
        upper = mean + z * std

    elif method == 'percentile':
        lower_q = (1 - confidence) / 2 * 100
        upper_q = (1 + confidence) / 2 * 100

        lower = np.percentile(predictions, lower_q, axis=0)
        upper = np.percentile(predictions, upper_q, axis=0)

    elif method == 'bootstrap':
        n_bootstrap = 1000
        n_samples, n_points = predictions.shape

        bootstrap_means = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_means.append(np.mean(predictions[indices], axis=0))

        bootstrap_means = np.array(bootstrap_means)

        lower_q = (1 - confidence) / 2 * 100
        upper_q = (1 + confidence) / 2 * 100

        lower = np.percentile(bootstrap_means, lower_q, axis=0)
        upper = np.percentile(bootstrap_means, upper_q, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return lower, upper
