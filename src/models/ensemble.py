"""
MODEL-011: Ensemble 모델 구현
============================

LSTM, BiLSTM, TFT 모델의 예측을 결합하는 앙상블 기법

주요 기능:
1. Weighted Average Ensemble - 가중 평균
2. Stacking Ensemble - Meta-learner 기반
3. Blending Ensemble - Hold-out 기반
4. 가중치 자동 최적화 (Optuna)

Author: Claude Code
Date: 2025-12
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings
from scipy.optimize import minimize
import json
from pathlib import Path


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    ensemble_type: str = 'weighted_average'  # weighted_average, stacking, blending
    optimization_method: str = 'scipy'  # scipy, optuna, grid
    meta_learner_type: str = 'ridge'  # ridge, mlp, xgboost
    blend_ratio: float = 0.2  # blending용 hold-out 비율
    n_folds: int = 5  # stacking용 cross-validation folds
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


class BaseEnsemble(ABC, nn.Module):
    """앙상블 모델 기본 클래스"""

    def __init__(self, models: List[nn.Module], config: Optional[EnsembleConfig] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.config = config or EnsembleConfig()
        self.n_models = len(models)
        self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        self._is_fitted = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """앙상블 예측"""
        pass

    @abstractmethod
    def fit(self, train_loader, val_loader=None, **kwargs):
        """앙상블 가중치 학습"""
        pass

    def get_individual_predictions(
        self,
        *args,
        **kwargs
    ) -> List[torch.Tensor]:
        """개별 모델 예측 수집"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(*args, **kwargs)
                # TFT의 경우 tuple 반환 (predictions, attention, weights)
                if isinstance(pred, tuple):
                    pred = pred[0]  # predictions만 사용
                    # quantile 출력인 경우 median (0.5) 사용
                    if pred.dim() == 3:
                        pred = pred[:, :, 1]  # 중간값 (0.5 quantile)
                predictions.append(pred)
        return predictions

    def get_weights(self) -> np.ndarray:
        """정규화된 가중치 반환"""
        weights = torch.softmax(self.weights, dim=0)
        return weights.detach().cpu().numpy()

    def set_weights(self, weights: Union[List[float], np.ndarray, torch.Tensor]):
        """가중치 설정"""
        if isinstance(weights, (list, np.ndarray)):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.weights.data = weights

    def save(self, path: Union[str, Path]):
        """앙상블 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 메타데이터 저장
        metadata = {
            'n_models': self.n_models,
            'weights': self.get_weights().tolist(),
            'config': {
                'ensemble_type': self.config.ensemble_type,
                'optimization_method': self.config.optimization_method,
                'meta_learner_type': self.config.meta_learner_type,
            },
            'is_fitted': self._is_fitted
        }

        with open(path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # 개별 모델 저장
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f'model_{i}.pt')

    @classmethod
    def load(cls, path: Union[str, Path], model_classes: List[type], model_configs: List[dict]):
        """앙상블 로드"""
        path = Path(path)

        with open(path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)

        # 모델 인스턴스 생성 및 로드
        models = []
        for i, (model_cls, config) in enumerate(zip(model_classes, model_configs)):
            model = model_cls(**config)
            model.load_state_dict(torch.load(path / f'model_{i}.pt'))
            models.append(model)

        # 앙상블 생성
        config = EnsembleConfig(**metadata['config'])
        ensemble = cls(models, config)
        ensemble.set_weights(metadata['weights'])
        ensemble._is_fitted = metadata['is_fitted']

        return ensemble


class WeightedAverageEnsemble(BaseEnsemble):
    """
    가중 평균 앙상블

    각 모델의 예측에 가중치를 적용하여 평균을 계산합니다.
    가중치는 검증 성능 기반으로 최적화됩니다.

    Example:
        >>> models = [lstm_model, bilstm_model, tft_model]
        >>> ensemble = WeightedAverageEnsemble(models)
        >>> ensemble.fit(train_loader, val_loader)
        >>> predictions = ensemble(x)
    """

    def __init__(self, models: List[nn.Module], config: Optional[EnsembleConfig] = None):
        super().__init__(models, config)

    def forward(
        self,
        *args,
        return_individual: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        앙상블 예측 수행

        Args:
            *args: 모델 입력
            return_individual: 개별 모델 예측도 반환할지 여부
            **kwargs: 추가 인자

        Returns:
            ensemble_pred: 앙상블 예측
            individual_preds: (optional) 개별 모델 예측 리스트
        """
        predictions = self.get_individual_predictions(*args, **kwargs)

        # 가중치 정규화
        weights = torch.softmax(self.weights, dim=0)

        # 가중 평균 계산
        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, ...)
        weights_expanded = weights.view(-1, *([1] * (stacked.dim() - 1)))
        ensemble_pred = (stacked * weights_expanded).sum(dim=0)

        if return_individual:
            return ensemble_pred, predictions
        return ensemble_pred

    def fit(
        self,
        val_predictions: List[torch.Tensor],
        val_targets: torch.Tensor,
        method: str = 'scipy',
        **kwargs
    ) -> Dict[str, float]:
        """
        검증 데이터에서 최적 가중치 학습

        Args:
            val_predictions: 각 모델의 검증 예측 리스트
            val_targets: 검증 타겟
            method: 최적화 방법 ('scipy', 'grid')

        Returns:
            최적화 결과 딕셔너리
        """
        # numpy로 변환
        preds = [p.detach().cpu().numpy().flatten() for p in val_predictions]
        targets = val_targets.detach().cpu().numpy().flatten()
        preds_array = np.stack(preds, axis=0)  # (n_models, n_samples)

        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # 정규화
            ensemble_pred = (preds_array * weights.reshape(-1, 1)).sum(axis=0)
            mse = np.mean((ensemble_pred - targets) ** 2)
            return mse

        if method == 'scipy':
            # 제약 조건: 가중치 합 = 1, 각 가중치 >= 0
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.n_models)]

            result = minimize(
                objective,
                x0=np.ones(self.n_models) / self.n_models,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            optimal_weights = result.x

        elif method == 'grid':
            # 그리드 서치
            best_mse = float('inf')
            optimal_weights = None

            # 가중치 조합 생성
            step = 0.1
            for w1 in np.arange(0, 1 + step, step):
                for w2 in np.arange(0, 1 - w1 + step, step):
                    w3 = 1 - w1 - w2
                    if w3 >= 0:
                        weights = [w1, w2, w3][:self.n_models]
                        if len(weights) < self.n_models:
                            weights.extend([0] * (self.n_models - len(weights)))
                        mse = objective(weights)
                        if mse < best_mse:
                            best_mse = mse
                            optimal_weights = np.array(weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 가중치 설정
        self.set_weights(optimal_weights)
        self._is_fitted = True

        # 최종 성능 계산
        final_mse = objective(optimal_weights)
        final_rmse = np.sqrt(final_mse)

        return {
            'optimal_weights': optimal_weights.tolist(),
            'final_mse': float(final_mse),
            'final_rmse': float(final_rmse)
        }


class StackingEnsemble(BaseEnsemble):
    """
    Stacking 앙상블

    개별 모델의 예측을 피처로 사용하는 Meta-learner를 학습합니다.
    K-Fold Cross-validation으로 out-of-fold 예측을 생성합니다.

    Example:
        >>> models = [lstm_model, bilstm_model, tft_model]
        >>> ensemble = StackingEnsemble(models, meta_learner='ridge')
        >>> ensemble.fit(train_loader, val_loader)
        >>> predictions = ensemble(x)
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
        meta_learner_type: str = 'ridge'
    ):
        super().__init__(models, config)
        self.meta_learner_type = meta_learner_type
        self.meta_learner = self._create_meta_learner()

    def _create_meta_learner(self) -> nn.Module:
        """Meta-learner 생성"""
        if self.meta_learner_type == 'ridge':
            # 간단한 선형 모델
            return nn.Linear(self.n_models, 1)
        elif self.meta_learner_type == 'mlp':
            return nn.Sequential(
                nn.Linear(self.n_models, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        else:
            # 기본: 선형
            return nn.Linear(self.n_models, 1)

    def forward(
        self,
        *args,
        return_individual: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Stacking 앙상블 예측

        개별 모델 예측을 meta-learner 입력으로 사용
        """
        predictions = self.get_individual_predictions(*args, **kwargs)

        # 예측을 meta-learner 입력 형태로 변환
        # predictions: List[(batch, seq) or (batch,)]
        stacked = torch.stack(predictions, dim=-1)  # (batch, ..., n_models)

        # 원래 shape 저장
        original_shape = stacked.shape[:-1]

        # meta-learner 입력 형태로 flatten
        stacked_flat = stacked.view(-1, self.n_models)

        # meta-learner 예측
        meta_pred = self.meta_learner(stacked_flat)

        # 원래 shape으로 복원
        ensemble_pred = meta_pred.view(*original_shape)

        if return_individual:
            return ensemble_pred, predictions
        return ensemble_pred

    def fit(
        self,
        val_predictions: List[torch.Tensor],
        val_targets: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs
    ) -> Dict[str, float]:
        """
        Meta-learner 학습

        Args:
            val_predictions: 각 모델의 검증 예측
            val_targets: 검증 타겟
            epochs: 학습 에포크
            lr: 학습률

        Returns:
            학습 결과 딕셔너리
        """
        # 입력 준비
        stacked = torch.stack(val_predictions, dim=-1)  # (batch, ..., n_models)
        stacked_flat = stacked.view(-1, self.n_models)
        targets_flat = val_targets.view(-1, 1)

        # 디바이스 설정
        device = stacked_flat.device
        self.meta_learner = self.meta_learner.to(device)

        # 학습
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        for epoch in range(epochs):
            self.meta_learner.train()
            optimizer.zero_grad()

            pred = self.meta_learner(stacked_flat)
            loss = criterion(pred, targets_flat)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        self._is_fitted = True

        return {
            'final_loss': float(best_loss),
            'final_rmse': float(np.sqrt(best_loss))
        }


class BlendingEnsemble(BaseEnsemble):
    """
    Blending 앙상블

    Hold-out 데이터셋에서 meta-learner를 학습합니다.
    Stacking보다 단순하지만 데이터 효율성이 낮습니다.

    Example:
        >>> models = [lstm_model, bilstm_model, tft_model]
        >>> ensemble = BlendingEnsemble(models, blend_ratio=0.2)
        >>> ensemble.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
        blend_ratio: float = 0.2
    ):
        super().__init__(models, config)
        self.blend_ratio = blend_ratio
        self.meta_learner = nn.Linear(self.n_models, 1)

    def forward(
        self,
        *args,
        return_individual: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Blending 앙상블 예측"""
        predictions = self.get_individual_predictions(*args, **kwargs)

        stacked = torch.stack(predictions, dim=-1)
        original_shape = stacked.shape[:-1]
        stacked_flat = stacked.view(-1, self.n_models)

        meta_pred = self.meta_learner(stacked_flat)
        ensemble_pred = meta_pred.view(*original_shape)

        if return_individual:
            return ensemble_pred, predictions
        return ensemble_pred

    def fit(
        self,
        val_predictions: List[torch.Tensor],
        val_targets: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        **kwargs
    ) -> Dict[str, float]:
        """Meta-learner 학습 (BlendingEnsemble)"""
        stacked = torch.stack(val_predictions, dim=-1)
        stacked_flat = stacked.view(-1, self.n_models)
        targets_flat = val_targets.view(-1, 1)

        device = stacked_flat.device
        self.meta_learner = self.meta_learner.to(device)

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        for epoch in range(epochs):
            self.meta_learner.train()
            optimizer.zero_grad()

            pred = self.meta_learner(stacked_flat)
            loss = criterion(pred, targets_flat)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        self._is_fitted = True

        return {
            'final_loss': float(best_loss),
            'final_rmse': float(np.sqrt(best_loss))
        }


class UncertaintyEnsemble(BaseEnsemble):
    """
    불확실성 추정 앙상블

    개별 모델의 예측 분산을 활용하여 불확실성을 추정합니다.

    Example:
        >>> ensemble = UncertaintyEnsemble(models)
        >>> mean, std, lower, upper = ensemble.predict_with_uncertainty(x)
    """

    def __init__(self, models: List[nn.Module], config: Optional[EnsembleConfig] = None):
        super().__init__(models, config)

    def forward(
        self,
        *args,
        return_uncertainty: bool = False,
        confidence: float = 0.9,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        불확실성 포함 앙상블 예측

        Args:
            return_uncertainty: 불확실성 정보 반환 여부
            confidence: 신뢰구간 수준

        Returns:
            mean: 평균 예측
            std: 표준편차 (if return_uncertainty)
            lower: 하한 (if return_uncertainty)
            upper: 상한 (if return_uncertainty)
        """
        predictions = self.get_individual_predictions(*args, **kwargs)
        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, ...)

        # 가중 평균
        weights = torch.softmax(self.weights, dim=0)
        weights_expanded = weights.view(-1, *([1] * (stacked.dim() - 1)))
        mean = (stacked * weights_expanded).sum(dim=0)

        if not return_uncertainty:
            return mean

        # 불확실성 추정 (모델 간 분산)
        variance = ((stacked - mean.unsqueeze(0)) ** 2 * weights_expanded).sum(dim=0)
        std = torch.sqrt(variance + 1e-8)

        # 신뢰구간 (정규분포 가정)
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = mean - z * std
        upper = mean + z * std

        return mean, std, lower, upper

    def fit(
        self,
        val_predictions: List[torch.Tensor],
        val_targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """불확실성 앙상블 가중치 학습"""
        # WeightedAverageEnsemble과 동일
        preds = [p.detach().cpu().numpy().flatten() for p in val_predictions]
        targets = val_targets.detach().cpu().numpy().flatten()
        preds_array = np.stack(preds, axis=0)

        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = (preds_array * weights.reshape(-1, 1)).sum(axis=0)
            mse = np.mean((ensemble_pred - targets) ** 2)
            return mse

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_models)]

        result = minimize(
            objective,
            x0=np.ones(self.n_models) / self.n_models,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.set_weights(result.x)
        self._is_fitted = True

        return {
            'optimal_weights': result.x.tolist(),
            'final_mse': float(result.fun)
        }


class EnsembleOptimizer:
    """
    앙상블 가중치 최적화 도구

    Optuna를 활용한 베이지안 최적화 지원

    Example:
        >>> optimizer = EnsembleOptimizer(ensemble)
        >>> best_weights = optimizer.optimize(val_preds, val_targets, n_trials=100)
    """

    def __init__(self, ensemble: BaseEnsemble):
        self.ensemble = ensemble

    def optimize_with_optuna(
        self,
        val_predictions: List[torch.Tensor],
        val_targets: torch.Tensor,
        n_trials: int = 100,
        metric: str = 'mse'
    ) -> Dict[str, Any]:
        """
        Optuna로 가중치 최적화

        Args:
            val_predictions: 검증 예측 리스트
            val_targets: 검증 타겟
            n_trials: 시도 횟수
            metric: 최적화 메트릭 ('mse', 'mae', 'mape')

        Returns:
            최적화 결과
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required for this method. pip install optuna")

        preds = [p.detach().cpu().numpy().flatten() for p in val_predictions]
        targets = val_targets.detach().cpu().numpy().flatten()
        preds_array = np.stack(preds, axis=0)
        n_models = len(preds)

        def objective(trial):
            # 가중치 샘플링
            weights = []
            remaining = 1.0
            for i in range(n_models - 1):
                w = trial.suggest_float(f'w{i}', 0.0, remaining)
                weights.append(w)
                remaining -= w
            weights.append(remaining)
            weights = np.array(weights)

            # 앙상블 예측
            ensemble_pred = (preds_array * weights.reshape(-1, 1)).sum(axis=0)

            # 메트릭 계산
            if metric == 'mse':
                return np.mean((ensemble_pred - targets) ** 2)
            elif metric == 'mae':
                return np.mean(np.abs(ensemble_pred - targets))
            elif metric == 'mape':
                return np.mean(np.abs((ensemble_pred - targets) / (targets + 1e-8))) * 100
            else:
                raise ValueError(f"Unknown metric: {metric}")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # 최적 가중치 추출
        best_weights = []
        remaining = 1.0
        for i in range(n_models - 1):
            w = study.best_params[f'w{i}']
            best_weights.append(w)
            remaining -= w
        best_weights.append(remaining)

        # 앙상블에 적용
        self.ensemble.set_weights(best_weights)
        self.ensemble._is_fitted = True

        return {
            'best_weights': best_weights,
            'best_value': study.best_value,
            'n_trials': n_trials
        }


def create_ensemble(
    models: List[nn.Module],
    ensemble_type: str = 'weighted_average',
    **kwargs
) -> BaseEnsemble:
    """
    앙상블 팩토리 함수

    Args:
        models: 기본 모델 리스트
        ensemble_type: 앙상블 유형
            - 'weighted_average': 가중 평균
            - 'stacking': Stacking (meta-learner)
            - 'blending': Blending (hold-out)
            - 'uncertainty': 불확실성 추정
        **kwargs: 추가 설정

    Returns:
        BaseEnsemble 인스턴스

    Example:
        >>> models = [lstm, bilstm, tft]
        >>> ensemble = create_ensemble(models, 'stacking', meta_learner_type='mlp')
    """
    config = EnsembleConfig(ensemble_type=ensemble_type)

    if ensemble_type == 'weighted_average':
        return WeightedAverageEnsemble(models, config)
    elif ensemble_type == 'stacking':
        meta_learner_type = kwargs.get('meta_learner_type', 'ridge')
        return StackingEnsemble(models, config, meta_learner_type=meta_learner_type)
    elif ensemble_type == 'blending':
        blend_ratio = kwargs.get('blend_ratio', 0.2)
        return BlendingEnsemble(models, config, blend_ratio=blend_ratio)
    elif ensemble_type == 'uncertainty':
        return UncertaintyEnsemble(models, config)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble(
    ensemble: BaseEnsemble,
    val_predictions: List[torch.Tensor],
    val_targets: torch.Tensor
) -> Dict[str, float]:
    """
    앙상블 성능 평가

    Args:
        ensemble: 학습된 앙상블
        val_predictions: 검증 예측
        val_targets: 검증 타겟

    Returns:
        성능 메트릭 딕셔너리
    """
    # 앙상블 예측
    stacked = torch.stack(val_predictions, dim=-1)
    stacked_flat = stacked.view(-1, ensemble.n_models)

    weights = torch.softmax(ensemble.weights, dim=0)
    ensemble_pred = (stacked_flat * weights).sum(dim=-1)

    # numpy 변환
    pred = ensemble_pred.detach().cpu().numpy()
    target = val_targets.detach().cpu().numpy().flatten()

    # 메트릭 계산
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    mape = np.mean(np.abs((pred - target) / (target + 1e-8))) * 100

    # R² 계산
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'MSE': float(mse),
        'RMSE': float(np.sqrt(mse)),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }


# 개별 모델 성능 대비 앙상블 개선율 분석
def compare_with_individual(
    ensemble: BaseEnsemble,
    val_predictions: List[torch.Tensor],
    val_targets: torch.Tensor,
    model_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    개별 모델 대비 앙상블 성능 비교

    Args:
        ensemble: 학습된 앙상블
        val_predictions: 각 모델의 검증 예측
        val_targets: 검증 타겟
        model_names: 모델 이름 리스트

    Returns:
        비교 결과 딕셔너리
    """
    if model_names is None:
        model_names = [f'Model_{i}' for i in range(len(val_predictions))]

    target = val_targets.detach().cpu().numpy().flatten()

    results = {'individual': {}, 'ensemble': {}, 'improvement': {}}

    # 개별 모델 성능
    for name, pred_tensor in zip(model_names, val_predictions):
        pred = pred_tensor.detach().cpu().numpy().flatten()
        mape = np.mean(np.abs((pred - target) / (target + 1e-8))) * 100
        results['individual'][name] = {'MAPE': mape}

    # 앙상블 성능
    ensemble_metrics = evaluate_ensemble(ensemble, val_predictions, val_targets)
    results['ensemble'] = ensemble_metrics

    # 개선율 계산
    best_individual_mape = min(r['MAPE'] for r in results['individual'].values())
    avg_individual_mape = np.mean([r['MAPE'] for r in results['individual'].values()])

    results['improvement'] = {
        'vs_best_individual': (best_individual_mape - ensemble_metrics['MAPE']) / best_individual_mape * 100,
        'vs_avg_individual': (avg_individual_mape - ensemble_metrics['MAPE']) / avg_individual_mape * 100
    }

    return results
