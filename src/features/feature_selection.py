"""
FEAT-010: 자동 피처 선택 (AutoML)
=================================

SHAP, Permutation Importance 기반 자동 피처 선택 시스템

주요 기능:
1. SHAP 기반 피처 중요도 분석
2. Permutation Importance 계산
3. 자동 피처 제거 로직
4. 피처 중요도 리포트 생성

Author: Claude Code
Date: 2025-12
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from datetime import datetime


@dataclass
class FeatureImportance:
    """피처 중요도 결과"""
    feature_names: List[str]
    importance_scores: np.ndarray
    method: str  # 'shap', 'permutation', 'gradient'
    std: Optional[np.ndarray] = None  # 표준편차 (permutation의 경우)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importance_scores
        })
        if self.std is not None:
            df['std'] = self.std
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def top_features(self, n: int = 10) -> List[str]:
        """상위 n개 피처"""
        indices = np.argsort(self.importance_scores)[::-1][:n]
        return [self.feature_names[i] for i in indices]

    def bottom_features(self, n: int = 10) -> List[str]:
        """하위 n개 피처 (제거 후보)"""
        indices = np.argsort(self.importance_scores)[:n]
        return [self.feature_names[i] for i in indices]


class PermutationImportance:
    """
    Permutation Importance 계산

    각 피처를 무작위로 섞어 모델 성능 변화를 측정합니다.

    Args:
        model: PyTorch 모델
        metric: 평가 메트릭 함수 (predictions, targets) -> score
        n_repeats: 반복 횟수 (안정성을 위해)
        random_state: 난수 시드

    Example:
        >>> pi = PermutationImportance(model, metric=mse_metric)
        >>> importance = pi.compute(X, y, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        metric: Callable,
        n_repeats: int = 10,
        random_state: int = 42
    ):
        self.model = model
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_state = random_state

    def compute(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: List[str],
        device: Optional[torch.device] = None
    ) -> FeatureImportance:
        """
        Permutation Importance 계산

        Args:
            X: 입력 텐서 (batch, seq, features)
            y: 타겟 텐서
            feature_names: 피처 이름 리스트
            device: 디바이스

        Returns:
            FeatureImportance 객체
        """
        if device is None:
            device = next(self.model.parameters()).device

        X = X.to(device)
        y = y.to(device)

        self.model.eval()
        np.random.seed(self.random_state)

        # 기준 성능 계산
        with torch.no_grad():
            baseline_pred = self.model(X)
            if isinstance(baseline_pred, tuple):
                baseline_pred = baseline_pred[0]
                if baseline_pred.dim() == 3:
                    baseline_pred = baseline_pred[:, :, 1]  # median

        baseline_score = self.metric(baseline_pred.cpu(), y.cpu())

        n_features = X.shape[-1]
        importance_scores = []
        importance_stds = []

        for feat_idx in range(n_features):
            scores = []

            for _ in range(self.n_repeats):
                # 피처 복사 및 셔플
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.shape[0])
                X_permuted[:, :, feat_idx] = X[perm_indices, :, feat_idx]

                # 예측
                with torch.no_grad():
                    pred = self.model(X_permuted)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                        if pred.dim() == 3:
                            pred = pred[:, :, 1]

                score = self.metric(pred.cpu(), y.cpu())
                scores.append(score)

            # 중요도 = 기준 성능 - 셔플 후 성능 (MSE는 높을수록 나쁨)
            mean_score = np.mean(scores)
            importance = mean_score - baseline_score  # 양수 = 중요
            importance_scores.append(importance)
            importance_stds.append(np.std(scores))

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=np.array(importance_scores),
            method='permutation',
            std=np.array(importance_stds)
        )


class GradientImportance:
    """
    Gradient 기반 피처 중요도

    입력에 대한 그래디언트 크기로 피처 중요도를 추정합니다.

    Args:
        model: PyTorch 모델
        aggregate: 집계 방법 ('mean', 'max', 'l2')

    Example:
        >>> gi = GradientImportance(model)
        >>> importance = gi.compute(X, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        aggregate: str = 'mean'
    ):
        self.model = model
        self.aggregate = aggregate

    def compute(
        self,
        X: torch.Tensor,
        feature_names: List[str],
        device: Optional[torch.device] = None
    ) -> FeatureImportance:
        """
        Gradient Importance 계산

        Args:
            X: 입력 텐서 (batch, seq, features)
            feature_names: 피처 이름 리스트
            device: 디바이스

        Returns:
            FeatureImportance 객체
        """
        if device is None:
            device = next(self.model.parameters()).device

        X = X.to(device).requires_grad_(True)
        self.model.eval()

        # Forward pass
        output = self.model(X)
        if isinstance(output, tuple):
            output = output[0]
            if output.dim() == 3:
                output = output[:, :, 1]

        # Backward pass
        output.sum().backward()

        # 그래디언트 집계
        gradients = X.grad.abs()  # (batch, seq, features)

        if self.aggregate == 'mean':
            importance = gradients.mean(dim=(0, 1))
        elif self.aggregate == 'max':
            importance = gradients.max(dim=1)[0].max(dim=0)[0]
        elif self.aggregate == 'l2':
            importance = torch.sqrt((gradients ** 2).sum(dim=(0, 1)))
        else:
            raise ValueError(f"Unknown aggregate: {self.aggregate}")

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=importance.detach().cpu().numpy(),
            method='gradient'
        )


class SHAPImportance:
    """
    SHAP 기반 피처 중요도

    Shapley 값을 근사하여 피처 중요도를 계산합니다.
    DeepExplainer 또는 KernelExplainer 사용

    Args:
        model: PyTorch 모델
        background_samples: 배경 데이터 샘플 수

    Example:
        >>> shap_imp = SHAPImportance(model)
        >>> importance = shap_imp.compute(X, background, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        background_samples: int = 100
    ):
        self.model = model
        self.background_samples = background_samples
        self._check_shap()

    def _check_shap(self):
        """SHAP 라이브러리 확인"""
        try:
            import shap
            self.shap = shap
        except ImportError:
            self.shap = None
            warnings.warn("SHAP not installed. Install with: pip install shap")

    def _model_wrapper(self, x: np.ndarray) -> np.ndarray:
        """SHAP용 모델 래퍼"""
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
            if isinstance(output, tuple):
                output = output[0]
                if output.dim() == 3:
                    output = output[:, :, 1]

        return output.cpu().numpy()

    def compute(
        self,
        X: torch.Tensor,
        background: Optional[torch.Tensor] = None,
        feature_names: List[str] = None
    ) -> FeatureImportance:
        """
        SHAP 값 계산

        Args:
            X: 설명할 입력 텐서
            background: 배경 데이터 (없으면 X에서 샘플링)
            feature_names: 피처 이름 리스트

        Returns:
            FeatureImportance 객체
        """
        if self.shap is None:
            raise ImportError("SHAP not installed. pip install shap")

        X_np = X.cpu().numpy()

        if background is None:
            n_samples = min(self.background_samples, len(X_np))
            indices = np.random.choice(len(X_np), n_samples, replace=False)
            background = X_np[indices]
        else:
            background = background.cpu().numpy()

        # 3D 입력을 2D로 변환 (마지막 시점만)
        X_2d = X_np[:, -1, :]  # (batch, features)
        bg_2d = background[:, -1, :]

        # 2D 모델 래퍼
        def model_2d(x):
            # 다시 3D로 확장
            x_3d = np.repeat(x[:, np.newaxis, :], X_np.shape[1], axis=1)
            return self._model_wrapper(x_3d)

        # KernelExplainer 사용
        explainer = self.shap.KernelExplainer(model_2d, bg_2d)
        shap_values = explainer.shap_values(X_2d[:min(100, len(X_2d))])

        # 피처별 평균 절대 SHAP 값
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=importance,
            method='shap'
        )


class AutoFeatureSelector:
    """
    자동 피처 선택기

    여러 방법을 조합하여 최적의 피처 세트를 선택합니다.

    Args:
        model: PyTorch 모델
        metric: 평가 메트릭
        methods: 사용할 방법들 ('permutation', 'gradient', 'shap')
        threshold: 제거 임계값 (상대적 중요도)
        min_features: 최소 유지 피처 수

    Example:
        >>> selector = AutoFeatureSelector(model, mse_metric)
        >>> selected = selector.select_features(X, y, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        metric: Callable,
        methods: List[str] = None,
        threshold: float = 0.01,
        min_features: int = 5
    ):
        self.model = model
        self.metric = metric
        self.methods = methods or ['permutation', 'gradient']
        self.threshold = threshold
        self.min_features = min_features

        self._importances: Dict[str, FeatureImportance] = {}

    def compute_all_importances(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: List[str],
        device: Optional[torch.device] = None
    ) -> Dict[str, FeatureImportance]:
        """
        모든 방법으로 중요도 계산

        Returns:
            {method: FeatureImportance} 딕셔너리
        """
        results = {}

        if 'permutation' in self.methods:
            pi = PermutationImportance(self.model, self.metric)
            results['permutation'] = pi.compute(X, y, feature_names, device)

        if 'gradient' in self.methods:
            gi = GradientImportance(self.model)
            results['gradient'] = gi.compute(X, feature_names, device)

        if 'shap' in self.methods:
            try:
                si = SHAPImportance(self.model)
                results['shap'] = si.compute(X, feature_names=feature_names)
            except ImportError:
                warnings.warn("SHAP not available, skipping")

        self._importances = results
        return results

    def aggregate_importance(
        self,
        importances: Dict[str, FeatureImportance],
        weights: Optional[Dict[str, float]] = None
    ) -> FeatureImportance:
        """
        여러 방법의 중요도를 집계

        Args:
            importances: 방법별 중요도
            weights: 방법별 가중치

        Returns:
            집계된 FeatureImportance
        """
        if not importances:
            raise ValueError("No importance scores to aggregate")

        if weights is None:
            weights = {m: 1.0 / len(importances) for m in importances}

        # 정규화 및 가중 합
        feature_names = list(importances.values())[0].feature_names
        n_features = len(feature_names)

        aggregated = np.zeros(n_features)

        for method, imp in importances.items():
            # Min-Max 정규화
            scores = imp.importance_scores
            if scores.max() - scores.min() > 0:
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = np.ones_like(scores)

            aggregated += weights.get(method, 0) * normalized

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=aggregated,
            method='aggregated'
        )

    def select_features(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: List[str],
        device: Optional[torch.device] = None,
        return_importance: bool = False
    ) -> Union[List[str], Tuple[List[str], FeatureImportance]]:
        """
        피처 선택 수행

        Args:
            X: 입력 텐서
            y: 타겟 텐서
            feature_names: 피처 이름
            device: 디바이스
            return_importance: 중요도도 반환할지

        Returns:
            selected_features: 선택된 피처 리스트
            importance: (optional) 집계된 중요도
        """
        # 중요도 계산
        importances = self.compute_all_importances(X, y, feature_names, device)

        # 집계
        aggregated = self.aggregate_importance(importances)

        # 선택
        scores = aggregated.importance_scores
        max_score = scores.max()

        # 임계값 이상인 피처 선택
        selected_mask = scores >= (max_score * self.threshold)

        # 최소 피처 수 보장
        if selected_mask.sum() < self.min_features:
            top_indices = np.argsort(scores)[::-1][:self.min_features]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True

        selected_features = [
            name for name, mask in zip(feature_names, selected_mask) if mask
        ]

        if return_importance:
            return selected_features, aggregated
        return selected_features

    def get_feature_ranking(self) -> pd.DataFrame:
        """
        피처 순위 DataFrame 반환

        모든 방법의 중요도와 순위를 포함
        """
        if not self._importances:
            raise ValueError("Call compute_all_importances first")

        dfs = []
        for method, imp in self._importances.items():
            df = imp.to_dataframe()
            df['method'] = method
            df['rank'] = range(1, len(df) + 1)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


class FeatureSelectionReport:
    """
    피처 선택 리포트 생성기

    Example:
        >>> report = FeatureSelectionReport()
        >>> report.generate(importances, output_dir='results/feature_selection/')
    """

    def __init__(self):
        self.timestamp = datetime.now().isoformat()

    def generate(
        self,
        importances: Dict[str, FeatureImportance],
        selected_features: List[str],
        all_features: List[str],
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        리포트 생성

        Args:
            importances: 방법별 중요도
            selected_features: 선택된 피처
            all_features: 전체 피처
            output_dir: 저장 디렉토리

        Returns:
            리포트 딕셔너리
        """
        removed_features = [f for f in all_features if f not in selected_features]

        report = {
            'timestamp': self.timestamp,
            'summary': {
                'total_features': len(all_features),
                'selected_features': len(selected_features),
                'removed_features': len(removed_features),
                'reduction_rate': len(removed_features) / len(all_features) * 100
            },
            'selected': selected_features,
            'removed': removed_features,
            'importance_by_method': {}
        }

        for method, imp in importances.items():
            df = imp.to_dataframe()
            report['importance_by_method'][method] = {
                'top_10': df.head(10).to_dict('records'),
                'bottom_10': df.tail(10).to_dict('records')
            }

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # JSON 리포트 저장
            with open(output_path / 'feature_selection_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # CSV 저장 (각 방법별)
            for method, imp in importances.items():
                df = imp.to_dataframe()
                df.to_csv(output_path / f'importance_{method}.csv', index=False)

        return report


def mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """MSE 메트릭 (Permutation Importance용)"""
    return float(torch.mean((predictions - targets) ** 2))


def mae_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """MAE 메트릭"""
    return float(torch.mean(torch.abs(predictions - targets)))


def create_feature_selector(
    model: nn.Module,
    metric: str = 'mse',
    methods: List[str] = None,
    **kwargs
) -> AutoFeatureSelector:
    """
    피처 선택기 팩토리 함수

    Args:
        model: PyTorch 모델
        metric: 메트릭 이름 ('mse', 'mae')
        methods: 중요도 계산 방법
        **kwargs: AutoFeatureSelector 추가 인자

    Returns:
        AutoFeatureSelector 인스턴스
    """
    metric_fn = mse_metric if metric == 'mse' else mae_metric
    methods = methods or ['permutation', 'gradient']

    return AutoFeatureSelector(
        model=model,
        metric=metric_fn,
        methods=methods,
        **kwargs
    )


def quick_feature_importance(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: List[str],
    method: str = 'gradient'
) -> pd.DataFrame:
    """
    빠른 피처 중요도 계산

    Args:
        model: PyTorch 모델
        X: 입력 텐서
        y: 타겟 텐서
        feature_names: 피처 이름
        method: 'gradient' 또는 'permutation'

    Returns:
        중요도 DataFrame
    """
    if method == 'gradient':
        gi = GradientImportance(model)
        importance = gi.compute(X, feature_names)
    elif method == 'permutation':
        pi = PermutationImportance(model, mse_metric, n_repeats=5)
        importance = pi.compute(X, y, feature_names)
    else:
        raise ValueError(f"Unknown method: {method}")

    return importance.to_dataframe()
