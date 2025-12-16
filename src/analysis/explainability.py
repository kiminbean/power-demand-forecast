"""
설명 가능한 AI (XAI) 모듈 (Task 23)
====================================

모델 예측의 해석 가능성을 제공합니다.

주요 기능:
1. SHAP 기반 피처 기여도
2. LIME 로컬 설명
3. Attention Weight 시각화
4. 피처 중요도 리포트

Author: Claude Code
Date: 2025-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """피처 기여도"""
    feature_name: str
    contribution: float  # 기여도 값 (양수: 증가, 음수: 감소)
    base_value: float    # 기준 값
    feature_value: float # 실제 피처 값

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'contribution': self.contribution,
            'base_value': self.base_value,
            'feature_value': self.feature_value
        }


@dataclass
class PredictionExplanation:
    """예측 설명"""
    prediction: float
    base_value: float
    contributions: List[FeatureContribution]
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction': self.prediction,
            'base_value': self.base_value,
            'contributions': [c.to_dict() for c in self.contributions],
            'method': self.method,
            'timestamp': self.timestamp
        }

    def top_contributors(self, n: int = 5) -> List[FeatureContribution]:
        """상위 기여 피처"""
        sorted_contribs = sorted(
            self.contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        return sorted_contribs[:n]

    def positive_contributors(self) -> List[FeatureContribution]:
        """양의 기여 피처"""
        return [c for c in self.contributions if c.contribution > 0]

    def negative_contributors(self) -> List[FeatureContribution]:
        """음의 기여 피처"""
        return [c for c in self.contributions if c.contribution < 0]


# ============================================================================
# Gradient 기반 설명
# ============================================================================

class GradientExplainer:
    """
    Gradient 기반 설명

    입력에 대한 그래디언트를 사용하여 피처 기여도를 계산합니다.

    Args:
        model: PyTorch 모델
        baseline: 기준 입력 (None이면 0)

    Example:
        >>> explainer = GradientExplainer(model)
        >>> explanation = explainer.explain(input_tensor, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        baseline: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.baseline = baseline

    def explain(
        self,
        inputs: torch.Tensor,
        feature_names: List[str],
        target_idx: int = 0
    ) -> PredictionExplanation:
        """
        입력에 대한 설명 생성

        Args:
            inputs: 입력 텐서 (1, seq_len, features)
            feature_names: 피처 이름 리스트
            target_idx: 타겟 출력 인덱스

        Returns:
            PredictionExplanation
        """
        device = next(self.model.parameters()).device
        inputs = inputs.to(device).requires_grad_(True)

        self.model.eval()

        # Forward pass
        output = self.model(inputs)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:
            output = output[:, -1, target_idx]
        else:
            output = output[:, target_idx] if output.dim() == 2 else output

        prediction = output.item() if output.numel() == 1 else output[0].item()

        # Backward pass
        output.sum().backward()

        # Gradient * Input (integrated gradients 근사)
        gradients = inputs.grad
        contributions = (gradients * inputs).detach()

        # 마지막 시점의 기여도 사용
        if contributions.dim() == 3:
            contributions = contributions[0, -1, :]  # (features,)
        else:
            contributions = contributions[0, :]

        contributions = contributions.cpu().numpy()

        # 기준 값 (평균 예측)
        base_value = prediction - contributions.sum()

        # FeatureContribution 생성
        feature_contribs = []
        inputs_np = inputs.detach().cpu().numpy()

        for i, name in enumerate(feature_names):
            feature_contribs.append(FeatureContribution(
                feature_name=name,
                contribution=float(contributions[i]),
                base_value=float(base_value),
                feature_value=float(inputs_np[0, -1, i]) if inputs_np.ndim == 3 else float(inputs_np[0, i])
            ))

        return PredictionExplanation(
            prediction=prediction,
            base_value=base_value,
            contributions=feature_contribs,
            method='gradient'
        )


class IntegratedGradientsExplainer:
    """
    Integrated Gradients 설명

    기준 입력에서 실제 입력까지의 그래디언트를 적분합니다.

    Args:
        model: PyTorch 모델
        steps: 적분 스텝 수

    Example:
        >>> explainer = IntegratedGradientsExplainer(model, steps=50)
        >>> explanation = explainer.explain(input_tensor, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.steps = steps
        self.baseline = baseline

    def explain(
        self,
        inputs: torch.Tensor,
        feature_names: List[str],
        target_idx: int = 0
    ) -> PredictionExplanation:
        """Integrated Gradients 계산"""
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        # 기준 입력 (0 또는 지정된 값)
        if self.baseline is None:
            baseline = torch.zeros_like(inputs)
        else:
            baseline = self.baseline.to(device)

        self.model.eval()

        # 경로 적분
        scaled_inputs = [
            baseline + (float(i) / self.steps) * (inputs - baseline)
            for i in range(self.steps + 1)
        ]

        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().requires_grad_(True)

            output = self.model(scaled_input)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                output = output[:, -1, target_idx]
            elif output.dim() == 2:
                output = output[:, target_idx]

            output.sum().backward()
            gradients.append(scaled_input.grad.clone())

        # 평균 그래디언트
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated Gradients = (input - baseline) * avg_gradients
        integrated_grads = (inputs - baseline) * avg_gradients

        # 마지막 시점의 기여도 사용
        if integrated_grads.dim() == 3:
            contributions = integrated_grads[0, -1, :].cpu().numpy()
            input_values = inputs[0, -1, :].cpu().numpy()
        else:
            contributions = integrated_grads[0, :].cpu().numpy()
            input_values = inputs[0, :].cpu().numpy()

        # 예측 값
        with torch.no_grad():
            output = self.model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                prediction = output[0, -1, target_idx].item()
            elif output.dim() == 2:
                prediction = output[0, target_idx].item()
            else:
                prediction = output.item()

        base_value = prediction - contributions.sum()

        feature_contribs = []
        for i, name in enumerate(feature_names):
            feature_contribs.append(FeatureContribution(
                feature_name=name,
                contribution=float(contributions[i]),
                base_value=float(base_value),
                feature_value=float(input_values[i])
            ))

        return PredictionExplanation(
            prediction=prediction,
            base_value=base_value,
            contributions=feature_contribs,
            method='integrated_gradients'
        )


# ============================================================================
# Perturbation 기반 설명
# ============================================================================

class PerturbationExplainer:
    """
    Perturbation 기반 설명

    피처를 교란하여 예측 변화를 측정합니다.

    Args:
        model: PyTorch 모델
        n_samples: 샘플 수
        perturbation_std: 교란 표준편차

    Example:
        >>> explainer = PerturbationExplainer(model, n_samples=100)
        >>> explanation = explainer.explain(input_tensor, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        perturbation_std: float = 0.1
    ):
        self.model = model
        self.n_samples = n_samples
        self.perturbation_std = perturbation_std

    def explain(
        self,
        inputs: torch.Tensor,
        feature_names: List[str],
        target_idx: int = 0
    ) -> PredictionExplanation:
        """Perturbation 기반 설명"""
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        self.model.eval()

        # 원본 예측
        with torch.no_grad():
            output = self.model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                original_pred = output[0, -1, target_idx].item()
            elif output.dim() == 2:
                original_pred = output[0, target_idx].item()
            else:
                original_pred = output.item()

        n_features = inputs.shape[-1]
        importance_scores = []

        for feat_idx in range(n_features):
            diffs = []

            for _ in range(self.n_samples):
                # 피처 교란
                perturbed = inputs.clone()
                noise = torch.randn(1, device=device) * self.perturbation_std * inputs[..., feat_idx].std()
                perturbed[..., feat_idx] = perturbed[..., feat_idx] + noise

                # 교란된 예측
                with torch.no_grad():
                    output = self.model(perturbed)
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.dim() == 3:
                        perturbed_pred = output[0, -1, target_idx].item()
                    elif output.dim() == 2:
                        perturbed_pred = output[0, target_idx].item()
                    else:
                        perturbed_pred = output.item()

                diffs.append(abs(original_pred - perturbed_pred))

            importance_scores.append(np.mean(diffs))

        # 정규화
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]

        # 입력 값
        if inputs.dim() == 3:
            input_values = inputs[0, -1, :].cpu().numpy()
        else:
            input_values = inputs[0, :].cpu().numpy()

        # 기여도 (중요도 * 부호)
        mean_values = inputs.mean().item()
        contributions = []
        for i, score in enumerate(importance_scores):
            sign = 1 if input_values[i] > mean_values else -1
            contributions.append(score * sign * (original_pred - 0))

        base_value = original_pred - sum(contributions)

        feature_contribs = []
        for i, name in enumerate(feature_names):
            feature_contribs.append(FeatureContribution(
                feature_name=name,
                contribution=float(contributions[i]),
                base_value=float(base_value),
                feature_value=float(input_values[i])
            ))

        return PredictionExplanation(
            prediction=original_pred,
            base_value=base_value,
            contributions=feature_contribs,
            method='perturbation'
        )


# ============================================================================
# SHAP Wrapper
# ============================================================================

class SHAPExplainer:
    """
    SHAP 기반 설명

    SHAP 라이브러리를 활용하여 피처 기여도를 계산합니다.

    Args:
        model: PyTorch 모델
        background_data: 배경 데이터

    Example:
        >>> explainer = SHAPExplainer(model, background_data)
        >>> explanation = explainer.explain(input_tensor, feature_names)
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.background_data = background_data
        self._shap = None
        self._check_shap()

    def _check_shap(self):
        """SHAP 라이브러리 확인"""
        try:
            import shap
            self._shap = shap
        except ImportError:
            warnings.warn("SHAP not installed. Install with: pip install shap")

    def _model_fn(self, x: np.ndarray) -> np.ndarray:
        """SHAP용 모델 래퍼"""
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                output = output[:, -1, :]

        return output.cpu().numpy()

    def explain(
        self,
        inputs: torch.Tensor,
        feature_names: List[str],
        target_idx: int = 0
    ) -> PredictionExplanation:
        """SHAP 설명"""
        if self._shap is None:
            raise ImportError("SHAP not installed. pip install shap")

        inputs_np = inputs.cpu().numpy()

        # 마지막 시점만 사용 (2D로 변환)
        if inputs_np.ndim == 3:
            inputs_2d = inputs_np[:, -1, :]
        else:
            inputs_2d = inputs_np

        # 배경 데이터
        if self.background_data is not None:
            bg = self.background_data.cpu().numpy()
            if bg.ndim == 3:
                bg = bg[:, -1, :]
        else:
            bg = np.zeros((1, inputs_2d.shape[1]))

        # 2D 모델 래퍼
        def model_2d(x):
            if x.ndim == 2:
                x_3d = np.repeat(x[:, np.newaxis, :], inputs_np.shape[1] if inputs_np.ndim == 3 else 1, axis=1)
            else:
                x_3d = x
            return self._model_fn(x_3d)[:, target_idx]

        # SHAP 계산
        explainer = self._shap.KernelExplainer(model_2d, bg)
        shap_values = explainer.shap_values(inputs_2d)

        # 예측 값
        device = next(self.model.parameters()).device
        with torch.no_grad():
            output = self.model(inputs.to(device))
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                prediction = output[0, -1, target_idx].item()
            elif output.dim() == 2:
                prediction = output[0, target_idx].item()
            else:
                prediction = output.item()

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
        else:
            base_value = float(base_value)

        contributions = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        feature_contribs = []
        for i, name in enumerate(feature_names):
            feature_contribs.append(FeatureContribution(
                feature_name=name,
                contribution=float(contributions[i]),
                base_value=base_value,
                feature_value=float(inputs_2d[0, i])
            ))

        return PredictionExplanation(
            prediction=prediction,
            base_value=base_value,
            contributions=feature_contribs,
            method='shap'
        )


# ============================================================================
# Attention 시각화
# ============================================================================

class AttentionExplainer:
    """
    Attention Weight 기반 설명

    Transformer 모델의 Attention 가중치를 시각화합니다.

    Args:
        model: Attention이 있는 PyTorch 모델

    Example:
        >>> explainer = AttentionExplainer(model)
        >>> weights = explainer.get_attention_weights(input_tensor)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_weights = []

    def get_attention_weights(
        self,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Attention 가중치 추출"""
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        self.model.eval()

        # 후크로 attention 가중치 수집
        attention_weights = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    attention_weights[name] = module.attention_weights.detach()
            return hook

        # Attention 모듈에 후크 등록
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            self.model(inputs)

        # 후크 제거
        for hook in hooks:
            hook.remove()

        return attention_weights

    def explain_with_attention(
        self,
        inputs: torch.Tensor,
        feature_names: List[str],
        time_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Attention 기반 설명"""
        attention_weights = self.get_attention_weights(inputs)

        if not attention_weights:
            return {'error': 'No attention weights found'}

        # 시간별 중요도 계산
        time_importance = {}
        for name, weights in attention_weights.items():
            # (batch, heads, seq, seq) -> (seq,) 평균
            if weights.dim() == 4:
                avg_weights = weights.mean(dim=(0, 1)).mean(dim=0)
            elif weights.dim() == 3:
                avg_weights = weights.mean(dim=0).mean(dim=0)
            else:
                avg_weights = weights.mean(dim=0)

            time_importance[name] = avg_weights.cpu().numpy().tolist()

        return {
            'attention_weights': {k: v.cpu().numpy().tolist() for k, v in attention_weights.items()},
            'time_importance': time_importance,
            'feature_names': feature_names
        }


# ============================================================================
# 설명 리포트 생성
# ============================================================================

class ExplanationReport:
    """
    설명 리포트 생성기

    모델 예측에 대한 종합적인 설명 리포트를 생성합니다.

    Example:
        >>> report = ExplanationReport()
        >>> report.add_explanation(explanation)
        >>> report.save('report.json')
    """

    def __init__(self):
        self.explanations: List[PredictionExplanation] = []
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat()
        }

    def add_explanation(self, explanation: PredictionExplanation) -> None:
        """설명 추가"""
        self.explanations.append(explanation)

    def get_summary(self) -> Dict[str, Any]:
        """요약 생성"""
        if not self.explanations:
            return {}

        all_contributions = {}
        for exp in self.explanations:
            for contrib in exp.contributions:
                if contrib.feature_name not in all_contributions:
                    all_contributions[contrib.feature_name] = []
                all_contributions[contrib.feature_name].append(contrib.contribution)

        # 평균 기여도
        avg_contributions = {
            name: np.mean(values)
            for name, values in all_contributions.items()
        }

        # 정렬
        sorted_features = sorted(
            avg_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            'n_explanations': len(self.explanations),
            'top_positive_features': [
                (name, float(val)) for name, val in sorted_features[:5] if val > 0
            ],
            'top_negative_features': [
                (name, float(val)) for name, val in sorted_features if val < 0
            ][:5],
            'avg_prediction': float(np.mean([e.prediction for e in self.explanations])),
            'avg_base_value': float(np.mean([e.base_value for e in self.explanations]))
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'metadata': self.metadata,
            'summary': self.get_summary(),
            'explanations': [e.to_dict() for e in self.explanations]
        }

    def save(self, path: str) -> None:
        """파일로 저장"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Explanation report saved to {path}")


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_explainer(
    method: str,
    model: nn.Module,
    **kwargs
) -> Union[GradientExplainer, IntegratedGradientsExplainer, PerturbationExplainer, SHAPExplainer]:
    """
    설명기 팩토리 함수

    Args:
        method: 'gradient', 'integrated_gradients', 'perturbation', 'shap'
        model: PyTorch 모델

    Returns:
        해당 설명기 인스턴스
    """
    explainers = {
        'gradient': GradientExplainer,
        'integrated_gradients': IntegratedGradientsExplainer,
        'perturbation': PerturbationExplainer,
        'shap': SHAPExplainer
    }

    explainer_class = explainers.get(method)
    if explainer_class is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(explainers.keys())}")

    return explainer_class(model, **kwargs)


def explain_prediction(
    model: nn.Module,
    inputs: torch.Tensor,
    feature_names: List[str],
    method: str = 'gradient',
    **kwargs
) -> PredictionExplanation:
    """
    간편한 예측 설명 함수

    Args:
        model: PyTorch 모델
        inputs: 입력 텐서
        feature_names: 피처 이름 리스트
        method: 설명 방법

    Returns:
        PredictionExplanation
    """
    explainer = create_explainer(method, model, **kwargs)
    return explainer.explain(inputs, feature_names)
