"""
XAI (Explainable AI) 테스트 (Task 23)
=====================================
Explainability 모듈 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 테스트용 모델
# ============================================================================

class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AttentionModel(nn.Module):
    """어텐션이 있는 테스트 모델"""

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self._attention_weights = None

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        h = self.fc1(x).unsqueeze(1)
        attn_out, self._attention_weights = self.attention(h, h, h)
        out = self.fc2(attn_out.squeeze(1))
        return out

    def get_attention_weights(self):
        return self._attention_weights


@pytest.fixture
def simple_model():
    """간단한 모델 fixture"""
    model = SimpleModel(input_size=10, hidden_size=32, output_size=1)
    model.eval()
    return model


@pytest.fixture
def attention_model():
    """어텐션 모델 fixture"""
    model = AttentionModel(input_size=10, hidden_size=32, output_size=1)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """샘플 입력"""
    torch.manual_seed(42)
    return torch.randn(1, 10)


@pytest.fixture
def feature_names():
    """피처 이름"""
    return [f"feature_{i}" for i in range(10)]


# ============================================================================
# 데이터클래스 테스트
# ============================================================================

class TestDataClasses:
    """데이터클래스 테스트"""

    def test_feature_contribution_creation(self):
        """FeatureContribution 생성"""
        from src.analysis.explainability import FeatureContribution

        contrib = FeatureContribution(
            feature_name="temperature",
            contribution=0.5,
            base_value=50.0,
            feature_value=25.0
        )

        assert contrib.feature_name == "temperature"
        assert contrib.contribution == 0.5
        assert contrib.base_value == 50.0
        assert contrib.feature_value == 25.0

    def test_feature_contribution_to_dict(self):
        """FeatureContribution 딕셔너리 변환"""
        from src.analysis.explainability import FeatureContribution

        contrib = FeatureContribution(
            feature_name="temperature",
            contribution=0.5,
            base_value=50.0,
            feature_value=25.0
        )

        result = contrib.to_dict()

        assert 'feature_name' in result
        assert result['contribution'] == 0.5
        assert result['base_value'] == 50.0
        assert result['feature_value'] == 25.0

    def test_prediction_explanation_creation(self):
        """PredictionExplanation 생성"""
        from src.analysis.explainability import PredictionExplanation, FeatureContribution

        contributions = [
            FeatureContribution("temp", 0.5, 50.0, 25.0),
            FeatureContribution("humidity", 0.3, 50.0, 60.0)
        ]

        explanation = PredictionExplanation(
            prediction=100.0,
            base_value=50.0,
            contributions=contributions,
            method="gradient"
        )

        assert explanation.prediction == 100.0
        assert explanation.base_value == 50.0
        assert len(explanation.contributions) == 2
        assert explanation.method == "gradient"

    def test_prediction_explanation_top_contributors(self):
        """상위 기여 피처 추출"""
        from src.analysis.explainability import PredictionExplanation, FeatureContribution

        contributions = [
            FeatureContribution("f1", 0.1, 50.0, 1.0),
            FeatureContribution("f2", 0.5, 50.0, 2.0),
            FeatureContribution("f3", -0.3, 50.0, 3.0),
            FeatureContribution("f4", 0.2, 50.0, 4.0),
        ]

        explanation = PredictionExplanation(
            prediction=100.0,
            base_value=50.0,
            contributions=contributions,
            method="gradient"
        )

        top = explanation.top_contributors(n=2)

        assert len(top) == 2
        assert top[0].feature_name == "f2"  # 가장 높은 절대값 기여도

    def test_prediction_explanation_positive_contributors(self):
        """양의 기여 피처 필터"""
        from src.analysis.explainability import PredictionExplanation, FeatureContribution

        contributions = [
            FeatureContribution("f1", 0.5, 50.0, 1.0),
            FeatureContribution("f2", -0.3, 50.0, 2.0),
            FeatureContribution("f3", 0.2, 50.0, 3.0),
        ]

        explanation = PredictionExplanation(
            prediction=100.0,
            base_value=50.0,
            contributions=contributions,
            method="gradient"
        )

        positive = explanation.positive_contributors()
        assert len(positive) == 2
        assert all(c.contribution > 0 for c in positive)

    def test_prediction_explanation_negative_contributors(self):
        """음의 기여 피처 필터"""
        from src.analysis.explainability import PredictionExplanation, FeatureContribution

        contributions = [
            FeatureContribution("f1", 0.5, 50.0, 1.0),
            FeatureContribution("f2", -0.3, 50.0, 2.0),
            FeatureContribution("f3", -0.2, 50.0, 3.0),
        ]

        explanation = PredictionExplanation(
            prediction=100.0,
            base_value=50.0,
            contributions=contributions,
            method="gradient"
        )

        negative = explanation.negative_contributors()
        assert len(negative) == 2
        assert all(c.contribution < 0 for c in negative)


# ============================================================================
# Gradient Explainer 테스트
# ============================================================================

class TestGradientExplainer:
    """Gradient Explainer 테스트"""

    def test_explainer_creation(self, simple_model):
        """Explainer 생성"""
        from src.analysis.explainability import GradientExplainer

        explainer = GradientExplainer(simple_model)
        assert explainer.model is not None

    def test_explain(self, simple_model, sample_input, feature_names):
        """설명 생성"""
        from src.analysis.explainability import GradientExplainer

        explainer = GradientExplainer(simple_model)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation.method == "gradient"
        assert len(explanation.contributions) == len(feature_names)
        assert explanation.prediction is not None

    def test_explain_with_baseline(self, simple_model, sample_input, feature_names):
        """baseline 지정"""
        from src.analysis.explainability import GradientExplainer

        baseline = torch.zeros_like(sample_input)
        explainer = GradientExplainer(simple_model, baseline=baseline)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation is not None
        assert len(explanation.contributions) == len(feature_names)


# ============================================================================
# Integrated Gradients Explainer 테스트
# ============================================================================

class TestIntegratedGradientsExplainer:
    """Integrated Gradients Explainer 테스트"""

    def test_explainer_creation(self, simple_model):
        """Explainer 생성"""
        from src.analysis.explainability import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(simple_model, steps=50)
        assert explainer.steps == 50

    def test_explain(self, simple_model, sample_input, feature_names):
        """설명 생성"""
        from src.analysis.explainability import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(simple_model, steps=20)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation.method == "integrated_gradients"
        assert len(explanation.contributions) == len(feature_names)

    def test_explain_with_baseline(self, simple_model, sample_input, feature_names):
        """커스텀 baseline 사용"""
        from src.analysis.explainability import IntegratedGradientsExplainer

        baseline = torch.zeros_like(sample_input)
        explainer = IntegratedGradientsExplainer(simple_model, steps=20, baseline=baseline)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation is not None


# ============================================================================
# Perturbation Explainer 테스트
# ============================================================================

class TestPerturbationExplainer:
    """Perturbation Explainer 테스트"""

    def test_explainer_creation(self, simple_model):
        """Explainer 생성"""
        from src.analysis.explainability import PerturbationExplainer

        explainer = PerturbationExplainer(simple_model, n_samples=50)
        assert explainer.n_samples == 50

    def test_explain(self, simple_model, sample_input, feature_names):
        """설명 생성"""
        from src.analysis.explainability import PerturbationExplainer

        explainer = PerturbationExplainer(simple_model, n_samples=30)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation.method == "perturbation"
        assert len(explanation.contributions) == len(feature_names)

    def test_perturbation_std(self, simple_model, sample_input, feature_names):
        """섭동 표준편차 설정"""
        from src.analysis.explainability import PerturbationExplainer

        explainer = PerturbationExplainer(
            simple_model,
            n_samples=20,
            perturbation_std=0.2
        )
        assert explainer.perturbation_std == 0.2

        explanation = explainer.explain(sample_input, feature_names)
        assert explanation is not None


# ============================================================================
# SHAP Explainer 테스트
# ============================================================================

class TestSHAPExplainer:
    """SHAP Explainer 테스트"""

    def test_explainer_creation(self, simple_model):
        """Explainer 생성"""
        from src.analysis.explainability import SHAPExplainer

        torch.manual_seed(42)
        background = torch.randn(10, 10)

        explainer = SHAPExplainer(simple_model, background_data=background)
        assert explainer is not None

    @pytest.mark.skipif(True, reason="SHAP can be slow and may not be installed")
    def test_explain(self, simple_model, sample_input, feature_names):
        """설명 생성"""
        from src.analysis.explainability import SHAPExplainer

        torch.manual_seed(42)
        background = torch.randn(10, 10)

        explainer = SHAPExplainer(simple_model, background_data=background)
        explanation = explainer.explain(sample_input, feature_names)

        assert explanation.method == "shap"
        assert len(explanation.contributions) == len(feature_names)


# ============================================================================
# Attention Explainer 테스트
# ============================================================================

class TestAttentionExplainer:
    """Attention Explainer 테스트"""

    def test_explainer_creation(self, attention_model):
        """Explainer 생성"""
        from src.analysis.explainability import AttentionExplainer

        explainer = AttentionExplainer(attention_model)
        assert explainer is not None

    def test_get_attention_weights(self, attention_model, sample_input):
        """어텐션 가중치 추출"""
        from src.analysis.explainability import AttentionExplainer

        explainer = AttentionExplainer(attention_model)

        # Forward pass to generate attention weights
        with torch.no_grad():
            _ = attention_model(sample_input)

        weights = explainer.get_attention_weights(sample_input)

        assert weights is not None
        assert isinstance(weights, dict)


# ============================================================================
# Explanation Report 테스트
# ============================================================================

class TestExplanationReport:
    """Explanation Report 테스트"""

    def test_report_creation(self):
        """리포트 생성"""
        from src.analysis.explainability import ExplanationReport

        report = ExplanationReport()
        assert report is not None
        assert len(report.explanations) == 0

    def test_report_add_explanation(self):
        """설명 추가"""
        from src.analysis.explainability import (
            ExplanationReport, PredictionExplanation, FeatureContribution
        )

        contributions = [
            FeatureContribution("temp", 0.5, 50.0, 25.0),
            FeatureContribution("humidity", -0.3, 50.0, 60.0)
        ]

        explanation = PredictionExplanation(
            prediction=100.0,
            base_value=50.0,
            contributions=contributions,
            method="gradient"
        )

        report = ExplanationReport()
        report.add_explanation(explanation)

        assert len(report.explanations) == 1

    def test_report_to_dict(self):
        """리포트 딕셔너리 변환"""
        from src.analysis.explainability import (
            ExplanationReport, PredictionExplanation, FeatureContribution
        )

        contributions = [FeatureContribution("temp", 0.5, 50.0, 25.0)]
        explanation = PredictionExplanation(100.0, 50.0, contributions, "gradient")

        report = ExplanationReport()
        report.add_explanation(explanation)

        result = report.to_dict()

        assert 'metadata' in result
        assert 'explanations' in result
        assert 'summary' in result

    def test_report_summary(self):
        """리포트 요약"""
        from src.analysis.explainability import (
            ExplanationReport, PredictionExplanation, FeatureContribution
        )

        contributions = [
            FeatureContribution("f1", 0.5, 50.0, 1.0),
            FeatureContribution("f2", 0.3, 50.0, 2.0),
            FeatureContribution("f3", -0.1, 50.0, 3.0),
        ]
        explanation = PredictionExplanation(100.0, 50.0, contributions, "gradient")

        report = ExplanationReport()
        report.add_explanation(explanation)

        summary = report.get_summary()

        assert 'n_explanations' in summary
        assert 'top_positive_features' in summary
        assert 'top_negative_features' in summary
        assert summary['n_explanations'] == 1

    def test_report_save_json(self):
        """JSON 저장"""
        from src.analysis.explainability import (
            ExplanationReport, PredictionExplanation, FeatureContribution
        )

        contributions = [FeatureContribution("temp", 0.5, 50.0, 25.0)]
        explanation = PredictionExplanation(100.0, 50.0, contributions, "gradient")

        report = ExplanationReport()
        report.add_explanation(explanation)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "explanation_report.json"
            report.save(str(json_path))

            assert json_path.exists()


# ============================================================================
# 팩토리 함수 테스트
# ============================================================================

class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_gradient_explainer(self, simple_model):
        """Gradient explainer 생성"""
        from src.analysis.explainability import create_explainer

        explainer = create_explainer('gradient', simple_model)
        assert explainer is not None

    def test_create_integrated_gradients_explainer(self, simple_model):
        """Integrated gradients explainer 생성"""
        from src.analysis.explainability import create_explainer

        explainer = create_explainer('integrated_gradients', simple_model, steps=30)
        assert explainer.steps == 30

    def test_create_perturbation_explainer(self, simple_model):
        """Perturbation explainer 생성"""
        from src.analysis.explainability import create_explainer

        explainer = create_explainer('perturbation', simple_model, n_samples=50)
        assert explainer.n_samples == 50

    def test_create_unknown_explainer(self, simple_model):
        """알 수 없는 explainer"""
        from src.analysis.explainability import create_explainer

        with pytest.raises(ValueError):
            create_explainer('unknown_method', simple_model)

    def test_explain_prediction_function(self, simple_model, sample_input, feature_names):
        """간편 설명 함수"""
        from src.analysis.explainability import explain_prediction

        explanation = explain_prediction(
            model=simple_model,
            inputs=sample_input,
            feature_names=feature_names,
            method='gradient'
        )

        assert explanation is not None
        assert explanation.method == 'gradient'


# ============================================================================
# 통합 테스트
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_explanation_pipeline(self, simple_model, sample_input, feature_names):
        """전체 설명 파이프라인"""
        from src.analysis.explainability import (
            GradientExplainer,
            IntegratedGradientsExplainer,
            ExplanationReport
        )

        # 여러 방법으로 설명 생성
        grad_explainer = GradientExplainer(simple_model)
        ig_explainer = IntegratedGradientsExplainer(simple_model, steps=20)

        # clone().detach()로 독립적인 텐서 사용
        grad_exp = grad_explainer.explain(sample_input.clone().detach(), feature_names)
        ig_exp = ig_explainer.explain(sample_input.clone().detach(), feature_names)

        # 리포트 생성
        report = ExplanationReport()
        report.add_explanation(grad_exp)
        report.add_explanation(ig_exp)

        # 요약 확인
        summary = report.get_summary()
        assert summary['n_explanations'] == 2

    def test_compare_explanation_methods(self, simple_model, sample_input, feature_names):
        """설명 방법 비교"""
        from src.analysis.explainability import (
            GradientExplainer,
            PerturbationExplainer
        )

        # clone().detach()로 독립적인 텐서 사용
        grad_exp = GradientExplainer(simple_model).explain(sample_input.clone().detach(), feature_names)
        pert_exp = PerturbationExplainer(simple_model, n_samples=30).explain(
            sample_input.clone().detach(), feature_names
        )

        # 두 방법 모두 설명 생성
        assert len(grad_exp.contributions) == len(pert_exp.contributions)

        # 상위 기여자 확인
        grad_top = grad_exp.top_contributors(3)
        pert_top = pert_exp.top_contributors(3)

        assert len(grad_top) == 3
        assert len(pert_top) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
