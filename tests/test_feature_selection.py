"""
자동 피처 선택 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class SimpleModel(nn.Module):
    """테스트용 모델"""
    def __init__(self, input_size: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        return self.fc(x[:, -1, :])


class TestFeatureImportance:
    """FeatureImportance 테스트"""

    def test_creation(self):
        """중요도 객체 생성"""
        from src.features.feature_selection import FeatureImportance

        imp = FeatureImportance(
            feature_names=['a', 'b', 'c'],
            importance_scores=np.array([0.5, 0.3, 0.2]),
            method='test'
        )

        assert len(imp.feature_names) == 3
        assert imp.method == 'test'

    def test_to_dataframe(self):
        """DataFrame 변환"""
        from src.features.feature_selection import FeatureImportance

        imp = FeatureImportance(
            feature_names=['a', 'b', 'c'],
            importance_scores=np.array([0.5, 0.3, 0.2]),
            method='test'
        )

        df = imp.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'feature' in df.columns
        assert 'importance' in df.columns
        # 정렬 확인 (내림차순)
        assert df.iloc[0]['feature'] == 'a'

    def test_top_features(self):
        """상위 피처"""
        from src.features.feature_selection import FeatureImportance

        imp = FeatureImportance(
            feature_names=['a', 'b', 'c', 'd', 'e'],
            importance_scores=np.array([0.5, 0.1, 0.4, 0.2, 0.3]),
            method='test'
        )

        top2 = imp.top_features(n=2)
        assert top2 == ['a', 'c']

    def test_bottom_features(self):
        """하위 피처"""
        from src.features.feature_selection import FeatureImportance

        imp = FeatureImportance(
            feature_names=['a', 'b', 'c', 'd', 'e'],
            importance_scores=np.array([0.5, 0.1, 0.4, 0.2, 0.3]),
            method='test'
        )

        bottom2 = imp.bottom_features(n=2)
        assert 'b' in bottom2


class TestPermutationImportance:
    """Permutation Importance 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.feature_selection import PermutationImportance, mse_metric

        model = SimpleModel()
        pi = PermutationImportance(model, mse_metric, n_repeats=5)

        assert pi.n_repeats == 5

    def test_compute(self):
        """중요도 계산"""
        from src.features.feature_selection import PermutationImportance, mse_metric

        model = SimpleModel(input_size=5)
        pi = PermutationImportance(model, mse_metric, n_repeats=3)

        X = torch.randn(32, 24, 5)
        y = torch.randn(32, 1)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

        importance = pi.compute(X, y, feature_names)

        assert importance.method == 'permutation'
        assert len(importance.importance_scores) == 5
        assert importance.std is not None

    def test_compute_with_device(self):
        """디바이스 지정"""
        from src.features.feature_selection import PermutationImportance, mse_metric

        model = SimpleModel(input_size=3)
        pi = PermutationImportance(model, mse_metric, n_repeats=2)

        X = torch.randn(16, 12, 3)
        y = torch.randn(16, 1)
        feature_names = ['a', 'b', 'c']

        importance = pi.compute(X, y, feature_names, device=torch.device('cpu'))

        assert len(importance.feature_names) == 3


class TestGradientImportance:
    """Gradient Importance 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.feature_selection import GradientImportance

        model = SimpleModel()
        gi = GradientImportance(model, aggregate='mean')

        assert gi.aggregate == 'mean'

    def test_compute_mean(self):
        """평균 집계"""
        from src.features.feature_selection import GradientImportance

        model = SimpleModel(input_size=5)
        gi = GradientImportance(model, aggregate='mean')

        X = torch.randn(32, 24, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

        importance = gi.compute(X, feature_names)

        assert importance.method == 'gradient'
        assert len(importance.importance_scores) == 5

    def test_compute_max(self):
        """최대값 집계"""
        from src.features.feature_selection import GradientImportance

        model = SimpleModel(input_size=3)
        gi = GradientImportance(model, aggregate='max')

        X = torch.randn(16, 12, 3)
        feature_names = ['a', 'b', 'c']

        importance = gi.compute(X, feature_names)

        assert len(importance.importance_scores) == 3

    def test_compute_l2(self):
        """L2 노름 집계"""
        from src.features.feature_selection import GradientImportance

        model = SimpleModel(input_size=4)
        gi = GradientImportance(model, aggregate='l2')

        X = torch.randn(8, 10, 4)
        feature_names = ['a', 'b', 'c', 'd']

        importance = gi.compute(X, feature_names)

        assert len(importance.importance_scores) == 4


class TestAutoFeatureSelector:
    """AutoFeatureSelector 테스트"""

    def test_creation(self):
        """생성"""
        from src.features.feature_selection import AutoFeatureSelector, mse_metric

        model = SimpleModel()
        selector = AutoFeatureSelector(model, mse_metric)

        assert selector.threshold == 0.01
        assert selector.min_features == 5

    def test_compute_all_importances(self):
        """모든 중요도 계산"""
        from src.features.feature_selection import AutoFeatureSelector, mse_metric

        model = SimpleModel(input_size=5)
        selector = AutoFeatureSelector(
            model, mse_metric,
            methods=['permutation', 'gradient']
        )

        X = torch.randn(32, 24, 5)
        y = torch.randn(32, 1)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']

        importances = selector.compute_all_importances(X, y, feature_names)

        assert 'permutation' in importances
        assert 'gradient' in importances

    def test_aggregate_importance(self):
        """중요도 집계"""
        from src.features.feature_selection import (
            AutoFeatureSelector, FeatureImportance, mse_metric
        )

        model = SimpleModel()
        selector = AutoFeatureSelector(model, mse_metric)

        importances = {
            'method1': FeatureImportance(
                feature_names=['a', 'b', 'c'],
                importance_scores=np.array([0.6, 0.3, 0.1]),
                method='method1'
            ),
            'method2': FeatureImportance(
                feature_names=['a', 'b', 'c'],
                importance_scores=np.array([0.4, 0.4, 0.2]),
                method='method2'
            )
        }

        aggregated = selector.aggregate_importance(importances)

        assert aggregated.method == 'aggregated'
        assert len(aggregated.importance_scores) == 3

    def test_select_features(self):
        """피처 선택"""
        from src.features.feature_selection import AutoFeatureSelector, mse_metric

        model = SimpleModel(input_size=10)
        selector = AutoFeatureSelector(
            model, mse_metric,
            methods=['gradient'],
            threshold=0.1,
            min_features=3
        )

        X = torch.randn(32, 24, 10)
        y = torch.randn(32, 1)
        feature_names = [f'f{i}' for i in range(10)]

        selected = selector.select_features(X, y, feature_names)

        assert len(selected) >= 3
        assert all(f in feature_names for f in selected)

    def test_select_features_with_importance(self):
        """중요도와 함께 피처 선택"""
        from src.features.feature_selection import AutoFeatureSelector, mse_metric

        model = SimpleModel(input_size=5)
        selector = AutoFeatureSelector(
            model, mse_metric,
            methods=['gradient'],
            min_features=2
        )

        X = torch.randn(16, 12, 5)
        y = torch.randn(16, 1)
        feature_names = ['a', 'b', 'c', 'd', 'e']

        selected, importance = selector.select_features(
            X, y, feature_names, return_importance=True
        )

        assert len(selected) >= 2
        assert importance is not None

    def test_get_feature_ranking(self):
        """피처 순위"""
        from src.features.feature_selection import AutoFeatureSelector, mse_metric

        model = SimpleModel(input_size=5)
        selector = AutoFeatureSelector(
            model, mse_metric,
            methods=['gradient']
        )

        X = torch.randn(16, 12, 5)
        y = torch.randn(16, 1)
        feature_names = ['a', 'b', 'c', 'd', 'e']

        selector.compute_all_importances(X, y, feature_names)
        ranking = selector.get_feature_ranking()

        assert isinstance(ranking, pd.DataFrame)
        assert 'rank' in ranking.columns


class TestFeatureSelectionReport:
    """FeatureSelectionReport 테스트"""

    def test_generate(self):
        """리포트 생성"""
        from src.features.feature_selection import (
            FeatureSelectionReport, FeatureImportance
        )

        importances = {
            'gradient': FeatureImportance(
                feature_names=['a', 'b', 'c', 'd', 'e'],
                importance_scores=np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
                method='gradient'
            )
        }

        report_gen = FeatureSelectionReport()
        report = report_gen.generate(
            importances,
            selected_features=['a', 'b', 'c'],
            all_features=['a', 'b', 'c', 'd', 'e']
        )

        assert 'summary' in report
        assert report['summary']['total_features'] == 5
        assert report['summary']['selected_features'] == 3
        assert report['summary']['removed_features'] == 2

    def test_generate_with_output(self, tmp_path):
        """파일 저장과 함께 리포트 생성"""
        from src.features.feature_selection import (
            FeatureSelectionReport, FeatureImportance
        )

        importances = {
            'gradient': FeatureImportance(
                feature_names=['a', 'b', 'c'],
                importance_scores=np.array([0.5, 0.3, 0.2]),
                method='gradient'
            )
        }

        report_gen = FeatureSelectionReport()
        report = report_gen.generate(
            importances,
            selected_features=['a', 'b'],
            all_features=['a', 'b', 'c'],
            output_dir=str(tmp_path)
        )

        assert (tmp_path / 'feature_selection_report.json').exists()
        assert (tmp_path / 'importance_gradient.csv').exists()


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_feature_selector(self):
        """피처 선택기 생성"""
        from src.features.feature_selection import create_feature_selector

        model = SimpleModel()
        selector = create_feature_selector(
            model, metric='mse', methods=['gradient']
        )

        assert selector is not None

    def test_quick_feature_importance_gradient(self):
        """빠른 그래디언트 중요도"""
        from src.features.feature_selection import quick_feature_importance

        model = SimpleModel(input_size=5)
        X = torch.randn(16, 12, 5)
        y = torch.randn(16, 1)
        feature_names = ['a', 'b', 'c', 'd', 'e']

        df = quick_feature_importance(model, X, y, feature_names, method='gradient')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_quick_feature_importance_permutation(self):
        """빠른 순열 중요도"""
        from src.features.feature_selection import quick_feature_importance

        model = SimpleModel(input_size=3)
        X = torch.randn(16, 12, 3)
        y = torch.randn(16, 1)
        feature_names = ['a', 'b', 'c']

        df = quick_feature_importance(model, X, y, feature_names, method='permutation')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3


class TestMetricFunctions:
    """메트릭 함수 테스트"""

    def test_mse_metric(self):
        """MSE 메트릭"""
        from src.features.feature_selection import mse_metric

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        mse = mse_metric(pred, target)
        assert mse == 0.0

    def test_mae_metric(self):
        """MAE 메트릭"""
        from src.features.feature_selection import mae_metric

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])

        mae = mae_metric(pred, target)
        assert mae == 1.0


class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline(self):
        """전체 파이프라인"""
        from src.features.feature_selection import (
            AutoFeatureSelector, mse_metric, FeatureSelectionReport
        )

        # 모델 생성
        model = SimpleModel(input_size=10)

        # 피처 선택기
        selector = AutoFeatureSelector(
            model, mse_metric,
            methods=['gradient'],
            threshold=0.05,
            min_features=3
        )

        # 데이터
        X = torch.randn(64, 24, 10)
        y = torch.randn(64, 1)
        feature_names = [f'feature_{i}' for i in range(10)]

        # 중요도 계산
        importances = selector.compute_all_importances(X, y, feature_names)

        # 피처 선택
        selected, agg_imp = selector.select_features(
            X, y, feature_names, return_importance=True
        )

        # 리포트 생성
        report_gen = FeatureSelectionReport()
        report = report_gen.generate(
            importances,
            selected_features=selected,
            all_features=feature_names
        )

        assert len(selected) >= 3
        assert 'summary' in report

    def test_with_lstm_model(self):
        """LSTM 모델과 통합"""
        from src.models.lstm import LSTMModel
        from src.features.feature_selection import quick_feature_importance

        model = LSTMModel(input_size=5, hidden_size=16, num_layers=1)

        X = torch.randn(16, 24, 5)
        y = torch.randn(16, 1)
        feature_names = ['temp', 'humidity', 'wind', 'solar', 'demand']

        df = quick_feature_importance(model, X, y, feature_names, method='gradient')

        assert len(df) == 5
        assert 'importance' in df.columns
