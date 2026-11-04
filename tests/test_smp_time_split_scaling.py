"""Tests for leak-free scaling utilities in SMP training scripts.

SMP 학습 스크립트의 스케일러 누출 방지(시간 분할 이전 fit 금지) 회귀 테스트.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.smp.models.time_split_scaling import (
    carve_validation_from_train,
    fit_transform_time_split,
    temporal_split_bounds,
)


class TestFitTransformTimeSplit:
    """fit_transform_time_split 테스트."""

    def test_fits_on_train_slice_only(self):
        """스케일러 통계가 훈련 구간 통계와 동일해야 한다."""
        rng = np.random.default_rng(42)
        X = rng.normal(loc=100.0, scale=20.0, size=(500, 4))
        X[400:] += 500.0  # test 구간에 큰 이동 부여 → 누출 시 통계 왜곡

        scaler = StandardScaler()
        scaled = fit_transform_time_split(scaler, X, train_end=350)

        assert scaler.mean_ == pytest.approx(X[:350].mean(axis=0))
        assert scaler.scale_ == pytest.approx(X[:350].std(axis=0))

    def test_matches_manual_fit_transform(self):
        """수동(fit→transform 분리) 결과와 정확히 일치해야 한다."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 3))
        train_end = 210

        scaler = StandardScaler()
        scaled = fit_transform_time_split(scaler, X, train_end)

        ref_scaler = StandardScaler()
        expected = np.empty_like(X, dtype=np.float64)
        expected[:train_end] = ref_scaler.fit_transform(X[:train_end])
        expected[train_end:] = ref_scaler.transform(X[train_end:])

        np.testing.assert_allclose(scaled, expected)

    def test_output_shape_preserved(self):
        scaled = fit_transform_time_split(StandardScaler(), np.arange(40.0).reshape(20, 2), 12)
        assert scaled.shape == (20, 2)

    def test_invalid_train_end_raises(self):
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="train_end"):
            fit_transform_time_split(StandardScaler(), X, 0)
        with pytest.raises(ValueError, match="train_end"):
            fit_transform_time_split(StandardScaler(), X, 11)


class TestTemporalSplitBounds:
    """temporal_split_bounds 테스트."""

    def test_default_split(self):
        assert temporal_split_bounds(100, 0.8, 0.1) == (80, 90)

    def test_70_15_split(self):
        assert temporal_split_bounds(1000, 0.7, 0.15) == (700, 850)

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            temporal_split_bounds(100, 0.9, 0.2)  # 합이 1 이상
        with pytest.raises(ValueError):
            temporal_split_bounds(100, 0.0, 0.1)
        with pytest.raises(ValueError):
            temporal_split_bounds(100, 0.8, 0.0)

    def test_too_small_data_raises(self):
        with pytest.raises(ValueError):
            temporal_split_bounds(2, 0.8, 0.1)


class TestCarveValidationFromTrain:
    """carve_validation_from_train 테스트."""

    def test_carves_tail_fraction(self):
        assert carve_validation_from_train(1000, 0.1) == 100

    def test_val_excluded_from_fit_range(self):
        """fit 구간(train_end - val_size)과 val 구간이 겹치지 않아야 한다."""
        train_start, train_end = 0, 1000
        val_size = carve_validation_from_train(train_end - train_start, 0.1)
        fit_end = train_end - val_size
        assert fit_end == 900
        assert train_end - val_size >= fit_end

    def test_invalid_val_frac_raises(self):
        with pytest.raises(ValueError):
            carve_validation_from_train(100, 1.0)
        with pytest.raises(ValueError):
            carve_validation_from_train(100, -0.1)


class TestTrainingScriptsImportable:
    """수정된 학습 스크립트가 모듈 수준 오류 없이 import되는지 확인."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "src.smp.models.time_split_scaling",
            "src.smp.models.train_smp_v4",
            "src.smp.models.train_smp_v4_enhanced",
            "src.smp.models.train_smp_v4_smp_only",
            "src.smp.models.train_smp_v3_4_kpx_generation",
        ],
    )
    def test_import(self, module_name):
        import importlib

        module = importlib.import_module(module_name)
        assert module is not None
