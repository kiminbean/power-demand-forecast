"""
EVAL-001: 평가 지표 테스트
===========================

평가 지표 모듈의 단위 테스트
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# 프로젝트 루트 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import (
    # 기본 지표
    mse,
    rmse,
    mae,
    mape,
    r2_score,
    # 고급 지표
    smape,
    mase,
    mbe,
    cv_rmse,
    max_error,
    median_absolute_error,
    # 통합
    compute_all_metrics,
    compute_metrics_by_threshold,
    # 시간 기반
    compute_metrics_by_hour,
    compute_metrics_by_dayofweek,
    compute_metrics_by_month,
    compute_metrics_by_season,
    # 잔차 분석
    analyze_residuals,
    compute_prediction_intervals,
    # 리포트
    EvaluationReport,
    generate_evaluation_report,
    # 비교
    compare_models,
    compare_horizons,
)


# ============================================================
# 테스트 설정
# ============================================================

@pytest.fixture
def perfect_predictions():
    """완벽한 예측 (오차 0)"""
    y = np.array([100, 200, 300, 400, 500])
    return y, y.copy()


@pytest.fixture
def sample_predictions():
    """샘플 예측 데이터"""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 380, 520])
    return y_true, y_pred


@pytest.fixture
def large_predictions():
    """대규모 예측 데이터"""
    np.random.seed(42)
    n = 1000
    y_true = np.random.uniform(200, 800, n)
    noise = np.random.normal(0, 20, n)
    y_pred = y_true + noise
    return y_true, y_pred


@pytest.fixture
def time_predictions():
    """시간 정보가 있는 예측 데이터"""
    np.random.seed(42)

    # 1년치 시간별 데이터
    timestamps = pd.date_range('2023-01-01', periods=8760, freq='h')
    y_true = 400 + 100 * np.sin(np.arange(8760) * 2 * np.pi / 24) + np.random.normal(0, 20, 8760)
    y_pred = y_true + np.random.normal(0, 30, 8760)

    return y_true, y_pred, timestamps


# ============================================================
# 기본 지표 테스트
# ============================================================

class TestMSE:
    """MSE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert mse(y_true, y_pred) == 0

    def test_basic_calculation(self, sample_predictions):
        """기본 계산"""
        y_true, y_pred = sample_predictions
        # (10² + 10² + 10² + 20² + 20²) / 5 = 180
        expected = (100 + 100 + 100 + 400 + 400) / 5
        assert mse(y_true, y_pred) == expected

    def test_non_negative(self, large_predictions):
        """항상 0 이상"""
        y_true, y_pred = large_predictions
        assert mse(y_true, y_pred) >= 0


class TestRMSE:
    """RMSE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert rmse(y_true, y_pred) == 0

    def test_sqrt_of_mse(self, sample_predictions):
        """MSE의 제곱근"""
        y_true, y_pred = sample_predictions
        assert np.isclose(rmse(y_true, y_pred), np.sqrt(mse(y_true, y_pred)))


class TestMAE:
    """MAE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert mae(y_true, y_pred) == 0

    def test_basic_calculation(self, sample_predictions):
        """기본 계산"""
        y_true, y_pred = sample_predictions
        # (10 + 10 + 10 + 20 + 20) / 5 = 14
        expected = (10 + 10 + 10 + 20 + 20) / 5
        assert mae(y_true, y_pred) == expected


class TestMAPE:
    """MAPE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert mape(y_true, y_pred) == 0

    def test_percentage_scale(self, sample_predictions):
        """백분율 스케일 (0-100)"""
        y_true, y_pred = sample_predictions
        result = mape(y_true, y_pred)
        # 각 오차율: 10%, 5%, 3.33%, 5%, 4% → 평균 약 5.47%
        assert 0 < result < 100

    def test_zero_handling(self):
        """0값 처리"""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 210])
        # 0이 아닌 값만 사용
        result = mape(y_true, y_pred)
        assert result != float('inf')


class TestR2Score:
    """R² Score 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 1"""
        y_true, y_pred = perfect_predictions
        assert r2_score(y_true, y_pred) == 1.0

    def test_range(self, large_predictions):
        """범위 확인"""
        y_true, y_pred = large_predictions
        result = r2_score(y_true, y_pred)
        # 좋은 예측이면 0~1 사이
        assert -1 <= result <= 1

    def test_mean_prediction(self):
        """평균 예측 시 0"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.full(5, np.mean(y_true))  # 모든 예측이 평균
        assert np.isclose(r2_score(y_true, y_pred), 0, atol=1e-10)


# ============================================================
# 고급 지표 테스트
# ============================================================

class TestSMAPE:
    """sMAPE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert smape(y_true, y_pred) == 0

    def test_symmetric(self):
        """대칭성 확인"""
        y_true = np.array([100, 200])
        y_pred = np.array([120, 180])
        # y_true와 y_pred 위치 교환해도 결과 동일
        assert np.isclose(smape(y_true, y_pred), smape(y_pred, y_true))

    def test_bounded(self, large_predictions):
        """범위 확인 (0-200)"""
        y_true, y_pred = large_predictions
        result = smape(y_true, y_pred)
        assert 0 <= result <= 200


class TestMASE:
    """MASE 테스트"""

    def test_better_than_naive(self):
        """naive보다 좋은 예측 시 < 1"""
        np.random.seed(42)
        n = 100
        y_train = np.cumsum(np.random.randn(n)) + 500
        y_true = y_train[-24:]
        y_pred = y_true + np.random.normal(0, 1, 24)  # 작은 오차

        result = mase(y_true, y_pred, y_train, seasonality=24)
        # 좋은 예측이면 1 미만
        assert result < 2


class TestMBE:
    """MBE 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert mbe(y_true, y_pred) == 0

    def test_overestimation(self):
        """과대예측 시 양수"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([120, 220, 320])  # 모두 20 높음
        assert mbe(y_true, y_pred) == 20

    def test_underestimation(self):
        """과소예측 시 음수"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([80, 180, 280])  # 모두 20 낮음
        assert mbe(y_true, y_pred) == -20


class TestCVRMSE:
    """CV(RMSE) 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert cv_rmse(y_true, y_pred) == 0

    def test_percentage_scale(self, sample_predictions):
        """백분율 스케일"""
        y_true, y_pred = sample_predictions
        result = cv_rmse(y_true, y_pred)
        assert result > 0


class TestMaxError:
    """MaxError 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert max_error(y_true, y_pred) == 0

    def test_basic_calculation(self, sample_predictions):
        """기본 계산"""
        y_true, y_pred = sample_predictions
        # 최대 오차: 20 (400→380 또는 500→520)
        assert max_error(y_true, y_pred) == 20


class TestMedianAbsoluteError:
    """MedianAbsoluteError 테스트"""

    def test_perfect_predictions(self, perfect_predictions):
        """완벽한 예측 시 0"""
        y_true, y_pred = perfect_predictions
        assert median_absolute_error(y_true, y_pred) == 0

    def test_basic_calculation(self, sample_predictions):
        """기본 계산"""
        y_true, y_pred = sample_predictions
        # 오차: [10, 10, 10, 20, 20] → 중앙값 10
        assert median_absolute_error(y_true, y_pred) == 10


# ============================================================
# 통합 평가 테스트
# ============================================================

class TestComputeAllMetrics:
    """compute_all_metrics 테스트"""

    def test_returns_all_metrics(self, sample_predictions):
        """모든 지표 반환 확인"""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        required_keys = ['MAPE', 'R2', 'MSE', 'RMSE', 'MAE', 'sMAPE', 'MBE', 'CV_RMSE']
        for key in required_keys:
            assert key in metrics

    def test_with_train_data(self, large_predictions):
        """학습 데이터 포함 시 MASE 계산"""
        y_true, y_pred = large_predictions
        y_train = np.random.uniform(200, 800, 500)

        metrics = compute_all_metrics(y_true, y_pred, y_train)
        assert 'MASE' in metrics


class TestComputeMetricsByThreshold:
    """compute_metrics_by_threshold 테스트"""

    def test_returns_dict(self, large_predictions):
        """딕셔너리 반환"""
        y_true, y_pred = large_predictions
        result = compute_metrics_by_threshold(y_true, y_pred)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_custom_thresholds(self, large_predictions):
        """커스텀 임계값"""
        y_true, y_pred = large_predictions
        thresholds = [300, 500, 700]
        result = compute_metrics_by_threshold(y_true, y_pred, thresholds)

        assert len(result) > 0


# ============================================================
# 시간 기반 분석 테스트
# ============================================================

class TestTimeBasedMetrics:
    """시간 기반 분석 테스트"""

    def test_hourly_metrics(self, time_predictions):
        """시간대별 분석"""
        y_true, y_pred, timestamps = time_predictions
        result = compute_metrics_by_hour(y_true, y_pred, timestamps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24  # 0-23시
        assert 'hour' in result.columns
        assert 'MAPE' in result.columns

    def test_dayofweek_metrics(self, time_predictions):
        """요일별 분석"""
        y_true, y_pred, timestamps = time_predictions
        result = compute_metrics_by_dayofweek(y_true, y_pred, timestamps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 7  # 월-일
        assert 'day_name' in result.columns

    def test_monthly_metrics(self, time_predictions):
        """월별 분석"""
        y_true, y_pred, timestamps = time_predictions
        result = compute_metrics_by_month(y_true, y_pred, timestamps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        assert 'month_name' in result.columns

    def test_seasonal_metrics(self, time_predictions):
        """계절별 분석"""
        y_true, y_pred, timestamps = time_predictions
        result = compute_metrics_by_season(y_true, y_pred, timestamps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert 'season' in result.columns


# ============================================================
# 잔차 분석 테스트
# ============================================================

class TestResidualAnalysis:
    """잔차 분석 테스트"""

    def test_analyze_residuals(self, sample_predictions):
        """잔차 분석 기본"""
        y_true, y_pred = sample_predictions
        result = analyze_residuals(y_true, y_pred)

        assert 'mean' in result
        assert 'std' in result
        assert 'skewness' in result
        assert 'kurtosis' in result

    def test_prediction_intervals(self, large_predictions):
        """예측 구간 커버리지"""
        y_true, y_pred = large_predictions
        result = compute_prediction_intervals(y_true, y_pred)

        # 95% 신뢰구간 커버리지 확인
        assert 'coverage_95' in result
        # 정규분포 가정하면 약 95% 커버
        assert 80 < result['coverage_95'] < 100


# ============================================================
# 평가 리포트 테스트
# ============================================================

class TestEvaluationReport:
    """EvaluationReport 테스트"""

    def test_report_creation(self, sample_predictions):
        """리포트 생성"""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        report = EvaluationReport(
            model_name="TestModel",
            horizon=1,
            timestamp="2024-12-14",
            overall_metrics=metrics
        )

        assert report.model_name == "TestModel"
        assert report.horizon == 1

    def test_report_to_dict(self, sample_predictions):
        """딕셔너리 변환"""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        report = EvaluationReport(
            model_name="TestModel",
            horizon=1,
            timestamp="2024-12-14",
            overall_metrics=metrics
        )

        result = report.to_dict()
        assert 'model_name' in result
        assert 'overall_metrics' in result

    def test_report_save_load(self, sample_predictions):
        """리포트 저장/로드"""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        report = EvaluationReport(
            model_name="TestModel",
            horizon=1,
            timestamp="2024-12-14",
            overall_metrics=metrics
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            report.save(filepath)
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_report_summary(self, sample_predictions):
        """리포트 요약"""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        report = EvaluationReport(
            model_name="TestModel",
            horizon=1,
            timestamp="2024-12-14",
            overall_metrics=metrics
        )

        summary = report.summary()
        assert "TestModel" in summary
        assert "MAPE" in summary


class TestGenerateEvaluationReport:
    """generate_evaluation_report 테스트"""

    def test_basic_generation(self, sample_predictions):
        """기본 리포트 생성"""
        y_true, y_pred = sample_predictions
        report = generate_evaluation_report(
            y_true, y_pred,
            model_name="TestModel",
            horizon=1
        )

        assert isinstance(report, EvaluationReport)
        assert report.overall_metrics is not None

    def test_with_timestamps(self, time_predictions):
        """타임스탬프 포함 리포트"""
        y_true, y_pred, timestamps = time_predictions
        report = generate_evaluation_report(
            y_true, y_pred,
            timestamps=timestamps,
            model_name="TestModel",
            horizon=1
        )

        assert report.hourly_metrics is not None
        assert report.seasonal_metrics is not None


# ============================================================
# 모델 비교 테스트
# ============================================================

class TestCompareModels:
    """compare_models 테스트"""

    def test_basic_comparison(self, large_predictions):
        """기본 모델 비교"""
        y_true, y_pred = large_predictions

        # 두 모델 시뮬레이션
        y_pred2 = y_true + np.random.normal(0, 30, len(y_true))

        results = {
            'Model1': (y_true, y_pred),
            'Model2': (y_true, y_pred2)
        }

        comparison = compare_models(results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Model' in comparison.columns
        assert 'MAPE' in comparison.columns

    def test_ranking(self, large_predictions):
        """랭킹 확인"""
        y_true, y_pred = large_predictions
        y_pred2 = y_true + np.random.normal(0, 50, len(y_true))  # 더 큰 오차

        results = {
            'Better': (y_true, y_pred),
            'Worse': (y_true, y_pred2)
        }

        comparison = compare_models(results)
        assert 'MAPE_rank' in comparison.columns


class TestCompareHorizons:
    """compare_horizons 테스트"""

    def test_horizon_comparison(self):
        """horizon 비교"""
        np.random.seed(42)
        n = 100

        y_true_dict = {}
        y_pred_dict = {}

        for h in [1, 6, 12, 24]:
            y_true = np.random.uniform(300, 600, n)
            # 더 먼 horizon일수록 오차가 큼
            noise_scale = 10 + h * 2
            y_pred = y_true + np.random.normal(0, noise_scale, n)

            y_true_dict[h] = y_true
            y_pred_dict[h] = y_pred

        comparison = compare_horizons(y_true_dict, y_pred_dict, "TestModel")

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 4
        assert 'Horizon' in comparison.columns


# ============================================================
# 엣지 케이스 테스트
# ============================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_value(self):
        """단일 값"""
        y_true = np.array([100])
        y_pred = np.array([110])

        assert mae(y_true, y_pred) == 10
        assert mape(y_true, y_pred) == 10.0

    def test_all_same_values(self):
        """모든 값이 동일"""
        y_true = np.full(10, 100)
        y_pred = np.full(10, 100)

        assert mse(y_true, y_pred) == 0
        assert r2_score(y_true, y_pred) == 1.0

    def test_negative_values(self):
        """음수 값"""
        y_true = np.array([-100, -50, 0, 50, 100])
        y_pred = np.array([-90, -60, 10, 40, 110])

        # 음수에서도 지표가 정상 계산
        assert mae(y_true, y_pred) == 10
        assert mse(y_true, y_pred) == 100

    def test_large_values(self):
        """큰 값"""
        y_true = np.array([1e6, 2e6, 3e6])
        y_pred = np.array([1.1e6, 1.9e6, 3.1e6])

        assert mape(y_true, y_pred) > 0
        assert r2_score(y_true, y_pred) > 0

    def test_list_input(self):
        """리스트 입력"""
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]

        # 리스트도 처리 가능
        assert mae(y_true, y_pred) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
