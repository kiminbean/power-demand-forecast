"""
Evaluation Module
=================
모델 평가를 위한 지표 및 리포트 생성

주요 구성요소:
- 기본 지표: MSE, RMSE, MAE, MAPE, R²
- 고급 지표: sMAPE, MASE, MBE, CV(RMSE)
- 시간 기반 분석: 시간대별, 요일별, 월별, 계절별
- 평가 리포트 생성 및 모델 비교
"""

from .metrics import (
    # 기본 평가 지표
    mse,
    rmse,
    mae,
    mape,
    r2_score,

    # 고급 평가 지표
    smape,
    mase,
    mbe,
    cv_rmse,
    max_error,
    median_absolute_error,

    # 통합 평가
    compute_all_metrics,
    compute_metrics_by_threshold,

    # 시간 기반 분석
    compute_metrics_by_hour,
    compute_metrics_by_dayofweek,
    compute_metrics_by_month,
    compute_metrics_by_season,

    # 잔차 분석
    analyze_residuals,
    compute_prediction_intervals,

    # 평가 리포트
    EvaluationReport,
    generate_evaluation_report,

    # 모델 비교
    compare_models,
    compare_horizons,
)

__all__ = [
    # 기본 평가 지표
    'mse',
    'rmse',
    'mae',
    'mape',
    'r2_score',

    # 고급 평가 지표
    'smape',
    'mase',
    'mbe',
    'cv_rmse',
    'max_error',
    'median_absolute_error',

    # 통합 평가
    'compute_all_metrics',
    'compute_metrics_by_threshold',

    # 시간 기반 분석
    'compute_metrics_by_hour',
    'compute_metrics_by_dayofweek',
    'compute_metrics_by_month',
    'compute_metrics_by_season',

    # 잔차 분석
    'analyze_residuals',
    'compute_prediction_intervals',

    # 평가 리포트
    'EvaluationReport',
    'generate_evaluation_report',

    # 모델 비교
    'compare_models',
    'compare_horizons',
]
