"""
Analysis Module
===============
데이터 분석 및 모델 평가

- correlation_analysis: 상관관계 분석
- check_distribution: 분포 검사
- inflection_point_analysis: 변곡점(급변구간) 잔차 분석
"""

from .correlation_analysis import (
    load_and_merge_data,
    create_correlation_heatmap,
    analyze_top_correlations,
)

from .inflection_point_analysis import (
    run_inflection_point_analysis,
    identify_inflection_points,
    analyze_residuals,
    print_analysis_report,
)

__all__ = [
    # correlation_analysis
    'load_and_merge_data',
    'create_correlation_heatmap',
    'analyze_top_correlations',
    # inflection_point_analysis
    'run_inflection_point_analysis',
    'identify_inflection_points',
    'analyze_residuals',
    'print_analysis_report',
]
