"""
Experiments Module
==================
모델 성능 비교 실험

EVAL-002: 기상변수 포함/미포함 비교
EVAL-003: Horizon 변화에 따른 기상변수 효과 실험
"""

from .weather_comparison import (
    run_weather_comparison_experiment,
    run_experiment_group,
    analyze_results,
    create_boxplot,
    FEATURE_GROUPS,
    DEFAULT_CONFIG,
)

from .horizon_comparison import (
    run_full_horizon_comparison,
    run_horizon_experiment,
    analyze_horizon_results,
    calculate_weather_improvement,
    create_horizon_plots,
    HORIZONS,
)

__all__ = [
    # EVAL-002
    'run_weather_comparison_experiment',
    'run_experiment_group',
    'analyze_results',
    'create_boxplot',
    'FEATURE_GROUPS',
    'DEFAULT_CONFIG',
    # EVAL-003
    'run_full_horizon_comparison',
    'run_horizon_experiment',
    'analyze_horizon_results',
    'calculate_weather_improvement',
    'create_horizon_plots',
    'HORIZONS',
]
