"""
Training Module
===============
모델 학습을 위한 Trainer 및 콜백

주요 구성요소:
- Trainer: 학습/검증/테스트 파이프라인
- EarlyStopping: 조기 종료 콜백
- TrainingHistory: 학습 히스토리 기록
- Schedulers: 학습률 스케줄러
- ModelSelection: AutoML 모델 선택 (Task 19)
"""

from .trainer import (
    # Trainer
    Trainer,
    # Callbacks
    EarlyStopping,
    TrainingHistory,
    # Schedulers
    create_scheduler,
    # Metrics
    compute_metrics,
)

from .model_selection import (
    # Types
    ModelType,
    ModelConfig,
    ModelResult,
    SearchSpace,
    # Factory
    ModelFactory,
    # AutoML
    HyperparameterTuner,
    ModelComparator,
    AutoMLPipeline,
    # Search spaces
    LSTM_SEARCH_SPACE,
    TFT_SEARCH_SPACE,
    ENSEMBLE_SEARCH_SPACE,
    get_search_space,
    # Utilities
    simple_train_fn,
    create_automl_pipeline,
)

__all__ = [
    # Trainer
    'Trainer',
    # Callbacks
    'EarlyStopping',
    'TrainingHistory',
    # Schedulers
    'create_scheduler',
    # Metrics
    'compute_metrics',
    # Model Selection (Task 19)
    'ModelType',
    'ModelConfig',
    'ModelResult',
    'SearchSpace',
    'ModelFactory',
    'HyperparameterTuner',
    'ModelComparator',
    'AutoMLPipeline',
    'LSTM_SEARCH_SPACE',
    'TFT_SEARCH_SPACE',
    'ENSEMBLE_SEARCH_SPACE',
    'get_search_space',
    'simple_train_fn',
    'create_automl_pipeline',
]
