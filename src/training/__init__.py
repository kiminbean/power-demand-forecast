"""
Training Module
===============
모델 학습을 위한 Trainer 및 콜백

주요 구성요소:
- Trainer: 학습/검증/테스트 파이프라인
- EarlyStopping: 조기 종료 콜백
- TrainingHistory: 학습 히스토리 기록
- Schedulers: 학습률 스케줄러
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
]
