"""
Models Module
=============
시계열 예측을 위한 딥러닝 모델

MODEL-002: Vanilla RNN (pending)
MODEL-003: LSTM
MODEL-004: BiLSTM
MODEL-005: Multi-horizon LSTM
MODEL-006: Conditional Predictor (겨울철 변곡점 특화)
"""

from .lstm import (
    # 모델 클래스
    LSTMModel,
    MultiHorizonLSTM,
    ResidualLSTM,
    # 팩토리 함수
    create_model,
    # 유틸리티
    model_summary,
)

from .conditional import (
    # 조건부 모델
    Season,
    PredictionContext,
    SeasonClassifier,
    InflectionDetector,
    ConditionalPredictor,
    AdaptiveConditionalPredictor,
    create_conditional_predictor,
)

__all__ = [
    # MODEL-003: LSTM
    'LSTMModel',
    'MultiHorizonLSTM',
    'ResidualLSTM',
    'create_model',
    'model_summary',
    # MODEL-006: Conditional
    'Season',
    'PredictionContext',
    'SeasonClassifier',
    'InflectionDetector',
    'ConditionalPredictor',
    'AdaptiveConditionalPredictor',
    'create_conditional_predictor',
]
