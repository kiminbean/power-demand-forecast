"""
Models Module
=============
시계열 예측을 위한 딥러닝 모델

MODEL-002: Vanilla RNN (pending)
MODEL-003: LSTM
MODEL-004: BiLSTM
MODEL-005: Multi-horizon LSTM
MODEL-006: Conditional Predictor (겨울철 변곡점 특화)
MODEL-007: Temporal Fusion Transformer (TFT)
MODEL-011: Ensemble (Weighted Average, Stacking, Blending)
MODEL-012: Probabilistic Forecasting (MC Dropout, Deep Ensemble, Quantile)
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

from .transformer import (
    # TFT 컴포넌트
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    StaticCovariateEncoder,
    TemporalVariableSelection,
    PositionalEncoding,
    InterpretableMultiHeadAttention,
    TemporalSelfAttention,
    StaticEnrichmentLayer,
    LSTMEncoder,
    LSTMDecoder,
    # TFT 모델
    TemporalFusionTransformer,
    # 손실 함수
    QuantileLoss,
    # 유틸리티
    generate_causal_mask,
    generate_encoder_decoder_mask,
)

from .ensemble import (
    # 앙상블 설정
    EnsembleConfig,
    # 앙상블 모델
    BaseEnsemble,
    WeightedAverageEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    UncertaintyEnsemble,
    # 최적화
    EnsembleOptimizer,
    # 팩토리 및 유틸리티
    create_ensemble,
    evaluate_ensemble,
    compare_with_individual,
)

from .probabilistic import (
    # 데이터 클래스
    PredictionInterval,
    # 확률적 모델
    QuantileRegressor,
    MCDropout,
    DeepEnsembleUncertainty,
    # 손실 함수
    PinballLoss,
    # 보정 및 평가
    CalibrationMetrics,
    ProbabilisticWrapper,
    # 팩토리 및 유틸리티
    create_probabilistic_model,
    calculate_prediction_intervals,
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
    # MODEL-007: TFT
    'GatedLinearUnit',
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'StaticCovariateEncoder',
    'TemporalVariableSelection',
    'PositionalEncoding',
    'InterpretableMultiHeadAttention',
    'TemporalSelfAttention',
    'StaticEnrichmentLayer',
    'LSTMEncoder',
    'LSTMDecoder',
    'TemporalFusionTransformer',
    'QuantileLoss',
    'generate_causal_mask',
    'generate_encoder_decoder_mask',
    # MODEL-011: Ensemble
    'EnsembleConfig',
    'BaseEnsemble',
    'WeightedAverageEnsemble',
    'StackingEnsemble',
    'BlendingEnsemble',
    'UncertaintyEnsemble',
    'EnsembleOptimizer',
    'create_ensemble',
    'evaluate_ensemble',
    'compare_with_individual',
    # MODEL-012: Probabilistic
    'PredictionInterval',
    'QuantileRegressor',
    'MCDropout',
    'DeepEnsembleUncertainty',
    'PinballLoss',
    'CalibrationMetrics',
    'ProbabilisticWrapper',
    'create_probabilistic_model',
    'calculate_prediction_intervals',
]


def create_tft_model(
    num_known_vars: int,
    num_unknown_vars: int,
    hidden_size: int = 64,
    lstm_layers: int = 2,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    encoder_length: int = 48,
    decoder_length: int = 24,
    quantiles: list = None,
    num_static_vars: int = 0
) -> TemporalFusionTransformer:
    """
    TFT 모델 팩토리 함수

    Args:
        num_known_vars: Known 변수 수 (시간 관련 피처)
        num_unknown_vars: Unknown 변수 수 (기상/수요 피처)
        hidden_size: Hidden 차원
        lstm_layers: LSTM 레이어 수
        num_attention_heads: Attention head 수
        dropout: Dropout 비율
        encoder_length: Encoder 시퀀스 길이
        decoder_length: Decoder 시퀀스 길이
        quantiles: Quantile 출력 리스트
        num_static_vars: Static 변수 수

    Returns:
        TemporalFusionTransformer 모델

    Example:
        >>> model = create_tft_model(
        ...     num_known_vars=8,
        ...     num_unknown_vars=25,
        ...     hidden_size=64,
        ...     encoder_length=48,
        ...     decoder_length=24
        ... )
    """
    return TemporalFusionTransformer(
        num_static_vars=num_static_vars,
        num_known_vars=num_known_vars,
        num_unknown_vars=num_unknown_vars,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        quantiles=quantiles
    )
