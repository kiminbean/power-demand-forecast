"""
SMP 예측 모델 모듈
==================

LSTM, TFT 기반 SMP 예측 모델 및 앙상블

Classes:
    SMPLSTMModel: LSTM 기반 SMP 예측 모델
    SMPQuantileLSTM: Quantile 예측 LSTM 모델
    QuantileLoss: Quantile 손실 함수
    SMPTFTModel: Temporal Fusion Transformer 기반 SMP 예측 모델
    SMPTFTQuantileLoss: TFT Quantile 손실 함수
    SMPEnsemble: 앙상블 모델 (예정)
    GenerationPredictor: 태양광/풍력 발전량 예측기 (예정)
"""

from .smp_lstm import (
    SMPLSTMModel,
    SMPQuantileLSTM,
    QuantileLoss,
    create_smp_model,
    model_summary as lstm_model_summary,
    get_device,
)

from .smp_tft import (
    SMPTFTModel,
    SMPTFTQuantileLoss,
    create_smp_tft_model,
    model_summary as tft_model_summary,
)

from .generation_predictor import (
    PlantConfig,
    GenerationPrediction,
    SolarPowerCalculator,
    WindPowerCalculator,
    GenerationPredictor,
    GenerationLSTM,
    create_generation_predictor,
)

# 통합 model_summary 함수
def model_summary(model):
    """모델 요약 (LSTM/TFT 공통)"""
    if isinstance(model, SMPTFTModel):
        return tft_model_summary(model)
    return lstm_model_summary(model)

__all__ = [
    # LSTM
    "SMPLSTMModel",
    "SMPQuantileLSTM",
    "QuantileLoss",
    "create_smp_model",
    # TFT
    "SMPTFTModel",
    "SMPTFTQuantileLoss",
    "create_smp_tft_model",
    # Generation Predictor
    "PlantConfig",
    "GenerationPrediction",
    "SolarPowerCalculator",
    "WindPowerCalculator",
    "GenerationPredictor",
    "GenerationLSTM",
    "create_generation_predictor",
    # Common
    "model_summary",
    "get_device",
]
