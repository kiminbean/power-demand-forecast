"""
Inference Module
================

Production 모델 추론 모듈

Usage:
------
>>> from inference import predict, predict_batch, ProductionPredictor
>>>
>>> # 간편 예측
>>> result = predict(df, model="conditional")
>>> print(f"Predicted: {result.predicted_demand:.2f} MW")
>>>
>>> # 배치 예측
>>> batch_result = predict_batch(df, model="demand_only")
>>> print(f"Predictions: {len(batch_result.predictions)}")
>>>
>>> # 상세 제어
>>> predictor = ProductionPredictor()
>>> predictor.load_models()
>>> pred = predictor.predict_demand_only(df)
"""

from .predict import (
    ProductionPredictor,
    PredictionResult,
    BatchPredictionResult,
    predict,
    predict_batch,
    get_predictor,
)

__all__ = [
    'ProductionPredictor',
    'PredictionResult',
    'BatchPredictionResult',
    'predict',
    'predict_batch',
    'get_predictor',
]
