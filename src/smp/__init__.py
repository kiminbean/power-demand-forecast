"""
SMP (System Marginal Price) 예측 및 입찰 지원 모듈
=================================================

제주도 민간 태양광/풍력 발전사업자를 위한 SMP 예측 및 입찰 전략 지원 시스템

Modules:
    crawlers: SMP 및 연료비 데이터 크롤링
    models: SMP 예측 모델 (LSTM, TFT, Ensemble)
    features: SMP 피처 엔지니어링
    bidding: 입찰 전략 최적화
    data: 데이터셋 및 스케일러

Version: 2.0.0
Author: Power Demand Forecast Team
Date: 2025-12
"""

__version__ = "2.0.0"

from . import crawlers
from . import models
from . import features
from . import bidding
from . import data

__all__ = [
    "crawlers",
    "models",
    "features",
    "bidding",
    "data",
    "__version__",
]
