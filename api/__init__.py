"""
Power Demand Forecast API
==========================

제주도 전력 수요 예측 REST API 서비스

Endpoints:
- GET  /health         - 서비스 상태 확인
- GET  /models         - 로드된 모델 정보
- POST /predict        - 단일 예측
- POST /predict/batch  - 배치 예측
- POST /predict/conditional - 조건부 예측

Author: Power Demand Forecast Team
Date: 2024-12
"""

__version__ = "1.0.0"

from . import service
from . import schemas
from . import config
from . import main
