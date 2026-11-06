"""
Prediction Service
==================

예측 비즈니스 로직을 담당하는 서비스 레이어
"""

import sys
import time
import logging
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predict import (
    ProductionPredictor, PredictionResult, BatchPredictionResult,
    _scaled_tensor, _validated_prediction,
)
from features.time_features import LUNAR_HOLIDAY_LAST_YEAR
from data.dataset import get_device

from .config import settings
from .schemas import (
    TimeSeriesData,
    ModelType,
    ConditionalMode,
    PredictionResponse,
    ConditionalPredictionResponse,
    BatchPredictionResponse,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """요청한 선택형 모델의 산출물이 로드되지 않았다 (API는 503으로 응답)."""


class PredictionService:
    """
    예측 서비스 클래스

    ProductionPredictor를 래핑하여 API에 필요한 형태로 변환
    """

    def __init__(self):
        self.predictor: Optional[ProductionPredictor] = None
        # 선택형 AR + GBM (inference.ar_gbm_predictor.ArGbmPredictor, scikit-learn 필요)
        self.ar_gbm = None
        self._ar_gbm_error: Optional[str] = None
        self.start_time: datetime = datetime.now()
        self._is_initialized: bool = False

    def initialize(self) -> None:
        """서비스 초기화 및 모델 로드"""
        if self._is_initialized:
            logger.info("Service already initialized")
            return

        logger.info(f"Initializing prediction service...")
        logger.info(f"Model directory: {settings.model_path}")

        try:
            self.predictor = ProductionPredictor(
                model_dir=str(settings.model_path)
            )
            self.predictor.load_models()
            self._load_ar_gbm()
            self._is_initialized = True
            logger.info("Prediction service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}")
            raise

    def _load_ar_gbm(self) -> None:
        """선택형 AR + GBM 모델을 불러온다. 실패해도 기본(LSTM) 서비스는 그대로 연다."""
        try:
            # scikit-learn이 없거나 깨져도 LSTM 서비스는 떠야 하므로 여기서 불러온다
            from inference.ar_gbm_predictor import ArGbmPredictor

            predictor = ArGbmPredictor(settings.ar_gbm_model_path)
            predictor.load()
        except Exception as e:  # noqa: BLE001
            # 산출물 없음·게이트 미통과·형식/버전 불일치·pickle 오류 등 무엇이든 선택형 모델만 끈다.
            # 이유는 로그에 남기고, 응답에는 DEBUG일 때만 붙인다 (경로·버전 노출 방지).
            self._ar_gbm_error = f"{type(e).__name__}: {e}"
            logger.warning(f"AR+GBM model unavailable: {self._ar_gbm_error}")
            return
        self.ar_gbm = predictor
        logger.info(f"AR+GBM model loaded from {settings.ar_gbm_model_path}")

    def _require_ar_gbm(self):
        if self.ar_gbm is None:
            reason = f": {self._ar_gbm_error}" if settings.DEBUG and self._ar_gbm_error else ""
            raise ModelNotLoadedError(f"Model 'ar_gbm' is not available{reason}")
        return self.ar_gbm

    def is_ready(self) -> bool:
        """서비스 준비 상태 확인"""
        return self._is_initialized and self.predictor is not None

    def get_uptime(self) -> float:
        """서비스 가동 시간 (초)"""
        return (datetime.now() - self.start_time).total_seconds()

    def get_device(self) -> str:
        """연산 디바이스"""
        if self.predictor:
            return str(self.predictor.device)
        return str(get_device())

    def get_model_info(self) -> List[Dict]:
        """로드된 모델 정보"""
        if not self.predictor:
            return []

        models = []

        # demand_only model
        if self.predictor.model_demand is not None:
            config = self.predictor.config_demand
            models.append({
                "name": "demand_only",
                "type": config.get('model_config', {}).get('model_type', 'LSTM'),
                "n_features": config.get('n_features', 0),
                "seq_length": config.get('training_config', {}).get('seq_length', 168),
                "hidden_size": config.get('model_config', {}).get('hidden_size', 0),
                "num_layers": config.get('model_config', {}).get('num_layers', 0),
                "status": "loaded"
            })

        # weather_full model
        if self.predictor.model_weather is not None:
            config = self.predictor.config_weather
            models.append({
                "name": "weather_full",
                "type": config.get('model_config', {}).get('model_type', 'LSTM'),
                "n_features": config.get('n_features', 0),
                "seq_length": config.get('training_config', {}).get('seq_length', 168),
                "hidden_size": config.get('model_config', {}).get('hidden_size', 0),
                "num_layers": config.get('model_config', {}).get('num_layers', 0),
                "status": "loaded"
            })

        # 선택형 AR + GBM (model_type="ar_gbm")
        if self.ar_gbm is not None and self.ar_gbm.model is not None:
            info = self.ar_gbm.model.info
            models.append({
                "name": "ar_gbm",
                "type": "HourlyAR+HistGBM",
                "n_features": len(self.ar_gbm.model.design_columns),
                "seq_length": info.get('min_history_hours'),
                "min_history_hours": info.get('min_history_hours'),
                # 계수는 train_end까지로 적합했고, val_end까지는 트리 수 선택에만 썼다
                "train_end": info.get('train_end'),
                "val_end": info.get('val_end'),
                "holiday_table_through": LUNAR_HOLIDAY_LAST_YEAR,
                "status": "loaded"
            })

        return models

    def _convert_to_dataframe(self, data: List[TimeSeriesData]) -> pd.DataFrame:
        """TimeSeriesData 리스트를 DataFrame으로 변환"""
        records = []
        for item in data:
            record = {
                'datetime': item.datetime,
                'power_demand': item.power_demand,
            }
            if item.temperature is not None:
                record['기온'] = item.temperature
            if item.humidity is not None:
                record['습도'] = item.humidity
            if item.wind_speed is not None:
                record['풍속'] = item.wind_speed
            if item.precipitation is not None:
                record['강수량'] = item.precipitation
            records.append(record)

        df = pd.DataFrame(records)
        aware = [value.utcoffset() is not None for value in df['datetime']]
        if any(aware) and not all(aware):
            raise ValueError("Cannot mix naive and timezone-aware timestamps")
        if all(aware) and aware:
            from inference.feature_preprocessing import MODEL_TIMEZONE

            df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(MODEL_TIMEZONE)
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        return df

    def _prepare_features_for_api(
        self, df: pd.DataFrame, include_weather: bool = False,
        selected_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """API용 피처 준비 — 직접 추론 경로와 동일한 공유 정책 사용

        기존 bfill().ffill() + fillna(0)은 미래 정보 누수(bfill)와
        스케일러 왜곡(0 채움) 문제가 있어 src/inference/feature_preprocessing.py
        의 인과적 정책(ffill 후 선행 warm-up 행 제거)로 통일됨.
        """
        from inference.feature_preprocessing import prepare_inference_features

        return prepare_inference_features(
            df, include_weather=include_weather, selected_features=selected_features
        )

    def _run_prediction(
        self,
        df: pd.DataFrame,
        model_name: str = "demand_only"
    ) -> float:
        """직접 예측 실행 (NaN 처리 포함)"""
        import torch

        # 모델 및 설정 선택
        if model_name == "weather_full":
            model = self.predictor.model_weather
            scaler = self.predictor.scaler_weather
            config = self.predictor.config_weather
            include_weather = True
        else:
            model = self.predictor.model_demand
            scaler = self.predictor.scaler_demand
            config = self.predictor.config_demand
            include_weather = False

        if model is None:
            raise RuntimeError(f"Model '{model_name}' not loaded")

        features = config['features']
        # 피처 준비 (NaN 처리 포함)
        df_prep = self._prepare_features_for_api(
            df, include_weather=include_weather, selected_features=features
        )

        # 필요한 피처만 추출
        seq_length = config['training_config']['seq_length']

        available = [f for f in features if f in df_prep.columns]
        if len(available) < len(features) * 0.5:
            logger.warning(f"Only {len(available)}/{len(features)} features available")

        data = df_prep[available].values

        if len(data) < seq_length:
            raise ValueError(f"Insufficient data: need {seq_length}, got {len(data)}")

        # 시퀀스 추출
        sequence = data[-seq_length:]

        # 텐서 변환 및 예측
        X = _scaled_tensor(scaler, sequence, self.predictor.device)

        with torch.no_grad():
            pred = model(X)

        return _validated_prediction(pred, scaler)

    def predict(
        self,
        data: List[TimeSeriesData],
        model_type: ModelType = ModelType.DEMAND_ONLY
    ) -> PredictionResponse:
        """
        단일 예측 수행

        Parameters
        ----------
        data : List[TimeSeriesData]
            시계열 데이터 (최소 168개)
        model_type : ModelType
            사용할 모델

        Returns
        -------
        PredictionResponse
            예측 결과
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized")

        start_time = time.perf_counter()

        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        logger.debug(f"Input data shape: {df.shape}")

        # Get timestamp
        timestamp = df.index[-1]

        # Predict based on model type
        if model_type == ModelType.DEMAND_ONLY:
            prediction = self._run_prediction(df, "demand_only")
            model_used = "demand_only"
        elif model_type == ModelType.WEATHER_FULL:
            prediction = self._run_prediction(df, "weather_full")
            model_used = "weather_full"
        elif model_type == ModelType.AR_GBM:
            prediction = self._require_ar_gbm().forecast_next(df)
            model_used = "ar_gbm"
        else:
            # conditional - blend predictions based on season
            pred_demand = self._run_prediction(df, "demand_only")
            month = timestamp.month
            is_winter = month in [12, 1, 2]

            if is_winter and self.predictor.model_weather is not None:
                try:
                    pred_weather = self._run_prediction(df, "weather_full")
                    # 겨울철 블렌딩
                    if month == 1:
                        weight = 0.3
                    else:
                        weight = 0.2
                    prediction = (1 - weight) * pred_demand + weight * pred_weather
                    model_used = f"conditional_soft (w={weight})"
                except Exception as e:
                    logger.warning(f"Weather prediction failed: {e}")
                    prediction = pred_demand
                    model_used = "demand_only (fallback)"
            else:
                prediction = pred_demand
                model_used = "demand_only"

        processing_time = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            success=True,
            prediction=round(prediction, 2),
            model_used=model_used,
            timestamp=timestamp,
            processing_time_ms=round(processing_time, 2)
        )

    def predict_conditional(
        self,
        data: List[TimeSeriesData],
        mode: ConditionalMode = ConditionalMode.SOFT
    ) -> ConditionalPredictionResponse:
        """
        조건부 예측 수행 (겨울철 자동 최적화)

        Parameters
        ----------
        data : List[TimeSeriesData]
            시계열 데이터
        mode : ConditionalMode
            예측 모드

        Returns
        -------
        ConditionalPredictionResponse
            조건부 예측 결과
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized")

        start_time = time.perf_counter()

        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        timestamp = df.index[-1]

        # Predict demand_only
        pred_demand = self._run_prediction(df, "demand_only")

        month = timestamp.month
        is_winter = month in [12, 1, 2]
        weather_weight = 0.0
        pred_weather = None

        if is_winter and self.predictor.model_weather is not None:
            try:
                pred_weather = self._run_prediction(df, "weather_full")

                if mode == ConditionalMode.HARD:
                    prediction = pred_weather
                    model_used = "weather_full"
                    weather_weight = 1.0
                else:
                    # Soft blending
                    if month == 1:
                        weather_weight = 0.3
                    else:
                        weather_weight = 0.2
                    prediction = (1 - weather_weight) * pred_demand + weather_weight * pred_weather
                    model_used = f"conditional_soft (w={weather_weight})"
            except Exception as e:
                logger.warning(f"Weather prediction failed: {e}")
                prediction = pred_demand
                model_used = "demand_only (fallback)"
        else:
            prediction = pred_demand
            model_used = "demand_only"

        processing_time = (time.perf_counter() - start_time) * 1000

        return ConditionalPredictionResponse(
            success=True,
            prediction=round(prediction, 2),
            model_used=model_used,
            timestamp=timestamp,
            context={
                'is_winter': is_winter,
                'month': month,
                'weather_weight': weather_weight,
                'pred_demand_only': round(pred_demand, 2),
                'pred_weather_full': round(pred_weather, 2) if pred_weather else None
            },
            processing_time_ms=round(processing_time, 2)
        )

    def predict_batch(
        self,
        data: List[TimeSeriesData],
        model_type: ModelType = ModelType.DEMAND_ONLY,
        step: int = 1
    ) -> BatchPredictionResponse:
        """
        배치 예측 수행

        Parameters
        ----------
        data : List[TimeSeriesData]
            시계열 데이터
        model_type : ModelType
            사용할 모델
        step : int
            슬라이딩 윈도우 스텝

        Returns
        -------
        BatchPredictionResponse
            배치 예측 결과
        """
        import torch

        if not self.is_ready():
            raise RuntimeError("Service not initialized")

        if not isinstance(step, Integral) or isinstance(step, bool) or step <= 0:
            raise ValueError("Batch step must be a positive integer")
        step = int(step)

        start_time = time.perf_counter()

        # Convert to DataFrame
        df = self._convert_to_dataframe(data)

        if model_type == ModelType.AR_GBM:
            forecasts = self._require_ar_gbm().forecast(df)
            if not np.isfinite(forecasts.to_numpy()).all():
                raise ValueError("AR+GBM predictions must be finite")
            # LSTM 배치와 같은 계약: 각 예측에는 입력 창의 마지막 시각(예측 대상 1시간 전)을 붙인다
            labels = (forecasts.index - pd.Timedelta(hours=1))[::step]
            return self._batch_response(list(labels), forecasts.to_numpy()[::step], "ar_gbm", start_time)

        # Map model type
        model_name = model_type.value if model_type != ModelType.CONDITIONAL else "demand_only"

        # Select model and config
        if model_name == "weather_full":
            model = self.predictor.model_weather
            scaler = self.predictor.scaler_weather
            config = self.predictor.config_weather
            include_weather = True
        else:
            model = self.predictor.model_demand
            scaler = self.predictor.scaler_demand
            config = self.predictor.config_demand
            include_weather = False

        features = config['features']
        # Prepare features
        df_prep = self._prepare_features_for_api(
            df, include_weather=include_weather, selected_features=features
        )

        seq_length = config['training_config']['seq_length']

        if len(df_prep) < seq_length:
            raise ValueError(
                f"Insufficient preprocessed data: need {seq_length} rows, got {len(df_prep)}"
            )

        available = [f for f in features if f in df_prep.columns]
        data_array = df_prep[available].values

        # Generate batch predictions
        predictions_list = []
        timestamps_list = []
        n_samples = len(data_array) - seq_length + 1

        for i in range(0, n_samples, step):
            sequence = data_array[i:i + seq_length]
            X = _scaled_tensor(scaler, sequence, self.predictor.device)

            with torch.no_grad():
                pred = model(X)

            pred_value = _validated_prediction(pred, scaler)

            predictions_list.append(pred_value)
            timestamps_list.append(df_prep.index[i + seq_length - 1])

        predictions_array = np.array(predictions_list)
        return self._batch_response(timestamps_list, predictions_array, model_name, start_time)

    def _batch_response(
        self,
        timestamps_list: List,
        predictions_array: np.ndarray,
        model_name: str,
        start_time: float
    ) -> BatchPredictionResponse:
        """배치 예측 결과를 응답 형식으로 만든다."""
        # Format predictions
        predictions = [
            {
                "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "prediction": round(float(pred), 2)
            }
            for ts, pred in zip(timestamps_list, predictions_array)
        ]

        # Calculate statistics
        with np.errstate(over="ignore", invalid="ignore"):
            statistics = {
                "mean": round(float(predictions_array.mean()), 2),
                "std": round(float(predictions_array.std()), 2),
                "min": round(float(predictions_array.min()), 2),
                "max": round(float(predictions_array.max()), 2)
            }
        if not all(np.isfinite(value) for value in statistics.values()):
            raise ValueError("Batch statistics must contain finite values")

        processing_time = (time.perf_counter() - start_time) * 1000

        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            model_used=model_name,
            total_predictions=len(predictions),
            statistics=statistics,
            processing_time_ms=round(processing_time, 2)
        )


# 서비스 싱글톤
_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """예측 서비스 싱글톤 반환"""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


def initialize_service() -> PredictionService:
    """서비스 초기화 및 반환"""
    service = get_prediction_service()
    if not service.is_ready():
        service.initialize()
    return service
