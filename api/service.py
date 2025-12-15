"""
Prediction Service
==================

예측 비즈니스 로직을 담당하는 서비스 레이어
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predict import ProductionPredictor, PredictionResult, BatchPredictionResult
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


class PredictionService:
    """
    예측 서비스 클래스

    ProductionPredictor를 래핑하여 API에 필요한 형태로 변환
    """

    def __init__(self):
        self.predictor: Optional[ProductionPredictor] = None
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
            self._is_initialized = True
            logger.info("Prediction service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}")
            raise

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
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        return df

    def _prepare_features_for_api(self, df: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
        """API용 피처 준비 (NaN 처리 포함)"""
        from features import add_time_features, add_lag_features, add_weather_features

        df = df.copy()

        # 시간 피처 추가
        df = add_time_features(df, include_holiday=True)

        # 지연 피처 추가
        df = add_lag_features(df, demand_col='power_demand')

        # 기상 피처 추가 (선택적)
        if include_weather and '기온' in df.columns:
            df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)

        # NaN 처리: backward fill 후 forward fill (시작과 끝 부분 모두 처리)
        df = df.bfill().ffill()

        # 여전히 NaN이 있으면 0으로 채움
        df = df.fillna(0)

        return df

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

        # 피처 준비 (NaN 처리 포함)
        df_prep = self._prepare_features_for_api(df, include_weather=include_weather)

        # 필요한 피처만 추출
        features = config['features']
        seq_length = config['training_config']['seq_length']

        available = [f for f in features if f in df_prep.columns]
        if len(available) < len(features) * 0.5:
            logger.warning(f"Only {len(available)}/{len(features)} features available")

        data = df_prep[available].values

        if len(data) < seq_length:
            raise ValueError(f"Insufficient data: need {seq_length}, got {len(data)}")

        # 시퀀스 추출
        sequence = data[-seq_length:]

        # 스케일링
        scaled = scaler.transform(sequence)

        # 텐서 변환 및 예측
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.predictor.device)

        with torch.no_grad():
            pred = model(X)

        # 역변환
        pred_value = scaler.inverse_transform_target(
            pred.cpu().numpy().flatten()
        )[0]

        return float(pred_value)

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

        start_time = time.perf_counter()

        # Convert to DataFrame
        df = self._convert_to_dataframe(data)

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

        # Prepare features
        df_prep = self._prepare_features_for_api(df, include_weather=include_weather)

        features = config['features']
        seq_length = config['training_config']['seq_length']

        available = [f for f in features if f in df_prep.columns]
        data_array = df_prep[available].values

        # Generate batch predictions
        predictions_list = []
        timestamps_list = []
        n_samples = len(data_array) - seq_length + 1

        for i in range(0, n_samples, step):
            sequence = data_array[i:i + seq_length]
            scaled = scaler.transform(sequence)

            X = torch.FloatTensor(scaled).unsqueeze(0).to(self.predictor.device)

            with torch.no_grad():
                pred = model(X)

            pred_value = scaler.inverse_transform_target(
                pred.cpu().numpy().flatten()
            )[0]

            predictions_list.append(pred_value)
            timestamps_list.append(df_prep.index[i + seq_length - 1])

        predictions_array = np.array(predictions_list)

        # Format predictions
        predictions = [
            {
                "timestamp": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                "prediction": round(float(pred), 2)
            }
            for ts, pred in zip(timestamps_list, predictions_array)
        ]

        # Calculate statistics
        statistics = {
            "mean": round(float(predictions_array.mean()), 2),
            "std": round(float(predictions_array.std()), 2),
            "min": round(float(predictions_array.min()), 2),
            "max": round(float(predictions_array.max()), 2)
        }

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
