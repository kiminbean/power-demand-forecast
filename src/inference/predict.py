"""
Production Inference Script
===========================

Production 모델을 사용한 전력 수요 예측

주요 기능:
1. 단일 예측 (Single Prediction)
2. 배치 예측 (Batch Prediction)
3. Conditional 예측 (겨울철 자동 최적화)

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import TimeSeriesScaler, get_device
from models.lstm import create_model
from features import (
    add_time_features,
    add_weather_features,
    add_lag_features,
)

warnings.filterwarnings('ignore')


# ============================================================
# Data Classes
# ============================================================

@dataclass
class PredictionResult:
    """예측 결과"""
    timestamp: datetime
    predicted_demand: float
    model_used: str
    confidence: Optional[float] = None
    context: Optional[Dict] = None


@dataclass
class BatchPredictionResult:
    """배치 예측 결과"""
    timestamps: List[datetime]
    predictions: np.ndarray
    model_used: str
    metrics: Optional[Dict] = None


# ============================================================
# Production Predictor Class
# ============================================================

class ProductionPredictor:
    """
    Production 모델 예측기

    Usage:
    ------
    >>> predictor = ProductionPredictor()
    >>> predictor.load_models()
    >>> result = predictor.predict(input_data)
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Parameters
        ----------
        model_dir : str, optional
            모델 디렉토리 경로
        device : torch.device, optional
            연산 디바이스
        """
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models" / "production"
        self.device = device if device else get_device()

        self.model_demand = None
        self.model_weather = None
        self.scaler_demand = None
        self.scaler_weather = None
        self.config_demand = None
        self.config_weather = None

        self.is_loaded = False

    def load_models(self) -> None:
        """모델 로드"""
        print(f"Loading models from: {self.model_dir}")

        # Load demand_only model
        demand_path = self.model_dir / "demand_only.pt"
        if demand_path.exists():
            checkpoint = torch.load(demand_path, map_location=self.device)
            self.config_demand = checkpoint

            self.model_demand = create_model(
                model_type=checkpoint['model_config']['model_type'],
                input_size=checkpoint['n_features'],
                hidden_size=checkpoint['model_config']['hidden_size'],
                num_layers=checkpoint['model_config']['num_layers'],
                dropout=checkpoint['model_config']['dropout']
            ).to(self.device)
            self.model_demand.load_state_dict(checkpoint['model_state_dict'])
            self.model_demand.eval()

            self.scaler_demand = TimeSeriesScaler()
            self.scaler_demand.load(str(self.model_dir / "demand_only_scaler.pkl"))

            print(f"  ✓ demand_only loaded ({checkpoint['n_features']} features)")
        else:
            raise FileNotFoundError(f"Model not found: {demand_path}")

        # Load weather_full model
        weather_path = self.model_dir / "weather_full.pt"
        if weather_path.exists():
            checkpoint = torch.load(weather_path, map_location=self.device)
            self.config_weather = checkpoint

            self.model_weather = create_model(
                model_type=checkpoint['model_config']['model_type'],
                input_size=checkpoint['n_features'],
                hidden_size=checkpoint['model_config']['hidden_size'],
                num_layers=checkpoint['model_config']['num_layers'],
                dropout=checkpoint['model_config']['dropout']
            ).to(self.device)
            self.model_weather.load_state_dict(checkpoint['model_state_dict'])
            self.model_weather.eval()

            self.scaler_weather = TimeSeriesScaler()
            self.scaler_weather.load(str(self.model_dir / "weather_full_scaler.pkl"))

            print(f"  ✓ weather_full loaded ({checkpoint['n_features']} features)")
        else:
            print(f"  ⚠ weather_full not found (conditional mode disabled)")

        self.is_loaded = True
        print(f"Device: {self.device}")

    def _check_loaded(self) -> None:
        """모델 로드 확인"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

    def _prepare_features(
        self,
        df: pd.DataFrame,
        include_weather: bool = False
    ) -> pd.DataFrame:
        """피처 준비"""
        df = df.copy()

        # Ensure datetime index
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)

        # Feature engineering
        df = add_time_features(df, include_holiday=True)
        df = add_lag_features(df, demand_col='power_demand')

        if include_weather and '기온' in df.columns:
            df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)

        return df.dropna()

    def _get_sequence(
        self,
        df: pd.DataFrame,
        features: List[str],
        seq_length: int = 168
    ) -> np.ndarray:
        """시퀀스 데이터 추출"""
        available = [f for f in features if f in df.columns]
        data = df[available].values

        if len(data) < seq_length:
            raise ValueError(f"Insufficient data: need {seq_length}, got {len(data)}")

        return data[-seq_length:]

    def predict_demand_only(
        self,
        df: pd.DataFrame,
        return_tensor: bool = False
    ) -> Union[float, torch.Tensor]:
        """
        demand_only 모델로 예측

        Parameters
        ----------
        df : pd.DataFrame
            입력 데이터 (최소 168시간)
        return_tensor : bool
            텐서 반환 여부

        Returns
        -------
        float or Tensor
            예측 수요
        """
        self._check_loaded()

        # Prepare features
        df_prep = self._prepare_features(df, include_weather=False)
        features = self.config_demand['features']
        seq_length = self.config_demand['training_config']['seq_length']

        # Get sequence
        sequence = self._get_sequence(df_prep, features, seq_length)

        # Scale
        scaled = self.scaler_demand.transform(sequence)

        # To tensor
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            pred = self.model_demand(X)

        if return_tensor:
            return pred

        # Inverse transform
        pred_value = self.scaler_demand.inverse_transform_target(
            pred.cpu().numpy().flatten()
        )[0]

        return float(pred_value)

    def predict_weather_full(
        self,
        df: pd.DataFrame,
        return_tensor: bool = False
    ) -> Union[float, torch.Tensor]:
        """
        weather_full 모델로 예측

        Parameters
        ----------
        df : pd.DataFrame
            입력 데이터 (기상 데이터 포함)

        Returns
        -------
        float
            예측 수요
        """
        self._check_loaded()

        if self.model_weather is None:
            raise RuntimeError("weather_full model not loaded")

        # Prepare features
        df_prep = self._prepare_features(df, include_weather=True)
        features = self.config_weather['features']
        seq_length = self.config_weather['training_config']['seq_length']

        # Get sequence
        sequence = self._get_sequence(df_prep, features, seq_length)

        # Scale
        scaled = self.scaler_weather.transform(sequence)

        # To tensor
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            pred = self.model_weather(X)

        if return_tensor:
            return pred

        # Inverse transform
        pred_value = self.scaler_weather.inverse_transform_target(
            pred.cpu().numpy().flatten()
        )[0]

        return float(pred_value)

    def predict_conditional(
        self,
        df: pd.DataFrame,
        mode: str = "soft"
    ) -> PredictionResult:
        """
        Conditional 모델로 예측 (겨울철 자동 최적화)

        Parameters
        ----------
        df : pd.DataFrame
            입력 데이터
        mode : str
            'soft' (확률적 블렌딩) 또는 'hard' (이진 선택)

        Returns
        -------
        PredictionResult
            예측 결과
        """
        self._check_loaded()

        # Get timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            timestamp = df.index[-1]
        elif 'datetime' in df.columns:
            timestamp = df['datetime'].iloc[-1]
        else:
            timestamp = datetime.now()

        # Check if winter
        month = timestamp.month
        is_winter = month in [12, 1, 2]

        # Predict with demand_only
        pred_demand = self.predict_demand_only(df)

        # If weather model available and winter
        if self.model_weather is not None and is_winter:
            try:
                pred_weather = self.predict_weather_full(df)

                if mode == "hard":
                    # Use weather_full in winter
                    final_pred = pred_weather
                    model_used = "weather_full"
                    weather_weight = 1.0
                else:
                    # Soft blending based on month
                    if month == 1:  # Peak winter
                        weather_weight = 0.3
                    elif month in [12, 2]:
                        weather_weight = 0.2
                    else:
                        weather_weight = 0.0

                    final_pred = (1 - weather_weight) * pred_demand + weather_weight * pred_weather
                    model_used = f"conditional_soft (w={weather_weight:.1f})"

            except Exception as e:
                # Fallback to demand_only
                final_pred = pred_demand
                model_used = "demand_only (fallback)"
                weather_weight = 0.0
        else:
            final_pred = pred_demand
            model_used = "demand_only"
            weather_weight = 0.0

        return PredictionResult(
            timestamp=timestamp,
            predicted_demand=final_pred,
            model_used=model_used,
            context={
                'is_winter': is_winter,
                'month': month,
                'weather_weight': weather_weight,
                'pred_demand_only': pred_demand
            }
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        model: str = "demand_only",
        step: int = 1
    ) -> BatchPredictionResult:
        """
        배치 예측 (슬라이딩 윈도우)

        Parameters
        ----------
        df : pd.DataFrame
            입력 데이터
        model : str
            사용할 모델 ('demand_only', 'weather_full', 'conditional')
        step : int
            슬라이딩 스텝

        Returns
        -------
        BatchPredictionResult
            배치 예측 결과
        """
        self._check_loaded()

        # Prepare data
        include_weather = model in ['weather_full', 'conditional']
        df_prep = self._prepare_features(df, include_weather=include_weather)

        if model == 'demand_only':
            features = self.config_demand['features']
            seq_length = self.config_demand['training_config']['seq_length']
            scaler = self.scaler_demand
            model_obj = self.model_demand
        else:
            features = self.config_weather['features']
            seq_length = self.config_weather['training_config']['seq_length']
            scaler = self.scaler_weather
            model_obj = self.model_weather

        available = [f for f in features if f in df_prep.columns]
        data = df_prep[available].values

        # Generate predictions
        predictions = []
        timestamps = []

        n_samples = len(data) - seq_length + 1

        for i in range(0, n_samples, step):
            sequence = data[i:i + seq_length]
            scaled = scaler.transform(sequence)

            X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = model_obj(X)

            pred_value = scaler.inverse_transform_target(
                pred.cpu().numpy().flatten()
            )[0]

            predictions.append(pred_value)
            timestamps.append(df_prep.index[i + seq_length - 1])

        return BatchPredictionResult(
            timestamps=timestamps,
            predictions=np.array(predictions),
            model_used=model
        )


# ============================================================
# Convenience Functions
# ============================================================

_predictor: Optional[ProductionPredictor] = None


def get_predictor(model_dir: Optional[str] = None) -> ProductionPredictor:
    """싱글톤 예측기 반환"""
    global _predictor
    if _predictor is None:
        _predictor = ProductionPredictor(model_dir)
        _predictor.load_models()
    return _predictor


def predict(
    df: pd.DataFrame,
    model: str = "conditional",
    mode: str = "soft"
) -> Union[float, PredictionResult]:
    """
    간편 예측 함수

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터 (최소 168시간)
    model : str
        'demand_only', 'weather_full', 'conditional'
    mode : str
        conditional 모드 ('soft', 'hard')

    Returns
    -------
    float or PredictionResult
        예측 결과

    Examples
    --------
    >>> from inference.predict import predict
    >>> result = predict(df, model="conditional")
    >>> print(f"Predicted: {result.predicted_demand:.2f} MW")
    """
    predictor = get_predictor()

    if model == "demand_only":
        return predictor.predict_demand_only(df)
    elif model == "weather_full":
        return predictor.predict_weather_full(df)
    else:
        return predictor.predict_conditional(df, mode=mode)


def predict_batch(
    df: pd.DataFrame,
    model: str = "demand_only",
    step: int = 1
) -> BatchPredictionResult:
    """
    배치 예측 함수

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터
    model : str
        사용할 모델
    step : int
        슬라이딩 스텝

    Returns
    -------
    BatchPredictionResult
        배치 예측 결과
    """
    predictor = get_predictor()
    return predictor.predict_batch(df, model=model, step=step)


# ============================================================
# CLI Interface
# ============================================================

def main():
    """CLI 인터페이스"""
    import argparse

    parser = argparse.ArgumentParser(description='Production Inference')
    parser.add_argument('--data', type=str, required=True, help='Input data path')
    parser.add_argument('--model', type=str, default='conditional',
                        choices=['demand_only', 'weather_full', 'conditional'])
    parser.add_argument('--mode', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--batch', action='store_true', help='Batch prediction')
    parser.add_argument('--output', type=str, default=None, help='Output path')

    args = parser.parse_args()

    # Load data
    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, parse_dates=['datetime'])

    # Initialize predictor
    predictor = ProductionPredictor()
    predictor.load_models()

    if args.batch:
        # Batch prediction
        print(f"\nRunning batch prediction with {args.model}...")
        result = predictor.predict_batch(df, model=args.model)

        print(f"\nPredictions: {len(result.predictions)}")
        print(f"Mean: {result.predictions.mean():.2f} MW")
        print(f"Std: {result.predictions.std():.2f} MW")
        print(f"Min: {result.predictions.min():.2f} MW")
        print(f"Max: {result.predictions.max():.2f} MW")

        if args.output:
            output_df = pd.DataFrame({
                'timestamp': result.timestamps,
                'predicted_demand': result.predictions
            })
            output_df.to_csv(args.output, index=False)
            print(f"\nSaved to: {args.output}")

    else:
        # Single prediction
        if args.model == 'conditional':
            result = predictor.predict_conditional(df, mode=args.mode)
            print(f"\n[Prediction Result]")
            print(f"  Timestamp: {result.timestamp}")
            print(f"  Predicted: {result.predicted_demand:.2f} MW")
            print(f"  Model: {result.model_used}")
            print(f"  Context: {result.context}")
        else:
            if args.model == 'demand_only':
                pred = predictor.predict_demand_only(df)
            else:
                pred = predictor.predict_weather_full(df)

            print(f"\n[Prediction Result]")
            print(f"  Model: {args.model}")
            print(f"  Predicted: {pred:.2f} MW")


if __name__ == '__main__':
    main()
