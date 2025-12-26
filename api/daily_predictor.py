"""
Daily Power Demand Predictor
============================

일간 전력 수요 예측을 위한 BiLSTM 모델 로더 및 예측기

v18 BiLSTM 모델 사양:
- input_size: 31 (features)
- hidden_size: 96
- num_layers: 1
- bidirectional: True
- seq_length: 7 (일)

성능:
- MAPE: 6.17%
- R²: 0.726
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class DailyBiLSTM(nn.Module):
    """
    일간 전력 수요 예측용 BiLSTM 모델

    v18 모델 구조에 맞춘 아키텍처:
    - BiLSTM + BatchNorm + FC
    """

    def __init__(
        self,
        input_size: int = 31,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # BatchNorm (bidirectional이므로 hidden_size * 2)
        self.bn = nn.BatchNorm1d(hidden_size * 2)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝만 사용
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size * 2)

        # BatchNorm
        bn_out = self.bn(last_hidden)

        # FC
        output = self.fc(bn_out)

        return output


@dataclass
class DailyPredictionConfig:
    """일간 예측 설정"""
    model_path: str = "models/v18_bilstm_btm.pt"
    seq_length: int = 7  # 7일
    input_size: int = 31
    hidden_size: int = 96
    num_layers: int = 1

    # 피처 목록 (v18 모델 기준)
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [
                # 래그 피처
                'power_rolling_mean_3', 'power_rolling_max_3', 'power_rolling_min_3',
                'power_rolling_mean_7', 'power_lag_1', 'power_rolling_max_7',
                'power_rolling_min_7', 'power_rolling_mean_14', 'power_rolling_max_14',
                'power_rolling_min_14', 'power_lag_2', 'power_lag_3', 'power_lag_7',
                'year', 'power_lag_365',
                # 기상 피처
                'CDD', 'HDD',
                # 시간 피처
                'month_sin', 'month_cos', 'is_weekend',
                # 추가 피처 (31개 맞추기)
                'power_mwh', 'avg_temp', 'min_temp', 'max_temp',
                'sunlight', 'dew_point', 'visitors', 'ev_cumulative',
                'dayofweek_sin', 'dayofweek_cos', 'THI'
            ]


class DailyPredictor:
    """
    일간 전력 수요 예측기

    Usage:
        predictor = DailyPredictor()
        predictor.load_model()
        prediction = predictor.predict(daily_data)
    """

    def __init__(self, config: Optional[DailyPredictionConfig] = None):
        self.config = config or DailyPredictionConfig()
        self.model: Optional[DailyBiLSTM] = None
        self.device = self._get_device()
        self.scaler_params: Optional[Dict] = None
        self._is_loaded = False

    def _get_device(self) -> torch.device:
        """Get best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        모델 로드

        Args:
            model_path: 모델 파일 경로 (없으면 config 사용)

        Returns:
            성공 여부
        """
        path = Path(model_path or self.config.model_path)

        if not path.exists():
            logger.error(f"Model file not found: {path}")
            return False

        try:
            # 모델 생성
            self.model = DailyBiLSTM(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers
            )

            # 가중치 로드
            state_dict = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            logger.info(f"Daily BiLSTM model loaded from {path}")
            logger.info(f"Device: {self.device}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def is_ready(self) -> bool:
        """모델 로드 상태 확인"""
        return self._is_loaded and self.model is not None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        일간 예측용 피처 준비

        Args:
            df: 원본 데이터프레임 (일별 데이터)

        Returns:
            피처가 추가된 데이터프레임
        """
        df = df.copy()

        # 전력 수요 컬럼 확인
        if 'power_mwh' not in df.columns and 'power_demand' in df.columns:
            df['power_mwh'] = df['power_demand']

        # 래그 피처
        for lag in [1, 2, 3, 7, 14, 365]:
            col = f'power_lag_{lag}'
            if col not in df.columns:
                df[col] = df['power_mwh'].shift(lag)

        # 롤링 피처
        for window in [3, 7, 14]:
            if f'power_rolling_mean_{window}' not in df.columns:
                df[f'power_rolling_mean_{window}'] = df['power_mwh'].rolling(window).mean()
                df[f'power_rolling_max_{window}'] = df['power_mwh'].rolling(window).max()
                df[f'power_rolling_min_{window}'] = df['power_mwh'].rolling(window).min()

        # 시간 피처
        if hasattr(df.index, 'month'):
            dt = df.index
        else:
            dt = pd.to_datetime(df.index)

        if 'year' not in df.columns:
            df['year'] = dt.year
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        if 'month_cos' not in df.columns:
            df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        if 'dayofweek_sin' not in df.columns:
            df['dayofweek_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        if 'dayofweek_cos' not in df.columns:
            df['dayofweek_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (dt.dayofweek >= 5).astype(int)

        # 기온 관련 피처
        if 'avg_temp' in df.columns:
            if 'HDD' not in df.columns:
                df['HDD'] = np.maximum(18 - df['avg_temp'], 0)
            if 'CDD' not in df.columns:
                df['CDD'] = np.maximum(df['avg_temp'] - 18, 0)
            if 'THI' not in df.columns and 'humidity' in df.columns:
                # Temperature-Humidity Index
                T = df['avg_temp']
                RH = df.get('humidity', 50)
                df['THI'] = 1.8 * T - 0.55 * (1 - RH/100) * (1.8 * T - 26) + 32

        # 결측치 처리
        df = df.bfill().ffill().fillna(0)

        return df

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1
    ) -> Tuple[float, Dict]:
        """
        일간 전력 수요 예측

        Args:
            df: 일별 전력 수요 데이터 (최소 7일)
            horizon: 예측 기간 (일), 현재는 1일만 지원

        Returns:
            (예측값, 메타데이터)
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 피처 준비
        df_prep = self._prepare_features(df)

        # 사용 가능한 피처 확인
        available_features = [f for f in self.config.features if f in df_prep.columns]

        if len(available_features) < self.config.input_size * 0.5:
            logger.warning(
                f"Only {len(available_features)}/{self.config.input_size} features available. "
                f"Missing: {set(self.config.features) - set(available_features)}"
            )

        # 부족한 피처는 0으로 채움
        for feat in self.config.features:
            if feat not in df_prep.columns:
                df_prep[feat] = 0

        # 시퀀스 추출
        data = df_prep[self.config.features].values

        if len(data) < self.config.seq_length:
            raise ValueError(
                f"Insufficient data: need {self.config.seq_length} days, got {len(data)}"
            )

        sequence = data[-self.config.seq_length:]

        # Min-Max 스케일링 (간단한 정규화)
        seq_min = sequence.min(axis=0, keepdims=True)
        seq_max = sequence.max(axis=0, keepdims=True)
        seq_range = seq_max - seq_min
        seq_range[seq_range == 0] = 1  # 0으로 나누기 방지

        scaled = (sequence - seq_min) / seq_range

        # 텐서 변환
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        # 예측
        with torch.no_grad():
            pred_scaled = self.model(X)

        # 역변환 (전력 수요 컬럼 기준)
        power_idx = self.config.features.index('power_mwh') if 'power_mwh' in self.config.features else 0
        pred_value = pred_scaled.cpu().numpy().flatten()[0]

        # 스케일 복원 (대략적인 범위 사용)
        power_min = df['power_mwh'].min() if 'power_mwh' in df.columns else df.iloc[:, 0].min()
        power_max = df['power_mwh'].max() if 'power_mwh' in df.columns else df.iloc[:, 0].max()
        prediction = pred_value * (power_max - power_min) + power_min

        metadata = {
            'model': 'BiLSTM (v18)',
            'input_days': self.config.seq_length,
            'features_used': len(available_features),
            'device': str(self.device),
            'horizon_days': horizon
        }

        return float(prediction), metadata

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'name': 'daily_bilstm',
            'type': 'BiLSTM',
            'version': 'v18',
            'input_size': self.config.input_size,
            'hidden_size': self.config.hidden_size,
            'seq_length': self.config.seq_length,
            'performance': {
                'MAPE': '6.17%',
                'R2': '0.726'
            },
            'status': 'loaded' if self._is_loaded else 'not_loaded'
        }


# 싱글톤 인스턴스
_daily_predictor: Optional[DailyPredictor] = None


def get_daily_predictor() -> DailyPredictor:
    """일간 예측기 싱글톤 반환"""
    global _daily_predictor
    if _daily_predictor is None:
        _daily_predictor = DailyPredictor()
    return _daily_predictor


def initialize_daily_predictor(model_path: Optional[str] = None) -> DailyPredictor:
    """일간 예측기 초기화 및 모델 로드"""
    predictor = get_daily_predictor()
    if not predictor.is_ready():
        predictor.load_model(model_path)
    return predictor
