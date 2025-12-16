"""
Multi-horizon 예측 모듈 (Task 12)
================================
다중 시간대 예측을 위한 모델 및 유틸리티를 제공합니다.

주요 컴포넌트:
- MultiHorizonPredictor: 다중 시간대 예측기
- HorizonSpecificModel: 시간대별 특화 모델
- RecursivePredictor: 재귀적 예측기
- DirectMultiOutput: 직접 다중 출력 모델
- HybridPredictor: 하이브리드 예측기
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PredictionStrategy(Enum):
    """예측 전략"""
    RECURSIVE = 'recursive'          # 재귀적: 이전 예측을 다음 입력으로
    DIRECT = 'direct'                # 직접: 각 시간대별 독립 모델
    MULTI_OUTPUT = 'multi_output'    # 다중 출력: 하나의 모델이 모든 시간대 예측
    HYBRID = 'hybrid'                # 하이브리드: 전략 조합


@dataclass
class HorizonConfig:
    """시간대 설정"""
    horizons: List[int]              # 예측 시간대 (시간 단위)
    strategy: PredictionStrategy = PredictionStrategy.MULTI_OUTPUT
    aggregation: str = 'mean'        # 앙상블 집계 방식
    confidence_level: float = 0.95   # 신뢰 수준


@dataclass
class HorizonPrediction:
    """시간대별 예측 결과"""
    horizon: int                     # 예측 시간대
    prediction: np.ndarray           # 예측값
    lower_bound: Optional[np.ndarray] = None  # 하한
    upper_bound: Optional[np.ndarray] = None  # 상한
    confidence: float = 1.0          # 신뢰도
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHorizonResult:
    """다중 시간대 예측 결과"""
    predictions: Dict[int, HorizonPrediction]  # horizon -> prediction
    timestamps: Optional[List] = None
    model_name: str = ''

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame 변환"""
        data = {}
        for horizon, pred in self.predictions.items():
            data[f'pred_h{horizon}'] = pred.prediction
            if pred.lower_bound is not None:
                data[f'lower_h{horizon}'] = pred.lower_bound
            if pred.upper_bound is not None:
                data[f'upper_h{horizon}'] = pred.upper_bound

        df = pd.DataFrame(data)
        if self.timestamps is not None:
            df['timestamp'] = self.timestamps
            df = df.set_index('timestamp')

        return df


class DirectMultiOutputNet(nn.Module):
    """
    직접 다중 출력 신경망

    하나의 모델이 모든 시간대를 동시에 예측합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizons: List[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: 입력 피처 수
            hidden_size: 히든 크기
            num_layers: 레이어 수
            horizons: 예측 시간대 리스트
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizons = horizons or [1, 6, 12, 24]
        self.num_horizons = len(self.horizons)

        # 공유 인코더
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 시간대별 디코더 헤드
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in self.horizons
        ])

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, features)

        Returns:
            시간대별 예측 딕셔너리
        """
        # 인코딩
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]  # 마지막 레이어 히든 상태

        # 시간대별 예측
        predictions = {}
        for i, horizon in enumerate(self.horizons):
            pred = self.horizon_heads[i](hidden)
            predictions[horizon] = pred

        return predictions


class RecursiveLSTM(nn.Module):
    """
    재귀적 LSTM

    한 스텝씩 예측하고 결과를 다음 입력으로 사용합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """단일 스텝 예측"""
        output, (hidden, cell) = self.lstm(x)
        pred = self.output_layer(output[:, -1, :])
        return pred

    def predict_multi_step(
        self,
        x: torch.Tensor,
        horizons: List[int],
        feature_idx: int = 0
    ) -> Dict[int, torch.Tensor]:
        """
        다중 스텝 예측

        Args:
            x: 입력 텐서
            horizons: 예측 시간대
            feature_idx: 타겟 피처 인덱스

        Returns:
            시간대별 예측
        """
        batch_size = x.size(0)
        max_horizon = max(horizons)

        predictions = {}
        current_input = x.clone()

        # 재귀적 예측
        all_preds = []
        for step in range(max_horizon):
            pred = self.forward(current_input)
            all_preds.append(pred)

            # 다음 입력 준비
            new_step = current_input[:, -1:, :].clone()
            new_step[:, 0, feature_idx] = pred.squeeze(-1)
            current_input = torch.cat([current_input[:, 1:, :], new_step], dim=1)

        # 요청된 시간대의 예측 추출
        all_preds = torch.stack(all_preds, dim=1)  # (batch, max_horizon, 1)
        for horizon in horizons:
            predictions[horizon] = all_preds[:, horizon - 1, :]

        return predictions


class HorizonSpecificModel(nn.Module):
    """
    시간대별 특화 모델

    각 시간대마다 독립적인 모델을 사용합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        horizons: List[int] = None
    ):
        super().__init__()

        self.input_size = input_size
        self.horizons = horizons or [1, 6, 12, 24]

        # 시간대별 독립 모델
        self.models = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            for h in self.horizons
        })

    def forward(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, features)
            horizon: 특정 시간대 (None이면 모두)

        Returns:
            예측값
        """
        if horizon is not None:
            return self.models[str(horizon)](x)

        return {h: self.models[str(h)](x) for h in self.horizons}


class AttentionMultiHorizon(nn.Module):
    """
    Attention 기반 다중 시간대 모델

    시간대별로 서로 다른 attention을 적용합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        horizons: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizons = horizons or [1, 6, 12, 24]

        # 입력 임베딩
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # 시간대별 어텐션
        self.horizon_attention = nn.ModuleDict({
            str(h): nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for h in self.horizons
        })

        # 출력 레이어
        self.output_layers = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for h in self.horizons
        })

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """순전파"""
        # 입력 임베딩
        embedded = self.input_embedding(x)  # (batch, seq, hidden)

        predictions = {}
        for horizon in self.horizons:
            # 어텐션 적용
            attn_output, _ = self.horizon_attention[str(horizon)](
                embedded, embedded, embedded
            )

            # 마지막 타임스텝 사용
            final_repr = attn_output[:, -1, :]

            # 예측
            pred = self.output_layers[str(horizon)](final_repr)
            predictions[horizon] = pred

        return predictions


class MultiHorizonPredictor:
    """
    다중 시간대 예측기

    다양한 전략을 사용하여 다중 시간대 예측을 수행합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        config: HorizonConfig,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: 예측 모델
            config: 시간대 설정
            device: 연산 디바이스
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')

        self.model.to(self.device)

    def predict(
        self,
        X: np.ndarray,
        return_intervals: bool = False
    ) -> MultiHorizonResult:
        """
        다중 시간대 예측

        Args:
            X: 입력 데이터 (batch, seq_len, features)
            return_intervals: 신뢰 구간 반환 여부

        Returns:
            다중 시간대 예측 결과
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            raw_predictions = self.model(X_tensor)

        predictions = {}
        for horizon, pred in raw_predictions.items():
            pred_np = pred.cpu().numpy().flatten()

            # 신뢰 구간 계산 (선택적)
            lower, upper = None, None
            if return_intervals:
                # 간단한 신뢰 구간 (실제로는 불확실성 추정 필요)
                std = np.std(pred_np) * (1 + horizon * 0.1)
                z = 1.96  # 95% 신뢰 구간
                lower = pred_np - z * std
                upper = pred_np + z * std

            predictions[horizon] = HorizonPrediction(
                horizon=horizon,
                prediction=pred_np,
                lower_bound=lower,
                upper_bound=upper,
                confidence=1.0 - horizon * 0.01  # 시간대가 멀수록 신뢰도 감소
            )

        return MultiHorizonResult(
            predictions=predictions,
            model_name=self.model.__class__.__name__
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> MultiHorizonResult:
        """
        불확실성을 포함한 예측 (MC Dropout)

        Args:
            X: 입력 데이터
            n_samples: 샘플 수

        Returns:
            다중 시간대 예측 결과
        """
        self.model.train()  # 드롭아웃 활성화

        X_tensor = torch.FloatTensor(X).to(self.device)

        # 여러 번 예측
        all_predictions = {h: [] for h in self.config.horizons}

        with torch.no_grad():
            for _ in range(n_samples):
                raw_predictions = self.model(X_tensor)
                for horizon, pred in raw_predictions.items():
                    all_predictions[horizon].append(pred.cpu().numpy())

        # 통계 계산
        predictions = {}
        for horizon in self.config.horizons:
            preds = np.array(all_predictions[horizon])
            mean_pred = np.mean(preds, axis=0).flatten()
            std_pred = np.std(preds, axis=0).flatten()

            z = 1.96  # 95% 신뢰 구간
            predictions[horizon] = HorizonPrediction(
                horizon=horizon,
                prediction=mean_pred,
                lower_bound=mean_pred - z * std_pred,
                upper_bound=mean_pred + z * std_pred,
                confidence=1.0 - std_pred.mean() / (np.abs(mean_pred).mean() + 1e-8)
            )

        self.model.eval()

        return MultiHorizonResult(
            predictions=predictions,
            model_name=f"{self.model.__class__.__name__}_MCDropout"
        )


class HybridMultiHorizonPredictor:
    """
    하이브리드 다중 시간대 예측기

    단기/장기 예측에 서로 다른 전략을 적용합니다.
    """

    def __init__(
        self,
        short_term_model: nn.Module,
        long_term_model: nn.Module,
        short_term_horizons: List[int],
        long_term_horizons: List[int],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            short_term_model: 단기 예측 모델
            long_term_model: 장기 예측 모델
            short_term_horizons: 단기 시간대
            long_term_horizons: 장기 시간대
            device: 연산 디바이스
        """
        self.short_term_model = short_term_model
        self.long_term_model = long_term_model
        self.short_term_horizons = short_term_horizons
        self.long_term_horizons = long_term_horizons
        self.device = device or torch.device('cpu')

        self.short_term_model.to(self.device)
        self.long_term_model.to(self.device)

    def predict(self, X: np.ndarray) -> MultiHorizonResult:
        """다중 시간대 예측"""
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.short_term_model.eval()
        self.long_term_model.eval()

        predictions = {}

        with torch.no_grad():
            # 단기 예측
            short_preds = self.short_term_model(X_tensor)
            for horizon, pred in short_preds.items():
                if horizon in self.short_term_horizons:
                    predictions[horizon] = HorizonPrediction(
                        horizon=horizon,
                        prediction=pred.cpu().numpy().flatten(),
                        metadata={'model': 'short_term'}
                    )

            # 장기 예측
            long_preds = self.long_term_model(X_tensor)
            for horizon, pred in long_preds.items():
                if horizon in self.long_term_horizons:
                    predictions[horizon] = HorizonPrediction(
                        horizon=horizon,
                        prediction=pred.cpu().numpy().flatten(),
                        metadata={'model': 'long_term'}
                    )

        return MultiHorizonResult(
            predictions=predictions,
            model_name='Hybrid'
        )


class EnsembleMultiHorizonPredictor:
    """
    앙상블 다중 시간대 예측기

    여러 모델의 예측을 결합합니다.
    """

    def __init__(
        self,
        models: List[nn.Module],
        horizons: List[int],
        weights: Optional[Dict[int, List[float]]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            models: 모델 리스트
            horizons: 예측 시간대
            weights: 시간대별 모델 가중치
            device: 연산 디바이스
        """
        self.models = models
        self.horizons = horizons
        self.weights = weights or {h: [1.0 / len(models)] * len(models) for h in horizons}
        self.device = device or torch.device('cpu')

        for model in self.models:
            model.to(self.device)

    def predict(self, X: np.ndarray) -> MultiHorizonResult:
        """앙상블 예측"""
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 각 모델의 예측 수집
        all_predictions = {h: [] for h in self.horizons}

        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds = model(X_tensor)
                for horizon, pred in preds.items():
                    if horizon in self.horizons:
                        all_predictions[horizon].append(pred.cpu().numpy())

        # 가중 평균
        predictions = {}
        for horizon in self.horizons:
            preds = np.array(all_predictions[horizon])  # (n_models, batch, 1)
            weights = np.array(self.weights[horizon])

            # 평균 계산
            weighted_mean = np.average(preds, axis=0, weights=weights).flatten()

            # 표준편차 계산 (broadcasting 수정)
            deviations = preds - weighted_mean.reshape(1, -1, 1)
            weighted_var = np.average(deviations ** 2, axis=0, weights=weights)
            weighted_std = np.sqrt(weighted_var).flatten()

            predictions[horizon] = HorizonPrediction(
                horizon=horizon,
                prediction=weighted_mean,
                lower_bound=weighted_mean - 1.96 * weighted_std,
                upper_bound=weighted_mean + 1.96 * weighted_std
            )

        return MultiHorizonResult(
            predictions=predictions,
            model_name='Ensemble'
        )


class MultiHorizonTrainer:
    """
    다중 시간대 모델 학습기
    """

    def __init__(
        self,
        model: nn.Module,
        horizons: List[int],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: 학습할 모델
            horizons: 예측 시간대
            device: 연산 디바이스
        """
        self.model = model
        self.horizons = horizons
        self.device = device or torch.device('cpu')

        self.model.to(self.device)

    def train(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        모델 학습

        Args:
            X: 입력 데이터 (n_samples, seq_len, features)
            y: 시간대별 타겟 (horizon -> (n_samples,))
            epochs: 에포크 수
            learning_rate: 학습률
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율

        Returns:
            학습 이력
        """
        # 데이터 분할
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)

        y_train = {h: torch.FloatTensor(y[h][train_idx]).to(self.device) for h in self.horizons}
        y_val = {h: torch.FloatTensor(y[h][val_idx]).to(self.device) for h in self.horizons}

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # 학습
            self.model.train()
            n_batches = len(train_idx) // batch_size
            epoch_loss = 0.0

            for i in range(0, len(train_idx), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = {h: y_train[h][i:i + batch_size] for h in self.horizons}

                optimizer.zero_grad()
                predictions = self.model(batch_X)

                loss = 0
                for horizon in self.horizons:
                    pred = predictions[horizon].squeeze()
                    target = batch_y[horizon]
                    loss += criterion(pred, target)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            history['train_loss'].append(epoch_loss / max(1, n_batches))

            # 검증
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val)
                val_loss = 0
                for horizon in self.horizons:
                    pred = val_preds[horizon].squeeze()
                    val_loss += criterion(pred, y_val[horizon]).item()

            history['val_loss'].append(val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {history['train_loss'][-1]:.6f}, "
                    f"Val Loss: {history['val_loss'][-1]:.6f}"
                )

        return history


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_direct_multi_output_model(
    input_size: int,
    hidden_size: int = 128,
    horizons: List[int] = None
) -> DirectMultiOutputNet:
    """직접 다중 출력 모델 생성"""
    return DirectMultiOutputNet(
        input_size=input_size,
        hidden_size=hidden_size,
        horizons=horizons or [1, 6, 12, 24]
    )


def create_attention_multi_horizon_model(
    input_size: int,
    hidden_size: int = 128,
    num_heads: int = 4,
    horizons: List[int] = None
) -> AttentionMultiHorizon:
    """Attention 기반 다중 시간대 모델 생성"""
    return AttentionMultiHorizon(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        horizons=horizons or [1, 6, 12, 24]
    )


def create_multi_horizon_predictor(
    model: nn.Module,
    horizons: List[int] = None,
    strategy: str = 'multi_output',
    device: Optional[torch.device] = None
) -> MultiHorizonPredictor:
    """다중 시간대 예측기 생성"""
    strategy_map = {
        'recursive': PredictionStrategy.RECURSIVE,
        'direct': PredictionStrategy.DIRECT,
        'multi_output': PredictionStrategy.MULTI_OUTPUT,
        'hybrid': PredictionStrategy.HYBRID
    }

    config = HorizonConfig(
        horizons=horizons or [1, 6, 12, 24],
        strategy=strategy_map.get(strategy, PredictionStrategy.MULTI_OUTPUT)
    )

    return MultiHorizonPredictor(model, config, device)
