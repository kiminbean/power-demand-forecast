"""
SMP LSTM 예측 모델
==================

LSTM 기반 24시간 SMP(계통한계가격) 예측 모델

주요 기능:
1. 48시간 입력 → 24시간 예측
2. 다중 출력 (24개 시간별 SMP)
3. 불확실성 추정을 위한 Quantile 예측 (10%, 50%, 90%)
4. BiLSTM 및 Attention 지원
5. MPS (Apple Silicon) 최적화

Architecture:
    Input (48h) → LSTM Encoder → Attention → FC → 24h Predictions

Author: Claude Code
Date: 2025-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


def get_device() -> torch.device:
    """최적 디바이스 반환 (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TemporalAttention(nn.Module):
    """시간 축 Attention 레이어

    LSTM 출력에서 중요한 시간 단계에 가중치를 부여합니다.

    Args:
        hidden_size: LSTM hidden state 크기
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) - 가중 평균된 컨텍스트 벡터
            weights: (batch, seq_len) - 어텐션 가중치
        """
        # (batch, seq_len, 1)
        scores = self.attention(lstm_output)
        weights = F.softmax(scores.squeeze(-1), dim=-1)

        # 가중 합계
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights


class SMPLSTMModel(nn.Module):
    """SMP 예측용 LSTM 모델

    48시간 과거 데이터로 24시간 미래 SMP를 예측합니다.

    Architecture:
        Input → LSTM → Temporal Attention → FC Decoder → 24h Output

    Args:
        input_size: 입력 피처 수
        hidden_size: LSTM hidden state 크기 (default: 128)
        num_layers: LSTM 레이어 수 (default: 2)
        dropout: Dropout 비율 (default: 0.2)
        bidirectional: 양방향 LSTM 사용 여부 (default: True)
        prediction_hours: 예측 시간 수 (default: 24)
        use_attention: Attention 사용 여부 (default: True)

    Example:
        >>> model = SMPLSTMModel(input_size=20, hidden_size=128)
        >>> x = torch.randn(32, 48, 20)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> print(output.shape)  # (32, 24)  # 24시간 예측
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        prediction_hours: int = 24,
        use_attention: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.prediction_hours = prediction_hours
        self.use_attention = use_attention

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        if use_attention:
            self.attention = TemporalAttention(lstm_output_size)

        # FC Decoder
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_hours)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """순전파

        Args:
            x: (batch, seq_len, input_size)
            return_attention: Attention 가중치 반환 여부

        Returns:
            output: (batch, prediction_hours) or
            (output, attention_weights) if return_attention=True
        """
        # Input normalization
        x = self.input_norm(x)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Attention or last hidden state
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            if self.bidirectional:
                context = torch.cat([
                    lstm_out[:, -1, :self.hidden_size],
                    lstm_out[:, 0, self.hidden_size:]
                ], dim=1)
            else:
                context = lstm_out[:, -1, :]
            attn_weights = None

        # Decoder
        output = self.decoder(context)

        if return_attention and attn_weights is not None:
            return output, attn_weights
        return output

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SMPQuantileLSTM(nn.Module):
    """SMP Quantile 예측 모델

    불확실성 추정을 위해 여러 분위수(10%, 50%, 90%)를 동시 예측합니다.

    Args:
        input_size: 입력 피처 수
        hidden_size: LSTM hidden state 크기
        num_layers: LSTM 레이어 수
        dropout: Dropout 비율
        bidirectional: 양방향 LSTM 사용 여부
        prediction_hours: 예측 시간 수
        quantiles: 예측할 분위수 리스트

    Example:
        >>> model = SMPQuantileLSTM(input_size=20, quantiles=[0.1, 0.5, 0.9])
        >>> x = torch.randn(32, 48, 20)
        >>> q10, q50, q90 = model(x)
        >>> print(q50.shape)  # (32, 24)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        prediction_hours: int = 24,
        quantiles: List[float] = None
    ):
        super().__init__()

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.prediction_hours = prediction_hours
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Shared LSTM encoder
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = TemporalAttention(lstm_output_size)

        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Quantile heads (각 분위수별 독립 출력)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, prediction_hours)
            for _ in quantiles
        ])

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in self.shared_fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)

        for head in self.quantile_heads:
            nn.init.kaiming_normal_(head.weight, nonlinearity='linear')
            if head.bias is not None:
                head.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """순전파

        Args:
            x: (batch, seq_len, input_size)
            return_dict: 딕셔너리로 반환 여부

        Returns:
            tuple of tensors: (q1, q2, q3, ...) 각 (batch, prediction_hours)
            or dict: {'q10': tensor, 'q50': tensor, ...} if return_dict=True
        """
        # Input normalization
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context, _ = self.attention(lstm_out)

        # Shared features
        features = self.shared_fc(context)

        # Quantile outputs
        outputs = []
        for head in self.quantile_heads:
            outputs.append(head(features))

        if return_dict:
            return {
                f'q{int(q*100)}': out
                for q, out in zip(self.quantiles, outputs)
            }

        return tuple(outputs)

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QuantileLoss(nn.Module):
    """Quantile Loss (Pinball Loss)

    분위수 회귀를 위한 손실 함수

    Args:
        quantiles: 분위수 리스트 (예: [0.1, 0.5, 0.9])
    """

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: 각 분위수별 예측값 튜플
            targets: 실제값 (batch, prediction_hours)

        Returns:
            loss: 평균 Quantile Loss
        """
        losses = []
        for q, pred in zip(self.quantiles, predictions):
            errors = targets - pred
            loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss.mean())

        return sum(losses) / len(losses)


def create_smp_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
    prediction_hours: int = 24,
    quantiles: List[float] = None,
    **kwargs
) -> nn.Module:
    """SMP 모델 팩토리 함수

    Args:
        model_type: 모델 타입 ('lstm', 'quantile')
        input_size: 입력 피처 수
        hidden_size: hidden state 크기
        num_layers: LSTM 레이어 수
        dropout: Dropout 비율
        bidirectional: 양방향 LSTM 사용 여부
        prediction_hours: 예측 시간 수
        quantiles: 분위수 리스트 (quantile 모델용)

    Returns:
        nn.Module: 생성된 모델
    """
    model_type = model_type.lower()

    if model_type == 'lstm':
        return SMPLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            prediction_hours=prediction_hours,
            **kwargs
        )

    elif model_type == 'quantile':
        return SMPQuantileLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            prediction_hours=prediction_hours,
            quantiles=quantiles,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_summary(model: nn.Module) -> str:
    """모델 구조 요약"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable:,}")
    lines.append("-" * 60)

    for name, module in model.named_children():
        lines.append(f"  {name}: {module.__class__.__name__}")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # 테스트
    print("SMP LSTM Model Test")
    print("=" * 60)

    # 모델 생성
    model = SMPLSTMModel(input_size=20, hidden_size=128, num_layers=2)
    print(model_summary(model))

    # 테스트 입력
    batch_size = 8
    seq_len = 48
    input_size = 20

    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Quantile 모델 테스트
    print("\n" + "=" * 60)
    print("Quantile Model Test")
    print("=" * 60)

    q_model = SMPQuantileLSTM(input_size=20, quantiles=[0.1, 0.5, 0.9])
    print(model_summary(q_model))

    q10, q50, q90 = q_model(x)
    print(f"\nQ10 shape: {q10.shape}")
    print(f"Q50 shape: {q50.shape}")
    print(f"Q90 shape: {q90.shape}")

    # Loss 테스트
    targets = torch.randn(batch_size, 24)
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    loss = loss_fn((q10, q50, q90), targets)
    print(f"\nQuantile Loss: {loss.item():.4f}")
