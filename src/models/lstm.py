"""
MODEL-003: LSTM 모델 구현
========================

시계열 전력 수요 예측을 위한 LSTM 모델

주요 기능:
1. LSTM/BiLSTM 지원
2. Single/Multi-horizon 예측
3. Dropout 및 Residual Connection
4. MPS (Apple Silicon) 지원

하이퍼파라미터 (feature_list.json 기준):
- hidden_size: 64
- num_layers: 2
- dropout: 0.2

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class LSTMModel(nn.Module):
    """
    LSTM 기반 시계열 예측 모델

    단일 horizon 예측용 (1시간, 6시간, 12시간, 24시간 중 하나)

    Architecture:
        Input → LSTM → Dropout → FC Layers → Output

    Args:
        input_size: 입력 특성 수
        hidden_size: LSTM hidden state 크기 (default: 64)
        num_layers: LSTM 레이어 수 (default: 2)
        dropout: Dropout 비율 (default: 0.2)
        bidirectional: 양방향 LSTM 사용 여부 (default: False)
        output_size: 출력 크기 (default: 1)

    Example:
        >>> model = LSTMModel(input_size=38, hidden_size=64, num_layers=2)
        >>> x = torch.randn(32, 48, 38)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> print(output.shape)  # (32, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 양방향인 경우 hidden_size * 2
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 (Xavier/He initialization)"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias를 1로 설정 (gradient vanishing 방지)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, seq_length, input_size)
            hidden: 초기 hidden state (optional)

        Returns:
            output: 예측값 (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # 마지막 타임스텝의 출력 사용
        if self.bidirectional:
            # 양방향인 경우: forward의 마지막 + backward의 첫 번째
            last_output = torch.cat([
                lstm_out[:, -1, :self.hidden_size],
                lstm_out[:, 0, self.hidden_size:]
            ], dim=1)
        else:
            last_output = lstm_out[:, -1, :]

        # FC layers
        output = self.fc(last_output)

        return output

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiHorizonLSTM(nn.Module):
    """
    다중 예측 시간대 LSTM 모델

    여러 horizon (1h, 6h, 12h, 24h)을 동시에 예측

    Architecture:
        Input → LSTM → Shared FC → Multi-Head Outputs

    Args:
        input_size: 입력 특성 수
        hidden_size: LSTM hidden state 크기
        num_layers: LSTM 레이어 수
        dropout: Dropout 비율
        horizons: 예측 시간대 리스트 (default: [1, 6, 12, 24])
        bidirectional: 양방향 LSTM 사용 여부

    Example:
        >>> model = MultiHorizonLSTM(input_size=38, horizons=[1, 6, 12, 24])
        >>> x = torch.randn(32, 48, 38)
        >>> output = model(x)
        >>> print(output.shape)  # (32, 4)  # 4 horizons
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizons: List[int] = None,
        bidirectional: bool = False
    ):
        super().__init__()

        if horizons is None:
            horizons = [1, 6, 12, 24]

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.horizons = horizons
        self.num_horizons = len(horizons)
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 양방향인 경우 hidden_size * 2
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head outputs (각 horizon별 독립적인 출력 레이어)
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in horizons
        ])

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
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in self.shared_fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)

        for head in self.horizon_heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, seq_length, input_size)
            hidden: 초기 hidden state (optional)

        Returns:
            output: 예측값 (batch_size, num_horizons)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, hidden)

        # 마지막 타임스텝의 출력 사용
        if self.bidirectional:
            last_output = torch.cat([
                lstm_out[:, -1, :self.hidden_size],
                lstm_out[:, 0, self.hidden_size:]
            ], dim=1)
        else:
            last_output = lstm_out[:, -1, :]

        # Shared features
        shared_features = self.shared_fc(last_output)

        # Multi-head outputs
        outputs = []
        for head in self.horizon_heads:
            output = head(shared_features)
            outputs.append(output)

        # Concatenate all horizons
        return torch.cat(outputs, dim=1)

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualLSTM(nn.Module):
    """
    Residual Connection이 있는 LSTM 모델

    Skip connection을 통해 gradient flow 개선

    Architecture:
        Input → LSTM Block 1 → (+) → LSTM Block 2 → (+) → Output
                     ↓                     ↓
                (skip connection)     (skip connection)

    Args:
        input_size: 입력 특성 수
        hidden_size: LSTM hidden state 크기
        num_blocks: Residual block 수 (default: 2)
        dropout: Dropout 비율
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.output_size = output_size

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Residual LSTM blocks
        self.lstm_blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_blocks):
            self.lstm_blocks.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, seq_length, input_size)

        Returns:
            output: 예측값 (batch_size, output_size)
        """
        # Input projection
        out = self.input_proj(x)

        # Residual LSTM blocks
        for lstm, ln in zip(self.lstm_blocks, self.layer_norms):
            residual = out
            out, _ = lstm(out)
            out = self.dropout_layer(out)
            out = ln(out + residual)  # Residual connection

        # 마지막 타임스텝
        last_output = out[:, -1, :]

        # Output
        return self.fc(last_output)

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    horizons: List[int] = None,
    output_size: int = 1
) -> nn.Module:
    """
    모델 팩토리 함수

    Args:
        model_type: 모델 타입 ('lstm', 'bilstm', 'multi_horizon', 'residual')
        input_size: 입력 특성 수
        hidden_size: hidden state 크기
        num_layers: 레이어 수
        dropout: Dropout 비율
        bidirectional: 양방향 여부 (lstm/multi_horizon만 해당)
        horizons: 예측 시간대 리스트 (multi_horizon만 해당)
        output_size: 출력 크기 (lstm/bilstm만 해당)

    Returns:
        nn.Module: 생성된 모델

    Example:
        >>> model = create_model('lstm', input_size=38)
        >>> model = create_model('multi_horizon', input_size=38, horizons=[1, 6, 12, 24])
    """
    model_type = model_type.lower()

    if model_type == 'lstm':
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            output_size=output_size
        )

    elif model_type == 'bilstm':
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            output_size=output_size
        )

    elif model_type == 'multi_horizon':
        return MultiHorizonLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizons=horizons,
            bidirectional=bidirectional
        )

    elif model_type == 'residual':
        return ResidualLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_layers,
            dropout=dropout,
            output_size=output_size
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported: 'lstm', 'bilstm', 'multi_horizon', 'residual'")


# ============================================================
# 모델 정보 출력 함수
# ============================================================

def model_summary(model: nn.Module, input_shape: Tuple[int, int, int] = None) -> str:
    """
    모델 구조 요약 출력

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape (batch_size, seq_length, input_size)

    Returns:
        str: 모델 요약 문자열
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")

    if input_shape is not None:
        # Memory estimation (rough)
        param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        lines.append(f"Parameter memory: ~{param_memory:.2f} MB")

    lines.append("-" * 60)
    lines.append("Layer structure:")

    for name, module in model.named_modules():
        if name:
            lines.append(f"  {name}: {module.__class__.__name__}")

    lines.append("=" * 60)

    return "\n".join(lines)
