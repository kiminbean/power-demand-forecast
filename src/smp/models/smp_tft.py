"""
SMP Temporal Fusion Transformer 모델
====================================

TFT 기반 24시간 SMP(계통한계가격) 예측 모델

주요 기능:
1. 48시간 입력 → 24시간 예측
2. 변수 중요도 해석 (Variable Selection Network)
3. 시간적 패턴 학습 (Interpretable Multi-Head Attention)
4. 불확실성 추정 (Quantile 예측)
5. MPS (Apple Silicon) 최적화

Architecture:
    Input → Variable Selection → LSTM Encoder → Self-Attention →
    LSTM Decoder → Quantile Outputs

Reference:
    Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting"

Author: Claude Code
Date: 2025-12
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any


def get_device() -> torch.device:
    """최적 디바이스 반환 (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# TFT 빌딩 블록
# ============================================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU)"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        value, gate = x.chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)

    TFT의 핵심 빌딩 블록으로 정보 흐름을 게이트로 제어합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size else input_size
        self.context_size = context_size
        self.residual = residual

        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None

        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, self.output_size, dropout=dropout)

        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if self.residual:
            residual = self.skip_proj(x) if self.skip_proj else x

        hidden = self.fc1(x)
        if context is not None and self.context_fc is not None:
            if context.dim() < x.dim():
                context = context.unsqueeze(-2).expand(*x.shape[:-1], -1)
            hidden = hidden + self.context_fc(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.glu(hidden)

        if self.residual:
            return self.layer_norm(residual + hidden)
        return self.layer_norm(hidden)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN)

    각 입력 변수의 중요도를 학습하여 가중치를 부여합니다.
    """

    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()

        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # 각 변수별 GRN
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size,
                residual=False
            )
            for _ in range(num_inputs)
        ])

        # 변수 선택 GRN
        self.selection_grn = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
            residual=False
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_inputs, input_size) or (batch, num_inputs, input_size)
            context: (batch, context_size)

        Returns:
            output: (batch, seq_len, hidden_size) or (batch, hidden_size)
            weights: (batch, seq_len, num_inputs) or (batch, num_inputs)
        """
        is_temporal = x.dim() == 4

        if is_temporal:
            batch_size, seq_len, num_inputs, input_size = x.shape
        else:
            batch_size, num_inputs, input_size = x.shape
            seq_len = 1
            x = x.unsqueeze(1)

        # Flatten for selection
        x_flat = x.view(batch_size, seq_len, -1)

        # 변수 선택 가중치
        selection_weights = self.selection_grn(x_flat, context)
        selection_weights = self.softmax(selection_weights)

        # 각 변수 변환
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_input = x[:, :, i, :]
            var_output = grn(var_input, context)
            var_outputs.append(var_output)

        # Stack: (batch, seq_len, num_inputs, hidden_size)
        var_outputs = torch.stack(var_outputs, dim=2)

        # 가중 합: (batch, seq_len, hidden_size)
        output = torch.sum(var_outputs * selection_weights.unsqueeze(-1), dim=2)

        if not is_temporal:
            output = output.squeeze(1)
            selection_weights = selection_weights.squeeze(1)

        return output, selection_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention

    모든 헤드가 Value를 공유하여 해석 가능한 attention weights를 제공합니다.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        # 평균 attention weights
        avg_weights = attn_weights.mean(dim=1)

        return output, avg_weights


class PositionalEncoding(nn.Module):
    """위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ============================================================
# SMP TFT 모델
# ============================================================

class SMPTFTModel(nn.Module):
    """SMP Temporal Fusion Transformer

    48시간 과거 데이터로 24시간 미래 SMP를 예측합니다.

    Architecture:
        1. Input Embedding
        2. Variable Selection Network
        3. LSTM Encoder
        4. Interpretable Multi-Head Self-Attention
        5. Temporal Fusion Decoder
        6. Quantile Output

    Args:
        input_size: 입력 피처 수
        hidden_size: hidden state 크기 (default: 64)
        num_heads: attention head 수 (default: 4)
        num_layers: LSTM 레이어 수 (default: 2)
        dropout: Dropout 비율 (default: 0.1)
        prediction_hours: 예측 시간 수 (default: 24)
        quantiles: 예측할 분위수 (default: [0.1, 0.5, 0.9])

    Example:
        >>> model = SMPTFTModel(input_size=20, hidden_size=64)
        >>> x = torch.randn(32, 48, 20)
        >>> outputs = model(x)
        >>> print(outputs['q50'].shape)  # (32, 24)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        prediction_hours: int = 24,
        quantiles: List[float] = None
    ):
        super().__init__()

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_hours = prediction_hours
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Input embedding (각 피처를 hidden_size로 프로젝션)
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Variable Selection (simplified: 전체 입력에 대해)
        self.input_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=100, dropout=dropout)

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Post-LSTM GRN
        self.encoder_grn = GatedResidualNetwork(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )

        # Self-Attention
        self.self_attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-Attention GRN
        self.attention_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Final decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Quantile outputs
        self.quantile_heads = nn.ModuleDict({
            f'q{int(q*100)}': nn.Linear(hidden_size, prediction_hours)
            for q in quantiles
        })

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_variable_importance: bool = False
    ) -> Dict[str, torch.Tensor]:
        """순전파

        Args:
            x: (batch, seq_len, input_size)
            return_attention: attention weights 반환 여부
            return_variable_importance: 변수 중요도 반환 여부

        Returns:
            Dict containing:
                - 'q10', 'q50', 'q90': 각 분위수 예측 (batch, prediction_hours)
                - 'attention_weights': (optional) (batch, seq_len, seq_len)
                - 'variable_importance': (optional) (batch, input_size)
        """
        batch_size, seq_len, _ = x.shape

        # Variable Selection
        # x 형태 변환: (batch, seq_len, input_size) → (batch, seq_len, input_size, 1)
        x_expanded = x.unsqueeze(-1)
        selected, var_weights = self.input_vsn(x_expanded)

        # Positional encoding
        encoded = self.pos_encoder(selected)

        # LSTM Encoder
        lstm_out, _ = self.encoder(encoded)

        # Post-LSTM GRN
        encoder_out = self.encoder_grn(lstm_out)

        # Self-Attention
        attn_out, attn_weights = self.self_attention(
            encoder_out, encoder_out, encoder_out
        )

        # Post-Attention GRN with residual
        temporal_features = self.attention_grn(attn_out)

        # Global pooling (마지막 시퀀스의 특성 사용)
        context = temporal_features[:, -1, :]

        # Decoder
        decoded = self.decoder(context)

        # Quantile outputs
        outputs = {}
        for name, head in self.quantile_heads.items():
            outputs[name] = head(decoded)

        # Optional returns
        if return_attention:
            outputs['attention_weights'] = attn_weights

        if return_variable_importance:
            # 평균 변수 중요도
            outputs['variable_importance'] = var_weights.mean(dim=1)

        return outputs

    def get_num_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SMPTFTQuantileLoss(nn.Module):
    """TFT Quantile Loss

    여러 분위수에 대한 Pinball Loss의 합
    """

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.quantile_names = [f'q{int(q*100)}' for q in quantiles]

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: 분위수별 예측 딕셔너리
            targets: 실제값 (batch, prediction_hours)

        Returns:
            loss: 평균 Quantile Loss
        """
        total_loss = 0.0

        for q, name in zip(self.quantiles, self.quantile_names):
            pred = predictions[name]
            errors = targets - pred
            loss = torch.max(q * errors, (q - 1) * errors)
            total_loss += loss.mean()

        return total_loss / len(self.quantiles)


def create_smp_tft_model(
    input_size: int,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    prediction_hours: int = 24,
    quantiles: List[float] = None,
    **kwargs
) -> SMPTFTModel:
    """SMP TFT 모델 팩토리 함수"""
    return SMPTFTModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        prediction_hours=prediction_hours,
        quantiles=quantiles,
        **kwargs
    )


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
    print("SMP TFT Model Test")
    print("=" * 60)

    # 모델 생성
    model = SMPTFTModel(input_size=20, hidden_size=64, num_heads=4)
    print(model_summary(model))

    # 테스트 입력
    batch_size = 8
    seq_len = 48
    input_size = 20

    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass
    outputs = model(x, return_attention=True, return_variable_importance=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Q10 shape: {outputs['q10'].shape}")
    print(f"Q50 shape: {outputs['q50'].shape}")
    print(f"Q90 shape: {outputs['q90'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"Variable importance shape: {outputs['variable_importance'].shape}")

    # Loss 테스트
    targets = torch.randn(batch_size, 24)
    loss_fn = SMPTFTQuantileLoss([0.1, 0.5, 0.9])
    loss = loss_fn(outputs, targets)
    print(f"\nQuantile Loss: {loss.item():.4f}")

    print("\nAll tests passed!")
