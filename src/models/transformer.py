"""
MODEL-004: Temporal Fusion Transformer (TFT) 구현
=================================================

시계열 전력 수요 예측을 위한 Temporal Fusion Transformer 모델

주요 컴포넌트:
1. Gated Residual Network (GRN) - 정보 흐름 제어
2. Variable Selection Network (VSN) - 피처 중요도 학습
3. Interpretable Multi-Head Attention - 시간적 의존성 학습
4. LSTM Encoder-Decoder - 시퀀스 처리

Reference:
- Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting"
- arXiv: https://arxiv.org/abs/1912.09363

Author: Claude Code
Date: 2025-12
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Gated Linear Unit (GLU)
# ============================================================

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU)

    GLU(x) = sigmoid(x[:, :d]) * x[:, d:]

    입력을 두 부분으로 나누어 하나는 게이트로, 하나는 값으로 사용
    """

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0.0):
        super().__init__()

        if hidden_size is None:
            hidden_size = input_size

        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        # Split into value and gate
        value, gate = x.chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


# ============================================================
# Gated Residual Network (GRN)
# ============================================================

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)

    TFT의 핵심 빌딩 블록으로, 정보 흐름을 게이트로 제어합니다.

    GRN(a, c) = LayerNorm(a + GLU(η₁))
    where:
        η₁ = W₁·η₂ + b₁
        η₂ = ELU(W₂·a + W₃·c + b₂)  (c is optional context)

    Args:
        input_size: 입력 차원
        hidden_size: 은닉층 차원
        output_size: 출력 차원 (None이면 input_size와 동일)
        dropout: Dropout 비율
        context_size: 컨텍스트 벡터 차원 (옵션)
        residual: Residual connection 사용 여부

    Example:
        >>> grn = GatedResidualNetwork(input_size=64, hidden_size=64)
        >>> x = torch.randn(32, 48, 64)  # (batch, seq_len, features)
        >>> output = grn(x)
        >>> print(output.shape)  # (32, 48, 64)
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
        self.output_size = output_size if output_size is not None else input_size
        self.context_size = context_size
        self.residual = residual

        # Primary dense layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Context projection (optional)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None

        # ELU activation
        self.elu = nn.ELU()

        # Second dense layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Gated Linear Unit
        self.glu = GatedLinearUnit(hidden_size, self.output_size, dropout=dropout)

        # Skip connection projection (if dimensions differ)
        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (..., input_size)
            context: 컨텍스트 벡터 (..., context_size) - optional

        Returns:
            output: 출력 텐서 (..., output_size)
        """
        # Skip connection
        if self.residual:
            if self.skip_proj is not None:
                residual = self.skip_proj(x)
            else:
                residual = x

        # Primary transformation
        hidden = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_fc is not None:
            # Expand context to match x dimensions if needed
            if context.dim() < x.dim():
                # Add sequence dimension
                context = context.unsqueeze(-2).expand(*x.shape[:-1], -1)
            hidden = hidden + self.context_fc(context)

        # ELU activation
        hidden = self.elu(hidden)

        # Second transformation
        hidden = self.fc2(hidden)

        # Gated Linear Unit
        hidden = self.glu(hidden)

        # Add residual and normalize
        if self.residual:
            output = self.layer_norm(residual + hidden)
        else:
            output = self.layer_norm(hidden)

        return output


# ============================================================
# Variable Selection Network (VSN)
# ============================================================

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)

    각 입력 변수의 중요도를 학습하여 가중치를 부여합니다.
    Softmax를 통해 변수 선택 가중치를 계산하고, 가중 합으로 최종 표현을 생성합니다.

    Architecture:
        1. 각 변수별 GRN 적용 (개별 변환)
        2. 플래튼된 변수들에 GRN 적용 → Softmax → 변수 가중치
        3. 변환된 변수들의 가중 합 계산

    Args:
        input_size: 각 변수의 입력 차원
        num_inputs: 변수 개수
        hidden_size: 은닉층 차원
        dropout: Dropout 비율
        context_size: 컨텍스트 벡터 차원 (옵션)

    Example:
        >>> vsn = VariableSelectionNetwork(
        ...     input_size=1,      # 각 변수는 스칼라
        ...     num_inputs=10,     # 10개 변수
        ...     hidden_size=64
        ... )
        >>> x = torch.randn(32, 48, 10, 1)  # (batch, seq, num_vars, var_dim)
        >>> output, weights = vsn(x)
        >>> print(output.shape)   # (32, 48, 64)
        >>> print(weights.shape)  # (32, 48, 10)
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
        self.context_size = context_size

        # Individual variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size,
                residual=False  # No residual for dimension change
            )
            for _ in range(num_inputs)
        ])

        # Variable selection GRN
        # Input: flattened variables, Output: num_inputs weights
        self.selection_grn = GatedResidualNetwork(
            input_size=num_inputs * input_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
            residual=False
        )

        # Softmax for variable weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파

        Args:
            x: 입력 텐서
               - 3D: (batch, num_inputs, input_size) - static variables
               - 4D: (batch, seq_len, num_inputs, input_size) - temporal variables
            context: 컨텍스트 벡터 (batch, context_size) - optional

        Returns:
            output: 선택된 변수들의 가중 합 (..., hidden_size)
            weights: 변수 선택 가중치 (..., num_inputs)
        """
        is_temporal = x.dim() == 4

        if is_temporal:
            batch_size, seq_len, num_inputs, input_size = x.shape
        else:
            batch_size, num_inputs, input_size = x.shape
            seq_len = 1
            x = x.unsqueeze(1)  # Add sequence dimension

        # Flatten for selection GRN
        # Shape: (batch, seq_len, num_inputs * input_size)
        x_flat = x.view(batch_size, seq_len, -1)

        # Compute variable selection weights
        # Shape: (batch, seq_len, num_inputs)
        selection_weights = self.selection_grn(x_flat, context)
        selection_weights = self.softmax(selection_weights)

        # Transform each variable through its GRN
        # List of (batch, seq_len, hidden_size)
        transformed_vars = []
        for i, grn in enumerate(self.var_grns):
            var_i = x[:, :, i, :]  # (batch, seq_len, input_size)
            transformed = grn(var_i, context)
            transformed_vars.append(transformed)

        # Stack: (batch, seq_len, num_inputs, hidden_size)
        transformed = torch.stack(transformed_vars, dim=2)

        # Apply selection weights
        # weights: (batch, seq_len, num_inputs, 1)
        weights = selection_weights.unsqueeze(-1)

        # Weighted sum: (batch, seq_len, hidden_size)
        output = (transformed * weights).sum(dim=2)

        # Remove sequence dimension if input was not temporal
        if not is_temporal:
            output = output.squeeze(1)
            selection_weights = selection_weights.squeeze(1)

        return output, selection_weights


# ============================================================
# Static Covariate Encoder
# ============================================================

class StaticCovariateEncoder(nn.Module):
    """
    Static Covariate Encoder

    정적 변수들을 인코딩하여 4개의 컨텍스트 벡터를 생성합니다:
    1. c_s: Variable Selection 컨텍스트
    2. c_e: LSTM Encoder 초기 상태
    3. c_d: LSTM Decoder 초기 상태
    4. c_h: Static Enrichment 컨텍스트

    Args:
        input_size: 각 정적 변수의 차원
        num_static_vars: 정적 변수 개수
        hidden_size: 은닉층 및 출력 차원
        dropout: Dropout 비율
    """

    def __init__(
        self,
        input_size: int,
        num_static_vars: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_static_vars = num_static_vars
        self.hidden_size = hidden_size

        # Variable selection for static variables
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            num_inputs=num_static_vars,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Context generators (4 different contexts)
        self.context_grns = nn.ModuleDict({
            'selection': GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            ),
            'encoder': GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            ),
            'decoder': GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            ),
            'enrichment': GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            )
        })

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        순전파

        Args:
            x: 정적 변수 텐서 (batch, num_static_vars, input_size)

        Returns:
            contexts: 컨텍스트 딕셔너리
                - 'selection': (batch, hidden_size)
                - 'encoder': (batch, hidden_size)
                - 'decoder': (batch, hidden_size)
                - 'enrichment': (batch, hidden_size)
            weights: 변수 선택 가중치 (batch, num_static_vars)
        """
        # Variable selection
        static_rep, weights = self.vsn(x)

        # Generate 4 contexts
        contexts = {}
        for name, grn in self.context_grns.items():
            contexts[name] = grn(static_rep)

        return contexts, weights


# ============================================================
# Temporal Variable Selection
# ============================================================

class TemporalVariableSelection(nn.Module):
    """
    Temporal Variable Selection

    시간에 따라 변하는 변수들(known, unknown)에 대한 VSN

    Args:
        input_sizes: 각 변수 그룹의 입력 차원 딕셔너리
            - 'known': known future 변수 차원
            - 'unknown': unknown 변수 차원
        num_inputs: 각 변수 그룹의 변수 개수 딕셔너리
        hidden_size: 은닉층 차원
        dropout: Dropout 비율
        context_size: Static context 차원
    """

    def __init__(
        self,
        input_sizes: Dict[str, int],
        num_inputs: Dict[str, int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()

        self.input_sizes = input_sizes
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # VSN for each variable group
        self.vsns = nn.ModuleDict()

        for group_name in ['known', 'unknown']:
            if group_name in num_inputs and num_inputs[group_name] > 0:
                self.vsns[group_name] = VariableSelectionNetwork(
                    input_size=input_sizes.get(group_name, 1),
                    num_inputs=num_inputs[group_name],
                    hidden_size=hidden_size,
                    dropout=dropout,
                    context_size=context_size
                )

    def forward(
        self,
        known_inputs: torch.Tensor = None,
        unknown_inputs: torch.Tensor = None,
        context: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        순전파

        Args:
            known_inputs: Known future 변수 (batch, seq_len, num_known, input_size)
            unknown_inputs: Unknown 변수 (batch, seq_len, num_unknown, input_size)
            context: Static context (batch, hidden_size)

        Returns:
            outputs: 각 그룹의 선택된 표현 딕셔너리
            weights: 각 그룹의 변수 가중치 딕셔너리
        """
        outputs = {}
        weights = {}

        if known_inputs is not None and 'known' in self.vsns:
            outputs['known'], weights['known'] = self.vsns['known'](
                known_inputs, context
            )

        if unknown_inputs is not None and 'unknown' in self.vsns:
            outputs['unknown'], weights['unknown'] = self.vsns['unknown'](
                unknown_inputs, context
            )

        return outputs, weights


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Temporal Sequences

    시퀀스의 위치 정보를 인코딩합니다. 시계열에서 시간적 순서를 학습에 반영합니다.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: 임베딩 차원
        max_len: 최대 시퀀스 길이
        dropout: Dropout 비율

    Example:
        >>> pe = PositionalEncoding(d_model=64, max_len=100)
        >>> x = torch.randn(32, 48, 64)  # (batch, seq_len, d_model)
        >>> output = pe(x)
        >>> print(output.shape)  # (32, 48, 64)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, d_model)

        Returns:
            output: 위치 인코딩이 추가된 텐서 (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================
# Interpretable Multi-Head Attention
# ============================================================

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention (TFT 논문)

    일반적인 Multi-Head Attention과 달리, 모든 헤드가 Value를 공유합니다.
    이를 통해 Attention weight가 직접 해석 가능해집니다.

    Standard MHA: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    TFT MHA: 모든 헤드가 동일한 V 사용 → 가중치 평균 가능

    수식:
        InterpretableMultiHead(Q, K, V) = (1/H) Σ_h Attention_h(Q, K, V) W^O

    Args:
        d_model: 모델 차원 (입력/출력)
        num_heads: Attention head 수
        dropout: Dropout 비율

    Example:
        >>> attn = InterpretableMultiHeadAttention(d_model=64, num_heads=4)
        >>> x = torch.randn(32, 48, 64)  # (batch, seq_len, d_model)
        >>> output, weights = attn(x, x, x)
        >>> print(output.shape)   # (32, 48, 64)
        >>> print(weights.shape)  # (32, 48, 48) - average attention weights
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Query projections for each head
        self.W_q = nn.Linear(d_model, d_model)

        # Key projections for each head
        self.W_k = nn.Linear(d_model, d_model)

        # Single Value projection (shared across heads for interpretability)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale factor
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        순전파

        Args:
            query: Query 텐서 (batch, seq_len_q, d_model)
            key: Key 텐서 (batch, seq_len_k, d_model)
            value: Value 텐서 (batch, seq_len_v, d_model)
            mask: Attention mask (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
                  True인 위치는 마스킹됨 (-inf로 대체)
            return_attention: Attention weight 반환 여부

        Returns:
            output: Attention 결과 (batch, seq_len_q, d_model)
            attention_weights: 평균 Attention 가중치 (batch, seq_len_q, seq_len_k)
                              return_attention=False면 None
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Linear projections
        # Q, K: (batch, seq_len, num_heads, d_k)
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k)

        # V: (batch, seq_len, d_model) - shared across heads
        V = self.W_v(value)

        # Transpose for attention: (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (seq_len_q, seq_len_k) -> (1, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(mask, float('-inf'))

        # Attention weights: (batch, num_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Average attention across heads: (batch, seq_len_q, seq_len_k)
        avg_attention = attention_weights.mean(dim=1)

        # Apply attention to shared values
        # avg_attention: (batch, seq_len_q, seq_len_k)
        # V: (batch, seq_len_v, d_model)
        # output: (batch, seq_len_q, d_model)
        output = torch.matmul(avg_attention, V)

        # Final projection
        output = self.W_o(output)

        if return_attention:
            return output, avg_attention
        return output, None


# ============================================================
# Temporal Self-Attention Layer
# ============================================================

class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention Layer

    시계열 데이터의 장기 의존성을 학습하는 Self-Attention 레이어입니다.
    GRN으로 감싸진 Attention을 사용하여 정보 흐름을 제어합니다.

    Architecture:
        1. Positional Encoding (선택적)
        2. Interpretable Multi-Head Attention
        3. GRN (Gate + Residual + LayerNorm)

    Args:
        d_model: 모델 차원
        num_heads: Attention head 수
        dropout: Dropout 비율
        use_positional_encoding: 위치 인코딩 사용 여부

    Example:
        >>> layer = TemporalSelfAttention(d_model=64, num_heads=4)
        >>> x = torch.randn(32, 48, 64)
        >>> output, attention = layer(x)
        >>> print(output.shape)     # (32, 48, 64)
        >>> print(attention.shape)  # (32, 48, 48)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding

        # Positional Encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                dropout=dropout
            )

        # Pre-attention GRN
        self.pre_attention_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            dropout=dropout
        )

        # Interpretable Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-attention GRN (Gated skip connection)
        self.post_attention_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        순전파

        Args:
            x: 입력 텐서 (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)
            return_attention: Attention weight 반환 여부

        Returns:
            output: Self-Attention 결과 (batch, seq_len, d_model)
            attention_weights: Attention 가중치 (batch, seq_len, seq_len)
        """
        # Add positional encoding
        if self.use_positional_encoding:
            x_pe = self.positional_encoding(x)
        else:
            x_pe = x

        # Pre-attention GRN
        x_grn = self.pre_attention_grn(x_pe)

        # Self-Attention
        attn_output, attention_weights = self.attention(
            query=x_grn,
            key=x_grn,
            value=x_grn,
            mask=mask,
            return_attention=return_attention
        )

        # Post-attention GRN with residual from input
        # output = LayerNorm(x + GLU(attn_output))
        output = self.post_attention_grn(attn_output)

        # Add residual connection from original input
        output = output + x

        return output, attention_weights


# ============================================================
# Causal Mask Generator
# ============================================================

def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    인과적 마스크 생성 (미래 정보 차단)

    시퀀스의 각 위치에서 미래 위치에 접근하지 못하도록 마스킹합니다.
    디코더나 자기회귀 모델에서 사용됩니다.

    Args:
        seq_len: 시퀀스 길이
        device: 텐서를 생성할 디바이스

    Returns:
        mask: 인과적 마스크 (seq_len, seq_len)
              True인 위치는 마스킹됨 (Attention에서 -inf로 대체)

    Example:
        >>> mask = generate_causal_mask(4)
        >>> print(mask)
        tensor([[False,  True,  True,  True],
                [False, False,  True,  True],
                [False, False, False,  True],
                [False, False, False, False]])
    """
    # Upper triangular matrix (excluding diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    if device is not None:
        mask = mask.to(device)

    return mask


def generate_encoder_decoder_mask(
    encoder_len: int,
    decoder_len: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Encoder-Decoder용 Attention 마스크 생성

    Encoder에서는 과거 시점만 보고 (causal mask),
    Decoder에서는 전체 Encoder 출력과 과거 Decoder 위치만 봅니다.

    Args:
        encoder_len: Encoder 시퀀스 길이
        decoder_len: Decoder 시퀀스 길이
        device: 텐서를 생성할 디바이스

    Returns:
        mask: 통합 마스크 (encoder_len + decoder_len, encoder_len + decoder_len)

    Example:
        >>> mask = generate_encoder_decoder_mask(3, 2)
        >>> # Encoder (3) can see all encoder positions
        >>> # Decoder (2) can see all encoder + causal decoder positions
    """
    total_len = encoder_len + decoder_len

    # Start with all visible
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)

    # Encoder: causal mask (optional, can see all for bidirectional)
    # For TFT, encoder typically sees all past

    # Decoder: causal mask for decoder positions
    decoder_mask = torch.triu(
        torch.ones(decoder_len, decoder_len),
        diagonal=1
    ).bool()

    mask[encoder_len:, encoder_len:] = decoder_mask

    if device is not None:
        mask = mask.to(device)

    return mask


# ============================================================
# Static Enrichment Layer
# ============================================================

class StaticEnrichmentLayer(nn.Module):
    """
    Static Enrichment Layer

    정적 컨텍스트 정보를 시간적 표현에 통합합니다.
    GRN을 사용하여 정적 정보가 시간적 특성에 미치는 영향을 학습합니다.

    Args:
        d_model: 모델 차원
        dropout: Dropout 비율

    Example:
        >>> layer = StaticEnrichmentLayer(d_model=64)
        >>> temporal = torch.randn(32, 48, 64)
        >>> static_context = torch.randn(32, 64)
        >>> output = layer(temporal, static_context)
        >>> print(output.shape)  # (32, 48, 64)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.enrichment_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            dropout=dropout,
            context_size=d_model  # Static context size
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        static_context: torch.Tensor
    ) -> torch.Tensor:
        """
        순전파

        Args:
            temporal_features: 시간적 특성 (batch, seq_len, d_model)
            static_context: 정적 컨텍스트 (batch, d_model)

        Returns:
            enriched: 정적 정보가 통합된 특성 (batch, seq_len, d_model)
        """
        return self.enrichment_grn(temporal_features, static_context)


# ============================================================
# LSTM Encoder
# ============================================================

class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for Past Sequence Processing

    과거 시퀀스(encoder_length)를 처리하여 hidden state를 생성합니다.
    Static context를 사용하여 초기 hidden state를 설정합니다.

    Args:
        input_size: 입력 특성 차원
        hidden_size: LSTM hidden 차원
        num_layers: LSTM 레이어 수
        dropout: Dropout 비율

    Example:
        >>> encoder = LSTMEncoder(input_size=64, hidden_size=64, num_layers=2)
        >>> x = torch.randn(32, 48, 64)  # (batch, encoder_len, features)
        >>> init_state = (torch.randn(2, 32, 64), torch.randn(2, 32, 64))
        >>> output, (h_n, c_n) = encoder(x, init_state)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        init_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        순전파

        Args:
            x: 입력 시퀀스 (batch, seq_len, input_size)
            init_state: 초기 상태 (h_0, c_0), 각각 (num_layers, batch, hidden_size)

        Returns:
            output: 모든 시점의 출력 (batch, seq_len, hidden_size)
            (h_n, c_n): 마지막 hidden state
        """
        output, (h_n, c_n) = self.lstm(x, init_state)
        return output, (h_n, c_n)


# ============================================================
# LSTM Decoder
# ============================================================

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for Future Sequence Processing

    미래 시퀀스(decoder_length)를 처리합니다.
    Encoder의 마지막 hidden state로 초기화됩니다.

    Args:
        input_size: 입력 특성 차원 (known future features)
        hidden_size: LSTM hidden 차원
        num_layers: LSTM 레이어 수
        dropout: Dropout 비율

    Example:
        >>> decoder = LSTMDecoder(input_size=64, hidden_size=64, num_layers=2)
        >>> x = torch.randn(32, 24, 64)  # (batch, decoder_len, features)
        >>> encoder_state = (torch.randn(2, 32, 64), torch.randn(2, 32, 64))
        >>> output, (h_n, c_n) = decoder(x, encoder_state)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        순전파

        Args:
            x: Known future 입력 시퀀스 (batch, decoder_len, input_size)
            encoder_state: Encoder의 마지막 상태 (h_n, c_n)

        Returns:
            output: 모든 시점의 출력 (batch, decoder_len, hidden_size)
            (h_n, c_n): 마지막 hidden state
        """
        output, (h_n, c_n) = self.lstm(x, encoder_state)
        return output, (h_n, c_n)


# ============================================================
# Temporal Fusion Transformer (Full Model)
# ============================================================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT)

    Google의 TFT 논문을 기반으로 한 시계열 예측 모델입니다.
    다중 시간대 예측과 해석 가능성을 제공합니다.

    Architecture:
        1. Input Embeddings (Static, Known, Unknown)
        2. Variable Selection Networks
        3. LSTM Encoder-Decoder
        4. Static Enrichment
        5. Temporal Self-Attention
        6. Position-wise Feed-Forward
        7. Quantile Outputs

    Args:
        num_static_vars: 정적 변수 개수 (0이면 없음)
        num_known_vars: Known future 변수 개수
        num_unknown_vars: Unknown 변수 개수 (과거에만 사용)
        hidden_size: 모델 hidden 차원
        lstm_layers: LSTM 레이어 수
        num_attention_heads: Attention head 수
        dropout: Dropout 비율
        encoder_length: Encoder 시퀀스 길이 (과거)
        decoder_length: Decoder 시퀀스 길이 (미래 예측)
        quantiles: 예측할 분위수 리스트

    Example:
        >>> model = TemporalFusionTransformer(
        ...     num_static_vars=0,
        ...     num_known_vars=8,
        ...     num_unknown_vars=25,
        ...     hidden_size=64,
        ...     encoder_length=48,
        ...     decoder_length=24
        ... )
        >>> # known: (batch, encoder_len + decoder_len, num_known, 1)
        >>> # unknown: (batch, encoder_len, num_unknown, 1)
        >>> known = torch.randn(32, 72, 8, 1)
        >>> unknown = torch.randn(32, 48, 25, 1)
        >>> output = model(known_inputs=known, unknown_inputs=unknown)
        >>> print(output['predictions'].shape)  # (32, 24, 3) for 3 quantiles
    """

    def __init__(
        self,
        num_static_vars: int = 0,
        num_known_vars: int = 8,
        num_unknown_vars: int = 25,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        encoder_length: int = 48,
        decoder_length: int = 24,
        quantiles: List[float] = None
    ):
        super().__init__()

        self.num_static_vars = num_static_vars
        self.num_known_vars = num_known_vars
        self.num_unknown_vars = num_unknown_vars
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)

        # ============================================================
        # 1. Input Embeddings (Linear projections to hidden_size)
        # ============================================================

        # Static variable embeddings (if any)
        if num_static_vars > 0:
            self.static_encoder = StaticCovariateEncoder(
                input_size=1,
                num_static_vars=num_static_vars,
                hidden_size=hidden_size,
                dropout=dropout
            )
        else:
            self.static_encoder = None

        # ============================================================
        # 2. Variable Selection Networks
        # ============================================================

        # Temporal variable selection
        self.temporal_vsn = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': num_known_vars, 'unknown': num_unknown_vars},
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size if num_static_vars > 0 else None
        )

        # ============================================================
        # 3. LSTM Encoder-Decoder
        # ============================================================

        # Encoder (processes past: known + unknown combined)
        self.lstm_encoder = LSTMEncoder(
            input_size=hidden_size,  # After VSN, features are hidden_size
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # Decoder (processes future: known only)
        self.lstm_decoder = LSTMDecoder(
            input_size=hidden_size,  # After VSN
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # GRN for LSTM output processing
        self.post_lstm_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # ============================================================
        # 4. Static Enrichment
        # ============================================================

        self.static_enrichment = StaticEnrichmentLayer(
            d_model=hidden_size,
            dropout=dropout
        )

        # ============================================================
        # 5. Temporal Self-Attention
        # ============================================================

        self.temporal_attention = TemporalSelfAttention(
            d_model=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            use_positional_encoding=True
        )

        # ============================================================
        # 6. Position-wise Feed-Forward (GRN)
        # ============================================================

        self.position_wise_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # ============================================================
        # 7. Output Layer (Quantile predictions)
        # ============================================================

        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _get_static_context(
        self,
        static_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """정적 컨텍스트 생성"""
        if self.static_encoder is not None and static_inputs is not None:
            contexts, static_weights = self.static_encoder(static_inputs)
            return contexts, static_weights
        else:
            return None, None

    def _get_lstm_init_state(
        self,
        batch_size: int,
        device: torch.device,
        static_context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM 초기 상태 생성"""
        if static_context is not None:
            # Use static context for initialization
            # Expand to (num_layers, batch, hidden)
            h_0 = static_context.unsqueeze(0).expand(
                self.lstm_layers, -1, -1
            ).contiguous()
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)
        else:
            # Zero initialization
            h_0 = torch.zeros(
                self.lstm_layers, batch_size, self.hidden_size,
                device=device
            )
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)

    def forward(
        self,
        known_inputs: torch.Tensor,
        unknown_inputs: torch.Tensor,
        static_inputs: torch.Tensor = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        순전파

        Args:
            known_inputs: Known future 변수
                Shape: (batch, encoder_length + decoder_length, num_known_vars, 1)
            unknown_inputs: Unknown 변수 (과거만)
                Shape: (batch, encoder_length, num_unknown_vars, 1)
            static_inputs: 정적 변수 (옵션)
                Shape: (batch, num_static_vars, 1)
            return_attention: Attention 가중치 반환 여부

        Returns:
            dict containing:
                - 'predictions': 예측값 (batch, decoder_length, num_quantiles)
                - 'attention_weights': Attention 가중치 (optional)
                - 'variable_weights': 변수 선택 가중치 (optional)
        """
        batch_size = known_inputs.size(0)
        device = known_inputs.device

        # ============================================================
        # 1. Static Context
        # ============================================================

        static_contexts, static_weights = self._get_static_context(static_inputs)
        selection_context = static_contexts['selection'] if static_contexts else None
        encoder_context = static_contexts['encoder'] if static_contexts else None
        enrichment_context = static_contexts['enrichment'] if static_contexts else None

        # ============================================================
        # 2. Variable Selection
        # ============================================================

        # Split known inputs into encoder and decoder parts
        known_encoder = known_inputs[:, :self.encoder_length, :, :]
        known_decoder = known_inputs[:, self.encoder_length:, :, :]

        # Encoder period: known + unknown
        encoder_vsn_outputs, encoder_vsn_weights = self.temporal_vsn(
            known_inputs=known_encoder,
            unknown_inputs=unknown_inputs,
            context=selection_context
        )

        # Combine known and unknown for encoder
        # Both are (batch, encoder_length, hidden_size)
        encoder_features = encoder_vsn_outputs['known'] + encoder_vsn_outputs['unknown']

        # Decoder period: known only
        decoder_vsn_outputs, decoder_vsn_weights = self.temporal_vsn(
            known_inputs=known_decoder,
            unknown_inputs=None,
            context=selection_context
        )
        decoder_features = decoder_vsn_outputs['known']

        # ============================================================
        # 3. LSTM Encoder-Decoder
        # ============================================================

        # Initialize LSTM state
        init_state = self._get_lstm_init_state(
            batch_size, device, encoder_context
        )

        # Encode past sequence
        encoder_output, encoder_state = self.lstm_encoder(
            encoder_features, init_state
        )

        # Decode future sequence
        decoder_output, _ = self.lstm_decoder(
            decoder_features, encoder_state
        )

        # Concatenate encoder and decoder outputs
        # Shape: (batch, encoder_length + decoder_length, hidden_size)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)

        # Post-LSTM GRN
        lstm_output = self.post_lstm_grn(lstm_output)

        # ============================================================
        # 4. Static Enrichment
        # ============================================================

        if enrichment_context is not None:
            enriched = self.static_enrichment(lstm_output, enrichment_context)
        else:
            # Use zero context if no static variables
            zero_context = torch.zeros(batch_size, self.hidden_size, device=device)
            enriched = self.static_enrichment(lstm_output, zero_context)

        # ============================================================
        # 5. Temporal Self-Attention
        # ============================================================

        # Generate mask for encoder-decoder
        total_length = self.encoder_length + self.decoder_length
        mask = generate_encoder_decoder_mask(
            self.encoder_length,
            self.decoder_length,
            device=device
        )

        attention_output, attention_weights = self.temporal_attention(
            enriched, mask=mask, return_attention=True
        )

        # ============================================================
        # 6. Position-wise Feed-Forward
        # ============================================================

        output = self.position_wise_grn(attention_output)

        # Add residual from attention
        output = output + attention_output

        # ============================================================
        # 7. Output Layer (Decoder part only)
        # ============================================================

        # Take only decoder outputs
        decoder_output_final = output[:, self.encoder_length:, :]

        # Project to quantiles
        predictions = self.output_layer(decoder_output_final)

        # ============================================================
        # Build output dictionary
        # ============================================================

        result = {
            'predictions': predictions,  # (batch, decoder_length, num_quantiles)
        }

        if return_attention:
            result['attention_weights'] = attention_weights
            result['encoder_variable_weights'] = encoder_vsn_weights
            result['decoder_variable_weights'] = decoder_vsn_weights
            if static_weights is not None:
                result['static_variable_weights'] = static_weights

        return result

    def get_attention_weights(
        self,
        known_inputs: torch.Tensor,
        unknown_inputs: torch.Tensor,
        static_inputs: torch.Tensor = None
    ) -> torch.Tensor:
        """Attention 가중치만 추출"""
        result = self.forward(
            known_inputs, unknown_inputs, static_inputs,
            return_attention=True
        )
        return result['attention_weights']


# ============================================================
# Quantile Loss
# ============================================================

class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss)

    분위수 회귀를 위한 손실 함수입니다.

    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)

    Args:
        quantiles: 예측할 분위수 리스트

    Example:
        >>> loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> predictions = torch.randn(32, 24, 3)  # 3 quantiles
        >>> targets = torch.randn(32, 24)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.register_buffer(
            'quantile_tensor',
            torch.tensor(self.quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        순전파

        Args:
            predictions: 예측값 (batch, seq_len, num_quantiles)
            targets: 실제값 (batch, seq_len) or (batch, seq_len, 1)

        Returns:
            loss: 평균 Quantile Loss
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # (batch, seq_len, 1)

        # Expand targets to match predictions
        # (batch, seq_len, num_quantiles)
        targets = targets.expand_as(predictions)

        # Compute errors
        errors = targets - predictions

        # Quantile loss for each quantile
        # quantile_tensor shape: (num_quantiles,)
        quantiles = self.quantile_tensor.view(1, 1, -1)

        losses = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors
        )

        return losses.mean()


# ============================================================
# 테스트 및 검증 함수
# ============================================================

def test_grn():
    """GRN 테스트"""
    print("Testing GRN...")

    batch_size, seq_len, input_size = 32, 48, 64
    hidden_size = 64

    # Without context
    grn = GatedResidualNetwork(
        input_size=input_size,
        hidden_size=hidden_size
    )
    x = torch.randn(batch_size, seq_len, input_size)
    output = grn(x)
    assert output.shape == (batch_size, seq_len, input_size), f"Expected {(batch_size, seq_len, input_size)}, got {output.shape}"
    print(f"  Without context: input {x.shape} -> output {output.shape}")

    # With context
    context_size = 32
    grn_ctx = GatedResidualNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        context_size=context_size
    )
    context = torch.randn(batch_size, context_size)
    output_ctx = grn_ctx(x, context)
    assert output_ctx.shape == (batch_size, seq_len, input_size)
    print(f"  With context: input {x.shape}, context {context.shape} -> output {output_ctx.shape}")

    # Different output size
    output_size = 128
    grn_diff = GatedResidualNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    output_diff = grn_diff(x)
    assert output_diff.shape == (batch_size, seq_len, output_size)
    print(f"  Different output: input {x.shape} -> output {output_diff.shape}")

    print("  GRN tests passed!")
    return True


def test_vsn():
    """VSN 테스트"""
    print("Testing VSN...")

    batch_size, seq_len = 32, 48
    num_inputs, input_size = 10, 1
    hidden_size = 64

    vsn = VariableSelectionNetwork(
        input_size=input_size,
        num_inputs=num_inputs,
        hidden_size=hidden_size
    )

    # Temporal input (4D)
    x_temporal = torch.randn(batch_size, seq_len, num_inputs, input_size)
    output, weights = vsn(x_temporal)

    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len, num_inputs), f"Expected {(batch_size, seq_len, num_inputs)}, got {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len)), "Weights should sum to 1"
    print(f"  Temporal: input {x_temporal.shape} -> output {output.shape}, weights {weights.shape}")

    # Static input (3D)
    x_static = torch.randn(batch_size, num_inputs, input_size)
    output_static, weights_static = vsn(x_static)

    assert output_static.shape == (batch_size, hidden_size)
    assert weights_static.shape == (batch_size, num_inputs)
    print(f"  Static: input {x_static.shape} -> output {output_static.shape}, weights {weights_static.shape}")

    # With context
    context_size = 32
    vsn_ctx = VariableSelectionNetwork(
        input_size=input_size,
        num_inputs=num_inputs,
        hidden_size=hidden_size,
        context_size=context_size
    )
    context = torch.randn(batch_size, context_size)
    output_ctx, _ = vsn_ctx(x_temporal, context)
    assert output_ctx.shape == (batch_size, seq_len, hidden_size)
    print(f"  With context: context {context.shape} -> output {output_ctx.shape}")

    print("  VSN tests passed!")
    return True


def test_static_encoder():
    """Static Covariate Encoder 테스트"""
    print("Testing Static Covariate Encoder...")

    batch_size = 32
    num_static_vars, input_size = 5, 8
    hidden_size = 64

    encoder = StaticCovariateEncoder(
        input_size=input_size,
        num_static_vars=num_static_vars,
        hidden_size=hidden_size
    )

    x = torch.randn(batch_size, num_static_vars, input_size)
    contexts, weights = encoder(x)

    assert len(contexts) == 4, f"Expected 4 contexts, got {len(contexts)}"
    for name, ctx in contexts.items():
        assert ctx.shape == (batch_size, hidden_size), f"Context '{name}' shape mismatch"
    assert weights.shape == (batch_size, num_static_vars)

    print(f"  Input: {x.shape}")
    print(f"  Contexts: {list(contexts.keys())}, shape: {contexts['selection'].shape}")
    print(f"  Weights: {weights.shape}")
    print("  Static Covariate Encoder tests passed!")
    return True


def test_positional_encoding():
    """Positional Encoding 테스트"""
    print("Testing Positional Encoding...")

    batch_size, seq_len, d_model = 32, 48, 64

    pe = PositionalEncoding(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pe(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"  Input: {x.shape} -> Output: {output.shape}")

    # Check that positional encoding is added (output != input)
    assert not torch.allclose(x, output), "Output should differ from input"
    print("  Positional encoding is properly added")

    print("  Positional Encoding tests passed!")
    return True


def test_interpretable_attention():
    """Interpretable Multi-Head Attention 테스트"""
    print("Testing Interpretable Multi-Head Attention...")

    batch_size, seq_len, d_model = 32, 48, 64
    num_heads = 4

    attn = InterpretableMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-attention
    output, weights = attn(x, x, x)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), \
        f"Expected {(batch_size, seq_len, seq_len)}, got {weights.shape}"
    print(f"  Self-attention: input {x.shape} -> output {output.shape}, weights {weights.shape}")

    # Check attention weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Attention weights should sum to 1"
    print("  Attention weights sum to 1: OK")

    # With mask
    mask = generate_causal_mask(seq_len)
    output_masked, weights_masked = attn(x, x, x, mask=mask)
    assert output_masked.shape == output.shape
    print(f"  With causal mask: output {output_masked.shape}")

    # Check that future positions are masked (weights should be 0)
    # For position 0, only position 0 should have non-zero weight
    first_pos_weights = weights_masked[:, 0, :]  # (batch, seq_len)
    assert first_pos_weights[:, 1:].sum() < 1e-5, "Future positions should have ~0 weight"
    print("  Causal masking verified: future weights are ~0")

    print("  Interpretable Multi-Head Attention tests passed!")
    return True


def test_temporal_self_attention():
    """Temporal Self-Attention Layer 테스트"""
    print("Testing Temporal Self-Attention...")

    batch_size, seq_len, d_model = 32, 48, 64
    num_heads = 4

    layer = TemporalSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output, attention = layer(x)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attention.shape == (batch_size, seq_len, seq_len), \
        f"Expected {(batch_size, seq_len, seq_len)}, got {attention.shape}"
    print(f"  Output: {output.shape}, Attention: {attention.shape}")

    # With mask
    mask = generate_causal_mask(seq_len)
    output_masked, _ = layer(x, mask=mask)
    assert output_masked.shape == output.shape
    print(f"  With causal mask: output {output_masked.shape}")

    # Without positional encoding
    layer_no_pe = TemporalSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_positional_encoding=False
    )
    output_no_pe, _ = layer_no_pe(x)
    assert output_no_pe.shape == output.shape
    print(f"  Without positional encoding: output {output_no_pe.shape}")

    print("  Temporal Self-Attention tests passed!")
    return True


def test_causal_mask():
    """Causal Mask 테스트"""
    print("Testing Causal Mask...")

    seq_len = 5
    mask = generate_causal_mask(seq_len)

    assert mask.shape == (seq_len, seq_len), f"Expected {(seq_len, seq_len)}, got {mask.shape}"
    print(f"  Mask shape: {mask.shape}")

    # Check structure
    expected = torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False],
    ])
    assert torch.equal(mask, expected), "Causal mask structure incorrect"
    print("  Mask structure verified:")
    print(f"    {mask[0].tolist()}")
    print(f"    {mask[1].tolist()}")
    print("    ...")

    print("  Causal Mask tests passed!")
    return True


def test_static_enrichment():
    """Static Enrichment Layer 테스트"""
    print("Testing Static Enrichment Layer...")

    batch_size, seq_len, d_model = 32, 48, 64

    layer = StaticEnrichmentLayer(d_model=d_model)
    temporal = torch.randn(batch_size, seq_len, d_model)
    static_context = torch.randn(batch_size, d_model)

    output = layer(temporal, static_context)

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
    print(f"  Temporal: {temporal.shape}, Static: {static_context.shape} -> Output: {output.shape}")

    print("  Static Enrichment Layer tests passed!")
    return True


def test_lstm_encoder():
    """LSTM Encoder 테스트"""
    print("Testing LSTM Encoder...")

    batch_size, seq_len, input_size = 32, 48, 64
    hidden_size = 64
    num_layers = 2

    encoder = LSTMEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    x = torch.randn(batch_size, seq_len, input_size)
    output, (h_n, c_n) = encoder(x)

    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)
    print(f"  Input: {x.shape} -> Output: {output.shape}, h_n: {h_n.shape}")

    # With initial state
    h_0 = torch.randn(num_layers, batch_size, hidden_size)
    c_0 = torch.randn(num_layers, batch_size, hidden_size)
    output2, _ = encoder(x, (h_0, c_0))
    assert output2.shape == output.shape
    print(f"  With init state: Output: {output2.shape}")

    print("  LSTM Encoder tests passed!")
    return True


def test_lstm_decoder():
    """LSTM Decoder 테스트"""
    print("Testing LSTM Decoder...")

    batch_size, seq_len, input_size = 32, 24, 64
    hidden_size = 64
    num_layers = 2

    decoder = LSTMDecoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    x = torch.randn(batch_size, seq_len, input_size)
    encoder_state = (
        torch.randn(num_layers, batch_size, hidden_size),
        torch.randn(num_layers, batch_size, hidden_size)
    )

    output, (h_n, c_n) = decoder(x, encoder_state)

    assert output.shape == (batch_size, seq_len, hidden_size), \
        f"Expected {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    print(f"  Input: {x.shape} -> Output: {output.shape}")

    print("  LSTM Decoder tests passed!")
    return True


def test_temporal_fusion_transformer():
    """Temporal Fusion Transformer 전체 모델 테스트"""
    print("Testing Temporal Fusion Transformer...")

    batch_size = 32
    encoder_length = 48
    decoder_length = 24
    num_known_vars = 8
    num_unknown_vars = 25
    hidden_size = 64
    num_quantiles = 3

    model = TemporalFusionTransformer(
        num_static_vars=0,
        num_known_vars=num_known_vars,
        num_unknown_vars=num_unknown_vars,
        hidden_size=hidden_size,
        encoder_length=encoder_length,
        decoder_length=decoder_length
    )

    # Create inputs
    known_inputs = torch.randn(
        batch_size,
        encoder_length + decoder_length,
        num_known_vars,
        1
    )
    unknown_inputs = torch.randn(
        batch_size,
        encoder_length,
        num_unknown_vars,
        1
    )

    # Forward pass
    result = model(known_inputs=known_inputs, unknown_inputs=unknown_inputs)

    assert 'predictions' in result
    assert result['predictions'].shape == (batch_size, decoder_length, num_quantiles), \
        f"Expected {(batch_size, decoder_length, num_quantiles)}, got {result['predictions'].shape}"
    print(f"  Predictions shape: {result['predictions'].shape}")

    # With attention weights
    result_attn = model(
        known_inputs=known_inputs,
        unknown_inputs=unknown_inputs,
        return_attention=True
    )
    assert 'attention_weights' in result_attn
    total_len = encoder_length + decoder_length
    assert result_attn['attention_weights'].shape == (batch_size, total_len, total_len)
    print(f"  Attention weights shape: {result_attn['attention_weights'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("  Temporal Fusion Transformer tests passed!")
    return True


def test_tft_with_static():
    """정적 변수가 있는 TFT 테스트"""
    print("Testing TFT with static variables...")

    batch_size = 32
    encoder_length = 48
    decoder_length = 24
    num_static_vars = 5
    num_known_vars = 8
    num_unknown_vars = 25

    model = TemporalFusionTransformer(
        num_static_vars=num_static_vars,
        num_known_vars=num_known_vars,
        num_unknown_vars=num_unknown_vars,
        hidden_size=64,
        encoder_length=encoder_length,
        decoder_length=decoder_length
    )

    known_inputs = torch.randn(batch_size, encoder_length + decoder_length, num_known_vars, 1)
    unknown_inputs = torch.randn(batch_size, encoder_length, num_unknown_vars, 1)
    static_inputs = torch.randn(batch_size, num_static_vars, 1)

    result = model(
        known_inputs=known_inputs,
        unknown_inputs=unknown_inputs,
        static_inputs=static_inputs,
        return_attention=True
    )

    assert result['predictions'].shape == (batch_size, decoder_length, 3)
    assert 'static_variable_weights' in result
    assert result['static_variable_weights'].shape == (batch_size, num_static_vars)
    print(f"  With static vars: predictions {result['predictions'].shape}")
    print(f"  Static weights: {result['static_variable_weights'].shape}")

    print("  TFT with static variables tests passed!")
    return True


def test_quantile_loss():
    """Quantile Loss 테스트"""
    print("Testing Quantile Loss...")

    batch_size, seq_len = 32, 24
    num_quantiles = 3
    quantiles = [0.1, 0.5, 0.9]

    loss_fn = QuantileLoss(quantiles=quantiles)

    predictions = torch.randn(batch_size, seq_len, num_quantiles)
    targets = torch.randn(batch_size, seq_len)

    loss = loss_fn(predictions, targets)

    assert loss.dim() == 0, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"
    print(f"  Loss value: {loss.item():.4f}")

    # Test with 3D targets
    targets_3d = targets.unsqueeze(-1)
    loss_3d = loss_fn(predictions, targets_3d)
    assert loss_3d.dim() == 0
    print(f"  Loss with 3D targets: {loss_3d.item():.4f}")

    # Test gradient flow
    predictions_grad = predictions.clone().requires_grad_(True)
    loss_grad = loss_fn(predictions_grad, targets)
    loss_grad.backward()
    assert predictions_grad.grad is not None
    print("  Gradient flow: OK")

    print("  Quantile Loss tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TFT Components Test Suite")
    print("=" * 60)

    # Subtask 1.2 tests
    test_grn()
    print()
    test_vsn()
    print()
    test_static_encoder()
    print()

    # Subtask 1.3 tests
    test_positional_encoding()
    print()
    test_interpretable_attention()
    print()
    test_temporal_self_attention()
    print()
    test_causal_mask()
    print()
    test_static_enrichment()
    print()

    # Subtask 1.4 tests
    test_lstm_encoder()
    print()
    test_lstm_decoder()
    print()
    test_temporal_fusion_transformer()
    print()
    test_tft_with_static()
    print()
    test_quantile_loss()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
