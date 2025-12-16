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
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
