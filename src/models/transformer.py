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


if __name__ == "__main__":
    print("=" * 60)
    print("TFT Components Test Suite")
    print("=" * 60)

    test_grn()
    print()
    test_vsn()
    print()
    test_static_encoder()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
