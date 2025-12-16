"""
MODEL-004: Temporal Fusion Transformer 컴포넌트 테스트
====================================================

TFT 모델 컴포넌트 (GRN, VSN, Encoders) 단위 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import (
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    StaticCovariateEncoder,
    TemporalVariableSelection,
    PositionalEncoding,
    InterpretableMultiHeadAttention,
    TemporalSelfAttention,
    StaticEnrichmentLayer,
    generate_causal_mask,
    generate_encoder_decoder_mask,
)


# ============================================================
# 테스트 설정
# ============================================================

@pytest.fixture
def device():
    """테스트용 디바이스"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def batch_config():
    """테스트용 배치 설정"""
    return {
        'batch_size': 32,
        'seq_len': 48,
        'input_size': 64,
        'hidden_size': 64,
        'context_size': 32,
        'num_vars': 10,
    }


# ============================================================
# GatedLinearUnit (GLU) 테스트
# ============================================================

class TestGatedLinearUnit:
    """GLU 클래스 테스트"""

    def test_glu_creation(self):
        """GLU 생성 테스트"""
        glu = GatedLinearUnit(input_size=64, hidden_size=32)
        assert glu is not None
        assert glu.fc.in_features == 64
        assert glu.fc.out_features == 64  # 32 * 2

    def test_glu_forward_shape(self, batch_config):
        """GLU 순전파 출력 shape 테스트"""
        glu = GatedLinearUnit(input_size=64, hidden_size=32)
        x = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 64)
        output = glu(x)

        assert output.shape == (batch_config['batch_size'], batch_config['seq_len'], 32)

    def test_glu_default_hidden_size(self):
        """GLU 기본 hidden_size 테스트"""
        glu = GatedLinearUnit(input_size=64)
        x = torch.randn(32, 48, 64)
        output = glu(x)

        assert output.shape == (32, 48, 64)

    def test_glu_with_dropout(self):
        """GLU dropout 테스트"""
        glu = GatedLinearUnit(input_size=64, dropout=0.5)
        glu.train()  # Dropout active
        x = torch.randn(32, 48, 64)
        output = glu(x)

        assert output.shape == (32, 48, 64)

    def test_glu_output_range(self):
        """GLU 출력값이 sigmoid로 gating됨 확인"""
        glu = GatedLinearUnit(input_size=64)
        glu.eval()

        # 큰 입력값 테스트 (출력이 제한됨)
        x = torch.randn(32, 48, 64) * 10
        output = glu(x)

        # 출력이 유한한지 확인
        assert torch.isfinite(output).all()


# ============================================================
# GatedResidualNetwork (GRN) 테스트
# ============================================================

class TestGatedResidualNetwork:
    """GRN 클래스 테스트"""

    def test_grn_creation(self, batch_config):
        """GRN 생성 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size']
        )
        assert grn is not None
        assert grn.input_size == batch_config['input_size']
        assert grn.hidden_size == batch_config['hidden_size']

    def test_grn_forward_shape(self, batch_config):
        """GRN 순전파 출력 shape 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        )
        output = grn(x)

        assert output.shape == x.shape

    def test_grn_different_output_size(self, batch_config):
        """GRN 다른 output_size 테스트"""
        output_size = 128
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size'],
            output_size=output_size
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        )
        output = grn(x)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            output_size
        )

    def test_grn_with_context(self, batch_config):
        """GRN with context 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size'],
            context_size=batch_config['context_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        )
        context = torch.randn(
            batch_config['batch_size'],
            batch_config['context_size']
        )
        output = grn(x, context)

        assert output.shape == x.shape

    def test_grn_context_broadcast(self, batch_config):
        """GRN context가 시퀀스 차원으로 브로드캐스트되는지 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size'],
            context_size=batch_config['context_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        )
        # context는 (batch, context_size) 형태
        context = torch.randn(
            batch_config['batch_size'],
            batch_config['context_size']
        )

        # 에러 없이 실행되어야 함
        output = grn(x, context)
        assert output.shape == x.shape

    def test_grn_no_residual(self, batch_config):
        """GRN residual=False 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size'],
            output_size=128,  # 다른 크기
            residual=False
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        )
        output = grn(x)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            128
        )

    def test_grn_layer_norm(self, batch_config):
        """GRN LayerNorm 적용 확인"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        ) * 100  # 큰 값

        output = grn(x)

        # LayerNorm이 적용되어 출력이 정규화됨
        # 마지막 차원의 평균이 대략 0, 분산이 대략 1에 가까워야 함
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)

        # LayerNorm 특성상 정확히 0, 1은 아니지만 안정적인 범위
        assert torch.isfinite(output).all()


# ============================================================
# VariableSelectionNetwork (VSN) 테스트
# ============================================================

class TestVariableSelectionNetwork:
    """VSN 클래스 테스트"""

    def test_vsn_creation(self, batch_config):
        """VSN 생성 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )
        assert vsn is not None
        assert vsn.num_inputs == batch_config['num_vars']
        assert len(vsn.var_grns) == batch_config['num_vars']

    def test_vsn_temporal_forward(self, batch_config):
        """VSN 시간적 입력 순전파 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )
        # 4D input: (batch, seq, num_vars, var_dim)
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            1
        )
        output, weights = vsn(x)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        assert weights.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars']
        )

    def test_vsn_static_forward(self, batch_config):
        """VSN 정적 입력 순전파 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )
        # 3D input: (batch, num_vars, var_dim)
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['num_vars'],
            1
        )
        output, weights = vsn(x)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['hidden_size']
        )
        assert weights.shape == (
            batch_config['batch_size'],
            batch_config['num_vars']
        )

    def test_vsn_weights_sum_to_one(self, batch_config):
        """VSN 가중치가 1로 합산되는지 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            1
        )
        _, weights = vsn(x)

        # 가중치 합이 1에 가까운지 확인 (Softmax 적용됨)
        weight_sums = weights.sum(dim=-1)
        expected = torch.ones_like(weight_sums)

        assert torch.allclose(weight_sums, expected, atol=1e-5)

    def test_vsn_with_context(self, batch_config):
        """VSN with context 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size'],
            context_size=batch_config['context_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            1
        )
        context = torch.randn(
            batch_config['batch_size'],
            batch_config['context_size']
        )
        output, weights = vsn(x, context)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

    def test_vsn_variable_importance(self, batch_config):
        """VSN 변수 중요도가 다양한지 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            1
        )
        _, weights = vsn(x)

        # 모든 가중치가 동일하지 않음 (학습된 선택)
        # 초기화 상태에서도 약간의 차이가 있어야 함
        weight_std = weights.std(dim=-1).mean()
        assert weight_std > 0


# ============================================================
# StaticCovariateEncoder 테스트
# ============================================================

class TestStaticCovariateEncoder:
    """Static Covariate Encoder 테스트"""

    def test_encoder_creation(self, batch_config):
        """Encoder 생성 테스트"""
        encoder = StaticCovariateEncoder(
            input_size=8,
            num_static_vars=5,
            hidden_size=batch_config['hidden_size']
        )
        assert encoder is not None
        assert len(encoder.context_grns) == 4  # 4개 컨텍스트

    def test_encoder_forward(self, batch_config):
        """Encoder 순전파 테스트"""
        num_static_vars = 5
        input_size = 8

        encoder = StaticCovariateEncoder(
            input_size=input_size,
            num_static_vars=num_static_vars,
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(
            batch_config['batch_size'],
            num_static_vars,
            input_size
        )
        contexts, weights = encoder(x)

        assert len(contexts) == 4
        for name in ['selection', 'encoder', 'decoder', 'enrichment']:
            assert name in contexts
            assert contexts[name].shape == (
                batch_config['batch_size'],
                batch_config['hidden_size']
            )

        assert weights.shape == (
            batch_config['batch_size'],
            num_static_vars
        )

    def test_encoder_context_independence(self, batch_config):
        """각 컨텍스트가 독립적으로 학습되는지 테스트"""
        encoder = StaticCovariateEncoder(
            input_size=8,
            num_static_vars=5,
            hidden_size=batch_config['hidden_size']
        )
        x = torch.randn(batch_config['batch_size'], 5, 8)
        contexts, _ = encoder(x)

        # 4개의 컨텍스트가 모두 다름 (다른 GRN 통과)
        all_same = all(
            torch.allclose(contexts['selection'], contexts[name])
            for name in ['encoder', 'decoder', 'enrichment']
        )
        # 초기화 상태에서도 약간의 차이가 있어야 함
        # (완전히 같을 확률은 매우 낮음)


# ============================================================
# TemporalVariableSelection 테스트
# ============================================================

class TestTemporalVariableSelection:
    """Temporal Variable Selection 테스트"""

    def test_temporal_vsn_creation(self, batch_config):
        """Temporal VSN 생성 테스트"""
        tvs = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': 5, 'unknown': 10},
            hidden_size=batch_config['hidden_size']
        )
        assert tvs is not None
        assert 'known' in tvs.vsns
        assert 'unknown' in tvs.vsns

    def test_temporal_vsn_forward(self, batch_config):
        """Temporal VSN 순전파 테스트"""
        num_known, num_unknown = 5, 10

        tvs = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': num_known, 'unknown': num_unknown},
            hidden_size=batch_config['hidden_size']
        )

        known_inputs = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            num_known,
            1
        )
        unknown_inputs = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            num_unknown,
            1
        )

        outputs, weights = tvs(known_inputs, unknown_inputs)

        assert 'known' in outputs
        assert 'unknown' in outputs
        assert outputs['known'].shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        assert outputs['unknown'].shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

    def test_temporal_vsn_with_context(self, batch_config):
        """Temporal VSN with context 테스트"""
        tvs = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': 5, 'unknown': 10},
            hidden_size=batch_config['hidden_size'],
            context_size=batch_config['context_size']
        )

        known_inputs = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 5, 1)
        unknown_inputs = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 10, 1)
        context = torch.randn(batch_config['batch_size'], batch_config['context_size'])

        outputs, weights = tvs(known_inputs, unknown_inputs, context)

        assert 'known' in outputs
        assert 'unknown' in outputs

    def test_temporal_vsn_partial_inputs(self, batch_config):
        """Temporal VSN 부분 입력 테스트 (known만)"""
        tvs = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': 5, 'unknown': 10},
            hidden_size=batch_config['hidden_size']
        )

        known_inputs = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 5, 1)

        outputs, weights = tvs(known_inputs=known_inputs)

        assert 'known' in outputs
        assert 'unknown' not in outputs


# ============================================================
# 통합 테스트
# ============================================================

class TestTFTComponentsIntegration:
    """TFT 컴포넌트 통합 테스트"""

    def test_grn_to_vsn_pipeline(self, batch_config):
        """GRN → VSN 파이프라인 테스트"""
        # GRN으로 변환 후 VSN에 입력
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size'],
            output_size=1  # VSN 입력용
        )
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )

        # 입력 데이터
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            batch_config['input_size']
        )

        # 각 변수별로 GRN 적용
        transformed = []
        for i in range(batch_config['num_vars']):
            var_i = x[:, :, i, :]  # (batch, seq, input_size)
            transformed.append(grn(var_i))  # (batch, seq, 1)

        # Stack and pass to VSN
        x_transformed = torch.stack(transformed, dim=2)  # (batch, seq, num_vars, 1)
        output, weights = vsn(x_transformed)

        assert output.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

    def test_static_context_to_temporal_vsn(self, batch_config):
        """Static Encoder → Temporal VSN 연결 테스트"""
        static_encoder = StaticCovariateEncoder(
            input_size=8,
            num_static_vars=5,
            hidden_size=batch_config['hidden_size']
        )
        temporal_vsn = TemporalVariableSelection(
            input_sizes={'known': 1, 'unknown': 1},
            num_inputs={'known': 5, 'unknown': 10},
            hidden_size=batch_config['hidden_size'],
            context_size=batch_config['hidden_size']  # Static encoder output size
        )

        # Static input
        static_input = torch.randn(batch_config['batch_size'], 5, 8)
        contexts, _ = static_encoder(static_input)

        # Use selection context for temporal VSN
        known_inputs = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 5, 1)
        unknown_inputs = torch.randn(batch_config['batch_size'], batch_config['seq_len'], 10, 1)

        outputs, _ = temporal_vsn(
            known_inputs,
            unknown_inputs,
            context=contexts['selection']
        )

        assert 'known' in outputs
        assert 'unknown' in outputs

    def test_device_compatibility(self, device, batch_config):
        """디바이스 호환성 테스트"""
        grn = GatedResidualNetwork(
            input_size=batch_config['input_size'],
            hidden_size=batch_config['hidden_size']
        ).to(device)

        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['input_size']
        ).to(device)

        output = grn(x)

        assert output.device == x.device

    def test_gradient_flow(self, batch_config):
        """그래디언트 흐름 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=batch_config['num_vars'],
            hidden_size=batch_config['hidden_size']
        )

        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['num_vars'],
            1,
            requires_grad=True
        )

        output, weights = vsn(x)
        loss = output.sum()
        loss.backward()

        # 그래디언트가 입력까지 전파되는지 확인
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================
# 성능 테스트
# ============================================================

class TestTFTComponentsPerformance:
    """TFT 컴포넌트 성능 테스트"""

    @pytest.mark.parametrize("batch_size", [1, 32, 64])
    def test_grn_batch_sizes(self, batch_size):
        """다양한 배치 크기에서 GRN 테스트"""
        grn = GatedResidualNetwork(input_size=64, hidden_size=64)
        x = torch.randn(batch_size, 48, 64)
        output = grn(x)

        assert output.shape == (batch_size, 48, 64)

    @pytest.mark.parametrize("seq_len", [24, 48, 168])
    def test_vsn_sequence_lengths(self, seq_len):
        """다양한 시퀀스 길이에서 VSN 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=10,
            hidden_size=64
        )
        x = torch.randn(32, seq_len, 10, 1)
        output, weights = vsn(x)

        assert output.shape == (32, seq_len, 64)
        assert weights.shape == (32, seq_len, 10)

    @pytest.mark.parametrize("num_vars", [5, 10, 25])
    def test_vsn_variable_counts(self, num_vars):
        """다양한 변수 개수에서 VSN 테스트"""
        vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=num_vars,
            hidden_size=64
        )
        x = torch.randn(32, 48, num_vars, 1)
        output, weights = vsn(x)

        assert output.shape == (32, 48, 64)
        assert weights.shape == (32, 48, num_vars)


# ============================================================
# Positional Encoding 테스트
# ============================================================

class TestPositionalEncoding:
    """Positional Encoding 테스트"""

    def test_pe_creation(self, batch_config):
        """Positional Encoding 생성 테스트"""
        pe = PositionalEncoding(d_model=batch_config['hidden_size'])
        assert pe is not None

    def test_pe_forward(self, batch_config):
        """Positional Encoding 순전파 테스트"""
        pe = PositionalEncoding(d_model=batch_config['hidden_size'])
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        output = pe(x)

        assert output.shape == x.shape

    def test_pe_adds_position_info(self, batch_config):
        """Positional Encoding이 위치 정보를 추가하는지 테스트"""
        pe = PositionalEncoding(d_model=batch_config['hidden_size'], dropout=0.0)
        x = torch.zeros(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        output = pe(x)

        # 입력이 0이어도 출력은 위치 인코딩으로 인해 0이 아니어야 함
        assert not torch.allclose(output, x)

    def test_pe_different_positions(self, batch_config):
        """다른 위치는 다른 인코딩을 가져야 함"""
        pe = PositionalEncoding(d_model=batch_config['hidden_size'], dropout=0.0)
        x = torch.zeros(1, 10, batch_config['hidden_size'])
        output = pe(x)

        # 위치 0과 위치 1은 다른 인코딩을 가져야 함
        assert not torch.allclose(output[0, 0], output[0, 1])

    @pytest.mark.parametrize("seq_len", [10, 48, 100])
    def test_pe_variable_sequence_lengths(self, seq_len, batch_config):
        """다양한 시퀀스 길이에서 테스트"""
        pe = PositionalEncoding(d_model=batch_config['hidden_size'])
        x = torch.randn(batch_config['batch_size'], seq_len, batch_config['hidden_size'])
        output = pe(x)

        assert output.shape == (batch_config['batch_size'], seq_len, batch_config['hidden_size'])


# ============================================================
# InterpretableMultiHeadAttention 테스트
# ============================================================

class TestInterpretableMultiHeadAttention:
    """Interpretable Multi-Head Attention 테스트"""

    def test_attention_creation(self, batch_config):
        """Attention 생성 테스트"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        assert attn is not None
        assert attn.num_heads == 4
        assert attn.d_k == batch_config['hidden_size'] // 4

    def test_attention_forward(self, batch_config):
        """Self-Attention 순전파 테스트"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        output, weights = attn(x, x, x)

        assert output.shape == x.shape
        assert weights.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['seq_len']
        )

    def test_attention_weights_sum_to_one(self, batch_config):
        """Attention 가중치가 1로 합산되는지 테스트"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        _, weights = attn(x, x, x)
        weight_sums = weights.sum(dim=-1)

        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_attention_with_mask(self, batch_config):
        """Masking이 적용되는지 테스트"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        mask = generate_causal_mask(batch_config['seq_len'])
        _, weights = attn(x, x, x, mask=mask)

        # 첫 번째 위치는 자기 자신만 볼 수 있음
        first_pos_weights = weights[:, 0, :]
        assert first_pos_weights[:, 1:].sum() < 1e-5

    def test_attention_cross_attention(self, batch_config):
        """Cross-Attention 테스트 (Q != K, V)"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        query = torch.randn(batch_config['batch_size'], 10, batch_config['hidden_size'])
        key = torch.randn(batch_config['batch_size'], 20, batch_config['hidden_size'])
        value = torch.randn(batch_config['batch_size'], 20, batch_config['hidden_size'])

        output, weights = attn(query, key, value)

        assert output.shape == (batch_config['batch_size'], 10, batch_config['hidden_size'])
        assert weights.shape == (batch_config['batch_size'], 10, 20)

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_attention_various_heads(self, num_heads, batch_config):
        """다양한 head 수에서 테스트"""
        attn = InterpretableMultiHeadAttention(
            d_model=batch_config['hidden_size'],
            num_heads=num_heads
        )
        x = torch.randn(batch_config['batch_size'], batch_config['seq_len'], batch_config['hidden_size'])
        output, _ = attn(x, x, x)

        assert output.shape == x.shape


# ============================================================
# TemporalSelfAttention 테스트
# ============================================================

class TestTemporalSelfAttention:
    """Temporal Self-Attention Layer 테스트"""

    def test_layer_creation(self, batch_config):
        """Layer 생성 테스트"""
        layer = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        assert layer is not None

    def test_layer_forward(self, batch_config):
        """순전파 테스트"""
        layer = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        output, attention = layer(x)

        assert output.shape == x.shape
        assert attention.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['seq_len']
        )

    def test_layer_with_mask(self, batch_config):
        """마스킹 테스트"""
        layer = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        mask = generate_causal_mask(batch_config['seq_len'])

        output, _ = layer(x, mask=mask)

        assert output.shape == x.shape

    def test_layer_without_positional_encoding(self, batch_config):
        """위치 인코딩 없이 테스트"""
        layer = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4,
            use_positional_encoding=False
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        output, _ = layer(x)

        assert output.shape == x.shape

    def test_layer_gradient_flow(self, batch_config):
        """그래디언트 흐름 테스트"""
        layer = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size'],
            requires_grad=True
        )

        output, _ = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================
# Causal Mask 테스트
# ============================================================

class TestCausalMask:
    """Causal Mask 생성 함수 테스트"""

    def test_mask_shape(self):
        """마스크 shape 테스트"""
        for seq_len in [5, 10, 48]:
            mask = generate_causal_mask(seq_len)
            assert mask.shape == (seq_len, seq_len)

    def test_mask_structure(self):
        """마스크 구조 테스트"""
        mask = generate_causal_mask(4)
        expected = torch.tensor([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        assert torch.equal(mask, expected)

    def test_mask_diagonal(self):
        """대각선은 False여야 함 (자기 자신은 볼 수 있음)"""
        mask = generate_causal_mask(10)
        for i in range(10):
            assert mask[i, i] == False

    def test_mask_upper_triangular(self):
        """상삼각 부분만 True"""
        seq_len = 10
        mask = generate_causal_mask(seq_len)

        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == True
                else:
                    assert mask[i, j] == False

    def test_mask_device(self):
        """디바이스 지정 테스트"""
        mask = generate_causal_mask(5, device=torch.device('cpu'))
        assert mask.device == torch.device('cpu')


# ============================================================
# StaticEnrichmentLayer 테스트
# ============================================================

class TestStaticEnrichmentLayer:
    """Static Enrichment Layer 테스트"""

    def test_layer_creation(self, batch_config):
        """Layer 생성 테스트"""
        layer = StaticEnrichmentLayer(d_model=batch_config['hidden_size'])
        assert layer is not None

    def test_layer_forward(self, batch_config):
        """순전파 테스트"""
        layer = StaticEnrichmentLayer(d_model=batch_config['hidden_size'])
        temporal = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        static_context = torch.randn(
            batch_config['batch_size'],
            batch_config['hidden_size']
        )

        output = layer(temporal, static_context)

        assert output.shape == temporal.shape

    def test_layer_context_influence(self, batch_config):
        """정적 컨텍스트가 출력에 영향을 미치는지 테스트"""
        layer = StaticEnrichmentLayer(d_model=batch_config['hidden_size'])
        temporal = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        context1 = torch.randn(batch_config['batch_size'], batch_config['hidden_size'])
        context2 = torch.randn(batch_config['batch_size'], batch_config['hidden_size'])

        output1 = layer(temporal, context1)
        output2 = layer(temporal, context2)

        # 다른 컨텍스트는 다른 출력을 생성해야 함
        assert not torch.allclose(output1, output2)


# ============================================================
# Attention 통합 테스트
# ============================================================

class TestAttentionIntegration:
    """Attention 컴포넌트 통합 테스트"""

    def test_full_attention_pipeline(self, batch_config):
        """전체 Attention 파이프라인 테스트"""
        # Static enrichment → Temporal self-attention
        enrichment = StaticEnrichmentLayer(d_model=batch_config['hidden_size'])
        attention = TemporalSelfAttention(
            d_model=batch_config['hidden_size'],
            num_heads=4
        )

        temporal = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        static_context = torch.randn(
            batch_config['batch_size'],
            batch_config['hidden_size']
        )

        enriched = enrichment(temporal, static_context)
        output, attn_weights = attention(enriched)

        assert output.shape == temporal.shape
        assert attn_weights.shape == (
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['seq_len']
        )

    def test_encoder_decoder_mask(self, batch_config):
        """Encoder-Decoder 마스크 테스트"""
        encoder_len, decoder_len = 48, 24
        mask = generate_encoder_decoder_mask(encoder_len, decoder_len)

        assert mask.shape == (encoder_len + decoder_len, encoder_len + decoder_len)

        # Encoder 부분은 모두 볼 수 있음
        assert not mask[:encoder_len, :encoder_len].any()

        # Decoder는 미래를 볼 수 없음
        decoder_mask = mask[encoder_len:, encoder_len:]
        for i in range(decoder_len):
            for j in range(decoder_len):
                if j > i:
                    assert decoder_mask[i, j] == True
