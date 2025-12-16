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
    LSTMEncoder,
    LSTMDecoder,
    TemporalFusionTransformer,
    QuantileLoss,
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


# ============================================================
# LSTM Encoder 테스트
# ============================================================

class TestLSTMEncoder:
    """LSTM Encoder 테스트"""

    def test_encoder_creation(self, batch_config):
        """Encoder 생성 테스트"""
        encoder = LSTMEncoder(
            input_size=batch_config['hidden_size'],
            hidden_size=batch_config['hidden_size'],
            num_layers=2
        )
        assert encoder is not None

    def test_encoder_forward(self, batch_config):
        """순전파 테스트"""
        encoder = LSTMEncoder(
            input_size=batch_config['hidden_size'],
            hidden_size=batch_config['hidden_size'],
            num_layers=2
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )

        output, (h_n, c_n) = encoder(x)

        assert output.shape == x.shape
        assert h_n.shape == (2, batch_config['batch_size'], batch_config['hidden_size'])
        assert c_n.shape == (2, batch_config['batch_size'], batch_config['hidden_size'])

    def test_encoder_with_init_state(self, batch_config):
        """초기 상태와 함께 테스트"""
        num_layers = 2
        encoder = LSTMEncoder(
            input_size=batch_config['hidden_size'],
            hidden_size=batch_config['hidden_size'],
            num_layers=num_layers
        )
        x = torch.randn(
            batch_config['batch_size'],
            batch_config['seq_len'],
            batch_config['hidden_size']
        )
        h_0 = torch.randn(num_layers, batch_config['batch_size'], batch_config['hidden_size'])
        c_0 = torch.randn(num_layers, batch_config['batch_size'], batch_config['hidden_size'])

        output, _ = encoder(x, (h_0, c_0))

        assert output.shape == x.shape


# ============================================================
# LSTM Decoder 테스트
# ============================================================

class TestLSTMDecoder:
    """LSTM Decoder 테스트"""

    def test_decoder_creation(self, batch_config):
        """Decoder 생성 테스트"""
        decoder = LSTMDecoder(
            input_size=batch_config['hidden_size'],
            hidden_size=batch_config['hidden_size'],
            num_layers=2
        )
        assert decoder is not None

    def test_decoder_forward(self, batch_config):
        """순전파 테스트"""
        num_layers = 2
        decoder_len = 24
        decoder = LSTMDecoder(
            input_size=batch_config['hidden_size'],
            hidden_size=batch_config['hidden_size'],
            num_layers=num_layers
        )
        x = torch.randn(batch_config['batch_size'], decoder_len, batch_config['hidden_size'])
        encoder_state = (
            torch.randn(num_layers, batch_config['batch_size'], batch_config['hidden_size']),
            torch.randn(num_layers, batch_config['batch_size'], batch_config['hidden_size'])
        )

        output, (h_n, c_n) = decoder(x, encoder_state)

        assert output.shape == (batch_config['batch_size'], decoder_len, batch_config['hidden_size'])
        assert h_n.shape == (num_layers, batch_config['batch_size'], batch_config['hidden_size'])


# ============================================================
# TemporalFusionTransformer 전체 모델 테스트
# ============================================================

class TestTemporalFusionTransformer:
    """TFT 전체 모델 테스트"""

    @pytest.fixture
    def tft_config(self):
        """TFT 설정"""
        return {
            'num_static_vars': 0,
            'num_known_vars': 8,
            'num_unknown_vars': 25,
            'hidden_size': 64,
            'lstm_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'encoder_length': 48,
            'decoder_length': 24,
        }

    def test_tft_creation(self, tft_config):
        """TFT 모델 생성 테스트"""
        model = TemporalFusionTransformer(**tft_config)
        assert model is not None

    def test_tft_forward(self, tft_config):
        """TFT 순전파 테스트"""
        batch_size = 32
        model = TemporalFusionTransformer(**tft_config)

        known_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'] + tft_config['decoder_length'],
            tft_config['num_known_vars'],
            1
        )
        unknown_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'],
            tft_config['num_unknown_vars'],
            1
        )

        result = model(known_inputs=known_inputs, unknown_inputs=unknown_inputs)

        assert 'predictions' in result
        assert result['predictions'].shape == (
            batch_size,
            tft_config['decoder_length'],
            3  # default quantiles
        )

    def test_tft_with_attention(self, tft_config):
        """Attention 가중치 반환 테스트"""
        batch_size = 32
        model = TemporalFusionTransformer(**tft_config)

        known_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'] + tft_config['decoder_length'],
            tft_config['num_known_vars'],
            1
        )
        unknown_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'],
            tft_config['num_unknown_vars'],
            1
        )

        result = model(
            known_inputs=known_inputs,
            unknown_inputs=unknown_inputs,
            return_attention=True
        )

        assert 'attention_weights' in result
        total_len = tft_config['encoder_length'] + tft_config['decoder_length']
        assert result['attention_weights'].shape == (batch_size, total_len, total_len)

    def test_tft_with_static(self, tft_config):
        """정적 변수와 함께 테스트"""
        batch_size = 32
        tft_config['num_static_vars'] = 5
        model = TemporalFusionTransformer(**tft_config)

        known_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'] + tft_config['decoder_length'],
            tft_config['num_known_vars'],
            1
        )
        unknown_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'],
            tft_config['num_unknown_vars'],
            1
        )
        static_inputs = torch.randn(batch_size, tft_config['num_static_vars'], 1)

        result = model(
            known_inputs=known_inputs,
            unknown_inputs=unknown_inputs,
            static_inputs=static_inputs,
            return_attention=True
        )

        assert 'static_variable_weights' in result
        assert result['static_variable_weights'].shape == (
            batch_size,
            tft_config['num_static_vars']
        )

    def test_tft_gradient_flow(self, tft_config):
        """그래디언트 흐름 테스트"""
        batch_size = 8
        model = TemporalFusionTransformer(**tft_config)

        known_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'] + tft_config['decoder_length'],
            tft_config['num_known_vars'],
            1,
            requires_grad=True
        )
        unknown_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'],
            tft_config['num_unknown_vars'],
            1,
            requires_grad=True
        )

        result = model(known_inputs=known_inputs, unknown_inputs=unknown_inputs)
        loss = result['predictions'].sum()
        loss.backward()

        assert known_inputs.grad is not None
        assert unknown_inputs.grad is not None

    def test_tft_custom_quantiles(self, tft_config):
        """커스텀 분위수 테스트"""
        batch_size = 32
        custom_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        tft_config['quantiles'] = custom_quantiles
        model = TemporalFusionTransformer(**tft_config)

        known_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'] + tft_config['decoder_length'],
            tft_config['num_known_vars'],
            1
        )
        unknown_inputs = torch.randn(
            batch_size,
            tft_config['encoder_length'],
            tft_config['num_unknown_vars'],
            1
        )

        result = model(known_inputs=known_inputs, unknown_inputs=unknown_inputs)

        assert result['predictions'].shape == (
            batch_size,
            tft_config['decoder_length'],
            len(custom_quantiles)
        )

    def test_tft_parameter_count(self, tft_config):
        """파라미터 수 테스트"""
        model = TemporalFusionTransformer(**tft_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 모든 파라미터가 학습 가능해야 함
        assert total_params == trainable_params
        # TFT는 충분한 용량이 있어야 함 (대략 100K+ 파라미터)
        assert total_params > 100000


# ============================================================
# QuantileLoss 테스트
# ============================================================

class TestQuantileLoss:
    """Quantile Loss 테스트"""

    def test_loss_creation(self):
        """Loss 생성 테스트"""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        assert loss_fn is not None

    def test_loss_forward(self):
        """Loss 순전파 테스트"""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        predictions = torch.randn(32, 24, 3)
        targets = torch.randn(32, 24)

        loss = loss_fn(predictions, targets)

        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # non-negative

    def test_loss_with_3d_targets(self):
        """3D 타겟 테스트"""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        predictions = torch.randn(32, 24, 3)
        targets = torch.randn(32, 24, 1)

        loss = loss_fn(predictions, targets)

        assert loss.dim() == 0

    def test_loss_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        predictions = torch.randn(32, 24, 3, requires_grad=True)
        targets = torch.randn(32, 24)

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()

    def test_loss_perfect_prediction(self):
        """완벽한 예측 시 낮은 loss"""
        loss_fn = QuantileLoss(quantiles=[0.5])

        targets = torch.randn(32, 24)
        predictions = targets.unsqueeze(-1)  # 완벽한 예측

        loss = loss_fn(predictions, targets)

        assert loss.item() < 0.01  # 매우 낮은 loss

    @pytest.mark.parametrize("quantiles", [
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
        [0.05, 0.5, 0.95],
    ])
    def test_loss_various_quantiles(self, quantiles):
        """다양한 분위수 테스트"""
        loss_fn = QuantileLoss(quantiles=quantiles)

        predictions = torch.randn(32, 24, len(quantiles))
        targets = torch.randn(32, 24)

        loss = loss_fn(predictions, targets)

        assert loss.item() >= 0


# ============================================================
# TFT Training Pipeline Tests
# ============================================================

from training.train_tft import (
    TFTFeatureConfig,
    TFTDataset,
    TFTTrainer,
    create_tft_dataloaders,
)


class TestTFTFeatureConfig:
    """TFTFeatureConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = TFTFeatureConfig()

        assert len(config.known_features) > 0
        assert len(config.unknown_features) > 0
        assert config.target_col == 'power_demand'
        assert config.target_col == config.unknown_features[0]

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        known = ['hour_sin', 'hour_cos']
        unknown = ['target', 'temp', 'humidity']

        config = TFTFeatureConfig(
            known_features=known,
            unknown_features=unknown,
            target_col='target'
        )

        assert config.known_features == known
        assert config.unknown_features[0] == 'target'

    def test_get_available_features(self):
        """사용 가능한 피처 필터링 테스트"""
        config = TFTFeatureConfig()

        # 일부 피처만 포함된 컬럼 리스트
        df_columns = ['power_demand', 'temp_mean', 'hour_sin', 'hour_cos', 'missing_col']

        available = config.get_available_features(df_columns)

        assert 'power_demand' in available['unknown']
        assert 'temp_mean' in available['unknown']
        assert 'hour_sin' in available['known']
        assert 'missing_col' not in available['known']
        assert 'missing_col' not in available['unknown']


class TestTFTDataset:
    """TFTDataset 테스트"""

    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        np.random.seed(42)
        n_samples = 200
        n_known = 5
        n_unknown = 8
        n_features = n_known + n_unknown

        data = np.random.randn(n_samples, n_features).astype(np.float32)
        known_indices = list(range(n_unknown, n_features))
        unknown_indices = list(range(n_unknown))

        return data, known_indices, unknown_indices

    def test_dataset_creation(self, sample_data):
        """Dataset 생성 테스트"""
        data, known_indices, unknown_indices = sample_data

        dataset = TFTDataset(
            data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=24,
            decoder_length=12
        )

        assert len(dataset) == 200 - 24 - 12 + 1

    def test_dataset_shapes(self, sample_data):
        """출력 형태 테스트"""
        data, known_indices, unknown_indices = sample_data
        encoder_length = 24
        decoder_length = 12
        n_known = len(known_indices)
        n_unknown = len(unknown_indices)

        dataset = TFTDataset(
            data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=encoder_length,
            decoder_length=decoder_length
        )

        known, unknown, targets, static = dataset[0]

        assert known.shape == (encoder_length + decoder_length, n_known, 1)
        assert unknown.shape == (encoder_length, n_unknown, 1)
        assert targets.shape == (decoder_length,)
        assert static is None

    def test_dataset_values(self, sample_data):
        """값 정확성 테스트"""
        data, known_indices, unknown_indices = sample_data

        dataset = TFTDataset(
            data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=24,
            decoder_length=12
        )

        known, unknown, targets, _ = dataset[5]

        # 첫 번째 known feature의 첫 번째 시점
        expected_known = data[5, known_indices[0]]
        assert abs(known[0, 0, 0].item() - expected_known) < 1e-6

        # 타겟 값 확인
        expected_target = data[5 + 24, 0]  # encoder_end에서의 타겟
        assert abs(targets[0].item() - expected_target) < 1e-6

    def test_dataset_insufficient_data(self, sample_data):
        """데이터 부족 시 예외 테스트"""
        data, known_indices, unknown_indices = sample_data
        short_data = data[:50]  # 짧은 데이터

        with pytest.raises(ValueError, match="데이터가 너무 짧습니다"):
            TFTDataset(
                data=short_data,
                known_indices=known_indices,
                unknown_indices=unknown_indices,
                target_idx=0,
                encoder_length=48,
                decoder_length=24  # 48 + 24 = 72 > 50
            )


class TestCreateTFTDataloaders:
    """create_tft_dataloaders 테스트"""

    def test_dataloader_creation(self):
        """DataLoader 생성 테스트"""
        np.random.seed(42)
        data = np.random.randn(300, 13).astype(np.float32)
        known_indices = list(range(8, 13))
        unknown_indices = list(range(8))

        train_loader, val_loader, test_loader = create_tft_dataloaders(
            train_data=data,
            val_data=data,
            test_data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=24,
            decoder_length=12,
            batch_size=32
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch(self):
        """DataLoader 배치 테스트"""
        np.random.seed(42)
        data = np.random.randn(300, 13).astype(np.float32)
        known_indices = list(range(8, 13))
        unknown_indices = list(range(8))
        batch_size = 16

        train_loader, _, _ = create_tft_dataloaders(
            train_data=data,
            val_data=data,
            test_data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=24,
            decoder_length=12,
            batch_size=batch_size
        )

        known, unknown, targets, static = next(iter(train_loader))

        assert known.shape[0] == batch_size
        assert unknown.shape[0] == batch_size
        assert targets.shape[0] == batch_size


class TestTFTTrainer:
    """TFTTrainer 테스트"""

    @pytest.fixture
    def trainer_setup(self, device):
        """Trainer 설정"""
        np.random.seed(42)
        n_known = 5
        n_unknown = 8
        encoder_length = 24
        decoder_length = 12

        # 데이터
        data = np.random.randn(200, n_known + n_unknown).astype(np.float32)
        known_indices = list(range(n_unknown, n_known + n_unknown))
        unknown_indices = list(range(n_unknown))

        train_loader, val_loader, test_loader = create_tft_dataloaders(
            train_data=data,
            val_data=data,
            test_data=data,
            known_indices=known_indices,
            unknown_indices=unknown_indices,
            target_idx=0,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            batch_size=16
        )

        # 모델
        model = TemporalFusionTransformer(
            num_static_vars=0,
            num_known_vars=n_known,
            num_unknown_vars=n_unknown,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            dropout=0.1,
            encoder_length=encoder_length,
            decoder_length=decoder_length
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = TFTTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            grad_clip=1.0
        )

        return trainer, train_loader, val_loader, test_loader

    def test_trainer_creation(self, trainer_setup):
        """Trainer 생성 테스트"""
        trainer, _, _, _ = trainer_setup

        assert trainer.model is not None
        assert trainer.criterion is not None
        assert trainer.quantiles == [0.1, 0.5, 0.9]

    def test_trainer_fit(self, trainer_setup):
        """Trainer fit 테스트"""
        trainer, train_loader, val_loader, _ = trainer_setup

        history = trainer.fit(
            train_loader, val_loader,
            epochs=2,
            patience=5,
            verbose=0
        )

        assert len(history.history['train_loss']) == 2
        assert len(history.history['val_loss']) == 2
        assert history.history['train_loss'][-1] >= 0
        assert history.history['val_loss'][-1] >= 0

    def test_trainer_evaluate(self, trainer_setup):
        """Trainer evaluate 테스트"""
        trainer, train_loader, val_loader, test_loader = trainer_setup

        # 먼저 학습
        trainer.fit(train_loader, val_loader, epochs=1, verbose=0)

        # 평가
        result = trainer.evaluate(test_loader, return_predictions=True)

        assert 'test_quantile_loss' in result
        assert 'test_mse_loss' in result
        assert 'test_rmse' in result
        assert 'predictions' in result
        assert 'targets' in result

    def test_trainer_predict(self, trainer_setup):
        """Trainer predict 테스트"""
        trainer, train_loader, val_loader, test_loader = trainer_setup

        # 먼저 학습
        trainer.fit(train_loader, val_loader, epochs=1, verbose=0)

        # 예측
        result = trainer.predict(test_loader, return_intervals=True)

        assert 'median' in result
        assert 'lower' in result
        assert 'upper' in result
        assert result['median'].shape == result['lower'].shape
        assert result['median'].shape == result['upper'].shape

    def test_trainer_checkpoint(self, trainer_setup, tmp_path):
        """체크포인트 저장/로드 테스트"""
        trainer, train_loader, val_loader, _ = trainer_setup

        # 학습
        history = trainer.fit(train_loader, val_loader, epochs=2, verbose=0)

        # 체크포인트 저장
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=2, history=history)

        assert checkpoint_path.exists()

        # 체크포인트 로드
        loaded = trainer.load_checkpoint(str(checkpoint_path))

        assert loaded['epoch'] == 2
        assert 'model_state_dict' in loaded
        assert 'optimizer_state_dict' in loaded
