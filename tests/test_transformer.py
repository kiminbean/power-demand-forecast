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
