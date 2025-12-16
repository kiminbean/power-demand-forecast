"""
통합 파이프라인 테스트 (Task 25)
=================================
전체 파이프라인 통합 테스트
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """샘플 데이터"""
    np.random.seed(42)
    n_hours = 500

    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=n_hours),
        periods=n_hours,
        freq='h'
    )

    # 패턴이 있는 수요 데이터
    hour = dates.hour
    base = 1000 + 200 * np.sin(2 * np.pi * hour / 24)
    noise = np.random.randn(n_hours) * 30

    return pd.DataFrame({
        'datetime': dates,
        'demand': base + noise,
        'temperature': 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.randn(n_hours) * 2,
        'humidity': 60 + np.random.randn(n_hours) * 10,
        'hour': hour,
        'day_of_week': dates.dayofweek
    })


@pytest.fixture
def temp_dir():
    """임시 디렉토리"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# PipelineStage 테스트
# ============================================================================

class TestPipelineStage:
    """PipelineStage 테스트"""

    def test_stage_values(self):
        """단계 값 확인"""
        from src.pipeline import PipelineStage

        assert PipelineStage.DATA_LOAD.value == "data_load"
        assert PipelineStage.MODEL_TRAINING.value == "model_training"
        assert PipelineStage.PREDICTION.value == "prediction"


# ============================================================================
# PipelineConfig 테스트
# ============================================================================

class TestPipelineConfig:
    """PipelineConfig 테스트"""

    def test_default_config(self):
        """기본 설정"""
        from src.pipeline import PipelineConfig

        config = PipelineConfig()

        assert config.name == "power_demand_forecast"
        assert config.sequence_length == 168
        assert config.prediction_horizon == 24
        assert config.model_type == "lstm"

    def test_custom_config(self):
        """커스텀 설정"""
        from src.pipeline import PipelineConfig

        config = PipelineConfig(
            name="custom_pipeline",
            sequence_length=24,
            hidden_size=64,
            epochs=50
        )

        assert config.name == "custom_pipeline"
        assert config.sequence_length == 24
        assert config.hidden_size == 64

    def test_config_to_dict(self):
        """딕셔너리 변환"""
        from src.pipeline import PipelineConfig

        config = PipelineConfig()
        result = config.to_dict()

        assert 'name' in result
        assert 'sequence_length' in result
        assert 'model_type' in result


# ============================================================================
# PipelineResult 테스트
# ============================================================================

class TestPipelineResult:
    """PipelineResult 테스트"""

    def test_result_creation(self):
        """결과 생성"""
        from src.pipeline import PipelineResult, PipelineStage

        result = PipelineResult(
            success=True,
            stage=PipelineStage.DATA_LOAD,
            message="Loaded 1000 records",
            metrics={'n_records': 1000}
        )

        assert result.success
        assert result.stage == PipelineStage.DATA_LOAD
        assert result.metrics['n_records'] == 1000

    def test_result_to_dict(self):
        """딕셔너리 변환"""
        from src.pipeline import PipelineResult, PipelineStage

        result = PipelineResult(
            success=True,
            stage=PipelineStage.PREDICTION,
            message="Predictions generated"
        )

        result_dict = result.to_dict()

        assert 'success' in result_dict
        assert 'stage' in result_dict
        assert result_dict['stage'] == 'prediction'


# ============================================================================
# PowerDemandPipeline 테스트
# ============================================================================

class TestPowerDemandPipeline:
    """PowerDemandPipeline 테스트"""

    def test_pipeline_creation(self, temp_dir):
        """파이프라인 생성"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir,
            log_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)

        assert pipeline is not None
        assert pipeline.model is None
        assert pipeline.results == []

    def test_load_data_from_dataframe(self, sample_data, temp_dir):
        """DataFrame에서 데이터 로드"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        result = pipeline.load_data(sample_data)

        assert result.success
        assert len(pipeline.data) == len(sample_data)

    def test_preprocess(self, sample_data, temp_dir):
        """전처리"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.load_data(sample_data)
        result = pipeline.preprocess()

        assert result.success
        assert len(pipeline.feature_names) > 0

    def test_engineer_features(self, sample_data, temp_dir):
        """피처 엔지니어링"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.load_data(sample_data)
        pipeline.preprocess()
        result = pipeline.engineer_features()

        assert result.success
        # 지연 피처와 롤링 피처가 추가됨
        assert len(pipeline.feature_names) > 4

    def test_prepare_sequences(self, sample_data, temp_dir):
        """시퀀스 준비"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=12,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)

        data = np.random.rand(100, 5)
        target = np.random.rand(100)

        X, y = pipeline.prepare_sequences(data, target)

        assert X.shape[0] == 100 - 24 - 12 + 1
        assert X.shape[1] == 24
        assert X.shape[2] == 5
        assert y.shape[1] == 12

    def test_train_model(self, sample_data, temp_dir):
        """모델 학습"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            num_layers=1,
            epochs=5,
            early_stopping_patience=3,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.load_data(sample_data)
        pipeline.preprocess()
        pipeline.engineer_features()
        result = pipeline.train_model()

        assert result.success
        assert pipeline.model is not None
        assert 'train_loss' in result.metrics
        assert 'val_loss' in result.metrics

    def test_predict(self, sample_data, temp_dir):
        """예측"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            num_layers=1,
            epochs=5,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.load_data(sample_data)
        pipeline.preprocess()
        pipeline.engineer_features()
        pipeline.train_model()
        result = pipeline.predict()

        assert result.success
        assert result.data is not None
        assert len(result.data) == config.prediction_horizon

    def test_predict_without_model(self, temp_dir):
        """모델 없이 예측 시도"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        result = pipeline.predict()

        assert not result.success


# ============================================================================
# SimpleLSTM 테스트
# ============================================================================

class TestSimpleLSTM:
    """SimpleLSTM 테스트"""

    def test_model_creation(self):
        """모델 생성"""
        from src.pipeline import SimpleLSTM

        model = SimpleLSTM(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            output_size=24
        )

        assert model is not None

    def test_forward_pass(self):
        """순전파"""
        from src.pipeline import SimpleLSTM

        model = SimpleLSTM(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            output_size=24
        )

        batch = torch.randn(4, 48, 10)  # (batch, seq_len, features)
        output = model(batch)

        assert output.shape == (4, 24)


# ============================================================================
# 헬퍼 함수 테스트
# ============================================================================

class TestHelperFunctions:
    """헬퍼 함수 테스트"""

    def test_create_default_config(self):
        """기본 설정 생성"""
        from src.pipeline import create_default_config

        config = create_default_config()

        assert config.name == "power_demand_forecast"

    def test_get_pipeline_status(self, sample_data, temp_dir):
        """파이프라인 상태 조회"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig, get_pipeline_status

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.load_data(sample_data)

        status = get_pipeline_status(pipeline)

        assert 'stages_completed' in status
        assert 'model_trained' in status
        assert 'data_loaded' in status
        assert status['data_loaded']

    def test_run_pipeline_function(self, sample_data, temp_dir):
        """run_pipeline 함수"""
        from src.pipeline import run_pipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir,
            run_anomaly_detection=False,
            run_scenario_analysis=False
        )

        result = run_pipeline(config, sample_data)

        assert 'success' in result
        assert 'stages_completed' in result


# ============================================================================
# 통합 테스트
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline_run(self, sample_data, temp_dir):
        """전체 파이프라인 실행"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir,
            run_anomaly_detection=True,
            run_scenario_analysis=False  # 모델 의존성으로 인해 비활성화
        )

        pipeline = PowerDemandPipeline(config)
        result = pipeline.run(sample_data)

        assert result['stages_completed'] > 0
        assert result['success'] or result['stages_failed'] == 0

    def test_pipeline_with_report(self, sample_data, temp_dir):
        """리포트 포함 파이프라인"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            epochs=3,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir,
            run_anomaly_detection=False,
            run_scenario_analysis=False
        )

        pipeline = PowerDemandPipeline(config)
        pipeline.run(sample_data)

        # 리포트 파일 확인
        output_path = Path(temp_dir)
        report_files = list(output_path.glob("pipeline_report_*.json"))

        assert len(report_files) >= 1

    def test_pipeline_skip_training(self, sample_data, temp_dir):
        """학습 스킵"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir
        )

        pipeline = PowerDemandPipeline(config)
        result = pipeline.run(sample_data, skip_training=True)

        assert pipeline.model is None
        # 예측은 모델이 없어서 실패
        prediction_result = next(
            (r for r in pipeline.results if r.stage.value == 'prediction'),
            None
        )
        if prediction_result:
            assert not prediction_result.success

    def test_pipeline_sample_data_generation(self, temp_dir):
        """샘플 데이터 생성 테스트"""
        from src.pipeline import PowerDemandPipeline, PipelineConfig

        config = PipelineConfig(
            sequence_length=24,
            prediction_horizon=6,
            hidden_size=32,
            epochs=2,
            data_path=temp_dir,
            model_path=temp_dir,
            output_path=temp_dir,
            run_anomaly_detection=False,
            run_scenario_analysis=False
        )

        pipeline = PowerDemandPipeline(config)
        # 데이터 소스 없이 로드 (샘플 데이터 생성)
        result = pipeline.load_data()

        assert result.success
        assert len(pipeline.data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
