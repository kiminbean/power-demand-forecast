"""
MLflow 유틸리티 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


class SimpleModel(nn.Module):
    """테스트용 모델"""
    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestModelStage:
    """ModelStage 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.training.mlflow_utils import ModelStage

        assert ModelStage.NONE.value == 'None'
        assert ModelStage.PRODUCTION.value == 'Production'
        assert ModelStage.STAGING.value == 'Staging'
        assert ModelStage.ARCHIVED.value == 'Archived'


class TestRunConfig:
    """RunConfig 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.mlflow_utils import RunConfig

        config = RunConfig(
            experiment_name='test_exp',
            run_name='run_1',
            tags={'version': '1.0'}
        )

        assert config.experiment_name == 'test_exp'
        assert config.run_name == 'run_1'


class TestMetricRecord:
    """MetricRecord 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.mlflow_utils import MetricRecord

        record = MetricRecord(name='loss', value=0.5, step=1)

        assert record.name == 'loss'
        assert record.value == 0.5
        assert record.step == 1


class TestExperimentTracker:
    """ExperimentTracker 테스트"""

    def test_creation(self, tmp_path):
        """생성"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')

        assert tracker.tracking_uri.exists()

    def test_create_experiment(self, tmp_path):
        """실험 생성"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        exp_id = tracker.create_experiment('my_experiment')

        assert exp_id is not None
        assert tracker.get_experiment('my_experiment') is not None

    def test_list_experiments(self, tmp_path):
        """실험 목록"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        tracker.create_experiment('exp1')
        tracker.create_experiment('exp2')

        experiments = tracker.list_experiments()

        assert len(experiments) == 2

    def test_start_run(self, tmp_path):
        """실행 시작"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment', run_name='test_run')

        assert run is not None
        assert run.status == 'RUNNING'
        assert tracker.active_run == run

    def test_end_run(self, tmp_path):
        """실행 종료"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        tracker.start_run('my_experiment')
        tracker.end_run()

        assert tracker.active_run is None

    def test_log_param(self, tmp_path):
        """파라미터 로깅"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment')

        tracker.log_param('learning_rate', 0.001)
        tracker.log_param('batch_size', 32)

        assert run.params['learning_rate'] == 0.001
        assert run.params['batch_size'] == 32

    def test_log_params(self, tmp_path):
        """여러 파라미터 로깅"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment')

        tracker.log_params({
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        })

        assert len(run.params) == 3

    def test_log_metric(self, tmp_path):
        """메트릭 로깅"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment')

        tracker.log_metric('loss', 0.5, step=0)
        tracker.log_metric('loss', 0.3, step=1)

        assert len(run.metrics['loss']) == 2

    def test_log_metrics(self, tmp_path):
        """여러 메트릭 로깅"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment')

        tracker.log_metrics({'loss': 0.5, 'accuracy': 0.8}, step=0)

        assert 'loss' in run.metrics
        assert 'accuracy' in run.metrics

    def test_set_tag(self, tmp_path):
        """태그 설정"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('my_experiment')

        tracker.set_tag('model_type', 'LSTM')

        assert run.tags['model_type'] == 'LSTM'

    def test_log_artifact(self, tmp_path):
        """아티팩트 로깅"""
        from src.training.mlflow_utils import ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        tracker.start_run('my_experiment')

        # 임시 파일 생성
        artifact_file = tmp_path / 'test_artifact.txt'
        artifact_file.write_text('test content')

        tracker.log_artifact(artifact_file)

        assert len(tracker.active_run.artifacts) == 1

    def test_persistence(self, tmp_path):
        """저장 및 로드"""
        from src.training.mlflow_utils import ExperimentTracker

        # 첫 번째 인스턴스
        tracker1 = ExperimentTracker(tmp_path / 'mlruns')
        tracker1.create_experiment('persistent_exp')
        run = tracker1.start_run('persistent_exp')
        tracker1.log_param('key', 'value')
        tracker1.end_run()

        # 두 번째 인스턴스 (새로 로드)
        tracker2 = ExperimentTracker(tmp_path / 'mlruns')

        assert tracker2.get_experiment('persistent_exp') is not None


class TestModelRegistry:
    """ModelRegistry 테스트"""

    def test_creation(self, tmp_path):
        """생성"""
        from src.training.mlflow_utils import ModelRegistry

        registry = ModelRegistry(tmp_path / 'registry')

        assert registry.registry_path.exists()

    def test_register_model(self, tmp_path):
        """모델 등록"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        # 모델 저장
        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        # 등록
        version = registry.register_model(
            name='my_model',
            run_id='run_123',
            model_path=model_path,
            metrics={'rmse': 0.5}
        )

        assert version.name == 'my_model'
        assert version.version == 1

    def test_register_multiple_versions(self, tmp_path):
        """여러 버전 등록"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        v1 = registry.register_model('my_model', 'run_1', model_path)
        v2 = registry.register_model('my_model', 'run_2', model_path)

        assert v1.version == 1
        assert v2.version == 2

    def test_get_model_version(self, tmp_path):
        """모델 버전 조회"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        registry.register_model('my_model', 'run_1', model_path)

        version = registry.get_model_version('my_model', 1)

        assert version is not None
        assert version.version == 1

    def test_get_latest_version(self, tmp_path):
        """최신 버전 조회"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        registry.register_model('my_model', 'run_1', model_path)
        registry.register_model('my_model', 'run_2', model_path)
        registry.register_model('my_model', 'run_3', model_path)

        latest = registry.get_latest_version('my_model')

        assert latest.version == 3

    def test_list_model_versions(self, tmp_path):
        """모델 버전 목록"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        for i in range(3):
            registry.register_model('my_model', f'run_{i}', model_path)

        versions = registry.list_model_versions('my_model')

        assert len(versions) == 3

    def test_transition_stage(self, tmp_path):
        """스테이지 전환"""
        from src.training.mlflow_utils import (
            ModelRegistry, ModelStage, PyTorchModelSaver
        )

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        registry.register_model('my_model', 'run_1', model_path)

        version = registry.transition_stage('my_model', 1, ModelStage.STAGING)

        assert version.stage == ModelStage.STAGING

    def test_transition_to_production(self, tmp_path):
        """Production 전환 시 기존 모델 아카이브"""
        from src.training.mlflow_utils import (
            ModelRegistry, ModelStage, PyTorchModelSaver
        )

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        registry.register_model('my_model', 'run_1', model_path)
        registry.register_model('my_model', 'run_2', model_path)

        registry.transition_stage('my_model', 1, ModelStage.PRODUCTION)
        registry.transition_stage('my_model', 2, ModelStage.PRODUCTION)

        v1 = registry.get_model_version('my_model', 1)
        v2 = registry.get_model_version('my_model', 2)

        assert v1.stage == ModelStage.ARCHIVED
        assert v2.stage == ModelStage.PRODUCTION

    def test_delete_model_version(self, tmp_path):
        """모델 버전 삭제"""
        from src.training.mlflow_utils import ModelRegistry, PyTorchModelSaver

        registry = ModelRegistry(tmp_path / 'registry')

        model = SimpleModel()
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(model, model_path)

        registry.register_model('my_model', 'run_1', model_path)
        result = registry.delete_model_version('my_model', 1)

        assert result is True
        assert registry.get_model_version('my_model', 1) is None


class TestMetricLogger:
    """MetricLogger 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.mlflow_utils import MetricLogger

        logger = MetricLogger()

        assert logger._step == 0

    def test_log(self):
        """메트릭 로깅"""
        from src.training.mlflow_utils import MetricLogger

        logger = MetricLogger()

        logger.log({'loss': 0.5, 'accuracy': 0.8})
        logger.log({'loss': 0.3, 'accuracy': 0.9})

        assert len(logger.get_history('loss')) == 2
        assert len(logger.get_history('accuracy')) == 2

    def test_get_best_minimize(self):
        """최소값 조회"""
        from src.training.mlflow_utils import MetricLogger, MetricGoal

        logger = MetricLogger()

        logger.log({'loss': 0.5})
        logger.log({'loss': 0.3})
        logger.log({'loss': 0.4})

        best = logger.get_best('loss', MetricGoal.MINIMIZE)

        assert best == 0.3

    def test_get_best_maximize(self):
        """최대값 조회"""
        from src.training.mlflow_utils import MetricLogger, MetricGoal

        logger = MetricLogger()

        logger.log({'accuracy': 0.7})
        logger.log({'accuracy': 0.9})
        logger.log({'accuracy': 0.8})

        best = logger.get_best('accuracy', MetricGoal.MAXIMIZE)

        assert best == 0.9

    def test_to_dataframe(self):
        """DataFrame 변환"""
        from src.training.mlflow_utils import MetricLogger

        logger = MetricLogger()

        for i in range(5):
            logger.log({'loss': 0.5 - i * 0.1, 'accuracy': 0.5 + i * 0.1})

        df = logger.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'loss' in df.columns
        assert 'accuracy' in df.columns
        assert len(df) == 5

    def test_reset(self):
        """초기화"""
        from src.training.mlflow_utils import MetricLogger

        logger = MetricLogger()

        logger.log({'loss': 0.5})
        logger.reset()

        assert len(logger.get_history('loss')) == 0
        assert logger._step == 0

    def test_with_tracker(self, tmp_path):
        """트래커 연동"""
        from src.training.mlflow_utils import MetricLogger, ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('test_exp')

        logger = MetricLogger(tracker)
        logger.log({'loss': 0.5})

        assert 'loss' in run.metrics


class TestPyTorchModelSaver:
    """PyTorchModelSaver 테스트"""

    def test_save(self, tmp_path):
        """모델 저장"""
        from src.training.mlflow_utils import PyTorchModelSaver

        model = SimpleModel()
        path = tmp_path / 'model'

        PyTorchModelSaver.save(
            model, path,
            epoch=10,
            metrics={'loss': 0.5},
            config={'hidden_size': 32}
        )

        assert (path / 'model.pt').exists()
        assert (path / 'meta.json').exists()

    def test_save_with_optimizer(self, tmp_path):
        """옵티마이저와 함께 저장"""
        from src.training.mlflow_utils import PyTorchModelSaver

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        path = tmp_path / 'model'

        PyTorchModelSaver.save(model, path, optimizer=optimizer)

        checkpoint = torch.load(path / 'model.pt', weights_only=False)
        assert 'optimizer_state_dict' in checkpoint

    def test_load(self, tmp_path):
        """모델 로드"""
        from src.training.mlflow_utils import PyTorchModelSaver

        # 저장
        model = SimpleModel()
        path = tmp_path / 'model'
        PyTorchModelSaver.save(model, path, epoch=10, metrics={'loss': 0.5})

        # 로드
        new_model = SimpleModel()
        info = PyTorchModelSaver.load(path, new_model)

        assert info['epoch'] == 10
        assert info['metrics']['loss'] == 0.5

    def test_load_with_optimizer(self, tmp_path):
        """옵티마이저와 함께 로드"""
        from src.training.mlflow_utils import PyTorchModelSaver

        # 저장
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        path = tmp_path / 'model'
        PyTorchModelSaver.save(model, path, optimizer=optimizer)

        # 로드
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        PyTorchModelSaver.load(path, new_model, new_optimizer)

        # 옵티마이저 상태 확인
        assert new_optimizer.state_dict() is not None


class TestTrainingCallbacks:
    """TrainingCallback 테스트"""

    def test_mlflow_callback(self, tmp_path):
        """MLflow 콜백"""
        from src.training.mlflow_utils import MLflowCallback, ExperimentTracker

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        run = tracker.start_run('test_exp')

        callback = MLflowCallback(tracker)
        callback.on_epoch_end(0, {'loss': 0.5, 'accuracy': 0.8})

        assert 'loss' in run.metrics
        assert 'accuracy' in run.metrics

    def test_early_stopping_callback_no_improvement(self):
        """조기 종료 - 개선 없음"""
        from src.training.mlflow_utils import EarlyStoppingCallback

        callback = EarlyStoppingCallback(monitor='val_loss', patience=3)

        # 개선 없는 에포크들
        for i in range(5):
            callback.on_epoch_end(i, {'val_loss': 1.0})

        assert callback.should_stop is True
        assert callback.counter >= 3

    def test_early_stopping_callback_with_improvement(self):
        """조기 종료 - 개선 있음"""
        from src.training.mlflow_utils import EarlyStoppingCallback

        callback = EarlyStoppingCallback(monitor='val_loss', patience=3)

        # 지속적 개선
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for i, loss in enumerate(losses):
            callback.on_epoch_end(i, {'val_loss': loss})

        assert callback.should_stop is False
        assert callback.counter == 0


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_experiment_tracker(self, tmp_path):
        """실험 추적기 생성"""
        from src.training.mlflow_utils import create_experiment_tracker

        tracker = create_experiment_tracker(tmp_path / 'mlruns')

        assert tracker is not None

    def test_create_model_registry(self, tmp_path):
        """모델 레지스트리 생성"""
        from src.training.mlflow_utils import create_model_registry

        registry = create_model_registry(tmp_path / 'registry')

        assert registry is not None

    def test_create_metric_logger(self):
        """메트릭 로거 생성"""
        from src.training.mlflow_utils import create_metric_logger

        logger = create_metric_logger()

        assert logger is not None


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self, tmp_path):
        """전체 워크플로우"""
        from src.training.mlflow_utils import (
            ExperimentTracker, ModelRegistry, MetricLogger,
            PyTorchModelSaver, ModelStage
        )

        # 1. 설정
        tracker = ExperimentTracker(tmp_path / 'mlruns')
        registry = ModelRegistry(tmp_path / 'registry')
        logger = MetricLogger(tracker)

        # 2. 실험 시작
        run = tracker.start_run('power_forecast', run_name='lstm_v1')

        # 3. 파라미터 로깅
        tracker.log_params({
            'model_type': 'LSTM',
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001
        })

        # 4. 학습 시뮬레이션
        model = SimpleModel()
        for epoch in range(5):
            loss = 1.0 - epoch * 0.15
            accuracy = 0.5 + epoch * 0.1
            logger.log({'loss': loss, 'accuracy': accuracy})

        # 5. 모델 저장 및 등록
        model_path = tmp_path / 'model'
        PyTorchModelSaver.save(
            model, model_path,
            epoch=5,
            metrics={'loss': logger.get_best('loss'), 'accuracy': logger.get_best('accuracy')}
        )

        version = registry.register_model(
            name='power_forecast_lstm',
            run_id=run.run_id,
            model_path=model_path,
            metrics={'loss': logger.get_best('loss')}
        )

        # 6. 스테이지 전환
        registry.transition_stage('power_forecast_lstm', version.version, ModelStage.STAGING)
        registry.transition_stage('power_forecast_lstm', version.version, ModelStage.PRODUCTION)

        # 7. 실행 종료
        tracker.end_run()

        # 검증
        prod_model = registry.get_latest_version('power_forecast_lstm', ModelStage.PRODUCTION)
        assert prod_model is not None
        assert prod_model.stage == ModelStage.PRODUCTION

    def test_model_comparison(self, tmp_path):
        """모델 비교"""
        from src.training.mlflow_utils import (
            ExperimentTracker, ModelRegistry, PyTorchModelSaver
        )

        tracker = ExperimentTracker(tmp_path / 'mlruns')
        registry = ModelRegistry(tmp_path / 'registry')

        models_metrics = []

        # 여러 모델 학습
        for i in range(3):
            run = tracker.start_run('comparison_exp', run_name=f'model_{i}')

            tracker.log_params({'model_id': i})

            # 각 모델마다 다른 성능
            metrics = {'rmse': 0.5 - i * 0.1, 'mae': 0.3 - i * 0.05}
            tracker.log_metrics(metrics)

            model = SimpleModel(hidden_size=32 + i * 16)
            model_path = tmp_path / f'model_{i}'
            PyTorchModelSaver.save(model, model_path, metrics=metrics)

            version = registry.register_model(
                name='comparison_model',
                run_id=run.run_id,
                model_path=model_path,
                metrics=metrics
            )

            models_metrics.append(metrics)
            tracker.end_run()

        # 최적 모델 찾기
        versions = registry.list_model_versions('comparison_model')
        best_version = min(versions, key=lambda v: v.metrics.get('rmse', float('inf')))

        assert best_version.version == 3  # 마지막 모델이 가장 좋음

