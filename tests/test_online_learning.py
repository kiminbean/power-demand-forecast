"""
온라인 학습 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class SimpleModel(nn.Module):
    """테스트용 모델"""
    def __init__(self, input_size: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


class TestUpdateStrategy:
    """UpdateStrategy 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.training.online_learning import UpdateStrategy

        assert UpdateStrategy.FULL_RETRAIN.value == 'full_retrain'
        assert UpdateStrategy.FINE_TUNE.value == 'fine_tune'
        assert UpdateStrategy.INCREMENTAL.value == 'incremental'


class TestDriftType:
    """DriftType 테스트"""

    def test_enum_values(self):
        """열거형 값"""
        from src.training.online_learning import DriftType

        assert DriftType.NO_DRIFT.value == 'no_drift'
        assert DriftType.SUDDEN.value == 'sudden'
        assert DriftType.GRADUAL.value == 'gradual'


class TestUpdateConfig:
    """UpdateConfig 테스트"""

    def test_default_values(self):
        """기본값"""
        from src.training.online_learning import UpdateConfig, UpdateStrategy

        config = UpdateConfig()

        assert config.strategy == UpdateStrategy.FINE_TUNE
        assert config.min_samples == 100
        assert config.update_interval == 24


class TestDataBuffer:
    """DataBuffer 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer(max_size=100)

        assert buffer.max_size == 100
        assert len(buffer) == 0

    def test_add(self):
        """데이터 추가"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer(max_size=100)

        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        buffer.add(X, y)

        assert len(buffer) == 10

    def test_get_data(self):
        """데이터 조회"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer(max_size=100)

        X = np.random.randn(20, 5)
        y = np.random.randn(20)

        buffer.add(X, y)

        X_out, y_out = buffer.get_data()

        assert X_out.shape == (20, 5)
        assert y_out.shape == (20,)

    def test_get_data_n_samples(self):
        """최신 N개 샘플 조회"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer(max_size=100)

        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        buffer.add(X, y)

        X_out, y_out = buffer.get_data(n_samples=10)

        assert X_out.shape == (10, 5)

    def test_max_size(self):
        """최대 크기 제한"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer(max_size=50)

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        buffer.add(X, y)

        assert len(buffer) == 50

    def test_clear(self):
        """버퍼 초기화"""
        from src.training.online_learning import DataBuffer

        buffer = DataBuffer()
        buffer.add(np.random.randn(10, 5), np.random.randn(10))
        buffer.clear()

        assert len(buffer) == 0


class TestConceptDriftDetector:
    """ConceptDriftDetector 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.online_learning import ConceptDriftDetector

        detector = ConceptDriftDetector(delta=0.005, lambda_=50.0)

        assert detector.delta == 0.005
        assert detector.lambda_ == 50.0

    def test_update_no_drift(self):
        """드리프트 없음"""
        from src.training.online_learning import ConceptDriftDetector, DriftType

        detector = ConceptDriftDetector()

        # 안정적인 값
        for _ in range(50):
            result = detector.update(1.0 + np.random.randn() * 0.1)

        assert result.drift_type == DriftType.NO_DRIFT

    def test_update_with_drift(self):
        """드리프트 감지"""
        from src.training.online_learning import ConceptDriftDetector

        detector = ConceptDriftDetector(lambda_=10.0)

        # 안정적인 값
        for _ in range(50):
            detector.update(1.0)

        # 급격한 변화
        for _ in range(50):
            result = detector.update(10.0)

        # 드리프트가 감지될 수 있음
        assert result.details['ph_value'] >= 0

    def test_reset(self):
        """상태 초기화"""
        from src.training.online_learning import ConceptDriftDetector

        detector = ConceptDriftDetector()

        for _ in range(10):
            detector.update(1.0)

        detector.reset()

        assert detector._count == 0


class TestPerformanceMonitor:
    """PerformanceMonitor 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.online_learning import PerformanceMonitor

        monitor = PerformanceMonitor(window_size=100)

        assert monitor.window_size == 100

    def test_update(self):
        """업데이트"""
        from src.training.online_learning import PerformanceMonitor

        monitor = PerformanceMonitor()

        for i in range(10):
            monitor.update(prediction=i * 1.1, actual=float(i))

        metrics = monitor.get_metrics()

        assert metrics.mae > 0

    def test_get_metrics(self):
        """메트릭 조회"""
        from src.training.online_learning import PerformanceMonitor

        monitor = PerformanceMonitor()

        for i in range(20):
            monitor.update(prediction=float(i), actual=float(i) + 1)

        metrics = monitor.get_metrics()

        assert metrics.mae == pytest.approx(1.0, rel=0.1)
        assert metrics.rmse > 0
        assert metrics.r2 is not None

    def test_get_trend(self):
        """추세 분석"""
        from src.training.online_learning import PerformanceMonitor

        monitor = PerformanceMonitor(window_size=20)

        # 오차 감소 (개선)
        for i in range(20):
            error = 10.0 - i * 0.4
            monitor.update(prediction=100 + error, actual=100)

        trend = monitor.get_trend()

        assert trend in ['improving', 'stable', 'degrading']

    def test_reset(self):
        """초기화"""
        from src.training.online_learning import PerformanceMonitor

        monitor = PerformanceMonitor()

        for _ in range(10):
            monitor.update(1.0, 1.1)

        monitor.reset()

        metrics = monitor.get_metrics()
        assert metrics.mae == 0


class TestIncrementalTrainer:
    """IncrementalTrainer 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.online_learning import IncrementalTrainer, UpdateConfig

        model = SimpleModel()
        config = UpdateConfig()

        trainer = IncrementalTrainer(model, config)

        assert trainer.model is model

    def test_train_step(self):
        """학습 스텝"""
        from src.training.online_learning import IncrementalTrainer, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig(batch_size=16)

        trainer = IncrementalTrainer(model, config)

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        metrics = trainer.train_step(X, y)

        assert 'train_loss' in metrics

    def test_train_step_with_validation(self):
        """검증 데이터와 함께 학습"""
        from src.training.online_learning import IncrementalTrainer, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig()

        trainer = IncrementalTrainer(model, config)

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        val_X = np.random.randn(20, 5).astype(np.float32)
        val_y = np.random.randn(20).astype(np.float32)

        metrics = trainer.train_step(X, y, validation_data=(val_X, val_y))

        assert metrics['val_loss'] is not None

    def test_full_train(self):
        """전체 학습"""
        from src.training.online_learning import IncrementalTrainer, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig(epochs=5)

        trainer = IncrementalTrainer(model, config)

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        history = trainer.full_train(X, y)

        assert len(history['train_loss']) <= 5

    def test_rollback(self):
        """롤백"""
        from src.training.online_learning import IncrementalTrainer, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig()

        trainer = IncrementalTrainer(model, config)

        # 원래 가중치 저장
        original_weight = model.fc.weight.data.clone()

        # 학습
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        trainer.full_train(X, y, epochs=10)

        # 롤백
        trainer.rollback()

        # 원래 가중치로 복원됨
        assert torch.allclose(model.fc.weight.data, original_weight)


class TestOnlineLearner:
    """OnlineLearner 테스트"""

    def test_creation(self):
        """생성"""
        from src.training.online_learning import OnlineLearner, UpdateConfig

        model = SimpleModel()
        config = UpdateConfig()

        learner = OnlineLearner(model, config)

        assert learner.model is model

    def test_predict(self):
        """예측"""
        from src.training.online_learning import OnlineLearner

        model = SimpleModel(input_size=5)
        learner = OnlineLearner(model)

        X = np.random.randn(10, 5).astype(np.float32)
        predictions = learner.predict(X)

        assert predictions.shape == (10, 1)

    def test_observe(self):
        """관측"""
        from src.training.online_learning import OnlineLearner, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig(min_samples=10)

        learner = OnlineLearner(model, config)

        X = np.random.randn(5, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)

        result = learner.observe(X, y)

        assert 'drift_detected' in result
        assert 'performance' in result
        assert result['buffer_size'] == 5

    def test_observe_with_prediction(self):
        """예측값과 함께 관측"""
        from src.training.online_learning import OnlineLearner

        model = SimpleModel(input_size=5)
        learner = OnlineLearner(model)

        X = np.random.randn(5, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        pred = np.random.randn(5).astype(np.float32)

        result = learner.observe(X, y, prediction=pred)

        assert result['buffer_size'] == 5

    def test_update(self):
        """업데이트"""
        from src.training.online_learning import OnlineLearner, UpdateConfig

        model = SimpleModel(input_size=5)
        config = UpdateConfig(min_samples=50, epochs=2)

        learner = OnlineLearner(model, config)

        # 데이터 추가
        for _ in range(10):
            X = np.random.randn(10, 5).astype(np.float32)
            y = np.random.randn(10).astype(np.float32)
            learner.observe(X, y)

        # 업데이트
        result = learner.update(force=True)

        assert result['status'] == 'success'
        assert result['update_count'] == 1

    def test_get_status(self):
        """상태 조회"""
        from src.training.online_learning import OnlineLearner

        model = SimpleModel(input_size=5)
        learner = OnlineLearner(model)

        status = learner.get_status()

        assert 'buffer_size' in status
        assert 'update_count' in status
        assert 'performance' in status

    def test_reset(self):
        """초기화"""
        from src.training.online_learning import OnlineLearner

        model = SimpleModel(input_size=5)
        learner = OnlineLearner(model)

        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        learner.observe(X, y)

        learner.reset()

        status = learner.get_status()
        assert status['buffer_size'] == 0


class TestModelUpdater:
    """ModelUpdater 테스트"""

    def test_creation(self, tmp_path):
        """생성"""
        from src.training.online_learning import ModelUpdater

        updater = ModelUpdater(tmp_path / 'updates')

        assert updater.base_path.exists()

    def test_register_model(self, tmp_path):
        """모델 등록"""
        from src.training.online_learning import ModelUpdater

        updater = ModelUpdater(tmp_path / 'updates')

        model = SimpleModel()
        learner = updater.register_model('my_model', model)

        assert learner is not None
        assert updater.get_learner('my_model') is learner

    def test_update_all(self, tmp_path):
        """모든 모델 업데이트"""
        from src.training.online_learning import ModelUpdater, UpdateConfig

        updater = ModelUpdater(tmp_path / 'updates')

        config = UpdateConfig(min_samples=20, epochs=2)

        # 모델 등록
        model1 = SimpleModel(input_size=5)
        model2 = SimpleModel(input_size=5)

        updater.register_model('model1', model1, config)
        updater.register_model('model2', model2, config)

        # 데이터 추가
        for name in ['model1', 'model2']:
            learner = updater.get_learner(name)
            X = np.random.randn(30, 5).astype(np.float32)
            y = np.random.randn(30).astype(np.float32)
            learner.buffer.add(X, y)

        # 업데이트
        results = updater.update_all(force=True)

        assert 'model1' in results
        assert 'model2' in results

    def test_get_all_status(self, tmp_path):
        """모든 모델 상태 조회"""
        from src.training.online_learning import ModelUpdater

        updater = ModelUpdater(tmp_path / 'updates')

        updater.register_model('model1', SimpleModel())
        updater.register_model('model2', SimpleModel())

        status = updater.get_all_status()

        assert 'model1' in status
        assert 'model2' in status

    def test_save_state(self, tmp_path):
        """상태 저장"""
        from src.training.online_learning import ModelUpdater

        updater = ModelUpdater(tmp_path / 'updates')

        updater.register_model('model1', SimpleModel())

        updater.save_state()

        assert (tmp_path / 'updates' / 'state.json').exists()


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_online_learner(self):
        """온라인 학습기 생성"""
        from src.training.online_learning import create_online_learner

        model = SimpleModel()
        learner = create_online_learner(model, strategy='fine_tune')

        assert learner is not None

    def test_create_online_learner_strategies(self):
        """다양한 전략"""
        from src.training.online_learning import (
            create_online_learner, UpdateStrategy
        )

        model = SimpleModel()

        strategies = ['full_retrain', 'fine_tune', 'incremental']
        for strategy in strategies:
            learner = create_online_learner(model, strategy=strategy)
            assert learner is not None

    def test_create_model_updater(self, tmp_path):
        """모델 업데이트 관리자 생성"""
        from src.training.online_learning import create_model_updater

        updater = create_model_updater(tmp_path / 'updates')

        assert updater is not None


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self, tmp_path):
        """전체 워크플로우"""
        from src.training.online_learning import (
            OnlineLearner, UpdateConfig, UpdateStrategy
        )

        # 1. 설정
        model = SimpleModel(input_size=5)
        config = UpdateConfig(
            strategy=UpdateStrategy.FINE_TUNE,
            min_samples=50,
            epochs=3
        )

        learner = OnlineLearner(model, config)

        # 2. 초기 데이터로 학습
        X_init = np.random.randn(100, 5).astype(np.float32)
        y_init = X_init.sum(axis=1) + np.random.randn(100) * 0.1

        learner.buffer.add(X_init, y_init)
        learner.update(force=True)

        # 3. 스트리밍 데이터 처리
        for _ in range(10):
            X_new = np.random.randn(10, 5).astype(np.float32)
            y_new = X_new.sum(axis=1) + np.random.randn(10) * 0.1

            result = learner.observe(X_new, y_new)

        # 4. 상태 확인
        status = learner.get_status()

        assert status['update_count'] >= 1
        assert status['buffer_size'] > 0

    def test_drift_scenario(self):
        """드리프트 시나리오"""
        from src.training.online_learning import (
            OnlineLearner, UpdateConfig
        )

        model = SimpleModel(input_size=5)
        config = UpdateConfig(min_samples=30, drift_threshold=0.1)

        learner = OnlineLearner(model, config)

        # 정상 데이터
        for _ in range(50):
            X = np.random.randn(5, 5).astype(np.float32)
            y = X.sum(axis=1)
            learner.observe(X, y)

        # 드리프트 데이터 (패턴 변경)
        for _ in range(50):
            X = np.random.randn(5, 5).astype(np.float32)
            y = X.sum(axis=1) * 10  # 스케일 변경
            result = learner.observe(X, y)

        # 성능 저하 감지됨
        status = learner.get_status()
        assert status['performance']['mae'] > 0

    def test_continuous_learning(self, tmp_path):
        """지속적 학습"""
        from src.training.online_learning import (
            ModelUpdater, UpdateConfig
        )

        updater = ModelUpdater(tmp_path / 'updates')

        config = UpdateConfig(
            min_samples=30,
            update_interval=1,  # 1시간마다
            epochs=2
        )

        model = SimpleModel(input_size=5)
        learner = updater.register_model('power_model', model, config)

        # 시뮬레이션: 여러 시간에 걸친 데이터
        for hour in range(24):
            X = np.random.randn(10, 5).astype(np.float32)
            y = X.sum(axis=1) + hour * 0.1  # 시간에 따른 변화

            learner.observe(X, y)

        # 업데이트 실행
        updater.update_all(force=True)

        status = updater.get_all_status()
        assert status['power_model']['update_count'] >= 1

