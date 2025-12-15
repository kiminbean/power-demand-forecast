"""
MODEL-003: LSTM 모델 테스트
===========================

LSTM 모델 및 학습 파이프라인 단위 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys

# 프로젝트 루트 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm import (
    LSTMModel,
    MultiHorizonLSTM,
    ResidualLSTM,
    create_model,
    model_summary
)
from training.trainer import (
    Trainer,
    EarlyStopping,
    TrainingHistory,
    create_scheduler,
    compute_metrics
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
def sample_batch():
    """테스트용 배치 데이터"""
    batch_size = 32
    seq_length = 48
    n_features = 38

    X = torch.randn(batch_size, seq_length, n_features)
    y = torch.randn(batch_size, 1)
    return X, y


@pytest.fixture
def sample_multi_horizon_batch():
    """테스트용 multi-horizon 배치 데이터"""
    batch_size = 32
    seq_length = 48
    n_features = 38
    n_horizons = 4

    X = torch.randn(batch_size, seq_length, n_features)
    y = torch.randn(batch_size, n_horizons)
    return X, y


# ============================================================
# LSTMModel 테스트
# ============================================================

class TestLSTMModel:
    """LSTMModel 클래스 테스트"""

    def test_model_creation(self):
        """모델 생성 테스트"""
        model = LSTMModel(input_size=38, hidden_size=64, num_layers=2)
        assert model is not None
        assert model.input_size == 38
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_forward_shape(self, sample_batch):
        """순전파 출력 shape 테스트"""
        X, _ = sample_batch
        model = LSTMModel(input_size=38, hidden_size=64)
        output = model(X)

        assert output.shape == (32, 1)

    def test_forward_dtype(self, sample_batch):
        """순전파 출력 dtype 테스트"""
        X, _ = sample_batch
        model = LSTMModel(input_size=38)
        output = model(X)

        assert output.dtype == torch.float32

    def test_bidirectional(self, sample_batch):
        """양방향 LSTM 테스트"""
        X, _ = sample_batch
        model = LSTMModel(input_size=38, hidden_size=64, bidirectional=True)
        output = model(X)

        assert output.shape == (32, 1)
        # BiLSTM은 FC 입력이 hidden_size * 2
        assert model.lstm.bidirectional == True

    def test_multi_output(self, sample_batch):
        """다중 출력 테스트"""
        X, _ = sample_batch
        model = LSTMModel(input_size=38, output_size=4)
        output = model(X)

        assert output.shape == (32, 4)

    def test_dropout(self):
        """Dropout 적용 테스트"""
        model = LSTMModel(input_size=38, dropout=0.5)
        assert model.dropout == 0.5

    def test_num_parameters(self):
        """파라미터 수 계산 테스트"""
        model = LSTMModel(input_size=38, hidden_size=64, num_layers=2)
        n_params = model.get_num_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_device_move(self, sample_batch, device):
        """디바이스 이동 테스트"""
        X, _ = sample_batch
        model = LSTMModel(input_size=38).to(device)
        X = X.to(device)

        output = model(X)
        assert output.device.type == device.type


# ============================================================
# MultiHorizonLSTM 테스트
# ============================================================

class TestMultiHorizonLSTM:
    """MultiHorizonLSTM 클래스 테스트"""

    def test_model_creation(self):
        """모델 생성 테스트"""
        model = MultiHorizonLSTM(input_size=38, horizons=[1, 6, 12, 24])
        assert model is not None
        assert model.num_horizons == 4

    def test_forward_shape(self, sample_batch):
        """순전파 출력 shape 테스트"""
        X, _ = sample_batch
        horizons = [1, 6, 12, 24]
        model = MultiHorizonLSTM(input_size=38, horizons=horizons)
        output = model(X)

        assert output.shape == (32, len(horizons))

    def test_different_horizons(self, sample_batch):
        """다양한 horizon 설정 테스트"""
        X, _ = sample_batch

        for horizons in [[1], [1, 24], [1, 6, 12, 24, 48]]:
            model = MultiHorizonLSTM(input_size=38, horizons=horizons)
            output = model(X)
            assert output.shape == (32, len(horizons))

    def test_default_horizons(self, sample_batch):
        """기본 horizon 테스트"""
        X, _ = sample_batch
        model = MultiHorizonLSTM(input_size=38)
        output = model(X)

        # 기본값: [1, 6, 12, 24]
        assert output.shape == (32, 4)


# ============================================================
# ResidualLSTM 테스트
# ============================================================

class TestResidualLSTM:
    """ResidualLSTM 클래스 테스트"""

    def test_model_creation(self):
        """모델 생성 테스트"""
        model = ResidualLSTM(input_size=38, hidden_size=64, num_blocks=2)
        assert model is not None

    def test_forward_shape(self, sample_batch):
        """순전파 출력 shape 테스트"""
        X, _ = sample_batch
        model = ResidualLSTM(input_size=38)
        output = model(X)

        assert output.shape == (32, 1)


# ============================================================
# create_model 팩토리 테스트
# ============================================================

class TestCreateModel:
    """create_model 팩토리 함수 테스트"""

    def test_create_lstm(self, sample_batch):
        """LSTM 생성 테스트"""
        X, _ = sample_batch
        model = create_model('lstm', input_size=38)
        output = model(X)

        assert isinstance(model, LSTMModel)
        assert output.shape == (32, 1)

    def test_create_bilstm(self, sample_batch):
        """BiLSTM 생성 테스트"""
        X, _ = sample_batch
        model = create_model('bilstm', input_size=38)
        output = model(X)

        assert isinstance(model, LSTMModel)
        assert model.bidirectional == True

    def test_create_multi_horizon(self, sample_batch):
        """Multi-horizon 모델 생성 테스트"""
        X, _ = sample_batch
        model = create_model('multi_horizon', input_size=38, horizons=[1, 6, 12, 24])
        output = model(X)

        assert isinstance(model, MultiHorizonLSTM)
        assert output.shape == (32, 4)

    def test_create_residual(self, sample_batch):
        """Residual 모델 생성 테스트"""
        X, _ = sample_batch
        model = create_model('residual', input_size=38)
        output = model(X)

        assert isinstance(model, ResidualLSTM)

    def test_invalid_model_type(self):
        """잘못된 모델 타입 테스트"""
        with pytest.raises(ValueError):
            create_model('invalid_type', input_size=38)


# ============================================================
# EarlyStopping 테스트
# ============================================================

class TestEarlyStopping:
    """EarlyStopping 클래스 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        es = EarlyStopping(patience=10)
        assert es.patience == 10
        assert es.counter == 0
        assert es.early_stop == False

    def test_improvement_detection(self):
        """개선 감지 테스트"""
        es = EarlyStopping(patience=5, mode='min')
        model = LSTMModel(input_size=38)

        # 첫 번째 호출 (항상 개선)
        improved = es(0.5, model)
        assert improved == True
        assert es.counter == 0

        # 개선된 경우
        improved = es(0.3, model)
        assert improved == True
        assert es.counter == 0

        # 개선되지 않은 경우
        improved = es(0.4, model)
        assert improved == False
        assert es.counter == 1

    def test_early_stop_trigger(self):
        """Early stopping 트리거 테스트"""
        es = EarlyStopping(patience=3, mode='min')
        model = LSTMModel(input_size=38)

        es(0.5, model)  # best
        es(0.6, model)  # counter = 1
        es(0.7, model)  # counter = 2
        es(0.8, model)  # counter = 3, early_stop = True

        assert es.early_stop == True

    def test_best_weights_restore(self):
        """최적 가중치 복원 테스트"""
        es = EarlyStopping(patience=3, restore_best_weights=True)
        model = LSTMModel(input_size=38)

        # 가중치 변경 전 상태
        es(0.5, model)

        # 가중치 변경
        for param in model.parameters():
            param.data = torch.zeros_like(param.data)

        # 복원
        es.restore_best(model)

        # 가중치가 0이 아닌지 확인 (복원됨)
        first_param = next(model.parameters())
        assert not torch.all(first_param == 0)

    def test_max_mode(self):
        """max 모드 테스트"""
        es = EarlyStopping(patience=3, mode='max')
        model = LSTMModel(input_size=38)

        es(0.5, model)
        improved = es(0.6, model)  # 증가 = 개선
        assert improved == True

        improved = es(0.5, model)  # 감소 = 개선 아님
        assert improved == False

    def test_reset(self):
        """리셋 테스트"""
        es = EarlyStopping(patience=3)
        model = LSTMModel(input_size=38)

        es(0.5, model)
        es(0.6, model)
        es.reset()

        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop == False


# ============================================================
# TrainingHistory 테스트
# ============================================================

class TestTrainingHistory:
    """TrainingHistory 클래스 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        history = TrainingHistory()
        assert 'train_loss' in history.history
        assert 'val_loss' in history.history

    def test_update(self):
        """업데이트 테스트"""
        history = TrainingHistory()
        history.update(epoch=1, train_loss=0.5, val_loss=0.4, lr=0.001)

        assert history.history['epoch'] == [1]
        assert history.history['train_loss'] == [0.5]
        assert history.history['val_loss'] == [0.4]

    def test_best_tracking(self):
        """최적값 추적 테스트"""
        history = TrainingHistory()
        history.update(epoch=1, train_loss=0.5, val_loss=0.4, lr=0.001)
        history.update(epoch=2, train_loss=0.4, val_loss=0.3, lr=0.001)
        history.update(epoch=3, train_loss=0.3, val_loss=0.35, lr=0.001)

        best = history.get_best()
        assert best['epoch'] == 2
        assert best['val_loss'] == 0.3

    def test_save_load(self):
        """저장/로드 테스트"""
        history = TrainingHistory()
        history.update(epoch=1, train_loss=0.5, val_loss=0.4, lr=0.001)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            history.save(filepath)

            new_history = TrainingHistory()
            new_history.load(filepath)

            assert new_history.history['train_loss'] == [0.5]
        finally:
            os.unlink(filepath)


# ============================================================
# Scheduler 테스트
# ============================================================

class TestSchedulers:
    """학습률 스케줄러 테스트"""

    def test_plateau_scheduler(self):
        """ReduceLROnPlateau 스케줄러 테스트"""
        model = LSTMModel(input_size=38)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = create_scheduler(optimizer, 'plateau', patience=5)

        assert scheduler is not None
        # Step 호출 테스트
        scheduler.step(0.5)

    def test_cosine_scheduler(self):
        """CosineAnnealing 스케줄러 테스트"""
        model = LSTMModel(input_size=38)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = create_scheduler(optimizer, 'cosine', T_max=50)

        assert scheduler is not None
        initial_lr = optimizer.param_groups[0]['lr']

        # 몇 스텝 후 LR 변경 확인
        for _ in range(25):
            scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr != initial_lr

    def test_step_scheduler(self):
        """StepLR 스케줄러 테스트"""
        model = LSTMModel(input_size=38)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = create_scheduler(optimizer, 'step', step_size=10, gamma=0.1)

        assert scheduler is not None

    def test_invalid_scheduler(self):
        """잘못된 스케줄러 타입 테스트"""
        model = LSTMModel(input_size=38)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with pytest.raises(ValueError):
            create_scheduler(optimizer, 'invalid_scheduler')


# ============================================================
# Metrics 테스트
# ============================================================

class TestMetrics:
    """평가 지표 테스트"""

    def test_compute_metrics_basic(self):
        """기본 지표 계산 테스트"""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])

        metrics = compute_metrics(predictions, actuals)

        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics

    def test_perfect_predictions(self):
        """완벽한 예측 테스트"""
        values = np.array([100, 200, 300])
        metrics = compute_metrics(values, values)

        assert metrics['MSE'] == 0
        assert metrics['RMSE'] == 0
        assert metrics['MAE'] == 0
        assert metrics['MAPE'] == 0
        assert metrics['R2'] == 1.0

    def test_mape_zero_handling(self):
        """MAPE 0값 처리 테스트"""
        predictions = np.array([100, 200])
        actuals = np.array([0, 200])  # 0 포함

        # 0이 아닌 값만으로 MAPE 계산
        metrics = compute_metrics(predictions, actuals)
        assert metrics['MAPE'] != float('inf')


# ============================================================
# Trainer 테스트
# ============================================================

class TestTrainer:
    """Trainer 클래스 테스트"""

    @pytest.fixture
    def simple_loaders(self):
        """간단한 DataLoader"""
        from torch.utils.data import TensorDataset, DataLoader

        n_samples = 100
        seq_length = 48
        n_features = 38

        X = torch.randn(n_samples, seq_length, n_features)
        y = torch.randn(n_samples, 1)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        return loader, loader, loader  # train, val, test

    def test_trainer_creation(self, device):
        """Trainer 생성 테스트"""
        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)
        assert trainer is not None

    def test_single_epoch_train(self, simple_loaders, device):
        """단일 에포크 학습 테스트"""
        train_loader, _, _ = simple_loaders

        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)
        loss = trainer._train_epoch(train_loader)

        assert isinstance(loss, float)
        assert loss > 0

    def test_single_epoch_validate(self, simple_loaders, device):
        """단일 에포크 검증 테스트"""
        _, val_loader, _ = simple_loaders

        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)
        loss = trainer._validate_epoch(val_loader)

        assert isinstance(loss, float)
        assert loss > 0

    def test_fit_short(self, simple_loaders, device):
        """짧은 학습 테스트"""
        train_loader, val_loader, _ = simple_loaders

        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)
        history = trainer.fit(
            train_loader, val_loader,
            epochs=3, patience=10, verbose=0
        )

        assert len(history.history['train_loss']) == 3

    def test_evaluate(self, simple_loaders, device):
        """평가 테스트"""
        _, _, test_loader = simple_loaders

        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)
        result = trainer.evaluate(test_loader, return_predictions=True)

        assert 'test_loss' in result
        assert 'predictions' in result
        assert 'actuals' in result

    def test_checkpoint_save_load(self, simple_loaders, device):
        """체크포인트 저장/로드 테스트"""
        train_loader, val_loader, _ = simple_loaders

        model = LSTMModel(input_size=38)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, device)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            filepath = f.name

        try:
            trainer.save_checkpoint(filepath, epoch=1)

            # 새 trainer에서 로드
            new_model = LSTMModel(input_size=38)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            new_trainer = Trainer(new_model, criterion, new_optimizer, device)

            checkpoint = new_trainer.load_checkpoint(filepath)
            assert checkpoint['epoch'] == 1
        finally:
            os.unlink(filepath)


# ============================================================
# 통합 테스트
# ============================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline_single_horizon(self, device):
        """단일 horizon 전체 파이프라인 테스트"""
        from torch.utils.data import TensorDataset, DataLoader

        # 데이터 생성
        n_samples = 200
        seq_length = 48
        n_features = 38

        X_train = torch.randn(n_samples, seq_length, n_features)
        y_train = torch.randn(n_samples, 1)
        X_val = torch.randn(50, seq_length, n_features)
        y_val = torch.randn(50, 1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=32
        )

        # 모델 생성
        model = create_model('lstm', input_size=n_features, hidden_size=32)

        # Trainer 설정
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = create_scheduler(optimizer, 'plateau')

        trainer = Trainer(
            model, criterion, optimizer, device,
            scheduler=scheduler, grad_clip=1.0
        )

        # 학습
        history = trainer.fit(
            train_loader, val_loader,
            epochs=5, patience=3, verbose=0
        )

        # 검증
        assert len(history.history['train_loss']) > 0
        assert history.history['train_loss'][-1] < history.history['train_loss'][0]

    def test_full_pipeline_multi_horizon(self, device):
        """다중 horizon 전체 파이프라인 테스트"""
        from torch.utils.data import TensorDataset, DataLoader

        # 데이터 생성
        n_samples = 200
        seq_length = 48
        n_features = 38
        n_horizons = 4

        X_train = torch.randn(n_samples, seq_length, n_features)
        y_train = torch.randn(n_samples, n_horizons)
        X_val = torch.randn(50, seq_length, n_features)
        y_val = torch.randn(50, n_horizons)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=32
        )

        # 모델 생성
        model = create_model(
            'multi_horizon', input_size=n_features,
            hidden_size=32, horizons=[1, 6, 12, 24]
        )

        # Trainer 설정
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, criterion, optimizer, device)

        # 학습
        history = trainer.fit(
            train_loader, val_loader,
            epochs=5, patience=3, verbose=0
        )

        # 평가
        result = trainer.evaluate(val_loader, return_predictions=True)

        assert result['predictions'].shape[1] == n_horizons


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
