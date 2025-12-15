"""
MODEL-003: LSTM 학습 파이프라인
===============================

LSTM 모델 학습을 위한 Trainer 클래스 및 콜백

주요 기능:
1. Early Stopping
2. Learning Rate Scheduler (ReduceLROnPlateau, CosineAnnealing)
3. Gradient Clipping
4. 체크포인트 저장/로드
5. 학습 히스토리 기록

하이퍼파라미터 (feature_list.json 기준):
- learning_rate: 0.001
- epochs: 100
- patience: 15

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Callable, Tuple, Any
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np


# ============================================================
# Early Stopping 콜백
# ============================================================

class EarlyStopping:
    """
    Early Stopping 콜백

    검증 손실이 개선되지 않으면 학습을 조기 종료

    Args:
        patience: 개선 없이 허용되는 에포크 수
        min_delta: 개선으로 인정되는 최소 변화량
        mode: 'min' (손실 감소) 또는 'max' (지표 증가)
        restore_best_weights: 최적 가중치 복원 여부

    Example:
        >>> early_stopping = EarlyStopping(patience=15)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch(...)
        ...     early_stopping(val_loss, model)
        ...     if early_stopping.early_stop:
        ...         break
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

        if mode == 'min':
            self.is_improvement = lambda current, best: current < best - min_delta
        else:
            self.is_improvement = lambda current, best: current > best + min_delta

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Early Stopping 체크

        Args:
            score: 현재 검증 점수
            model: 현재 모델

        Returns:
            bool: 개선 여부
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True

        if self.is_improvement(score, self.best_score):
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def restore_best(self, model: nn.Module) -> None:
        """최적 가중치 복원"""
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    def reset(self) -> None:
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None


# ============================================================
# 학습 히스토리
# ============================================================

class TrainingHistory:
    """
    학습 히스토리 기록 및 관리

    학습 과정의 손실, 지표, 학습률 등을 기록

    Example:
        >>> history = TrainingHistory()
        >>> history.update(epoch=1, train_loss=0.5, val_loss=0.4, lr=0.001)
        >>> history.save('training_history.json')
    """

    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'best_epoch': None,
            'best_val_loss': float('inf'),
            'training_time': [],
            'metrics': {}
        }
        self.start_time = None

    def start_epoch(self) -> None:
        """에포크 시작 시간 기록"""
        self.start_time = time.time()

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        metrics: Dict[str, float] = None
    ) -> None:
        """
        히스토리 업데이트

        Args:
            epoch: 현재 에포크
            train_loss: 학습 손실
            val_loss: 검증 손실
            lr: 현재 학습률
            metrics: 추가 평가 지표
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)

        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.history['training_time'].append(elapsed)

        if val_loss < self.history['best_val_loss']:
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch

        if metrics:
            for key, value in metrics.items():
                if key not in self.history['metrics']:
                    self.history['metrics'][key] = []
                self.history['metrics'][key].append(value)

    def get_best(self) -> Dict[str, Any]:
        """최적 에포크 정보 반환"""
        return {
            'epoch': self.history['best_epoch'],
            'val_loss': self.history['best_val_loss']
        }

    def save(self, filepath: str) -> None:
        """히스토리를 JSON 파일로 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, filepath: str) -> None:
        """JSON 파일에서 히스토리 로드"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


# ============================================================
# Trainer 클래스
# ============================================================

class Trainer:
    """
    LSTM 모델 학습 Trainer

    학습, 검증, 테스트의 전체 파이프라인을 관리

    Args:
        model: PyTorch 모델
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 학습 디바이스
        scheduler: 학습률 스케줄러 (optional)
        grad_clip: Gradient clipping 값 (optional)
        checkpoint_dir: 체크포인트 저장 디렉토리

    Example:
        >>> trainer = Trainer(model, criterion, optimizer, device)
        >>> history = trainer.fit(train_loader, val_loader, epochs=100, patience=15)
        >>> test_loss = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        grad_clip: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """단일 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """단일 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        verbose: int = 1,
        log_interval: int = 10,
        callbacks: List[Callable] = None
    ) -> TrainingHistory:
        """
        모델 학습

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            epochs: 최대 에포크 수
            patience: Early stopping patience
            verbose: 출력 레벨 (0: silent, 1: progress, 2: detailed)
            log_interval: 로그 출력 간격
            callbacks: 추가 콜백 함수 리스트

        Returns:
            TrainingHistory: 학습 히스토리
        """
        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=patience, mode='min')

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Epochs: {epochs}, Patience: {patience}")
            print(f"Device: {self.device}")
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            history.start_epoch()

            # Training
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_loss = self._validate_epoch(val_loader)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            history.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=current_lr
            )

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            improved = early_stopping(val_loss, self.model)

            # Logging
            if verbose >= 1 and (epoch % log_interval == 0 or epoch == 1 or early_stopping.early_stop):
                status = "✓" if improved else ""
                print(f"Epoch {epoch:4d} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"ES: {early_stopping.counter}/{patience} {status}")

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_loss, val_loss, self.model)

            # Early stopping
            if early_stopping.early_stop:
                if verbose >= 1:
                    print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                break

        # Restore best weights
        early_stopping.restore_best(self.model)

        if verbose >= 1:
            best = history.get_best()
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Best epoch: {best['epoch']} (val_loss: {best['val_loss']:.6f})")
            print(f"{'='*60}")

        return history

    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            test_loader: 테스트 데이터 로더
            return_predictions: 예측값 반환 여부

        Returns:
            Dict: 평가 결과 (loss, predictions, actuals)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                n_batches += 1

                if return_predictions:
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())

        result = {
            'test_loss': total_loss / n_batches
        }

        if return_predictions:
            result['predictions'] = np.array(predictions)
            result['actuals'] = np.array(actuals)

        return result

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        예측 수행

        Args:
            data_loader: 데이터 로더

        Returns:
            np.ndarray: 예측값
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        history: TrainingHistory = None,
        additional_info: Dict = None
    ) -> None:
        """
        체크포인트 저장

        Args:
            filepath: 저장 경로
            epoch: 현재 에포크
            history: 학습 히스토리
            additional_info: 추가 정보
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if history:
            checkpoint['history'] = history.history

        if additional_info:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict:
        """
        체크포인트 로드

        Args:
            filepath: 체크포인트 경로

        Returns:
            Dict: 체크포인트 정보
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint


# ============================================================
# 학습률 스케줄러 생성 함수
# ============================================================

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'plateau',
    **kwargs
) -> _LRScheduler:
    """
    학습률 스케줄러 생성

    Args:
        optimizer: 옵티마이저
        scheduler_type: 스케줄러 타입
            - 'plateau': ReduceLROnPlateau
            - 'cosine': CosineAnnealingLR
            - 'step': StepLR
            - 'exponential': ExponentialLR
        **kwargs: 스케줄러별 추가 인자

    Returns:
        _LRScheduler: 학습률 스케줄러
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6)
        )

    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )

    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )

    elif scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        total_epochs = kwargs.get('total_epochs', 100)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================
# 평가 지표 계산
# ============================================================

def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    scaler=None,
    target_idx: int = 0
) -> Dict[str, float]:
    """
    평가 지표 계산

    Args:
        predictions: 예측값
        actuals: 실제값
        scaler: 역스케일링용 스케일러 (optional)
        target_idx: 타겟 인덱스

    Returns:
        Dict[str, float]: 평가 지표
    """
    # 역스케일링 (필요한 경우)
    if scaler is not None:
        predictions = scaler.inverse_transform_target(predictions, target_idx)
        actuals = scaler.inverse_transform_target(actuals, target_idx)

    # Flatten
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # MSE
    mse = np.mean((predictions - actuals) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # MAPE (0 값 처리)
    non_zero_mask = actuals != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
    else:
        mape = float('inf')

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
