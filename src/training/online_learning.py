"""
온라인 학습 모듈 (Task 10)
========================
실시간 데이터를 활용한 모델 업데이트 및 적응형 학습을 구현합니다.

주요 컴포넌트:
- OnlineLearner: 온라인 학습 관리자
- IncrementalTrainer: 점진적 학습
- ConceptDriftDetector: 개념 드리프트 감지
- ModelUpdater: 모델 업데이트 관리
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from collections import deque
from enum import Enum
import json
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """업데이트 전략"""
    FULL_RETRAIN = 'full_retrain'        # 전체 재학습
    FINE_TUNE = 'fine_tune'               # 파인튜닝
    INCREMENTAL = 'incremental'           # 점진적 학습
    ENSEMBLE_UPDATE = 'ensemble_update'   # 앙상블 업데이트


class DriftType(Enum):
    """드리프트 유형"""
    NO_DRIFT = 'no_drift'
    GRADUAL = 'gradual'
    SUDDEN = 'sudden'
    RECURRING = 'recurring'


@dataclass
class DriftDetectionResult:
    """드리프트 감지 결과"""
    drift_detected: bool
    drift_type: DriftType
    confidence: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateConfig:
    """업데이트 설정"""
    strategy: UpdateStrategy = UpdateStrategy.FINE_TUNE
    min_samples: int = 100              # 최소 샘플 수
    update_interval: int = 24           # 업데이트 주기 (시간)
    learning_rate: float = 0.0001       # 파인튜닝 학습률
    epochs: int = 5                     # 파인튜닝 에포크
    batch_size: int = 32                # 배치 크기
    drift_threshold: float = 0.05       # 드리프트 임계값
    max_buffer_size: int = 10000        # 최대 버퍼 크기
    validation_split: float = 0.2       # 검증 데이터 비율
    early_stopping_patience: int = 3    # 조기 종료 인내심


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    mae: float
    rmse: float
    mape: float
    r2: float
    timestamp: datetime = field(default_factory=datetime.now)


class DataBuffer:
    """
    데이터 버퍼

    새로운 데이터를 저장하고 관리합니다.
    """

    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: 최대 버퍼 크기
        """
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)

    def add(self, X: np.ndarray, y: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """
        데이터 추가

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            timestamp: 타임스탬프
        """
        if timestamp is None:
            timestamp = datetime.now()

        for i in range(len(X)):
            self._buffer.append((X[i], y[i] if len(y.shape) > 0 else y))
            self._timestamps.append(timestamp)

    def get_data(
        self,
        n_samples: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 조회

        Args:
            n_samples: 샘플 수 (최신 n개)
            since: 시작 시간 이후 데이터

        Returns:
            (X, y) 튜플
        """
        if len(self._buffer) == 0:
            return np.array([]), np.array([])

        data = list(self._buffer)
        timestamps = list(self._timestamps)

        if since:
            filtered = [(d, t) for d, t in zip(data, timestamps) if t >= since]
            if filtered:
                data = [d for d, _ in filtered]

        if n_samples:
            data = data[-n_samples:]

        if not data:
            return np.array([]), np.array([])

        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])

        return X, y

    def clear(self) -> None:
        """버퍼 초기화"""
        self._buffer.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class ConceptDriftDetector:
    """
    개념 드리프트 감지기

    Page-Hinkley 테스트 및 ADWIN을 기반으로 드리프트를 감지합니다.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        window_size: int = 100
    ):
        """
        Args:
            delta: Page-Hinkley 감도
            lambda_: 임계값
            window_size: 윈도우 크기
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.window_size = window_size

        # 상태
        self._sum = 0.0
        self._mean = 0.0
        self._min_value = float('inf')
        self._count = 0
        self._values: deque = deque(maxlen=window_size)

    def update(self, value: float) -> DriftDetectionResult:
        """
        값 업데이트 및 드리프트 확인

        Args:
            value: 새 값 (예: 예측 오차)

        Returns:
            드리프트 감지 결과
        """
        self._count += 1
        self._values.append(value)

        # Page-Hinkley 테스트
        self._mean = self._mean + (value - self._mean) / self._count
        self._sum = self._sum + value - self._mean - self.delta

        if self._sum < self._min_value:
            self._min_value = self._sum

        ph_value = self._sum - self._min_value

        # 드리프트 판단
        drift_detected = ph_value > self.lambda_

        # 드리프트 유형 분류
        if drift_detected:
            drift_type = self._classify_drift()
            confidence = min(1.0, ph_value / (self.lambda_ * 2))
        else:
            drift_type = DriftType.NO_DRIFT
            confidence = 0.0

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=confidence,
            timestamp=datetime.now(),
            details={
                'ph_value': ph_value,
                'threshold': self.lambda_,
                'mean': self._mean
            }
        )

    def _classify_drift(self) -> DriftType:
        """드리프트 유형 분류"""
        if len(self._values) < self.window_size // 2:
            return DriftType.SUDDEN

        values = list(self._values)
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])

        change_rate = abs(second_half - first_half) / (abs(first_half) + 1e-8)

        if change_rate > 0.5:
            return DriftType.SUDDEN
        else:
            return DriftType.GRADUAL

    def reset(self) -> None:
        """상태 초기화"""
        self._sum = 0.0
        self._mean = 0.0
        self._min_value = float('inf')
        self._count = 0
        self._values.clear()


class PerformanceMonitor:
    """
    성능 모니터

    모델 성능을 추적하고 분석합니다.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 이동 윈도우 크기
        """
        self.window_size = window_size
        self._errors: deque = deque(maxlen=window_size)
        self._predictions: deque = deque(maxlen=window_size)
        self._actuals: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

    def update(
        self,
        prediction: float,
        actual: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        예측 결과 업데이트

        Args:
            prediction: 예측값
            actual: 실제값
            timestamp: 타임스탬프
        """
        if timestamp is None:
            timestamp = datetime.now()

        error = abs(prediction - actual)
        self._errors.append(error)
        self._predictions.append(prediction)
        self._actuals.append(actual)
        self._timestamps.append(timestamp)

    def get_metrics(self) -> PerformanceMetrics:
        """현재 성능 메트릭 계산"""
        if len(self._errors) == 0:
            return PerformanceMetrics(
                mae=0.0, rmse=0.0, mape=0.0, r2=0.0
            )

        predictions = np.array(self._predictions)
        actuals = np.array(self._actuals)
        errors = np.array(self._errors)

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAPE (0 제외)
        mask = actuals != 0
        if mask.any():
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = 0.0

        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return PerformanceMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2)

    def get_trend(self) -> str:
        """성능 추세 분석"""
        if len(self._errors) < self.window_size // 2:
            return 'insufficient_data'

        errors = list(self._errors)
        first_half = np.mean(errors[:len(errors)//2])
        second_half = np.mean(errors[len(errors)//2:])

        change = (second_half - first_half) / (first_half + 1e-8)

        if change > 0.1:
            return 'degrading'
        elif change < -0.1:
            return 'improving'
        else:
            return 'stable'

    def reset(self) -> None:
        """상태 초기화"""
        self._errors.clear()
        self._predictions.clear()
        self._actuals.clear()
        self._timestamps.clear()


class IncrementalTrainer:
    """
    점진적 학습 트레이너

    새로운 데이터로 모델을 점진적으로 업데이트합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        config: UpdateConfig,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: PyTorch 모델
            config: 업데이트 설정
            device: 연산 디바이스
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')

        self._original_model_state = copy.deepcopy(model.state_dict())
        self._best_model_state = copy.deepcopy(model.state_dict())
        self._best_loss = float('inf')

    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        학습 스텝 수행

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            validation_data: 검증 데이터

        Returns:
            학습 메트릭
        """
        self.model.train()
        self.model.to(self.device)

        # 데이터 준비
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 옵티마이저
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()

        # 학습
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = self.model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # 검증
        val_loss = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_loss = self._validate(val_X, val_y)

            if val_loss < self._best_loss:
                self._best_loss = val_loss
                self._best_model_state = copy.deepcopy(self.model.state_dict())

        return {
            'train_loss': avg_loss,
            'val_loss': val_loss
        }

    def _validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """검증 수행"""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        with torch.no_grad():
            output = self.model(X_tensor)
            loss = nn.MSELoss()(output, y_tensor)

        return loss.item()

    def full_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        전체 학습 수행

        Args:
            X: 입력 데이터
            y: 타겟 데이터
            epochs: 에포크 수

        Returns:
            학습 이력
        """
        if epochs is None:
            epochs = self.config.epochs

        # 검증 데이터 분할
        n_val = int(len(X) * self.config.validation_split)
        indices = np.random.permutation(len(X))

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        history = {'train_loss': [], 'val_loss': []}
        patience_counter = 0

        for epoch in range(epochs):
            metrics = self.train_step(X_train, y_train, (X_val, y_val))

            history['train_loss'].append(metrics['train_loss'])
            if metrics['val_loss'] is not None:
                history['val_loss'].append(metrics['val_loss'])

            # 조기 종료
            if metrics['val_loss'] is not None:
                if metrics['val_loss'] >= self._best_loss:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                else:
                    patience_counter = 0

        # 최적 모델 복원
        self.model.load_state_dict(self._best_model_state)

        return history

    def rollback(self) -> None:
        """원래 모델로 롤백"""
        self.model.load_state_dict(self._original_model_state)

    def save_checkpoint(self) -> Dict[str, Any]:
        """체크포인트 저장"""
        return {
            'model_state': self.model.state_dict(),
            'best_loss': self._best_loss,
            'original_state': self._original_model_state,
            'best_state': self._best_model_state
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """체크포인트 로드"""
        self.model.load_state_dict(checkpoint['model_state'])
        self._best_loss = checkpoint['best_loss']
        self._original_model_state = checkpoint['original_state']
        self._best_model_state = checkpoint['best_state']


class OnlineLearner:
    """
    온라인 학습 관리자

    실시간 데이터 수집, 드리프트 감지, 모델 업데이트를 통합 관리합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[UpdateConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: PyTorch 모델
            config: 업데이트 설정
            device: 연산 디바이스
        """
        self.model = model
        self.config = config or UpdateConfig()
        self.device = device or torch.device('cpu')

        # 컴포넌트 초기화
        self.buffer = DataBuffer(self.config.max_buffer_size)
        self.drift_detector = ConceptDriftDetector(
            delta=self.config.drift_threshold / 10,
            lambda_=50.0
        )
        self.performance_monitor = PerformanceMonitor()
        self.trainer = IncrementalTrainer(model, config, device)

        # 상태
        self._last_update = datetime.now()
        self._update_count = 0
        self._is_updating = False

    def observe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prediction: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        새 데이터 관측

        Args:
            X: 입력 데이터
            y: 실제 타겟
            prediction: 예측값 (없으면 모델로 예측)

        Returns:
            관측 결과
        """
        # 예측
        if prediction is None:
            prediction = self.predict(X)

        # 버퍼에 추가
        self.buffer.add(X, y)

        # 성능 모니터 업데이트
        for p, a in zip(prediction.flatten(), y.flatten()):
            self.performance_monitor.update(p, a)

        # 드리프트 감지
        error = np.mean(np.abs(prediction - y))
        drift_result = self.drift_detector.update(error)

        # 업데이트 필요성 확인
        should_update = self._check_update_needed(drift_result)

        result = {
            'drift_detected': drift_result.drift_detected,
            'drift_type': drift_result.drift_type.value,
            'performance': self.performance_monitor.get_metrics().__dict__,
            'buffer_size': len(self.buffer),
            'should_update': should_update
        }

        # 자동 업데이트
        if should_update:
            update_result = self.update()
            result['update_result'] = update_result

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        self.model.eval()
        self.model.to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)

        return output.cpu().numpy()

    def _check_update_needed(self, drift_result: DriftDetectionResult) -> bool:
        """업데이트 필요 여부 확인"""
        # 업데이트 중이면 건너뜀
        if self._is_updating:
            return False

        # 최소 샘플 수 확인
        if len(self.buffer) < self.config.min_samples:
            return False

        # 드리프트 감지됨
        if drift_result.drift_detected:
            return True

        # 주기적 업데이트
        hours_since_update = (datetime.now() - self._last_update).total_seconds() / 3600
        if hours_since_update >= self.config.update_interval:
            return True

        return False

    def update(self, force: bool = False) -> Dict[str, Any]:
        """
        모델 업데이트

        Args:
            force: 강제 업데이트

        Returns:
            업데이트 결과
        """
        if self._is_updating and not force:
            return {'status': 'already_updating'}

        self._is_updating = True

        try:
            # 데이터 준비
            X, y = self.buffer.get_data()
            if len(X) < self.config.min_samples:
                return {'status': 'insufficient_data', 'samples': len(X)}

            # 업데이트 전 성능
            before_metrics = self.performance_monitor.get_metrics()

            # 전략에 따른 업데이트
            if self.config.strategy == UpdateStrategy.FULL_RETRAIN:
                history = self.trainer.full_train(X, y)
            elif self.config.strategy == UpdateStrategy.FINE_TUNE:
                history = self.trainer.full_train(X, y, epochs=self.config.epochs)
            elif self.config.strategy == UpdateStrategy.INCREMENTAL:
                history = {'train_loss': []}
                for _ in range(self.config.epochs):
                    metrics = self.trainer.train_step(X, y)
                    history['train_loss'].append(metrics['train_loss'])
            else:
                history = self.trainer.full_train(X, y)

            # 업데이트 후 상태 갱신
            self._last_update = datetime.now()
            self._update_count += 1
            self.drift_detector.reset()

            return {
                'status': 'success',
                'strategy': self.config.strategy.value,
                'samples_used': len(X),
                'history': history,
                'before_mae': before_metrics.mae,
                'update_count': self._update_count
            }

        finally:
            self._is_updating = False

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        metrics = self.performance_monitor.get_metrics()

        return {
            'buffer_size': len(self.buffer),
            'update_count': self._update_count,
            'last_update': self._last_update.isoformat(),
            'is_updating': self._is_updating,
            'performance': {
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'mape': metrics.mape,
                'r2': metrics.r2
            },
            'trend': self.performance_monitor.get_trend()
        }

    def reset(self) -> None:
        """상태 초기화"""
        self.buffer.clear()
        self.drift_detector.reset()
        self.performance_monitor.reset()
        self._update_count = 0


class ModelUpdater:
    """
    모델 업데이트 관리자

    여러 모델의 업데이트를 관리합니다.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Args:
            base_path: 저장 기본 경로
        """
        self.base_path = Path(base_path) if base_path else Path('./model_updates')
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._learners: Dict[str, OnlineLearner] = {}
        self._update_history: List[Dict[str, Any]] = []

    def register_model(
        self,
        name: str,
        model: nn.Module,
        config: Optional[UpdateConfig] = None,
        device: Optional[torch.device] = None
    ) -> OnlineLearner:
        """
        모델 등록

        Args:
            name: 모델 이름
            model: PyTorch 모델
            config: 업데이트 설정
            device: 연산 디바이스

        Returns:
            OnlineLearner 인스턴스
        """
        learner = OnlineLearner(model, config, device)
        self._learners[name] = learner

        logger.info(f"Registered model '{name}' for online learning")
        return learner

    def get_learner(self, name: str) -> Optional[OnlineLearner]:
        """학습기 조회"""
        return self._learners.get(name)

    def update_all(self, force: bool = False) -> Dict[str, Any]:
        """모든 모델 업데이트"""
        results = {}
        for name, learner in self._learners.items():
            result = learner.update(force=force)
            results[name] = result

            self._update_history.append({
                'model': name,
                'timestamp': datetime.now().isoformat(),
                'result': result
            })

        return results

    def get_all_status(self) -> Dict[str, Any]:
        """모든 모델 상태 조회"""
        return {
            name: learner.get_status()
            for name, learner in self._learners.items()
        }

    def save_state(self, path: Optional[Path] = None) -> None:
        """상태 저장"""
        path = path or self.base_path / 'state.json'

        state = {
            'models': list(self._learners.keys()),
            'update_history': self._update_history[-100:]  # 최근 100개
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_online_learner(
    model: nn.Module,
    strategy: str = 'fine_tune',
    min_samples: int = 100,
    update_interval: int = 24,
    device: Optional[torch.device] = None
) -> OnlineLearner:
    """
    온라인 학습기 생성

    Args:
        model: PyTorch 모델
        strategy: 업데이트 전략
        min_samples: 최소 샘플 수
        update_interval: 업데이트 주기 (시간)
        device: 연산 디바이스

    Returns:
        OnlineLearner 인스턴스
    """
    strategy_map = {
        'full_retrain': UpdateStrategy.FULL_RETRAIN,
        'fine_tune': UpdateStrategy.FINE_TUNE,
        'incremental': UpdateStrategy.INCREMENTAL,
        'ensemble_update': UpdateStrategy.ENSEMBLE_UPDATE
    }

    config = UpdateConfig(
        strategy=strategy_map.get(strategy, UpdateStrategy.FINE_TUNE),
        min_samples=min_samples,
        update_interval=update_interval
    )

    return OnlineLearner(model, config, device)


def create_model_updater(base_path: Optional[Path] = None) -> ModelUpdater:
    """모델 업데이트 관리자 생성"""
    return ModelUpdater(base_path)
