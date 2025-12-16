"""
Task 19: AutoML 모델 선택 시스템
================================

자동 모델 비교, 선택, 하이퍼파라미터 튜닝을 제공합니다.

주요 기능:
1. 여러 모델 자동 비교
2. Optuna 기반 하이퍼파라미터 튜닝
3. 최적 모델 자동 선택
4. 모델 비교 리포트 생성

Author: Claude Code
Date: 2025-12
"""

import time
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # Fallback for type hints when optuna not available
    warnings.warn("Optuna not available. Install with: pip install optuna")

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """모델 타입"""
    LSTM = 'lstm'
    BILSTM = 'bilstm'
    TFT = 'tft'
    ENSEMBLE = 'ensemble'


@dataclass
class ModelConfig:
    """모델 설정"""
    model_type: ModelType
    name: str
    hyperparameters: Dict[str, Any]
    search_space: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'name': self.name,
            'hyperparameters': self.hyperparameters
        }


@dataclass
class ModelResult:
    """모델 결과"""
    config: ModelConfig
    metrics: Dict[str, float]
    training_time: float
    model_path: Optional[str] = None
    best_epoch: int = 0
    history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'training_time': self.training_time,
            'model_path': self.model_path,
            'best_epoch': self.best_epoch
        }


@dataclass
class SearchSpace:
    """하이퍼파라미터 탐색 공간"""
    name: str
    param_type: str  # 'int', 'float', 'categorical', 'loguniform'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    def suggest(self, trial: 'Trial') -> Any:
        """Optuna trial에서 값 제안"""
        if self.param_type == 'int':
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        elif self.param_type == 'float':
            return trial.suggest_float(self.name, self.low, self.high, log=self.log)
        elif self.param_type == 'loguniform':
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        elif self.param_type == 'categorical':
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


# ============================================================================
# 탐색 공간 프리셋
# ============================================================================

LSTM_SEARCH_SPACE = [
    SearchSpace('hidden_size', 'categorical', choices=[32, 64, 128, 256]),
    SearchSpace('num_layers', 'int', low=1, high=4),
    SearchSpace('dropout', 'float', low=0.0, high=0.5),
    SearchSpace('learning_rate', 'loguniform', low=1e-5, high=1e-2),
    SearchSpace('batch_size', 'categorical', choices=[32, 64, 128]),
]

TFT_SEARCH_SPACE = [
    SearchSpace('hidden_size', 'categorical', choices=[32, 64, 128, 256]),
    SearchSpace('lstm_layers', 'int', low=1, high=3),
    SearchSpace('num_attention_heads', 'categorical', choices=[2, 4, 8]),
    SearchSpace('dropout', 'float', low=0.05, high=0.3),
    SearchSpace('learning_rate', 'loguniform', low=1e-4, high=1e-2),
    SearchSpace('batch_size', 'categorical', choices=[32, 64, 128]),
]

ENSEMBLE_SEARCH_SPACE = [
    SearchSpace('n_models', 'int', low=3, high=7),
    SearchSpace('diversity_weight', 'float', low=0.0, high=0.5),
    SearchSpace('aggregation', 'categorical', choices=['mean', 'weighted', 'stacking']),
]


def get_search_space(model_type: ModelType) -> List[SearchSpace]:
    """모델 타입에 따른 탐색 공간 반환"""
    spaces = {
        ModelType.LSTM: LSTM_SEARCH_SPACE,
        ModelType.BILSTM: LSTM_SEARCH_SPACE,
        ModelType.TFT: TFT_SEARCH_SPACE,
        ModelType.ENSEMBLE: ENSEMBLE_SEARCH_SPACE,
    }
    return spaces.get(model_type, LSTM_SEARCH_SPACE)


# ============================================================================
# 모델 팩토리
# ============================================================================

class ModelFactory:
    """모델 생성 팩토리"""

    def __init__(self, input_size: int, output_size: int = 1, device: str = 'cpu'):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    def create_lstm(self, **kwargs) -> nn.Module:
        """LSTM 모델 생성"""
        from src.models.lstm import LSTMModel

        model = LSTMModel(
            input_size=self.input_size,
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1),
            bidirectional=kwargs.get('bidirectional', False),
            output_size=self.output_size
        )
        return model.to(self.device)

    def create_bilstm(self, **kwargs) -> nn.Module:
        """BiLSTM 모델 생성"""
        kwargs['bidirectional'] = True
        return self.create_lstm(**kwargs)

    def create_tft(self, **kwargs) -> nn.Module:
        """TFT 모델 생성"""
        from src.models.transformer import TemporalFusionTransformer

        model = TemporalFusionTransformer(
            num_known_vars=kwargs.get('num_known_vars', 8),
            num_unknown_vars=kwargs.get('num_unknown_vars', self.input_size),
            hidden_size=kwargs.get('hidden_size', 64),
            lstm_layers=kwargs.get('lstm_layers', 2),
            num_attention_heads=kwargs.get('num_attention_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
            encoder_length=kwargs.get('encoder_length', 48),
            decoder_length=kwargs.get('decoder_length', 24),
        )
        return model.to(self.device)

    def create(self, model_type: ModelType, **kwargs) -> nn.Module:
        """모델 타입에 따라 생성"""
        creators = {
            ModelType.LSTM: self.create_lstm,
            ModelType.BILSTM: self.create_bilstm,
            ModelType.TFT: self.create_tft,
        }

        creator = creators.get(model_type)
        if creator is None:
            raise ValueError(f"Unknown model type: {model_type}")

        return creator(**kwargs)


# ============================================================================
# 하이퍼파라미터 튜너
# ============================================================================

class HyperparameterTuner:
    """
    Optuna 기반 하이퍼파라미터 튜너

    Args:
        model_factory: 모델 팩토리
        train_fn: 학습 함수 (model, train_loader, val_loader, **kwargs) -> metrics
        search_space: 탐색 공간
        metric_name: 최적화할 메트릭 이름
        direction: 'minimize' 또는 'maximize'
        n_trials: 시도 횟수

    Example:
        >>> tuner = HyperparameterTuner(factory, train_fn, search_space)
        >>> best_params, best_value = tuner.tune(train_loader, val_loader)
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        train_fn: Callable,
        search_space: List[SearchSpace],
        metric_name: str = 'val_loss',
        direction: str = 'minimize',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. pip install optuna")

        self.model_factory = model_factory
        self.train_fn = train_fn
        self.search_space = search_space
        self.metric_name = metric_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

        self._study: Optional[optuna.Study] = None
        self._best_model: Optional[nn.Module] = None

    def _objective(
        self,
        trial: Trial,
        model_type: ModelType,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **train_kwargs
    ) -> float:
        """Optuna objective 함수"""
        # 하이퍼파라미터 제안
        params = {}
        for space in self.search_space:
            params[space.name] = space.suggest(trial)

        # 모델 생성
        model = self.model_factory.create(model_type, **params)

        # 학습
        try:
            metrics = self.train_fn(
                model,
                train_loader,
                val_loader,
                learning_rate=params.get('learning_rate', 1e-3),
                **train_kwargs
            )

            value = metrics.get(self.metric_name)
            if value is None:
                raise ValueError(f"Metric {self.metric_name} not found")

            return value

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf') if self.direction == 'minimize' else float('-inf')

    def tune(
        self,
        model_type: ModelType,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **train_kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        하이퍼파라미터 튜닝 실행

        Returns:
            best_params: 최적 하이퍼파라미터
            best_value: 최적 메트릭 값
        """
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self._study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )

        self._study.optimize(
            lambda trial: self._objective(
                trial, model_type, train_loader, val_loader, **train_kwargs
            ),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        return self._study.best_params, self._study.best_value

    def get_study_results(self) -> pd.DataFrame:
        """스터디 결과 DataFrame 반환"""
        if self._study is None:
            raise ValueError("No study available. Run tune() first.")

        return self._study.trials_dataframe()

    def get_param_importances(self) -> Dict[str, float]:
        """파라미터 중요도 반환"""
        if self._study is None:
            raise ValueError("No study available. Run tune() first.")

        return optuna.importance.get_param_importances(self._study)


# ============================================================================
# 모델 비교기
# ============================================================================

class ModelComparator:
    """
    여러 모델을 비교하고 최적 모델을 선택합니다.

    Args:
        model_factory: 모델 팩토리
        train_fn: 학습 함수
        metrics: 평가 메트릭 이름 리스트
        primary_metric: 주요 비교 메트릭
        direction: 'minimize' 또는 'maximize'

    Example:
        >>> comparator = ModelComparator(factory, train_fn)
        >>> comparator.add_model(ModelConfig(ModelType.LSTM, 'lstm_default', {...}))
        >>> results = comparator.compare(train_loader, val_loader)
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        train_fn: Callable,
        metrics: List[str] = None,
        primary_metric: str = 'val_loss',
        direction: str = 'minimize'
    ):
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.metrics = metrics or ['val_loss', 'val_mse', 'val_mae']
        self.primary_metric = primary_metric
        self.direction = direction

        self._configs: List[ModelConfig] = []
        self._results: List[ModelResult] = []

    def add_model(self, config: ModelConfig) -> None:
        """모델 설정 추가"""
        self._configs.append(config)

    def add_models(self, configs: List[ModelConfig]) -> None:
        """여러 모델 설정 추가"""
        self._configs.extend(configs)

    def compare(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
        **train_kwargs
    ) -> List[ModelResult]:
        """
        모델 비교 실행

        Returns:
            결과 리스트 (primary_metric 기준 정렬)
        """
        self._results = []

        for i, config in enumerate(self._configs):
            if verbose:
                logger.info(f"Training model {i+1}/{len(self._configs)}: {config.name}")

            start_time = time.time()

            try:
                # 모델 생성
                model = self.model_factory.create(
                    config.model_type,
                    **config.hyperparameters
                )

                # 학습
                metrics = self.train_fn(
                    model,
                    train_loader,
                    val_loader,
                    **train_kwargs
                )

                training_time = time.time() - start_time

                result = ModelResult(
                    config=config,
                    metrics=metrics,
                    training_time=training_time,
                    best_epoch=metrics.get('best_epoch', 0),
                    history=metrics.get('history', {})
                )

            except Exception as e:
                logger.error(f"Failed to train {config.name}: {e}")
                result = ModelResult(
                    config=config,
                    metrics={self.primary_metric: float('inf')},
                    training_time=time.time() - start_time
                )

            self._results.append(result)

            if verbose:
                logger.info(f"  {self.primary_metric}: {result.metrics.get(self.primary_metric, 'N/A'):.6f}")

        # 정렬
        self._results.sort(
            key=lambda r: r.metrics.get(self.primary_metric, float('inf')),
            reverse=(self.direction == 'maximize')
        )

        return self._results

    def get_best_model(self) -> ModelResult:
        """최적 모델 결과 반환"""
        if not self._results:
            raise ValueError("No results available. Run compare() first.")
        return self._results[0]

    def get_comparison_table(self) -> pd.DataFrame:
        """비교 테이블 DataFrame 반환"""
        if not self._results:
            raise ValueError("No results available. Run compare() first.")

        rows = []
        for result in self._results:
            row = {
                'model_name': result.config.name,
                'model_type': result.config.model_type.value,
                'training_time': result.training_time,
                **result.metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, output_path: str) -> None:
        """결과 저장"""
        if not self._results:
            raise ValueError("No results available. Run compare() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'primary_metric': self.primary_metric,
            'direction': self.direction,
            'results': [r.to_dict() for r in self._results]
        }

        with open(output_path / 'comparison_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # CSV 저장
        df = self.get_comparison_table()
        df.to_csv(output_path / 'comparison_table.csv', index=False)

        logger.info(f"Results saved to {output_path}")


# ============================================================================
# AutoML 파이프라인
# ============================================================================

class AutoMLPipeline:
    """
    AutoML 파이프라인: 자동 모델 선택 및 튜닝

    1. 여러 모델 타입 비교
    2. 최적 모델 타입 선택
    3. 하이퍼파라미터 튜닝
    4. 최종 모델 선택

    Args:
        input_size: 입력 피처 수
        output_size: 출력 크기
        train_fn: 학습 함수
        device: 디바이스

    Example:
        >>> pipeline = AutoMLPipeline(input_size=25, train_fn=train_model)
        >>> best_model, best_params = pipeline.run(train_loader, val_loader)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        train_fn: Callable = None,
        device: str = 'cpu',
        primary_metric: str = 'val_loss',
        direction: str = 'minimize'
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.train_fn = train_fn
        self.device = device
        self.primary_metric = primary_metric
        self.direction = direction

        self.model_factory = ModelFactory(input_size, output_size, device)

        self._comparison_results: Optional[List[ModelResult]] = None
        self._tuning_results: Optional[Tuple[Dict, float]] = None
        self._best_model: Optional[nn.Module] = None
        self._best_config: Optional[ModelConfig] = None

    def _create_default_configs(self) -> List[ModelConfig]:
        """기본 모델 설정 생성"""
        return [
            ModelConfig(
                model_type=ModelType.LSTM,
                name='lstm_small',
                hyperparameters={'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1}
            ),
            ModelConfig(
                model_type=ModelType.LSTM,
                name='lstm_medium',
                hyperparameters={'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}
            ),
            ModelConfig(
                model_type=ModelType.LSTM,
                name='lstm_large',
                hyperparameters={'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2}
            ),
            ModelConfig(
                model_type=ModelType.BILSTM,
                name='bilstm_medium',
                hyperparameters={'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}
            ),
        ]

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_configs: List[ModelConfig] = None,
        tune_best: bool = True,
        n_tuning_trials: int = 30,
        output_dir: Optional[str] = None,
        **train_kwargs
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        AutoML 파이프라인 실행

        Args:
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            model_configs: 비교할 모델 설정 (없으면 기본값)
            tune_best: 최적 모델 튜닝 여부
            n_tuning_trials: 튜닝 시도 횟수
            output_dir: 결과 저장 디렉토리

        Returns:
            best_model: 최적 모델
            best_params: 최적 하이퍼파라미터
        """
        if self.train_fn is None:
            raise ValueError("train_fn must be provided")

        configs = model_configs or self._create_default_configs()

        # Step 1: 모델 비교
        logger.info("Step 1: Comparing models...")
        comparator = ModelComparator(
            self.model_factory,
            self.train_fn,
            primary_metric=self.primary_metric,
            direction=self.direction
        )
        comparator.add_models(configs)

        self._comparison_results = comparator.compare(
            train_loader, val_loader, **train_kwargs
        )

        best_result = comparator.get_best_model()
        best_model_type = best_result.config.model_type

        logger.info(f"Best model type: {best_model_type.value}")
        logger.info(f"Best {self.primary_metric}: {best_result.metrics.get(self.primary_metric):.6f}")

        # Step 2: 하이퍼파라미터 튜닝 (선택적)
        if tune_best and OPTUNA_AVAILABLE:
            logger.info("Step 2: Tuning hyperparameters...")

            search_space = get_search_space(best_model_type)
            tuner = HyperparameterTuner(
                self.model_factory,
                self.train_fn,
                search_space,
                metric_name=self.primary_metric,
                direction=self.direction,
                n_trials=n_tuning_trials
            )

            best_params, best_value = tuner.tune(
                best_model_type, train_loader, val_loader, **train_kwargs
            )

            self._tuning_results = (best_params, best_value)

            logger.info(f"Best tuned {self.primary_metric}: {best_value:.6f}")
            logger.info(f"Best params: {best_params}")
        else:
            best_params = best_result.config.hyperparameters
            logger.info("Skipping tuning (Optuna not available or tune_best=False)")

        # Step 3: 최종 모델 생성
        logger.info("Step 3: Creating final model...")
        self._best_model = self.model_factory.create(best_model_type, **best_params)
        self._best_config = ModelConfig(
            model_type=best_model_type,
            name=f'{best_model_type.value}_tuned',
            hyperparameters=best_params
        )

        # 결과 저장
        if output_dir:
            self._save_results(output_dir, comparator)

        return self._best_model, best_params

    def _save_results(self, output_dir: str, comparator: ModelComparator) -> None:
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 비교 결과 저장
        comparator.save_results(output_path / 'comparison')

        # 전체 파이프라인 결과 저장
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'primary_metric': self.primary_metric,
            'direction': self.direction,
            'best_model': self._best_config.to_dict() if self._best_config else None,
            'tuning_results': {
                'best_params': self._tuning_results[0] if self._tuning_results else None,
                'best_value': self._tuning_results[1] if self._tuning_results else None
            } if self._tuning_results else None
        }

        with open(output_path / 'automl_results.json', 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)

        logger.info(f"AutoML results saved to {output_path}")

    def get_comparison_table(self) -> pd.DataFrame:
        """비교 테이블 반환"""
        if not self._comparison_results:
            raise ValueError("No comparison results. Run pipeline first.")

        rows = []
        for result in self._comparison_results:
            row = {
                'model_name': result.config.name,
                'model_type': result.config.model_type.value,
                'training_time': result.training_time,
                **result.metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)


# ============================================================================
# 유틸리티 함수
# ============================================================================

def simple_train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    간단한 학습 함수 (AutoML용)

    Returns:
        metrics: {'val_loss': ..., 'val_mse': ..., 'best_epoch': ...}
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                output = output[:, -1, :]

            loss = criterion(output.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() == 3:
                    output = output[:, -1, :]

                loss = criterion(output.squeeze(), y.squeeze())
                val_loss += loss.item()

                val_predictions.append(output.cpu())
                val_targets.append(y.cpu())

        val_loss /= len(val_loader)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 메트릭 계산
    predictions = torch.cat(val_predictions)
    targets = torch.cat(val_targets)

    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    return {
        'val_loss': best_val_loss,
        'val_mse': mse,
        'val_mae': mae,
        'best_epoch': best_epoch
    }


def create_automl_pipeline(
    input_size: int,
    output_size: int = 1,
    device: str = 'cpu',
    **kwargs
) -> AutoMLPipeline:
    """AutoML 파이프라인 팩토리 함수"""
    return AutoMLPipeline(
        input_size=input_size,
        output_size=output_size,
        train_fn=simple_train_fn,
        device=device,
        **kwargs
    )
