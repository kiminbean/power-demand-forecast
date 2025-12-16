"""
MLflow 유틸리티 모듈 (Task 9)
=============================
실험 추적, 모델 버전 관리, 모델 레지스트리 기능을 제공합니다.

주요 컴포넌트:
- ExperimentTracker: 실험 추적 관리
- ModelRegistry: 모델 레지스트리 관리
- ArtifactManager: 아티팩트 관리
- MetricLogger: 메트릭 로깅
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from enum import Enum
import json
import shutil
import hashlib
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """모델 스테이지"""
    NONE = 'None'
    STAGING = 'Staging'
    PRODUCTION = 'Production'
    ARCHIVED = 'Archived'


class MetricGoal(Enum):
    """메트릭 최적화 목표"""
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


@dataclass
class RunConfig:
    """실행 설정"""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ''


@dataclass
class MetricRecord:
    """메트릭 기록"""
    name: str
    value: float
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelVersion:
    """모델 버전"""
    name: str
    version: int
    stage: ModelStage
    run_id: str
    created_at: datetime
    description: str = ''
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Experiment:
    """실험"""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str = 'active'
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Run:
    """실행"""
    run_id: str
    experiment_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[MetricRecord]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    실험 추적 관리자

    MLflow 스타일의 실험 추적 기능을 제공합니다.
    """

    def __init__(self, tracking_uri: Optional[Path] = None):
        """
        Args:
            tracking_uri: 추적 저장소 경로
        """
        self.tracking_uri = Path(tracking_uri) if tracking_uri else Path('./mlruns')
        self.tracking_uri.mkdir(parents=True, exist_ok=True)

        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, Run] = {}
        self._active_run: Optional[Run] = None

        self._load_experiments()

    def _load_experiments(self) -> None:
        """저장된 실험 로드"""
        meta_file = self.tracking_uri / 'experiments.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                data = json.load(f)
            for exp_data in data:
                exp = Experiment(**exp_data)
                self._experiments[exp.name] = exp

    def _save_experiments(self) -> None:
        """실험 정보 저장"""
        meta_file = self.tracking_uri / 'experiments.json'
        data = [
            {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'tags': exp.tags
            }
            for exp in self._experiments.values()
        ]
        with open(meta_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        실험 생성

        Args:
            name: 실험 이름
            tags: 태그

        Returns:
            실험 ID
        """
        if name in self._experiments:
            return self._experiments[name].experiment_id

        experiment_id = hashlib.md5(name.encode()).hexdigest()[:8]
        artifact_location = str(self.tracking_uri / experiment_id / 'artifacts')

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            tags=tags or {}
        )

        # 디렉토리 생성
        (self.tracking_uri / experiment_id).mkdir(parents=True, exist_ok=True)

        self._experiments[name] = experiment
        self._save_experiments()

        logger.info(f"Created experiment '{name}' with id '{experiment_id}'")
        return experiment_id

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """실험 조회"""
        return self._experiments.get(name)

    def list_experiments(self) -> List[Experiment]:
        """실험 목록"""
        return list(self._experiments.values())

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Run:
        """
        실행 시작

        Args:
            experiment_name: 실험 이름
            run_name: 실행 이름
            tags: 태그

        Returns:
            실행 객체
        """
        # 실험 확인/생성
        if experiment_name not in self._experiments:
            self.create_experiment(experiment_name)

        experiment = self._experiments[experiment_name]
        run_id = hashlib.md5(f"{experiment_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        run = Run(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            status='RUNNING',
            start_time=datetime.now(),
            tags=tags or {}
        )

        if run_name:
            run.tags['mlflow.runName'] = run_name

        self._runs[run_id] = run
        self._active_run = run

        # 실행 디렉토리 생성
        run_dir = self.tracking_uri / experiment.experiment_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'artifacts').mkdir(exist_ok=True)

        logger.info(f"Started run '{run_id}' in experiment '{experiment_name}'")
        return run

    def end_run(self, status: str = 'FINISHED') -> None:
        """실행 종료"""
        if self._active_run:
            self._active_run.status = status
            self._active_run.end_time = datetime.now()
            self._save_run(self._active_run)
            logger.info(f"Ended run '{self._active_run.run_id}' with status '{status}'")
            self._active_run = None

    def _save_run(self, run: Run) -> None:
        """실행 정보 저장"""
        run_dir = self.tracking_uri / run.experiment_id / run.run_id
        meta_file = run_dir / 'meta.json'

        # 메트릭을 직렬화 가능한 형태로 변환
        metrics_data = {}
        for name, records in run.metrics.items():
            metrics_data[name] = [
                {
                    'value': r.value,
                    'step': r.step,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in records
            ]

        data = {
            'run_id': run.run_id,
            'experiment_id': run.experiment_id,
            'status': run.status,
            'start_time': run.start_time.isoformat(),
            'end_time': run.end_time.isoformat() if run.end_time else None,
            'params': run.params,
            'metrics': metrics_data,
            'tags': run.tags,
            'artifacts': run.artifacts
        }

        with open(meta_file, 'w') as f:
            json.dump(data, f, indent=2)

    def log_param(self, key: str, value: Any) -> None:
        """파라미터 로깅"""
        if self._active_run:
            self._active_run.params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """여러 파라미터 로깅"""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """메트릭 로깅"""
        if self._active_run:
            if key not in self._active_run.metrics:
                self._active_run.metrics[key] = []

            record = MetricRecord(name=key, value=value, step=step)
            self._active_run.metrics[key].append(record)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """여러 메트릭 로깅"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def set_tag(self, key: str, value: str) -> None:
        """태그 설정"""
        if self._active_run:
            self._active_run.tags[key] = value

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        """아티팩트 로깅"""
        if not self._active_run:
            return

        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning(f"Artifact path does not exist: {local_path}")
            return

        run_dir = self.tracking_uri / self._active_run.experiment_id / self._active_run.run_id
        artifacts_dir = run_dir / 'artifacts'

        if artifact_path:
            dest_dir = artifacts_dir / artifact_path
            dest_dir.mkdir(parents=True, exist_ok=True)
        else:
            dest_dir = artifacts_dir

        if local_path.is_file():
            shutil.copy2(local_path, dest_dir / local_path.name)
            self._active_run.artifacts.append(str(dest_dir / local_path.name))
        else:
            shutil.copytree(local_path, dest_dir / local_path.name)
            self._active_run.artifacts.append(str(dest_dir / local_path.name))

    def get_run(self, run_id: str) -> Optional[Run]:
        """실행 조회"""
        return self._runs.get(run_id)

    @property
    def active_run(self) -> Optional[Run]:
        """현재 활성 실행"""
        return self._active_run


class ModelRegistry:
    """
    모델 레지스트리

    모델 버전 관리 및 스테이지 전환을 담당합니다.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Args:
            registry_path: 레지스트리 저장 경로
        """
        self.registry_path = Path(registry_path) if registry_path else Path('./model_registry')
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, Dict[int, ModelVersion]] = {}  # name -> version -> ModelVersion
        self._load_registry()

    def _load_registry(self) -> None:
        """레지스트리 로드"""
        meta_file = self.registry_path / 'registry.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                data = json.load(f)

            for model_name, versions in data.items():
                self._models[model_name] = {}
                for version_str, version_data in versions.items():
                    version_data['stage'] = ModelStage(version_data['stage'])
                    version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                    self._models[model_name][int(version_str)] = ModelVersion(**version_data)

    def _save_registry(self) -> None:
        """레지스트리 저장"""
        meta_file = self.registry_path / 'registry.json'

        data = {}
        for model_name, versions in self._models.items():
            data[model_name] = {}
            for version, mv in versions.items():
                data[model_name][str(version)] = {
                    'name': mv.name,
                    'version': mv.version,
                    'stage': mv.stage.value,
                    'run_id': mv.run_id,
                    'created_at': mv.created_at.isoformat(),
                    'description': mv.description,
                    'tags': mv.tags,
                    'metrics': mv.metrics
                }

        with open(meta_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        name: str,
        run_id: str,
        model_path: Path,
        description: str = '',
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> ModelVersion:
        """
        모델 등록

        Args:
            name: 모델 이름
            run_id: 실행 ID
            model_path: 모델 파일 경로
            description: 설명
            tags: 태그
            metrics: 메트릭

        Returns:
            모델 버전
        """
        if name not in self._models:
            self._models[name] = {}
            (self.registry_path / name).mkdir(exist_ok=True)

        version = max(self._models[name].keys(), default=0) + 1

        model_version = ModelVersion(
            name=name,
            version=version,
            stage=ModelStage.NONE,
            run_id=run_id,
            created_at=datetime.now(),
            description=description,
            tags=tags or {},
            metrics=metrics or {}
        )

        # 모델 파일 복사
        version_dir = self.registry_path / name / f'v{version}'
        version_dir.mkdir(exist_ok=True)

        model_path = Path(model_path)
        if model_path.is_file():
            shutil.copy2(model_path, version_dir / model_path.name)
        else:
            shutil.copytree(model_path, version_dir / 'model')

        self._models[name][version] = model_version
        self._save_registry()

        logger.info(f"Registered model '{name}' version {version}")
        return model_version

    def get_model_version(self, name: str, version: int) -> Optional[ModelVersion]:
        """모델 버전 조회"""
        if name in self._models and version in self._models[name]:
            return self._models[name][version]
        return None

    def get_latest_version(self, name: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """최신 버전 조회"""
        if name not in self._models:
            return None

        versions = self._models[name]
        if stage:
            versions = {v: mv for v, mv in versions.items() if mv.stage == stage}

        if not versions:
            return None

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_model_versions(
        self,
        name: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """모델 버전 목록"""
        if name not in self._models:
            return []

        versions = list(self._models[name].values())
        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda x: x.version, reverse=True)

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage
    ) -> ModelVersion:
        """
        스테이지 전환

        Args:
            name: 모델 이름
            version: 버전
            stage: 새 스테이지

        Returns:
            업데이트된 모델 버전
        """
        if name not in self._models or version not in self._models[name]:
            raise ValueError(f"Model '{name}' version {version} not found")

        # Production 전환 시 기존 Production 모델을 Archived로
        if stage == ModelStage.PRODUCTION:
            for v, mv in self._models[name].items():
                if mv.stage == ModelStage.PRODUCTION:
                    mv.stage = ModelStage.ARCHIVED
                    logger.info(f"Archived model '{name}' version {v}")

        model_version = self._models[name][version]
        model_version.stage = stage
        self._save_registry()

        logger.info(f"Transitioned model '{name}' version {version} to {stage.value}")
        return model_version

    def get_model_path(self, name: str, version: int) -> Path:
        """모델 경로 조회"""
        return self.registry_path / name / f'v{version}'

    def delete_model_version(self, name: str, version: int) -> bool:
        """모델 버전 삭제"""
        if name not in self._models or version not in self._models[name]:
            return False

        # 파일 삭제
        version_dir = self.registry_path / name / f'v{version}'
        if version_dir.exists():
            shutil.rmtree(version_dir)

        del self._models[name][version]
        self._save_registry()

        logger.info(f"Deleted model '{name}' version {version}")
        return True


class MetricLogger:
    """
    메트릭 로거

    학습 중 메트릭을 추적하고 시각화합니다.
    """

    def __init__(self, tracker: Optional[ExperimentTracker] = None):
        """
        Args:
            tracker: 실험 추적기
        """
        self.tracker = tracker
        self._history: Dict[str, List[float]] = {}
        self._step = 0

    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        메트릭 로깅

        Args:
            metrics: 메트릭 딕셔너리
            step: 스텝 번호
        """
        if step is None:
            step = self._step
            self._step += 1

        for name, value in metrics.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(value)

            if self.tracker:
                self.tracker.log_metric(name, value, step)

    def get_history(self, metric_name: str) -> List[float]:
        """메트릭 이력 조회"""
        return self._history.get(metric_name, [])

    def get_best(
        self,
        metric_name: str,
        goal: MetricGoal = MetricGoal.MINIMIZE
    ) -> Optional[float]:
        """최적 메트릭 값 조회"""
        history = self._history.get(metric_name, [])
        if not history:
            return None

        if goal == MetricGoal.MINIMIZE:
            return min(history)
        else:
            return max(history)

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame 변환"""
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        """이력 초기화"""
        self._history.clear()
        self._step = 0


class PyTorchModelSaver:
    """
    PyTorch 모델 저장/로드

    모델 저장 시 상태와 메타데이터를 함께 저장합니다.
    """

    @staticmethod
    def save(
        model: nn.Module,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        모델 저장

        Args:
            model: PyTorch 모델
            path: 저장 경로
            optimizer: 옵티마이저
            epoch: 에포크
            metrics: 메트릭
            config: 설정
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
            'config': config or {},
            'model_class': model.__class__.__name__
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path / 'model.pt')

        # 메타데이터 저장
        meta = {
            'model_class': model.__class__.__name__,
            'epoch': epoch,
            'metrics': metrics or {},
            'config': config or {},
            'saved_at': datetime.now().isoformat()
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved model to {path}")

    @staticmethod
    def load(
        path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        모델 로드

        Args:
            path: 저장 경로
            model: PyTorch 모델 (구조)
            optimizer: 옵티마이저
            device: 디바이스

        Returns:
            체크포인트 정보
        """
        path = Path(path)
        checkpoint = torch.load(path / 'model.pt', map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded model from {path}")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }


class TrainingCallback:
    """학습 콜백 베이스 클래스"""

    def on_epoch_start(self, epoch: int) -> None:
        """에포크 시작"""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """에포크 종료"""
        pass

    def on_train_end(self) -> None:
        """학습 종료"""
        pass


class MLflowCallback(TrainingCallback):
    """MLflow 로깅 콜백"""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """에포크 종료 시 메트릭 로깅"""
        self.tracker.log_metrics(metrics, step=epoch)

    def on_train_end(self) -> None:
        """학습 종료 시 실행 종료"""
        self.tracker.end_run()


class EarlyStoppingCallback(TrainingCallback):
    """조기 종료 콜백"""

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """에포크 종료 시 조기 종료 확인"""
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_experiment_tracker(tracking_uri: Optional[Path] = None) -> ExperimentTracker:
    """실험 추적기 생성"""
    return ExperimentTracker(tracking_uri)


def create_model_registry(registry_path: Optional[Path] = None) -> ModelRegistry:
    """모델 레지스트리 생성"""
    return ModelRegistry(registry_path)


def create_metric_logger(tracker: Optional[ExperimentTracker] = None) -> MetricLogger:
    """메트릭 로거 생성"""
    return MetricLogger(tracker)


def log_pytorch_model(
    tracker: ExperimentTracker,
    registry: ModelRegistry,
    model: nn.Module,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> ModelVersion:
    """
    PyTorch 모델 로깅 및 등록

    Args:
        tracker: 실험 추적기
        registry: 모델 레지스트리
        model: PyTorch 모델
        model_name: 모델 이름
        metrics: 메트릭
        config: 설정

    Returns:
        등록된 모델 버전
    """
    if not tracker.active_run:
        raise ValueError("No active run. Call tracker.start_run() first.")

    run = tracker.active_run

    # 임시 저장
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / 'model'
        PyTorchModelSaver.save(model, tmp_path, metrics=metrics, config=config)

        # 아티팩트 로깅
        tracker.log_artifact(tmp_path, 'model')

        # 레지스트리 등록
        model_version = registry.register_model(
            name=model_name,
            run_id=run.run_id,
            model_path=tmp_path,
            metrics=metrics or {}
        )

    return model_version
