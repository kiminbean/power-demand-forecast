"""
MODEL-006: TFT 학습 파이프라인
==============================

Temporal Fusion Transformer 모델 학습을 위한 Dataset, Trainer, 설정

주요 기능:
1. TFTDataset: Known/Unknown 피처 분리
2. TFTTrainer: Quantile Loss 기반 학습
3. 피처 설정 (Static, Known, Unknown)
4. Multi-horizon 예측 지원

Author: Claude Code
Date: 2024-12
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import TimeSeriesScaler, get_device, split_data_by_time
from training.trainer import EarlyStopping, TrainingHistory, create_scheduler
from models.transformer import TemporalFusionTransformer, QuantileLoss


# ============================================================
# TFT 피처 설정
# ============================================================

class TFTFeatureConfig:
    """
    TFT 모델을 위한 피처 설정

    피처를 Static, Known, Unknown으로 분류합니다.

    제주도 전력수요 예측 기준:
    - Static: 없음 (단일 지역)
    - Known: 시간 관련 피처 (미래 값을 알 수 있음)
    - Unknown: 기상/수요 관련 피처 (현재까지만 알 수 있음)
    """

    # 시간 관련 피처 (미래 값을 알 수 있음)
    TIME_VARYING_KNOWN = [
        # 주기 인코딩
        'hour_sin', 'hour_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos',
        # 범주형 시간 피처
        'hour', 'dayofweek', 'month',
        'is_weekend', 'is_holiday',
    ]

    # 기상/수요 관련 피처 (현재까지만 알 수 있음)
    TIME_VARYING_UNKNOWN = [
        # Target (첫 번째)
        'power_demand',
        # 기상 변수
        'temp_mean', 'ground_temp_mean', 'soil_temp_5cm_mean',
        'humidity_mean', 'wind_speed_mean', 'irradiance_mean',
        'total_cloud_cover_mean', 'dewpoint_mean',
        # 파생 변수
        'THI', 'HDD', 'CDD', 'wind_chill',
        # Lag 피처
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
        'demand_ma_24h', 'demand_ma_168h',
        'temp_lag_1h', 'temp_lag_24h',
        # 외부 변수
        'ev_cumsum', 'visitors_daily',
    ]

    # Static 피처 (현재 프로젝트에서는 없음)
    STATIC = []

    def __init__(
        self,
        known_features: List[str] = None,
        unknown_features: List[str] = None,
        static_features: List[str] = None,
        target_col: str = 'power_demand'
    ):
        """
        Args:
            known_features: Known 피처 목록 (None이면 기본값)
            unknown_features: Unknown 피처 목록 (None이면 기본값)
            static_features: Static 피처 목록 (None이면 기본값)
            target_col: 타겟 컬럼명
        """
        self.known_features = known_features or self.TIME_VARYING_KNOWN.copy()
        self.unknown_features = unknown_features or self.TIME_VARYING_UNKNOWN.copy()
        self.static_features = static_features or self.STATIC.copy()
        self.target_col = target_col

        # 타겟을 unknown_features 첫 번째로 보장
        if target_col in self.unknown_features:
            self.unknown_features.remove(target_col)
        self.unknown_features = [target_col] + self.unknown_features

    def get_available_features(self, df_columns: List[str]) -> Dict[str, List[str]]:
        """
        DataFrame에서 사용 가능한 피처만 필터링

        Args:
            df_columns: DataFrame 컬럼 목록

        Returns:
            Dict: {'known': [...], 'unknown': [...], 'static': [...]}
        """
        available_known = [f for f in self.known_features if f in df_columns]
        available_unknown = [f for f in self.unknown_features if f in df_columns]
        available_static = [f for f in self.static_features if f in df_columns]

        return {
            'known': available_known,
            'unknown': available_unknown,
            'static': available_static
        }

    def validate(self, df_columns: List[str]) -> None:
        """피처 설정 검증"""
        available = self.get_available_features(df_columns)

        missing_known = set(self.known_features) - set(df_columns)
        missing_unknown = set(self.unknown_features) - set(df_columns)

        if missing_known:
            print(f"Warning: Missing known features: {missing_known}")
        if missing_unknown:
            print(f"Warning: Missing unknown features: {missing_unknown}")

        print(f"Available features - Known: {len(available['known'])}, "
              f"Unknown: {len(available['unknown'])}, Static: {len(available['static'])}")


# ============================================================
# TFT Dataset
# ============================================================

class TFTDataset(Dataset):
    """
    Temporal Fusion Transformer를 위한 PyTorch Dataset

    Known/Unknown 피처를 분리하여 TFT 모델 입력 형식으로 변환

    입력 형식:
        - known_inputs: (encoder_length + decoder_length, num_known, 1)
        - unknown_inputs: (encoder_length, num_unknown, 1)
        - static_inputs: (num_static, 1) (optional)

    출력:
        - targets: (decoder_length,) 미래 타겟 값들

    Args:
        data: 전체 데이터 (numpy array)
        known_indices: Known 피처 인덱스 리스트
        unknown_indices: Unknown 피처 인덱스 리스트
        target_idx: 타겟 피처 인덱스
        encoder_length: Encoder 시퀀스 길이 (과거)
        decoder_length: Decoder 시퀀스 길이 (미래)
        static_indices: Static 피처 인덱스 리스트 (optional)

    Example:
        >>> dataset = TFTDataset(
        ...     data=scaled_data,
        ...     known_indices=[0, 1, 2],
        ...     unknown_indices=[3, 4, 5],
        ...     target_idx=3,
        ...     encoder_length=48,
        ...     decoder_length=24
        ... )
        >>> known, unknown, targets = dataset[0]
    """

    def __init__(
        self,
        data: np.ndarray,
        known_indices: List[int],
        unknown_indices: List[int],
        target_idx: int,
        encoder_length: int = 48,
        decoder_length: int = 24,
        static_indices: List[int] = None
    ):
        self.data = data.astype(np.float32)
        self.known_indices = known_indices
        self.unknown_indices = unknown_indices
        self.target_idx = target_idx
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.static_indices = static_indices or []

        # 유효한 샘플 수 계산
        # 시퀀스: [t-encoder_length : t+decoder_length]
        total_length = encoder_length + decoder_length
        self.n_samples = len(data) - total_length + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"데이터가 너무 짧습니다. "
                f"필요: {total_length}, 실제: {len(data)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        인덱스에 해당하는 샘플 반환

        Returns:
            Tuple: (known_inputs, unknown_inputs, targets, static_inputs)
                - known_inputs: (encoder_len + decoder_len, num_known, 1)
                - unknown_inputs: (encoder_len, num_unknown, 1)
                - targets: (decoder_len,)
                - static_inputs: (num_static, 1) or None
        """
        # 시작/종료 인덱스 계산
        encoder_start = idx
        encoder_end = idx + self.encoder_length
        decoder_end = encoder_end + self.decoder_length

        # Known features: 전체 시퀀스 (encoder + decoder)
        # Shape: (enc_len + dec_len, num_known, 1)
        known_data = self.data[encoder_start:decoder_end, self.known_indices]
        known_inputs = torch.from_numpy(known_data).unsqueeze(-1)

        # Unknown features: encoder 부분만 (미래는 알 수 없음)
        # Shape: (enc_len, num_unknown, 1)
        unknown_data = self.data[encoder_start:encoder_end, self.unknown_indices]
        unknown_inputs = torch.from_numpy(unknown_data).unsqueeze(-1)

        # Targets: decoder 부분의 타겟 값
        # Shape: (dec_len,)
        targets = torch.from_numpy(
            self.data[encoder_end:decoder_end, self.target_idx].copy()
        )

        # Static features (선택적)
        if self.static_indices:
            # 첫 번째 시점의 static 값 사용
            static_data = self.data[encoder_start, self.static_indices]
            static_inputs = torch.from_numpy(static_data).unsqueeze(-1)
        else:
            static_inputs = None

        return known_inputs, unknown_inputs, targets, static_inputs


# ============================================================
# TFT Trainer
# ============================================================

class TFTTrainer:
    """
    Temporal Fusion Transformer 학습 Trainer

    TFT 모델의 학습, 검증, 테스트를 관리합니다.
    QuantileLoss를 기본으로 사용하며, MSE 호환 모드도 지원합니다.

    Args:
        model: TFT 모델
        optimizer: 옵티마이저
        device: 학습 디바이스
        scheduler: 학습률 스케줄러 (optional)
        grad_clip: Gradient clipping 값 (optional)
        checkpoint_dir: 체크포인트 저장 디렉토리
        quantiles: Quantile 손실에 사용할 분위수 (default: [0.1, 0.5, 0.9])

    Example:
        >>> model = TemporalFusionTransformer(...)
        >>> trainer = TFTTrainer(model, optimizer, device)
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        grad_clip: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None,
        quantiles: List[float] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        # Quantile Loss
        self.criterion = QuantileLoss(quantiles=self.quantiles)

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        단일 에포크 학습

        Returns:
            Tuple[float, float]: (quantile_loss, mse_loss)
        """
        self.model.train()
        total_q_loss = 0.0
        total_mse_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            known, unknown, targets, static = batch

            # 디바이스로 이동
            known = known.to(self.device)
            unknown = unknown.to(self.device)
            targets = targets.to(self.device)
            if static is not None:
                static = static.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(
                known_inputs=known,
                unknown_inputs=unknown,
                static_inputs=static
            )
            predictions = output['predictions']  # (batch, decoder_len, num_quantiles)

            # Quantile Loss
            q_loss = self.criterion(predictions, targets)
            q_loss.backward()

            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            self.optimizer.step()

            # MSE 계산 (median 사용)
            median_idx = len(self.quantiles) // 2
            median_pred = predictions[:, :, median_idx]
            mse_loss = torch.mean((median_pred - targets) ** 2)

            total_q_loss += q_loss.item()
            total_mse_loss += mse_loss.item()
            n_batches += 1

        return total_q_loss / n_batches, total_mse_loss / n_batches

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        단일 에포크 검증

        Returns:
            Tuple[float, float]: (quantile_loss, mse_loss)
        """
        self.model.eval()
        total_q_loss = 0.0
        total_mse_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                known, unknown, targets, static = batch

                known = known.to(self.device)
                unknown = unknown.to(self.device)
                targets = targets.to(self.device)
                if static is not None:
                    static = static.to(self.device)

                output = self.model(
                    known_inputs=known,
                    unknown_inputs=unknown,
                    static_inputs=static
                )
                predictions = output['predictions']

                # Quantile Loss
                q_loss = self.criterion(predictions, targets)

                # MSE (median)
                median_idx = len(self.quantiles) // 2
                median_pred = predictions[:, :, median_idx]
                mse_loss = torch.mean((median_pred - targets) ** 2)

                total_q_loss += q_loss.item()
                total_mse_loss += mse_loss.item()
                n_batches += 1

        return total_q_loss / n_batches, total_mse_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        verbose: int = 1,
        log_interval: int = 10
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

        Returns:
            TrainingHistory: 학습 히스토리
        """
        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=patience, mode='min')

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"TFT Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"Epochs: {epochs}, Patience: {patience}")
            print(f"Device: {self.device}")
            print(f"Quantiles: {self.quantiles}")
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            history.start_epoch()

            # Training
            train_q_loss, train_mse = self._train_epoch(train_loader)

            # Validation
            val_q_loss, val_mse = self._validate_epoch(val_loader)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history (using quantile loss as main loss)
            history.update(
                epoch=epoch,
                train_loss=train_q_loss,
                val_loss=val_q_loss,
                lr=current_lr,
                metrics={
                    'train_mse': train_mse,
                    'val_mse': val_mse
                }
            )

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_q_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            improved = early_stopping(val_q_loss, self.model)

            # Logging
            if verbose >= 1 and (epoch % log_interval == 0 or epoch == 1 or early_stopping.early_stop):
                status = "✓" if improved else ""
                print(f"Epoch {epoch:4d} | "
                      f"Train Q: {train_q_loss:.6f} MSE: {train_mse:.6f} | "
                      f"Val Q: {val_q_loss:.6f} MSE: {val_mse:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"ES: {early_stopping.counter}/{patience} {status}")

            # Early stopping
            if early_stopping.early_stop:
                if verbose >= 1:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
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
            Dict: 평가 결과
        """
        self.model.eval()
        total_q_loss = 0.0
        total_mse_loss = 0.0
        n_batches = 0

        all_predictions = []
        all_targets = []
        all_attention = []

        with torch.no_grad():
            for batch in test_loader:
                known, unknown, targets, static = batch

                known = known.to(self.device)
                unknown = unknown.to(self.device)
                targets = targets.to(self.device)
                if static is not None:
                    static = static.to(self.device)

                output = self.model(
                    known_inputs=known,
                    unknown_inputs=unknown,
                    static_inputs=static,
                    return_attention=True
                )
                predictions = output['predictions']

                # Losses
                q_loss = self.criterion(predictions, targets)
                median_idx = len(self.quantiles) // 2
                median_pred = predictions[:, :, median_idx]
                mse_loss = torch.mean((median_pred - targets) ** 2)

                total_q_loss += q_loss.item()
                total_mse_loss += mse_loss.item()
                n_batches += 1

                if return_predictions:
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    if 'attention_weights' in output:
                        all_attention.append(output['attention_weights'].cpu().numpy())

        result = {
            'test_quantile_loss': total_q_loss / n_batches,
            'test_mse_loss': total_mse_loss / n_batches,
            'test_rmse': np.sqrt(total_mse_loss / n_batches)
        }

        if return_predictions:
            result['predictions'] = np.concatenate(all_predictions, axis=0)
            result['targets'] = np.concatenate(all_targets, axis=0)
            if all_attention:
                result['attention_weights'] = np.concatenate(all_attention, axis=0)

        return result

    def predict(
        self,
        data_loader: DataLoader,
        return_intervals: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        예측 수행

        Args:
            data_loader: 데이터 로더
            return_intervals: 예측 구간 반환 여부

        Returns:
            Dict: {
                'median': 중앙값 예측,
                'lower': 하위 quantile,
                'upper': 상위 quantile
            }
        """
        self.model.eval()
        predictions_list = []

        with torch.no_grad():
            for batch in data_loader:
                known, unknown, _, static = batch

                known = known.to(self.device)
                unknown = unknown.to(self.device)
                if static is not None:
                    static = static.to(self.device)

                output = self.model(
                    known_inputs=known,
                    unknown_inputs=unknown,
                    static_inputs=static
                )
                predictions_list.append(output['predictions'].cpu().numpy())

        predictions = np.concatenate(predictions_list, axis=0)

        result = {
            'all_quantiles': predictions
        }

        if return_intervals and len(self.quantiles) >= 3:
            result['lower'] = predictions[:, :, 0]
            result['median'] = predictions[:, :, len(self.quantiles) // 2]
            result['upper'] = predictions[:, :, -1]

        return result

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        history: TrainingHistory = None,
        additional_info: Dict = None
    ) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'quantiles': self.quantiles,
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
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint


# ============================================================
# DataLoader 생성 함수
# ============================================================

def create_tft_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    known_indices: List[int],
    unknown_indices: List[int],
    target_idx: int,
    encoder_length: int = 48,
    decoder_length: int = 24,
    batch_size: int = 64,
    static_indices: List[int] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    TFT용 Train/Val/Test DataLoader 생성

    Args:
        train_data: 학습 데이터 (정규화됨)
        val_data: 검증 데이터 (정규화됨)
        test_data: 테스트 데이터 (정규화됨)
        known_indices: Known 피처 인덱스
        unknown_indices: Unknown 피처 인덱스
        target_idx: 타겟 인덱스
        encoder_length: Encoder 시퀀스 길이
        decoder_length: Decoder 시퀀스 길이
        batch_size: 배치 크기
        static_indices: Static 피처 인덱스
        num_workers: 데이터 로딩 워커 수

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]
    """
    # Custom collate function to handle None static inputs
    def collate_fn(batch):
        known = torch.stack([item[0] for item in batch])
        unknown = torch.stack([item[1] for item in batch])
        targets = torch.stack([item[2] for item in batch])

        if batch[0][3] is not None:
            static = torch.stack([item[3] for item in batch])
        else:
            static = None

        return known, unknown, targets, static

    train_dataset = TFTDataset(
        train_data, known_indices, unknown_indices, target_idx,
        encoder_length, decoder_length, static_indices
    )
    val_dataset = TFTDataset(
        val_data, known_indices, unknown_indices, target_idx,
        encoder_length, decoder_length, static_indices
    )
    test_dataset = TFTDataset(
        test_data, known_indices, unknown_indices, target_idx,
        encoder_length, decoder_length, static_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 통합 파이프라인 함수
# ============================================================

def prepare_tft_data_pipeline(
    data_path: Union[str, Path],
    feature_config: TFTFeatureConfig = None,
    encoder_length: int = 48,
    decoder_length: int = 24,
    batch_size: int = 64,
    train_end: str = '2022-12-31 23:00:00',
    val_end: str = '2023-06-30 23:00:00',
    scaler_save_path: Optional[Union[str, Path]] = None
) -> Dict:
    """
    TFT 데이터 준비 전체 파이프라인

    파이프라인:
    1. 데이터 로드
    2. 피처 분류 (Known/Unknown)
    3. Train/Val/Test 분할
    4. 정규화 (학습 데이터 기준)
    5. DataLoader 생성

    Args:
        data_path: 데이터 파일 경로
        feature_config: 피처 설정 (None이면 기본값)
        encoder_length: Encoder 시퀀스 길이
        decoder_length: Decoder 시퀀스 길이
        batch_size: 배치 크기
        train_end: 학습 데이터 종료 시점
        val_end: 검증 데이터 종료 시점
        scaler_save_path: 스케일러 저장 경로

    Returns:
        Dict: {
            'train_loader': DataLoader,
            'val_loader': DataLoader,
            'test_loader': DataLoader,
            'scaler': TimeSeriesScaler,
            'feature_config': TFTFeatureConfig,
            'known_indices': List[int],
            'unknown_indices': List[int],
            'target_idx': int,
            'num_known': int,
            'num_unknown': int,
            'device': torch.device
        }
    """
    print("=" * 60)
    print("MODEL-006: TFT Data Pipeline")
    print("=" * 60)

    if feature_config is None:
        feature_config = TFTFeatureConfig()

    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  - Shape: {df.shape}")
    print(f"  - Date range: {df.index.min()} ~ {df.index.max()}")

    # 2. 피처 분류
    print("\n[Step 2] Classifying features...")
    available = feature_config.get_available_features(df.columns.tolist())
    known_features = available['known']
    unknown_features = available['unknown']
    static_features = available['static']

    print(f"  - Known features: {len(known_features)}")
    print(f"  - Unknown features: {len(unknown_features)}")
    print(f"  - Static features: {len(static_features)}")

    # 모든 피처 결합 (unknown 먼저, 그 다음 known)
    # 이렇게 하면 target_idx = 0
    all_features = unknown_features + known_features

    # 인덱스 계산
    unknown_indices = list(range(len(unknown_features)))
    known_indices = list(range(len(unknown_features), len(all_features)))
    target_idx = 0  # 첫 번째 unknown 피처가 타겟

    print(f"  - Target: {unknown_features[0]} (idx={target_idx})")

    # 3. 데이터 분할
    print("\n[Step 3] Splitting data...")
    train_df, val_df, test_df = split_data_by_time(df, train_end, val_end)
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")

    # 4. 피처 선택 및 결측치 처리
    print("\n[Step 4] Preparing features...")
    train_data = train_df[all_features].ffill().bfill().fillna(0).values
    val_data = val_df[all_features].ffill().bfill().fillna(0).values
    test_data = test_df[all_features].ffill().bfill().fillna(0).values

    # 5. 정규화
    print("\n[Step 5] Normalizing data...")
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    print(f"  - Train range: [{train_scaled.min():.3f}, {train_scaled.max():.3f}]")

    if scaler_save_path:
        scaler.save(scaler_save_path)
        print(f"  - Scaler saved to: {scaler_save_path}")

    # 6. DataLoader 생성
    print("\n[Step 6] Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_tft_dataloaders(
        train_scaled, val_scaled, test_scaled,
        known_indices=known_indices,
        unknown_indices=unknown_indices,
        target_idx=target_idx,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        batch_size=batch_size
    )
    print(f"  - Encoder length: {encoder_length}")
    print(f"  - Decoder length: {decoder_length}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # 7. 디바이스 확인
    device = get_device()
    print(f"\n[Device] {device}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'feature_config': feature_config,
        'feature_names': all_features,
        'known_indices': known_indices,
        'unknown_indices': unknown_indices,
        'target_idx': target_idx,
        'num_known': len(known_features),
        'num_unknown': len(unknown_features),
        'device': device
    }


# ============================================================
# 테스트 함수
# ============================================================

def test_tft_dataset():
    """TFTDataset 테스트"""
    print("\nTesting TFTDataset...")

    # 합성 데이터
    np.random.seed(42)
    n_samples = 500
    n_known = 8
    n_unknown = 10
    n_features = n_known + n_unknown

    data = np.random.randn(n_samples, n_features).astype(np.float32)
    known_indices = list(range(n_unknown, n_features))
    unknown_indices = list(range(n_unknown))
    target_idx = 0

    encoder_length = 48
    decoder_length = 24

    dataset = TFTDataset(
        data=data,
        known_indices=known_indices,
        unknown_indices=unknown_indices,
        target_idx=target_idx,
        encoder_length=encoder_length,
        decoder_length=decoder_length
    )

    print(f"  Dataset size: {len(dataset)}")

    known, unknown, targets, static = dataset[0]
    print(f"  Known shape: {known.shape}")  # (72, 8, 1)
    print(f"  Unknown shape: {unknown.shape}")  # (48, 10, 1)
    print(f"  Targets shape: {targets.shape}")  # (24,)
    print(f"  Static: {static}")

    assert known.shape == (encoder_length + decoder_length, n_known, 1)
    assert unknown.shape == (encoder_length, n_unknown, 1)
    assert targets.shape == (decoder_length,)

    print("  TFTDataset test passed!")


def test_tft_trainer():
    """TFTTrainer 테스트 (합성 데이터)"""
    print("\nTesting TFTTrainer...")

    device = get_device()

    # 합성 데이터
    np.random.seed(42)
    n_samples = 200
    n_known = 8
    n_unknown = 10
    n_features = n_known + n_unknown

    data = np.random.randn(n_samples, n_features).astype(np.float32)
    known_indices = list(range(n_unknown, n_features))
    unknown_indices = list(range(n_unknown))
    target_idx = 0

    encoder_length = 24
    decoder_length = 12
    batch_size = 16

    # DataLoader 생성
    train_loader, val_loader, _ = create_tft_dataloaders(
        data, data, data,
        known_indices, unknown_indices, target_idx,
        encoder_length, decoder_length, batch_size
    )

    # 모델 생성
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

    # Trainer 생성
    trainer = TFTTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        grad_clip=1.0
    )

    # 짧은 학습 테스트
    history = trainer.fit(
        train_loader, val_loader,
        epochs=3,
        patience=5,
        verbose=1,
        log_interval=1
    )

    print(f"  Final train loss: {history.history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

    # 예측 테스트
    result = trainer.predict(val_loader)
    print(f"  Prediction shape: {result['median'].shape}")

    print("  TFTTrainer test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL-006: TFT Training Pipeline Test")
    print("=" * 60)

    test_tft_dataset()
    test_tft_trainer()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
