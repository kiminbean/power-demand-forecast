"""
SMP 시계열 데이터셋
==================

SMP 예측 모델 학습을 위한 PyTorch Dataset

주요 기능:
1. 48시간 입력 → 24시간 타겟 시퀀스 생성
2. 시간/날짜 피처 자동 추가
3. 데이터 누수 방지 (train/val/test 분할)
4. 다중 피처 지원

Author: Claude Code
Date: 2025-12
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SMPScaler:
    """SMP 데이터용 스케일러

    Min-Max 정규화를 사용하며 데이터 누수를 방지합니다.

    Attributes:
        min_vals: 각 피처의 최소값
        max_vals: 각 피처의 최대값
        feature_names: 피처 이름 목록
    """

    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.range_vals: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.target_idx: int = 0

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: str = 'smp_mainland'
    ) -> 'SMPScaler':
        """학습 데이터로 스케일러 학습

        Args:
            data: 학습 데이터
            target_col: 타겟 컬럼명 (역변환용)

        Returns:
            self
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            if target_col in self.feature_names:
                self.target_idx = self.feature_names.index(target_col)
            data = data.values

        self.min_vals = np.nanmin(data, axis=0)
        self.max_vals = np.nanmax(data, axis=0)
        self.range_vals = self.max_vals - self.min_vals

        # 상수 컬럼 처리 (0으로 나누기 방지)
        self.range_vals[self.range_vals == 0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """데이터 정규화 (0~1)"""
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")

        if isinstance(data, pd.DataFrame):
            data = data.values

        return (data - self.min_vals) / self.range_vals

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """학습 및 변환"""
        return self.fit(data, **kwargs).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """역변환 (0~1 → 원래 스케일)"""
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")
        return data * self.range_vals + self.min_vals

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """타겟 변수만 역변환

        Args:
            data: 정규화된 타겟 데이터 (batch, prediction_hours)

        Returns:
            원래 스케일의 타겟 데이터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")

        target_min = self.min_vals[self.target_idx]
        target_range = self.range_vals[self.target_idx]

        return data * target_range + target_min

    def save(self, path: Union[str, Path]):
        """스케일러 저장"""
        import json
        path = Path(path)

        data = {
            'min_vals': self.min_vals.tolist(),
            'max_vals': self.max_vals.tolist(),
            'range_vals': self.range_vals.tolist(),
            'feature_names': self.feature_names,
            'target_idx': self.target_idx,
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SMPScaler':
        """스케일러 로드"""
        import json
        path = Path(path)

        with open(path, 'r') as f:
            data = json.load(f)

        scaler = cls()
        scaler.min_vals = np.array(data['min_vals'])
        scaler.max_vals = np.array(data['max_vals'])
        scaler.range_vals = np.array(data['range_vals'])
        scaler.feature_names = data['feature_names']
        scaler.target_idx = data['target_idx']
        scaler.is_fitted = True

        return scaler


class SMPDataset(Dataset):
    """SMP 시계열 데이터셋

    48시간 입력 시퀀스로 24시간 SMP를 예측하는 데이터셋

    Args:
        data: 시계열 데이터 (DataFrame 또는 ndarray)
        sequence_length: 입력 시퀀스 길이 (default: 48)
        prediction_hours: 예측 시간 수 (default: 24)
        target_col: 타겟 컬럼 이름 (default: 'smp_mainland')
        scaler: 사용할 스케일러 (None이면 내부 생성)
        is_train: 학습 데이터 여부 (스케일러 fit용)

    Example:
        >>> df = pd.read_csv('smp_data.csv')
        >>> dataset = SMPDataset(df, sequence_length=48, prediction_hours=24)
        >>> x, y = dataset[0]
        >>> print(x.shape, y.shape)  # (48, n_features), (24,)
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        sequence_length: int = 48,
        prediction_hours: int = 24,
        target_col: str = 'smp_mainland',
        scaler: Optional[SMPScaler] = None,
        is_train: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_hours = prediction_hours
        self.target_col = target_col

        # DataFrame 처리
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            self.target_idx = self.feature_names.index(target_col) if target_col in self.feature_names else 0
            data = data.values.astype(np.float32)
        else:
            self.feature_names = None
            self.target_idx = 0
            data = data.astype(np.float32)

        # 스케일링
        if scaler is None:
            self.scaler = SMPScaler()
            if is_train:
                self.data = self.scaler.fit_transform(data)
            else:
                raise ValueError("테스트 데이터에는 학습된 스케일러가 필요합니다.")
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(data)

        self.data = self.data.astype(np.float32)

        # 유효한 시퀀스 수 계산
        self.n_samples = len(self.data) - self.sequence_length - self.prediction_hours + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"데이터 길이({len(data)})가 시퀀스({sequence_length}) + "
                f"예측({prediction_hours})보다 작습니다."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터 샘플 반환

        Returns:
            x: (sequence_length, n_features) 입력 시퀀스
            y: (prediction_hours,) 타겟 시퀀스
        """
        # 입력 시퀀스
        x_start = idx
        x_end = idx + self.sequence_length
        x = self.data[x_start:x_end]

        # 타겟 시퀀스 (다음 24시간의 SMP)
        y_start = x_end
        y_end = y_start + self.prediction_hours
        y = self.data[y_start:y_end, self.target_idx]

        return torch.from_numpy(x), torch.from_numpy(y)


class SMPDataModule:
    """SMP 데이터 모듈

    데이터 로딩, 분할, DataLoader 생성을 관리합니다.

    Args:
        data_path: 데이터 파일 경로
        sequence_length: 입력 시퀀스 길이
        prediction_hours: 예측 시간 수
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        target_col: 타겟 컬럼명
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        sequence_length: int = 48,
        prediction_hours: int = 24,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        target_col: str = 'smp_mainland',
        num_workers: int = 0
    ):
        self.sequence_length = sequence_length
        self.prediction_hours = prediction_hours
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_col = target_col
        self.num_workers = num_workers

        # 데이터 로드
        if data is not None:
            self.raw_data = data
        elif data_path is not None:
            self.raw_data = self._load_data(data_path)
        else:
            raise ValueError("data_path 또는 data를 제공해야 합니다.")

        self.scaler = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """데이터 파일 로드"""
        path = Path(path)

        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")

    def setup(self) -> None:
        """데이터 분할 및 데이터셋 생성"""
        n = len(self.raw_data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # 시간순 분할 (데이터 누수 방지)
        train_data = self.raw_data.iloc[:train_end]
        val_data = self.raw_data.iloc[train_end:val_end]
        test_data = self.raw_data.iloc[val_end:]

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # 학습 데이터로 스케일러 학습
        self.train_dataset = SMPDataset(
            train_data,
            sequence_length=self.sequence_length,
            prediction_hours=self.prediction_hours,
            target_col=self.target_col,
            is_train=True
        )
        self.scaler = self.train_dataset.scaler

        # 검증/테스트 데이터셋
        if len(val_data) > self.sequence_length + self.prediction_hours:
            self.val_dataset = SMPDataset(
                val_data,
                sequence_length=self.sequence_length,
                prediction_hours=self.prediction_hours,
                target_col=self.target_col,
                scaler=self.scaler,
                is_train=False
            )

        if len(test_data) > self.sequence_length + self.prediction_hours:
            self.test_dataset = SMPDataset(
                test_data,
                sequence_length=self.sequence_length,
                prediction_hours=self.prediction_hours,
                target_col=self.target_col,
                scaler=self.scaler,
                is_train=False
            )

    def train_dataloader(self) -> DataLoader:
        """학습용 DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """검증용 DataLoader"""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """테스트용 DataLoader"""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def input_size(self) -> int:
        """입력 피처 수"""
        return self.raw_data.shape[1]


def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """시간 피처 추가

    Args:
        df: 데이터프레임
        timestamp_col: 시간 컬럼명

    Returns:
        시간 피처가 추가된 데이터프레임
    """
    df = df.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        ts = df[timestamp_col]
    else:
        # timestamp 컬럼이 없으면 인덱스 사용
        ts = pd.to_datetime(df.index)

    # 주기적 시간 피처 (sin/cos)
    df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)

    # 이진 피처
    df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(float)

    return df


if __name__ == "__main__":
    # 테스트
    print("SMP Dataset Test")
    print("=" * 60)

    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'smp_mainland': 100 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.randn(n_samples) * 5,
        'smp_jeju': 95 + 18 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.randn(n_samples) * 4,
        'demand': 600 + 100 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.randn(n_samples) * 20,
    })

    # 시간 피처 추가
    data = add_time_features(data)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # timestamp 제거 (숫자 피처만)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data_numeric = data[numeric_cols]

    # 데이터 모듈 생성
    dm = SMPDataModule(
        data=data_numeric,
        sequence_length=48,
        prediction_hours=24,
        batch_size=16,
        target_col='smp_mainland'
    )
    dm.setup()

    print(f"\nTrain samples: {len(dm.train_dataset)}")
    print(f"Input size: {dm.input_size}")

    # 샘플 확인
    x, y = dm.train_dataset[0]
    print(f"\nSample X shape: {x.shape}")
    print(f"Sample Y shape: {y.shape}")

    # DataLoader 테스트
    train_loader = dm.train_dataloader()
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
