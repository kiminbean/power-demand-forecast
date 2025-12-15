"""
MODEL-001: PyTorch Dataset 및 DataLoader 구현

시계열 전력 수요 예측을 위한 데이터 파이프라인

주요 기능:
1. TimeSeriesDataset: PyTorch Dataset 클래스
2. TimeSeriesScaler: Min-Max 정규화 (데이터 누수 방지)
3. 시퀀스 생성: 48시간 입력 → 1~24시간 예측
4. Train/Val/Test 분할: 시간 기반 분할

하이퍼파라미터 (feature_list.json 기준):
- sequence_length: 48 (입력 시퀀스 길이)
- batch_size: 32
- prediction_horizons: [1, 6, 12, 24]
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json


# ============================================================
# 디바이스 설정 (M1 MPS 지원)
# ============================================================

def get_device() -> torch.device:
    """
    최적의 디바이스를 반환합니다.

    우선순위: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        torch.device: 사용 가능한 최적 디바이스
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ============================================================
# 스케일러 클래스 (Min-Max 정규화)
# ============================================================

class TimeSeriesScaler:
    """
    시계열 데이터를 위한 Min-Max 스케일러

    데이터 누수 방지:
    - fit()은 학습 데이터에만 적용
    - transform()은 fit된 파라미터로 모든 데이터에 적용

    Attributes:
        min_vals: 각 특성의 최소값 (학습 데이터 기준)
        max_vals: 각 특성의 최대값 (학습 데이터 기준)
        feature_names: 특성 이름 목록
    """

    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'TimeSeriesScaler':
        """
        학습 데이터로 스케일러를 학습합니다.

        Args:
            data: 학습 데이터 (DataFrame 또는 ndarray)

        Returns:
            self: 학습된 스케일러
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            data = data.values

        self.min_vals = np.nanmin(data, axis=0)
        self.max_vals = np.nanmax(data, axis=0)

        # 0으로 나누기 방지 (상수 컬럼 처리)
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        데이터를 정규화합니다.

        Args:
            data: 변환할 데이터

        Returns:
            np.ndarray: 정규화된 데이터 (0~1 범위)

        Raises:
            ValueError: fit()이 호출되지 않은 경우
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        if isinstance(data, pd.DataFrame):
            data = data.values

        return (data - self.min_vals) / self.range_vals

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        데이터를 학습하고 변환합니다.

        Args:
            data: 학습 및 변환할 데이터

        Returns:
            np.ndarray: 정규화된 데이터
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        정규화된 데이터를 원래 스케일로 복원합니다.

        Args:
            data: 정규화된 데이터

        Returns:
            np.ndarray: 원래 스케일의 데이터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")

        return data * self.range_vals + self.min_vals

    def inverse_transform_target(
        self,
        data: np.ndarray,
        target_idx: int = 0
    ) -> np.ndarray:
        """
        타겟 변수만 역변환합니다.

        Args:
            data: 정규화된 타겟 데이터
            target_idx: 타겟 변수의 인덱스

        Returns:
            np.ndarray: 원래 스케일의 타겟 데이터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")

        return data * self.range_vals[target_idx] + self.min_vals[target_idx]

    def save(self, path: Union[str, Path]) -> None:
        """스케일러 파라미터를 저장합니다."""
        params = {
            'min_vals': self.min_vals.tolist() if self.min_vals is not None else None,
            'max_vals': self.max_vals.tolist() if self.max_vals is not None else None,
            'range_vals': self.range_vals.tolist() if hasattr(self, 'range_vals') else None,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        with open(path, 'w') as f:
            json.dump(params, f)

    def load(self, path: Union[str, Path]) -> 'TimeSeriesScaler':
        """저장된 스케일러 파라미터를 로드합니다."""
        with open(path, 'r') as f:
            params = json.load(f)

        self.min_vals = np.array(params['min_vals']) if params['min_vals'] else None
        self.max_vals = np.array(params['max_vals']) if params['max_vals'] else None
        self.range_vals = np.array(params['range_vals']) if params['range_vals'] else None
        self.feature_names = params['feature_names']
        self.is_fitted = params['is_fitted']
        return self


# ============================================================
# PyTorch Dataset 클래스
# ============================================================

class TimeSeriesDataset(Dataset):
    """
    시계열 예측을 위한 PyTorch Dataset

    시퀀스 생성:
    - 입력: X[t-seq_len:t] (과거 seq_len 시간의 특성)
    - 출력: y[t+horizon] (미래 horizon 시간 후의 타겟)

    Args:
        data: 입력 데이터 (numpy array, [samples, features])
        target_idx: 타겟 변수의 인덱스 (기본 0)
        seq_length: 입력 시퀀스 길이 (기본 48)
        horizon: 예측 시점 (기본 1, 즉 t+1 예측)

    Attributes:
        data: 정규화된 데이터
        seq_length: 시퀀스 길이
        horizon: 예측 시점
    """

    def __init__(
        self,
        data: np.ndarray,
        target_idx: int = 0,
        seq_length: int = 48,
        horizon: int = 1
    ):
        self.data = data.astype(np.float32)
        self.target_idx = target_idx
        self.seq_length = seq_length
        self.horizon = horizon

        # 유효한 샘플 수 계산
        self.n_samples = len(data) - seq_length - horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"데이터가 너무 짧습니다. "
                f"필요: {seq_length + horizon}, 실제: {len(data)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 (입력, 타겟) 쌍을 반환합니다.

        Args:
            idx: 샘플 인덱스

        Returns:
            Tuple[Tensor, Tensor]: (X, y)
                - X: [seq_length, n_features]
                - y: [1] (scalar target)
        """
        # 입력 시퀀스: [t-seq_len, t-1]
        x_start = idx
        x_end = idx + self.seq_length

        # 타겟: t + horizon - 1 (0-indexed)
        y_idx = x_end + self.horizon - 1

        X = torch.from_numpy(self.data[x_start:x_end])
        y = torch.tensor([self.data[y_idx, self.target_idx]], dtype=torch.float32)

        return X, y


class MultiHorizonDataset(Dataset):
    """
    다중 예측 시점을 위한 PyTorch Dataset

    여러 horizon (1h, 6h, 12h, 24h)을 동시에 예측

    Args:
        data: 입력 데이터
        target_idx: 타겟 변수 인덱스
        seq_length: 입력 시퀀스 길이
        horizons: 예측 시점 리스트 (예: [1, 6, 12, 24])
    """

    def __init__(
        self,
        data: np.ndarray,
        target_idx: int = 0,
        seq_length: int = 48,
        horizons: List[int] = None
    ):
        if horizons is None:
            horizons = [1, 6, 12, 24]

        self.data = data.astype(np.float32)
        self.target_idx = target_idx
        self.seq_length = seq_length
        self.horizons = horizons
        self.max_horizon = max(horizons)

        # 유효한 샘플 수
        self.n_samples = len(data) - seq_length - self.max_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"데이터가 너무 짧습니다. "
                f"필요: {seq_length + self.max_horizon}, 실제: {len(data)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[Tensor, Tensor]: (X, y)
                - X: [seq_length, n_features]
                - y: [n_horizons] (각 horizon의 타겟)
        """
        x_start = idx
        x_end = idx + self.seq_length

        X = torch.from_numpy(self.data[x_start:x_end])

        # 각 horizon에 대한 타겟
        y_values = []
        for h in self.horizons:
            y_idx = x_end + h - 1
            y_values.append(self.data[y_idx, self.target_idx])

        y = torch.tensor(y_values, dtype=torch.float32)

        return X, y


# ============================================================
# 데이터 분할 함수
# ============================================================

def split_data_by_time(
    df: pd.DataFrame,
    train_end: str = '2022-12-31 23:00:00',
    val_end: str = '2023-06-30 23:00:00'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시간 기반으로 데이터를 Train/Val/Test로 분할합니다.

    기본 분할 (feature_list.json 기준):
    - Train: 2013-01-01 ~ 2022-12-31
    - Validation: 2023-01-01 ~ 2023-06-30
    - Test: 2023-07-01 ~ 2024-12-31

    Args:
        df: 입력 DataFrame (datetime 인덱스)
        train_end: 학습 데이터 종료 시점
        val_end: 검증 데이터 종료 시점

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: train, val, test
    """
    train_df = df.loc[:train_end]
    val_df = df.loc[train_end:val_end].iloc[1:]  # 중복 제외
    test_df = df.loc[val_end:].iloc[1:]

    return train_df, val_df, test_df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'power_demand',
    feature_cols: List[str] = None,
    exclude_cols: List[str] = None
) -> Tuple[np.ndarray, List[str], int]:
    """
    모델 입력을 위한 특성을 준비합니다.

    Args:
        df: 입력 DataFrame
        target_col: 타겟 컬럼명
        feature_cols: 사용할 특성 목록 (None이면 모든 숫자형 컬럼)
        exclude_cols: 제외할 컬럼 목록

    Returns:
        Tuple[ndarray, List[str], int]:
            - 데이터 배열
            - 특성 이름 목록
            - 타겟 인덱스
    """
    if exclude_cols is None:
        exclude_cols = []

    # 기본 제외 컬럼 (비숫자형, 결측 많음)
    default_exclude = ['일시', 'clfmAbbrCd', 'lcsCh', 'gndSttCd', 'dmstMtphNo']
    exclude_cols = list(set(exclude_cols + default_exclude))

    if feature_cols is None:
        # 숫자형 컬럼만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]

    # 타겟을 첫 번째 컬럼으로
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    feature_cols = [target_col] + feature_cols

    # 결측치 처리
    data_df = df[feature_cols].copy()

    # 결측치 보간 (forward fill + backward fill)
    data_df = data_df.ffill().bfill()

    # 여전히 NaN이 있으면 0으로
    data_df = data_df.fillna(0)

    target_idx = 0  # 타겟은 첫 번째 컬럼

    return data_df.values, feature_cols, target_idx


# ============================================================
# DataLoader 생성 함수
# ============================================================

def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    target_idx: int = 0,
    seq_length: int = 48,
    horizon: int = 1,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train/Val/Test DataLoader를 생성합니다.

    Args:
        train_data: 학습 데이터 (정규화됨)
        val_data: 검증 데이터 (정규화됨)
        test_data: 테스트 데이터 (정규화됨)
        target_idx: 타겟 변수 인덱스
        seq_length: 시퀀스 길이
        horizon: 예측 시점
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
        pin_memory: 메모리 고정 여부

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test 로더
    """
    train_dataset = TimeSeriesDataset(
        train_data, target_idx, seq_length, horizon
    )
    val_dataset = TimeSeriesDataset(
        val_data, target_idx, seq_length, horizon
    )
    test_dataset = TimeSeriesDataset(
        test_data, target_idx, seq_length, horizon
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def create_multi_horizon_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    target_idx: int = 0,
    seq_length: int = 48,
    horizons: List[int] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    다중 예측 시점용 DataLoader를 생성합니다.

    Args:
        horizons: 예측 시점 리스트 (기본: [1, 6, 12, 24])

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test 로더
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    train_dataset = MultiHorizonDataset(
        train_data, target_idx, seq_length, horizons
    )
    val_dataset = MultiHorizonDataset(
        val_data, target_idx, seq_length, horizons
    )
    test_dataset = MultiHorizonDataset(
        test_data, target_idx, seq_length, horizons
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 통합 파이프라인 함수
# ============================================================

def prepare_data_pipeline(
    data_path: Union[str, Path],
    target_col: str = 'power_demand',
    feature_cols: List[str] = None,
    seq_length: int = 48,
    horizon: int = 1,
    batch_size: int = 32,
    train_end: str = '2022-12-31 23:00:00',
    val_end: str = '2023-06-30 23:00:00',
    scaler_save_path: Optional[Union[str, Path]] = None
) -> Dict:
    """
    데이터 준비 전체 파이프라인을 실행합니다.

    파이프라인:
    1. 데이터 로드
    2. 특성 선택 및 전처리
    3. Train/Val/Test 분할
    4. 정규화 (학습 데이터 기준)
    5. DataLoader 생성

    Args:
        data_path: 데이터 파일 경로
        target_col: 타겟 컬럼명
        feature_cols: 사용할 특성 목록
        seq_length: 시퀀스 길이
        horizon: 예측 시점
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
            'feature_names': List[str],
            'target_idx': int,
            'n_features': int,
            'device': torch.device
        }
    """
    print("=" * 60)
    print("MODEL-001: Data Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  - Shape: {df.shape}")
    print(f"  - Date range: {df.index.min()} ~ {df.index.max()}")

    # 2. 데이터 분할
    print("\n[Step 2] Splitting data...")
    train_df, val_df, test_df = split_data_by_time(df, train_end, val_end)
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")

    # 3. 특성 준비
    print("\n[Step 3] Preparing features...")
    train_data, feature_names, target_idx = prepare_features(
        train_df, target_col, feature_cols
    )
    val_data, _, _ = prepare_features(val_df, target_col, feature_cols)
    test_data, _, _ = prepare_features(test_df, target_col, feature_cols)
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Target: {feature_names[target_idx]} (idx={target_idx})")

    # 4. 정규화
    print("\n[Step 4] Normalizing data...")
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    print(f"  - Train range: [{train_scaled.min():.3f}, {train_scaled.max():.3f}]")

    if scaler_save_path:
        scaler.save(scaler_save_path)
        print(f"  - Scaler saved to: {scaler_save_path}")

    # 5. DataLoader 생성
    print("\n[Step 5] Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_scaled, val_scaled, test_scaled,
        target_idx=target_idx,
        seq_length=seq_length,
        horizon=horizon,
        batch_size=batch_size
    )
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # 6. 디바이스 확인
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
        'feature_names': feature_names,
        'target_idx': target_idx,
        'n_features': len(feature_names),
        'device': device
    }


# ============================================================
# 데모 및 테스트
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

    print("=" * 60)
    print("MODEL-001: Dataset Demo")
    print("=" * 60)

    # 프로젝트 루트
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data/processed/jeju_hourly_merged.csv"

    if data_path.exists():
        # 전체 파이프라인 실행
        result = prepare_data_pipeline(
            data_path=data_path,
            seq_length=48,
            horizon=1,
            batch_size=32
        )

        # 샘플 배치 확인
        print("\n=== Sample Batch ===")
        train_loader = result['train_loader']
        for X, y in train_loader:
            print(f"X shape: {X.shape}")  # [batch, seq_len, features]
            print(f"y shape: {y.shape}")  # [batch, 1]
            print(f"X dtype: {X.dtype}")
            print(f"y dtype: {y.dtype}")
            break

        # 디바이스로 이동 테스트
        device = result['device']
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            print(f"\nTensor on device: {X.device}")
            break
    else:
        print(f"Data file not found: {data_path}")
        print("Running synthetic data test...")

        # 합성 데이터 테스트
        np.random.seed(42)
        synthetic_data = np.random.randn(1000, 10).astype(np.float32)

        dataset = TimeSeriesDataset(synthetic_data, seq_length=48, horizon=1)
        print(f"\nSynthetic dataset: {len(dataset)} samples")

        X, y = dataset[0]
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
