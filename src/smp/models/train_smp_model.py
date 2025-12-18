"""
SMP LSTM 모델 학습 스크립트
===========================

실제 KPX SMP 데이터로 LSTM 모델을 학습합니다.

Usage:
    python -m src.smp.models.train_smp_model

Author: Claude Code
Date: 2025-12
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.smp.models.smp_lstm import SMPLSTMModel, get_device, model_summary

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SMPDataset(Dataset):
    """SMP 시계열 데이터셋

    Args:
        data: SMP 시계열 데이터 (정규화된 numpy array)
        input_hours: 입력 시퀀스 길이
        output_hours: 출력 시퀀스 길이
        step: 슬라이딩 윈도우 스텝
    """

    def __init__(
        self,
        data: np.ndarray,
        input_hours: int = 24,
        output_hours: int = 24,
        step: int = 1
    ):
        self.data = torch.FloatTensor(data)
        self.input_hours = input_hours
        self.output_hours = output_hours
        self.step = step

        # 샘플 인덱스 생성
        total_len = len(data)
        self.indices = []
        for i in range(0, total_len - input_hours - output_hours + 1, step):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start:start + self.input_hours]
        y = self.data[start + self.input_hours:start + self.input_hours + self.output_hours, 0]  # SMP only
        return x, y


def load_smp_data(data_path: str) -> pd.DataFrame:
    """SMP 데이터 로드 및 전처리

    Args:
        data_path: CSV 파일 경로

    Returns:
        전처리된 DataFrame
    """
    df = pd.read_csv(data_path)

    # 0이 아닌 유효 데이터만 필터링
    df = df[df['smp_mainland'] > 0].copy()

    # 24:00 → 00:00 변환 (다음날로 처리)
    def fix_hour_24(timestamp):
        if ' 24:00' in str(timestamp):
            # 24:00을 다음날 00:00으로 변환
            date_part = str(timestamp).replace(' 24:00', '')
            dt = pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return dt
        return pd.to_datetime(timestamp)

    df['datetime'] = df['timestamp'].apply(fix_hour_24)

    # 시간순 정렬
    df = df.sort_values('datetime').reset_index(drop=True)

    logger.info(f"데이터 로드 완료: {len(df)}건")
    logger.info(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    return df


def create_features(df: pd.DataFrame) -> np.ndarray:
    """피처 엔지니어링

    Args:
        df: SMP DataFrame

    Returns:
        피처 배열 (n_samples, n_features)
    """
    features = []

    # 기본 SMP 피처
    features.append(df['smp_mainland'].values)  # 육지 SMP
    features.append(df['smp_jeju'].values)      # 제주 SMP
    features.append(df['smp_max'].values)       # 최고가
    features.append(df['smp_min'].values)       # 최저가

    # 시간 피처
    hour = df['hour'].values
    features.append(np.sin(2 * np.pi * hour / 24))  # 시간 사인
    features.append(np.cos(2 * np.pi * hour / 24))  # 시간 코사인

    # 요일 피처 (datetime에서 추출)
    day_of_week = df['datetime'].dt.dayofweek.values
    features.append(np.sin(2 * np.pi * day_of_week / 7))  # 요일 사인
    features.append(np.cos(2 * np.pi * day_of_week / 7))  # 요일 코사인

    # 주말 여부
    features.append((day_of_week >= 5).astype(float))  # 주말 = 1

    # SMP 변화율 (lag features)
    smp = df['smp_mainland'].values
    smp_diff = np.diff(smp, prepend=smp[0])
    features.append(smp_diff)  # SMP 변화량

    # 이동 평균 (짧은 데이터에 맞춤)
    smp_ma3 = pd.Series(smp).rolling(3, min_periods=1).mean().values
    smp_ma6 = pd.Series(smp).rolling(6, min_periods=1).mean().values
    features.append(smp_ma3)
    features.append(smp_ma6)

    # 스택
    feature_array = np.column_stack(features)

    logger.info(f"피처 생성 완료: {feature_array.shape}")

    return feature_array


def normalize_data(
    data: np.ndarray,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, MinMaxScaler]:
    """데이터 정규화

    Args:
        data: 입력 데이터
        scaler: 기존 스케일러 (없으면 새로 생성)

    Returns:
        정규화된 데이터, 스케일러
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)

    return normalized, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    device: torch.device = None
) -> Dict[str, Any]:
    """모델 학습

    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        epochs: 최대 에폭 수
        learning_rate: 학습률
        patience: Early stopping patience
        device: 학습 디바이스

    Returns:
        학습 결과 딕셔너리
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_mape': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    logger.info(f"학습 시작 (device: {device})")
    logger.info(f"학습 샘플: {len(train_loader.dataset)}, 검증 샘플: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mape = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

                # MAPE 계산 (정규화된 값 기준)
                mape = torch.mean(torch.abs((batch_y - output) / (batch_y + 1e-8))) * 100
                val_mape += mape.item()

        val_loss /= len(val_loader)
        val_mape /= len(val_loader)

        # Learning rate 조정
        scheduler.step(val_loss)

        # History 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mape'].append(val_mape)

        # 로깅
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAPE: {val_mape:.2f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Best 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device = None
) -> Dict[str, float]:
    """모델 평가

    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        scaler: 역정규화용 스케일러
        device: 디바이스

    Returns:
        평가 지표 딕셔너리
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)

            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 역정규화 (첫 번째 피처가 SMP)
    # 예측값과 실제값 역정규화
    smp_min = scaler.data_min_[0]
    smp_max = scaler.data_max_[0]
    smp_range = smp_max - smp_min

    preds_real = preds * smp_range + smp_min
    targets_real = targets * smp_range + smp_min

    # 평가 지표
    mae = np.mean(np.abs(preds_real - targets_real))
    rmse = np.sqrt(np.mean((preds_real - targets_real) ** 2))
    mape = np.mean(np.abs((targets_real - preds_real) / (targets_real + 1e-8))) * 100

    # R² Score
    ss_res = np.sum((targets_real - preds_real) ** 2)
    ss_tot = np.sum((targets_real - np.mean(targets_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }


def save_model(
    model: nn.Module,
    scaler: MinMaxScaler,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str
):
    """모델 저장

    Args:
        model: 저장할 모델
        scaler: 스케일러
        metrics: 평가 지표
        config: 모델 설정
        output_dir: 저장 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    model_path = output_path / 'smp_lstm_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, model_path)

    # 스케일러 저장
    scaler_path = output_path / 'smp_scaler.npy'
    np.save(scaler_path, {
        'min_': scaler.data_min_,
        'max_': scaler.data_max_,
        'scale_': scaler.scale_,
        'data_min_': scaler.data_min_,
        'data_max_': scaler.data_max_,
        'feature_range': scaler.feature_range
    })

    # 메트릭 저장
    metrics_path = output_path / 'smp_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"모델 저장 완료: {output_path}")


def main():
    """메인 학습 함수"""
    print("=" * 60)
    print("SMP LSTM 모델 학습")
    print("=" * 60)

    # 설정
    config = {
        'data_path': 'data/smp/smp_history_real.csv',
        'output_dir': 'models/smp',
        'input_hours': 24,      # 24시간 입력
        'output_hours': 24,     # 24시간 예측
        'hidden_size': 64,      # 작은 데이터셋에 맞춤
        'num_layers': 2,
        'dropout': 0.3,         # 강한 regularization
        'bidirectional': True,
        'batch_size': 8,
        'epochs': 150,
        'learning_rate': 0.001,
        'patience': 20,
        'test_size': 0.2,
        'val_size': 0.2,
    }

    print("\n📋 설정:")
    for k, v in config.items():
        print(f"   {k}: {v}")

    # 1. 데이터 로드
    print("\n" + "=" * 60)
    print("1. 데이터 로드")
    print("=" * 60)
    df = load_smp_data(config['data_path'])

    # 2. 피처 생성
    print("\n" + "=" * 60)
    print("2. 피처 엔지니어링")
    print("=" * 60)
    features = create_features(df)
    print(f"피처 수: {features.shape[1]}")

    # 3. 정규화
    print("\n" + "=" * 60)
    print("3. 데이터 정규화")
    print("=" * 60)
    normalized_data, scaler = normalize_data(features)
    print(f"정규화 완료: {normalized_data.shape}")

    # 4. 데이터셋 분할
    print("\n" + "=" * 60)
    print("4. 데이터셋 분할")
    print("=" * 60)

    # 시계열이므로 순차 분할
    n_samples = len(normalized_data)
    train_end = int(n_samples * (1 - config['test_size'] - config['val_size']))
    val_end = int(n_samples * (1 - config['test_size']))

    train_data = normalized_data[:train_end]
    val_data = normalized_data[train_end:val_end]
    test_data = normalized_data[val_end:]

    print(f"학습: {len(train_data)}, 검증: {len(val_data)}, 테스트: {len(test_data)}")

    # 5. 데이터로더 생성 (작은 데이터셋에 맞춤)
    # 전체 데이터로 하나의 데이터셋 생성
    full_dataset = SMPDataset(normalized_data, config['input_hours'], config['output_hours'], step=1)
    n_samples = len(full_dataset)
    print(f"전체 샘플 수: {n_samples}")

    if n_samples < 20:
        logger.warning("데이터가 매우 적습니다. 교차 검증 없이 학습합니다.")
        train_dataset = full_dataset
        val_dataset = full_dataset
        test_dataset = full_dataset
    else:
        # 시계열 순서대로 분할
        train_size = int(n_samples * 0.7)
        val_size = int(n_samples * 0.15)

        # 인덱스 기반 분할
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, n_samples))

        # Subset으로 분할
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices) if val_indices else train_dataset
        test_dataset = Subset(full_dataset, test_indices) if test_indices else train_dataset

    print(f"학습 샘플: {len(train_dataset)}, 검증 샘플: {len(val_dataset)}, 테스트 샘플: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 6. 모델 생성
    print("\n" + "=" * 60)
    print("5. 모델 생성")
    print("=" * 60)

    input_size = features.shape[1]
    model = SMPLSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        prediction_hours=config['output_hours']
    )

    print(model_summary(model))

    device = get_device()
    print(f"Device: {device}")

    # 7. 모델 학습
    print("\n" + "=" * 60)
    print("6. 모델 학습")
    print("=" * 60)

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device
    )

    print(f"\n학습 완료: {train_result['epochs_trained']} epochs")
    print(f"Best Val Loss: {train_result['best_val_loss']:.6f}")

    # 8. 모델 평가
    print("\n" + "=" * 60)
    print("7. 모델 평가")
    print("=" * 60)

    metrics = evaluate_model(model, test_loader, scaler, device)

    print(f"\n📊 테스트 결과:")
    print(f"   MAE:  {metrics['mae']:.2f} 원/kWh")
    print(f"   RMSE: {metrics['rmse']:.2f} 원/kWh")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   R²:   {metrics['r2']:.4f}")

    # 9. 모델 저장
    print("\n" + "=" * 60)
    print("8. 모델 저장")
    print("=" * 60)

    config['input_size'] = input_size
    save_model(model, scaler, metrics, config, config['output_dir'])

    # 10. 예측 샘플 출력
    print("\n" + "=" * 60)
    print("9. 예측 샘플")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        sample_x, sample_y = test_dataset[0]
        sample_x = sample_x.unsqueeze(0).to(device)
        pred = model(sample_x).cpu().numpy()[0]
        actual = sample_y.numpy()

        # 역정규화
        smp_min = scaler.data_min_[0]
        smp_max = scaler.data_max_[0]
        smp_range = smp_max - smp_min

        pred_real = pred * smp_range + smp_min
        actual_real = actual * smp_range + smp_min

        print("\n시간별 예측 vs 실제 (첫 샘플):")
        print("-" * 50)
        for h in range(min(12, len(pred_real))):  # 처음 12시간만
            error = abs(pred_real[h] - actual_real[h])
            print(f"  {h+1:2d}시: 예측 {pred_real[h]:7.2f} | 실제 {actual_real[h]:7.2f} | 오차 {error:6.2f}")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

    return model, scaler, metrics


if __name__ == "__main__":
    main()
