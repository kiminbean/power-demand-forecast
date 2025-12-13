"""
LSTM Power Demand Forecasting with THI Feature
===============================================
THI(불쾌지수) 변수 통합 및 성능 검증

Model: LSTM (Long Short-Term Memory)
Target: 일별 전력 수요 예측
Features: 기상 데이터 + THI + 시간 파생 변수

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from pathlib import Path
import sys

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.weather_features import calculate_humidity_and_thi

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'data_path': PROJECT_ROOT / 'data' / 'processed' / 'jeju_daily_dataset.csv',
    'sequence_length': 14,  # 14일 시퀀스
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,  # Early stopping
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'random_seed': 42,
}

# Feature columns (THI 미포함 기본 버전)
BASE_FEATURES = [
    'temp_mean', 'temp_max', 'temp_min', 'temp_range',
    'dewpoint_mean', 'sunshine_hours', 'solar_radiation',
    'soil_temp_5cm', 'soil_temp_10cm', 'soil_temp_20cm',
    'CDD', 'HDD', 'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'is_holiday'
]

# THI 포함 버전
THI_FEATURES = BASE_FEATURES + ['humidity', 'THI']

TARGET = 'power_sum'


# ============================================================
# Device Setup (MPS for Apple Silicon)
# ============================================================
def get_device():
    """최적 디바이스 선택 (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    return device


# ============================================================
# Data Loading & Preprocessing
# ============================================================
def load_data(include_thi: bool = True) -> pd.DataFrame:
    """
    데이터 로드 및 THI 계산
    
    Args:
        include_thi: THI 변수 포함 여부
    """
    print(f"\n📂 Loading data from {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG['data_path'], parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    if include_thi:
        print("🔧 Calculating THI and Humidity...")
        df = calculate_humidity_and_thi(df)
    
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    return df


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    sequence_length: int
) -> tuple:
    """
    시계열 시퀀스 데이터 생성
    
    Returns:
        X: (samples, sequence_length, n_features)
        y: (samples,)
    """
    # 결측치 처리
    df = df.dropna(subset=feature_cols + [target_col])
    
    # 스케일링
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features = feature_scaler.fit_transform(df[feature_cols])
    targets = target_scaler.fit_transform(df[[target_col]])
    
    # 시퀀스 생성
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y).flatten()
    
    return X, y, feature_scaler, target_scaler


def split_data(X: np.ndarray, y: np.ndarray) -> dict:
    """시간 순서 기반 데이터 분할"""
    n = len(X)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
    
    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
    }


# ============================================================
# PyTorch Dataset & Model
# ============================================================
class PowerDemandDataset(Dataset):
    """전력 수요 예측 데이터셋"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM 기반 전력 수요 예측 모델"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # Fully connected
        out = self.fc(last_output)
        
        return out.squeeze()


# ============================================================
# Training & Evaluation
# ============================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float
) -> dict:
    """모델 학습"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    
    print(f"\n🚀 Training for {epochs} epochs (patience={patience})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    target_scaler: MinMaxScaler,
    device: torch.device
) -> dict:
    """모델 평가"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # 역스케일링
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'predictions': predictions,
        'actuals': actuals
    }


# ============================================================
# Main Experiment
# ============================================================
def run_experiment(include_thi: bool) -> dict:
    """
    실험 실행
    
    Args:
        include_thi: THI 변수 포함 여부
    """
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    feature_cols = THI_FEATURES if include_thi else BASE_FEATURES
    experiment_name = "WITH_THI" if include_thi else "WITHOUT_THI"
    
    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"   Features: {len(feature_cols)} columns")
    print(f"{'='*60}")
    
    # 데이터 준비
    df = load_data(include_thi=include_thi)
    X, y, feature_scaler, target_scaler = prepare_sequences(
        df, feature_cols, TARGET, CONFIG['sequence_length']
    )
    
    print(f"   Sequence shape: {X.shape}")
    
    # 데이터 분할
    data = split_data(X, y)
    
    # DataLoader 생성
    train_dataset = PowerDemandDataset(data['X_train'], data['y_train'])
    val_dataset = PowerDemandDataset(data['X_val'], data['y_val'])
    test_dataset = PowerDemandDataset(data['X_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # 모델 생성
    device = get_device()
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습
    history = train_model(
        model, train_loader, val_loader, device,
        CONFIG['epochs'], CONFIG['patience'], CONFIG['learning_rate']
    )
    
    # 평가
    metrics = evaluate_model(model, test_loader, target_scaler, device)
    
    print(f"\n📊 {experiment_name} Results:")
    print(f"   RMSE: {metrics['RMSE']:.2f} MW")
    print(f"   MAE:  {metrics['MAE']:.2f} MW")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")
    
    return {
        'name': experiment_name,
        'features': len(feature_cols),
        'metrics': metrics,
        'history': history
    }


def main():
    """메인 실험 실행"""
    print("="*60)
    print("🔬 LSTM Power Demand Forecasting: THI Feature Impact Study")
    print("="*60)
    
    # 실험 1: THI 미포함
    result_without_thi = run_experiment(include_thi=False)
    
    # 실험 2: THI 포함
    result_with_thi = run_experiment(include_thi=True)
    
    # 비교 리포트
    print("\n" + "="*60)
    print("📋 COMPARISON REPORT: THI Feature Impact")
    print("="*60)
    
    print(f"\n{'Metric':<10} {'Without THI':>15} {'With THI':>15} {'Improvement':>15}")
    print("-" * 60)
    
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    for m in metrics:
        v1 = result_without_thi['metrics'][m]
        v2 = result_with_thi['metrics'][m]
        
        if m == 'R2':
            improvement = ((v2 - v1) / abs(v1)) * 100 if v1 != 0 else 0
            symbol = "↑" if v2 > v1 else "↓"
        else:
            improvement = ((v1 - v2) / v1) * 100 if v1 != 0 else 0
            symbol = "↓" if v2 < v1 else "↑"
        
        print(f"{m:<10} {v1:>15.4f} {v2:>15.4f} {improvement:>+13.2f}% {symbol}")
    
    print("-" * 60)
    print(f"{'Features':<10} {result_without_thi['features']:>15} {result_with_thi['features']:>15}")
    
    # 결론
    r2_improvement = result_with_thi['metrics']['R2'] - result_without_thi['metrics']['R2']
    print(f"\n✅ Conclusion: R² improved by {r2_improvement:.4f} ({r2_improvement*100:.2f}%)")
    
    if r2_improvement > 0:
        print("   THI 변수가 전력 수요 예측 성능 향상에 기여했습니다!")
    else:
        print("   THI 변수 추가가 유의미한 개선을 보이지 않았습니다.")
    
    return result_without_thi, result_with_thi


if __name__ == "__main__":
    main()
