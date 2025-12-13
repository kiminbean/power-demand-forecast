"""
LSTM with Differencing (MODEL-002)
==================================
비정상성 해결을 위한 1차 차분 적용

Key Changes from MODEL-001:
1. Target: y_t → Δy_t = y_t - y_{t-1}
2. Scaler: StandardScaler (for diff data)
3. Restoration: y_hat_t = y_{t-1}(actual) + Δy_hat_t

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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
    'output_dir': PROJECT_ROOT / 'results' / 'model_002',
    'sequence_length': 14,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 100,
    'patience': 20,  # 더 긴 patience
    'n_splits': 5,   # TimeSeriesSplit
    'random_seed': 42,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# Feature columns (THI 포함)
FEATURES = [
    'temp_mean', 'temp_max', 'temp_min', 'temp_range',
    'dewpoint_mean', 'sunshine_hours', 'solar_radiation',
    'soil_temp_5cm', 'soil_temp_10cm', 'soil_temp_20cm',
    'CDD', 'HDD', 'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos', 'is_weekend', 'is_holiday',
    'humidity', 'THI'
]

TARGET = 'power_sum'


# ============================================================
# Device Setup
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Data Processing with Differencing
# ============================================================
def load_and_preprocess_data() -> pd.DataFrame:
    """데이터 로드 및 전처리 (THI 계산 + 차분)"""
    print(f"\n📂 Loading data from {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG['data_path'], parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # THI 계산
    print("🔧 Calculating THI and Humidity...")
    df = calculate_humidity_and_thi(df)
    
    # 1차 차분 (Target)
    print("🔧 Applying 1st order differencing to target...")
    df['power_diff'] = df[TARGET].diff()
    
    # 첫 행 제거 (NaN)
    df = df.dropna(subset=['power_diff']).reset_index(drop=True)
    
    print(f"   Total records after diff: {len(df)}")
    
    # 차분 데이터 분포 확인
    diff_mean = df['power_diff'].mean()
    diff_std = df['power_diff'].std()
    print(f"   Diff distribution: mean={diff_mean:.2f}, std={diff_std:.2f}")
    
    return df


def prepare_sequences_with_diff(
    df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int
) -> tuple:
    """
    차분된 타겟을 사용한 시퀀스 데이터 생성
    
    Returns:
        X: (samples, sequence_length, n_features)
        y_diff: (samples,) - 차분된 타겟
        y_prev: (samples,) - 이전 시점 실제값 (복원용)
        y_actual: (samples,) - 현재 시점 실제값 (평가용)
    """
    # 결측치 처리
    df = df.dropna(subset=feature_cols + ['power_diff', TARGET])
    
    # 스케일링
    feature_scaler = MinMaxScaler()  # Features는 MinMax
    diff_scaler = StandardScaler()   # 차분된 Target은 StandardScaler
    
    features = feature_scaler.fit_transform(df[feature_cols])
    diff_scaled = diff_scaler.fit_transform(df[['power_diff']])
    
    # 원본 타겟 (복원 및 평가용)
    original_target = df[TARGET].values
    
    # 시퀀스 생성
    X, y_diff, y_prev, y_actual = [], [], [], []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y_diff.append(diff_scaled[i])
        y_prev.append(original_target[i-1])  # 이전 시점 실제값
        y_actual.append(original_target[i])  # 현재 시점 실제값
    
    X = np.array(X)
    y_diff = np.array(y_diff).flatten()
    y_prev = np.array(y_prev)
    y_actual = np.array(y_actual)
    
    return X, y_diff, y_prev, y_actual, feature_scaler, diff_scaler


# ============================================================
# PyTorch Components
# ============================================================
class DiffDataset(Dataset):
    def __init__(self, X, y_diff, y_prev, y_actual):
        self.X = torch.FloatTensor(X)
        self.y_diff = torch.FloatTensor(y_diff)
        self.y_prev = torch.FloatTensor(y_prev)
        self.y_actual = torch.FloatTensor(y_actual)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_diff[idx], self.y_prev[idx], self.y_actual[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
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
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


# ============================================================
# Training with TimeSeriesSplit
# ============================================================
def train_fold(
    model, train_loader, val_loader, device, epochs, patience, lr
) -> tuple:
    """단일 Fold 학습"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_diff_batch, _, _ in train_loader:
            X_batch = X_batch.to(device)
            y_diff_batch = y_diff_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_diff_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_diff_batch, _, _ in val_loader:
                X_batch = X_batch.to(device)
                y_diff_batch = y_diff_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_diff_batch)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_val_loss


def evaluate_fold(
    model, test_loader, diff_scaler, device
) -> dict:
    """
    Fold 평가 (Teacher Forcing 방식 복원)
    
    복원 로직: y_hat_t = y_{t-1}(actual) + inverse_transform(diff_hat_t)
    """
    model.eval()
    
    all_diff_pred = []
    all_y_prev = []
    all_y_actual = []
    
    with torch.no_grad():
        for X_batch, _, y_prev_batch, y_actual_batch in test_loader:
            X_batch = X_batch.to(device)
            diff_pred = model(X_batch).cpu().numpy()
            
            all_diff_pred.extend(diff_pred)
            all_y_prev.extend(y_prev_batch.numpy())
            all_y_actual.extend(y_actual_batch.numpy())
    
    # 역스케일링 (차분 예측값)
    all_diff_pred = np.array(all_diff_pred).reshape(-1, 1)
    diff_pred_unscaled = diff_scaler.inverse_transform(all_diff_pred).flatten()
    
    # 복원: y_hat = y_prev(actual) + diff_pred
    y_prev = np.array(all_y_prev)
    y_actual = np.array(all_y_actual)
    y_pred = y_prev + diff_pred_unscaled
    
    # 평가 지표
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'predictions': y_pred,
        'actuals': y_actual
    }


def run_cross_validation():
    """TimeSeriesSplit 교차 검증 실행"""
    print("="*60)
    print("🔬 MODEL-002: LSTM with Differencing + TimeSeriesSplit")
    print("="*60)
    
    device = get_device()
    print(f"✅ Device: {device}")
    
    # 데이터 준비
    df = load_and_preprocess_data()
    X, y_diff, y_prev, y_actual, feature_scaler, diff_scaler = prepare_sequences_with_diff(
        df, FEATURES, CONFIG['sequence_length']
    )
    
    print(f"\n📊 Data shape: X={X.shape}, y_diff={y_diff.shape}")
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=CONFIG['n_splits'])
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    print(f"\n🔄 Running {CONFIG['n_splits']}-Fold TimeSeriesSplit Cross-Validation...\n")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"--- Fold {fold}/{CONFIG['n_splits']} ---")
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        # 데이터 분할
        X_train, X_test = X[train_idx], X[test_idx]
        y_diff_train, y_diff_test = y_diff[train_idx], y_diff[test_idx]
        y_prev_train, y_prev_test = y_prev[train_idx], y_prev[test_idx]
        y_actual_train, y_actual_test = y_actual[train_idx], y_actual[test_idx]
        
        # Validation split (Train의 마지막 10%)
        val_size = int(len(X_train) * 0.1)
        X_val, y_diff_val = X_train[-val_size:], y_diff_train[-val_size:]
        y_prev_val, y_actual_val = y_prev_train[-val_size:], y_actual_train[-val_size:]
        X_train, y_diff_train = X_train[:-val_size], y_diff_train[:-val_size]
        y_prev_train, y_actual_train = y_prev_train[:-val_size], y_actual_train[:-val_size]
        
        # DataLoader
        train_dataset = DiffDataset(X_train, y_diff_train, y_prev_train, y_actual_train)
        val_dataset = DiffDataset(X_val, y_diff_val, y_prev_val, y_actual_val)
        test_dataset = DiffDataset(X_test, y_diff_test, y_prev_test, y_actual_test)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
        
        # 모델 생성 및 학습
        model = LSTMModel(
            input_dim=len(FEATURES),
            hidden_dim=CONFIG['hidden_dim'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        )
        
        model, _ = train_fold(
            model, train_loader, val_loader, device,
            CONFIG['epochs'], CONFIG['patience'], CONFIG['learning_rate']
        )
        
        # 평가
        metrics = evaluate_fold(model, test_loader, diff_scaler, device)
        fold_results.append(metrics)
        all_predictions.extend(metrics['predictions'])
        all_actuals.extend(metrics['actuals'])
        
        print(f"    R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%\n")
    
    # 전체 결과 집계
    print("="*60)
    print("📋 CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    avg_metrics = {
        'R2': np.mean([r['R2'] for r in fold_results]),
        'RMSE': np.mean([r['RMSE'] for r in fold_results]),
        'MAE': np.mean([r['MAE'] for r in fold_results]),
        'MAPE': np.mean([r['MAPE'] for r in fold_results]),
    }
    
    std_metrics = {
        'R2': np.std([r['R2'] for r in fold_results]),
        'RMSE': np.std([r['RMSE'] for r in fold_results]),
    }
    
    print(f"\n{'Metric':<10} {'Mean':>15} {'Std':>15}")
    print("-" * 45)
    print(f"{'R²':<10} {avg_metrics['R2']:>15.4f} {std_metrics['R2']:>15.4f}")
    print(f"{'RMSE':<10} {avg_metrics['RMSE']:>15.2f} {std_metrics['RMSE']:>15.2f}")
    print(f"{'MAE':<10} {avg_metrics['MAE']:>15.2f}")
    print(f"{'MAPE':<10} {avg_metrics['MAPE']:>15.2f}%")
    
    # 시각화
    plot_results(all_predictions, all_actuals, fold_results)
    
    # 결론
    print("\n" + "="*60)
    print("📝 CONCLUSION")
    print("="*60)
    
    if avg_metrics['R2'] > 0:
        print(f"\n   ✅ SUCCESS! R² = {avg_metrics['R2']:.4f} (positive)")
        print(f"   차분(Differencing) 적용으로 비정상성 문제 해결!")
    else:
        print(f"\n   ⚠️ R² still negative: {avg_metrics['R2']:.4f}")
        print(f"   추가적인 Feature Engineering 또는 모델 개선 필요")
    
    return avg_metrics, fold_results


def plot_results(predictions, actuals, fold_results):
    """결과 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 예측 vs 실제
    ax1 = axes[0]
    ax1.scatter(actuals, predictions, alpha=0.5, s=10)
    ax1.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect')
    ax1.set_xlabel('Actual Power (MW)')
    ax1.set_ylabel('Predicted Power (MW)')
    ax1.set_title('Prediction vs Actual (All Folds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fold별 R² 비교
    ax2 = axes[1]
    r2_scores = [r['R2'] for r in fold_results]
    ax2.bar(range(1, len(r2_scores)+1), r2_scores, color='steelblue')
    ax2.axhline(y=np.mean(r2_scores), color='red', linestyle='--', label=f'Mean: {np.mean(r2_scores):.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CONFIG['output_dir'] / 'model_002_results.png', dpi=150)
    print(f"\n✅ Saved: {CONFIG['output_dir'] / 'model_002_results.png'}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    avg_metrics, fold_results = run_cross_validation()
