#!/usr/bin/env python3
"""
SMP v3.7: Long-term Moving Averages for Price Level

Based on v3.2 Optuna best model with added long-term moving averages.
The key insight from Gemini discussion:
- SMP is determined by fuel costs, not just demand/generation
- Without fuel price data, use long-term MA to capture price LEVEL changes
- 30-day and 90-day MAs help model understand baseline price trends

New features:
- smp_ma720 (30-day = 720 hours moving average)
- smp_ma2160 (90-day = 2160 hours moving average)
- smp_level_ratio (current SMP / 30-day MA - relative to baseline)
- smp_trend (30-day MA / 90-day MA - trend direction)
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SMPDataPipeline:
    """Data pipeline with long-term moving averages."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, train_start: str = '2020-12-19', train_end: str = '2024-12-31') -> pd.DataFrame:
        """Load EPSIS data"""
        data_path = PROJECT_ROOT / 'data' / 'smp' / 'smp_5years_epsis.csv'
        df = pd.read_csv(data_path)

        df = df[df['smp_mainland'] > 0].copy()

        def fix_hour_24(ts):
            if ' 24:00' in str(ts):
                date_part = str(ts).replace(' 24:00', '')
                return pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return pd.to_datetime(ts)

        df['datetime'] = df['timestamp'].apply(fix_hour_24)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)]

        return df

    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create features with long-term moving averages."""
        features = []
        self.feature_names = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        # ===== Base price features (4) =====
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)
        self.feature_names.extend(['smp_mainland', 'smp_jeju', 'smp_max', 'smp_min'])

        # ===== Time cyclical (8) =====
        hour = df['hour'].values
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        self.feature_names.extend(['hour_sin', 'hour_cos'])

        day_of_week = df['datetime'].dt.dayofweek.values
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        features.append((day_of_week >= 5).astype(float))
        self.feature_names.extend(['dow_sin', 'dow_cos', 'is_weekend'])

        month = df['datetime'].dt.month.values
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        self.feature_names.extend(['month_sin', 'month_cos'])

        # ===== Season/Peak (5) =====
        is_summer = ((month >= 6) & (month <= 8)).astype(float)
        is_winter = ((month == 12) | (month <= 2)).astype(float)
        features.append(is_summer)
        features.append(is_winter)
        self.feature_names.extend(['is_summer', 'is_winter'])

        peak_morning = ((hour >= 9) & (hour <= 12)).astype(float)
        peak_evening = ((hour >= 17) & (hour <= 21)).astype(float)
        off_peak = ((hour >= 1) & (hour <= 6)).astype(float)
        features.append(peak_morning)
        features.append(peak_evening)
        features.append(off_peak)
        self.feature_names.extend(['peak_morning', 'peak_evening', 'off_peak'])

        # ===== Short-term Statistical (4) - same as v3.2 =====
        smp_ma24 = smp_series.rolling(24, min_periods=1).mean().values
        smp_std24 = smp_series.rolling(24, min_periods=1).std().fillna(0).values
        features.append(smp_ma24)
        features.append(smp_std24)
        self.feature_names.extend(['smp_ma24', 'smp_std24'])

        smp_diff = np.diff(smp, prepend=smp[0])
        smp_range = df['smp_max'].values - df['smp_min'].values
        features.append(smp_diff)
        features.append(smp_range)
        self.feature_names.extend(['smp_diff', 'smp_range'])

        # ===== NEW: Long-term Moving Averages (6) =====
        # 30-day MA (720 hours) - captures monthly price level
        smp_ma720 = smp_series.rolling(720, min_periods=24).mean().values
        features.append(smp_ma720)
        self.feature_names.append('smp_ma720_30d')

        # 90-day MA (2160 hours) - captures quarterly price level
        smp_ma2160 = smp_series.rolling(2160, min_periods=24).mean().values
        features.append(smp_ma2160)
        self.feature_names.append('smp_ma2160_90d')

        # 7-day MA (168 hours) - weekly baseline
        smp_ma168 = smp_series.rolling(168, min_periods=24).mean().values
        features.append(smp_ma168)
        self.feature_names.append('smp_ma168_7d')

        # Price level ratio: current vs 30-day baseline
        # This tells model "is current price above or below recent average?"
        smp_level_ratio = smp / (smp_ma720 + 1e-6)
        features.append(smp_level_ratio)
        self.feature_names.append('smp_level_ratio')

        # Trend indicator: 30-day MA vs 90-day MA
        # > 1 means upward trend, < 1 means downward trend
        smp_trend = smp_ma720 / (smp_ma2160 + 1e-6)
        features.append(smp_trend)
        self.feature_names.append('smp_trend')

        # Weekly volatility (std of last 168 hours)
        smp_std168 = smp_series.rolling(168, min_periods=24).std().fillna(0).values
        features.append(smp_std168)
        self.feature_names.append('smp_std168_7d')

        # Stack features
        X = np.column_stack(features)
        y = df['smp_jeju'].values  # Target: Jeju SMP

        # Remove NaN rows (from rolling calculations)
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"  Features: {len(self.feature_names)}, Records: {len(X)}")
        logger.info(f"  New long-term features: smp_ma720_30d, smp_ma2160_90d, smp_ma168_7d, smp_level_ratio, smp_trend, smp_std168_7d")

        return X, y, self.feature_names


class SMPDataset(Dataset):
    """PyTorch Dataset for SMP prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 96):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return X_seq, y_val


class BiLSTMAttention(nn.Module):
    """BiLSTM with Multi-Head Attention (same as v3.2)."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2, n_heads: int = 4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        return self.fc(out).squeeze(-1)


def train_epoch(model, loader, criterion, optimizer, device, noise_std=0.0):
    """Train one epoch with optional noise injection."""
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Add noise for regularization
        if noise_std > 0:
            noise = torch.randn_like(X) * noise_std
            X = X + noise

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, scaler_y):
    """Evaluate model."""
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            preds.extend(pred)
            actuals.extend(y.numpy())

    # Inverse transform
    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    # MAPE (exclude near-zero values)
    mask = actuals > 10
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(actuals[mask], preds[mask]) * 100
    else:
        mape = float('inf')

    r2 = r2_score(actuals, preds)
    mae = np.mean(np.abs(actuals - preds))

    return mape, r2, preds, actuals, mae


def main():
    logger.info("=" * 60)
    logger.info("SMP v3.7: Long-term Moving Averages for Price Level")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load and prepare data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipeline()
    df = pipeline.load_data()
    X, y, feature_names = pipeline.create_features(df)

    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Total records: {len(X)}")
    logger.info(f"Target (smp_jeju): mean={y.mean():.2f}, std={y.std():.2f}")

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train/Val/Test split (70/15/15)
    n = len(X_scaled)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X_scaled[:train_end], y_scaled[:train_end]
    X_val, y_val = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
    X_test, y_test = X_scaled[val_end:], y_scaled[val_end:]

    logger.info(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Best hyperparameters from v3.2 Optuna
    config = {
        'seq_len': 96,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.198,
        'n_heads': 4,
        'lr': 0.000165,
        'weight_decay': 0.000476,
        'batch_size': 32,
        'noise_std': 0.0099,
        'epochs': 150,
        'patience': 20
    }

    logger.info(f"\nConfig (from v3.2 Optuna): {config}")

    # Create datasets
    train_ds = SMPDataset(X_train, y_train, config['seq_len'])
    val_ds = SMPDataset(X_val, y_val, config['seq_len'])
    test_ds = SMPDataset(X_test, y_test, config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])

    # Create model
    model = BiLSTMAttention(
        input_size=len(feature_names),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        n_heads=config['n_heads']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training
    logger.info("\nTraining...")
    best_val_mape = float('inf')
    best_val_r2 = 0
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            noise_std=config['noise_std']
        )
        val_mape, val_r2, _, _, val_mae = evaluate(model, val_loader, device, scaler_y)

        scheduler.step(val_mae)

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_val_r2 = val_r2
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: Val MAPE={val_mape:.2f}%, R²={val_r2:.4f}")

        if patience_counter >= config['patience']:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    test_mape, test_r2, preds, actuals, test_mae = evaluate(model, test_loader, device, scaler_y)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Validation: MAPE={best_val_mape:.2f}%, R²={best_val_r2:.4f}")
    logger.info(f"Test:       MAPE={test_mape:.2f}%, R²={test_r2:.4f}, MAE={test_mae:.2f} won/kWh")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_7_longterm_ma"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")

    metrics = {
        'val_mape': best_val_mape,
        'val_r2': best_val_r2,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'features': len(feature_names),
        'feature_names': feature_names,
        'records': len(X),
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'improvement': 'Added 30d/90d moving averages to capture price level changes'
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH v3.2 BASELINE")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Features | Key Change |")
    logger.info("|-------|------|-----|----------|------------|")
    logger.info(f"| v3.2 (Optuna) | 7.42% | 0.760 | 22 | Baseline |")
    logger.info(f"| v3.7 (Long MA) | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_names)} | +30d/90d MA |")

    if test_r2 > 0.760:
        improvement = test_r2 - 0.760
        logger.info(f"\n✅ IMPROVED: R² +{improvement:.3f}")
    elif test_r2 > 0.74:
        logger.info(f"\n⚠️ Similar performance (within margin)")
    else:
        logger.info(f"\n❌ No improvement from long-term MA")

    # Feature importance hint
    logger.info("\n" + "=" * 60)
    logger.info("NEW FEATURES ADDED")
    logger.info("=" * 60)
    logger.info("- smp_ma720_30d: 30-day moving average (price baseline)")
    logger.info("- smp_ma2160_90d: 90-day moving average (quarterly level)")
    logger.info("- smp_ma168_7d: 7-day moving average (weekly baseline)")
    logger.info("- smp_level_ratio: Current price / 30d MA (relative position)")
    logger.info("- smp_trend: 30d MA / 90d MA (trend direction)")
    logger.info("- smp_std168_7d: 7-day volatility")


if __name__ == "__main__":
    main()
