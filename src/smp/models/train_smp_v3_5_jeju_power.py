#!/usr/bin/env python3
"""
SMP v3.5: With Jeju Hourly Power Trading Data

Uses Jeju-specific power trading data (2013-2024) for SMP prediction.
- Data: jeju_hourly_power_2013_2024.csv
- Period: 2013-01-01 ~ 2024-12-31 (12 years)
- Overlap with SMP: 2020-12-19 ~ 2024-12-31 (~4 years)

This should significantly improve R² as we have:
1. Jeju-specific power data
2. ~4 years of overlapping data
3. Hourly granularity
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jeju_power_data():
    """Load Jeju hourly power trading data."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    power_path = raw_dir / "jeju_hourly_power_2013_2024.csv"

    df = pd.read_csv(power_path)

    # Create datetime
    df['datetime'] = pd.to_datetime(df['거래일자']) + pd.to_timedelta(df['시간'], unit='h')

    # Rename columns
    df = df.rename(columns={'전력거래량(MWh)': 'jeju_power_mwh'})

    # Select relevant columns
    df = df[['datetime', 'jeju_power_mwh']].copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    logger.info(f"  Jeju Power: {len(df)} records ({df['datetime'].min()} ~ {df['datetime'].max()})")

    return df


def load_smp_data():
    """Load SMP data."""
    smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"

    df = pd.read_csv(smp_path)

    # Handle 24:00 timestamps
    df['timestamp'] = df['timestamp'].str.replace(' 24:00', ' 00:00')
    df['datetime'] = pd.to_datetime(df['timestamp'])

    # Adjust dates where hour was 24
    mask = df['hour'] == 24
    df.loc[mask, 'datetime'] = df.loc[mask, 'datetime'] + pd.Timedelta(days=1)

    df = df.sort_values('datetime').reset_index(drop=True)

    # Rename columns
    if 'smp_mainland' in df.columns:
        df['smp_land'] = df['smp_mainland']

    logger.info(f"  SMP data: {len(df)} records ({df['datetime'].min()} ~ {df['datetime'].max()})")

    return df


def create_features(smp_df, power_df):
    """Create features for model training."""

    # Merge SMP with power data (inner join - overlap period)
    df = smp_df.merge(power_df, on='datetime', how='inner')
    logger.info(f"  After merge (overlap period): {len(df)} records")
    logger.info(f"  Period: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # SMP features
    smp_cols = ['smp_land', 'smp_jeju', 'smp_max', 'smp_min']
    available_smp = [c for c in smp_cols if c in df.columns]

    for col in available_smp:
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Rolling features
        for window in [6, 24]:
            df[f'{col}_roll{window}_mean'] = df[col].rolling(window).mean()

    # Jeju power features (KEY ADDITION!)
    df['power_lag1'] = df['jeju_power_mwh'].shift(1)
    df['power_lag2'] = df['jeju_power_mwh'].shift(2)
    df['power_lag3'] = df['jeju_power_mwh'].shift(3)
    df['power_lag6'] = df['jeju_power_mwh'].shift(6)
    df['power_lag12'] = df['jeju_power_mwh'].shift(12)
    df['power_lag24'] = df['jeju_power_mwh'].shift(24)

    # Rolling features
    df['power_roll6_mean'] = df['jeju_power_mwh'].rolling(6).mean()
    df['power_roll12_mean'] = df['jeju_power_mwh'].rolling(12).mean()
    df['power_roll24_mean'] = df['jeju_power_mwh'].rolling(24).mean()
    df['power_roll24_std'] = df['jeju_power_mwh'].rolling(24).std()

    # Change rates
    df['power_change_1h'] = df['jeju_power_mwh'].pct_change(1)
    df['power_change_6h'] = df['jeju_power_mwh'].pct_change(6)
    df['power_change_24h'] = df['jeju_power_mwh'].pct_change(24)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    # Target
    target_col = 'smp_jeju' if 'smp_jeju' in df.columns else 'smp_land'

    # Feature columns
    exclude_cols = ['datetime', 'timestamp', 'date', target_col, 'source', 'is_synthetic']
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    # Remove constant features
    feature_cols = [c for c in feature_cols if df[c].std() > 1e-6]

    logger.info(f"  Features: {len(feature_cols)}, Records: {len(df)}")

    return df, feature_cols, target_col


class SMPDataset(Dataset):
    """Dataset for SMP prediction."""

    def __init__(self, X, y, seq_len=96):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_val = self.y[idx+self.seq_len]
        return X_seq, y_val


class BiLSTMAttention(nn.Module):
    """BiLSTM with Multi-Head Attention."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, n_heads=4):
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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

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

    # Filter out near-zero actuals for MAPE (to avoid division by zero)
    # SMP can be 0 during high solar generation
    mask = actuals > 10  # Only consider SMP > 10 won/kWh for MAPE
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(actuals[mask], preds[mask]) * 100
    else:
        mape = float('inf')

    r2 = r2_score(actuals, preds)
    mae = np.mean(np.abs(actuals - preds))

    return mape, r2, preds, actuals, mae


def main():
    logger.info("=" * 60)
    logger.info("SMP v3.5: With Jeju Hourly Power Trading Data")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading data...")
    smp_df = load_smp_data()
    power_df = load_jeju_power_data()

    # Create features
    logger.info("Creating features...")
    df, feature_cols, target_col = create_features(smp_df, power_df)

    logger.info(f"\nData period: {df['datetime'].min()} ~ {df['datetime'].max()}")
    logger.info(f"Features: {len(feature_cols)}, Records: {len(df)}")
    logger.info(f"Target: {target_col}, mean={df[target_col].mean():.2f}, std={df[target_col].std():.2f}")

    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Scale
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

    # Hyperparameters (from v3.2 Optuna)
    config = {
        'seq_len': 96,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.2,
        'n_heads': 4,
        'lr': 0.000165,
        'weight_decay': 0.000476,
        'batch_size': 32,
        'epochs': 150,
        'patience': 20
    }

    # Datasets
    train_ds = SMPDataset(X_train, y_train, config['seq_len'])
    val_ds = SMPDataset(X_val, y_val, config['seq_len'])
    test_ds = SMPDataset(X_test, y_test, config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])

    # Model
    model = BiLSTMAttention(
        input_size=len(feature_cols),
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
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mape, val_r2, _, _, val_mae = evaluate(model, val_loader, device, scaler_y)

        scheduler.step(val_mae)  # Use MAE for LR scheduling (more stable than MAPE)

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
    logger.info(f"Validation: MAPE={best_val_mape:.2f}% (SMP>10), R²={best_val_r2:.4f}")
    logger.info(f"Test:       MAPE={test_mape:.2f}% (SMP>10), R²={test_r2:.4f}, MAE={test_mae:.2f} won/kWh")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_5_jeju_power"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")

    import json
    metrics = {
        'val_mape': best_val_mape,
        'val_r2': best_val_r2,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'features': len(feature_cols),
        'records': len(df),
        'data_period': f"{df['datetime'].min()} ~ {df['datetime'].max()}",
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'note': 'MAPE calculated only for SMP > 10 won/kWh (excluding zero SMP periods)'
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH PREVIOUS MODELS")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Features | Data Period |")
    logger.info("|-------|------|-----|----------|-------------|")
    logger.info(f"| v3.2 (Optuna) | 7.42% | 0.760 | 22 | 5 years SMP only |")
    logger.info(f"| v3.5 (Jeju Power) | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_cols)} | ~4 years with power |")

    if test_r2 > 0.76:
        logger.info(f"\n✅ v3.5 IMPROVED: R² +{test_r2 - 0.76:.3f}")
    elif test_r2 > 0.70:
        logger.info(f"\n⚠️ v3.5 comparable (R² {0.76 - test_r2:.3f} lower)")
    else:
        logger.info(f"\n❌ v3.5 underperformed")

    # If improved, show feature importance
    if test_r2 > 0.75:
        logger.info("\n🎯 Jeju power data contributes to better prediction!")


if __name__ == "__main__":
    main()
