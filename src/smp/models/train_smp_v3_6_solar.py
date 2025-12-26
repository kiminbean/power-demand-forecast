#!/usr/bin/env python3
"""
SMP v3.6: With Jeju Solar Generation Data

Uses Jeju solar generation data (2018-2024) for SMP prediction.
- Data: 한국동서발전_제주_기상관측_태양광발전.csv
- Period: 2018-01-01 ~ 2024-05-31 (~6.5 years)
- Overlap with SMP: 2020-12-19 ~ 2024-05-31 (~3.5 years)

Key insight: SMP goes to 0 during high solar generation periods.
Solar generation is a KEY driver of SMP in Jeju.
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


def load_solar_data():
    """Load Jeju solar generation data with weather observations."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    solar_path = raw_dir / "한국동서발전_제주_기상관측_태양광발전.csv"

    df = pd.read_csv(solar_path)

    # Create datetime
    df['datetime'] = pd.to_datetime(df['일시'])

    # Rename columns to English
    df = df.rename(columns={
        '기온': 'temperature',
        '강수량(mm)': 'precipitation',
        '습도': 'humidity',
        '적설(cm)': 'snow',
        '전운량(10분위)': 'cloud_cover',
        '일조(hr)': 'sunshine_hours',
        '일사량': 'irradiance',
        '태양광 설비용량(MW)': 'solar_capacity_mw',
        '태양광 발전량(MWh)': 'solar_gen_mwh'
    })

    # Calculate capacity factor (generation / capacity)
    df['solar_capacity_factor'] = df['solar_gen_mwh'] / (df['solar_capacity_mw'] + 0.001)

    # Select relevant columns
    cols = ['datetime', 'temperature', 'precipitation', 'humidity', 'cloud_cover',
            'sunshine_hours', 'irradiance', 'solar_capacity_mw', 'solar_gen_mwh',
            'solar_capacity_factor']
    df = df[cols].copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Convert to numeric
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"  Solar data: {len(df)} records ({df['datetime'].min()} ~ {df['datetime'].max()})")

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


def create_features(smp_df, solar_df):
    """Create features for model training."""

    # Merge SMP with solar data (inner join - overlap period)
    df = smp_df.merge(solar_df, on='datetime', how='inner')
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

    # Solar generation features (KEY FEATURES!)
    # These directly explain low/zero SMP periods
    df['solar_lag1'] = df['solar_gen_mwh'].shift(1)
    df['solar_lag2'] = df['solar_gen_mwh'].shift(2)
    df['solar_lag3'] = df['solar_gen_mwh'].shift(3)
    df['solar_lag6'] = df['solar_gen_mwh'].shift(6)
    df['solar_lag24'] = df['solar_gen_mwh'].shift(24)

    # Rolling features
    df['solar_roll6_mean'] = df['solar_gen_mwh'].rolling(6).mean()
    df['solar_roll12_mean'] = df['solar_gen_mwh'].rolling(12).mean()
    df['solar_roll24_mean'] = df['solar_gen_mwh'].rolling(24).mean()
    df['solar_roll24_std'] = df['solar_gen_mwh'].rolling(24).std()

    # Change rates
    df['solar_change_1h'] = df['solar_gen_mwh'].pct_change(1)
    df['solar_change_6h'] = df['solar_gen_mwh'].pct_change(6)

    # Weather features (affect solar generation)
    df['irradiance_lag1'] = df['irradiance'].shift(1)
    df['irradiance_lag6'] = df['irradiance'].shift(6)
    df['cloud_lag1'] = df['cloud_cover'].shift(1)
    df['temp_lag1'] = df['temperature'].shift(1)

    # Solar capacity factor features
    df['cf_lag1'] = df['solar_capacity_factor'].shift(1)
    df['cf_lag24'] = df['solar_capacity_factor'].shift(24)
    df['cf_roll6_mean'] = df['solar_capacity_factor'].rolling(6).mean()

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
    logger.info("SMP v3.6: With Jeju Solar Generation Data")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading data...")
    smp_df = load_smp_data()
    solar_df = load_solar_data()

    # Create features
    logger.info("Creating features...")
    df, feature_cols, target_col = create_features(smp_df, solar_df)

    logger.info(f"\nData period: {df['datetime'].min()} ~ {df['datetime'].max()}")
    logger.info(f"Features: {len(feature_cols)}, Records: {len(df)}")
    logger.info(f"Target: {target_col}, mean={df[target_col].mean():.2f}, std={df[target_col].std():.2f}")

    # Check solar-SMP relationship
    corr = df['solar_gen_mwh'].corr(df[target_col])
    logger.info(f"Solar-SMP correlation: {corr:.3f}")

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
    output_dir = PROJECT_ROOT / "models" / "smp_v3_6_solar"
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
        'note': 'Uses solar generation data. MAPE calculated only for SMP > 10 won/kWh'
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
    logger.info(f"| v3.5 (Power demand) | 11.09% | 0.506 | 60 | ~4 years |")
    logger.info(f"| v3.6 (Solar gen) | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_cols)} | ~3.5 years |")

    if test_r2 > 0.76:
        logger.info(f"\n✅ v3.6 IMPROVED: R² +{test_r2 - 0.76:.3f}")
    elif test_r2 > 0.70:
        logger.info(f"\n⚠️ v3.6 comparable (R² {0.76 - test_r2:.3f} lower)")
    else:
        logger.info(f"\n❌ v3.6 underperformed")

    # If improved, highlight solar's importance
    if test_r2 > 0.75:
        logger.info("\n🎯 Solar generation is a key driver of SMP prediction!")


if __name__ == "__main__":
    main()
