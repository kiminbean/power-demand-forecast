#!/usr/bin/env python3
"""
SMP v3.3: With Jeju Power Generation Data (LNG + Oil)

Uses marginal generator data for improved SMP prediction.
- Jeju LNG generation (한계발전기)
- Jeju Oil generation (한계발전기)

Data period: 2023-05-01 ~ 2024-03-31 (11 months)
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


def load_generation_data():
    """Load and preprocess Jeju LNG/Oil generation data."""
    raw_dir = PROJECT_ROOT / "data" / "raw"

    # Load LNG data
    lng_path = raw_dir / "제주 시간대별 발전량(LNG)_240331.csv"
    lng_df = pd.read_csv(lng_path, encoding='cp949')
    lng_df.columns = ['date'] + [f'lng_h{i}' for i in range(1, 25)]

    # Load Oil data
    oil_path = raw_dir / "제주 시간대별 발전량(유류)_240331.csv"
    oil_df = pd.read_csv(oil_path, encoding='cp949')
    oil_df.columns = ['date'] + [f'oil_h{i}' for i in range(1, 25)]

    # Melt to hourly format
    lng_melted = lng_df.melt(
        id_vars=['date'],
        var_name='hour_col',
        value_name='lng_generation'
    )
    lng_melted['hour'] = lng_melted['hour_col'].str.extract(r'(\d+)').astype(int)

    oil_melted = oil_df.melt(
        id_vars=['date'],
        var_name='hour_col',
        value_name='oil_generation'
    )
    oil_melted['hour'] = oil_melted['hour_col'].str.extract(r'(\d+)').astype(int)

    # Create datetime
    lng_melted['datetime'] = pd.to_datetime(lng_melted['date']) + pd.to_timedelta(lng_melted['hour'], unit='h')
    oil_melted['datetime'] = pd.to_datetime(oil_melted['date']) + pd.to_timedelta(oil_melted['hour'], unit='h')

    # Merge LNG and Oil
    gen_df = lng_melted[['datetime', 'lng_generation']].merge(
        oil_melted[['datetime', 'oil_generation']],
        on='datetime',
        how='inner'
    )

    # Total thermal generation
    gen_df['thermal_generation'] = gen_df['lng_generation'] + gen_df['oil_generation']

    # Sort and set index
    gen_df = gen_df.sort_values('datetime').reset_index(drop=True)

    logger.info(f"  Generation data: {len(gen_df)} records ({gen_df['datetime'].min()} ~ {gen_df['datetime'].max()})")

    return gen_df


def load_smp_data():
    """Load SMP data."""
    smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"

    df = pd.read_csv(smp_path)

    # Handle 24:00 timestamps (convert to 00:00 next day)
    df['timestamp'] = df['timestamp'].str.replace(' 24:00', ' 00:00')
    df['datetime'] = pd.to_datetime(df['timestamp'])

    # Adjust dates where hour was 24 (now 00:00 but should be next day)
    mask = df['hour'] == 24
    df.loc[mask, 'datetime'] = df.loc[mask, 'datetime'] + pd.Timedelta(days=1)

    df = df.sort_values('datetime').reset_index(drop=True)

    # Rename columns for consistency
    if 'smp_mainland' in df.columns:
        df['smp_land'] = df['smp_mainland']

    logger.info(f"  SMP data: {len(df)} records ({df['datetime'].min()} ~ {df['datetime'].max()})")

    return df


def load_weather_data():
    """Load and merge weather data."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    weather_path = processed_dir / "jeju_weather_hourly_merged.csv"

    if weather_path.exists():
        df = pd.read_csv(weather_path)

        # Handle Korean column names
        if '일시' in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
            df['temperature'] = pd.to_numeric(df['기온'], errors='coerce')
            df['humidity'] = pd.to_numeric(df['습도'], errors='coerce')
            df['wind_speed'] = pd.to_numeric(df['풍속'], errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])

        logger.info(f"  Weather data: {len(df)} records")
        return df

    return None


def create_features(smp_df, gen_df, weather_df=None):
    """Create features for model training."""

    # Filter SMP to generation data period
    min_dt = gen_df['datetime'].min()
    max_dt = gen_df['datetime'].max()

    smp_filtered = smp_df[(smp_df['datetime'] >= min_dt) & (smp_df['datetime'] <= max_dt)].copy()
    logger.info(f"  SMP filtered to generation period: {len(smp_filtered)} records")

    # Merge SMP with generation data
    df = smp_filtered.merge(gen_df, on='datetime', how='inner')
    logger.info(f"  After merge with generation: {len(df)} records")

    # Merge with weather if available
    if weather_df is not None:
        df = df.merge(weather_df[['datetime', 'temperature', 'humidity', 'wind_speed']],
                      on='datetime', how='left')
        df['temperature'] = df['temperature'].ffill().bfill()
        df['humidity'] = df['humidity'].ffill().bfill()
        df['wind_speed'] = df['wind_speed'].ffill().bfill()

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

    # SMP features (use available columns)
    smp_cols = ['smp_land', 'smp_jeju', 'smp_max', 'smp_min']
    available_smp = [c for c in smp_cols if c in df.columns]

    for col in available_smp:
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Rolling features
        for window in [6, 12, 24]:
            df[f'{col}_roll{window}_mean'] = df[col].rolling(window).mean()
            df[f'{col}_roll{window}_std'] = df[col].rolling(window).std()

    # Generation features
    gen_cols = ['lng_generation', 'oil_generation', 'thermal_generation']
    for col in gen_cols:
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Rolling features
        for window in [6, 12, 24]:
            df[f'{col}_roll{window}_mean'] = df[col].rolling(window).mean()

    # Generation ratio features
    df['lng_ratio'] = df['lng_generation'] / (df['thermal_generation'] + 1e-6)
    df['oil_ratio'] = df['oil_generation'] / (df['thermal_generation'] + 1e-6)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    # Target
    target_col = 'smp_jeju' if 'smp_jeju' in df.columns else 'smp_land'

    # Feature columns (exclude datetime and target)
    exclude_cols = ['datetime', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32']]

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

    mape = mean_absolute_percentage_error(actuals, preds) * 100
    r2 = r2_score(actuals, preds)

    return mape, r2, preds, actuals


def main():
    logger.info("=" * 60)
    logger.info("SMP v3.3: With Jeju Power Generation Data")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading data...")
    smp_df = load_smp_data()
    gen_df = load_generation_data()
    weather_df = load_weather_data()

    # Create features
    logger.info("Creating features...")
    df, feature_cols, target_col = create_features(smp_df, gen_df, weather_df)

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
        val_mape, val_r2, _, _ = evaluate(model, val_loader, device, scaler_y)

        scheduler.step(val_mape)

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

    # Final evaluation on test set
    test_mape, test_r2, preds, actuals = evaluate(model, test_loader, device, scaler_y)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Validation: MAPE={best_val_mape:.2f}%, R²={best_val_r2:.4f}")
    logger.info(f"Test:       MAPE={test_mape:.2f}%, R²={test_r2:.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_3_generation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save metrics
    import json
    metrics = {
        'val_mape': best_val_mape,
        'val_r2': best_val_r2,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'features': len(feature_cols),
        'records': len(df),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison with v3.2
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH v3.2")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Features | Data Period |")
    logger.info("|-------|------|-----|----------|-------------|")
    logger.info(f"| v3.2 (Optuna) | 7.42% | 0.760 | 22 | 5 years |")
    logger.info(f"| v3.3 (Gen) | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_cols)} | 11 months |")

    if test_r2 > 0.76:
        logger.info(f"\n✅ v3.3 IMPROVED: R² +{test_r2 - 0.76:.3f}")
    else:
        logger.info(f"\n⚠️ v3.3 needs more data or tuning")


if __name__ == "__main__":
    main()
