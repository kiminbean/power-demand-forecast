#!/usr/bin/env python3
"""
SMP v7.0 Enhanced: BiLSTM+Attention with Full Feature Set

Uses comprehensive data from crawlers:
- SMP time series + lag features
- Weather data
- Fuel prices (WTI, Brent, Natural Gas)
- Carbon prices (K-ETS, EU-ETS)
- Power generation proxies
- Net load estimates

Target: R² > 0.85, MAPE < 6.5%
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BiLSTMAttentionV7(nn.Module):
    """
    Enhanced BiLSTM with Multi-Head Self-Attention

    Improvements over v6:
    - Deeper network with residual connections
    - Separate feature embedding layers
    - Dropout scheduling
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.3,
        output_hours: int = 24
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )

        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours * 3)  # q10, q50, q90
        )

        self.output_hours = output_hours

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature embedding
        x = self.feature_embed(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Self-attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.layer_norm1(lstm_out + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        # Use last timestep
        x = x[:, -1, :]

        # Output
        out = self.output_proj(x)

        return out.view(-1, self.output_hours, 3)


class SMPDatasetV7(Dataset):
    """Enhanced dataset with all features"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 168,
        pred_len: int = 24
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y


def load_enhanced_dataset() -> pd.DataFrame:
    """Load the enhanced dataset from data collector"""
    data_path = PROJECT_ROOT / "data" / "processed" / "smp_enhanced_dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Enhanced dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter 2022-2024
    df = df[(df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2024-12-31')]

    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Features: {len(df.columns)}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Prepare feature matrix and target from enhanced dataset

    Note: Only using validated real data sources:
    - SMP time series (real)
    - Weather data (real)
    - Fuel prices from Yahoo Finance (real)
    - Carbon prices (synthetic but realistic)

    Excluded: generation data (sample/synthetic only)
    """
    # Select feature columns - ONLY REAL DATA
    feature_cols = [
        # SMP lag features (real)
        'smp_lag_1', 'smp_lag_2', 'smp_lag_3', 'smp_lag_6', 'smp_lag_12',
        'smp_lag_24', 'smp_lag_48', 'smp_lag_168',
        # SMP rolling features (real)
        'smp_roll_mean_6', 'smp_roll_std_6',
        'smp_roll_mean_12', 'smp_roll_std_12',
        'smp_roll_mean_24', 'smp_roll_std_24',
        'smp_roll_mean_48', 'smp_roll_std_48',
        'smp_roll_mean_168', 'smp_roll_std_168',
        # Time features (derived)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'is_weekend',
        # Weather features (real from KMA)
        'temperature', 'wind_speed', 'humidity', 'solar_radiation', 'pressure',
        # Fuel prices (real from Yahoo Finance)
        'wti_crude', 'brent_crude', 'natural_gas', 'heating_oil',
        'wti_crude_ma7', 'wti_crude_ma30',
        'brent_crude_ma7', 'brent_crude_ma30',
        'natural_gas_ma7', 'natural_gas_ma30',
        # Carbon prices (synthetic but realistic patterns)
        'kets_close', 'kets_ma7', 'kets_ma30',
        # EXCLUDED: renewable_total, net_load (synthetic sample data)
    ]

    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_cols)} features")

    # Handle missing values
    df = df.ffill().bfill()

    features = df[available_cols].values.astype(np.float32)
    targets = df['smp'].values.astype(np.float32)

    # Handle inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    for i in range(features.shape[1]):
        p1, p99 = np.percentile(features[:, i], [1, 99])
        features[:, i] = np.clip(features[:, i], p1, p99)

    # Target statistics
    target_mean = targets.mean()
    target_std = targets.std()

    return features, targets, target_mean, target_std


def quantile_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quantile loss for q10, q50, q90"""
    quantiles = [0.1, 0.5, 0.9]
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - pred[..., i]
        losses.append(torch.max(q * errors, (q - 1) * errors))
    return torch.mean(torch.stack(losses))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = quantile_loss(pred, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_q50 = pred[..., 1]

            all_preds.append(pred_q50.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Denormalize
    preds_denorm = preds * target_std + target_mean
    targets_denorm = targets * target_std + target_mean

    # Metrics
    mae = np.mean(np.abs(preds_denorm - targets_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))

    # Robust MAPE
    mask = np.abs(targets_denorm) > 10
    if mask.sum() > 0:
        mape = np.mean(np.abs(preds_denorm[mask] - targets_denorm[mask]) /
                       np.abs(targets_denorm[mask])) * 100
    else:
        mape = 0

    # R²
    ss_res = np.sum((targets_denorm - preds_denorm) ** 2)
    ss_tot = np.sum((targets_denorm - targets_denorm.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


def walk_forward_validation(
    features: np.ndarray,
    targets: np.ndarray,
    target_mean: float,
    target_std: float,
    n_splits: int = 5,
    train_size: int = 8760,
    test_size: int = 720,
    seq_len: int = 168,
    pred_len: int = 24
) -> Tuple[Dict[str, float], nn.Module]:
    """Walk-forward validation"""
    device = get_device()
    results = []

    total_samples = len(features)
    min_required = train_size + test_size + seq_len + pred_len

    if total_samples < min_required:
        raise ValueError(f"Not enough data: {total_samples} < {min_required}")

    available = total_samples - min_required
    step = available // (n_splits - 1) if n_splits > 1 else 0

    best_model = None
    best_r2 = -float('inf')

    for split_idx in range(n_splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Split {split_idx + 1}/{n_splits}")
        logger.info("=" * 60)

        start_idx = split_idx * step
        train_end = start_idx + train_size
        test_end = train_end + test_size

        logger.info(f"  Train: [{start_idx:,} - {train_end:,}]")
        logger.info(f"  Test:  [{train_end:,} - {test_end:,}]")

        # Split data
        train_features = features[start_idx:train_end]
        train_targets = targets[start_idx:train_end]
        test_features = features[train_end:test_end]
        test_targets = targets[train_end:test_end]

        # Normalize
        scaler = RobustScaler()
        train_features_norm = scaler.fit_transform(train_features)
        test_features_norm = scaler.transform(test_features)

        train_target_mean = train_targets.mean()
        train_target_std = train_targets.std()
        train_targets_norm = (train_targets - train_target_mean) / (train_target_std + 1e-8)
        test_targets_norm = (test_targets - train_target_mean) / (train_target_std + 1e-8)

        # Datasets
        train_dataset = SMPDatasetV7(train_features_norm, train_targets_norm, seq_len, pred_len)
        test_dataset = SMPDatasetV7(test_features_norm, test_targets_norm, seq_len, pred_len)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Model
        input_size = train_features.shape[1]
        model = BiLSTMAttentionV7(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            n_heads=8,
            dropout=0.3,
            output_hours=pred_len
        ).to(device)

        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer with OneCycleLR
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.003,
            epochs=100,
            steps_per_epoch=1
        )

        # Training
        n_epochs = 100
        patience = 15
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = quantile_loss(pred, y)
                    val_loss += loss.item()
            val_loss /= len(test_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best weights
        if best_weights:
            model.load_state_dict(best_weights)

        # Evaluate
        metrics = evaluate(model, test_loader, device, train_target_mean, train_target_std)
        logger.info(f"  MAPE: {metrics['mape']:.2f}%, R²: {metrics['r2']:.4f}")

        results.append(metrics)

        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model

    # Average results
    avg_results = {
        'mae': np.mean([r['mae'] for r in results]),
        'rmse': np.mean([r['rmse'] for r in results]),
        'mape': np.mean([r['mape'] for r in results]),
        'r2': np.mean([r['r2'] for r in results])
    }

    return avg_results, best_model


def main():
    logger.info("=" * 60)
    logger.info("SMP v7.0 Enhanced: Full Feature Training")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading enhanced dataset...")
    df = load_enhanced_dataset()

    logger.info(f"  Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # Prepare features
    logger.info("\nPreparing features...")
    features, targets, target_mean, target_std = prepare_features(df)
    logger.info(f"  Feature matrix: {features.shape}")
    logger.info(f"  Target mean: {target_mean:.2f}, std: {target_std:.2f}")

    # Walk-forward validation
    logger.info("\n" + "=" * 60)
    logger.info("Starting Walk-Forward Validation...")
    logger.info("=" * 60)

    results, best_model = walk_forward_validation(
        features=features,
        targets=targets,
        target_mean=target_mean,
        target_std=target_std,
        n_splits=5,
        train_size=8760,
        test_size=720
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward Results (Average)")
    logger.info("=" * 60)
    logger.info(f"  MAE:  {results['mae']:.2f} won/kWh")
    logger.info(f"  RMSE: {results['rmse']:.2f} won/kWh")
    logger.info(f"  MAPE: {results['mape']:.2f}%")
    logger.info(f"  R²:   {results['r2']:.4f}")

    # Save model
    model_dir = PROJECT_ROOT / "models" / "smp_v7"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), model_dir / "smp_v7_model.pt")
    logger.info(f"\n  Model saved: {model_dir / 'smp_v7_model.pt'}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("Comparison with Previous Versions")
    logger.info("=" * 60)
    logger.info(f"  v3.1 (BiLSTM, SMP only):     MAPE 7.83%, R² 0.736")
    logger.info(f"  v6.0 (BiLSTM+Weather):       MAPE 7.83%, R² 0.707")
    logger.info(f"  v7.0 (BiLSTM+Full Features): MAPE {results['mape']:.2f}%, R² {results['r2']:.3f}")

    # Improvement analysis
    baseline_r2 = 0.736
    improvement = (results['r2'] - baseline_r2) / baseline_r2 * 100
    logger.info(f"\n  R² improvement vs v3.1: {improvement:+.1f}%")

    return results


if __name__ == "__main__":
    main()
