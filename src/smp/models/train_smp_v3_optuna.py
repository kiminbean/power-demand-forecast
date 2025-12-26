#!/usr/bin/env python3
"""
SMP v3.2 Training with Optuna Hyperparameter Tuning
====================================================

Based on v3.1 (best baseline) with Optuna optimization.

Hyperparameters to tune:
- input_hours: 24, 48, 72, 96
- hidden_size: 32, 64, 128, 256
- num_layers: 1, 2, 3
- dropout: 0.1 - 0.5
- n_heads: 2, 4, 8
- learning_rate: 1e-4 to 1e-2
- batch_size: 32, 64, 128
- noise_std: 0.0 - 0.05

Target: R² > 0.80, MAPE < 7%

Usage:
    python -m src.smp.models.train_smp_v3_optuna [--n_trials 50]
"""

import os
import sys
import json
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# Optuna
try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed. Run: pip install optuna")

# Project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.smp.models.smp_lstm import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Pipeline
# =============================================================================
class SMPDataPipeline:
    """Data pipeline for SMP model"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, train_start: str = '2022-01-01', train_end: str = '2024-12-31') -> pd.DataFrame:
        """Load EPSIS data"""
        data_path = project_root / 'data' / 'smp' / 'smp_5years_epsis.csv'
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

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features (same as v3.1)"""
        features = []
        smp = df['smp_mainland'].values

        # Base price features (4)
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)
        self.feature_names.extend(['smp_mainland', 'smp_jeju', 'smp_max', 'smp_min'])

        # Time cyclical (8)
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

        # Season/Peak (5)
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

        # Statistical (4)
        smp_ma24 = pd.Series(smp).rolling(24, min_periods=1).mean().values
        smp_std24 = pd.Series(smp).rolling(24, min_periods=1).std().fillna(0).values
        features.append(smp_ma24)
        features.append(smp_std24)
        self.feature_names.extend(['smp_ma24', 'smp_std24'])

        smp_diff = np.diff(smp, prepend=smp[0])
        smp_range = df['smp_max'].values - df['smp_min'].values
        features.append(smp_diff)
        features.append(smp_range)
        self.feature_names.extend(['smp_diff', 'smp_range'])

        # Lag (2)
        smp_lag_24 = pd.Series(smp).shift(24).bfill().values
        smp_lag_168 = pd.Series(smp).shift(168).bfill().values
        features.append(smp_lag_24)
        features.append(smp_lag_168)
        self.feature_names.extend(['smp_lag_24h', 'smp_lag_168h'])

        return np.column_stack(features), smp


# =============================================================================
# Model
# =============================================================================
class StableAttention(nn.Module):
    """Self-Attention with stability"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5

        for m in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        output = context.transpose(1, 2).contiguous().view(B, L, D)

        return self.W_O(output)


class SMPModel(nn.Module):
    """BiLSTM + Attention model"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_heads: int = 4,
        output_hours: int = 24
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        lstm_out_size = hidden_size * 2
        self.attention = StableAttention(lstm_out_size, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        return self.fc(out[:, -1, :])


# =============================================================================
# Dataset
# =============================================================================
class SMPDataset(Dataset):
    """SMP dataset with noise injection"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_hours: int = 48,
        output_hours: int = 24,
        noise_std: float = 0.0,
        training: bool = True
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.input_hours = input_hours
        self.output_hours = output_hours
        self.noise_std = noise_std
        self.training = training

        self.valid_indices = list(range(input_hours, len(features) - output_hours))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx - self.input_hours:real_idx].clone()
        y = self.targets[real_idx:real_idx + self.output_hours]

        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x, y


# =============================================================================
# Training Function
# =============================================================================
def train_model(
    config: Dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    verbose: bool = False
) -> Tuple[float, float]:
    """Train model with given config and return MAPE, R²"""

    # Normalize
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    target_mean, target_std = targets.mean(), targets.std()
    targets_norm = (targets - target_mean) / target_std

    # Split
    train_size = int(len(features) * 0.8)

    train_ds = SMPDataset(
        features_norm[:train_size],
        targets_norm[:train_size],
        config['input_hours'],
        24,
        config['noise_std'],
        training=True
    )
    val_ds = SMPDataset(
        features_norm[train_size:],
        targets_norm[train_size:],
        config['input_hours'],
        24,
        0.0,
        training=False
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # Model
    model = SMPModel(
        input_size=features.shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        n_heads=config['n_heads'],
        output_hours=24
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.SmoothL1Loss()

    best_mape = float('inf')
    patience = 0
    max_patience = 20

    for epoch in range(config['max_epochs']):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        scheduler.step(val_loss / len(val_loader))

        preds = np.concatenate(all_preds) * target_std + target_mean
        trues = np.concatenate(all_targets) * target_std + target_mean

        mask = trues.flatten() > 0
        mape = np.mean(np.abs((trues.flatten()[mask] - preds.flatten()[mask]) / trues.flatten()[mask])) * 100
        r2 = r2_score(trues.flatten()[mask], preds.flatten()[mask])

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: MAPE={mape:.2f}%, R²={r2:.4f}")

        if mape < best_mape:
            best_mape = mape
            best_r2 = r2
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    return best_mape, best_r2


# =============================================================================
# Optuna Objective
# =============================================================================
def objective(trial: Trial, features: np.ndarray, targets: np.ndarray, device: torch.device) -> float:
    """Optuna objective function"""

    config = {
        'input_hours': trial.suggest_categorical('input_hours', [24, 48, 72, 96]),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'noise_std': trial.suggest_float('noise_std', 0.0, 0.05),
        'max_epochs': 50
    }

    # Ensure n_heads divides hidden_size * 2
    lstm_out = config['hidden_size'] * 2
    if lstm_out % config['n_heads'] != 0:
        config['n_heads'] = 4  # Safe default

    mape, r2 = train_model(config, features, targets, device, verbose=False)

    # Optuna minimizes, so return MAPE (lower is better)
    # But also penalize if R² is negative
    if r2 < 0:
        return 100.0  # Bad trial

    return mape


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--train_best', action='store_true', help='Train with best params after tuning')
    args = parser.parse_args()

    if not HAS_OPTUNA:
        logger.error("Optuna required. Install with: pip install optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.2 Optuna Hyperparameter Tuning")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    pipeline = SMPDataPipeline()
    df = pipeline.load_data()
    features, targets = pipeline.create_features(df)

    logger.info(f"Data: {len(features)} samples, {features.shape[1]} features")
    logger.info(f"Target: mean={targets.mean():.2f}, std={targets.std():.2f}")

    # Optuna study
    logger.info(f"\nStarting {args.n_trials} trials...")

    study = optuna.create_study(
        direction='minimize',
        study_name='smp_v3_optuna',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, features, targets, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("Optuna Results")
    logger.info("=" * 60)
    logger.info(f"Best MAPE: {study.best_value:.2f}%")
    logger.info(f"Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Save results
    output_dir = project_root / 'models' / 'smp_v3_optuna'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'optuna_results.json', 'w') as f:
        json.dump({
            'best_mape': study.best_value,
            'best_params': study.best_params,
            'n_trials': args.n_trials,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Train best model with full epochs
    if args.train_best:
        logger.info("\n" + "=" * 60)
        logger.info("Training Best Model (Full)")
        logger.info("=" * 60)

        best_config = study.best_params.copy()
        best_config['max_epochs'] = 150

        # Ensure n_heads divides hidden_size * 2
        lstm_out = best_config['hidden_size'] * 2
        if lstm_out % best_config['n_heads'] != 0:
            best_config['n_heads'] = 4

        mape, r2 = train_model(best_config, features, targets, device, verbose=True)

        logger.info(f"\nFinal Best Model:")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")

        # Save final metrics
        with open(output_dir / 'best_model_metrics.json', 'w') as f:
            json.dump({
                'mape': mape,
                'r2': r2,
                'config': best_config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    # Top 5 trials
    logger.info("\nTop 5 Trials:")
    trials_df = study.trials_dataframe()
    top5 = trials_df.nsmallest(5, 'value')
    for i, row in top5.iterrows():
        logger.info(f"  #{row['number']}: MAPE={row['value']:.2f}%")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
