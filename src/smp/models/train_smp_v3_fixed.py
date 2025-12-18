"""
SMP v3.1 Training Pipeline - Fixed Version
==========================================

v3.0 학습 실패 원인 분석 및 수정:
- 클램핑 로직 제거 (gradient flow 방해)
- 손실 함수 단순화 (Huber + Quantile)
- v2.1 advanced 모델의 성공적인 구조 유지
- Attention 메커니즘 안정화

Target Metrics:
- MAPE: < 10% (v2.1: 10.68%)
- R²: > 0.65 (v2.1: 0.59)
- 80% Coverage: > 85% (v2.1: 82.5%)

Usage:
    python -m src.smp.models.train_smp_v3_fixed

Author: Claude Code
Date: 2025-12
Version: 3.1.0
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

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
# Configuration v3.1
# =============================================================================
@dataclass
class TrainingConfigV31:
    """v3.1 Training Configuration - Stabilized"""

    # Data
    data_path: str = 'data/smp/smp_5years_epsis.csv'
    output_dir: str = 'models/smp_v3'

    # Period
    train_start: str = '2022-01-01'
    train_end: str = '2024-12-31'

    # Sequence
    input_hours: int = 48
    output_hours: int = 24

    # Model (v2.1 proven architecture + Attention)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    n_heads: int = 4

    # Quantile
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Training
    batch_size: int = 64
    epochs: int = 150
    learning_rate: float = 0.001
    patience: int = 25

    # Noise Injection (proven in v2.1)
    noise_std: float = 0.02
    noise_prob: float = 0.5


# =============================================================================
# Data Pipeline (from v2.1)
# =============================================================================
class SMPDataPipelineV31:
    """v3.1 Data Pipeline - Based on v2.1 success"""

    def __init__(self, config: TrainingConfigV31):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
        """Load EPSIS data"""
        data_path = project_root / self.config.data_path
        df = pd.read_csv(data_path)

        # Valid data only
        df = df[df['smp_mainland'] > 0].copy()

        # Parse timestamp
        def fix_hour_24(ts):
            if ' 24:00' in str(ts):
                date_part = str(ts).replace(' 24:00', '')
                return pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return pd.to_datetime(ts)

        df['datetime'] = df['timestamp'].apply(fix_hour_24)
        df = df.sort_values('datetime').reset_index(drop=True)

        # Filter period
        df = df[(df['datetime'] >= self.config.train_start) &
                (df['datetime'] <= self.config.train_end)]

        logger.info(f"Data loaded: {len(df):,} records ({df['datetime'].min()} ~ {df['datetime'].max()})")

        return df

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features - Same as v2.1 (proven)"""
        features = []
        smp = df['smp_mainland'].values

        # === 1. Base price features (4) ===
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)
        self.feature_names.extend(['smp_mainland', 'smp_jeju', 'smp_max', 'smp_min'])

        # === 2. Time cyclical features (6) ===
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

        # === 3. Season/Peak features (6) ===
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

        # === 4. Statistical features (4) ===
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

        # === 5. Lag features (2) ===
        smp_lag_24 = pd.Series(smp).shift(24).bfill().values
        smp_lag_168 = pd.Series(smp).shift(168).bfill().values  # 1 week
        features.append(smp_lag_24)
        features.append(smp_lag_168)
        self.feature_names.extend(['smp_lag_24h', 'smp_lag_168h'])

        feature_array = np.column_stack(features)
        target_array = smp

        logger.info(f"Features created: {len(self.feature_names)}")

        return feature_array, target_array


# =============================================================================
# Attention Module (Stabilized)
# =============================================================================
class StableAttention(nn.Module):
    """Stabilized Self-Attention"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** 0.5

        # Initialize with small values for stability
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape

        # Project
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output
        context = torch.matmul(attn, V)
        output = context.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_O(output)

        # Average attention weights for XAI
        avg_attn = attn.mean(dim=1).mean(dim=1)  # (B, L)

        return output, avg_attn


# =============================================================================
# Model v3.1
# =============================================================================
class SMPModelV31(nn.Module):
    """v3.1 SMP Model - BiLSTM + Stable Attention"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        n_heads: int = 4,
        prediction_hours: int = 24,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prediction_hours = prediction_hours
        self.quantiles = quantiles

        lstm_output = hidden_size * 2 if bidirectional else hidden_size

        # Encoder: BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention
        self.attention = StableAttention(lstm_output, n_heads, dropout)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(lstm_output)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(lstm_output, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_hours)
        )

        # Quantile heads
        self.quantile_heads = nn.ModuleDict({
            f'q{int(q*100)}': nn.Sequential(
                nn.Linear(lstm_output, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, prediction_hours)
            ) for q in quantiles
        })

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, Any]:
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, attn_weights = self.attention(lstm_out)

        # Residual + LayerNorm
        out = self.layer_norm(lstm_out + attn_out)

        # Use last timestep
        features = out[:, -1, :]

        # Main output
        result = {
            'point': self.fc(features),
            'quantiles': {
                name: head(features) for name, head in self.quantile_heads.items()
            }
        }

        if return_attention:
            result['attention'] = attn_weights

        return result


# =============================================================================
# Loss Functions
# =============================================================================
class QuantileLoss(nn.Module):
    """Quantile Loss"""

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = []

        for q in self.quantiles:
            key = f'q{int(q*100)}'
            if key not in predictions:
                continue

            pred = predictions[key]
            errors = targets - pred

            loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss.mean())

        return sum(losses) / len(losses) if losses else torch.tensor(0.0)


class CombinedLossV31(nn.Module):
    """v3.1 Combined Loss - Stable version"""

    def __init__(self, quantiles: List[float], mse_weight: float = 0.5, quantile_weight: float = 0.5):
        super().__init__()
        self.mse_loss = nn.SmoothL1Loss()  # Huber Loss
        self.quantile_loss = QuantileLoss(quantiles)
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight

    def forward(self, predictions: Dict[str, Any], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # MSE on point prediction
        mse = self.mse_loss(predictions['point'], targets)

        # Quantile loss
        quantile = self.quantile_loss(predictions['quantiles'], targets)

        total = self.mse_weight * mse + self.quantile_weight * quantile

        return total, {
            'total': total.item(),
            'mse': mse.item(),
            'quantile': quantile.item()
        }


# =============================================================================
# Dataset
# =============================================================================
class SMPDatasetV31(Dataset):
    """v3.1 Dataset with noise injection"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_hours: int = 48,
        output_hours: int = 24,
        noise_std: float = 0.0,
        noise_prob: float = 0.0
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.input_hours = input_hours
        self.output_hours = output_hours
        self.noise_std = noise_std
        self.noise_prob = noise_prob

        self.valid_indices = list(range(
            input_hours,
            len(features) - output_hours
        ))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.valid_indices[idx]

        x = self.features[real_idx - self.input_hours:real_idx].clone()
        y = self.targets[real_idx:real_idx + self.output_hours]

        # Noise injection
        if self.training and self.noise_std > 0 and np.random.random() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return {'x': x, 'y': y}


# =============================================================================
# Trainer
# =============================================================================
class TrainerV31:
    """v3.1 Trainer"""

    def __init__(self, config: TrainingConfigV31):
        self.config = config
        self.device = get_device()
        self.best_mape = float('inf')
        self.patience_counter = 0

        logger.info(f"Device: {self.device}")

    def train(self):
        """Training pipeline"""
        # Data preparation
        pipeline = SMPDataPipelineV31(self.config)
        df = pipeline.load_data()

        # Feature engineering
        features, targets = pipeline.create_features(df)

        # Normalize
        features_normalized = pipeline.scaler.fit_transform(features)

        # Target normalization (simple - mean/std)
        target_mean = targets.mean()
        target_std = targets.std()
        targets_normalized = (targets - target_mean) / target_std

        # Dataset
        train_size = int(len(features) * 0.8)

        train_dataset = SMPDatasetV31(
            features_normalized[:train_size],
            targets_normalized[:train_size],
            self.config.input_hours,
            self.config.output_hours,
            self.config.noise_std,
            self.config.noise_prob
        )
        train_dataset.training = True

        val_dataset = SMPDatasetV31(
            features_normalized[train_size:],
            targets_normalized[train_size:],
            self.config.input_hours,
            self.config.output_hours
        )
        val_dataset.training = False

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Model
        model = SMPModelV31(
            input_size=len(pipeline.feature_names),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            n_heads=self.config.n_heads,
            prediction_hours=self.config.output_hours,
            quantiles=self.config.quantiles
        ).to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        # Loss & Optimizer
        criterion = CombinedLossV31(self.config.quantiles)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        best_model_state = None

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_losses = []

            for batch in train_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                optimizer.zero_grad()
                predictions = model(x)
                loss, loss_dict = criterion(predictions, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss_dict['total'])

            # Validation
            model.eval()
            val_predictions = []
            val_targets_list = []
            val_q10 = []
            val_q90 = []
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(self.device)
                    y = batch['y'].to(self.device)

                    predictions = model(x)
                    loss, _ = criterion(predictions, y)

                    val_losses.append(loss.item())
                    val_predictions.append(predictions['point'].cpu().numpy())
                    val_targets_list.append(y.cpu().numpy())
                    val_q10.append(predictions['quantiles']['q10'].cpu().numpy())
                    val_q90.append(predictions['quantiles']['q90'].cpu().numpy())

            val_predictions = np.concatenate(val_predictions)
            val_targets_arr = np.concatenate(val_targets_list)
            val_q10 = np.concatenate(val_q10)
            val_q90 = np.concatenate(val_q90)

            # Inverse transform
            val_pred_original = val_predictions * target_std + target_mean
            val_target_original = val_targets_arr * target_std + target_mean
            val_q10_original = val_q10 * target_std + target_mean
            val_q90_original = val_q90 * target_std + target_mean

            # Metrics
            valid_mask = val_target_original.flatten() > 0
            mape = np.mean(np.abs(val_pred_original.flatten()[valid_mask] - val_target_original.flatten()[valid_mask]) /
                          val_target_original.flatten()[valid_mask]) * 100
            r2 = r2_score(val_target_original.flatten()[valid_mask], val_pred_original.flatten()[valid_mask])

            # Coverage
            in_interval = (val_target_original >= val_q10_original) & (val_target_original <= val_q90_original)
            coverage = np.mean(in_interval) * 100

            scheduler.step(np.mean(val_losses))

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                           f"Train: {np.mean(train_losses):.4f}, Val: {np.mean(val_losses):.4f}, "
                           f"MAPE: {mape:.2f}%, R2: {r2:.4f}, Coverage: {coverage:.1f}%")

            # Best model
            if mape < self.best_mape:
                self.best_mape = mape
                best_model_state = model.state_dict().copy()
                self.patience_counter = 0
                best_metrics = {
                    'mape': mape,
                    'r2': r2,
                    'coverage': coverage,
                    'mae': mean_absolute_error(val_target_original.flatten(), val_pred_original.flatten()),
                    'rmse': np.sqrt(mean_squared_error(val_target_original.flatten(), val_pred_original.flatten()))
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Save
        self._save_model(model, pipeline, best_metrics, target_mean, target_std)

        return model, pipeline

    def _save_model(self, model, pipeline, metrics, target_mean, target_std):
        """Save model and results"""
        output_dir = project_root / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("v3.1 Final Results")
        logger.info("=" * 60)
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"MAE: {metrics['mae']:.2f} won/kWh")
        logger.info(f"RMSE: {metrics['rmse']:.2f} won/kWh")
        logger.info(f"R2: {metrics['r2']:.4f}")
        logger.info(f"80% Coverage: {metrics['coverage']:.1f}%")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config.__dict__,
            'model_kwargs': {
                'input_size': len(pipeline.feature_names),
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'bidirectional': self.config.bidirectional,
                'n_heads': self.config.n_heads,
                'prediction_hours': self.config.output_hours,
                'quantiles': self.config.quantiles
            },
            'feature_names': pipeline.feature_names,
            'target_mean': target_mean,
            'target_std': target_std,
            'version': '3.1.0'
        }, output_dir / 'smp_v3_model.pt')

        # Save scaler
        np.save(output_dir / 'smp_v3_scaler.npy', {
            'feature_scaler_mean': pipeline.scaler.mean_,
            'feature_scaler_scale': pipeline.scaler.scale_,
            'target_mean': target_mean,
            'target_std': target_std,
            'feature_names': pipeline.feature_names
        })

        # Save metrics
        with open(output_dir / 'smp_v3_metrics.json', 'w') as f:
            json.dump({
                'version': '3.1.0',
                'mape': float(metrics['mape']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'r2': float(metrics['r2']),
                'coverage_80': float(metrics['coverage']),
                'parameters': sum(p.numel() for p in model.parameters()),
                'features': len(pipeline.feature_names),
                'improvements': [
                    'BiLSTM + Stable Attention',
                    'Huber Loss + Quantile Loss',
                    'Noise Injection',
                    'Gradient Clipping'
                ],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Model saved: {output_dir}")


# =============================================================================
# Main
# =============================================================================
def main():
    """Train v3.1 model"""
    logger.info("=" * 60)
    logger.info("SMP v3.1 Training - Fixed Version")
    logger.info("=" * 60)

    config = TrainingConfigV31()
    trainer = TrainerV31(config)

    model, pipeline = trainer.train()

    logger.info("=" * 60)
    logger.info("v3.1 Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
