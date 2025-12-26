#!/usr/bin/env python3
"""
SMP v3.8: Serial Ensemble (LightGBM + LSTM Residual Learning)

Strategy from Gemini discussion:
1. LightGBM captures trends, seasonality, and linear patterns
2. LSTM learns residuals (what LightGBM couldn't predict)
3. Final = LightGBM + LSTM residuals

This approach allows:
- LightGBM to handle the "easy" predictable patterns
- LSTM to focus on "hard" spikes and anomalies
- Better handling of extreme values (R² killer)
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

# LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Run: pip install lightgbm")

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
    """Data pipeline for ensemble model."""

    def __init__(self):
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
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

        return df

    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create features for LightGBM (tabular) and LSTM (sequential)."""
        self.feature_names = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        # ===== NO current-time features! =====
        # For fair comparison with LSTM (which uses past to predict future),
        # LightGBM should only use LAG features, not current values
        # Target: smp_mainland[t], Features: everything at t-1 or earlier
        features = {}

        # ===== Time features =====
        hour = df['hour'].values
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        day_of_week = df['datetime'].dt.dayofweek.values
        features['dayofweek'] = day_of_week
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        features['is_weekend'] = (day_of_week >= 5).astype(float)

        month = df['datetime'].dt.month.values
        features['month'] = month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)

        # ===== Season/Peak =====
        features['is_summer'] = ((month >= 6) & (month <= 8)).astype(float)
        features['is_winter'] = ((month == 12) | (month <= 2)).astype(float)
        features['peak_morning'] = ((hour >= 9) & (hour <= 12)).astype(float)
        features['peak_evening'] = ((hour >= 17) & (hour <= 21)).astype(float)
        features['off_peak'] = ((hour >= 1) & (hour <= 6)).astype(float)

        # ===== Lag features (important for LightGBM) =====
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 96]:
            features[f'smp_lag{lag}'] = smp_series.shift(lag).values

        # ===== Rolling statistics (SHIFTED by 1 to avoid leakage!) =====
        # Standard rolling includes current value, so we shift by 1
        # Example: ma24[t] = mean(smp[t-24:t-1]), NOT mean(smp[t-23:t])
        for window in [6, 12, 24, 48, 72]:
            features[f'smp_ma{window}'] = smp_series.rolling(window, min_periods=1).mean().shift(1).values
            features[f'smp_std{window}'] = smp_series.rolling(window, min_periods=1).std().shift(1).fillna(0).values

        # ===== Diff features (SHIFTED by 1 to avoid leakage!) =====
        # diff[t] should be smp[t-1] - smp[t-2], NOT smp[t] - smp[t-1]
        features['smp_diff1'] = smp_series.diff(1).shift(1).fillna(0).values
        features['smp_diff24'] = smp_series.diff(24).shift(1).fillna(0).values

        # Create DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values  # Target: Mainland SMP (same as v3.2)

        # Drop rows with NaN (from lag/rolling)
        feature_df = feature_df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in feature_df.columns if c not in ['datetime', 'target']]

        logger.info(f"  Features: {len(self.feature_names)}, Records: {len(feature_df)}")

        return feature_df


class ResidualDataset(Dataset):
    """Dataset for LSTM residual learning."""

    def __init__(self, X: np.ndarray, residuals: np.ndarray, seq_len: int = 48):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(residuals)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return X_seq, y_val


class ResidualLSTM(nn.Module):
    """LSTM for learning residuals (simpler than full BiLSTM)."""

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze(-1)


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM for trend prediction."""
    logger.info("\n[Stage 1] Training LightGBM for trend prediction...")

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Metrics
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)

    logger.info(f"  LightGBM Train R²: {train_r2:.4f}")
    logger.info(f"  LightGBM Val R²: {val_r2:.4f}")

    return model, train_pred, val_pred


def train_residual_lstm(X_train, residuals_train, X_val, residuals_val,
                        feature_count, device, config):
    """Train LSTM on residuals."""
    logger.info("\n[Stage 2] Training LSTM on residuals...")

    # Scale features and residuals
    scaler_X = StandardScaler()
    scaler_res = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    res_train_scaled = scaler_res.fit_transform(residuals_train.reshape(-1, 1)).flatten()
    res_val_scaled = scaler_res.transform(residuals_val.reshape(-1, 1)).flatten()

    # Datasets
    train_ds = ResidualDataset(X_train_scaled, res_train_scaled, config['seq_len'])
    val_ds = ResidualDataset(X_val_scaled, res_val_scaled, config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    # Model
    model = ResidualLSTM(
        input_size=feature_count,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                pred = model(X)
                val_loss += criterion(pred, y.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            logger.info(f"  Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if patience_counter >= config['patience']:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)

    return model, scaler_X, scaler_res


def predict_residuals(model, X, scaler_X, scaler_res, seq_len, device):
    """Predict residuals using LSTM."""
    X_scaled = scaler_X.transform(X)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(seq_len, len(X_scaled)):
            X_seq = torch.FloatTensor(X_scaled[i-seq_len:i]).unsqueeze(0).to(device)
            pred = model(X_seq).cpu().numpy()[0]
            predictions.append(pred)

    # Inverse transform
    predictions = np.array(predictions)
    predictions = scaler_res.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Pad with zeros for the first seq_len samples
    full_predictions = np.zeros(len(X))
    full_predictions[seq_len:] = predictions

    return full_predictions


def main():
    if not HAS_LIGHTGBM:
        logger.error("LightGBM is required. Run: pip install lightgbm")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.8: Serial Ensemble (LightGBM + LSTM Residual)")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipeline()
    df = pipeline.load_data()
    feature_df = pipeline.create_features(df)

    feature_names = pipeline.feature_names
    logger.info(f"Total features: {len(feature_names)}")

    # Prepare data
    X = feature_df[feature_names].values
    y = feature_df['target'].values

    # Train/Val/Test split (70/15/15)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ===== Stage 1: Train LightGBM =====
    lgb_model, lgb_train_pred, lgb_val_pred = train_lightgbm(
        X_train, y_train, X_val, y_val
    )

    # Calculate residuals
    residuals_train = y_train - lgb_train_pred
    residuals_val = y_val - lgb_val_pred

    logger.info(f"\n  Residuals Train: mean={residuals_train.mean():.2f}, std={residuals_train.std():.2f}")
    logger.info(f"  Residuals Val: mean={residuals_val.mean():.2f}, std={residuals_val.std():.2f}")

    # ===== Stage 2: Train LSTM on residuals =====
    lstm_config = {
        'seq_len': 48,  # Shorter for residual learning
        'hidden_size': 32,
        'num_layers': 1,
        'dropout': 0.1,
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'patience': 15
    }

    lstm_model, scaler_X, scaler_res = train_residual_lstm(
        X_train, residuals_train, X_val, residuals_val,
        len(feature_names), device, lstm_config
    )

    # ===== Final Evaluation =====
    logger.info("\n[Stage 3] Final Evaluation...")

    # LightGBM predictions on test set
    lgb_test_pred = lgb_model.predict(X_test)

    # LSTM residual predictions on test set
    lstm_residuals = predict_residuals(
        lstm_model, X_test, scaler_X, scaler_res,
        lstm_config['seq_len'], device
    )

    # Final ensemble prediction
    final_pred = lgb_test_pred + lstm_residuals

    # Metrics
    # LightGBM only
    lgb_mape = mean_absolute_percentage_error(y_test[y_test > 10], lgb_test_pred[y_test > 10]) * 100
    lgb_r2 = r2_score(y_test, lgb_test_pred)

    # Ensemble (LightGBM + LSTM)
    # Only evaluate after seq_len
    test_start = lstm_config['seq_len']
    y_test_valid = y_test[test_start:]
    final_pred_valid = final_pred[test_start:]

    mask = y_test_valid > 10
    ensemble_mape = mean_absolute_percentage_error(y_test_valid[mask], final_pred_valid[mask]) * 100
    ensemble_r2 = r2_score(y_test_valid, final_pred_valid)
    ensemble_mae = np.mean(np.abs(y_test_valid - final_pred_valid))

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"LightGBM only:     MAPE={lgb_mape:.2f}%, R²={lgb_r2:.4f}")
    logger.info(f"Ensemble (LGB+LSTM): MAPE={ensemble_mape:.2f}%, R²={ensemble_r2:.4f}, MAE={ensemble_mae:.2f}")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_8_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM
    lgb_model.save_model(str(output_dir / "lightgbm_model.txt"))

    # Save LSTM
    torch.save(lstm_model.state_dict(), output_dir / "lstm_residual_model.pt")

    # Save metrics
    metrics = {
        'lgb_mape': lgb_mape,
        'lgb_r2': lgb_r2,
        'ensemble_mape': ensemble_mape,
        'ensemble_r2': ensemble_r2,
        'ensemble_mae': ensemble_mae,
        'features': len(feature_names),
        'records': len(feature_df),
        'lstm_config': lstm_config,
        'timestamp': datetime.now().isoformat(),
        'strategy': 'LightGBM for trend + LSTM for residuals'
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Strategy |")
    logger.info("|-------|------|-----|----------|")
    logger.info(f"| v3.2 (Optuna) | 7.42% | 0.760 | BiLSTM+Attention |")
    logger.info(f"| v3.8 LightGBM only | {lgb_mape:.2f}% | {lgb_r2:.3f} | Tree-based |")
    logger.info(f"| v3.8 Ensemble | {ensemble_mape:.2f}% | {ensemble_r2:.3f} | LGB + LSTM residual |")

    if ensemble_r2 > 0.760:
        logger.info(f"\n✅ ENSEMBLE IMPROVED: R² +{ensemble_r2 - 0.760:.3f}")
    elif lgb_r2 > 0.760:
        logger.info(f"\n✅ LightGBM alone improved: R² +{lgb_r2 - 0.760:.3f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.2 baseline")

    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("TOP 10 FEATURE IMPORTANCE (LightGBM)")
    logger.info("=" * 60)
    importance = lgb_model.feature_importance(importance_type='gain')
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for fname, imp in feature_imp[:10]:
        logger.info(f"  {fname}: {imp:.0f}")


if __name__ == "__main__":
    main()
