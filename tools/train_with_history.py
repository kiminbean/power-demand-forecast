"""
Quick Training with Loss History
================================

Train models with history logging for loss curve visualization.

Usage:
    python tools/train_with_history.py

Author: Claude Code
Date: 2025-12
"""

import sys
from pathlib import Path
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings('ignore')

from data.dataset import TimeSeriesScaler, TimeSeriesDataset, split_data_by_time, get_device
from models.lstm import create_model
from features import add_time_features, add_lag_features, add_weather_features

# Configuration
FULL_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 15

MODEL_CONFIG = {
    'model_type': 'lstm',
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False
}

TIME_FEATURES = [
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos', 'is_weekend', 'is_holiday'
]
LAG_FEATURES = [
    'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
    'demand_ma_6h', 'demand_ma_24h', 'demand_std_24h',
    'demand_diff_1h', 'demand_diff_24h'
]


def prepare_data(include_weather: bool = False):
    """Prepare data for training"""
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    df = add_time_features(df, include_holiday=True)
    df = add_lag_features(df, demand_col='power_demand')

    if include_weather and '기온' in df.columns:
        df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)

    df = df.dropna()

    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_weather:
        WEATHER_FEATURES = ['기온', 'THI', 'HDD', 'CDD']
        for col in WEATHER_FEATURES:
            if col in df.columns:
                features.append(col)

    available = [f for f in features if f in df.columns]
    return df[available].copy(), available


def train_with_history(
    train_df, val_df, n_features, device,
    epochs=FULL_EPOCHS, model_name="model"
):
    """Train model and return history"""

    # Scaling
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)

    # Datasets
    seq_length = 168
    train_ds = TimeSeriesDataset(train_scaled, seq_length=seq_length, horizon=1)
    val_ds = TimeSeriesDataset(val_scaled, seq_length=seq_length, horizon=1)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    # Model
    model = create_model(
        model_type=MODEL_CONFIG['model_type'],
        input_size=n_features,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    print(f"\n  Training {model_name} ({epochs} epochs)...")
    print(f"  {'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<12}")
    print("  " + "-" * 50)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Print progress
        marker = " *" if epoch == best_epoch else ""
        print(f"  {epoch:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<12.6f}{marker}")

        scheduler.step(val_loss)

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss

    print(f"  Best epoch: {best_epoch} (val_loss: {best_val_loss:.6f})")

    return model, scaler, history


def plot_loss_curves(histories: dict, output_path: Path):
    """Plot loss curves for all models"""
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')

    for ax, (model_name, history) in zip(axes, histories.items()):
        epochs = history['epoch']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        best_epoch = history['best_epoch']

        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax.axvline(x=best_epoch, color='green', linestyle='--',
                   label=f'Best Epoch ({best_epoch})', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_loss(histories: dict, output_path: Path):
    """Plot combined loss comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training Analysis', fontsize=16, fontweight='bold')

    colors = {'Demand Only': 'steelblue', 'Weather Full': 'coral'}

    # 1. Train Loss Comparison
    ax1 = axes[0, 0]
    for model_name, history in histories.items():
        ax1.plot(history['epoch'], history['train_loss'],
                label=model_name, linewidth=2, color=colors.get(model_name, 'gray'))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss (MSE)')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Validation Loss Comparison
    ax2 = axes[0, 1]
    for model_name, history in histories.items():
        ax2.plot(history['epoch'], history['val_loss'],
                label=model_name, linewidth=2, color=colors.get(model_name, 'gray'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss (MSE)')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Learning Rate
    ax3 = axes[1, 0]
    for model_name, history in histories.items():
        ax3.plot(history['epoch'], history['learning_rate'],
                label=model_name, linewidth=2, color=colors.get(model_name, 'gray'))
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Final Metrics Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [['Model', 'Best Epoch', 'Best Val Loss']]
    for model_name, history in histories.items():
        table_data.append([
            model_name,
            str(history['best_epoch']),
            f"{history['best_val_loss']:.6f}"
        ])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)

    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data)):
        for j in range(3):
            table[(i, j)].set_facecolor('#E2EFDA' if j == 2 else '#D9E2F3')

    ax4.set_title('Training Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("Quick Training with Loss History")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    figures_dir = PROJECT_ROOT / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    histories = {}

    # 1. Demand Only Model
    print("\n[1/2] Training Demand Only model...")
    df_demand, features_demand = prepare_data(include_weather=False)
    train_df, val_df, _ = split_data_by_time(df_demand)

    model_demand, scaler_demand, history_demand = train_with_history(
        train_df, val_df, len(features_demand), device,
        epochs=FULL_EPOCHS, model_name="Demand Only"
    )
    histories['Demand Only'] = history_demand

    # 2. Weather Full Model
    print("\n[2/2] Training Weather Full model...")
    df_weather, features_weather = prepare_data(include_weather=True)
    train_df_w, val_df_w, _ = split_data_by_time(df_weather)

    model_weather, scaler_weather, history_weather = train_with_history(
        train_df_w, val_df_w, len(features_weather), device,
        epochs=FULL_EPOCHS, model_name="Weather Full"
    )
    histories['Weather Full'] = history_weather

    # Save histories
    history_path = PROJECT_ROOT / 'results' / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"\n  History saved: {history_path}")

    # Generate plots
    print("\n[3/3] Generating loss curve plots...")
    plot_loss_curves(histories, figures_dir / 'training_loss_curves.png')
    plot_combined_loss(histories, figures_dir / 'training_analysis.png')

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
