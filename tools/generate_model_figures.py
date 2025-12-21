"""
Model Performance Visualization Script
======================================

Generate performance graphs for all production models:
1. Demand Only (Power Demand LSTM)
2. Weather Full (Power Demand LSTM with Temperature)
3. SMP Prediction (BiLSTM-Attention v3.1)

Usage:
    python tools/generate_model_figures.py

Author: Claude Code
Date: 2025-12
"""

import sys
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings('ignore')

# Korean font setup for macOS - Use Nanum Gothic or fallback
def setup_korean_font():
    """Setup Korean font for matplotlib"""
    font_candidates = [
        'NanumGothic',
        'NanumBarunGothic',
        'AppleGothic',
        'Malgun Gothic',
        'NotoSansCJK',
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"  Using font: {font}")
            break
    else:
        # Fallback to default with warning
        print("  Warning: Korean font not found, using default")

    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200

# Import project modules
from data.dataset import TimeSeriesScaler, TimeSeriesDataset, split_data_by_time, get_device
from models.lstm import create_model
from features import add_time_features, add_lag_features, add_weather_features


def load_demand_model(model_name: str, model_dir: Path):
    """Load demand prediction model"""
    checkpoint = torch.load(model_dir / f"{model_name}.pt", map_location='cpu')

    model = create_model(
        model_type=checkpoint['model_config']['model_type'],
        input_size=checkpoint['n_features'],
        hidden_size=checkpoint['model_config']['hidden_size'],
        num_layers=checkpoint['model_config']['num_layers'],
        dropout=checkpoint['model_config']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = TimeSeriesScaler()
    scaler.load(str(model_dir / f"{model_name}_scaler.pkl"))

    return model, scaler, checkpoint


def prepare_demand_data(include_weather: bool = False):
    """Prepare data for demand prediction models"""
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    # Feature engineering
    df = add_time_features(df, include_holiday=True)
    df = add_lag_features(df, demand_col='power_demand')

    if include_weather and '기온' in df.columns:
        df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)

    df = df.dropna()

    # Feature selection
    TIME_FEATURES = [
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos', 'is_weekend', 'is_holiday'
    ]
    LAG_FEATURES = [
        'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
        'demand_ma_6h', 'demand_ma_24h', 'demand_std_24h',
        'demand_diff_1h', 'demand_diff_24h'
    ]
    WEATHER_FEATURES = ['기온', 'THI', 'HDD', 'CDD']

    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_weather:
        for col in WEATHER_FEATURES:
            if col in df.columns:
                features.append(col)

    available = [f for f in features if f in df.columns]
    return df[available].copy(), available


def generate_demand_predictions(model, scaler, test_df, device, config):
    """Generate predictions for demand model"""
    test_scaled = scaler.transform(test_df.values)
    test_ds = TimeSeriesDataset(
        test_scaled,
        seq_length=config['seq_length'],
        horizon=config['horizon']
    )
    test_loader = DataLoader(test_ds, batch_size=64)

    model = model.to(device)
    model.eval()

    all_preds, all_actuals = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    actuals = np.concatenate(all_actuals).flatten()

    preds = scaler.inverse_transform_target(preds)
    actuals = scaler.inverse_transform_target(actuals)

    return actuals, preds


def plot_data_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    model_name: str, output_path: Path):
    """Plot train/validation/test data split visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{model_name} - Train/Validation/Test Data Split', fontsize=16, fontweight='bold')

    # 1. Full time series with split regions
    ax1 = axes[0]

    # Get datetime index
    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = test_df.index

    # Plot each segment
    ax1.plot(train_idx, train_df['power_demand'], 'b-', alpha=0.7, linewidth=0.5, label=f'Train ({len(train_df):,} samples)')
    ax1.plot(val_idx, val_df['power_demand'], 'g-', alpha=0.7, linewidth=0.5, label=f'Validation ({len(val_df):,} samples)')
    ax1.plot(test_idx, test_df['power_demand'], 'r-', alpha=0.7, linewidth=0.5, label=f'Test ({len(test_df):,} samples)')

    # Add vertical lines at split points
    ax1.axvline(x=val_idx[0], color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=test_idx[0], color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Power Demand (MW)')
    ax1.set_title('Time Series Data Split')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Distribution comparison
    ax2 = axes[1]

    # Plot distributions
    ax2.hist(train_df['power_demand'], bins=50, alpha=0.5, label=f'Train (mean={train_df["power_demand"].mean():.1f})', color='blue', density=True)
    ax2.hist(val_df['power_demand'], bins=50, alpha=0.5, label=f'Val (mean={val_df["power_demand"].mean():.1f})', color='green', density=True)
    ax2.hist(test_df['power_demand'], bins=50, alpha=0.5, label=f'Test (mean={test_df["power_demand"].mean():.1f})', color='red', density=True)

    ax2.set_xlabel('Power Demand (MW)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Comparison by Split')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_demand_model(model_name: str, display_name: str, actuals: np.ndarray,
                      preds: np.ndarray, metrics: dict, output_path: Path):
    """Create performance visualization for demand model"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{display_name} Model Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted (last 7 days)
    ax1 = axes[0, 0]
    n_points = min(168, len(actuals))  # 7 days
    x = np.arange(n_points)
    ax1.plot(x, actuals[-n_points:], 'b-', label='Actual', alpha=0.8, linewidth=1.5)
    ax1.plot(x, preds[-n_points:], 'r--', label='Predicted', alpha=0.8, linewidth=1.5)
    ax1.fill_between(x, actuals[-n_points:], preds[-n_points:], alpha=0.2, color='gray')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power Demand (MW)')
    ax1.set_title('Actual vs Predicted (Last 7 Days)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(actuals, preds, alpha=0.3, s=10, c='steelblue')
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax2.set_xlabel('Actual (MW)')
    ax2.set_ylabel('Predicted (MW)')
    ax2.set_title(f'Scatter Plot (R² = {metrics["R2"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = axes[1, 0]
    errors = preds - actuals
    ax3.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {errors.mean():.2f}')
    ax3.set_xlabel('Prediction Error (MW)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution (MAE = {metrics["MAE"]:.2f} MW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Metrics summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['MAPE', f'{metrics["MAPE"]:.2f} %'],
        ['MAE', f'{metrics["MAE"]:.2f} MW'],
        ['RMSE', f'{metrics["RMSE"]:.2f} MW'],
        ['R²', f'{metrics["R2"]:.4f}'],
        ['sMAPE', f'{metrics.get("sMAPE", 0):.2f} %'],
        ['MBE', f'{metrics.get("MBE", 0):.2f} MW'],
    ]

    # Create table
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style data cells
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#D9E2F3')
        table[(i, 1)].set_facecolor('#E2EFDA')

    ax4.set_title(f'{display_name} Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_smp_predictions():
    """Generate SMP model predictions and visualization"""
    # Load SMP data
    data_path = PROJECT_ROOT / 'data' / 'smp' / 'smp_5years_epsis.csv'
    df = pd.read_csv(data_path)
    df = df[df['smp_mainland'] > 0].copy()

    # Parse datetime
    def fix_hour_24(ts):
        if ' 24:00' in str(ts):
            date_part = str(ts).replace(' 24:00', '')
            return pd.to_datetime(date_part) + pd.Timedelta(days=1)
        return pd.to_datetime(ts)

    df['datetime'] = df['timestamp'].apply(fix_hour_24)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Filter date range
    df = df[(df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2024-12-31')]

    return df


def plot_smp_data_split(df: pd.DataFrame, output_path: Path):
    """Plot SMP train/validation/test data split"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('SMP Model - Train/Validation/Test Data Split', fontsize=16, fontweight='bold')

    # Split data (70/15/15)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # 1. Full time series with split regions
    ax1 = axes[0]
    ax1.plot(train_df['datetime'], train_df['smp_mainland'], 'b-', alpha=0.7, linewidth=0.5,
             label=f'Train ({len(train_df):,} samples)')
    ax1.plot(val_df['datetime'], val_df['smp_mainland'], 'g-', alpha=0.7, linewidth=0.5,
             label=f'Validation ({len(val_df):,} samples)')
    ax1.plot(test_df['datetime'], test_df['smp_mainland'], 'r-', alpha=0.7, linewidth=0.5,
             label=f'Test ({len(test_df):,} samples)')

    ax1.axvline(x=val_df['datetime'].iloc[0], color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=test_df['datetime'].iloc[0], color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('SMP (KRW/kWh)')
    ax1.set_title('SMP Time Series Data Split (2022-2024)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Distribution comparison
    ax2 = axes[1]
    ax2.hist(train_df['smp_mainland'], bins=50, alpha=0.5,
             label=f'Train (mean={train_df["smp_mainland"].mean():.1f})', color='blue', density=True)
    ax2.hist(val_df['smp_mainland'], bins=50, alpha=0.5,
             label=f'Val (mean={val_df["smp_mainland"].mean():.1f})', color='green', density=True)
    ax2.hist(test_df['smp_mainland'], bins=50, alpha=0.5,
             label=f'Test (mean={test_df["smp_mainland"].mean():.1f})', color='red', density=True)

    ax2.set_xlabel('SMP (KRW/kWh)')
    ax2.set_ylabel('Density')
    ax2.set_title('SMP Distribution Comparison by Split')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_smp_model(df: pd.DataFrame, metrics: dict, output_path: Path):
    """Create SMP model performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SMP Prediction Model (LSTM-Attention v3.1) Performance', fontsize=16, fontweight='bold')

    # Recent SMP data (last 30 days)
    recent_df = df.tail(720)  # 30 days * 24 hours

    # 1. SMP Time series (Jeju)
    ax1 = axes[0, 0]
    ax1.plot(recent_df['datetime'], recent_df['smp_jeju'], 'b-', alpha=0.8, linewidth=1)
    ax1.axhline(y=recent_df['smp_jeju'].mean(), color='r', linestyle='--',
                label=f"Mean: {recent_df['smp_jeju'].mean():.1f} KRW")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SMP (KRW/kWh)')
    ax1.set_title('Jeju SMP Time Series (Last 30 Days)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. SMP Distribution (Jeju)
    ax2 = axes[0, 1]
    ax2.hist(df['smp_jeju'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=df['smp_jeju'].mean(), color='r', linestyle='--', linewidth=2,
                label=f"Mean: {df['smp_jeju'].mean():.1f} KRW")
    ax2.axvline(x=df['smp_jeju'].median(), color='orange', linestyle='-', linewidth=2,
                label=f"Median: {df['smp_jeju'].median():.1f} KRW")
    ax2.set_xlabel('SMP (KRW/kWh)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Jeju SMP Distribution (2022-2024)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Hourly SMP pattern (Jeju)
    ax3 = axes[1, 0]
    hourly_avg = df.groupby(df['datetime'].dt.hour)['smp_jeju'].mean()
    ax3.bar(hourly_avg.index, hourly_avg.values, color='steelblue', edgecolor='white')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Average SMP (KRW/kWh)')
    ax3.set_title('Jeju Hourly Average SMP Pattern')
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Metrics summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['MAPE', f'{metrics["mape"]:.2f} %'],
        ['MAE', f'{metrics["mae"]:.2f} KRW'],
        ['RMSE', f'{metrics["rmse"]:.2f} KRW'],
        ['R²', f'{metrics["r2"]:.4f}'],
        ['Coverage 80%', f'{metrics["coverage_80"]:.1f} %'],
        ['Parameters', f'{metrics["parameters"]:,}'],
        ['Features', f'{metrics["features"]}'],
    ]

    # Create table
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style data cells
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#FFF2CC')
        table[(i, 1)].set_facecolor('#E2EFDA')

    ax4.set_title('SMP Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_comparison_chart(demand_metrics: dict, weather_metrics: dict,
                           smp_metrics: dict, output_path: Path):
    """Create model comparison chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # 1. Power Demand Models Comparison
    ax1 = axes[0]
    models = ['Demand Only', 'Weather Full']
    mape_values = [demand_metrics['MAPE'], weather_metrics['MAPE']]
    r2_values = [demand_metrics['R2'] * 100, weather_metrics['R2'] * 100]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mape_values, width, label='MAPE (%)', color='steelblue')
    ax1.bar_label(bars1, fmt='%.2f%%', padding=3)

    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, r2_values, width, label='R² (x100)', color='coral')
    ax1_twin.bar_label(bars2, fmt='%.1f', padding=3)

    ax1.set_ylabel('MAPE (%)')
    ax1_twin.set_ylabel('R² (x100)')
    ax1.set_xlabel('Model')
    ax1.set_title('Power Demand Model Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_ylim(0, 10)
    ax1_twin.set_ylim(0, 100)

    # 2. All Models Summary
    ax2 = axes[1]
    all_models = ['Demand Only', 'Weather Full', 'SMP v3.1']
    all_mape = [demand_metrics['MAPE'], weather_metrics['MAPE'], smp_metrics['mape']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax2.barh(all_models, all_mape, color=colors, edgecolor='white')
    ax2.bar_label(bars, fmt='%.2f%%', padding=5)
    ax2.set_xlabel('MAPE (%)')
    ax2.set_title('All Models MAPE Comparison')
    ax2.set_xlim(0, 12)
    ax2.grid(True, alpha=0.3, axis='x')

    # Best model indicator
    best_idx = np.argmin(all_mape)
    ax2.annotate('* Best', xy=(all_mape[best_idx] + 0.5, best_idx),
                fontsize=12, color='gold', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main function to generate all figures"""
    print("=" * 60)
    print("Model Performance Figure Generation")
    print("=" * 60)

    figures_dir = PROJECT_ROOT / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load training report
    training_report_path = PROJECT_ROOT / 'models' / 'production' / 'training_report.json'
    with open(training_report_path) as f:
        training_report = json.load(f)

    # Load SMP metrics
    smp_metrics_path = PROJECT_ROOT / 'models' / 'smp_v3' / 'smp_v3_metrics.json'
    with open(smp_metrics_path) as f:
        smp_metrics = json.load(f)

    config = {'seq_length': 168, 'horizon': 1}

    # =====================================================
    # 1. Demand Only Model
    # =====================================================
    print("\n[1/6] Generating Demand Only model figure...")

    try:
        model_demand, scaler_demand, ckpt_demand = load_demand_model(
            'demand_only', PROJECT_ROOT / 'models' / 'production'
        )

        df_demand, _ = prepare_demand_data(include_weather=False)
        train_df_demand, val_df_demand, test_df_demand = split_data_by_time(df_demand)

        actuals_demand, preds_demand = generate_demand_predictions(
            model_demand, scaler_demand, test_df_demand, device, config
        )

        metrics_demand = training_report['results']['demand_only']['metrics']
        plot_demand_model(
            'demand_only', 'Demand Only',
            actuals_demand, preds_demand, metrics_demand,
            figures_dir / 'demand_only_performance.png'
        )
    except Exception as e:
        print(f"  Error: {e}")

    # =====================================================
    # 2. Demand Only Data Split
    # =====================================================
    print("\n[2/6] Generating Demand Only data split figure...")

    try:
        plot_data_split(
            train_df_demand, val_df_demand, test_df_demand,
            'Demand Only',
            figures_dir / 'demand_only_data_split.png'
        )
    except Exception as e:
        print(f"  Error: {e}")

    # =====================================================
    # 3. Weather Full Model
    # =====================================================
    print("\n[3/6] Generating Weather Full model figure...")

    try:
        model_weather, scaler_weather, ckpt_weather = load_demand_model(
            'weather_full', PROJECT_ROOT / 'models' / 'production'
        )

        df_weather, _ = prepare_demand_data(include_weather=True)
        train_df_weather, val_df_weather, test_df_weather = split_data_by_time(df_weather)

        actuals_weather, preds_weather = generate_demand_predictions(
            model_weather, scaler_weather, test_df_weather, device, config
        )

        metrics_weather = training_report['results']['weather_full']['metrics']
        plot_demand_model(
            'weather_full', 'Weather Full',
            actuals_weather, preds_weather, metrics_weather,
            figures_dir / 'weather_full_performance.png'
        )
    except Exception as e:
        print(f"  Error: {e}")

    # =====================================================
    # 4. Weather Full Data Split
    # =====================================================
    print("\n[4/6] Generating Weather Full data split figure...")

    try:
        plot_data_split(
            train_df_weather, val_df_weather, test_df_weather,
            'Weather Full',
            figures_dir / 'weather_full_data_split.png'
        )
    except Exception as e:
        print(f"  Error: {e}")

    # =====================================================
    # 5. SMP Prediction Model
    # =====================================================
    print("\n[5/6] Generating SMP model figure...")

    try:
        smp_df = generate_smp_predictions()
        plot_smp_model(smp_df, smp_metrics, figures_dir / 'smp_v3_performance.png')
        plot_smp_data_split(smp_df, figures_dir / 'smp_v3_data_split.png')
    except Exception as e:
        print(f"  Error: {e}")

    # =====================================================
    # 6. Comparison Chart
    # =====================================================
    print("\n[6/6] Generating comparison chart...")

    try:
        create_comparison_chart(
            training_report['results']['demand_only']['metrics'],
            training_report['results']['weather_full']['metrics'],
            smp_metrics,
            figures_dir / 'model_comparison.png'
        )
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
