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

# Korean font setup for macOS
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
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


def plot_demand_model(model_name: str, display_name: str, actuals: np.ndarray,
                      preds: np.ndarray, metrics: dict, output_path: Path):
    """Create performance visualization for demand model"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{display_name} 모델 성능 분석', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted (last 7 days)
    ax1 = axes[0, 0]
    n_points = min(168, len(actuals))  # 7 days
    x = np.arange(n_points)
    ax1.plot(x, actuals[-n_points:], 'b-', label='실제값', alpha=0.8, linewidth=1.5)
    ax1.plot(x, preds[-n_points:], 'r--', label='예측값', alpha=0.8, linewidth=1.5)
    ax1.fill_between(x, actuals[-n_points:], preds[-n_points:], alpha=0.2, color='gray')
    ax1.set_xlabel('시간 (Hour)')
    ax1.set_ylabel('전력 수요 (MW)')
    ax1.set_title('실제값 vs 예측값 (최근 7일)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(actuals, preds, alpha=0.3, s=10, c='steelblue')
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax2.set_xlabel('실제값 (MW)')
    ax2.set_ylabel('예측값 (MW)')
    ax2.set_title(f'실제값 vs 예측값 산점도 (R² = {metrics["R2"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = axes[1, 0]
    errors = preds - actuals
    ax3.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2, label=f'평균: {errors.mean():.2f}')
    ax3.set_xlabel('예측 오차 (MW)')
    ax3.set_ylabel('빈도')
    ax3.set_title(f'예측 오차 분포 (MAE = {metrics["MAE"]:.2f} MW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
╔══════════════════════════════════════════╗
║        {display_name} 성능 지표         ║
╠══════════════════════════════════════════╣
║                                          ║
║   MAPE  :  {metrics['MAPE']:.2f}%                     ║
║   MAE   :  {metrics['MAE']:.2f} MW                   ║
║   RMSE  :  {metrics['RMSE']:.2f} MW                  ║
║   R²    :  {metrics['R2']:.4f}                      ║
║   sMAPE :  {metrics.get('sMAPE', 0):.2f}%                     ║
║   MBE   :  {metrics.get('MBE', 0):.2f} MW                    ║
║                                          ║
╚══════════════════════════════════════════╝
"""
    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


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


def plot_smp_model(df: pd.DataFrame, metrics: dict, output_path: Path):
    """Create SMP model performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SMP 예측 모델 (LSTM-Attention v3.1) 성능 분석', fontsize=16, fontweight='bold')

    # Recent SMP data (last 30 days)
    recent_df = df.tail(720)  # 30 days * 24 hours

    # 1. SMP Time series
    ax1 = axes[0, 0]
    ax1.plot(recent_df['datetime'], recent_df['smp_mainland'], 'b-', alpha=0.8, linewidth=1)
    ax1.axhline(y=recent_df['smp_mainland'].mean(), color='r', linestyle='--', label=f"평균: {recent_df['smp_mainland'].mean():.1f}원")
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('SMP (원/kWh)')
    ax1.set_title('육지 SMP 시계열 (최근 30일)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. SMP Distribution
    ax2 = axes[0, 1]
    ax2.hist(df['smp_mainland'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=df['smp_mainland'].mean(), color='r', linestyle='--', linewidth=2,
                label=f"평균: {df['smp_mainland'].mean():.1f}원")
    ax2.axvline(x=df['smp_mainland'].median(), color='orange', linestyle='-', linewidth=2,
                label=f"중앙값: {df['smp_mainland'].median():.1f}원")
    ax2.set_xlabel('SMP (원/kWh)')
    ax2.set_ylabel('빈도')
    ax2.set_title('SMP 분포 (2022-2024)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Hourly SMP pattern
    ax3 = axes[1, 0]
    hourly_avg = df.groupby(df['datetime'].dt.hour)['smp_mainland'].mean()
    ax3.bar(hourly_avg.index, hourly_avg.values, color='steelblue', edgecolor='white')
    ax3.set_xlabel('시간')
    ax3.set_ylabel('평균 SMP (원/kWh)')
    ax3.set_title('시간대별 평균 SMP')
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""
╔══════════════════════════════════════════╗
║     SMP 예측 모델 (LSTM-Attention v3.1)  ║
╠══════════════════════════════════════════╣
║                                          ║
║   MAPE      :  {metrics['mape']:.2f}%                   ║
║   MAE       :  {metrics['mae']:.2f} 원                 ║
║   RMSE      :  {metrics['rmse']:.2f} 원                ║
║   R²        :  {metrics['r2']:.4f}                     ║
║   Coverage  :  {metrics['coverage_80']:.1f}%                   ║
║                                          ║
║   Parameters:  {metrics['parameters']:,}개               ║
║   Features  :  {metrics['features']}개                     ║
║                                          ║
╚══════════════════════════════════════════╝
"""
    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_comparison_chart(demand_metrics: dict, weather_metrics: dict,
                           smp_metrics: dict, output_path: Path):
    """Create model comparison chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')

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
    bars2 = ax1_twin.bar(x + width/2, r2_values, width, label='R² (×100)', color='coral')
    ax1_twin.bar_label(bars2, fmt='%.1f', padding=3)

    ax1.set_ylabel('MAPE (%)')
    ax1_twin.set_ylabel('R² (×100)')
    ax1.set_xlabel('모델')
    ax1.set_title('전력 수요 예측 모델 비교')
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
    ax2.set_title('전체 모델 MAPE 비교')
    ax2.set_xlim(0, 12)
    ax2.grid(True, alpha=0.3, axis='x')

    # Best model indicator
    best_idx = np.argmin(all_mape)
    ax2.annotate('⭐ Best', xy=(all_mape[best_idx] + 0.5, best_idx),
                fontsize=12, color='gold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main function to generate all figures"""
    print("=" * 60)
    print("Model Performance Figure Generation")
    print("=" * 60)

    figures_dir = PROJECT_ROOT / 'figures'
    figures_dir.mkdir(exist_ok=True)

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

    # =====================================================
    # 1. Demand Only Model
    # =====================================================
    print("\n[1/4] Generating Demand Only model figure...")

    try:
        model_demand, scaler_demand, ckpt_demand = load_demand_model(
            'demand_only', PROJECT_ROOT / 'models' / 'production'
        )

        df_demand, _ = prepare_demand_data(include_weather=False)
        _, _, test_df_demand = split_data_by_time(df_demand)

        config = {'seq_length': 168, 'horizon': 1}
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
        print(f"  ⚠ Error: {e}")

    # =====================================================
    # 2. Weather Full Model
    # =====================================================
    print("\n[2/4] Generating Weather Full model figure...")

    try:
        model_weather, scaler_weather, ckpt_weather = load_demand_model(
            'weather_full', PROJECT_ROOT / 'models' / 'production'
        )

        df_weather, _ = prepare_demand_data(include_weather=True)
        _, _, test_df_weather = split_data_by_time(df_weather)

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
        print(f"  ⚠ Error: {e}")

    # =====================================================
    # 3. SMP Prediction Model
    # =====================================================
    print("\n[3/4] Generating SMP model figure...")

    try:
        smp_df = generate_smp_predictions()
        plot_smp_model(smp_df, smp_metrics, figures_dir / 'smp_v3_performance.png')
    except Exception as e:
        print(f"  ⚠ Error: {e}")

    # =====================================================
    # 4. Comparison Chart
    # =====================================================
    print("\n[4/4] Generating comparison chart...")

    try:
        create_comparison_chart(
            training_report['results']['demand_only']['metrics'],
            training_report['results']['weather_full']['metrics'],
            smp_metrics,
            figures_dir / 'model_comparison.png'
        )
    except Exception as e:
        print(f"  ⚠ Error: {e}")

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
