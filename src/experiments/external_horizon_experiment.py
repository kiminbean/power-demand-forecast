"""
EVAL-006: External Features Horizon Experiment
==============================================

외부 데이터(인구, 전기차)의 Horizon별 효과 검증

비교 대상:
- baseline vs external (각 horizon별)

Horizons: h=1, h=24, h=168

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import (
    TimeSeriesScaler,
    TimeSeriesDataset,
    split_data_by_time,
    get_device
)
from models.lstm import create_model
from training.trainer import Trainer, create_scheduler
from evaluation.metrics import compute_all_metrics

from features import (
    add_time_features,
    add_weather_features,
    add_lag_features,
    add_external_features,
)

warnings.filterwarnings('ignore')


# Feature Groups
TIME_FEATURES = [
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_holiday'
]

LAG_FEATURES = [
    'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
    'demand_ma_6h', 'demand_ma_24h',
    'demand_std_24h',
    'demand_diff_1h', 'demand_diff_24h'
]

EXTERNAL_FEATURES = [
    'estimated_population',
    'ev_cumulative',
    'tourist_ratio',
    'ev_penetration',
    'ev_cumulative_log',
]


def load_data(
    data_path: str,
    include_external: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """데이터 로드"""
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    df = add_time_features(df, include_holiday=True)
    df = add_lag_features(df, demand_col='power_demand')

    if include_external:
        df = add_external_features(df)

    df = df.dropna()

    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_external:
        for col in EXTERNAL_FEATURES:
            if col in df.columns:
                features.append(col)

    available = [f for f in features if f in df.columns]
    return df[available].copy(), available


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_features: int,
    horizon: int,
    device: torch.device,
    epochs: int,
    seed: int
) -> Dict[str, float]:
    """학습 및 평가"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    seq_len = 168
    batch_size = 64

    train_ds = TimeSeriesDataset(train_scaled, seq_length=seq_len, horizon=horizon)
    val_ds = TimeSeriesDataset(val_scaled, seq_length=seq_len, horizon=horizon)
    test_ds = TimeSeriesDataset(test_scaled, seq_length=seq_len, horizon=horizon)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    test_loader = DataLoader(test_ds, batch_size)

    model = create_model(
        model_type='lstm',
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(optimizer, 'plateau')

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=10,
        verbose=0
    )

    # Evaluate
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

    return compute_all_metrics(actuals, preds)


def run_horizon_experiment(
    data_path: str,
    horizons: List[int] = [1, 24, 168],
    n_trials: int = 3,
    epochs: int = 50,
    quick: bool = False
) -> Dict:
    """Horizon별 외부 피처 실험"""
    print("=" * 70)
    print("EVAL-006: External Features Horizon Experiment")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Horizons: {horizons}, Trials: {n_trials}, Epochs: {epochs}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if quick:
        n_trials = 2
        epochs = 10
        print("Quick mode enabled")

    # Load data
    print("\nLoading data...")
    df_base, feat_base = load_data(data_path, include_external=False)
    df_ext, feat_ext = load_data(data_path, include_external=True)

    print(f"  baseline: {len(feat_base)} features")
    print(f"  external: {len(feat_ext)} features")

    # Split
    train_base, val_base, test_base = split_data_by_time(df_base)
    train_ext, val_ext, test_ext = split_data_by_time(df_ext)

    results = {h: {'baseline': [], 'external': []} for h in horizons}

    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"Horizon: h={horizon}")
        print('='*50)

        for trial in range(1, n_trials + 1):
            seed = 41 + trial
            print(f"\n  Trial {trial}/{n_trials} (seed={seed})")

            # Baseline
            print(f"    baseline...", end=" ", flush=True)
            metrics_base = train_and_evaluate(
                train_base, val_base, test_base,
                len(feat_base), horizon, device, epochs, seed
            )
            results[horizon]['baseline'].append(metrics_base)
            print(f"MAPE={metrics_base['MAPE']:.2f}%")

            # External
            print(f"    external...", end=" ", flush=True)
            metrics_ext = train_and_evaluate(
                train_ext, val_ext, test_ext,
                len(feat_ext), horizon, device, epochs, seed
            )
            results[horizon]['external'].append(metrics_ext)
            print(f"MAPE={metrics_ext['MAPE']:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    print(f"\n{'Horizon':<10} {'baseline MAPE':<18} {'external MAPE':<18} {'Effect':<12}")
    print("-" * 60)

    for h in horizons:
        base_mapes = [r['MAPE'] for r in results[h]['baseline']]
        ext_mapes = [r['MAPE'] for r in results[h]['external']]

        base_mean, base_std = np.mean(base_mapes), np.std(base_mapes)
        ext_mean, ext_std = np.mean(ext_mapes), np.std(ext_mapes)

        effect = (base_mean - ext_mean) / base_mean * 100

        summary[f'h={h}'] = {
            'baseline': {'mape_mean': base_mean, 'mape_std': base_std},
            'external': {'mape_mean': ext_mean, 'mape_std': ext_std},
            'effect': effect
        }

        effect_str = f"+{effect:.2f}%" if effect > 0 else f"{effect:.2f}%"
        print(f"h={h:<7} {base_mean:.2f}%±{base_std:.2f}       "
              f"{ext_mean:.2f}%±{ext_std:.2f}       {effect_str}")

    # Trend analysis
    print("\n[External Features Effect by Horizon]")
    for h in horizons:
        effect = summary[f'h={h}']['effect']
        if effect > 0:
            print(f"  h={h}: {effect:+.2f}% (IMPROVED)")
        else:
            print(f"  h={h}: {effect:+.2f}% (worse)")

    # Save
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'experiment': 'EVAL-006',
        'description': 'External Features Horizon Experiment',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'horizons': horizons,
            'n_trials': n_trials,
            'epochs': epochs,
            'external_features': EXTERNAL_FEATURES
        },
        'summary': summary,
        'detailed_results': {str(k): v for k, v in results.items()}
    }

    with open(output_dir / "external_horizon_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n✓ Report saved: {output_dir / 'external_horizon_report.json'}")

    return report


def main():
    parser = argparse.ArgumentParser(description='EVAL-006: External Horizon Experiment')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 24, 168])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'))

    args = parser.parse_args()

    run_horizon_experiment(
        data_path=args.data,
        horizons=args.horizons,
        n_trials=args.trials,
        epochs=args.epochs,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
