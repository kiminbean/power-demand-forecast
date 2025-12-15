"""
MODEL-003: LSTM 모델 학습 스크립트
==================================

LSTM 기반 전력 수요 예측 모델의 학습, 평가, 저장을 수행

사용법:
    python -m src.training.train_lstm --horizon 1
    python -m src.training.train_lstm --horizon 24 --bidirectional
    python -m src.training.train_lstm --multi_horizon

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import (
    prepare_data_pipeline,
    TimeSeriesScaler,
    get_device,
    create_dataloaders,
    create_multi_horizon_dataloaders
)
from models.lstm import (
    LSTMModel,
    MultiHorizonLSTM,
    create_model,
    model_summary
)
from training.trainer import (
    Trainer,
    TrainingHistory,
    EarlyStopping,
    create_scheduler,
    compute_metrics
)


# ============================================================
# 설정
# ============================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv',
    'target_col': 'power_demand',
    'train_end': '2022-12-31 23:00:00',
    'val_end': '2023-06-30 23:00:00',

    # Model
    'sequence_length': 48,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,
    'scheduler_type': 'plateau',

    # Multi-horizon
    'horizons': [1, 6, 12, 24],

    # Output
    'output_dir': PROJECT_ROOT / 'models',
    'log_dir': PROJECT_ROOT / 'logs',
}


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train LSTM model for power demand forecasting')

    # Model
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon (1, 6, 12, or 24)')
    parser.add_argument('--multi_horizon', action='store_true',
                       help='Train multi-horizon model')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional LSTM')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # Output
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for saving')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, 2)')

    return parser.parse_args()


def train_single_horizon(
    horizon: int,
    config: dict,
    args
) -> dict:
    """
    단일 horizon LSTM 모델 학습

    Args:
        horizon: 예측 horizon (1, 6, 12, 24)
        config: 설정
        args: 커맨드라인 인자

    Returns:
        dict: 학습 결과
    """
    print(f"\n{'='*60}")
    print(f"Training LSTM Model (Horizon: {horizon}h)")
    print(f"{'='*60}")

    # 데이터 준비
    print("\n[1] Loading and preparing data...")
    pipeline = prepare_data_pipeline(
        data_path=str(config['data_path']),
        target_col=config['target_col'],
        seq_length=config['sequence_length'],
        horizon=horizon,
        batch_size=args.batch_size,
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    train_loader = pipeline['train_loader']
    val_loader = pipeline['val_loader']
    test_loader = pipeline['test_loader']
    scaler = pipeline['scaler']
    n_features = pipeline['n_features']
    device = pipeline['device']

    # 모델 생성
    print("\n[2] Creating model...")
    model_type = 'bilstm' if args.bidirectional else 'lstm'
    model = create_model(
        model_type=model_type,
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=1
    )

    print(model_summary(model))
    print(f"Parameters: {model.get_num_parameters():,}")

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config['scheduler_type'],
        patience=5,
        factor=0.5
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=config['grad_clip'],
        checkpoint_dir=str(config['output_dir'])
    )

    # 학습
    print("\n[3] Training model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        verbose=args.verbose,
        log_interval=10
    )

    # 평가
    print("\n[4] Evaluating model...")
    result = trainer.evaluate(test_loader, return_predictions=True)

    # 평가 지표 계산
    metrics = compute_metrics(
        predictions=result['predictions'],
        actuals=result['actuals'],
        scaler=scaler,
        target_idx=pipeline['target_idx']
    )

    print(f"\nTest Results (Horizon: {horizon}h):")
    print(f"  RMSE: {metrics['RMSE']:.2f} MWh")
    print(f"  MAE:  {metrics['MAE']:.2f} MWh")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")

    # 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f"lstm_h{horizon}"
    model_path = config['output_dir'] / f"{experiment_name}_{timestamp}.pt"

    trainer.save_checkpoint(
        filepath=str(model_path),
        epoch=history.get_best()['epoch'],
        history=history,
        additional_info={
            'horizon': horizon,
            'metrics': metrics,
            'config': {
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'bidirectional': args.bidirectional,
                'sequence_length': config['sequence_length']
            }
        }
    )
    print(f"\n✓ Model saved: {model_path}")

    # 히스토리 저장
    history_path = config['log_dir'] / f"{experiment_name}_{timestamp}_history.json"
    history.save(str(history_path))
    print(f"✓ History saved: {history_path}")

    return {
        'horizon': horizon,
        'metrics': metrics,
        'history': history.history,
        'model_path': str(model_path)
    }


def train_multi_horizon(config: dict, args) -> dict:
    """
    다중 horizon LSTM 모델 학습

    Args:
        config: 설정
        args: 커맨드라인 인자

    Returns:
        dict: 학습 결과
    """
    print(f"\n{'='*60}")
    print(f"Training Multi-Horizon LSTM Model")
    print(f"Horizons: {config['horizons']}")
    print(f"{'='*60}")

    # 데이터 준비
    print("\n[1] Loading and preparing data...")

    # Multi-horizon용 데이터 로드 (max horizon 기준)
    from data.dataset import (
        split_data_by_time,
        prepare_features,
        TimeSeriesScaler
    )
    import pandas as pd

    df = pd.read_csv(config['data_path'], index_col=0, parse_dates=True)

    train_df, val_df, test_df = split_data_by_time(
        df,
        config['train_end'],
        config['val_end']
    )

    # 특성 준비
    train_data, feature_names, target_idx = prepare_features(
        train_df, config['target_col']
    )
    val_data, _, _ = prepare_features(val_df, config['target_col'])
    test_data, _, _ = prepare_features(test_df, config['target_col'])

    n_features = len(feature_names)
    print(f"  Features: {n_features}")
    print(f"  Target: {feature_names[target_idx]} (idx={target_idx})")

    # 스케일링
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_multi_horizon_dataloaders(
        train_scaled,
        val_scaled,
        test_scaled,
        target_idx=target_idx,
        seq_length=config['sequence_length'],
        horizons=config['horizons'],
        batch_size=args.batch_size
    )

    device = get_device()

    # 모델 생성
    print("\n[2] Creating multi-horizon model...")
    model = MultiHorizonLSTM(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizons=config['horizons'],
        bidirectional=args.bidirectional
    )

    print(model_summary(model))
    print(f"Parameters: {model.get_num_parameters():,}")

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config['scheduler_type'],
        patience=5,
        factor=0.5
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=config['grad_clip'],
        checkpoint_dir=str(config['output_dir'])
    )

    # 학습
    print("\n[3] Training model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        verbose=args.verbose,
        log_interval=10
    )

    # 평가
    print("\n[4] Evaluating model...")
    result = trainer.evaluate(test_loader, return_predictions=True)

    # 각 horizon별 평가
    predictions = result['predictions']  # (N, num_horizons)
    actuals = result['actuals']  # (N, num_horizons)

    print(f"\nTest Results (Multi-Horizon):")
    print("-" * 50)

    all_metrics = {}
    for i, h in enumerate(config['horizons']):
        pred_h = predictions[:, i] if len(predictions.shape) > 1 else predictions
        actual_h = actuals[:, i] if len(actuals.shape) > 1 else actuals

        metrics = compute_metrics(
            predictions=pred_h,
            actuals=actual_h,
            scaler=scaler,
            target_idx=target_idx
        )
        all_metrics[f'h{h}'] = metrics

        print(f"  Horizon {h:2d}h: RMSE={metrics['RMSE']:.2f}, "
              f"MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")

    print("-" * 50)

    # 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or "lstm_multi_horizon"
    model_path = config['output_dir'] / f"{experiment_name}_{timestamp}.pt"

    trainer.save_checkpoint(
        filepath=str(model_path),
        epoch=history.get_best()['epoch'],
        history=history,
        additional_info={
            'horizons': config['horizons'],
            'metrics': all_metrics,
            'config': {
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'bidirectional': args.bidirectional,
                'sequence_length': config['sequence_length']
            }
        }
    )
    print(f"\n✓ Model saved: {model_path}")

    # 히스토리 저장
    history_path = config['log_dir'] / f"{experiment_name}_{timestamp}_history.json"
    history.save(str(history_path))
    print(f"✓ History saved: {history_path}")

    return {
        'horizons': config['horizons'],
        'metrics': all_metrics,
        'history': history.history,
        'model_path': str(model_path)
    }


def main():
    """메인 함수"""
    args = parse_args()

    # 설정 준비
    config = DEFAULT_CONFIG.copy()
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['log_dir'].mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LSTM Power Demand Forecasting")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {get_device()}")

    if args.multi_horizon:
        # Multi-horizon 학습
        result = train_multi_horizon(config, args)
    else:
        # Single horizon 학습
        result = train_single_horizon(args.horizon, config, args)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
