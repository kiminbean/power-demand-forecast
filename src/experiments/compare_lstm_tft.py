"""
MODEL-007: LSTM vs TFT 성능 비교 실험
=====================================

동일 데이터셋에서 LSTM과 TFT 모델의 성능을 비교합니다.

비교 항목:
1. 다중 시간대 예측 성능 (1h, 6h, 12h, 24h)
2. 평가 지표: RMSE, MAE, MAPE, R²
3. 학습 시간 및 추론 시간
4. 파라미터 수

사용법:
    python -m src.experiments.compare_lstm_tft
    python -m src.experiments.compare_lstm_tft --epochs 50 --quick

Author: Claude Code
Date: 2024-12
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import (
    TimeSeriesScaler,
    get_device,
    split_data_by_time,
    prepare_features,
    create_dataloaders
)
from models.lstm import LSTMModel
from models.transformer import TemporalFusionTransformer, QuantileLoss
from training.trainer import Trainer, create_scheduler, compute_metrics
from training.train_tft import (
    TFTFeatureConfig,
    TFTTrainer,
    create_tft_dataloaders,
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

    # LSTM 설정
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.2,

    # TFT 설정
    'tft_hidden_size': 64,
    'tft_lstm_layers': 2,
    'tft_num_heads': 4,
    'tft_dropout': 0.1,

    # 공통 설정
    'encoder_length': 48,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 15,

    # 출력
    'output_dir': PROJECT_ROOT / 'results' / 'experiments',
}


# ============================================================
# 데이터 준비
# ============================================================

def prepare_lstm_data(
    df: pd.DataFrame,
    target_col: str,
    train_end: str,
    val_end: str,
    seq_length: int,
    horizon: int,
    batch_size: int
) -> Dict:
    """LSTM용 데이터 준비"""
    # 데이터 분할
    train_df, val_df, test_df = split_data_by_time(df, train_end, val_end)

    # 피처 준비
    train_data, feature_names, target_idx = prepare_features(train_df, target_col)
    val_data, _, _ = prepare_features(val_df, target_col)
    test_data, _, _ = prepare_features(test_df, target_col)

    # 정규화
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        train_scaled, val_scaled, test_scaled,
        target_idx=target_idx,
        seq_length=seq_length,
        horizon=horizon,
        batch_size=batch_size
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'n_features': len(feature_names),
        'target_idx': target_idx
    }


def prepare_tft_data(
    df: pd.DataFrame,
    target_col: str,
    train_end: str,
    val_end: str,
    encoder_length: int,
    decoder_length: int,
    batch_size: int
) -> Dict:
    """TFT용 데이터 준비"""
    feature_config = TFTFeatureConfig(target_col=target_col)

    # 사용 가능한 피처만 필터링
    available = feature_config.get_available_features(df.columns.tolist())
    known_features = available['known']
    unknown_features = available['unknown']

    if not known_features or not unknown_features:
        # 피처가 없으면 기본 피처 사용
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 시간 관련 피처 추정
        time_features = [c for c in numeric_cols if any(x in c.lower() for x in ['hour', 'day', 'month', 'sin', 'cos', 'weekend', 'holiday'])]
        other_features = [c for c in numeric_cols if c not in time_features]

        if target_col in other_features:
            other_features.remove(target_col)
        unknown_features = [target_col] + other_features[:10]  # 최대 10개
        known_features = time_features[:8] if time_features else numeric_cols[:5]  # 최대 8개

    all_features = unknown_features + known_features

    # 인덱스 계산
    unknown_indices = list(range(len(unknown_features)))
    known_indices = list(range(len(unknown_features), len(all_features)))
    target_idx = 0

    # 데이터 분할
    train_df, val_df, test_df = split_data_by_time(df, train_end, val_end)

    # 피처 선택 및 결측치 처리
    train_data = train_df[all_features].ffill().bfill().fillna(0).values.astype(np.float32)
    val_data = val_df[all_features].ffill().bfill().fillna(0).values.astype(np.float32)
    test_data = test_df[all_features].ffill().bfill().fillna(0).values.astype(np.float32)

    # 정규화
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_tft_dataloaders(
        train_scaled, val_scaled, test_scaled,
        known_indices=known_indices,
        unknown_indices=unknown_indices,
        target_idx=target_idx,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        batch_size=batch_size
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'num_known': len(known_features),
        'num_unknown': len(unknown_features),
        'target_idx': target_idx,
        'known_features': known_features,
        'unknown_features': unknown_features
    }


# ============================================================
# 모델 학습 및 평가
# ============================================================

def train_lstm_model(
    data: Dict,
    config: Dict,
    device: torch.device,
    horizon: int,
    verbose: int = 1
) -> Tuple[nn.Module, Dict]:
    """LSTM 모델 학습"""
    if verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Training LSTM (horizon={horizon}h)")
        print(f"{'='*50}")

    # 모델 생성
    model = LSTMModel(
        input_size=data['n_features'],
        hidden_size=config['lstm_hidden_size'],
        num_layers=config['lstm_num_layers'],
        dropout=config['lstm_dropout'],
        output_size=1
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = create_scheduler(optimizer, scheduler_type='plateau', patience=5)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=1.0
    )

    # 학습 시간 측정
    start_time = time.time()

    history = trainer.fit(
        data['train_loader'],
        data['val_loader'],
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=verbose,
        log_interval=10
    )

    train_time = time.time() - start_time

    # 평가
    result = trainer.evaluate(data['test_loader'], return_predictions=True)

    # 추론 시간 측정
    start_time = time.time()
    with torch.no_grad():
        for batch_X, _ in data['test_loader']:
            batch_X = batch_X.to(device)
            _ = model(batch_X)
    inference_time = time.time() - start_time

    # 메트릭 계산
    metrics = compute_metrics(
        result['predictions'],
        result['actuals'],
        scaler=data['scaler'],
        target_idx=data['target_idx']
    )

    return model, {
        'metrics': metrics,
        'train_time': train_time,
        'inference_time': inference_time,
        'n_parameters': model.get_num_parameters(),
        'best_epoch': history.get_best()['epoch'],
        'best_val_loss': history.get_best()['val_loss']
    }


def train_tft_model(
    data: Dict,
    config: Dict,
    device: torch.device,
    decoder_length: int,
    verbose: int = 1
) -> Tuple[nn.Module, Dict]:
    """TFT 모델 학습"""
    if verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Training TFT (decoder_length={decoder_length}h)")
        print(f"{'='*50}")

    # 모델 생성
    model = TemporalFusionTransformer(
        num_static_vars=0,
        num_known_vars=data['num_known'],
        num_unknown_vars=data['num_unknown'],
        hidden_size=config['tft_hidden_size'],
        lstm_layers=config['tft_lstm_layers'],
        num_attention_heads=config['tft_num_heads'],
        dropout=config['tft_dropout'],
        encoder_length=config['encoder_length'],
        decoder_length=decoder_length
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = create_scheduler(optimizer, scheduler_type='plateau', patience=5)

    trainer = TFTTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=1.0
    )

    # 학습 시간 측정
    start_time = time.time()

    history = trainer.fit(
        data['train_loader'],
        data['val_loader'],
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=verbose,
        log_interval=10
    )

    train_time = time.time() - start_time

    # 평가
    result = trainer.evaluate(data['test_loader'], return_predictions=True)

    # 추론 시간 측정
    start_time = time.time()
    _ = trainer.predict(data['test_loader'])
    inference_time = time.time() - start_time

    # 메트릭 계산 (median 예측 사용)
    predictions = result['predictions'][:, :, 1]  # median (q=0.5)
    targets = result['targets']

    # 평균화 (multi-horizon -> single value per sample)
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # 역스케일링
    pred_original = data['scaler'].inverse_transform_target(pred_flat, data['target_idx'])
    target_original = data['scaler'].inverse_transform_target(target_flat, data['target_idx'])

    # 메트릭 계산
    mse = np.mean((pred_original - target_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_original - target_original))

    non_zero = target_original != 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((target_original[non_zero] - pred_original[non_zero]) / target_original[non_zero])) * 100
    else:
        mape = float('inf')

    ss_res = np.sum((target_original - pred_original) ** 2)
    ss_tot = np.sum((target_original - np.mean(target_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

    # 파라미터 수 계산
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, {
        'metrics': metrics,
        'train_time': train_time,
        'inference_time': inference_time,
        'n_parameters': n_params,
        'best_epoch': history.get_best()['epoch'],
        'best_val_loss': history.get_best()['val_loss'],
        'quantile_loss': result['test_quantile_loss']
    }


# ============================================================
# 메인 실험 함수
# ============================================================

def run_comparison_experiment(
    config: Dict,
    horizons: List[int] = None,
    verbose: int = 1
) -> Dict:
    """LSTM vs TFT 비교 실험 실행"""
    if horizons is None:
        horizons = [1, 6, 12, 24]

    device = get_device()
    results = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'horizons': horizons,
        'lstm': {},
        'tft': {}
    }

    print("=" * 60)
    print("MODEL-007: LSTM vs TFT Comparison Experiment")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Horizons: {horizons}")
    print(f"Epochs: {config['epochs']}")
    print(f"Patience: {config['patience']}")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] Loading data...")
    df = pd.read_csv(config['data_path'], index_col=0, parse_dates=True)
    print(f"    Shape: {df.shape}")
    print(f"    Date range: {df.index.min()} ~ {df.index.max()}")

    # 각 horizon에 대해 실험
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon}h")
        print(f"{'='*60}")

        # LSTM 데이터 준비 및 학습
        print(f"\n[LSTM] Preparing data for horizon={horizon}h...")
        lstm_data = prepare_lstm_data(
            df=df,
            target_col=config['target_col'],
            train_end=config['train_end'],
            val_end=config['val_end'],
            seq_length=config['encoder_length'],
            horizon=horizon,
            batch_size=config['batch_size']
        )

        lstm_model, lstm_result = train_lstm_model(
            data=lstm_data,
            config=config,
            device=device,
            horizon=horizon,
            verbose=verbose
        )

        results['lstm'][str(horizon)] = lstm_result

        # TFT 데이터 준비 및 학습
        print(f"\n[TFT] Preparing data for decoder_length={horizon}h...")
        tft_data = prepare_tft_data(
            df=df,
            target_col=config['target_col'],
            train_end=config['train_end'],
            val_end=config['val_end'],
            encoder_length=config['encoder_length'],
            decoder_length=horizon,
            batch_size=config['batch_size']
        )

        tft_model, tft_result = train_tft_model(
            data=tft_data,
            config=config,
            device=device,
            decoder_length=horizon,
            verbose=verbose
        )

        results['tft'][str(horizon)] = tft_result

        # 결과 요약 출력
        print(f"\n[Summary] Horizon={horizon}h")
        print(f"  {'Metric':<12} {'LSTM':>12} {'TFT':>12} {'Diff':>12}")
        print(f"  {'-'*48}")

        for metric in ['RMSE', 'MAE', 'MAPE', 'R2']:
            lstm_val = lstm_result['metrics'][metric]
            tft_val = tft_result['metrics'][metric]
            diff = tft_val - lstm_val
            diff_pct = (diff / lstm_val * 100) if lstm_val != 0 else 0

            if metric == 'R2':
                better = "TFT" if diff > 0 else "LSTM"
            else:
                better = "TFT" if diff < 0 else "LSTM"

            print(f"  {metric:<12} {lstm_val:>12.4f} {tft_val:>12.4f} {diff:>+12.4f} ({better})")

        print(f"\n  {'Time':<12} {'LSTM':>12} {'TFT':>12}")
        print(f"  {'-'*36}")
        print(f"  {'Train (s)':<12} {lstm_result['train_time']:>12.1f} {tft_result['train_time']:>12.1f}")
        print(f"  {'Infer (s)':<12} {lstm_result['inference_time']:>12.3f} {tft_result['inference_time']:>12.3f}")
        print(f"  {'Params':<12} {lstm_result['n_parameters']:>12,} {tft_result['n_parameters']:>12,}")

    return results


def save_results(results: Dict, output_dir: Path) -> str:
    """실험 결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"lstm_vs_tft_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def print_final_summary(results: Dict):
    """최종 요약 출력"""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: LSTM vs TFT")
    print("=" * 60)

    horizons = results['horizons']

    # 표 형식 출력
    print(f"\n{'RMSE Comparison':^60}")
    print(f"  {'Horizon':<10} {'LSTM':>12} {'TFT':>12} {'Improvement':>15}")
    print(f"  {'-'*49}")

    total_lstm_rmse = 0
    total_tft_rmse = 0

    for h in horizons:
        h_str = str(h)
        lstm_rmse = results['lstm'][h_str]['metrics']['RMSE']
        tft_rmse = results['tft'][h_str]['metrics']['RMSE']
        improvement = (lstm_rmse - tft_rmse) / lstm_rmse * 100

        total_lstm_rmse += lstm_rmse
        total_tft_rmse += tft_rmse

        print(f"  {h_str + 'h':<10} {lstm_rmse:>12.2f} {tft_rmse:>12.2f} {improvement:>+14.1f}%")

    avg_improvement = (total_lstm_rmse - total_tft_rmse) / total_lstm_rmse * 100
    print(f"  {'-'*49}")
    print(f"  {'Average':<10} {total_lstm_rmse/len(horizons):>12.2f} {total_tft_rmse/len(horizons):>12.2f} {avg_improvement:>+14.1f}%")

    # 시간 비교
    print(f"\n{'Training Time (seconds)':^60}")
    print(f"  {'Horizon':<10} {'LSTM':>12} {'TFT':>12} {'Ratio':>15}")
    print(f"  {'-'*49}")

    for h in horizons:
        h_str = str(h)
        lstm_time = results['lstm'][h_str]['train_time']
        tft_time = results['tft'][h_str]['train_time']
        ratio = tft_time / lstm_time if lstm_time > 0 else 0

        print(f"  {h_str + 'h':<10} {lstm_time:>12.1f} {tft_time:>12.1f} {ratio:>14.1f}x")

    # 파라미터 수
    print(f"\n{'Model Parameters':^60}")
    lstm_params = results['lstm'][str(horizons[0])]['n_parameters']
    tft_params = results['tft'][str(horizons[0])]['n_parameters']
    print(f"  LSTM: {lstm_params:,}")
    print(f"  TFT:  {tft_params:,}")
    print(f"  Ratio: {tft_params/lstm_params:.1f}x")

    print("\n" + "=" * 60)


# ============================================================
# 메인
# ============================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='LSTM vs TFT Comparison Experiment'
    )
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs, single horizon)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1)')
    parser.add_argument('--horizons', type=str, default='1,6,12,24',
                       help='Comma-separated horizons')

    return parser.parse_args()


def test_with_synthetic_data():
    """합성 데이터로 빠른 테스트"""
    print("=" * 60)
    print("Quick Test with Synthetic Data")
    print("=" * 60)

    device = get_device()
    np.random.seed(42)

    # 합성 데이터 생성
    n_samples = 500
    n_features = 15
    n_known = 5
    n_unknown = 10

    # 시뮬레이션된 시계열 데이터
    t = np.arange(n_samples) / 24  # hours to days
    target = 100 + 20 * np.sin(2 * np.pi * t / 7) + np.random.randn(n_samples) * 5
    features = np.column_stack([target] + [np.random.randn(n_samples) for _ in range(n_features - 1)])
    data = features.astype(np.float32)

    # 분할
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 정규화
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # LSTM 테스트
    print("\n[LSTM Test]")
    from data.dataset import create_dataloaders

    lstm_train, lstm_val, lstm_test = create_dataloaders(
        train_scaled, val_scaled, test_scaled,
        target_idx=0, seq_length=24, horizon=1, batch_size=16
    )

    lstm_model = LSTMModel(input_size=n_features, hidden_size=32, num_layers=1, dropout=0.1)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_trainer = Trainer(lstm_model, nn.MSELoss(), lstm_optimizer, device)

    lstm_history = lstm_trainer.fit(lstm_train, lstm_val, epochs=3, patience=5, verbose=0)
    lstm_result = lstm_trainer.evaluate(lstm_test, return_predictions=True)

    print(f"  LSTM Test Loss: {lstm_result['test_loss']:.6f}")

    # TFT 테스트
    print("\n[TFT Test]")
    known_indices = list(range(n_unknown, n_features))
    unknown_indices = list(range(n_unknown))

    tft_train, tft_val, tft_test = create_tft_dataloaders(
        train_scaled, val_scaled, test_scaled,
        known_indices=known_indices,
        unknown_indices=unknown_indices,
        target_idx=0,
        encoder_length=24,
        decoder_length=6,
        batch_size=16
    )

    tft_model = TemporalFusionTransformer(
        num_static_vars=0,
        num_known_vars=n_known,
        num_unknown_vars=n_unknown,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=2,
        encoder_length=24,
        decoder_length=6
    )

    tft_optimizer = torch.optim.Adam(tft_model.parameters(), lr=0.001)
    tft_trainer = TFTTrainer(tft_model, tft_optimizer, device)

    tft_history = tft_trainer.fit(tft_train, tft_val, epochs=3, patience=5, verbose=0)
    tft_result = tft_trainer.evaluate(tft_test, return_predictions=True)

    print(f"  TFT Test Quantile Loss: {tft_result['test_quantile_loss']:.6f}")
    print(f"  TFT Test MSE: {tft_result['test_mse_loss']:.6f}")

    print("\n" + "=" * 60)
    print("Quick Test PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    args = parse_args()

    # 테스트 모드
    if args.quick and not DEFAULT_CONFIG['data_path'].exists():
        test_with_synthetic_data()
        sys.exit(0)

    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['patience'] = args.patience
    config['batch_size'] = args.batch_size

    if args.quick:
        config['epochs'] = 10
        config['patience'] = 5
        horizons = [1, 24]
    else:
        horizons = [int(h) for h in args.horizons.split(',')]

    # 데이터 파일 확인
    if not config['data_path'].exists():
        print(f"Error: Data file not found: {config['data_path']}")
        print("Running quick test with synthetic data instead...")
        test_with_synthetic_data()
        sys.exit(0)

    # 실험 실행
    results = run_comparison_experiment(
        config=config,
        horizons=horizons,
        verbose=args.verbose
    )

    # 결과 저장
    filepath = save_results(results, config['output_dir'])
    print(f"\nResults saved to: {filepath}")

    # 최종 요약
    print_final_summary(results)
