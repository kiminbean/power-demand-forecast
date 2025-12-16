"""
MODEL-008: TFT 하이퍼파라미터 튜닝
=================================

Optuna를 활용한 TFT 모델 하이퍼파라미터 최적화

탐색 공간:
- hidden_size: [32, 64, 128, 256]
- lstm_layers: [1, 2, 3]
- num_attention_heads: [2, 4, 8]
- dropout: [0.05, 0.3]
- learning_rate: [1e-4, 1e-2]
- batch_size: [32, 64, 128]

사용법:
    python -m src.experiments.tune_tft --n_trials 50
    python -m src.experiments.tune_tft --n_trials 10 --quick

Author: Claude Code
Date: 2024-12
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import TimeSeriesScaler, get_device, split_data_by_time
from models.transformer import TemporalFusionTransformer, QuantileLoss
from training.train_tft import (
    TFTFeatureConfig,
    TFTTrainer,
    create_tft_dataloaders,
)
from training.trainer import create_scheduler


# ============================================================
# 설정
# ============================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv',
    'target_col': 'power_demand',
    'train_end': '2022-12-31 23:00:00',
    'val_end': '2023-06-30 23:00:00',

    # 고정 하이퍼파라미터
    'encoder_length': 48,
    'decoder_length': 24,

    # 학습 설정
    'epochs': 50,
    'patience': 10,

    # 출력
    'output_dir': PROJECT_ROOT / 'results' / 'tuning',
}


# ============================================================
# 탐색 공간 정의
# ============================================================

SEARCH_SPACE = {
    'hidden_size': {
        'type': 'categorical',
        'choices': [32, 64, 128, 256]
    },
    'lstm_layers': {
        'type': 'int',
        'low': 1,
        'high': 3
    },
    'num_attention_heads': {
        'type': 'categorical',
        'choices': [2, 4, 8]
    },
    'dropout': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3
    },
    'learning_rate': {
        'type': 'loguniform',
        'low': 1e-4,
        'high': 1e-2
    },
    'batch_size': {
        'type': 'categorical',
        'choices': [32, 64, 128]
    }
}


def suggest_params(trial: Trial, search_space: Dict) -> Dict:
    """Trial에서 하이퍼파라미터 제안"""
    params = {}

    for name, config in search_space.items():
        if config['type'] == 'categorical':
            params[name] = trial.suggest_categorical(name, config['choices'])
        elif config['type'] == 'int':
            params[name] = trial.suggest_int(name, config['low'], config['high'])
        elif config['type'] == 'float':
            params[name] = trial.suggest_float(name, config['low'], config['high'])
        elif config['type'] == 'loguniform':
            params[name] = trial.suggest_float(name, config['low'], config['high'], log=True)

    return params


# ============================================================
# 데이터 준비
# ============================================================

def prepare_tuning_data(
    df: pd.DataFrame,
    config: Dict,
    batch_size: int = 64
) -> Dict:
    """튜닝용 데이터 준비"""
    feature_config = TFTFeatureConfig(target_col=config['target_col'])

    # 사용 가능한 피처만 필터링
    available = feature_config.get_available_features(df.columns.tolist())
    known_features = available['known']
    unknown_features = available['unknown']

    if not known_features or not unknown_features:
        # 피처가 없으면 기본 피처 사용
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        time_features = [c for c in numeric_cols if any(x in c.lower() for x in ['hour', 'day', 'month', 'sin', 'cos', 'weekend', 'holiday'])]
        other_features = [c for c in numeric_cols if c not in time_features]

        if config['target_col'] in other_features:
            other_features.remove(config['target_col'])
        unknown_features = [config['target_col']] + other_features[:10]
        known_features = time_features[:8] if time_features else numeric_cols[:5]

    all_features = unknown_features + known_features

    # 인덱스 계산
    unknown_indices = list(range(len(unknown_features)))
    known_indices = list(range(len(unknown_features), len(all_features)))
    target_idx = 0

    # 데이터 분할
    train_df, val_df, test_df = split_data_by_time(
        df, config['train_end'], config['val_end']
    )

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
        encoder_length=config['encoder_length'],
        decoder_length=config['decoder_length'],
        batch_size=batch_size
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'num_known': len(known_features),
        'num_unknown': len(unknown_features),
        'target_idx': target_idx
    }


# ============================================================
# Objective 함수
# ============================================================

class TFTObjective:
    """
    Optuna Objective 클래스

    모델 학습 및 검증 손실 반환
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict,
        search_space: Dict = None,
        device: torch.device = None,
        verbose: int = 0
    ):
        self.df = df
        self.config = config
        self.search_space = search_space or SEARCH_SPACE
        self.device = device or get_device()
        self.verbose = verbose

        # 데이터 캐시
        self._data_cache = {}

    def __call__(self, trial: Trial) -> float:
        """Objective 함수 실행"""
        # 하이퍼파라미터 제안
        params = suggest_params(trial, self.search_space)

        if self.verbose >= 1:
            print(f"\nTrial {trial.number}: {params}")

        try:
            # 데이터 준비 (배치 크기별 캐시)
            batch_size = params['batch_size']
            if batch_size not in self._data_cache:
                self._data_cache[batch_size] = prepare_tuning_data(
                    self.df, self.config, batch_size
                )

            data = self._data_cache[batch_size]

            # 모델 생성
            model = TemporalFusionTransformer(
                num_static_vars=0,
                num_known_vars=data['num_known'],
                num_unknown_vars=data['num_unknown'],
                hidden_size=params['hidden_size'],
                lstm_layers=params['lstm_layers'],
                num_attention_heads=params['num_attention_heads'],
                dropout=params['dropout'],
                encoder_length=self.config['encoder_length'],
                decoder_length=self.config['decoder_length']
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate']
            )

            scheduler = create_scheduler(optimizer, scheduler_type='plateau', patience=5)

            trainer = TFTTrainer(
                model=model,
                optimizer=optimizer,
                device=self.device,
                scheduler=scheduler,
                grad_clip=1.0
            )

            # 학습
            history = trainer.fit(
                data['train_loader'],
                data['val_loader'],
                epochs=self.config['epochs'],
                patience=self.config['patience'],
                verbose=0
            )

            # 최적 검증 손실 반환
            best_val_loss = history.get_best()['val_loss']

            if self.verbose >= 1:
                print(f"  Best val loss: {best_val_loss:.6f}")

            return best_val_loss

        except Exception as e:
            print(f"  Error: {e}")
            return float('inf')


# ============================================================
# Study 실행
# ============================================================

def run_tuning_study(
    config: Dict,
    n_trials: int = 50,
    study_name: str = None,
    verbose: int = 1
) -> Dict:
    """
    하이퍼파라미터 튜닝 Study 실행

    Args:
        config: 설정
        n_trials: Trial 수
        study_name: Study 이름
        verbose: 출력 레벨

    Returns:
        Dict: 튜닝 결과
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    device = get_device()

    print("=" * 60)
    print("MODEL-008: TFT Hyperparameter Tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {config['epochs']}")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] Loading data...")
    df = pd.read_csv(config['data_path'], index_col=0, parse_dates=True)
    print(f"    Shape: {df.shape}")

    # Objective 생성
    objective = TFTObjective(
        df=df,
        config=config,
        device=device,
        verbose=verbose
    )

    # Study 생성
    if study_name is None:
        study_name = f"tft_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    # 최적화 실행
    print("\n[2] Running optimization...")
    start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    tuning_time = time.time() - start_time

    # 결과 정리
    best_trial = study.best_trial
    best_params = best_trial.params

    results = {
        'study_name': study_name,
        'n_trials': n_trials,
        'best_trial_number': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_params,
        'tuning_time_seconds': tuning_time,
        'timestamp': datetime.now().isoformat(),
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }

    # 결과 출력
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.6f}")
    print(f"Tuning time: {tuning_time:.1f}s")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("=" * 60)

    return results


def save_tuning_results(results: Dict, output_dir: Path) -> str:
    """튜닝 결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{results['study_name']}.json"

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def create_best_model(
    best_params: Dict,
    data: Dict,
    config: Dict,
    device: torch.device
) -> TemporalFusionTransformer:
    """최적 파라미터로 모델 생성"""
    model = TemporalFusionTransformer(
        num_static_vars=0,
        num_known_vars=data['num_known'],
        num_unknown_vars=data['num_unknown'],
        hidden_size=best_params['hidden_size'],
        lstm_layers=best_params['lstm_layers'],
        num_attention_heads=best_params['num_attention_heads'],
        dropout=best_params['dropout'],
        encoder_length=config['encoder_length'],
        decoder_length=config['decoder_length']
    )

    return model.to(device)


# ============================================================
# 테스트 함수
# ============================================================

def test_tuning_with_synthetic_data():
    """합성 데이터로 튜닝 테스트"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Skipping test.")
        return False

    print("=" * 60)
    print("Quick Tuning Test with Synthetic Data")
    print("=" * 60)

    device = get_device()
    np.random.seed(42)

    # 합성 데이터 생성
    n_samples = 300
    n_features = 13
    n_known = 5
    n_unknown = 8

    data = np.random.randn(n_samples, n_features).astype(np.float32)

    # 분할 및 정규화
    train_data = data[:200]
    val_data = data[200:250]
    test_data = data[250:]

    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    known_indices = list(range(n_unknown, n_features))
    unknown_indices = list(range(n_unknown))

    train_loader, val_loader, test_loader = create_tft_dataloaders(
        train_scaled, val_scaled, test_scaled,
        known_indices=known_indices,
        unknown_indices=unknown_indices,
        target_idx=0,
        encoder_length=24,
        decoder_length=6,
        batch_size=16
    )

    # 간단한 Study
    def simple_objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64])
        dropout = trial.suggest_float('dropout', 0.1, 0.2)

        model = TemporalFusionTransformer(
            num_static_vars=0,
            num_known_vars=n_known,
            num_unknown_vars=n_unknown,
            hidden_size=hidden_size,
            lstm_layers=1,
            num_attention_heads=2,
            dropout=dropout,
            encoder_length=24,
            decoder_length=6
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = TFTTrainer(model, optimizer, device)

        history = trainer.fit(train_loader, val_loader, epochs=2, patience=3, verbose=0)

        return history.get_best()['val_loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(simple_objective, n_trials=3, show_progress_bar=False)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    print("\n" + "=" * 60)
    print("Quick Tuning Test PASSED!")
    print("=" * 60)

    return True


# ============================================================
# 메인
# ============================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='TFT Hyperparameter Tuning'
    )
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per trial')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not OPTUNA_AVAILABLE:
        print("Error: Optuna not installed.")
        print("Install with: pip install optuna")
        sys.exit(1)

    if args.quick:
        test_tuning_with_synthetic_data()
        sys.exit(0)

    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['patience'] = args.patience

    # 데이터 파일 확인
    if not config['data_path'].exists():
        print(f"Data file not found: {config['data_path']}")
        print("Running quick test instead...")
        test_tuning_with_synthetic_data()
        sys.exit(0)

    # 튜닝 실행
    results = run_tuning_study(
        config=config,
        n_trials=args.n_trials,
        verbose=args.verbose
    )

    # 결과 저장
    filepath = save_tuning_results(results, config['output_dir'])
    print(f"\nResults saved to: {filepath}")
