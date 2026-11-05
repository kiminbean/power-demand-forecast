"""
제주 전력수요 LSTM 재학습 + 시간외(out-of-time) 검증
=====================================================

2025-12 production 체크포인트는 training_report.json 수치(MAPE 6.06%)를
재현하지 못했다(재현 10.05%). 이 스크립트는 같은 구조·하이퍼파라미터로
다시 학습하고, 학습에 쓰지 않은 미래 구간에서만 평가한다.

평가 구성:
- fixed: Train ≤2022-12, Val 2023-01~06, Test 2023-07~2024-12
- walk-forward: 테스트 연도 Y ∈ {2021, 2022, 2023, 2024}
  Train ≤(Y-1)-06-30, Val (Y-1)-07~12, Test Y (확장 창)
- 기준선: persistence(t-1), seasonal naive(t-24, t-168)를 같은 타깃 시각에서 평가

합격 기준: fixed 테스트와 walk-forward 평균 모두에서
LSTM의 MAPE·RMSE가 persistence보다 낮아야 한다.

Usage:
    python src/training/retrain_demand_oot.py --output models/demand_retrain_2026-11-05
"""

import sys
import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import TimeSeriesDataset, get_device
from evaluation.metrics import compute_all_metrics
from models.lstm import LastValueResidual
from training.train_production import (
    load_data, train_model, save_model, TRAINING_CONFIG, MODEL_CONFIG,
    TIME_FEATURES, LAG_FEATURES,
)

TARGET = 'power_demand'
# 기존 체크포인트·API 추론과 같은 피처 구성 (weather_full = 기온 하나)
FEATURE_SETS = {
    'demand_only': [TARGET] + TIME_FEATURES + LAG_FEATURES,
    'weather_full': [TARGET] + TIME_FEATURES + LAG_FEATURES + ['기온'],
}
# 원본 CSV의 누락 시각은 2015년의 1시간짜리 2곳뿐이다. 그보다 긴 공백은 보간하지 않고 실패시킨다.
MAX_FILL_HOURS = 2
FIXED_SPLIT = ('2022-12-31 23:00', '2023-06-30 23:00', '2024-12-31 23:00')
WF_TEST_YEARS = [2021, 2022, 2023, 2024]
BASELINE_LAGS = {'persistence_t-1': 1, 'seasonal_t-24': 24, 'seasonal_t-168': 168}



def fill_short_gaps(data_path: Path, out_path: Path) -> List[str]:
    """누락된 시각을 시간 보간으로 채운 CSV를 쓴다. 채운 시각 목록을 반환한다."""
    df = pd.read_csv(data_path)
    ts = pd.to_datetime(df['datetime'], errors='raise')
    if ts.duplicated().any():
        raise ValueError("Duplicate datetime values in source data")
    df = df.assign(datetime=ts).set_index('datetime').sort_index()
    full = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='h'))
    missing = full.index[~full.index.isin(df.index)]
    # 연속 결측 길이를 확인해 긴 공백을 몰래 메우지 않는다
    if len(missing):
        gap_ids = (missing.to_series().diff() != pd.Timedelta(hours=1)).cumsum()
        if gap_ids.value_counts().max() > MAX_FILL_HOURS:
            raise ValueError(f"Gap longer than {MAX_FILL_HOURS}h in source data")
    num = full.select_dtypes('number').columns
    full[num] = full[num].interpolate(method='time', limit=MAX_FILL_HOURS)
    full = full.ffill(limit=MAX_FILL_HOURS)
    full.index.name = 'datetime'
    full.reset_index().to_csv(out_path, index=False)
    return [str(t) for t in missing]


def split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    """경계 시각 기준 시간 분할 (train ≤ train_end < val ≤ val_end < test ≤ test_end)."""
    tr = df.loc[df.index <= pd.Timestamp(train_end)]
    va = df.loc[(df.index > pd.Timestamp(train_end)) & (df.index <= pd.Timestamp(val_end))]
    te = df.loc[(df.index > pd.Timestamp(val_end)) & (df.index <= pd.Timestamp(test_end))]
    if min(len(tr), len(va), len(te)) <= TRAINING_CONFIG['seq_length']:
        raise ValueError(f"Split too short: {len(tr)}/{len(va)}/{len(te)}")
    return tr, va, te


def predict_period(model, scaler, df: pd.DataFrame, test_start: pd.Timestamp,
                   device, test_end: pd.Timestamp) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """[test_start, test_end] 구간 모든 시각을 예측한다. 입력 창은 직전 168시간(과거 관측)만 쓴다."""
    seq = TRAINING_CONFIG['seq_length']
    pos = df.index.get_indexer([test_start])[0]
    end_pos = df.index.get_indexer([test_end])[0]
    if pos < seq:
        raise ValueError("Not enough history before test start")
    if end_pos < pos:
        raise ValueError("test_end must be in the index and not before test_start")
    block = df.iloc[pos - seq:end_pos + 1]
    ds = TimeSeriesDataset(scaler.transform(block.values), seq_length=seq,
                           horizon=TRAINING_CONFIG['horizon'])
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for X, y in DataLoader(ds, 512):
            preds.append(model(X.to(device)).cpu().numpy())
            actuals.append(y.numpy())
    p = scaler.inverse_transform_target(np.concatenate(preds).flatten())
    a = scaler.inverse_transform_target(np.concatenate(actuals).flatten())
    idx = block.index[seq:seq + len(p)]
    if not np.allclose(a, block[TARGET].values[seq:seq + len(p)], atol=1e-2):
        raise AssertionError("Target alignment mismatch")
    return idx, p, a


def baselines(demand: pd.Series, idx: pd.DatetimeIndex, actual: np.ndarray) -> Dict:
    out = {}
    for name, lag in BASELINE_LAGS.items():
        pred = demand.shift(lag).reindex(idx).values
        out[name] = {k: float(v) for k, v in compute_all_metrics(actual, pred).items()}
    return out


def run_fold(name, df, bounds, device, seed, verbose, residual=False):
    tr, va, te = split(df, *bounds)
    model, scaler, hist = train_model(tr, va, df.shape[1], device, TRAINING_CONFIG,
                                      seed=seed, verbose=verbose,
                                      model_wrapper=LastValueResidual if residual else None)
    idx, p, a = predict_period(model, scaler, df, te.index[0], device, te.index[-1])
    metrics = {k: float(v) for k, v in compute_all_metrics(a, p).items()}
    h = hist.history
    result = {
        'bounds': {'train': [str(tr.index[0]), str(tr.index[-1]), len(tr)],
                   'val': [str(va.index[0]), str(va.index[-1]), len(va)],
                   'test': [str(idx[0]), str(idx[-1]), len(idx)]},
        'metrics': metrics,
        'baselines': baselines(df[TARGET], idx, a),
        'best_epoch': h.get('best_epoch'),
        'history': {'train_loss': h['train_loss'], 'val_loss': h['val_loss'],
                    'learning_rate': h['learning_rate']},
    }
    print(f"  [{name}] test {idx[0]:%Y-%m}~{idx[-1]:%Y-%m}  MAPE {metrics['MAPE']:.2f}% "
          f"(persistence {result['baselines']['persistence_t-1']['MAPE']:.2f}%)  "
          f"R2 {metrics['R2']:.3f}  best_epoch {h.get('best_epoch')}", flush=True)
    return result, model, scaler, (idx, p, a)


def beats(m: Dict, b: Dict) -> bool:
    return m['MAPE'] < b['MAPE'] and m['RMSE'] < b['RMSE']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(PROJECT_ROOT / 'data/processed/jeju_hourly_merged.csv'))
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'models/demand_retrain_2026-11-05'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default='demand_only,weather_full')
    ap.add_argument('--skip-wf', action='store_true')
    ap.add_argument('--verbose', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None, help='스모크 테스트용 epoch 상한')
    ap.add_argument('--residual', action='store_true', help='persistence 위의 보정량을 학습')
    args = ap.parse_args()
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        TRAINING_CONFIG['epochs'] = args.epochs

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        filled_path = Path(tmp) / 'filled.csv'
        filled = fill_short_gaps(Path(args.data), filled_path)
        print(f"Filled hours: {filled}", flush=True)
        report = {'created': datetime.now().isoformat(), 'device': str(device), 'seed': args.seed,
                  'residual_target': args.residual,
                  'model_config': MODEL_CONFIG, 'training_config': TRAINING_CONFIG,
                  'filled_hours': filled, 'models': {}}
        for name in [m.strip() for m in args.models.split(',') if m.strip()]:
            if name not in FEATURE_SETS:
                raise ValueError(f"Unknown model: {name}")
            df_all, _ = load_data(str(filled_path), include_weather=name == 'weather_full')
            df = df_all[FEATURE_SETS[name]].copy()
            print(f"\n=== {name} ({df.shape[1]} features, {df.index[0]} ~ {df.index[-1]})", flush=True)

            fixed, model, scaler, (idx, p, a) = run_fold('fixed', df, FIXED_SPLIT, device,
                                                         args.seed, args.verbose, args.residual)
            # 잔차 모델은 기본 LSTM 가중치만 저장하고, 추론 쪽이 앵커를 더해야 함을 체크포인트에 표시한다
            save_model(model.base if args.residual else model, scaler, FEATURE_SETS[name],
                       fixed['metrics'], TRAINING_CONFIG, name, out_dir)
            if args.residual:
                ck_path = out_dir / f'{name}.pt'
                ck = torch.load(ck_path, map_location='cpu', weights_only=False)
                ck['model_config'] = {**ck['model_config'], 'residual_target': True}
                torch.save(ck, ck_path)
            pd.DataFrame({'datetime': idx, 'actual': a, 'pred': p,
                          'persistence': df[TARGET].shift(1).reindex(idx).values}
                         ).to_csv(out_dir / f'{name}_fixed_test_predictions.csv', index=False)

            wf = {}
            if not args.skip_wf:
                for y in WF_TEST_YEARS:
                    bounds = (f'{y - 1}-06-30 23:00', f'{y - 1}-12-31 23:00', f'{y}-12-31 23:00')
                    wf[str(y)] = run_fold(f'wf-{y}', df, bounds, device, args.seed,
                                          args.verbose, args.residual)[0]
            entry = {'features': FEATURE_SETS[name], 'fixed': fixed, 'walk_forward': wf}
            if wf:
                keys = ['MAPE', 'RMSE', 'MAE', 'R2']
                entry['wf_mean'] = {k: float(np.mean([f['metrics'][k] for f in wf.values()])) for k in keys}
                entry['wf_mean_persistence'] = {k: float(np.mean([f['baselines']['persistence_t-1'][k]
                                                                   for f in wf.values()])) for k in keys}
                entry['wf_folds_beating_persistence'] = sum(
                    beats(f['metrics'], f['baselines']['persistence_t-1']) for f in wf.values())
            entry['passed'] = beats(fixed['metrics'], fixed['baselines']['persistence_t-1']) and (
                not wf or beats(entry['wf_mean'], entry['wf_mean_persistence']))
            report['models'][name] = entry
            print(f"  -> passed: {entry['passed']}", flush=True)
            # 모델 하나 끝날 때마다 기록해 중간 실패에도 결과가 남게 한다
            (out_dir / 'oot_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {out_dir / 'oot_report.json'}")


if __name__ == '__main__':
    main()
