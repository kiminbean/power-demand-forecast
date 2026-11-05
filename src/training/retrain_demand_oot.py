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
- 기준선: persistence(t-1), seasonal naive(t-24, t-168), 7일 ramp, 시간대별 선형 AR
  (그 fold의 학습 구간으로 적합)을 같은 타깃 시각에서 평가

합격 기준: fixed 테스트와 walk-forward 평균 모두에서 MAPE·RMSE가
persistence, 7일 ramp, 시간대별 AR보다 모두 낮고, MAE가 각각보다 1% 이상 낮아야 한다.
fixed 테스트에서는 그 차이가 Diebold-Mariano 검정(단측 p < 0.05)으로도 유의해야 한다.
(2026-11-06 경쟁 비교에서 잔차 LSTM이 시간대별 AR에 매년 졌다)

잔차 앵커(--anchor):
- last: 예측 = 직전 수요 + LSTM 보정량 (--residual과 같음)
- ar:   예측 = 시간대별 AR 예측 + LSTM 보정량

Usage:
    python src/training/retrain_demand_oot.py --output models/demand_retrain_2026-11-05
    python src/training/retrain_demand_oot.py --anchor ar --output models/demand_retrain_ar
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
from evaluation.metrics import compute_all_metrics, diebold_mariano
from models.hourly_ar import HourlyLinearAR, ramp7_forecast
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
# 배포하려면 이 기준선을 모두 MAPE·RMSE로 이겨야 한다
GATE_BASELINES = ('persistence_t-1', 'ramp7', 'hourly_ar')
# 지표 차이가 우연이 아니어야 한다. AR 앵커 모델은 거의 학습되지 않아도 0.0001%p 차이로 이길 수 있다
GATE_DM_ALPHA = 0.05
# 유의성만으로는 부족하다: 1에폭 AR 앵커 모델은 MAE 차이 0.0014 MWh(0.008%)로도 DM -3.85였다.
# 운영 모델을 바꿀 만한 실질 개선으로 MAE 1% 이상을 요구한다
GATE_MIN_MAE_GAIN = 0.01
AR_COLUMN = 'ar_forecast'



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


def baseline_predictions(demand: pd.Series, idx: pd.DatetimeIndex, train_end=None) -> Dict[str, np.ndarray]:
    """같은 타깃 시각의 기준선 예측. train_end가 있으면 그 시각까지로 적합한 시간대별 AR을 더한다."""
    preds = {name: demand.shift(lag).reindex(idx).values for name, lag in BASELINE_LAGS.items()}
    preds['ramp7'] = ramp7_forecast(demand).reindex(idx).values
    if train_end is not None:
        ar = HourlyLinearAR().fit(demand, train_end=train_end)
        preds['hourly_ar'] = ar.predict(demand).reindex(idx).values
    for name, pred in preds.items():
        if np.isnan(pred).any():
            raise ValueError(f"Baseline {name} lacks history for some test hours")
    return preds


def baselines(demand: pd.Series, idx: pd.DatetimeIndex, actual: np.ndarray, train_end=None) -> Dict:
    return {name: {k: float(v) for k, v in compute_all_metrics(actual, pred).items()}
            for name, pred in baseline_predictions(demand, idx, train_end).items()}


def add_ar_anchor(df: pd.DataFrame, train_end) -> Tuple[pd.DataFrame, HourlyLinearAR]:
    """학습 구간(≤ train_end)으로 AR을 적합하고, 행 t에 t+1 시각의 AR 예측을 붙인다.

    입력 창의 마지막 행(t-1)에서 타깃 t의 AR 예측을 바로 앵커로 쓰기 위한 배치다.
    시차 169시간이 모자란 앞부분 행만 빠지므로 시간 연속성은 유지된다.
    """
    ar = HourlyLinearAR().fit(df[TARGET], train_end=train_end)
    out = df.assign(**{AR_COLUMN: ar.predict_next(df[TARGET])}).dropna(subset=[AR_COLUMN])
    return out, ar


def anchor_affine(train_values: np.ndarray, anchor_idx: int) -> Tuple[float, float]:
    """TimeSeriesScaler(min-max)로 스케일된 앵커 열을 타깃(0번 열) 스케일로 옮기는 (배율, 이동)."""
    lo, hi = np.nanmin(train_values, axis=0), np.nanmax(train_values, axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return float(span[anchor_idx] / span[0]), float((lo[anchor_idx] - lo[0]) / span[0])


def run_fold(name, df, bounds, device, seed, verbose, anchor=None):
    """한 분할을 학습·평가한다. anchor: None, 'last'(직전 값), 'ar'(시간대별 AR 예측)."""
    demand, ar, wrapper = df[TARGET], None, None
    if anchor == 'ar':
        df, ar = add_ar_anchor(df, bounds[0])
    tr, va, te = split(df, *bounds)
    if anchor == 'last':
        wrapper = LastValueResidual
    elif anchor == 'ar':
        ar_idx = df.columns.get_loc(AR_COLUMN)
        scale, shift = anchor_affine(tr.values, ar_idx)

        def wrapper(base):
            return LastValueResidual(base, target_idx=ar_idx, anchor_scale=scale, anchor_shift=shift)
    elif anchor is not None:
        raise ValueError(f"Unknown anchor: {anchor}")
    model, scaler, hist = train_model(tr, va, df.shape[1], device, TRAINING_CONFIG,
                                      seed=seed, verbose=verbose, model_wrapper=wrapper)
    if anchor == 'ar':
        fitted = (scaler.range_vals[ar_idx] / scaler.range_vals[0],
                  (scaler.min_vals[ar_idx] - scaler.min_vals[0]) / scaler.range_vals[0])
        if not np.allclose((scale, shift), fitted):
            raise AssertionError("AR anchor scaling does not match the fitted scaler")
    idx, p, a = predict_period(model, scaler, df, te.index[0], device, te.index[-1])
    metrics = {k: float(v) for k, v in compute_all_metrics(a, p).items()}
    # 기준선은 앵커 열 때문에 앞부분이 잘리기 전의 수요로 계산해 앵커 종류와 상관없이 같게 한다
    base_preds = baseline_predictions(demand, idx, train_end=bounds[0])
    h = hist.history
    result = {
        'bounds': {'train': [str(tr.index[0]), str(tr.index[-1]), len(tr)],
                   'val': [str(va.index[0]), str(va.index[-1]), len(va)],
                   'test': [str(idx[0]), str(idx[-1]), len(idx)]},
        'metrics': metrics,
        'baselines': {n: {k: float(v) for k, v in compute_all_metrics(a, pr).items()}
                      for n, pr in base_preds.items()},
        'dm': {b: diebold_mariano(p - a, base_preds[b] - a) for b in GATE_BASELINES},
        'best_epoch': h.get('best_epoch'),
        'history': {'train_loss': h['train_loss'], 'val_loss': h['val_loss'],
                    'learning_rate': h['learning_rate']},
    }
    print(f"  [{name}] test {idx[0]:%Y-%m}~{idx[-1]:%Y-%m}  MAPE {metrics['MAPE']:.2f}% "
          f"(persistence {result['baselines']['persistence_t-1']['MAPE']:.2f}%, "
          f"AR {result['baselines']['hourly_ar']['MAPE']:.2f}%, "
          f"DM vs AR {result['dm']['hourly_ar']['stat']:+.2f})  "
          f"R2 {metrics['R2']:.3f}  best_epoch {h.get('best_epoch')}", flush=True)
    frame = pd.DataFrame({'datetime': idx, 'actual': a, 'pred': p,
                          **{'persistence' if n == 'persistence_t-1' else n: pr
                             for n, pr in base_preds.items()}})
    return result, model, scaler, frame, ar


def beats(m: Dict, b: Dict) -> bool:
    return m['MAPE'] < b['MAPE'] and m['RMSE'] < b['RMSE']


def passes_gate(metrics: Dict, base: Dict) -> bool:
    """MAPE·RMSE가 모든 게이트 기준선보다 낮아야 통과한다."""
    return all(beats(metrics, base[name]) for name in GATE_BASELINES)


def significant(dm: Dict) -> bool:
    """모든 게이트 기준선보다 절대오차가 유의하게 작아야 한다 (DM 단측 p < GATE_DM_ALPHA)."""
    return all(dm[name]['p_a_better'] < GATE_DM_ALPHA for name in GATE_BASELINES)


def mae_gain(metrics: Dict, baseline: Dict) -> float:
    """기준선 대비 MAE 상대 개선 (양수면 개선)."""
    return 1.0 - metrics['MAE'] / baseline['MAE']


def material(metrics: Dict, base: Dict) -> bool:
    """모든 게이트 기준선보다 MAE가 GATE_MIN_MAE_GAIN 이상 낮아야 한다."""
    return all(mae_gain(metrics, base[name]) >= GATE_MIN_MAE_GAIN for name in GATE_BASELINES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(PROJECT_ROOT / 'data/processed/jeju_hourly_merged.csv'))
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'models/demand_retrain_2026-11-05'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default='demand_only,weather_full')
    ap.add_argument('--skip-wf', action='store_true')
    ap.add_argument('--verbose', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None, help='스모크 테스트용 epoch 상한')
    ap.add_argument('--residual', action='store_true', help='persistence 위의 보정량을 학습 (--anchor last와 같음)')
    ap.add_argument('--anchor', choices=['last', 'ar'], default=None,
                    help='잔차 앵커: last=직전 수요, ar=시간대별 AR 예측')
    args = ap.parse_args()
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.residual and args.anchor not in (None, 'last'):
        raise ValueError("--residual means --anchor last; do not combine it with another anchor")
    anchor = args.anchor or ('last' if args.residual else None)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        filled_path = Path(tmp) / 'filled.csv'
        filled = fill_short_gaps(Path(args.data), filled_path)
        print(f"Filled hours: {filled}", flush=True)
        report = {'created': datetime.now().isoformat(), 'device': str(device), 'seed': args.seed,
                  'residual_target': anchor is not None, 'anchor': anchor,
                  'gate_baselines': list(GATE_BASELINES),
                  'model_config': MODEL_CONFIG, 'training_config': TRAINING_CONFIG,
                  'filled_hours': filled, 'models': {}}
        for name in [m.strip() for m in args.models.split(',') if m.strip()]:
            if name not in FEATURE_SETS:
                raise ValueError(f"Unknown model: {name}")
            df_all, _ = load_data(str(filled_path), include_weather=name == 'weather_full')
            df = df_all[FEATURE_SETS[name]].copy()
            features = FEATURE_SETS[name] + ([AR_COLUMN] if anchor == 'ar' else [])
            print(f"\n=== {name} ({len(features)} features, anchor {anchor}, "
                  f"{df.index[0]} ~ {df.index[-1]})", flush=True)

            fixed, model, scaler, frame, ar = run_fold('fixed', df, FIXED_SPLIT, device,
                                                       args.seed, args.verbose, anchor)
            # 잔차 모델은 기본 LSTM 가중치만 저장하고, 추론 쪽이 앵커를 더해야 함을 체크포인트에 표시한다
            save_model(model.base if anchor else model, scaler, features,
                       fixed['metrics'], TRAINING_CONFIG, name, out_dir)
            if anchor:
                ck_path = out_dir / f'{name}.pt'
                ck = torch.load(ck_path, map_location='cpu', weights_only=False)
                extra = {'residual_target': True}
                if anchor == 'ar':
                    # 서빙 경로가 앵커 열을 다시 만들 수 있도록 AR 계수와 스케일 변환을 함께 남긴다
                    extra.update({'residual_anchor': AR_COLUMN, 'anchor_scale': model.anchor_scale,
                                  'anchor_shift': model.anchor_shift, 'hourly_ar': ar.to_dict()})
                ck['model_config'] = {**ck['model_config'], **extra}
                torch.save(ck, ck_path)
            frame.to_csv(out_dir / f'{name}_fixed_test_predictions.csv', index=False)

            wf = {}
            if not args.skip_wf:
                for y in WF_TEST_YEARS:
                    bounds = (f'{y - 1}-06-30 23:00', f'{y - 1}-12-31 23:00', f'{y}-12-31 23:00')
                    wf[str(y)], _, _, wf_frame, _ = run_fold(f'wf-{y}', df, bounds, device, args.seed,
                                                             args.verbose, anchor)
                    # fold별 예측을 남겨 나중에 다른 기준(원본/절단 목표, DM 검정)으로 다시 채점할 수 있게 한다
                    wf_frame.to_csv(out_dir / f'{name}_wf_{y}_predictions.csv', index=False)
            entry = {'features': features, 'fixed': fixed, 'walk_forward': wf}
            if wf:
                keys = ['MAPE', 'RMSE', 'MAE', 'R2']
                entry['wf_mean'] = {k: float(np.mean([f['metrics'][k] for f in wf.values()])) for k in keys}
                entry['wf_mean_baselines'] = {
                    b: {k: float(np.mean([f['baselines'][b][k] for f in wf.values()])) for k in keys}
                    for b in GATE_BASELINES}
                entry['wf_folds_beating'] = {
                    b: sum(beats(f['metrics'], f['baselines'][b]) for f in wf.values()) for b in GATE_BASELINES}
                # 이전 보고서와 같은 키도 유지한다
                entry['wf_mean_persistence'] = entry['wf_mean_baselines']['persistence_t-1']
                entry['wf_folds_beating_persistence'] = entry['wf_folds_beating']['persistence_t-1']
            entry['gate'] = {b: {'fixed': beats(fixed['metrics'], fixed['baselines'][b]),
                                 'fixed_mae_gain': mae_gain(fixed['metrics'], fixed['baselines'][b]),
                                 'fixed_dm_p': fixed['dm'][b]['p_a_better'],
                                 'wf_mean': beats(entry['wf_mean'], entry['wf_mean_baselines'][b]) if wf else None,
                                 'wf_mean_mae_gain': (mae_gain(entry['wf_mean'], entry['wf_mean_baselines'][b])
                                                      if wf else None)}
                             for b in GATE_BASELINES}
            fixed_ok = (passes_gate(fixed['metrics'], fixed['baselines']) and significant(fixed['dm'])
                        and material(fixed['metrics'], fixed['baselines']))
            wf_ok = not wf or (passes_gate(entry['wf_mean'], entry['wf_mean_baselines'])
                               and material(entry['wf_mean'], entry['wf_mean_baselines']))
            entry['passed'] = fixed_ok and wf_ok
            report['models'][name] = entry
            print(f"  -> passed: {entry['passed']}", flush=True)
            # 모델 하나 끝날 때마다 기록해 중간 실패에도 결과가 남게 한다
            (out_dir / 'oot_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {out_dir / 'oot_report.json'}")


if __name__ == '__main__':
    main()
