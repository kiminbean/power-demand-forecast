"""
AR + GBM 잔차 보정 수요 모델: 게이트 재평가와 서빙 산출물 학습
================================================================

1. 고정 test와 walk-forward 4개 분할을 retrain_demand_oot.py와 같은 경계·기준선·게이트로 채점한다.
2. 같은 절차를 최신 데이터에 적용해 서빙 산출물을 만든다. 마지막 6개월을 검증(트리 수 선택)에 쓰고
   그 앞까지로 학습한다 (2024-12 데이터면 학습 ~2024-06, 검증 2024년 하반기).

산출물(--output, 기본 models/demand_ar_gbm/): ar_coefficients.json, gbm.joblib, metadata.json,
training_report.json. 운영 기본 모델은 바꾸지 않는다 (API에서는 model_type="ar_gbm"으로 선택).
게이트를 통과해야 산출물을 쓴다. --allow-ungated로 써도 서빙 로더는 게이트 미통과 산출물을 거부한다.

Usage (repo root):
    python src/training/train_demand_ar_gbm.py [--output DIR] [--pred-dir DIR] [--skip-eval --allow-ungated]
"""

import sys
import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from evaluation.metrics import compute_all_metrics, diebold_mariano  # noqa: E402
from features.time_features import LUNAR_HOLIDAY_LAST_YEAR  # noqa: E402
from models.ar_gbm import ArGbmModel, TARGET  # noqa: E402
from training.retrain_demand_oot import (  # noqa: E402
    FEATURE_SETS, FIXED_SPLIT, GATE_BASELINES, WF_TEST_YEARS, baseline_predictions, beats,
    fill_short_gaps, mae_gain, material, passes_gate, significant,
)
from training.train_production import TIME_FEATURES, load_data  # noqa: E402

FEATURES = FEATURE_SETS['demand_only']
# 서빙 산출물의 검증 구간 길이 (walk-forward 분할의 검증 구간과 같다)
SERVING_VAL_MONTHS = 6
METRIC_KEYS = ('MAPE', 'RMSE', 'MAE', 'R2')


def load_frame(data_path: Path) -> pd.DataFrame:
    """재학습 스크립트와 같은 경로(결측 보간 → load_data → 피처 선택)로 데이터를 만든다."""
    with tempfile.TemporaryDirectory() as tmp:
        filled = Path(tmp) / 'filled.csv'
        fill_short_gaps(data_path, filled)
        df_all, _ = load_data(str(filled), include_weather=False)
    return df_all[FEATURES].copy()


def evaluate_fold(frame: pd.DataFrame, bounds: Tuple[str, str, str], seed: int) -> Tuple[Dict, pd.DataFrame]:
    """한 분할을 학습·채점한다. 기준선과 DM 검정은 retrain_demand_oot.run_fold와 같다."""
    train_end, val_end, test_end = map(pd.Timestamp, bounds)
    model = ArGbmModel(FEATURES, TIME_FEATURES).fit(frame, train_end, val_end, seed=seed)
    pred = model.predict(frame)
    idx = frame.index[(frame.index > val_end) & (frame.index <= test_end)]
    if pred.reindex(idx).isna().any():
        raise ValueError(f"Missing AR+GBM predictions in test {idx[0]}~{idx[-1]}")
    p, a = pred.reindex(idx).to_numpy(), frame[TARGET].reindex(idx).to_numpy()
    base_preds = baseline_predictions(frame[TARGET], idx, train_end=bounds[0])
    result = {
        'bounds': {'train_end': bounds[0], 'val_end': bounds[1], 'test': [str(idx[0]), str(idx[-1]), len(idx)]},
        'metrics': {k: float(v) for k, v in compute_all_metrics(a, p).items()},
        'baselines': {n: {k: float(v) for k, v in compute_all_metrics(a, bp).items()}
                      for n, bp in base_preds.items()},
        'dm': {b: diebold_mariano(p - a, base_preds[b] - a) for b in GATE_BASELINES},
        'model': model.info,
    }
    frame_out = pd.DataFrame({'datetime': idx, 'actual': a, 'pred': p,
                              **{'persistence' if n == 'persistence_t-1' else n: bp
                                 for n, bp in base_preds.items()}})
    return result, frame_out


def gate(fixed: Dict, wf: Dict) -> Dict:
    """retrain_demand_oot.py와 같은 배포 게이트."""
    wf_mean = {k: float(np.mean([f['metrics'][k] for f in wf.values()])) for k in METRIC_KEYS}
    wf_base = {b: {k: float(np.mean([f['baselines'][b][k] for f in wf.values()])) for k in METRIC_KEYS}
               for b in GATE_BASELINES}
    fixed_ok = (passes_gate(fixed['metrics'], fixed['baselines']) and significant(fixed['dm'])
                and material(fixed['metrics'], fixed['baselines']))
    wf_ok = passes_gate(wf_mean, wf_base) and material(wf_mean, wf_base)
    detail = {b: {'fixed': beats(fixed['metrics'], fixed['baselines'][b]),
                  'fixed_mae_gain': mae_gain(fixed['metrics'], fixed['baselines'][b]),
                  'fixed_dm_p': fixed['dm'][b]['p_a_better'],
                  'wf_mean': beats(wf_mean, wf_base[b]), 'wf_mean_mae_gain': mae_gain(wf_mean, wf_base[b]),
                  'wf_folds_beating': sum(beats(f['metrics'], f['baselines'][b]) for f in wf.values())}
              for b in GATE_BASELINES}
    return {'wf_mean': wf_mean, 'wf_mean_baselines': wf_base, 'detail': detail, 'passed': fixed_ok and wf_ok}


def serving_bounds(frame: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """마지막 SERVING_VAL_MONTHS개월을 검증으로, 그 앞을 학습으로 쓴다."""
    val_end = frame.index[-1]
    return val_end - pd.DateOffset(months=SERVING_VAL_MONTHS), val_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(PROJECT_ROOT / 'data/processed/jeju_hourly_merged.csv'))
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'models/demand_ar_gbm'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-eval', action='store_true', help='게이트 재평가 없이 산출물만 학습 (--allow-ungated 필요)')
    ap.add_argument('--allow-ungated', action='store_true',
                    help='게이트를 통과하지 않은 산출물도 쓴다 (서빙 로더는 이런 산출물을 거부한다)')
    ap.add_argument('--pred-dir', default=None, help='분할별 예측 CSV를 저장할 폴더 (기본: 저장 안 함)')
    args = ap.parse_args()
    if args.skip_eval and not args.allow_ungated:
        ap.error('--skip-eval writes an unevaluated model; add --allow-ungated to do it anyway')

    frame = load_frame(Path(args.data))
    print(f"Data: {frame.index[0]} ~ {frame.index[-1]} ({len(frame)} rows, {len(FEATURES)} features)", flush=True)
    if frame.index[-1].year > LUNAR_HOLIDAY_LAST_YEAR:
        # 표에 없는 해의 설·추석이 평일로 학습되면 잔차 보정이 휴일을 거꾸로 배운다
        raise ValueError(f"Data reaches {frame.index[-1].year} but lunar holidays are listed only through "
                         f"{LUNAR_HOLIDAY_LAST_YEAR}; extend features.time_features.LUNAR_HOLIDAYS first")
    report = {'created': datetime.now().isoformat(), 'seed': args.seed, 'features': FEATURES,
              'data': {'start': str(frame.index[0]), 'end': str(frame.index[-1])}}
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_eval:
        fixed, fixed_frame = evaluate_fold(frame, FIXED_SPLIT, args.seed)
        wf = {}
        for y in WF_TEST_YEARS:
            # walk-forward 경계는 retrain_demand_oot.py와 같다: 전년 6월 말까지 학습, 하반기 검증, 해당 연도 test
            wf[str(y)], wf_frame = evaluate_fold(
                frame, (f'{y - 1}-06-30 23:00', f'{y - 1}-12-31 23:00', f'{y}-12-31 23:00'), args.seed)
            if pred_dir:
                wf_frame.to_csv(pred_dir / f'demand_only_wf_{y}_predictions.csv', index=False)
        if pred_dir:
            fixed_frame.to_csv(pred_dir / 'demand_only_fixed_test_predictions.csv', index=False)
        report.update({'fixed': fixed, 'walk_forward': wf, 'gate': gate(fixed, wf)})
        g = report['gate']
        print(f"  [fixed] MAPE {fixed['metrics']['MAPE']:.3f}% (AR {fixed['baselines']['hourly_ar']['MAPE']:.3f}%)"
              f"  WF mean MAPE {g['wf_mean']['MAPE']:.3f}%  -> passed: {g['passed']}", flush=True)
        if not g['passed'] and not args.allow_ungated:
            raise SystemExit(f"Gate failed; not writing the artifact to {args.output} "
                             "(--allow-ungated writes it, but serving refuses it)")

    train_end, val_end = serving_bounds(frame)
    model = ArGbmModel(FEATURES, TIME_FEATURES).fit(frame, train_end, val_end, seed=args.seed)
    gate_summary = {'passed': report['gate']['passed'] if 'gate' in report else None,
                    'evaluated': 'gate' in report}
    model.save(Path(args.output), extra={'gate': gate_summary, 'data_end': str(frame.index[-1])})
    report['serving'] = model.info
    print(f"  [serving] train ~{train_end}, val ~{val_end}: {model.info['best_iter']} trees, "
          f"val MAE {model.info['val_mae']:.3f} (AR {model.info['ar_val_mae']:.3f})", flush=True)
    (Path(args.output) / 'training_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
