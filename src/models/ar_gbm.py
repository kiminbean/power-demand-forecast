"""
시간대별 선형 AR + gradient boosting 잔차 보정: 1시간 앞 제주 전력거래량
==========================================================================

예측 = AR(t) + GBM이 예측한 AR 잔차. 잔차 모델의 입력은 t-1까지의 관측, t 시각 달력,
t 시각 AR 예측, 과거 AR 잔차뿐이다. docs/2026-11-06-demand-retrain-raw-target.md 7절의 변형 B로,
고정 test와 walk-forward 4개 연도 모두에서 배포 게이트를 통과했다.

학습과 추론이 같은 residual_features로 입력을 만들어 학습·서빙 불일치를 막는다.
GBM은 joblib(pickle)로 저장하므로 신뢰하는 산출물만 불러온다. 학습과 같은 scikit-learn
버전(numpy는 같은 주 버전)에서만 불러온다. 파일 sha256을 마지막에 쓰는 metadata.json에 남겨,
일부 파일만 바뀐 산출물(AR과 GBM 짝이 어긋난 경우)을 불러오지 않는다.
"""

import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .hourly_ar import AR_LAGS, HourlyLinearAR

TARGET = 'power_demand'
RESID_LAGS = (1, 2, 3, 24, 168)
# 어제·지난주 같은 시각이 휴일이면 AR의 24·168시간 시차가 낮게 끌려간다
HOLIDAY_LAGS = (24, 168)
GBM_PARAMS = {'learning_rate': 0.05, 'max_leaf_nodes': 31, 'min_samples_leaf': 200}
# 트리 수는 검증 MAE로 고른다 (sklearn 내장 조기 종료는 무작위 분할이라 시계열에 누수)
TREE_STEP = 25
MAX_TREES = 2000
PATIENCE_STEPS = 8
# 다음 시각 예측에 필요한 최소 이력: 168시간 전 잔차가 그 시각의 AR 예측(169시간 시차)을 요구한다
MIN_HISTORY_HOURS = max(RESID_LAGS) + max(AR_LAGS)
FORMAT_VERSION = 1
AR_FILE, GBM_FILE, META_FILE = 'ar_coefficients.json', 'gbm.joblib', 'metadata.json'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_with(path: Path, write) -> None:
    """임시 파일에 쓴 뒤 교체해, 쓰다 멈춰도 반쯤 쓴 파일이 남지 않게 한다."""
    tmp = path.with_name(f'.{path.name}.tmp')
    write(tmp)
    os.replace(tmp, path)


def residual_features(frame: pd.DataFrame, ar_pred: pd.Series, resid: pd.Series,
                      calendar: Sequence[str]) -> pd.DataFrame:
    """행 t의 잔차 모델 입력을 만든다.

    Args:
        frame: 연속된 시간 인덱스의 관측·달력 피처 (TARGET과 is_holiday 포함)
        ar_pred, resid: frame 인덱스를 덮는 연속 시계열의 AR 예측과 AR 잔차. frame보다 앞에서
            시작해도 된다 (추론 때는 피처 warm-up으로 잘리기 전 수요로 계산한다).
        calendar: 미리 알 수 있는 달력 열. t 시각 값을 그대로 쓴다.

    Returns:
        관측 피처는 t-1 값(_prev 접미사), 달력은 t 값, AR 예측과 과거 잔차를 더한 DataFrame.
        이력이 모자란 행은 NaN이 남는다.
    """
    calendar = list(calendar)
    x = frame.drop(columns=calendar).shift(1).add_suffix('_prev').join(frame[calendar])
    x['ar_forecast'] = ar_pred.reindex(frame.index)
    for k in RESID_LAGS:
        x[f'resid_lag_{k}'] = resid.shift(k).reindex(frame.index)
    for k in HOLIDAY_LAGS:
        x[f'is_holiday_lag_{k}'] = frame['is_holiday'].shift(k)
    return x


class ArGbmModel:
    """시간대별 AR과 그 잔차를 학습하는 HistGradientBoosting의 조합."""

    def __init__(self, features: Sequence[str], calendar: Sequence[str]):
        if TARGET not in features or 'is_holiday' not in calendar:
            raise ValueError(f"features must include {TARGET} and calendar must include is_holiday")
        if not set(calendar) <= set(features):
            raise ValueError("calendar columns must be a subset of features")
        self.features = list(features)
        self.calendar = list(calendar)
        self.ar: Optional[HourlyLinearAR] = None
        self.gbm: Optional[HistGradientBoostingRegressor] = None
        self.design_columns: Optional[List[str]] = None
        self.info: Dict = {}

    def design(self, frame: pd.DataFrame, demand: Optional[pd.Series] = None) -> pd.DataFrame:
        """demand가 없으면 frame의 수요로 AR·잔차를 계산한다 (학습 경로)."""
        if self.ar is None:
            raise RuntimeError("ArGbmModel must be fit or loaded first")
        demand = frame[TARGET] if demand is None else demand
        ar_pred = self.ar.predict(demand)
        return residual_features(frame[self.features], ar_pred, demand - ar_pred, self.calendar)

    def fit(self, frame: pd.DataFrame, train_end, val_end, seed: int = 42) -> "ArGbmModel":
        """frame의 train_end까지로 AR과 GBM을 적합하고, (train_end, val_end] 검증 MAE로 트리 수를 고른다."""
        train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
        if val_end <= train_end:
            raise ValueError("val_end must be after train_end")
        self.ar = HourlyLinearAR().fit(frame[TARGET], train_end=train_end)
        x = self.design(frame)
        r = frame[TARGET] - x['ar_forecast']
        ok = x.notna().all(axis=1) & r.notna()
        tr = ok & (x.index <= train_end)
        va = ok & (x.index > train_end) & (x.index <= val_end)
        if not tr.any() or not va.any():
            raise ValueError(f"Empty split: {int(tr.sum())} train / {int(va.sum())} validation rows")
        x_tr, r_tr, x_va, r_va = x[tr], r[tr], x[va], r[va]
        gbm = HistGradientBoostingRegressor(max_iter=TREE_STEP, early_stopping=False, warm_start=True,
                                            random_state=seed, **GBM_PARAMS)
        ar_val_mae = float(np.mean(np.abs(r_va)))
        # 0그루(보정 없음)에서 시작해 검증 MAE가 가장 낮은 시점의 모델을 그대로 보관한다
        best_mae, best_iter, best_gbm, stale = ar_val_mae, 0, None, 0
        for n_trees in range(TREE_STEP, MAX_TREES + 1, TREE_STEP):
            gbm.set_params(max_iter=n_trees)
            gbm.fit(x_tr, r_tr)
            val_mae = float(np.mean(np.abs(r_va - gbm.predict(x_va))))
            if val_mae < best_mae:
                best_mae, best_iter, best_gbm, stale = val_mae, n_trees, copy.deepcopy(gbm), 0
            else:
                stale += 1
                if stale >= PATIENCE_STEPS:
                    break
        self.gbm = best_gbm
        self.design_columns = list(x.columns)
        self.info = {'train_end': str(train_end), 'val_end': str(val_end), 'seed': seed,
                     'n_train': int(tr.sum()), 'n_val': int(va.sum()), 'best_iter': best_iter,
                     'val_mae': best_mae, 'ar_val_mae': ar_val_mae}
        return self

    def predict(self, frame: pd.DataFrame, demand: Optional[pd.Series] = None) -> pd.Series:
        """frame 각 시각 t의 1시간 앞 예측 (t-1까지의 정보만 사용). 입력이 모자란 행은 NaN."""
        if self.design_columns is None:
            raise RuntimeError("ArGbmModel must be fit or loaded first")
        x = self.design(frame, demand)[self.design_columns]
        ok = x.notna().all(axis=1)
        out = pd.Series(np.nan, index=frame.index, name='forecast')
        if ok.any():
            base = x.loc[ok, 'ar_forecast'].to_numpy()
            correction = self.gbm.predict(x.loc[ok]) if self.gbm is not None else 0.0
            out[ok] = base + correction
        return out

    def save(self, out_dir: Path, extra: Optional[Dict] = None) -> None:
        if self.ar is None or self.design_columns is None:
            raise RuntimeError("ArGbmModel must be fit before save")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _replace_with(out_dir / AR_FILE, lambda p: p.write_text(json.dumps(self.ar.to_dict())))
        files = {AR_FILE: _sha256(out_dir / AR_FILE)}
        if self.gbm is not None:
            _replace_with(out_dir / GBM_FILE, lambda p: joblib.dump(self.gbm, p))
            files[GBM_FILE] = _sha256(out_dir / GBM_FILE)
        # metadata를 마지막에 쓴다: 이것이 바뀌기 전까지 로더는 이전 파일 묶음의 해시로 검사한다
        meta = {'format_version': FORMAT_VERSION, 'created': datetime.now().isoformat(),
                'model': 'hourly_ar + hist_gradient_boosting residual', 'target': TARGET,
                'features': self.features, 'calendar': self.calendar, 'design_columns': self.design_columns,
                'resid_lags': list(RESID_LAGS), 'holiday_lags': list(HOLIDAY_LAGS),
                'gbm_params': GBM_PARAMS, 'tree_step': TREE_STEP, 'max_trees': MAX_TREES,
                'patience_steps': PATIENCE_STEPS, 'min_history_hours': MIN_HISTORY_HOURS,
                'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
                **self.info, **(extra or {}), 'files_sha256': files}
        _replace_with(out_dir / META_FILE,
                      lambda p: p.write_text(json.dumps(meta, indent=2, ensure_ascii=False)))

    @classmethod
    def load(cls, model_dir: Path) -> "ArGbmModel":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / META_FILE).read_text())
        if meta.get('format_version') != FORMAT_VERSION:
            raise ValueError(f"Unsupported AR+GBM format version: {meta.get('format_version')}")
        if meta.get('sklearn_version') != sklearn.__version__:
            # pickle은 버전이 다르면 조용히 다르게 동작할 수 있다 → 학습 버전으로 고정해야 한다
            raise RuntimeError(f"AR+GBM was trained with scikit-learn {meta.get('sklearn_version')}, "
                               f"running {sklearn.__version__}; install the training version")
        if str(meta.get('numpy_version', '')).split('.')[0] != np.__version__.split('.')[0]:
            raise RuntimeError(f"AR+GBM was trained with numpy {meta.get('numpy_version')}, "
                               f"running {np.__version__}; use the same numpy major version")
        # 코드의 입력 정의가 산출물과 다르면 요청 때가 아니라 지금 실패시킨다
        expected = {'resid_lags': list(RESID_LAGS), 'holiday_lags': list(HOLIDAY_LAGS),
                    'min_history_hours': MIN_HISTORY_HOURS}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"AR+GBM artifact {key}={meta.get(key)} does not match the code ({value})")
        files = meta.get('files_sha256')
        if not files or AR_FILE not in files:
            raise ValueError("AR+GBM metadata.json has no file checksums; retrain to write a complete artifact")
        for name, digest in files.items():
            if _sha256(model_dir / name) != digest:
                raise ValueError(f"AR+GBM file {name} does not match metadata.json (partial or mixed write)")
        if (meta['best_iter'] > 0) != (GBM_FILE in files):
            raise ValueError("AR+GBM metadata disagrees on whether a GBM was saved")
        model = cls(meta['features'], meta['calendar'])
        model.ar = HourlyLinearAR.from_dict(json.loads((model_dir / AR_FILE).read_text()))
        if GBM_FILE in files:
            model.gbm = joblib.load(model_dir / GBM_FILE)
            if list(model.gbm.feature_names_in_) != meta['design_columns']:
                raise ValueError("GBM feature names do not match the saved design columns")
        model.design_columns = list(meta['design_columns'])
        model.info = meta
        return model
