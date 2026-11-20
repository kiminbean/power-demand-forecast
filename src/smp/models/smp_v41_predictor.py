"""제주 하루전 SMP 서빙 예측기: 이력 제공자, v4.1 추론기, 전날 같은 시간 기준선.

이력(JejuDAHistoryProvider)은 다음 순서로 합친다 (같은 거래일·시간은 앞 원천 우선):
1. 제주시범사업 API 최근 페이지 (필요할 때만, 실패는 일정 시간 기억해 일일 한도를 아낀다)
2. data/smp/smp_real_epsis.csv의 smp_jeju (스케줄러가 갱신, KPX 하루전 공표값)
3. data/smp/jeju_da_smp_hourly.csv (jeju_da_history.py로 구축한 연속 이력)

- v4.1: train_smp_v41_serving.py 산출물(가중치·정규화 통계·평가 결과)과 같은 피처 모듈을 쓴다.
  오늘 0시까지 피처 참조 구간 포함 384시간이 끊김 없이 있어야 하고, 평가 기준(gate)을 통과해야 쓴다.
- 기준선: 오늘 각 거래시간 = 어제 같은 거래시간 가격. 학습 파라미터가 없어 최근 이력에서 재현한
  MAPE가 곧 표본 밖 성능이다. 2025-07~12 walk-forward에서 v4.1(16.82%)보다 나았다(11.75%).
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.smp.data import jeju_da_history
from src.smp.models.smp_v41_features import (
    FEATURE_WARMUP_HOURS, INPUT_HOURS, OUTPUT_HOURS, apply_normalization, build_features, denormalize,
    prepare_smp_frame,
)
from src.smp.models.train_smp_v4_smp_only import SMPModelV41

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = PROJECT_ROOT / "models/smp_v41_serving/smp_v41_serving.pt"
DEFAULT_HISTORY = PROJECT_ROOT / "data/smp/jeju_da_smp_hourly.csv"
DEFAULT_RECENT = PROJECT_ROOT / "data/smp/smp_real_epsis.csv"
REQUIRED_HOURS = FEATURE_WARMUP_HOURS + INPUT_HOURS
# 피처 계산에 쓰는 꼬리 행 수 (지연 336h + 이동창 168h보다 넉넉하게)
FEATURE_TAIL_ROWS = REQUIRED_HOURS + 200
HISTORY_CACHE_SECONDS = 3600
API_RETRY_AFTER_FAILURE_SECONDS = 1800
API_TOPUP_PAGES = 2  # 1000행/페이지 ≈ 10일씩
BASELINE_EVAL_DAYS = 150  # 기준선 성능을 재현하는 최근 평가 일수 (연속 이틀이 있는 날만)
BASELINE_COVERAGE_HOLDOUT_DAYS = 30  # 80% 구간 적중률을 마지막 30일 보류(holdout) 검증으로 계산
BASELINE_COVERAGE_MIN_FIT_DAYS = 60  # 분위수 적합에 최소한으로 쓸 일수 (이보다 부족하면 적중률 미계산)
BASELINE_VOL_WINDOW_DAYS = 14  # 구간 폭 조정에 참고하는 최근 변동성 창(일)
BASELINE_VOL_SCALE_MAX = 2.0  # 변동성 스케일 상한 (일시적 폭등이 구간을 무한히 넓히지 않게 한다)
MAPE_MIN_ABS = 10.0  # v4.1 평가와 같은 기준 (|실측| <= 10원은 MAPE에서 제외)


class InsufficientHistoryError(RuntimeError):
    """오늘 0시까지 필요한 연속 이력이 없다."""


def _tail_is_contiguous(df: pd.DataFrame, end_at: pd.Timestamp, hours: int) -> bool:
    if df.empty or df["datetime"].iloc[-1] != end_at or len(df) < hours:
        return False
    tail = df["datetime"].iloc[-hours:]
    return bool((tail.diff().iloc[1:] == pd.Timedelta(hours=1)).all())


class JejuDAHistoryProvider:
    """제주 하루전 SMP 시간별 이력 (오늘 0시까지). 예측기들이 함께 써서 API 호출을 아낀다."""

    def __init__(self, history_path: Path = DEFAULT_HISTORY, recent_path: Path = DEFAULT_RECENT,
                 api_topup: bool = True):
        self.history_path = Path(history_path)
        self.recent_path = Path(recent_path)
        self.api_topup = api_topup
        self._frame: Optional[pd.DataFrame] = None
        self._frame_key: Optional[pd.Timestamp] = None
        self._frame_at = 0.0
        self._api_failed_at = 0.0

    def _api_frame(self) -> pd.DataFrame:
        if not self.api_topup or time.time() - self._api_failed_at < API_RETRY_AFTER_FAILURE_SECONDS:
            return pd.DataFrame(columns=jeju_da_history.COLUMNS)
        try:
            key = jeju_da_history.api_key()
            items = []
            for page_no in range(1, API_TOPUP_PAGES + 1):
                page_items, _ = jeju_da_history.fetch_page(page_no, key)
                items.extend(page_items)
            return jeju_da_history.to_frame(jeju_da_history.hourly_from_items(items), "제주시범사업 API jsmp")
        except (RuntimeError, ValueError, KeyError) as exc:
            self._api_failed_at = time.time()
            logger.warning("Jeju SMP history API top-up failed: %s", exc)
            return pd.DataFrame(columns=jeju_da_history.COLUMNS)

    def _merged(self, end_at: pd.Timestamp, use_api: bool) -> pd.DataFrame:
        frames = [self._api_frame()] if use_api else []
        frames.append(jeju_da_history.from_local(self.recent_path, "EPSIS smp_real_epsis.csv (smp_jeju)"))
        frames.append(jeju_da_history.from_local(self.history_path, "jeju_da_smp_hourly.csv"))
        return prepare_smp_frame(jeju_da_history.merge_history(frames), end_at, source="merged Jeju history")

    def get(self, end_at: pd.Timestamp, required_hours: int) -> pd.DataFrame:
        """end_at에서 끝나는 required_hours 연속 이력을 보장하는 전체 이력 (부족하면 API로 한 번 보충)."""
        fresh = (self._frame is not None and self._frame_key == end_at
                 and time.time() - self._frame_at < HISTORY_CACHE_SECONDS)
        if fresh and _tail_is_contiguous(self._frame, end_at, required_hours):
            return self._frame
        df = self._frame if fresh else self._merged(end_at, use_api=False)
        if not _tail_is_contiguous(df, end_at, required_hours):
            df = self._merged(end_at, use_api=True)
        self._frame, self._frame_key, self._frame_at = df, end_at, time.time()
        if not _tail_is_contiguous(df, end_at, required_hours):
            latest = df["datetime"].iloc[-1] if len(df) else None
            raise InsufficientHistoryError(f"need {required_hours} contiguous hours ending {end_at}; latest {latest}")
        return df


class SMPV41Predictor:
    version = "v4.1"

    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT, history: Optional[JejuDAHistoryProvider] = None):
        self.artifact_path = Path(artifact_path)
        self.history = history or JejuDAHistoryProvider()
        self.model: Optional[SMPModelV41] = None
        self.meta: Dict[str, Any] = {}
        self.stats: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.artifact_path.exists():
            logger.info("v4.1 serving artifact not found: %s", self.artifact_path)
            return
        artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        model = SMPModelV41(**artifact["model_kwargs"])
        model.load_state_dict(artifact["model_state_dict"])
        model.eval()
        self.stats = {k: np.asarray(v, dtype=np.float64) for k, v in artifact["normalization"].items()}
        if self.stats["mean"].shape != (artifact["model_kwargs"]["input_size"],):
            raise ValueError("normalization statistics do not match the model input size")
        self.model = model
        self.meta = {k: artifact[k] for k in ("walk_forward", "gate_passed", "train_range", "trained_at", "eval_end")}

    def is_ready(self) -> bool:
        """산출물이 있고, 평가에서 단순 기준선(전날 같은 시간)보다 나았을 때만 서빙한다."""
        return self.model is not None and bool(self.meta.get("gate_passed"))

    @property
    def mape(self) -> Optional[float]:
        return self.meta.get("walk_forward", {}).get("v41", {}).get("mape")

    @property
    def coverage(self) -> Optional[float]:
        return self.meta.get("walk_forward", {}).get("v41", {}).get("coverage_80")

    def predict_24h(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("v4.1 serving model is not ready")
        end_at = pd.Timestamp((now or datetime.now()).date())  # 오늘 0시 = 어제 24시 구간 끝
        df = self.history.get(end_at, REQUIRED_HOURS).iloc[-FEATURE_TAIL_ROWS:].reset_index(drop=True)
        features, _, _ = build_features(df)
        x = apply_normalization(features[-INPUT_HOURS:], self.stats)
        with torch.no_grad():
            out = self.model(torch.as_tensor(x[None], dtype=torch.float32))
        point, q10, q90 = (denormalize(out[k].numpy()[0], self.stats) for k in ("point", "q10", "q90"))
        # 분위수 머리가 교차하지 않도록 중앙값 주변으로 정렬한다.
        q10, q90 = np.minimum(q10, point), np.maximum(q90, point)
        return _result(end_at, q10, point, q90, self.version, self.mape, self.coverage)


class SeasonalNaivePredictor:
    """오늘 거래시간 h의 예측 = 어제 거래시간 h의 제주 하루전 가격."""

    version = "baseline-24h"

    def __init__(self, history: Optional[JejuDAHistoryProvider] = None):
        self.history = history or JejuDAHistoryProvider()
        self.backtest: Dict[str, Any] = {}

    def is_ready(self) -> bool:
        return True

    @property
    def mape(self) -> Optional[float]:
        return self.backtest.get("mape")

    @property
    def coverage(self) -> Optional[float]:
        """80% 예측구간의 마지막 30일 보류(holdout) 적중률(%). 평가일이 부족하면 None."""
        return self.backtest.get("coverage_80")

    def _backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """연속 이틀(48시간)이 있는 최근 날들에서 '전날 값으로 다음 날 예측'을 재현한다."""
        series = df.set_index("datetime")["smp_jeju"]
        # 각 날의 0시(전날 24시 구간 끝)를 예측 시점으로 삼는다. 최근 날짜만 본다.
        day_ends = series.index[series.index.hour == 0][-(BASELINE_EVAL_DAYS * 3):]
        errors, actuals = [], []
        for day_end in day_ends:
            source = series.reindex(pd.date_range(day_end - pd.Timedelta(hours=OUTPUT_HOURS - 1), periods=OUTPUT_HOURS, freq="h"))
            actual = series.reindex(pd.date_range(day_end + pd.Timedelta(hours=1), periods=OUTPUT_HOURS, freq="h"))
            if source.isna().any() or actual.isna().any():
                continue
            errors.append(actual.to_numpy() - source.to_numpy())
            actuals.append(actual.to_numpy())
        if not errors:
            return {}
        errors, actuals = np.array(errors[-BASELINE_EVAL_DAYS:]), np.array(actuals[-BASELINE_EVAL_DAYS:])
        mask = np.abs(actuals) > MAPE_MIN_ABS
        result = {
            "mape": float(np.mean(np.abs(errors[mask]) / np.abs(actuals[mask])) * 100),
            "days": len(errors),
            # 시간대별 오차 분위수로 80% 구간을 만든다(서빙 구간은 전체 평가일로 적합).
            "q10_offset": np.percentile(errors, 10, axis=0),
            "q90_offset": np.percentile(errors, 90, axis=0),
        }
        # 최근 변동성이 적합기간보다 크면(체제 변화) 구간 폭을 그 비율만큼 넓힌다.
        # 고정 분위수만 쓰면 변동성이 커진 시기에 적중률이 명목 80%보다 크게 떨어진다.
        fit_sd = float(np.std(errors))
        recent_sd = float(np.std(errors[-BASELINE_VOL_WINDOW_DAYS:]))
        result["vol_scale"] = min(recent_sd / fit_sd, BASELINE_VOL_SCALE_MAX) if fit_sd > 0 else 1.0
        # 적중률은 같은 데이터로 재면하면 ~80%가 나오기 때문에, 마지막 30일을 떼어 놓고
        # 그 30일이 (적합에 쓰지 않은) 나머지 일수의 분위수 구간에 들어가는 비율로 계산한다.
        # 보류의 각 날도 서빙과 똑같이 '그날 직전 창'의 변동성으로 폭을 조정해 검증한다(인과적, 미래 정보 없음).
        if len(errors) >= BASELINE_COVERAGE_HOLDOUT_DAYS + BASELINE_COVERAGE_MIN_FIT_DAYS:
            fit, held = errors[:-BASELINE_COVERAGE_HOLDOUT_DAYS], errors[-BASELINE_COVERAGE_HOLDOUT_DAYS:]
            fit_q10 = np.percentile(fit, 10, axis=0)
            fit_q90 = np.percentile(fit, 90, axis=0)
            fit_sd = float(np.std(fit))
            hits = []
            for k in range(BASELINE_COVERAGE_HOLDOUT_DAYS):
                i = len(errors) - BASELINE_COVERAGE_HOLDOUT_DAYS + k
                scale = 1.0
                if fit_sd > 0 and i >= BASELINE_VOL_WINDOW_DAYS:
                    scale = min(float(np.std(errors[i - BASELINE_VOL_WINDOW_DAYS:i])) / fit_sd, BASELINE_VOL_SCALE_MAX)
                hits.append((held[k] >= fit_q10 * scale) & (held[k] <= fit_q90 * scale))
            result["coverage_80"] = float(np.mean(np.vstack(hits)) * 100)
        return result

    def predict_24h(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        end_at = pd.Timestamp((now or datetime.now()).date())
        df = self.history.get(end_at, OUTPUT_HOURS)
        if not self.backtest or self.backtest.get("end_at") != end_at:
            self.backtest = {**self._backtest(df), "end_at": end_at}
        point = df["smp_jeju"].to_numpy()[-OUTPUT_HOURS:].astype(np.float64)
        # 변동성 스케일을 구간 오프셋에 곱해 최근 체제 변화를 반영한다.
        scale = self.backtest.get("vol_scale", 1.0)
        q10 = point + scale * self.backtest.get("q10_offset", np.zeros(OUTPUT_HOURS))
        q90 = point + scale * self.backtest.get("q90_offset", np.zeros(OUTPUT_HOURS))
        q10, q90 = np.minimum(q10, point), np.maximum(q90, point)
        return _result(end_at, q10, point, q90, self.version, self.mape, self.coverage)


def _result(end_at: pd.Timestamp, q10, q50, q90, version: str, mape, coverage) -> Dict[str, Any]:
    start = end_at.to_pydatetime()
    return {
        # 구간 시작 시각 (00:00 = 거래시간 1). smp_routes(hour % 24 + 1)·입찰(hour)이 이 규칙을 가정한다.
        "times": [start + timedelta(hours=i) for i in range(OUTPUT_HOURS)],
        "q10": q10, "q50": q50, "q90": q90,
        "model_used": version, "mape": mape, "coverage": coverage,
        "forecast_date": end_at.date().isoformat(),
        "timestamp": datetime.now().isoformat(),
    }
