"""SMP v4.1 서빙 추론기 (제주 하루전 SMP, 오늘 거래시간 1~24 예측).

train_smp_v41_serving.py가 저장한 단일 산출물(가중치·정규화 통계·평가 결과)을 쓰고,
피처는 학습과 같은 smp_v41_features 모듈로 만든다.

입력 이력은 다음 순서로 합친다 (같은 거래일·시간은 앞 원천 우선):
1. 제주시범사업 API 최근 페이지 (필요할 때만, 실패는 일정 시간 기억해 일일 한도를 아낀다)
2. data/smp/smp_real_epsis.csv의 smp_jeju (스케줄러가 갱신, KPX 하루전 공표값)
3. data/smp/jeju_da_smp_hourly.csv (jeju_da_history.py로 구축한 연속 이력)
오늘 0시(어제 24시 구간 끝)까지 피처 참조 구간 포함 384시간이 끊김 없이 있어야 예측한다.
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


class InsufficientHistoryError(RuntimeError):
    """오늘 0시까지 연속 이력이 부족해 v4.1이 예측할 수 없다."""


class SMPV41Predictor:
    version = "v4.1"

    def __init__(self, artifact_path: Path = DEFAULT_ARTIFACT, history_path: Path = DEFAULT_HISTORY,
                 recent_path: Path = DEFAULT_RECENT, api_topup: bool = True):
        self.artifact_path = Path(artifact_path)
        self.history_path = Path(history_path)
        self.recent_path = Path(recent_path)
        self.api_topup = api_topup
        self.model: Optional[SMPModelV41] = None
        self.meta: Dict[str, Any] = {}
        self.stats: Dict[str, np.ndarray] = {}
        self._history: Optional[pd.DataFrame] = None
        self._history_at = 0.0
        self._api_failed_at = 0.0
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
            logger.warning("v4.1 history API top-up failed: %s", exc)
            return pd.DataFrame(columns=jeju_da_history.COLUMNS)

    def _merged_history(self, end_at: pd.Timestamp, use_api: bool) -> pd.DataFrame:
        frames = [self._api_frame()] if use_api else []
        frames.append(jeju_da_history.from_local(self.recent_path, "EPSIS smp_real_epsis.csv (smp_jeju)"))
        frames.append(jeju_da_history.from_local(self.history_path, "jeju_da_smp_hourly.csv"))
        return prepare_smp_frame(jeju_da_history.merge_history(frames), end_at, source="merged Jeju history")

    @staticmethod
    def _has_required_history(df: pd.DataFrame, end_at: pd.Timestamp) -> bool:
        if df.empty or df["datetime"].iloc[-1] != end_at or len(df) < REQUIRED_HOURS:
            return False
        tail = df["datetime"].iloc[-REQUIRED_HOURS:]
        return bool((tail.diff().iloc[1:] == pd.Timedelta(hours=1)).all())

    def _history_for(self, end_at: pd.Timestamp) -> pd.DataFrame:
        fresh = self._history is not None and time.time() - self._history_at < HISTORY_CACHE_SECONDS
        if fresh and self._has_required_history(self._history, end_at):
            return self._history
        df = self._merged_history(end_at, use_api=False)
        if not self._has_required_history(df, end_at):
            df = self._merged_history(end_at, use_api=True)
        if not self._has_required_history(df, end_at):
            raise InsufficientHistoryError(
                f"need {REQUIRED_HOURS} contiguous hours ending {end_at}; latest {df['datetime'].iloc[-1] if len(df) else None}")
        self._history, self._history_at = df, time.time()
        return df

    def predict_24h(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("v4.1 serving model is not ready")
        now = now or datetime.now()
        end_at = pd.Timestamp(now.date())  # 오늘 0시 = 어제 24시 구간 끝
        df = self._history_for(end_at).iloc[-FEATURE_TAIL_ROWS:].reset_index(drop=True)
        features, _, _ = build_features(df)
        x = apply_normalization(features[-INPUT_HOURS:], self.stats)
        with torch.no_grad():
            out = self.model(torch.as_tensor(x[None], dtype=torch.float32))
        point, q10, q90 = (denormalize(out[k].numpy()[0], self.stats) for k in ("point", "q10", "q90"))
        # 분위수 머리가 교차하지 않도록 중앙값 주변으로 정렬한다.
        q10, q90 = np.minimum(q10, point), np.maximum(q90, point)
        start = end_at.to_pydatetime()
        return {
            # 구간 시작 시각 (00:00 = 거래시간 1). smp_routes(hour % 24 + 1)·입찰(hour)이 이 규칙을 가정한다.
            "times": [start + timedelta(hours=i) for i in range(OUTPUT_HOURS)],
            "q10": q10, "q50": point, "q90": q90,
            "model_used": self.version,
            "mape": self.mape, "coverage": self.coverage,
            "forecast_date": end_at.date().isoformat(),
            "timestamp": datetime.now().isoformat(),
        }
