"""
AR + GBM 잔차 보정 모델의 서빙 경로 (1시간 앞)
================================================

LSTM 경로와 같은 공유 전처리(prepare_inference_features)로 관측·달력 피처를 만들고,
학습과 같은 models.ar_gbm.residual_features로 잔차 모델 입력을 만든다.
마지막 관측 시각 T의 다음 시각(T+1) 행에는 달력 피처만 채운다.

AR 예측과 잔차 시차는 피처 warm-up으로 잘리기 전의 수요로 계산한다.
그래서 이력은 MIN_HISTORY_HOURS(337)시간이면 된다.
배포 게이트를 통과한 산출물만 서빙한다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import add_time_features  # noqa: E402
from features.time_features import LUNAR_HOLIDAY_LAST_YEAR  # noqa: E402,F401  (서비스·테스트가 함께 쓴다)
from inference.feature_preprocessing import MODEL_TIMEZONE, prepare_inference_features  # noqa: E402
from models.ar_gbm import META_FILE, MIN_HISTORY_HOURS, TARGET, ArGbmModel  # noqa: E402

logger = logging.getLogger(__name__)


def _local_index(history: pd.DataFrame) -> pd.DatetimeIndex:
    """공유 전처리와 같은 규칙으로 시간대가 있는 입력을 제주 현지 시각으로 맞춘다."""
    index = pd.DatetimeIndex(history['datetime'] if 'datetime' in history.columns else history.index)
    return index.tz_convert(MODEL_TIMEZONE) if index.tz is not None else index


class ArGbmPredictor:
    """저장된 AR + GBM 산출물로 1시간 앞 수요를 예측한다."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model: Optional[ArGbmModel] = None
        self._warned_years: Set[int] = set()

    def load(self) -> None:
        if not (self.model_dir / META_FILE).exists():
            raise FileNotFoundError(f"AR+GBM model not found: {self.model_dir / META_FILE}")
        model = ArGbmModel.load(self.model_dir)
        # 평가를 건너뛰었거나 게이트에 떨어진 산출물이 같은 이름으로 서빙되지 않게 한다
        if model.info.get('gate', {}).get('passed') is not True:
            raise ValueError(f"AR+GBM artifact in {self.model_dir} has not passed the deployment gate")
        self.model = model

    def forecast(self, history: pd.DataFrame) -> pd.Series:
        """history로 만들 수 있는 모든 1시간 앞 예측.

        Args:
            history: 시간 단위로 연속된 관측 (datetime 인덱스 또는 'datetime' 열, power_demand 열)

        Returns:
            예측 대상 시각을 인덱스로 하는 Series. 마지막 값은 마지막 관측 다음 시각의 예측이다.
        """
        if self.model is None:
            raise RuntimeError("AR+GBM model not loaded. Call load() first.")
        df = history.set_index('datetime') if 'datetime' in history.columns else history
        if TARGET not in df.columns:
            raise ValueError(f"Input must contain {TARGET}")
        if len(df) < MIN_HISTORY_HOURS:
            raise ValueError(f"Insufficient data: AR+GBM needs at least {MIN_HISTORY_HOURS} hourly rows, "
                             f"got {len(df)}")
        # 검증·시간대 변환·과거 값 채우기·피처 계산은 LSTM 경로와 같은 공유 정책을 쓴다
        prepared = prepare_inference_features(df, include_weather=False, selected_features=self.model.features)
        if prepared.empty:
            raise ValueError("Insufficient data: no rows with complete AR+GBM input features")
        index = _local_index(df)
        demand = pd.Series(pd.to_numeric(df[TARGET], errors='raise').to_numpy(dtype=float), index=index).ffill()
        # 공유 전처리가 바뀌어 수요 값이 어긋나면 다른 입력으로 조용히 예측하지 않는다
        if prepared.index[-1] != index[-1] or not np.allclose(demand.reindex(prepared.index), prepared[TARGET]):
            raise RuntimeError("Inference preprocessing no longer matches the raw demand history")

        nxt = pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=1)])
        self._warn_if_holidays_missing(nxt[0])
        calendar = add_time_features(pd.DataFrame(index=nxt), include_holiday=True)[self.model.calendar]
        frame = pd.concat([prepared[self.model.features], calendar]).reindex(columns=self.model.features)
        demand = pd.concat([demand, pd.Series([np.nan], index=nxt)])
        out = self.model.predict(frame, demand=demand).dropna()
        if out.empty:
            raise ValueError("Insufficient data: history does not cover the AR+GBM input lags")
        return out

    def forecast_next(self, history: pd.DataFrame) -> float:
        """마지막 관측 다음 시각의 예측값."""
        out = self.forecast(history)
        if out.index[-1] != _local_index(history)[-1] + pd.Timedelta(hours=1):
            raise ValueError("Insufficient data: history does not cover the AR+GBM input lags")
        value = float(out.iloc[-1])
        if not np.isfinite(value):
            raise ValueError("AR+GBM prediction must be finite")
        return value

    def _warn_if_holidays_missing(self, target: pd.Timestamp) -> None:
        # 휴일 표가 없는 해에는 설·부처님오신날·추석이 평일로 들어가 잔차 보정이 휴일을 놓친다
        if target.year > LUNAR_HOLIDAY_LAST_YEAR and target.year not in self._warned_years:
            self._warned_years.add(target.year)
            logger.warning(f"Lunar holidays for {target.year} are not in the holiday table "
                           f"(last year {LUNAR_HOLIDAY_LAST_YEAR}); is_holiday misses Seollal, "
                           f"Buddha's Birthday and Chuseok")
