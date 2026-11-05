"""공개 API가 함께 쓰는 SMP 예측기 (SMP 예측·입찰 최적화·smp_routes가 같은 예측을 보도록).

순서대로 시도한다:
1. v4.1 (제주 하루전 SMP) — 산출물이 평가 기준(전날 같은 시간 기준선보다 나음)을 통과했을 때만
2. 기준선 — 오늘 거래시간 h = 어제 거래시간 h의 제주 하루전 가격 (2026-11-05 결정으로 v3.2 대신 사용)
3. v3.2 (성능 재검증 전) — 어제 24시간 이력도 없을 때만
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.smp.models.smp_predictor import SMPPredictor
from src.smp.models.smp_v41_predictor import JejuDAHistoryProvider, SeasonalNaivePredictor, SMPV41Predictor

logger = logging.getLogger(__name__)

V41_LABEL = "BiLSTM+Attention v4.1 (제주 하루전 SMP)"
BASELINE_LABEL = "기준선 예측 (전날 같은 시간 제주 하루전 가격)"
V32_LABEL = "BiLSTM+Attention v3.2 Optuna (성능 재검증 전)"


class ServingSMPPredictor:
    def __init__(self, primary: Optional[SMPV41Predictor], baseline: Optional[SeasonalNaivePredictor],
                 fallback: Optional[SMPPredictor]):
        self.primary = primary
        self.baseline = baseline
        self.fallback = fallback
        self.last_version: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None

    def _chain(self) -> List[Tuple[str, str, Any]]:
        return [("v4.1", V41_LABEL, self.primary), ("baseline-24h", BASELINE_LABEL, self.baseline),
                ("v3.2", V32_LABEL, self.fallback)]

    def is_ready(self) -> bool:
        return any(p is not None and p.is_ready() for _, _, p in self._chain())

    def predict_24h(self) -> Dict[str, Any]:
        """q10/q50/q90·times·forecast_date와 함께 실제로 쓴 모델(model_version·model_label)을 돌려준다."""
        reasons = []
        for version, label, predictor in self._chain():
            if predictor is None or not predictor.is_ready():
                reasons.append(f"{version} not ready")
                continue
            try:
                result = predictor.predict_24h()
            except Exception as exc:  # 이력 부족·입력 오류는 다음 예측기로 넘기되 이유를 남긴다
                reasons.append(f"{version}: {exc}")
                logger.warning("SMP %s unavailable: %s", version, exc)
                continue
            if version == "v3.2":
                # v3.2 기록 지표는 스케일러 누수로 무효이므로 내보내지 않는다.
                result = {**result, "mape": None, "coverage": None}
            self.last_version = version
            self.last_fallback_reason = "; ".join(reasons) or None
            return {**result, "model_version": version, "model_label": label,
                    "fallback_reason": self.last_fallback_reason}
        self.last_version = None
        self.last_fallback_reason = "; ".join(reasons)
        raise RuntimeError(f"no SMP model available ({self.last_fallback_reason})")

    def model_info(self) -> Dict[str, Any]:
        """현재 서빙 중인 모델 정보. 예측과 어긋나지 않도록 실제 서빙 경로를 한 번 확인한 뒤 보고한다."""
        try:
            self.predict_24h()  # 이력은 이력 제공자 안에서 1시간 캐시된다
        except Exception:
            return {"version": "none", "label": None, "mape": None, "coverage": None, "last_trained": None,
                    "fallback_reason": self.last_fallback_reason}
        base = {"version": self.last_version, "fallback_reason": self.last_fallback_reason}
        if self.last_version == "v4.1":
            meta = self.primary.meta
            return {**base, "label": V41_LABEL, "mape": self.primary.mape, "coverage": self.primary.coverage,
                    "last_trained": meta.get("trained_at"), "train_range": meta.get("train_range"),
                    "eval_end": meta.get("eval_end")}
        if self.last_version == "baseline-24h":
            return {**base, "label": BASELINE_LABEL, "mape": self.baseline.mape, "coverage": None,
                    "last_trained": None, "eval_days": self.baseline.backtest.get("days")}
        return {**base, "label": V32_LABEL, "mape": None, "coverage": None, "last_trained": None}


_lock = threading.Lock()
_instance: Optional[ServingSMPPredictor] = None


def get_serving_smp_predictor() -> ServingSMPPredictor:
    global _instance
    with _lock:
        if _instance is None:
            history = JejuDAHistoryProvider()
            primary = fallback = None
            try:
                primary = SMPV41Predictor(history=history)
            except Exception as exc:
                logger.error("v4.1 serving predictor failed to load: %s", exc)
            try:
                fallback = SMPPredictor(use_advanced=True)
            except Exception as exc:
                logger.error("v3.2 predictor failed to load: %s", exc)
            _instance = ServingSMPPredictor(primary, SeasonalNaivePredictor(history=history), fallback)
        return _instance
