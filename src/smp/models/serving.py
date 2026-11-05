"""공개 API가 함께 쓰는 SMP 예측기 (SMP 예측·입찰 최적화·smp_routes가 같은 예측을 보도록).

v4.1(제주 하루전 SMP, walk-forward 평가 통과 산출물)을 우선 쓰고, 산출물이 없거나 평가 기준을
통과하지 못했거나 오늘 0시까지 연속 이력이 부족하면 v3.2(성능 재검증 전)로 대체한다.
"""

import logging
import threading
from typing import Any, Dict, Optional

from src.smp.models.smp_predictor import SMPPredictor
from src.smp.models.smp_v41_predictor import SMPV41Predictor

logger = logging.getLogger(__name__)

V41_LABEL = "BiLSTM+Attention v4.1 (제주 하루전 SMP)"
V32_LABEL = "BiLSTM+Attention v3.2 Optuna (성능 재검증 전)"


class ServingSMPPredictor:
    def __init__(self, primary: Optional[SMPV41Predictor], fallback: Optional[SMPPredictor]):
        self.primary = primary
        self.fallback = fallback
        self.last_fallback_reason: Optional[str] = None

    def is_ready(self) -> bool:
        return bool((self.primary and self.primary.is_ready()) or (self.fallback and self.fallback.is_ready()))

    def predict_24h(self) -> Dict[str, Any]:
        """q10/q50/q90·times·forecast_date와 함께 실제로 쓴 모델(model_version·model_label)을 돌려준다."""
        reason = "v4.1 artifact missing or not passing its walk-forward gate"
        if self.primary is not None and self.primary.is_ready():
            try:
                result = self.primary.predict_24h()
                self.last_fallback_reason = None
                return {**result, "model_version": "v4.1", "model_label": V41_LABEL}
            except Exception as exc:  # 이력 부족·입력 오류 모두 v3.2로 대체하되 이유를 남긴다
                reason = f"v4.1 unavailable: {exc}"
                logger.warning(reason)
        self.last_fallback_reason = reason
        if self.fallback is None or not self.fallback.is_ready():
            raise RuntimeError(f"no SMP model available ({reason})")
        result = self.fallback.predict_24h()
        # v3.2 기록 지표는 스케일러 누수로 무효이므로 내보내지 않는다.
        result = {**result, "mape": None, "coverage": None}
        return {**result, "model_version": "v3.2", "model_label": V32_LABEL, "fallback_reason": reason}

    def model_info(self) -> Dict[str, Any]:
        """현재 서빙 중인 모델 정보. 예측과 어긋나지 않도록 실제 서빙 경로를 한 번 확인한 뒤 보고한다."""
        try:
            self.predict_24h()  # 이력은 추론기 안에서 1시간 캐시된다
        except Exception as exc:
            self.last_fallback_reason = self.last_fallback_reason or str(exc)
        if self.primary is not None and self.primary.is_ready() and self.last_fallback_reason is None:
            meta = self.primary.meta
            return {
                "version": "v4.1", "label": V41_LABEL, "mape": self.primary.mape, "coverage": self.primary.coverage,
                "last_trained": meta.get("trained_at"), "train_range": meta.get("train_range"),
                "eval_end": meta.get("eval_end"),
            }
        return {"version": "v3.2", "label": V32_LABEL, "mape": None, "coverage": None, "last_trained": None,
                "fallback_reason": self.last_fallback_reason}


_lock = threading.Lock()
_instance: Optional[ServingSMPPredictor] = None


def get_serving_smp_predictor() -> ServingSMPPredictor:
    global _instance
    with _lock:
        if _instance is None:
            primary = fallback = None
            try:
                primary = SMPV41Predictor()
            except Exception as exc:
                logger.error("v4.1 serving predictor failed to load: %s", exc)
            try:
                fallback = SMPPredictor(use_advanced=True)
            except Exception as exc:
                logger.error("v3.2 predictor failed to load: %s", exc)
            _instance = ServingSMPPredictor(primary, fallback)
        return _instance
