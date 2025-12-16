"""
이상 탐지 시스템 (Task 22)
===========================

전력 수요 데이터의 비정상 패턴 탐지

주요 기능:
1. 통계 기반 이상 탐지 (Z-score, IQR)
2. Isolation Forest
3. Autoencoder 기반 탐지
4. 실시간 이상치 알림

Author: Claude Code
Date: 2025-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """이상 유형"""
    SPIKE = 'spike'           # 급격한 상승
    DROP = 'drop'             # 급격한 하락
    DRIFT = 'drift'           # 점진적 변화
    PATTERN = 'pattern'       # 패턴 이상
    CONTEXTUAL = 'contextual' # 맥락적 이상 (시간대 고려)


class SeverityLevel(Enum):
    """심각도 수준"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


@dataclass
class Anomaly:
    """탐지된 이상"""
    timestamp: datetime
    value: float
    anomaly_type: AnomalyType
    severity: SeverityLevel
    score: float  # 이상 점수 (높을수록 이상)
    expected_range: Tuple[float, float] = (0.0, 0.0)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'score': self.score,
            'expected_range': self.expected_range,
            'context': self.context
        }


@dataclass
class AnomalyDetectionResult:
    """이상 탐지 결과"""
    method: str
    anomalies: List[Anomaly]
    total_points: int
    anomaly_rate: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'anomalies': [a.to_dict() for a in self.anomalies],
            'total_points': self.total_points,
            'anomaly_rate': self.anomaly_rate,
            'threshold': self.threshold,
            'metadata': self.metadata
        }


# ============================================================================
# 통계 기반 이상 탐지
# ============================================================================

class ZScoreDetector:
    """
    Z-score 기반 이상 탐지

    표준편차를 사용하여 이상치를 탐지합니다.

    Args:
        threshold: Z-score 임계값 (기본 3.0)
        window_size: 이동 평균/표준편차 윈도우 크기

    Example:
        >>> detector = ZScoreDetector(threshold=3.0)
        >>> result = detector.detect(data, timestamps)
    """

    def __init__(self, threshold: float = 3.0, window_size: Optional[int] = None):
        self.threshold = threshold
        self.window_size = window_size

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> AnomalyDetectionResult:
        """이상 탐지 수행"""
        if self.window_size:
            # 이동 통계 사용
            mean = pd.Series(data).rolling(window=self.window_size, min_periods=1).mean()
            std = pd.Series(data).rolling(window=self.window_size, min_periods=1).std()
            std = std.replace(0, 1e-10)  # 0으로 나누기 방지
        else:
            mean = np.mean(data)
            std = np.std(data) if np.std(data) > 0 else 1e-10

        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > self.threshold

        anomalies = []
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                ts = timestamps[i] if timestamps else datetime.now()
                value = float(data[i])
                score = float(z_scores[i])

                # 유형 판별
                if isinstance(mean, pd.Series):
                    m = mean.iloc[i]
                else:
                    m = mean

                anomaly_type = AnomalyType.SPIKE if value > m else AnomalyType.DROP
                severity = self._get_severity(score)

                anomalies.append(Anomaly(
                    timestamp=ts,
                    value=value,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    score=score,
                    expected_range=(float(m - self.threshold * std), float(m + self.threshold * std)) if not isinstance(mean, pd.Series) else (0, 0)
                ))

        return AnomalyDetectionResult(
            method='zscore',
            anomalies=anomalies,
            total_points=len(data),
            anomaly_rate=len(anomalies) / len(data) * 100 if len(data) > 0 else 0,
            threshold=self.threshold,
            metadata={'window_size': self.window_size}
        )

    def _get_severity(self, score: float) -> SeverityLevel:
        """점수에 따른 심각도"""
        if score > 5:
            return SeverityLevel.CRITICAL
        elif score > 4:
            return SeverityLevel.HIGH
        elif score > 3.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


class IQRDetector:
    """
    IQR(사분위수 범위) 기반 이상 탐지

    Args:
        multiplier: IQR 배수 (기본 1.5)

    Example:
        >>> detector = IQRDetector(multiplier=1.5)
        >>> result = detector.detect(data)
    """

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> AnomalyDetectionResult:
        """이상 탐지 수행"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        anomaly_mask = (data < lower_bound) | (data > upper_bound)

        anomalies = []
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                ts = timestamps[i] if timestamps else datetime.now()
                value = float(data[i])

                # 거리 기반 점수
                if value < lower_bound:
                    score = (lower_bound - value) / iqr
                    anomaly_type = AnomalyType.DROP
                else:
                    score = (value - upper_bound) / iqr
                    anomaly_type = AnomalyType.SPIKE

                anomalies.append(Anomaly(
                    timestamp=ts,
                    value=value,
                    anomaly_type=anomaly_type,
                    severity=self._get_severity(score),
                    score=float(score),
                    expected_range=(lower_bound, upper_bound)
                ))

        return AnomalyDetectionResult(
            method='iqr',
            anomalies=anomalies,
            total_points=len(data),
            anomaly_rate=len(anomalies) / len(data) * 100 if len(data) > 0 else 0,
            threshold=self.multiplier,
            metadata={'q1': q1, 'q3': q3, 'iqr': iqr, 'lower': lower_bound, 'upper': upper_bound}
        )

    def _get_severity(self, score: float) -> SeverityLevel:
        if score > 3:
            return SeverityLevel.CRITICAL
        elif score > 2:
            return SeverityLevel.HIGH
        elif score > 1.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


# ============================================================================
# Isolation Forest 기반 이상 탐지
# ============================================================================

class IsolationForestDetector:
    """
    Isolation Forest 기반 이상 탐지

    Args:
        contamination: 예상 이상치 비율 (0.0 ~ 0.5)
        n_estimators: 트리 수
        random_state: 랜덤 시드

    Example:
        >>> detector = IsolationForestDetector(contamination=0.05)
        >>> detector.fit(train_data)
        >>> result = detector.detect(test_data)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None

    def fit(self, data: np.ndarray) -> 'IsolationForestDetector':
        """모델 학습"""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        # 1D 데이터 처리
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self._model.fit(data)
        return self

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> AnomalyDetectionResult:
        """이상 탐지 수행"""
        if self._model is None:
            self.fit(data)

        # 1D 데이터 처리
        original_data = data
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        predictions = self._model.predict(data)
        scores = -self._model.score_samples(data)  # 높을수록 이상

        anomaly_mask = predictions == -1

        anomalies = []
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                ts = timestamps[i] if timestamps else datetime.now()
                value = float(original_data[i]) if original_data.ndim == 1 else float(original_data[i, 0])

                anomalies.append(Anomaly(
                    timestamp=ts,
                    value=value,
                    anomaly_type=AnomalyType.PATTERN,
                    severity=self._get_severity(scores[i]),
                    score=float(scores[i])
                ))

        return AnomalyDetectionResult(
            method='isolation_forest',
            anomalies=anomalies,
            total_points=len(original_data),
            anomaly_rate=len(anomalies) / len(original_data) * 100 if len(original_data) > 0 else 0,
            threshold=self.contamination,
            metadata={'n_estimators': self.n_estimators}
        )

    def _get_severity(self, score: float) -> SeverityLevel:
        if score > 0.7:
            return SeverityLevel.CRITICAL
        elif score > 0.6:
            return SeverityLevel.HIGH
        elif score > 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


# ============================================================================
# Autoencoder 기반 이상 탐지
# ============================================================================

class AutoencoderModel(nn.Module):
    """Autoencoder 모델"""

    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [32, 16, 8]

        # Encoder
        encoder_layers = []
        prev_size = input_size
        for size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # 마지막 Dropout 제거

        # Decoder
        decoder_layers = []
        for size in reversed(hidden_sizes[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AutoencoderDetector:
    """
    Autoencoder 기반 이상 탐지

    재구성 오류가 큰 데이터를 이상치로 판단합니다.

    Args:
        input_size: 입력 크기
        hidden_sizes: 은닉층 크기 리스트
        threshold_percentile: 이상치 판정 백분위수 (기본 95)
        epochs: 학습 에폭
        device: 디바이스

    Example:
        >>> detector = AutoencoderDetector(input_size=24)
        >>> detector.fit(train_data)
        >>> result = detector.detect(test_data)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        threshold_percentile: float = 95,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device

        self._model = AutoencoderModel(input_size, hidden_sizes).to(device)
        self._threshold = None

    def fit(self, data: np.ndarray) -> 'AutoencoderDetector':
        """모델 학습"""
        # 데이터 준비
        if data.ndim == 1:
            # 시퀀스로 변환
            data = self._create_sequences(data, self.input_size)

        X = torch.tensor(data, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed, _ = self._model(X)
            loss = criterion(reconstructed, X)
            loss.backward()
            optimizer.step()

        # 임계값 계산
        self._model.eval()
        with torch.no_grad():
            reconstructed, _ = self._model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()
            self._threshold = np.percentile(errors, self.threshold_percentile)

        return self

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> AnomalyDetectionResult:
        """이상 탐지 수행"""
        if self._threshold is None:
            self.fit(data)

        # 데이터 준비
        original_len = len(data)
        if data.ndim == 1:
            data = self._create_sequences(data, self.input_size)

        X = torch.tensor(data, dtype=torch.float32, device=self.device)

        self._model.eval()
        with torch.no_grad():
            reconstructed, _ = self._model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()

        anomaly_mask = errors > self._threshold

        anomalies = []
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly:
                ts = timestamps[i] if timestamps and i < len(timestamps) else datetime.now()

                anomalies.append(Anomaly(
                    timestamp=ts,
                    value=float(errors[i]),
                    anomaly_type=AnomalyType.PATTERN,
                    severity=self._get_severity(errors[i]),
                    score=float(errors[i] / self._threshold)
                ))

        return AnomalyDetectionResult(
            method='autoencoder',
            anomalies=anomalies,
            total_points=len(errors),
            anomaly_rate=len(anomalies) / len(errors) * 100 if len(errors) > 0 else 0,
            threshold=float(self._threshold),
            metadata={'epochs': self.epochs, 'threshold_percentile': self.threshold_percentile}
        )

    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """시퀀스 생성"""
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    def _get_severity(self, error: float) -> SeverityLevel:
        ratio = error / self._threshold if self._threshold else 0
        if ratio > 3:
            return SeverityLevel.CRITICAL
        elif ratio > 2:
            return SeverityLevel.HIGH
        elif ratio > 1.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


# ============================================================================
# 실시간 이상 탐지
# ============================================================================

class RealtimeAnomalyDetector:
    """
    실시간 이상 탐지기

    스트리밍 데이터에 대한 온라인 이상 탐지를 수행합니다.

    Args:
        window_size: 통계 계산 윈도우 크기
        z_threshold: Z-score 임계값
        ema_alpha: 지수 이동 평균 알파

    Example:
        >>> detector = RealtimeAnomalyDetector(window_size=100)
        >>> for value in stream:
        ...     anomaly = detector.update(value, timestamp)
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        ema_alpha: float = 0.1
    ):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.ema_alpha = ema_alpha

        self._history = deque(maxlen=window_size)
        self._ema = None
        self._ema_var = None

    def update(
        self,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[Anomaly]:
        """
        새 값으로 업데이트하고 이상 여부 확인

        Returns:
            Anomaly if detected, None otherwise
        """
        timestamp = timestamp or datetime.now()

        if self._ema is None:
            self._ema = value
            self._ema_var = 0.0
        else:
            # 지수 이동 평균 업데이트
            diff = value - self._ema
            self._ema += self.ema_alpha * diff
            self._ema_var = (1 - self.ema_alpha) * (self._ema_var + self.ema_alpha * diff ** 2)

        self._history.append(value)

        # 충분한 데이터가 있을 때만 탐지
        if len(self._history) < 10:
            return None

        # Z-score 계산
        std = np.sqrt(self._ema_var) if self._ema_var > 0 else 1e-10
        z_score = abs(value - self._ema) / std

        if z_score > self.z_threshold:
            anomaly_type = AnomalyType.SPIKE if value > self._ema else AnomalyType.DROP
            return Anomaly(
                timestamp=timestamp,
                value=value,
                anomaly_type=anomaly_type,
                severity=self._get_severity(z_score),
                score=z_score,
                expected_range=(self._ema - self.z_threshold * std, self._ema + self.z_threshold * std)
            )

        return None

    def _get_severity(self, score: float) -> SeverityLevel:
        if score > 5:
            return SeverityLevel.CRITICAL
        elif score > 4:
            return SeverityLevel.HIGH
        elif score > 3.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def get_stats(self) -> Dict[str, float]:
        """현재 통계 반환"""
        return {
            'ema': self._ema,
            'ema_std': np.sqrt(self._ema_var) if self._ema_var else 0,
            'window_mean': np.mean(list(self._history)) if self._history else 0,
            'window_std': np.std(list(self._history)) if len(self._history) > 1 else 0
        }


# ============================================================================
# 앙상블 이상 탐지
# ============================================================================

class EnsembleAnomalyDetector:
    """
    앙상블 이상 탐지기

    여러 탐지기의 결과를 결합합니다.

    Args:
        detectors: 탐지기 리스트
        voting: 'any', 'majority', 'all'

    Example:
        >>> ensemble = EnsembleAnomalyDetector([
        ...     ZScoreDetector(),
        ...     IQRDetector()
        ... ])
        >>> result = ensemble.detect(data)
    """

    def __init__(
        self,
        detectors: List[Union[ZScoreDetector, IQRDetector, IsolationForestDetector]] = None,
        voting: str = 'majority'
    ):
        self.detectors = detectors or [ZScoreDetector(), IQRDetector()]
        self.voting = voting

    def detect(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> AnomalyDetectionResult:
        """이상 탐지 수행"""
        # 각 탐지기 실행
        all_results = []
        for detector in self.detectors:
            result = detector.detect(data, timestamps)
            all_results.append(result)

        # 인덱스별 이상 투표
        anomaly_indices = {}
        for result in all_results:
            for anomaly in result.anomalies:
                idx = timestamps.index(anomaly.timestamp) if timestamps else 0
                if idx not in anomaly_indices:
                    anomaly_indices[idx] = []
                anomaly_indices[idx].append(anomaly)

        # 투표 기반 결합
        final_anomalies = []
        n_detectors = len(self.detectors)

        for idx, detected in anomaly_indices.items():
            n_votes = len(detected)

            if self.voting == 'any' and n_votes >= 1:
                pass  # 하나라도 감지하면 이상
            elif self.voting == 'majority' and n_votes < (n_detectors / 2):
                continue
            elif self.voting == 'all' and n_votes < n_detectors:
                continue

            # 가장 높은 점수의 이상 선택
            best_anomaly = max(detected, key=lambda a: a.score)

            # 메타데이터에 투표 정보 추가
            best_anomaly.context['votes'] = n_votes
            best_anomaly.context['detectors'] = [d.method for d in all_results if any(
                a.timestamp == best_anomaly.timestamp for a in d.anomalies
            )]

            final_anomalies.append(best_anomaly)

        return AnomalyDetectionResult(
            method=f'ensemble_{self.voting}',
            anomalies=final_anomalies,
            total_points=len(data),
            anomaly_rate=len(final_anomalies) / len(data) * 100 if len(data) > 0 else 0,
            threshold=0.0,
            metadata={
                'voting': self.voting,
                'n_detectors': n_detectors,
                'individual_rates': [r.anomaly_rate for r in all_results]
            }
        )


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_anomaly_detector(
    method: str = 'zscore',
    **kwargs
) -> Union[ZScoreDetector, IQRDetector, IsolationForestDetector, AutoencoderDetector]:
    """
    이상 탐지기 팩토리 함수

    Args:
        method: 'zscore', 'iqr', 'isolation_forest', 'autoencoder', 'ensemble'

    Returns:
        해당 탐지기 인스턴스
    """
    detectors = {
        'zscore': ZScoreDetector,
        'iqr': IQRDetector,
        'isolation_forest': IsolationForestDetector,
        'autoencoder': AutoencoderDetector,
        'ensemble': EnsembleAnomalyDetector
    }

    detector_class = detectors.get(method)
    if detector_class is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(detectors.keys())}")

    return detector_class(**kwargs)


def detect_anomalies(
    data: np.ndarray,
    method: str = 'zscore',
    timestamps: Optional[List[datetime]] = None,
    **kwargs
) -> AnomalyDetectionResult:
    """
    간편한 이상 탐지 함수

    Args:
        data: 시계열 데이터
        method: 탐지 방법
        timestamps: 타임스탬프 리스트

    Returns:
        AnomalyDetectionResult
    """
    detector = create_anomaly_detector(method, **kwargs)
    return detector.detect(data, timestamps)
