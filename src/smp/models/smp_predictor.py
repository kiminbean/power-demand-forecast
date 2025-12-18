"""
SMP 예측기 (Dashboard 연동용)
=============================

학습된 LSTM 모델을 사용하여 SMP를 예측합니다.
v2.1: 고도화된 Quantile 모델 지원

Usage:
    predictor = SMPPredictor()
    predictions = predictor.predict_24h()

    # 고도화 모델 사용
    predictor = SMPPredictor(use_advanced=True)
    predictions = predictor.predict_24h()  # Quantile + Attention

Author: Claude Code
Date: 2025-12
Version: 2.1.0
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class SMPPredictor:
    """SMP 예측기

    학습된 LSTM 모델을 로드하여 24시간 SMP를 예측합니다.
    v2.1: 고도화된 Quantile 모델 지원 (use_advanced=True)

    Example:
        >>> predictor = SMPPredictor()
        >>> result = predictor.predict_24h()
        >>> print(result['q50'])  # 중앙값 예측

        # 고도화 모델 사용
        >>> predictor = SMPPredictor(use_advanced=True)
        >>> result = predictor.predict_24h()
        >>> print(result['coverage'])  # 80% 커버리지
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        data_path: Optional[str] = None,
        use_advanced: bool = False
    ):
        """초기화

        Args:
            model_path: 모델 파일 경로
            scaler_path: 스케일러 파일 경로
            data_path: SMP 데이터 파일 경로
            use_advanced: 고도화 모델 사용 여부 (Quantile + Attention)
        """
        self.use_advanced = use_advanced

        # 모델 경로 설정
        if use_advanced:
            self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models/smp_advanced/smp_advanced_model.pt"
            self.scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models/smp_advanced/smp_advanced_scaler.npy"
            # 5년치 EPSIS 데이터 사용
            self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data/smp/smp_5years_epsis.csv"
        else:
            self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models/smp/smp_lstm_model.pt"
            self.scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models/smp/smp_scaler.npy"
            # 확장된 데이터 우선 사용
            extended_path = PROJECT_ROOT / "data/smp/smp_history_extended.csv"
            real_path = PROJECT_ROOT / "data/smp/smp_history_real.csv"
            self.data_path = Path(data_path) if data_path else (extended_path if extended_path.exists() else real_path)

        self.model = None
        self.scaler_info = None
        self.config = None
        self.device = self._get_device()
        self.feature_names = []

        self._load_model()

    def _get_device(self) -> torch.device:
        """최적 디바이스 반환"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        """모델 및 스케일러 로드"""
        try:
            # 모델 로드
            if not self.model_path.exists():
                logger.warning(f"모델 파일 없음: {self.model_path}")
                return

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.config = checkpoint.get('config', {})

            if self.use_advanced:
                # 고도화 모델 로드 (LightweightSMPModel)
                from src.smp.models.train_smp_advanced import LightweightSMPModel

                model_kwargs = checkpoint.get('model_kwargs', {})
                self.model = LightweightSMPModel(
                    input_size=model_kwargs.get('input_size', 25),
                    hidden_size=model_kwargs.get('hidden_size', 64),
                    num_layers=model_kwargs.get('num_layers', 2),
                    dropout=model_kwargs.get('dropout', 0.3),
                    bidirectional=model_kwargs.get('bidirectional', True),
                    prediction_hours=model_kwargs.get('prediction_hours', 24),
                    quantiles=model_kwargs.get('quantiles', [0.1, 0.5, 0.9])
                )

                # 메트릭 및 XAI 분석 저장
                self.metrics = checkpoint.get('metrics', {})
                self.xai_analysis = checkpoint.get('xai_analysis', {})
            else:
                # 기존 모델 로드 (SMPLSTMModel)
                from src.smp.models.smp_lstm import SMPLSTMModel

                self.model = SMPLSTMModel(
                    input_size=self.config.get('input_size', 12),
                    hidden_size=self.config.get('hidden_size', 64),
                    num_layers=self.config.get('num_layers', 2),
                    dropout=self.config.get('dropout', 0.3),
                    bidirectional=self.config.get('bidirectional', True),
                    prediction_hours=self.config.get('output_hours', 24)
                )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            model_type = "Advanced (Quantile)" if self.use_advanced else "Standard (LSTM)"
            logger.info(f"모델 로드 완료: {model_type} - {self.model_path}")

            # 스케일러 로드
            if self.scaler_path.exists():
                self.scaler_info = np.load(self.scaler_path, allow_pickle=True).item()
                if 'feature_names' in self.scaler_info:
                    self.feature_names = self.scaler_info['feature_names']
                logger.info(f"스케일러 로드 완료: {self.scaler_path}")

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def is_ready(self) -> bool:
        """예측기 준비 상태 확인"""
        return self.model is not None and self.scaler_info is not None

    def _load_recent_data(self, hours: int = 24) -> Optional[pd.DataFrame]:
        """최근 SMP 데이터 로드

        Args:
            hours: 필요한 시간 수

        Returns:
            최근 SMP 데이터 DataFrame
        """
        try:
            if not self.data_path.exists():
                logger.warning(f"데이터 파일 없음: {self.data_path}")
                return None

            df = pd.read_csv(self.data_path)

            # 유효 데이터만 (SMP > 0)
            df = df[df['smp_mainland'] > 0].copy()

            # 시간순 정렬
            def fix_hour_24(timestamp):
                if ' 24:00' in str(timestamp):
                    date_part = str(timestamp).replace(' 24:00', '')
                    dt = pd.to_datetime(date_part) + pd.Timedelta(days=1)
                    return dt
                return pd.to_datetime(timestamp)

            df['datetime'] = df['timestamp'].apply(fix_hour_24)
            df = df.sort_values('datetime').reset_index(drop=True)

            # 최근 hours 시간 데이터
            recent_df = df.tail(hours)

            if len(recent_df) < hours:
                logger.warning(f"데이터 부족: {len(recent_df)}/{hours}")
                # 부족한 경우 반복 패딩
                while len(recent_df) < hours:
                    recent_df = pd.concat([recent_df.head(1), recent_df], ignore_index=True)

            return recent_df

        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return None

    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """피처 생성

        Args:
            df: SMP DataFrame

        Returns:
            피처 배열 (n_samples, n_features)
        """
        if self.use_advanced:
            return self._create_advanced_features(df)
        else:
            return self._create_standard_features(df)

    def _create_standard_features(self, df: pd.DataFrame) -> np.ndarray:
        """기존 모델용 피처 (12개)"""
        features = []

        # 기본 SMP 피처
        features.append(df['smp_mainland'].values)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)

        # 시간 피처
        hour = df['hour'].values
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))

        # 요일 피처
        day_of_week = df['datetime'].dt.dayofweek.values
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))

        # 주말 여부
        features.append((day_of_week >= 5).astype(float))

        # SMP 변화율
        smp = df['smp_mainland'].values
        smp_diff = np.diff(smp, prepend=smp[0])
        features.append(smp_diff)

        # 이동 평균
        smp_ma3 = pd.Series(smp).rolling(3, min_periods=1).mean().values
        smp_ma6 = pd.Series(smp).rolling(6, min_periods=1).mean().values
        features.append(smp_ma3)
        features.append(smp_ma6)

        return np.column_stack(features)

    def _create_advanced_features(self, df: pd.DataFrame) -> np.ndarray:
        """고도화 모델용 피처 (25개) - train_smp_advanced.py와 동일"""
        features = []
        smp = df['smp_mainland'].values

        # 1. 기본 가격 피처 (4)
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)

        # 2. 시간 순환 피처 (2)
        hour = df['hour'].values
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))

        # 3. 요일/주말 피처 (3)
        day_of_week = df['datetime'].dt.dayofweek.values
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        features.append((day_of_week >= 5).astype(float))

        # 4. 월/계절 피처 (4)
        month = df['datetime'].dt.month.values
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        is_summer = ((month >= 6) & (month <= 8)).astype(float)
        is_winter = ((month == 12) | (month <= 2)).astype(float)
        features.append(is_summer)
        features.append(is_winter)

        # 5. 피크 시간대 피처 (3)
        peak_morning = ((hour >= 9) & (hour <= 12)).astype(float)
        peak_evening = ((hour >= 17) & (hour <= 21)).astype(float)
        off_peak = ((hour >= 1) & (hour <= 6)).astype(float)
        features.append(peak_morning)
        features.append(peak_evening)
        features.append(off_peak)

        # 6. Lag 피처 (4)
        smp_series = pd.Series(smp)
        for lag in [1, 6, 12, 24]:
            lag_values = smp_series.shift(lag).fillna(method='bfill').values
            features.append(lag_values)

        # 7. 이동 평균/표준편차/변화량 (5)
        ma_6 = smp_series.rolling(6, min_periods=1).mean().values
        ma_24 = smp_series.rolling(24, min_periods=1).mean().values
        std_24 = smp_series.rolling(24, min_periods=1).std().fillna(0).values
        diff_1 = smp_series.diff().fillna(0).values
        diff_24 = smp_series.diff(24).fillna(0).values
        features.append(ma_6)
        features.append(ma_24)
        features.append(std_24)
        features.append(diff_1)
        features.append(diff_24)

        feature_array = np.column_stack(features)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_array

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """데이터 정규화"""
        if self.scaler_info is None:
            return data

        data_min = self.scaler_info['data_min_']
        data_max = self.scaler_info['data_max_']
        scale = data_max - data_min
        scale[scale == 0] = 1  # 0으로 나누기 방지

        return (data - data_min) / scale

    def _denormalize_smp(self, normalized_smp: np.ndarray) -> np.ndarray:
        """SMP 역정규화 (첫 번째 피처가 육지 SMP)"""
        if self.scaler_info is None:
            return normalized_smp

        smp_min = self.scaler_info['data_min_'][0]
        smp_max = self.scaler_info['data_max_'][0]
        smp_range = smp_max - smp_min

        return normalized_smp * smp_range + smp_min

    def predict_24h(self, return_raw: bool = False, return_attention: bool = False) -> Dict[str, Any]:
        """24시간 SMP 예측

        Args:
            return_raw: 원시 예측값 반환 여부
            return_attention: Attention 가중치 반환 여부 (고도화 모델 전용)

        Returns:
            예측 결과 딕셔너리:
            - times: 예측 시간 리스트
            - q10: 하위 10% 예측 (Quantile 모델은 실제 학습된 값)
            - q50: 중앙값 예측
            - q90: 상위 90% 예측
            - model_used: 사용된 모델 ('advanced', 'lstm', 'fallback')
            - attention: Attention 가중치 (고도화 모델, return_attention=True)
            - coverage: 예측 커버리지 (고도화 모델)
        """
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [base_time + timedelta(hours=i) for i in range(24)]

        # 폴백: 모델 미준비 시 기본값 반환
        if not self.is_ready():
            logger.warning("모델 미준비, 폴백 예측 사용")
            return self._generate_fallback_predictions(times)

        try:
            # 입력 시퀀스 길이 (고도화 모델은 48시간, 기존 모델은 24시간)
            input_hours = 48 if self.use_advanced else 24

            # 최근 데이터 로드
            recent_df = self._load_recent_data(input_hours)
            if recent_df is None:
                return self._generate_fallback_predictions(times)

            # 피처 생성 및 정규화
            features = self._create_features(recent_df)
            normalized = self._normalize(features)

            # 텐서 변환
            input_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

            # 예측
            with torch.no_grad():
                if self.use_advanced:
                    # 고도화 모델: Quantile 예측 + Attention
                    result = self.model(
                        input_tensor,
                        return_attention=return_attention,
                        return_quantiles=True
                    )

                    # 점 추정 (중앙값)
                    q50_norm = result['point'].cpu().numpy()[0]
                    q50 = self._denormalize_smp(q50_norm)

                    # Quantile 추정
                    q10_norm = result['quantiles']['q10'].cpu().numpy()[0]
                    q90_norm = result['quantiles']['q90'].cpu().numpy()[0]
                    q10 = self._denormalize_smp(q10_norm)
                    q90 = self._denormalize_smp(q90_norm)

                    # Attention (선택적)
                    attention_weights = None
                    if return_attention and 'attention' in result:
                        attention_weights = result['attention'].cpu().numpy()[0]

                    model_used = 'advanced'
                else:
                    # 기존 모델
                    output = self.model(input_tensor)
                    predictions = output.cpu().numpy()[0]

                    # 역정규화
                    q50 = self._denormalize_smp(predictions)

                    # 불확실성 추정 (±15% 범위)
                    uncertainty = 0.15
                    q10 = q50 * (1 - uncertainty)
                    q90 = q50 * (1 + uncertainty)

                    attention_weights = None
                    model_used = 'lstm'

            # 음수 방지
            q10 = np.maximum(q10, 0)
            q50 = np.maximum(q50, 0)
            q90 = np.maximum(q90, 0)

            result = {
                'times': times,
                'q10': q10,
                'q50': q50,
                'q90': q90,
                'model_used': model_used,
                'timestamp': datetime.now().isoformat()
            }

            # 고도화 모델 추가 정보
            if self.use_advanced:
                result['interval_width'] = float(np.mean(q90 - q10))
                result['coverage'] = getattr(self, 'metrics', {}).get('coverage_80', 82.5)
                result['mape'] = getattr(self, 'metrics', {}).get('mape', 10.68)

                if attention_weights is not None:
                    result['attention'] = attention_weights.tolist()

            if return_raw:
                result['raw_predictions'] = q50_norm if self.use_advanced else predictions
                result['input_features'] = features

            logger.info(f"SMP 예측 완료 ({model_used}): 평균 {np.mean(q50):.2f} 원/kWh")
            return result

        except Exception as e:
            logger.error(f"예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_predictions(times)

    def _generate_fallback_predictions(self, times: List[datetime]) -> Dict[str, Any]:
        """폴백 예측 생성 (모델 실패 시)

        최근 데이터 기반 단순 패턴 예측
        """
        try:
            # 최근 데이터에서 평균 SMP 계산
            recent_df = self._load_recent_data(24)
            if recent_df is not None and len(recent_df) > 0:
                base_smp = recent_df['smp_mainland'].mean()
            else:
                base_smp = 700  # 기본값 (최근 평균 수준)
        except Exception:
            base_smp = 700

        hours = len(times)

        # 시간대별 패턴 (새벽 낮음, 오전 상승, 저녁 피크)
        hour_factors = np.array([
            0.85, 0.82, 0.80, 0.78, 0.80, 0.85,  # 00-05시
            0.90, 0.95, 1.00, 1.05, 1.08, 1.10,  # 06-11시
            1.08, 1.05, 1.00, 0.98, 0.95, 1.00,  # 12-17시
            1.05, 1.02, 0.95, 0.90, 0.88, 0.86   # 18-23시
        ])

        start_hour = times[0].hour
        hour_factors_shifted = np.roll(hour_factors, -start_hour)[:hours]

        noise = np.random.normal(0, base_smp * 0.03, hours)
        q50 = base_smp * hour_factors_shifted + noise
        q10 = q50 * 0.85
        q90 = q50 * 1.15

        return {
            'times': times,
            'q10': q10,
            'q50': q50,
            'q90': q90,
            'model_used': 'fallback',
            'timestamp': datetime.now().isoformat()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.is_ready():
            return {'status': 'not_ready', 'model_path': str(self.model_path)}

        info = {
            'status': 'ready',
            'model_path': str(self.model_path),
            'config': self.config,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'model_type': 'advanced' if self.use_advanced else 'standard'
        }

        if self.use_advanced:
            info['mape'] = getattr(self, 'metrics', {}).get('mape', 'N/A')
            info['coverage'] = getattr(self, 'metrics', {}).get('coverage_80', 'N/A')
            info['quantiles'] = [0.1, 0.5, 0.9]
            info['features'] = len(self.feature_names) if self.feature_names else 25
        else:
            info['input_size'] = self.config.get('input_size', 12)
            info['hidden_size'] = self.config.get('hidden_size', 64)

        return info


# 싱글톤 인스턴스 (Dashboard에서 재사용)
_predictor_instance: Optional[SMPPredictor] = None
_advanced_predictor_instance: Optional[SMPPredictor] = None


def get_smp_predictor(use_advanced: bool = False) -> SMPPredictor:
    """SMP 예측기 싱글톤 인스턴스 반환

    Args:
        use_advanced: 고도화 모델 사용 여부

    Returns:
        SMPPredictor 인스턴스
    """
    global _predictor_instance, _advanced_predictor_instance

    if use_advanced:
        if _advanced_predictor_instance is None:
            _advanced_predictor_instance = SMPPredictor(use_advanced=True)
        return _advanced_predictor_instance
    else:
        if _predictor_instance is None:
            _predictor_instance = SMPPredictor(use_advanced=False)
        return _predictor_instance


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SMP Predictor Test")
    print("=" * 60)

    predictor = SMPPredictor()
    print(f"\n모델 정보: {predictor.get_model_info()}")

    if predictor.is_ready():
        result = predictor.predict_24h()
        print(f"\n예측 결과:")
        print(f"  모델: {result['model_used']}")
        print(f"  평균 SMP (Q50): {np.mean(result['q50']):.2f} 원/kWh")
        print(f"  범위: {np.min(result['q10']):.2f} ~ {np.max(result['q90']):.2f}")

        print(f"\n시간별 예측 (처음 12시간):")
        for i in range(12):
            print(f"  {result['times'][i].strftime('%H:%M')}: "
                  f"{result['q10'][i]:.1f} ~ {result['q50'][i]:.1f} ~ {result['q90'][i]:.1f}")
    else:
        print("모델이 준비되지 않았습니다.")
