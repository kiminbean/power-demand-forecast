"""
재생에너지 발전량 ML 예측기
============================

기상 데이터 기반 태양광/풍력 발전량 ML 예측

모델:
- 태양광: LightGBM (기상 + 시간 피처)
- 풍력: Power Curve + 기상 보정

피처:
- 시간 피처: hour, month, day_of_year, is_weekend
- 기상 피처: 기온, 습도, 전운량, 일사량 (예측시 추정)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 모델 저장 경로
MODEL_DIR = Path(__file__).parent.parent / "models" / "renewable"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_MODEL_PATH = MODEL_DIR / "solar_lgbm_model.pkl"
SOLAR_SCALER_PATH = MODEL_DIR / "solar_scaler.pkl"

# 제주 설비 용량 (2024년 기준)
JEJU_SOLAR_CAPACITY_MW = 446.0
JEJU_WIND_CAPACITY_MW = 296.0


@dataclass
class WeatherForecast:
    """기상 예보 데이터"""
    hour: int
    temperature: float  # 기온 (°C)
    humidity: float     # 습도 (%)
    cloud_cover: float  # 전운량 (0-10)
    wind_speed: float   # 풍속 (m/s)

    # 추정 일사량 (구름량 기반)
    @property
    def estimated_radiation(self) -> float:
        """구름량 기반 일사량 추정 (MJ/m²)"""
        if self.hour < 6 or self.hour > 19:
            return 0.0
        # 맑은 날 최대 일사량 (정오 기준 약 3.5 MJ/m²)
        max_radiation = 3.5 * np.sin((self.hour - 6) * np.pi / 13)
        # 구름량에 따른 감소 (전운량 10 = 100% 구름)
        cloud_factor = 1.0 - (self.cloud_cover / 10) * 0.8
        return max(0, max_radiation * cloud_factor)


@dataclass
class RenewablePrediction:
    """재생에너지 예측 결과"""
    hour: int
    solar_mw: float
    wind_mw: float
    total_mw: float
    confidence: float  # 예측 신뢰도 (0-1)
    method: str  # 예측 방법


class SolarMLPredictor:
    """태양광 발전량 ML 예측기"""

    SOLAR_DATA_PATH = Path(__file__).parent.parent / "data/raw/한국동서발전_제주_기상관측_태양광발전.csv"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'hour', 'month', 'day_of_year', 'hour_sin', 'hour_cos',
            'temperature', 'humidity', 'cloud_cover', 'radiation',
            'capacity_mw'
        ]
        self._load_or_train_model()

    def _load_or_train_model(self):
        """모델 로드 또는 학습"""
        if SOLAR_MODEL_PATH.exists() and SOLAR_SCALER_PATH.exists():
            try:
                with open(SOLAR_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SOLAR_SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded existing solar ML model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

        # 모델 학습
        self._train_model()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링"""
        df = df.copy()

        # 시간 피처
        df['datetime'] = pd.to_datetime(df['일시'], errors='raise')
        if df['datetime'].isna().any():
            raise ValueError("Solar training contains missing timestamps")
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear

        # 순환 피처 (시간의 주기성)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 기상 피처 정리
        df['temperature'] = df['기온']
        df['humidity'] = df['습도']
        df['cloud_cover'] = df['전운량(10분위)']
        df['radiation'] = df['일사량']
        df['capacity_mw'] = df['태양광 설비용량(MW)']

        # 타겟
        df['target'] = df['태양광 발전량(MWh)']

        return df

    def _train_model(self):
        """LightGBM 모델 학습"""
        logger.info("Training solar ML model...")

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import GradientBoostingRegressor

            # 데이터 로드
            if not self.SOLAR_DATA_PATH.exists():
                logger.error(f"Solar data not found: {self.SOLAR_DATA_PATH}")
                return

            df = pd.read_csv(self.SOLAR_DATA_PATH)
            df = self._prepare_features(df)

            # 시간순으로 정렬하고 동일 시각이 양쪽에 섞이지 않게 분할
            df = df.dropna(subset=['target', 'datetime']).sort_values('datetime', kind='stable')
            split_index = int(len(df) * 0.8)
            if split_index == 0 or split_index == len(df):
                raise ValueError("Solar training requires at least two dated samples")
            split_time = df['datetime'].iloc[split_index]
            if split_time == df['datetime'].iloc[0]:
                later_times = df.loc[df['datetime'] > split_time, 'datetime']
                if later_times.empty:
                    raise ValueError("Solar training requires at least two distinct timestamps")
                split_time = later_times.iloc[0]
            train = df[df['datetime'] < split_time]
            test = df[df['datetime'] >= split_time]

            X_train = train[self.feature_names].fillna(0)
            X_test = test[self.feature_names].fillna(0)
            y_train = train['target'].values
            y_test = test['target'].values

            # 학습 데이터에서만 스케일러를 학습
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # 모델 학습 (GradientBoosting - sklearn 기본 제공)
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
            self.model.fit(X_train, y_train)

            # 관측 일사량 기반 평가는 추정 일사량을 사용하는 기본 예측 성능이 아님
            logger.info(
                "Solar model evaluation uses measured irradiance (missing values filled with zero); "
                "default inference estimates irradiance, so its performance is not evaluated here"
            )
            for label, features, targets in (
                ("train", X_train, y_train), ("holdout", X_test, y_test)
            ):
                if len(targets) < 2:
                    logger.info("Solar %s R² unavailable: fewer than two samples", label)
                else:
                    logger.info("Solar %s R²: %.3f", label, self.model.score(features, targets))

            # 학습 시 생성된 난수 상태는 예측에 사용되지 않지만 pickle에
            # 포함되면 numpy 1.x와 2.x 환경 간 로드가 실패한다(MT19937 호환성).
            # 앙상블과 개별 트리 모두 난수 상태를 가지므로 저장 전에 제거해
            # 어떤 numpy 환경에서도 모델 파일을 재사용한다.
            self.model._rng = None  # type: ignore[attr-defined]
            for tree in getattr(self.model, "estimators_", []) or []:
                tree.random_state = None

            # 모델 저장
            with open(SOLAR_MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SOLAR_SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Solar model saved to {SOLAR_MODEL_PATH}")

        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            import traceback
            traceback.print_exc()

    def predict(
        self,
        hour: int,
        temperature: float,
        humidity: float,
        cloud_cover: float = 5.0,
        radiation: Optional[float] = None,
        month: Optional[int] = None,
        day_of_year: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        태양광 발전량 예측

        Returns:
            (예측 발전량 MW, 신뢰도 0-1)
        """
        if self.model is None:
            # 폴백: 단순 패턴 기반
            return self._fallback_predict(hour, cloud_cover), 0.3

        now = datetime.now()
        if month is None:
            month = now.month
        if day_of_year is None:
            day_of_year = now.timetuple().tm_yday

        # 일사량 추정 (제공되지 않은 경우)
        if radiation is None:
            if hour < 6 or hour > 19:
                radiation = 0.0
            else:
                max_rad = 3.5 * np.sin((hour - 6) * np.pi / 13)
                radiation = max_rad * (1.0 - cloud_cover / 10 * 0.8)

        # 피처 준비
        features = pd.DataFrame([{
            'hour': hour,
            'month': month,
            'day_of_year': day_of_year,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'temperature': temperature,
            'humidity': humidity,
            'cloud_cover': cloud_cover,
            'radiation': radiation,
            'capacity_mw': JEJU_SOLAR_CAPACITY_MW
        }])

        # 스케일링 및 예측
        X_scaled = self.scaler.transform(features[self.feature_names])
        prediction = self.model.predict(X_scaled)[0]

        # 음수 방지 및 용량 제한
        prediction = max(0, min(prediction, JEJU_SOLAR_CAPACITY_MW))

        # 야간 발전량 0으로 강제
        if hour < 6 or hour > 19:
            prediction = 0.0

        # 신뢰도 (낮 시간대 높음, 밤 시간대 낮음)
        if 9 <= hour <= 16:
            confidence = 0.85
        elif 6 <= hour <= 19:
            confidence = 0.7
        else:
            confidence = 0.95  # 야간은 0 예측이 확실

        return prediction, confidence

    def _fallback_predict(self, hour: int, cloud_cover: float) -> float:
        """폴백 예측 (모델 없을 때)"""
        if hour < 6 or hour > 19:
            return 0.0

        # 단순 사인 커브 + 구름량 보정
        solar_angle = np.sin((hour - 6) * np.pi / 13)
        cloud_factor = 1.0 - (cloud_cover / 10) * 0.7

        return JEJU_SOLAR_CAPACITY_MW * solar_angle * 0.25 * cloud_factor


class WindPowerPredictor:
    """풍력 발전량 예측기 (물리 기반 Power Curve)"""

    def predict(self, wind_speed: float, temperature: float = 15.0) -> Tuple[float, float]:
        """
        풍력 발전량 예측

        Power Curve 기반 + 온도 보정

        Returns:
            (예측 발전량 MW, 신뢰도 0-1)
        """
        # Power Curve 파라미터
        cut_in = 3.0    # 발전 시작 풍속
        rated = 12.0    # 정격 풍속
        cut_out = 25.0  # 발전 정지 풍속

        if wind_speed < cut_in:
            capacity_factor = 0.0
            confidence = 0.9
        elif wind_speed < rated:
            # 3차 곡선 (풍력 = 풍속³)
            capacity_factor = ((wind_speed - cut_in) / (rated - cut_in)) ** 3
            capacity_factor = min(capacity_factor, 0.85)
            confidence = 0.8
        elif wind_speed <= cut_out:
            capacity_factor = 0.85
            confidence = 0.85
        else:
            capacity_factor = 0.0  # 강풍 정지
            confidence = 0.9

        # 온도 보정 (저온에서 공기밀도 증가 → 출력 증가)
        # 기준 온도 15°C
        temp_factor = 1.0 + (15 - temperature) * 0.003  # 1°C당 0.3% 변화
        temp_factor = max(0.9, min(1.1, temp_factor))

        power_mw = JEJU_WIND_CAPACITY_MW * capacity_factor * temp_factor

        return power_mw, confidence


class RenewableMLPredictor:
    """통합 재생에너지 ML 예측기"""

    def __init__(self):
        self.solar_predictor = SolarMLPredictor()
        self.wind_predictor = WindPowerPredictor()
        logger.info("Renewable ML Predictor initialized")

    def predict_hour(
        self,
        hour: int,
        temperature: float,
        humidity: float,
        wind_speed: float,
        cloud_cover: Optional[float] = None
    ) -> RenewablePrediction:
        """
        특정 시간 재생에너지 발전량 예측
        """
        # 구름량 추정 (습도 기반)
        if cloud_cover is None:
            if humidity >= 80:
                cloud_cover = 8.0
            elif humidity >= 60:
                cloud_cover = 5.0
            else:
                cloud_cover = 2.0

        # 태양광 예측
        solar_mw, solar_conf = self.solar_predictor.predict(
            hour=hour,
            temperature=temperature,
            humidity=humidity,
            cloud_cover=cloud_cover
        )

        # 풍력 예측
        wind_mw, wind_conf = self.wind_predictor.predict(
            wind_speed=wind_speed,
            temperature=temperature
        )

        # 평균 신뢰도
        avg_confidence = (solar_conf + wind_conf) / 2

        return RenewablePrediction(
            hour=hour,
            solar_mw=round(solar_mw, 1),
            wind_mw=round(wind_mw, 1),
            total_mw=round(solar_mw + wind_mw, 1),
            confidence=round(avg_confidence, 2),
            method="ML (GradientBoosting) + Power Curve"
        )

    def predict_24h(
        self,
        current_temp: float,
        current_humidity: float,
        current_wind: float,
        temp_forecast: Optional[List[float]] = None,
        humidity_forecast: Optional[List[float]] = None,
        wind_forecast: Optional[List[float]] = None
    ) -> List[RenewablePrediction]:
        """
        24시간 재생에너지 발전량 예측

        예보 데이터가 없으면 현재 값 기반으로 일반적인 패턴 적용
        """
        predictions = []

        for hour in range(24):
            # 기상 예보 값 또는 추정값
            if temp_forecast and len(temp_forecast) > hour:
                temp = temp_forecast[hour]
            else:
                # 일교차 패턴 적용 (새벽 최저, 오후 최고)
                temp_variation = 4 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -2
                temp = current_temp + temp_variation

            if humidity_forecast and len(humidity_forecast) > hour:
                humidity = humidity_forecast[hour]
            else:
                humidity = current_humidity

            if wind_forecast and len(wind_forecast) > hour:
                wind = wind_forecast[hour]
            else:
                # 풍속 일변화 패턴 (새벽/저녁 강함, 낮 약함)
                wind_variation = 1.0 + 0.3 * np.cos((hour - 3) * np.pi / 12)
                wind = current_wind * wind_variation

            pred = self.predict_hour(
                hour=hour,
                temperature=temp,
                humidity=humidity,
                wind_speed=wind
            )
            predictions.append(pred)

        return predictions


# 싱글톤 인스턴스
_ml_predictor = None

def get_ml_predictor() -> RenewableMLPredictor:
    """ML 예측기 싱글톤 인스턴스"""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = RenewableMLPredictor()
    return _ml_predictor


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)

    predictor = RenewableMLPredictor()

    print("=" * 60)
    print("재생에너지 ML 예측 테스트")
    print("=" * 60)

    # 현재 시간 예측
    now = datetime.now()
    pred = predictor.predict_hour(
        hour=now.hour,
        temperature=17.9,
        humidity=47.0,
        wind_speed=3.0
    )

    print(f"\n현재 시간 ({now.hour}시) 예측:")
    print(f"  태양광: {pred.solar_mw} MW")
    print(f"  풍력: {pred.wind_mw} MW")
    print(f"  합계: {pred.total_mw} MW")
    print(f"  신뢰도: {pred.confidence * 100:.0f}%")
    print(f"  방법: {pred.method}")

    # 24시간 예측
    print("\n24시간 예측:")
    predictions = predictor.predict_24h(
        current_temp=17.9,
        current_humidity=47.0,
        current_wind=3.0
    )

    print("\n시간 | 태양광  | 풍력   | 합계   | 신뢰도")
    print("-" * 50)
    for p in predictions:
        print(f"{p.hour:02d}:00 | {p.solar_mw:>6.1f} | {p.wind_mw:>6.1f} | {p.total_mw:>6.1f} | {p.confidence*100:>5.0f}%")

    print("=" * 60)
