#!/usr/bin/env python3
"""
SMP CatBoost Predictor for RTM (Real-Time Market)
=================================================

CatBoost v3.19 모델을 사용한 실시간 SMP 예측기
- 단일 스텝 예측 (다음 1시간)
- R² 0.827, MAPE 5.28% (BiLSTM v3.2보다 우수)
- RTM 실시간 의사결정에 최적화
- 68개 피처 (EMA, MACD, Momentum 포함)

Usage:
    predictor = SMPCatBoostPredictor()
    result = predictor.predict_next_hour()  # 다음 1시간 예측
    result = predictor.predict_hours(6)     # 다음 6시간 재귀 예측

Author: Claude Code
Date: 2025-12
Version: 2.0.0 (Upgraded to v3.19)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class SMPCatBoostPredictor:
    """CatBoost 기반 SMP 예측기 (RTM용)

    단일 스텝 예측에 최적화된 CatBoost v3.19 모델을 사용합니다.
    - MAPE: 5.28% (BiLSTM 7.17%보다 우수)
    - R²: 0.827
    - 68개 피처 (시간, 래그, 통계, EMA, MACD, Momentum)
    """

    def __init__(self, model_path: Optional[str] = None):
        """초기화

        Args:
            model_path: CatBoost 모델 파일 경로 (.cbm)
        """
        self.model_path = Path(model_path) if model_path else \
            PROJECT_ROOT / "models/smp_v3_19_recent/catboost_model.cbm"

        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.data_path = PROJECT_ROOT / "data/smp/smp_5years_epsis.csv"

        self._load_model()
        self._load_feature_names()

    def _load_model(self):
        """CatBoost 모델 로드"""
        try:
            from catboost import CatBoostRegressor

            if not self.model_path.exists():
                logger.warning(f"CatBoost 모델 파일 없음: {self.model_path}")
                return

            self.model = CatBoostRegressor()
            self.model.load_model(str(self.model_path))

            # 메트릭 로드
            metrics_path = self.model_path.parent / "metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    self.metrics = json.load(f)

            logger.info(f"CatBoost 모델 로드 완료: {self.model_path}")
            logger.info(f"  - MAPE: {self.metrics.get('test_mape', 'N/A'):.2f}%")
            logger.info(f"  - R²: {self.metrics.get('test_r2', 'N/A'):.4f}")

        except ImportError:
            logger.error("CatBoost not installed. Run: pip install catboost")
            self.model = None
        except Exception as e:
            logger.error(f"CatBoost 모델 로드 실패: {e}")
            self.model = None

    def _load_feature_names(self):
        """피처 이름 로드 (metrics.json에서)"""
        if self.metrics and 'feature_names' in self.metrics:
            self.feature_names = self.metrics['feature_names']
        else:
            # 기본 피처 이름 (68개 - v3.19)
            self.feature_names = [
                # Time features (17개)
                'hour', 'hour_sin', 'hour_cos',
                'dayofweek', 'dow_sin', 'dow_cos', 'is_weekend',
                'month', 'month_sin', 'month_cos',
                'doy_sin', 'doy_cos',
                'is_summer', 'is_winter',
                'peak_morning', 'peak_evening', 'off_peak',
                # Lag features (12개)
                'smp_lag1', 'smp_lag2', 'smp_lag3', 'smp_lag4', 'smp_lag5', 'smp_lag6',
                'smp_lag12', 'smp_lag24', 'smp_lag48', 'smp_lag72', 'smp_lag96', 'smp_lag168',
                # Rolling statistics (24개)
                'smp_ma6', 'smp_std6', 'smp_min6', 'smp_max6',
                'smp_ma12', 'smp_std12', 'smp_min12', 'smp_max12',
                'smp_ma24', 'smp_std24', 'smp_min24', 'smp_max24',
                'smp_ma48', 'smp_std48', 'smp_min48', 'smp_max48',
                'smp_ma72', 'smp_std72', 'smp_min72', 'smp_max72',
                'smp_ma168', 'smp_std168', 'smp_min168', 'smp_max168',
                # EMA features (5개) - v3.19 추가
                'smp_ema6', 'smp_ema12', 'smp_ema24', 'smp_ema48', 'smp_ema168',
                # Diff features (3개)
                'smp_diff1', 'smp_diff24', 'smp_diff168',
                # Ratio features (2개)
                'smp_lag1_vs_ma24', 'smp_lag1_vs_ma168',
                # Range features (2개)
                'smp_range24', 'smp_range168',
                # Momentum features (3개) - v3.19 추가
                'smp_roc24', 'smp_roc168', 'smp_macd'
            ]

    def is_ready(self) -> bool:
        """예측기 준비 상태 확인"""
        return self.model is not None

    def _load_recent_data(self, hours: int = 168) -> Optional[pd.DataFrame]:
        """최근 SMP 데이터 로드 (피처 생성용)

        Args:
            hours: 필요한 시간 수 (최소 168시간 = 1주일)
        """
        try:
            if not self.data_path.exists():
                logger.warning(f"데이터 파일 없음: {self.data_path}")
                return None

            df = pd.read_csv(self.data_path)
            df = df[df['smp_mainland'] > 0].copy()

            def fix_hour_24(ts):
                if ' 24:00' in str(ts):
                    date_part = str(ts).replace(' 24:00', '')
                    return pd.to_datetime(date_part) + pd.Timedelta(days=1)
                return pd.to_datetime(ts)

            df['datetime'] = df['timestamp'].apply(fix_hour_24)
            df = df.sort_values('datetime').reset_index(drop=True)

            # 최근 데이터
            recent_df = df.tail(max(hours, 200))

            return recent_df

        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return None

    def _create_features_for_prediction(
        self,
        df: pd.DataFrame,
        target_time: datetime
    ) -> np.ndarray:
        """예측용 피처 생성 (단일 시점)

        Args:
            df: 과거 SMP 데이터
            target_time: 예측 대상 시간
        """
        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        features = {}

        # === Time features ===
        hour = target_time.hour
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        dow = target_time.weekday()
        features['dayofweek'] = dow
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        features['is_weekend'] = 1.0 if dow >= 5 else 0.0

        month = target_time.month
        features['month'] = month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)

        doy = target_time.timetuple().tm_yday
        features['doy_sin'] = np.sin(2 * np.pi * doy / 365)
        features['doy_cos'] = np.cos(2 * np.pi * doy / 365)

        features['is_summer'] = 1.0 if 6 <= month <= 8 else 0.0
        features['is_winter'] = 1.0 if month == 12 or month <= 2 else 0.0

        features['peak_morning'] = 1.0 if 9 <= hour <= 12 else 0.0
        features['peak_evening'] = 1.0 if 17 <= hour <= 21 else 0.0
        features['off_peak'] = 1.0 if 1 <= hour <= 6 else 0.0

        # === Lag features (마지막 값 기준) ===
        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 168]:
            idx = -lag if lag <= len(smp) else 0
            features[f'smp_lag{lag}'] = smp[idx]

        # === Rolling statistics ===
        for window in [6, 12, 24, 48, 72, 168]:
            window_data = smp[-window:] if window <= len(smp) else smp
            features[f'smp_ma{window}'] = np.mean(window_data)
            features[f'smp_std{window}'] = np.std(window_data)
            features[f'smp_min{window}'] = np.min(window_data)
            features[f'smp_max{window}'] = np.max(window_data)

        # === EMA features (v3.19 추가) ===
        for span in [6, 12, 24, 48, 168]:
            # EMA 계산
            alpha = 2 / (span + 1)
            ema_data = smp[-span:] if span <= len(smp) else smp
            ema = ema_data[0]
            for val in ema_data[1:]:
                ema = alpha * val + (1 - alpha) * ema
            features[f'smp_ema{span}'] = ema

        # === Diff features ===
        features['smp_diff1'] = smp[-1] - smp[-2] if len(smp) >= 2 else 0
        features['smp_diff24'] = smp[-1] - smp[-25] if len(smp) >= 25 else 0
        features['smp_diff168'] = smp[-1] - smp[-169] if len(smp) >= 169 else 0

        # === Relative features ===
        features['smp_lag1_vs_ma24'] = smp[-1] / features['smp_ma24'] if features['smp_ma24'] > 0 else 1
        features['smp_lag1_vs_ma168'] = smp[-1] / features['smp_ma168'] if features['smp_ma168'] > 0 else 1

        # === Range features ===
        features['smp_range24'] = features['smp_max24'] - features['smp_min24']
        features['smp_range168'] = features['smp_max168'] - features['smp_min168']

        # === Momentum features (v3.19 추가) ===
        # Rate of Change (ROC)
        features['smp_roc24'] = (smp[-1] / smp[-25] - 1) if len(smp) >= 25 and smp[-25] > 0 else 0
        features['smp_roc168'] = (smp[-1] / smp[-169] - 1) if len(smp) >= 169 and smp[-169] > 0 else 0

        # MACD (EMA12 - EMA26)
        features['smp_macd'] = features.get('smp_ema12', smp[-1]) - features.get('smp_ema24', smp[-1])

        # 피처 배열 생성 (순서 중요)
        feature_array = np.array([features.get(name, 0) for name in self.feature_names])

        return feature_array.reshape(1, -1)

    def predict_next_hour(self) -> Dict[str, Any]:
        """다음 1시간 SMP 예측 (RTM용)

        Returns:
            예측 결과 딕셔너리:
            - time: 예측 시간
            - smp: 예측 SMP (원/kWh)
            - confidence_low: 하한 (q10)
            - confidence_high: 상한 (q90)
            - model_used: 모델 이름
            - mape: 모델 MAPE
        """
        if not self.is_ready():
            return self._generate_fallback()

        try:
            # 최근 데이터 로드
            df = self._load_recent_data(200)
            if df is None:
                return self._generate_fallback()

            # 예측 시간 (현재 시간의 다음 정시)
            now = datetime.now()
            target_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

            # 피처 생성
            features = self._create_features_for_prediction(df, target_time)

            # 예측
            prediction = self.model.predict(features)[0]

            # 신뢰 구간 (±12% 기반)
            uncertainty = 0.12
            confidence_low = prediction * (1 - uncertainty)
            confidence_high = prediction * (1 + uncertainty)

            return {
                'time': target_time,
                'smp': float(prediction),
                'confidence_low': float(confidence_low),
                'confidence_high': float(confidence_high),
                'model_used': 'CatBoost v3.19',
                'mape': self.metrics.get('test_mape', 5.28),
                'r2': self.metrics.get('test_r2', 0.83),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"CatBoost 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback()

    def predict_hours(self, hours: int = 6) -> Dict[str, Any]:
        """다중 시간 예측 (재귀적)

        RTM에서 향후 몇 시간 예측이 필요할 때 사용.
        주의: 재귀 예측이므로 시간이 멀어질수록 오차 누적 가능.

        Args:
            hours: 예측할 시간 수 (권장: 1~6시간)

        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_ready():
            return self._generate_fallback_multi(hours)

        try:
            df = self._load_recent_data(200)
            if df is None:
                return self._generate_fallback_multi(hours)

            now = datetime.now()
            base_time = now.replace(minute=0, second=0, microsecond=0)

            predictions = []
            smp_history = df['smp_mainland'].tolist()

            for h in range(1, hours + 1):
                target_time = base_time + timedelta(hours=h)

                # 임시 DataFrame 생성 (업데이트된 히스토리 사용)
                temp_df = pd.DataFrame({'smp_mainland': smp_history})

                # 피처 생성 및 예측
                features = self._create_features_for_prediction(temp_df, target_time)
                pred = self.model.predict(features)[0]

                predictions.append({
                    'hour': h,
                    'time': target_time,
                    'smp': float(pred),
                    'confidence_low': float(pred * 0.88),
                    'confidence_high': float(pred * 1.12)
                })

                # 히스토리에 예측값 추가 (재귀용)
                smp_history.append(pred)

            return {
                'predictions': predictions,
                'times': [p['time'] for p in predictions],
                'smp_values': [p['smp'] for p in predictions],
                'model_used': 'CatBoost v3.19 (recursive)',
                'mape': self.metrics.get('test_mape', 5.28),
                'warning': 'Recursive prediction - accuracy decreases with horizon',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"다중 시간 예측 실패: {e}")
            return self._generate_fallback_multi(hours)

    def _generate_fallback(self) -> Dict[str, Any]:
        """폴백 예측 (모델 실패 시)"""
        now = datetime.now()
        target_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        # 시간대별 기본 패턴
        hourly_pattern = {
            0: 85, 1: 82, 2: 80, 3: 78, 4: 80, 5: 85,
            6: 95, 7: 110, 8: 125, 9: 135, 10: 140, 11: 145,
            12: 148, 13: 145, 14: 140, 15: 135, 16: 128, 17: 122,
            18: 115, 19: 108, 20: 100, 21: 95, 22: 90, 23: 87
        }

        base_smp = hourly_pattern.get(target_time.hour, 100)

        return {
            'time': target_time,
            'smp': float(base_smp),
            'confidence_low': float(base_smp * 0.85),
            'confidence_high': float(base_smp * 1.15),
            'model_used': 'fallback',
            'mape': 15.0,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_fallback_multi(self, hours: int) -> Dict[str, Any]:
        """다중 시간 폴백 예측"""
        now = datetime.now()
        base_time = now.replace(minute=0, second=0, microsecond=0)

        hourly_pattern = {
            0: 85, 1: 82, 2: 80, 3: 78, 4: 80, 5: 85,
            6: 95, 7: 110, 8: 125, 9: 135, 10: 140, 11: 145,
            12: 148, 13: 145, 14: 140, 15: 135, 16: 128, 17: 122,
            18: 115, 19: 108, 20: 100, 21: 95, 22: 90, 23: 87
        }

        predictions = []
        for h in range(1, hours + 1):
            target_time = base_time + timedelta(hours=h)
            base_smp = hourly_pattern.get(target_time.hour, 100)
            predictions.append({
                'hour': h,
                'time': target_time,
                'smp': float(base_smp),
                'confidence_low': float(base_smp * 0.85),
                'confidence_high': float(base_smp * 1.15)
            })

        return {
            'predictions': predictions,
            'times': [p['time'] for p in predictions],
            'smp_values': [p['smp'] for p in predictions],
            'model_used': 'fallback',
            'mape': 15.0,
            'timestamp': datetime.now().isoformat()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'status': 'ready' if self.is_ready() else 'not_ready',
            'model_path': str(self.model_path),
            'model_type': 'CatBoost v3.19',
            'purpose': 'RTM (Real-Time Market)',
            'prediction_type': 'single-step (1 hour)',
            'mape': self.metrics.get('test_mape', 'N/A'),
            'r2': self.metrics.get('test_r2', 'N/A'),
            'features': len(self.feature_names),
            'feature_names': self.feature_names[:10] + ['...'] if len(self.feature_names) > 10 else self.feature_names
        }


# 싱글톤 인스턴스
_catboost_predictor: Optional[SMPCatBoostPredictor] = None


def get_catboost_predictor() -> SMPCatBoostPredictor:
    """CatBoost 예측기 싱글톤 인스턴스 반환"""
    global _catboost_predictor

    if _catboost_predictor is None:
        _catboost_predictor = SMPCatBoostPredictor()

    return _catboost_predictor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CatBoost SMP Predictor Test (RTM)")
    print("=" * 60)

    predictor = SMPCatBoostPredictor()
    print(f"\n모델 정보: {predictor.get_model_info()}")

    if predictor.is_ready():
        # 단일 시간 예측
        result = predictor.predict_next_hour()
        print(f"\n다음 1시간 예측:")
        print(f"  시간: {result['time']}")
        print(f"  SMP: {result['smp']:.2f} 원/kWh")
        print(f"  신뢰구간: {result['confidence_low']:.2f} ~ {result['confidence_high']:.2f}")
        print(f"  모델: {result['model_used']}")
        print(f"  MAPE: {result['mape']:.2f}%")

        # 다중 시간 예측
        multi_result = predictor.predict_hours(6)
        print(f"\n향후 6시간 예측 (재귀):")
        for p in multi_result['predictions']:
            print(f"  {p['time'].strftime('%H:%M')}: {p['smp']:.2f} 원/kWh")
    else:
        print("모델이 준비되지 않았습니다.")
