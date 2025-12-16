"""
통합 파이프라인 (Task 25)
==========================

전력 수요 예측 시스템의 전체 파이프라인 통합

주요 기능:
1. 데이터 로드 및 전처리
2. 모델 학습 및 예측
3. 분석 및 리포팅
4. 모니터링 및 알림

Author: Claude Code
Date: 2025-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 파이프라인 설정
# ============================================================================

class PipelineStage(Enum):
    """파이프라인 단계"""
    DATA_LOAD = "data_load"
    PREPROCESS = "preprocess"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    REPORTING = "reporting"


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    name: str = "power_demand_forecast"
    data_path: str = "data/"
    model_path: str = "models/"
    output_path: str = "results/"
    log_path: str = "logs/"

    # 데이터 설정
    sequence_length: int = 168  # 7일 * 24시간
    prediction_horizon: int = 24  # 24시간 예측

    # 모델 설정
    model_type: str = "lstm"  # lstm, tft, ensemble
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # 학습 설정
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10

    # 분석 설정
    run_anomaly_detection: bool = True
    run_explainability: bool = True
    run_scenario_analysis: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'data_path': self.data_path,
            'model_path': self.model_path,
            'output_path': self.output_path,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience
        }


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    stage: PipelineStage
    message: str = ""
    data: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'stage': self.stage.value,
            'message': self.message,
            'metrics': self.metrics,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp
        }


# ============================================================================
# 통합 파이프라인
# ============================================================================

class PowerDemandPipeline:
    """
    전력 수요 예측 통합 파이프라인

    모든 기능을 하나의 통합된 인터페이스로 제공합니다.

    Example:
        >>> config = PipelineConfig()
        >>> pipeline = PowerDemandPipeline(config)
        >>> result = pipeline.run()
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.results: List[PipelineResult] = []
        self._setup_directories()

    def _setup_directories(self):
        """디렉토리 설정"""
        for path_attr in ['data_path', 'model_path', 'output_path', 'log_path']:
            path = Path(getattr(self.config, path_attr))
            path.mkdir(parents=True, exist_ok=True)

    def _record_result(
        self,
        stage: PipelineStage,
        success: bool,
        message: str = "",
        data: Any = None,
        metrics: Dict[str, float] = None,
        duration: float = 0.0
    ):
        """결과 기록"""
        result = PipelineResult(
            success=success,
            stage=stage,
            message=message,
            data=data,
            metrics=metrics or {},
            duration_seconds=duration
        )
        self.results.append(result)
        logger.info(f"Stage {stage.value}: {'Success' if success else 'Failed'} - {message}")
        return result

    # =========================================================================
    # 데이터 처리
    # =========================================================================

    def load_data(self, data_source: Union[str, pd.DataFrame] = None) -> PipelineResult:
        """데이터 로드"""
        import time
        start = time.time()

        try:
            if isinstance(data_source, pd.DataFrame):
                self.data = data_source
            elif isinstance(data_source, str):
                self.data = pd.read_csv(data_source)
            else:
                # 기본 데이터 경로에서 로드
                data_path = Path(self.config.data_path)
                csv_files = list(data_path.glob("*.csv"))
                if csv_files:
                    self.data = pd.read_csv(csv_files[0])
                else:
                    # 샘플 데이터 생성
                    self.data = self._generate_sample_data()

            duration = time.time() - start
            return self._record_result(
                PipelineStage.DATA_LOAD,
                True,
                f"Loaded {len(self.data)} records",
                metrics={'n_records': len(self.data), 'n_features': len(self.data.columns)},
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.DATA_LOAD,
                False,
                str(e)
            )

    def _generate_sample_data(self, n_days: int = 365) -> pd.DataFrame:
        """샘플 데이터 생성"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_days),
            periods=n_days * 24,
            freq='h'
        )

        np.random.seed(42)

        # 시간 기반 수요 패턴
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month

        base_demand = 1000
        hourly_pattern = 200 * np.sin(2 * np.pi * hour / 24)
        weekly_pattern = 50 * (day_of_week < 5).astype(float)
        seasonal_pattern = 300 * np.sin(2 * np.pi * (month - 1) / 12)

        demand = (
            base_demand +
            hourly_pattern +
            weekly_pattern +
            seasonal_pattern +
            np.random.randn(len(dates)) * 50
        )

        # 기상 변수
        temperature = 15 + 15 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.randn(len(dates)) * 3
        humidity = 60 + np.random.randn(len(dates)) * 15

        return pd.DataFrame({
            'datetime': dates,
            'demand': demand,
            'temperature': temperature,
            'humidity': humidity,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month
        })

    def preprocess(self) -> PipelineResult:
        """데이터 전처리"""
        import time
        start = time.time()

        try:
            # 결측치 처리
            if self.data.isnull().any().any():
                self.data = self.data.ffill().bfill()

            # datetime 인덱스 설정
            if 'datetime' in self.data.columns:
                self.data.set_index('datetime', inplace=True)

            # 스케일링을 위한 피처 저장
            self.feature_names = [c for c in self.data.columns if c != 'demand']

            duration = time.time() - start
            return self._record_result(
                PipelineStage.PREPROCESS,
                True,
                f"Preprocessed data with {len(self.feature_names)} features",
                metrics={'n_features': len(self.feature_names)},
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.PREPROCESS,
                False,
                str(e)
            )

    def engineer_features(self) -> PipelineResult:
        """피처 엔지니어링"""
        import time
        start = time.time()

        try:
            # 추가 피처 생성
            if 'demand' in self.data.columns:
                # 지연 피처
                for lag in [1, 24, 168]:
                    self.data[f'demand_lag_{lag}'] = self.data['demand'].shift(lag)

                # 롤링 통계
                self.data['demand_rolling_mean_24'] = self.data['demand'].rolling(24).mean()
                self.data['demand_rolling_std_24'] = self.data['demand'].rolling(24).std()

            # 결측치 제거
            self.data = self.data.dropna()

            self.feature_names = [c for c in self.data.columns if c != 'demand']

            duration = time.time() - start
            return self._record_result(
                PipelineStage.FEATURE_ENGINEERING,
                True,
                f"Created {len(self.feature_names)} features",
                metrics={'n_features': len(self.feature_names)},
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.FEATURE_ENGINEERING,
                False,
                str(e)
            )

    # =========================================================================
    # 모델 학습
    # =========================================================================

    def prepare_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 준비"""
        X, y = [], []
        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_horizon

        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(target[i+seq_len:i+seq_len+pred_len])

        return np.array(X), np.array(y)

    def train_model(self, epochs: int = None) -> PipelineResult:
        """모델 학습"""
        import time
        start = time.time()

        try:
            epochs = epochs or self.config.epochs

            # 데이터 준비
            features = self.data[self.feature_names].values
            target = self.data['demand'].values

            X, y = self.prepare_sequences(features, target)

            # Train/Val 분할
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # 텐서 변환
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val)

            # 모델 생성
            input_size = X_train.shape[-1]
            output_size = y_train.shape[-1]

            self.model = SimpleLSTM(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=output_size,
                dropout=self.config.dropout
            )

            # 학습
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # 학습
                self.model.train()
                optimizer.zero_grad()
                output = self.model(X_train_t)
                loss = criterion(output, y_train_t)
                loss.backward()
                optimizer.step()

                # 검증
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val_t)
                    val_loss = criterion(val_output, y_val_t).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            duration = time.time() - start
            return self._record_result(
                PipelineStage.MODEL_TRAINING,
                True,
                f"Trained model for {epoch+1} epochs",
                metrics={
                    'train_loss': loss.item(),
                    'val_loss': best_val_loss,
                    'epochs_trained': epoch + 1
                },
                duration=duration
            )
        except Exception as e:
            import traceback
            return self._record_result(
                PipelineStage.MODEL_TRAINING,
                False,
                f"{str(e)}\n{traceback.format_exc()}"
            )

    # =========================================================================
    # 예측
    # =========================================================================

    def predict(self, input_data: np.ndarray = None) -> PipelineResult:
        """예측 수행"""
        import time
        start = time.time()

        try:
            if self.model is None:
                raise ValueError("Model not trained. Run train_model() first.")

            if input_data is None:
                # 마지막 시퀀스 사용
                features = self.data[self.feature_names].values
                input_data = features[-self.config.sequence_length:]

            # 텐서 변환
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
            else:
                input_tensor = input_data

            # 예측
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_tensor)

            predictions_np = predictions.cpu().numpy().flatten()

            duration = time.time() - start
            return self._record_result(
                PipelineStage.PREDICTION,
                True,
                f"Generated {len(predictions_np)} predictions",
                data=predictions_np,
                metrics={
                    'mean_prediction': float(np.mean(predictions_np)),
                    'max_prediction': float(np.max(predictions_np)),
                    'min_prediction': float(np.min(predictions_np))
                },
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.PREDICTION,
                False,
                str(e)
            )

    # =========================================================================
    # 분석
    # =========================================================================

    def run_analysis(
        self,
        predictions: np.ndarray = None,
        actuals: np.ndarray = None
    ) -> PipelineResult:
        """분석 실행"""
        import time
        start = time.time()

        try:
            analysis_results = {}

            # 이상 탐지
            if self.config.run_anomaly_detection:
                try:
                    from src.analysis.anomaly_detection import ZScoreDetector
                    detector = ZScoreDetector(threshold=3.0)
                    if 'demand' in self.data.columns:
                        result = detector.detect(self.data['demand'].values)
                        analysis_results['anomaly_detection'] = {
                            'n_anomalies': len(result.anomalies),
                            'anomaly_rate': result.anomaly_rate
                        }
                except ImportError:
                    pass

            # 시나리오 분석
            if self.config.run_scenario_analysis and self.model is not None:
                try:
                    from src.analysis.scenario_analysis import (
                        ScenarioGenerator, ScenarioRunner
                    )
                    generator = ScenarioGenerator()
                    runner = ScenarioRunner(self.model, feature_names=self.feature_names)

                    # 마지막 시퀀스로 시나리오 분석
                    features = self.data[self.feature_names].values
                    input_data = torch.FloatTensor(features[-self.config.sequence_length:]).unsqueeze(0)

                    scenarios = [
                        generator.get_predefined('heatwave_mild'),
                        generator.get_predefined('coldwave_mild'),
                    ]
                    results = runner.run_multiple_scenarios(scenarios, input_data)
                    analysis_results['scenario_analysis'] = {
                        'n_scenarios': len(results),
                        'scenarios': [r.scenario_config.name for r in results]
                    }
                except ImportError:
                    pass

            duration = time.time() - start
            return self._record_result(
                PipelineStage.ANALYSIS,
                True,
                f"Completed {len(analysis_results)} analyses",
                data=analysis_results,
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.ANALYSIS,
                False,
                str(e)
            )

    # =========================================================================
    # 리포팅
    # =========================================================================

    def generate_report(self) -> PipelineResult:
        """리포트 생성"""
        import time
        start = time.time()

        try:
            report = {
                'pipeline_name': self.config.name,
                'generated_at': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'stages': [r.to_dict() for r in self.results],
                'summary': {
                    'total_stages': len(self.results),
                    'successful_stages': sum(1 for r in self.results if r.success),
                    'total_duration': sum(r.duration_seconds for r in self.results)
                }
            }

            # 파일 저장
            output_path = Path(self.config.output_path) / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            duration = time.time() - start
            return self._record_result(
                PipelineStage.REPORTING,
                True,
                f"Report saved to {output_path}",
                data=str(output_path),
                duration=duration
            )
        except Exception as e:
            return self._record_result(
                PipelineStage.REPORTING,
                False,
                str(e)
            )

    # =========================================================================
    # 전체 실행
    # =========================================================================

    def run(
        self,
        data_source: Union[str, pd.DataFrame] = None,
        skip_training: bool = False
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info(f"Starting pipeline: {self.config.name}")

        # 1. 데이터 로드
        self.load_data(data_source)

        # 2. 전처리
        self.preprocess()

        # 3. 피처 엔지니어링
        self.engineer_features()

        # 4. 모델 학습
        if not skip_training:
            self.train_model()

        # 5. 예측
        if self.model is not None:
            self.predict()

        # 6. 분석
        self.run_analysis()

        # 7. 리포트 생성
        self.generate_report()

        # 결과 요약
        return {
            'success': all(r.success for r in self.results),
            'stages_completed': len(self.results),
            'stages_failed': sum(1 for r in self.results if not r.success),
            'total_duration': sum(r.duration_seconds for r in self.results),
            'results': self.results
        }


# ============================================================================
# 간단한 LSTM 모델
# ============================================================================

class SimpleLSTM(nn.Module):
    """간단한 LSTM 모델"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 24,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 마지막 출력 사용
        out = self.fc(lstm_out[:, -1, :])
        return out


# ============================================================================
# 헬퍼 함수
# ============================================================================

def run_pipeline(
    config: PipelineConfig = None,
    data_source: Union[str, pd.DataFrame] = None
) -> Dict[str, Any]:
    """파이프라인 간편 실행 함수"""
    pipeline = PowerDemandPipeline(config)
    return pipeline.run(data_source)


def create_default_config() -> PipelineConfig:
    """기본 설정 생성"""
    return PipelineConfig()


def get_pipeline_status(pipeline: PowerDemandPipeline) -> Dict[str, Any]:
    """파이프라인 상태 조회"""
    return {
        'stages_completed': len(pipeline.results),
        'model_trained': pipeline.model is not None,
        'data_loaded': hasattr(pipeline, 'data') and pipeline.data is not None,
        'feature_names': pipeline.feature_names,
        'last_result': pipeline.results[-1].to_dict() if pipeline.results else None
    }
