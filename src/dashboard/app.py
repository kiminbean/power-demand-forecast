"""
제주도 전력 수요 예측 대시보드 (통합 API 버전)
==============================================

FastAPI 서버와 연동하여 실시간 예측을 제공하는 Streamlit 대시보드
전력 수요 예측 + 신재생에너지(태양광/풍력) 발전량 예측 통합

주요 기능:
1. 실시간 예측 차트 (24시간) - API 연동
2. 기상 조건 입력 인터페이스
3. 시나리오 분석 (폭염/한파)
4. 과거 데이터 비교
5. 모델 성능 지표
6. 🌞 신재생에너지 발전량 예측 (태양광/풍력)
7. ⚡ 통합 에너지 현황 대시보드

Usage:
    # 전력 수요 예측 API 서버
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # 신재생에너지 API 서버 (별도)
    cd ../kpx-demand-forecast && uvicorn api.main:app --port 8001

    # 대시보드 실행
    streamlit run src/dashboard/app.py

Author: Power Demand Forecast Team
Date: 2025-12
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from pathlib import Path
import sys

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="제주도 전력 수요 예측",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# 스타일
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .api-connected {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .api-disconnected {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 설정
# ============================================================================

class Config:
    """대시보드 설정"""
    API_URL = "http://localhost:8000"  # 전력 수요 예측 API
    RENEWABLE_API_URL = "http://localhost:8001"  # 신재생에너지 발전량 예측 API (J-REF)
    DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODEL_PATH = PROJECT_ROOT / "models"

    # 시나리오 프리셋
    SCENARIOS = {
        "normal": {"name": "평년", "temp_delta": 0, "humidity_delta": 0, "demand_factor": 1.0},
        "heatwave_mild": {"name": "약한 폭염 (+3°C)", "temp_delta": 3, "humidity_delta": -5, "demand_factor": 1.08},
        "heatwave_severe": {"name": "심한 폭염 (+7°C)", "temp_delta": 7, "humidity_delta": -10, "demand_factor": 1.20},
        "coldwave_mild": {"name": "약한 한파 (-5°C)", "temp_delta": -5, "humidity_delta": 5, "demand_factor": 1.10},
        "coldwave_severe": {"name": "심한 한파 (-10°C)", "temp_delta": -10, "humidity_delta": 10, "demand_factor": 1.25},
    }

    # 신재생에너지 색상
    RENEWABLE_COLORS = {
        "solar": "#F59E0B",  # 태양광 - 노란색
        "wind": "#3B82F6",   # 풍력 - 파란색
        "total": "#10B981",  # 합계 - 초록색
    }


# ============================================================================
# API 클라이언트
# ============================================================================

class APIClient:
    """FastAPI 연동 클라이언트"""

    def __init__(self, base_url: str = Config.API_URL):
        self.base_url = base_url
        self._health_cache = None
        self._health_time = None

    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"status": "offline", "models_loaded": False}

    def get_models(self) -> Optional[Dict]:
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def predict(self, data: List[Dict], model_type: str = "conditional") -> Optional[Dict]:
        """단일 예측 API 호출"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"data": data, "model_type": model_type},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"예측 실패: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"API 연결 오류: {e}")
        return None

    def predict_conditional(self, data: List[Dict], mode: str = "soft") -> Optional[Dict]:
        """조건부 예측 API 호출"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/conditional",
                json={"data": data, "mode": mode},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"조건부 예측 오류: {e}")
        return None

    def predict_batch(self, data: List[Dict], model_type: str = "demand_only", step: int = 1) -> Optional[Dict]:
        """배치 예측 API 호출"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"data": data, "model_type": model_type, "step": step},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"배치 예측 오류: {e}")
        return None


# ============================================================================
# 신재생에너지 API 클라이언트 (J-REF)
# ============================================================================

class RenewableAPIClient:
    """신재생에너지 발전량 예측 API 클라이언트 (태양광/풍력) - J-REF API"""

    def __init__(self, base_url: str = Config.RENEWABLE_API_URL):
        self.base_url = base_url

    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"status": "offline", "models_loaded": {"wind": False, "solar": False}}

    def get_models(self) -> Optional[Dict]:
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def predict(
        self,
        weather: Dict[str, Any],
        energy_type: str = "both",
        include_uncertainty: bool = True
    ) -> Optional[Dict]:
        """
        단일 예측 API 호출

        Args:
            weather: WeatherInput 형식의 딕셔너리
                - datetime, temperature, humidity, wind_speed, wind_direction, pressure
                - (optional) solar_radiation, cloud_cover, visibility, precipitation
            energy_type: "solar", "wind", "both"
            include_uncertainty: 80% 신뢰구간 포함 여부
        """
        try:
            # datetime 문자열 변환
            if isinstance(weather.get("datetime"), datetime):
                weather = weather.copy()
                weather["datetime"] = weather["datetime"].isoformat()

            response = requests.post(
                f"{self.base_url}/predict",
                json={
                    "weather": weather,
                    "energy_type": energy_type,
                    "include_uncertainty": include_uncertainty
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"신재생 예측 실패: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"신재생 API 연결 오류: {e}")
        return None

    def predict_batch(
        self,
        weather_data: List[Dict[str, Any]],
        energy_type: str = "both"
    ) -> Optional[Dict]:
        """
        배치 예측 API 호출 (최대 168시간)

        Args:
            weather_data: WeatherInput 형식의 딕셔너리 리스트
            energy_type: "solar", "wind", "both"

        Returns:
            BatchPredictionResponse 형식:
            {
                "success": True,
                "predictions": [{"datetime": "...", "predictions": {"solar": ..., "wind": ...}}],
                "total_hours": 24,
                "statistics": {"solar": {...}, "wind": {...}},
                "processing_time_ms": 123.4
            }
        """
        try:
            # datetime 문자열 변환
            converted_data = []
            for w in weather_data:
                w_copy = w.copy()
                if isinstance(w_copy.get("datetime"), datetime):
                    w_copy["datetime"] = w_copy["datetime"].isoformat()
                converted_data.append(w_copy)

            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={
                    "weather_data": converted_data,
                    "energy_type": energy_type
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"신재생 배치 예측 실패: {response.status_code}")
        except Exception as e:
            st.error(f"신재생 배치 예측 오류: {e}")
        return None

    def predict_realtime(
        self,
        target_datetime: Optional[datetime] = None,
        energy_type: str = "both"
    ) -> Optional[Dict]:
        """실시간 예측 (기상청 API 자동 연동) - 지원 시"""
        try:
            payload = {"energy_type": energy_type}
            if target_datetime:
                payload["target_datetime"] = target_datetime.isoformat()

            response = requests.post(
                f"{self.base_url}/predict/realtime",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None


def convert_to_renewable_weather(df: pd.DataFrame, n_points: int = 24) -> List[Dict]:
    """
    과거 데이터 DataFrame을 신재생에너지 API 요청 형식으로 변환

    J-REF API WeatherInput 형식:
    - datetime, temperature, humidity, wind_speed, wind_direction,
    - pressure, solar_radiation, cloud_cover, visibility, precipitation
    """
    recent_data = df.tail(n_points).copy()
    weather_list = []

    for idx, row in recent_data.iterrows():
        weather = {
            "datetime": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
            "temperature": float(row.get('기온', row.get('temperature', 15.0))),
            "humidity": float(row.get('습도', row.get('humidity', 60.0))),
            "wind_speed": max(0, float(row.get('풍속', row.get('wind_speed', 3.0)))),
            "wind_direction": float(row.get('풍향', row.get('wind_direction', 180.0))) % 360,
            "pressure": float(row.get('기압', row.get('pressure', 1013.0))),
        }

        # 선택적 필드
        if '일사량' in row or 'solar_radiation' in row:
            val = row.get('일사량', row.get('solar_radiation'))
            if pd.notna(val):
                weather["solar_radiation"] = max(0, float(val))

        if '운량' in row or 'cloud_cover' in row:
            val = row.get('운량', row.get('cloud_cover'))
            if pd.notna(val):
                weather["cloud_cover"] = max(0, min(10, float(val)))

        if '시정' in row or 'visibility' in row:
            val = row.get('시정', row.get('visibility'))
            if pd.notna(val):
                weather["visibility"] = max(0, float(val))

        if '강수량' in row or 'precipitation' in row:
            val = row.get('강수량', row.get('precipitation'))
            if pd.notna(val):
                weather["precipitation"] = max(0, float(val))

        weather_list.append(weather)

    return weather_list


def create_sample_weather(
    base_datetime: datetime,
    hours: int = 24,
    temp: float = 15.0,
    humidity: float = 60.0,
    wind_speed: float = 5.0,
    wind_direction: float = 270.0
) -> List[Dict]:
    """샘플 기상 데이터 생성 (신재생 API용)"""
    weather_list = []

    for h in range(hours):
        dt = base_datetime + timedelta(hours=h)
        hour = dt.hour

        # 시간대별 변동 적용
        temp_adj = temp + 5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else temp - 3
        solar_rad = max(0, 3.5 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0

        weather = {
            "datetime": dt.isoformat(),
            "temperature": temp_adj,
            "humidity": humidity,
            "wind_speed": wind_speed + np.random.uniform(-1, 1),
            "wind_direction": wind_direction,
            "pressure": 1013.0 + np.random.uniform(-5, 5),
            "solar_radiation": solar_rad,
            "cloud_cover": np.random.randint(0, 5),
        }
        weather_list.append(weather)

    return weather_list


# ============================================================================
# 데이터 로더
# ============================================================================

@st.cache_data(ttl=300)
def load_historical_data() -> Optional[pd.DataFrame]:
    """과거 데이터 로드"""
    try:
        data_file = Config.DATA_PATH / "jeju_hourly_merged.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
    return None


def prepare_api_data(df: pd.DataFrame, n_points: int = 168) -> List[Dict]:
    """DataFrame을 API 요청 형식으로 변환"""
    # 최근 n_points개 데이터 선택
    recent_data = df.tail(n_points).copy()

    api_data = []
    for idx, row in recent_data.iterrows():
        record = {
            "datetime": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
            "power_demand": float(row['power_demand']),
        }

        # 기상 데이터 추가 (있는 경우)
        if '기온' in row and pd.notna(row['기온']):
            record["temperature"] = float(row['기온'])
        if '습도' in row and pd.notna(row['습도']):
            record["humidity"] = float(row['습도'])
        if '풍속' in row and pd.notna(row['풍속']):
            record["wind_speed"] = float(row['풍속'])
        if '강수량' in row and pd.notna(row['강수량']):
            record["precipitation"] = float(row['강수량'])

        api_data.append(record)

    return api_data


def apply_weather_modification(
    df: pd.DataFrame,
    temp_delta: float = 0,
    humidity_delta: float = 0
) -> pd.DataFrame:
    """기상 조건 수정 적용"""
    modified = df.copy()

    if '기온' in modified.columns:
        modified['기온'] = modified['기온'] + temp_delta
    if '습도' in modified.columns:
        modified['습도'] = (modified['습도'] + humidity_delta).clip(0, 100)

    return modified


# ============================================================================
# 차트 컴포넌트
# ============================================================================

class Charts:
    """차트 생성 클래스"""

    @staticmethod
    def create_realtime_prediction_chart(
        historical_df: pd.DataFrame,
        prediction_value: float,
        prediction_time: datetime,
        model_used: str
    ) -> go.Figure:
        """실시간 예측 차트"""
        fig = go.Figure()

        # 최근 48시간 실제 데이터
        recent = historical_df.tail(48)

        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent['power_demand'],
            mode='lines',
            name='실제 수요',
            line=dict(color='#10B981', width=2)
        ))

        # 예측 포인트
        fig.add_trace(go.Scatter(
            x=[prediction_time],
            y=[prediction_value],
            mode='markers+text',
            name=f'예측 ({model_used})',
            marker=dict(color='#EF4444', size=15, symbol='star'),
            text=[f'{prediction_value:.0f} MW'],
            textposition='top center',
            textfont=dict(size=14, color='#EF4444')
        ))

        # 예측선 연결
        last_actual = recent['power_demand'].iloc[-1]
        last_time = recent.index[-1]

        fig.add_trace(go.Scatter(
            x=[last_time, prediction_time],
            y=[last_actual, prediction_value],
            mode='lines',
            name='예측 추이',
            line=dict(color='#3B82F6', width=2, dash='dash')
        ))

        fig.update_layout(
            title="실시간 전력 수요 예측 (API 연동)",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_batch_prediction_chart(
        predictions: List[Dict],
        historical_df: pd.DataFrame
    ) -> go.Figure:
        """배치 예측 차트"""
        fig = go.Figure()

        # 최근 실제 데이터
        recent = historical_df.tail(72)

        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent['power_demand'],
            mode='lines',
            name='실제 수요',
            line=dict(color='#10B981', width=2)
        ))

        # 예측 데이터
        pred_times = [pd.to_datetime(p['timestamp']) for p in predictions]
        pred_values = [p['prediction'] for p in predictions]

        fig.add_trace(go.Scatter(
            x=pred_times,
            y=pred_values,
            mode='lines+markers',
            name='배치 예측',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="배치 예측 결과",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=400,
            template="plotly_white",
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_scenario_comparison_chart(
        scenarios_results: Dict[str, Dict]
    ) -> go.Figure:
        """시나리오 비교 차트"""
        fig = go.Figure()

        colors = {
            'normal': '#64748B',
            'heatwave_mild': '#F97316',
            'heatwave_severe': '#DC2626',
            'coldwave_mild': '#0EA5E9',
            'coldwave_severe': '#1D4ED8'
        }

        for scenario_name, result in scenarios_results.items():
            if result and 'predictions' in result:
                config = Config.SCENARIOS.get(scenario_name, {})
                display_name = config.get('name', scenario_name)
                color = colors.get(scenario_name, '#64748B')

                pred_times = [pd.to_datetime(p['timestamp']) for p in result['predictions']]
                pred_values = [p['prediction'] for p in result['predictions']]

                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_values,
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=2)
                ))

        fig.update_layout(
            title="시나리오별 예측 비교 (API 연동)",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_model_performance_chart(model_info: Dict) -> go.Figure:
        """모델 성능 차트"""
        models = model_info.get('models', [])

        if not models:
            return go.Figure()

        names = [m['name'] for m in models]
        features = [m.get('n_features', 0) for m in models]
        hidden_sizes = [m.get('hidden_size', 0) for m in models]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("피처 수", "Hidden Size")
        )

        fig.add_trace(
            go.Bar(x=names, y=features, marker_color='#3B82F6', name='Features'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=names, y=hidden_sizes, marker_color='#10B981', name='Hidden Size'),
            row=1, col=2
        )

        fig.update_layout(height=300, showlegend=False, template="plotly_white")

        return fig

    @staticmethod
    def create_hourly_pattern_chart(data: pd.DataFrame) -> go.Figure:
        """시간대별 패턴 차트"""
        df = data.copy()
        df['hour'] = df.index.hour

        hourly_avg = df.groupby('hour')['power_demand'].agg(['mean', 'std']).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'],
            error_y=dict(type='data', array=hourly_avg['std'], visible=True),
            marker_color='#3B82F6',
            name='평균 수요'
        ))

        fig.update_layout(
            title="시간대별 평균 전력 수요",
            xaxis_title="시간 (0-23)",
            yaxis_title="전력 수요 (MW)",
            height=350,
            template="plotly_white"
        )

        return fig

    # =========================================================================
    # 신재생에너지 차트
    # =========================================================================

    @staticmethod
    def create_renewable_prediction_chart(
        predictions: List[Dict],
        energy_type: str = "both"
    ) -> go.Figure:
        """신재생에너지 발전량 예측 차트"""
        fig = go.Figure()

        if not predictions:
            return fig

        timestamps = [pd.to_datetime(p.get('timestamp', p.get('datetime'))) for p in predictions]

        # 태양광
        if energy_type in ["solar", "both"]:
            solar_vals = []
            solar_lower = []
            solar_upper = []

            for p in predictions:
                preds = p.get('predictions', {})

                # J-REF API 형식: {"solar": 123.4, "wind": 56.7} (딕셔너리)
                if isinstance(preds, dict):
                    solar_vals.append(preds.get('solar', 0) or 0)
                    solar_lower.append(0)
                    solar_upper.append(0)
                # 기존 형식: [{"energy_type": "solar", "prediction_mw": 123.4}, ...] (리스트)
                elif isinstance(preds, list):
                    solar_pred = next(
                        (pred for pred in preds if isinstance(pred, dict) and pred.get('energy_type') == 'solar'),
                        None
                    )
                    if solar_pred:
                        solar_vals.append(solar_pred.get('prediction_mw', 0))
                        solar_lower.append(solar_pred.get('lower_bound_mw', 0))
                        solar_upper.append(solar_pred.get('upper_bound_mw', 0))
                    else:
                        solar_vals.append(p.get('solar_mw', 0) or 0)
                        solar_lower.append(p.get('solar_lower', 0) or 0)
                        solar_upper.append(p.get('solar_upper', 0) or 0)
                else:
                    solar_vals.append(p.get('solar_mw', 0) or 0)
                    solar_lower.append(p.get('solar_lower', 0) or 0)
                    solar_upper.append(p.get('solar_upper', 0) or 0)

            if solar_vals and any(v > 0 for v in solar_vals):
                # 신뢰구간 영역
                if solar_lower and solar_upper:
                    fig.add_trace(go.Scatter(
                        x=timestamps + timestamps[::-1],
                        y=solar_upper + solar_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(245, 158, 11, 0.2)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='태양광 80% CI',
                        showlegend=False
                    ))

                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=solar_vals,
                    mode='lines+markers',
                    name='태양광 발전',
                    line=dict(color=Config.RENEWABLE_COLORS["solar"], width=2),
                    marker=dict(size=4)
                ))

        # 풍력
        if energy_type in ["wind", "both"]:
            wind_vals = []
            wind_lower = []
            wind_upper = []

            for p in predictions:
                preds = p.get('predictions', {})

                # J-REF API 형식: {"solar": 123.4, "wind": 56.7} (딕셔너리)
                if isinstance(preds, dict):
                    wind_vals.append(preds.get('wind', 0) or 0)
                    wind_lower.append(0)
                    wind_upper.append(0)
                # 기존 형식: [{"energy_type": "wind", "prediction_mw": 56.7}, ...] (리스트)
                elif isinstance(preds, list):
                    wind_pred = next(
                        (pred for pred in preds if isinstance(pred, dict) and pred.get('energy_type') == 'wind'),
                        None
                    )
                    if wind_pred:
                        wind_vals.append(wind_pred.get('prediction_mw', 0))
                        wind_lower.append(wind_pred.get('lower_bound_mw', 0))
                        wind_upper.append(wind_pred.get('upper_bound_mw', 0))
                    else:
                        wind_vals.append(p.get('wind_mw', 0) or 0)
                        wind_lower.append(p.get('wind_lower', 0) or 0)
                        wind_upper.append(p.get('wind_upper', 0) or 0)
                else:
                    wind_vals.append(p.get('wind_mw', 0) or 0)
                    wind_lower.append(p.get('wind_lower', 0) or 0)
                    wind_upper.append(p.get('wind_upper', 0) or 0)

            if wind_vals and any(v > 0 for v in wind_vals):
                # 신뢰구간 영역
                if wind_lower and wind_upper:
                    fig.add_trace(go.Scatter(
                        x=timestamps + timestamps[::-1],
                        y=wind_upper + wind_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.2)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='풍력 80% CI',
                        showlegend=False
                    ))

                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=wind_vals,
                    mode='lines+markers',
                    name='풍력 발전',
                    line=dict(color=Config.RENEWABLE_COLORS["wind"], width=2),
                    marker=dict(size=4)
                ))

        fig.update_layout(
            title="🌞🌬️ 신재생에너지 발전량 예측",
            xaxis_title="시간",
            yaxis_title="발전량 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_renewable_pie_chart(solar_mw: float, wind_mw: float) -> go.Figure:
        """신재생에너지 구성 비율 파이 차트"""
        total = solar_mw + wind_mw
        if total == 0:
            total = 1

        fig = go.Figure(data=[go.Pie(
            labels=['태양광', '풍력'],
            values=[solar_mw, wind_mw],
            marker=dict(colors=[Config.RENEWABLE_COLORS["solar"], Config.RENEWABLE_COLORS["wind"]]),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=14
        )])

        fig.update_layout(
            title="신재생에너지 구성 비율",
            height=350,
            showlegend=True,
            annotations=[dict(
                text=f'{total:.0f}<br>MW',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )

        return fig

    @staticmethod
    def create_energy_overview_chart(
        demand_mw: float,
        solar_mw: float,
        wind_mw: float
    ) -> go.Figure:
        """통합 에너지 현황 차트"""
        renewable_total = solar_mw + wind_mw
        net_demand = demand_mw - renewable_total

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("에너지 수급 현황", "신재생 비율"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        # 막대 그래프
        fig.add_trace(
            go.Bar(
                x=['전력 수요', '태양광', '풍력', '순수요'],
                y=[demand_mw, solar_mw, wind_mw, max(0, net_demand)],
                marker_color=['#EF4444', Config.RENEWABLE_COLORS["solar"],
                             Config.RENEWABLE_COLORS["wind"], '#64748B'],
                text=[f'{v:.0f}' for v in [demand_mw, solar_mw, wind_mw, max(0, net_demand)]],
                textposition='outside'
            ),
            row=1, col=1
        )

        # 파이 차트 (신재생 비율)
        renewable_ratio = (renewable_total / demand_mw * 100) if demand_mw > 0 else 0
        fig.add_trace(
            go.Pie(
                labels=['신재생', '기타'],
                values=[renewable_total, max(0, net_demand)],
                marker=dict(colors=[Config.RENEWABLE_COLORS["total"], '#CBD5E1']),
                hole=0.4,
                textinfo='percent',
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            template="plotly_white",
            showlegend=False,
            annotations=[
                dict(
                    text=f'{renewable_ratio:.1f}%',
                    x=0.82, y=0.5,
                    font_size=16,
                    showarrow=False
                )
            ]
        )

        return fig

    @staticmethod
    def create_renewable_timeseries_combined(
        demand_predictions: List[Dict],
        renewable_predictions: List[Dict]
    ) -> go.Figure:
        """전력 수요 + 신재생 발전 통합 시계열 차트"""
        fig = go.Figure()

        # 전력 수요
        if demand_predictions:
            times = [pd.to_datetime(p.get('timestamp', p.get('datetime'))) for p in demand_predictions]
            values = [p.get('prediction', 0) for p in demand_predictions]

            fig.add_trace(go.Scatter(
                x=times,
                y=values,
                mode='lines',
                name='전력 수요 예측',
                line=dict(color='#EF4444', width=2)
            ))

        # 신재생 발전량 합계
        if renewable_predictions:
            times = [pd.to_datetime(p.get('timestamp', p.get('datetime'))) for p in renewable_predictions]
            total_renewable = []

            for p in renewable_predictions:
                preds = p.get('predictions', {})

                # J-REF API 형식: {"solar": 123.4, "wind": 56.7} (딕셔너리)
                if isinstance(preds, dict):
                    solar = preds.get('solar', 0) or 0
                    wind = preds.get('wind', 0) or 0
                # 기존 형식: [{"energy_type": "solar", "prediction_mw": 123.4}, ...] (리스트)
                elif isinstance(preds, list):
                    solar = 0
                    wind = 0
                    for pred in preds:
                        if isinstance(pred, dict):
                            if pred.get('energy_type') == 'solar':
                                solar = pred.get('prediction_mw', 0)
                            elif pred.get('energy_type') == 'wind':
                                wind = pred.get('prediction_mw', 0)
                else:
                    solar = p.get('solar_mw', 0) or 0
                    wind = p.get('wind_mw', 0) or 0

                total_renewable.append(solar + wind)

            fig.add_trace(go.Scatter(
                x=times,
                y=total_renewable,
                mode='lines',
                name='신재생 발전량',
                line=dict(color=Config.RENEWABLE_COLORS["total"], width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)'
            ))

        fig.update_layout(
            title="⚡ 전력 수요 vs 신재생 발전량 예측",
            xaxis_title="시간",
            yaxis_title="전력 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig


# ============================================================================
# 메인 대시보드
# ============================================================================

def main():
    """메인 함수"""

    # API 클라이언트 초기화
    api = APIClient()
    renewable_api = RenewableAPIClient()

    # 헤더
    st.markdown('<p class="main-header">⚡ 제주도 전력 수요 예측 시스템</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">전력 수요 예측 + 신재생에너지 발전량 예측 | FastAPI 연동</p>', unsafe_allow_html=True)

    # API 상태 확인
    health = api.health_check()
    api_online = health.get("status") == "healthy"

    renewable_health = renewable_api.health_check()
    renewable_online = renewable_health.get("status") == "healthy"

    # 상단 상태 표시
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if api_online:
            st.markdown('<div class="api-connected">🟢 수요예측 API</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-disconnected">🔴 수요예측 API</div>', unsafe_allow_html=True)

    with col2:
        if renewable_online:
            st.markdown('<div class="api-connected">🟢 신재생 API</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-disconnected">🔴 신재생 API</div>', unsafe_allow_html=True)

    with col3:
        st.metric("수요 모델", "✅" if health.get("models_loaded") else "❌")

    with col4:
        models_loaded = renewable_health.get("models_loaded", {})
        solar_loaded = models_loaded.get("solar", False)
        wind_loaded = models_loaded.get("wind", False)
        st.metric("태양광/풍력", f"{'☀️' if solar_loaded else '❌'}/{'💨' if wind_loaded else '❌'}")

    with col5:
        uptime = health.get("uptime_seconds", 0)
        st.metric("수요 업타임", f"{uptime/60:.1f}분")

    with col6:
        r_uptime = renewable_health.get("uptime_seconds", 0)
        st.metric("신재생 업타임", f"{r_uptime/60:.1f}분")

    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")

        # 모델 선택
        st.subheader("모델 선택")
        model_type = st.selectbox(
            "예측 모델",
            options=["conditional", "demand_only", "weather_full"],
            index=0,
            format_func=lambda x: {
                "conditional": "조건부 앙상블 (권장)",
                "demand_only": "수요 전용",
                "weather_full": "기상 포함"
            }.get(x, x)
        )

        st.markdown("---")

        # 기상 조건 수정
        st.subheader("기상 조건 수정")
        st.caption("시나리오 분석용 기상 조건 조정")

        temp_delta = st.slider(
            "온도 변화 (°C)",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5
        )

        humidity_delta = st.slider(
            "습도 변화 (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=1.0
        )

        st.markdown("---")

        # 시나리오 프리셋
        st.subheader("시나리오 프리셋")
        scenario_options = {v["name"]: k for k, v in Config.SCENARIOS.items()}
        selected_preset = st.selectbox(
            "프리셋 선택",
            options=["직접 설정"] + list(scenario_options.keys())
        )

        if selected_preset != "직접 설정":
            preset_key = scenario_options[selected_preset]
            preset = Config.SCENARIOS[preset_key]
            temp_delta = float(preset["temp_delta"])
            humidity_delta = float(preset["humidity_delta"])
            st.info(f"온도: {temp_delta:+.0f}°C, 습도: {humidity_delta:+.0f}%")

        st.markdown("---")

        # 데이터 범위
        st.subheader("데이터 범위")
        date_range = st.date_input(
            "기간 선택",
            value=(
                datetime.now().date() - timedelta(days=7),
                datetime.now().date()
            )
        )

        st.markdown("---")

        # API 연결 상태
        st.subheader("🔌 API 상태")

        # 전력 수요 API
        if api_online:
            st.success(f"✅ 수요 예측: 연결됨")
        else:
            st.error(f"❌ 수요 예측: 오프라인")

        # 신재생에너지 API
        if renewable_online:
            st.success(f"✅ 신재생에너지: 연결됨")
        else:
            st.error(f"❌ 신재생에너지: 오프라인")

        st.markdown("---")

        # API URL 정보
        with st.expander("🔗 API 서버 URL"):
            st.caption(f"**수요 예측**: {Config.API_URL}")
            st.caption(f"**신재생에너지**: {Config.RENEWABLE_API_URL}")

            st.markdown("---")
            st.markdown("**서버 실행 명령:**")
            st.code("""
# 전력 수요 예측 API
uvicorn api.main:app --port 8000

# 신재생에너지 API
cd ../kpx-demand-forecast
uvicorn api.main:app --port 8001
            """, language="bash")

    # 데이터 로드
    historical_data = load_historical_data()

    if historical_data is None or len(historical_data) == 0:
        st.error("과거 데이터를 로드할 수 없습니다.")
        return

    st.success(f"데이터 로드: {len(historical_data):,}개 레코드 (2013-2024)")

    # 탭 구성 (7개 탭: 전력수요 + 신재생에너지 + 통합)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔮 실시간 예측",
        "🌡️ 시나리오 분석",
        "🌞 신재생 발전",
        "⚡ 통합 현황",
        "📊 과거 데이터",
        "🤖 모델 정보",
        "ℹ️ 시스템 정보"
    ])

    # ==========================================================================
    # 탭 1: 실시간 예측
    # ==========================================================================
    with tab1:
        st.header("실시간 전력 수요 예측")

        if not api_online:
            st.warning("API 서버에 연결할 수 없습니다. API를 먼저 실행해주세요.")
            st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000")
        else:
            col1, col2 = st.columns([3, 1])

            with col2:
                st.subheader("예측 실행")

                if st.button("🚀 예측 실행", type="primary", width="stretch"):
                    with st.spinner("예측 중..."):
                        # 기상 조건 수정 적용
                        modified_data = apply_weather_modification(
                            historical_data,
                            temp_delta=temp_delta,
                            humidity_delta=humidity_delta
                        )

                        # API 데이터 준비
                        api_data = prepare_api_data(modified_data, n_points=168)

                        # API 호출
                        if model_type == "conditional":
                            result = api.predict_conditional(api_data, mode="soft")
                        else:
                            result = api.predict(api_data, model_type=model_type)

                        if result:
                            st.session_state['last_prediction'] = result
                            st.success("예측 완료!")

                # 결과 표시
                if 'last_prediction' in st.session_state:
                    result = st.session_state['last_prediction']

                    st.markdown("---")
                    st.subheader("예측 결과")

                    st.metric(
                        "예측 수요",
                        f"{result['prediction']:.1f} MW",
                        delta=f"{result['prediction'] - historical_data['power_demand'].iloc[-1]:.1f} MW"
                    )

                    st.caption(f"모델: {result.get('model_used', 'N/A')}")
                    st.caption(f"처리시간: {result.get('processing_time_ms', 0):.1f}ms")

                    if 'context' in result:
                        with st.expander("상세 컨텍스트"):
                            st.json(result['context'])

            with col1:
                if 'last_prediction' in st.session_state:
                    result = st.session_state['last_prediction']

                    pred_time = pd.to_datetime(result.get('timestamp', datetime.now()))

                    fig = Charts.create_realtime_prediction_chart(
                        historical_data,
                        result['prediction'],
                        pred_time,
                        result.get('model_used', 'unknown')
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("오른쪽의 '예측 실행' 버튼을 클릭하여 예측을 시작하세요.")

                    # 기본 차트 표시
                    recent = historical_data.tail(72)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=recent.index,
                        y=recent['power_demand'],
                        mode='lines',
                        name='최근 수요',
                        line=dict(color='#10B981', width=2)
                    ))
                    fig.update_layout(
                        title="최근 72시간 전력 수요",
                        xaxis_title="시간",
                        yaxis_title="전력 수요 (MW)",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, width="stretch")

    # ==========================================================================
    # 탭 2: 시나리오 분석
    # ==========================================================================
    with tab2:
        st.header("시나리오 분석 (What-If)")

        if not api_online:
            st.warning("API 서버에 연결할 수 없습니다.")
        else:
            st.markdown("""
            다양한 기상 시나리오에서의 전력 수요를 API를 통해 예측합니다.
            """)

            col1, col2 = st.columns(2)

            with col1:
                compare_scenarios = st.multiselect(
                    "비교할 시나리오 선택",
                    options=list(Config.SCENARIOS.keys()),
                    default=["normal", "heatwave_mild", "coldwave_mild"],
                    format_func=lambda x: Config.SCENARIOS[x]["name"]
                )

            with col2:
                batch_step = st.slider("예측 간격 (시간)", 1, 6, 1)

            if st.button("📊 시나리오 분석 실행", type="primary"):
                if compare_scenarios:
                    with st.spinner("시나리오 분석 중..."):
                        scenarios_results = {}

                        progress_bar = st.progress(0)

                        for i, scenario in enumerate(compare_scenarios):
                            config = Config.SCENARIOS[scenario]

                            # 기상 조건 수정
                            modified_data = apply_weather_modification(
                                historical_data,
                                temp_delta=config["temp_delta"],
                                humidity_delta=config["humidity_delta"]
                            )

                            # API 데이터 준비
                            api_data = prepare_api_data(modified_data, n_points=200)

                            # 배치 예측
                            result = api.predict_batch(api_data, model_type="demand_only", step=batch_step)

                            if result:
                                # 수요 계수 적용
                                for pred in result['predictions']:
                                    pred['prediction'] *= config["demand_factor"]

                                scenarios_results[scenario] = result

                            progress_bar.progress((i + 1) / len(compare_scenarios))

                        st.session_state['scenarios_results'] = scenarios_results
                        st.success("시나리오 분석 완료!")

            # 결과 표시
            if 'scenarios_results' in st.session_state:
                results = st.session_state['scenarios_results']

                if results:
                    # 비교 차트
                    fig = Charts.create_scenario_comparison_chart(results)
                    st.plotly_chart(fig, width="stretch")

                    # 통계 테이블
                    st.subheader("시나리오 비교 통계")

                    comparison_data = []
                    for scenario_name, result in results.items():
                        if result and 'predictions' in result:
                            config = Config.SCENARIOS[scenario_name]
                            predictions = [p['prediction'] for p in result['predictions']]

                            comparison_data.append({
                                "시나리오": config["name"],
                                "온도 변화": f"{config['temp_delta']:+d}°C",
                                "습도 변화": f"{config['humidity_delta']:+d}%",
                                "평균 수요": f"{np.mean(predictions):.1f} MW",
                                "피크 수요": f"{np.max(predictions):.1f} MW",
                                "최소 수요": f"{np.min(predictions):.1f} MW",
                            })

                    if comparison_data:
                        st.dataframe(
                            pd.DataFrame(comparison_data),
                            width="stretch",
                            hide_index=True
                        )

    # ==========================================================================
    # 탭 3: 신재생에너지 발전량 예측
    # ==========================================================================
    with tab3:
        st.header("🌞🌬️ 신재생에너지 발전량 예측")
        st.markdown("태양광 및 풍력 발전량 예측 (J-REF API 연동)")

        if not renewable_online:
            st.warning("⚠️ 신재생에너지 API 서버에 연결할 수 없습니다.")
            st.code("cd ../kpx-demand-forecast && uvicorn api.main:app --port 8001", language="bash")
            st.info("API 서버 실행 후 새로고침해주세요.")
        else:
            col1, col2 = st.columns([3, 1])

            with col2:
                st.subheader("⚙️ 예측 설정")

                # 예측 타입 선택
                energy_type = st.selectbox(
                    "에너지 타입",
                    options=["both", "solar", "wind"],
                    format_func=lambda x: {
                        "both": "☀️💨 태양광 + 풍력",
                        "solar": "☀️ 태양광만",
                        "wind": "💨 풍력만"
                    }.get(x, x)
                )

                # 예측 시간
                forecast_hours = st.slider("예측 시간 (h)", 6, 168, 24, step=6)

                st.markdown("---")
                st.subheader("🌤️ 기상 조건")

                # 수동 기상 입력
                input_temp = st.number_input("기온 (°C)", value=15.0, min_value=-20.0, max_value=45.0)
                input_humidity = st.number_input("습도 (%)", value=60.0, min_value=0.0, max_value=100.0)
                input_wind_speed = st.number_input("풍속 (m/s)", value=5.0, min_value=0.0, max_value=50.0)
                input_wind_dir = st.number_input("풍향 (°)", value=270.0, min_value=0.0, max_value=359.0)
                input_solar_rad = st.number_input("일사량 (MJ/m²)", value=2.0, min_value=0.0, max_value=5.0)

                # 예측 실행 버튼
                if st.button("🚀 신재생 발전량 예측", type="primary", width="stretch"):
                    with st.spinner("신재생에너지 발전량 예측 중..."):
                        # 기상 데이터 생성
                        base_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
                        weather_data = create_sample_weather(
                            base_datetime=base_dt,
                            hours=forecast_hours,
                            temp=input_temp,
                            humidity=input_humidity,
                            wind_speed=input_wind_speed,
                            wind_direction=input_wind_dir
                        )

                        # 일사량 적용
                        for w in weather_data:
                            w["solar_radiation"] = max(0, input_solar_rad * np.sin(
                                np.pi * (pd.to_datetime(w["datetime"]).hour - 6) / 12
                            )) if 6 <= pd.to_datetime(w["datetime"]).hour <= 18 else 0

                        # API 호출
                        result = renewable_api.predict_batch(weather_data, energy_type)

                        if result and result.get("success"):
                            st.session_state['renewable_prediction'] = result
                            st.session_state['renewable_weather'] = weather_data
                            st.success(f"✅ 예측 완료! ({result.get('total_hours', 0)}시간)")

            with col1:
                if 'renewable_prediction' in st.session_state:
                    result = st.session_state['renewable_prediction']
                    stats = result.get('statistics', {})

                    # 요약 메트릭
                    st.subheader("📊 예측 결과 요약")
                    metric_cols = st.columns(4)

                    solar_stats = stats.get('solar', {})
                    wind_stats = stats.get('wind', {})

                    with metric_cols[0]:
                        st.metric(
                            "☀️ 태양광 평균",
                            f"{solar_stats.get('mean_mw', 0):.1f} MW",
                            help="예측 기간 평균 태양광 발전량"
                        )
                    with metric_cols[1]:
                        st.metric(
                            "☀️ 태양광 피크",
                            f"{solar_stats.get('max_mw', 0):.1f} MW"
                        )
                    with metric_cols[2]:
                        st.metric(
                            "💨 풍력 평균",
                            f"{wind_stats.get('mean_mw', 0):.1f} MW",
                            help="예측 기간 평균 풍력 발전량"
                        )
                    with metric_cols[3]:
                        st.metric(
                            "💨 풍력 피크",
                            f"{wind_stats.get('max_mw', 0):.1f} MW"
                        )

                    # 발전량 예측 차트
                    st.subheader("📈 시간별 발전량 예측")

                    predictions = result.get('predictions', [])
                    if predictions:
                        # 데이터 변환
                        chart_data = []
                        for p in predictions:
                            dt_str = p.get('datetime', '')
                            preds = p.get('predictions', {})
                            chart_data.append({
                                'datetime': pd.to_datetime(dt_str),
                                'solar': preds.get('solar', 0),
                                'wind': preds.get('wind', 0),
                                'total': preds.get('solar', 0) + preds.get('wind', 0)
                            })

                        chart_df = pd.DataFrame(chart_data)

                        # Plotly 차트
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=chart_df['datetime'],
                            y=chart_df['solar'],
                            mode='lines+markers',
                            name='☀️ 태양광',
                            line=dict(color=Config.RENEWABLE_COLORS["solar"], width=2),
                            fill='tozeroy',
                            fillcolor='rgba(245, 158, 11, 0.2)'
                        ))

                        fig.add_trace(go.Scatter(
                            x=chart_df['datetime'],
                            y=chart_df['wind'],
                            mode='lines+markers',
                            name='💨 풍력',
                            line=dict(color=Config.RENEWABLE_COLORS["wind"], width=2),
                            fill='tozeroy',
                            fillcolor='rgba(59, 130, 246, 0.2)'
                        ))

                        fig.add_trace(go.Scatter(
                            x=chart_df['datetime'],
                            y=chart_df['total'],
                            mode='lines',
                            name='합계',
                            line=dict(color=Config.RENEWABLE_COLORS["total"], width=3, dash='dash')
                        ))

                        fig.update_layout(
                            title="신재생에너지 발전량 예측",
                            xaxis_title="시간",
                            yaxis_title="발전량 (MW)",
                            height=450,
                            template="plotly_white",
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig, width="stretch")

                    # 구성 비율 파이 차트
                    col_pie1, col_pie2 = st.columns(2)

                    with col_pie1:
                        total_solar = solar_stats.get('total_mwh', 0)
                        total_wind = wind_stats.get('total_mwh', 0)

                        fig_pie = Charts.create_renewable_pie_chart(total_solar, total_wind)
                        st.plotly_chart(fig_pie, width="stretch")

                    with col_pie2:
                        # 통계 테이블
                        st.markdown("### 📋 상세 통계")
                        stats_table = []
                        if solar_stats:
                            stats_table.append({
                                "타입": "☀️ 태양광",
                                "평균 (MW)": f"{solar_stats.get('mean_mw', 0):.1f}",
                                "최소 (MW)": f"{solar_stats.get('min_mw', 0):.1f}",
                                "최대 (MW)": f"{solar_stats.get('max_mw', 0):.1f}",
                                "총량 (MWh)": f"{solar_stats.get('total_mwh', 0):.1f}",
                            })
                        if wind_stats:
                            stats_table.append({
                                "타입": "💨 풍력",
                                "평균 (MW)": f"{wind_stats.get('mean_mw', 0):.1f}",
                                "최소 (MW)": f"{wind_stats.get('min_mw', 0):.1f}",
                                "최대 (MW)": f"{wind_stats.get('max_mw', 0):.1f}",
                                "총량 (MWh)": f"{wind_stats.get('total_mwh', 0):.1f}",
                            })
                        if stats_table:
                            st.dataframe(pd.DataFrame(stats_table), width="stretch", hide_index=True)

                else:
                    st.info("👈 오른쪽에서 기상 조건을 입력하고 '신재생 발전량 예측' 버튼을 클릭하세요.")

                    # 모델 정보 표시
                    model_info = renewable_api.get_models()
                    if model_info:
                        st.subheader("🤖 신재생에너지 예측 모델")
                        for model in model_info.get('models', []):
                            with st.expander(f"{model['name'].upper()} 모델"):
                                st.markdown(f"""
                                - **타입**: {model.get('type', 'N/A')}
                                - **R²**: {model.get('r2', 0):.4f}
                                - **RMSE**: {model.get('rmse', 0):.2f} MW
                                - **피처 수**: {model.get('features', 0)}
                                - **설명**: {model.get('description', '')}
                                """)

    # ==========================================================================
    # 탭 4: 통합 에너지 현황
    # ==========================================================================
    with tab4:
        st.header("⚡ 통합 에너지 현황")
        st.markdown("전력 수요 예측 + 신재생에너지 발전량 예측 통합 분석")

        # API 상태 확인
        both_online = api_online and renewable_online

        if not both_online:
            missing = []
            if not api_online:
                missing.append("전력 수요 예측 API (포트 8000)")
            if not renewable_online:
                missing.append("신재생에너지 API (포트 8001)")

            st.warning(f"⚠️ 다음 API가 오프라인입니다: {', '.join(missing)}")
            st.code("""
# 전력 수요 예측 API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 신재생에너지 API (별도 터미널)
cd ../kpx-demand-forecast && uvicorn api.main:app --port 8001
            """, language="bash")
        else:
            # 예측 설정
            st.subheader("⚙️ 통합 예측 설정")
            col_set1, col_set2, col_set3 = st.columns(3)

            with col_set1:
                integrated_hours = st.slider("예측 기간 (시간)", 6, 48, 24, step=6, key="integrated_hours")

            with col_set2:
                integrated_temp = st.number_input("기온 변화 (°C)", value=0.0, min_value=-15.0, max_value=15.0, key="int_temp")

            with col_set3:
                integrated_wind = st.number_input("풍속 (m/s)", value=5.0, min_value=0.0, max_value=30.0, key="int_wind")

            if st.button("🔄 통합 예측 실행", type="primary"):
                with st.spinner("전력 수요 및 신재생에너지 통합 예측 중..."):
                    # 1. 전력 수요 예측
                    modified_data = apply_weather_modification(
                        historical_data,
                        temp_delta=integrated_temp,
                        humidity_delta=0
                    )
                    api_data = prepare_api_data(modified_data, n_points=200)
                    demand_result = api.predict_batch(api_data, model_type="demand_only", step=1)

                    # 2. 신재생에너지 예측
                    base_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
                    weather_data = create_sample_weather(
                        base_datetime=base_dt,
                        hours=integrated_hours,
                        temp=15.0 + integrated_temp,
                        humidity=60.0,
                        wind_speed=integrated_wind,
                        wind_direction=270.0
                    )
                    renewable_result = renewable_api.predict_batch(weather_data, "both")

                    if demand_result and renewable_result:
                        st.session_state['integrated_demand'] = demand_result
                        st.session_state['integrated_renewable'] = renewable_result
                        st.success("✅ 통합 예측 완료!")

            # 결과 표시
            if 'integrated_demand' in st.session_state and 'integrated_renewable' in st.session_state:
                demand_result = st.session_state['integrated_demand']
                renewable_result = st.session_state['integrated_renewable']

                # 요약 메트릭
                st.subheader("📊 에너지 수급 현황")

                # 평균 계산
                demand_preds = demand_result.get('predictions', [])
                avg_demand = np.mean([p['prediction'] for p in demand_preds]) if demand_preds else 0

                renewable_stats = renewable_result.get('statistics', {})
                avg_solar = renewable_stats.get('solar', {}).get('mean_mw', 0)
                avg_wind = renewable_stats.get('wind', {}).get('mean_mw', 0)
                avg_renewable = avg_solar + avg_wind

                # 신재생 비율
                renewable_ratio = (avg_renewable / avg_demand * 100) if avg_demand > 0 else 0

                metric_cols = st.columns(5)
                with metric_cols[0]:
                    st.metric("⚡ 평균 수요", f"{avg_demand:.0f} MW")
                with metric_cols[1]:
                    st.metric("☀️ 태양광", f"{avg_solar:.0f} MW")
                with metric_cols[2]:
                    st.metric("💨 풍력", f"{avg_wind:.0f} MW")
                with metric_cols[3]:
                    st.metric("🌱 신재생 합계", f"{avg_renewable:.0f} MW")
                with metric_cols[4]:
                    st.metric("📈 신재생 비율", f"{renewable_ratio:.1f}%",
                             delta=f"{renewable_ratio - 20:.1f}%" if renewable_ratio > 20 else None)

                st.markdown("---")

                # 통합 차트
                st.subheader("📈 수요 vs 신재생 발전량")

                fig = Charts.create_renewable_timeseries_combined(
                    demand_preds,
                    renewable_result.get('predictions', [])
                )
                st.plotly_chart(fig, width="stretch")

                # 에너지 현황 차트
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    fig_overview = Charts.create_energy_overview_chart(
                        avg_demand, avg_solar, avg_wind
                    )
                    st.plotly_chart(fig_overview, width="stretch")

                with col_chart2:
                    # 순수요 분석
                    st.markdown("### 📋 순수요 분석")
                    net_demand = avg_demand - avg_renewable

                    st.markdown(f"""
                    | 항목 | 값 |
                    |---|---|
                    | 총 전력 수요 | **{avg_demand:.0f} MW** |
                    | 신재생 발전량 | **{avg_renewable:.0f} MW** |
                    | 순수요 (기타 발전) | **{max(0, net_demand):.0f} MW** |
                    | 신재생 비율 | **{renewable_ratio:.1f}%** |
                    """)

                    if renewable_ratio >= 30:
                        st.success("🎉 신재생에너지 비율 30% 이상 달성!")
                    elif renewable_ratio >= 20:
                        st.info("📊 신재생에너지 비율 20% 이상")
                    else:
                        st.warning("⚠️ 신재생에너지 비율 20% 미만")

            else:
                st.info("👆 '통합 예측 실행' 버튼을 클릭하여 전력 수요와 신재생에너지 발전량을 함께 분석하세요.")

    # ==========================================================================
    # 탭 5: 과거 데이터
    # ==========================================================================
    with tab5:
        st.header("과거 데이터 분석")

        # 날짜 필터링
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            mask = (historical_data.index.date >= start_date) & (historical_data.index.date <= end_date)
            filtered_data = historical_data[mask]
        else:
            filtered_data = historical_data.tail(168)

        if len(filtered_data) > 0:
            st.success(f"선택 기간: {len(filtered_data):,}개 레코드")

            # 통계
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("평균 수요", f"{filtered_data['power_demand'].mean():.1f} MW")
            with col2:
                st.metric("최대 수요", f"{filtered_data['power_demand'].max():.1f} MW")
            with col3:
                st.metric("최소 수요", f"{filtered_data['power_demand'].min():.1f} MW")
            with col4:
                st.metric("표준편차", f"{filtered_data['power_demand'].std():.1f} MW")

            # 차트
            col1, col2 = st.columns(2)

            with col1:
                # 시계열 차트
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data['power_demand'],
                    mode='lines',
                    name='전력 수요',
                    line=dict(color='#3B82F6')
                ))
                fig.update_layout(
                    title="전력 수요 추이",
                    xaxis_title="시간",
                    yaxis_title="MW",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                # 시간대별 패턴
                fig = Charts.create_hourly_pattern_chart(filtered_data)
                st.plotly_chart(fig, width="stretch")

            # 상세 데이터
            with st.expander("상세 데이터 보기"):
                st.dataframe(
                    filtered_data[['power_demand', '기온', '습도', '풍속']].round(2),
                    width="stretch"
                )
        else:
            st.warning("선택한 기간에 데이터가 없습니다.")

    # ==========================================================================
    # 탭 6: 모델 정보
    # ==========================================================================
    with tab6:
        st.header("모델 정보")

        if api_online:
            model_info = api.get_models()

            if model_info:
                # 모델 목록
                st.subheader("로드된 모델")

                for model in model_info.get('models', []):
                    with st.container():
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"### {model['name']}")
                            st.caption(f"타입: {model['type'].upper()}")

                        with col2:
                            st.metric("피처 수", model.get('n_features', 'N/A'))
                            st.metric("시퀀스 길이", model.get('seq_length', 'N/A'))

                        with col3:
                            st.metric("Hidden Size", model.get('hidden_size', 'N/A'))
                            st.metric("레이어 수", model.get('num_layers', 'N/A'))

                        st.markdown("---")

                # 모델 비교 차트
                fig = Charts.create_model_performance_chart(model_info)
                st.plotly_chart(fig, width="stretch")

                st.info(f"기본 모델: **{model_info.get('default_model', 'conditional')}**")
        else:
            st.warning("API에 연결할 수 없습니다.")

    # ==========================================================================
    # 탭 7: 시스템 정보
    # ==========================================================================
    with tab7:
        st.header("시스템 정보")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("⚡ 전력 수요 예측 API")
            st.caption(f"URL: {Config.API_URL}")
            st.json(health)

        with col2:
            st.subheader("🌞🌬️ 신재생에너지 API")
            st.caption(f"URL: {Config.RENEWABLE_API_URL}")
            st.json(renewable_health)

        with col3:
            st.subheader("📊 데이터 정보")

            if historical_data is not None:
                st.markdown(f"""
                - **총 레코드**: {len(historical_data):,}
                - **기간**: {historical_data.index.min()} ~ {historical_data.index.max()}
                - **컬럼 수**: {len(historical_data.columns)}
                - **수요 범위**: {historical_data['power_demand'].min():.1f} ~ {historical_data['power_demand'].max():.1f} MW
                """)

            if st.button("🔄 새로고침"):
                st.rerun()

        st.markdown("---")

        # API 엔드포인트 - 두 API 모두 표시
        col_ep1, col_ep2 = st.columns(2)

        with col_ep1:
            st.subheader("⚡ 전력 수요 API 엔드포인트")
            st.caption(f"Base URL: {Config.API_URL}")

            demand_endpoints = [
                {"Method": "GET", "Endpoint": "/health", "설명": "상태 확인"},
                {"Method": "GET", "Endpoint": "/models", "설명": "모델 정보"},
                {"Method": "POST", "Endpoint": "/predict", "설명": "단일 예측"},
                {"Method": "POST", "Endpoint": "/predict/conditional", "설명": "조건부 예측"},
                {"Method": "POST", "Endpoint": "/predict/batch", "설명": "배치 예측"},
            ]
            st.dataframe(pd.DataFrame(demand_endpoints), width="stretch", hide_index=True)

        with col_ep2:
            st.subheader("🌞🌬️ 신재생에너지 API 엔드포인트")
            st.caption(f"Base URL: {Config.RENEWABLE_API_URL}")

            renewable_endpoints = [
                {"Method": "GET", "Endpoint": "/health", "설명": "상태 확인"},
                {"Method": "GET", "Endpoint": "/models", "설명": "모델 정보 (R², RMSE)"},
                {"Method": "POST", "Endpoint": "/predict", "설명": "단일 예측 (태양광/풍력)"},
                {"Method": "POST", "Endpoint": "/predict/batch", "설명": "배치 예측 (최대 168h)"},
            ]
            st.dataframe(pd.DataFrame(renewable_endpoints), width="stretch", hide_index=True)

        st.markdown("---")

        # 사용 가이드
        col_guide1, col_guide2 = st.columns(2)

        with col_guide1:
            with st.expander("💡 전력 수요 API 사용 예시"):
                st.code("""
import requests

# 1. 상태 확인
health = requests.get("http://localhost:8000/health").json()
print(f"Status: {health['status']}")

# 2. 조건부 예측
response = requests.post(
    "http://localhost:8000/predict/conditional",
    json={
        "data": [...],  # 168개 이상의 시계열 데이터
        "mode": "soft"
    }
)
result = response.json()
print(f"예측: {result['prediction']} MW")
print(f"모델: {result['model_used']}")
                """, language="python")

        with col_guide2:
            with st.expander("💡 신재생에너지 API 사용 예시"):
                st.code("""
import requests

# 1. 상태 확인
health = requests.get("http://localhost:8001/health").json()
print(f"Status: {health['status']}")
print(f"Models: {health['models_loaded']}")

# 2. 단일 예측
response = requests.post(
    "http://localhost:8001/predict",
    json={
        "weather": {
            "datetime": "2024-12-17T14:00:00",
            "temperature": 8.5,
            "humidity": 65.0,
            "wind_speed": 5.2,
            "wind_direction": 270.0,
            "pressure": 1013.5,
            "solar_radiation": 2.5
        },
        "energy_type": "both",
        "include_uncertainty": True
    }
)
result = response.json()
print(f"태양광: {result['predictions'][0]['prediction_mw']} MW")
print(f"풍력: {result['predictions'][1]['prediction_mw']} MW")
                """, language="python")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
