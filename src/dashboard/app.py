"""
제주도 전력 수요 예측 대시보드 (API 연동 버전)
==============================================

FastAPI 서버와 연동하여 실시간 예측을 제공하는 Streamlit 대시보드

주요 기능:
1. 실시간 예측 차트 (24시간) - API 연동
2. 기상 조건 입력 인터페이스
3. 시나리오 분석 (폭염/한파)
4. 과거 데이터 비교
5. 모델 성능 지표

Usage:
    # API 서버 먼저 실행
    uvicorn api.main:app --host 0.0.0.0 --port 8000

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
    API_URL = "http://localhost:8000"
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


# ============================================================================
# 메인 대시보드
# ============================================================================

def main():
    """메인 함수"""

    # API 클라이언트 초기화
    api = APIClient()

    # 헤더
    st.markdown('<p class="main-header">⚡ 제주도 전력 수요 예측 시스템</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">FastAPI 연동 | 실시간 예측 | 시나리오 분석</p>', unsafe_allow_html=True)

    # API 상태 확인
    health = api.health_check()
    api_online = health.get("status") == "healthy"

    # 상단 상태 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if api_online:
            st.markdown('<div class="api-connected">🟢 API 연결됨</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-disconnected">🔴 API 오프라인</div>', unsafe_allow_html=True)

    with col2:
        st.metric("디바이스", health.get("device", "N/A"))

    with col3:
        st.metric("모델 로드", "✅" if health.get("models_loaded") else "❌")

    with col4:
        uptime = health.get("uptime_seconds", 0)
        st.metric("업타임", f"{uptime/60:.1f}분")

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
        st.caption(f"API: {Config.API_URL}")

    # 데이터 로드
    historical_data = load_historical_data()

    if historical_data is None or len(historical_data) == 0:
        st.error("과거 데이터를 로드할 수 없습니다.")
        return

    st.success(f"데이터 로드: {len(historical_data):,}개 레코드 (2013-2024)")

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 실시간 예측",
        "🌡️ 시나리오 분석",
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

                if st.button("🚀 예측 실행", type="primary", use_container_width=True):
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
                    st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(fig, use_container_width=True)

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
                            use_container_width=True,
                            hide_index=True
                        )

    # ==========================================================================
    # 탭 3: 과거 데이터
    # ==========================================================================
    with tab3:
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
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # 시간대별 패턴
                fig = Charts.create_hourly_pattern_chart(filtered_data)
                st.plotly_chart(fig, use_container_width=True)

            # 상세 데이터
            with st.expander("상세 데이터 보기"):
                st.dataframe(
                    filtered_data[['power_demand', '기온', '습도', '풍속']].round(2),
                    use_container_width=True
                )
        else:
            st.warning("선택한 기간에 데이터가 없습니다.")

    # ==========================================================================
    # 탭 4: 모델 정보
    # ==========================================================================
    with tab4:
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
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"기본 모델: **{model_info.get('default_model', 'conditional')}**")
        else:
            st.warning("API에 연결할 수 없습니다.")

    # ==========================================================================
    # 탭 5: 시스템 정보
    # ==========================================================================
    with tab5:
        st.header("시스템 정보")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("API 서버")

            st.json(health)

            if st.button("새로고침"):
                st.rerun()

        with col2:
            st.subheader("데이터 정보")

            if historical_data is not None:
                st.markdown(f"""
                - **총 레코드**: {len(historical_data):,}
                - **기간**: {historical_data.index.min()} ~ {historical_data.index.max()}
                - **컬럼 수**: {len(historical_data.columns)}
                - **수요 범위**: {historical_data['power_demand'].min():.1f} ~ {historical_data['power_demand'].max():.1f} MW
                """)

        # API 엔드포인트
        st.subheader("API 엔드포인트")

        endpoints = [
            {"Method": "GET", "Endpoint": "/health", "설명": "상태 확인"},
            {"Method": "GET", "Endpoint": "/models", "설명": "모델 정보"},
            {"Method": "POST", "Endpoint": "/predict", "설명": "단일 예측"},
            {"Method": "POST", "Endpoint": "/predict/conditional", "설명": "조건부 예측"},
            {"Method": "POST", "Endpoint": "/predict/batch", "설명": "배치 예측"},
        ]

        st.dataframe(pd.DataFrame(endpoints), use_container_width=True, hide_index=True)

        # 사용 가이드
        with st.expander("API 사용 예시"):
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


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
