"""
대시보드 UI (Task 14)
====================
전력 수요 예측 시각화 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DashboardConfig:
    """대시보드 설정"""
    api_url: str = "http://localhost:8000"
    refresh_interval: int = 60  # seconds
    default_location: str = "jeju"
    theme: str = "light"
    chart_height: int = 400
    max_history_days: int = 365


def get_config() -> DashboardConfig:
    """설정 로드"""
    return DashboardConfig()


# ============================================================================
# Data Fetchers
# ============================================================================

class DataFetcher:
    """데이터 수집기"""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self._cache: Dict[str, Any] = {}

    def get_predictions(
        self,
        location: str = "jeju",
        horizons: List[str] = None,
        model_type: str = "ensemble"
    ) -> Optional[Dict]:
        """예측 데이터 조회"""
        if horizons is None:
            horizons = ["1h", "6h", "24h"]

        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={
                    "location": location,
                    "horizons": horizons,
                    "model_type": model_type
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Prediction fetch failed: {e}")
            return None

    def get_historical_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        resolution: str = "hourly"
    ) -> Optional[Dict]:
        """과거 데이터 조회"""
        try:
            response = requests.get(
                f"{self.api_url}/data/historical",
                params={
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date,
                    "resolution": resolution
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            return None

    def get_health(self) -> Optional[Dict]:
        """상태 확인"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def get_models(self) -> Optional[Dict]:
        """모델 목록"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def get_metrics(self) -> Optional[Dict]:
        """메트릭 조회"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


# ============================================================================
# Mock Data Generator (for offline mode)
# ============================================================================

class MockDataGenerator:
    """오프라인 모드용 모의 데이터 생성기"""

    @staticmethod
    def generate_predictions(
        horizons: List[str] = None
    ) -> Dict:
        """예측 데이터 생성"""
        if horizons is None:
            horizons = ["1h", "6h", "24h"]

        now = datetime.now()
        predictions = []

        for horizon in horizons:
            hours = int(horizon.replace("h", ""))
            target_time = now + timedelta(hours=hours)

            # 시간대별 수요 패턴 시뮬레이션
            hour = target_time.hour
            base_demand = 800 + 200 * np.sin(hour * np.pi / 12)
            variation = np.random.randn() * 30

            pred_value = base_demand + variation
            std = 30 + hours * 2

            predictions.append({
                "timestamp": target_time.isoformat(),
                "horizon": horizon,
                "prediction": round(pred_value, 2),
                "lower_bound": round(pred_value - 1.96 * std, 2),
                "upper_bound": round(pred_value + 1.96 * std, 2),
                "confidence": round(0.95 - hours * 0.01, 3)
            })

        return {
            "request_id": "mock123",
            "location": "jeju",
            "model_type": "ensemble",
            "created_at": now.isoformat(),
            "predictions": predictions
        }

    @staticmethod
    def generate_historical_data(
        start_date: date,
        end_date: date,
        resolution: str = "hourly"
    ) -> Dict:
        """과거 데이터 생성"""
        data = []
        current = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())

        delta = timedelta(hours=1) if resolution == "hourly" else timedelta(days=1)

        while current <= end:
            hour = current.hour
            day_of_year = current.timetuple().tm_yday

            # 계절 패턴
            seasonal = 100 * np.sin(2 * np.pi * day_of_year / 365)
            # 일간 패턴
            daily = 200 * np.sin((hour - 6) * np.pi / 12)
            # 기본 수요
            base = 850

            demand = base + seasonal + daily + np.random.randn() * 30
            temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.randn() * 3
            humidity = 60 + np.random.randn() * 10

            data.append({
                "timestamp": current.isoformat(),
                "demand": round(demand, 2),
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1)
            })
            current += delta

        return {
            "location": "jeju",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data": data,
            "count": len(data)
        }


# ============================================================================
# Chart Components
# ============================================================================

class ChartFactory:
    """차트 생성 팩토리"""

    @staticmethod
    def create_prediction_chart(predictions: List[Dict], height: int = 400) -> go.Figure:
        """예측 차트 생성"""
        if not predictions:
            return go.Figure()

        timestamps = [p["timestamp"] for p in predictions]
        values = [p["prediction"] for p in predictions]
        lower = [p.get("lower_bound", p["prediction"] - 50) for p in predictions]
        upper = [p.get("upper_bound", p["prediction"] + 50) for p in predictions]

        fig = go.Figure()

        # 신뢰 구간
        fig.add_trace(go.Scatter(
            x=timestamps + timestamps[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% 신뢰구간",
            showlegend=True
        ))

        # 예측값
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines+markers",
            name="예측값",
            line=dict(color="blue", width=2),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="전력 수요 예측",
            xaxis_title="시간",
            yaxis_title="수요 (MW)",
            height=height,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig

    @staticmethod
    def create_historical_chart(
        data: List[Dict],
        height: int = 400
    ) -> go.Figure:
        """과거 데이터 차트 생성"""
        if not data:
            return go.Figure()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("전력 수요", "기온")
        )

        # 전력 수요
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["demand"],
                mode="lines",
                name="수요",
                line=dict(color="blue")
            ),
            row=1, col=1
        )

        # 기온
        if "temperature" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["temperature"],
                    mode="lines",
                    name="기온",
                    line=dict(color="red")
                ),
                row=2, col=1
            )

        fig.update_layout(height=height, showlegend=True)
        fig.update_yaxes(title_text="MW", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)

        return fig

    @staticmethod
    def create_demand_pattern_chart(
        data: List[Dict],
        height: int = 400
    ) -> go.Figure:
        """수요 패턴 분석 차트"""
        if not data:
            return go.Figure()

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # 시간대별 평균
        hourly_avg = df.groupby("hour")["demand"].mean()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=hourly_avg.index,
            y=hourly_avg.values,
            name="시간대별 평균",
            marker_color="steelblue"
        ))

        fig.update_layout(
            title="시간대별 평균 전력 수요",
            xaxis_title="시간",
            yaxis_title="평균 수요 (MW)",
            height=height
        )

        return fig

    @staticmethod
    def create_model_comparison_chart(
        models: List[Dict],
        height: int = 300
    ) -> go.Figure:
        """모델 비교 차트"""
        if not models:
            return go.Figure()

        names = [m["name"] for m in models]
        rmse_values = [m.get("metrics", {}).get("rmse", 0) for m in models]
        mape_values = [m.get("metrics", {}).get("mape", 0) for m in models]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "MAPE (%)"))

        fig.add_trace(
            go.Bar(x=names, y=rmse_values, name="RMSE", marker_color="steelblue"),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=names, y=mape_values, name="MAPE", marker_color="indianred"),
            row=1, col=2
        )

        fig.update_layout(height=height, showlegend=False)

        return fig

    @staticmethod
    def create_gauge_chart(
        value: float,
        title: str,
        max_value: float = 100,
        height: int = 200
    ) -> go.Figure:
        """게이지 차트 생성"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "lightgreen"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "yellow"},
                    {"range": [max_value * 0.8, max_value], "color": "red"}
                ]
            }
        ))

        fig.update_layout(height=height)
        return fig


# ============================================================================
# Dashboard Components
# ============================================================================

class DashboardComponents:
    """대시보드 컴포넌트"""

    @staticmethod
    def render_header():
        """헤더 렌더링"""
        st.title("🔌 제주도 전력 수요 예측 대시보드")
        st.markdown("---")

    @staticmethod
    def render_sidebar(config: DashboardConfig) -> Dict:
        """사이드바 렌더링"""
        st.sidebar.title("⚙️ 설정")

        # 위치 선택
        location = st.sidebar.selectbox(
            "위치",
            ["jeju", "seoul", "busan"],
            index=0
        )

        # 모델 선택
        model_type = st.sidebar.selectbox(
            "모델",
            ["ensemble", "lstm", "tft"],
            index=0
        )

        # 예측 시간대
        horizons = st.sidebar.multiselect(
            "예측 시간대",
            ["1h", "6h", "12h", "24h", "48h"],
            default=["1h", "6h", "24h"]
        )

        # 날짜 범위
        st.sidebar.subheader("과거 데이터")
        date_range = st.sidebar.date_input(
            "날짜 범위",
            value=(
                datetime.now().date() - timedelta(days=7),
                datetime.now().date()
            )
        )

        # 해상도
        resolution = st.sidebar.radio(
            "해상도",
            ["hourly", "daily"],
            horizontal=True
        )

        st.sidebar.markdown("---")
        st.sidebar.info(f"API: {config.api_url}")

        return {
            "location": location,
            "model_type": model_type,
            "horizons": horizons,
            "date_range": date_range,
            "resolution": resolution
        }

    @staticmethod
    def render_status_cards(health: Optional[Dict], metrics: Optional[Dict]):
        """상태 카드 렌더링"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = health.get("status", "unknown") if health else "offline"
            color = "🟢" if status == "healthy" else "🔴"
            st.metric("상태", f"{color} {status}")

        with col2:
            uptime = health.get("uptime", 0) if health else 0
            hours = uptime / 3600
            st.metric("업타임", f"{hours:.1f}h")

        with col3:
            total_preds = metrics.get("total_predictions", 0) if metrics else 0
            st.metric("총 예측 수", total_preds)

        with col4:
            models_loaded = health.get("models_loaded", 0) if health else 0
            st.metric("로드된 모델", models_loaded)

    @staticmethod
    def render_prediction_section(
        predictions: Optional[Dict],
        chart_factory: ChartFactory,
        height: int = 400
    ):
        """예측 섹션 렌더링"""
        st.subheader("📈 전력 수요 예측")

        if predictions and predictions.get("predictions"):
            # 예측 차트
            fig = chart_factory.create_prediction_chart(
                predictions["predictions"],
                height=height
            )
            st.plotly_chart(fig, use_container_width=True)

            # 예측 테이블
            with st.expander("상세 데이터"):
                df = pd.DataFrame(predictions["predictions"])
                st.dataframe(df)

            # 메타데이터
            st.caption(
                f"모델: {predictions.get('model_type')} | "
                f"생성: {predictions.get('created_at')} | "
                f"ID: {predictions.get('request_id')}"
            )
        else:
            st.warning("예측 데이터를 불러올 수 없습니다.")

    @staticmethod
    def render_historical_section(
        historical_data: Optional[Dict],
        chart_factory: ChartFactory,
        height: int = 400
    ):
        """과거 데이터 섹션 렌더링"""
        st.subheader("📊 과거 데이터")

        if historical_data and historical_data.get("data"):
            # 과거 데이터 차트
            tab1, tab2 = st.tabs(["시계열", "패턴 분석"])

            with tab1:
                fig = chart_factory.create_historical_chart(
                    historical_data["data"],
                    height=height
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = chart_factory.create_demand_pattern_chart(
                    historical_data["data"],
                    height=height
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption(f"데이터 수: {historical_data.get('count', 0)}개")
        else:
            st.warning("과거 데이터를 불러올 수 없습니다.")

    @staticmethod
    def render_model_section(
        models: Optional[Dict],
        chart_factory: ChartFactory
    ):
        """모델 섹션 렌더링"""
        st.subheader("🤖 모델 정보")

        if models and models.get("models"):
            # 모델 비교 차트
            fig = chart_factory.create_model_comparison_chart(
                models["models"],
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            # 모델 상세
            with st.expander("모델 상세 정보"):
                for model in models["models"]:
                    st.write(f"**{model['name']}** ({model['type']})")
                    st.write(f"- 버전: {model.get('version', 'N/A')}")
                    st.write(f"- 상태: {model.get('status', 'N/A')}")
                    st.write(f"- 메트릭: {model.get('metrics', {})}")
                    st.markdown("---")
        else:
            st.warning("모델 정보를 불러올 수 없습니다.")


# ============================================================================
# Main Dashboard
# ============================================================================

class Dashboard:
    """메인 대시보드 클래스"""

    def __init__(self, config: DashboardConfig = None):
        self.config = config or get_config()
        self.fetcher = DataFetcher(self.config.api_url)
        self.mock_generator = MockDataGenerator()
        self.chart_factory = ChartFactory()
        self.components = DashboardComponents()
        self._use_mock = False

    def check_api_status(self) -> bool:
        """API 상태 확인"""
        health = self.fetcher.get_health()
        return health is not None

    def run(self):
        """대시보드 실행"""
        # 페이지 설정
        st.set_page_config(
            page_title="전력 수요 예측",
            page_icon="🔌",
            layout="wide"
        )

        # 헤더
        self.components.render_header()

        # 사이드바
        settings = self.components.render_sidebar(self.config)

        # API 상태 확인
        if not self.check_api_status():
            st.warning("⚠️ API 서버에 연결할 수 없습니다. 모의 데이터를 사용합니다.")
            self._use_mock = True

        # 데이터 로드
        if self._use_mock:
            health = {"status": "mock", "uptime": 0, "models_loaded": 0}
            metrics = {"total_predictions": 0}
            predictions = self.mock_generator.generate_predictions(settings["horizons"])
            historical = self.mock_generator.generate_historical_data(
                settings["date_range"][0],
                settings["date_range"][1],
                settings["resolution"]
            )
            models = {
                "models": [
                    {"name": "lstm_v1", "type": "lstm", "metrics": {"rmse": 45, "mape": 3.5}, "status": "mock"},
                    {"name": "ensemble_v1", "type": "ensemble", "metrics": {"rmse": 40, "mape": 3.0}, "status": "mock"}
                ],
                "count": 2
            }
        else:
            health = self.fetcher.get_health()
            metrics = self.fetcher.get_metrics()
            predictions = self.fetcher.get_predictions(
                location=settings["location"],
                horizons=settings["horizons"],
                model_type=settings["model_type"]
            )
            historical = self.fetcher.get_historical_data(
                location=settings["location"],
                start_date=settings["date_range"][0].isoformat(),
                end_date=settings["date_range"][1].isoformat(),
                resolution=settings["resolution"]
            )
            models = self.fetcher.get_models()

        # 상태 카드
        self.components.render_status_cards(health, metrics)

        st.markdown("---")

        # 메인 컨텐츠
        col1, col2 = st.columns([2, 1])

        with col1:
            self.components.render_prediction_section(
                predictions,
                self.chart_factory,
                self.config.chart_height
            )

        with col2:
            self.components.render_model_section(
                models,
                self.chart_factory
            )

        st.markdown("---")

        # 과거 데이터 섹션
        self.components.render_historical_section(
            historical,
            self.chart_factory,
            self.config.chart_height
        )

        # 푸터
        st.markdown("---")
        st.caption(
            f"📅 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"🔄 자동 새로고침: {self.config.refresh_interval}초"
        )


def main():
    """메인 함수"""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
