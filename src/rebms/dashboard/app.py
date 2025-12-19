"""
RE-BMS v5.0 Dashboard
=====================

Mobile-first command center for renewable energy bidding.

Features:
1. Command Center Dashboard (dark theme)
2. 10-Segment Bidding Matrix with step chart
3. Portfolio Management
4. Real-time Market Monitoring
5. Settlement Analytics

Usage:
    streamlit run src/rebms/dashboard/app.py --server.port 8507
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from uuid import uuid4

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import RE-BMS models
from src.rebms.models.bid import (
    BidSegment, HourlyBid, DailyBid, MarketType, BidStatus,
    create_optimized_segments
)
from src.rebms.validators.bid_validator import BidValidator


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RE-BMS v5.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# Dark Theme CSS
# ============================================================================

DARK_THEME_CSS = """
<style>
    /* Command Center Dark Theme */
    .stApp {
        background-color: #0e1117;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }

    .bid-status-draft {
        color: #fbbf24;
    }

    .bid-status-submitted {
        color: #10b981;
    }

    .bid-status-accepted {
        color: #3b82f6;
    }

    .segment-high { background-color: rgba(239, 68, 68, 0.2); }
    .segment-medium { background-color: rgba(251, 191, 36, 0.2); }
    .segment-low { background-color: rgba(34, 197, 94, 0.2); }

    /* Mobile optimizations */
    @media (max-width: 768px) {
        .metric-card { padding: 12px; }
        h1 { font-size: 1.5rem !important; }
    }

    /* Step chart container */
    .step-chart {
        background-color: #1a1f2c;
        border-radius: 8px;
        padding: 15px;
    }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'current_bid' not in st.session_state:
    st.session_state.current_bid = None

if 'bids' not in st.session_state:
    st.session_state.bids = {}

if 'resources' not in st.session_state:
    # Sample resources
    st.session_state.resources = {
        "solar-001": {"name": "Jeju Solar Plant A", "type": "solar", "capacity": 50},
        "wind-001": {"name": "Jeju Wind Farm B", "type": "wind", "capacity": 40},
        "solar-002": {"name": "Jeju Solar Plant C", "type": "solar", "capacity": 35},
    }


# ============================================================================
# Helper Functions
# ============================================================================

def get_smp_forecast() -> Dict[str, List[float]]:
    """Get SMP forecast from predictor or generate default"""
    try:
        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)
        predictions = predictor.predict_24h()

        q10 = predictions.get('q10', [90.0] * 24)
        q50 = predictions.get('q50', [100.0] * 24)
        q90 = predictions.get('q90', [110.0] * 24)

        if hasattr(q10, 'tolist'):
            q10 = q10.tolist()
        if hasattr(q50, 'tolist'):
            q50 = q50.tolist()
        if hasattr(q90, 'tolist'):
            q90 = q90.tolist()

        return {'q10': q10, 'q50': q50, 'q90': q90}

    except Exception:
        # Generate default pattern
        base = 95.0
        hours = range(24)
        q10 = [base + np.sin(h * np.pi / 12) * 10 - 10 for h in hours]
        q50 = [base + np.sin(h * np.pi / 12) * 10 for h in hours]
        q90 = [base + np.sin(h * np.pi / 12) * 10 + 10 for h in hours]
        return {'q10': q10, 'q50': q50, 'q90': q90}


def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    now = datetime.now()

    return {
        "dam": {
            "status": "open" if now.hour < 10 else "closed",
            "deadline": f"{(now + timedelta(days=1)).strftime('%Y-%m-%d')} 10:00",
            "trading_date": (now + timedelta(days=1)).date() if now.hour >= 10 else now.date(),
        },
        "rtm": {
            "status": "open",
            "next_interval": (now + timedelta(minutes=15 - now.minute % 15)).strftime("%H:%M"),
        },
    }


def create_step_chart(segments: List[BidSegment], smp_forecast: float = None) -> go.Figure:
    """Create step chart visualization for bid segments"""
    if not segments:
        return go.Figure()

    # Sort by segment_id
    segments = sorted(segments, key=lambda s: s.segment_id)

    # Calculate cumulative quantities
    quantities = [s.quantity_mw for s in segments]
    prices = [s.price_krw_mwh for s in segments]
    cumulative = np.cumsum([0] + quantities)

    fig = go.Figure()

    # Create step lines
    for i, seg in enumerate(segments):
        if seg.quantity_mw > 0:
            x_start = cumulative[i]
            x_end = cumulative[i + 1]

            # Horizontal line
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[prices[i], prices[i]],
                mode='lines',
                line=dict(color='#10b981', width=3),
                name=f'Seg {seg.segment_id}',
                showlegend=False,
                hovertemplate=f"Seg {seg.segment_id}: {seg.quantity_mw:.1f} MW @ {prices[i]:.1f} KRW<extra></extra>",
            ))

            # Vertical connector
            if i < len(segments) - 1 and segments[i + 1].quantity_mw > 0:
                fig.add_trace(go.Scatter(
                    x=[x_end, x_end],
                    y=[prices[i], prices[i + 1]],
                    mode='lines',
                    line=dict(color='#10b981', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    # Add SMP forecast line
    if smp_forecast:
        fig.add_hline(
            y=smp_forecast,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"SMP Forecast: ₩{smp_forecast:.1f}",
            annotation_position="right",
        )

    fig.update_layout(
        title="Bid Curve (Step Chart)",
        xaxis_title="Cumulative Quantity (MW)",
        yaxis_title="Price (KRW/MWh)",
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def create_smp_forecast_chart(forecast: Dict[str, List[float]]) -> go.Figure:
    """Create 24h SMP forecast chart"""
    hours = list(range(24))

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=forecast['q90'] + forecast['q10'][::-1],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% CI',
        hoverinfo='skip',
    ))

    # Q50 line
    fig.add_trace(go.Scatter(
        x=hours,
        y=forecast['q50'],
        mode='lines',
        line=dict(color='#6366f1', width=2),
        name='Forecast (Q50)',
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="Hour",
        yaxis_title="SMP (KRW/MWh)",
    )

    return fig


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main dashboard entry point"""

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("⚡ RE-BMS Command Center")

    with col2:
        market_status = get_market_status()
        status_color = "#10b981" if market_status["dam"]["status"] == "open" else "#ef4444"
        st.markdown(f"""
            <div style="text-align: right; padding-top: 15px;">
                <span style="color: {status_color}; font-size: 20px;">●</span>
                <span style="color: white;"> DAM: {market_status["dam"]["status"].upper()}</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="text-align: right; padding-top: 15px; color: white;">
                🕐 {datetime.now().strftime("%H:%M:%S")}
            </div>
        """, unsafe_allow_html=True)

    # Navigation tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "📝 Bidding",
        "🏭 Portfolio",
        "💰 Settlement",
    ])

    with tabs[0]:
        render_dashboard()

    with tabs[1]:
        render_bidding()

    with tabs[2]:
        render_portfolio()

    with tabs[3]:
        render_settlement()


def render_dashboard():
    """Command center dashboard"""

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_capacity = sum(r["capacity"] for r in st.session_state.resources.values())
    current_output = total_capacity * np.random.uniform(0.6, 0.8)

    with col1:
        st.metric(
            label="Portfolio Capacity",
            value=f"{total_capacity} MW",
            delta=f"+{len(st.session_state.resources)} resources",
        )

    with col2:
        st.metric(
            label="Current Output",
            value=f"{current_output:.1f} MW",
            delta=f"{(current_output/total_capacity*100):.1f}%",
        )

    with col3:
        revenue = current_output * 95 * 24 / 1000  # Approximate daily revenue in thousands
        st.metric(
            label="Est. Daily Revenue",
            value=f"₩{revenue:.1f}M",
            delta="+8.3%",
        )

    with col4:
        smp = 95 + np.random.randn() * 5
        st.metric(
            label="SMP (Current)",
            value=f"₩{smp:.1f}/kWh",
            delta=f"{np.random.randn()*2:.1f}%",
            delta_color="inverse" if np.random.random() > 0.5 else "normal",
        )

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("24h SMP Forecast")
        forecast = get_smp_forecast()
        fig = create_smp_forecast_chart(forecast)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Status")

        market_status = get_market_status()

        st.markdown(f"""
        **Day-Ahead Market (DAM)**
        - Status: **{market_status["dam"]["status"].upper()}**
        - Deadline: {market_status["dam"]["deadline"]}
        - Trading Date: {market_status["dam"]["trading_date"]}

        **Real-Time Market (RTM)**
        - Status: **{market_status["rtm"]["status"].upper()}**
        - Next Interval: {market_status["rtm"]["next_interval"]}
        """)

    # Active Bids Summary
    st.subheader("Active Bids")

    if st.session_state.bids:
        bids_data = []
        for bid_id, bid in st.session_state.bids.items():
            bids_data.append({
                "Bid ID": bid_id[:8] + "...",
                "Resource": bid.resource_id,
                "Date": str(bid.trading_date),
                "Status": bid.status.value,
                "Total MW": f"{bid.total_daily_mwh:.1f}",
            })

        df = pd.DataFrame(bids_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No active bids. Go to Bidding tab to create one.")


def render_bidding():
    """10-segment bidding interface"""

    st.subheader("📝 10-Segment Bidding Matrix")

    # Bid Configuration
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        trading_date = st.date_input(
            "Trading Date",
            value=datetime.now().date() + timedelta(days=1),
        )

    with col2:
        resource_options = list(st.session_state.resources.keys())
        resource_id = st.selectbox(
            "Resource",
            options=resource_options,
            format_func=lambda x: st.session_state.resources[x]["name"],
        )

    with col3:
        risk_level = st.select_slider(
            "Risk Level",
            options=["conservative", "moderate", "aggressive"],
            value="moderate",
        )

    with col4:
        capacity = st.session_state.resources[resource_id]["capacity"]
        st.metric("Capacity", f"{capacity} MW")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🆕 Create New Bid", type="primary"):
            bid = DailyBid.create_empty(
                bid_id=str(uuid4()),
                resource_id=resource_id,
                trading_date=trading_date,
            )
            bid.risk_level = risk_level
            st.session_state.current_bid = bid
            st.session_state.bids[bid.bid_id] = bid
            st.success(f"Created bid: {bid.bid_id[:8]}...")

    with col2:
        if st.button("🤖 AI Optimize"):
            if st.session_state.current_bid:
                with st.spinner("Optimizing with SMP forecast..."):
                    forecast = get_smp_forecast()
                    bid = st.session_state.current_bid

                    # Optimize each hour
                    for i, hb in enumerate(bid.hourly_bids):
                        hour = hb.hour
                        segments = create_optimized_segments(
                            total_capacity_mw=capacity,
                            smp_q10=forecast['q10'][hour - 1],
                            smp_q50=forecast['q50'][hour - 1],
                            smp_q90=forecast['q90'][hour - 1],
                            risk_level=risk_level,
                        )
                        bid.hourly_bids[i] = HourlyBid(hour=hour, segments=segments)

                    bid.smp_forecast_used = True
                    bid._calculate_summary()
                    st.success("Bid optimized!")
                    st.rerun()
            else:
                st.warning("Create a bid first")

    with col3:
        if st.button("✅ Submit to KPX"):
            if st.session_state.current_bid:
                bid = st.session_state.current_bid
                bid.status = BidStatus.SUBMITTED
                bid.submitted_at = datetime.now()
                bid.kpx_reference_id = f"KPX-{uuid4().hex[:8].upper()}"
                st.success(f"Submitted! Ref: {bid.kpx_reference_id}")
            else:
                st.warning("Create a bid first")

    st.divider()

    # Bid Visualization
    if st.session_state.current_bid:
        bid = st.session_state.current_bid

        st.markdown(f"""
        **Current Bid**: {bid.bid_id[:8]}... |
        **Status**: {bid.status.value} |
        **Total**: {bid.total_daily_mwh:.1f} MWh |
        **Avg Price**: ₩{bid.average_price:.1f}
        """)

        # Hour selector
        selected_hour = st.slider("Select Hour", 1, 24, 12)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Hour {selected_hour} Segments")

            hourly_bid = bid.get_hourly_bid(selected_hour)
            if hourly_bid and hourly_bid.segments:
                # Create editable dataframe
                seg_data = []
                for seg in hourly_bid.segments:
                    seg_data.append({
                        "Segment": seg.segment_id,
                        "Quantity (MW)": seg.quantity_mw,
                        "Price (KRW)": seg.price_krw_mwh,
                    })

                df = pd.DataFrame(seg_data)
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Segment"],
                )

                # Update segments if edited
                if not df.equals(edited_df):
                    new_segments = []
                    for _, row in edited_df.iterrows():
                        new_segments.append(BidSegment(
                            segment_id=int(row["Segment"]),
                            quantity_mw=float(row["Quantity (MW)"]),
                            price_krw_mwh=float(row["Price (KRW)"]),
                        ))
                    bid.update_hourly_bid(selected_hour, HourlyBid(hour=selected_hour, segments=new_segments))
            else:
                st.info("No segments for this hour")

        with col2:
            st.subheader("Step Chart")

            if hourly_bid and hourly_bid.segments:
                forecast = get_smp_forecast()
                smp_forecast = forecast['q50'][selected_hour - 1]
                fig = create_step_chart(hourly_bid.segments, smp_forecast)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Optimize bid to see step chart")

    else:
        st.info("Create a new bid to start")


def render_portfolio():
    """Portfolio management view"""

    st.subheader("🏭 Resource Portfolio")

    # Resource cards
    cols = st.columns(3)

    for i, (resource_id, resource) in enumerate(st.session_state.resources.items()):
        with cols[i % 3]:
            output = resource["capacity"] * np.random.uniform(0.5, 0.9)
            utilization = (output / resource["capacity"]) * 100

            st.markdown(f"""
            <div class="metric-card">
                <h4>{resource["name"]}</h4>
                <p>Type: {resource["type"].title()} | <span style="color: #10b981;">●</span> Online</p>
                <p>Capacity: {resource["capacity"]} MW</p>
                <p>Current: {output:.1f} MW ({utilization:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Add resource form
    st.subheader("Add New Resource")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        new_id = st.text_input("Resource ID", value=f"res-{len(st.session_state.resources)+1:03d}")

    with col2:
        new_name = st.text_input("Name", value="New Resource")

    with col3:
        new_type = st.selectbox("Type", options=["solar", "wind"])

    with col4:
        new_capacity = st.number_input("Capacity (MW)", min_value=1, max_value=500, value=50)

    if st.button("➕ Add Resource"):
        st.session_state.resources[new_id] = {
            "name": new_name,
            "type": new_type,
            "capacity": new_capacity,
        }
        st.success(f"Added resource: {new_name}")
        st.rerun()


def render_settlement():
    """Settlement analytics view"""

    st.subheader("💰 Settlement & Penalties")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Generation Revenue", "₩145.2M", "+12.3%")

    with col2:
        st.metric("Imbalance Charges", "₩3.5M", "-15.2%", delta_color="inverse")

    with col3:
        st.metric("Net Revenue", "₩141.7M", "+14.1%")

    with col4:
        st.metric("Forecast Accuracy", "94.2%", "+2.1%")

    st.divider()

    # Settlement table
    st.markdown("**Recent Settlements**")

    dates = pd.date_range(end=datetime.now().date(), periods=7, freq='D')
    data = {
        "Date": dates.strftime("%Y-%m-%d"),
        "Generation (MWh)": np.random.uniform(800, 1200, 7).round(1),
        "Revenue (₩M)": np.random.uniform(18, 25, 7).round(2),
        "Imbalance (₩M)": np.random.uniform(0.2, 1.5, 7).round(2),
        "Accuracy (%)": np.random.uniform(90, 98, 7).round(1),
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Imbalance chart
    st.subheader("Imbalance Analysis")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

    # Penalty distribution pie
    fig.add_trace(
        go.Pie(
            labels=["No Penalty", "Over-Gen", "Under-Gen"],
            values=[18, 3, 3],
            marker_colors=["#10b981", "#3b82f6", "#ef4444"],
            hole=0.4,
        ),
        row=1, col=1
    )

    # Daily imbalance bar
    fig.add_trace(
        go.Bar(
            x=list(range(1, 25)),
            y=np.random.uniform(-5, 5, 24),
            marker_color=["#10b981" if x > 0 else "#ef4444" for x in np.random.uniform(-5, 5, 24)],
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    main()
