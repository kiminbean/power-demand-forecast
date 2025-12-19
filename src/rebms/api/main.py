"""
RE-BMS v5.0 API
===============

Mobile-first REST API for Renewable Energy Bidding Management System.

Features:
- 10-segment bidding with validation
- AI-powered bid optimization
- Market status and SMP forecast
- Settlement management

Usage:
    uvicorn src.rebms.api.main:app --host 0.0.0.0 --port 8506 --reload
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import bids_router
from .schemas import HealthResponse, MarketStatusResponse, SMPForecastResponse


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="RE-BMS v5.0 API",
        description="""
## Renewable Energy Bidding Management System

Mobile-first API for KPX Day-Ahead and Real-Time Market bidding.

### Features
- **10-Segment Bidding**: Create and manage bids with monotonic price constraint
- **AI Optimization**: Optimize bids using SMP forecast predictions
- **Market Integration**: Submit bids to KPX (simulation mode)
- **Settlement Tracking**: Track revenue and imbalance penalties

### Markets
- **DAM (Day-Ahead Market)**: Submit by D-1 10:00, hourly granularity
- **RTM (Real-Time Market)**: Submit 15 min before, 15-minute granularity

### Quick Start
1. Create bid template: `POST /api/v1/bids/template/{resource_id}`
2. Optimize with AI: `POST /api/v1/bids/{bid_id}/optimize`
3. Validate: `POST /api/v1/bids/{bid_id}/validate`
4. Submit: `POST /api/v1/bids/{bid_id}/submit`
        """,
        version="5.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Bidding", "description": "10-segment bid management"},
            {"name": "Market", "description": "Market status and SMP data"},
            {"name": "Health", "description": "API health and status"},
        ],
    )

    # CORS for mobile apps
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.start_time = datetime.now()
    app.state.request_count = 0

    # Include routers
    app.include_router(
        bids_router,
        prefix="/api/v1/bids",
        tags=["Bidding"],
    )

    return app


app = create_app()


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns API status, version, and uptime.
    """
    uptime = (datetime.now() - app.state.start_time).total_seconds()

    return HealthResponse(
        status="healthy",
        version="5.0.0",
        uptime=uptime,
        active_connections=0,  # TODO: Track WebSocket connections
    )


@app.get("/api/v1/market-status", response_model=MarketStatusResponse, tags=["Market"])
async def get_market_status():
    """
    Get current market status and deadlines

    Returns DAM/RTM status, deadlines, and trading dates.
    """
    now = datetime.now()

    # DAM deadline: D-1 10:00
    dam_deadline = datetime.combine(
        (now + timedelta(days=1)).date(),
        datetime.strptime("10:00", "%H:%M").time()
    )

    # Trading date for DAM
    if now.hour < 10:
        dam_trading_date = now.date()
    else:
        dam_trading_date = (now + timedelta(days=1)).date()

    # RTM next interval (15-minute)
    minutes_to_next = 15 - (now.minute % 15)
    rtm_next_interval = now + timedelta(minutes=minutes_to_next)

    return MarketStatusResponse(
        current_time=now.isoformat(),
        dam={
            "status": "open" if now.hour < 10 else "closed",
            "deadline": dam_deadline.strftime("%Y-%m-%d %H:%M"),
            "trading_date": dam_trading_date.isoformat(),
            "hours_remaining": max(0, (dam_deadline - now).total_seconds() / 3600),
        },
        rtm={
            "status": "open",
            "next_interval": rtm_next_interval.strftime("%Y-%m-%d %H:%M"),
            "interval_minutes": 15,
        },
    )


@app.get("/api/v1/smp-forecast", response_model=SMPForecastResponse, tags=["Market"])
async def get_smp_forecast():
    """
    Get 24-hour SMP forecast

    Returns quantile predictions (q10, q50, q90) from the SMP model.
    Falls back to default values if model is unavailable.
    """
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)
        predictions = predictor.predict_24h()

        q10 = predictions.get('q10', [90.0] * 24)
        q50 = predictions.get('q50', [100.0] * 24)
        q90 = predictions.get('q90', [110.0] * 24)

        # Convert to lists if numpy arrays
        if hasattr(q10, 'tolist'):
            q10 = q10.tolist()
        if hasattr(q50, 'tolist'):
            q50 = q50.tolist()
        if hasattr(q90, 'tolist'):
            q90 = q90.tolist()

        model_used = predictions.get('model_used', 'smp_v3.1')
        confidence = 0.85

    except Exception as e:
        # Fallback to default pattern
        import numpy as np

        base = 95.0
        q10 = [base + np.sin(h * np.pi / 12) * 10 - 10 for h in range(24)]
        q50 = [base + np.sin(h * np.pi / 12) * 10 for h in range(24)]
        q90 = [base + np.sin(h * np.pi / 12) * 10 + 10 for h in range(24)]

        model_used = "fallback_pattern"
        confidence = 0.5

    return SMPForecastResponse(
        q10=q10,
        q50=q50,
        q90=q90,
        model_used=model_used,
        confidence=confidence,
        created_at=datetime.now().isoformat(),
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "code": "INTERNAL_ERROR",
        },
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

# ============================================================================
# Real Jeju Power Plant Data Loader
# ============================================================================

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
JEJU_PLANTS_CSV = PROJECT_ROOT / "data/jeju_plants/jeju_power_plants.csv"

def load_jeju_plants():
    """Load real Jeju power plant data from CSV"""
    try:
        df = pd.read_csv(JEJU_PLANTS_CSV)
        # Filter only renewable energy (wind, solar) with status 운영중
        renewables = df[
            (df['type'].isin(['wind', 'solar'])) &
            (df['status'] == '운영중')
        ].copy()

        plants = []
        for _, row in renewables.iterrows():
            plants.append({
                "id": row['id'],
                "name": row['name'],
                "name_en": row['name_en'],
                "type": row['type'],
                "subtype": row['subtype'],
                "capacity": float(row['capacity_mw']),
                "operator": row['operator'],
                "location": row['address'],
                "latitude": float(row['latitude']),
                "longitude": float(row['longitude']),
            })
        return plants
    except Exception as e:
        print(f"Error loading Jeju plants: {e}")
        return []

# Cache the plants data
_jeju_plants_cache = None

def get_jeju_plants():
    """Get cached Jeju plants data"""
    global _jeju_plants_cache
    if _jeju_plants_cache is None:
        _jeju_plants_cache = load_jeju_plants()
    return _jeju_plants_cache


# ============================================================================
# Mobile API Endpoints
# ============================================================================

@app.get("/api/v1/dashboard/kpis", tags=["Market"])
async def get_dashboard_kpis():
    """Get dashboard KPI metrics for mobile app (using real Jeju plant data)"""
    import numpy as np

    # Load real Jeju power plants
    plants = get_jeju_plants()
    if not plants:
        # Fallback to sample data
        plants = [
            {"id": "solar-001", "name": "Jeju Solar Plant A", "type": "solar", "capacity": 50},
            {"id": "wind-001", "name": "Jeju Wind Farm B", "type": "wind", "capacity": 40},
        ]

    total_capacity = sum(p["capacity"] for p in plants)
    utilization = np.random.uniform(0.65, 0.85)
    current_output = total_capacity * utilization

    # Get current SMP from predictor
    current_smp = 95.0
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)
        result = predictor.predict_24h()
        current_smp = float(result['q50'][0])
    except:
        pass

    daily_revenue = current_output * current_smp * 24 / 1000

    return {
        "total_capacity_mw": total_capacity,
        "current_output_mw": round(current_output, 1),
        "utilization_pct": round(utilization * 100, 1),
        "daily_revenue_million": round(daily_revenue, 2),
        "revenue_change_pct": round(np.random.uniform(5, 15), 1),
        "current_smp": round(current_smp, 1),
        "smp_change_pct": round(np.random.uniform(-5, 5), 1),
        "resource_count": len(plants),
        "wind_count": len([p for p in plants if p['type'] == 'wind']),
        "solar_count": len([p for p in plants if p['type'] == 'solar']),
    }


@app.get("/api/v1/resources", tags=["Market"])
async def get_resources():
    """Get all resources in portfolio (real Jeju power plants)"""
    import numpy as np

    # Load real Jeju power plants
    plants = get_jeju_plants()
    if not plants:
        return []

    result = []
    for p in plants:
        # Simulate utilization based on type
        if p['type'] == 'solar':
            # Solar: higher during day (assume current time affects this)
            from datetime import datetime
            hour = datetime.now().hour
            if 6 <= hour <= 18:
                utilization = np.random.uniform(0.4, 0.8)
            else:
                utilization = np.random.uniform(0.0, 0.1)
        else:
            # Wind: more variable
            utilization = np.random.uniform(0.3, 0.9)

        result.append({
            "id": p["id"],
            "name": p["name"],
            "name_en": p.get("name_en", p["name"]),
            "type": p["type"],
            "subtype": p.get("subtype", "unknown"),
            "capacity": p["capacity"],
            "current_output": round(p["capacity"] * utilization, 2),
            "utilization": round(utilization * 100, 1),
            "status": "online",
            "location": p["location"],
            "operator": p.get("operator", "Unknown"),
            "latitude": p.get("latitude"),
            "longitude": p.get("longitude"),
        })

    # Sort by capacity descending
    result.sort(key=lambda x: x['capacity'], reverse=True)
    return result


@app.get("/api/v1/bidding/optimized-segments", tags=["Bidding"])
async def get_optimized_segments(
    capacity_mw: float = 50.0,
    risk_level: str = "moderate",
):
    """Get AI-optimized bid segments for all 24 hours"""
    import numpy as np
    import sys
    from pathlib import Path

    # Get SMP forecast
    q10, q50, q90 = [], [], []
    model_version = "fallback"

    try:
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)
        result = predictor.predict_24h()
        q10 = result['q10'].tolist() if hasattr(result['q10'], 'tolist') else list(result['q10'])
        q50 = result['q50'].tolist() if hasattr(result['q50'], 'tolist') else list(result['q50'])
        q90 = result['q90'].tolist() if hasattr(result['q90'], 'tolist') else list(result['q90'])
        model_version = result.get('model_used', 'v3.1')
    except:
        q10 = [85 + np.sin(h * np.pi / 12) * 10 for h in range(24)]
        q50 = [95 + np.sin(h * np.pi / 12) * 10 for h in range(24)]
        q90 = [105 + np.sin(h * np.pi / 12) * 10 for h in range(24)]

    hourly_bids = []

    for hour in range(24):
        segments = []
        price_min, price_mid, price_max = q10[hour], q50[hour], q90[hour]

        if risk_level == "conservative":
            qty_weights = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
        elif risk_level == "aggressive":
            qty_weights = [0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        else:
            qty_weights = [0.10] * 10

        for seg_id in range(1, 11):
            seg_ratio = (seg_id - 1) / 9
            price = price_min + (price_max - price_min) * seg_ratio
            qty = capacity_mw * qty_weights[seg_id - 1]
            segments.append({
                "segment_id": seg_id,
                "quantity_mw": round(qty, 2),
                "price_krw_mwh": round(price, 1),
            })

        total_mw = sum(s["quantity_mw"] for s in segments)
        avg_price = sum(s["price_krw_mwh"] * s["quantity_mw"] for s in segments) / total_mw if total_mw > 0 else 0

        hourly_bids.append({
            "hour": hour + 1,
            "segments": segments,
            "total_mw": round(total_mw, 2),
            "avg_price": round(avg_price, 1),
            "smp_forecast": {"q10": round(q10[hour], 1), "q50": round(q50[hour], 1), "q90": round(q90[hour], 1)}
        })

    return {
        "trading_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        "capacity_mw": capacity_mw,
        "risk_level": risk_level,
        "hourly_bids": hourly_bids,
        "total_daily_mwh": round(sum(b["total_mw"] for b in hourly_bids), 1),
        "model_used": model_version,
    }


@app.get("/api/v1/settlements/recent", tags=["Market"])
async def get_recent_settlements(days: int = 7):
    """Get recent settlement summaries"""
    import numpy as np

    settlements = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        generation = np.random.uniform(800, 1200)
        smp = np.random.uniform(85, 110)
        revenue = generation * smp / 1000
        imbalance = np.random.uniform(0.5, 2.5)

        settlements.append({
            "date": date,
            "generation_mwh": round(generation, 1),
            "revenue_million": round(revenue, 2),
            "imbalance_million": round(imbalance, 2),
            "net_revenue_million": round(revenue - imbalance, 2),
            "accuracy_pct": round(np.random.uniform(90, 98), 1),
        })
    return settlements


@app.get("/api/v1/settlements/summary", tags=["Market"])
async def get_settlement_summary():
    """Get settlement summary statistics"""
    import numpy as np

    return {
        "generation_revenue_million": round(np.random.uniform(140, 150), 1),
        "generation_change_pct": round(np.random.uniform(8, 15), 1),
        "imbalance_charges_million": round(np.random.uniform(2, 5), 1),
        "imbalance_change_pct": round(np.random.uniform(-20, -10), 1),
        "net_revenue_million": round(np.random.uniform(135, 148), 1),
        "net_change_pct": round(np.random.uniform(10, 18), 1),
        "forecast_accuracy_pct": round(np.random.uniform(92, 96), 1),
        "accuracy_change_pct": round(np.random.uniform(1, 3), 1),
    }


@app.get("/api/v1/model/info", tags=["Market"])
async def get_model_info():
    """Get information about the SMP prediction model"""
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.smp.models.smp_predictor import get_smp_predictor
        predictor = get_smp_predictor(use_advanced=True)

        if predictor.is_ready():
            return {
                "status": "ready",
                "version": predictor.model_version,
                "type": "advanced" if predictor.use_advanced else "standard",
                "device": str(predictor.device),
                "mape": getattr(predictor, 'metrics', {}).get('mape', 'N/A'),
                "coverage": getattr(predictor, 'metrics', {}).get('coverage_80', 'N/A'),
            }
    except:
        pass

    return {"status": "fallback", "version": "fallback", "message": "Using fallback predictions"}


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("=" * 60)
    print("RE-BMS v5.0 API Starting...")
    print("=" * 60)
    print(f"  Docs: http://localhost:8506/docs")
    print(f"  ReDoc: http://localhost:8506/redoc")
    print(f"  Health: http://localhost:8506/health")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("RE-BMS v5.0 API Shutting down...")


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.rebms.api.main:app",
        host="0.0.0.0",
        port=8506,
        reload=True,
    )
