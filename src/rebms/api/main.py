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
