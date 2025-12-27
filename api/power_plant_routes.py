"""
Power Plant Management API Routes (v6.2.0)
==========================================

API endpoints for small-scale power plant (solar/wind/ESS) registration and management.
Supports VPP (Virtual Power Plant) aggregation concept.

Author: Claude Code
Date: 2025-12
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Data storage path
DATA_DIR = Path(__file__).parent.parent / "data" / "power_plants"
DATA_FILE = DATA_DIR / "power_plants.json"

router = APIRouter(
    prefix="/api/v1",
    tags=["Power Plants"],
)


# ============================================================
# Pydantic Models
# ============================================================

class LocationModel(BaseModel):
    """Location information"""
    address: str = Field(description="Address")
    lat: Optional[float] = Field(None, description="Latitude")
    lng: Optional[float] = Field(None, description="Longitude")


class PowerPlantCreate(BaseModel):
    """Power plant creation request"""
    name: str = Field(description="Plant name (e.g., '우리집 태양광 1호')")
    type: str = Field(description="Plant type: solar, wind, ess")
    capacity: float = Field(description="Capacity in kW")
    installDate: str = Field(description="Installation date (ISO format)")
    contractType: str = Field(description="Contract type: net_metering, ppa")
    location: LocationModel = Field(description="Location information")
    roofDirection: Optional[str] = Field(None, description="Roof direction: south, east, west, flat")
    status: Optional[str] = Field("active", description="Operating status: active, maintenance, paused")


class PowerPlant(PowerPlantCreate):
    """Power plant with ID and timestamps"""
    id: str = Field(description="Unique identifier")
    createdAt: str = Field(description="Creation timestamp")
    updatedAt: str = Field(description="Last update timestamp")


class PowerPlantUpdate(BaseModel):
    """Power plant update request (partial)"""
    name: Optional[str] = None
    type: Optional[str] = None
    capacity: Optional[float] = None
    installDate: Optional[str] = None
    contractType: Optional[str] = None
    location: Optional[LocationModel] = None
    roofDirection: Optional[str] = None
    status: Optional[str] = None


# ============================================================
# Storage Helpers
# ============================================================

def ensure_data_dir():
    """Ensure data directory exists"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_power_plants() -> List[dict]:
    """Load power plants from JSON file"""
    ensure_data_dir()
    if not DATA_FILE.exists():
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load power plants: {e}")
        return []


def save_power_plants(plants: List[dict]):
    """Save power plants to JSON file"""
    ensure_data_dir()
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(plants, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save power plants: {e}")
        raise HTTPException(status_code=500, detail="Failed to save power plant data")


# ============================================================
# API Endpoints
# ============================================================

@router.get(
    "/power-plants",
    response_model=List[PowerPlant],
    summary="Get all power plants",
    description="Retrieve list of all registered power plants"
)
async def get_power_plants():
    """Get all registered power plants"""
    plants = load_power_plants()
    return plants


@router.get(
    "/power-plants/{plant_id}",
    response_model=PowerPlant,
    summary="Get power plant by ID",
    description="Retrieve a specific power plant by its ID"
)
async def get_power_plant(plant_id: str):
    """Get a specific power plant by ID"""
    plants = load_power_plants()
    for plant in plants:
        if plant["id"] == plant_id:
            return plant
    raise HTTPException(status_code=404, detail=f"Power plant {plant_id} not found")


@router.post(
    "/power-plants",
    response_model=PowerPlant,
    summary="Create power plant",
    description="Register a new power plant"
)
async def create_power_plant(plant: PowerPlantCreate):
    """Create a new power plant"""
    plants = load_power_plants()

    # Validate plant type
    if plant.type not in ["solar", "wind", "ess"]:
        raise HTTPException(status_code=400, detail="Invalid plant type. Must be: solar, wind, ess")

    # Validate contract type
    if plant.contractType not in ["net_metering", "ppa"]:
        raise HTTPException(status_code=400, detail="Invalid contract type. Must be: net_metering, ppa")

    # Validate roof direction if provided
    if plant.roofDirection and plant.roofDirection not in ["south", "east", "west", "flat"]:
        raise HTTPException(status_code=400, detail="Invalid roof direction. Must be: south, east, west, flat")

    # Validate status if provided
    status = plant.status or "active"
    if status not in ["active", "maintenance", "paused"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be: active, maintenance, paused")

    # Create new plant with ID and timestamps
    now = datetime.now().isoformat()
    new_plant = {
        "id": str(uuid.uuid4()),
        "name": plant.name,
        "type": plant.type,
        "capacity": plant.capacity,
        "installDate": plant.installDate,
        "contractType": plant.contractType,
        "location": plant.location.dict(),
        "roofDirection": plant.roofDirection,
        "status": status,
        "createdAt": now,
        "updatedAt": now,
    }

    plants.append(new_plant)
    save_power_plants(plants)

    logger.info(f"Created power plant: {new_plant['name']} ({new_plant['id']})")
    return new_plant


@router.put(
    "/power-plants/{plant_id}",
    response_model=PowerPlant,
    summary="Update power plant",
    description="Update an existing power plant"
)
async def update_power_plant(plant_id: str, update: PowerPlantUpdate):
    """Update an existing power plant"""
    plants = load_power_plants()

    for i, plant in enumerate(plants):
        if plant["id"] == plant_id:
            # Update only provided fields
            update_data = update.dict(exclude_unset=True)

            # Validate plant type if provided
            if "type" in update_data and update_data["type"] not in ["solar", "wind", "ess"]:
                raise HTTPException(status_code=400, detail="Invalid plant type")

            # Validate contract type if provided
            if "contractType" in update_data and update_data["contractType"] not in ["net_metering", "ppa"]:
                raise HTTPException(status_code=400, detail="Invalid contract type")

            # Validate status if provided
            if "status" in update_data and update_data["status"] not in ["active", "maintenance", "paused"]:
                raise HTTPException(status_code=400, detail="Invalid status. Must be: active, maintenance, paused")

            # Handle nested location update
            if "location" in update_data and update_data["location"]:
                update_data["location"] = update_data["location"].dict() if hasattr(update_data["location"], "dict") else update_data["location"]

            # Apply updates
            for key, value in update_data.items():
                if value is not None:
                    plant[key] = value

            plant["updatedAt"] = datetime.now().isoformat()
            plants[i] = plant
            save_power_plants(plants)

            logger.info(f"Updated power plant: {plant['name']} ({plant_id})")
            return plant

    raise HTTPException(status_code=404, detail=f"Power plant {plant_id} not found")


@router.delete(
    "/power-plants/{plant_id}",
    summary="Delete power plant",
    description="Delete a power plant by ID"
)
async def delete_power_plant(plant_id: str):
    """Delete a power plant by ID"""
    plants = load_power_plants()

    for i, plant in enumerate(plants):
        if plant["id"] == plant_id:
            deleted_plant = plants.pop(i)
            save_power_plants(plants)
            logger.info(f"Deleted power plant: {deleted_plant['name']} ({plant_id})")
            return {"success": True, "message": f"Power plant {plant_id} deleted"}

    raise HTTPException(status_code=404, detail=f"Power plant {plant_id} not found")


# ============================================================
# VPP Aggregation Endpoints
# ============================================================

@router.get(
    "/power-plants/aggregate/capacity",
    summary="Get aggregated capacity",
    description="Get total aggregated capacity from all registered plants"
)
async def get_aggregated_capacity():
    """Get total aggregated capacity from all registered plants"""
    plants = load_power_plants()

    # Filter active plants for bidding calculations
    active_plants = [p for p in plants if p.get("status", "active") == "active"]

    # Total capacity includes all plants
    total_capacity = sum(p.get("capacity", 0) for p in plants)
    # Active capacity only counts active plants (for VPP bidding)
    active_capacity = sum(p.get("capacity", 0) for p in active_plants)

    by_type = {}
    by_status = {"active": 0, "maintenance": 0, "paused": 0}

    for plant in plants:
        plant_type = plant.get("type", "unknown")
        plant_status = plant.get("status", "active")
        by_type[plant_type] = by_type.get(plant_type, 0) + plant.get("capacity", 0)
        if plant_status in by_status:
            by_status[plant_status] += plant.get("capacity", 0)

    return {
        "total_capacity_kw": total_capacity,
        "active_capacity_kw": active_capacity,
        "total_plants": len(plants),
        "active_plants": len(active_plants),
        "capacity_by_type": by_type,
        "capacity_by_status": by_status,
        "can_bid_kpx": active_capacity >= 1000,  # KPX requires 1MW minimum (only active plants)
        "vpp_status": "active" if active_capacity >= 1000 else "aggregating"
    }
