"""
RE-BMS Resource Models
======================

Renewable energy resource models for VPP portfolio management.

Resource Types:
- Solar: Photovoltaic power plants
- Wind: Wind farms
- Solar+ESS: Solar with Energy Storage System
- Wind+ESS: Wind with Energy Storage System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class ResourceType(Enum):
    """Renewable resource type"""
    SOLAR = "solar"
    WIND = "wind"
    SOLAR_ESS = "solar_ess"    # Solar + ESS hybrid
    WIND_ESS = "wind_ess"      # Wind + ESS hybrid


class ConnectionStatus(Enum):
    """Grid connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    MAINTENANCE = "maintenance"
    CURTAILED = "curtailed"


@dataclass
class GenerationForecast:
    """24-hour generation forecast

    Attributes:
        hourly_mw: List of 24 hourly generation values (MW)
        confidence: Forecast confidence (0-1)
        model_used: ML model used for prediction
        created_at: Forecast creation timestamp
    """
    hourly_mw: List[float]      # 24 hours
    confidence: float = 0.8
    model_used: str = "lstm"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if len(self.hourly_mw) != 24:
            raise ValueError(f"Expected 24 hourly values, got {len(self.hourly_mw)}")

    def total_mwh(self) -> float:
        """Total daily generation in MWh"""
        return sum(self.hourly_mw)

    def peak_mw(self) -> float:
        """Peak hourly generation"""
        return max(self.hourly_mw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hourly_mw': self.hourly_mw,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'created_at': self.created_at.isoformat(),
            'total_mwh': self.total_mwh(),
            'peak_mw': self.peak_mw(),
        }


@dataclass
class RenewableResource:
    """Renewable energy generator resource

    Represents a solar or wind generator in the VPP portfolio.

    Attributes:
        resource_id: Unique identifier
        name: Display name
        resource_type: Solar/Wind/Hybrid
        installed_capacity_mw: Installed capacity in MW
        latitude: GPS latitude
        longitude: GPS longitude
        region: KPX region code (default: "jeju")

    Example:
        >>> resource = RenewableResource(
        ...     resource_id="solar-001",
        ...     name="Jeju Solar Plant A",
        ...     resource_type=ResourceType.SOLAR,
        ...     installed_capacity_mw=50.0,
        ...     latitude=33.489,
        ...     longitude=126.498,
        ... )
    """
    resource_id: str                        # Unique identifier
    name: str                               # Display name
    resource_type: ResourceType             # Solar/Wind

    # Capacity
    installed_capacity_mw: float            # Installed capacity
    current_output_mw: float = 0.0          # Real-time output
    availability_percent: float = 100.0     # Current availability (0-100)

    # Location
    latitude: float = 33.489               # Default: Jeju
    longitude: float = 126.498
    region: str = "jeju"                    # KPX region code

    # Grid connection
    connection_status: ConnectionStatus = ConnectionStatus.CONNECTED
    grid_point: str = ""                    # Connection point ID

    # Forecasting
    forecast: Optional[GenerationForecast] = None

    # Performance metrics
    capacity_factor: float = 0.25           # Historical capacity factor (0-1)
    curtailment_rate: float = 0.0           # Curtailment percentage (0-1)

    # Metadata
    registered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def effective_capacity(self) -> float:
        """Calculate effective capacity considering availability"""
        return self.installed_capacity_mw * (self.availability_percent / 100)

    def utilization_rate(self) -> float:
        """Current utilization rate"""
        if self.installed_capacity_mw > 0:
            return (self.current_output_mw / self.installed_capacity_mw) * 100
        return 0.0

    def update_output(self, output_mw: float):
        """Update current output"""
        self.current_output_mw = min(output_mw, self.installed_capacity_mw)
        self.last_updated = datetime.now()

    def set_curtailed(self, curtailment_percent: float):
        """Set curtailment status"""
        self.curtailment_rate = curtailment_percent / 100
        if curtailment_percent > 0:
            self.connection_status = ConnectionStatus.CURTAILED
        else:
            self.connection_status = ConnectionStatus.CONNECTED
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'resource_id': self.resource_id,
            'name': self.name,
            'resource_type': self.resource_type.value,
            'installed_capacity_mw': self.installed_capacity_mw,
            'current_output_mw': self.current_output_mw,
            'availability_percent': self.availability_percent,
            'effective_capacity_mw': self.effective_capacity(),
            'utilization_rate': self.utilization_rate(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'region': self.region,
            'connection_status': self.connection_status.value,
            'grid_point': self.grid_point,
            'forecast': self.forecast.to_dict() if self.forecast else None,
            'capacity_factor': self.capacity_factor,
            'curtailment_rate': self.curtailment_rate,
            'registered_at': self.registered_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenewableResource':
        """Create from dictionary"""
        forecast = None
        if data.get('forecast'):
            forecast = GenerationForecast(
                hourly_mw=data['forecast']['hourly_mw'],
                confidence=data['forecast'].get('confidence', 0.8),
            )

        return cls(
            resource_id=data['resource_id'],
            name=data['name'],
            resource_type=ResourceType(data['resource_type']),
            installed_capacity_mw=data['installed_capacity_mw'],
            current_output_mw=data.get('current_output_mw', 0.0),
            availability_percent=data.get('availability_percent', 100.0),
            latitude=data.get('latitude', 33.489),
            longitude=data.get('longitude', 126.498),
            region=data.get('region', 'jeju'),
            connection_status=ConnectionStatus(data.get('connection_status', 'connected')),
            forecast=forecast,
            capacity_factor=data.get('capacity_factor', 0.25),
            curtailment_rate=data.get('curtailment_rate', 0.0),
        )


@dataclass
class ResourcePortfolio:
    """VPP resource portfolio

    Aggregates multiple renewable resources for VPP operation.

    Attributes:
        portfolio_id: Unique identifier
        name: Portfolio name
        operator_id: VPP operator ID
        resources: List of managed resources

    Example:
        >>> portfolio = ResourcePortfolio(
        ...     portfolio_id="vpp-001",
        ...     name="Jeju VPP",
        ...     operator_id="operator-001",
        ...     resources=[solar_plant, wind_farm],
        ... )
        >>> portfolio.calculate_aggregates()
        >>> portfolio.total_capacity_mw
        90.0
    """
    portfolio_id: str
    name: str
    operator_id: str                        # VPP operator ID
    resources: List[RenewableResource] = field(default_factory=list)

    # Aggregated metrics (auto-calculated)
    total_capacity_mw: float = 0.0
    current_output_mw: float = 0.0
    aggregate_forecast: Optional[GenerationForecast] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_aggregates(self):
        """Calculate portfolio-level aggregates"""
        self.total_capacity_mw = sum(r.installed_capacity_mw for r in self.resources)
        self.current_output_mw = sum(r.current_output_mw for r in self.resources)

        # Aggregate forecasts if all resources have forecasts
        if all(r.forecast is not None for r in self.resources) and self.resources:
            hourly_totals = [0.0] * 24
            for resource in self.resources:
                for h in range(24):
                    hourly_totals[h] += resource.forecast.hourly_mw[h]

            # Average confidence
            avg_confidence = sum(r.forecast.confidence for r in self.resources) / len(self.resources)

            self.aggregate_forecast = GenerationForecast(
                hourly_mw=hourly_totals,
                confidence=avg_confidence,
                model_used="aggregate",
            )

        self.last_updated = datetime.now()

    def add_resource(self, resource: RenewableResource):
        """Add resource to portfolio"""
        self.resources.append(resource)
        self.calculate_aggregates()

    def remove_resource(self, resource_id: str) -> bool:
        """Remove resource by ID"""
        for i, r in enumerate(self.resources):
            if r.resource_id == resource_id:
                self.resources.pop(i)
                self.calculate_aggregates()
                return True
        return False

    def get_resource(self, resource_id: str) -> Optional[RenewableResource]:
        """Get resource by ID"""
        for r in self.resources:
            if r.resource_id == resource_id:
                return r
        return None

    def get_by_type(self, resource_type: ResourceType) -> List[RenewableResource]:
        """Get resources by type"""
        return [r for r in self.resources if r.resource_type == resource_type]

    def utilization_rate(self) -> float:
        """Portfolio-level utilization rate"""
        if self.total_capacity_mw > 0:
            return (self.current_output_mw / self.total_capacity_mw) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'operator_id': self.operator_id,
            'resources': [r.to_dict() for r in self.resources],
            'total_capacity_mw': self.total_capacity_mw,
            'current_output_mw': self.current_output_mw,
            'utilization_rate': self.utilization_rate(),
            'aggregate_forecast': self.aggregate_forecast.to_dict() if self.aggregate_forecast else None,
            'resource_count': len(self.resources),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourcePortfolio':
        """Create from dictionary"""
        resources = [RenewableResource.from_dict(r) for r in data.get('resources', [])]

        portfolio = cls(
            portfolio_id=data['portfolio_id'],
            name=data['name'],
            operator_id=data['operator_id'],
            resources=resources,
        )
        portfolio.calculate_aggregates()
        return portfolio
