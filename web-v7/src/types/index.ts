/**
 * RE-BMS v6.0 Type Definitions
 */

// SMP Forecast Data
export interface SMPForecast {
  q10: number[];
  q50: number[];
  q90: number[];
  model_used: string;
  confidence: number;
  created_at: string;
}

// Market Status
export interface MarketStatus {
  current_time: string;
  dam: {
    status: string;
    deadline: string;
    trading_date: string;
    hours_remaining: number;
  };
  rtm: {
    status: string;
    next_interval: string;
    interval_minutes: number;
  };
}

// Weather Data
export interface WeatherData {
  temperature: number;  // 기온 (°C)
  wind_speed: number;   // 풍속 (m/s)
  humidity: number;     // 습도 (%)
  condition: string;    // 날씨 상태 (맑음, 흐림, 비 등)
}

// Dashboard KPIs
export interface DashboardKPIs {
  total_capacity_mw: number;
  current_output_mw: number;
  utilization_pct: number;
  supply_reserve_rate: number;   // 공급 예비율 (%)
  daily_revenue_million: number;
  revenue_change_pct: number;
  current_smp: number;
  smp_change_pct: number;
  current_demand_mw: number;     // 현재 수요 (MW)
  renewable_ratio_pct: number;   // 재생에너지 비율 (%)
  grid_frequency: number;        // 계통 주파수 (Hz)
  weather: WeatherData;          // 기상 현황
  resource_count: number;
  data_source?: string;
}

// Resource (Power Plant)
export interface Resource {
  id: string;
  name: string;
  name_en?: string;
  type: string;
  subtype?: string;
  capacity: number;
  current_output: number;
  utilization: number;
  status: string;
  location: string;
  operator?: string;
  latitude?: number;
  longitude?: number;
}

// Bid Segment
export interface BidSegment {
  segment_id: number;
  quantity_mw: number;
  price_krw_mwh: number;
}

// Hourly Bid
export interface HourlyBid {
  hour: number;
  segments: BidSegment[];
  total_mw: number;
  avg_price: number;
  smp_forecast: {
    q10: number;
    q50: number;
    q90: number;
  };
}

// Optimized Bids
export interface OptimizedBids {
  trading_date: string;
  capacity_mw: number;
  risk_level: string;
  hourly_bids: HourlyBid[];
  total_daily_mwh: number;
  model_used: string;
}

// Settlement Record (KPX Jeju Pilot - Gemini Verified)
export interface SettlementRecord {
  date: string;
  cleared_mwh: number;           // DA market cleared amount
  generation_mwh: number;        // Actual generation
  imbalance_mwh: number;         // |cleared - actual|
  revenue_million: number;       // DA market revenue
  imbalance_million: number;     // Imbalance penalty
  capacity_payment_million: number; // Capacity Payment (CP)
  net_revenue_million: number;   // Total net revenue
  accuracy_pct: number;          // Forecast accuracy
  avg_da_smp: number;            // Average DA-SMP
  avg_rt_smp: number;            // Average RT-SMP
  avg_deviation: number;         // Average deviation %
  // 3-tier penalty statistics
  hours_tier1: number;           // ±8% - No penalty
  hours_tier2: number;           // ±8~15% - Mild penalty
  hours_tier3: number;           // >±15% - Severe penalty
  hours_zero_risk: number;       // RT-SMP 0원 risk hours
  // Legacy fields (for backward compatibility)
  avg_smp?: number;
  hours_no_penalty?: number;
  hours_over_generation?: number;
  hours_under_generation?: number;
}

// Settlement Stats (KPX Jeju Pilot - Gemini Verified)
export interface SettlementStats {
  generation_revenue_million: number;
  generation_change_pct: number;
  imbalance_charges_million: number;
  imbalance_change_pct: number;
  capacity_payment_million: number;  // Capacity Payment (CP)
  net_revenue_million: number;
  net_change_pct: number;
  forecast_accuracy_pct: number;
  accuracy_change_pct: number;
  // DA/RT market info
  total_cleared_mwh: number;
  total_actual_mwh: number;
  avg_da_smp: number;
  avg_rt_smp: number;
  avg_deviation_pct: number;
  // 3-tier penalty statistics
  total_hours_tier1: number;     // ±8% - No penalty
  total_hours_tier2: number;     // ±8~15% - Mild penalty
  total_hours_tier3: number;     // >±15% - Severe penalty
  total_hours_zero_risk: number; // RT-SMP 0원 risk hours
  // Legacy fields
  total_hours_no_penalty?: number;
  total_hours_over_gen?: number;
  total_hours_under_gen?: number;
}

// Model Info
export interface ModelInfo {
  status: string;
  version: string;
  type?: string;
  device?: string;
  mape?: number | string;
  coverage?: number | string;
  message?: string;
}

// Jeju Realtime Power Data
export interface JejuRealtimeData {
  timestamp: string;
  demand_mw: number;
  supply_mw: number;
  reserve_mw: number;
  reserve_rate: number;
  solar_mw: number;
  wind_mw: number;
}

// Power Supply Chart Data
export interface PowerChartData {
  time: string;
  demand: number;
  supply: number;
  reserve: number;
  solar: number;
  wind: number;
  forecast?: boolean;
}

// Power Supply API Response (실측 + 예측)
export interface PowerSupplyHourlyData {
  hour: number;
  time: string;
  supply: number;
  demand: number;
  solar: number;
  wind: number;
  is_forecast: boolean;
}

export interface PowerSupplyResponse {
  current_hour: number;
  data: PowerSupplyHourlyData[];
  data_source: string;
}

// Alert
export interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
}

// Bid Submission Status
export type BidStatus = 'draft' | 'pending_review' | 'approved' | 'submitted_kpx' | 'matched' | 'rejected';

// Bid Submission
export interface BidSubmission {
  id: string;
  tradingDate: string;
  tradingHour: number;
  segments: {
    id: number;
    quantity: number;
    price: number;
  }[];
  totalQuantity: number;
  avgPrice: number;
  status: BidStatus;
  submittedAt?: string;
  approvedAt?: string;
  kpxSubmittedAt?: string;
  approver?: string;
  remarks?: string;
}

// KPX Market Bid (for simulation)
export interface KPXMarketBid {
  bidder: string;
  bidderType: 'generator' | 'consumer';
  quantity: number;
  price: number;
  isOurs?: boolean;
}

// KPX Matching Result
export interface KPXMatchingResult {
  hour: number;
  clearingPrice: number;
  totalDemand: number;
  totalSupply: number;
  matchedQuantity: number;
  ourAcceptedQuantity: number;
  ourRevenue: number;
  status: 'cleared' | 'partial' | 'rejected';
}

// API Connection Status (for each API endpoint)
export interface APIConnectionStatus {
  status: 'connected' | 'error' | 'unknown';
  last_update: string | null;
  error_message?: string;
}

// Realtime Status Response (from /api/v1/realtime-status)
export interface RealtimeStatus {
  smp_api: APIConnectionStatus;
  power_supply_api: APIConnectionStatus;
  weather_api: APIConnectionStatus;
  overall_status: 'all_connected' | 'partial' | 'all_error';
}

// RTM (Real-Time Market) Prediction Types
export interface RTMPrediction {
  status: string;
  prediction: {
    time: string;
    smp: number;
    confidence_low: number;
    confidence_high: number;
  };
  model: {
    name: string;
    mape: number;
    r2: number;
  };
  data_source: string;
}

export interface RTMMultiPrediction {
  status: string;
  predictions: Array<{
    time: string;
    smp: number;
    confidence_low: number;
    confidence_high: number;
    is_recursive: boolean;
  }>;
  model: {
    name: string;
    mape: number;
    note: string;
  };
  data_source: string;
}

export interface RTMModelInfo {
  status: string;
  model: {
    name: string;
    version: string;
    type: string;
    device: string;
    mape: number;
    r2: number;
    prediction_type: string;
  };
}

// Power Plant Types (v6.2.0)
export type PlantType = 'solar' | 'wind' | 'ess';
export type ContractType = 'net_metering' | 'ppa';
export type RoofDirection = 'south' | 'east' | 'west' | 'flat';
export type PlantStatus = 'active' | 'maintenance' | 'paused';
export type WeatherCondition = 'clear' | 'partly_cloudy' | 'cloudy' | 'rainy';

export interface PowerPlantLocation {
  address: string;
  lat?: number;
  lng?: number;
}

export interface PowerPlant {
  id: string;
  name: string;
  type: PlantType;
  capacity: number;
  installDate: string;
  contractType: ContractType;
  location: PowerPlantLocation;
  roofDirection?: RoofDirection;
  status?: PlantStatus;  // Operating status (default: active)
  createdAt: string;
  updatedAt: string;
}

export interface PowerPlantCreate {
  name: string;
  type: PlantType;
  capacity: number;
  installDate: string;
  contractType: ContractType;
  location: PowerPlantLocation;
  roofDirection?: RoofDirection;
}

// Power Plant UI Labels
export const PLANT_TYPE_LABELS: Record<PlantType, { label: string; icon: string }> = {
  solar: { label: '태양광', icon: '☀️' },
  wind: { label: '풍력', icon: '💨' },
  ess: { label: 'ESS', icon: '🔋' },
};

export const CONTRACT_TYPE_LABELS: Record<ContractType, { label: string; description: string }> = {
  net_metering: { label: '상계거래', description: '전기요금 차감' },
  ppa: { label: 'PPA', description: '현금 수익' },
};

export const ROOF_DIRECTION_LABELS: Record<RoofDirection, string> = {
  south: '남향',
  east: '동향',
  west: '서향',
  flat: '평지',
};

export const PLANT_STATUS_LABELS: Record<PlantStatus, { label: string; color: string; icon: string }> = {
  active: { label: '운영중', color: 'green', icon: '✓' },
  maintenance: { label: '점검중', color: 'yellow', icon: '🔧' },
  paused: { label: '중지', color: 'gray', icon: '⏸' },
};
