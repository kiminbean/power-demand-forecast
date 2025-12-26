/**
 * RE-BMS Mobile API Service
 * Connects to the FastAPI backend for real SMP predictions
 * Uses fetch for web compatibility
 */

import { Platform } from 'react-native';

// API Base URL - different for web vs native
// Uses FastAPI server on port 8000 with /api/v1 prefix
const API_BASE_URL = Platform.OS === 'web'
  ? 'http://localhost:8000'
  : 'http://localhost:8000';  // Change to your server IP for device testing

// Types
export interface SMPForecast {
  q10: number[];
  q50: number[];
  q90: number[];
  model_used: string;
  confidence: number;
  created_at: string;
}

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

export interface DashboardKPIs {
  total_capacity_mw: number;
  current_output_mw: number;
  utilization_pct: number;
  daily_revenue_million: number;
  revenue_change_pct: number;
  current_smp: number;
  smp_change_pct: number;
  resource_count: number;
}

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

export interface BidSegment {
  segment_id: number;
  quantity_mw: number;
  price_krw_mwh: number;
}

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

export interface OptimizedBids {
  trading_date: string;
  capacity_mw: number;
  risk_level: string;
  hourly_bids: HourlyBid[];
  total_daily_mwh: number;
  model_used: string;
}

export interface SettlementRecord {
  date: string;
  generation_mwh: number;
  revenue_million: number;
  imbalance_million: number;
  net_revenue_million: number;
  accuracy_pct: number;
}

export interface SettlementStats {
  generation_revenue_million: number;
  generation_change_pct: number;
  imbalance_charges_million: number;
  imbalance_change_pct: number;
  net_revenue_million: number;
  net_change_pct: number;
  forecast_accuracy_pct: number;
  accuracy_change_pct: number;
}

export interface ModelInfo {
  status: string;
  version: string;
  type?: string;
  device?: string;
  mape?: number | string;
  coverage?: number | string;
  message?: string;
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

export interface CurrentSMP {
  current_smp: number;
  hour: number;
  region: string;
  comparison: {
    daily_avg: number;
    weekly_avg: number;
    daily_change_pct: number;
    weekly_change_pct: number;
  };
}

export interface BiddingStrategy {
  risk_level: string;
  recommended_hours: number[];
  total_generation_kwh: number;
  total_revenue: number;
  average_smp: number;
  revenue_per_kwh: number;
  hourly_details: Array<{
    hour: number;
    smp: number;
    generation_kwh: number;
    revenue: number;
    recommendation: string;
  }>;
}

export interface SimulationResult {
  expected_revenue: number;
  best_case: number;
  worst_case: number;
  risk_adjusted: number;
  scenarios: Array<{
    scenario: string;
    revenue: number;
    generation_kwh: number;
  }>;
}

export interface PowerSupplyData {
  current_hour: number;
  data: Array<{
    hour: number;
    time: string;
    supply: number;
    demand: number;
    solar: number;
    wind: number;
    is_forecast: boolean;
  }>;
  data_source: string;
}

export interface DAMSimulationRequest {
  segments: Array<{
    segment_id: number;
    quantity_mw: number;
    price_krw_mwh: number;
  }>;
  hour: number;
}

export interface DAMSimulationResult {
  hour: number;
  submitted_segments: number;
  total_quantity_mw: number;
  total_cleared_mw: number;
  clearing_rate: number;
  expected_revenue_million: number;
  results: Array<{
    segment_id: number;
    quantity_mw: number;
    price_krw_mwh: number;
    cleared_mw: number;
    clearing_price: number;
    status: string;
    revenue_million: number;
  }>;
  market_clearing_price: number;
}

// API Client using fetch (web compatible)
class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API call failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Health check
  async healthCheck(): Promise<{ status: string; version: string }> {
    return this.fetch('/health');
  }

  // SMP Forecast (24h predictions with quantiles)
  async getSMPForecast(): Promise<SMPForecast> {
    return this.fetch('/api/v1/smp-forecast');
  }

  // Market Status (DAM/RTM status and deadlines)
  async getMarketStatus(): Promise<MarketStatus> {
    return this.fetch('/api/v1/market-status');
  }

  // Dashboard KPIs
  async getDashboardKPIs(): Promise<DashboardKPIs> {
    return this.fetch('/api/v1/dashboard/kpis');
  }

  // Resources Portfolio
  async getResources(): Promise<Resource[]> {
    return this.fetch('/api/v1/resources');
  }

  // AI-Optimized Bid Segments
  async getOptimizedSegments(
    capacityMw: number = 50,
    riskLevel: string = 'moderate'
  ): Promise<OptimizedBids> {
    return this.fetch(
      `/api/v1/bidding/optimized-segments?capacity_mw=${capacityMw}&risk_level=${riskLevel}`
    );
  }

  // Recent Settlements
  async getRecentSettlements(days: number = 7): Promise<SettlementRecord[]> {
    return this.fetch(`/api/v1/settlements/recent?days=${days}`);
  }

  // Settlement Summary Statistics
  async getSettlementSummary(): Promise<SettlementStats> {
    return this.fetch('/api/v1/settlements/summary');
  }

  // Model Info
  async getModelInfo(): Promise<ModelInfo> {
    return this.fetch('/api/v1/model/info');
  }

  // Current SMP (real-time)
  async getCurrentSMP(region: string = 'jeju'): Promise<CurrentSMP> {
    return this.fetch(`/api/v1/smp/current?region=${region}`);
  }

  // Power Supply Data (24-hour)
  async getPowerSupply(): Promise<PowerSupplyData> {
    return this.fetch('/api/v1/power-supply');
  }

  // Bidding Strategy (AI optimization)
  async getBiddingStrategy(
    capacityKw: number = 50000,
    energyType: string = 'solar',
    riskLevel: string = 'moderate'
  ): Promise<BiddingStrategy> {
    return this.fetch('/api/v1/bidding/strategy', {
      method: 'POST',
      body: JSON.stringify({
        capacity_kw: capacityKw,
        energy_type: energyType,
        risk_level: riskLevel,
        location: 'Jeju',
        prediction_hours: 24,
      }),
    });
  }

  // Revenue Simulation
  async simulateRevenue(
    capacityKw: number = 50000,
    energyType: string = 'solar',
    hours: number = 24
  ): Promise<SimulationResult> {
    return this.fetch('/api/v1/bidding/simulate', {
      method: 'POST',
      body: JSON.stringify({
        capacity_kw: capacityKw,
        energy_type: energyType,
        hours: hours,
      }),
    });
  }

  // DAM Simulation (KPX submission simulation)
  async simulateDAM(request: DAMSimulationRequest): Promise<DAMSimulationResult> {
    return this.fetch('/api/v1/bidding/dam-simulate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // RTM Prediction - Single Hour (CatBoost v3.10)
  async getRTMPrediction(): Promise<RTMPrediction> {
    return this.fetch('/smp/rtm/predict');
  }

  // RTM Prediction - Multi Hour (CatBoost v3.10, recursive)
  async getRTMMultiPrediction(hours: number = 6): Promise<RTMMultiPrediction> {
    return this.fetch(`/smp/rtm/predict/${hours}`);
  }

  // RTM Model Info
  async getRTMModelInfo(): Promise<RTMModelInfo> {
    return this.fetch('/smp/rtm/model/info');
  }
}

// Singleton instance
export const apiService = new ApiService();

// Helper function to check API availability
export async function isApiAvailable(): Promise<boolean> {
  try {
    const health = await apiService.healthCheck();
    return health.status === 'healthy';
  } catch {
    return false;
  }
}

export default apiService;
