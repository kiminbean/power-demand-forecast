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

// Dashboard KPIs
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

// Settlement Record
export interface SettlementRecord {
  date: string;
  generation_mwh: number;
  revenue_million: number;
  imbalance_million: number;
  net_revenue_million: number;
  accuracy_pct: number;
}

// Settlement Stats
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
