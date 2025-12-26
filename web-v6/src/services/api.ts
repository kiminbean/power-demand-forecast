/**
 * RE-BMS v6.0 API Service
 * Connects to the FastAPI backend at port 8506
 */

import type {
  SMPForecast,
  MarketStatus,
  DashboardKPIs,
  Resource,
  OptimizedBids,
  SettlementRecord,
  SettlementStats,
  ModelInfo,
  RTMPrediction,
  RTMMultiPrediction,
  RTMModelInfo,
} from '../types';

const API_BASE_URL = '/api/v1';

class ApiService {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
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
  async healthCheck(): Promise<{ status: string; service: string }> {
    return this.fetch('/health');
  }

  // SMP Forecast (24h predictions with quantiles)
  async getSMPForecast(): Promise<SMPForecast> {
    return this.fetch('/smp-forecast');
  }

  // Market Status (DAM/RTM status and deadlines)
  async getMarketStatus(): Promise<MarketStatus> {
    return this.fetch('/market-status');
  }

  // Dashboard KPIs
  async getDashboardKPIs(): Promise<DashboardKPIs> {
    return this.fetch('/dashboard/kpis');
  }

  // Resources Portfolio (Jeju Power Plants)
  async getResources(): Promise<Resource[]> {
    return this.fetch('/resources');
  }

  // AI-Optimized Bid Segments
  async getOptimizedSegments(
    capacityMw: number = 50,
    riskLevel: string = 'moderate'
  ): Promise<OptimizedBids> {
    return this.fetch(
      `/bidding/optimized-segments?capacity_mw=${capacityMw}&risk_level=${riskLevel}`
    );
  }

  // Recent Settlements
  async getRecentSettlements(days: number = 7): Promise<SettlementRecord[]> {
    return this.fetch(`/settlements/recent?days=${days}`);
  }

  // Settlement Summary Statistics
  async getSettlementSummary(): Promise<SettlementStats> {
    return this.fetch('/settlements/summary');
  }

  // Model Info
  async getModelInfo(): Promise<ModelInfo> {
    return this.fetch('/model/info');
  }

  // RTM Prediction - Single Hour (CatBoost v3.10, MAPE 5.25%)
  async getRTMPrediction(): Promise<RTMPrediction> {
    // RTM uses direct /smp prefix, not /api/v1
    const response = await fetch('/smp/rtm/predict', {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }

  // RTM Prediction - Multi Hour (CatBoost v3.10, recursive)
  async getRTMMultiPrediction(hours: number = 6): Promise<RTMMultiPrediction> {
    const response = await fetch(`/smp/rtm/predict/${hours}`, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }

  // RTM Model Info
  async getRTMModelInfo(): Promise<RTMModelInfo> {
    const response = await fetch('/smp/rtm/model/info', {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
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

// Mock data generators for development/fallback
export const mockData = {
  getSMPForecast(): SMPForecast {
    const basePrice = 95;
    const hours = 24;
    const q50 = Array.from({ length: hours }, (_, i) => {
      const hourFactor = Math.sin((i - 6) * Math.PI / 12) * 30;
      return Math.round(basePrice + hourFactor + Math.random() * 10);
    });
    const q10 = q50.map(v => Math.round(v * 0.85));
    const q90 = q50.map(v => Math.round(v * 1.15));

    return {
      q10,
      q50,
      q90,
      model_used: 'BiLSTM+Attention v3.2 Optuna',
      confidence: 0.87,
      created_at: new Date().toISOString(),
    };
  },

  getDashboardKPIs(): DashboardKPIs {
    return {
      total_capacity_mw: 224.9,
      current_output_mw: 168.5,
      utilization_pct: 74.9,
      daily_revenue_million: 245.8,
      revenue_change_pct: 3.2,
      current_smp: 102.5,
      smp_change_pct: -2.1,
      resource_count: 20,
    };
  },

  getMarketStatus(): MarketStatus {
    return {
      current_time: new Date().toISOString(),
      dam: {
        status: 'open',
        deadline: '10:00',
        trading_date: new Date(Date.now() + 86400000).toISOString().split('T')[0],
        hours_remaining: 4,
      },
      rtm: {
        status: 'active',
        next_interval: '15분',
        interval_minutes: 15,
      },
    };
  },

  getResources(): Resource[] {
    return [
      { id: '1', name: '가시리풍력', type: 'wind', capacity: 15.0, current_output: 12.3, utilization: 82, status: 'active', location: '서귀포시', latitude: 33.3823, longitude: 126.7632 },
      { id: '2', name: '김녕풍력', type: 'wind', capacity: 12.0, current_output: 9.8, utilization: 81.7, status: 'active', location: '제주시', latitude: 33.5575, longitude: 126.7631 },
      { id: '3', name: '한경풍력', type: 'wind', capacity: 21.0, current_output: 18.2, utilization: 86.7, status: 'active', location: '제주시', latitude: 33.3343, longitude: 126.1727 },
      { id: '4', name: '삼달풍력', type: 'wind', capacity: 6.1, current_output: 5.1, utilization: 83.6, status: 'active', location: '서귀포시', latitude: 33.3489, longitude: 126.8347 },
      { id: '5', name: '월정태양광', type: 'solar', capacity: 5.0, current_output: 3.8, utilization: 76, status: 'active', location: '제주시', latitude: 33.5556, longitude: 126.7889 },
      { id: '6', name: '성산태양광', type: 'solar', capacity: 8.0, current_output: 6.2, utilization: 77.5, status: 'active', location: '서귀포시', latitude: 33.4586, longitude: 126.9312 },
    ];
  },

  getModelInfo(): ModelInfo {
    return {
      status: 'active',
      version: 'v3.2',
      type: 'BiLSTM+Attention (Optuna)',
      device: 'MPS',
      mape: 7.17,
      coverage: 94.5,
    };
  },

  // RTM Model Mock Data (CatBoost v3.10)
  getRTMModelInfo(): RTMModelInfo {
    return {
      status: 'active',
      model: {
        name: 'CatBoost v3.10',
        version: '3.10',
        type: 'CatBoost Gradient Boosting',
        device: 'CPU',
        mape: 5.25,
        r2: 0.8264,
        prediction_type: 'single-step (RTM)',
      },
    };
  },
};

export default apiService;
