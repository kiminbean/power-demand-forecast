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
  DayAheadSMP,
  PowerSupplyResponse,
  RTMPrediction,
  RTMMultiPrediction,
  RTMModelInfo,
  PowerPlant,
  PowerPlantCreate,
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

  // KPX 제주 하루전시장 SMP (현재 거래시간) — v7·모바일과 같은 원천
  async getDayAheadSMP(): Promise<DayAheadSMP> {
    return this.fetch('/smp/day-ahead');
  }

  // 전력수급 24시간 (v7 차트와 같은 원천)
  async getPowerSupply(): Promise<PowerSupplyResponse> {
    return this.fetch('/power-supply');
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

  // ============================================
  // Power Plant Management (v6.2.0)
  // ============================================

  // Get all power plants
  async getPowerPlants(): Promise<PowerPlant[]> {
    return this.fetch('/power-plants');
  }

  // Get a single power plant
  async getPowerPlant(id: string): Promise<PowerPlant> {
    return this.fetch(`/power-plants/${id}`);
  }

  // Create a new power plant
  async createPowerPlant(plant: PowerPlantCreate): Promise<PowerPlant> {
    return this.fetch('/power-plants', {
      method: 'POST',
      body: JSON.stringify(plant),
    });
  }

  // Update an existing power plant
  async updatePowerPlant(id: string, plant: Partial<PowerPlantCreate>): Promise<PowerPlant> {
    return this.fetch(`/power-plants/${id}`, {
      method: 'PUT',
      body: JSON.stringify(plant),
    });
  }

  // Delete a power plant
  async deletePowerPlant(id: string): Promise<{ success: boolean }> {
    return this.fetch(`/power-plants/${id}`, {
      method: 'DELETE',
    });
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
      model_used: 'BiLSTM+Attention v3.2 Optuna (데모)',
      confidence: null,
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
    // 예전 데모 목록(가시리풍력 15MW 등)은 기준 목록과 달라 실제처럼 보일 수 있으므로 지어내지 않는다.
    return [];
  },

  getModelInfo(): ModelInfo {
    return {
      status: 'active',
      version: 'v3.2',
      type: 'BiLSTM+Attention (Optuna)',
      device: 'MPS',
      // 누수 시절 지표(7.17%)·근거 없는 coverage는 표시하지 않는다.
      mape: null,
      coverage: null,
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
