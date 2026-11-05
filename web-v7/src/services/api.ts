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
  PowerSupplyResponse,
  RealtimeStatus,
  RTMPrediction,
  RTMMultiPrediction,
  RTMModelInfo,
  PowerPlant,
  PowerPlantCreate,
  CurrentSMP,
  DayAheadSMP,
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

  async getDayAheadSMP(): Promise<DayAheadSMP> {
    return this.fetch('/smp/day-ahead');
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

  // Power Supply (실측 + 예측 데이터)
  async getPowerSupply(): Promise<PowerSupplyResponse> {
    return this.fetch('/power-supply');
  }

  // Realtime Status (API connection status)
  async getRealtimeStatus(): Promise<RealtimeStatus> {
    return this.fetch('/realtime-status');
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

  // Current SMP (Real-time)
  async getCurrentSMP(region: string = 'jeju'): Promise<CurrentSMP> {
    return this.fetch(`/smp/current?region=${region}`);
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
  async updatePowerPlant(id: string, plant: Partial<PowerPlantCreate> & Pick<Partial<PowerPlant>, 'status'>): Promise<PowerPlant> {
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
      supply_reserve_rate: 128.5,  // 공급 예비율 (%)
      daily_revenue_million: 245.8,
      revenue_change_pct: 3.2,
      current_smp: 102.5,
      smp_change_pct: -2.1,
      current_demand_mw: 685.0,
      renewable_ratio_pct: 24.6,
      grid_frequency: 60.01,
      weather: {
        temperature: 5.5,
        wind_speed: 3.2,
        humidity: 58.0,
        condition: '맑음',
      },
      resource_count: 20,
      data_source: 'Mock Data',
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


  getPowerSupply(): PowerSupplyResponse {
    const currentHour = new Date().getHours();
    const demandPattern = [520, 495, 480, 475, 485, 510, 550, 590, 650, 700, 730, 760, 750, 720, 710, 680, 640, 620, 610, 600, 590, 570, 550, 530];
    const windPattern = [185, 180, 175, 172, 178, 188, 170, 148, 132, 118, 105, 92, 85, 90, 108, 125, 145, 168, 182, 195, 200, 195, 190, 188];
    const solarPattern = [0, 0, 0, 0, 0, 0, 2, 25, 65, 105, 140, 165, 175, 168, 145, 95, 35, 5, 0, 0, 0, 0, 0, 0];

    return {
      current_hour: currentHour,
      data: Array.from({ length: 24 }, (_, hour) => ({
        hour,
        time: `${String(hour).padStart(2, '0')}:00`,
        supply: Math.round(demandPattern[hour] * 1.15),
        demand: demandPattern[hour],
        solar: solarPattern[hour],
        wind: windPattern[hour],
        is_forecast: hour > currentHour,
      })),
      data_source: 'Mock Data',
    };
  },
};

export default apiService;
