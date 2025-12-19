/**
 * RE-BMS API Service
 * Connects to FastAPI backend at port 8506
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';

// Configuration
const API_BASE_URL = 'http://localhost:8506/api/v1';

// Types
export interface BidSegment {
  segmentId: number;
  quantityMw: number;
  priceKrwMwh: number;
  cumulativeMw?: number;
}

export interface HourlyBid {
  hour: number;
  segments: BidSegment[];
}

export interface DailyBid {
  bidId: string;
  resourceId: string;
  resourceName?: string;
  marketType: 'DAM' | 'RTM';
  tradingDate: string;
  status: 'DRAFT' | 'PENDING' | 'SUBMITTED' | 'ACCEPTED' | 'REJECTED';
  hourlyBids: HourlyBid[];
  createdAt: string;
  updatedAt: string;
}

export interface Resource {
  resourceId: string;
  name: string;
  type: 'solar' | 'wind';
  capacityMw: number;
  location: string;
  status: 'active' | 'maintenance' | 'offline';
}

export interface SMPForecast {
  hour: number;
  q10: number;
  q50: number;
  q90: number;
}

export interface MarketStatus {
  damStatus: 'open' | 'closed' | 'pending';
  rtmStatus: 'open' | 'closed' | 'pending';
  damDeadline: string;
  rtmDeadline: string;
  currentSmp: number;
}

export interface Settlement {
  date: string;
  resourceId: string;
  revenue: number;
  penalty: number;
  netRevenue: number;
  generationMwh: number;
  imbalanceRate: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

// API Client
class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        // const token = await getAuthToken();
        // if (token) config.headers.Authorization = `Bearer ${token}`;
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health Check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.data.status === 'healthy';
    } catch {
      return false;
    }
  }

  // Market Status
  async getMarketStatus(): Promise<MarketStatus> {
    const response = await this.client.get<MarketStatus>('/market-status');
    return response.data;
  }

  // SMP Forecast
  async getSMPForecast(date?: string): Promise<SMPForecast[]> {
    const params = date ? { date } : {};
    const response = await this.client.get<SMPForecast[]>('/smp-forecast', { params });
    return response.data;
  }

  // Bids
  async getBids(status?: string): Promise<DailyBid[]> {
    const params = status ? { status } : {};
    const response = await this.client.get<DailyBid[]>('/bids', { params });
    return response.data;
  }

  async getBid(bidId: string): Promise<DailyBid> {
    const response = await this.client.get<DailyBid>(`/bids/${bidId}`);
    return response.data;
  }

  async createBid(bid: Partial<DailyBid>): Promise<DailyBid> {
    const response = await this.client.post<DailyBid>('/bids', bid);
    return response.data;
  }

  async updateBid(bidId: string, updates: Partial<DailyBid>): Promise<DailyBid> {
    const response = await this.client.put<DailyBid>(`/bids/${bidId}`, updates);
    return response.data;
  }

  async optimizeBid(
    bidId: string,
    strategy: 'conservative' | 'moderate' | 'aggressive'
  ): Promise<DailyBid> {
    const response = await this.client.post<DailyBid>(`/bids/${bidId}/optimize`, {
      strategy,
    });
    return response.data;
  }

  async submitBid(bidId: string): Promise<DailyBid> {
    const response = await this.client.post<DailyBid>(`/bids/${bidId}/submit`);
    return response.data;
  }

  // Resources
  async getResources(): Promise<Resource[]> {
    const response = await this.client.get<Resource[]>('/resources');
    return response.data;
  }

  async getResource(resourceId: string): Promise<Resource> {
    const response = await this.client.get<Resource>(`/resources/${resourceId}`);
    return response.data;
  }

  // Settlements
  async getSettlements(params: {
    startDate?: string;
    endDate?: string;
    resourceId?: string;
  }): Promise<Settlement[]> {
    const response = await this.client.get<Settlement[]>('/settlements', { params });
    return response.data;
  }

  async getSettlementSummary(period: '7d' | '30d' | '90d'): Promise<{
    totalRevenue: number;
    totalPenalty: number;
    netRevenue: number;
    avgImbalance: number;
  }> {
    const response = await this.client.get('/settlements/summary', {
      params: { period },
    });
    return response.data;
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
