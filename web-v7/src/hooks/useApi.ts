/**
 * Custom hooks for API data fetching
 */

import { useState, useEffect, useCallback } from 'react';
import { apiService, mockData, isApiAvailable } from '../services/api';
import type {
  SMPForecast,
  MarketStatus,
  DashboardKPIs,
  Resource,
  ModelInfo,
  SettlementRecord,
  SettlementStats,
  OptimizedBids,
  PowerSupplyResponse,
  RealtimeStatus,
  CurrentSMP,
} from '../types';

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

// Generic hook for API calls with fallback
function useApiData<T>(
  fetchFn: () => Promise<T>,
  mockFn: () => T,
  deps: unknown[] = []
): UseApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      console.error('API Error, using mock data:', err);
      setError('API unavailable, showing demo data');
      setData(mockFn());
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

// SMP Forecast Hook
export function useSMPForecast(): UseApiState<SMPForecast> {
  return useApiData(
    () => apiService.getSMPForecast(),
    () => mockData.getSMPForecast()
  );
}

// Current SMP Hook (Real-time)
export function useCurrentSMP(): UseApiState<CurrentSMP> {
  return useApiData(
    () => apiService.getCurrentSMP(),
    () => mockData.getCurrentSMP()
  );
}

// Dashboard KPIs Hook
export function useDashboardKPIs(): UseApiState<DashboardKPIs> {
  return useApiData(
    () => apiService.getDashboardKPIs(),
    () => mockData.getDashboardKPIs()
  );
}

// Market Status Hook
export function useMarketStatus(): UseApiState<MarketStatus> {
  return useApiData(
    () => apiService.getMarketStatus(),
    () => mockData.getMarketStatus()
  );
}

// Resources Hook
export function useResources(): UseApiState<Resource[]> {
  return useApiData(
    () => apiService.getResources(),
    () => mockData.getResources()
  );
}

// Model Info Hook
export function useModelInfo(): UseApiState<ModelInfo> {
  return useApiData(
    () => apiService.getModelInfo(),
    () => mockData.getModelInfo()
  );
}

// Power Supply Hook (실측 + 예측)
export function usePowerSupply(): UseApiState<PowerSupplyResponse> {
  return useApiData(
    () => apiService.getPowerSupply(),
    () => mockData.getPowerSupply()
  );
}

// Optimized Bids Hook
export function useOptimizedBids(
  capacityMw: number = 50,
  riskLevel: string = 'moderate'
): UseApiState<OptimizedBids> {
  return useApiData(
    () => apiService.getOptimizedSegments(capacityMw, riskLevel),
    () => ({
      trading_date: new Date(Date.now() + 86400000).toISOString().split('T')[0],
      capacity_mw: capacityMw,
      risk_level: riskLevel,
      hourly_bids: [],
      total_daily_mwh: 0,
      model_used: 'Mock',
    }),
    [capacityMw, riskLevel]
  );
}

// Settlements Hook
export function useSettlements(days: number = 7): UseApiState<SettlementRecord[]> {
  return useApiData(
    () => apiService.getRecentSettlements(days),
    () => [],
    [days]
  );
}

// Settlement Summary Hook
export function useSettlementSummary(): UseApiState<SettlementStats> {
  return useApiData(
    () => apiService.getSettlementSummary(),
    () => ({
      generation_revenue_million: 1250.5,
      generation_change_pct: 5.2,
      imbalance_charges_million: -45.3,
      imbalance_change_pct: -12.5,
      net_revenue_million: 1205.2,
      net_change_pct: 6.8,
      forecast_accuracy_pct: 94.5,
      accuracy_change_pct: 1.2,
    })
  );
}

// API Status Hook
export function useApiStatus() {
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      const available = await isApiAvailable();
      setIsAvailable(available);
    };
    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  return isAvailable;
}

// Auto-refresh hook
export function useAutoRefresh(callback: () => void, intervalMs: number = 60000) {
  useEffect(() => {
    const interval = setInterval(callback, intervalMs);
    return () => clearInterval(interval);
  }, [callback, intervalMs]);
}

// Realtime Status Hook (API connection status)
export function useRealtimeStatus(): UseApiState<RealtimeStatus> {
  const [data, setData] = useState<RealtimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.getRealtimeStatus();
      setData(result);
    } catch (err) {
      console.error('Realtime status fetch failed:', err);
      setError('Unable to fetch API status');
      // Fallback mock status
      setData({
        smp_api: { status: 'unknown', last_update: null },
        power_supply_api: { status: 'unknown', last_update: null },
        weather_api: { status: 'unknown', last_update: null },
        overall_status: 'all_error',
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
    // Check status every 30 seconds
    const interval = setInterval(fetch, 30000);
    return () => clearInterval(interval);
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}
