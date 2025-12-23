/**
 * RE-BMS API Hooks
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback } from 'react';
import apiService, {
  Resource,
  SMPForecast,
  MarketStatus,
  DashboardKPIs,
  OptimizedBids,
  SettlementRecord,
  SettlementStats,
  ModelInfo,
  CurrentSMP,
  BiddingStrategy,
  SimulationResult,
  PowerSupplyData,
} from '../services/api';

// Generic fetch hook
interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

function useFetch<T>(
  fetchFn: () => Promise<T>,
  deps: any[] = []
): FetchState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setLoading(false);
    }
  }, deps);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { data, loading, error, refetch };
}

// Market Status Hook
export function useMarketStatus() {
  return useFetch<MarketStatus>(() => apiService.getMarketStatus(), []);
}

// SMP Forecast Hook
export function useSMPForecast() {
  return useFetch<SMPForecast>(() => apiService.getSMPForecast(), []);
}

// Current SMP Hook
export function useCurrentSMP(region: string = 'jeju') {
  return useFetch<CurrentSMP>(() => apiService.getCurrentSMP(region), [region]);
}

// Dashboard KPIs Hook
export function useDashboardKPIs() {
  return useFetch<DashboardKPIs>(() => apiService.getDashboardKPIs(), []);
}

// Resources Hook
export function useResources() {
  return useFetch<Resource[]>(() => apiService.getResources(), []);
}

// Model Info Hook
export function useModelInfo() {
  return useFetch<ModelInfo>(() => apiService.getModelInfo(), []);
}

// Optimized Segments Hook
export function useOptimizedSegments(capacityMw: number = 50, riskLevel: string = 'moderate') {
  return useFetch<OptimizedBids>(
    () => apiService.getOptimizedSegments(capacityMw, riskLevel),
    [capacityMw, riskLevel]
  );
}

// Settlements Hooks
export function useRecentSettlements(days: number = 7) {
  return useFetch<SettlementRecord[]>(
    () => apiService.getRecentSettlements(days),
    [days]
  );
}

export function useSettlementSummary() {
  return useFetch<SettlementStats>(() => apiService.getSettlementSummary(), []);
}

// Power Supply Hook
export function usePowerSupply() {
  return useFetch<PowerSupplyData>(() => apiService.getPowerSupply(), []);
}

// Bidding Strategy Hook
export function useBiddingStrategy(
  capacityKw: number = 50000,
  energyType: string = 'solar',
  riskLevel: string = 'moderate'
) {
  return useFetch<BiddingStrategy>(
    () => apiService.getBiddingStrategy(capacityKw, energyType, riskLevel),
    [capacityKw, energyType, riskLevel]
  );
}

// Revenue Simulation Hook
export function useRevenueSimulation(
  capacityKw: number = 50000,
  energyType: string = 'solar',
  hours: number = 24
) {
  return useFetch<SimulationResult>(
    () => apiService.simulateRevenue(capacityKw, energyType, hours),
    [capacityKw, energyType, hours]
  );
}

// Mutation hooks for bidding
export function useBidActions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const getOptimizedSegments = useCallback(
    async (capacityMw: number = 50, riskLevel: string = 'moderate') => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.getOptimizedSegments(capacityMw, riskLevel);
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to get optimized segments'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const simulateDAM = useCallback(
    async (segments: Array<{ segment_id: number; quantity_mw: number; price_krw_mwh: number }>, hour: number) => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.simulateDAM({ segments, hour });
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to simulate DAM'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const getBiddingStrategy = useCallback(
    async (capacityKw: number = 50000, energyType: string = 'solar', riskLevel: string = 'moderate') => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.getBiddingStrategy(capacityKw, energyType, riskLevel);
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to get bidding strategy'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const simulateRevenue = useCallback(
    async (capacityKw: number = 50000, energyType: string = 'solar', hours: number = 24) => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.simulateRevenue(capacityKw, energyType, hours);
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to simulate revenue'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    getOptimizedSegments,
    simulateDAM,
    getBiddingStrategy,
    simulateRevenue,
  };
}
