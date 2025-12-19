/**
 * RE-BMS API Hooks
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback } from 'react';
import apiService, {
  DailyBid,
  Resource,
  Settlement,
  SMPForecast,
  MarketStatus,
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
export function useSMPForecast(date?: string) {
  return useFetch<SMPForecast[]>(
    () => apiService.getSMPForecast(date),
    [date]
  );
}

// Bids Hooks
export function useBids(status?: string) {
  return useFetch<DailyBid[]>(() => apiService.getBids(status), [status]);
}

export function useBid(bidId: string) {
  return useFetch<DailyBid>(() => apiService.getBid(bidId), [bidId]);
}

// Resources Hook
export function useResources() {
  return useFetch<Resource[]>(() => apiService.getResources(), []);
}

export function useResource(resourceId: string) {
  return useFetch<Resource>(
    () => apiService.getResource(resourceId),
    [resourceId]
  );
}

// Settlements Hook
export function useSettlements(params: {
  startDate?: string;
  endDate?: string;
  resourceId?: string;
}) {
  return useFetch<Settlement[]>(
    () => apiService.getSettlements(params),
    [params.startDate, params.endDate, params.resourceId]
  );
}

// Mutation hooks
export function useBidMutations() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const createBid = useCallback(async (bid: Partial<DailyBid>) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.createBid(bid);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to create bid'));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateBid = useCallback(
    async (bidId: string, updates: Partial<DailyBid>) => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.updateBid(bidId, updates);
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to update bid'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const optimizeBid = useCallback(
    async (
      bidId: string,
      strategy: 'conservative' | 'moderate' | 'aggressive'
    ) => {
      setLoading(true);
      setError(null);
      try {
        const result = await apiService.optimizeBid(bidId, strategy);
        return result;
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to optimize bid'));
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const submitBid = useCallback(async (bidId: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.submitBid(bidId);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to submit bid'));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    createBid,
    updateBid,
    optimizeBid,
    submitBid,
  };
}
