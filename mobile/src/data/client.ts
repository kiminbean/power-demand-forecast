// 허용 목록 GET 중 앱이 쓰는 것만 감싼다(명세 2.1절 3, 6.3절). 메서드 이름은 옛 src/services/api.ts와 같다.
// 기준 URL은 src/config/environment.ts 규칙(웹은 같은 출처 상대 경로)을 따른다. 처리한 실패는 상태로 보여 주므로 로그를 남기지 않는다.
import { API_URL } from '../config/environment';
import { EXAMPLE_PLANT } from './examples';
import {
  ApiShapeError, parseDayAhead, parseForecast, parseKpis, parseMarketStatus, parseModelInfo, parsePowerSupply,
  parseRealtimeStatus, parseSegments,
} from './types';

export const ENDPOINTS = Object.freeze({
  dayAhead: '/api/v1/smp/day-ahead',
  forecast: '/api/v1/smp-forecast',
  modelInfo: '/api/v1/model/info',
  kpis: '/api/v1/dashboard/kpis',
  powerSupply: '/api/v1/power-supply',
  marketStatus: '/api/v1/market-status',
  segments: '/api/v1/bidding/optimized-segments',
  realtimeStatus: '/api/v1/realtime-status',
});

export const RISK_LEVELS = ['conservative', 'moderate', 'aggressive'] as const;
export type RiskLevel = (typeof RISK_LEVELS)[number];

export class ApiHttpError extends Error {
  readonly endpoint: string;
  readonly status: number;
  constructor(endpoint: string, status: number) {
    super(`API 오류 ${status}: ${endpoint}`);
    this.name = 'ApiHttpError';
    this.endpoint = endpoint;
    this.status = status;
  }
  /** 502·503만 재시도할 가치가 있다(6.3절). 4xx는 다시 보내도 같다. */
  get retryable(): boolean {
    return this.status === 502 || this.status === 503;
  }
}

export class ApiNetworkError extends Error {
  readonly endpoint: string;
  readonly retryable = true;
  constructor(endpoint: string) {
    super(`API에 닿지 못했습니다: ${endpoint}`);
    this.name = 'ApiNetworkError';
    this.endpoint = endpoint;
  }
}

export type FetchResponseLike = { ok: boolean; status: number; json(): Promise<unknown> };
export type FetchLike = (url: string, init: { method: 'GET'; signal?: AbortSignal }) => Promise<FetchResponseLike>;
export type ClientOptions = { baseUrl?: string; fetchImpl?: FetchLike };

const isAbort = (error: unknown) => typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'AbortError';

export function createClient(options: ClientOptions = {}) {
  const base = (options.baseUrl ?? API_URL).replace(/\/+$/, '');
  const doFetch: FetchLike = options.fetchImpl ?? ((url, init) => fetch(url, init));

  // Content-Type을 붙이지 않는다: GET 본문이 없고, 붙이면 다른 출처 설정에서 사전 요청이 생긴다.
  async function get<T>(path: string, parse: (json: unknown) => T, signal?: AbortSignal): Promise<T> {
    let response: FetchResponseLike;
    try {
      response = await doFetch(base + path, { method: 'GET', signal });
    } catch (error) {
      if (isAbort(error)) throw error;
      throw new ApiNetworkError(path);
    }
    if (!response.ok) throw new ApiHttpError(path, response.status);
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new ApiShapeError(path, '$', 'JSON');
    }
    return parse(body);
  }

  return {
    getDayAheadSMP: (signal?: AbortSignal) => get(ENDPOINTS.dayAhead, parseDayAhead, signal),
    getSMPForecast: (signal?: AbortSignal) => get(ENDPOINTS.forecast, parseForecast, signal),
    getModelInfo: (signal?: AbortSignal) => get(ENDPOINTS.modelInfo, parseModelInfo, signal),
    getDashboardKPIs: (signal?: AbortSignal) => get(ENDPOINTS.kpis, parseKpis, signal),
    getPowerSupply: (signal?: AbortSignal) => get(ENDPOINTS.powerSupply, parsePowerSupply, signal),
    getMarketStatus: (signal?: AbortSignal) => get(ENDPOINTS.marketStatus, parseMarketStatus, signal),
    getOptimizedSegments: (risk: RiskLevel, signal?: AbortSignal) => {
      if (!RISK_LEVELS.includes(risk)) throw new TypeError(`알 수 없는 입찰 성향: ${String(risk)}`);
      const query = `?capacity_mw=${EXAMPLE_PLANT.capacityMw.value}&risk_level=${risk}`;
      return get(ENDPOINTS.segments + query, parseSegments, signal);
    },
    getRealtimeStatus: (signal?: AbortSignal) => get(ENDPOINTS.realtimeStatus, parseRealtimeStatus, signal),
  };
}

export type ApiClient = ReturnType<typeof createClient>;
