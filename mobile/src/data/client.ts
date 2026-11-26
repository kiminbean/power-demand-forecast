// 허용 목록 GET 중 앱이 쓰는 것만 감싼다(명세 2.1절 3, 6.3절). 메서드 이름은 옛 src/services/api.ts와 같다.
// 기준 URL은 src/config/environment.ts 규칙(웹은 같은 출처 상대 경로)을 따른다. 처리한 실패는 상태로 보여 주므로 로그를 남기지 않는다.
import { API_URL } from '../config/environment';
import { markArrived } from '../lib/resource';
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
    super(`API HTTP ${status}: ${endpoint}`);
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
    super(`API unreachable: ${endpoint}`);
    this.name = 'ApiNetworkError';
    this.endpoint = endpoint;
  }
}

export type FetchResponseLike = { ok: boolean; status: number; json(): Promise<unknown> };
export type FetchLike = (url: string, init: { method: 'GET'; signal?: AbortSignal }) => Promise<FetchResponseLike>;
/**
 * index.html이 번들을 받는 동안 미리 시작한 첫 화면 GET(round 1 D3, 명세 8.1·9절): 시작 시각 t, URL → 응답 약속 r, URL → 응답이 도착한
 * 기기 시각 a(round 2 NOW-KST), 닫힘 x. 그 스크립트는 경로마다 그 화면이 처음 그릴 때 보내는 요청만 시작한다(tests/shell-html이
 * ENDPOINTS·예시 용량과 맞춰 본다).
 */
export type PrefetchTable = {
  t: number; r: Record<string, Promise<FetchResponseLike> | undefined>; a?: Record<string, number | undefined>; x?: boolean;
};
/** 미리 받은 응답을 쓰는 나이 한도: 자료 저장소의 새로 받기 주기(lib/resource REFRESH_EVERY_MS, 5분)와 같다. */
export const PREFETCH_MAX_AGE_MS = 5 * 60_000;
/** signal: 이 클라이언트의 모든 요청에 붙는 중단 신호(자료 저장소가 요청마다 10초 신호를 묶은 클라이언트를 만든다). */
export type ClientOptions = { baseUrl?: string; fetchImpl?: FetchLike; signal?: AbortSignal; prefetch?: () => PrefetchTable | null; now?: () => number };

const isAbort = (error: unknown) => typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'AbortError';
const globalPrefetch = () => (globalThis as { __PD_PREFETCH__?: PrefetchTable }).__PD_PREFETCH__ ?? null;

/**
 * 미리 시작한 요청을 URL마다 한 번만 꺼낸다(꺼내면 지운다). 처음 부를 때 표를 닫아 아직 시작하지 않은 미리 받기를 막는다(같은 GET이
 * 두 번 나가지 않게). 없거나 5분 넘었으면 null이라서 보통 요청을 보낸다.
 */
export function takePrefetch(table: PrefetchTable | null, url: string, now: number): Promise<FetchResponseLike> | null {
  if (!table || typeof table !== 'object' || !table.r) return null;
  table.x = true;
  const p = Object.prototype.hasOwnProperty.call(table.r, url) ? table.r[url] : undefined;
  delete table.r[url];
  const age = now - table.t;
  return p && typeof p.then === 'function' && age >= 0 && age < PREFETCH_MAX_AGE_MS ? p : null;
}

export function createClient(options: ClientOptions = {}) {
  const base = (options.baseUrl ?? API_URL).replace(/\/+$/, '');
  const doFetch: FetchLike = options.fetchImpl ?? ((url, init) => fetch(url, init));
  const prefetch = options.prefetch ?? (options.fetchImpl ? () => null : globalPrefetch);
  const now = options.now ?? Date.now;

  // Content-Type을 붙이지 않는다: GET 본문이 없고, 붙이면 다른 출처 설정에서 사전 요청이 생긴다.
  async function get<T>(path: string, parse: (json: unknown) => T, signal: AbortSignal | undefined = options.signal): Promise<T> {
    let response: FetchResponseLike;
    const url = base + path;
    const table = prefetch();
    const early = takePrefetch(table, url, now());
    try {
      // 미리 받은 응답은 상태(503 포함)를 그대로 쓴다(재시도 규칙이 같다). 연결이 실패했으면 바로 보통 요청으로 다시 받는다.
      response = await (early ? early.catch(() => doFetch(url, { method: 'GET', signal })) : doFetch(url, { method: 'GET', signal }));
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
    // 미리 받은 응답이면 도착한 기기 시각을 값에 붙인다(자료 저장소의 조회 시각, round 2 NOW-KST). 미리 받기가 실패해 보통 요청으로
    // 받았으면 도착 시각이 없다(그때는 저장소가 받은 순간을 쓴다).
    const arrived = early && table && table.a ? table.a[url] : undefined;
    return typeof arrived === 'number' ? markArrived(parse(body), arrived) : parse(body);
  }

  return {
    getDayAheadSMP: (signal?: AbortSignal) => get(ENDPOINTS.dayAhead, parseDayAhead, signal),
    getSMPForecast: (signal?: AbortSignal) => get(ENDPOINTS.forecast, parseForecast, signal),
    getModelInfo: (signal?: AbortSignal) => get(ENDPOINTS.modelInfo, parseModelInfo, signal),
    getDashboardKPIs: (signal?: AbortSignal) => get(ENDPOINTS.kpis, parseKpis, signal),
    getPowerSupply: (signal?: AbortSignal) => get(ENDPOINTS.powerSupply, parsePowerSupply, signal),
    getMarketStatus: (signal?: AbortSignal) => get(ENDPOINTS.marketStatus, parseMarketStatus, signal),
    getOptimizedSegments: (risk: RiskLevel, signal?: AbortSignal) => {
      if (!RISK_LEVELS.includes(risk)) throw new TypeError(`unknown risk level: ${String(risk)}`);
      const query = `?capacity_mw=${EXAMPLE_PLANT.capacityMw.value}&risk_level=${risk}`;
      return get(ENDPOINTS.segments + query, parseSegments, signal);
    },
    getRealtimeStatus: (signal?: AbortSignal) => get(ENDPOINTS.realtimeStatus, parseRealtimeStatus, signal),
  };
}

export type ApiClient = ReturnType<typeof createClient>;
