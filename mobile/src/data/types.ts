// 허용 목록 API 응답의 모양과 검사기(명세 6.3절). 앱이 쓰는 필드만 검사하고, 틀리면 ApiShapeError로 실패한다
// (조용히 0이나 기본값으로 바꾸지 않는다). 값이 없을 수 있는 필드는 null을 허용하고, 그 판단은 어댑터가 한다.
export class ApiShapeError extends Error {
  readonly endpoint: string;
  readonly path: string;
  constructor(endpoint: string, path: string, expected: string) {
    super(`API response shape: ${endpoint} ${path} (${expected})`);
    this.name = 'ApiShapeError';
    this.endpoint = endpoint;
    this.path = path;
  }
}

type Rec = Record<string, unknown>;
type Ctx = { endpoint: string };
const fail = (c: Ctx, path: string, expected: string): never => {
  throw new ApiShapeError(c.endpoint, path, expected);
};
const isRec = (v: unknown): v is Rec => typeof v === 'object' && v !== null && !Array.isArray(v);
const rec = (v: unknown, c: Ctx, p: string): Rec => (isRec(v) ? v : fail(c, p, 'object'));
const arr = (v: unknown, c: Ctx, p: string): unknown[] => (Array.isArray(v) ? v : fail(c, p, 'array'));
const str = (v: unknown, c: Ctx, p: string): string => (typeof v === 'string' ? v : fail(c, p, 'string'));
const num = (v: unknown, c: Ctx, p: string): number => (typeof v === 'number' && Number.isFinite(v) ? v : fail(c, p, 'finite number'));
const int = (v: unknown, c: Ctx, p: string): number => (Number.isInteger(v) ? (v as number) : fail(c, p, 'integer'));
const strOrNull = (v: unknown, c: Ctx, p: string): string | null => (v === undefined || v === null ? null : str(v, c, p));
const numOrNull = (v: unknown, c: Ctx, p: string): number | null => (v === undefined || v === null ? null : num(v, c, p));
const intOrNull = (v: unknown, c: Ctx, p: string): number | null => (v === undefined || v === null ? null : int(v, c, p));
const series = (v: unknown, c: Ctx, p: string): (number | null)[] => arr(v, c, p).map((x, i) => numOrNull(x, c, `${p}[${i}]`));

/** market_type은 오늘 화면의 isAvailableDayAhead(utils/smpPresentation)가 'day_ahead'인지 본다(없으면 확정 가격 없음). */
export type DayAheadSmp = {
  status: string; current_smp: number | null; trade_day: string | null; trade_hour: number | null;
  market_type: string | null; data_source: string | null; source_url: string | null;
};
export function parseDayAhead(json: unknown): DayAheadSmp {
  const c = { endpoint: '/api/v1/smp/day-ahead' };
  const r = rec(json, c, '$');
  return {
    status: str(r.status, c, 'status'), current_smp: numOrNull(r.current_smp, c, 'current_smp'),
    trade_day: strOrNull(r.trade_day, c, 'trade_day'), trade_hour: intOrNull(r.trade_hour, c, 'trade_hour'),
    market_type: strOrNull(r.market_type, c, 'market_type'),
    data_source: strOrNull(r.data_source, c, 'data_source'), source_url: strOrNull(r.source_url, c, 'source_url'),
  };
}

export type SmpForecast = {
  q10: (number | null)[]; q50: (number | null)[]; q90: (number | null)[]; model_used: string;
  created_at: string | null; data_source: string | null; forecast_date: string | null;
};
export function parseForecast(json: unknown): SmpForecast {
  const c = { endpoint: '/api/v1/smp-forecast' };
  const r = rec(json, c, '$');
  const q10 = series(r.q10, c, 'q10');
  const q50 = series(r.q50, c, 'q50');
  const q90 = series(r.q90, c, 'q90');
  if (q10.length !== q50.length || q90.length !== q50.length || q50.length > 48) fail(c, 'q10/q50/q90', 'same length ≤ 48');
  return {
    q10, q50, q90, model_used: str(r.model_used, c, 'model_used'), created_at: strOrNull(r.created_at, c, 'created_at'),
    data_source: strOrNull(r.data_source, c, 'data_source'), forecast_date: strOrNull(r.forecast_date, c, 'forecast_date'),
  };
}

/** data_source: API의 수치 설명(기준선이면 "최근 150일 재현 MAPE · … 마지막 30일 보류 검증"). 정확도 시트의 API 표기 기간에 쓴다(D8). */
export type ModelInfo = {
  status: string; version: string | null; type: string | null; mape: number | null; coverage: number | null; data_source: string | null;
};
export function parseModelInfo(json: unknown): ModelInfo {
  const c = { endpoint: '/api/v1/model/info' };
  const r = rec(json, c, '$');
  return {
    status: str(r.status, c, 'status'), version: strOrNull(r.version, c, 'version'), type: strOrNull(r.type, c, 'type'),
    mape: numOrNull(r.mape, c, 'mape'), coverage: numOrNull(r.coverage, c, 'coverage'),
    data_source: strOrNull(r.data_source, c, 'data_source'),
  };
}

export type KpiWeather = { temperature: number | null; wind_speed: number | null; humidity: number | null; condition: string | null };
export type DashboardKpis = {
  current_demand_mw: number | null; supply_reserve_rate: number | null; weather: KpiWeather | null;
  /** 출처 판정(7.4절)은 비문자열도 따로 다루므로 원값 그대로 둔다. */
  data_source: unknown;
};
export function parseKpis(json: unknown): DashboardKpis {
  const c = { endpoint: '/api/v1/dashboard/kpis' };
  const r = rec(json, c, '$');
  let weather: KpiWeather | null = null;
  if (r.weather !== undefined && r.weather !== null) {
    const w = rec(r.weather, c, 'weather');
    weather = {
      temperature: numOrNull(w.temperature, c, 'weather.temperature'), wind_speed: numOrNull(w.wind_speed, c, 'weather.wind_speed'),
      humidity: numOrNull(w.humidity, c, 'weather.humidity'), condition: strOrNull(w.condition, c, 'weather.condition'),
    };
  }
  return {
    current_demand_mw: numOrNull(r.current_demand_mw, c, 'current_demand_mw'),
    supply_reserve_rate: numOrNull(r.supply_reserve_rate, c, 'supply_reserve_rate'), weather, data_source: r.data_source,
  };
}

export type PowerSupplyRow = { hour: number; supply: number | null; demand: number | null };
/** data_source: 공급능력이 실시간 값으로 맞춘 것인지 밝히는 글자(round 2 integrity, provenance.supplyPair가 읽는다). */
export type PowerSupply = { current_hour: number | null; data: PowerSupplyRow[]; data_source: string | null };
export function parsePowerSupply(json: unknown): PowerSupply {
  const c = { endpoint: '/api/v1/power-supply' };
  const r = rec(json, c, '$');
  const data = arr(r.data, c, 'data').map((row, i) => {
    const x = rec(row, c, `data[${i}]`);
    return { hour: int(x.hour, c, `data[${i}].hour`), supply: numOrNull(x.supply, c, `data[${i}].supply`), demand: numOrNull(x.demand, c, `data[${i}].demand`) };
  });
  return { current_hour: intOrNull(r.current_hour, c, 'current_hour'), data, data_source: strOrNull(r.data_source, c, 'data_source') };
}

export type MarketStatus = {
  current_time: string;
  dam: { status: string; deadline: string; trading_date: string; hours_remaining: number | null };
  rtm: { status: string; next_interval: string | null; interval_minutes: number | null };
};
export function parseMarketStatus(json: unknown): MarketStatus {
  const c = { endpoint: '/api/v1/market-status' };
  const r = rec(json, c, '$');
  const dam = rec(r.dam, c, 'dam');
  const rtm = rec(r.rtm, c, 'rtm');
  return {
    current_time: str(r.current_time, c, 'current_time'),
    dam: {
      status: str(dam.status, c, 'dam.status'), deadline: str(dam.deadline, c, 'dam.deadline'),
      trading_date: str(dam.trading_date, c, 'dam.trading_date'), hours_remaining: numOrNull(dam.hours_remaining, c, 'dam.hours_remaining'),
    },
    rtm: {
      status: str(rtm.status, c, 'rtm.status'), next_interval: strOrNull(rtm.next_interval, c, 'rtm.next_interval'),
      interval_minutes: numOrNull(rtm.interval_minutes, c, 'rtm.interval_minutes'),
    },
  };
}

export type BidSegment = { segment_id: number; quantity_mw: number; price_krw_mwh: number; clearing_probability: number; expected_revenue: number | null };
export type HourlyBid = { hour: number; segments: BidSegment[]; smp_forecast: { q10: number | null; q50: number | null; q90: number | null }; expected_revenue: number | null };
export type OptimizedSegments = {
  trading_date: string; capacity_mw: number; risk_level: string; hourly_bids: HourlyBid[];
  model_used: string | null; optimization_method: string | null;
};
export function parseSegments(json: unknown): OptimizedSegments {
  const c = { endpoint: '/api/v1/bidding/optimized-segments' };
  const r = rec(json, c, '$');
  const hourly_bids = arr(r.hourly_bids, c, 'hourly_bids').map((item, i) => {
    const h = rec(item, c, `hourly_bids[${i}]`);
    const p = `hourly_bids[${i}]`;
    const fc = rec(h.smp_forecast, c, `${p}.smp_forecast`);
    return {
      hour: int(h.hour, c, `${p}.hour`),
      segments: arr(h.segments, c, `${p}.segments`).map((s, j) => {
        const g = rec(s, c, `${p}.segments[${j}]`);
        const q = `${p}.segments[${j}]`;
        return {
          segment_id: int(g.segment_id, c, `${q}.segment_id`), quantity_mw: num(g.quantity_mw, c, `${q}.quantity_mw`),
          price_krw_mwh: num(g.price_krw_mwh, c, `${q}.price_krw_mwh`), clearing_probability: num(g.clearing_probability, c, `${q}.clearing_probability`),
          expected_revenue: numOrNull(g.expected_revenue, c, `${q}.expected_revenue`),
        };
      }),
      smp_forecast: { q10: numOrNull(fc.q10, c, `${p}.smp_forecast.q10`), q50: numOrNull(fc.q50, c, `${p}.smp_forecast.q50`), q90: numOrNull(fc.q90, c, `${p}.smp_forecast.q90`) },
      expected_revenue: numOrNull(h.expected_revenue, c, `${p}.expected_revenue`),
    };
  });
  return {
    trading_date: str(r.trading_date, c, 'trading_date'), capacity_mw: num(r.capacity_mw, c, 'capacity_mw'),
    risk_level: str(r.risk_level, c, 'risk_level'), hourly_bids, model_used: strOrNull(r.model_used, c, 'model_used'),
    optimization_method: strOrNull(r.optimization_method, c, 'optimization_method'),
  };
}

export type ApiPartStatus = { status: string; data_date_status: string | null };
export type RealtimeStatus = {
  timestamp: string | null; overall_status: string;
  smp_api: ApiPartStatus | null; power_supply_api: ApiPartStatus | null; weather_api: ApiPartStatus | null;
};
export function parseRealtimeStatus(json: unknown): RealtimeStatus {
  const c = { endpoint: '/api/v1/realtime-status' };
  const r = rec(json, c, '$');
  // 일부만 오는 응답(4.4절)도 그대로 그리므로 각 부분은 없을 수 있다. 있으면 status는 문자열이어야 한다.
  const part = (key: string): ApiPartStatus | null => {
    if (r[key] === undefined || r[key] === null) return null;
    const x = rec(r[key], c, key);
    return { status: str(x.status, c, `${key}.status`), data_date_status: strOrNull(x.data_date_status, c, `${key}.data_date_status`) };
  };
  return {
    timestamp: strOrNull(r.timestamp, c, 'timestamp'), overall_status: str(r.overall_status, c, 'overall_status'),
    smp_api: part('smp_api'), power_supply_api: part('power_supply_api'), weather_api: part('weather_api'),
  };
}
