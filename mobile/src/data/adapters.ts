// 응답 → Provenanced<T>(명세 7.2·7.3절). 칩 없는 숫자가 화면에 가지 않도록, 값·단위·상태·칩·출처·시각을 한 덩어리로 묶는다.
// 브랜드는 내보내지 않는 unique symbol이라 이 파일 밖에서는 객체 리터럴로 Provenanced를 만들 수 없다.
import { clockSkew, parseApiTime, parseDayKey } from '../lib/kst';
import type { ClockSkew } from '../lib/kst';
import { kstTime, tradeHour } from '../lib/format';
import {
  CHIPS, demandProvenance, forecastChip, forecastShortName, reserveLevel, smpForecastKind, supplyPair, SUPPLY_MISSING, weatherProvenance,
} from '../lib/provenance';
import type { Chip, DemandProvenance, ForecastKind, ReserveLevel, WeatherKind } from '../lib/provenance';
import { createdAtText, forecastDateText, lookupAt, observedLookup, serverTimeText } from '../content/strings';
import { exampleFromCalculator } from './examples';
import type { Example } from './examples';
import { ApiShapeError } from './types';
import type { DashboardKpis, DayAheadSmp, MarketStatus, OptimizedSegments, PowerSupply, SmpForecast } from './types';

declare const provenancedBrand: unique symbol;

export type ProvState = 'confirmed' | 'measured' | 'forecast' | 'estimated' | 'default' | 'national';
export type TimeKind = '거래시간' | '조회' | '관측' | '만든 시각' | '서버 시각' | '평가 기간';
export type Unit = '원/kWh' | 'MW' | '%' | '°C' | 'm/s' | '';
/** 명세 7.2절의 모양에 칩(같은 상태라도 문구가 다르다: "추정 · 실측 아님"·"시뮬레이션 · 실측 아님")을 더했다. */
export type ProvenancedFields<T> = {
  value: T; unit: Unit; state: ProvState; chip: Chip; source: string; time: { kind: TimeKind; kst: string }; note?: string;
};
export type Provenanced<T> = Readonly<ProvenancedFields<T>> & { readonly [provenancedBrand]: true };

function provenanced<T>(fields: ProvenancedFields<T>): Provenanced<T> {
  return Object.freeze({ ...fields, time: Object.freeze({ ...fields.time }) }) as unknown as Provenanced<T>;
}

export type DayAheadView =
  | { status: 'ready'; price: Provenanced<number>; tradeHour: number; tradeDay: string; sourceUrl: string | null }
  | { status: 'unavailable' };

/** 하루전 확정 가격. "실측"이 아니라 "확정 가격"이다(7.1절). */
export function adaptDayAhead(res: DayAheadSmp, fetchedAt: number): DayAheadView {
  const day = parseDayKey(res.trade_day);
  const hour = res.trade_hour;
  if (res.status !== 'available' || res.current_smp === null || day === null || hour === null || hour < 1 || hour > 24) {
    return { status: 'unavailable' };
  }
  const price = provenanced({
    value: res.current_smp, unit: '원/kWh', state: 'confirmed', chip: CHIPS.confirmedPrice,
    source: res.data_source ?? 'API 서버 하루전 가격 (출처 표기 없음)', time: { kind: '거래시간', kst: tradeHour(hour) }, note: lookupAt(fetchedAt),
  });
  return { status: 'ready', price, tradeHour: hour, tradeDay: day, sourceUrl: res.source_url };
}

export type ForecastView = {
  kind: ForecastKind; modelUsed: string; shortName: string; forecastDate: string | null; createdAt: number | null;
  q50: Provenanced<(number | null)[]>;
  /** 띠(q10–q90)는 기준선·학습 모델 예측에만. API 대체값의 분위수는 예측 구간이 아니다. */
  band: { q10: (number | null)[]; q90: (number | null)[] } | null;
};
const FORECAST_SOURCE: Record<ForecastKind, string> = {
  baseline: 'API 서버 기준선 예측', model: 'API 서버 학습 모델 예측', fallback: 'API 서버 대체값 (모델 예측 아님)',
};

/** 모델 문자열은 이 시리즈를 그린 같은 응답의 model_used에서만 가져온다(7.8절 3). */
export function adaptForecast(res: SmpForecast): ForecastView {
  const kind = smpForecastKind(res);
  const createdAt = parseApiTime(res.created_at);
  const forecastDate = parseDayKey(res.forecast_date);
  const q50 = provenanced({
    value: res.q50.slice(), unit: '원/kWh', state: kind === 'fallback' ? 'estimated' : 'forecast', chip: forecastChip(kind),
    source: FORECAST_SOURCE[kind], time: { kind: '만든 시각', kst: createdAtText(createdAt) }, note: forecastDateText(forecastDate),
  });
  return {
    kind, modelUsed: res.model_used, shortName: forecastShortName(res.model_used), forecastDate, createdAt, q50,
    band: kind === 'fallback' ? null : { q10: res.q10.slice(), q90: res.q90.slice() },
  };
}

export type ForecastPoint = {
  tradeHour: number;
  kind: 'value' | 'zero' | 'missing';
  q50: Provenanced<number> | null;
  band: { q10: Provenanced<number>; q90: Provenanced<number> } | null;
};
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * 거래시간 하나의 예측 값(판독 줄·표). 시리즈와 같은 칩·출처·시각을 물려받는다. 0.00 입력은 값으로 내보내지 않고(7.8절 4),
 * 띠는 띠가 있는 예측(기준선·학습 모델)에서 두 분위수가 다 있을 때만 준다.
 */
export function forecastPoint(view: ForecastView, tradeHour: number): ForecastPoint {
  const i = tradeHour - 1;
  const v = Number.isInteger(tradeHour) && tradeHour >= 1 && tradeHour <= 24 ? view.q50.value[i] : null;
  if (!finite(v)) return { tradeHour, kind: 'missing', q50: null, band: null };
  if (v === 0) return { tradeHour, kind: 'zero', q50: null, band: null };
  const { unit, state, chip, source, time, note } = view.q50;
  const base = { unit, state, chip, source, time };
  const q10 = view.band ? view.band.q10[i] : null;
  const q90 = view.band ? view.band.q90[i] : null;
  const band = finite(q10) && finite(q90)
    ? { q10: provenanced({ ...base, value: q10, note: '80% 구간 아래' }), q90: provenanced({ ...base, value: q90, note: '80% 구간 위' }) }
    : null;
  return { tradeHour, kind: 'value', q50: provenanced({ ...base, value: v, note }), band };
}

export type WeatherView = {
  kind: Exclude<WeatherKind, 'none'>;
  temperature: Provenanced<number> | null; windSpeed: Provenanced<number> | null; humidity: Provenanced<number> | null;
  condition: Provenanced<string> | null;
};
export type KpiView = {
  provenance: DemandProvenance; demand: Provenanced<number> | null; reserveRate: Provenanced<number> | null;
  /** 단계는 ③(제주 실측)일 때만. 끊김 철회는 화면 view model이 신선도로 한다. */
  level: ReserveLevel | null; weather: WeatherView | null;
};
const WEATHER_STATE: Record<Exclude<WeatherKind, 'none'>, ProvState> = { live: 'measured', file: 'estimated', default: 'default' };

export function adaptKpis(res: DashboardKpis, fetchedAt: number): KpiView {
  const prov = demandProvenance(res);
  const time = { kind: '조회' as const, kst: observedLookup(fetchedAt) };
  const demand = res.current_demand_mw === null ? null : provenanced({
    value: res.current_demand_mw, unit: 'MW', state: prov.demandState, chip: prov.demandChip, source: prov.source, time,
  });
  const reserveRate = res.supply_reserve_rate === null ? null : provenanced({
    value: res.supply_reserve_rate, unit: '%', state: prov.reserveState, chip: prov.reserveChip, source: prov.source, time,
    note: prov.reserveNote || undefined,
  });
  const level = prov.judge && reserveRate !== null ? reserveLevel(reserveRate.value) : null;
  const wp = weatherProvenance(res);
  let weather: WeatherView | null = null;
  if (wp.kind !== 'none' && wp.chip && wp.source && res.weather) {
    const kind = wp.kind;
    const w = res.weather;
    const base = { state: WEATHER_STATE[kind], chip: wp.chip, source: wp.source, time };
    const one = (value: number | null, unit: Unit) => (value === null ? null : provenanced({ ...base, value, unit }));
    weather = {
      kind, temperature: one(w.temperature, '°C'), windSpeed: one(w.wind_speed, 'm/s'), humidity: one(w.humidity, '%'),
      condition: w.condition === null ? null : provenanced({ ...base, value: w.condition, unit: '' }),
    };
  }
  return { provenance: prov, demand, reserveRate, level, weather };
}

export type SupplyView =
  | { status: 'paired'; hour: number; supply: Provenanced<number>; reserve: Provenanced<number>; watchBoundary: Provenanced<number> }
  | { status: 'missing'; text: typeof SUPPLY_MISSING };

/** 공급능력·예비력·관심 경계: 지금 시각 한 쌍이 ±1 MW로 맞을 때만 실측(7.4절). */
export function adaptSupply(kpis: DashboardKpis, powerSupply: PowerSupply, fetchedAt: number): SupplyView {
  const pair = supplyPair(kpis, powerSupply);
  if (!pair.ok) return { status: 'missing', text: pair.reason };
  const base = {
    unit: 'MW' as const, state: 'measured' as const, chip: CHIPS.measured, source: demandProvenance(kpis).source,
    time: { kind: '조회' as const, kst: observedLookup(fetchedAt) }, note: '수급 자료의 지금 시각 값',
  };
  return {
    status: 'paired', hour: pair.hour,
    supply: provenanced({ ...base, value: pair.supplyMw }),
    reserve: provenanced({ ...base, value: pair.reserveMw }),
    watchBoundary: provenanced({ ...base, value: pair.watchBoundaryMw }),
  };
}

export type MarketView = {
  serverTime: number | null; skew: ClockSkew | null; serverText: string;
  dam: MarketStatus['dam']; tradingDate: string | null; rtm: MarketStatus['rtm'];
};

/** 하루전 개장·마감의 기준은 서버 시각이다(7.5절). receivedAt은 응답을 받은 기기 시각. */
export function adaptMarketStatus(res: MarketStatus, receivedAt: number): MarketView {
  const serverTime = parseApiTime(res.current_time);
  return {
    serverTime, skew: serverTime === null ? null : clockSkew(serverTime, receivedAt), serverText: serverTimeText(serverTime),
    dam: { ...res.dam }, tradingDate: parseDayKey(res.dam.trading_date), rtm: { ...res.rtm },
  };
}

export type SegmentView = { id: number; price: Provenanced<number>; probability: Provenanced<number>; quantity: Example<number>; revenue: Example<number> | null };
export type HourBidView = { tradeHour: number; baselineQ50: number | null; revenue: Example<number> | null; segments: SegmentView[] };
export type SegmentsView = {
  tradingDate: string | null; risk: string; method: string | null; modelUsed: string | null; capacityMw: number;
  /** 계산기가 쓴 SMP 예측의 종류(같은 응답의 model_used, 7.8절 3). 칩 문구가 여기서 갈린다. */
  kind: ForecastKind; chip: Chip; hours: HourBidView[];
};
// 계산기 분위수의 출처: 기준선이면 "기준선 분위수", 학습 모델이면 "학습 모델 분위수", 대체값(패턴·규칙)이면 모델 예측이 아니다.
const CALC_CHIP: Readonly<Record<ForecastKind, Chip>> = { baseline: CHIPS.estQuantile, model: CHIPS.estModelQuantile, fallback: CHIPS.estApiFallback };

/**
 * 입찰 계산기 결과. hourly_bids[i].hour는 0부터라 거래시간 = hour + 1(7.8절 6). price_krw_mwh에는 원/kWh 값이 들어 있고
 * expected_revenue는 천 원 단위다(7.8절 5). 예시 용량에 기대는 물량·기대 수입은 Example, 가격·낙찰 가능성은 추정이다.
 */
export function adaptSegments(res: OptimizedSegments, fetchedAt: number): SegmentsView {
  const time = { kind: '조회' as const, kst: kstTime(fetchedAt) };
  const kind = smpForecastKind({ model_used: res.model_used });
  const chip = CALC_CHIP[kind];
  const hours = res.hourly_bids.map((bid, i): HourBidView => {
    if (bid.hour < 0 || bid.hour > 23) throw new ApiShapeError('/api/v1/bidding/optimized-segments', `hourly_bids[${i}].hour`, '0–23');
    return {
      tradeHour: bid.hour + 1,
      baselineQ50: bid.smp_forecast.q50,
      revenue: bid.expected_revenue === null ? null : exampleFromCalculator(bid.expected_revenue, '천원', '기대 수입'),
      segments: bid.segments.map(seg => ({
        id: seg.segment_id,
        price: provenanced({ value: seg.price_krw_mwh, unit: '원/kWh', state: 'estimated', chip, source: 'API 서버 입찰 계산', time }),
        probability: provenanced({
          value: seg.clearing_probability * 100, unit: '%', state: 'estimated', chip, source: 'API 서버 입찰 계산', time,
          note: '실제 낙찰 확률 아님',
        }),
        quantity: exampleFromCalculator(seg.quantity_mw, 'MW', '구간 물량'),
        revenue: seg.expected_revenue === null ? null : exampleFromCalculator(seg.expected_revenue, '천원', '기대 수입'),
      })),
    };
  });
  return {
    tradingDate: parseDayKey(res.trading_date), risk: res.risk_level, method: res.optimization_method, modelUsed: res.model_used,
    capacityMw: res.capacity_mw, kind, chip, hours,
  };
}
