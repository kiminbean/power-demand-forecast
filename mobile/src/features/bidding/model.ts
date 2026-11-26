// 입찰 화면 view model(명세 4.2·7.3·7.5·7.6·7.8절). (응답들, 지금 시각) → 화면 글자·구조를 만드는 순수 함수다. React·DOM이 없고
// 컴포넌트는 배치만 한다. tests/bidding-tab.test.cjs가 round 0 스냅숏과 상태 변형(variants.json의 B6)을 넣어 글자·상태를 고정한다.
// 하루전시장 개장·마감은 서버(/market-status)가 기준이고, 앱 시각(기기 시계 + 서버 시계 차이)은 응답 사이의 카운트다운만 움직인다.
// 화면에 보이는 숫자는 모두 strings.ts(format.ts)를 거친 메타 글자이거나 값(Value)·예시(SampleValue) 표지 안에 있다(I2).
import { adaptSegments } from '../../data/adapters';
import type { HourBidView, Provenanced, SegmentsView } from '../../data/adapters';
import { EXAMPLE_MERIT_BLOCKS, EXAMPLE_MERIT_DEMAND_MW, EXAMPLE_PLANT, exampleFromCalculator } from '../../data/examples';
import type { Example } from '../../data/examples';
import type { MarketStatus, OptimizedSegments, SmpForecast } from '../../data/types';
import { HOUR_MS, MINUTE_MS, addDays, kstAt, nextIntervalAt, parseApiTime, parseDayKey } from '../../lib/kst';
import { num } from '../../lib/format';
import { CHIPS, smpForecastKind } from '../../lib/provenance';
import type { Chip, ForecastKind } from '../../lib/provenance';
import { ladderStats } from '../../ui/charts/chartMath';
import type { PillTone } from '../../ui/tokens';
import {
  ANSWER_DEADLINE_CHECKING, ANSWER_DEADLINE_ERROR, ANSWER_DEADLINE_MISMATCH, CALC_BAD_REQUEST_TEXT, CALC_BUSY_TEXT, CALC_EMPTY_TEXT, CLOSED_TODAY_TEXT, NEXT_DEADLINE_LEAD,
  CALC_FAILED_TEXT, CALC_UNREADABLE_TEXT, DEADLINE_FAILED_TEXT, DEADLINE_MISMATCH_TEXT, DEADLINE_PILL, RTM_NO_INTERVAL, SOURCE_ITEMS,
  SOURCE_PENDING, SOURCE_TEXT, answerClosedText, answerOpenText, basisLagText, bidTableCaption, calcFooterSource, calcInputText,
  calcMethodText, calcSourceText, countdownText, deadlineApiText, deadlineCheckText, deadlineWhenText, freshness, ladderChartLabel,
  ladderNotRealText, ladderSentenceText, ladderTrendText, lookupAt, lookupBasisText, meritSummaryText, modelUsedText, nextCountdownText, nextDeadlineText,
  priceBasisText, rtmLineText, rtmStatusText, serverTimeText,
} from '../../content/strings';
import type { Risk, RouteState } from '../../app/routes';
import type { ChipSpec, Fetched, Footer, MethodFacts, SourceRowView } from '../today/model';

/** 남은 시간이 이보다 적으면 «마감 임박»(4.2절). */
export const IMMINENT_MS = 60 * MINUTE_MS;
/** 서버 hours_remaining과 다시 계산한 남은 시간의 허용 차이(시간, 4.2절). 서버는 소수 첫째 자리로 반올림해 보낸다. */
export const HOURS_TOLERANCE = 0.2;
/** 가격 근거 줄(7.8절 2): 계산기 q50(소수 첫째 자리)과 기준선 예측 q50이 이 안이면 같은 시리즈다. */
export const Q50_TOLERANCE = 0.05;
export const DEFAULT_RISK: Risk = 'moderate';
/** ?hour=가 없을 때의 거래시간: 4.2절 와이어프레임의 예시(19h · 18–19시, 저녁). 지금 시각과 무관해 같은 화면이 재현된다. */
export const DEFAULT_BID_HOUR = 19;
const EPS = 1e-9;
const HOURS_IN_DAY = 24;

/** URL ?risk=(없거나 다른 경로면 균형). */
export const riskOf = (s: RouteState): Risk => (s.key === 'bidding' && s.risk) || DEFAULT_RISK;

export type Recheck = 'idle' | 'pending' | 'done';
/** 시각은 모두 앱 시각(KST epoch ms, 기기 시계 + 타당한 서버 시계 차이)이다. 화면(BiddingTab)이 바꿔서 넘긴다. */
export type BiddingInput = {
  now: number;
  market: Fetched<MarketStatus>;
  calc: Fetched<OptimizedSegments> & { busy: boolean };
  /** 오늘 화면이 받아 둔 기준선 예측(엿보기). 입찰 경로는 시장 상태·입찰 계산만 요청한다(6.3절). 없으면 가격 근거 줄을 확인할 수 없다. */
  forecast: SmpForecast | null;
  /** 지금 계산 결과를 받을 때 엿본 예측일(그 뒤 예측일이 바뀌면 오래된 입력). */
  calcForecastDate: string | null;
  risk: Risk | null;
  selectedHour: number | null;
  /** 서버 상태와 계산이 어긋나 시장 상태를 한 번 더 받는 중(pending)이거나 받았다(done). */
  recheck: Recheck;
  /** 오래된 입력이라 계산을 한 번 다시 받았다. */
  staleRetried: boolean;
};

export type DeadlinePhase = 'checking' | 'open' | 'imminent' | 'closed' | 'mismatch' | 'error';
export type Pill = { tone: PillTone; text: string };
export type DeadlineCardView = {
  phase: DeadlinePhase;
  pill: Pill | null;
  /** 제목 옆 상태(마감 뒤 «오늘 마감됨», D13: 마감 pill이 "다음 마감" 줄 밑에 있어 다음 마감이 닫힌 것처럼 읽혔다). */
  status: Pill | null;
  chips: ChipSpec[];
  /** 접수 중·마감 임박: "3시간 56분 남음", 마감 뒤: "15시간 48분"(D13, 서버 시각 기준 앱 시각에서 센다). */
  countdown: string | null;
  /** 큰 숫자 위 줄: 마감 뒤에만 "다음 마감까지". */
  countdownLead: string | null;
  /** "마감 11-26(목) 10:00 KST · 입찰일 11-27(금)" / 마감 뒤 "다음 마감 11-27(금) 10:00 KST · 입찰일 11-28(토)". */
  when: string | null;
  /** 불일치·오류 문장. */
  note: string | null;
  rtm: string | null;
  footer: Footer | null;
};

export type BidRow = { id: number; segment: string; quantity: Example<number>; price: Provenanced<number>; probability: Provenanced<number>; revenue: Example<number> | null };
export type BidCardView =
  | { state: 'loading' }
  | { state: 'error'; text: string; retry: boolean }
  | { state: 'empty'; text: string; footer: Footer }
  | {
      state: 'ready'; kind: ForecastKind; chip: { chip: Chip; source: string }; headChips: ChipSpec[]; hour: HourBidView; selected: number;
      first: number; last: number; tradingDate: string | null; chartLabel: string; summary: string | null; trend: string | null;
      notReal: string; basis: string | null; rows: BidRow[]; total: { quantity: Example<number>; revenue: Example<number> | null };
      tableCaption: string; footer: Footer;
    };

export type BiddingCardId = 'bid-deadline' | 'bid-ladder' | 'bid-merit';
export type BiddingView = {
  answer: string;
  /** NowPill·셸(새로고침 결과·오프라인 띠)의 마지막 조회: 시장 상태(앱 시각). 계산기는 폴링하지 않아 나이로 지연을 말하지 않는다. */
  fetchedAt: number | null;
  deadline: DeadlineCardView;
  /** 서버 상태와 계산이 어긋남(확인 조회 전후 모두 참). 화면이 즉시 한 번 다시 조회한다. */
  mismatch: boolean;
  calc: BidCardView;
  /** 계산이 오래된 입력으로 만들어졌다(입찰일·예측일이 바뀜). 화면이 한 번 다시 계산한다. */
  calcStale: boolean;
  facts: Record<BiddingCardId, MethodFacts>;
  sources: SourceRowView[];
  /** ≥ 1280 "입찰 읽는 법" 패널의 범례 값. */
  legend: { when: string | null; probability: number | null; capacityMw: number };
};

const hasData = <T>(r: Fetched<T>): r is Fetched<T> & { data: T; fetchedAt: number } => r.data !== null && r.fetchedAt !== null;
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const validHour = (h: unknown): h is number => typeof h === 'number' && Number.isInteger(h) && h >= 1 && h <= HOURS_IN_DAY;

// ---------------------------------------------------------------- 하루전 마감(4.2절 상태표)
type MarketFacts = {
  /** 서버가 hours_remaining을 셈한 시각(current_time, 읽지 못하면 받은 시각). */
  ref: number; serverTime: number | null; status: 'open' | 'closed' | 'other'; trading: string | null;
  /** 입찰일 하루 전 HH:MM(이번 마감)과 입찰일 HH:MM(마감 뒤의 다음 마감). */
  deadline: number | null; next: number | null; fetchedAt: number; raw: MarketStatus;
};

function readMarket(r: Fetched<MarketStatus>): MarketFacts | null {
  if (!hasData(r)) return null;
  const serverTime = parseApiTime(r.data.current_time);
  const trading = parseDayKey(r.data.dam.trading_date);
  const s = r.data.dam.status;
  return {
    ref: serverTime ?? r.fetchedAt, serverTime, status: s === 'open' ? 'open' : s === 'closed' ? 'closed' : 'other', trading,
    deadline: trading ? kstAt(trading, r.data.dam.deadline, -1) : null, next: trading ? kstAt(trading, r.data.dam.deadline) : null,
    fetchedAt: r.fetchedAt, raw: r.data,
  };
}

/** 서버 시각에서 다시 계산한 남은 시간(ms): 접수 중이면 이번 마감까지, 마감이면 다음 마감까지. 셀 수 없으면 null. */
function serverLeft(f: MarketFacts): number | null {
  const target = f.status === 'open' ? f.deadline : f.status === 'closed' ? f.next : null;
  return target === null ? null : target - f.ref;
}

/**
 * 불일치(4.2절): 모르는 상태·읽을 수 없는 날짜·시각, 서버는 open인데 계산상 마감이 지남(서버 시각 또는 지금), 서버는 마감인데
 * 이번 마감이 아직 오지 않음, hours_remaining과 다시 계산한 값의 차이가 0.2시간 넘음.
 */
function deadlineMismatch(f: MarketFacts, now: number): boolean {
  const left = serverLeft(f);
  if (left === null || f.deadline === null || left <= 0) return true;
  if (f.status === 'open' && f.deadline <= now) return true;
  if (f.status === 'closed' && f.deadline > f.ref) return true;
  const hours = f.raw.dam.hours_remaining;
  return hours !== null && Math.abs(hours - left / HOUR_MS) > HOURS_TOLERANCE + EPS;
}

function serverPill(status: MarketFacts['status']): Pill {
  if (status === 'open') return { tone: 'ok', text: DEADLINE_PILL.open };
  return { tone: 'neutral', text: status === 'closed' ? DEADLINE_PILL.closed : DEADLINE_PILL.unknown };
}

/** 실시간시장 줄. 다음 구간은 API의 next_interval 문자열이 아니라 KST 지금 뒤 첫 interval분 배수다(4.2절). */
function rtmText(rtm: MarketStatus['rtm'], now: number): string {
  if (rtm.status !== 'active') return rtmStatusText(rtm.status);
  const minutes = rtm.interval_minutes;
  const next = finite(minutes) ? nextIntervalAt(now, minutes) : null;
  return finite(minutes) && next !== null ? rtmLineText(minutes, next) : RTM_NO_INTERVAL;
}

const CHECKING: DeadlineCardView = {
  phase: 'checking', pill: { tone: 'neutral', text: DEADLINE_PILL.checking }, status: null, chips: [], countdown: null, countdownLead: null, when: null,
  note: null, rtm: null,
  footer: null,
};

function deadlineOf(input: BiddingInput, f: MarketFacts | null): { view: DeadlineCardView; answer: string; mismatch: boolean } {
  const { now } = input;
  if (!f) {
    // 첫 응답 전에는 «확인 중»(절대 "마감"이라 쓰지 않는다). 받지 못했으면 오류(카운트다운 없음).
    if (input.market.status !== 'error') return { view: CHECKING, answer: ANSWER_DEADLINE_CHECKING, mismatch: false };
    return { view: { ...CHECKING, phase: 'error', pill: null, note: DEADLINE_FAILED_TEXT }, answer: ANSWER_DEADLINE_ERROR, mismatch: false };
  }
  const stage = freshness(now - f.fetchedAt);
  const lag: ChipSpec[] = stage === 'fresh' ? [] : [{ lag: stage, at: f.fetchedAt, text: basisLagText(f.fetchedAt, stage) }];
  const base = {
    chips: [{ chip: CHIPS.marketRule, source: SOURCE_TEXT.market }, ...lag], rtm: rtmText(f.raw.rtm, now),
    footer: { source: `출처 ${SOURCE_TEXT.market}`, time: serverTimeText(f.serverTime) },
  };
  if (deadlineMismatch(f, now) || f.deadline === null || f.next === null || f.trading === null) {
    // 즉시 한 번 다시 조회하는 동안은 «확인 중», 그래도 다르면 서버 pill + ‹시각 확인 필요›, 카운트다운 숨김.
    if (input.recheck !== 'done') return { view: CHECKING, answer: ANSWER_DEADLINE_CHECKING, mismatch: true };
    const when = f.deadline !== null && f.trading !== null ? deadlineWhenText(f.deadline, f.trading) : null;
    return {
      view: { phase: 'mismatch', pill: serverPill(f.status), status: null, ...base, chips: [...base.chips, { chip: CHIPS.checkTime }], countdown: null,
        countdownLead: null, when,
        note: DEADLINE_MISMATCH_TEXT },
      answer: ANSWER_DEADLINE_MISMATCH, mismatch: true,
    };
  }
  if (f.status === 'closed') {
    // 다음 마감은 입찰일(trading_date) 10:00이고 그 마감의 입찰일은 하루 뒤다(시장 규칙, round 1 integrity: 마감된 입찰일과 짝지어 읽혔다).
    return {
      view: { phase: 'closed', pill: null, status: { tone: 'neutral', text: CLOSED_TODAY_TEXT }, ...base, countdown: nextCountdownText(f.next - now),
        countdownLead: NEXT_DEADLINE_LEAD,
        when: nextDeadlineText(f.next, addDays(f.trading, 1)), note: null },
      answer: answerClosedText(f.next), mismatch: false,
    };
  }
  const left = f.deadline - now;
  const imminent = left <= IMMINENT_MS;
  return {
    view: { phase: imminent ? 'imminent' : 'open', pill: imminent ? { tone: 'watch', text: DEADLINE_PILL.imminent } : serverPill('open'), status: null, ...base,
      countdown: countdownText(left), countdownLead: null, when: deadlineWhenText(f.deadline, f.trading), note: null },
    answer: answerOpenText(f.deadline, left, imminent), mismatch: false,
  };
}

// ---------------------------------------------------------------- 예시 발전소 입찰 계산(4.2·7.8절)
/** 오래된 입력(4.2절): 계산의 입찰일이 시장 상태의 입찰일과 다르거나, 계산을 받은 뒤 기준선 예측일이 바뀌었다. */
export function calcInputStale(input: Pick<BiddingInput, 'market' | 'calc' | 'forecast' | 'calcForecastDate'>): boolean {
  const calc = input.calc.data;
  if (!calc) return false;
  const calcDay = parseDayKey(calc.trading_date);
  const marketDay = input.market.data ? parseDayKey(input.market.data.dam.trading_date) : null;
  if (calcDay && marketDay && calcDay !== marketDay) return true;
  const fd = input.forecast ? parseDayKey(input.forecast.forecast_date) : null;
  return Boolean(input.calcForecastDate && fd && input.calcForecastDate !== fd);
}

function calcError(error: unknown): BidCardView {
  const e = (typeof error === 'object' && error !== null ? error : {}) as { name?: unknown; status?: unknown };
  const status = e.name === 'ApiHttpError' && typeof e.status === 'number' ? e.status : null;
  // 503은 서버가 다른 계산을 하는 중(저장소가 1.5초 뒤 한 번 다시 보낸 뒤에도). 4xx는 다시 보내도 같아 버튼이 없다(M8).
  if (status === 503) return { state: 'error', text: CALC_BUSY_TEXT, retry: true };
  if (status === 400 || status === 422) return { state: 'error', text: CALC_BAD_REQUEST_TEXT, retry: false };
  if (status !== null && status >= 400 && status < 500) return { state: 'error', text: CALC_FAILED_TEXT, retry: false };
  return { state: 'error', text: e.name === 'ApiShapeError' ? CALC_UNREADABLE_TEXT : CALC_FAILED_TEXT, retry: true };
}

/**
 * 가격 근거 공개 줄(7.8절 2): 계산의 입찰일이 기준선 예측일과 다르고, 보이는 시간의 계산기 q50이 같은 예측의 q50과 0.05 안이면
 * "가격 근거: 11-26 기준선(=11-25 확정 가격) · 입찰일 11-27과 하루 차이". 백엔드가 D+1 기준선을 주면(P5) 저절로 사라진다.
 * 예측은 엿본 것뿐이라 오늘 화면을 거치지 않은 방문에서는 확인할 수 없어 쓰지 않는다.
 */
function priceBasis(seg: SegmentsView, hour: HourBidView, forecast: SmpForecast | null): string | null {
  const kind = seg.kind;
  if (!forecast || kind === 'fallback' || smpForecastKind(forecast) !== kind) return null;
  const fd = parseDayKey(forecast.forecast_date);
  if (!fd || !seg.tradingDate || fd === seg.tradingDate) return null;
  const mine = hour.baselineQ50;
  const theirs = forecast.q50[hour.tradeHour - 1];
  if (!finite(mine) || !finite(theirs) || Math.abs(mine - theirs) > Q50_TOLERANCE + EPS) return null;
  return priceBasisText(fd, seg.tradingDate, kind);
}

function calcCard(input: BiddingInput, stale: boolean): { view: BidCardView; seg: SegmentsView | null; hour: HourBidView | null } {
  const r = input.calc;
  if (!hasData(r)) return { view: r.status === 'error' ? calcError(r.error) : { state: 'loading' }, seg: null, hour: null };
  const seg = adaptSegments(r.data, r.fetchedAt);
  // 오래된 입력: 한 번 다시 계산하는 동안은 스켈레톤(4.2절 "스켈레톤 유지"). 다시 받아도 다르면 값과 ‹시각 확인 필요›.
  if (stale && (!input.staleRetried || r.busy)) return { view: { state: 'loading' }, seg, hour: null };
  const footer: Footer = { source: calcFooterSource(seg.kind), time: lookupAt(r.fetchedAt) };
  const usable = seg.hours.filter(h => h.segments.length > 0);
  if (!usable.length) return { view: { state: 'empty', text: CALC_EMPTY_TEXT, footer }, seg, hour: null };
  const want = validHour(input.selectedHour) ? input.selectedHour : DEFAULT_BID_HOUR;
  const hour = usable.find(h => h.tradeHour === want) ?? usable.find(h => h.tradeHour === DEFAULT_BID_HOUR) ?? usable[0];
  const items = hour.segments.map(s => ({ price: s.price.value, probability: s.probability.value, quantity: s.quantity.value }));
  const stats = ladderStats(items);
  const totalMw = items.reduce((a, it) => a + (it.quantity > 0 ? it.quantity : 0), 0);
  const byPrice = items.slice().sort((a, b) => a.price - b.price);
  const falling = byPrice.every((it, i) => i === 0 || it.probability <= byPrice[i - 1].probability);
  const source = calcSourceText(seg.kind);
  return {
    seg, hour,
    view: {
      state: 'ready', kind: seg.kind, chip: { chip: seg.chip, source }, headChips: stale ? [{ chip: CHIPS.checkTime }] : [], hour,
      selected: hour.tradeHour, first: usable[0].tradeHour, last: usable[usable.length - 1].tradeHour, tradingDate: seg.tradingDate,
      chartLabel: ladderChartLabel(hour.tradeHour),
      summary: stats ? ladderSentenceText(totalMw, stats.count, stats.minPrice, stats.maxPrice) : null,
      // 가격이 낮을수록 낙찰 가능성이 높을 때만 그렇게 말한다(계산기가 그렇게 만들지만 응답으로 확인한다).
      trend: stats && falling && stats.probAtMin > stats.probAtMax ? ladderTrendText(stats.probAtMin, stats.probAtMax) : null,
      notReal: ladderNotRealText(seg.kind), basis: priceBasis(seg, hour, input.forecast),
      rows: hour.segments.map(s => ({ id: s.id, segment: num(s.id, 0), quantity: s.quantity, price: s.price, probability: s.probability, revenue: s.revenue })),
      total: { quantity: exampleFromCalculator(totalMw, 'MW', '합계 물량'), revenue: hour.revenue },
      tableCaption: bidTableCaption(hour.tradeHour), footer,
    },
  };
}

// ---------------------------------------------------------------- 시트(산출 방식의 이번 값 · 출처 표의 입찰 계산 행)
function factsOf(f: MarketFacts | null, seg: SegmentsView | null): Record<BiddingCardId, MethodFacts> {
  const deadline: string[] = [];
  if (f) {
    const d = f.raw.dam;
    deadline.push(deadlineApiText(d.status, d.deadline, d.trading_date, d.hours_remaining));
    const left = serverLeft(f);
    if (left !== null) deadline.push(deadlineCheckText(left / HOUR_MS, HOURS_TOLERANCE));
  }
  const ladder = seg ? [calcMethodText(seg.method), modelUsedText(seg.modelUsed ?? '표기 없음'), calcInputText(seg.tradingDate, seg.risk, seg.capacityMw)] : [];
  return { 'bid-deadline': { lines: deadline, link: null }, 'bid-ladder': { lines: ladder, link: null }, 'bid-merit': { lines: [], link: null } };
}

function sourceRow(r: Fetched<OptimizedSegments>, seg: SegmentsView | null): SourceRowView {
  const item = SOURCE_ITEMS.bidding;
  if (seg && r.fetchedAt !== null) {
    const source = calcSourceText(seg.kind);
    return { id: 'bidding', item, source, basis: lookupBasisText(r.fetchedAt), chip: { chip: CHIPS.estShort, source }, status: null, link: null };
  }
  const basis = r.status === 'error' ? SOURCE_PENDING.error : r.status === 'idle' ? SOURCE_PENDING.idle : SOURCE_PENDING.loading;
  return { id: 'bidding', item, source: calcSourceText('baseline'), basis, chip: null, status: null, link: null };
}

// ---------------------------------------------------------------- 가격 원리 예시(4.2절 MeritOrderCard)
/** 예시 수요를 채우는 마지막 발전기(싼 것부터 쌓아 처음으로 수요에 닿는 블록)와 그 가격. */
export function meritExample() {
  const demand = EXAMPLE_MERIT_DEMAND_MW.value;
  let cum = 0;
  const marginal = EXAMPLE_MERIT_BLOCKS.findIndex(b => {
    cum += b.capacityMw.value;
    return cum >= demand;
  });
  const block = marginal >= 0 ? EXAMPLE_MERIT_BLOCKS[marginal] : null;
  return { marginal, summary: block ? meritSummaryText(demand, block.name.value, block.bidPrice.value) : null };
}

// ---------------------------------------------------------------- 묶기
export function biddingView(input: BiddingInput): BiddingView {
  const f = readMarket(input.market);
  const d = deadlineOf(input, f);
  const stale = calcInputStale(input);
  const c = calcCard(input, stale);
  const first = c.hour ? c.hour.segments[0] : undefined;
  return {
    answer: d.answer, fetchedAt: f ? f.fetchedAt : null, deadline: d.view, mismatch: d.mismatch, calc: c.view, calcStale: stale,
    facts: factsOf(f, c.seg), sources: [sourceRow(input.calc, c.seg)],
    legend: { when: d.view.when, probability: first ? first.probability.value : null, capacityMw: EXAMPLE_PLANT.capacityMw.value },
  };
}
