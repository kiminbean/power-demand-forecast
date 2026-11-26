// 오늘 화면 view model(명세 4.1·6.3·7.4·7.6·7.8절). (응답들, 지금 시각) → 화면 글자·구조를 만드는 순수 함수다. React·DOM이 없고
// 컴포넌트는 배치만 한다. tests/view-models.test.cjs가 round 0 스냅숏과 상태 변형 픽스처를 재생해 이 함수의 글자를 고정한다.
// 글자 조각(Seg) 규칙(I2): 평문 조각 t에는 숫자를 두지 않는다. 숫자는 API 값이면 v(data-pd-value), 계산값·시각·라벨이면 m(data-pd-meta).
import { adaptDayAhead, adaptForecast, adaptKpis, adaptSupply, forecastPoint } from '../../data/adapters';
import type { ForecastPoint, ForecastView, KpiView, Provenanced, SupplyView } from '../../data/adapters';
import type { DashboardKpis, DayAheadSmp, MarketStatus, ModelInfo, PowerSupply, SmpForecast } from '../../data/types';
import type { ResourceStatus } from '../../lib/resource';
import { HOUR_MS, MINUTE_MS, kstDateKey, kstParts, parseApiTime, tradeHourAt } from '../../lib/kst';
import { mw, num, price, tradeHour } from '../../lib/format';
import { CHIPS, DOCUMENTED_SMP_ACCURACY, forecastChip, weatherProvenance } from '../../lib/provenance';
import type { Chip, ForecastKind, ReserveTone } from '../../lib/provenance';
import { forecastStats, periodStats, widestBand } from '../../ui/charts/chartMath';
import {
  ACCURACY_SOURCE, ACCURACY_UNREGISTERED, ANSWER_ALL_FAILED, ANSWER_LOADING, BAND_LABEL, BOUNDARY_CHECKING_TEXT, CONFIRMED_LABEL,
  FAIL_CAUSE, FORECAST_DATE_LAG, FORECAST_EMPTY, FORECAST_SUBJECT_SHORT, KPI_LABELS, KPX_LINK_LABEL, LEVEL_NOTE, MODEL_INFO_FAILED,
  NO_JUDGE_BROKEN, NO_JUDGE_EMPTY, NO_JUDGE_UNKNOWN_SOURCE, NO_PRICE_TEXT, SOURCE_ITEMS, SOURCE_PENDING, SOURCE_TEXT, ZERO_INPUT_LABEL,
  accuracyBasisPeriod, accuracyBasisText, accuracyExampleText, accuracyText, allFailedText, answerBrokenText, answerStaleText,
  apiAccuracyText, compareDeltaParts, compareRatioText, dayAheadBasisText, forecastBasisText, forecastLegendText, forecastSourceLine,
  forecastSummaryText, forecastTimeLine, freshness, hourStaleChipText, hourStepLabel, kpxOriginalText, loadFailedText, lookupAt,
  lookupBasisText, modelInfoFactText, modelUsedText, observedBasisText, observedLookup, periodLabelText, periodNoteText, priceHourLine,
  reserveCalcText, serverBasisText, supplyMissingText, todayAnswerText, uncertaintyText, weatherSourceText,
} from '../../content/strings';
import type { AnswerPower, AnswerPrice, Freshness } from '../../content/strings';
import type { SourceRow } from '../../app/routes';
import type { Line, Seg } from '../../ui/Rich';

export type { Line, Seg };

export type Fetched<T> = { data: T | null; fetchedAt: number | null; status: ResourceStatus; error: unknown };
/** 시각은 모두 앱 시각(기기 시계 + 타당한 서버 시계 차이, KST epoch ms)이다. 화면(SMPForecastScreen)이 바꿔서 넘긴다. */
export type TodayInput = {
  now: number;
  dayAhead: Fetched<DayAheadSmp>;
  /** isAvailableDayAhead(dayAheadSmp)와 describeDayAhead(dayAheadSmp)(utils/smpPresentation) 결과. 화면이 계산해 넘긴다. */
  priceAvailable: boolean;
  kpxText: string;
  forecast: Fetched<SmpForecast>;
  modelInfo: Fetched<ModelInfo>;
  kpis: Fetched<DashboardKpis>;
  powerSupply: Fetched<PowerSupply>;
  market: Fetched<MarketStatus>;
  /** URL ?hour=. 없거나 틀리면 지금 거래시간. */
  selectedHour: number | null;
};

export type ChipSpec = { chip: Chip; source?: string | null } | { lag: 'stale' | 'broken'; at: number; text?: string };
export type Footer = { source: string; time: string; extra?: string };

export type PriceCardView =
  | { state: 'loading' }
  | { state: 'error'; text: string }
  | { state: 'unavailable'; text: string; footer: Footer }
  | { state: 'ready'; price: Provenanced<number>; chips: ChipSpec[]; hourLine: string; checking: string | null; compare: Line[]; footer: Footer };

export type KpiItem = { label: string; value: Provenanced<number> | null; chip: ChipSpec | null; note: string | null };
export type SupplyPart =
  | { kind: 'paired'; supply: Provenanced<number>; demand: Provenanced<number>; reserve: Provenanced<number>; boundary: Provenanced<number>; formula: Line }
  | { kind: 'missing'; text: string };
export type PowerCardView =
  | { state: 'loading' }
  | { state: 'error'; text: string }
  | {
      state: 'ready'; chips: ChipSpec[]; demand: KpiItem; reserve: KpiItem;
      level: { tone: ReserveTone; text: string; note: string } | null; judgeNote: string | null;
      supply: SupplyPart; weather: { line: Line; chip: ChipSpec } | null; footer: Footer;
    };

export type HourRow = { hour: number; label: string; point: ForecastPoint; confirmed: Provenanced<number> | null };
export type PeriodRow = { id: string; label: string; mean: string; min: string; max: string };
export type ForecastCardView =
  | { state: 'loading' }
  | { state: 'error'; text: string }
  | { state: 'empty'; text: string; chips: ChipSpec[]; footer: Footer }
  | {
      state: 'ready'; view: ForecastView; kind: ForecastKind; chips: ChipSpec[];
      confirmed: { tradeHour: number; price: Provenanced<number> } | null; legend: string;
      selected: number; nowHour: number; isNow: boolean; stepLabel: (hour: number) => string;
      reading: Line; summary: string; uncertainty: string | null; periods: PeriodRow[]; periodNote: string;
      rows: HourRow[]; window: { from: number; to: number }; footer: Footer;
    };

export type AccuracyCardView =
  | { state: 'loading' }
  | { state: 'error'; text: string }
  | {
      state: 'ready'; documented: { main: string; basis: string; example: string; r2?: string } | null; unregistered: string | null;
      api: string | null; footer: Footer;
    };

export type MethodFacts = { lines: string[]; link: { label: string; url: string } | null };
export type TodayCardId = 'today-price' | 'today-power' | 'today-forecast' | 'today-accuracy';
export type SourceRowView = { id: SourceRow; item: string; source: string; basis: string; chip: ChipSpec | null; status: string | null; link: string | null };

export type TodayView = {
  heroState: 'loading' | 'ready' | 'unavailable' | 'error';
  answer: string;
  /** NowPill·셸(오프라인 띠·새로고침 결과)의 마지막 조회: 가격과 전력 중 더 오래된 성공 조회(앱 시각). */
  fetchedAt: number | null;
  stage: Freshness;
  /** 세 요청(가격·예측·전력)이 모두 실패: StateView 글자. 값이 하나도 없으면 카드 대신 이것만 둔다. */
  allFailed: { text: string; showCards: boolean } | null;
  price: PriceCardView;
  power: PowerCardView;
  forecast: ForecastCardView;
  accuracy: AccuracyCardView;
  facts: Record<TodayCardId, MethodFacts>;
  sources: SourceRowView[];
  /** ≥ 1280 "이 화면 읽는 법" 패널의 4상태 범례에 쓰는 이 화면의 값. */
  legend: { confirmed: number | null; forecast: number | null };
  /** 거래시간이 어긋났고 가격을 이번 정시(+15초) 전에 받았다: 한 번 다시 받는다(다른 탭에서 정시를 넘겨 돌아온 경우). */
  recheckPrice: boolean;
};

/** 정시 뒤 이 시간 안에 거래시간이 어긋나면 이전 값을 유지하고 "다음 거래시간 값 확인 중"(4.1절 거래시간 경계). */
export const HOUR_GRACE_MS = 5 * MINUTE_MS;
/** 정시 뒤 다시 조회하는 두 시점(4.1절): HH:00:15와 HH:02:00. */
export const BOUNDARY_RECHECK_MS = [15_000, 2 * MINUTE_MS] as const;
/** 12행 표(4.1절): 선택한 시간부터, 하루 끝을 넘으면 마지막 12행. */
export const TABLE_ROWS = 12;
const HOURS_IN_DAY = 24;
const PCT_DIFF_SHOWN = 0.1;
const one = (n: number) => num(n, 1);

const hasData = <T>(r: Fetched<T>): r is Fetched<T> & { data: T; fetchedAt: number } => r.data !== null && r.fetchedAt !== null;
const failed = (r: Fetched<unknown>) => r.data === null && r.status === 'error';
const validHour = (h: unknown): h is number => typeof h === 'number' && Number.isInteger(h) && h >= 1 && h <= HOURS_IN_DAY;

/** KST 정시부터 지난 시간(ms). */
export function sinceHourStart(now: number): number {
  const p = kstParts(now);
  return p ? p.minute * MINUTE_MS + p.second * 1000 : 0;
}

/** 이번 KST 정시(앱 시각). */
export const hourStartOf = (now: number) => now - sinceHourStart(now) - (((now % 1000) + 1000) % 1000);

/** 다음 재조회 시각(앱 시각): 이번 시의 HH:00:15·HH:02:00 중 아직 안 온 것, 없으면 다음 시의 HH:00:15. */
export function nextBoundaryCheck(now: number): number {
  const start = hourStartOf(now);
  const upcoming = BOUNDARY_RECHECK_MS.map(ms => start + ms).find(t => t > now);
  return upcoming ?? start + HOUR_MS + BOUNDARY_RECHECK_MS[0];
}

function failCause(errors: unknown[]): string {
  const names = errors.map(e => (typeof e === 'object' && e !== null ? (e as { name?: unknown }).name : null));
  if (names.length && names.every(n => n === 'ApiNetworkError')) return FAIL_CAUSE.network;
  if (names.some(n => n === 'ApiShapeError')) return FAIL_CAUSE.shape;
  return FAIL_CAUSE.server;
}

const lagChip = (stage: Freshness, at: number): ChipSpec | null => (stage === 'fresh' ? null : { lag: stage, at });

type Ctx = { input: TodayInput; now: number; today: string | null; nowHour: number };

// ---------------------------------------------------------------- 가격
type PriceFacts = { price: Provenanced<number>; tradeHour: number; tradeDay: string; sourceUrl: string | null; mismatch: boolean; grace: boolean };

function readPrice(c: Ctx): PriceFacts | null {
  const r = c.input.dayAhead;
  if (!hasData(r) || !c.input.priceAvailable) return null;
  const view = adaptDayAhead(r.data, r.fetchedAt);
  if (view.status !== 'ready') return null;
  const mismatch = view.tradeDay !== c.today || view.tradeHour !== c.nowHour;
  // 유예는 "API가 아직 다음 거래시간으로 넘어가지 않은" 경우만: 값이 바로 앞 거래시간(자정이면 어제 24h)이고 정시 뒤 5분 안.
  const before = hourStartOf(c.now) - 1;
  const previous = view.tradeDay === kstDateKey(before) && view.tradeHour === tradeHourAt(before);
  return { price: view.price, tradeHour: view.tradeHour, tradeDay: view.tradeDay, sourceUrl: view.sourceUrl, mismatch,
    grace: mismatch && previous && sinceHourStart(c.now) < HOUR_GRACE_MS };
}

type Comparison = { q50: Provenanced<number>; diff: number; kind: ForecastKind };
/** 7.8절 1: 같은 날짜·유한값·0.00 입력 아님. 예측일이 KST 오늘이 아니거나(예측일 지연) 대체값이면 비교하지 않는다. */
function readComparison(c: Ctx, p: PriceFacts | null): Comparison | null {
  const fc = readForecast(c);
  if (!p || !fc || fc.kind === 'fallback' || fc.forecastDate !== p.tradeDay || fc.forecastDate !== c.today) return null;
  const point = forecastPoint(fc, p.tradeHour);
  return point.kind === 'value' && point.q50 ? { q50: point.q50, diff: point.q50.value - p.price.value, kind: fc.kind } : null;
}

function readForecast(c: Ctx): ForecastView | null {
  const r = c.input.forecast;
  return r.data === null ? null : adaptForecast(r.data);
}

function priceCard(c: Ctx, p: PriceFacts | null, cmp: Comparison | null): PriceCardView {
  const r = c.input.dayAhead;
  if (!hasData(r)) return failed(r) ? { state: 'error', text: loadFailedText('가격', c.now) } : { state: 'loading' };
  const footer: Footer = { source: `출처 ${r.data.data_source ?? 'API 서버 하루전 가격'}`, time: lookupAt(r.fetchedAt) };
  if (!p) return { state: 'unavailable', text: NO_PRICE_TEXT, footer };
  const stage = freshness(c.now - r.fetchedAt);
  const chip: ChipSpec = stage === 'broken' ? { lag: 'broken', at: r.fetchedAt }
    : p.mismatch && !p.grace ? { lag: 'stale', at: r.fetchedAt, text: hourStaleChipText(p.tradeHour) }
      : { chip: p.price.chip, source: p.price.source };
  let compare: Line[] = [];
  if (cmp) {
    const d = compareDeltaParts(cmp.diff, cmp.kind === 'baseline' ? DOCUMENTED_SMP_ACCURACY.maeKrwPerKwh : null);
    compare = [
      [{ t: `같은 시간 ${FORECAST_SUBJECT_SHORT[cmp.kind]}은 ` }, { v: cmp.q50, f: price }, { t: '원이었습니다.' }],
      d.value === null ? [{ m: d.lead, tag: 'compare' }]
        : [{ t: d.lead }, { m: d.value, tag: 'derived', strong: true }, { m: d.tail, tag: 'compare' }],
    ];
  }
  // 끊김(60분)이면 "확인 중"을 쓰지 않는다: AnswerLine과 칩이 이미 "그때 값"이라고 말한다.
  return { state: 'ready', price: p.price, chips: [chip], hourLine: priceHourLine(p.tradeHour),
    checking: p.grace && stage !== 'broken' ? BOUNDARY_CHECKING_TEXT : null, compare, footer };
}

// ---------------------------------------------------------------- 전력
type PowerFacts = { kpi: KpiView; supply: SupplyView; stage: Freshness; fetchedAt: number };

function readPower(c: Ctx): PowerFacts | null {
  const k = c.input.kpis;
  if (!hasData(k)) return null;
  const ps = c.input.powerSupply;
  const supply: SupplyView = ps.data !== null ? adaptSupply(k.data, ps.data, k.fetchedAt) : { status: 'missing', text: '공급능력 미수신' };
  return { kpi: adaptKpis(k.data, k.fetchedAt), supply, stage: freshness(c.now - k.fetchedAt), fetchedAt: k.fetchedAt };
}

function powerCard(c: Ctx, w: PowerFacts | null): PowerCardView {
  const k = c.input.kpis;
  if (!w) return failed(k) ? { state: 'error', text: loadFailedText('전력', c.now) } : { state: 'loading' };
  const { kpi, supply, stage, fetchedAt } = w;
  const prov = kpi.provenance;
  const allMeasured = prov.case === 3;
  // 보이는 값이 하나도 없으면(수요·예비율 모두 null) 머리 칩을 두지 않는다: 잰 값이 없는데 "● 실측"이라 쓰지 않는다.
  const anyValue = kpi.demand !== null || kpi.reserveRate !== null;
  const lag = anyValue ? lagChip(stage, fetchedAt) : null;
  const headChip: ChipSpec[] = !anyValue ? [] : allMeasured ? (lag ? [] : [{ chip: CHIPS.measured, source: prov.source }]) : [{ chip: CHIPS.estBasis }];
  const chips: ChipSpec[] = [...(lag ? [lag] : []), ...headChip];
  const item = (label: string, value: Provenanced<number> | null, chip: Chip, note: string | null): KpiItem =>
    ({ label, value, chip: allMeasured ? null : { chip, source: prov.source }, note });
  const judged = Boolean(kpi.level) && stage !== 'broken';
  const level = judged && kpi.level ? { tone: kpi.level.tone, text: kpi.level.label, note: LEVEL_NOTE } : null;
  const judgeNote = level ? null
    : stage === 'broken' && kpi.level ? NO_JUDGE_BROKEN
      : prov.why ?? (allMeasured ? NO_JUDGE_EMPTY : NO_JUDGE_UNKNOWN_SOURCE);
  const part: SupplyPart = supply.status === 'paired' && kpi.demand
    ? {
        kind: 'paired', supply: supply.supply, demand: kpi.demand, reserve: supply.reserve, boundary: supply.watchBoundary,
        formula: [{ t: '공급능력 ' }, { v: supply.supply, f: mw, unit: true }, { t: ' − 수요 ' }, { v: kpi.demand, f: mw, unit: true },
          { t: ' = 예비력 ' }, { v: supply.reserve, f: mw, unit: true, derived: true }],
      }
    : { kind: 'missing', text: supplyMissingText(allMeasured && kpi.reserveRate !== null) };
  const wp = weatherProvenance(k.data);
  const wx = kpi.weather;
  let weather: { line: Line; chip: ChipSpec } | null = null;
  if (wx && wp.chip) {
    const valueOrDash = (v: Provenanced<number> | null): Seg => (v ? { v, f: one, unit: true } : { t: '—' });
    weather = {
      line: [{ t: '날씨 ' }, valueOrDash(wx.temperature), { t: ' · 바람 ' }, valueOrDash(wx.windSpeed), { t: ' · ' },
        { m: wx.condition ? wx.condition.value : '—', tag: 'condition' }],
      chip: lag ?? { chip: wp.chip, source: wp.source },
    };
  }
  const reserveNote = prov.reserveNote ? prov.reserveNote.replace(/^ · /, '') : null;
  return {
    state: 'ready', chips,
    demand: item(KPI_LABELS.demand, kpi.demand, prov.demandChip, null),
    reserve: item(KPI_LABELS.reserve, kpi.reserveRate, prov.reserveChip, reserveNote),
    level, judgeNote, supply: part, weather,
    footer: { source: `출처 ${prov.source}`, time: observedLookup(fetchedAt), extra: weather && wp.source ? weatherSourceText(wp.source) : undefined },
  };
}

// ---------------------------------------------------------------- 예측
function readingLine(fc: ForecastView, point: ForecastPoint, confirmed: { tradeHour: number; price: Provenanced<number> } | null): Line {
  const subject = FORECAST_SUBJECT_SHORT[fc.kind];
  const segs: Seg[] = [];
  if (point.kind === 'zero') segs.push({ m: ZERO_INPUT_LABEL, tag: 'zero' });
  else if (point.kind === 'missing' || !point.q50) segs.push({ t: `${subject} —` });
  else {
    segs.push({ t: `${subject} ` }, { v: point.q50, f: price }, { t: '원' });
    if (point.band) segs.push({ t: ' · ' }, { m: BAND_LABEL, tag: 'label' }, { t: ' ' }, { v: point.band.q10, f: price }, { t: '–' }, { v: point.band.q90, f: price }, { t: '원' });
  }
  if (confirmed && confirmed.tradeHour === point.tradeHour) segs.push({ t: ` · ${CONFIRMED_LABEL} ` }, { v: confirmed.price, f: price }, { t: '원' });
  return segs;
}

function forecastCard(c: Ctx, p: PriceFacts | null, cmp: Comparison | null): ForecastCardView {
  const r = c.input.forecast;
  if (!hasData(r)) return failed(r) ? { state: 'error', text: loadFailedText('예측', c.now) } : { state: 'loading' };
  const fc = adaptForecast(r.data);
  const dateLag = fc.forecastDate !== c.today;
  const footer: Footer = { source: forecastSourceLine(fc.kind, fc.shortName), time: forecastTimeLine(fc.forecastDate, fc.createdAt) };
  const kindChip: ChipSpec = { chip: forecastChip(fc.kind), source: fc.q50.source };
  const lagChips: ChipSpec[] = dateLag ? [{ lag: 'stale', at: r.fetchedAt, text: FORECAST_DATE_LAG }] : [];
  const stats = forecastStats(fc.q50.value);
  if (!stats) return { state: 'empty', text: FORECAST_EMPTY, chips: [kindChip, ...lagChips], footer };
  const confirmed = p && cmp ? { tradeHour: p.tradeHour, price: p.price } : null;
  const chips: ChipSpec[] = [kindChip, ...(confirmed ? [{ chip: CHIPS.confirmedPoint, source: p ? p.price.source : null }] : []), ...lagChips];
  const selected = validHour(c.input.selectedHour) ? c.input.selectedHour : c.nowHour;
  const widest = fc.band ? widestBand(fc.q50.value, fc.band.q10, fc.band.q90) : null;
  const periods = periodStats(fc.q50.value);
  const rows: HourRow[] = Array.from({ length: HOURS_IN_DAY }, (_, i) => {
    const hour = i + 1;
    return { hour, label: tradeHour(hour), point: forecastPoint(fc, hour), confirmed: confirmed && confirmed.tradeHour === hour ? confirmed.price : null };
  });
  const from = Math.min(selected, HOURS_IN_DAY - TABLE_ROWS + 1);
  return {
    state: 'ready', view: fc, kind: fc.kind, chips, confirmed, legend: forecastLegendText(fc.kind),
    selected, nowHour: c.nowHour, isNow: selected === c.nowHour, stepLabel: hour => hourStepLabel(hour, hour === c.nowHour),
    reading: readingLine(fc, forecastPoint(fc, selected), confirmed),
    summary: forecastSummaryText(stats, fc.kind), uncertainty: widest ? uncertaintyText(widest) : null,
    periods: periods.map(s => ({ id: s.id, label: periodLabelText(s.label, s.from, s.to), mean: price(s.mean), min: price(s.min), max: price(s.max) })),
    periodNote: periodNoteText(stats.zeroHours), rows, window: { from, to: from + TABLE_ROWS - 1 }, footer,
  };
}

function accuracyCard(c: Ctx): AccuracyCardView {
  const r = c.input.forecast;
  if (r.data === null) return failed(r) ? { state: 'error', text: loadFailedText('예측', c.now) } : { state: 'loading' };
  const kind = adaptForecast(r.data).kind;
  const m = c.input.modelInfo;
  const api = m.data !== null ? apiAccuracyText(m.data.mape, m.data.coverage) : failed(m) ? MODEL_INFO_FAILED : null;
  const at = m.fetchedAt ?? r.fetchedAt;
  return {
    state: 'ready',
    documented: kind === 'baseline' ? { main: accuracyText(), basis: accuracyBasisText(), example: accuracyExampleText() } : null,
    unregistered: kind === 'baseline' ? null : ACCURACY_UNREGISTERED, api,
    footer: { source: ACCURACY_SOURCE, time: at === null ? '조회 전' : lookupAt(at) },
  };
}

// ---------------------------------------------------------------- AnswerLine(4.1절 표)
function answerPower(c: Ctx, w: PowerFacts | null): AnswerPower {
  if (!w) return failed(c.input.kpis) ? { kind: 'error' } : { kind: 'loading' };
  const { kpi, supply } = w;
  if (kpi.provenance.case !== 3) return { kind: 'nojudge' };
  if (!kpi.level || !kpi.reserveRate) return { kind: 'empty' };
  if (supply.status === 'paired') return { kind: 'paired', reserveMw: supply.reserve.value };
  return { kind: 'rate', rate: kpi.reserveRate.value, verdict: kpi.level.verdict };
}

function answerPrice(c: Ctx, p: PriceFacts | null): AnswerPrice {
  const r = c.input.dayAhead;
  if (r.data === null) return failed(r) ? { kind: 'error' } : { kind: 'loading' };
  if (!p) return { kind: 'none' };
  return { kind: p.mismatch ? 'hour' : 'now', value: p.price.value, hour: p.tradeHour };
}

// ---------------------------------------------------------------- 시트(산출 방식 살아 있는 값 · 출처 표)
function methodFacts(c: Ctx, p: PriceFacts | null, cmp: Comparison | null, w: PowerFacts | null): Record<TodayCardId, MethodFacts> {
  const pricesLines = [kpxOriginalText(c.input.kpxText), ...(p && cmp ? [compareRatioText(cmp.diff, p.price.value)] : [])];
  const powerLines: string[] = [];
  if (w && w.supply.status === 'paired' && w.kpi.reserveRate && w.kpi.demand && w.kpi.demand.value > 0) {
    const calc = (w.supply.reserve.value / w.kpi.demand.value) * 100;
    if (Math.abs(calc - w.kpi.reserveRate.value) >= PCT_DIFF_SHOWN) powerLines.push(reserveCalcText(calc, w.kpi.reserveRate.value));
  }
  const fc = readForecast(c);
  const m = c.input.modelInfo.data;
  return {
    'today-price': { lines: pricesLines, link: p && p.sourceUrl ? { label: KPX_LINK_LABEL, url: p.sourceUrl } : null },
    'today-power': { lines: powerLines, link: null },
    'today-forecast': { lines: fc ? [modelUsedText(fc.modelUsed)] : [], link: null },
    'today-accuracy': { lines: m ? [modelInfoFactText(m.type, m.version)] : [], link: null },
  };
}

/** 아직 값이 없는 행. 시트는 저장소를 엿보기만 하므로 오늘 화면을 열지 않은 방문(딥링크)에서는 요청 자체가 없다(idle). */
function pendingRow(id: SourceRow, source: string, r: Fetched<unknown>): SourceRowView {
  const basis = failed(r) ? SOURCE_PENDING.error : r.status === 'idle' ? SOURCE_PENDING.idle : SOURCE_PENDING.loading;
  return { id, item: SOURCE_ITEMS[id as keyof typeof SOURCE_ITEMS], source, basis, chip: null, status: null, link: null };
}

function sourceRows(c: Ctx, p: PriceFacts | null, w: PowerFacts | null): SourceRowView[] {
  const { dayAhead, forecast, kpis, market } = c.input;
  const rows: SourceRowView[] = [];
  if (hasData(dayAhead) && p) {
    rows.push({ id: 'day-ahead', item: SOURCE_ITEMS['day-ahead'], source: p.price.source, basis: dayAheadBasisText(p.tradeDay, p.tradeHour, dayAhead.fetchedAt),
      chip: { chip: p.price.chip, source: p.price.source }, status: null, link: p.sourceUrl });
  } else if (hasData(dayAhead)) {
    rows.push({ id: 'day-ahead', item: SOURCE_ITEMS['day-ahead'], source: dayAhead.data.data_source ?? 'API 서버 하루전 가격',
      basis: lookupBasisText(dayAhead.fetchedAt), chip: null, status: NO_PRICE_TEXT, link: null });
  } else rows.push(pendingRow('day-ahead', 'KPX 제주 하루전시장', dayAhead));
  const fc = readForecast(c);
  if (fc) {
    rows.push({ id: 'forecast', item: SOURCE_ITEMS.forecast, source: fc.q50.source, basis: forecastBasisText(fc.forecastDate, fc.createdAt),
      chip: { chip: forecastChip(fc.kind), source: fc.q50.source }, status: null, link: null });
  } else rows.push(pendingRow('forecast', 'API 서버 기준선 예측', forecast));
  rows.push({ id: 'accuracy', item: SOURCE_ITEMS.accuracy, source: SOURCE_TEXT.accuracy, basis: accuracyBasisPeriod(),
    chip: { chip: CHIPS.verified }, status: null, link: null });
  if (w && hasData(kpis)) {
    const prov = w.kpi.provenance;
    rows.push({ id: 'demand', item: SOURCE_ITEMS.demand, source: prov.source, basis: observedBasisText(w.fetchedAt),
      chip: { chip: prov.demandChip, source: prov.source }, status: null, link: null });
    rows.push(w.supply.status === 'paired'
      ? { id: 'supply', item: SOURCE_ITEMS.supply, source: SOURCE_TEXT.supply, basis: lookupBasisText(w.fetchedAt), chip: { chip: CHIPS.measured }, status: null, link: null }
      : { id: 'supply', item: SOURCE_ITEMS.supply, source: SOURCE_TEXT.supply, basis: lookupBasisText(w.fetchedAt), chip: null, status: SOURCE_PENDING.missing, link: null });
    const wp = weatherProvenance(kpis.data);
    rows.push(wp.chip && wp.source && w.kpi.weather
      ? { id: 'weather', item: SOURCE_ITEMS.weather, source: wp.source, basis: observedBasisText(w.fetchedAt), chip: { chip: wp.chip, source: wp.source }, status: null, link: null }
      : { id: 'weather', item: SOURCE_ITEMS.weather, source: '기상청 초단기실황 (API 서버)', basis: lookupBasisText(w.fetchedAt), chip: null, status: SOURCE_PENDING.none, link: null });
  } else {
    rows.push(pendingRow('demand', 'KPX 제주 실시간 (API 서버)', kpis), pendingRow('supply', SOURCE_TEXT.supply, kpis),
      pendingRow('weather', '기상청 초단기실황 (API 서버)', kpis));
  }
  if (hasData(market)) {
    rows.push({ id: 'market', item: SOURCE_ITEMS.market, source: SOURCE_TEXT.market, basis: serverBasisText(parseApiTime(market.data.current_time)),
      chip: { chip: CHIPS.marketRule }, status: null, link: null });
  } else rows.push(pendingRow('market', SOURCE_TEXT.market, market));
  return rows;
}

// ---------------------------------------------------------------- 묶기
export function todayView(input: TodayInput): TodayView {
  const now = input.now;
  const c: Ctx = { input, now, today: kstDateKey(now), nowHour: tradeHourAt(now) ?? 1 };
  const p = readPrice(c);
  const cmp = readComparison(c, p);
  const w = readPower(c);
  const live: Fetched<unknown>[] = [input.dayAhead, input.kpis];
  const fresh = live.filter(hasData).map(r => r.fetchedAt);
  const fetchedAt = fresh.length ? Math.min(...fresh) : null;
  const stage = fetchedAt === null ? 'fresh' : freshness(now - fetchedAt);
  const core = [input.dayAhead, input.forecast, input.kpis];
  const everyFailed = core.every(r => r.status === 'error');
  const anyData = core.some(r => r.data !== null);
  const allFailed = everyFailed ? { text: allFailedText(failCause(core.map(r => r.error))), showCards: anyData } : null;

  let answer: string;
  if (core.every(r => r.data === null && r.status !== 'error')) answer = ANSWER_LOADING;
  else if (everyFailed && !anyData) answer = ANSWER_ALL_FAILED;
  else if (stage === 'broken' && fetchedAt !== null) answer = answerBrokenText(fetchedAt);
  else if (stage === 'stale' && fetchedAt !== null && p) answer = answerStaleText(fetchedAt, p.price.value);
  else answer = todayAnswerText(answerPrice(c, p), answerPower(c, w));

  const priceView = priceCard(c, p, cmp);
  const forecastView = forecastCard(c, p, cmp);
  const nowPoint = forecastView.state === 'ready' ? forecastPoint(forecastView.view, c.nowHour) : null;
  return {
    heroState: priceView.state,
    answer, fetchedAt, stage, allFailed,
    price: priceView, power: powerCard(c, w), forecast: forecastView, accuracy: accuracyCard(c),
    facts: methodFacts(c, p, cmp, w), sources: sourceRows(c, p, w),
    legend: { confirmed: p && !p.mismatch ? p.price.value : null, forecast: nowPoint && nowPoint.q50 ? nowPoint.q50.value : null },
    recheckPrice: Boolean(p && p.mismatch && input.dayAhead.fetchedAt !== null && input.dayAhead.fetchedAt < hourStartOf(now) + BOUNDARY_RECHECK_MS[0]),
  };
}
