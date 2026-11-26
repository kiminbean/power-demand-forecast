// 차트 계산(명세 4.1·4.2·5.3·7.5·7.8절). React·DOM 없이 순수 함수만 둔다: 눈금·계단 경계·지금선·띠·0.00 빗금 열 같은
// 기하와 최고·최저·평균·가장 불확실한 시간·시간대 요약·계단 요약을 node 테스트(tests/charts.test.cjs)가 그대로 고정한다.
// "같은 값"은 화면 글자(format.ts price)로 판단한다: 화면에서 같아 보이는 값이 이어지면 한 구간으로 묶는다.
import { CHART } from '../tokens';
import { kstDateKey, nowLineFraction } from '../../lib/kst';
import { price } from '../../lib/format';
import { priceTickText } from '../../content/strings';

export const HOURS = 24;
/** x 눈금: 0·6·12·18·24시, 위치는 i/24. */
export const X_TICK_HOURS = [0, 6, 12, 18, 24] as const;
/** y 눈금 간격 후보. 모두 50의 배수라 맨 위 눈금은 늘 50 단위로 올림된다(4.2절). */
const Y_STEPS = [50, 100, 200, 250, 500, 1000, 2000, 5000, 10000] as const;
/** 계단 차트 x(누적 MW) 눈금 간격 후보. */
const MW_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500] as const;
const MW_INTERVALS = 5;
/** 1·4·7·10구간: 360 미만에서는 세 칸마다 하나만 글자를 단다(4.2절). */
const THIN_EVERY = 3;

export type Box = { left: number; top: number; width: number; height: number };
export type Scale = { bottom: number; top: number; step: number; ticks: number[] };
/** 거래시간(1–24) 닫힌 구간. */
export type Run = { from: number; to: number };
/** 시간 칸: 값 / 0.00 입력(선을 끊고 빗금) / 값 없음(선만 끊는다). */
export type HourKind = 'value' | 'zero' | 'missing';
export type HourGroup = { value: number; runs: Run[] };

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));
/** SVG 좌표는 소수 둘째 자리까지만 쓴다(경로 문자열 길이). 화면 글자가 아니라 format.ts를 거치지 않는다. */
export const r2 = (n: number) => Math.round(n * 100) / 100;

/** 앞 24시간만 본다(거래시간 1–24). 응답이 짧으면 모자란 시간은 '값 없음'이다. */
export function hourKinds(q50: readonly (number | null)[]): HourKind[] {
  return Array.from({ length: HOURS }, (_, i): HourKind => {
    const v = q50[i];
    if (!isFiniteNumber(v)) return 'missing';
    return v === 0 ? 'zero' : 'value';
  });
}

/** 참인 칸들을 이어진 거래시간 구간으로 묶는다. flags[i]는 거래시간 i + 1. */
export function runsOf(flags: readonly boolean[]): Run[] {
  const out: Run[] = [];
  flags.forEach((on, i) => {
    if (!on) return;
    const last = out[out.length - 1];
    if (last && last.to === i) last.to = i + 1;
    else out.push({ from: i + 1, to: i + 1 });
  });
  return out;
}

export const zeroRuns = (kinds: readonly HourKind[]) => runsOf(kinds.map(k => k === 'zero'));

/** y 눈금: 0부터(음수가 있으면 그 아래 눈금부터), 간격은 maxIntervals 칸 안에 들어가는 가장 작은 후보. */
export function niceScale(values: readonly (number | null | undefined)[], maxIntervals: number = CHART.yIntervals): Scale {
  const finite = values.filter(isFiniteNumber);
  const hi = finite.length ? Math.max(0, ...finite) : 0;
  const lo = finite.length ? Math.min(0, ...finite) : 0;
  const intervals = (s: number) => Math.max(1, Math.ceil(hi / s)) - Math.floor(lo / s);
  const step = Y_STEPS.find(s => intervals(s) <= maxIntervals) ?? Y_STEPS[Y_STEPS.length - 1];
  const top = Math.max(1, Math.ceil(hi / step)) * step;
  const bottom = Math.floor(lo / step) * step;
  const ticks: number[] = [];
  for (let k = 0; bottom + k * step <= top; k += 1) ticks.push(bottom + k * step);
  return { bottom, top, step, ticks };
}

export function yOf(value: number, scale: Scale, plot: Box): number {
  const span = scale.top - scale.bottom || 1;
  return plot.top + ((scale.top - value) / span) * plot.height;
}

/** 시 경계 k(0–24)의 x. 거래시간 H는 경계 H−1과 H 사이다. */
export const boundaryX = (k: number, plot: Box) => plot.left + (k / HOURS) * plot.width;
export const hourCenterX = (hour: number, plot: Box) => boundaryX(hour - 0.5, plot);

/** 누른 자리 x(차트 상자 기준)에서 가장 가까운 거래시간. 그림 밖은 양 끝 시간으로 붙인다. */
export function hourAtX(x: number, plot: Box): number {
  return clamp(Math.floor(((x - plot.left) / plot.width) * HOURS) + 1, 1, HOURS);
}

/** ←/→는 한 칸(양 끝에서 멈춤), Home·End는 처음·끝. 해당 키가 아니면 null. */
export function keyStep(key: string, current: number, min: number, max: number): number | null {
  if (key === 'ArrowLeft') return Math.max(min, current - 1);
  if (key === 'ArrowRight') return Math.min(max, current + 1);
  if (key === 'Home') return min;
  if (key === 'End') return max;
  return null;
}

/** 지금선 위치(0–1). 예측일이 KST 오늘일 때만 그린다(4.1절 상태표 "예측일 불일치"). */
export function nowFraction(now: number | null, forecastDate: string | null): number | null {
  if (now === null || !isFiniteNumber(now) || !forecastDate || kstDateKey(now) !== forecastDate) return null;
  return nowLineFraction(now);
}

/** 기준선 계단(값 시간만). 0.00 입력·값 없음에서 선을 끊으므로 구간마다 새 M으로 시작한다. */
export function stepPath(values: readonly (number | null)[], kinds: readonly HourKind[], scale: Scale, plot: Box): string {
  let d = '';
  for (const run of runsOf(kinds.map(k => k === 'value'))) {
    for (let h = run.from; h <= run.to; h += 1) {
      const y = r2(yOf(values[h - 1] as number, scale, plot));
      const x1 = r2(boundaryX(h, plot));
      d += h === run.from ? `M${r2(boundaryX(h - 1, plot))} ${y}H${x1}` : `V${y}H${x1}`;
    }
  }
  return d;
}

/** 80% 띠(q10–q90): 값 시간 가운데 두 분위수가 다 있는 구간만. q90 계단을 따라 갔다가 q10 계단으로 돌아와 닫는다. */
export function bandPath(kinds: readonly HourKind[], q10: readonly (number | null)[], q90: readonly (number | null)[], scale: Scale, plot: Box): string {
  const ok = kinds.map((k, i) => k === 'value' && isFiniteNumber(q10[i]) && isFiniteNumber(q90[i]));
  let d = '';
  for (const run of runsOf(ok)) {
    let upper = '';
    let lower = '';
    for (let h = run.from; h <= run.to; h += 1) {
      const yHi = r2(yOf(q90[h - 1] as number, scale, plot));
      upper += h === run.from ? `M${r2(boundaryX(h - 1, plot))} ${yHi}H${r2(boundaryX(h, plot))}` : `V${yHi}H${r2(boundaryX(h, plot))}`;
    }
    for (let h = run.to; h >= run.from; h -= 1) {
      lower += `V${r2(yOf(q10[h - 1] as number, scale, plot))}H${r2(boundaryX(h - 1, plot))}`;
    }
    d += `${upper}${lower}Z`;
  }
  return d;
}

/** 화면 글자 기준 같은 값(pick과 같은 price 글자)인 시간들을 이어진 구간으로 묶는다. */
function groupOf(values: readonly (number | null)[], eligible: readonly boolean[], pick: number): HourGroup {
  const key = price(pick);
  return { value: pick, runs: runsOf(eligible.map((on, i) => on && isFiniteNumber(values[i]) && price(values[i]) === key)) };
}

export type ForecastStats = { max: HourGroup; min: HourGroup; mean: number; counted: number; zeroHours: number };

/** 최고·최저·평균. 0.00 입력 시간은 빼고 그 수를 남긴다(7.8절 4). 값이 하나도 없으면 null. */
export function forecastStats(q50: readonly (number | null)[]): ForecastStats | null {
  const kinds = hourKinds(q50);
  const eligible = kinds.map(k => k === 'value');
  const values = q50.slice(0, HOURS).filter((v, i): v is number => eligible[i]);
  if (!values.length) return null;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return {
    max: groupOf(q50, eligible, Math.max(...values)), min: groupOf(q50, eligible, Math.min(...values)),
    mean, counted: values.length, zeroHours: kinds.filter(k => k === 'zero').length,
  };
}

/** 80% 구간 폭(q90 − q10)이 가장 큰 시간. 0.00 입력·값 없음은 뺀다. */
export function widestBand(q50: readonly (number | null)[], q10: readonly (number | null)[], q90: readonly (number | null)[]): HourGroup | null {
  const kinds = hourKinds(q50);
  const widths = kinds.map((k, i) => (k === 'value' && isFiniteNumber(q10[i]) && isFiniteNumber(q90[i]) ? (q90[i] as number) - (q10[i] as number) : null));
  const finite = widths.filter(isFiniteNumber);
  if (!finite.length) return null;
  return groupOf(widths, widths.map(isFiniteNumber), Math.max(...finite));
}

export const PERIODS = [
  { id: 'night', label: '심야', from: 1, to: 8 }, { id: 'day', label: '주간', from: 9, to: 17 }, { id: 'evening', label: '저녁', from: 18, to: 24 },
] as const;
export type PeriodStat = { id: (typeof PERIODS)[number]['id']; label: string; from: number; to: number;
  mean: number | null; min: number | null; max: number | null; zeroHours: number };

/** 시간대 요약(심야 01–08h · 주간 09–17h · 저녁 18–24h): 평균·최저·최고, 0.00 입력 시간 제외. */
export function periodStats(q50: readonly (number | null)[]): PeriodStat[] {
  const kinds = hourKinds(q50);
  return PERIODS.map(p => {
    const hours = Array.from({ length: p.to - p.from + 1 }, (_, i) => p.from + i);
    const values = hours.filter(h => kinds[h - 1] === 'value').map(h => q50[h - 1] as number);
    const zeroHours = hours.filter(h => kinds[h - 1] === 'zero').length;
    const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    return { id: p.id, label: p.label, from: p.from, to: p.to, mean,
      min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null, zeroHours };
  });
}

/** 12 px 글자 폭 추정(넉넉하게). 라벨 상자 크기와 y 눈금 자리 폭에 쓴다. */
export function textWidth(text: string): number {
  let w = 0;
  for (const ch of text) {
    if (/[가-힣]/.test(ch)) w += CHART.glyph.hangul;
    else if (ch === ' ') w += CHART.glyph.space;
    else if (/[MW%m]/.test(ch)) w += CHART.glyph.wide;
    else w += CHART.glyph.other;
  }
  return Math.ceil(w);
}

/** 라벨 상자를 가운데 x에 두되 차트 상자 [0, width] 밖으로 나가지 않게 붙인다. */
export const labelLeft = (centerX: number, boxWidth: number, width: number) => clamp(centerX - boxWidth / 2, 0, Math.max(0, width - boxWidth));

/** y 눈금 글자 자리: 가장 긴 눈금 글자 + 간격. */
export const yGutter = (scale: Scale) => CHART.edge + Math.max(...scale.ticks.map(t => textWidth(priceTickText(t)))) + CHART.tickGap;

const pickHeight = (buckets: readonly { below: number; height: number }[], width: number) =>
  (buckets.find(b => width < b.below) ?? buckets[buckets.length - 1]).height;
/** 기준선 차트 높이: 카드 폭 296 → 200, 358 → 220, 398 → 230, 608 → 260(4.1절). */
export const stepChartHeight = (width: number) => pickHeight(CHART.stepHeights, width);
/** 계단 차트 높이: 296 → 180, 358 → 200, 608 → 240(4.2절). */
export const ladderChartHeight = (width: number) => pickHeight(CHART.ladderHeights, width);

export type StepLayout = { width: number; height: number; plot: Box; scale: Scale; note: boolean };
/** 위에서부터: 단위·지금 pill 줄 → 그림 → x 눈금 줄 → (0.00 입력이 있으면) 각주 줄. */
export function stepLayout(width: number, scale: Scale, note: boolean): StepLayout {
  const height = stepChartHeight(width);
  const left = yGutter(scale);
  const bottom = height - CHART.xRow - (note ? CHART.noteRow : 0);
  return { width, height, scale, note, plot: { left, top: CHART.topRow, width: Math.max(1, width - left - CHART.edge), height: Math.max(1, bottom - CHART.topRow) } };
}

/** 지금 pill: 지금선 가운데에 두되 왼쪽 위 단위 글자와 겹치지 않고 차트 밖으로 나가지 않게 붙인다. */
export function pillLeft(lineX: number, pillWidth: number, unitWidth: number, width: number): number {
  const min = CHART.edge + unitWidth + CHART.tickGap;
  return clamp(lineX - pillWidth / 2, min, Math.max(min, width - CHART.edge - pillWidth));
}

export type LadderItem = { price: number; probability: number; quantity: number };
export type LadderBar = { index: number; x0: number; x1: number; y: number };

/** 계단 막대: x는 누적 물량(0–합계), 높이는 가격(0부터). 합계가 0 이하이면 같은 폭으로 나눈다. */
export function ladderBars(items: readonly LadderItem[], scale: Scale, plot: Box): LadderBar[] {
  const total = items.reduce((a, it) => a + (isFiniteNumber(it.quantity) && it.quantity > 0 ? it.quantity : 0), 0);
  let cum = 0;
  return items.map((it, index) => {
    const q = total > 0 ? (isFiniteNumber(it.quantity) && it.quantity > 0 ? it.quantity : 0) : 1;
    const span = total > 0 ? total : items.length;
    const x0 = plot.left + (cum / span) * plot.width;
    cum += q;
    return { index, x0, x1: plot.left + (cum / span) * plot.width, y: yOf(it.price, scale, plot) };
  });
}

/** 누른 자리 x의 막대. 막대 밖은 가장 가까운 끝 막대로 붙인다. 막대가 없으면 null. */
export function barAtX(x: number, bars: readonly LadderBar[]): number | null {
  if (!bars.length) return null;
  const hit = bars.find(b => x >= b.x0 && x < b.x1);
  if (hit) return hit.index;
  return x < bars[0].x0 ? 0 : bars.length - 1;
}

/** 확률 글자를 달 구간. 360 미만이면 1·4·7·10구간(세 칸마다)과, 떨어져 있을 때만 마지막 구간. */
export function ladderLabelIndices(count: number, narrow: boolean): number[] {
  const all = Array.from({ length: Math.max(0, count) }, (_, i) => i);
  if (!narrow) return all;
  const thin = all.filter(i => i % THIN_EVERY === 0);
  const last = count - 1;
  if (last > 0 && !thin.includes(last) && last - thin[thin.length - 1] >= 2) thin.push(last);
  return thin;
}

export type LadderStats = { count: number; minPrice: number; maxPrice: number; probAtMin: number; probAtMax: number };
/** 요약 "10구간, 171.7–244.1원, 낙찰 가능성 95%에서 5%(추정)": 가장 싼·비싼 구간의 가격과 그 구간의 낙찰 가능성. */
export function ladderStats(items: readonly LadderItem[]): LadderStats | null {
  const valid = items.filter(it => isFiniteNumber(it.price) && isFiniteNumber(it.probability));
  if (!valid.length) return null;
  const sorted = valid.slice().sort((a, b) => a.price - b.price);
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  return { count: items.length, minPrice: lo.price, maxPrice: hi.price, probAtMin: lo.probability, probAtMax: hi.probability };
}

/** 누적 물량 눈금(0부터 합계까지, 다섯 칸 안). */
export function mwTicks(total: number): number[] {
  if (!isFiniteNumber(total) || total <= 0) return [0];
  const step = MW_STEPS.find(s => total / s <= MW_INTERVALS) ?? MW_STEPS[MW_STEPS.length - 1];
  const ticks: number[] = [];
  for (let v = 0; v <= total + 1e-9; v += step) ticks.push(v);
  return ticks;
}

export type LadderLayout = { width: number; height: number; plot: Box; scale: Scale };
/** 위에서부터: 단위 줄과 가장 높은 막대의 확률 글자 자리 → 그림 → x 눈금 줄. */
export function ladderLayout(width: number, scale: Scale): LadderLayout {
  const height = ladderChartHeight(width);
  const left = yGutter(scale);
  const bottom = height - CHART.xRow;
  return { width, height, scale, plot: { left, top: CHART.ladderTop, width: Math.max(1, width - left - CHART.edge), height: Math.max(1, bottom - CHART.ladderTop) } };
}

export type RangeGeometry = { demandX: number; boundaryX: number; width: number };
/** 범위 막대: 0…공급능력. 수요·관심 경계 위치는 [0, 폭] 안으로 붙인다. 공급능력이 0 이하이면 그리지 않는다. */
export function rangeBarGeometry(supply: number, demand: number, boundary: number, width: number): RangeGeometry | null {
  if (![supply, demand, boundary, width].every(isFiniteNumber) || supply <= 0 || width <= 0) return null;
  return { width, demandX: clamp(demand / supply, 0, 1) * width, boundaryX: clamp(boundary / supply, 0, 1) * width };
}

/** 눈금 글자 상자 폭(넉넉하게): 추정 폭 + 좌우 2 px. */
export const tickBox = (text: string) => textWidth(text) + CHART.labelGap * 2;
