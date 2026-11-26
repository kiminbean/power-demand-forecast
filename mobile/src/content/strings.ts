// 숫자가 든 짧은 UI 문구(명세 6.1절). .tsx의 JSX 글자에는 숫자를 직접 쓰지 않고 여기서 가져온다(7.10절).
// 숫자·시각은 모두 format.ts를 거친다. 긴 설명 글은 public/content/guide.ko.json(guide.ts가 읽는다)에 둔다.
import { age, clockHour, kstClock, kstTime, mw, num, pct, price, tradeHour, tradeHourRange } from '../lib/format';
import { MINUTE_MS } from '../lib/kst';
import { DOCUMENTED_SMP_ACCURACY, WATCH_RATIO } from '../lib/provenance';
import type { ForecastKind } from '../lib/provenance';

export const APP_NAME = 'eXeco 제주 전력시장';
export const APP_VERSION = 'v6.1';

/** 마지막 성공 조회가 10분을 넘으면 지연, 60분을 넘으면 끊김(7.6절). */
export const STALE_AFTER_MS = 10 * MINUTE_MS;
export const BROKEN_AFTER_MS = 60 * MINUTE_MS;
export type Freshness = 'fresh' | 'stale' | 'broken';
export function freshness(ageMs: number): Freshness {
  if (ageMs > BROKEN_AFTER_MS) return 'broken';
  return ageMs > STALE_AFTER_MS ? 'stale' : 'fresh';
}

export const lookupAt = (fetchedAt: number) => `조회 ${kstTime(fetchedAt)}`;
/** KPX·기상 값: 게시 시각이 없어 조회 시각만 쓴다. */
export const observedLookup = (fetchedAt: number) => `관측 시각 미제공 · 조회 ${kstTime(fetchedAt)}`;
export const createdAtText = (createdAt: number | null) => `만든 시각 ${kstTime(createdAt)}`;
export const serverTimeText = (serverTime: number | null) => `서버 시각 ${kstTime(serverTime)}`;
export const forecastDateText = (dayKey: string | null) => `예측일 ${dayKey ?? '미제공'}`;

export type NowPillInput = { now: number; fetchedAt: number | null; skewNotable?: boolean };
/** NowPill 글자: "06:04 KST · 조회 1분 전" / "지연 · 06:04 조회 (12분 전)" / "끊김 · 마지막 조회 06:04" / 조회 없는 화면 "06:04 KST". */
export function nowPillText({ now, fetchedAt, skewNotable = false }: NowPillInput): string {
  const tail = skewNotable ? ' · 서버 시각 기준' : '';
  if (fetchedAt === null) return kstTime(now) + tail;
  const elapsed = now - fetchedAt;
  const stage = freshness(elapsed);
  if (stage === 'broken') return `끊김 · 마지막 조회 ${kstClock(fetchedAt)}${tail}`;
  if (stage === 'stale') return `지연 · ${kstClock(fetchedAt)} 조회 (${age(elapsed)})${tail}`;
  return `${kstTime(now)} · 조회 ${age(elapsed)}${tail}`;
}

/** 지연이면 "지연 · 06:04 값", 끊김이면 실측 글리프를 거두고 "06:04 값". */
export const staleChipText = (fetchedAt: number, stage: Exclude<Freshness, 'fresh'>) =>
  (stage === 'stale' ? `지연 · ${kstClock(fetchedAt)} 값` : `${kstClock(fetchedAt)} 값`);
export const offlineText = (lastData: number) => `오프라인 · 마지막 자료 ${kstTime(lastData)}`;
export const refreshDoneText = (at: number) => `${kstTime(at)} 기준으로 새로 불러왔습니다`;
export const refreshFailedText = (shown: number) => `새로고침 실패 · ${kstClock(shown)} 값 표시 중`;
export const clockSkewText = (skewMs: number) => `기기 시계와 ${num(Math.abs(skewMs) / MINUTE_MS, 0)}분 차이`;

/** 문서화된 정확도(7.7절). 수치는 provenance.ts의 상수 하나에서만 온다. */
export const accuracyText = () =>
  `시간마다 평균 ${num(DOCUMENTED_SMP_ACCURACY.maeKrwPerKwh, 2)}원/kWh 빗나갑니다 (MAPE ${num(DOCUMENTED_SMP_ACCURACY.mapePct, 2)}%)`;
export const accuracyBasisText = () =>
  `${DOCUMENTED_SMP_ACCURACY.method} ${DOCUMENTED_SMP_ACCURACY.period} · ${DOCUMENTED_SMP_ACCURACY.basis}`;
/** 예측 방법 화면에서만 더한다. */
export const accuracyR2Text = () => `R² ${num(DOCUMENTED_SMP_ACCURACY.r2, 3)}`;
/** API가 스스로 밝힌 수치는 회색 "API 표기" 줄로만. */
export const apiAccuracyText = (mape: number | null, coverage: number | null) =>
  `API 표기: 검증 MAPE ${pct(mape, 2)}% (최근 150일 재현) · 80% 구간 적중률 ${pct(coverage, 1)}% (최근 30일 보류 검증)`;

/** 0.00 입력 시간(7.8절 4). */
export const ZERO_INPUT_LABEL = '기준선 입력 0.00 · 예측 아님';
export const ZERO_INPUT_CELL = '— (입력 0.00)';
export const zeroInputExcludedText = (hours: number) => `0.00 입력 ${num(hours, 0)}시간 제외`;

// 차트 문구(F5). 차트 계산(src/ui/charts/chartMath.ts)이 넘기는 값만 받는다. 모양만 맞으면 되므로 여기서 형식을 따로 적는다.
type HourRun = { from: number; to: number };
type HourGroup = { value: number; runs: readonly HourRun[] };
export type ForecastSummaryInput = { max: HourGroup; min: HourGroup; mean: number; zeroHours: number };
export type LadderSummaryInput = { count: number; minPrice: number; maxPrice: number; probAtMin: number; probAtMax: number };

/** 차트 지금선 pill: "지금 06:04". */
export const nowMarkText = (now: number) => `지금 ${kstClock(now)}`;
/** x 눈금: "00 06 12 18 24시"(마지막에만 단위). */
export const hourTickText = (hour: number, last: boolean) => `${clockHour(hour)}${last ? '시' : ''}`;
/** 계단 차트 x 눈금: "0 10 … 50 MW"(누적 물량, 마지막에만 단위). */
export const mwTickText = (value: number, last: boolean) => `${num(value, 0)}${last ? ' MW' : ''}`;
export const priceTickText = (value: number) => num(value, 0);
/** 같은 값이 이어진 시간은 한 구간으로: "03–05h · 02–05시". 떨어진 구간은 쉼표로 잇는다. */
export const hourGroupText = (runs: readonly HourRun[]) => runs.map(r => tradeHourRange(r.from, r.to)).join(', ');

const FORECAST_SUBJECT: Readonly<Record<ForecastKind, string>> = {
  baseline: '기준선 예측으로는', model: '학습 모델 예측으로는', fallback: 'API 대체값으로는',
};
/** "기준선 예측으로는 08h · 07–08시가 가장 비싸고(212.3원), 03–05h · 02–05시가 가장 쌉니다(102.3원). 하루 평균 169.2원." */
export function forecastSummaryText(stats: ForecastSummaryInput | null, kind: ForecastKind): string {
  if (!stats) return '그릴 수 있는 예측 값이 없습니다.';
  const excluded = stats.zeroHours > 0 ? ` (${zeroInputExcludedText(stats.zeroHours)})` : '';
  return `${FORECAST_SUBJECT[kind]} ${hourGroupText(stats.max.runs)}가 가장 비싸고(${price(stats.max.value)}원), `
    + `${hourGroupText(stats.min.runs)}가 가장 쌉니다(${price(stats.min.value)}원). 하루 평균 ${price(stats.mean)}원${excluded}.`;
}
/** "가장 불확실한 시간: 13h · 12–13시 (80% 구간 폭 326.4원)" */
export const uncertaintyText = (widest: HourGroup) =>
  `가장 불확실한 시간: ${hourGroupText(widest.runs)} (80% 구간 폭 ${price(widest.value)}원)`;
export const confirmedPointText = (tradeHourValue: number, value: number) =>
  `점은 KPX 확정 가격 ${tradeHour(tradeHourValue)} ${price(value)}원`;

export const segmentText = (index: number) => `${num(index + 1, 0)}구간`;
export const probabilityText = (value: number) => `${pct(value, 0)}%`;
/** 계단 차트 요약 이름: "10구간, 171.7–244.1원, 낙찰 가능성 95%에서 5%(추정)" */
export const ladderSummaryText = (s: LadderSummaryInput) =>
  `${num(s.count, 0)}구간, ${price(s.minPrice)}–${price(s.maxPrice)}원, 낙찰 가능성 ${probabilityText(s.probAtMin)}에서 ${probabilityText(s.probAtMax)}(추정)`;
/** 계단 판독 줄: "4구간 · 5 MW · 134.5원 · 낙찰 가능성 62% (추정)". 물량은 예시 발전소 기준이다. */
export const ladderReadingText = (index: number, quantityMw: number, priceValue: number, probability: number) =>
  `${segmentText(index)} · ${mw(quantityMw)} MW · ${price(priceValue)}원 · 낙찰 가능성 ${probabilityText(probability)} (추정)`;

/** 수급 식: "공급능력 1,191 MW − 수요 664 MW = 예비력 527 MW". 접근 이름에는 unit으로 '메가와트'를 넘긴다. */
export const supplyFormulaText = (supply: number, demand: number, reserve: number, unit = 'MW') =>
  `공급능력 ${mw(supply)} ${unit} − 수요 ${mw(demand)} ${unit} = 예비력 ${mw(reserve)} ${unit}`;
/** 관심 경계 = 공급능력 ÷ 1.15. "(예비율 15%)"의 15도 같은 상수에서 나온다. */
export const watchReservePctText = () => `예비율 ${num((WATCH_RATIO - 1) * 100, 0)}%`;
export const watchBoundaryText = (boundary: number, unit = 'MW') => `관심 경계 ${mw(boundary)} ${unit} (${watchReservePctText()})`;

export const tourProgressText = (step: number, total: number) => `${num(step, 0)} / ${num(total, 0)}`;
export const routeTitle = (screen: string) => `${screen} · ${APP_NAME}`;
export const tourTitle = (step: number, total: number) => `둘러보기 ${num(step, 0)}/${num(total, 0)} · ${APP_NAME}`;

/** 용어 사전 12항목의 제목. 숫자가 든 제목이 있어 JSON(숫자 금지)이 아니라 여기 둔다. 뜻풀이는 guide.ko.json. */
export const GLOSSARY_TITLES = Object.freeze({
  smp: 'SMP(계통한계가격)',
  dam: '하루전시장',
  rtm: '실시간시장',
  'trade-hour': '거래시간(01h–24h)',
  demand: '수요',
  supply: '공급능력',
  reserve: '예비율과 예비력',
  baseline: '기준선 예측',
  band: '80% 구간',
  error: '평균 오차(MAE·MAPE)',
  clearing: '낙찰 가능성',
  settlement: '정산',
});
export type GlossaryId = keyof typeof GLOSSARY_TITLES;
