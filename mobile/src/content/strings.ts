// 숫자가 든 짧은 UI 문구(명세 6.1절). .tsx의 JSX 글자에는 숫자를 직접 쓰지 않고 여기서 가져온다(7.10절).
// 숫자·시각은 모두 format.ts를 거친다. 긴 설명 글은 public/content/guide.ko.json(guide.ts가 읽는다)에 둔다.
import {
  age, clockHour, clockPhrase, dayShort, duration, kstClock, kstDate, kstDayLabel, kstTime, mw, num, pct, price, tradeHour, tradeHourRange,
} from '../lib/format';
import { MINUTE_MS, addDays, daysBetween } from '../lib/kst';
import { DOCUMENTED_SMP_ACCURACY, WATCH_RATIO } from '../lib/provenance';
import type { ForecastKind } from '../lib/provenance';
import type { MethodCardId } from './guide';

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
/** 마지막 자료 시각을 모르면(화면이 아직 알려 주지 않음) 시각 없이 쓴다. */
export const offlineText = (lastData: number | null) =>
  (lastData === null ? '오프라인 · 새 자료를 받을 수 없습니다' : `오프라인 · 마지막 자료 ${kstTime(lastData)}`);
export const refreshDoneText = (at: number) => `${kstTime(at)} 기준으로 새로 불러왔습니다`;
export const refreshFailedText = (shown: number) => `새로고침 실패 · ${kstClock(shown)} 값 표시 중`;
export const REFRESH_FAILED_TEXT = '새로고침 실패';
// 헤더 새로고침을 숨기지 않고 비활성으로 둘 때 이름 뒤에 붙는 까닭(3.4·7.6절).
export const REFRESH_NONE_REASON = '이 화면은 새로 불러올 자료가 없습니다';
export const REFRESH_OFFLINE_REASON = '오프라인이라 새로 불러올 수 없습니다';
/** 자기 새로고침을 셸에 알리지 않는 레거시 화면은 다시 그려서 다시 받게 한다. 결과는 그 화면이 보여 준다. */
export const screenReloadingText = (screen: string) => `${screen} 화면을 다시 불러오는 중입니다`;
/** 라우트가 바뀔 때 live region 알림: "입찰 화면". */
export const screenAnnouncement = (screen: string) => `${screen} 화면`;
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
export const TOUR_NAME = '1분 둘러보기';
/** 둘러보기 단계의 화면 이름: "둘러보기 2/5". */
export const tourLabel = (step: number, total: number) => `둘러보기 ${num(step, 0)}/${num(total, 0)}`;
export const routeTitle = (screen: string) => `${screen} · ${APP_NAME}`;
export const tourTitle = (step: number, total: number) => routeTitle(tourLabel(step, total));

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

// ---------------------------------------------------------------- 오늘 화면(F7, 4.1절). 화면 view model(features/today/model.ts)이 부른다.
/** 카드 제목 = [i] 이름("산출 방식 보기: …")과 산출 방식 시트 제목의 원천. 숫자가 든 제목이 있어 여기 둔다. */
export const CARD_TITLES: Readonly<Record<MethodCardId, string>> = Object.freeze({
  'today-price': '지금 시간 전기값',
  'today-power': '지금 제주 전력',
  'today-forecast': '오늘 24시간 가격',
  'today-accuracy': '예측은 얼마나 맞나요?',
  'bid-deadline': '하루전시장 입찰 마감',
  'bid-ladder': '예시 발전소로 입찰 계산해 보기',
  'bid-merit': '가격은 어떻게 정해지나요?',
  'settle-sample': '예시 정산서',
});
export const methodSheetTitle = (card: MethodCardId) => `산출 방식 · ${CARD_TITLES[card]}`;
export const SOURCES_SHEET_TITLE = '출처와 기준 시각';
/** 산출 방식·용어 시트 끝의 표시(7.9절, P7). */
export const SIMPLIFIED_NOTE = '단순화한 설명 · 도메인 검토 전';

export const ANSWER_LOADING = '가격과 전력 정보를 불러오는 중입니다.';
export const ANSWER_ALL_FAILED = '지금은 시장 정보를 불러올 수 없습니다.';
export const ANSWER_NO_PRICE = '지금 거래시간의 확정 가격이 아직 없습니다.';
/** 지연(10–60분): "06:04 기준 제주 전기 도매가격은 1kWh에 115.3원이었습니다." */
export const answerStaleText = (at: number, value: number) => `${kstClock(at)} 기준 제주 전기 도매가격은 1kWh에 ${price(value)}원이었습니다.`;
/** 끊김(60분 넘음). */
export const answerBrokenText = (at: number) => `${kstClock(at)} 뒤로 새 자료를 받지 못했습니다. 아래 값은 그때 값입니다.`;

/** AnswerLine 재료(4.1절 표). hour: 거래시간이 지금과 어긋나 그 거래시간 이름을 붙이고 과거형으로 쓴다. */
export type AnswerPrice = { kind: 'now' | 'hour'; value: number; hour: number } | { kind: 'none' | 'loading' | 'error' };
export type AnswerPower =
  | { kind: 'paired'; reserveMw: number } | { kind: 'rate'; rate: number; verdict: string }
  | { kind: 'nojudge' | 'empty' | 'loading' | 'error' };

function powerSentence(w: AnswerPower): string {
  switch (w.kind) {
    case 'paired': {
      const shown = mw(Math.abs(w.reserveMw));
      if (shown === '0') return '낼 수 있는 전력과 쓰는 전력이 같습니다.';
      return w.reserveMw > 0 ? `낼 수 있는 전력이 쓰는 전력보다 ${shown} MW 더 많습니다.` : `쓰는 전력이 낼 수 있는 전력보다 ${shown} MW 더 많습니다.`;
    }
    case 'rate': return `예비율은 ${pct(w.rate)}%로 ${w.verdict}(앱 기준).`;
    case 'nojudge': return '전력 여유는 실측 값이 없어 판정하지 않습니다.';
    case 'empty': return '전력 값이 비어 있어 판정하지 않습니다.';
    case 'loading': return '전력 정보를 불러오는 중입니다.';
    default: return '전력 정보를 불러오지 못했습니다.';
  }
}

/** "지금 시간 제주 전기 도매가격은 1kWh에 115.3원이고, 낼 수 있는 전력이 쓰는 전력보다 527 MW 더 많습니다." */
export function todayAnswerText(p: AnswerPrice, w: AnswerPower): string {
  if (p.kind === 'now' || p.kind === 'hour') {
    const subject = p.kind === 'now' ? '지금 시간' : tradeHour(p.hour);
    const head = `${subject} 제주 전기 도매가격은 1kWh에 ${price(p.value)}원`;
    const past = p.kind === 'hour';
    if (w.kind === 'paired' || w.kind === 'rate') return `${head}${past ? '이었고' : '이고'}, ${powerSentence(w)}`;
    return `${head}${past ? '이었습니다' : '입니다'}. ${powerSentence(w)}`;
  }
  const lead = p.kind === 'none' ? ANSWER_NO_PRICE : p.kind === 'loading' ? '가격 정보를 불러오는 중입니다.' : '가격 정보를 불러오지 못했습니다.';
  return `${lead} ${powerSentence(w)}`;
}

export const loadFailedText = (what: '가격' | '전력' | '예측', now: number) => `${what} 정보를 불러오지 못했습니다 · ${kstTime(now)}`;
/** 전부 실패 StateView: "지금은 시장 정보를 불러올 수 없습니다 · 서버 응답 없음". */
export const allFailedText = (cause: string) => `지금은 시장 정보를 불러올 수 없습니다 · ${cause}`;
export const FAIL_CAUSE = { network: '네트워크 연결을 확인해 주세요', server: '서버 응답 없음', shape: '서버 응답을 읽지 못했습니다' } as const;

export const priceHourLine = (hour: number) => `거래시간 ${tradeHour(hour)} · 하루전시장`;
export const NO_PRICE_TEXT = 'KPX 확정 가격 없음 · 지금 거래시간 값이 아직 없습니다';
export const BOUNDARY_CHECKING_TEXT = '다음 거래시간 값 확인 중';
/** 가격 지연 칩: "지연 · 07h 값". */
export const hourStaleChipText = (hour: number) => `지연 · ${clockHour(hour)}h 값`;
export const FORECAST_DATE_LAG = '예측일 지연';
/**
 * 비교 둘째 줄 "확정 가격보다 29.9원 높았습니다 (평소 오차 약 12원)."의 조각: 앞말 · 차이 숫자 · 뒷말. 차이(예측 − 확정)는 원값으로
 * 한 번 반올림한다. 평소 오차는 문서화 MAE(기준선일 때만). 반올림해서 0.0이면 숫자 없이 "확정 가격과 같았습니다".
 */
export function compareDeltaParts(diff: number, usualError: number | null): { lead: string; value: string | null; tail: string } {
  const shown = price(Math.abs(diff));
  const usual = usualError === null ? '' : ` (평소 오차 약 ${num(usualError, 0)}원)`;
  if (shown === '0.0') return { lead: `확정 가격과 같았습니다${usual}.`, value: null, tail: '' };
  return { lead: '확정 가격보다 ', value: shown, tail: `원 ${diff > 0 ? '높았습니다' : '낮았습니다'}${usual}.` };
}
/** 산출 방식 시트에만 두는 비율(4.1절): "기준선 예측과 확정 가격의 차이 29.9원은 확정 가격의 25.9%입니다." */
export const compareRatioText = (diff: number, confirmed: number) =>
  `기준선 예측과 확정 가격의 차이 ${price(Math.abs(diff))}원은 확정 가격의 ${pct((Math.abs(diff) / confirmed) * 100)}%입니다.`;
export const kpxOriginalText = (described: string) => `KPX 원문 표기: ${described}`;
export const KPX_LINK_LABEL = 'KPX 원문 보기 (새 창)';

export const KPI_LABELS = Object.freeze({ demand: '쓰는 전력(수요)', reserve: '예비율' });
export const LEVEL_NOTE = '앱 기준 · KPX 공식 경보 단계 아님';
export const supplyMissingText = (kpxRate: boolean) => (kpxRate ? '공급능력 미수신 · 예비율은 KPX 발표값입니다' : '공급능력 미수신');
export const NO_JUDGE_UNKNOWN_SOURCE = '출처를 확인할 수 없어 예비율 단계를 판정하지 않습니다';
export const NO_JUDGE_EMPTY = '예비율 값이 없어 단계를 판정하지 않습니다';
export const NO_JUDGE_BROKEN = '새 자료를 받지 못해 예비율 단계 판정을 거뒀습니다';
export const VALUE_EMPTY = '값 없음';
/** 산출 방식 시트에만(4.1절): MW로 계산한 예비율과 KPX 발표값이 0.1 %p 이상 다를 때. */
export const reserveCalcText = (calc: number, kpx: number) =>
  `화면 MW로 계산하면 ${pct(calc)}% · KPX 발표값 ${pct(kpx)}%와 ${pct(Math.abs(calc - kpx))}%p 차이 (MW 반올림·조회 시점 차이)`;
export const weatherSourceText = (source: string) => `날씨 ${source}`;

const FORECAST_LEGEND: Readonly<Record<ForecastKind, string>> = {
  baseline: '점선 = 기준선 예측(어제 같은 시간 가격) · 띠 = 80% 구간 · 점 = KPX 확정 가격 · 세로선 = 지금',
  model: '점선 = 학습 모델 예측 · 띠 = 80% 구간 · 점 = KPX 확정 가격 · 세로선 = 지금',
  fallback: '점선 = API 대체값(모델 예측 아님) · 점 = KPX 확정 가격 · 세로선 = 지금',
};
export const forecastLegendText = (kind: ForecastKind) => FORECAST_LEGEND[kind];
export const FORECAST_SUBJECT_SHORT: Readonly<Record<ForecastKind, string>> = { baseline: '기준선 예측', model: '학습 모델 예측', fallback: 'API 대체값' };
export const BAND_LABEL = '80% 구간';
export const CONFIRMED_LABEL = '확정 가격';
/** 스텝퍼 값: "07h · 06–07시 · 지금". */
export const hourStepLabel = (hour: number, isNow: boolean) => `${tradeHour(hour)}${isNow ? ' · 지금' : ''}`;
export const WRAP_FIRST = '(처음으로)';
export const WRAP_LAST = '(끝으로)';
export const TO_NOW_LABEL = '지금으로';
export const periodLabelText = (label: string, from: number, to: number) => `${label} ${clockHour(from)}–${clockHour(to)}h`;
export const periodNoteText = (zeroHours: number) =>
  `평균·최저·최고는 0.00 입력 시간을 빼고 계산합니다${zeroHours > 0 ? ` (${zeroInputExcludedText(zeroHours)})` : ''}`;
export const FULL_DAY_LABEL = '24시간 전체 보기';
export const SHORT_DAY_LABEL = '12시간만 보기';
export const TABLE_CAPTIONS = Object.freeze({ periods: '시간대 요약', hours: '시간별 기준선 예측' });
export const PERIOD_COLUMNS = Object.freeze({ period: '시간대', mean: '평균', min: '최저', max: '최고' });
export const HOUR_COLUMNS = Object.freeze({ hour: '거래시간', band: '80% 구간', confirmed: '확정 가격' });
const FORECAST_SOURCE_LINE: Readonly<Record<ForecastKind, string>> = {
  baseline: '출처 API 서버 기준선 예측', model: '출처 API 서버 학습 모델 예측', fallback: '출처 API 서버 대체값 (모델 예측 아님)',
};
/** 기준선은 짧은 이름이 출처와 같아 붙이지 않는다(7.3절 "model_used 짧은 이름"). */
export const forecastSourceLine = (kind: ForecastKind, shortName: string) =>
  (kind === 'baseline' ? FORECAST_SOURCE_LINE[kind] : `${FORECAST_SOURCE_LINE[kind]} (${shortName})`);
export const forecastTimeLine = (forecastDate: string | null, createdAt: number | null) =>
  `${forecastDateText(forecastDate)} · ${createdAtText(createdAt)}`;
export const modelUsedText = (modelUsed: string) => `API 모델 이름: ${modelUsed}`;
export const FORECAST_EMPTY = '그릴 수 있는 예측 값이 없습니다';

/** 4.1절: "가격이 100원인 시간이라면 보통 12원쯤 빗나간다는 뜻입니다." (MAPE 11.75 %의 100원 기준). */
export const accuracyExampleText = () =>
  `가격이 100원인 시간이라면 보통 ${num(DOCUMENTED_SMP_ACCURACY.mapePct, 0)}원쯤 빗나간다는 뜻입니다.`;
export const ACCURACY_UNREGISTERED = '이 모델의 검증 수치는 앱에 등록되어 있지 않습니다';
export const ACCURACY_SOURCE = '출처 프로젝트 벤치마크 문서(2026-11-06) · API 서버 모델 정보';
export const MODEL_INFO_FAILED = 'API 표기: 모델 정보를 불러오지 못했습니다';
export const modelInfoFactText = (type: string | null, version: string | null) =>
  `API 모델 정보: ${[type, version].filter(Boolean).join(' · ') || '이름 없음'}`;

/** 오늘 화면 용어 줄(4.1절). 보이는 글자는 짧게, 시트 제목은 GLOSSARY_TITLES. */
export const TODAY_TERMS: ReadonlyArray<{ id: GlossaryId; label: string }> = Object.freeze([
  { id: 'smp', label: 'SMP' }, { id: 'dam', label: '하루전시장' }, { id: 'reserve', label: '예비율' },
  { id: 'supply', label: '공급능력' }, { id: 'band', label: '80% 구간' },
]);
export const TERMS_HEADING = '용어';

export const TOUR_BANNER = Object.freeze({ title: '처음이신가요? 1분 둘러보기', body: '다섯 단계로 숫자와 표시를 짚어 봅니다.' });

export const TODAY_PANEL_TITLE = '오늘 화면 읽는 법';
export const PANEL_HEADINGS = Object.freeze({ legend: '숫자 옆 표시', terms: '용어', method: '이렇게 계산합니다' });
export const TODAY_PANEL_METHOD = Object.freeze([
  '지금 가격은 KPX가 하루 전에 정한 그 시간의 확정 가격입니다.',
  '24시간 그래프는 어제 같은 시간 가격을 오늘 예측으로 쓴 기준선입니다.',
  `예비력은 공급능력에서 수요를 뺀 값이고, ${watchReservePctText()} 아래로 내려가면 관심 단계입니다.`,
]);
/** 4상태 범례(4.6절): 이 화면의 실제 값이 있으면 값과 함께, 없으면 표시와 설명만. */
export const legendConfirmedText = (value: number | null) => (value === null ? '확정 가격 · KPX가 정한 값' : `${price(value)} 원/kWh 확정 가격`);
export const legendForecastText = (value: number | null) => (value === null ? '기준선 예측 · 어제 같은 시간 가격' : `${price(value)} 원/kWh 기준선 예측`);
export const LEGEND_ESTIMATED = '추정 · 계산·가정·대체값 (이 화면에서는 API가 실측을 못 줄 때만)';
export const LEGEND_EXAMPLE = '예시 · 설명용 고정 값 (오늘 화면에는 없음)';

/** 출처와 기준 시각 표(4.4절)의 오늘 행. 기준 글자는 KST 시각만 쓴다. */
export const SOURCE_ITEMS = Object.freeze({
  'day-ahead': '하루전 SMP', forecast: '기준선 예측', accuracy: '정확도', demand: '수요·예비율', supply: '공급능력', weather: '날씨',
  market: '시장 일정', bidding: '입찰 계산',
});
export const SOURCE_TEXT = Object.freeze({
  accuracy: '프로젝트 벤치마크(walk-forward) · API 모델 정보',
  supply: 'API 서버 수급 자료의 지금 시각 값(수요 ±1 MW 일치 때만)',
  market: 'API 서버 시장 상태',
});
export const SOURCE_HEADINGS = Object.freeze({ item: '항목', source: '출처', basis: '기준 (KST)', status: '상태' });
export const dayAheadBasisText = (tradeDay: string, hour: number, fetchedAt: number) => `${tradeDay} ${clockHour(hour)}h · 조회 ${kstClock(fetchedAt)}`;
export const forecastBasisText = (forecastDate: string | null, createdAt: number | null) =>
  `예측일 ${forecastDate ? kstDayLabel(forecastDate) : '미제공'} · 만든 시각 ${kstClock(createdAt)}`;
export const accuracyBasisPeriod = () => `평가 기간 ${DOCUMENTED_SMP_ACCURACY.period}`;
export const observedBasisText = (fetchedAt: number) => `관측 시각 미제공 · 조회 ${kstClock(fetchedAt)}`;
export const lookupBasisText = (fetchedAt: number) => `조회 ${kstClock(fetchedAt)}`;
export const serverBasisText = (serverTime: number | null) => `서버 시각 ${kstClock(serverTime)}`;
export const SOURCE_PENDING = Object.freeze({
  loading: '불러오는 중', error: '불러오지 못함', missing: '미수신', none: '이 응답에 없음', idle: '이번 방문에서 아직 조회하지 않음',
});
export const SOURCES_NOTES = Object.freeze([
  'KPX가 게시 시각을 넘겨주지 않아 조회 시각만 표시합니다.',
  '아직 연결되지 않은 자료: 오늘 확정 가격 24시간 전체, KPX 게시 시각, 실제 정산·발전소 자료(공개하지 않음).',
]);
export const LIVE_FACTS_HEADING = '이번 화면의 값';

// ---------------------------------------------------------------- 입찰 화면(F8, 4.2절). 화면 view model(features/bidding/model.ts)이 부른다.
export const BIDDING_EXAMPLE_NOTE = '시장 값은 실제, 발전소 값은 예시입니다.';
export const ANSWER_DEADLINE_CHECKING = '입찰 마감 시각을 확인하는 중입니다.';
export const ANSWER_DEADLINE_MISMATCH = '입찰 마감 시각을 확인하지 못해 남은 시간을 표시하지 않습니다.';
export const ANSWER_DEADLINE_ERROR = '시장 상태를 확인하지 못해 남은 시간을 표시하지 않습니다.';
/** 분 내림. 1분이 안 남았으면 "0분"이 아니라 "1분 미만". */
const leftText = (ms: number) => (ms < MINUTE_MS ? '1분 미만' : duration(ms));
/** "발전소는 내일 만들 전기를 오늘 오전 10시까지 미리 팝니다. 마감까지 3시간 56분 남았습니다." 시각은 서버의 마감 표기에서. */
export const answerOpenText = (deadline: number, leftMs: number, imminent: boolean) =>
  `발전소는 내일 만들 전기를 오늘 ${clockPhrase(deadline)}까지 미리 팝니다. 마감까지 ${leftText(leftMs)} 남았습니다${imminent ? '(마감 임박)' : ''}.`;
export const answerClosedText = (next: number) => `오늘 입찰은 마감됐습니다. 다음 마감은 내일 ${clockPhrase(next)}입니다.`;
/** 카운트다운(32/700): "3시간 56분 남음". */
export const countdownText = (leftMs: number) => `${leftText(leftMs)} 남음`;
export const deadlineWhenText = (deadline: number, tradingDate: string) =>
  `마감 ${kstDate(deadline)} ${kstTime(deadline)} · 입찰일 ${kstDayLabel(tradingDate)}`;
export const nextDeadlineText = (next: number) => `다음 마감 ${kstDate(next)} ${kstTime(next)}`;
export const DEADLINE_PILL = Object.freeze({ checking: '확인 중', open: '접수 중', imminent: '마감 임박', closed: '마감', unknown: '상태 알 수 없음' });
/** 시장 상태 조회가 10분(지연)·60분(끊김)을 넘으면 "지연 · 06:04 기준" / "06:04 기준". 카운트다운은 계속 움직인다. */
export const basisLagText = (at: number, stage: Exclude<Freshness, 'fresh'>) =>
  (stage === 'stale' ? `지연 · ${kstClock(at)} 기준` : `${kstClock(at)} 기준`);
export const DEADLINE_CHECKING_TEXT = '시장 상태를 확인하는 중입니다';
export const DEADLINE_FAILED_TEXT = '시장 상태를 확인하지 못했습니다';
export const DEADLINE_MISMATCH_TEXT = '서버의 시장 상태와 계산한 마감 시각이 맞지 않아 남은 시간을 표시하지 않습니다';
/** "실시간시장은 15분 단위입니다 (API 기준) · 다음 구간 06:15 KST" */
export const rtmLineText = (minutes: number, next: number) => `실시간시장은 ${num(minutes, 0)}분 단위입니다 (API 기준) · 다음 구간 ${kstTime(next)}`;
/** rtm.status가 active가 아닐 때. 모르는 값은 닫힌 쪽으로 "알 수 없음". */
export function rtmStatusText(status: string): string {
  const name = status === 'closed' ? '닫힘' : status === 'inactive' ? '쉬는 중' : status === 'suspended' ? '중단' : '알 수 없음';
  return `실시간시장 상태: ${name}`;
}
export const RTM_NO_INTERVAL = '실시간시장 구간 길이를 받지 못했습니다 (API)';
export const RTM_PRICE_NOTE = '실시간시장 가격 예측은 검증된 제주 모델이 없어 표시하지 않습니다.';
/** 산출 방식 시트의 이번 값(마감): API 원문과 서버 시각으로 다시 계산한 남은 시간. */
export const deadlineApiText = (status: string, deadline: string, tradingDate: string, hours: number | null) =>
  `API 표기: 하루전시장 ${status} · 마감 ${deadline} · 입찰일 ${tradingDate} · 남은 시간 ${num(hours, 1)}시간`;
export const deadlineCheckText = (hours: number, tolerance: number) =>
  `서버 시각으로 다시 계산한 남은 시간 ${num(hours, 1)}시간 (차이가 ${num(tolerance, 1)}시간을 넘으면 남은 시간을 숨깁니다)`;

const CALC_SHORT: Readonly<Record<ForecastKind, string>> = { baseline: '기준선', model: '학습 모델', fallback: 'API 대체값' };
const CALC_BASIS: Readonly<Record<ForecastKind, string>> = { baseline: '기준선 분위수', model: '학습 모델 분위수', fallback: 'API 대체값' };
export const calcFooterSource = (kind: ForecastKind) => `출처 API 서버 입찰 계산 · ${CALC_SHORT[kind]}`;
export const calcSourceText = (kind: ForecastKind) => `API 서버 입찰 계산 (${CALC_BASIS[kind]})`;
/** 스텝퍼 값: "19h · 18–19시 · 11-27"(320에서는 날짜를 카드 부제로 옮긴다). */
export const bidHourLabel = (hour: number, tradingDate: string | null) => (tradingDate ? `${tradeHour(hour)} · ${dayShort(tradingDate)}` : tradeHour(hour));
export const bidDayText = (tradingDate: string) => `입찰일 ${kstDayLabel(tradingDate)}`;
export const ladderChartLabel = (hour: number) => `${tradeHour(hour)} 입찰 계단`;
/** "50 MW를 10구간으로 나눠 171.7–244.1원에 내는 계산입니다." */
export const ladderSentenceText = (totalMw: number, count: number, min: number, max: number) =>
  `${mw(totalMw)} MW를 ${num(count, 0)}구간으로 나눠 ${price(min)}–${price(max)}원에 내는 계산입니다.`;
/** "가격이 낮은 구간일수록 낙찰 가능성(추정)이 높습니다 (95%→5%)." */
export const ladderTrendText = (atMin: number, atMax: number) =>
  `가격이 낮은 구간일수록 낙찰 가능성(추정)이 높습니다 (${probabilityText(atMin)}→${probabilityText(atMax)}).`;
export const ladderNotRealText = (kind: ForecastKind) =>
  `낙찰 가능성은 ${CALC_BASIS[kind]}${kind === 'fallback' ? '으로' : '로'} 계산한 값이며 실제 낙찰 확률이 아닙니다.`;
export const NO_SEND_NOTE = '이 화면은 KPX에 아무것도 보내지 않습니다.';
// 숫자로 끝나는 말 뒤의 '과/와': 2·4·5·9(이·사·오·구)로 끝나면 받침이 없다.
const waGwa = (text: string) => ('2459'.includes(text.slice(-1)) ? '와' : '과');
/** 가격 근거 공개 줄(7.8절 2): "가격 근거: 11-26 기준선(=11-25 확정 가격) · 입찰일 11-27과 하루 차이". */
export function priceBasisText(forecastDate: string, tradingDate: string, kind: 'baseline' | 'model'): string {
  const basis = kind === 'baseline'
    ? `${dayShort(forecastDate)} 기준선(=${dayShort(addDays(forecastDate, -1))} 확정 가격)` : `${dayShort(forecastDate)} 학습 모델 예측`;
  const gap = Math.abs(daysBetween(forecastDate, tradingDate) ?? 0);
  const day = dayShort(tradingDate);
  return `가격 근거: ${basis} · 입찰일 ${day}${waGwa(day)} ${gap === 1 ? '하루' : `${num(gap, 0)}일`} 차이`;
}
export const CALC_BUSY_TEXT = '다른 요청을 처리하는 중입니다 · 잠시 뒤 다시 계산해 주세요';
export const CALC_BAD_REQUEST_TEXT = '계산 조건 오류';
export const CALC_FAILED_TEXT = '계산 결과를 불러오지 못했습니다';
export const CALC_UNREADABLE_TEXT = '계산 결과를 읽지 못했습니다';
export const CALC_EMPTY_TEXT = '계산된 입찰 구간이 없습니다';
export const RECALC_LABEL = '다시 계산';
export const bidTableCaption = (hour: number) => `${tradeHour(hour)} 구간별 입찰 계산`;
export const BID_COLUMNS = Object.freeze({
  segment: '구간', quantity: '물량(MW, 예시)', price: '가격(원/kWh)', probability: '낙찰 가능성(추정)', revenue: '기대 수입',
});
export const calcMethodText = (method: string | null) => `API 계산 방법: ${method ?? '표기 없음'}`;
export const calcInputText = (tradingDate: string | null, risk: string, capacityMw: number) =>
  `API 입찰일 ${kstDayLabel(tradingDate)} · 성향 ${risk} · 예시 용량 ${mw(capacityMw)} MW`;

export const MERIT_TOGGLE = Object.freeze({ open: '펼치기', close: '접기' });
export const meritDemandText = (demandMw: number) => `예시 수요 ${mw(demandMw)} MW`;
export const meritPriceText = (value: number) => `${num(value, 0)}원`;
/** 그림 이름이자 요약: 마지막으로 필요한 발전기의 가격이 그 시간 가격이 된다(예시 값). */
export const meritSummaryText = (demandMw: number, name: string, value: number) =>
  `${meritDemandText(demandMw)}를 채우는 마지막 발전기(${name})의 가격 ${num(value, 0)}원/kWh가 이 시간 모두의 가격(SMP)이 됩니다.`;
export const MY_PLANT_TITLE = '내 발전소';
export const MY_PLANT_BODY = '내 발전소 관리와 실제 입찰 제출은 공개 데모에서 제공하지 않습니다. 이 화면은 자료를 저장하거나 KPX에 보내지 않습니다. '
  + '위의 예시 발전소로 계산 방식을 볼 수 있습니다.';

export const BIDDING_PANEL_TITLE = '입찰 읽는 법';
export const BIDDING_TERMS: ReadonlyArray<{ id: GlossaryId; label: string }> = Object.freeze([
  { id: 'dam', label: '하루전시장' }, { id: 'clearing', label: '낙찰 가능성' }, { id: 'rtm', label: '실시간시장' },
]);
export const BIDDING_PANEL_METHOD = Object.freeze([
  '계단 하나가 구간 하나이고, 높이가 그 구간에 내는 가격입니다.',
  '계단 위 숫자는 분위수로 계산한 낙찰 가능성(추정)이며 실제 낙찰 확률이 아닙니다.',
  '호박색 점선 표시는 예시, 빗금 표시는 추정, 표시 없는 칩은 시장 규칙입니다.',
]);
export const legendRuleText = (when: string | null) => when ?? '마감 시각 · 서버의 시장 상태 기준';
export const legendProbabilityText = (value: number | null) =>
  (value === null ? '낙찰 가능성 · 분위수로 계산한 추정' : `${probabilityText(value)} 낙찰 가능성 (예시 발전소 1구간)`);
export const legendPlantText = (capacityMw: number) => `${mw(capacityMw)} MW 예시 풍력발전소 (가상)`;
