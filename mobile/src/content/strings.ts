// 숫자가 든 짧은 UI 문구(명세 6.1절). .tsx의 JSX 글자에는 숫자를 직접 쓰지 않고 여기서 가져온다(7.10절).
// 숫자·시각은 모두 format.ts를 거친다. 긴 설명 글은 public/content/guide.ko.json(guide.ts가 읽는다)에 둔다.
import {
  MINUS, age, clockHour, clockPhrase, dayShort, duration, krw, kstClock, kstDate, kstDayLabel, kstTime, mw, num, pct, price, tradeClockRange,
  tradeHour,
} from '../lib/format';
import { HOUR_MS, MINUTE_MS, SKEW_PLAUSIBLE_MS, addDays, daysBetween } from '../lib/kst';
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

export type NowPillInput = { now: number; fetchedAt: number | null; skewNotable?: boolean; offline?: boolean };
/**
 * NowPill 글자: "06:04 KST · 조회 1분 전" / "지연 · 06:04 조회 (12분 전)" / "끊김 · 마지막 조회 06:04" / 조회 없는 화면 "06:04 KST".
 * 오프라인이면 지금 시각 대신 마지막 조회를 앞에 둔다: "06:03 조회 · 오프라인"(오프라인 띠의 "마지막 자료 06:03"과 같은 시각).
 */
export function nowPillText({ now, fetchedAt, skewNotable = false, offline = false }: NowPillInput): string {
  const tail = skewNotable ? ' · 서버 시각 기준' : '';
  if (fetchedAt === null) return kstTime(now) + tail;
  if (offline) return `${kstClock(fetchedAt)} 조회 · 오프라인${tail}`;
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
/** 카드의 [다시 시도]·[다시 계산]이 성공했을 때(round 1 D4): 헤더 새로고침 알림과 같은 꼴. */
export const retryDoneText = (at: number) => `다시 불러왔습니다 · ${kstTime(at)}`;
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
/** 평가 방식 한 줄 풀이(D8): 기간은 같은 카드의 출처 줄("평가 기간 …")에 둔다. */
export const accuracyBasisText = () =>
  `${DOCUMENTED_SMP_ACCURACY.method} 평가(지난 자료만 보고 하루씩 맞혀 봄) · ${DOCUMENTED_SMP_ACCURACY.basis}`;
/** 예측 방법 화면에서만 더한다. "약 20%"는 같은 상수(0.203)를 한 번 반올림한 값이다. */
export const accuracyR2Text = () =>
  `R² ${num(DOCUMENTED_SMP_ACCURACY.r2, 3)} · 가격이 오르내리는 폭의 약 ${num(DOCUMENTED_SMP_ACCURACY.r2 * 100, 0)}%를 설명합니다(1이면 전부)`;
/** /model/info의 버전 중 기간 글자("최근 N일 재현", "마지막 N일 보류 검증")가 맞는 것: 기준선(api/v6_routes.py get_model_info). */
const BASELINE_MODEL_VERSION = 'baseline-24h';
export type ApiAccuracyInput = { version: string | null; mape: number | null; coverage: number | null; data_source: string | null };
const daysIn = (text: string, re: RegExp) => {
  const m = re.exec(text);
  return m ? num(Number(m[1]), 0) : null;
};
/**
 * API가 스스로 밝힌 정확도(7.7절, D8): 카드가 아니라 산출 방식 시트의 회색 "API 표기" 줄로만. 기간은 API 설명(data_source)에서 읽어
 * 기준선 버전일 때만 붙이고, 다른 버전은 그 설명을 그대로 옮긴다. 적중률은 "10번 중 N번꼴"로도 쓴다(같은 API 값, 반올림 1회).
 */
export function apiAccuracyLines(info: ApiAccuracyInput): string[] {
  const src = info.data_source ?? '';
  const baseline = info.version === BASELINE_MODEL_VERSION;
  const evalDays = baseline ? daysIn(src, /최근 (\d+)일 재현/) : null;
  const holdDays = baseline ? daysIn(src, /마지막 (\d+)일 보류 검증/) : null;
  const lines: string[] = [];
  if (info.mape !== null) lines.push(`API 표기: 검증 MAPE ${pct(info.mape, 2)}%${evalDays ? ` (최근 ${evalDays}일 재현 · 문서 수치와 기간이 다릅니다)` : ''}`);
  if (info.coverage !== null) {
    lines.push(`API 표기: 80% 구간 적중률 ${pct(info.coverage, 1)}%${holdDays ? ` (마지막 ${holdDays}일 보류 검증)` : ''}`
      + ` · 10번 중 ${num(info.coverage / 10, 0)}번꼴로 실제 가격이 띠 안에 들었습니다`);
  }
  if (!lines.length) lines.push('API 표기: 검증 수치 없음');
  if (!baseline && src) lines.push(`API 설명: ${src}`);
  return lines;
}

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
/** 같은 값이 이어진 시간은 한 구간으로: "02–05시"(문장 속이라 시각 범위만, D11). 떨어진 구간은 쉼표로 잇는다. */
export const hourGroupText = (runs: readonly HourRun[]) => runs.map(r => tradeClockRange(r.from, r.to)).join(', ');

const FORECAST_SUBJECT: Readonly<Record<ForecastKind, string>> = {
  baseline: '기준선 예측으로는', model: '학습 모델 예측으로는', fallback: 'API 대체값으로는',
};
/** "기준선 예측으로는 07–08시가 가장 비싸고(212.3원), 02–05시가 가장 쌉니다(102.3원). 하루 평균 169.2원." */
export function forecastSummaryText(stats: ForecastSummaryInput | null, kind: ForecastKind): string {
  if (!stats) return '그릴 수 있는 예측 값이 없습니다.';
  const excluded = stats.zeroHours > 0 ? ` (${zeroInputExcludedText(stats.zeroHours)})` : '';
  return `${FORECAST_SUBJECT[kind]} ${hourGroupText(stats.max.runs)}가 가장 비싸고(${price(stats.max.value)}원), `
    + `${hourGroupText(stats.min.runs)}가 가장 쌉니다(${price(stats.min.value)}원). 하루 평균 ${price(stats.mean)}원${excluded}.`;
}
/** "가장 불확실한 시간: 12–13시 (80% 구간 폭 326.4원)" */
export const uncertaintyText = (widest: HourGroup) =>
  `가장 불확실한 시간: ${hourGroupText(widest.runs)} (80% 구간 폭 ${price(widest.value)}원)`;
/** 차트 이름의 확정 가격 점(D10: 이번 방문에서 받은 거래시간마다 하나): "점은 KPX 확정 가격 18–19시 151.6원, 19–20시 160.2원". */
export const confirmedPointText = (points: readonly { hour: number; value: number }[]) =>
  `점은 KPX 확정 가격 ${points.map(p => `${tradeClockRange(p.hour)} ${price(p.value)}원`).join(', ')}`;

export const segmentText = (index: number) => `${num(index + 1, 0)}구간`;
/** 계산기 구간 번호(segment_id)로: "1구간"(입찰 행 카드의 머리, round 1 D2). */
export const segmentIdText = (id: number) => `${num(id, 0)}구간`;
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
/** 예비율 경계 = 공급능력 ÷ 1.15. "예비율 15%"의 15도 같은 상수에서 나온다. */
export const watchReservePctText = () => `예비율 ${num((WATCH_RATIO - 1) * 100, 0)}%`;
/**
 * 범위 막대의 경계 표지(D9): KPX 경보 단계 낱말(관심·경계)을 쓰지 않고 수요 수준이라고 밝힌다. 화면 "수요 1,036 MW · 넘으면 예비율 15% 미만",
 * 접근 이름은 boundaryText("수요가 1,036 메가와트를 넘으면 예비율 15% 미만").
 */
export const boundaryRuleText = () => `넘으면 ${watchReservePctText()} 미만`;
export const boundaryText = (boundary: number, unit = 'MW') => `수요가 ${mw(boundary)} ${unit}를 ${boundaryRuleText()}`;

export const tourProgressText = (step: number, total: number) => `${num(step, 0)} / ${num(total, 0)}`;
export const TOUR_NAME = '1분 둘러보기';
/** 둘러보기 단계의 화면 이름: "둘러보기 2/5". */
export const tourLabel = (step: number, total: number) => `둘러보기 ${num(step, 0)}/${num(total, 0)}`;
export const routeTitle = (screen: string) => `${screen} · ${APP_NAME}`;
export const tourTitle = (step: number, total: number) => routeTitle(tourLabel(step, total));

/**
 * 용어 사전 항목의 제목. 숫자가 든 제목이 있어 JSON(숫자 금지)이 아니라 여기 둔다. 뜻풀이는 guide.ko.json.
 * 4.4절 목록 12개 + 정산 화면 용어 줄(4.3절)의 "kWh와 MW"(F9). 4.4절 목록에는 없지만 정산서의 "120 MWh × 150원/kWh"를 읽으려면
 * 단위 풀이가 필요해 13번째로 더했다(F9 증거의 판단 1).
 */
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
  units: 'kWh와 MW',
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
/** 설명 글의 짝·검증 규칙 한 줄(D7): 뜻과 읽는 법 뒤 흐린 글자로. */
export const DETAIL_LABEL = '자세히';

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
    const subject = p.kind === 'now' ? '지금 시간' : tradeClockRange(p.hour);
    const head = `${subject} 제주 전기 도매가격은 1kWh에 ${price(p.value)}원`;
    const past = p.kind === 'hour';
    if (w.kind === 'paired' || w.kind === 'rate') return `${head}${past ? '이었고' : '이고'}, ${powerSentence(w)}`;
    return `${head}${past ? '이었습니다' : '입니다'}. ${powerSentence(w)}`;
  }
  const lead = p.kind === 'none' ? ANSWER_NO_PRICE : p.kind === 'loading' ? '가격 정보를 불러오는 중입니다.' : '가격 정보를 불러오지 못했습니다.';
  return `${lead} ${powerSentence(w)}`;
}

export const loadFailedText = (what: '가격' | '전력' | '예측', now: number) => `${what} 정보를 불러오지 못했습니다 · ${kstTime(now)}`;
/**
 * 전부 실패 StateView. 값이 하나도 없으면 AnswerLine이 이미 "지금은 시장 정보를 불러올 수 없습니다."라서 원인만 쓴다(같은 문장 두 번 금지).
 * 이전 값이 남아 있으면(AnswerLine이 그때 값을 말한다) 문장과 원인을 함께: "지금은 시장 정보를 불러올 수 없습니다 · 서버 응답 없음".
 */
export const allFailedText = (cause: string, withLead: boolean) => (withLead ? `지금은 시장 정보를 불러올 수 없습니다 · ${cause}` : cause);
export const FAIL_CAUSE = { network: '네트워크 연결을 확인해 주세요', server: '서버 응답 없음', shape: '서버 응답을 읽지 못했습니다' } as const;

/** 카드 메타 줄(D11): 시각 범위가 먼저, 거래시간 번호는 이 줄에만. "06–07시 · 거래시간 07h · 하루전시장". */
export const priceHourLine = (hour: number) => `${tradeClockRange(hour)} · 거래시간 ${clockHour(hour)}h · 하루전시장`;
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
export const NO_JUDGE_BROKEN = '새 자료를 받지 못해 예비율 단계 판정을 멈췄습니다';
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
/** 띠가 그림 위로 넘쳤으면 띠 항목에 ▲ 뜻을 붙인다(D10). 확정 가격 점의 한계(서버가 지금 시간 값만 준다)를 같은 줄에 밝힌다. */
export const forecastLegendText = (kind: ForecastKind, clipped = false) =>
  `${clipped ? FORECAST_LEGEND[kind].replace('띠 = 80% 구간', '띠 = 80% 구간(▲ 위쪽 잘림)') : FORECAST_LEGEND[kind]}. ${CONFIRMED_ONLY_NOW}`;
export const CONFIRMED_ONLY_NOW = '확정 가격은 지금 시간 값만 받습니다(서버 제한).';
export const FORECAST_SUBJECT_SHORT: Readonly<Record<ForecastKind, string>> = { baseline: '기준선 예측', model: '학습 모델 예측', fallback: 'API 대체값' };
export const BAND_LABEL = '80% 구간';
export const CONFIRMED_LABEL = '확정 가격';
/** 스텝퍼 값: "07h · 06–07시 · 지금". */
export const hourStepLabel = (hour: number, isNow: boolean) => `${tradeHour(hour)}${isNow ? ' · 지금' : ''}`;
export const WRAP_FIRST = '(처음으로)';
export const WRAP_LAST = '(끝으로)';
export const TO_NOW_LABEL = '지금으로';
/** 이미 지금 거래시간이면 [지금으로]를 지우지 않고 까닭 있는 비활성으로 둔다(round 1 D4: 누른 버튼이 사라져 초점이 body로 떨어졌다). */
export const TO_NOW_REASON = '지금 거래시간입니다';
/** 시간대 요약 행 이름(D11): "밤·아침 00–08시"(거래시간 01–08h). */
export const periodLabelText = (label: string, from: number, to: number) => `${label} ${tradeClockRange(from, to)}`;
/** 0.00 입력 시간이 있을 때만 각주를 단다(없으면 null). */
export const periodNoteText = (zeroHours: number) =>
  (zeroHours > 0 ? `평균·최저·최고에서 0.00 입력 ${num(zeroHours, 0)}시간을 뺐습니다` : null);
export const FULL_DAY_LABEL = '24시간 전체 보기';
export const SHORT_DAY_LABEL = '12시간만 보기';
export const TABLE_CAPTIONS = Object.freeze({ periods: '시간대 요약', hours: '시간별 기준선 예측' });
export const PERIOD_COLUMNS = Object.freeze({ period: '시간대', mean: '평균', min: '최저', max: '최고' });
export const HOUR_COLUMNS = Object.freeze({ hour: '거래시간', band: '80% 구간', confirmed: '확정 가격' });
/** 600 px 이상 표가 카드보다 넓을 때 가장자리 흐림과 함께(D2). */
export const SCROLL_CUE = '옆으로 밀어 보기';
const FORECAST_SOURCE_LINE: Readonly<Record<ForecastKind, string>> = {
  baseline: '출처 API 서버 기준선 예측', model: '출처 API 서버 학습 모델 예측', fallback: '출처 API 서버 대체값 (모델 예측 아님)',
};
/** 기준선은 짧은 이름이 출처와 같아 붙이지 않는다(7.3절 "model_used 짧은 이름"). */
export const forecastSourceLine = (kind: ForecastKind, shortName: string) =>
  (kind === 'baseline' ? FORECAST_SOURCE_LINE[kind] : `${FORECAST_SOURCE_LINE[kind]} (${shortName})`);
export const forecastTimeLine = (forecastDate: string | null, createdAt: number | null) =>
  `${forecastDateText(forecastDate)} · ${createdAtText(createdAt)}`;
/**
 * 모델 이름 줄(D8): API 이름 끝의 " · 검증 MAPE 12.68%"는 떼어 낸다. API 정확도는 정확도 시트의 회색 "API 표기" 줄에만 둔다(I5·7.7절).
 */
export const modelUsedText = (modelUsed: string) => `API 모델 이름: ${modelUsed.replace(/\s*·\s*검증 MAPE [\d.]+%\s*$/, '')}`;
export const FORECAST_EMPTY = '그릴 수 있는 예측 값이 없습니다';

/** 4.1절: "가격이 100원인 시간이라면 보통 12원쯤 빗나간다는 뜻입니다." (MAPE 11.75 %의 100원 기준). */
export const accuracyExampleText = () =>
  `가격이 100원인 시간이라면 보통 ${num(DOCUMENTED_SMP_ACCURACY.mapePct, 0)}원쯤 빗나간다는 뜻입니다.`;
export const ACCURACY_UNREGISTERED = '이 모델의 검증 수치는 앱에 등록되어 있지 않습니다';
export const ACCURACY_SOURCE = '출처 프로젝트 벤치마크 문서(2026-11-06)';
export const MODEL_INFO_FAILED = 'API 표기: 모델 정보를 불러오지 못했습니다';
export const modelInfoFactText = (type: string | null, version: string | null) =>
  `API 모델 정보: ${type || '이름 없음'}${version ? ` (API 버전 ${version})` : ''}`;

/** 오늘 화면 용어 줄(4.1절). 보이는 글자는 짧게, 시트 제목은 GLOSSARY_TITLES. */
export const TODAY_TERMS: ReadonlyArray<{ id: GlossaryId; label: string }> = Object.freeze([
  { id: 'smp', label: 'SMP' }, { id: 'dam', label: '하루전시장' }, { id: 'reserve', label: '예비율' },
  { id: 'supply', label: '공급능력' }, { id: 'band', label: '80% 구간' },
]);
export const TERMS_HEADING = '용어';

/** 오늘 화면의 배너는 제목 한 줄만(round 1 D17), 설명 문장(body)은 알아보기의 둘러보기 카드에서만 쓴다. */
export const TOUR_BANNER = Object.freeze({ title: '처음이신가요? 1분 둘러보기', body: '다섯 단계로 숫자와 표시를 짚어 봅니다.' });

export const TODAY_PANEL_TITLE = '오늘 화면 읽는 법';
export const PANEL_HEADINGS = Object.freeze({ legend: '숫자 옆 표시', terms: '용어', method: '이렇게 계산합니다' });
export const TODAY_PANEL_METHOD = Object.freeze([
  '지금 가격은 KPX가 하루 전에 정한 그 시간의 확정 가격입니다.',
  '24시간 그래프는 어제 같은 시간 가격을 오늘 예측으로 쓴 기준선입니다.',
  `예비력은 공급능력에서 수요를 뺀 값이고, 수요가 막대 아래 표지를 넘으면 ${watchReservePctText()} 아래로 내려갑니다.`,
]);
/** 범례 값(시각 범위와 함께): 둘러보기 5단계·≥ 1280 패널. */
export type LegendValue = { value: number; hour: number };
/**
 * 4상태 범례(4.6절)의 실측·확정 행: 이 화면의 실제 값이 있으면 둘 다 예로 든다("766 MW 실측 · 151.6원/kWh 확정 (18–19시)").
 * 가격 글자는 카드 문장과 같은 꼴(단위를 붙여 씀)이다. 값이 없으면 표시와 설명만.
 */
export function legendMeasuredText(demandMw: number | null, confirmed: LegendValue | null): string {
  const parts = [demandMw === null ? null : `${mw(demandMw)} MW 실측`,
    confirmed === null ? null : `${price(confirmed.value)}원/kWh 확정 (${tradeClockRange(confirmed.hour)})`].filter((p): p is string => p !== null);
  return parts.length ? parts.join(' · ') : '지금 잰 값과 KPX가 정한 확정 가격';
}
export const legendForecastText = (f: LegendValue | null) =>
  (f === null ? '기준선 예측 · 어제 같은 시간 가격' : `${price(f.value)}원/kWh 기준선 예측 (${tradeClockRange(f.hour)})`);
export const LEGEND_ESTIMATED = '추정 · 계산·가정·대체값 (이 화면에서는 API가 실측을 못 줄 때만)';
export const LEGEND_EXAMPLE = '예시 · 설명용 고정 값 (오늘 화면에는 없음)';

/** 출처와 기준 시각 표(4.4절)의 오늘 행. 기준 글자는 KST 시각만 쓴다. */
export const SOURCE_ITEMS = Object.freeze({
  'day-ahead': '하루전 SMP', forecast: '기준선 예측', accuracy: '정확도', demand: '수요·예비율', supply: '공급능력', weather: '날씨',
  market: '시장 일정', bidding: '입찰 계산',
});
export const SOURCE_TEXT = Object.freeze({
  accuracy: '프로젝트 벤치마크(walk-forward) · API 모델 정보',
  supply: 'API 서버 수급 자료(지금 시각 값)',
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
/** 마감 뒤(D13): "다음 마감 11-27(금) 10:00 KST · 입찰일 11-28(토)". 다음 마감의 입찰일은 시장 상태의 입찰일 + 1일(시장 규칙). */
export const nextDeadlineText = (next: number, tradingDate: string | null) =>
  `다음 마감 ${kstDate(next)} ${kstTime(next)}${tradingDate ? ` · 입찰일 ${kstDayLabel(tradingDate)}` : ''}`;
/** 마감 뒤 큰 숫자(D13): 위 줄 "다음 마감까지" + 큰 숫자 "15시간 48분"(숫자가 두 줄로 갈리지 않게 따로 둔다). 서버 시각으로 맞춘 앱 시각에서 센다. */
export const NEXT_DEADLINE_LEAD = '다음 마감까지';
export const nextCountdownText = (leftMs: number) => leftText(leftMs);
export const CLOSED_TODAY_TEXT = '오늘 마감됨';
/** 마감 뒤 예시 계산의 날짜가 마감된 입찰일임을 밝힌다(round 1 integrity). */
export const closedBidDayText = (tradingDate: string) => `마감된 입찰일 ${dayShort(tradingDate)} 기준`;
export const DEADLINE_PILL = Object.freeze({ checking: '확인 중', open: '접수 중', imminent: '마감 임박', closed: '마감', unknown: '상태 알 수 없음' });
/** 시장 상태 조회가 10분(지연)·60분(끊김)을 넘으면 "지연 · 06:04 기준" / "06:04 기준". 카운트다운은 계속 움직인다. */
export const basisLagText = (at: number, stage: Exclude<Freshness, 'fresh'>) =>
  (stage === 'stale' ? `지연 · ${kstClock(at)} 기준` : `${kstClock(at)} 기준`);
export const DEADLINE_CHECKING_TEXT = '시장 상태를 확인하는 중입니다';
export const DEADLINE_FAILED_TEXT = '시장 상태를 확인하지 못했습니다';
export const DEADLINE_MISMATCH_TEXT = '서버의 시장 상태와 계산한 마감 시각이 맞지 않아 남은 시간을 표시하지 않습니다';
/** "실시간시장은 15분 단위입니다 · 다음 구간 06:15 KST"(출처 줄이 이미 API 서버 시장 상태라 "(API 기준)"을 뺐다, round 1 멘토). */
export const rtmLineText = (minutes: number, next: number) => `실시간시장은 ${num(minutes, 0)}분 단위입니다 · 다음 구간 ${kstTime(next)}`;
/** rtm.status가 active가 아닐 때. 모르는 값은 닫힌 쪽으로 "알 수 없음". */
export function rtmStatusText(status: string): string {
  const name = status === 'closed' ? '닫힘' : status === 'inactive' ? '쉬는 중' : status === 'suspended' ? '중단' : '알 수 없음';
  return `실시간시장 상태: ${name}`;
}
export const RTM_NO_INTERVAL = '실시간시장 구간 길이를 받지 못했습니다 (API)';
export const RTM_PRICE_NOTE = '실시간시장 가격 예측은 검증된 제주 모델이 없어 표시하지 않습니다.';
/** API 원문 낱말은 한국어 이름 뒤 괄호에만 둔다(D6): "접수 중 (API 값 open)". 모르는 값은 "알 수 없음". */
const apiWord = (korean: string, raw: string) => `${korean} (API 값 ${raw})`;
const damStatusText = (status: string) => apiWord(status === 'open' ? DEADLINE_PILL.open : status === 'closed' ? DEADLINE_PILL.closed : '알 수 없음', status);
/** 산출 방식 시트의 이번 값(마감): API 원문과 서버 시각으로 다시 계산한 남은 시간. */
export const deadlineApiText = (status: string, deadline: string, tradingDate: string, hours: number | null) =>
  `API 표기: 하루전시장 ${damStatusText(status)} · 마감 ${deadline} · 입찰일 ${tradingDate} · 남은 시간 ${num(hours, 1)}시간`;
export const deadlineCheckText = (hours: number, tolerance: number) =>
  `서버 시각으로 다시 계산한 남은 시간 ${num(hours, 1)}시간 (차이가 ${num(tolerance, 1)}시간을 넘으면 남은 시간을 숨깁니다)`;

const CALC_SHORT: Readonly<Record<ForecastKind, string>> = { baseline: '기준선', model: '학습 모델', fallback: 'API 대체값' };
const CALC_BASIS: Readonly<Record<ForecastKind, string>> = { baseline: '기준선 분위수', model: '학습 모델 분위수', fallback: 'API 대체값' };
export const calcFooterSource = (kind: ForecastKind) => `출처 API 서버 입찰 계산 · ${CALC_SHORT[kind]}`;
export const calcSourceText = (kind: ForecastKind) => `API 서버 입찰 계산 (${CALC_BASIS[kind]})`;
/** 스텝퍼 값: "19h · 18–19시 · 11-27"(320에서는 날짜를 카드 부제로 옮긴다). */
export const bidHourLabel = (hour: number, tradingDate: string | null) => (tradingDate ? `${tradeHour(hour)} · ${dayShort(tradingDate)}` : tradeHour(hour));
export const bidDayText = (tradingDate: string) => `입찰일 ${kstDayLabel(tradingDate)}`;
export const ladderChartLabel = (hour: number) => `${tradeClockRange(hour)} 입찰 계단`;
/** "50 MW를 10구간으로 나눠 171.7–244.1원에 내는 계산입니다." */
export const ladderSentenceText = (totalMw: number, count: number, min: number, max: number) =>
  `${mw(totalMw)} MW를 ${num(count, 0)}구간으로 나눠 ${price(min)}–${price(max)}원에 내는 계산입니다.`;
/** "가격이 낮은 구간일수록 낙찰 가능성(추정)이 높습니다 (95%→5%)." */
export const ladderTrendText = (atMin: number, atMax: number) =>
  `가격이 낮은 구간일수록 낙찰 가능성(추정)이 높습니다 (${probabilityText(atMin)}→${probabilityText(atMax)}).`;
export const ladderNotRealText = (kind: ForecastKind) =>
  `낙찰 가능성은 ${CALC_BASIS[kind]}${kind === 'fallback' ? '으로' : '로'} 계산한 값이며 실제 낙찰 확률이 아닙니다.`;
export const NO_SEND_NOTE = '이 화면은 KPX에 아무것도 보내지 않습니다.';
/** 계단 차트 위 범례(D12): 점선 = 그 시간 기준선, 띠 = 80% 구간, 빗금 막대 = 구간별 입찰 가격(추정). */
export function ladderLegendText(kind: ForecastKind, q50: number | null, q10: number | null, q90: number | null): string {
  const line = q50 === null ? null : `점선 = ${CALC_SHORT[kind]} ${price(q50)}원`;
  const band = q10 === null || q90 === null ? null : `띠 = 80% 구간 ${price(q10)}–${price(q90)}원`;
  return [line, band, '빗금 막대 = 입찰 가격(추정)'].filter((x): x is string => x !== null).join(' · ');
}
/** 계단 차트 단위 옆 설명(D12, round 1 멘토: % 글자의 뜻이 차트에 없었다). */
export const LADDER_CAPTION = '막대 위 % = 낙찰 가능성(추정)';
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
export const bidTableCaption = (hour: number) => `${tradeClockRange(hour)} 구간별 입찰 계산`;
export const BID_COLUMNS = Object.freeze({
  segment: '구간', quantity: '물량(MW, 예시)', price: '가격(원/kWh)', probability: '낙찰 가능성(추정)', revenue: '기대 수입',
});
/** 입찰 성향(4.2절): 세그먼트 글자와 산출 방식 시트가 같은 표를 쓴다. API 값은 conservative·moderate·aggressive. */
export const RISK_LABELS = Object.freeze({ conservative: '보수적', moderate: '균형', aggressive: '공격적' });
// API가 보낸 글자로 표를 찾을 때는 자기 키만 본다("constructor" 같은 값이 Object.prototype의 함수를 읽지 않게).
const ownName = (table: Readonly<Record<string, string>>, key: string, fallback: string) =>
  (Object.prototype.hasOwnProperty.call(table, key) ? table[key] : fallback);
const riskLabel = (risk: string) => ownName(RISK_LABELS, risk, '알 수 없음');
/** 계산 방법의 한국어 이름(api/ai_bidding_optimizer.py optimization_method). 영어 이름은 괄호 안 "API 이름"으로만(D6). */
const CALC_METHOD_NAMES: Readonly<Record<string, string>> = { 'Quantile-based Probabilistic Clearing Model': '분위수 기반 확률 낙찰 모델' };
export function calcMethodText(method: string | null): string {
  if (method === null) return '계산 방법: API 표기 없음';
  if (!/[A-Za-z]/.test(method)) return `계산 방법: ${method}`;
  return `계산 방법: ${ownName(CALC_METHOD_NAMES, method, '한국어 이름 없음')} (API 이름: ${method})`;
}
export const calcInputText = (tradingDate: string | null, risk: string, capacityMw: number) =>
  `API 입찰일 ${kstDayLabel(tradingDate)} · 성향 ${apiWord(riskLabel(risk), risk)} · 예시 용량 ${mw(capacityMw)} MW`;

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

// ---------------------------------------------------------------- 정산 화면(F9, 4.3절). 화면 view model(features/settlement/model.ts)이 부른다.
// 정산은 공개 자료가 아니라서 API를 부르지 않고, 화면의 숫자는 모두 data/examples.ts의 예시 값이다.
export const SETTLEMENT_ANSWER = '정산은 판 전기값을 KPX가 계산해 주는 과정입니다. 아래 숫자는 계산 방식을 보여 주는 예시입니다.';
export const SETTLEMENT_RESTRICTED_TITLE = '실제 정산 자료';
export const SETTLEMENT_RESTRICTED_BODY = '발전소별 실제 정산 자료는 공개 데모에서 제공하지 않습니다. '
  + '이 서비스의 내부 정산 계산도 KPX 정산이 아닌 모의 계산입니다.';
/** 정산서 행 이름. ①·②(원문자 U+2460대)는 금지 기호 범위 밖이고, 아래 설명 문장과 그림이 같은 번호로 가리킨다. */
export const STATEMENT_ROWS = Object.freeze({ dayAhead: '① 하루전시장에서 판 양', deviation: '② 실제 발전과의 차이', total: '합계' });
export const STATEMENT_PERIOD = '하루';
export const STATEMENT_NOTES = Object.freeze([
  '①은 하루전 가격, ②는 실시간 가격으로 계산하는 단순화한 예시입니다.',
  '실제 규칙은 KPX 정산 규정을 따릅니다.',
]);
/** 정산 화면 용어 줄(4.3절): 보이는 글자, 시트 제목은 GLOSSARY_TITLES. */
export const SETTLEMENT_TERMS: ReadonlyArray<{ id: GlossaryId; label: string }> = Object.freeze([
  { id: 'settlement', label: '정산' }, { id: 'dam', label: '하루전시장' }, { id: 'rtm', label: '실시간시장' }, { id: 'units', label: 'kWh와 MW' },
]);
/** 산출 방식 시트의 이번 화면 값(예시): MWh를 kWh로 바꿔 원 단위까지 다시 계산한다(4.3절 "예시 계산은 맞는다"). */
// 정산 문장 속 값과 단위("1 MWh")·연산자와 값("× 150원/kWh")은 줄바꿈하지 않는 빈칸(U+00A0)으로 붙인다(5.4절 값+단위 NoWrap).
// 문장 하나가 Text 하나라 SampleValue의 nowrap을 쓸 수 없다. 화면 읽기에는 보통 빈칸과 같다.
const NB = '\u00A0';
export const unitFactText = (kwhPerMwh: number) =>
  `단위: 1${NB}MWh =${NB}${num(kwhPerMwh, 0)}${NB}kWh (가격이 kWh당이라 MWh로 적은 양을 kWh로 바꿔 곱합니다)`;
export const statementFactText = (label: string, mwh: number, kwh: number, unitPrice: number, won: number) =>
  `${label}: ${num(mwh, 0)}${NB}MWh =${NB}${num(kwh, 0)}${NB}kWh, ×${NB}${num(unitPrice, 0)}원/kWh =${NB}${num(won, 0)}원 (${krw(won, 0)})`;
/** "합계: 18,000,000원 − 840,000원 = 17,160,000원 (1,716만원)". */
export const statementTotalFactText = (first: number, second: number, total: number) =>
  `합계: ${num(first, 0)}원 ${second < 0 ? MINUS : '+'}${NB}${num(Math.abs(second), 0)}원 =${NB}${num(total, 0)}원 (${krw(total, 0)})`;

export const SETTLEMENT_PANEL_TITLE = '정산 읽는 법';
/** 읽는 법 세 줄(4.6절). 셋째 줄의 환산 수는 data/examples.ts의 KWH_PER_MWH에서 받는다. */
export const settlementPanelMethod = (kwhPerMwh: number): readonly string[] => [
  '①은 하루전시장에서 판 양에 하루전 가격을 곱한 돈입니다.',
  '②는 실제로 만든 양과 판 양의 차이에 실시간 가격을 곱한 돈입니다. 덜 만들면 음수라서 빠집니다.',
  `①${NB}+${NB}②가 그날 받는 돈입니다. 가격이 kWh당이라 1${NB}MWh는 ${num(kwhPerMwh, 0)}${NB}kWh로 바꿔 곱합니다.`,
];
/** 4상태 범례의 예시 행. 둘러보기 5단계의 "┆1,716만원 예시 정산서┆"와 같은 글자. */
export const legendStatementText = (won: number) => `${krw(won, 0)} 예시 정산서`;
export const LEGEND_LOCKED = '실제 정산 자료 · 공개 데모에서 제공하지 않음';
/** ①+② 식 그림(≥ 1280 패널)의 칸 이름과 그림 이름(요약). */
export const STATEMENT_FIGURE_LABELS = Object.freeze({ dayAhead: '① 하루전', deviation: '② 실시간 차이', total: '합계' });
export const statementFigureText = (dayAhead: number, deviation: number, total: number) =>
  `예시 정산서 그림: ① 하루전 ${krw(dayAhead, 0)} + ② 실시간 차이 ${krw(deviation, 0)} = 합계 ${krw(total, 0)}`;

// ---------------------------------------------------------------- 알아보기(F10, 4.4절). 긴 글은 guide.ko.json, 여기는 숫자가 들거나 상태로 바뀌는 짧은 글만.
export const GUIDE_ANSWER = '이 앱이 무엇을 보여 주는지, 숫자를 어떻게 읽는지 설명합니다.';
export const tourCardTitle = (steps: number) => `${TOUR_NAME} · ${num(steps, 0)}단계`;
export const glossaryCountText = (count: number) => `용어 ${num(count, 0)}개`;
export const whereText = (tab: string) => `이 앱에서 보는 곳: ${tab}`;
export const ABOUT_TITLE = `${APP_NAME} (공개 데모) ${APP_VERSION}`;
export const ABOUT_ROW = `eXeco ${APP_VERSION} 공개 데모`;
/** 앱 정보의 화면 갱신일. 평가에 넘길 번들을 만들 때 그 날짜로 맞춘다(명세 13절, 뒤에 고치면 sha가 바뀐다). 자료 이용 조건 문장은 guide.ko.json(topics.about). */
export const APP_UPDATED_AT = '2026-11-26';
export const appUpdatedText = () => `화면 갱신 ${APP_UPDATED_AT}`;
export const aboutSkewText = (skewMs: number) => `${clockSkewText(skewMs)} · 서버 시각 기준으로 표시`;
export const ABOUT_SKEW_IMPLAUSIBLE = `서버 시각이 기기 시계와 ${num(SKEW_PLAUSIBLE_MS / HOUR_MS, 0)}시간 넘게 달라 기기 시계로 표시합니다`;
export const MODEL_NOT_USED = '학습한 모델(v4.1)은 제주 재검증에서 이 기준선보다 나빠 쓰지 않습니다.';
/** 예측 방법 화면의 이번 방문 서버 값(7.9절: 설명 문구의 숫자는 API가 말하는 사실에서만). 아무것도 모르면 null. */
export function marketFactsText(deadline: number | null, tradingDate: string | null, minutes: number | null): string | null {
  const parts = [deadline === null ? null : `하루전시장 마감 ${kstDate(deadline)} ${kstTime(deadline)}`,
    tradingDate === null ? null : `입찰일 ${kstDayLabel(tradingDate)}`, minutes === null ? null : `실시간시장 ${num(minutes, 0)}분 단위`];
  const known = parts.filter((p): p is string => p !== null);
  return known.length ? `이번 방문의 서버 값: ${known.join(' · ')}` : null;
}
export const READING_HEADINGS = Object.freeze({ states: '네 가지 표시', other: '그 밖의 표시', time: '시각' });
export const tradeHourExampleText = () => `${tradeHour(7)} = ${clockHour(6)}:00–${clockHour(7)}:00에 거래하는 1시간`;
export const TOUR_CALC_FAILED = '계산 예시를 불러오지 못했습니다';
export const TOUR_FINISH = '직접 둘러보기';
export const TOUR_LEGEND_TITLE = '이번 방문의 표시';
/** /realtime-status(4.4절). 목록 밖 값은 닫힌 쪽("알 수 없음")으로 둔다. 이 행들에는 "연결됨"이라는 말을 쓰지 않는다. */
export const RS_OVERALL = Object.freeze({ all_connected: '모두 응답', partial: '일부 응답', disconnected: '응답 없음' });
export const RS_PART = Object.freeze({ connected: '응답 정상', no_data: '자료 없음', error: '오류', unknown: '알 수 없음' });
export const RS_DATE = Object.freeze({ current_date: '오늘 자료', historical: '과거 날짜 자료', future: '미래 날짜 자료', unknown: '날짜 미상' });
export const RS_DATE_OTHER = '날짜 알 수 없음';
export const RS_ITEMS = Object.freeze({ smp: '응답 상태 · SMP', supply: '응답 상태 · 수급', weather: '응답 상태 · 기상' });
export const RS_SOURCE = 'API 서버 응답 상태(realtime-status)';
export const rsBasisText = (at: number | null) => (at === null ? '응답 시각 미제공' : `응답 시각 ${kstClock(at)}`);
export const rsOverallText = (overall: string) => `API 서버: ${overall}`;
export const RS_FAILED = '응답 상태를 불러오지 못했습니다';
