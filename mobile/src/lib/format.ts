// 숫자·시각이 글자가 되는 유일한 곳(명세 6.3절). 반올림은 여기서 한 번만 한다.
// null·NaN·±Infinity는 언제나 '—'(기존 formatNullableNumber와 같은 규칙), 진짜 0은 '0.0'.
import { formatNullableNumber } from '../utils/smpPresentation';
import { kstDayStart, kstParts, MINUTE_MS } from './kst';

export const DASH = '—';
/** 음수는 하이픈이 아닌 빼기 기호를 쓴다("−17.2"). */
export const MINUS = '−';
export const PRICE_UNIT = '원/kWh';
const WEEKDAYS = ['일', '월', '화', '수', '목', '금', '토'] as const;

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const pad2 = (n: number) => String(n).padStart(2, '0');
const isZeroText = (text: string) => !/[1-9]/.test(text);

/**
 * API가 보낸 십진 값을 그대로 반올림한다(0.5는 0에서 먼 쪽): 58.05 → 58.1, 2.675 → 2.68. toFixed는 이진 근삿값(58.0499…)을
 * 반올림해 "58.0"이 되어 I3(화면 값 = API 값을 한 번 반올림)과 어긋난다. 지수 표기로 자릿수를 옮기면 십진 글자 그대로 계산된다.
 * 아주 크거나 작은 값(글자가 이미 지수 표기)은 손대지 않는다.
 */
function roundDecimal(value: number, decimals: number): number {
  const text = String(Math.abs(value));
  if (text.includes('e')) return value;
  const rounded = Math.round(Number(`${text}e${decimals}`));
  return Math.sign(value) * Number(`${rounded}e-${decimals}`);
}

/** 소수 decimals자리로 한 번 반올림하고 정수부에 천 단위 쉼표를 넣는다. 반올림 결과가 0이면 부호를 떼어 '0.0'. */
export function num(value: number | null | undefined, decimals = 1): string {
  const raw = formatNullableNumber(isFiniteNumber(value) ? roundDecimal(value, decimals) : value, decimals);
  if (raw === DASH) return DASH;
  const negative = raw.startsWith('-');
  const [int, frac] = (negative ? raw.slice(1) : raw).split('.');
  const text = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (frac === undefined ? '' : `.${frac}`);
  return negative && !isZeroText(text) ? MINUS + text : text;
}

/** 원/kWh 값. 115.27 → "115.3" */
export const price = (v: number | null | undefined) => num(v, 1);
/** MW 값. 1191.3 → "1,191" */
export const mw = (v: number | null | undefined) => num(v, 0);
/** 백분율 값(이미 % 단위). 79.5 → "79.5" */
export const pct = (v: number | null | undefined, decimals = 1) => num(v, decimals);

/** 원 → 만원·억원(1억 이상은 소수 둘째 자리 억원). 반올림 1회. */
export function krw(won: number | null | undefined, decimals = 1): string {
  if (!isFiniteNumber(won)) return DASH;
  const abs = Math.abs(won);
  // 만원 자릿수: 입찰 기대 수입은 소수 첫째 자리("468.6만원"), 정산 예시처럼 만원 단위로 떨어지는 금액은 0("1,716만원", 4.3절).
  const man = num(abs / 1e4, decimals);
  const body = Number(man.replace(/,/g, '')) >= 1e4 ? `${num(abs / 1e8, 2)}억원` : `${man}만원`;
  return won < 0 && !isZeroText(body) ? MINUS + body : body;
}

/** API expected_revenue는 천 원 단위다(7.8절 5). 4686.0 → "468.6만원", 560.98 → "56.1만원". */
export const krwFromThousand = (thousand: number | null | undefined) => (isFiniteNumber(thousand) ? krw(thousand * 1000) : DASH);

/** 차이. +3.2 / −17.2 / 0.0 ("+-"는 나오지 않는다). */
export function delta(value: number | null | undefined, decimals = 1): string {
  if (!isFiniteNumber(value)) return DASH;
  const text = num(Math.abs(value), decimals);
  if (isZeroText(text)) return text;
  return (value > 0 ? '+' : MINUS) + text;
}

const validHour = (h: unknown): h is number => typeof h === 'number' && Number.isInteger(h) && h >= 1 && h <= 24;

/** 거래시간 H는 (H−1)시–H시 구간이다. 7 → "07h · 06–07시" */
export function tradeHour(hour: number | null | undefined): string {
  return validHour(hour) ? `${pad2(hour)}h · ${pad2(hour - 1)}–${pad2(hour)}시` : DASH;
}

/** 거래시간 번호만: 7 → "07h"(행 카드의 둘째 줄, round 1 D2·D11). */
export function tradeHourNo(hour: number | null | undefined): string {
  return validHour(hour) ? `${pad2(hour)}h` : DASH;
}

/** 하루 안의 시(0–24) 두 자리. 차트 x 눈금 "00 06 12 18 24". */
export function clockHour(hour: number | null | undefined): string {
  return typeof hour === 'number' && Number.isInteger(hour) && hour >= 0 && hour <= 24 ? pad2(hour) : DASH;
}

/** 연속 거래시간 구간. (3, 5) → "03–05h · 02–05시" */
export function tradeHourRange(from: number | null | undefined, to: number | null | undefined): string {
  if (!validHour(from) || !validHour(to) || from > to) return DASH;
  if (from === to) return tradeHour(from);
  return `${pad2(from)}–${pad2(to)}h · ${pad2(from - 1)}–${pad2(to)}시`;
}

/**
 * 문장·요약·첫 줄의 거래시간(명세 2.3절, round 1 D11): 시각 범위만. 거래시간 H는 (H−1)시–H시다. 7 → "06–07시", (3, 5) → "02–05시".
 * "07h" 번호는 카드 메타 줄·스텝퍼·표 칸·칩에만 쓴다(tradeHour).
 */
export function tradeClockRange(from: number | null | undefined, to: number | null | undefined = from): string {
  if (!validHour(from) || !validHour(to) || from > to) return DASH;
  return `${pad2(from - 1)}–${pad2(to)}시`;
}

/** "06:04" — 시계처럼 초는 버린다(06:03:57 → "06:03"). */
export function kstClock(epochMs: number | null | undefined): string {
  const p = isFiniteNumber(epochMs) ? kstParts(epochMs) : null;
  return p ? `${pad2(p.hour)}:${pad2(p.minute)}` : DASH;
}

/** "06:04 KST" */
export function kstTime(epochMs: number | null | undefined): string {
  const clock = kstClock(epochMs);
  return clock === DASH ? DASH : `${clock} KST`;
}

/** "11-26(목)" */
export function kstDate(epochMs: number | null | undefined): string {
  const p = isFiniteNumber(epochMs) ? kstParts(epochMs) : null;
  return p ? `${pad2(p.month)}-${pad2(p.day)}(${WEEKDAYS[p.weekday]})` : DASH;
}

/** 거래일 키 '2026-11-27' → "11-27(금)" */
export function kstDayLabel(dayKey: string | null | undefined): string {
  const start = typeof dayKey === 'string' ? kstDayStart(dayKey) : null;
  return start === null ? DASH : kstDate(start);
}

/** 거래일 키 '2026-11-26' → "11-26"(요일 없이: 입찰 스텝퍼 · 가격 근거 줄). */
export function dayShort(dayKey: string | null | undefined): string {
  const start = typeof dayKey === 'string' ? kstDayStart(dayKey) : null;
  const p = start === null ? null : kstParts(start);
  return p ? `${pad2(p.month)}-${pad2(p.day)}` : DASH;
}

/** 시각을 말로(문장 속): 10:00 → "오전 10시", 13:30 → "오후 1시 30분". */
export function clockPhrase(epochMs: number | null | undefined): string {
  const p = isFiniteNumber(epochMs) ? kstParts(epochMs) : null;
  if (!p) return DASH;
  const hour12 = p.hour % 12 === 0 ? 12 : p.hour % 12;
  return `${p.hour < 12 ? '오전' : '오후'} ${hour12}시${p.minute ? ` ${p.minute}분` : ''}`;
}

/** 조회 나이. 1분 미만 "방금", "12분 전", "1시간 5분 전". 분은 내림. */
export function age(ms: number | null | undefined): string {
  if (!isFiniteNumber(ms)) return DASH;
  if (ms < MINUTE_MS) return '방금';
  return `${duration(ms)} 전`;
}

/** 남은·지난 시간. 분 내림: 3h56m03s → "3시간 56분", 45s → "0분". 음수는 '—'. */
export function duration(ms: number | null | undefined): string {
  if (!isFiniteNumber(ms) || ms < 0) return DASH;
  const minutes = Math.floor(ms / MINUTE_MS);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h === 0) return `${m}분`;
  return m === 0 ? `${h}시간` : `${h}시간 ${m}분`;
}
