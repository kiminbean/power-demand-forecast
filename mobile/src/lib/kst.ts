// 한국 표준시 계산(명세 7.5절). 기기 시간대·Intl 시간대 자료를 쓰지 않는다: 고정 +9시간을 더한 뒤 getUTC*로 읽는다
// (한국은 1988년 뒤 서머타임이 없다). 이 파일만 Date의 날짜·시각 읽기를 한다(7.10절 금지 검사 예외).
export const MINUTE_MS = 60 * 1000;
export const HOUR_MS = 60 * MINUTE_MS;
export const KST_OFFSET_MS = 9 * HOUR_MS;
/** 기기와 서버 시계 차이가 이보다 크면 "서버 시각 기준"을 덧붙인다(7.6절). */
export const SKEW_NOTABLE_MS = 120 * 1000;
/** 이보다 크면 서버 시각이 타당하지 않다고 보고 기기 시계를 쓴다(7.6절). */
export const SKEW_PLAUSIBLE_MS = 12 * HOUR_MS;

export type KstParts = { year: number; month: number; day: number; hour: number; minute: number; second: number; weekday: number };

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const pad2 = (n: number) => String(n).padStart(2, '0');

/** epoch ms → KST 달력 값. weekday는 0=일요일. 유한하지 않거나 범위를 넘으면 null. */
export function kstParts(epochMs: number): KstParts | null {
  if (!isFiniteNumber(epochMs)) return null;
  const d = new Date(epochMs + KST_OFFSET_MS);
  if (!Number.isFinite(d.getTime())) return null;
  return {
    year: d.getUTCFullYear(), month: d.getUTCMonth() + 1, day: d.getUTCDate(),
    hour: d.getUTCHours(), minute: d.getUTCMinutes(), second: d.getUTCSeconds(), weekday: d.getUTCDay(),
  };
}

function utcIfValid(y: number, mo: number, d: number, h = 0, mi = 0, s = 0, ms = 0): number | null {
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 59) return null;
  const t = Date.UTC(y, mo - 1, d, h, mi, s, ms);
  // 2월 30일처럼 넘친 날짜는 Date.UTC가 다음 달로 굴리므로 되읽어서 거른다.
  return new Date(t).getUTCDate() === d ? t : null;
}

const API_TIME = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|[+-]\d{2}:?\d{2})?$/;

/**
 * API 시각 문자열 → epoch ms. 오프셋이 없으면 KST로 본다(market-status.current_time, created_at, realtime-status.timestamp).
 * new Date(문자열)을 쓰지 않는다(엔진마다 오프셋 없는 문자열을 다르게 읽는다). 모양이 틀리면 null.
 */
export function parseApiTime(text: unknown): number | null {
  if (typeof text !== 'string') return null;
  const m = API_TIME.exec(text.trim());
  if (!m) return null;
  const ms = m[7] ? Number(m[7].slice(0, 3).padEnd(3, '0')) : 0;
  const wall = utcIfValid(Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4]), Number(m[5]), m[6] ? Number(m[6]) : 0, ms);
  if (wall === null) return null;
  const zone = m[8];
  if (!zone) return wall - KST_OFFSET_MS;
  if (zone === 'Z') return wall;
  const digits = zone.slice(1).replace(':', '');
  const offset = (Number(digits.slice(0, 2)) * 60 + Number(digits.slice(2))) * MINUTE_MS;
  return zone.startsWith('-') ? wall + offset : wall - offset;
}

/** 거래일 '20261126' 또는 '2026-11-26' → '2026-11-26'. 없는 날짜·다른 모양은 null. */
export function parseDayKey(text: unknown): string | null {
  if (typeof text !== 'string') return null;
  const m = /^(\d{4})-?(\d{2})-?(\d{2})$/.exec(text.trim());
  if (!m || utcIfValid(Number(m[1]), Number(m[2]), Number(m[3])) === null) return null;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

/** 'YYYY-MM-DD'(KST 날짜)의 0시 KST epoch ms. */
export function kstDayStart(dayKey: string): number | null {
  const key = parseDayKey(dayKey);
  if (!key) return null;
  const [y, mo, d] = key.split('-').map(Number);
  return Date.UTC(y, mo - 1, d) - KST_OFFSET_MS;
}

export function kstDateKey(epochMs: number): string | null {
  const p = kstParts(epochMs);
  return p ? `${p.year}-${pad2(p.month)}-${pad2(p.day)}` : null;
}

/** 지금 거래시간 = KST 시 + 1(1–24). 06:03:57 → 7. */
export function tradeHourAt(epochMs: number): number | null {
  const p = kstParts(epochMs);
  return p ? p.hour + 1 : null;
}

/** 차트 지금선의 가로 위치(0–1) = (KST 시 + 분/60)/24. 분 아래는 버린다. */
export function nowLineFraction(epochMs: number): number | null {
  const p = kstParts(epochMs);
  return p ? (p.hour + p.minute / 60) / 24 : null;
}

/** 같은 KST 날짜의 'HH:MM'까지 남은 ms(지났으면 음수). 시각 모양이 틀리면 null. */
export function msUntilKstTime(epochMs: number, hhmm: string): number | null {
  const key = kstDateKey(epochMs);
  const at = key ? kstAt(key, hhmm) : null;
  return at === null ? null : at - epochMs;
}

const DAY_MS = 24 * HOUR_MS;

/** 날짜 키에 n일을 더한다('2026-11-26', −1 → '2026-11-25'). KST는 서머타임이 없어 하루는 늘 24시간이다. */
export function addDays(dayKey: string, days: number): string | null {
  const start = kstDayStart(dayKey);
  return start === null || !Number.isInteger(days) ? null : kstDateKey(start + days * DAY_MS);
}

/** 두 날짜 키의 날 수 차이(to − from). */
export function daysBetween(from: string, to: string): number | null {
  const a = kstDayStart(from);
  const b = kstDayStart(to);
  return a === null || b === null ? null : Math.round((b - a) / DAY_MS);
}

/** 날짜 키(+dayOffset일)의 'HH:MM' KST 시각(epoch ms). 하루전시장 마감 = 거래일 하루 전 10:00(4.2절). 모양이 틀리면 null. */
export function kstAt(dayKey: string, hhmm: string, dayOffset = 0): number | null {
  const m = /^(\d{2}):(\d{2})$/.exec(String(hhmm).trim());
  const day = addDays(dayKey, dayOffset);
  const start = day === null ? null : kstDayStart(day);
  if (!m || start === null || Number(m[1]) > 23 || Number(m[2]) > 59) return null;
  return start + Number(m[1]) * HOUR_MS + Number(m[2]) * MINUTE_MS;
}

/** 실시간시장 다음 구간(4.2절): KST 지금 뒤 첫 interval분 배수(자정 기준). 06:03:57, 15 → 06:15. 간격이 틀리면 null. */
export function nextIntervalAt(epochMs: number, minutes: number): number | null {
  const key = kstDateKey(epochMs);
  const start = key === null ? null : kstDayStart(key);
  if (start === null || !Number.isInteger(minutes) || minutes <= 0 || minutes > 24 * 60) return null;
  const step = minutes * MINUTE_MS;
  return start + (Math.floor((epochMs - start) / step) + 1) * step;
}

export type ClockSkew = { skewMs: number; plausible: boolean; notable: boolean };

/** 서버 − 기기. 12시간을 넘으면 서버 시각을 믿지 않는다. */
export function clockSkew(serverEpochMs: number, deviceEpochMs: number): ClockSkew | null {
  if (!isFiniteNumber(serverEpochMs) || !isFiniteNumber(deviceEpochMs)) return null;
  const skewMs = serverEpochMs - deviceEpochMs;
  const plausible = Math.abs(skewMs) <= SKEW_PLAUSIBLE_MS;
  return { skewMs, plausible, notable: plausible && Math.abs(skewMs) > SKEW_NOTABLE_MS };
}

/** 앱의 "지금" = 기기 시계 + 타당한 시계 차이. */
export function kstNow(skew: ClockSkew | null = null, deviceNow: number = Date.now()): number {
  return skew && skew.plausible ? deviceNow + skew.skewMs : deviceNow;
}
