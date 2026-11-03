import type { CurrentSMP, DayAheadSMP } from '../services/api';

export function formatNullableNumber(value: number | null | undefined, decimals = 1): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(decimals) : '—';
}

export function publishedSmpPrice(data: Partial<CurrentSMP> | null | undefined): number | null {
  const price = data?.latest_smp?.price ?? data?.current_smp;
  return typeof price === 'number' && Number.isFinite(price) ? price : null;
}

export function describeCurrentSmp(data: Partial<CurrentSMP> | null | undefined): string {
  if (publishedSmpPrice(data) === null) return '공표 SMP 데이터 없음';
  const status = data?.data_date_status === 'historical' ? '과거 거래일' : '거래일 기준';
  const latest = data?.latest_smp;
  const hour = typeof latest?.trade_hour === 'number' && Number.isInteger(latest.trade_hour) && latest.trade_hour >= 1 && latest.trade_hour <= 24 ? `${latest.trade_hour}h` : '시간 미상';
  return `공표 SMP (${status}: ${latest?.trade_day || data?.trade_day || '날짜 미상'} ${hour}; 시장 미상; 출처: ${latest?.data_source || data?.data_source || '미상'})`;
}

function normalizedTradeDay(value: string | null | undefined): string | null {
  if (typeof value !== 'string') return null;
  const compact = /^(\d{4})(\d{2})(\d{2})$/.exec(value);
  const day = compact ? `${compact[1]}-${compact[2]}-${compact[3]}` : value;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) return null;
  const date = new Date(`${day}T00:00:00Z`);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === day ? day : null;
}

export function isAvailableDayAhead(data: DayAheadSMP | null | undefined): data is DayAheadSMP & { current_smp: number; trade_day: string; trade_hour: number } {
  return data?.status === 'available' && data.market_type === 'day_ahead' &&
    typeof data.current_smp === 'number' && Number.isFinite(data.current_smp) &&
    normalizedTradeDay(data.trade_day) !== null &&
    typeof data.trade_hour === 'number' && Number.isInteger(data.trade_hour) &&
    data.trade_hour >= 1 && data.trade_hour <= 24;
}

export function describeDayAhead(data: DayAheadSMP | null | undefined): string {
  if (!isAvailableDayAhead(data)) return '제주 하루전시장 공표 SMP 데이터 없음';
  return `제주 하루전시장 공표 SMP (${normalizedTradeDay(data.trade_day)} ${String(data.trade_hour).padStart(2, '0')}h; 출처: ${data.data_source || '미상'})`;
}
