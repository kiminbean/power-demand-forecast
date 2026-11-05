import type { CurrentSMP, DayAheadSMP } from '../../types';

const dateStatusLabels: Record<CurrentSMP['data_date_status'], string> = {
  current_date: '당일 자료',
  historical: '과거 자료',
  future: '미래 날짜 자료',
  unknown: '날짜 확인 불가',
  no_data: '자료 없음',
};

export function PublishedSMPCard({ data }: { data: CurrentSMP | null }) {
  const price = data?.latest_smp?.price ?? data?.current_smp;
  const hasValue = data?.data_date_status !== 'no_data' && typeof price === 'number' && Number.isFinite(price);
  const tradeHour = data?.latest_smp?.trade_hour;
  const region = data?.region?.toLowerCase() === 'jeju' ? '제주' : data?.region || '지역 미확인';
  return (
    <div className="card" data-testid="published-smp">
      <p className="text-sm text-text-muted mb-1">공표 SMP ({region})</p>
      <div className="text-2xl font-bold text-smp">{hasValue ? `${price.toFixed(1)} 원/kWh` : 'SMP 데이터 없음'}</div>
      {data && <p className="text-xs text-text-muted mt-1">{dateStatusLabels[data.data_date_status] ?? '날짜 확인 불가'} · 거래일 {data.latest_smp?.trade_day || data.trade_day || '확인 불가'} · 거래시간 {tradeHour != null ? `${String(tradeHour).padStart(2, '0')}h` : '확인 불가'}</p>}
      <p className="text-xs text-text-muted mt-1">{data?.latest_smp?.market === 'real_time' ? '실시간시장(RT)' : '시장 미확인'} · 출처: {data?.latest_smp?.data_source || data?.data_source || '확인 불가'} · 실시간 여부 확인 불가</p>
      {hasValue && data?.comparison && <p className="text-xs text-text-muted mt-1">일평균 대비 {data.comparison.daily_change_pct >= 0 ? '+' : ''}{data.comparison.daily_change_pct.toFixed(1)}%</p>}
    </div>
  );
}

export function DayAheadSMPCard({ data }: { data: DayAheadSMP | null }) {
  const hasValue = data?.status === 'available' && typeof data.current_smp === 'number' && Number.isFinite(data.current_smp);
  return (
    <div className="card" data-testid="day-ahead-smp">
      <p className="text-sm text-text-muted mb-1">제주 하루전시장 SMP (Day-ahead)</p>
      <div className="text-2xl font-bold text-smp">{hasValue ? `${data.current_smp!.toFixed(1)} 원/kWh` : 'SMP 데이터 없음'}</div>
      <p className="text-xs text-text-muted mt-1">거래일 {data?.trade_day || '확인 불가'} · 거래시간 {data?.trade_hour != null ? `${String(data.trade_hour).padStart(2, '0')}h` : '확인 불가'}</p>
      <p className="text-xs text-text-muted mt-1">출처: {data?.data_source || '확인 불가'} {data?.source_url && <a href={data.source_url} target="_blank" rel="noreferrer">원문</a>}</p>
    </div>
  );
}
