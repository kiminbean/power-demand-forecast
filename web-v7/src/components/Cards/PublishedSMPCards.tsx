import type { ReactNode } from 'react';
import type { CurrentSMP, DayAheadSMP } from '../../types';

const dateStatusLabels: Record<CurrentSMP['data_date_status'], string> = {
  current_date: '당일 자료',
  historical: '과거 자료',
  future: '미래 날짜 자료',
  unknown: '날짜 확인 불가',
  no_data: '자료 없음',
};

// 'kpi'는 eXeco 대시보드 KPI 행(현재 수요·계통 주파수 등)과 같은 레이아웃을 쓴다.
type CardVariant = 'card' | 'kpi';

const KPI_BOX = 'flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5 min-w-0';
const KPI_TITLE = 'text-2xl font-medium text-black tracking-[-0.8px] text-center';
const KPI_VALUE = 'text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]';
const KPI_UNIT = 'text-2xl font-bold text-black leading-9 tracking-[-0.8px]';
const KPI_EMPTY = 'text-2xl font-bold text-[#9a9a9a] leading-[60px] tracking-[-0.8px]';
const KPI_BADGE = 'bg-white rounded-lg px-3.5 py-1';
const KPI_BADGE_TEXT = 'text-xl text-[#272727] tracking-[-0.72px]';
// 거래 정보는 한 줄로 자르고(title에 전체), 출처는 숨기지 않도록 단어 단위로 줄바꿈한다.
const KPI_META = 'w-full text-center text-sm text-gray-700 leading-5';

function KpiMeta({ lines, children }: { lines: [string, string]; children?: ReactNode }) {
  return (
    <div className="w-full min-w-0 flex flex-col gap-0.5">
      <p className={`${KPI_META} truncate`} title={lines[0]}>{lines[0]}</p>
      <p className={`${KPI_META} break-keep`}>{lines[1]} {children}</p>
    </div>
  );
}

function formatHour(hour: number | null | undefined) {
  return hour != null ? `${String(hour).padStart(2, '0')}h` : '확인 불가';
}

function KpiValue({ value }: { value: number | null }) {
  if (value === null) return <span className={KPI_EMPTY}>SMP 데이터 없음</span>;
  return (
    <div className="flex items-end gap-1">
      <span className={KPI_VALUE}>{value.toFixed(1)}</span>
      <span className={KPI_UNIT}>원/kWh</span>
    </div>
  );
}

export function PublishedSMPCard({ data, variant = 'card' }: { data: CurrentSMP | null; variant?: CardVariant }) {
  const price = data?.latest_smp?.price ?? data?.current_smp;
  const hasValue = data?.data_date_status !== 'no_data' && typeof price === 'number' && Number.isFinite(price);
  const tradeHour = data?.latest_smp?.trade_hour;
  const region = data?.region?.toLowerCase() === 'jeju' ? '제주' : data?.region || '지역 미확인';
  const dateLabel = data ? dateStatusLabels[data.data_date_status] ?? '날짜 확인 불가' : null;
  const tradeDay = data?.latest_smp?.trade_day || data?.trade_day || '확인 불가';
  const market = data?.latest_smp?.market === 'real_time' ? '실시간시장(RT)' : '시장 미확인';
  const source = data?.latest_smp?.data_source || data?.data_source || '확인 불가';
  const comparison = hasValue && data?.comparison
    ? `일평균 대비 ${data.comparison.daily_change_pct >= 0 ? '+' : ''}${data.comparison.daily_change_pct.toFixed(1)}%`
    : null;

  if (variant === 'kpi') {
    const meta: [string, string] = [
      `거래일 ${tradeDay} · 거래시간 ${formatHour(tradeHour)} · ${market}`,
      `출처: ${source} · 실시간 여부 확인 불가`,
    ];
    return (
      <div className={KPI_BOX} data-testid="published-smp">
        <span className={KPI_TITLE}>공표 SMP ({region})</span>
        <div className="flex flex-wrap items-center justify-center gap-3.5 px-3.5">
          <KpiValue value={hasValue ? price : null} />
          {dateLabel && (
            <div className={KPI_BADGE}>
              <span className={KPI_BADGE_TEXT}>{comparison ? `${dateLabel} · ${comparison}` : dateLabel}</span>
            </div>
          )}
        </div>
        <KpiMeta lines={meta} />
      </div>
    );
  }

  return (
    <div className="card" data-testid="published-smp">
      <p className="text-sm text-text-muted mb-1">공표 SMP ({region})</p>
      <div className="text-2xl font-bold text-smp">{hasValue ? `${price.toFixed(1)} 원/kWh` : 'SMP 데이터 없음'}</div>
      {data && <p className="text-xs text-text-muted mt-1">{dateLabel} · 거래일 {tradeDay} · 거래시간 {formatHour(tradeHour)}</p>}
      <p className="text-xs text-text-muted mt-1">{market} · 출처: {source} · 실시간 여부 확인 불가</p>
      {comparison && <p className="text-xs text-text-muted mt-1">{comparison}</p>}
    </div>
  );
}

export function DayAheadSMPCard({ data, variant = 'card' }: { data: DayAheadSMP | null; variant?: CardVariant }) {
  const hasValue = data?.status === 'available' && typeof data.current_smp === 'number' && Number.isFinite(data.current_smp);
  const link = data?.source_url && <a href={data.source_url} target="_blank" rel="noreferrer">원문</a>;

  if (variant === 'kpi') {
    const meta: [string, string] = [
      `거래일 ${data?.trade_day || '확인 불가'} · 거래시간 ${formatHour(data?.trade_hour)}`,
      `출처: ${data?.data_source || '확인 불가'}`,
    ];
    return (
      <div className={KPI_BOX} data-testid="day-ahead-smp">
        <span className={KPI_TITLE}>제주 하루전시장 SMP (Day-ahead)</span>
        <div className="flex flex-wrap items-center justify-center gap-3.5 px-3.5">
          <KpiValue value={hasValue ? data!.current_smp : null} />
          <div className={KPI_BADGE}>
            <span className={KPI_BADGE_TEXT}>하루전시장(DA)</span>
          </div>
        </div>
        <KpiMeta lines={meta}>{link}</KpiMeta>
      </div>
    );
  }

  return (
    <div className="card" data-testid="day-ahead-smp">
      <p className="text-sm text-text-muted mb-1">제주 하루전시장 SMP (Day-ahead)</p>
      <div className="text-2xl font-bold text-smp">{hasValue ? `${data.current_smp!.toFixed(1)} 원/kWh` : 'SMP 데이터 없음'}</div>
      <p className="text-xs text-text-muted mt-1">거래일 {data?.trade_day || '확인 불가'} · 거래시간 {formatHour(data?.trade_hour)}</p>
      <p className="text-xs text-text-muted mt-1">출처: {data?.data_source || '확인 불가'} {link}</p>
    </div>
  );
}
