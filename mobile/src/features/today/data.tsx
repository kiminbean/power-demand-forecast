/**
 * 오늘 화면의 자료와 view model(명세 4.1·6.3절). 오늘 화면·둘러보기·산출 방식과 출처 시트·≥ 1280 패널이 이 훅 하나로 같은 저장소를 본다.
 * 자료 입구(허용 목록 GET 여섯 개, 하루전 가격 판정, 차트 접근 이름)는 화면 파일 src/screens/SMPForecastScreen.tsx의 TODAY_ENTRY이고
 * App.tsx가 TodayEntryProvider로 넘긴다. 그래서 features는 src/screens를 가져오지 않는다(round 1 R8, D18).
 * 입구가 화면 파일에 남는 까닭: 잠긴 기존 테스트(smp-truthfulness)가 그 파일이 getDayAheadSMP()·isAvailableDayAhead(dayAheadSmp)·
 * describeDayAhead(dayAheadSmp)·"예측 SMP"를 쓰는지 본다(명세 10절). 공급자가 없으면 조용히 비우지 않고 바로 실패한다.
 */
import { createContext, useContext, useEffect, useMemo } from 'react';
import type React from 'react';
import { todayView } from './model';
import type { Fetched, SeenPrice } from './model';
import { useClock, useNav } from '../../app/Shell';
import { ENDPOINTS } from '../../data/client';
import type { DashboardKpis, DayAheadSmp, MarketStatus, ModelInfo, PowerSupply, SmpForecast } from '../../data/types';
import { parseApiTime } from '../../lib/kst';
import type { ForecastKind } from '../../lib/provenance';
import { useResource } from '../../lib/resource';
import type { Loader, Resource } from '../../lib/resource';

export type TodayLoaders = Readonly<{
  dayAhead: Loader<DayAheadSmp>;
  forecast: Loader<SmpForecast>;
  modelInfo: Loader<ModelInfo>;
  kpis: Loader<DashboardKpis>;
  powerSupply: Loader<PowerSupply>;
  market: Loader<MarketStatus>;
}>;
/** 하루전 가격의 판정(utils/smpPresentation의 isAvailableDayAhead)과 KPX 원문 표기(describeDayAhead). */
export type DayAheadJudgment = Readonly<{ available: boolean; kpxText: string }>;
export type TodayEntry = Readonly<{
  load: TodayLoaders;
  judgeDayAhead: (quote: DayAheadSmp | null) => DayAheadJudgment;
  /** 기준선 계단 차트의 접근 이름 머리(요약 문장이 뒤에 붙는다). 예측 종류마다 다르다. */
  chartLabel: (kind: ForecastKind) => string;
}>;

const EntryContext = createContext<TodayEntry | null>(null);

/**
 * 이번 방문에서 받은 하루전 확정 가격(round 1 D10): 서버는 지금 거래시간 값만 주므로 머무는 동안 받은 값을 거래시간마다 기억해 차트에
 * 점으로 남긴다. 메모리에만 두고(새 요청·저장소 없음) 새로고침하면 사라진다. 날짜가 바뀌어도 view model이 오늘 것만 그린다.
 */
const SEEN = new Map<string, SeenPrice>();
const SEEN_MAX = 48;
function rememberPrice(data: DayAheadSmp | null, fetchedAt: number | null, available: boolean): readonly SeenPrice[] {
  if (data && fetchedAt !== null && available && data.trade_day !== null && data.trade_hour !== null) {
    const key = `${data.trade_day}|${data.trade_hour}`;
    SEEN.delete(key);
    SEEN.set(key, { data, fetchedAt, available });
    while (SEEN.size > SEEN_MAX) {
      const oldest = SEEN.keys().next();
      if (oldest.done) break;
      SEEN.delete(oldest.value);
    }
  }
  return Array.from(SEEN.values());
}

/** App.tsx가 화면 파일의 TODAY_ENTRY로 셸 전체를 감싼다(시트·패널·둘러보기도 셸 안에서 그려진다). */
export function TodayEntryProvider({ value, children }: { value: TodayEntry; children: React.ReactNode }) {
  return <EntryContext.Provider value={value}>{children}</EntryContext.Provider>;
}

export function useTodayEntry(): TodayEntry {
  const entry = useContext(EntryContext);
  if (!entry) throw new Error('TodayEntryProvider missing: App.tsx must pass TODAY_ENTRY from the Today screen file');
  return entry;
}

/**
 * 출처와 기준 시각 표의 오늘 행이 쓰는 자료(round 1 D15): 이 표를 여는 화면(알아보기 › 데이터 출처와 연결 상태, 출처 시트)은 이번 방문에
 * 아직 받지 않았거나 실패했거나 5분 지난 것만 같은 저장소로 받는다(lib/resource 규칙, 같은 키는 한 번에 하나). 모델 정보는 표에 없어
 * 받지 않는다. 값은 useTodayView(false)가 엿본다.
 */
export function useSourceRowsLoad(): void {
  const { load } = useTodayEntry();
  useResource(ENDPOINTS.dayAhead, load.dayAhead);
  useResource(ENDPOINTS.forecast, load.forecast);
  useResource(ENDPOINTS.kpis, load.kpis);
  useResource(ENDPOINTS.powerSupply, load.powerSupply);
  useResource(ENDPOINTS.marketStatus, load.market);
}

/**
 * 오늘 화면의 자료와 view model. fetching이면 처음 마운트에 받고(lib/resource 규칙) 다시 받을 손잡이를 준다. 아니면 엿보기만 한다.
 * 저장소의 조회 시각은 기기 시계라서 서버 시계 차이를 더해 앱 시각으로 바꾼 뒤 넘긴다(나이는 어느 쪽이든 같다).
 */
export function useTodayView(fetching = false) {
  const entry = useTodayEntry();
  const { load } = entry;
  const dayAhead = useResource(ENDPOINTS.dayAhead, fetching ? load.dayAhead : undefined);
  const forecast = useResource(ENDPOINTS.forecast, fetching ? load.forecast : undefined);
  const modelInfo = useResource(ENDPOINTS.modelInfo, fetching ? load.modelInfo : undefined);
  const kpis = useResource(ENDPOINTS.kpis, fetching ? load.kpis : undefined);
  const powerSupply = useResource(ENDPOINTS.powerSupply, fetching ? load.powerSupply : undefined);
  const market = useResource(ENDPOINTS.marketStatus, fetching ? load.market : undefined);
  const clock = useClock();
  const { state } = useNav();
  const { reportServerTime } = clock;
  // 시장 상태의 서버 시각으로 기기 시계 차이를 맞춘다(7.5·7.6절). 받은 기기 시각과 짝을 짓는다.
  useEffect(() => {
    if (!fetching || !market.data || market.fetchedAt === null) return;
    const server = parseApiTime(market.data.current_time);
    if (server !== null) reportServerTime(server, market.fetchedAt);
  }, [fetching, market.data, market.fetchedAt, reportServerTime]);
  const skewMs = clock.skew && clock.skew.plausible ? clock.skew.skewMs : 0;
  const judged = useMemo(() => entry.judgeDayAhead(dayAhead.data), [entry, dayAhead.data]);
  const seen = useMemo(() => rememberPrice(dayAhead.data, dayAhead.fetchedAt, judged.available), [dayAhead.data, dayAhead.fetchedAt, judged]);
  const hour = state.key === 'smp' ? state.hour : null;
  const view = useMemo(() => {
    const app = <T,>(r: Resource<T>): Fetched<T> => ({ data: r.data, status: r.status, error: r.error, fetchedAt: r.fetchedAt === null ? null : r.fetchedAt + skewMs });
    return todayView({
      now: clock.now, dayAhead: app(dayAhead), priceAvailable: judged.available, kpxText: judged.kpxText, forecast: app(forecast),
      modelInfo: app(modelInfo), kpis: app(kpis), powerSupply: app(powerSupply), market: app(market), selectedHour: hour,
      seen: seen.map(x => ({ data: x.data, available: x.available, fetchedAt: x.fetchedAt + skewMs })),
    });
  }, [clock.now, skewMs, dayAhead, judged, forecast, modelInfo, kpis, powerSupply, market, hour, seen]);
  return { view, res: { dayAhead, forecast, modelInfo, kpis, powerSupply, market }, skew: clock.skew, chartLabel: entry.chartLabel };
}
