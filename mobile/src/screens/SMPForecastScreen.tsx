/**
 * 오늘(명세 4.1절, 경로 / · 라우트 키 smp). 이 파일이 오늘 화면의 자료 입구다: 허용 목록 GET 여섯 개(하루전 확정 가격 · 기준선 예측 ·
 * 모델 정보 · KPI · 수급 · 시장 상태)를 자료 저장소(lib/resource)로 받고, 하루전 가격은 utils/smpPresentation의 isAvailableDayAhead·
 * describeDayAhead로 판정과 KPX 원문 표기를 만든 뒤 순수 함수 features/today/model.ts에 넘긴다. 카드는 배치만 한다.
 * 산출 방식·출처 시트와 ≥ 1280 패널은 useTodayView()로 같은 저장소를 엿본다(새 요청 없음, App.tsx가 시트로 등록한다).
 * 기존 화면 경로(App.tsx의 `case 'smp': return <SMPForecastScreen`)와 기본 export(필수 prop 없음)는 그대로다.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import type { DayAheadSMP } from '../services/api';
import { describeDayAhead, isAvailableDayAhead } from '../utils/smpPresentation';
import { useClock, useNav, useShellData } from '../app/Shell';
import type { RefreshResult } from '../app/Shell';
import { ROUTES, rootState } from '../app/routes';
import type { MethodCard, SourceRow } from '../app/routes';
import { isVisible, watchVisible } from '../app/head';
import { createClient, ENDPOINTS } from '../data/client';
import { kstNow, kstParts, parseApiTime } from '../lib/kst';
import { CHIPS } from '../lib/provenance';
import type { ForecastKind } from '../lib/provenance';
import { appResources, useRefreshWhileVisible, useResource } from '../lib/resource';
import type { Resource } from '../lib/resource';
import { TOUR_BANNER } from '../content/strings';
import { nextBoundaryCheck, todayView } from '../features/today/model';
import type { Fetched, TodayCardId } from '../features/today/model';
import { AccuracyCard } from '../features/today/AccuracyCard';
import { ForecastCard } from '../features/today/ForecastCard';
import { PowerCard } from '../features/today/PowerCard';
import { PriceCard } from '../features/today/PriceCard';
import { AllFailedNotice, TermRow, TodayPanelBody } from '../features/today/TodayParts';
import { LiveFacts, SourcesTable } from '../features/guide/sheets';
import type { SourceRowItem } from '../features/guide/sheets';
import { NowPill, ProvenanceChip } from '../ui/Chips';
import { AnswerLine, Screen } from '../ui/Screen';
import { TourBanner } from '../ui/Tour';
import { SPACE } from '../ui/tokens';

/** 기준선 계단 차트의 접근 이름 머리(요약 문장이 뒤에 붙는다). 예측 종류마다 이름이 다르다. */
const FORECAST_CHART_LABEL: Readonly<Record<ForecastKind, string>> = {
  baseline: '기준선 예측 SMP 24시간',
  model: '학습 모델 예측 SMP 24시간',
  fallback: 'API 대체값 24시간 (예측 SMP 아님)',
};
const chartLabel = (kind: ForecastKind) => FORECAST_CHART_LABEL[kind];

// 요청마다 자료 저장소가 만든 10초 중단 신호를 묶은 클라이언트로 부른다.
const api = (signal: AbortSignal) => createClient({ signal });
const LOAD = {
  dayAhead: (signal: AbortSignal) => api(signal).getDayAheadSMP(),
  forecast: (signal: AbortSignal) => api(signal).getSMPForecast(),
  modelInfo: (signal: AbortSignal) => api(signal).getModelInfo(),
  kpis: (signal: AbortSignal) => api(signal).getDashboardKPIs(),
  powerSupply: (signal: AbortSignal) => api(signal).getPowerSupply(),
  market: (signal: AbortSignal) => api(signal).getMarketStatus(),
};
const VISIBILITY = { isVisible, watchVisible };

/**
 * 오늘 화면의 자료와 view model. fetching이면 처음 마운트에 받고(lib/resource 규칙) 다시 받을 손잡이를 준다. 아니면 엿보기만 한다.
 * 저장소의 조회 시각은 기기 시계라서 서버 시계 차이를 더해 앱 시각으로 바꾼 뒤 넘긴다(나이는 어느 쪽이든 같다).
 */
export function useTodayView(fetching = false) {
  const dayAhead = useResource(ENDPOINTS.dayAhead, fetching ? LOAD.dayAhead : undefined);
  const forecast = useResource(ENDPOINTS.forecast, fetching ? LOAD.forecast : undefined);
  const modelInfo = useResource(ENDPOINTS.modelInfo, fetching ? LOAD.modelInfo : undefined);
  const kpis = useResource(ENDPOINTS.kpis, fetching ? LOAD.kpis : undefined);
  const powerSupply = useResource(ENDPOINTS.powerSupply, fetching ? LOAD.powerSupply : undefined);
  const market = useResource(ENDPOINTS.marketStatus, fetching ? LOAD.market : undefined);
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
  // isAvailableDayAhead는 런타임 타입 가드다(status·market_type·값·거래일·거래시간을 다 본다). 파싱 결과는 모양이 같고
  // 문자열 리터럴 타입만 넓어서 옛 타입으로 넘긴다.
  const dayAheadSmp = dayAhead.data as DayAheadSMP | null;
  const priceAvailable = isAvailableDayAhead(dayAheadSmp);
  const kpxText = describeDayAhead(dayAheadSmp);
  const hour = state.key === 'smp' ? state.hour : null;
  const view = useMemo(() => {
    const app = <T,>(r: Resource<T>): Fetched<T> => ({ data: r.data, status: r.status, error: r.error, fetchedAt: r.fetchedAt === null ? null : r.fetchedAt + skewMs });
    return todayView({
      now: clock.now, dayAhead: app(dayAhead), priceAvailable, kpxText, forecast: app(forecast), modelInfo: app(modelInfo), kpis: app(kpis),
      powerSupply: app(powerSupply), market: app(market), selectedHour: hour,
    });
  }, [clock.now, skewMs, dayAhead, priceAvailable, kpxText, forecast, modelInfo, kpis, powerSupply, market, hour]);
  return { view, res: { dayAhead, forecast, modelInfo, kpis, powerSupply, market }, skew: clock.skew };
}

// 오늘 화면의 둘러보기 배너는 첫 방문에만(4.4절). 저장소를 못 쓰면 그 방문 동안만 닫힌 것을 기억한다.
const TOUR_SEEN_KEY = 'pd-tour-seen';
let tourSeenThisVisit = false;
type StorageLike = { getItem(k: string): string | null; setItem(k: string, v: string): void };
const storage = (): StorageLike | null => {
  try {
    return (globalThis as { localStorage?: StorageLike }).localStorage ?? null;
  } catch {
    return null;
  }
};
function tourSeen(): boolean {
  if (tourSeenThisVisit) return true;
  try {
    const s = storage();
    return s ? s.getItem(TOUR_SEEN_KEY) === '1' : false;
  } catch {
    return false;
  }
}
function markTourSeen(): void {
  tourSeenThisVisit = true;
  try {
    const s = storage();
    if (s) s.setItem(TOUR_SEEN_KEY, '1');
  } catch {
    // 저장하지 못해도 이번 방문 동안은 닫힌 채로 둔다.
  }
}

export default function SMPForecastScreen() {
  const { view, res, skew } = useTodayView(true);
  const nav = useNav();
  const clock = useClock();
  const [tourHidden, setTourHidden] = useState(tourSeen);
  const { dayAhead, forecast, modelInfo, kpis, powerSupply, market } = res;

  const reloadPrice = dayAhead.reload;
  const reloadForecast = forecast.reload;
  const reloadAll = useCallback(() => Promise.all([dayAhead.reload(), forecast.reload(), kpis.reload(), modelInfo.reload(), powerSupply.reload(), market.reload()]),
    [dayAhead.reload, forecast.reload, kpis.reload, modelInfo.reload, powerSupply.reload, market.reload]);
  // 헤더 새로고침: 가격·예측·전력이 모두 새로 오면 "06:09 KST 기준으로 새로 불러왔습니다", 아니면 보이는 값의 시각과 함께 실패를 알린다.
  const refresh = useCallback(async (): Promise<RefreshResult> => {
    const [d, f, k] = await reloadAll();
    const offset = skew && skew.plausible ? skew.skewMs : 0;
    const shown = [ENDPOINTS.dayAhead, ENDPOINTS.kpis].map(key => appResources.get(key).fetchedAt).filter((t): t is number => t !== null);
    return d && f && k ? { ok: true, at: kstNow(skew) } : { ok: false, shownAt: shown.length ? Math.min(...shown) + offset : null };
  }, [reloadAll, skew]);
  useShellData(useMemo(() => ({ refresh, fetchedAt: view.fetchedAt }), [refresh, view.fetchedAt]));

  // 화면이 보이는 동안 5분마다, 숨었다 보이면 5분 넘은 것만(6.3절).
  const oldest = useCallback(() => {
    const list = [dayAhead.fetchedAt, kpis.fetchedAt].filter((t): t is number => t !== null);
    return list.length ? Math.min(...list) : null;
  }, [dayAhead.fetchedAt, kpis.fetchedAt]);
  useRefreshWhileVisible(() => { reloadAll(); }, oldest, VISIBILITY);

  // 거래시간 경계(4.1절): 정시 뒤 HH:00:15와 HH:02:00에 가격을 다시 받는다(자정에는 예측도). 보이지 않으면 건너뛴다.
  useEffect(() => {
    const offset = skew && skew.plausible ? skew.skewMs : 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const plan = () => {
      const now = Date.now() + offset;
      const at = nextBoundaryCheck(now);
      timer = setTimeout(() => {
        if (isVisible()) {
          reloadPrice();
          const p = kstParts(at);
          if (p && p.hour === 0) reloadForecast();
        }
        plan();
      }, Math.max(0, at - now));
    };
    plan();
    return () => {
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [skew, reloadPrice, reloadForecast]);
  // 다른 탭에 있다가 정시를 넘겨 돌아왔으면 한 번 다시 받는다.
  useEffect(() => {
    if (view.recheckPrice) reloadPrice();
  }, [view.recheckPrice, reloadPrice]);

  const selectHour = useCallback((hour: number) => nav.replace({ ...nav.state, hour }), [nav]);
  const toNow = useCallback(() => nav.replace({ ...nav.state, hour: null }), [nav]);
  const openMethod = (card: TodayCardId) => () => nav.openSheet({ kind: 'method', card });
  const openSources = (row: SourceRow) => () => nav.openSheet({ kind: 'sources', row });
  const showTour = nav.tabs.includes('guide') && !tourHidden;
  const closeTour = () => {
    markTourSeen();
    setTourHidden(true);
  };
  const startTour = () => {
    closeTour();
    nav.push({ ...rootState('guide'), tour: 1 });
  };

  const pill = (
    <View style={styles.pills}>
      <NowPill now={clock.now} fetchedAt={view.fetchedAt} skewNotable={clock.skewNotable} />
      {skew && !skew.plausible ? <ProvenanceChip chip={CHIPS.checkTime} /> : null}
    </View>
  );
  const failedAll = view.allFailed;
  return (
    <Screen title={ROUTES.smp.heading} pill={pill} answer={<AnswerLine>{view.answer}</AnswerLine>} testID="today-screen">
      {failedAll ? <AllFailedNotice text={failedAll.text} onRetry={() => { reloadAll(); }} hero={!failedAll.showCards} /> : null}
      {failedAll && !failedAll.showCards ? null : (
        <>
          <PriceCard view={view.price} onMethod={openMethod('today-price')} onSources={openSources('day-ahead')} onRetry={() => { dayAhead.reload(); }} />
          <PowerCard view={view.power} onMethod={openMethod('today-power')} onSources={openSources('demand')}
            onRetry={() => { kpis.reload(); powerSupply.reload(); }} />
          {showTour ? <TourBanner title={TOUR_BANNER.title} body={TOUR_BANNER.body} onStart={startTour} onDismiss={closeTour} /> : null}
          <ForecastCard view={view.forecast} chartLabel={chartLabel} now={clock.now} onSelectHour={selectHour} onToNow={toNow}
            onMethod={openMethod('today-forecast')} onSources={openSources('forecast')} onRetry={() => { forecast.reload(); }} />
          <AccuracyCard view={view.accuracy} onMethod={openMethod('today-accuracy')} onSources={openSources('accuracy')}
            onRetry={() => { forecast.reload(); modelInfo.reload(); }} />
        </>
      )}
      <TermRow onOpen={id => nav.openSheet({ kind: 'term', id })} />
    </Screen>
  );
}

const isTodayCard = (card: MethodCard): card is TodayCardId => card.startsWith('today-');

/** 산출 방식 시트의 "이번 화면의 값"(4.1·4.4절: KPX 원문 표기·25.9 % 비율·MW 계산 예비율 차이·모델 원문). */
export function TodayMethodFacts({ card }: { card: MethodCard }) {
  const { view } = useTodayView(false);
  if (!isTodayCard(card)) return null;
  const facts = view.facts[card];
  return <LiveFacts lines={facts.lines} link={facts.link} />;
}

/** 출처와 기준 시각 시트(4.4절): 오늘 행 + 다른 화면이 넘기는 행(입찰 계산 F8, 응답 상태 F10). */
export function TodaySourcesSheet({ row, extra = [] }: { row: SourceRow | null; extra?: readonly SourceRowItem[] }) {
  const { view } = useTodayView(false);
  return <SourcesTable rows={[...view.sources, ...extra]} highlight={row} />;
}

/** ≥ 1280 "오늘 화면 읽는 법" 패널 본문(4.6절). */
export function TodayPanel() {
  const { view } = useTodayView(false);
  const nav = useNav();
  return <TodayPanelBody confirmed={view.legend.confirmed} forecast={view.legend.forecast} onTerm={id => nav.openSheet({ kind: 'term', id })} />;
}

const styles = StyleSheet.create({
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
});
