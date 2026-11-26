/**
 * 오늘(명세 4.1절, 경로 / · 라우트 키 smp). 이 파일이 오늘 화면의 자료 입구다: TODAY_ENTRY가 허용 목록 GET 여섯 개(하루전 확정 가격 ·
 * 기준선 예측 · 모델 정보 · KPI · 수급 · 시장 상태)를 부르는 방법, 하루전 가격 판정(utils/smpPresentation의 isAvailableDayAhead·
 * describeDayAhead), 기준선 차트의 접근 이름을 정한다. App.tsx가 이것을 features/today/data의 TodayEntryProvider로 넘기고, 오늘 화면·
 * 둘러보기·시트·패널은 useTodayView로 같은 저장소를 본다(features는 이 파일을 가져오지 않는다, D18). 카드는 배치만 한다.
 * 기존 화면 경로(App.tsx의 `case 'smp': return <SMPForecastScreen`)와 기본 export(필수 prop 없음)는 그대로다.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import type { DayAheadSMP } from '../services/api';
import { describeDayAhead, isAvailableDayAhead } from '../utils/smpPresentation';
import { useClock, useNav, useShellData } from '../app/Shell';
import type { RefreshResult } from '../app/Shell';
import { ROUTES, rootState } from '../app/routes';
import type { SourceRow } from '../app/routes';
import { isVisible, watchVisible } from '../app/head';
import { createClient, ENDPOINTS } from '../data/client';
import { kstNow, kstParts } from '../lib/kst';
import { CHIPS } from '../lib/provenance';
import type { ForecastKind } from '../lib/provenance';
import { appResources, useRefreshWhileVisible } from '../lib/resource';
import { TOUR_BANNER } from '../content/strings';
import { useTodayView } from '../features/today/data';
import type { TodayEntry } from '../features/today/data';
import { nextBoundaryCheck } from '../features/today/model';
import type { TodayCardId } from '../features/today/model';
import { markTourSeen, tourSeen } from '../features/today/tourSeen';
import { AccuracyCard } from '../features/today/AccuracyCard';
import { ForecastCard } from '../features/today/ForecastCard';
import { PowerCard } from '../features/today/PowerCard';
import { PriceCard } from '../features/today/PriceCard';
import { AllFailedNotice, TermRow } from '../features/today/TodayParts';
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

// 요청마다 자료 저장소가 만든 10초 중단 신호를 묶은 클라이언트로 부른다.
const api = (signal: AbortSignal) => createClient({ signal });

/** 오늘 화면의 자료 입구(App.tsx → TodayEntryProvider). */
export const TODAY_ENTRY: TodayEntry = {
  load: {
    dayAhead: signal => api(signal).getDayAheadSMP(),
    forecast: signal => api(signal).getSMPForecast(),
    modelInfo: signal => api(signal).getModelInfo(),
    kpis: signal => api(signal).getDashboardKPIs(),
    powerSupply: signal => api(signal).getPowerSupply(),
    market: signal => api(signal).getMarketStatus(),
  },
  // isAvailableDayAhead는 런타임 타입 가드다(status·market_type·값·거래일·거래시간을 다 본다). 파싱 결과는 모양이 같고
  // 문자열 리터럴 타입만 넓어서 옛 타입으로 넘긴다.
  judgeDayAhead: quote => {
    const dayAheadSmp = quote as DayAheadSMP | null;
    return { available: isAvailableDayAhead(dayAheadSmp), kpxText: describeDayAhead(dayAheadSmp) };
  },
  chartLabel: kind => FORECAST_CHART_LABEL[kind],
};

const VISIBILITY = { isVisible, watchVisible };

export default function SMPForecastScreen() {
  const { view, res, skew, chartLabel } = useTodayView(true);
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
      {skew && !skew.plausible ? <ProvenanceChip chip={CHIPS.clockImplausible} /> : null}
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
            onRetry={() => { forecast.reload(); }} />
        </>
      )}
      <TermRow onOpen={id => nav.openSheet({ kind: 'term', id })} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
});
