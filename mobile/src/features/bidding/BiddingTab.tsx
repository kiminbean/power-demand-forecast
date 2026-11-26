/**
 * 입찰(명세 4.2절, 경로 /bidding · 라우트 키 bidding). 이 파일이 입찰 화면의 자료 입구다: 허용 목록 GET 두 개(시장 상태 · 입찰 계산)만
 * 보낸다(6.3절). 시장 상태는 오늘 화면과 같은 저장소 키라 함께 쓰고, 기준선 예측은 오늘 화면이 받아 둔 것을 엿보기만 한다(가격 근거
 * 줄, 7.8절 2). 입찰 계산은 성향마다 저장소 키를 따로 두어 5분 동안 다시 쓰고 한 번에 하나만 보낸다(calculator.ts). 폴링하지 않는다.
 * 순수 함수 model.ts가 글자·상태를 만들고 카드는 배치만 한다. 산출 방식·출처 시트와 ≥ 1280 패널은 useBiddingView()로 같은 저장소를
 * 엿본다(새 요청 없음, App.tsx가 시트로 등록한다).
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { useClock, useNav, useShellData } from '../../app/Shell';
import type { RefreshResult } from '../../app/Shell';
import { ROUTES } from '../../app/routes';
import type { MethodCard, Risk, SourceRow } from '../../app/routes';
import { isVisible, watchVisible } from '../../app/head';
import { createClient, ENDPOINTS } from '../../data/client';
import type { SmpForecast } from '../../data/types';
import { kstNow, parseApiTime, parseDayKey } from '../../lib/kst';
import { CHIPS } from '../../lib/provenance';
import { appResources, shouldLoadOnMount, useRefreshWhileVisible, useResource } from '../../lib/resource';
import type { Resource } from '../../lib/resource';
import {
  BIDDING_EXAMPLE_NOTE, BIDDING_PANEL_METHOD, BIDDING_TERMS, MY_PLANT_BODY, MY_PLANT_TITLE, legendPlantText, legendProbabilityText,
  legendRuleText,
} from '../../content/strings';
import { LiveFacts } from '../guide/sheets';
import { ReadingPanelBody } from '../today/TodayParts';
import { calcKey, calcLoader } from './calculator';
import { biddingView, riskOf } from './model';
import type { BiddingCardId, Recheck } from './model';
import { DeadlineCard } from './DeadlineCard';
import { MeritOrderCard } from './MeritOrderCard';
import { SampleBidCard } from './SampleBidCard';
import { RestrictedCard } from '../../ui/Card';
import { NowPill, ProvenanceChip } from '../../ui/Chips';
import { AnswerLine, Screen } from '../../ui/Screen';
import { Text } from '../../ui/Text';
import { SPACE } from '../../ui/tokens';

const loadMarket = (signal: AbortSignal) => createClient({ signal }).getMarketStatus();
const VISIBILITY = { isVisible, watchVisible };
// 계산 결과를 받을 때 엿본 기준선 예측일(그 뒤 예측일이 바뀌면 오래된 입력). 응답 객체는 바뀌지 않아 객체를 열쇠로 기억한다.
const CALC_FORECAST_DATE = new WeakMap<object, string | null>();

export type BiddingFlow = { recheck: Recheck; staleRetried: boolean };
const AT_REST: BiddingFlow = { recheck: 'idle', staleRetried: false };

/**
 * 입찰 화면의 자료와 view model. fetching이면 처음 마운트에 받고(lib/resource 규칙) 다시 받을 손잡이를 준다. 아니면 엿보기만 한다.
 * 저장소의 조회 시각은 기기 시계라서 서버 시계 차이를 더해 앱 시각으로 바꾼 뒤 넘긴다(나이는 어느 쪽이든 같다).
 */
export function useBiddingView(fetching = false, flow: BiddingFlow = AT_REST) {
  const nav = useNav();
  const clock = useClock();
  const risk = riskOf(nav.state);
  const hour = nav.state.key === 'bidding' ? nav.state.hour : null;
  const market = useResource(ENDPOINTS.marketStatus, fetching ? loadMarket : undefined);
  const calc = useResource(calcKey(risk), fetching ? calcLoader(risk) : undefined);
  const forecast = useResource<SmpForecast>(ENDPOINTS.forecast);
  const { reportServerTime } = clock;
  // 시장 상태의 서버 시각으로 기기 시계 차이를 맞춘다(7.5·7.6절). 받은 기기 시각과 짝을 짓는다.
  useEffect(() => {
    if (!fetching || !market.data || market.fetchedAt === null) return;
    const server = parseApiTime(market.data.current_time);
    if (server !== null) reportServerTime(server, market.fetchedAt);
  }, [fetching, market.data, market.fetchedAt, reportServerTime]);
  const skewMs = clock.skew && clock.skew.plausible ? clock.skew.skewMs : 0;
  const forecastDate = forecast.data ? parseDayKey(forecast.data.forecast_date) : null;
  if (calc.data && !CALC_FORECAST_DATE.has(calc.data)) CALC_FORECAST_DATE.set(calc.data, forecastDate);
  const calcForecastDate = calc.data ? CALC_FORECAST_DATE.get(calc.data) ?? null : null;
  const view = useMemo(() => {
    const app = <T,>(r: Resource<T>) => ({ data: r.data, status: r.status, error: r.error, fetchedAt: r.fetchedAt === null ? null : r.fetchedAt + skewMs });
    return biddingView({
      now: clock.now, market: app(market), calc: { ...app(calc), busy: calc.busy }, forecast: forecast.data, calcForecastDate, risk,
      selectedHour: hour, recheck: flow.recheck, staleRetried: flow.staleRetried,
    });
  }, [clock.now, skewMs, market, calc, forecast.data, calcForecastDate, risk, hour, flow.recheck, flow.staleRetried]);
  return { view, market, calc, risk, skew: clock.skew };
}

export default function BiddingTab() {
  const nav = useNav();
  const clock = useClock();
  const [recheck, setRecheck] = useState<Recheck>('idle');
  // 오래된 입력으로 한 번 다시 계산한 성향(계산 저장소 키). 성향마다 한 번.
  const [staleFor, setStaleFor] = useState<string | null>(null);
  const key = calcKey(riskOf(nav.state));
  const { view, market, calc, risk, skew } = useBiddingView(true, { recheck, staleRetried: staleFor === key });
  const reloadMarket = market.reload;
  const reloadCalc = calc.reload;

  // 서버 상태와 계산이 어긋나면 즉시 한 번 다시 조회하고, 그래도 다르면 남은 시간을 숨긴다(4.2절 불일치). 맞으면 되돌린다.
  useEffect(() => {
    if (!view.mismatch) {
      if (recheck !== 'idle') setRecheck('idle');
      return;
    }
    if (recheck !== 'idle') return;
    setRecheck('pending');
    reloadMarket().finally(() => setRecheck('done'));
  }, [view.mismatch, recheck, reloadMarket]);

  // 계산이 오래된 입력(입찰일·예측일이 바뀜)이면 한 번 다시 계산한다. 그동안 카드는 스켈레톤이다.
  useEffect(() => {
    if (!view.calcStale) {
      if (staleFor === key) setStaleFor(null);
      return;
    }
    if (staleFor === key || calc.busy) return;
    setStaleFor(key);
    reloadCalc();
  }, [view.calcStale, staleFor, key, calc.busy, reloadCalc]);

  // 화면이 보이는 동안 5분마다 시장 상태만 다시 받는다(계산기는 폴링하지 않는다, 4.2절).
  const marketAge = useCallback(() => market.fetchedAt, [market.fetchedAt]);
  useRefreshWhileVisible(() => { reloadMarket(); }, marketAge, VISIBILITY);

  // 헤더 새로고침: 시장 상태는 늘, 계산은 성향별 5분 기억이 지났거나 실패했을 때만(서버가 한 번에 하나만 계산한다).
  const refresh = useCallback(async (): Promise<RefreshResult> => {
    const due = shouldLoadOnMount(appResources.get(key), appResources.now());
    const [m, c] = await Promise.all([reloadMarket(), due ? reloadCalc() : Promise.resolve(true)]);
    const offset = skew && skew.plausible ? skew.skewMs : 0;
    const shown = appResources.get(ENDPOINTS.marketStatus).fetchedAt;
    return m && c ? { ok: true, at: kstNow(skew) } : { ok: false, shownAt: shown === null ? null : shown + offset };
  }, [key, reloadMarket, reloadCalc, skew]);
  useShellData(useMemo(() => ({ refresh, fetchedAt: view.fetchedAt }), [refresh, view.fetchedAt]));

  // 거래시간·성향 선택은 URL에 남는다(같은 항목을 고친다, 3.1절). 성향을 바꾸면 그 성향의 계산을 받는다(5분 안이면 기억한 값).
  const selectHour = useCallback((hour: number) => nav.replace({ ...nav.state, hour }), [nav]);
  const selectRisk = useCallback((next: Risk) => nav.replace({ ...nav.state, risk: next }), [nav]);
  const openMethod = (card: BiddingCardId) => () => nav.openSheet({ kind: 'method', card });
  const openSources = (row: SourceRow) => () => nav.openSheet({ kind: 'sources', row });

  const pill = (
    <View style={styles.pills}>
      <NowPill now={clock.now} fetchedAt={view.fetchedAt} skewNotable={clock.skewNotable} />
      {skew && !skew.plausible ? <ProvenanceChip chip={CHIPS.checkTime} /> : null}
    </View>
  );
  const answer = (
    <View style={styles.answer}>
      <AnswerLine>{view.answer}</AnswerLine>
      <Text variant="label" tone="text2">{BIDDING_EXAMPLE_NOTE}</Text>
    </View>
  );
  return (
    <Screen title={ROUTES.bidding.heading} pill={pill} answer={answer} testID="bidding-screen">
      <DeadlineCard view={view.deadline} onMethod={openMethod('bid-deadline')} onSources={openSources('market')} onRetry={() => { reloadMarket(); }} />
      <SampleBidCard view={view.calc} risk={risk} onRisk={selectRisk} onHour={selectHour} onRetry={() => { reloadCalc(); }}
        onMethod={openMethod('bid-ladder')} onSources={openSources('bidding')} />
      <MeritOrderCard onMethod={openMethod('bid-merit')} />
      <RestrictedCard title={MY_PLANT_TITLE} body={MY_PLANT_BODY} testID="bid-restricted" />
    </Screen>
  );
}

const isBidCard = (card: MethodCard): card is BiddingCardId => card === 'bid-deadline' || card === 'bid-ladder' || card === 'bid-merit';

/** 산출 방식 시트의 "이번 화면의 값"(마감: API 원문과 서버 시각으로 다시 계산한 남은 시간 · 계단: 계산 방법·모델 원문·입력). */
export function BiddingMethodFacts({ card }: { card: MethodCard }) {
  const { view } = useBiddingView(false);
  if (!isBidCard(card)) return null;
  const facts = view.facts[card];
  return <LiveFacts lines={facts.lines} link={facts.link} />;
}

/** 출처와 기준 시각 표의 입찰 계산 행(4.4절). 오늘 화면의 행 뒤에 붙는다. */
export function useBiddingSourceRows() {
  return useBiddingView(false).view.sources;
}

/** ≥ 1280 "입찰 읽는 법" 패널(4.2·4.6절): 이 화면의 표시 예 · 용어 3개 · 읽는 법 3줄. */
export function BiddingPanel() {
  const { view } = useBiddingView(false);
  const nav = useNav();
  const legend = [
    { chip: CHIPS.marketRule, text: legendRuleText(view.legend.when) },
    { chip: CHIPS.estShort, text: legendProbabilityText(view.legend.probability) },
    { chip: CHIPS.exampleVirtual, text: legendPlantText(view.legend.capacityMw) },
  ];
  return <ReadingPanelBody legend={legend} terms={BIDDING_TERMS} method={BIDDING_PANEL_METHOD} onTerm={id => nav.openSheet({ kind: 'term', id })} testID="bidding-panel" />;
}

const styles = StyleSheet.create({
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
  answer: { gap: SPACE.xs },
});
