/**
 * 1분 둘러보기(명세 4.4절, /guide?tour=1..5). 단계마다 h1(단계 제목, 설명 글을 받기 전에는 "둘러보기 2/5"), 진행 "2 / 5", [이전] [다음]
 * [건너뛰기](모두 44 px). 마지막 단계는 [이전] [직접 둘러보기](→ 오늘). 단계 이동은 새 히스토리 항목이라 브라우저 뒤로가 앞 단계로 간다(3.3절).
 * 단계 예시는 실제 화면과 같은 컴포넌트다: 1 PriceCard · 2 보기 전용 StepChart와 정확도 문장 · 3 DeadlineCard와 계단 요약(▨ 추정,
 * 가상 발전소, 기대 수입은 쓰지 않는다) · 4 SampleStatementCard · 5 이번 방문 값의 표시 범례(값이 없으면 표시와 설명만).
 * 1–3단계는 그 화면과 같은 저장소 키로 받는다(5분 안에 받은 값이 있으면 요청 없음). 5단계는 엿보기만 한다.
 */
import React, { useEffect } from 'react';
import { View } from 'react-native';
import { tourLegend } from './model';
import { GuideGate } from './sheets';
import { useGuide } from './useGuide';
import { useClock, useNav } from '../../app/Shell';
import { TOUR_STEPS, rootState } from '../../app/routes';
import type { MethodCard, SourceRow } from '../../app/routes';
import { CHIPS } from '../../lib/provenance';
import {
  CARD_TITLES, NO_SEND_NOTE, RECALC_LABEL, TOUR_CALC_FAILED, TOUR_LEGEND_TITLE, tourLabel,
} from '../../content/strings';
import { useBiddingView } from '../bidding/BiddingTab';
import type { BiddingFlow } from '../bidding/BiddingTab';
import { DeadlineCard } from '../bidding/DeadlineCard';
import { settlementView } from '../settlement/model';
import { SampleStatementCard } from '../settlement/SampleStatementCard';
import { ChipList, LegendRow } from '../today/TodayParts';
import { PriceCard } from '../today/PriceCard';
import { useTodayView } from '../today/data';
import { markTourSeen } from '../today/tourSeen';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ProvenanceChip, SampleRibbon } from '../../ui/Chips';
import { ChartBleed } from '../../ui/charts/ChartFrame';
import { Screen } from '../../ui/Screen';
import { StepChart } from '../../ui/charts/StepChart';
import { Text, keepTogether } from '../../ui/Text';
import { TourStep } from '../../ui/Tour';

// 둘러보기에서는 입찰 화면처럼 다시 조회·다시 계산을 걸지 않는다: 불일치·오래된 입력이면 그 상태(‹시각 확인 필요›)를 그대로 보인다.
const TOUR_FLOW: BiddingFlow = { recheck: 'done', staleRetried: true };

function useSheets() {
  const nav = useNav();
  return { method: (card: MethodCard) => () => nav.openSheet({ kind: 'method', card }), sources: (row: SourceRow) => () => nav.openSheet({ kind: 'sources', row }) };
}

function TourPrice() {
  const open = useSheets();
  const { view, res } = useTodayView(true);
  return <PriceCard view={view.price} onMethod={open.method('today-price')} onSources={open.sources('day-ahead')} onRetry={() => { res.dayAhead.reload(); }} />;
}

/** 2단계: 상호작용 없는 기준선 계단(스텝퍼·표 없음) + 정확도 문장. */
function TourForecast() {
  const open = useSheets();
  const { now } = useClock();
  const { view, res, chartLabel } = useTodayView(true);
  const f = view.forecast;
  const a = view.accuracy.state === 'ready' ? view.accuracy : null;
  const title = CARD_TITLES['today-forecast'];
  if (f.state !== 'ready') {
    return (
      <Card title={title} onMethod={open.method('today-forecast')} testID="tour-forecast">
        <StateView kind={f.state === 'loading' ? 'loading' : f.state} text={f.state === 'loading' ? undefined : f.text}
          onRetry={f.state === 'error' ? () => { res.forecast.reload(); } : undefined} />
      </Card>
    );
  }
  return (
    <Card title={title} chips={<ChipList chips={f.chips} />} onMethod={open.method('today-forecast')} testID="tour-forecast">
      <Text variant="meta" tone="text2" dataSet={{ pdMeta: 'legend' }}>{f.legend}</Text>
      <ChartBleed><StepChart forecast={f.view} label={chartLabel(f.kind)} now={now} dots={f.dots} /></ChartBleed>
      {a && a.documented ? <Text variant="body" weight="700" dataSet={{ pdMeta: 'documented' }}>{a.documented.main}</Text> : null}
      {a && a.documented ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'period' }}>{a.documented.basis}</Text> : null}
      {a && a.unregistered ? <Text variant="body" tone="text2">{a.unregistered}</Text> : null}
      <SourceFooter source={f.footer.source} time={f.footer.time} onSources={open.sources('forecast')} context={title} />
    </Card>
  );
}

/** 3단계: 실제 마감 카드 + 예시 발전소 계단 요약. 계산을 못 불러오면 마감 카드와 "계산 예시를 불러오지 못했습니다"(+ 다시 계산). */
function TourBidding() {
  const open = useSheets();
  const { view, market, calc } = useBiddingView(true, TOUR_FLOW);
  const c = view.calc;
  const title = CARD_TITLES['bid-ladder'];
  return (
    <>
      <DeadlineCard view={view.deadline} onMethod={open.method('bid-deadline')} onSources={open.sources('market')} onRetry={() => { market.reload(); }} />
      <Card title={title} ribbon={<SampleRibbon chip={CHIPS.exampleVirtual} />} chips={c.state === 'ready' ? <ProvenanceChip {...c.chip} /> : undefined}
        onMethod={open.method('bid-ladder')} testID="tour-ladder">
        {c.state === 'ready' ? (
          <>
            {c.summary ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{keepTogether(c.summary, { variant: 'body' })}</Text> : null}
            {c.trend ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{c.trend}</Text> : null}
            <Text variant="meta" tone="text3">{c.notReal}</Text>
          </>
        ) : c.state === 'loading' ? <StateView kind="loading" />
          : <StateView kind="error" text={c.state === 'empty' ? c.text : TOUR_CALC_FAILED} retryLabel={RECALC_LABEL}
              onRetry={c.state === 'error' && !c.retry ? undefined : () => { calc.reload(); }} />}
        <Text variant="meta" tone="text3">{NO_SEND_NOTE}</Text>
        {c.state === 'ready' ? <SourceFooter source={c.footer.source} time={c.footer.time} onSources={open.sources('bidding')} context={title} /> : null}
      </Card>
    </>
  );
}

/** 5단계: 이번 방문에 받은 값으로 네 표시(엿보기만, 새 요청 없음). */
function TourLegend() {
  const today = useTodayView(false).view;
  const bidding = useBiddingView(false).view;
  const rows = tourLegend({
    today: today.legend, forecastKind: today.forecast.state === 'ready' ? today.forecast.kind : null,
    probability: bidding.legend.probability, statementTotal: settlementView().total.value,
  });
  return <Card title={TOUR_LEGEND_TITLE} testID="tour-legend">{rows.map(r => <LegendRow key={r.chip.text} chip={r.chip} text={r.text} />)}</Card>;
}

function TourExample({ step }: { step: number }) {
  const open = useSheets();
  if (step === 1) return <TourPrice />;
  if (step === 2) return <TourForecast />;
  if (step === 3) return <TourBidding />;
  return step === 4 ? <SampleStatementCard onMethod={open.method('settle-sample')} /> : <TourLegend />;
}

/** pill: 알아보기 화면과 같은 NowPill(GuideTab이 넘긴다, 두 파일이 서로 가져오지 않게). */
export function TourScreen({ step, pill }: { step: number; pill: React.ReactNode }) {
  const nav = useNav();
  const guide = useGuide();
  useEffect(() => { markTourSeen(); }, []);
  const section = guide.content ? guide.content.tour[step - 1] : null;
  const toStep = (n: number) => nav.push({ ...rootState('guide'), tour: n });
  return (
    <Screen title={section ? section.title : tourLabel(step, TOUR_STEPS)} pill={pill} testID="guide-tour">
      <TourStep step={step} total={TOUR_STEPS} onPrev={() => toStep(step - 1)} onSkip={() => nav.push(rootState('guide'))}
        onNext={() => (step < TOUR_STEPS ? toStep(step + 1) : nav.push(rootState('smp')))}>
        <GuideGate>{c => <View>{c.tour[step - 1].body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}</View>}</GuideGate>
      </TourStep>
      <TourExample step={step} />
    </Screen>
  );
}
