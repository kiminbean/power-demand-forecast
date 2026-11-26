/**
 * 알아보기(명세 4.4절, 경로 /guide · 라우트 키 guide). 하위 상태는 하나만(routes.ts): 둘러보기 단계(?tour=, TourScreen.tsx) > 용어 한 개
 * (?term=) > 주제(?topic=). 첫 화면은 둘러보기 입구와 목록 6행이고 요청이 없다. 긴 글은 guide.ko.json(처음 필요할 때 한 번)에서,
 * 숫자는 화면 값과 strings.ts에서만 온다. 요청은 "데이터 출처와 연결 상태"의 /realtime-status 하나뿐이고(6.3절), 다른 값(예측 종류·
 * 모델 정보·시장 상태·시계 차이·출처 표의 오늘·입찰 행)은 이번 방문의 저장소를 엿보기만 한다. 출처와 기준 시각 시트도 응답 상태 행을
 * 엿보기만 해서(useStatusRows) 알아보기를 거치지 않은 방문에서는 "이번 방문에서 아직 조회하지 않음"이다.
 */
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { GuideGate, TermSheetBody, WhereLink } from './sheets';
import { aboutClockLines, marketFacts, methodAccuracy, statusView } from './model';
import { TourScreen } from './TourScreen';
import { useClock, useNav } from '../../app/Shell';
import { GUIDE_TOPIC_TITLES, ROUTES, TOUR_STEPS, rootState } from '../../app/routes';
import type { GuideTopic } from '../../app/routes';
import { THEME_OPTIONS } from '../../app/SheetHost';
import { useTheme } from '../../app/theme';
import { createClient, ENDPOINTS } from '../../data/client';
import type { MarketStatus, RealtimeStatus, SmpForecast } from '../../data/types';
import { CHIPS, smpForecastKind } from '../../lib/provenance';
import { useResource } from '../../lib/resource';
import type { GuideContent } from '../../content/guide';
import {
  ABOUT_ROW, ABOUT_TITLE, GLOSSARY_TITLES, GUIDE_ANSWER, MODEL_NOT_USED, READING_HEADINGS, RS_FAILED, SIMPLIFIED_NOTE, SOURCES_SHEET_TITLE,
  TOUR_BANNER, TOUR_NAME, appUpdatedText, glossaryCountText, rsOverallText, tourCardTitle, tradeHourExampleText,
} from '../../content/strings';
import type { GlossaryId } from '../../content/strings';
import { useBiddingSourceRows } from '../bidding/BiddingTab';
import { AccuracyCard } from '../today/AccuracyCard';
import { TodaySourcesSheet } from '../today/sheets';
import { Button } from '../../ui/Button';
import { Card, StateView } from '../../ui/Card';
import { NowPill, ProvenanceChip, StatusPill } from '../../ui/Chips';
import { ListRow, TextLink } from '../../ui/Controls';
import { AnswerLine, Screen } from '../../ui/Screen';
import { Text } from '../../ui/Text';
import { LAYOUT, RADIUS, SPACE } from '../../ui/tokens';

const GLOSSARY_IDS = Object.keys(GLOSSARY_TITLES) as GlossaryId[];
const loadStatus = (signal: AbortSignal) => createClient({ signal }).getRealtimeStatus();
const topicState = (topic: GuideTopic) => ({ ...rootState('guide'), topic });

/** 조회가 없는 화면의 NowPill: KST 시각만(3.4절). 서버 시계가 타당하지 않으면 ‹시각 확인 필요›(7.6절). 둘러보기 단계에도 넘긴다. */
export function GuidePill() {
  const { now, skew, skewNotable } = useClock();
  return (
    <View style={styles.pills}>
      <NowPill now={now} fetchedAt={null} skewNotable={skewNotable} />
      {skew && !skew.plausible ? <ProvenanceChip chip={CHIPS.clockImplausible} /> : null}
    </View>
  );
}

/** 목록 행 묶음(카드와 같은 면·모서리). 마지막 행은 아래 선이 없다. */
function List({ children, testID }: { children: React.ReactNode; testID?: string }) {
  const { theme } = useTheme();
  return <View style={[styles.list, { backgroundColor: theme.color.surface, borderColor: theme.color.border }]} testID={testID}>{children}</View>;
}

/**
 * 이어 볼 주제 링크(화면 끝). 링크·버튼이 없는 글 화면(숫자 읽는 법, 딥링크로 연 출처)도 키보드로 스크롤 영역에 들어올 수 있게 한다
 * (axe scrollable-region-focusable, F10 측정).
 */
function TopicLink({ topic }: { topic: GuideTopic }) {
  const nav = useNav();
  return <TextLink label={GUIDE_TOPIC_TITLES[topic]} onPress={() => nav.push(topicState(topic))} />;
}

export function GuideHome() {
  const nav = useNav();
  const { mode } = useTheme();
  const open = (topic: GuideTopic) => () => nav.push(topicState(topic));
  const theme = THEME_OPTIONS.find(o => o.value === mode);
  return (
    <Screen title={ROUTES.guide.heading} pill={<GuidePill />} answer={<AnswerLine>{GUIDE_ANSWER}</AnswerLine>} testID="guide-screen">
      <Card title={tourCardTitle(TOUR_STEPS)} testID="guide-tour-card">
        <Text variant="label" tone="text2">{TOUR_BANNER.body}</Text>
        <View style={styles.start}>
          <Button tone="primary" label="시작하기" accessibilityLabel={`시작하기: ${TOUR_NAME}`} onPress={() => nav.push({ ...rootState('guide'), tour: 1 })} />
        </View>
      </Card>
      <List testID="guide-list">
        <ListRow title={GUIDE_TOPIC_TITLES.glossary} detail={glossaryCountText(GLOSSARY_IDS.length)} onPress={open('glossary')} />
        <ListRow title={GUIDE_TOPIC_TITLES.reading} onPress={open('reading')} />
        <ListRow title={GUIDE_TOPIC_TITLES.method} onPress={open('method')} />
        <ListRow title={GUIDE_TOPIC_TITLES.sources} onPress={open('sources')} />
        <ListRow title="화면 테마" detail={theme ? theme.label : undefined} onPress={() => nav.openSheet({ kind: 'theme' })} />
        <ListRow title={GUIDE_TOPIC_TITLES.about} detail={ABOUT_ROW} onPress={open('about')} last />
      </List>
    </Screen>
  );
}

/** 용어 사전(?topic=glossary): 항목마다 이름(→ ?term=) · 한 줄 뜻 · "이 앱에서 보는 곳 ›". */
export function GlossaryList({ content }: { content: GuideContent }) {
  const nav = useNav();
  const { theme } = useTheme();
  const last = content.glossary.length - 1;
  return (
    <List testID="glossary">
      {content.glossary.map((g, i) => (
        <View key={g.id} style={[styles.term, { borderBottomColor: i === last ? 'transparent' : theme.color.border }]}>
          <ListRow title={GLOSSARY_TITLES[g.id]} detail={g.short} onPress={() => nav.push({ ...rootState('guide'), term: g.id })} last />
          <WhereLink id={g.id} />
        </View>
      ))}
    </List>
  );
}

/** 숫자 읽는 법(?topic=reading): 4상태(칩 모양 그대로) · 일반 칩 · 줄표 · 지연·끊김 · KST와 거래시간 표기. */
export function ReadingBody({ content }: { content: GuideContent }) {
  const L = content.legend;
  const item = (key: string, mark: React.ReactNode, text: string) => (
    <View key={key} style={styles.mark}><View style={styles.marks}>{mark}</View><Text variant="label" tone="text2">{text}</Text></View>
  );
  return (
    <>
      {content.topics.reading.body.map(p => <AnswerLine key={p}>{p}</AnswerLine>)}
      <Card title={READING_HEADINGS.states} testID="reading-states">
        {item('measured', <ProvenanceChip chip={CHIPS.measured} />, L.measured)}
        {item('forecast', <ProvenanceChip chip={CHIPS.forecastBaseline} />, L.forecast)}
        {item('estimated', <ProvenanceChip chip={CHIPS.estNotMeasured} />, L.estimated)}
        {item('example', <ProvenanceChip chip={CHIPS.example} />, L.example)}
      </Card>
      <Card title={READING_HEADINGS.other} testID="reading-other">
        {item('plain', <><ProvenanceChip chip={CHIPS.verified} /><ProvenanceChip chip={CHIPS.marketRule} /></>, L.plain)}
        {item('missing', <Text variant="title" tone="text3">—</Text>, L.missing)}
        {item('stale', <ProvenanceChip lag="stale" at={0} text="지연" />, L.stale)}
        {item('broken', <ProvenanceChip lag="broken" at={0} text="끊김" />, L.broken)}
      </Card>
      <Card title={READING_HEADINGS.time} testID="reading-time">
        <Text variant="body" tone="text2">{L.time}</Text>
        <Text variant="label" weight="600" dataSet={{ pdMeta: 'example' }}>{tradeHourExampleText()}</Text>
      </Card>
      <TopicLink topic="glossary" />
    </>
  );
}

/** 예측 방법과 정확도(?topic=method): 지금 쓰는 방법 · 정확도(오늘 카드 + R²) · 단순화한 설명 목록(도메인 검토 전, P7). */
function MethodScreen() {
  const nav = useNav();
  const forecast = useResource<SmpForecast>(ENDPOINTS.forecast);
  const market = useResource<MarketStatus>(ENDPOINTS.marketStatus);
  const accuracy = methodAccuracy(forecast.data ? smpForecastKind(forecast.data) : null);
  const facts = marketFacts(market.data);
  return (
    <Screen title={GUIDE_TOPIC_TITLES.method} pill={<GuidePill />} testID="guide-method">
      <GuideGate>
        {c => (
          <>
            <AnswerLine>{c.topics.method.body[0]}</AnswerLine>
            {c.topics.method.body.slice(1).map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
            <Text variant="label" tone="text2" dataSet={{ pdMeta: 'model' }}>{MODEL_NOT_USED}</Text>
            <AccuracyCard view={accuracy} onMethod={() => nav.openSheet({ kind: 'method', card: 'today-accuracy' })}
              onSources={() => nav.openSheet({ kind: 'sources', row: 'accuracy' })} onRetry={() => { forecast.reload(); }} />
            <Card title={c.simplified.title} chips={<ProvenanceChip chip={CHIPS.simplified} />} testID="method-simplified">
              {c.simplified.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
              {facts ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'facts' }}>{facts}</Text> : null}
              <Text variant="meta" tone="text3">{SIMPLIFIED_NOTE}</Text>
            </Card>
          </>
        )}
      </GuideGate>
    </Screen>
  );
}

/** 출처 표의 응답 상태 3행(엿보기만): 출처와 기준 시각 시트가 쓴다. */
export function useStatusRows() {
  return statusView(useResource<RealtimeStatus>(ENDPOINTS.realtimeStatus)).rows;
}

/** 데이터 출처와 연결 상태(?topic=sources): 출처 시트와 같은 표 + /realtime-status 전체 상태. 이 화면만 응답 상태를 받는다. */
function SourcesScreen() {
  const rs = useResource(ENDPOINTS.realtimeStatus, loadStatus);
  const status = statusView(rs);
  const rows = [...useBiddingSourceRows(), ...status.rows];
  return (
    <Screen title={GUIDE_TOPIC_TITLES.sources} pill={<GuidePill />} testID="guide-sources">
      <GuideGate>{c => c.topics.sources.body.map((p, i) => (i ? <Text key={p} variant="label" tone="text2">{p}</Text> : <AnswerLine key={p}>{p}</AnswerLine>))}</GuideGate>
      <Card title={SOURCES_SHEET_TITLE} testID="sources-card"
        chips={status.overall ? <StatusPill tone={status.overall.tone} text={rsOverallText(status.overall.text)} /> : undefined}>
        {rs.status === 'error' && !rs.data ? <StateView kind="error" text={RS_FAILED} onRetry={() => { rs.reload(); }} /> : null}
        <TodaySourcesSheet row={null} extra={rows} page />
      </Card>
      <TopicLink topic="reading" />
    </Screen>
  );
}

/** 앱 정보(?topic=about): 이름·버전 · 읽기 전용 · 자료 이용 조건 · 화면 갱신일 · 시계 차이(120초 넘을 때) · 데이터 출처와 연결 상태 ›. */
function AboutScreen() {
  const nav = useNav();
  const { skew } = useClock();
  return (
    <Screen title={GUIDE_TOPIC_TITLES.about} pill={<GuidePill />} testID="guide-about">
      <Card title={ABOUT_TITLE} testID="about-card">
        <GuideGate>{c => c.topics.about.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}</GuideGate>
        <Text variant="label" tone="text2" dataSet={{ pdMeta: 'updated' }}>{appUpdatedText()}</Text>
        {aboutClockLines(skew).map(l => <Text key={l} variant="label" tone="text2" dataSet={{ pdMeta: 'clock' }}>{l}</Text>)}
      </Card>
      <List><ListRow title={GUIDE_TOPIC_TITLES.sources} onPress={() => nav.push(topicState('sources'))} last /></List>
    </Screen>
  );
}

export default function GuideTab() {
  const { state } = useNav();
  if (state.tour !== null) return <TourScreen step={state.tour} pill={<GuidePill />} />;
  const term = state.term;
  if (term !== null) {
    return (
      <Screen title={GLOSSARY_TITLES[term]} pill={<GuidePill />} testID="guide-term">
        <GuideGate>{c => <AnswerLine>{(c.glossary.find(g => g.id === term) ?? { short: '' }).short}</AnswerLine>}</GuideGate>
        <Card title="자세한 풀이"><TermSheetBody id={term} sheet={false} /></Card>
      </Screen>
    );
  }
  switch (state.topic) {
    case 'glossary':
      return <Screen title={GUIDE_TOPIC_TITLES.glossary} pill={<GuidePill />} testID="guide-glossary"><GuideGate>{c => <GlossaryList content={c} />}</GuideGate></Screen>;
    case 'reading':
      return <Screen title={GUIDE_TOPIC_TITLES.reading} pill={<GuidePill />} testID="guide-reading"><GuideGate>{c => <ReadingBody content={c} />}</GuideGate></Screen>;
    case 'method':
      return <MethodScreen />;
    case 'sources':
      return <SourcesScreen />;
    case 'about':
      return <AboutScreen />;
    default:
      return <GuideHome />;
  }
}

const styles = StyleSheet.create({
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
  start: { alignSelf: 'flex-start' },
  list: { borderWidth: 1, borderRadius: RADIUS.card, paddingHorizontal: SPACE.lg, width: '100%', maxWidth: LAYOUT.column, alignSelf: 'center' },
  term: { borderBottomWidth: 1, paddingBottom: SPACE.xs },
  mark: { gap: SPACE.xs },
  marks: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
});
