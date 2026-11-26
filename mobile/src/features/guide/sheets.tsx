// 시트 본문(명세 4.4절): 산출 방식(카드별 설명 글 + 화면이 넘기는 이번 화면의 값), 용어 한 개, 출처와 기준 시각 표.
// 시트 틀(dialog·aria-modal·초점 가둠·Esc·복귀·히스토리)은 ui/Sheet와 app/SheetHost가 맡는다. 긴 글은 guide.ko.json에서만 오고,
// 숫자는 화면 값에서만 온다(JSON에는 숫자가 없다). 행은 화면마다 모듈이 넘긴다: 오늘(F7) · 입찰 계산(F8) · 응답 상태(F10, guide/model).
// 용어 시트와 용어 화면(?term=)은 같은 본문을 쓴다: 한 줄 뜻 + 풀이 + "이 앱에서 보는 곳 ›"(시트는 "용어 사전 전체 ›"를 더한다).
import type React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { useGuide } from './useGuide';
import { TERM_WHERE } from './model';
import { useNav } from '../../app/Shell';
import { ROUTES, rootState } from '../../app/routes';
import type { GuideContent, GuideSection, MethodCardId } from '../../content/guide';
import { DETAIL_LABEL, GLOSSARY_TITLES, LIVE_FACTS_HEADING, SIMPLIFIED_NOTE, SOURCE_HEADINGS, SOURCES_NOTES, whereText } from '../../content/strings';
import type { GlossaryId } from '../../content/strings';
import { useTheme } from '../../app/theme';
import { RetryScope, StateView, useInRetryScope } from '../../ui/Card';
import { ProvenanceChip, StatusPill } from '../../ui/Chips';
import type { ChipProps } from '../../ui/Chips';
import { TextLink } from '../../ui/Controls';
import { DataTable } from '../../ui/DataTable';
import { Text, keepTogether, labelData } from '../../ui/Text';
import { HIT, LAYOUT, RADIUS, SPACE } from '../../ui/tokens';
import type { PillTone } from '../../ui/tokens';
import { focusScreenHeading, web } from '../../ui/web';

const GUIDE_LOADING = '설명을 불러오는 중입니다';
const GUIDE_FAILED = '설명을 불러오지 못했습니다';
/** 표 보기는 768 이상(4.4절). 다만 ≥ 1024의 시트는 오른쪽 패널(400)이라 두 줄 목록이다(알아보기 화면은 본문 640이라 표). */
const SOURCES_TABLE_FROM = 768;

/**
 * 설명 글이 준비되면 그린다. 읽는 동안 스켈레톤, 실패하면 "설명을 불러오지 못했습니다" + [다시 시도]. 입찰의 가격 원리 카드도 쓴다.
 * [다시 시도](round 2 M5·FOCUS): 버튼이 사라지기 전에 초점을 가장 가까운 제목(카드 h2 · 시트 제목 · 그 밖은 화면 h1)으로 옮기고, 다시 받으면
 * "다시 불러왔습니다 · HH:MM KST"를 알린다. 카드·시트 밖이면 화면 h1 범위를 여기서 만든다. 오류와 불러오는 중이 같은 자리(같은 꼴)라
 * 누른 StateView가 다시 만들어지지 않는다. 한 화면에는 이 문이 하나다(용어 화면은 뜻과 풀이를 한 문으로 묶는다).
 */
export function GuideGate({ children }: { children: (content: GuideContent) => React.ReactNode }) {
  const guide = useGuide();
  const scoped = useInRetryScope();
  if (guide.status === 'ready' && guide.content) return <>{children(guide.content)}</>;
  const view = guide.status === 'error'
    ? <StateView kind="error" text={GUIDE_FAILED} onRetry={guide.retry} /> : <StateView kind="loading" text={GUIDE_LOADING} />;
  return scoped ? view : <RetryScope focus={focusScreenHeading}>{view}</RetryScope>;
}

/** 짝·검증 규칙 한 줄(D7): 뜻과 읽는 법 뒤에 흐린 글자로. */
function Detail({ text }: { text?: string }) {
  return text ? <Text variant="meta" tone="text3" testID="guide-detail">{`${DETAIL_LABEL} · ${text}`}</Text> : null;
}

function Section({ section }: { section: GuideSection }) {
  return (
    <View style={styles.section}>
      <Text variant="body" weight="700">{section.title}</Text>
      {section.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
      <Detail text={section.detail} />
    </View>
  );
}

/** KPX 원문 같은 바깥 주소(새 창). 진짜 <a href>라서 Enter로 열리고 Tab으로 닿는다. 상자는 44 이상. */
export function ExternalLink({ label, url }: { label: string; url: string }) {
  return (
    <View role="link" style={styles.link} {...web({ href: url, hrefAttrs: { target: '_blank', rel: 'noopener noreferrer' } })}>
      <Text variant="body" weight="600" tone="link" style={styles.underline}>{label}</Text>
    </View>
  );
}

export type LiveFactsProps = { lines: readonly string[]; api?: readonly string[]; link: { label: string; url: string } | null };
/**
 * 이번 화면의 값(KPX 원문 표기·비율·MW 계산 차이·모델 원문). 숫자가 들어 있어 글자는 메타다. api는 API가 스스로 밝힌 수치(정확도)라
 * 회색 "API 표기" 줄로 따로 그린다(7.7절·D8). 값이 없으면 그리지 않는다.
 */
export function LiveFacts({ lines, api = [], link }: LiveFactsProps) {
  if (!lines.length && !api.length && !link) return null;
  return (
    <View style={styles.section} testID="live-facts">
      <Text variant="label" weight="600" tone="text2">{LIVE_FACTS_HEADING}</Text>
      {lines.map(line => <Text key={line} variant="label" tone="text2" dataSet={{ pdMeta: 'fact' }}>{keepTogether(line, { variant: 'label', tone: 'text2' })}</Text>)}
      {api.map(line => <Text key={line} variant="meta" tone="text3" dataSet={{ pdMeta: 'api' }}>{keepTogether(line, { variant: 'meta', tone: 'text3' })}</Text>)}
      {link ? <ExternalLink label={link.label} url={link.url} /> : null}
    </View>
  );
}

/** 산출 방식 시트(?sheet=method&card=): 설명 글 → 이번 화면의 값 → "단순화한 설명 · 도메인 검토 전". */
export function MethodSheetBody({ card, facts }: { card: MethodCardId; facts?: React.ReactNode }) {
  return (
    <View style={styles.body} testID={`method-${card}`}>
      <GuideGate>{content => <Section section={content.cards[card]} />}</GuideGate>
      {facts}
      <Text variant="meta" tone="text3">{SIMPLIFIED_NOTE}</Text>
    </View>
  );
}

/**
 * "이 앱에서 보는 곳: 입찰 ›"(4.4절): 그 용어가 보이는 탭으로 간다. 접근 이름에 용어를 넣는다(round 2: 용어 사전의 링크 13개 중 8개가
 * 같은 이름 "…오늘"이라 링크 목록에서 구별되지 않았다, WCAG 2.4.9).
 */
export function WhereLink({ id }: { id: GlossaryId }) {
  const nav = useNav();
  const key = TERM_WHERE[id];
  const label = whereText(ROUTES[key].tab);
  return <TextLink label={label} accessibilityLabel={`${GLOSSARY_TITLES[id]} · ${label}`} onPress={() => nav.push(rootState(key))} />;
}

/**
 * 용어 한 항목. sheet: 용어 시트(?sheet=term&id=)라서 한 줄 뜻을 굵게 앞에 두고, 지금 탭이 그 용어의 탭이면 "보는 곳"을 빼고,
 * 알아보기 탭이 있으면 "용어 사전 전체 ›"를 단다. 용어 화면(?term=)은 한 줄 뜻을 AnswerLine으로 따로 그린다.
 */
export function TermSheetBody({ id, sheet = true }: { id: GlossaryId; sheet?: boolean }) {
  return <GuideGate>{content => <TermBody content={content} id={id} sheet={sheet} />}</GuideGate>;
}

/** 용어 한 항목의 본문(설명 글을 받은 뒤). 용어 화면(?term=)은 한 줄 뜻과 이것을 한 GuideGate로 묶는다. */
export function TermBody({ content, id, sheet }: { content: GuideContent; id: GlossaryId; sheet: boolean }) {
  const nav = useNav();
  const entry = content.glossary.find(g => g.id === id);
  if (!entry) return <StateView kind="empty" />;
  return (
    <View style={styles.body} testID={`term-${id}`}>
      {sheet ? <Text variant="body" weight="600">{entry.short}</Text> : null}
      {entry.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
      <Detail text={entry.detail} />
      {sheet && TERM_WHERE[id] === nav.state.key ? null : <WhereLink id={id} />}
      {sheet && nav.tabs.includes('guide')
        ? <TextLink label="용어 사전 전체" onPress={() => nav.push({ ...rootState('guide'), topic: 'glossary' })} /> : null}
    </View>
  );
}

/** tone: 수신 상태 pill(응답 상태 행, 5.2절 "채운 pill은 … 수신 상태에만"). 없으면 상태 글자만. */
export type SourceRowItem = {
  id: string; item: string; source: string; basis: string; chip: ChipProps | null; status: string | null; link: string | null; tone?: PillTone;
};

function Status({ row }: { row: SourceRowItem }) {
  if (row.chip) return <ProvenanceChip {...row.chip} />;
  if (row.status && row.tone) return <StatusPill tone={row.tone} text={row.status} />;
  return row.status ? <Text variant="meta" weight="600" tone="text2" dataSet={{ pdMeta: 'status' }}>{row.status}</Text> : null;
}

/** [출처 표]를 누른 카드의 행. 응답 상태는 세 행(status:smp …)이 한 묶음이다. */
const isHit = (id: string, highlight: string | null) => highlight !== null && (id === highlight || id.startsWith(`${highlight}:`));

/** 출처와 기준 시각 표(?sheet=sources, 알아보기 ?topic=sources와 같은 표). highlight는 [출처 표]를 누른 카드의 행. page: 알아보기 화면. */
export function SourcesTable({ rows, highlight, page = false }: { rows: readonly SourceRowItem[]; highlight: string | null; page?: boolean }) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const asTable = width >= SOURCES_TABLE_FROM && (page || width < LAYOUT.railFrom);
  const notes = SOURCES_NOTES.map(n => <Text key={n} variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{n}</Text>);
  if (asTable) {
    return (
      <View style={styles.body} testID="sources-table">
        <DataTable fit caption={SOURCE_HEADINGS.item} columns={[
          { key: 'item', title: SOURCE_HEADINGS.item }, { key: 'source', title: SOURCE_HEADINGS.source, meta: true },
          { key: 'basis', title: SOURCE_HEADINGS.basis, meta: true }, { key: 'status', title: SOURCE_HEADINGS.status },
        ]} rows={rows.map(r => ({
          key: r.id, current: isHit(r.id, highlight),
          cells: { item: r.item, source: r.link ? <View><Text variant="label" dataSet={{ pdMeta: 'cell' }}>{r.source}</Text>
            <ExternalLink label="원문 (새 창)" url={r.link} /></View> : r.source,
          basis: <Text variant="label" dataSet={{ pdMeta: 'cell' }}>{keepTogether(r.basis, { variant: 'label' })}</Text>, status: <Status row={r} /> },
        }))} />
        {notes}
      </View>
    );
  }
  return (
    <View style={styles.body} testID="sources-list">
      {rows.map(r => (
        <View key={r.id} style={[styles.row, { borderBottomColor: theme.color.border }, isHit(r.id, highlight) ? { backgroundColor: theme.color.well } : null]}
          {...web(isHit(r.id, highlight) ? { 'aria-current': 'true', dataSet: { pdRow: r.id } } : { dataSet: { pdRow: r.id } })}>
          <View style={styles.rowHead}>
            <Text variant="body" weight="600" dataSet={labelData(r.item)}>{r.item}</Text>
            <Status row={r} />
          </View>
          <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'source' }}>{keepTogether(`${r.source} · ${r.basis}`, { variant: 'meta', tone: 'text3' })}</Text>
          {r.link ? <ExternalLink label="원문 (새 창)" url={r.link} /> : null}
        </View>
      ))}
      {notes}
    </View>
  );
}

const styles = StyleSheet.create({
  body: { gap: SPACE.md },
  section: { gap: SPACE.sm },
  link: { minHeight: HIT, minWidth: HIT, justifyContent: 'center', alignSelf: 'flex-start' },
  underline: { textDecorationLine: 'underline' },
  row: { gap: SPACE.xs, paddingVertical: SPACE.sm, paddingHorizontal: SPACE.sm, borderBottomWidth: 1, borderRadius: RADIUS.control },
  rowHead: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: SPACE.sm },
});
