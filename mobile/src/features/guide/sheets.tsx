// 시트 본문(명세 4.4절): 산출 방식(카드별 설명 글 + 화면이 넘기는 이번 화면의 값), 용어 한 개, 출처와 기준 시각 표.
// 시트 틀(dialog·aria-modal·초점 가둠·Esc·복귀·히스토리)은 ui/Sheet와 app/SheetHost가 맡는다. 긴 글은 guide.ko.json에서만 오고,
// 숫자는 화면 값에서만 온다(JSON에는 숫자가 없다). F7은 오늘 화면의 행·값을 넘기고, F8–F10이 입찰·응답 상태 행과 용어 사전 링크를 더한다.
import React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { useGuide } from './useGuide';
import type { GuideContent, GuideSection, MethodCardId } from '../../content/guide';
import { LIVE_FACTS_HEADING, SIMPLIFIED_NOTE, SOURCE_HEADINGS, SOURCES_NOTES } from '../../content/strings';
import type { GlossaryId } from '../../content/strings';
import { useTheme } from '../../app/theme';
import { Button } from '../../ui/Button';
import { StateView } from '../../ui/Card';
import { ProvenanceChip } from '../../ui/Chips';
import type { ChipProps } from '../../ui/Chips';
import { DataTable } from '../../ui/DataTable';
import { Text, labelData } from '../../ui/Text';
import { HIT, LAYOUT, RADIUS, SPACE } from '../../ui/tokens';
import { web } from '../../ui/web';

const GUIDE_LOADING = '설명을 불러오는 중입니다';
const GUIDE_FAILED = '설명을 불러오지 못했습니다';
/** 표 보기는 아래쪽 640 시트(768–1023)에서만. 휴대폰과 ≥ 1024 오른쪽 패널(400)은 두 줄 목록이다. */
const SOURCES_TABLE_FROM = 768;

/** 설명 글이 준비되면 그린다. 읽는 동안 스켈레톤, 실패하면 "설명을 불러오지 못했습니다" + [다시 시도]. */
function GuideGate({ children }: { children: (content: GuideContent) => React.ReactNode }) {
  const guide = useGuide();
  if (guide.status === 'ready' && guide.content) return <>{children(guide.content)}</>;
  if (guide.status === 'error') return <StateView kind="error" text={GUIDE_FAILED} onRetry={guide.retry} />;
  return <StateView kind="loading" text={GUIDE_LOADING} />;
}

function Section({ section }: { section: GuideSection }) {
  return (
    <View style={styles.section}>
      <Text variant="body" weight="700">{section.title}</Text>
      {section.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
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

export type LiveFactsProps = { lines: readonly string[]; link: { label: string; url: string } | null };
/** 이번 화면의 값(KPX 원문 표기·비율·MW 계산 차이·모델 원문). 숫자가 들어 있어 글자는 메타다. 값이 없으면 그리지 않는다. */
export function LiveFacts({ lines, link }: LiveFactsProps) {
  if (!lines.length && !link) return null;
  return (
    <View style={styles.section} testID="live-facts">
      <Text variant="label" weight="600" tone="text2">{LIVE_FACTS_HEADING}</Text>
      {lines.map(line => <Text key={line} variant="label" tone="text2" dataSet={{ pdMeta: 'fact' }}>{line}</Text>)}
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

/** 용어 시트(?sheet=term&id=): 한 줄 뜻 + 풀이. 용어 사전 전체 링크는 알아보기 탭이 생기면(F10) onGlossary로 단다. */
export function TermSheetBody({ id, onGlossary }: { id: GlossaryId; onGlossary?: () => void }) {
  return (
    <GuideGate>
      {content => {
        const entry = content.glossary.find(g => g.id === id);
        if (!entry) return <StateView kind="empty" />;
        return (
          <View style={styles.body} testID={`term-${id}`}>
            <Text variant="body" weight="600">{entry.short}</Text>
            {entry.body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}
            {onGlossary ? <Button tone="ghost" label="용어 사전 전체" accessibilityLabel="용어 사전 전체 보기" onPress={onGlossary} /> : null}
          </View>
        );
      }}
    </GuideGate>
  );
}

export type SourceRowItem = {
  id: string; item: string; source: string; basis: string; chip: ChipProps | null; status: string | null; link: string | null;
};

function Status({ row }: { row: SourceRowItem }) {
  if (row.chip) return <ProvenanceChip {...row.chip} />;
  return row.status ? <Text variant="meta" weight="600" tone="text2" dataSet={{ pdMeta: 'status' }}>{row.status}</Text> : null;
}

/** 출처와 기준 시각 표(?sheet=sources, 알아보기 ?topic=sources와 같은 표). highlight는 [출처 표]를 누른 카드의 행. */
export function SourcesTable({ rows, highlight }: { rows: readonly SourceRowItem[]; highlight: string | null }) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const asTable = width >= SOURCES_TABLE_FROM && width < LAYOUT.railFrom;
  const notes = SOURCES_NOTES.map(n => <Text key={n} variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{n}</Text>);
  if (asTable) {
    return (
      <View style={styles.body} testID="sources-table">
        <DataTable caption={SOURCE_HEADINGS.item} columns={[
          { key: 'item', title: SOURCE_HEADINGS.item }, { key: 'source', title: SOURCE_HEADINGS.source, meta: true },
          { key: 'basis', title: SOURCE_HEADINGS.basis, meta: true }, { key: 'status', title: SOURCE_HEADINGS.status },
        ]} rows={rows.map(r => ({
          key: r.id, current: r.id === highlight,
          cells: { item: r.item, source: r.link ? <View><Text variant="label" dataSet={{ pdMeta: 'cell' }}>{r.source}</Text>
            <ExternalLink label="원문 (새 창)" url={r.link} /></View> : r.source, basis: r.basis, status: <Status row={r} /> },
        }))} />
        {notes}
      </View>
    );
  }
  return (
    <View style={styles.body} testID="sources-list">
      {rows.map(r => (
        <View key={r.id} style={[styles.row, { borderBottomColor: theme.color.border }, r.id === highlight ? { backgroundColor: theme.color.well } : null]}
          {...web(r.id === highlight ? { 'aria-current': 'true', dataSet: { pdRow: r.id } } : { dataSet: { pdRow: r.id } })}>
          <View style={styles.rowHead}>
            <Text variant="body" weight="600" dataSet={labelData(r.item)}>{r.item}</Text>
            <Status row={r} />
          </View>
          <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'source' }}>{`${r.source} · ${r.basis}`}</Text>
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
