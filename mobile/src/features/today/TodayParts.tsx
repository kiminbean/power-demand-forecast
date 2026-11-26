// 오늘 화면 조각(명세 4.1·4.6절): 칩 묶음, 용어 줄, 전부 실패 안내, ≥ 1280 "이 화면 읽는 법" 패널 본문.
// 글자와 상태는 model.ts가 정하고 여기서는 배치만 한다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import type { ChipSpec, TodayLegend } from './model';
import { CHIPS } from '../../lib/provenance';
import type { Chip } from '../../lib/provenance';
import {
  LEGEND_ESTIMATED, LEGEND_EXAMPLE, PANEL_HEADINGS, TERMS_HEADING, TODAY_PANEL_METHOD, TODAY_TERMS, legendForecastText,
  legendMeasuredText,
} from '../../content/strings';
import type { GlossaryId } from '../../content/strings';
import { useTheme } from '../../app/theme';
import { StateView } from '../../ui/Card';
import { ProvenanceChip } from '../../ui/Chips';
import { TermLink } from '../../ui/Controls';
import { Text, keepTogether } from '../../ui/Text';
import { RADIUS, SPACE } from '../../ui/tokens';
import { web } from '../../ui/web';

export function ChipList({ chips }: { chips: readonly ChipSpec[] }) {
  if (!chips.length) return null;
  return <>{chips.map((c, i) => <ProvenanceChip key={'lag' in c ? `lag${i}` : `${c.chip.text}${i}`} {...c} />)}</>;
}

type TermRowProps = { onOpen: (id: GlossaryId) => void; terms?: typeof TODAY_TERMS; testID?: string };
/** 용어 줄(4.1절): 각 높이 44, 줄바꿈 허용. 누르면 ?sheet=term&id=… 정산 화면(4.3절)도 자기 용어로 쓴다. */
export function TermRow({ onOpen, terms = TODAY_TERMS, testID = 'today-terms' }: TermRowProps) {
  return (
    <View style={styles.terms} testID={testID}>
      <Text variant="label" weight="600" tone="text2">{TERMS_HEADING}</Text>
      {terms.map(t => <TermLink key={t.id} label={t.label} onPress={() => onOpen(t.id)} />)}
    </View>
  );
}

/**
 * 세 요청(가격·예측·전력)이 모두 실패(4.1절 "전부 실패"): 원인 + [다시 시도]. 값이 하나도 없으면 AnswerLine이 "지금은 시장 정보를
 * 불러올 수 없습니다."라서 원인만, 이전 값이 남았으면 "지금은 시장 정보를 불러올 수 없습니다 · {원인}"(글자는 model.ts가 정한다).
 * 카드가 하나도 없을 때는 이것이 영웅 자리라서 today-hero(data-state="error")를 단다(M9 측정 표지).
 */
export function AllFailedNotice({ text, onRetry, hero }: { text: string; onRetry: () => void; hero: boolean }) {
  const { theme } = useTheme();
  return (
    <View style={[styles.notice, { backgroundColor: theme.color.surface, borderColor: theme.color.border }]}
      testID={hero ? 'today-hero' : 'today-all-failed'} {...web({ dataSet: { state: 'error' } })}>
      <StateView kind="error" text={text} onRetry={onRetry} />
    </View>
  );
}

export type PanelBodyProps = {
  legend: ReadonlyArray<{ chip: Chip; text: string }>;
  terms: ReadonlyArray<{ id: GlossaryId; label: string }>;
  method: readonly string[];
  onTerm: (id: GlossaryId) => void;
  testID: string;
  /** 산출 방식 요약 앞의 그림(정산: ①+② 식을 그림으로, 4.3절). */
  figure?: React.ReactNode;
};
/** ≥ 1280 "이 화면 읽는 법" 패널 본문(4.6절): 상태 범례(그 화면의 실제 값 예) · 용어 3개 · 산출 방식 요약 3줄. 입찰·정산 화면도 쓴다. */
export function ReadingPanelBody({ legend, terms, method, onTerm, testID, figure }: PanelBodyProps) {
  return (
    <View style={styles.panel} testID={testID}>
      <Text variant="label" weight="600" tone="text2">{PANEL_HEADINGS.legend}</Text>
      {legend.map(l => <LegendRow key={l.chip.text} chip={l.chip} text={l.text} />)}
      <Text variant="label" weight="600" tone="text2">{PANEL_HEADINGS.terms}</Text>
      <View style={styles.terms}>
        {terms.map(t => <TermLink key={t.id} label={t.label} onPress={() => onTerm(t.id)} />)}
      </View>
      <Text variant="label" weight="600" tone="text2">{PANEL_HEADINGS.method}</Text>
      {figure}
      {method.map(m => <Text key={m} variant="label" tone="text2" dataSet={{ pdMeta: 'method' }}>{m}</Text>)}
    </View>
  );
}

/** 범례 한 줄: 칩 + 그 표시의 이번 화면 값 또는 설명(값이 있으면 숫자는 메타). 패널과 둘러보기 5단계가 쓴다. */
export function LegendRow({ chip, text }: { chip: Chip; text: string }) {
  return (
    <View style={styles.legendRow}>
      <ProvenanceChip chip={chip} />
      <Text variant="label" tone="text2" style={styles.grow} dataSet={{ pdMeta: 'legend' }}>{keepTogether(text, { variant: 'label', tone: 'text2' })}</Text>
    </View>
  );
}

/** 오늘 패널: 4상태 범례(이 화면의 실제 값으로, 없으면 표시와 설명만) · 용어 3개 · 산출 방식 요약 3줄. */
export function TodayPanelBody({ legend: values, onTerm }: { legend: TodayLegend; onTerm: (id: GlossaryId) => void }) {
  const legend = [
    { chip: CHIPS.measuredConfirmed, text: legendMeasuredText(values.demand, values.confirmed) },
    { chip: CHIPS.forecastBaseline, text: legendForecastText(values.forecast) },
    { chip: CHIPS.estNotMeasured, text: LEGEND_ESTIMATED },
    { chip: CHIPS.example, text: LEGEND_EXAMPLE },
  ];
  const terms = TODAY_TERMS.filter(t => t.id === 'smp' || t.id === 'reserve' || t.id === 'band');
  return <ReadingPanelBody legend={legend} terms={terms} method={TODAY_PANEL_METHOD} onTerm={onTerm} testID="today-panel" />;
}

const styles = StyleSheet.create({
  terms: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: SPACE.md },
  notice: { borderWidth: 1, borderRadius: RADIUS.card, padding: SPACE.lg, width: '100%' },
  panel: { gap: SPACE.sm },
  legendRow: { flexDirection: 'row', alignItems: 'center', gap: SPACE.sm },
  grow: { flex: 1, minWidth: 0 },
});
