// 카드 골격은 하나다(4.0절): 머리(제목 h2 · 칩 · [i]) → 본문 → SourceFooter. I2 검사용 data-pd-card를 단다.
// 모든 카드는 자기 상태를 가진다: 한 카드가 실패해도 StateView가 그 자리를 같은 크기로 채운다.
import React, { useEffect } from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { HIT, LAYOUT, RADIUS, SIZE, SPACE } from './tokens';
import { Button, IconButton } from './Button';
import { ProvenanceChip } from './Chips';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { useAnnounce } from './Live';
import { Text } from './Text';
import { web, webStyle } from './web';
import { useTheme } from '../app/theme';
import { CHIPS } from '../lib/provenance';

export type CardProps = {
  title: string;
  /** 칩 묶음(ProvenanceChip·StatusPill). 360 미만에서는 제목 아래 줄로 내려간다. */
  chips?: React.ReactNode;
  /** 있으면 [i] 버튼("산출 방식 보기: {제목}")을 단다. */
  onMethod?: () => void;
  /** SampleRibbon. 머리보다 먼저 읽힌다. */
  ribbon?: React.ReactNode;
  children?: React.ReactNode;
  testID?: string;
  /** data-state(예: today-hero의 loading·ready·unavailable·error, M9 측정 표지). */
  state?: string;
};

/** 카드 테두리와 안쪽 여백(360 미만 12, 그 밖 16). 차트를 카드 폭으로 넓힐 때(ChartBleed)도 같은 값을 쓴다. */
export const CARD_BORDER = 1;
export const cardPadding = (windowWidth: number) => (windowWidth < LAYOUT.narrowBelow ? SPACE.md : SPACE.lg);

export function Card({ title, chips, onMethod, ribbon, children, testID, state }: CardProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < LAYOUT.narrowBelow;
  const shadow = theme.elevation.card ? webStyle({ boxShadow: theme.elevation.card }) : null;
  return (
    <View testID={testID} style={[styles.card, { backgroundColor: theme.color.surface, borderColor: theme.color.border,
      padding: cardPadding(width) }, shadow]} {...web({ dataSet: state ? { pdCard: 'true', state } : { pdCard: 'true' } })}>
      {ribbon}
      <CardHeader title={title} chips={chips} onMethod={onMethod} narrow={narrow} />
      {children}
    </View>
  );
}

export type CardHeaderProps = Pick<CardProps, 'title' | 'chips' | 'onMethod'> & { narrow?: boolean };
/** 제목(h2 · 15/700) · 칩 · [i](44×44, link 색). */
export function CardHeader({ title, chips, onMethod, narrow = false }: CardHeaderProps) {
  const { theme } = useTheme();
  return (
    <View style={styles.head}>
      <View style={[styles.titleRow, narrow ? styles.stack : null]}>
        <Text role="heading" {...web({ 'aria-level': 2 })} variant="body" weight="700">{title}</Text>
        {chips ? <View style={styles.chips}>{chips}</View> : null}
      </View>
      {onMethod ? <IconButton icon="info" color={theme.color.link} accessibilityLabel={`산출 방식 보기: ${title}`} onPress={onMethod} /> : null}
    </View>
  );
}

export type SourceFooterProps = { source: string; time: string; onSources?: () => void; context?: string };
/** 출처 · 시각 · [출처 표]. 나이 글자는 부모가 60초마다 새 time으로 다시 그린다(알림 없음). */
export function SourceFooter({ source, time, onSources, context }: SourceFooterProps) {
  return (
    <View role="note" style={styles.footer}>
      <Text variant="meta" tone="text3" style={styles.grow} dataSet={{ pdMeta: 'source' }}>
        {source} · <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'time' }}>{time}</Text>
      </Text>
      {onSources ? <Button tone="ghost" label="출처 표" accessibilityLabel={`출처 표: ${context ?? source}`} onPress={onSources} /> : null}
    </View>
  );
}

/** 공개되지 않는 자료(정산·발전소). lock 아이콘은 장식, 버튼은 없다. */
export function RestrictedCard({ title, body, testID }: { title: string; body: string; testID?: string }) {
  const { theme } = useTheme();
  return (
    <Card title={title} chips={<ProvenanceChip chip={CHIPS.locked} />} testID={testID}>
      <View style={styles.row}>
        <Icon name="lock" size={SIZE.icon} color={theme.color.text3} />
        <Text variant="body" tone="text2" style={styles.grow}>{body}</Text>
      </View>
    </Card>
  );
}

export type StateKind = 'loading' | 'empty' | 'error' | 'restricted' | 'stale';
const STATE_TEXT: Readonly<Record<StateKind, string>> = {
  loading: '불러오는 중입니다', empty: '표시할 자료가 없습니다', error: '자료를 불러오지 못했습니다',
  restricted: '공개 데모에서 제공하지 않습니다', stale: '새 자료를 받지 못해 마지막 값을 보여 줍니다',
};
const STATE_ICON: Readonly<Record<Exclude<StateKind, 'loading'>, IconName>> = { empty: 'info', error: 'alert', restricted: 'lock', stale: 'clock' };
const SKELETON_LINES = ['100%', '72%', '48%'] as const;

export type StateViewProps = {
  kind: StateKind;
  text?: string;
  /** 다시 시도가 성공할 수 있을 때만 준다. 제한 상태에서는 주어도 버튼을 그리지 않는다(M8). */
  onRetry?: () => void;
  /** 최종 카드 본문과 같은 높이(레이아웃 점프 0). */
  minHeight?: number;
};

export function StateView({ kind, text, onRetry, minHeight = SIZE.stateMin }: StateViewProps) {
  const { theme } = useTheme();
  const announce = useAnnounce();
  const message = text ?? STATE_TEXT[kind];
  // 오류·지연 글자는 셸의 live region 하나로 알린다(카드마다 live region을 두지 않는다).
  useEffect(() => {
    if (kind === 'error' || kind === 'stale') announce(message);
  }, [kind, message, announce]);
  if (kind === 'loading') {
    return (
      <View aria-busy style={[styles.state, { minHeight }]} {...web({ dataSet: { pdState: 'loading' } })}>
        {SKELETON_LINES.map(w => <View key={w} style={[styles.bone, { width: w, backgroundColor: theme.color.well }]} />)}
        <Text variant="meta" tone="text3">{message}</Text>
      </View>
    );
  }
  const retry = kind === 'restricted' ? undefined : onRetry;
  return (
    <View style={[styles.state, { minHeight }]} {...web({ dataSet: { pdState: kind } })}>
      <View style={styles.row}>
        <Icon name={STATE_ICON[kind]} size={SIZE.icon} color={theme.color.text2} />
        <Text variant="body" tone="text2" style={styles.grow}>{message}</Text>
      </View>
      {retry ? <View style={styles.start}><Button label="다시 시도" accessibilityLabel="다시 시도" onPress={retry} /></View> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: RADIUS.card, borderWidth: CARD_BORDER, gap: SPACE.md, width: '100%', maxWidth: LAYOUT.column, alignSelf: 'center' },
  head: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm },
  titleRow: { flex: 1, flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm, minHeight: HIT },
  stack: { flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACE.xs, maxWidth: '100%' },
  footer: { flexDirection: 'row', alignItems: 'center', gap: SPACE.sm, minHeight: HIT },
  grow: { flex: 1, minWidth: 0 },
  row: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm },
  state: { justifyContent: 'center', gap: SPACE.sm },
  start: { alignSelf: 'flex-start' },
  bone: { height: SIZE.skeletonLine, borderRadius: RADIUS.control },
});
