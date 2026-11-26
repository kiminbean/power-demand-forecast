// 상태 칩·pill(5.2절). 상태는 색만으로 전하지 않는다: 글자 + 테두리 모양 + 글리프(또는 아이콘)가 같은 정보를 준다.
// 출처 칩은 외곽선·빗금, 채운 pill은 예비율 단계·접수 상태·수신 상태에만 쓴다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { RADIUS, SIZE, SPACE } from './tokens';
import type { PillTone } from './tokens';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { Text, keepNegations } from './Text';
import { web } from './web';
import { useTheme } from '../app/theme';
import { CHIPS, chipA11yLabel, chipText } from '../lib/provenance';
import type { Chip } from '../lib/provenance';
import { freshness, nowPillText, staleChipText } from '../content/strings';
import type { NowPillInput } from '../content/strings';

export type ChipProps = { chip: Chip; source?: string | null } | { lag: 'stale' | 'broken'; at: number };

// 누를 수 없는 표지라 role=img + 긴 이름을 준다(글리프 "●"를 "검은 원"으로 읽지 않게). 보이는 글자는 늘 있다.
export function ProvenanceChip(props: ChipProps) {
  const { theme } = useTheme();
  if ('lag' in props) {
    // 지연은 관심 색 + clock, 끊김은 위험 색 + alert, 끊김에서는 실측 글리프를 거둔다(7.6절).
    const tone = theme.pill[props.lag === 'stale' ? 'watch' : 'danger'];
    const text = staleChipText(props.at, props.lag);
    return (
      <View role="img" aria-label={`상태: ${text}`} style={[styles.chip, { backgroundColor: tone.bg }]} {...web({ dataSet: { pdChip: props.lag } })}>
        <Icon name={props.lag === 'stale' ? 'clock' : 'alert'} size={SIZE.chipIcon} color={tone.text} />
        <Text variant="meta" weight="600" color={tone.text} nowrap>{text}</Text>
      </View>
    );
  }
  const { chip, source } = props;
  const tone = theme.chip[chip.kind];
  const outline = tone.border ? { borderWidth: 1, borderColor: tone.border, borderStyle: tone.borderStyle } : null;
  const data = tone.hatch ? { pdChip: chip.kind, pdHatch: 'est' } : { pdChip: chip.kind };
  return (
    <View role="img" aria-label={chipA11yLabel(chip, source)} style={[styles.chip, { backgroundColor: tone.bg }, outline]} {...web({ dataSet: data })}>
      {chip.kind === 'plain' && chip.text === CHIPS.locked.text ? <Icon name="lock" size={SIZE.chipIcon} color={tone.text} /> : null}
      <Text variant="meta" weight="600" color={tone.text}>{keepNegations(chipText(chip), { variant: 'meta', weight: '600', color: tone.text })}</Text>
    </View>
  );
}

/** 예비율 단계·접수 상태: 채움 + 색 점(또는 아이콘) + 글자. 글자에 시각이 들어갈 수 있어 메타로 표시한다. */
export function StatusPill({ tone, text, icon }: { tone: PillTone; text: string; icon?: IconName }) {
  const { theme } = useTheme();
  const c = theme.pill[tone];
  return (
    <View style={[styles.chip, { backgroundColor: c.bg }]} {...web({ dataSet: { pdPill: tone } })}>
      {icon ? <Icon name={icon} size={SIZE.chipIcon} color={c.text} /> : <View style={[styles.dot, { backgroundColor: c.text }]} />}
      <Text variant="meta" weight="600" color={c.text} dataSet={{ pdMeta: 'pill' }}>{text}</Text>
    </View>
  );
}

/** h1 행 오른쪽의 KST 시각 + 조회 나이. 글자 그대로 읽히고 live가 아니다(60초 갱신을 다시 알리지 않는다). */
export function NowPill(input: NowPillInput) {
  const { theme } = useTheme();
  const stage = input.fetchedAt === null ? 'fresh' : freshness(input.now - input.fetchedAt);
  const tone = stage === 'fresh' ? { bg: theme.color.nowPillBg, text: theme.color.nowPillText } : theme.pill[stage === 'stale' ? 'watch' : 'danger'];
  return (
    <View style={[styles.chip, styles.now, { backgroundColor: tone.bg }]} testID="now-pill" {...web({ dataSet: { pdMeta: 'now', pdStage: stage } })}>
      {stage === 'fresh' ? null : <Icon name={stage === 'stale' ? 'clock' : 'alert'} size={SIZE.chipIcon} color={tone.text} />}
      <Text variant="meta" weight="600" color={tone.text}>{nowPillText(input)}</Text>
    </View>
  );
}

/** ┆예시 · 실제 값 아님┆ 카드 머리 리본. 카드의 첫 자식으로 두어 본문보다 먼저 읽힌다. */
export function SampleRibbon({ chip = CHIPS.exampleRibbon }: { chip?: Chip }) {
  const { theme } = useTheme();
  const t = theme.chip.example;
  return (
    <View style={[styles.ribbon, { backgroundColor: t.bg, borderColor: t.border ?? t.bg }]} {...web({ dataSet: { pdChip: 'example', pdRibbon: 'true' } })}>
      <Text variant="meta" weight="600" color={t.text}>{keepNegations(chipText(chip), { variant: 'meta', weight: '600', color: t.text })}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  chip: {
    minHeight: SIZE.chip, paddingHorizontal: SPACE.sm, borderRadius: RADIUS.pill, flexDirection: 'row', alignItems: 'center',
    alignSelf: 'flex-start', gap: SPACE.xs, maxWidth: '100%',
  },
  now: { paddingHorizontal: SPACE.md - SPACE.xs },
  dot: { width: SIZE.dot, height: SIZE.dot, borderRadius: RADIUS.pill },
  ribbon: {
    borderWidth: 1, borderStyle: 'dotted', borderRadius: RADIUS.control, paddingHorizontal: SPACE.md, paddingVertical: SPACE.xs,
    alignSelf: 'stretch',
  },
});
