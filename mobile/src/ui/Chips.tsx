// 상태 칩·pill(5.2절). 상태는 색만으로 전하지 않는다: 글자 + 테두리 모양 + 글리프(또는 아이콘)가 같은 정보를 준다.
// 출처 칩은 외곽선·빗금, 채운 pill은 예비율 단계·접수 상태·수신 상태에만 쓴다.
import { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { RADIUS, SIZE, SPACE } from './tokens';
import type { PillTone } from './tokens';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { Text, keepNegations } from './Text';
import { web } from './web';
import { isOnline, watchOnline } from '../app/head';
import { useTheme } from '../app/theme';
import { CHIPS, chipA11yLabel, chipText } from '../lib/provenance';
import type { Chip } from '../lib/provenance';
import { freshness, nowPillText, staleChipText } from '../content/strings';
import type { NowPillInput } from '../content/strings';
import type { ClockSkew } from '../lib/kst';

/** lag의 text: 조회 나이가 아닌 지연(가격 거래시간 불일치 "지연 · 07h 값", "예측일 지연")에 쓰는 글자. 없으면 조회 시각으로 만든다. */
export type ChipProps = { chip: Chip; source?: string | null } | { lag: 'stale' | 'broken'; at: number; text?: string };

// 누를 수 없는 표지라 role=img + 긴 이름을 준다(글리프 "●"를 "검은 원"으로 읽지 않게). 보이는 글자는 늘 있다.
export function ProvenanceChip(props: ChipProps) {
  const { theme } = useTheme();
  if ('lag' in props) {
    // 지연은 관심 색 + clock, 끊김은 위험 색 + alert, 끊김에서는 실측 글리프를 거둔다(7.6절).
    const tone = theme.pill[props.lag === 'stale' ? 'watch' : 'danger'];
    const text = props.text ?? staleChipText(props.at, props.lag);
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
      <Text variant="meta" weight="600" color={tone.text} style={styles.shrink}>
        {keepNegations(chipText(chip), { variant: 'meta', weight: '600', color: tone.text })}
      </Text>
    </View>
  );
}

/**
 * 예비율 단계·접수 상태·수신 상태: 채움 + 작은 네모(또는 아이콘) + 글자. 글자에 시각이 들어갈 수 있어 메타로 표시한다. 둥근 점(●)은 실측·확정
 * 값의 표시라서 상태 pill에는 쓰지 않는다(round 2 멘토: "● 오늘 마감됨"이 실측 칩처럼 보였다).
 */
export function StatusPill({ tone, text, icon }: { tone: PillTone; text: string; icon?: IconName }) {
  const { theme } = useTheme();
  const c = theme.pill[tone];
  return (
    <View style={[styles.chip, { backgroundColor: c.bg }]} {...web({ dataSet: { pdPill: tone } })}>
      {icon ? <Icon name={icon} size={SIZE.chipIcon} color={c.text} /> : <View style={[styles.dot, { backgroundColor: c.text }]} />}
      <Text variant="meta" weight="600" color={c.text} dataSet={{ pdMeta: 'pill' }} style={styles.shrink}>{text}</Text>
    </View>
  );
}

/**
 * h1 행 오른쪽의 KST 시각 + 조회 나이. 글자 그대로 읽히고 live가 아니다(60초 갱신을 다시 알리지 않는다).
 * 오프라인이면 "06:03 조회 · 오프라인"(위 오프라인 띠와 같은 시각, round 1 멘토: "조회 방금"이 띠와 어긋났다).
 */
export function NowPill(input: NowPillInput) {
  const { theme } = useTheme();
  const [online, setOnline] = useState(isOnline);
  useEffect(() => watchOnline(setOnline), []);
  const stage = input.fetchedAt === null ? 'fresh' : freshness(input.now - input.fetchedAt);
  // 평소에는 옅은 바탕 + 1 px 테두리(round 2 parity: 짙은 남색 캡슐이 h1과 큰 숫자보다 먼저 눈에 띄었다). 채운 색은 지연·끊김에만.
  const fresh = stage === 'fresh';
  const tone = fresh ? { bg: theme.color.well, text: theme.color.text1 } : theme.pill[stage === 'stale' ? 'watch' : 'danger'];
  return (
    <View style={[styles.chip, styles.now, { backgroundColor: tone.bg }, fresh ? { borderWidth: 1, borderColor: theme.color.border } : null]} testID="now-pill"
      {...web({ dataSet: { pdMeta: 'now', pdStage: stage } })}>
      {stage === 'fresh' ? null : <Icon name={stage === 'stale' ? 'clock' : 'alert'} size={SIZE.chipIcon} color={tone.text} />}
      <Text variant="meta" weight="600" color={tone.text}>{nowPillText({ ...input, offline: !online })}</Text>
    </View>
  );
}

/** h1 행 오른쪽 묶음(3.4·7.6절): NowPill + 서버 시계가 타당하지 않으면 ‹서버 시각 확인 필요 · 기기 시계 기준›. 네 화면이 같이 쓴다. */
export function ClockPills({ skew, ...input }: NowPillInput & { skew: ClockSkew | null }) {
  return (
    <View style={styles.pills}>
      <NowPill {...input} />
      {skew && !skew.plausible ? <ProvenanceChip chip={CHIPS.clockImplausible} /> : null}
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
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
  // RN-web은 flex 항목의 flex-shrink를 0으로 둔다: 칩·pill 글자가 줄지 않으면 좁은 표 칸(출처 표의 상태 칸, 768에서 약 140 px)을
  // 넘친다(F10 측정). 줄 수 있게 해 두면 칩은 칸 폭(maxWidth 100%) 안에서 글자를 줄바꿈한다. 자리가 넉넉한 곳의 모양은 그대로다.
  shrink: { flexShrink: 1 },
  dot: { width: SIZE.dot, height: SIZE.dot, borderRadius: RADIUS.mark },
  ribbon: {
    borderWidth: 1, borderStyle: 'dotted', borderRadius: RADIUS.control, paddingHorizontal: SPACE.md, paddingVertical: SPACE.xs,
    alignSelf: 'stretch',
  },
});
