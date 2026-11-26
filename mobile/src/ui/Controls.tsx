// 입력 컨트롤(6.2절): 시간 스텝퍼, 세그먼트(radiogroup), 용어 링크, 목록 행. 모두 상자 ≥ 44(M2), 한국어 이름.
import React, { useRef } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { HIT, RADIUS, SIZE, SPACE } from './tokens';
import { IconButton } from './Button';
import { Icon } from './Icon';
import { useAnnounce } from './Live';
import { Text } from './Text';
import { domOf, keyOf, prevent, rovingIndex, web } from './web';
import { useTheme } from '../app/theme';
import { tradeHour } from '../lib/format';

export const FIRST_TRADE_HOUR = 1;
export const LAST_TRADE_HOUR = 24;

export type HourStepperProps = { hour: number; onChange: (hour: number) => void; min?: number; max?: number };
/** 거래시간 ‹ ›. 바뀐 값은 셸 live region 하나로 알린다(값에 aria-live를 따로 두지 않는다, M5 "polite 1개"). */
export function HourStepper({ hour, onChange, min = FIRST_TRADE_HOUR, max = LAST_TRADE_HOUR }: HourStepperProps) {
  const announce = useAnnounce();
  const move = (step: number) => {
    const next = hour + step;
    if (next < min || next > max) return;
    onChange(next);
    announce(tradeHour(next));
  };
  return (
    <View style={styles.stepper}>
      <IconButton icon="back" accessibilityLabel="이전 거래시간" onPress={() => move(-1)} disabled={hour <= min} disabledReason="첫 거래시간입니다" />
      <Text variant="body" weight="600" style={styles.stepValue} dataSet={{ pdMeta: 'hour' }}>{tradeHour(hour)}</Text>
      <IconButton icon="chev" accessibilityLabel="다음 거래시간" onPress={() => move(1)} disabled={hour >= max} disabledReason="마지막 거래시간입니다" />
    </View>
  );
}

export type SegmentOption<T extends string> = { value: T; label: string };
export type SegmentedControlProps<T extends string> = { label: string; options: readonly SegmentOption<T>[]; value: T; onChange: (value: T) => void };
/** role=radiogroup > radio(aria-checked). 화살표·Home·End로 고르고 초점도 옮긴다(roving tabindex). */
export function SegmentedControl<T extends string>({ label, options, value, onChange }: SegmentedControlProps<T>) {
  const { theme } = useTheme();
  const refs = useRef<Array<View | null>>([]);
  const current = Math.max(0, options.findIndex(o => o.value === value));
  const onKey = (e: unknown, index: number) => {
    const next = rovingIndex(keyOf(e), index, options.length, 'both');
    if (next === null) return;
    prevent(e);
    onChange(options[next].value);
    const node = domOf({ current: refs.current[next] });
    if (node) node.focus();
  };
  return (
    <View role="radiogroup" aria-label={label} style={[styles.segment, { borderColor: theme.color.borderStrong }]}>
      {options.map((o, i) => {
        const on = i === current;
        return (
          <Pressable key={o.value} ref={el => { refs.current[i] = el; }} role="radio" aria-checked={on} aria-label={o.label}
            onPress={() => onChange(o.value)} style={[styles.segItem, on ? { backgroundColor: theme.color.primary } : null]}
            {...web({ tabIndex: on ? 0 : -1, onKeyDown: (e: unknown) => onKey(e, i) })}>
            <Text variant="label" weight="600" color={on ? theme.color.onPrimary : theme.color.text2}>{o.label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

/** 용어 시트를 여는 링크. 시각적으로 작아도 상자는 44. */
export function TermLink({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable role="link" aria-label={`용어 풀이: ${label}`} onPress={onPress} style={styles.term}>
      <Text variant="body" weight="500" tone="link" style={styles.underline}>{label}</Text>
    </Pressable>
  );
}

/** 알아보기 목록 행(높이 56). 이름은 보이는 글자 전체. */
export function ListRow({ title, detail, onPress }: { title: string; detail?: string; onPress: () => void }) {
  const { theme } = useTheme();
  return (
    <Pressable role="link" aria-label={detail ? `${title}, ${detail}` : title} onPress={onPress}
      style={[styles.listRow, { borderBottomColor: theme.color.border }]}>
      <View style={styles.grow}>
        <Text variant="body" weight="600">{title}</Text>
        {detail ? <Text variant="label" tone="text2">{detail}</Text> : null}
      </View>
      <Icon name="chev" size={SIZE.icon} color={theme.color.text3} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  stepper: { flexDirection: 'row', alignItems: 'center', gap: SPACE.xs },
  stepValue: { minWidth: HIT * 3, textAlign: 'center' },
  // overflow:hidden을 쓰지 않는다: 가장자리 칸의 포커스 링(간격 2 px)이 잘린다.
  segment: { flexDirection: 'row', borderWidth: 1, borderRadius: RADIUS.control, alignSelf: 'flex-start', maxWidth: '100%' },
  segItem: {
    minHeight: HIT, minWidth: HIT, paddingHorizontal: SPACE.lg, borderRadius: RADIUS.control, alignItems: 'center', justifyContent: 'center',
    flexShrink: 1,
  },
  term: { minHeight: HIT, minWidth: HIT, justifyContent: 'center', alignSelf: 'flex-start' },
  underline: { textDecorationLine: 'underline' },
  listRow: { minHeight: SIZE.listRow, flexDirection: 'row', alignItems: 'center', gap: SPACE.md, borderBottomWidth: 1, paddingVertical: SPACE.sm },
  grow: { flex: 1, minWidth: 0 },
});
