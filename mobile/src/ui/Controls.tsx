// 입력 컨트롤(6.2절): 시간 스텝퍼, 세그먼트(radiogroup), 용어 링크, 목록 행. 모두 상자 ≥ 44(M2), 한국어 이름.
import { useRef } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { HIT, RADIUS, SIZE, SPACE } from './tokens';
import { IconButton } from './Button';
import { Icon } from './Icon';
import { useAnnounce } from './Live';
import { Text, labelData } from './Text';
import { domOf, keyOf, prevent, rovingIndex, web } from './web';
import { useTheme } from '../app/theme';
import { tradeHour } from '../lib/format';
import { WRAP_FIRST, WRAP_LAST } from '../content/strings';

export const FIRST_TRADE_HOUR = 1;
export const LAST_TRADE_HOUR = 24;

export type HourStepperProps = {
  hour: number; onChange: (hour: number) => void; min?: number; max?: number;
  /** 오늘 화면(4.1절): 24h 다음은 01h, 01h 이전은 24h로 돌며 "(처음으로)"·"(끝으로)"를 함께 알린다. 없으면 양 끝에서 멈춘다. */
  wrap?: boolean;
  /** 값 글자(기본 "07h · 06–07시"). 오늘 화면은 지금 거래시간에 " · 지금"을 붙인다. 알림도 같은 글자다. */
  label?: (hour: number) => string;
};
/** 거래시간 ‹ ›. 바뀐 값은 셸 live region 하나로 알린다(값에 aria-live를 따로 두지 않는다, M5 "polite 1개"). */
export function HourStepper({ hour, onChange, min = FIRST_TRADE_HOUR, max = LAST_TRADE_HOUR, wrap = false, label = tradeHour }: HourStepperProps) {
  const announce = useAnnounce();
  const move = (step: number) => {
    let next = hour + step;
    let note = '';
    if (next < min || next > max) {
      if (!wrap) return;
      next = next < min ? max : min;
      note = ` ${next === min ? WRAP_FIRST : WRAP_LAST}`;
    }
    onChange(next);
    announce(`${label(next)}${note}`);
  };
  return (
    <View style={styles.stepper}>
      <IconButton icon="back" accessibilityLabel="이전 거래시간" onPress={() => move(-1)} disabled={!wrap && hour <= min} disabledReason="첫 거래시간입니다" />
      <Text variant="body" weight="600" style={styles.stepValue} dataSet={{ pdMeta: 'hour' }}>{label(hour)}</Text>
      <IconButton icon="chev" accessibilityLabel="다음 거래시간" onPress={() => move(1)} disabled={!wrap && hour >= max} disabledReason="마지막 거래시간입니다" />
    </View>
  );
}

export type SegmentOption<T extends string> = { value: T; label: string };
export type SegmentedControlProps<T extends string> = {
  label: string; options: readonly SegmentOption<T>[]; value: T; onChange: (value: T) => void;
  /** 긴 선택지(테마 시트)는 위아래로 쌓는다. 키는 네 방향 모두 같다. */
  vertical?: boolean;
};
/** role=radiogroup > radio(aria-checked). 화살표·Home·End로 고르고 초점도 옮긴다(roving tabindex). */
export function SegmentedControl<T extends string>({ label, options, value, onChange, vertical = false }: SegmentedControlProps<T>) {
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
    <View role="radiogroup" aria-label={label} style={[styles.segment, vertical ? styles.segmentV : null, { borderColor: theme.color.borderStrong }]}>
      {options.map((o, i) => {
        const on = i === current;
        return (
          <Pressable key={o.value} ref={el => { refs.current[i] = el; }} role="radio" aria-checked={on} aria-label={o.label}
            onPress={() => onChange(o.value)}
            style={[styles.segItem, vertical ? styles.segItemV : null, on ? { backgroundColor: theme.color.primary } : null]}
            {...web({ tabIndex: on ? 0 : -1, onKeyDown: (e: unknown) => onKey(e, i) })}>
            <Text variant="label" weight="600" color={on ? theme.color.onPrimary : theme.color.text2}>{o.label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

/**
 * react-native-web 0.19.13은 role=link인 div에서 Enter를 브라우저 링크 클릭에 맡기는데 div에는 클릭이 생기지 않아 열리지 않는다
 * (F6 탐침): 링크는 Enter를 직접 받는다(링크라서 Space로는 열지 않는다).
 */
const enterOpens = (onPress: () => void) => web({
  onKeyDown: (e: unknown) => {
    if (keyOf(e) !== 'Enter') return;
    prevent(e);
    onPress();
  },
});

/** 글자 링크(화면 안 이동: "이 앱에서 보는 곳: 입찰 ›", "용어 사전 전체 ›"). 시각적으로 작아도 상자는 44. 이름은 보이는 글자 그대로. */
/** accessibilityLabel: 같은 글자의 링크가 여럿일 때 구별되는 이름(보이는 글자를 품는다, WCAG 2.5.3). */
export function TextLink({ label, onPress, accessibilityLabel }: { label: string; onPress: () => void; accessibilityLabel?: string }) {
  const { theme } = useTheme();
  return (
    <Pressable role="link" aria-label={accessibilityLabel ?? label} onPress={onPress} style={styles.term} {...enterOpens(onPress)}>
      <View style={styles.linkRow}>
        <Text variant="body" weight="500" tone="link" style={styles.underline} dataSet={labelData(label)}>{label}</Text>
        <Icon name="chev" size={SIZE.chipIcon} color={theme.color.link} />
      </View>
    </Pressable>
  );
}

/** 용어 시트를 여는 링크. 라벨에 숫자가 있을 수 있어("80% 구간") 글자를 data-pd-meta="term"으로 둔다(I2). */
export function TermLink({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable role="link" aria-label={`용어 풀이: ${label}`} onPress={onPress} style={styles.term} {...enterOpens(onPress)}>
      <Text variant="body" weight="500" tone="link" style={styles.underline} dataSet={{ pdMeta: 'term' }}>{label}</Text>
    </Pressable>
  );
}

/** 알아보기 목록 행(높이 56). 이름은 보이는 글자 전체. 숫자가 든 글자("낱말 13개", "거래시간(01h–24h)")는 이름표라 메타로 둔다. */
export function ListRow({ title, detail, onPress, last = false }: { title: string; detail?: string; onPress: () => void; last?: boolean }) {
  const { theme } = useTheme();
  return (
    <Pressable role="link" aria-label={detail ? `${title}, ${detail}` : title} onPress={onPress} {...enterOpens(onPress)}
      style={[styles.listRow, { borderBottomColor: last ? 'transparent' : theme.color.border }]}>
      <View style={styles.grow}>
        <Text variant="body" weight="600" dataSet={labelData(title)}>{title}</Text>
        {detail ? <Text variant="label" tone="text2" dataSet={labelData(detail)}>{detail}</Text> : null}
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
  segmentV: { flexDirection: 'column', alignSelf: 'stretch' },
  segItemV: { alignItems: 'flex-start' },
  // 44보다 짧은 글자("정산")는 상자 가운데에 둔다: 왼쪽에 붙이면 다음 링크와의 간격만 넓어 보인다(F9 정산 용어 줄).
  term: { minHeight: HIT, minWidth: HIT, justifyContent: 'center', alignItems: 'center', alignSelf: 'flex-start' },
  underline: { textDecorationLine: 'underline' },
  linkRow: { flexDirection: 'row', alignItems: 'center', gap: SPACE.xs },
  listRow: { minHeight: SIZE.listRow, flexDirection: 'row', alignItems: 'center', gap: SPACE.md, borderBottomWidth: 1, paddingVertical: SPACE.sm },
  grow: { flex: 1, minWidth: 0 },
});
