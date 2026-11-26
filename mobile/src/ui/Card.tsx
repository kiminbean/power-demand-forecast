// 카드 골격은 하나다(4.0절): 머리(제목 h2 · 칩 · [i]) → 본문 → SourceFooter. I2 검사용 data-pd-card를 단다.
// 모든 카드는 자기 상태를 가진다: 한 카드가 실패해도 StateView가 그 자리를 같은 크기로 채운다.
// 카드 제목은 tabIndex=-1이라 초점을 받을 수 있다(round 1 D4): [다시 시도]·[다시 계산]을 누르면 버튼이 사라지기 전에 초점을 제목으로 옮기고,
// 다시 받은 뒤에는 "다시 불러왔습니다 · HH:MM KST"를 알린다(초점이 body로 떨어지지 않는다).
import { createContext, useContext, useEffect, useMemo, useRef } from 'react';
import type React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { Text as RNText } from 'react-native';
import { HIT, LAYOUT, RADIUS, SIZE, SPACE, skeletonBucket } from './tokens';
import { Button, IconButton } from './Button';
import { ProvenanceChip } from './Chips';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { useAnnounce } from './Live';
import { Text, keepTogether, labelData } from './Text';
import { focusRef, web, webStyle } from './web';
import { useClock } from '../app/Shell';
import { useTheme } from '../app/theme';
import { kstNow } from '../lib/kst';
import { CHIPS } from '../lib/provenance';
import { retryDoneText } from '../content/strings';

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
  /** 두 칸 줄(768–1023 오늘의 가격·전력, D14)에서 옆 카드와 같은 높이로 늘인다. */
  fill?: boolean;
};

/** 카드 테두리와 안쪽 여백(360 미만 12, 그 밖 16). 차트를 카드 폭으로 넓힐 때(ChartBleed)도 같은 값을 쓴다. */
export const CARD_BORDER = 1;
export const cardPadding = (windowWidth: number) => (windowWidth < LAYOUT.narrowBelow ? SPACE.md : SPACE.lg);

/**
 * [다시 시도]가 눌렸음을 기억했다가(arm) 그 StateView가 끝나면(report) 결과를 알린다. 카드는 제목(h2), 시트는 시트 제목, 카드 밖 설명 글
 * (알아보기 화면의 GuideGate)은 화면 h1이 초점 자리다(round 2 M5·FOCUS: 카드 밖 [다시 시도]는 초점이 body로 떨어지고 알림이 없었다).
 * 오늘 화면의 전부 실패 안내는 화면이 따로 맡는다(범위 밖).
 */
type RetryFocus = { arm(): void; report(kind: StateKind | 'gone'): void };
const RetryContext = createContext<RetryFocus | null>(null);
/** 가장 가까운 초점 자리(카드·시트·화면 범위)가 이미 있는가. */
export const useInRetryScope = () => useContext(RetryContext) !== null;

function useRetryFocus(focusTarget: () => void): RetryFocus {
  const announce = useAnnounce();
  const { skew } = useClock();
  const armed = useRef(false);
  const latest = useRef({ announce, skew, focusTarget });
  latest.current = { announce, skew, focusTarget };
  return useMemo<RetryFocus>(() => ({
    arm: () => {
      armed.current = true;
      latest.current.focusTarget();
    },
    report: kind => {
      if (!armed.current || kind === 'loading') return;
      armed.current = false;
      // 다시 실패하면 StateView가 오류 글자를 알린다. 성공(내용·빈 값)이면 옛 오류 글자를 이 알림으로 바꾼다.
      if (kind !== 'error') latest.current.announce(retryDoneText(kstNow(latest.current.skew)));
    },
  }), []);
}

/** 이 범위 안 StateView의 [다시 시도]가 초점을 focus()로 옮기고, 다시 받으면 "다시 불러왔습니다 · HH:MM KST"를 알린다. */
export function RetryScope({ focus, children }: { focus: () => void; children?: React.ReactNode }) {
  return <RetryContext.Provider value={useRetryFocus(focus)}>{children}</RetryContext.Provider>;
}

export function Card({ title, chips, onMethod, ribbon, children, testID, state, fill = false }: CardProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < LAYOUT.narrowBelow;
  const shadow = theme.elevation.card ? webStyle({ boxShadow: theme.elevation.card }) : null;
  const titleRef = useRef<RNText>(null);
  return (
    <View testID={testID} style={[styles.card, { backgroundColor: theme.color.surface, borderColor: theme.color.border,
      padding: cardPadding(width) }, shadow, fill ? styles.fill : null]} {...web({ dataSet: state ? { pdCard: 'true', state } : { pdCard: 'true' } })}>
      {ribbon}
      <CardHeader title={title} chips={chips} onMethod={onMethod} narrow={narrow} titleRef={titleRef} />
      <RetryScope focus={() => focusRef(titleRef)}>{children}</RetryScope>
    </View>
  );
}

export type CardHeaderProps = Pick<CardProps, 'title' | 'chips' | 'onMethod'> & { narrow?: boolean; titleRef?: React.Ref<RNText> };
/**
 * 제목(h2 · 15/700) · 칩 · [i](44×44, link 색). 머리 줄은 칩 높이(24)만 차지하고 [i]의 44×44 상자는 위아래로 10씩 카드 안쪽 여백과
 * 머리 아래 간격(12)으로 내민다: DOM 상자는 그대로 44(M2)이고, 다른 누름 대상과 겹치지 않으며, 카드마다 20 px을 아낀다(4.1절 첫 화면).
 */
export function CardHeader({ title, chips, onMethod, narrow = false, titleRef }: CardHeaderProps) {
  const { theme } = useTheme();
  return (
    <View style={styles.head}>
      <View style={[styles.titleRow, narrow ? styles.stack : null]}>
        <Text ref={titleRef} role="heading" {...web({ 'aria-level': 2, tabIndex: -1 })} variant="body" weight="700" dataSet={labelData(title)}>{title}</Text>
        {chips ? <View style={styles.chips}>{chips}</View> : null}
      </View>
      {onMethod ? (
        <View style={styles.overhang}>
          <IconButton icon="info" color={theme.color.link} accessibilityLabel={`산출 방식 보기: ${title}`} onPress={onMethod} />
        </View>
      ) : null}
    </View>
  );
}

const META3 = { variant: 'meta', tone: 'text3' } as const;
/** extra: 같은 카드의 둘째 출처(전력 카드의 "날씨 기상청 초단기실황 (API 서버)"). 시각 뒤에 " · "로 잇는다. */
export type SourceFooterProps = { source: string; time: string; onSources?: () => void; context?: string; extra?: string };
/** 출처 · 시각 · [출처 표]. 나이 글자는 부모가 60초마다 새 time으로 다시 그린다(알림 없음). */
export function SourceFooter({ source, time, onSources, context, extra }: SourceFooterProps) {
  return (
    <View role="note" style={styles.footer}>
      <Text variant="meta" tone="text3" style={styles.grow} dataSet={{ pdMeta: 'source' }}>
        {keepTogether(source, META3)} · <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'time' }}>{keepTogether(time, META3)}</Text>
        {extra ? <>{' · '}{keepTogether(extra, META3)}</> : null}
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
  /** 버튼 글자(기본 "다시 시도"). 입찰 계산은 "다시 계산"(4.2절). */
  retryLabel?: string;
  /** 최종 카드 본문과 같은 높이(레이아웃 점프 0). 폭 구간 표(tokens SKELETON)를 주면 그 화면 폭의 값을 쓴다(round 2 CLS). */
  minHeight?: number | readonly number[];
};

export function StateView({ kind, text, onRetry, retryLabel = '다시 시도', minHeight: height = SIZE.stateMin }: StateViewProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const minHeight = typeof height === 'number' ? height : height[skeletonBucket(width)];
  const announce = useAnnounce();
  const retryFocus = useContext(RetryContext);
  const message = text ?? STATE_TEXT[kind];
  // 오류·지연 글자는 셸의 live region 하나로 알린다(카드마다 live region을 두지 않는다).
  useEffect(() => {
    if (kind === 'error' || kind === 'stale') announce(message);
  }, [kind, message, announce]);
  // 다시 시도 뒤의 결과(D4): 불러오는 중은 건너뛰고, 오류면 위 알림, 내용으로 바뀌면(이 StateView가 사라지면) "다시 불러왔습니다".
  useEffect(() => {
    if (retryFocus) retryFocus.report(kind);
  }, [kind, retryFocus]);
  useEffect(() => () => {
    if (retryFocus) retryFocus.report('gone');
  }, [retryFocus]);
  if (kind === 'loading') {
    return (
      <View aria-busy style={[styles.state, { minHeight }]} {...web({ dataSet: { pdState: 'loading' } })}>
        {SKELETON_LINES.map(w => <View key={w} style={[styles.bone, { width: w, backgroundColor: theme.color.well }]} />)}
        <Text variant="meta" tone="text3">{message}</Text>
      </View>
    );
  }
  const given = kind === 'restricted' ? undefined : onRetry;
  // 누르면 버튼이 곧 스켈레톤으로 바뀐다: 그 전에 초점을 카드 제목으로 옮긴다(D4). 카드 밖(전부 실패 안내)은 화면의 onRetry가 h1로 옮긴다.
  const retry = given ? () => {
    if (retryFocus) retryFocus.arm();
    given();
  } : undefined;
  // 오류 글자에는 KST 시각이 들어간다("… · 06:04 KST"): 숫자가 있으면 메타로 둔다(I2).
  return (
    <View style={[styles.state, { minHeight }]} {...web({ dataSet: { pdState: kind } })}>
      <View style={styles.row}>
        <Icon name={STATE_ICON[kind]} size={SIZE.icon} color={theme.color.text2} />
        <Text variant="body" tone="text2" style={styles.grow} dataSet={/\d/.test(message) ? { pdMeta: 'state' } : undefined}>{message}</Text>
      </View>
      {retry ? <View style={styles.start}><Button label={retryLabel} accessibilityLabel={retryLabel} onPress={retry} /></View> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: RADIUS.card, borderWidth: CARD_BORDER, gap: SPACE.md, width: '100%', maxWidth: LAYOUT.column, alignSelf: 'center' },
  fill: { flexGrow: 1 },
  head: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm },
  titleRow: { flex: 1, flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm, minHeight: SIZE.chip },
  overhang: { marginVertical: -(HIT - SIZE.chip) / 2 },
  stack: { flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACE.xs, maxWidth: '100%' },
  footer: { flexDirection: 'row', alignItems: 'center', gap: SPACE.sm, minHeight: HIT },
  grow: { flex: 1, minWidth: 0 },
  row: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm },
  state: { justifyContent: 'center', gap: SPACE.sm },
  start: { alignSelf: 'flex-start' },
  bone: { height: SIZE.skeletonLine, borderRadius: RADIUS.control },
});
