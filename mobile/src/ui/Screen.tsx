// 화면 골격(4.0절): h1 행(제목 20/700 + NowPill) → AnswerLine → 카드들. 내용 영역이 role=main(<main>)이고
// h1은 tabIndex=-1이라 라우트가 바뀌면 셸(F6)이 headingRef로 초점을 옮긴다. 여백 16(360 미만 12), 아래 24, 본문 열 최대 640.
import React from 'react';
import { ScrollView, StyleSheet, View, useWindowDimensions } from 'react-native';
import type { Text as RNText } from 'react-native';
import { LAYOUT, SPACE, TYPE_SCALE } from './tokens';
import { Button } from './Button';
import { Text, keepTogether, labelData } from './Text';
import { web } from './web';
import { useTheme } from '../app/theme';

export type ScreenProps = {
  title: string;
  /** h1 오른쪽(NowPill). */
  pill?: React.ReactNode;
  /** AnswerLine. */
  answer?: React.ReactNode;
  children?: React.ReactNode;
  headingRef?: React.Ref<RNText>;
  scrollRef?: React.Ref<ScrollView>;
  testID?: string;
};

export function Screen({ title, pill, answer, children, headingRef, scrollRef, testID }: ScreenProps) {
  const { width } = useWindowDimensions();
  const gutter = width < LAYOUT.narrowBelow ? LAYOUT.gutterNarrow : LAYOUT.gutter;
  return (
    <ScrollView ref={scrollRef} role="main" testID={testID} style={styles.scroll} contentContainerStyle={[styles.content, { paddingHorizontal: gutter }]}>
      <View style={styles.h1row}>
        {/* 숫자가 든 제목("거래시간(01h–24h)", "둘러보기 2/5")은 이름표라 메타로 둔다(I2). */}
        <Text ref={headingRef} role="heading" {...web({ 'aria-level': 1, tabIndex: -1 })} variant="title" style={styles.h1}
          dataSet={labelData(title)}>{title}</Text>
        {pill ? <View style={styles.pill}>{pill}</View> : null}
      </View>
      {answer}
      {children}
    </ScrollView>
  );
}

/**
 * 5초 문장(15/500 text1). 화면 view model이 데이터 상태로 만든 글자를 그대로 받는다. 숫자는 바로 아래 카드 값과 같은 포맷 결과라서
 * (view model이 같은 값·같은 함수로 만든다) 문장 전체를 data-pd-meta="answer"로 둔다(I2: 카드 밖 숫자는 메타 안에만).
 */
export function AnswerLine({ children, minLines }: { children: React.ReactNode; minLines?: number }) {
  // 글자로 오면 값+단위·시각·짧은 괄호를 한 줄로 묶는다(keepTogether, round 1: "684" / "MW"가 갈라짐).
  const body = typeof children === 'string' ? keepTogether(children, { variant: 'body', weight: '500' }) : children;
  // minLines: 불러오는 동안 다 받은 문장의 줄 수만큼 자리를 잡는다(round 2: 한 줄에서 두세 줄로 늘며 아래 카드가 모두 밀렸다).
  const reserve = minLines ? { minHeight: minLines * TYPE_SCALE.body.lineHeight } : null;
  return <Text variant="body" weight="500" style={reserve} dataSet={{ pdAnswer: 'true', pdMeta: 'answer' }}>{body}</Text>;
}

/** 한국어 오류 화면: h1 + [다시 불러오기]. 콘솔에 쓰지 않는다(7.10절). */
export function ErrorScreen() {
  const { theme } = useTheme();
  const reload = () => {
    const loc = (globalThis as { location?: { reload?: () => void } }).location;
    if (loc && typeof loc.reload === 'function') loc.reload();
  };
  return (
    <View role="main" style={[styles.error, { backgroundColor: theme.color.canvas }]} testID="error-screen">
      <Text role="heading" {...web({ 'aria-level': 1, tabIndex: -1 })} variant="title">화면을 그리는 중 문제가 생겼습니다</Text>
      <Button tone="primary" label="다시 불러오기" accessibilityLabel="다시 불러오기" onPress={reload} />
    </View>
  );
}

type BoundaryState = { failed: boolean };
export class ErrorBoundary extends React.Component<{ children?: React.ReactNode }, BoundaryState> {
  state: BoundaryState = { failed: false };

  static getDerivedStateFromError(): BoundaryState {
    return { failed: true };
  }

  render() {
    return this.state.failed ? <ErrorScreen /> : this.props.children;
  }
}

const styles = StyleSheet.create({
  scroll: { flex: 1 },
  content: { width: '100%', maxWidth: LAYOUT.column, alignSelf: 'center', paddingTop: SPACE.lg, paddingBottom: SPACE.xxl, gap: SPACE.md },
  h1row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: SPACE.sm },
  h1: { flexShrink: 1 },
  // NowPill 묶음은 줄 폭을 넘지 않는다: 넘으면 그 안에서 칩이 다음 줄로 내려간다(320에서 시계 칩 "서버 시각 확인 필요 · 기기 시계 기준"이
  // NowPill 옆에서 화면 밖으로 잘렸다, F13a 측정). RN-web은 flex 항목을 줄이지 않으므로 줄일 수 있게 둔다.
  pill: { flexShrink: 1, minWidth: 0, maxWidth: '100%' },
  error: { flex: 1, padding: LAYOUT.gutter, gap: SPACE.lg, justifyContent: 'center', alignItems: 'flex-start' },
});
