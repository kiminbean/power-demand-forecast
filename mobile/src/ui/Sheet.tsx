// 시트(6.2절): 휴대폰은 아래 시트(최대 85 %, 위 모서리 20), 600–1023은 아래쪽 640 폭, ≥ 1024는 오른쪽 패널 400.
// role=dialog + aria-modal + aria-labelledby, 초점 가둠(Tab·Shift+Tab), Esc, 스크림 클릭, 닫히면 연 요소로 초점 복귀.
// 열고 닫는 것과 히스토리(3.3절)는 SheetHost(F6)가 맡는다: 이 부품은 열린 동안만 그려진다. RN Modal은 쓰지 않는다.
import React, { useEffect, useId, useRef } from 'react';
import { ScrollView, StyleSheet, View, useWindowDimensions } from 'react-native';
import type { ViewStyle } from 'react-native';
import { LAYOUT, MOTION, RADIUS, SPACE } from './tokens';
import { IconButton } from './Button';
import { Text, labelData } from './Text';
import { domOf, tabStops, useScrollFocus, web, webStyle } from './web';
import type { DomEl } from './web';
import { useTheme } from '../app/theme';

type DocLike = {
  activeElement: unknown;
  contains(node: unknown): boolean;
  addEventListener(type: 'keydown', fn: (e: KeyEventLike) => void, capture: boolean): void;
  removeEventListener(type: 'keydown', fn: (e: KeyEventLike) => void, capture: boolean): void;
};
type KeyEventLike = { key: string; shiftKey: boolean; preventDefault(): void };

export type SheetProps = { title: string; onClose: () => void; children?: React.ReactNode; testID?: string };

export function Sheet({ title, onClose, children, testID = 'sheet' }: SheetProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const panelRef = useRef<View>(null);
  const closeRef = useRef<View>(null);
  // 부모가 렌더마다 새 함수를 넘겨도 가둠·복귀 효과를 다시 걸지 않게 최신 값만 들고 있는다.
  const closeLatest = useRef(onClose);
  closeLatest.current = onClose;
  const titleId = `pd-sheet-${useId().replace(/[^A-Za-z0-9_-]/g, '')}`;
  // 본문이 시트 높이를 넘으면 본문 자체가 Tab 정지점이 된다(region, 이름 = 시트 제목). 링크·버튼이 없는 본문(오늘 24시간 가격의
  // 산출 방식)은 그러지 않으면 키보드로 끝까지 읽을 수 없다(F11: 320×568에서 axe scrollable-region-focusable).
  const body = useScrollFocus('y');

  useEffect(() => {
    const doc = (globalThis as { document?: DocLike }).document;
    if (!doc) return undefined;
    const opener = doc.activeElement as Partial<DomEl> | null;
    const close = domOf(closeRef);
    if (close) close.focus();
    const onKey = (e: KeyEventLike) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeLatest.current();
        return;
      }
      const panel = domOf(panelRef);
      if (e.key !== 'Tab' || !panel) return;
      const stops = tabStops(panel);
      const at = stops.indexOf(doc.activeElement as DomEl);
      if (!stops.length) {
        e.preventDefault();
      } else if (e.shiftKey && at <= 0) {
        e.preventDefault();
        stops[stops.length - 1].focus();
      } else if (!e.shiftKey && (at === -1 || at === stops.length - 1)) {
        e.preventDefault();
        stops[0].focus();
      }
    };
    doc.addEventListener('keydown', onKey, true);
    return () => {
      doc.removeEventListener('keydown', onKey, true);
      if (opener && typeof opener.focus === 'function' && doc.contains(opener)) opener.focus();
    };
  }, []);

  const place: ViewStyle = width >= LAYOUT.railFrom ? styles.side : width >= LAYOUT.sheetCenterFrom ? styles.center : styles.bottom;
  const lift = theme.elevation.sheet ? webStyle({ boxShadow: theme.elevation.sheet })
    : { borderWidth: 1, borderColor: theme.elevation.sheetBorder ?? theme.color.borderStrong };
  return (
    <View style={styles.overlay} testID={testID}>
      <View aria-hidden style={[StyleSheet.absoluteFill, { backgroundColor: theme.color.scrim }]}
        {...web({ onClick: () => closeLatest.current(), dataSet: { pdScrim: 'true' } })} />
      <View ref={panelRef} role="dialog" aria-modal aria-labelledby={titleId}
        style={[styles.panel, place, { backgroundColor: theme.color.surface }, lift, motion.enter]}
        {...web({ dataSet: width >= LAYOUT.railFrom ? {} : { pdSafe: 'bottom' } })}>
        <View style={[styles.head, { borderBottomColor: theme.color.border }]}>
          {/* 숫자가 든 제목("산출 방식 · 오늘 24시간 가격", "80% 구간")은 이름표라 메타로 둔다(I2). */}
          <Text role="heading" {...web({ 'aria-level': 2 })} nativeID={titleId} variant="body" weight="700" style={styles.title}
            dataSet={labelData(title)}>{title}</Text>
          <IconButton pressRef={closeRef} icon="close" accessibilityLabel="닫기" onPress={() => closeLatest.current()} testID="sheet-close" />
        </View>
        <ScrollView style={styles.body} contentContainerStyle={styles.content} onLayout={body.onLayout} onContentSizeChange={body.onContentSizeChange}
          {...web({ role: 'region', 'aria-labelledby': titleId, ...body.focus })}>{children}</ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 10 },
  panel: { position: 'absolute', maxHeight: `${Math.round(LAYOUT.sheetMaxHeight * 100)}%` },
  bottom: { left: 0, right: 0, bottom: 0, borderTopLeftRadius: RADIUS.sheet, borderTopRightRadius: RADIUS.sheet },
  // 600–639 px에서는 640이 화면보다 넓어 문서가 40 px 넘쳤다(F11 경계 폭 측정): 폭은 화면을 넘지 않고 최대 640.
  center: {
    left: 0, right: 0, bottom: 0, width: '100%', maxWidth: LAYOUT.column, marginHorizontal: 'auto',
    borderTopLeftRadius: RADIUS.sheet, borderTopRightRadius: RADIUS.sheet,
  },
  side: { top: 0, right: 0, bottom: 0, width: LAYOUT.sheetPanel, maxHeight: '100%' },
  head: { flexDirection: 'row', alignItems: 'center', gap: SPACE.sm, paddingLeft: SPACE.lg, paddingRight: SPACE.xs, borderBottomWidth: 1 },
  title: { flex: 1, minWidth: 0 },
  body: { flexShrink: 1 },
  content: { padding: SPACE.lg, gap: SPACE.md },
});

// 200 ms ease-out으로 한 번 들어온다(반복 없음). index.html의 prefers-reduced-motion 규칙이 끈다.
const motion = StyleSheet.create({
  enter: webStyle({
    animationDuration: `${MOTION.sheetMs}ms`, animationTimingFunction: 'ease-out',
    animationKeyframes: [{ '0%': { opacity: 0, transform: `translateY(${SPACE.lg}px)` }, '100%': { opacity: 1, transform: 'translateY(0px)' } }],
  }),
});
