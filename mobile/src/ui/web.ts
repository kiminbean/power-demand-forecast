// RN 0.73 타입에 없는 웹 전용 prop(dataSet·aria-level·aria-orientation·aria-current·tabIndex·onKeyDown·onClick)을
// 넘기는 유일한 통로다. react-native-web 0.19.13의 createDOMProps가 이 이름 그대로 DOM 속성으로 내보낸다
// (F4 의미 탐침 scratchpad/mobile/eval/steps/F4/probe.md에서 확인).
import { useCallback, useRef, useState } from 'react';
import type { LayoutChangeEvent, ViewStyle } from 'react-native';

export const web = (props: Record<string, unknown>): object => props;
/** RN 0.73 ViewStyle에 없는 CSS 값(boxShadow, 애니메이션)을 스타일 배열에 넣는다. */
export const webStyle = (style: Record<string, unknown>) => style as unknown as ViewStyle;

/** 화면 읽기 프로그램에만 들리는 글자. opacity 0은 접근성 트리에서 빠지지 않고 측정 도구(probe.js)의 잘림 검사에서도 빠진다. */
export const SR_ONLY = { position: 'absolute', width: 1, height: 1, overflow: 'hidden', opacity: 0 } as const;

export type DomEl = {
  focus(options?: { preventScroll?: boolean }): void;
  contains(node: unknown): boolean;
  querySelectorAll(selector: string): ArrayLike<DomEl>;
  getClientRects(): ArrayLike<unknown>;
};
export type ScrollEl = DomEl & { scrollTop: number; scrollHeight: number; clientHeight: number };

/** 웹에서 호스트 컴포넌트의 ref는 DOM 요소다. 네이티브·VM 테스트에서는 null이라 아무 일도 하지 않는다. */
export function domOf(ref: { current: unknown } | null | undefined): DomEl | null {
  const node = ref ? (ref.current as Partial<DomEl> | null) : null;
  return node && typeof node.focus === 'function' && typeof node.querySelectorAll === 'function' ? (node as DomEl) : null;
}

/**
 * root 안의 세로 스크롤 상자, 스크롤 범위가 큰 것부터. 셸이 화면을 모르고도(레거시 화면의 ScrollView까지) 맨 위로 올리고
 * 위치를 기억·복원하는 통로다(3.3절). getComputedStyle이 없는 환경(네이티브·node)에서는 빈 배열.
 */
export function scrollBoxes(root: DomEl | null): ScrollEl[] {
  if (!root || typeof getComputedStyle !== 'function') return [];
  const boxes = Array.from(root.querySelectorAll('*')) as ScrollEl[];
  const range = (el: ScrollEl) => el.scrollHeight - el.clientHeight;
  return boxes
    .filter(el => /^(auto|scroll)$/.test(getComputedStyle(el as unknown as Element).overflowY))
    .sort((a, b) => range(b) - range(a));
}

/**
 * 스크롤 상자가 실제로 넘칠 때만 Tab 정지점으로 만든다(axe scrollable-region-focusable, F11 측정): 링크·버튼이 없는 시트 본문(세로),
 * 카드 안 가로 스크롤 표. onLayout(보이는 크기)과 onContentSizeChange(내용 크기)로 잰다. role·이름은 부르는 쪽이 늘 붙이고 tabIndex만
 * 바꾼다(role이 바뀌면 RN-web이 div↔section으로 요소를 바꿔 내용을 다시 그린다).
 */
export function useScrollFocus(axis: 'x' | 'y') {
  const sizes = useRef({ view: 0, content: 0 });
  const [scrolls, setScrolls] = useState(false);
  const measure = useCallback(() => setScrolls(sizes.current.content > sizes.current.view + 1), []);
  const onLayout = useCallback((e: LayoutChangeEvent) => {
    sizes.current.view = axis === 'x' ? e.nativeEvent.layout.width : e.nativeEvent.layout.height;
    measure();
  }, [axis, measure]);
  const onContentSizeChange = useCallback((w: number, h: number) => {
    sizes.current.content = axis === 'x' ? w : h;
    measure();
  }, [axis, measure]);
  return { onLayout, onContentSizeChange, scrolls, focus: scrolls ? { tabIndex: 0 } : {} };
}

type AttrEl = { setAttribute?: (name: string, value: string) => void; removeAttribute?: (name: string) => void };
/**
 * 시트가 열린 동안 바깥(헤더·본문·탭 막대)을 inert로 둔다(round 1 접근성: Chromium은 aria-modal만으로 바깥을 접근성 트리에서 빼지 않아
 * 화면 읽기 프로그램이 시트 밖으로 나갈 수 있었다). inert는 접근성 트리와 초점에서 함께 뺀다. 네이티브·VM에서는 아무 일도 하지 않는다.
 */
export function setInert(ref: { current: unknown } | null | undefined, on: boolean): void {
  const node = ref ? (ref.current as AttrEl | null) : null;
  if (!node || typeof node.setAttribute !== 'function' || typeof node.removeAttribute !== 'function') return;
  if (on) node.setAttribute('inert', '');
  else node.removeAttribute('inert');
}

/**
 * 같은 화면의 카드 제목(h2, tabIndex=-1)으로 초점을 옮긴다(round 1 D4: 둘러보기 안내를 닫으면 다음 카드 제목으로). testID는 카드의 것이다.
 * 찾지 못하면 false(부르는 쪽이 화면 h1로 보낸다).
 */
export function focusCardTitle(testID: string): boolean {
  const doc = (globalThis as { document?: { querySelector(s: string): unknown } }).document;
  const heading = doc ? (doc.querySelector(`[data-testid="${testID}"] [role="heading"]`) as Partial<DomEl> | null) : null;
  if (!heading || typeof heading.focus !== 'function') return false;
  heading.focus();
  return true;
}

/** 라우트가 바뀌면 h1(tabIndex=-1)로 초점을 옮긴다. 초점이 스크롤을 움직이지 않게 한다(복원한 위치를 지킨다). */
export function focusHeading(root: DomEl | null): boolean {
  const heading = root ? root.querySelectorAll('h1, [role="heading"][aria-level="1"]')[0] : undefined;
  if (!heading) return false;
  heading.focus({ preventScroll: true });
  return true;
}

/** 문서에서 화면 h1로 초점을 옮긴다(카드 밖 StateView의 [다시 시도], round 1 D4). */
export function focusScreenHeading(): void {
  const doc = (globalThis as unknown as { document?: DomEl }).document;
  if (doc && typeof doc.querySelectorAll === 'function') focusHeading(doc);
}

/** Tab으로 갈 수 있는 것: Pressable은 늘 tabindex를 0 또는 -1로 쓴다(비활성·roving 칸은 -1). */
export function tabStops(root: DomEl): DomEl[] {
  return Array.from(root.querySelectorAll('[tabindex="0"]')).filter(el => el.getClientRects().length > 0);
}

type KeyLike = { key?: string; nativeEvent?: { key?: string }; preventDefault?: () => void };
export const keyOf = (e: unknown): string => {
  const ev = (e ?? {}) as KeyLike;
  return ev.key ?? (ev.nativeEvent ? ev.nativeEvent.key ?? '' : '');
};
export const prevent = (e: unknown) => {
  const ev = (e ?? {}) as KeyLike;
  if (typeof ev.preventDefault === 'function') ev.preventDefault();
};

/** roving tabindex: 화살표·Home·End가 고르는 다음 칸. 해당 키가 아니면 null. 'both'는 radiogroup(네 방향). */
export function rovingIndex(key: string, index: number, count: number, axis: 'horizontal' | 'vertical' | 'both'): number | null {
  if (count <= 0) return null;
  const back = (axis !== 'vertical' && key === 'ArrowLeft') || (axis !== 'horizontal' && key === 'ArrowUp');
  const forward = (axis !== 'vertical' && key === 'ArrowRight') || (axis !== 'horizontal' && key === 'ArrowDown');
  if (back) return (index - 1 + count) % count;
  if (forward) return (index + 1) % count;
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  return null;
}
