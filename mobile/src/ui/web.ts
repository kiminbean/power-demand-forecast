// RN 0.73 타입에 없는 웹 전용 prop(dataSet·aria-level·aria-orientation·aria-current·tabIndex·onKeyDown·onClick)을
// 넘기는 유일한 통로다. react-native-web 0.19.13의 createDOMProps가 이 이름 그대로 DOM 속성으로 내보낸다
// (F4 의미 탐침 scratchpad/mobile/eval/steps/F4/probe.md에서 확인).
import type { ViewStyle } from 'react-native';

export const web = (props: Record<string, unknown>): object => props;
/** RN 0.73 ViewStyle에 없는 CSS 값(boxShadow, 애니메이션)을 스타일 배열에 넣는다. */
export const webStyle = (style: Record<string, unknown>) => style as unknown as ViewStyle;

/** 화면 읽기 프로그램에만 들리는 글자. opacity 0은 접근성 트리에서 빠지지 않고 측정 도구(probe.js)의 잘림 검사에서도 빠진다. */
export const SR_ONLY = { position: 'absolute', width: 1, height: 1, overflow: 'hidden', opacity: 0 } as const;

export type DomEl = {
  focus(): void;
  contains(node: unknown): boolean;
  querySelectorAll(selector: string): ArrayLike<DomEl>;
  getClientRects(): ArrayLike<unknown>;
};

/** 웹에서 호스트 컴포넌트의 ref는 DOM 요소다. 네이티브·VM 테스트에서는 null이라 아무 일도 하지 않는다. */
export function domOf(ref: { current: unknown } | null | undefined): DomEl | null {
  const node = ref ? (ref.current as Partial<DomEl> | null) : null;
  return node && typeof node.focus === 'function' && typeof node.querySelectorAll === 'function' ? (node as DomEl) : null;
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
