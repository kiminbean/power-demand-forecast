// 시트 자리(3.3·6.2절): 경로 상태에 sheet가 있고 그 종류가 등록되어 있을 때만 Sheet를 그린다. 열고 닫는 히스토리 규칙은
// routes.ts(openSheet·closeSheet), 초점 가둠·Esc·스크림은 Sheet가 맡는다. 여기서는 연 요소를 기록해 두었다가 닫힌 뒤 돌려준다:
// Safari는 버튼을 눌러도 초점을 주지 않아 열 때 document.activeElement가 body일 수 있으므로 마지막으로 누른 요소도 함께 본다(F4 증거).
// 화면 테마 시트(4.4·8.4절)는 셸 설정이라 여기에 붙박이로 둔다. 산출 방식·출처·용어 시트는 그 화면 단계가 registry로 넘긴다.
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import type { SheetKind, SheetState } from './routes';
import { useTheme } from './theme';
import type { ThemeMode } from './theme';
import { SegmentedControl } from '../ui/Controls';
import type { SegmentOption } from '../ui/Controls';
import { Sheet } from '../ui/Sheet';
import { Text } from '../ui/Text';
import { SPACE } from '../ui/tokens';
import type { DomEl } from '../ui/web';

// 메서드 표기라 매개변수가 이변(bivariant)이다: 종류별 항목을 SheetEntry 하나로 꺼내 쓸 수 있다.
export type SheetEntry<S extends SheetState = SheetState> = { title(sheet: S): string; render(sheet: S): React.ReactNode };
export type SheetRegistry = { [K in SheetKind]?: SheetEntry<Extract<SheetState, { kind: K }>> };

const THEME_OPTIONS: readonly SegmentOption<ThemeMode>[] = [
  { value: 'system', label: '시스템 설정 따름 (기본)' },
  { value: 'light', label: '밝게' },
  { value: 'dark', label: '어둡게' },
];

/** 고르면 바로 적용된다. 저장소를 못 쓰면 그 사실을 그대로 쓴다. */
function ThemeSheet() {
  const { mode, setMode } = useTheme();
  const [saved, setSaved] = useState(true);
  return (
    <View style={styles.theme}>
      <SegmentedControl label="화면 테마" options={THEME_OPTIONS} value={mode} vertical onChange={next => setSaved(setMode(next))} />
      <Text variant="label" tone="text2">{saved ? '이 기기에만 저장됩니다' : '이 기기에 저장하지 못해 이번 방문 동안만 적용됩니다'}</Text>
    </View>
  );
}

export const BUILT_IN_SHEETS: SheetRegistry = { theme: { title: () => '화면 테마', render: () => <ThemeSheet /> } };

type PointerListener = (e: { target: unknown }) => void;
type DocLike = {
  activeElement: unknown;
  body: unknown;
  contains(node: unknown): boolean;
  addEventListener(type: 'pointerdown', fn: PointerListener, capture: boolean): void;
  removeEventListener(type: 'pointerdown', fn: PointerListener, capture: boolean): void;
};
const PRESSABLE = 'button, a, [role="button"], [role="link"], [role="tab"], [role="radio"], [tabindex]';
const domDoc = (): DocLike | null => (typeof document === 'undefined' ? null : (document as unknown as DocLike));
const isFocusable = (el: unknown): el is DomEl => Boolean(el) && typeof (el as DomEl).focus === 'function';
const sheetKey = (s: SheetState) => (s.kind === 'method' ? s.card : s.kind === 'sources' ? s.row ?? '' : s.kind === 'term' ? s.id : '');

export type SheetHostProps = {
  sheet: SheetState | null;
  registry: SheetRegistry;
  onClose: () => void;
  /** 연 요소가 없거나(딥링크로 들어온 시트) 사라졌을 때 닫힌 뒤 초점을 둘 곳. 셸은 화면 h1을 준다. */
  onNoOpener: () => void;
};

export function SheetHost({ sheet, registry, onClose, onNoOpener }: SheetHostProps) {
  const entry = sheet ? (registry[sheet.kind] as SheetEntry | undefined) : undefined;
  const open = Boolean(sheet && entry);
  const pressed = useRef<DomEl | null>(null);
  const opener = useRef<DomEl | null>(null);
  const wasOpen = useRef(false);

  useEffect(() => {
    const d = domDoc();
    if (!d) return undefined;
    const onDown: PointerListener = e => {
      const target = e.target as { closest?: (selector: string) => unknown } | null;
      const hit = target && typeof target.closest === 'function' ? target.closest(PRESSABLE) : null;
      if (isFocusable(hit)) pressed.current = hit;
    };
    d.addEventListener('pointerdown', onDown, true);
    return () => d.removeEventListener('pointerdown', onDown, true);
  }, []);

  // 열리는 순간, Sheet가 닫기 버튼으로 초점을 옮기기(Sheet의 효과) 전에 연 요소를 잡는다.
  useLayoutEffect(() => {
    if (!open || wasOpen.current) return;
    const d = domDoc();
    const active = d ? d.activeElement : null;
    opener.current = d && active !== d.body && isFocusable(active) ? active : pressed.current;
  }, [open]);

  // 닫힌 뒤(Sheet의 정리 효과 다음) 연 요소로 초점을 돌려준다.
  useEffect(() => {
    if (open) {
      wasOpen.current = true;
      return;
    }
    if (!wasOpen.current) return;
    wasOpen.current = false;
    const d = domDoc();
    const target = opener.current;
    opener.current = null;
    if (d && target && d.contains(target)) target.focus();
    else onNoOpener();
  }, [open, onNoOpener]);

  if (!sheet || !entry) return null;
  // 시트끼리 바뀌면(용어 → 용어 사전) 새로 열어 초점·가둠을 다시 건다.
  return <Sheet key={`${sheet.kind}:${sheetKey(sheet)}`} title={entry.title(sheet)} onClose={onClose}>{entry.render(sheet)}</Sheet>;
}

const styles = StyleSheet.create({
  theme: { gap: SPACE.md },
});
