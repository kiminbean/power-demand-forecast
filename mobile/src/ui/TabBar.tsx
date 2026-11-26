// 4목적지 내비게이션(3.4절). 휴대폰은 하단 탭 막대, ≥ 1024는 왼쪽 레일(같은 4항목, aria-orientation=vertical).
// nav "주요 메뉴" > tablist > tab(aria-selected). 화살표·Home·End는 초점만 옮기고(roving tabindex) Enter·Space가 연다.
// react-native-web 0.19.13은 role=tab에서 Space를 누름으로 바꾸지 않아(Enter만) Space는 여기서 직접 받는다(round 1 접근성).
// 600–1023에서는 탭 칸이 본문 카드 열(최대 640 − 좌우 여백 16)과 같은 폭·x에 선다(round 1 D14). 막대 바탕은 화면 끝까지.
import { useRef } from 'react';
import { Pressable, StyleSheet, View, useWindowDimensions } from 'react-native';
import { HIT, LAYOUT, SIZE, SPACE } from './tokens';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { Text } from './Text';
import { domOf, keyOf, prevent, rovingIndex, web } from './web';
import { useTheme } from '../app/theme';

export type TabItem<K extends string> = { key: K; label: string; icon: IconName };
export type TabBarProps<K extends string> = {
  items: readonly TabItem<K>[];
  active: K;
  /** 지금 탭을 다시 누르면 셸이 맨 위로 올린다(3.3절). */
  onSelect: (key: K) => void;
  orientation?: 'horizontal' | 'vertical';
};

export function TabBar<K extends string>({ items, active, onSelect, orientation = 'horizontal' }: TabBarProps<K>) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const vertical = orientation === 'vertical';
  const tablet = !vertical && width >= LAYOUT.sheetCenterFrom;
  const refs = useRef<Array<View | null>>([]);
  const current = items.findIndex(item => item.key === active);
  const onKey = (e: unknown, index: number) => {
    const key = keyOf(e);
    if (key === ' ' || key === 'Spacebar') {
      prevent(e);
      onSelect(items[index].key);
      return;
    }
    const next = rovingIndex(key, index, items.length, orientation);
    if (next === null) return;
    prevent(e);
    const node = domOf({ current: refs.current[next] });
    if (node) node.focus();
  };
  // 아래 탭 막대만 안전 영역 아래쪽 패딩을 받는다(index.html의 [data-pd-safe~="bottom"]). 가로 모드 노치는 막대 안쪽(x)에.
  const safe = vertical ? {} : { dataSet: { pdSafe: 'bottom x' } };
  return (
    <View role="navigation" aria-label="주요 메뉴" testID={vertical ? 'rail' : 'tab-bar'}
      style={[vertical ? styles.rail : styles.bar, { backgroundColor: theme.color.surface, borderColor: theme.color.border }]} {...web(safe)}>
      <View role="tablist" style={vertical ? styles.listV : [styles.listH, tablet ? styles.listTablet : null]} {...web({ 'aria-orientation': orientation })}>
        {items.map((item, i) => {
          const on = i === current;
          const tint = on ? theme.color.primary : theme.color.text2;
          return (
            <Pressable key={item.key} ref={el => { refs.current[i] = el; }} role="tab" aria-selected={on} aria-label={item.label}
              onPress={() => onSelect(item.key)} style={vertical ? styles.tabV : styles.tabH}
              {...web({ tabIndex: on || (current < 0 && i === 0) ? 0 : -1, onKeyDown: (e: unknown) => onKey(e, i) })}>
              <View style={[vertical ? styles.markV : styles.markH, on ? { backgroundColor: theme.color.primary } : null]} />
              <Icon name={item.icon} size={SIZE.icon} color={tint} />
              <Text variant={vertical ? 'body' : 'meta'} weight="600" color={tint}>{item.label}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

/** ≥ 1024 왼쪽 레일(240 px): 같은 4항목, 위아래 화살표. */
export function Rail<K extends string>(props: Omit<TabBarProps<K>, 'orientation'>) {
  return <TabBar {...props} orientation="vertical" />;
}

const styles = StyleSheet.create({
  bar: { borderTopWidth: 1 },
  rail: { width: LAYOUT.rail, borderRightWidth: 1, paddingVertical: SPACE.sm },
  listH: { flexDirection: 'row', width: '100%', alignSelf: 'center' },
  // 본문(Screen)과 같은 상자 규칙: 최대 640, 좌우 16 → 카드 열과 같은 x·폭.
  listTablet: { maxWidth: LAYOUT.column, paddingHorizontal: LAYOUT.gutter },
  listV: { flexDirection: 'column', gap: SPACE.xs },
  tabH: { flex: 1, minHeight: LAYOUT.tabBar, minWidth: HIT, alignItems: 'center', justifyContent: 'center', gap: SPACE.xs / 2 },
  tabV: { minHeight: HIT + SPACE.sm, flexDirection: 'row', alignItems: 'center', gap: SPACE.md, paddingHorizontal: SPACE.xl },
  markH: { position: 'absolute', top: 0, left: SPACE.md, right: SPACE.md, height: SIZE.tabMark },
  markV: { position: 'absolute', left: 0, top: SPACE.sm, bottom: SPACE.sm, width: SIZE.tabMark },
});
