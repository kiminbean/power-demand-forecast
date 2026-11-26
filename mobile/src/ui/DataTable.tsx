// 표 보기(6.2절): role=table/row/columnheader/cell(RN-web은 div에 role 속성으로 내보낸다, F4 탐침),
// 머리 고정 없음, 카드 안에서만 가로 스크롤(M1의 유일한 예외: data-pd-card 안의 표 스크롤러).
// cards: 600 px 아래(휴대폰)에서는 같은 열·행을 행 카드로 쌓는다(round 1 D2: 표가 카드보다 넓어 끝 칸이 화면 밖에 있었다). 행 하나가
// 목록 항목 하나이고, 첫 열이 왼쪽 머리, 나머지 열은 "열 이름 값" 조각으로 이어진다. 조각마다 한 줄로 묶여(값·범위가 갈리지 않는다)
// 조각 사이 " · "에서만 줄이 바뀐다. 빈 칸(null)은 행 카드에서 빼고 표에서는 "—".
import { useState } from 'react';
import type React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SIZE, SPACE } from './tokens';
import { Text, keepTogether, labelData } from './Text';
import { useScrollFocus, web, webStyle } from './web';
import { useTheme } from '../app/theme';
import { SCROLL_CUE } from '../content/strings';

export type Column = {
  key: string;
  title: string;
  align?: 'left' | 'right';
  /** 시각·거래시간처럼 메타로 표시할 글자 칸(I2: 숫자는 data-pd-value·meta 안에만). */
  meta?: boolean;
  /** 머리 칸 제목 아래의 칩(입찰 표의 기대 수입: ┆예시┆ [▨ 추정], 4.2절). 행 카드에서는 목록 위 한 줄. */
  extra?: React.ReactNode;
  /** 칸 폭 비율(기본 1). 시간대 요약 표의 이름 칸처럼 글자가 긴 칸에. */
  grow?: number;
};
/** head·sub: 행 카드의 왼쪽 머리 두 줄(없으면 첫 열 글자). 시간별 표는 "06–07시" / "07h"(D11: 번호는 둘째 줄에만). */
export type Row = { key: string; cells: Readonly<Record<string, React.ReactNode>>; current?: boolean; head?: string; sub?: string };
/**
 * fit: 가로 스크롤 없이 칸이 폭을 나눠 갖고 글자는 칸 안에서 줄바꿈한다(긴 글 칸이 있는 출처 표, 짧은 4칸 시간대 요약 표). 가로 스크롤
 * 안에서는 행 폭이 칸마다 한 줄 글자 폭의 합이 되어(크롬 내재 크기) 긴 글 표가 늘 넘치고 끝 칸이 잘린다(F10 측정: 4칸 × 165 = 658 > 574).
 * cellMin: 칸 최소 폭(기본 72). 360 미만에서 시간대 요약 4칸이 272 px에 들게 56으로 낮춘다(round 1 접근성).
 */
export type DataTableProps = {
  caption: string; columns: readonly Column[]; rows: readonly Row[]; testID?: string; fit?: boolean; cellMin?: number; cards?: boolean;
};
const DASH = '—';

export function DataTable({ caption, columns, rows, testID, fit = false, cellMin = SIZE.cellMin, cards = false }: DataTableProps) {
  const { theme } = useTheme();
  const c = theme.color;
  // 고른 행은 표와 행 카드 모두 aria-current로 알리고 well 바탕 + 왼쪽 2 px primary로 보인다(4.1절). aria-selected는 table의 row에 허용되지 않는다.
  const rowLook = (row: Row) => [styles.line, { borderTopColor: c.border, borderLeftColor: 'transparent' },
    row.current ? { backgroundColor: c.well, borderLeftColor: c.primary } : null];
  const current = (row: Row) => web(row.current ? { 'aria-current': 'true' } : {});
  const text = (column: Column, content: React.ReactNode) => (typeof content === 'string'
    ? <Text variant="label" tone="text1" dataSet={column.meta ? { pdMeta: 'cell' } : undefined}>{keepTogether(content, { variant: 'label', tone: 'text1' })}</Text>
    : content);
  if (cards) {
    const first = columns[0];
    const rest = columns.slice(1);
    return (
      <View style={styles.cards} testID={testID}>
        {rest.filter(col => col.extra).map(col => (
          <View key={col.key} style={styles.extra}><Text variant="meta" tone="text3" dataSet={labelData(col.title)}>{col.title}</Text>{col.extra}</View>
        ))}
        <View role="list" aria-label={caption}>
          {rows.map(row => (
            <View key={row.key} role="listitem" style={[styles.card, rowLook(row)]} {...current(row)}>
              <View style={styles.head}>
                {row.head ? <Text variant="label" weight="600" nowrap dataSet={labelData(row.head)}>{row.head}</Text> : text(first, row.cells[first.key])}
                {row.sub ? <Text variant="meta" tone="text3" nowrap dataSet={{ pdMeta: 'hour' }}>{row.sub}</Text> : null}
              </View>
              <Text variant="label" tone="text1" style={styles.grow}>
                {rest.filter(col => row.cells[col.key] != null).flatMap((col, i) => [i ? ' · ' : null,
                  <Text key={col.key} variant="label" tone="text2" nowrap dataSet={labelData(col.title)}>{`${col.title} `}<Text variant="label">{text(col, row.cells[col.key])}</Text></Text>])}
              </Text>
            </View>
          ))}
        </View>
      </View>
    );
  }
  const width = (column: Column) => ({ flex: column.grow ?? 1, minWidth: cellMin });
  const table = (
    <View role="table" aria-label={caption} style={styles.table}>
      <View role="row" style={[styles.row, { backgroundColor: c.well }]}>
        {columns.map(column => (
          <View key={column.key} role="columnheader" style={[styles.cell, width(column), column.align === 'right' ? styles.right : null]}>
            <Text variant="label" weight="600" tone="text2" dataSet={labelData(column.title)}>{column.title}</Text>
            {column.extra ? <View style={styles.extra}>{column.extra}</View> : null}
          </View>
        ))}
      </View>
      {rows.map(row => (
        <View key={row.key} role="row" style={[styles.row, rowLook(row)]} {...current(row)}>
          {columns.map(column => (
            <View key={column.key} role="cell" style={[styles.cell, width(column), column.align === 'right' ? styles.right : null]}>
              {/* 글자 칸도 줄바꿈 단위(keepTogether)를 지킨다: "밤·아침 00–" / "08시"처럼 시각 범위가 갈리지 않게(round 1). */}
              {text(column, row.cells[column.key] ?? DASH)}
            </View>
          ))}
        </View>
      ))}
    </View>
  );
  if (fit) return <View style={styles.fit} testID={testID}>{table}</View>;
  return <ScrollingTable caption={caption} testID={testID}>{table}</ScrollingTable>;
}

/**
 * 카드 안 가로 스크롤(M1의 유일한 예외). 표가 넘치면 스크롤 상자가 Tab 정지점이 되어 화살표로 옆 칸을 볼 수 있다(F11: 320·390의
 * 입찰 표에서 axe scrollable-region-focusable). 이름은 표 캡션이다. 넘치는 동안은 오른쪽 가장자리를 흐리게 하고 "옆으로 밀어 보기"를
 * 붙인다(round 1 D2: 밀 수 있다는 표시가 없었다). 끝까지 밀면 흐림은 걷힌다.
 */
function ScrollingTable({ caption, testID, children }: { caption: string; testID?: string; children: React.ReactNode }) {
  const { theme } = useTheme();
  const box = useScrollFocus('x');
  const [atEnd, setAtEnd] = useState(false);
  return (
    <View style={styles.cards}>
      <View>
        <ScrollView horizontal style={styles.scroll} contentContainerStyle={styles.scrollContent} testID={testID} scrollEventThrottle={64}
          onScroll={e => { const n = e.nativeEvent; setAtEnd(n.contentOffset.x + n.layoutMeasurement.width >= n.contentSize.width - 1); }}
          onLayout={box.onLayout} onContentSizeChange={box.onContentSizeChange} {...web({ role: 'region', 'aria-label': caption, ...box.focus })}>
          {children}
        </ScrollView>
        {box.scrolls && !atEnd ? (
          <View aria-hidden pointerEvents="none" testID="table-edge-fade"
            style={[styles.fade, webStyle({ backgroundImage: `linear-gradient(to right, transparent, ${theme.color.surface})` })]} />
        ) : null}
      </View>
      {box.scrolls ? <Text variant="meta" tone="text3">{SCROLL_CUE}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flexGrow: 0, maxWidth: '100%' },
  fit: { width: '100%' },
  scrollContent: { minWidth: '100%' },
  table: { flex: 1 },
  row: { flexDirection: 'row' },
  line: { borderTopWidth: 1, borderLeftWidth: SIZE.selectedEdge },
  cell: { paddingVertical: SPACE.sm, paddingHorizontal: SPACE.sm, justifyContent: 'center' },
  right: { alignItems: 'flex-end' },
  extra: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs, marginTop: SPACE.xs },
  fade: { position: 'absolute', top: 0, right: 0, bottom: 0, width: SIZE.edgeFade },
  cards: { gap: SPACE.xs },
  card: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm, paddingVertical: SPACE.sm, paddingRight: SPACE.xs, paddingLeft: SPACE.sm - SIZE.selectedEdge },
  head: { width: SIZE.rowHead },
  grow: { flex: 1, minWidth: 0 },
});
