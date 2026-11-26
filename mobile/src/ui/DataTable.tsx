// 표 보기(6.2절): role=table/row/columnheader/cell(RN-web은 div에 role 속성으로 내보낸다, F4 탐침),
// 머리 고정 없음, 카드 안에서만 가로 스크롤(M1의 유일한 예외: data-pd-card 안의 표 스크롤러).
import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SIZE, SPACE } from './tokens';
import { Text, labelData } from './Text';
import { useScrollFocus, web } from './web';
import { useTheme } from '../app/theme';

export type Column = {
  key: string;
  title: string;
  align?: 'left' | 'right';
  /** 시각·거래시간처럼 메타로 표시할 글자 칸(I2: 숫자는 data-pd-value·meta 안에만). */
  meta?: boolean;
  /** 머리 칸 제목 아래의 칩(입찰 표의 기대 수입: ┆예시┆ [▨ 추정], 4.2절). */
  extra?: React.ReactNode;
};
export type Row = { key: string; cells: Readonly<Record<string, React.ReactNode>>; current?: boolean };
/**
 * fit: 가로 스크롤 없이 칸이 폭을 나눠 갖고 글자는 칸 안에서 줄바꿈한다(긴 글 칸이 있는 출처 표). 가로 스크롤 안에서는 행 폭이 칸마다
 * 한 줄 글자 폭의 합이 되어(크롬 내재 크기) 긴 글 표가 늘 넘치고 끝 칸이 잘린다(F10 측정: 4칸 × 165 = 658 > 574).
 */
export type DataTableProps = { caption: string; columns: readonly Column[]; rows: readonly Row[]; testID?: string; fit?: boolean };

export function DataTable({ caption, columns, rows, testID, fit = false }: DataTableProps) {
  const { theme } = useTheme();
  const cell = (column: Column, content: React.ReactNode) => (
    <View key={column.key} role="cell" style={[styles.cell, column.align === 'right' ? styles.right : null]}>
      {typeof content === 'string'
        ? <Text variant="label" tone="text1" dataSet={column.meta ? { pdMeta: 'cell' } : undefined}>{content}</Text>
        : content}
    </View>
  );
  const table = (
    <View role="table" aria-label={caption} style={styles.table}>
      <View role="row" style={[styles.row, { backgroundColor: theme.color.well }]}>
        {columns.map(column => (
          <View key={column.key} role="columnheader" style={[styles.cell, column.align === 'right' ? styles.right : null]}>
            <Text variant="label" weight="600" tone="text2" dataSet={labelData(column.title)}>{column.title}</Text>
            {column.extra ? <View style={styles.extra}>{column.extra}</View> : null}
          </View>
        ))}
      </View>
      {rows.map(row => (
        // 선택 행은 aria-current로 알리고 well 바탕 + 왼쪽 2 px primary로 보인다(4.1절). aria-selected는 table의 row에 허용되지 않는다.
        <View key={row.key} role="row" style={[styles.row, styles.line, { borderTopColor: theme.color.border, borderLeftColor: 'transparent' },
          row.current ? { backgroundColor: theme.color.well, borderLeftColor: theme.color.primary } : null]} {...web(row.current ? { 'aria-current': 'true' } : {})}>
          {columns.map(column => cell(column, row.cells[column.key] ?? null))}
        </View>
      ))}
    </View>
  );
  if (fit) return <View style={styles.fit} testID={testID}>{table}</View>;
  return <ScrollingTable caption={caption} testID={testID}>{table}</ScrollingTable>;
}

/**
 * 카드 안 가로 스크롤(M1의 유일한 예외). 표가 넘치면 스크롤 상자가 Tab 정지점이 되어 화살표로 옆 칸을 볼 수 있다(F11: 320·390의
 * 입찰 표에서 axe scrollable-region-focusable). 이름은 표 캡션이다.
 */
function ScrollingTable({ caption, testID, children }: { caption: string; testID?: string; children: React.ReactNode }) {
  const box = useScrollFocus('x');
  return (
    <ScrollView horizontal style={styles.scroll} contentContainerStyle={styles.scrollContent} testID={testID}
      onLayout={box.onLayout} onContentSizeChange={box.onContentSizeChange} {...web({ role: 'region', 'aria-label': caption, ...box.focus })}>
      {children}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flexGrow: 0, maxWidth: '100%' },
  fit: { width: '100%' },
  scrollContent: { minWidth: '100%' },
  table: { flex: 1 },
  row: { flexDirection: 'row' },
  line: { borderTopWidth: 1, borderLeftWidth: SIZE.selectedEdge },
  cell: { flex: 1, minWidth: SIZE.cellMin, paddingVertical: SPACE.sm, paddingHorizontal: SPACE.sm, justifyContent: 'center' },
  right: { alignItems: 'flex-end' },
  extra: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACE.xs, marginTop: SPACE.xs },
});
