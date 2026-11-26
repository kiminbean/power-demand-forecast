// 표 보기(6.2절): role=table/row/columnheader/cell(RN-web은 div에 role 속성으로 내보낸다, F4 탐침),
// 머리 고정 없음, 카드 안에서만 가로 스크롤(M1의 유일한 예외: data-pd-card 안의 표 스크롤러).
import React from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { SIZE, SPACE } from './tokens';
import { Text } from './Text';
import { web } from './web';
import { useTheme } from '../app/theme';

export type Column = {
  key: string;
  title: string;
  align?: 'left' | 'right';
  /** 시각·거래시간처럼 메타로 표시할 글자 칸(I2: 숫자는 data-pd-value·meta 안에만). */
  meta?: boolean;
};
export type Row = { key: string; cells: Readonly<Record<string, React.ReactNode>>; current?: boolean };
export type DataTableProps = { caption: string; columns: readonly Column[]; rows: readonly Row[]; testID?: string };

export function DataTable({ caption, columns, rows, testID }: DataTableProps) {
  const { theme } = useTheme();
  const cell = (column: Column, content: React.ReactNode) => (
    <View key={column.key} role="cell" style={[styles.cell, column.align === 'right' ? styles.right : null]}>
      {typeof content === 'string'
        ? <Text variant="label" tone="text1" dataSet={column.meta ? { pdMeta: 'cell' } : undefined}>{content}</Text>
        : content}
    </View>
  );
  return (
    <ScrollView horizontal style={styles.scroll} contentContainerStyle={styles.scrollContent} testID={testID}>
      <View role="table" aria-label={caption} style={styles.table}>
        <View role="row" style={[styles.row, { backgroundColor: theme.color.well }]}>
          {columns.map(column => (
            <View key={column.key} role="columnheader" style={[styles.cell, column.align === 'right' ? styles.right : null]}>
              <Text variant="label" weight="600" tone="text2">{column.title}</Text>
            </View>
          ))}
        </View>
        {rows.map(row => (
          // 선택 행은 aria-current로 알린다(aria-selected는 table의 row에 허용되지 않는다).
          <View key={row.key} role="row" style={[styles.row, styles.line, { borderTopColor: theme.color.border },
            row.current ? { backgroundColor: theme.color.well } : null]} {...web(row.current ? { 'aria-current': 'true' } : {})}>
            {columns.map(column => cell(column, row.cells[column.key] ?? null))}
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flexGrow: 0, maxWidth: '100%' },
  scrollContent: { minWidth: '100%' },
  table: { flex: 1 },
  row: { flexDirection: 'row' },
  line: { borderTopWidth: 1 },
  cell: { flex: 1, minWidth: SIZE.cellMin, paddingVertical: SPACE.sm, paddingHorizontal: SPACE.sm, justifyContent: 'center' },
  right: { alignItems: 'flex-end' },
});
