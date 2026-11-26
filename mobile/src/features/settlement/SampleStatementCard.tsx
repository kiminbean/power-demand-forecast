// 예시 정산서(명세 4.3·7.3·7.9절): ┆예시 · 실제 값 아님┆ 리본 → 머리(제목 · [i]) → 가상 발전소 한 줄 → well 면의 정산서
// (① 하루전시장 · ② 실시간 차이 · 합계) → 계산 설명 두 줄. 숫자는 모두 examples.ts의 예시 값이라 SampleValue(text2, 한 단계 작게,
// 화면 읽기에 "(예시)")로만 보이고 출처 줄이 없다(7.3절 정산 행: 출처·시각 "—"). 행은 휴대폰 폭(600 미만, 4.6절)에서 두 줄
// (라벨 / 식 = 값, 4.3절은 320을 들었다), 그보다 넓으면 한 줄이고 식은 오른쪽에 붙는다. 들어가는 만큼 자연히 줄을 바꾸면 430에서
// ① 행은 두 줄 · ② 행은 한 줄로 섞여서 폭 구간으로 정한다. 둘러보기 4단계(F10)도 이 카드를 그대로 쓴다.
// StatementFigure는 ①+② 식 그림(폭포)이다. round 1 D14: 1280 패널에만 있던 그림을 모든 폭에서 이 카드 안에 둔다(768 정산 화면이 반쯤
// 비었다). 빈 상자 둘과 머리카락 선처럼 보이지 않게 ①·합계는 예시 호박색으로 채우고 ②는 진한 호박색 칸으로 내려간다. y는 0부터다.
import React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { settlementView } from './model';
import type { SettlementView } from './model';
import { useTheme } from '../../app/theme';
import { EXAMPLE_PLANT } from '../../data/examples';
import { krw, mw, num } from '../../lib/format';
import { CARD_TITLES, STATEMENT_NOTES, STATEMENT_PERIOD, STATEMENT_ROWS } from '../../content/strings';
import { Card } from '../../ui/Card';
import { SampleRibbon } from '../../ui/Chips';
import { Text } from '../../ui/Text';
import { CHART, LAYOUT, RADIUS, SPACE } from '../../ui/tokens';
import { SampleValue } from '../../ui/Values';
import { web } from '../../ui/web';

const TITLE = CARD_TITLES['settle-sample'];
const whole = (v: number) => num(v, 0);
/** 정산 예시 금액은 만원 단위로 떨어진다: "1,800만원"(4.3절). */
const manwon = (v: number) => krw(v, 0);
/** 그림 치수: 막대 칸 높이(②의 −84만원도 5 px 넘게 보인다), 막대 바닥 높이, 칸 안 막대 좌우 여백. */
const FIGURE = { plot: 112, minBar: 3, inset: '18%' } as const;
/** 정산서 행을 한 줄로 두는 폭(4.6절 폭 구간: 600 미만은 휴대폰). 시트가 가운데로 오는 폭과 같은 경계다. */
const ONE_LINE_FROM = LAYOUT.sheetCenterFrom;

export function SampleStatementCard({ onMethod }: { onMethod?: () => void }) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const stacked = width < ONE_LINE_FROM;
  const view = settlementView();
  return (
    <Card title={TITLE} ribbon={<SampleRibbon />} onMethod={onMethod} testID="settle-sample">
      <Text variant="label" tone="text2" testID="settle-plant">
        {EXAMPLE_PLANT.name.value} · <SampleValue v={EXAMPLE_PLANT.capacityMw} format={mw} size="label" /> (가상) · {STATEMENT_PERIOD}
      </Text>
      <View style={[styles.statement, { backgroundColor: theme.color.well }]} testID="settle-statement">
        {view.rows.map(r => (
          <View key={r.id} style={[styles.row, stacked ? styles.stack : null]} testID={`settle-row-${r.id}`}>
            <Text variant="label" tone="text2">{r.label}</Text>
            <Text variant="label" tone="text2" style={styles.end}>
              <SampleValue v={r.quantity} format={whole} size="label" />{' × '}<SampleValue v={r.unitPrice} format={whole} size="label" />
              {' = '}<SampleValue v={r.amount} format={manwon} size="label" showUnit={false} />
            </Text>
          </View>
        ))}
        <View style={[styles.rule, { backgroundColor: theme.color.border }]} />
        <View style={styles.row} testID="settle-row-total">
          <Text variant="body" weight="700">{STATEMENT_ROWS.total}</Text>
          <View style={styles.end}><SampleValue v={view.total} format={manwon} size="title" showUnit={false} /></View>
        </View>
      </View>
      <StatementFigure view={view} />
      <View style={styles.notes}>
        {STATEMENT_NOTES.map(n => <Text key={n} variant="label" tone="text2">{n}</Text>)}
      </View>
    </Card>
  );
}

/** ①+② 식 그림: 위 줄은 "1,800만원 + −84만원 = 1,716만원", 아래는 막대(②는 ①의 끝에서 내려가는 칸), 칸 이름. 이름은 식 문장. */
export function StatementFigure({ view }: { view: SettlementView }) {
  const { theme } = useTheme();
  const ink = theme.chip.example;
  const fill = theme.chart.exampleFill;
  const { bars, top, summary } = view.figure;
  const px = (v: number) => (top > 0 ? Math.round((v / top) * FIGURE.plot) : 0);
  return (
    <View role="img" aria-label={summary} testID="settle-figure" {...web({ dataSet: { pdChart: 'settlement' } })}>
      <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={styles.figure}>
        {bars.map((b, i) => (
          <React.Fragment key={b.id}>
            {i > 0 ? <Text variant="meta" weight="600" tone="text2">{i === bars.length - 1 ? '=' : '+'}</Text> : null}
            <View style={styles.col}>
              <Text variant="meta" weight="600" tone="text2" nowrap dataSet={{ pdExample: 'true' }}>{manwon(b.amount.value)}</Text>
              <View style={[styles.plot, { borderBottomColor: theme.color.borderStrong }]}>
                <View testID="settle-figure-bar" style={[styles.bar, {
                  bottom: px(b.from), height: Math.max(FIGURE.minBar, px(b.to) - px(b.from)), borderColor: ink.text,
                  backgroundColor: b.id === 'deviation' ? theme.chart.exampleMark : fill,
                }]} />
              </View>
              <Text variant="meta" tone="text3" style={styles.center}>{b.label}</Text>
            </View>
          </React.Fragment>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  statement: { borderRadius: RADIUS.control, padding: SPACE.md, gap: SPACE.sm },
  row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'baseline', columnGap: SPACE.md, rowGap: SPACE.xs },
  stack: { flexDirection: 'column', alignItems: 'flex-start' },
  end: { marginLeft: 'auto', textAlign: 'right' },
  rule: { height: StyleSheet.hairlineWidth, alignSelf: 'stretch' },
  notes: { gap: SPACE.xs },
  figure: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.xs },
  col: { flex: 1, minWidth: 0, alignItems: 'center', gap: SPACE.xs },
  plot: { alignSelf: 'stretch', height: FIGURE.plot, borderBottomWidth: CHART.axis },
  bar: { position: 'absolute', left: FIGURE.inset, right: FIGURE.inset, borderWidth: CHART.barStroke },
  center: { textAlign: 'center' },
});
