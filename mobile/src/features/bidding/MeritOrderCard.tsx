// 가격은 어떻게 정해지나요?(명세 4.2·7.1·7.9절, 처음에는 접힘): 펼치면 예시 발전기 6개(examples.ts)를 싼 순서로 쌓은 정적 그림과
// 예시 수요선, 설명 글(guide.ko.json의 bid-merit, 숫자 없음), 그림의 글 대안 6행 목록. 모두 예시라 ┆예시┆ 칩을 달고, 그림은 예시 글자색
// (7.1절)으로 그리며, 숫자는 메타·SampleValue로만 보인다. 실제 발전소 이름은 쓰지 않는다.
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Line, Rect } from 'react-native-svg';
import { meritExample } from './model';
import { useTheme } from '../../app/theme';
import { EXAMPLE_MERIT_BLOCKS, EXAMPLE_MERIT_DEMAND_MW } from '../../data/examples';
import { PRICE_UNIT, mw, num } from '../../lib/format';
import { CHIPS } from '../../lib/provenance';
import { CARD_TITLES, MERIT_TOGGLE, meritDemandText, meritPriceText, mwTickText, priceTickText } from '../../content/strings';
import { GuideGate } from '../guide/sheets';
import { Button } from '../../ui/Button';
import { Card } from '../../ui/Card';
import { ProvenanceChip } from '../../ui/Chips';
import { ChartLabel, LABEL_HEIGHT, chartStyles, useChartWidth } from '../../ui/charts/ChartFrame';
import { labelLeft, ladderBars, ladderLayout, mwTicks, niceScale, r2, tickBox, yOf } from '../../ui/charts/chartMath';
import { Text, keepTogether } from '../../ui/Text';
import { CHART, SPACE } from '../../ui/tokens';
import { SampleValue } from '../../ui/Values';
import { web } from '../../ui/web';

const TITLE = CARD_TITLES['bid-merit'];
/** 0원 발전기도 보이게 막대 높이의 바닥을 둔다. */
const MIN_BAR = 3;
const won = (v: number) => num(v, 0);

/** 예시 메리트 오더 그림: x = 누적 공급 가능량, 높이 = 입찰 가격, 세로 점선 = 예시 수요. 마지막으로 필요한 발전기는 채워 그린다. */
function MeritChart({ marginal, summary }: { marginal: number; summary: string }) {
  const { theme } = useTheme();
  const { width, onLayout } = useChartWidth(undefined, true);
  const items = EXAMPLE_MERIT_BLOCKS.map(b => ({ price: b.bidPrice.value, probability: 0, quantity: b.capacityMw.value }));
  const scale = niceScale(items.map(it => it.price));
  const { height, plot } = ladderLayout(width, scale);
  const bottom = plot.top + plot.height;
  const bars = ladderBars(items, scale, plot);
  const total = items.reduce((a, it) => a + it.quantity, 0);
  const xOf = (v: number) => plot.left + (total > 0 ? v / total : 0) * plot.width;
  const demandX = xOf(EXAMPLE_MERIT_DEMAND_MW.value);
  const ink = theme.chip.example;
  const demandText = meritDemandText(EXAMPLE_MERIT_DEMAND_MW.value);
  // 수요선은 늘 마지막 발전기의 폭을 지나므로 그 가격 글자 위아래로 나눠 그린다(글자를 가로지르지 않게).
  const markTop = bars[marginal] ? Math.min(bars[marginal].y, bottom - MIN_BAR) : bottom;
  const gapTop = markTop - LABEL_HEIGHT - CHART.labelGap * 2;
  const demandLines = [[LABEL_HEIGHT, gapTop], [markTop, bottom]].filter(([a, b]) => b > a);
  return (
    <View role="img" aria-label={summary} testID="merit-chart" onLayout={onLayout} style={[chartStyles.frame, { height }]}
      {...web({ dataSet: { pdChart: 'merit' } })}>
      <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={chartStyles.layer}>
        <Svg width={width} height={height}>
          {bars.map(b => {
            const top = Math.min(b.y, bottom - MIN_BAR);
            return (
              <Rect key={b.index} testID="merit-bar" x={r2(b.x0)} y={r2(top)} width={r2(Math.max(0, b.x1 - b.x0 - CHART.barGap))} height={r2(bottom - top)}
                fill={b.index === marginal ? ink.text : ink.bg} stroke={ink.text} strokeWidth={CHART.barStroke} />
            );
          })}
          <Line x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(bottom)} y2={r2(bottom)} stroke={theme.color.borderStrong} strokeWidth={CHART.axis} />
          {demandLines.map(([a, b]) => (
            <Line key={a} testID="merit-demand" x1={r2(demandX)} x2={r2(demandX)} y1={r2(a)} y2={r2(b)} stroke={theme.color.text1}
              strokeWidth={CHART.nowLine} strokeDasharray={CHART.dash} />
          ))}
        </Svg>
        <ChartLabel text={PRICE_UNIT} meta="unit" left={CHART.edge} top={0} width={tickBox(PRICE_UNIT)} align="left" />
        <ChartLabel text={demandText} meta="chart" tone="text1" weight="600" top={0}
          left={r2(labelLeft(demandX, tickBox(demandText), width))} width={tickBox(demandText)} />
        {scale.ticks.map(t => (
          <ChartLabel key={`y${t}`} text={priceTickText(t)} meta="axis" left={0} top={r2(yOf(t, scale, plot) - LABEL_HEIGHT / 2)}
            width={r2(plot.left - CHART.tickGap)} align="right" />
        ))}
        {mwTicks(total).map((v, i, all) => {
          const text = mwTickText(v, i === all.length - 1);
          return <ChartLabel key={`x${v}`} text={text} meta="axis" left={r2(labelLeft(xOf(v), tickBox(text), width))} top={r2(bottom + CHART.xLabelTop)} width={tickBox(text)} />;
        })}
        {bars.map(b => {
          const text = meritPriceText(items[b.index].price);
          const box = Math.max(b.x1 - b.x0, tickBox(text));
          return <ChartLabel key={`p${b.index}`} text={text} meta="chart" tone="text2" weight="600" width={r2(box)}
            left={r2(labelLeft((b.x0 + b.x1 - CHART.barGap) / 2, box, width))} top={r2(Math.min(b.y, bottom - MIN_BAR) - LABEL_HEIGHT - CHART.labelGap)} />;
        })}
      </View>
    </View>
  );
}

export function MeritOrderCard({ onMethod }: { onMethod: () => void }) {
  const [open, setOpen] = useState(false);
  const toggle = open ? MERIT_TOGGLE.close : MERIT_TOGGLE.open;
  const { marginal, summary } = meritExample();
  return (
    <Card title={TITLE} chips={<><ProvenanceChip chip={CHIPS.example} /><ProvenanceChip chip={CHIPS.simplified} /></>} onMethod={onMethod} testID="bid-merit">
      <View style={styles.start}>
        {/* 이름 = 카드 제목 + 보이는 글자(WCAG 2.5.3). 펼침 상태는 aria-expanded로 알린다. */}
        <Button tone="ghost" label={toggle} accessibilityLabel={`${TITLE} ${toggle}`} expanded={open} onPress={() => setOpen(v => !v)} />
      </View>
      {open ? (
        <View style={styles.body} testID="merit-body">
          {summary && marginal >= 0 ? <MeritChart marginal={marginal} summary={summary} /> : null}
          {summary ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'summary' }}>{keepTogether(summary, { variant: 'label', tone: 'text2' })}</Text> : null}
          <GuideGate>{content => content.cards['bid-merit'].body.map(p => <Text key={p} variant="body" tone="text2">{p}</Text>)}</GuideGate>
          <View style={styles.list} testID="merit-list">
            {EXAMPLE_MERIT_BLOCKS.map(b => (
              <Text key={b.name.value} variant="label" tone="text2">
                {b.name.value} · <SampleValue v={b.capacityMw} format={mw} size="label" /> · <SampleValue v={b.bidPrice} format={won} size="label" />
              </Text>
            ))}
          </View>
        </View>
      ) : null}
    </Card>
  );
}

const styles = StyleSheet.create({
  start: { alignSelf: 'flex-start' },
  body: { gap: SPACE.sm },
  list: { gap: SPACE.xs },
});
