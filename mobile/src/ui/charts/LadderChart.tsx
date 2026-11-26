// 입찰 계단(명세 4.2·5.3·6.2절): x = 누적 물량(예시 발전소, 구간당 5 MW), y = 가격 0부터(맨 위 눈금은 50 단위 올림),
// 막대 위에 낙찰 가능성(추정) 글자 12 px. 360 미만에서는 1·4·7·10구간 글자만 보인다(표에는 전부 있다).
// round 1 D12: 막대는 채우지 않고 외곽선 + ▨ 추정 칩과 같은 빗금으로 그린다(10개 단색 막대가 한 덩어리로 읽혔고, 자홍은 오늘 차트의 예측
// 색이었다). 그 시간 기준선은 자홍 점선, 80% 구간은 옅은 띠로 깔아 어느 구간이 기준선 아래인지 보인다. 고른 구간만 primary로 채운다.
// 단위 옆에 "막대 위 % = 낙찰 가능성(추정)". 불투명도로 확률을 나타내지 않는다. role=img + 요약 이름, ←/→/Home/End와 누름으로 고른다.
import React from 'react';
import { View, useWindowDimensions } from 'react-native';
import { Defs, Line, Rect, Svg } from 'react-native-svg';
import { CHART, LAYOUT } from '../tokens';
import { useAnnounce } from '../Live';
import { web } from '../web';
import { useTheme } from '../../app/theme';
import type { HourBidView } from '../../data/adapters';
import { PRICE_UNIT } from '../../lib/format';
import { LADDER_CAPTION, ladderReadingText, ladderSummaryText, mwTickText, priceTickText, probabilityText } from '../../content/strings';
import { ChartLabel, Hatch, LABEL_HEIGHT, chartStyles, pickHandlers, useChartWidth, useSvgId } from './ChartFrame';
import {
  barAtX, labelAlign, labelLeft, ladderBars, ladderLabelIndices, ladderLayout, ladderStats, mwTicks, niceScale, r2, tickBox, yOf,
} from './chartMath';

export type LadderChartProps = {
  /** 한 거래시간의 계산기 결과(adaptSegments). 가격·낙찰 가능성은 추정, 물량은 예시다. */
  hour: HourBidView;
  /** 접근 이름의 꼬리(예: "18–19시 입찰 계단"). 이름은 6.2절 요약("10구간, …")으로 시작한다. */
  label: string;
  /** 고른 구간(0부터). */
  selected?: number | null;
  onSelect?: (index: number) => void;
  width?: number;
  testID?: string;
};

export function LadderChart({ hour, label, selected = null, onSelect, width: fixedWidth, testID = 'ladder-chart' }: LadderChartProps) {
  const { theme } = useTheme();
  const announce = useAnnounce();
  const { width: windowWidth } = useWindowDimensions();
  const { width, onLayout } = useChartWidth(fixedWidth);
  const hatchId = useSvgId('pd-est');
  const items = hour.segments.map(s => ({ price: s.price.value, probability: s.probability.value, quantity: s.quantity.value }));
  const q50 = hour.baselineQ50;
  const band = hour.baselineBand;
  // 눈금은 막대·기준선·띠를 모두 담는다(0부터, 50 단위 올림).
  const scale = niceScale([...items.map(it => it.price), q50, band ? band.q90 : null]);
  const { height, plot } = ladderLayout(width, scale);
  const bottom = plot.top + plot.height;
  const bars = ladderBars(items, scale, plot);
  const shown = new Set(ladderLabelIndices(items.length, windowWidth < LAYOUT.narrowBelow));
  const total = items.reduce((a, it) => a + (it.quantity > 0 ? it.quantity : 0), 0);
  const stats = ladderStats(items);
  const name = `${stats ? ladderSummaryText(stats) : '계산된 구간이 없습니다'} · ${label}`;
  const current = selected !== null && selected >= 0 && selected < items.length ? selected : null;
  const c = theme.color;

  const pick = (index: number) => {
    if (!onSelect || !items[index]) return;
    onSelect(index);
    const it = items[index];
    announce(ladderReadingText(index, it.quantity, it.price, it.probability));
  };
  const handlers = onSelect && items.length
    ? pickHandlers({ min: 0, max: items.length - 1, current, pick, fromX: x => barAtX(x, bars) })
    : {};
  const xOf = (mwValue: number) => plot.left + (total > 0 ? mwValue / total : 0) * plot.width;
  const unitBox = tickBox(PRICE_UNIT);
  const captionBox = tickBox(LADDER_CAPTION);
  const inset = CHART.barStrokeLadder / 2;

  return (
    <View role="img" aria-label={name} testID={testID} onLayout={onLayout}
      style={[chartStyles.frame, { height }, onSelect ? chartStyles.pointer : null]}
      {...web({ dataSet: { pdChart: 'ladder', pdPlot: `${r2(plot.left)} ${r2(plot.top)} ${r2(plot.width)} ${r2(plot.height)}`,
        pdScale: `${scale.bottom} ${scale.top}`, pdSelected: current === null ? 'none' : String(current) } })}
      {...handlers}>
      <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={chartStyles.layer}>
        <Svg width={width} height={height}>
          <Defs><Hatch id={hatchId} color={theme.chip.estimated.hatch ?? c.border} /></Defs>
          {scale.ticks.filter(t => t !== scale.bottom).map(t => (
            <Line key={`g${t}`} x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(yOf(t, scale, plot))} y2={r2(yOf(t, scale, plot))}
              stroke={c.border} strokeWidth={CHART.grid} />
          ))}
          {band ? (
            <Rect testID="ladder-band" x={r2(plot.left)} y={r2(yOf(band.q90, scale, plot))} width={r2(plot.width)}
              height={r2(Math.max(0, yOf(band.q10, scale, plot) - yOf(band.q90, scale, plot)))} fill={theme.series.band}
              stroke={theme.series.bandEdge} strokeWidth={CHART.axis} />
          ) : null}
          {bars.map(b => {
            const on = b.index === current;
            const w = Math.max(0, b.x1 - b.x0 - CHART.barGap);
            const top = Math.min(b.y, bottom);
            return (
              <Rect key={`b${b.index}`} testID={on ? 'ladder-bar-selected' : 'ladder-bar'} x={r2(b.x0 + inset)} y={r2(top + inset)}
                width={r2(Math.max(0, w - CHART.barStrokeLadder))} height={r2(Math.max(0, bottom - top - inset))}
                fill={on ? c.primary : `url(#${hatchId})`} stroke={on ? c.primary : c.borderStrong} strokeWidth={CHART.barStrokeLadder} />
            );
          })}
          {q50 !== null ? (
            <Line testID="ladder-baseline" x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(yOf(q50, scale, plot))} y2={r2(yOf(q50, scale, plot))}
              stroke={theme.series.smp} strokeWidth={CHART.line} strokeDasharray={CHART.dash} />
          ) : null}
          <Line testID="ladder-axis" x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(bottom)} y2={r2(bottom)}
            stroke={c.borderStrong} strokeWidth={CHART.axis} />
          {current !== null && onSelect
            ? <Rect testID="ladder-selected-mark" x={r2(bars[current].x0)} y={r2(bottom + CHART.axis + 1)}
                width={r2(Math.max(0, bars[current].x1 - bars[current].x0 - CHART.barGap))} height={CHART.selMark} fill={c.primary} />
            : null}
        </Svg>
        <ChartLabel text={PRICE_UNIT} meta="unit" left={CHART.edge} top={0} width={unitBox} align="left" />
        {/* 단위 줄 오른쪽: 막대 위 숫자의 뜻(round 1 멘토). 오른쪽 끝에 붙여 글자 간격을 넓혀도 안쪽으로만 자란다. */}
        <ChartLabel text={LADDER_CAPTION} meta="caption" testID="ladder-caption" tone="text2" align="right"
          left={r2(Math.max(CHART.edge + unitBox, width - CHART.edge - captionBox))} top={0} width={captionBox} />
        {scale.ticks.map(t => (
          <ChartLabel key={`y${t}`} text={priceTickText(t)} meta="axis" left={0} top={r2(yOf(t, scale, plot) - LABEL_HEIGHT / 2)}
            width={r2(plot.left - CHART.tickGap)} align="right" />
        ))}
        {mwTicks(total).map((v, i, all) => {
          const text = mwTickText(v, i === all.length - 1);
          const box = tickBox(text);
          const x = xOf(v);
          return <ChartLabel key={`x${v}`} text={text} meta="axis" left={r2(labelLeft(x, box, width))} align={labelAlign(x, box, width)}
            top={r2(bottom + CHART.xLabelTop)} width={box} />;
        })}
        {bars.filter(b => shown.has(b.index)).map(b => {
          const text = probabilityText(items[b.index].probability);
          const box = Math.max(b.x1 - b.x0, tickBox(text));
          const x = (b.x0 + b.x1 - CHART.barGap) / 2;
          return <ChartLabel key={`p${b.index}`} testID="ladder-prob" text={text} meta="chart" tone="text2" weight="600" halo={c.surface}
            left={r2(labelLeft(x, box, width))} align={labelAlign(x, box, width)}
            top={r2(Math.min(b.y, bottom) - LABEL_HEIGHT - CHART.labelGap)} width={r2(box)} />;
        })}
      </View>
    </View>
  );
}
