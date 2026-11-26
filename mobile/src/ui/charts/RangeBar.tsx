// 수급 범위 막대(명세 4.1·5.3·6.2절): 0…공급능력, 수요 구간은 series.demand 채움, 예비력 구간은 series.reserve 빗금 + 외곽선,
// 관심 경계(공급능력 ÷ 1.15)는 watch 색 눈금 + 삼각 표지 + 글자. 짝이 맞는 실시간 값일 때만 그린다(화면이 SupplyView로 판단).
// 막대 안에 글자를 두지 않는다(12 px 글자가 파랑 채움 위에서 두 테마 모두 4.5:1을 넘지 못한다): 수요·예비력은 위 줄, 경계는 아래 줄.
// role=img, 이름 = 식 문장 + 경계 문장. 누르는 기능이 없다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Defs, Line, Path, Rect } from 'react-native-svg';
import { CHART, SPACE } from '../tokens';
import { Text } from '../Text';
import { Value } from '../Values';
import { web } from '../web';
import { useTheme } from '../../app/theme';
import type { Provenanced } from '../../data/adapters';
import { mw } from '../../lib/format';
import { supplyFormulaText, watchBoundaryText, watchReservePctText } from '../../content/strings';
import { Hatch, chartStyles, useChartWidth, useSvgId } from './ChartFrame';
import { r2, rangeBarGeometry } from './chartMath';

export type RangeBarProps = {
  supply: Provenanced<number>;
  demand: Provenanced<number>;
  reserve: Provenanced<number>;
  /** 관심 경계 = 공급능력 ÷ 1.15(adaptSupply의 watchBoundary). */
  boundary: Provenanced<number>;
  width?: number;
  testID?: string;
};

const SPOKEN_MW = '메가와트';

function Swatch({ kind, hatchId }: { kind: 'demand' | 'reserve' | 'watch'; hatchId: string }) {
  const { theme } = useTheme();
  const s = CHART.swatch;
  const inset = CHART.barStroke / 2;
  return (
    <Svg width={s} height={s}>
      {kind === 'demand' ? <Rect x={0} y={0} width={s} height={s} fill={theme.series.demand} /> : null}
      {kind === 'reserve' ? <Rect x={inset} y={inset} width={s - CHART.barStroke} height={s - CHART.barStroke} fill={`url(#${hatchId})`}
        stroke={theme.series.reserve} strokeWidth={CHART.barStroke} /> : null}
      {kind === 'watch' ? <Path d={`M0 ${s}L${s / 2} 0L${s} ${s}Z`} fill={theme.series.watch} /> : null}
    </Svg>
  );
}

export function RangeBar({ supply, demand, reserve, boundary, width: fixedWidth, testID = 'range-bar' }: RangeBarProps) {
  const { theme } = useTheme();
  // 범위 막대는 카드 안쪽 여백 안에 둔다(ChartBleed 없이).
  const { width, onLayout } = useChartWidth(fixedWidth, true);
  const hatchId = useSvgId('pd-reserve');
  const geo = rangeBarGeometry(supply.value, demand.value, boundary.value, width);
  const name = `${supplyFormulaText(supply.value, demand.value, reserve.value, SPOKEN_MW)}. ${watchBoundaryText(boundary.value, SPOKEN_MW)}`;
  const barTop = CHART.barTick;
  const svgHeight = CHART.bar + 2 * CHART.barTick + CHART.triangle;
  const inset = CHART.barStroke / 2;
  return (
    <View role="img" aria-label={name} testID={testID} onLayout={onLayout} style={chartStyles.frame} {...web({ dataSet: { pdChart: 'range' } })}>
      {geo ? (
        <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={styles.stack}>
          <View style={styles.legend}>
            <View style={styles.item}>
              <Swatch kind="demand" hatchId={hatchId} />
              <Text variant="label" tone="text2">수요 <Value v={demand} format={mw} size="label" /></Text>
            </View>
            <View style={styles.item}>
              <Swatch kind="reserve" hatchId={hatchId} />
              <Text variant="label" tone="text2">예비력 <Value v={reserve} format={mw} size="label" derived /></Text>
            </View>
          </View>
          <Svg width={width} height={svgHeight}>
            <Defs><Hatch id={hatchId} color={theme.series.reserve} /></Defs>
            <Rect testID="range-demand" x={0} y={barTop} width={r2(geo.demandX)} height={CHART.bar} fill={theme.series.demand} />
            {geo.width - geo.demandX > CHART.barStroke
              ? <Rect testID="range-reserve" x={r2(geo.demandX + inset)} y={barTop + inset} width={r2(geo.width - geo.demandX - CHART.barStroke)}
                  height={CHART.bar - CHART.barStroke} fill={`url(#${hatchId})`} stroke={theme.series.reserve} strokeWidth={CHART.barStroke} />
              : null}
            <Line testID="range-boundary" x1={r2(geo.boundaryX)} x2={r2(geo.boundaryX)} y1={0} y2={CHART.bar + 2 * CHART.barTick}
              stroke={theme.series.watch} strokeWidth={CHART.dotRing} />
            <Path testID="range-boundary-mark" fill={theme.series.watch}
              d={`M${r2(geo.boundaryX)} ${CHART.bar + 2 * CHART.barTick}L${r2(geo.boundaryX + CHART.triangle / 2)} ${svgHeight}L${r2(geo.boundaryX - CHART.triangle / 2)} ${svgHeight}Z`} />
          </Svg>
          <View style={[styles.boundaryRow, { paddingRight: r2(Math.max(0, geo.width - geo.boundaryX - CHART.triangle / 2)) }]}>
            <Swatch kind="watch" hatchId={hatchId} />
            <Text variant="label" tone="text2" style={styles.shrink}>
              관심 경계 <Value v={boundary} format={mw} size="label" derived />{' '}
              <Text variant="label" tone="text2" dataSet={{ pdMeta: 'rule' }}>{`(${watchReservePctText()})`}</Text>
            </Text>
          </View>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  stack: { gap: SPACE.xs },
  legend: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', columnGap: SPACE.md, rowGap: SPACE.xs },
  item: { flexDirection: 'row', alignItems: 'center', gap: SPACE.xs },
  boundaryRow: { flexDirection: 'row', justifyContent: 'flex-end', alignItems: 'center', gap: SPACE.xs },
  shrink: { flexShrink: 1, textAlign: 'right' },
});
