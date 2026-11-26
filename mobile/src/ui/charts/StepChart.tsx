// 24시간 기준선 계단(명세 4.1·5.3·6.2·7.5·7.8절): 자홍 점선 계단(q50) + 같은 색 8 % 띠(q10–q90, 1 px 테두리) + 확정 가격 점 +
// KST 지금선과 "지금 HH:MM" pill + 지난 시간 6 % 음영 + 0.00 입력 빗금 열(선·띠를 끊고 각주 한 번),
// x 눈금 00 06 12 18 24시(i/24 위치), y축 0부터, 단위 원/kWh는 맨 위 눈금 위.
// round 1 D10: y 위쪽은 기준선과 확정 가격으로 정하고(띠는 넣지 않는다) 그림 위로 넘친 띠는 그 시간 맨 위에 ▲로 표시한다. 확정 가격 점은
// 이번 방문에서 받은 값마다 하나씩 남는다(서버는 지금 시간 값만 준다).
// role=img + 요약 이름(안쪽 그림은 aria-hidden), 표 보기가 대안이다. 초점을 두면 ←/→/Home/End로, 누르면 가까운 시간으로
// 선택이 바뀌고 판독 줄은 부모가 그린다(떠 있는 툴팁 없음). 세로로 끌면 페이지가 스크롤된다(응답자 없음).
import React from 'react';
import { View } from 'react-native';
import { Circle, Defs, Line, Path, Rect, Svg } from 'react-native-svg';
import { CHART, RADIUS, SPACE } from '../tokens';
import { useAnnounce } from '../Live';
import { Text } from '../Text';
import { web } from '../web';
import { useTheme } from '../../app/theme';
import type { ForecastView, Provenanced } from '../../data/adapters';
import { PRICE_UNIT, tradeHour } from '../../lib/format';
import {
  ZERO_INPUT_LABEL, confirmedPointText, forecastSummaryText, hourTickText, nowMarkText, priceTickText, uncertaintyText,
} from '../../content/strings';
import { ChartLabel, Hatch, LABEL_HEIGHT, chartStyles, pickHandlers, useChartWidth, useSvgId } from './ChartFrame';
import {
  HOURS, X_TICK_HOURS, bandPath, boundaryX, clippedRuns, forecastStats, hourAtX, hourCenterX, hourKinds, labelAlign, labelLeft, nowFraction,
  pillLeft, r2, stepLayout, stepPath, stepScale, textWidth, tickBox, widestBand, yOf, zeroRuns,
} from './chartMath';

export type ConfirmedDot = { tradeHour: number; price: Provenanced<number> };

export type StepChartProps = {
  forecast: ForecastView;
  /** 접근 이름의 머리. 오늘 화면은 "기준선 예측 SMP 24시간"을 넘긴다(smp-truthfulness가 그 화면 소스의 '예측 SMP'를 본다). */
  label: string;
  /** 앱의 지금(kstNow). 예측일이 KST 오늘일 때만 지금선·지난 시간 음영을 그린다. */
  now: number | null;
  /** 7.8절 1 조건(같은 날짜·유한값·0.00 입력 아님)을 화면이 확인한 확정 가격들: 이번 방문에서 받은 거래시간마다 하나(D10). */
  dots?: readonly ConfirmedDot[];
  selectedHour?: number | null;
  /** 없으면 보기 전용(둘러보기 2단계): 초점·키·누름을 받지 않는다. */
  onSelectHour?: (hour: number) => void;
  /** 고정 폭(테스트·둘러보기). 없으면 onLayout으로 잰다. */
  width?: number;
  testID?: string;
};

const validHour = (h: unknown): h is number => typeof h === 'number' && Number.isInteger(h) && h >= 1 && h <= HOURS;
const NO_DOTS: readonly ConfirmedDot[] = [];

export function StepChart({
  forecast, label, now, dots: given = NO_DOTS, selectedHour = null, onSelectHour, width: fixedWidth, testID = 'step-chart',
}: StepChartProps) {
  const { theme } = useTheme();
  const announce = useAnnounce();
  const { width, onLayout } = useChartWidth(fixedWidth);
  const hatchId = useSvgId('pd-zero');
  const q50 = forecast.q50.value;
  const band = forecast.band;
  const kinds = hourKinds(q50);
  const zero = zeroRuns(kinds);
  // 확정 가격 점은 그 시간 기준선이 그릴 수 있는 값일 때만 둔다(7.8절 1의 마지막 안전장치. 날짜 조건은 화면이 본다).
  const dots = given.filter(d => validHour(d.tradeHour) && kinds[d.tradeHour - 1] === 'value').sort((a, b) => a.tradeHour - b.tradeHour);
  const scale = stepScale(q50, dots.map(d => d.price.value));
  const clipped = band ? clippedRuns(kinds, band.q90, scale) : [];
  const { height, plot } = stepLayout(width, scale, zero.length > 0);
  const bottom = plot.top + plot.height;
  const cell = plot.width / HOURS;
  const fraction = nowFraction(now, forecast.forecastDate);
  const nowX = fraction === null ? null : plot.left + fraction * plot.width;
  const selected = validHour(selectedHour) ? selectedHour : null;
  const c = theme.color;

  const stats = forecastStats(q50);
  const widest = band ? widestBand(q50, band.q10, band.q90) : null;
  const name = [`${label}:`, forecastSummaryText(stats, forecast.kind), widest ? `${uncertaintyText(widest)}.` : '',
    dots.length ? `${confirmedPointText(dots.map(d => ({ hour: d.tradeHour, value: d.price.value })))}.` : ''].filter(Boolean).join(' ');

  const pick = (hour: number) => {
    if (!onSelectHour) return;
    onSelectHour(hour);
    announce(tradeHour(hour));
  };
  const handlers = onSelectHour ? pickHandlers({ min: 1, max: HOURS, current: selected, pick, fromX: x => hourAtX(x, plot) }) : {};
  const unitBox = tickBox(PRICE_UNIT);
  const nowText = now === null ? '' : nowMarkText(now);
  const pillWidth = textWidth(nowText) + 2 * CHART.pillPad;

  return (
    <View role="img" aria-label={name} testID={testID} onLayout={onLayout}
      style={[chartStyles.frame, { height }, onSelectHour ? chartStyles.pointer : null]}
      {...web({ dataSet: { pdChart: 'step', pdPlot: `${r2(plot.left)} ${r2(plot.top)} ${r2(plot.width)} ${r2(plot.height)}`,
        pdScale: `${scale.bottom} ${scale.top}`, pdSelected: selected === null ? 'none' : String(selected) } })}
      {...handlers}>
      <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={chartStyles.layer}>
        <Svg width={width} height={height}>
          <Defs><Hatch id={hatchId} color={theme.chart.zeroHatch} /></Defs>
          {nowX !== null && nowX > plot.left
            ? <Rect testID="step-past" x={r2(plot.left)} y={r2(plot.top)} width={r2(nowX - plot.left)} height={r2(plot.height)} fill={theme.chart.pastShade} />
            : null}
          {zero.map(run => (
            <Rect key={`z${run.from}`} testID="step-zero" x={r2(boundaryX(run.from - 1, plot))} y={r2(plot.top)}
              width={r2((run.to - run.from + 1) * cell)} height={r2(plot.height)} fill={`url(#${hatchId})`} />
          ))}
          {scale.ticks.filter(t => t !== scale.bottom).map(t => (
            <Line key={`g${t}`} x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(yOf(t, scale, plot))} y2={r2(yOf(t, scale, plot))}
              stroke={c.border} strokeWidth={CHART.grid} />
          ))}
          {band ? <Path testID="step-band" d={bandPath(kinds, band.q10, band.q90, scale, plot)} fill={theme.series.band}
            stroke={theme.series.bandEdge} strokeWidth={CHART.axis} /> : null}
          {clipped.flatMap(run => Array.from({ length: run.to - run.from + 1 }, (_, k) => run.from + k)).map(hour => {
            const x = hourCenterX(hour, plot);
            const half = Math.min(CHART.clipMark, cell) / 2;
            return <Path key={`c${hour}`} testID="step-clipped" fill={theme.series.smp}
              d={`M${r2(x - half)} ${r2(plot.top + CHART.clipMark)}L${r2(x)} ${r2(plot.top)}L${r2(x + half)} ${r2(plot.top + CHART.clipMark)}Z`} />;
          })}
          <Path testID="step-line" d={stepPath(q50, kinds, scale, plot)} fill="none" stroke={theme.series.smp}
            strokeWidth={CHART.line} strokeDasharray={CHART.dash} />
          <Line testID="step-axis" x1={r2(plot.left)} x2={r2(plot.left + plot.width)} y1={r2(bottom)} y2={r2(bottom)}
            stroke={c.borderStrong} strokeWidth={CHART.axis} />
          {nowX !== null
            ? <Line testID="step-now-line" x1={r2(nowX)} x2={r2(nowX)} y1={r2(plot.top)} y2={r2(bottom)} stroke={c.borderStrong} strokeWidth={CHART.nowLine} />
            : null}
          {selected !== null && onSelectHour
            ? <Rect testID="step-selected-mark" x={r2(boundaryX(selected - 1, plot))} y={r2(bottom + CHART.axis + 1)} width={r2(cell)}
                height={CHART.selMark} fill={c.primary} />
            : null}
          {selected !== null && onSelectHour && kinds[selected - 1] === 'value'
            ? <Circle testID="step-selected" cx={r2(hourCenterX(selected, plot))} cy={r2(yOf(q50[selected - 1] as number, scale, plot))}
                r={CHART.dotR} fill={c.surface} stroke={c.primary} strokeWidth={CHART.dotRing} />
            : null}
          {dots.map(d => (
            <Circle key={`d${d.tradeHour}`} testID="step-confirmed" cx={r2(hourCenterX(d.tradeHour, plot))} cy={r2(yOf(d.price.value, scale, plot))}
              r={CHART.dotR} fill={c.text1} stroke={c.surface} strokeWidth={CHART.dotRing} />
          ))}
          {zero.length
            ? <Rect x={CHART.edge} y={r2(bottom + CHART.xRow + (CHART.noteRow - CHART.swatch) / 2)} width={CHART.swatch} height={CHART.swatch}
                fill={`url(#${hatchId})`} stroke={c.borderStrong} strokeWidth={CHART.axis} />
            : null}
        </Svg>
        <ChartLabel text={PRICE_UNIT} meta="unit" left={CHART.edge} top={0} width={unitBox} align="left" />
        {scale.ticks.map(t => (
          <ChartLabel key={`y${t}`} text={priceTickText(t)} meta="axis" left={0} top={r2(yOf(t, scale, plot) - LABEL_HEIGHT / 2)}
            width={r2(plot.left - CHART.tickGap)} align="right" />
        ))}
        {X_TICK_HOURS.map(h => {
          const text = hourTickText(h, h === HOURS);
          const box = tickBox(text);
          const x = boundaryX(h, plot);
          return <ChartLabel key={`x${h}`} text={text} meta="axis" left={r2(labelLeft(x, box, width))} align={labelAlign(x, box, width)}
            top={r2(bottom + CHART.xLabelTop)} width={box} />;
        })}
        {zero.length
          ? <ChartLabel text={ZERO_INPUT_LABEL} meta="note" testID="step-zero-note" align="left" left={CHART.edge + CHART.swatch + SPACE.xs}
              top={r2(bottom + CHART.xRow + (CHART.noteRow - LABEL_HEIGHT) / 2)} width={tickBox(ZERO_INPUT_LABEL)} />
          : null}
        {nowX !== null ? (
          <View testID="step-now-pill" style={{ position: 'absolute', top: CHART.pillTop, height: CHART.pill, width: pillWidth,
            left: r2(pillLeft(nowX, pillWidth, unitBox, width)), borderRadius: RADIUS.pill, backgroundColor: c.nowPillBg, justifyContent: 'center' }}>
            <Text variant="meta" weight="600" color={c.nowPillText} nowrap dataSet={{ pdMeta: 'now' }} style={{ textAlign: 'center' }}>{nowText}</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}
