// 오늘 24시간 가격(명세 4.1·7.8절): 기준선 계단 차트 또는 표, 거래시간 스텝퍼(24h 다음 01h로 돈다) + [지금으로], 판독 줄,
// 최고·최저·평균 요약, 가장 불확실한 시간, 출처. 고른 거래시간은 URL ?hour=에 남는다(부모가 바꾼다).
// [차트 | 표]는 보이는 그림 바로 위에 둔다: 표가 길어(시간대 3행 + 12행) 아래에 두면 다시 차트로 바꾸려고 끝까지 내려가야 한다.
// round 1 수정: 600 px 아래 시간별 표는 행 카드(D2), 표의 열은 거래시간 · 확정 · 기준선 · 80% 구간(D2), [지금으로]는 지금이어도 남아
// 까닭 있는 비활성(D4), 요약·불확실성은 본문 크기 한 문단(D17: 차트를 읽는 문장이라 12–13 px 메타가 아니다).
import { useState } from 'react';
import type React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { ChipSpec, ForecastCardView, HourRow } from './model';
import { ChipList } from './TodayParts';
import { price, tradeClockRange, tradeHourNo } from '../../lib/format';
import {
  CARD_TITLES, FORECAST_SUBJECT_SHORT, FULL_DAY_LABEL, HOUR_COLUMNS, PERIOD_COLUMNS, SHORT_DAY_LABEL, TABLE_CAPTIONS, TO_NOW_LABEL, TO_NOW_REASON,
  ZERO_INPUT_CELL,
} from '../../content/strings';
import { Button } from '../../ui/Button';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ChartBleed } from '../../ui/charts/ChartFrame';
import { StepChart } from '../../ui/charts/StepChart';
import { HourStepper, SegmentedControl } from '../../ui/Controls';
import type { SegmentOption } from '../../ui/Controls';
import { DataTable } from '../../ui/DataTable';
import { RichText } from '../../ui/Rich';
import { Text, keepTogether } from '../../ui/Text';
import { LAYOUT, SIZE, SPACE } from '../../ui/tokens';
import { Value } from '../../ui/Values';

const TITLE = CARD_TITLES['today-forecast'];
/** 스켈레톤 = 차트(220) + 스텝퍼(44) + 판독·요약 줄: 레이아웃이 뛰지 않게. */
const SKELETON_HEIGHT = 380;
const DASH = '—';
type Mode = 'chart' | 'table';
const MODES: readonly SegmentOption<Mode>[] = [{ value: 'chart', label: '차트' }, { value: 'table', label: '표' }];

function hourCells(row: HourRow): Record<string, React.ReactNode> {
  const p = row.point;
  const q50 = p.kind === 'zero' ? ZERO_INPUT_CELL : p.q50 ? <Value v={p.q50} format={price} size="label" showUnit={false} /> : DASH;
  const band = p.band ? (
    <Text variant="label" nowrap><Value v={p.band.q10} format={price} size="label" showUnit={false} />–<Value v={p.band.q90} format={price} size="label" showUnit={false} /></Text>
  ) : DASH;
  // 확정 가격은 받은 시간에만(없으면 표는 "—", 행 카드는 그 조각을 뺀다).
  const confirmed = row.confirmed ? <Value v={row.confirmed} format={price} size="label" showUnit={false} /> : null;
  return { hour: row.label, confirmed, q50, band };
}

export type ForecastCardProps = {
  view: ForecastCardView;
  /** StepChart 접근 이름 머리(오늘 화면 소스가 넘긴다: smp-truthfulness가 '예측 SMP'를 그 파일에서 찾는다). */
  chartLabel: (kind: 'baseline' | 'model' | 'fallback') => string;
  now: number;
  onSelectHour: (hour: number) => void;
  onToNow: () => void;
  onMethod: () => void;
  onSources: () => void;
  onRetry: () => void;
};

export function ForecastCard({ view, chartLabel, now, onSelectHour, onToNow, onMethod, onSources, onRetry }: ForecastCardProps) {
  const { width } = useWindowDimensions();
  const [mode, setMode] = useState<Mode>('chart');
  const [fullDay, setFullDay] = useState(false);
  const chips = (list: readonly ChipSpec[]) => <ChipList chips={list} />;
  if (view.state === 'loading') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-forecast"><StateView kind="loading" minHeight={SKELETON_HEIGHT} /></Card>;
  }
  if (view.state === 'error') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-forecast"><StateView kind="error" text={view.text} onRetry={onRetry} /></Card>;
  }
  if (view.state === 'empty') {
    return (
      <Card title={TITLE} chips={chips(view.chips)} onMethod={onMethod} testID="today-forecast">
        <StateView kind="empty" text={view.text} />
        <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
      </Card>
    );
  }
  const subject = FORECAST_SUBJECT_SHORT[view.kind];
  const rows = fullDay ? view.rows : view.rows.slice(view.window.from - 1, view.window.to);
  const phone = width < LAYOUT.rowCardsBelow;
  // 요약과 불확실성은 차트를 읽는 문장이라 본문 크기 한 문단(D17).
  const summary = view.uncertainty ? `${view.summary} ${view.uncertainty}.` : view.summary;
  return (
    <Card title={TITLE} chips={chips(view.chips)} onMethod={onMethod} testID="today-forecast">
      <SegmentedControl label="보기 방식" options={MODES} value={mode} onChange={setMode} />
      {mode === 'chart' ? (
        <View style={styles.group}>
          <Text variant="meta" tone="text2" dataSet={{ pdMeta: 'legend' }}>{view.legend}</Text>
          <ChartBleed>
            <StepChart forecast={view.view} label={chartLabel(view.kind)} now={now} dots={view.dots}
              selectedHour={view.selected} onSelectHour={onSelectHour} />
          </ChartBleed>
        </View>
      ) : (
        <View style={styles.group} testID="today-tables">
          {/* 시간대 요약은 짧은 4칸이라 늘 카드 폭에 맞춘다(320에서도 가로로 밀지 않는다, round 1). */}
          <DataTable caption={TABLE_CAPTIONS.periods} fit cellMin={width < LAYOUT.narrowBelow ? SIZE.cellMinNarrow : SIZE.cellMin} columns={[
            { key: 'period', title: PERIOD_COLUMNS.period, meta: true, grow: 1.5 }, { key: 'mean', title: PERIOD_COLUMNS.mean, align: 'right', meta: true },
            { key: 'min', title: PERIOD_COLUMNS.min, align: 'right', meta: true }, { key: 'max', title: PERIOD_COLUMNS.max, align: 'right', meta: true },
          ]} rows={view.periods.map(p => ({ key: p.id, cells: { period: p.label, mean: p.mean, min: p.min, max: p.max } }))} />
          {view.periodNote ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{view.periodNote}</Text> : null}
          {/* 열 순서(D2): 거래시간 · 확정 · 기준선 · 80% 구간. 600 px 아래는 같은 열의 행 카드("06–07시" / "07h" · 확정 가격 115.3 · …). */}
          <DataTable caption={TABLE_CAPTIONS.hours} cards={phone} testID={phone ? 'today-hour-cards' : undefined} columns={[
            { key: 'hour', title: HOUR_COLUMNS.hour, meta: true }, { key: 'confirmed', title: HOUR_COLUMNS.confirmed, align: 'right', meta: true },
            { key: 'q50', title: subject, align: 'right', meta: true }, { key: 'band', title: HOUR_COLUMNS.band, align: 'right', meta: true },
          ]} rows={rows.map(r => ({ key: String(r.hour), current: r.hour === view.selected, cells: hourCells(r), head: tradeClockRange(r.hour), sub: tradeHourNo(r.hour) }))} />
          <View style={styles.start}>
            <Button tone="ghost" label={fullDay ? SHORT_DAY_LABEL : FULL_DAY_LABEL} accessibilityLabel={fullDay ? SHORT_DAY_LABEL : FULL_DAY_LABEL}
              onPress={() => setFullDay(v => !v)} />
          </View>
        </View>
      )}
      <View style={styles.group}>
        <View style={styles.stepRow}>
          <HourStepper hour={view.selected} onChange={onSelectHour} wrap label={view.stepLabel} />
          {/* 지금이어도 지우지 않는다(D4): 누른 버튼이 사라지면 초점이 body로 떨어진다. 이름은 보이는 글자 그대로(WCAG 2.5.3). */}
          <Button tone="ghost" label={TO_NOW_LABEL} accessibilityLabel={TO_NOW_LABEL} onPress={onToNow}
            disabled={view.isNow} disabledReason={TO_NOW_REASON} />
        </View>
        <RichText line={view.reading} testID="today-reading" />
      </View>
      <Text variant="body" tone="text2" dataSet={{ pdMeta: 'summary' }} testID="today-summary">{keepTogether(summary, { variant: 'body', tone: 'text2' })}</Text>
      <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
    </Card>
  );
}

const styles = StyleSheet.create({
  group: { gap: SPACE.xs },
  stepRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm },
  start: { alignSelf: 'flex-start' },
});
