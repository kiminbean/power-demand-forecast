// 오늘 24시간 가격(명세 4.1·7.8절): 기준선 계단 차트 또는 표, 거래시간 스텝퍼(24h 다음 01h로 돈다) + [지금으로], 판독 줄,
// 최고·최저·평균 요약, 가장 불확실한 시간, 출처. 고른 거래시간은 URL ?hour=에 남는다(부모가 바꾼다).
// [차트 | 표]는 보이는 그림 바로 위에 둔다: 표가 길어(시간대 3행 + 12행) 아래에 두면 다시 차트로 바꾸려고 끝까지 내려가야 한다.
import React, { useState } from 'react';
import { StyleSheet, View } from 'react-native';
import type { ChipSpec, ForecastCardView, HourRow } from './model';
import { ChipList } from './TodayParts';
import { price } from '../../lib/format';
import {
  CARD_TITLES, FORECAST_SUBJECT_SHORT, FULL_DAY_LABEL, HOUR_COLUMNS, PERIOD_COLUMNS, SHORT_DAY_LABEL, TABLE_CAPTIONS, TO_NOW_LABEL,
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
import { SPACE } from '../../ui/tokens';
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
    <Text variant="label"><Value v={p.band.q10} format={price} size="label" showUnit={false} />–<Value v={p.band.q90} format={price} size="label" showUnit={false} /></Text>
  ) : DASH;
  const confirmed = row.confirmed ? <Value v={row.confirmed} format={price} size="label" showUnit={false} /> : DASH;
  return { hour: row.label, q50, band, confirmed };
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
  return (
    <Card title={TITLE} chips={chips(view.chips)} onMethod={onMethod} testID="today-forecast">
      <SegmentedControl label="보기 방식" options={MODES} value={mode} onChange={setMode} />
      {mode === 'chart' ? (
        <View style={styles.group}>
          <Text variant="meta" tone="text2" dataSet={{ pdMeta: 'legend' }}>{view.legend}</Text>
          <ChartBleed>
            <StepChart forecast={view.view} label={chartLabel(view.kind)} now={now} confirmed={view.confirmed}
              selectedHour={view.selected} onSelectHour={onSelectHour} />
          </ChartBleed>
        </View>
      ) : (
        <View style={styles.group} testID="today-tables">
          <DataTable caption={TABLE_CAPTIONS.periods} columns={[
            { key: 'period', title: PERIOD_COLUMNS.period, meta: true }, { key: 'mean', title: PERIOD_COLUMNS.mean, align: 'right', meta: true },
            { key: 'min', title: PERIOD_COLUMNS.min, align: 'right', meta: true }, { key: 'max', title: PERIOD_COLUMNS.max, align: 'right', meta: true },
          ]} rows={view.periods.map(p => ({ key: p.id, cells: { period: p.label, mean: p.mean, min: p.min, max: p.max } }))} />
          {view.periodNote ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{view.periodNote}</Text> : null}
          <DataTable caption={TABLE_CAPTIONS.hours} columns={[
            // 글자 칸("—", "— (입력 0.00)")은 메타, 값 칸은 Value(data-pd-value)다.
            { key: 'hour', title: HOUR_COLUMNS.hour, meta: true }, { key: 'q50', title: subject, align: 'right', meta: true },
            { key: 'band', title: HOUR_COLUMNS.band, align: 'right', meta: true }, { key: 'confirmed', title: HOUR_COLUMNS.confirmed, align: 'right', meta: true },
          ]} rows={rows.map(r => ({ key: String(r.hour), current: r.hour === view.selected, cells: hourCells(r) }))} />
          <View style={styles.start}>
            <Button tone="ghost" label={fullDay ? SHORT_DAY_LABEL : FULL_DAY_LABEL} accessibilityLabel={fullDay ? SHORT_DAY_LABEL : FULL_DAY_LABEL}
              onPress={() => setFullDay(v => !v)} />
          </View>
        </View>
      )}
      <View style={styles.group}>
        <View style={styles.stepRow}>
          <HourStepper hour={view.selected} onChange={onSelectHour} wrap label={view.stepLabel} />
          {/* 접근 이름은 보이는 글자 그대로(WCAG 2.5.3 이름 안의 라벨): 스텝퍼 옆이라 뜻이 분명하다. */}
          {view.isNow ? null : <Button tone="ghost" label={TO_NOW_LABEL} accessibilityLabel={TO_NOW_LABEL} onPress={onToNow} />}
        </View>
        <RichText line={view.reading} testID="today-reading" />
      </View>
      <View style={styles.group}>
        <Text variant="label" tone="text2" dataSet={{ pdMeta: 'summary' }}>{keepTogether(view.summary, { variant: 'label', tone: 'text2' })}</Text>
        {view.uncertainty
          ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'summary' }}>{keepTogether(view.uncertainty, { variant: 'label', tone: 'text2' })}</Text>
          : null}
      </View>
      <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
    </Card>
  );
}

const styles = StyleSheet.create({
  group: { gap: SPACE.xs },
  stepRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm },
  start: { alignSelf: 'flex-start' },
});
