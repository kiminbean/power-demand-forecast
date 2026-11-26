// 예시 발전소로 입찰 계산해 보기(명세 4.2·7.2·7.8절): ┆예시 · 가상 발전소┆ 리본, 거래시간 스텝퍼(캐시된 24시간을 읽고 새로 요청하지
// 않는다, 양 끝에서 멈춤), 성향(바꿀 때만 요청), 계단 차트 또는 표, 요약·낙찰 가능성 주의·가격 근거 공개 줄, "KPX에 아무것도 보내지
// 않습니다". 물량·기대 수입은 예시(SampleValue), 가격·낙찰 가능성은 추정이다. 기대 수입은 표에만 둔다(4.2절).
import React, { useState } from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { BidCardView, BidRow } from './model';
import { RISKS as RISK_KEYS } from '../../app/routes';
import type { Risk } from '../../app/routes';
import { useTheme } from '../../app/theme';
import { EXAMPLE_PLANT } from '../../data/examples';
import { krwFromThousand, mw, pct, price } from '../../lib/format';
import { CHIPS } from '../../lib/provenance';
import {
  BID_COLUMNS, CARD_TITLES, NO_SEND_NOTE, RECALC_LABEL, RISK_LABELS, bidDayText, bidHourLabel, ladderReadingText,
} from '../../content/strings';
import { ChipList } from '../today/TodayParts';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ProvenanceChip, SampleRibbon } from '../../ui/Chips';
import { ChartBleed } from '../../ui/charts/ChartFrame';
import { LadderChart } from '../../ui/charts/LadderChart';
import { HourStepper, SegmentedControl } from '../../ui/Controls';
import type { SegmentOption } from '../../ui/Controls';
import { DataTable } from '../../ui/DataTable';
import { Icon } from '../../ui/Icon';
import { Text, keepTogether } from '../../ui/Text';
import { LAYOUT, SIZE, SPACE } from '../../ui/tokens';
import { SampleValue, Value } from '../../ui/Values';

const TITLE = CARD_TITLES['bid-ladder'];
/** 스켈레톤 = 계단 차트(200) + 판독·요약·주의 줄: 레이아웃이 뛰지 않게. */
const SKELETON_HEIGHT = 330;
const DASH = '—';
const TOTAL_LABEL = '합계';
type Mode = 'chart' | 'table';
const MODES: readonly SegmentOption<Mode>[] = [{ value: 'chart', label: '차트' }, { value: 'table', label: '표' }];
// 성향 글자는 산출 방식 시트("성향 균형 (API 값 moderate)")와 같은 표 하나에서 온다(D6).
const RISKS: readonly SegmentOption<Risk>[] = RISK_KEYS.map(value => ({ value, label: RISK_LABELS[value] }));
const probability = (v: number) => pct(v, 0);

function rowCells(r: BidRow): Record<string, React.ReactNode> {
  return {
    segment: r.segment,
    quantity: <SampleValue v={r.quantity} format={mw} size="label" showUnit={false} />,
    price: <Value v={r.price} format={price} size="label" showUnit={false} />,
    // 낙찰 가능성은 API의 0–1 값을 %로 바꾼 계산값이라 I3 비교 대상(data-pd-value)이 아니다.
    probability: <Value v={r.probability} format={probability} size="label" derived />,
    revenue: r.revenue ? <SampleValue v={r.revenue} format={krwFromThousand} size="label" showUnit={false} /> : DASH,
  };
}

export type SampleBidCardProps = {
  view: BidCardView;
  risk: Risk;
  onRisk: (risk: Risk) => void;
  onHour: (hour: number) => void;
  onRetry: () => void;
  onMethod: () => void;
  onSources: () => void;
};

export function SampleBidCard({ view, risk, onRisk, onHour, onRetry, onMethod, onSources }: SampleBidCardProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < LAYOUT.narrowBelow;
  const [mode, setMode] = useState<Mode>('chart');
  const [picked, setPicked] = useState<number | null>(null);
  const ready = view.state === 'ready' ? view : null;
  // 320에서는 스텝퍼에서 날짜를 빼고 카드 부제로 옮긴다(4.2절 폭별).
  const plant = (
    <Text variant="label" tone="text2" testID="bid-plant">
      {EXAMPLE_PLANT.name.value} · <SampleValue v={EXAMPLE_PLANT.capacityMw} format={mw} size="label" /> (가상)
      {narrow && ready && ready.tradingDate ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'day' }} nowrap>{` · ${bidDayText(ready.tradingDate)}`}</Text> : null}
    </Text>
  );
  const riskRow = (
    <View style={styles.control}>
      <Text variant="label" weight="600" tone="text2">성향</Text>
      <SegmentedControl label="성향" options={RISKS} value={risk} onChange={onRisk} />
    </View>
  );
  // 카드 골격(4.0절): 리본 → 머리 → 본문 → "KPX에 아무것도 보내지 않습니다"(늘) → 출처 줄.
  const shell = (body: React.ReactNode, footer: { source: string; time: string } | null, chips?: React.ReactNode) => (
    <Card title={TITLE} ribbon={<SampleRibbon chip={CHIPS.exampleVirtual} />} chips={chips} onMethod={onMethod} testID="bid-ladder">
      {plant}
      {body}
      <Text variant="meta" tone="text3">{NO_SEND_NOTE}</Text>
      {footer ? <SourceFooter source={footer.source} time={footer.time} onSources={onSources} context={TITLE} /> : null}
    </Card>
  );
  if (view.state === 'loading') return shell(<>{riskRow}<StateView kind="loading" minHeight={SKELETON_HEIGHT} /></>, null);
  if (view.state === 'error') {
    return shell(<>{riskRow}<StateView kind="error" text={view.text} onRetry={view.retry ? onRetry : undefined} retryLabel={RECALC_LABEL} /></>, null);
  }
  if (view.state === 'empty') {
    return shell(<>{riskRow}<StateView kind="empty" text={view.text} onRetry={onRetry} retryLabel={RECALC_LABEL} /></>, view.footer);
  }
  const seg = picked !== null ? view.hour.segments[picked] : undefined;
  const columns = [
    { key: 'segment', title: BID_COLUMNS.segment, meta: true },
    { key: 'quantity', title: BID_COLUMNS.quantity, align: 'right' as const },
    { key: 'price', title: BID_COLUMNS.price, align: 'right' as const },
    { key: 'probability', title: BID_COLUMNS.probability, align: 'right' as const },
    {
      key: 'revenue', title: BID_COLUMNS.revenue, align: 'right' as const,
      extra: <><ProvenanceChip chip={CHIPS.example} /><ProvenanceChip chip={CHIPS.estShort} source={view.chip.source} /></>,
    },
  ];
  const total = {
    segment: TOTAL_LABEL, quantity: <SampleValue v={view.total.quantity} format={mw} size="label" showUnit={false} />, price: DASH, probability: DASH,
    revenue: view.total.revenue ? <SampleValue v={view.total.revenue} format={krwFromThousand} size="label" showUnit={false} /> : DASH,
  };
  return shell(
    <>
      <View style={styles.control}>
        <Text variant="label" weight="600" tone="text2">거래시간</Text>
        <HourStepper hour={view.selected} min={view.first} max={view.last} onChange={onHour}
          label={hour => bidHourLabel(hour, narrow ? null : view.tradingDate)} />
      </View>
      {riskRow}
      <View style={styles.control}>
        <Text variant="label" weight="600" tone="text2">계단 가격</Text>
        <ProvenanceChip {...view.chip} />
      </View>
      <SegmentedControl label="보기 방식" options={MODES} value={mode} onChange={setMode} />
      {mode === 'chart' ? (
        <ChartBleed>
          <LadderChart hour={view.hour} label={view.chartLabel} selected={picked} onSelect={setPicked} />
        </ChartBleed>
      ) : (
        <DataTable caption={view.tableCaption} columns={columns} testID="bid-table"
          rows={[...view.rows.map(r => ({ key: String(r.id), current: r.id === (seg ? seg.id : null), cells: rowCells(r) })), { key: 'total', cells: total }]} />
      )}
      {seg ? (
        <Text variant="label" weight="600" dataSet={{ pdMeta: 'reading' }} testID="bid-reading">
          {ladderReadingText(picked ?? 0, seg.quantity.value, seg.price.value, seg.probability.value)}
        </Text>
      ) : null}
      <View style={styles.group}>
        {view.summary ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{keepTogether(view.summary, { variant: 'body' })}</Text> : null}
        {view.trend ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{view.trend}</Text> : null}
        <Text variant="meta" tone="text3">{view.notReal}</Text>
        {view.basis ? (
          <View style={styles.basis} testID="bid-basis">
            <Icon name="alert" size={SIZE.chipIcon} color={theme.color.text2} />
            <Text variant="meta" tone="text2" style={styles.grow} dataSet={{ pdMeta: 'basis' }}>{keepTogether(view.basis, { variant: 'meta', tone: 'text2' })}</Text>
          </View>
        ) : null}
      </View>
    </>,
    view.footer,
    view.headChips.length ? <ChipList chips={view.headChips} /> : undefined,
  );
}

const styles = StyleSheet.create({
  control: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: SPACE.sm, rowGap: SPACE.xs },
  group: { gap: SPACE.xs },
  basis: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.xs },
  grow: { flex: 1, minWidth: 0 },
});
