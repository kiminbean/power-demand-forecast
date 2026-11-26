// 예시 발전소로 입찰 계산해 보기(명세 4.2·7.2·7.8절): ┆예시 · 가상 발전소┆ 리본, 거래시간 스텝퍼(캐시된 24시간을 읽고 새로 요청하지
// 않는다, 양 끝에서 멈춤), 성향(바꿀 때만 요청), 계단 차트 또는 표, 요약·낙찰 가능성 주의·가격 근거 공개 줄, "KPX에 아무것도 보내지
// 않습니다". 물량·기대 수입은 예시(SampleValue), 가격·낙찰 가능성은 추정이다. 기대 수입은 표에만 둔다(4.2절).
// round 1 수정: 성향 줄은 모든 상태에서 같은 자리(controls 칸, 열쇠 'risk')에 있다(D4: 상태마다 자리가 달라 성향을 바꾸면 세그먼트가
// 다시 만들어져 초점이 body로 떨어졌다). 600 px 아래 표는 행 카드(D2). 차트 위 범례에 기준선·80% 구간 값, 고른 구간은 판독 칩(D12).
import { useState } from 'react';
import type React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { BidCardView, BidRow } from './model';
import { RISKS as RISK_KEYS } from '../../app/routes';
import type { Risk } from '../../app/routes';
import { useTheme } from '../../app/theme';
import { EXAMPLE_PLANT } from '../../data/examples';
import { krwFromThousand, mw, pct, price } from '../../lib/format';
import { CHIPS } from '../../lib/provenance';
import {
  BID_COLUMNS, CARD_TITLES, NO_SEND_NOTE, RECALC_LABEL, RISK_LABELS, bidDayText, bidHourLabel, closedBidDayText,
  ladderKeyItems, ladderReadingText, segmentIdText,
} from '../../content/strings';
import { ChipList } from '../today/TodayParts';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ProvenanceChip, SampleRibbon } from '../../ui/Chips';
import { ChartBleed, ChartKey } from '../../ui/charts/ChartFrame';
import { LadderChart } from '../../ui/charts/LadderChart';
import { HourStepper, SegmentedControl } from '../../ui/Controls';
import type { SegmentOption } from '../../ui/Controls';
import { DataTable } from '../../ui/DataTable';
import { Icon } from '../../ui/Icon';
import { Text, keepTogether } from '../../ui/Text';
import { LAYOUT, RADIUS, SIZE, SPACE } from '../../ui/tokens';
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
  /** 하루전시장이 오늘 마감됐다(마감 카드의 단계): 계산의 거래일이 마감된 거래일임을 밝힌다(D13). */
  closed?: boolean;
};

export function SampleBidCard({ view, risk, onRisk, onHour, onRetry, onMethod, onSources, closed = false }: SampleBidCardProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const narrow = width < LAYOUT.narrowBelow;
  const phone = width < LAYOUT.rowCardsBelow;
  const [mode, setMode] = useState<Mode>('chart');
  const [picked, setPicked] = useState<number | null>(null);
  const ready = view.state === 'ready' ? view : null;
  // 320에서는 스텝퍼에서 날짜를 빼고 카드 부제로 옮긴다(4.2절 폭별). 마감 뒤에는 마감된 거래일 기준이라고 붙인다(D13).
  const day = ready && ready.tradingDate ? (closed ? closedBidDayText(ready.tradingDate) : narrow ? bidDayText(ready.tradingDate) : null) : null;
  const plant = (
    <Text variant="label" tone="text2" testID="bid-plant">
      {EXAMPLE_PLANT.name.value} · <SampleValue v={EXAMPLE_PLANT.capacityMw} format={mw} size="label" /> (가상)
      {day ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'day' }} nowrap>{` · ${day}`}</Text> : null}
    </Text>
  );
  // 거래시간·성향 칸은 모든 상태에서 카드의 같은 자리에 있다: 성향 세그먼트(열쇠 'risk')는 계산 중에도 다시 만들어지지 않아 초점이 남는다.
  const controls = (
    <View style={styles.controls}>
      {ready ? (
        <View key="hour" style={styles.control}>
          <Text variant="label" weight="600" tone="text2">거래시간</Text>
          <HourStepper hour={ready.selected} min={ready.first} max={ready.last} onChange={onHour}
            label={hour => bidHourLabel(hour, narrow || closed ? null : ready.tradingDate)} />
        </View>
      ) : null}
      <View key="risk" style={styles.control}>
        <Text variant="label" weight="600" tone="text2">성향</Text>
        <SegmentedControl label="성향" options={RISKS} value={risk} onChange={onRisk} />
      </View>
    </View>
  );
  // 카드 골격(4.0절): 리본 → 머리 → 본문 → "KPX에 아무것도 보내지 않습니다"(늘) → 출처 줄.
  const shell = (body: React.ReactNode, footer: { source: string; time: string } | null, chips?: React.ReactNode) => (
    <Card title={TITLE} ribbon={<SampleRibbon chip={CHIPS.exampleVirtual} />} chips={chips} onMethod={onMethod} testID="bid-ladder">
      {plant}
      {controls}
      {body}
      <Text variant="meta" tone="text3">{NO_SEND_NOTE}</Text>
      {footer ? <SourceFooter source={footer.source} time={footer.time} onSources={onSources} context={TITLE} /> : null}
    </Card>
  );
  if (view.state === 'loading') return shell(<StateView kind="loading" minHeight={SKELETON_HEIGHT} />, null);
  if (view.state === 'error') {
    return shell(<StateView kind="error" text={view.text} onRetry={view.retry ? onRetry : undefined} retryLabel={RECALC_LABEL} />, null);
  }
  if (view.state === 'empty') {
    return shell(<StateView kind="empty" text={view.text} onRetry={onRetry} retryLabel={RECALC_LABEL} />, view.footer);
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
  // 합계 행의 가격·낙찰 가능성은 없다(표 "—", 행 카드에서는 빠진다).
  const total = {
    segment: TOTAL_LABEL, quantity: <SampleValue v={view.total.quantity} format={mw} size="label" showUnit={false} />,
    revenue: view.total.revenue ? <SampleValue v={view.total.revenue} format={krwFromThousand} size="label" showUnit={false} /> : DASH,
  };
  const band = view.hour.baselineBand;
  const key = ladderKeyItems(view.kind, view.hour.baselineQ50, band ? band.q10 : null, band ? band.q90 : null);
  // 600 px 아래는 같은 열의 행 카드(D2): "1구간 · 물량(MW, 예시) 5 · 가격(원/kWh) 171.7 · 낙찰 가능성(추정) 95% · 기대 수입 56.1만원".
  const table = (
    <DataTable caption={view.tableCaption} columns={columns} cards={phone} testID={phone ? 'bid-cards' : 'bid-table'}
      rows={[...view.rows.map(r => ({ key: String(r.id), current: r.id === (seg ? seg.id : null), cells: rowCells(r), head: segmentIdText(r.id) })),
        { key: 'total', cells: total, head: TOTAL_LABEL }]} />
  );
  return shell(
    <>
      <View style={styles.control}>
        <Text variant="label" weight="600" tone="text2">계단 가격</Text>
        <ProvenanceChip {...view.chip} />
      </View>
      <SegmentedControl label="보기 방식" options={MODES} value={mode} onChange={setMode} />
      {mode === 'chart' ? (
        <View style={styles.group}>
          <ChartKey items={key} testID="bid-legend" />
          <ChartBleed>
            <LadderChart hour={view.hour} label={view.chartLabel} selected={picked} onSelect={setPicked} />
          </ChartBleed>
        </View>
      ) : table}
      {seg ? (
        // 고른 구간의 판독 칩(D12): 차트 아래에 둔다(위에 두면 처음 고를 때 누른 차트가 밀린다).
        <View style={[styles.readout, { backgroundColor: theme.color.well }]} testID="bid-reading">
          <Text variant="label" weight="600" dataSet={{ pdMeta: 'reading' }}>
            {ladderReadingText(picked ?? 0, seg.quantity.value, seg.price.value, seg.probability.value)}
          </Text>
        </View>
      ) : null}
      <View style={styles.group}>
        {view.summary ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{keepTogether(view.summary, { variant: 'body' })}</Text> : null}
        {view.trend ? <Text variant="body" dataSet={{ pdMeta: 'summary' }}>{keepTogether(view.trend, { variant: 'body' })}</Text> : null}
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
  controls: { gap: SPACE.md },
  control: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: SPACE.sm, rowGap: SPACE.xs },
  group: { gap: SPACE.xs },
  readout: { alignSelf: 'flex-start', maxWidth: '100%', borderRadius: RADIUS.control, paddingHorizontal: SPACE.md, paddingVertical: SPACE.xs },
  basis: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.xs },
  grow: { flex: 1, minWidth: 0 },
});
