// 지금 제주 전력(명세 4.1·7.4절): 수요·예비율 KPI, 예비율 단계(앱 기준), 수급 식과 범위 막대(짝이 맞는 실시간 값일 때만), 날씨, 출처.
// 폭: 320은 KPI를 위아래로, 360 이상은 나란히. 날씨는 모든 폭에서 범위 막대 아래 자기 줄이다(round 1 D14: 768·1280에서 셋째 칸에
// 넣으면 "· 맑음"이 홀로 줄을 바꿔 세 줄이 됐다).
import React from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { KpiItem, PowerCardView } from './model';
import { ChipList } from './TodayParts';
import { mw, pct } from '../../lib/format';
import { CARD_TITLES, VALUE_EMPTY } from '../../content/strings';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ProvenanceChip, StatusPill } from '../../ui/Chips';
import { RangeBar } from '../../ui/charts/RangeBar';
import { RichText } from '../../ui/Rich';
import { Text } from '../../ui/Text';
import { LAYOUT, SPACE } from '../../ui/tokens';
import { KpiBlock } from '../../ui/Values';

const TITLE = CARD_TITLES['today-power'];
/** 스켈레톤 = 최종 본문 높이(KPI 48 + 단계 24 + 식 18 + 막대 83 + 날씨 24 + 간격 + 출처 줄): 레이아웃이 뛰지 않게. */
const SKELETON_HEIGHT = 300;
const DASH = '—';

function Kpi({ item, format }: { item: KpiItem; format: (n: number) => string }) {
  return (
    <View style={styles.kpi}>
      {item.value ? (
        <KpiBlock label={item.label} v={item.value} format={format} chip={item.chip ? <ProvenanceChip {...item.chip} /> : false} inline />
      ) : (
        <View style={styles.block}>
          <Text variant="label" weight="600" tone="text2">{item.label}</Text>
          <Text variant="title" weight="700" tone="text3">{DASH}</Text>
          <Text variant="meta" tone="text3">{VALUE_EMPTY}</Text>
        </View>
      )}
      {item.note ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{item.note}</Text> : null}
    </View>
  );
}

export type PowerCardProps = { view: PowerCardView; onMethod: () => void; onSources: () => void; onRetry: () => void; fill?: boolean };

export function PowerCard({ view, onMethod, onSources, onRetry, fill = false }: PowerCardProps) {
  const { width } = useWindowDimensions();
  if (view.state === 'loading') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-power" fill={fill}><StateView kind="loading" minHeight={SKELETON_HEIGHT} /></Card>;
  }
  if (view.state === 'error') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-power" fill={fill}><StateView kind="error" text={view.text} onRetry={onRetry} /></Card>;
  }
  const narrow = width < LAYOUT.narrowBelow;
  const weather = view.weather ? (
    <View style={styles.weather} testID="today-weather">
      <RichText line={view.weather.line} variant="label" tone="text2" />
      <ProvenanceChip {...view.weather.chip} />
    </View>
  ) : null;
  return (
    <Card title={TITLE} chips={<ChipList chips={view.chips} />} onMethod={onMethod} testID="today-power" fill={fill}>
      <View style={styles.body}>
        <View style={[styles.kpis, narrow ? styles.stack : null]} testID="today-kpis">
          <Kpi item={view.demand} format={mw} />
          <Kpi item={view.reserve} format={pct} />
        </View>
        {view.level ? (
          <View style={styles.levelRow}>
            <StatusPill tone={view.level.tone} text={view.level.text} />
            <Text variant="meta" tone="text3">{view.level.note}</Text>
          </View>
        ) : view.judgeNote ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'judge' }}>{view.judgeNote}</Text> : null}
        {view.supply.kind === 'paired' ? (
          <>
            <RichText line={view.supply.formula} variant="label" testID="today-formula" />
            <RangeBar supply={view.supply.supply} demand={view.supply.demand} reserve={view.supply.reserve} boundary={view.supply.boundary} />
          </>
        ) : <Text variant="meta" tone="text3" testID="today-supply-missing">{view.supply.text}</Text>}
        {weather}
      </View>
      <SourceFooter source={view.footer.source} time={view.footer.time} extra={view.footer.extra} onSources={onSources} context={TITLE} />
    </Card>
  );
}

const styles = StyleSheet.create({
  body: { gap: SPACE.sm },
  kpis: { flexDirection: 'row', gap: SPACE.sm },
  stack: { flexDirection: 'column' },
  kpi: { flex: 1, minWidth: 0, gap: SPACE.xs },
  block: { gap: SPACE.xs, alignItems: 'flex-start' },
  levelRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm },
  weather: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', columnGap: SPACE.sm, rowGap: SPACE.xs },
});
