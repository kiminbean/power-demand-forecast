// 지금 제주 전력(명세 4.1·7.4절): 수요·예비율 KPI, 예비율 단계(앱 기준), 수급 식과 범위 막대(짝이 맞는 실시간 값일 때만), 날씨, 출처.
// KPI 둘은 글자 폭만큼 나란히 두고, 둘째가 들어가지 않으면 통째로 다음 줄로 간다(round 2 parity: 반 칸씩 나누면 360·768에서 한쪽만 값이
// 이름 아래로 내려가 둘이 어긋났다). 날씨는 모든 폭에서 범위 막대 아래 자기 줄이고, 칩은 글자 옆 첫 줄에 붙는다(칩만 홀로 줄을 바꾸지 않게).
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
import { LAYOUT, SKELETON, SPACE } from '../../ui/tokens';
import { KpiBlock } from '../../ui/Values';

const TITLE = CARD_TITLES['today-power'];
const DASH = '—';

/**
 * 실측이 아닌 KPI(item.chip이 있다: API 기본값·전국 값·시뮬레이션)는 값을 본문 크기 흐린 글자로 두고, 같은 사실을 되풀이하는 덧말은 빼고
 * 칩 하나와 판정하지 않는 까닭 한 문장만 남긴다(round 2 멘토: 칩·덧말·문장이 같은 말을 세 번 했고, 기본값이 실측처럼 굵게 보였다).
 */
function Kpi({ item, format }: { item: KpiItem; format: (n: number) => string }) {
  return (
    <View style={styles.kpi}>
      {item.value ? (
        <KpiBlock label={item.label} v={item.value} format={format} chip={item.chip ? <ProvenanceChip {...item.chip} /> : false} inline muted={Boolean(item.chip)} />
      ) : (
        <View style={styles.block}>
          <Text variant="label" weight="600" tone="text2">{item.label}</Text>
          <Text variant="title" weight="700" tone="text3">{DASH}</Text>
          <Text variant="meta" tone="text3">{VALUE_EMPTY}</Text>
        </View>
      )}
      {item.note && !item.chip ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'note' }}>{item.note}</Text> : null}
    </View>
  );
}

export type PowerCardProps = { view: PowerCardView; onMethod: () => void; onSources: () => void; onRetry: () => void; fill?: boolean };

export function PowerCard({ view, onMethod, onSources, onRetry, fill = false }: PowerCardProps) {
  const { width } = useWindowDimensions();
  if (view.state === 'loading') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-power" fill={fill}><StateView kind="loading" minHeight={SKELETON.power} /></Card>;
  }
  if (view.state === 'error') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-power" fill={fill}><StateView kind="error" text={view.text} onRetry={onRetry} /></Card>;
  }
  const narrow = width < LAYOUT.narrowBelow;
  const weather = view.weather ? (
    <View style={styles.weather} testID="today-weather">
      <View style={styles.grow}><RichText line={view.weather.line} variant="label" tone="text2" /></View>
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
  kpis: { flexDirection: 'row', flexWrap: 'wrap', columnGap: SPACE.xl, rowGap: SPACE.xs },
  stack: { flexDirection: 'column' },
  kpi: { maxWidth: '100%', gap: SPACE.xs },
  block: { gap: SPACE.xs, alignItems: 'flex-start' },
  levelRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.sm },
  weather: { flexDirection: 'row', alignItems: 'flex-start', gap: SPACE.sm },
  grow: { flex: 1, minWidth: 0 },
});
