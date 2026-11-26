// 지금 시간 전기값(명세 4.1절, 영웅 카드). today-hero의 data-state가 loading → ready·unavailable·error로 바뀐다(M9 측정 표지).
// 확정 가격(● 확정 가격) 32/700 + 거래시간 줄 + 같은 날짜·거래시간 기준선과의 비교 두 줄(7.8절 1 조건) + 출처 줄.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import type { PriceCardView } from './model';
import { ChipList } from './TodayParts';
import { price } from '../../lib/format';
import { chipA11yLabel } from '../../lib/provenance';
import { CARD_TITLES } from '../../content/strings';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { RichText } from '../../ui/Rich';
import type { Line } from '../../ui/Rich';
import { Text } from '../../ui/Text';
import { SPACE } from '../../ui/tokens';
import { Value, spokenValue } from '../../ui/Values';

const TITLE = CARD_TITLES['today-price'];
const joinLines = (lines: readonly Line[]): Line => lines.flatMap((line, i) => (i === 0 ? [...line] : [{ t: ' ' }, ...line]));
/** 스켈레톤 = 최종 본문 높이(값 37 + 거래시간 18 + 비교 세 줄 69 + 간격 + 출처 줄 44): 레이아웃이 뛰지 않게. */
const SKELETON_HEIGHT = 196;
const DASH = '—';

export type PriceCardProps = { view: PriceCardView; onMethod: () => void; onSources: () => void; onRetry: () => void };

export function PriceCard({ view, onMethod, onSources, onRetry }: PriceCardProps) {
  if (view.state === 'loading') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-hero" state="loading"><StateView kind="loading" minHeight={SKELETON_HEIGHT} /></Card>;
  }
  if (view.state === 'error') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-hero" state="error"><StateView kind="error" text={view.text} onRetry={onRetry} /></Card>;
  }
  if (view.state === 'unavailable') {
    return (
      <Card title={TITLE} onMethod={onMethod} testID="today-hero" state="unavailable">
        <View style={styles.hero}>
          <Text variant="display" weight="700" tone="text3">{DASH}</Text>
          <Text variant="body" tone="text2">{view.text}</Text>
        </View>
        <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
      </Card>
    );
  }
  const name = `${TITLE} ${spokenValue(price(view.price.value), view.price.unit)}, ${chipA11yLabel(view.price.chip, view.price.source)}`;
  return (
    <Card title={TITLE} chips={<ChipList chips={view.chips} />} onMethod={onMethod} testID="today-hero" state="ready">
      <View style={styles.hero}>
        <View role="group" aria-label={name}>
          <Value v={view.price} format={price} size="display" />
        </View>
        <Text variant="label" tone="text2" dataSet={{ pdMeta: 'hour' }}>{view.hourLine}</Text>
        {view.checking ? <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'checking' }}>{view.checking}</Text> : null}
      </View>
      {/* 비교 두 문장은 한 문단으로 흐르게 둔다(같은 글자·크기, 390에서 세 줄 → 두 줄). */}
      {view.compare.length ? <RichText line={joinLines(view.compare)} testID="today-compare" /> : null}
      <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
    </Card>
  );
}

const styles = StyleSheet.create({
  hero: { gap: SPACE.xs },
});
