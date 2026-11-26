// 예측은 얼마나 맞나요?(명세 4.1·7.7절, D8): 쉬운 말이 먼저다. 문서화된 기준선 정확도(‹검증값›, MAPE 하나)를 굵게, 그 뜻("100원이면
// 12원쯤"), 평가 방식 풀이 순서. API가 밝힌 수치(12.68 % 등)는 카드에 두지 않고 산출 방식 시트의 회색 "API 표기" 줄로만 간다.
// 기준선이 아닌 예측이면 문서화 수치를 쓰지 않고 "이 모델의 검증 수치는 앱에 등록되어 있지 않습니다". R²는 예측 방법 화면(알아보기)에서만 더한다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import type { AccuracyCardView } from './model';
import { CHIPS } from '../../lib/provenance';
import { CARD_TITLES } from '../../content/strings';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { ProvenanceChip } from '../../ui/Chips';
import { Text, keepTogether } from '../../ui/Text';
import { SPACE } from '../../ui/tokens';

const TITLE = CARD_TITLES['today-accuracy'];

export type AccuracyCardProps = { view: AccuracyCardView; onMethod: () => void; onSources: () => void; onRetry: () => void };

export function AccuracyCard({ view, onMethod, onSources, onRetry }: AccuracyCardProps) {
  if (view.state === 'loading') return <Card title={TITLE} onMethod={onMethod} testID="today-accuracy"><StateView kind="loading" /></Card>;
  if (view.state === 'error') {
    return <Card title={TITLE} onMethod={onMethod} testID="today-accuracy"><StateView kind="error" text={view.text} onRetry={onRetry} /></Card>;
  }
  const d = view.documented;
  return (
    <Card title={TITLE} chips={d ? <ProvenanceChip chip={CHIPS.verified} /> : undefined} onMethod={onMethod} testID="today-accuracy">
      {d ? (
        <View style={styles.group}>
          <Text variant="body" weight="700" dataSet={{ pdMeta: 'documented' }}>{keepTogether(d.main, { variant: 'body', weight: '700' })}</Text>
          <Text variant="body" tone="text2" dataSet={{ pdMeta: 'documented' }}>{d.example}</Text>
          <Text variant="meta" tone="text3" dataSet={{ pdMeta: 'period' }}>{d.basis}</Text>
          {d.r2 ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'documented' }}>{keepTogether(d.r2, { variant: 'label', tone: 'text2' })}</Text> : null}
        </View>
      ) : <Text variant="body" tone="text2">{view.unregistered}</Text>}
      <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} />
    </Card>
  );
}

const styles = StyleSheet.create({
  group: { gap: SPACE.xs },
});
