// 하루전시장 입찰 마감(명세 4.2절): 서버 시장 상태로 정한 단계(확인 중 · 접수 중 · 마감 임박 · 마감 · 불일치 · 오류), 카운트다운
// (마감 뒤에는 다음 마감까지, round 1 D13: 하루 14시간 동안 이 카드에 큰 숫자가 없었다. «오늘 마감됨»은 제목 옆)
// (분 내림, 셸 시계가 분이 바뀔 때마다 다시 그린다 · 알리지 않는다), 마감·입찰일, 실시간시장 구간, 출처. 글자와 단계는 model.ts가
// 정하고 여기서는 배치만 한다. 둘러보기 3단계(F10)도 이 카드를 그대로 쓴다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import type { DeadlineCardView } from './model';
import { ChipList } from '../today/TodayParts';
import { CARD_TITLES, DEADLINE_CHECKING_TEXT, RTM_PRICE_NOTE } from '../../content/strings';
import { Card, SourceFooter, StateView } from '../../ui/Card';
import { StatusPill } from '../../ui/Chips';
import { Text, keepTogether } from '../../ui/Text';
import { SPACE } from '../../ui/tokens';

const TITLE = CARD_TITLES['bid-deadline'];
/** 스켈레톤 = 접수 중 본문(카운트다운 37 + 마감 줄 18 + 실시간시장 두 줄 35 + 간격)에서 pill 줄을 뺀 높이: 레이아웃이 뛰지 않게. */
const SKELETON_HEIGHT = 112;

export type DeadlineCardProps = { view: DeadlineCardView; onMethod: () => void; onSources: () => void; onRetry: () => void };

export function DeadlineCard({ view, onMethod, onSources, onRetry }: DeadlineCardProps) {
  if (view.phase === 'error') {
    return (
      <Card title={TITLE} onMethod={onMethod} testID="bid-deadline" state="error">
        <StateView kind="error" text={view.note ?? undefined} onRetry={onRetry} />
      </Card>
    );
  }
  if (view.phase === 'checking') {
    return (
      <Card title={TITLE} onMethod={onMethod} testID="bid-deadline" state="checking">
        {view.pill ? <StatusPill tone={view.pill.tone} text={view.pill.text} /> : null}
        <StateView kind="loading" text={DEADLINE_CHECKING_TEXT} minHeight={SKELETON_HEIGHT} />
      </Card>
    );
  }
  const chips = <>{view.status ? <StatusPill tone={view.status.tone} text={view.status.text} /> : null}<ChipList chips={view.chips} /></>;
  return (
    <Card title={TITLE} chips={chips} onMethod={onMethod} testID="bid-deadline" state={view.phase}>
      <View style={styles.group}>
        {view.countdownLead ? <Text variant="body" weight="600" tone="text2">{view.countdownLead}</Text> : null}
        {view.countdown
          ? <Text variant="display" weight="700" nowrap dataSet={{ pdMeta: 'countdown' }} testID="bid-countdown">{view.countdown}</Text>
          : null}
        {view.note ? <Text variant="body" tone="text2">{view.note}</Text> : null}
        {view.when ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'deadline' }}>{keepTogether(view.when, { variant: 'label', tone: 'text2' })}</Text> : null}
        {view.pill ? <StatusPill tone={view.pill.tone} text={view.pill.text} /> : null}
      </View>
      <View style={styles.group}>
        {view.rtm ? <Text variant="label" tone="text2" dataSet={{ pdMeta: 'rtm' }}>{keepTogether(view.rtm, { variant: 'label', tone: 'text2' })}</Text> : null}
        <Text variant="meta" tone="text3">{RTM_PRICE_NOTE}</Text>
      </View>
      {view.footer ? <SourceFooter source={view.footer.source} time={view.footer.time} onSources={onSources} context={TITLE} /> : null}
    </Card>
  );
}

const styles = StyleSheet.create({
  group: { gap: SPACE.xs },
});
