/**
 * 정산(명세 4.3절, 경로 /settlement · 라우트 키 settlement). 발전소별 정산·발전소 자료는 공개되지 않아 이 화면은 요청을 하나도 보내지
 * 않는다(6.3절 "정산: 없음"): 자료 클라이언트·저장소를 가져오지 않고, 제한 카드는 정적이며 재시도가 없다. 숫자는 모두 data/examples.ts의
 * 예시 값이고 순수 함수 model.ts가 글자를 만든다. 헤더 새로고침은 셸이 비활성으로 둔다(ROUTES.settlement.refreshable = false, 3.4절).
 * 산출 방식 시트(settle-sample)의 이번 화면 값과 ≥ 1280 "정산 읽는 법" 패널도 여기서 내보낸다(App.tsx가 시트·패널로 등록한다).
 */
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { useClock, useNav } from '../../app/Shell';
import { ROUTES } from '../../app/routes';
import type { MethodCard } from '../../app/routes';
import { CHIPS } from '../../lib/provenance';
import { LEGEND_LOCKED, SETTLEMENT_RESTRICTED_BODY, SETTLEMENT_RESTRICTED_TITLE, SETTLEMENT_TERMS } from '../../content/strings';
import { LiveFacts } from '../guide/sheets';
import { ReadingPanelBody, TermRow } from '../today/TodayParts';
import { settlementView } from './model';
import { SampleStatementCard } from './SampleStatementCard';
import { RestrictedCard } from '../../ui/Card';
import { NowPill, ProvenanceChip, SampleRibbon } from '../../ui/Chips';
import { AnswerLine, Screen } from '../../ui/Screen';
import { SPACE } from '../../ui/tokens';

/** 패널의 용어 셋(4.6절): 정산 · 하루전시장 · 실시간시장. 단위 풀이는 패널의 읽는 법 셋째 줄이 같은 내용을 쓴다. */
const PANEL_TERMS = SETTLEMENT_TERMS.filter(t => t.id !== 'units');

export default function SettlementTab() {
  const nav = useNav();
  const clock = useClock();
  const view = settlementView();
  // 조회가 없는 화면이라 NowPill은 KST 시각만(3.4절). 서버 시계가 타당하지 않으면 입찰 화면처럼 ‹시각 확인 필요›를 붙인다(7.6절).
  const pill = (
    <View style={styles.pills}>
      <NowPill now={clock.now} fetchedAt={null} skewNotable={clock.skewNotable} />
      {clock.skew && !clock.skew.plausible ? <ProvenanceChip chip={CHIPS.clockImplausible} /> : null}
    </View>
  );
  return (
    <Screen title={ROUTES.settlement.heading} pill={pill} answer={<AnswerLine>{view.answer}</AnswerLine>} testID="settlement-screen">
      <RestrictedCard title={SETTLEMENT_RESTRICTED_TITLE} body={SETTLEMENT_RESTRICTED_BODY} testID="settle-restricted" />
      <SampleStatementCard onMethod={() => nav.openSheet({ kind: 'method', card: 'settle-sample' })} />
      <TermRow terms={SETTLEMENT_TERMS} onOpen={id => nav.openSheet({ kind: 'term', id })} testID="settle-terms" />
    </Screen>
  );
}

/** 산출 방식 시트(settle-sample)의 이번 화면 값: MWh를 kWh로 바꿔 원 단위까지 다시 계산한 예시(리본으로 예시임을 먼저 알린다). */
export function SettlementMethodFacts({ card }: { card: MethodCard }) {
  if (card !== 'settle-sample') return null;
  return (
    <View style={styles.facts} testID="settle-facts">
      <SampleRibbon />
      <LiveFacts lines={settlementView().facts} link={null} />
    </View>
  );
}

/** ≥ 1280 "정산 읽는 법" 패널(4.3·4.6절): 이 화면의 표시(예시 · 잠김) · 용어 셋 · 읽는 법 세 줄. ①+② 그림은 모든 폭에서 예시 정산서 카드 안에 있다(D14). */
export function SettlementPanel() {
  const nav = useNav();
  const view = settlementView();
  const legend = [{ chip: CHIPS.example, text: view.legend }, { chip: CHIPS.locked, text: LEGEND_LOCKED }];
  return (
    <ReadingPanelBody legend={legend} terms={PANEL_TERMS} method={view.panelMethod} onTerm={id => nav.openSheet({ kind: 'term', id })} testID="settlement-panel" />
  );
}

const styles = StyleSheet.create({
  pills: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: SPACE.xs },
  facts: { gap: SPACE.sm },
});
