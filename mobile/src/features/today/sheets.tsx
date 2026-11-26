// 오늘 화면이 시트와 패널에 넘기는 것(명세 4.1·4.4·4.6절). 모두 엿보기만 한다(새 요청 없음): 산출 방식 시트의 "이번 화면의 값",
// 출처와 기준 시각 표의 오늘 행(다른 화면이 넘기는 행을 뒤에 붙인다), ≥ 1280 "오늘 화면 읽는 법" 패널. App.tsx와 알아보기가 쓴다.
import React from 'react';
import { useTodayView } from './data';
import type { TodayCardId } from './model';
import { TodayPanelBody } from './TodayParts';
import { useNav } from '../../app/Shell';
import type { MethodCard, SourceRow } from '../../app/routes';
import { LiveFacts, SourcesTable } from '../guide/sheets';
import type { SourceRowItem } from '../guide/sheets';

const isTodayCard = (card: MethodCard): card is TodayCardId => card.startsWith('today-');

/** 산출 방식 시트의 "이번 화면의 값"(4.1·4.4절: KPX 원문 표기·25.9 % 비율·MW 계산 예비율 차이·모델 원문·API 표기 줄). */
export function TodayMethodFacts({ card }: { card: MethodCard }) {
  const { view } = useTodayView(false);
  if (!isTodayCard(card)) return null;
  const facts = view.facts[card];
  return <LiveFacts lines={facts.lines} api={facts.api} link={facts.link} />;
}

/** 출처와 기준 시각 시트(4.4절): 오늘 행 + 다른 화면이 넘기는 행(입찰 계산 F8, 응답 상태 F10). */
export function TodaySourcesSheet({ row, extra = [], page }: { row: SourceRow | null; extra?: readonly SourceRowItem[]; page?: boolean }) {
  const { view } = useTodayView(false);
  return <SourcesTable rows={[...view.sources, ...extra]} highlight={row} page={page} />;
}

/** ≥ 1280 "오늘 화면 읽는 법" 패널 본문(4.6절). */
export function TodayPanel() {
  const { view } = useTodayView(false);
  const nav = useNav();
  return <TodayPanelBody legend={view.legend} onTerm={id => nav.openSheet({ kind: 'term', id })} />;
}
