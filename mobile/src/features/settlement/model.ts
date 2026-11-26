// 정산 화면 view model(명세 4.3·7.3·7.9절): 예시 정산서의 행·합계, 산출 방식 시트의 이번 화면 값, ≥ 1280 패널의 범례와 ①+② 식 그림
// 재료를 만드는 순수 함수다. 정산·발전소 자료는 공개되지 않아 API를 부르지 않는다(6.3절 "정산: 없음"). 숫자는 모두 data/examples.ts의
// 예시 값(Example<T>)이고 이 파일에는 숫자를 두지 않는다(tests/settlement-tab.test.cjs가 확인한다). 컴포넌트는 배치만 한다.
import { EXAMPLE_STATEMENT, EXAMPLE_STATEMENT_AMOUNTS, KWH_PER_MWH } from '../../data/examples';
import type { Example } from '../../data/examples';
import {
  SETTLEMENT_ANSWER, STATEMENT_FIGURE_LABELS, STATEMENT_ROWS, legendStatementText, settlementUnitLine, statementFactText,
  statementFigureText, statementTotalFactText, unitFactText,
} from '../../content/strings';

export type StatementRowId = 'day-ahead' | 'deviation';
/** 정산서 한 행: 양(MWh) × 가격(원/kWh) = 금액(원). 셋 다 예시다. */
export type StatementRow = { id: StatementRowId; label: string; quantity: Example<number>; unitPrice: Example<number>; amount: Example<number> };
/** ①+② 그림의 막대 하나: 금액 구간 [from, to](원). ②는 ①의 끝에서 그만큼 내려가거나 올라가는 칸이다(폭포 그림). */
export type FigureBar = { id: StatementRowId | 'total'; label: string; amount: Example<number>; from: number; to: number };
export type SettlementView = {
  answer: string;
  rows: readonly StatementRow[];
  total: Example<number>;
  /** 산출 방식 시트의 이번 화면 값(예시): 단위 환산과 원 단위 계산. */
  facts: readonly string[];
  /** ≥ 1280 패널 범례의 예시 행과 읽는 법 세 줄. */
  legend: string;
  /** ≥ 1280 패널 읽는 법의 단위 줄(앞 두 줄은 guide.ko.json panels.settlement, round 2). */
  panelUnit: string;
  /** y는 0부터, top은 가장 높은 막대 끝(그림 높이의 기준). */
  figure: { summary: string; bars: readonly FigureBar[]; top: number };
};

const st = EXAMPLE_STATEMENT;
const won = EXAMPLE_STATEMENT_AMOUNTS;
/** 그림의 바닥. 지어낸 값이 아니라 축의 원점이다(차트 y는 0부터, 4.1·4.2절과 같은 규칙). */
const AXIS_ZERO = 0;
const span = (a: number, b: number) => ({ from: Math.min(a, b), to: Math.max(a, b) });

export function settlementView(): SettlementView {
  const rows: StatementRow[] = [
    { id: 'day-ahead', label: STATEMENT_ROWS.dayAhead, quantity: st.dayAheadMwh, unitPrice: st.dayAheadPrice, amount: won.dayAhead },
    { id: 'deviation', label: STATEMENT_ROWS.deviation, quantity: st.realtimeDeviationMwh, unitPrice: st.realtimePrice, amount: won.deviation },
  ];
  const first = won.dayAhead.value;
  const bars: FigureBar[] = [
    { id: 'day-ahead', label: STATEMENT_FIGURE_LABELS.dayAhead, amount: won.dayAhead, ...span(AXIS_ZERO, first) },
    { id: 'deviation', label: STATEMENT_FIGURE_LABELS.deviation, amount: won.deviation, ...span(first, first + won.deviation.value) },
    { id: 'total', label: STATEMENT_FIGURE_LABELS.total, amount: won.total, ...span(AXIS_ZERO, won.total.value) },
  ];
  return {
    answer: SETTLEMENT_ANSWER,
    rows,
    total: won.total,
    facts: [
      unitFactText(KWH_PER_MWH),
      ...rows.map(r => statementFactText(r.label, r.quantity.value, r.quantity.value * KWH_PER_MWH, r.unitPrice.value, r.amount.value)),
      statementTotalFactText(won.dayAhead.value, won.deviation.value, won.total.value),
    ],
    legend: legendStatementText(won.total.value),
    panelUnit: settlementUnitLine(KWH_PER_MWH),
    figure: {
      summary: statementFigureText(won.dayAhead.value, won.deviation.value, won.total.value),
      bars,
      top: Math.max(...bars.map(b => b.to)),
    },
  };
}
