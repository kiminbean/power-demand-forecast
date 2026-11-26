// 타입 수준 검사(명세 7.2절 "차트는 Provenanced만 받는다", F5). `tsc --noEmit -p .`(check.sh 3단계)가 본다.
// 아래 오류 기대 지시문 줄이 실제로 오류를 내지 않으면 tsc가 TS2578로 실패한다.
import React from 'react';
import { LadderChart } from '../../src/ui/charts/LadderChart';
import { RangeBar } from '../../src/ui/charts/RangeBar';
import { StepChart } from '../../src/ui/charts/StepChart';
import type { ForecastView, HourBidView, Provenanced } from '../../src/data/adapters';
import { EXAMPLE_PLANT } from '../../src/data/examples';

declare const real: Provenanced<number>;
declare const forecast: ForecastView;
declare const hour: HourBidView;
const sample = EXAMPLE_PLANT.capacityMw;

export const allowed = [
  React.createElement(StepChart, { forecast, label: '기준선 예측 SMP 24시간', now: null }),
  React.createElement(StepChart, { forecast, label: '기준선 예측 SMP 24시간', now: 0, confirmed: { tradeHour: 7, price: real }, selectedHour: 7, onSelectHour: () => undefined }),
  React.createElement(LadderChart, { hour, label: '19h · 18–19시 입찰 계단', selected: 3, onSelect: () => undefined }),
  React.createElement(RangeBar, { supply: real, demand: real, reserve: real, boundary: real }),
];

// @ts-expect-error 칩 없는 숫자 배열은 기준선 차트에 넣을 수 없다(ForecastView의 q50은 Provenanced)
React.createElement(StepChart, { forecast: { q50: [101.2, 99.8], band: null }, label: 'x', now: null });
// @ts-expect-error 확정 가격 점도 Provenanced여야 한다
React.createElement(StepChart, { forecast, label: 'x', now: null, confirmed: { tradeHour: 7, price: 115.3 } });
// @ts-expect-error 범위 막대에 칩 없는 숫자를 넣을 수 없다
React.createElement(RangeBar, { supply: 1191, demand: real, reserve: real, boundary: real });
// @ts-expect-error 예시 값은 실제 계열 차트에 넣을 수 없다(7.9절)
React.createElement(RangeBar, { supply: sample, demand: real, reserve: real, boundary: real });
// @ts-expect-error 계단 차트는 계산기 결과(HourBidView)만 받는다
React.createElement(LadderChart, { hour: { segments: [{ price: 171.7, probability: 95, quantity: 5 }] }, label: 'x' });
