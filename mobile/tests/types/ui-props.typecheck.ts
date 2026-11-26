// 타입 수준 검사(명세 6.2·7.2절, F4 "모든 버튼이 이름 필수(타입)"). `tsc --noEmit -p .`(check.sh 3단계)가 본다.
// 아래 오류 기대 지시문 줄이 실제로 오류를 내지 않으면 tsc가 TS2578로 실패한다.
import React from 'react';
import { Button, IconButton } from '../../src/ui/Button';
import { HeroValue, KpiBlock, SampleValue, Value } from '../../src/ui/Values';
import type { Provenanced } from '../../src/data/adapters';
import { EXAMPLE_PLANT } from '../../src/data/examples';
import { mw } from '../../src/lib/format';

declare const real: Provenanced<number>;
const sample = EXAMPLE_PLANT.capacityMw;

export const allowed = [
  React.createElement(Button, { label: '다시 시도', accessibilityLabel: '다시 시도' }),
  React.createElement(IconButton, { icon: 'refresh', accessibilityLabel: '가격·전력 정보 새로고침' }),
  React.createElement(Value, { v: real, format: mw }),
  React.createElement(KpiBlock, { label: '수요', v: real, format: mw }),
  React.createElement(HeroValue, { label: '확정 가격', v: real, format: mw }),
  React.createElement(SampleValue, { v: sample, format: mw }),
];

// @ts-expect-error 버튼은 한국어 이름(accessibilityLabel)이 필수다
React.createElement(Button, { label: '다시 시도' });
// @ts-expect-error 아이콘만 있는 버튼도 이름이 필수다
React.createElement(IconButton, { icon: 'close' });
// @ts-expect-error 칩 없는 숫자는 Value에 넣을 수 없다
React.createElement(Value, { v: 664, format: mw });
// @ts-expect-error 예시 값은 실제 값 자리(Value)에 넣을 수 없다
React.createElement(Value, { v: sample, format: mw });
// @ts-expect-error 예시 값은 KPI 타일에 넣을 수 없다(7.9절)
React.createElement(KpiBlock, { label: '용량', v: sample, format: mw });
// @ts-expect-error 실제 값은 SampleValue 자리에 넣을 수 없다
React.createElement(SampleValue, { v: real, format: mw });
