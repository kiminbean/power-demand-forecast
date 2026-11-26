// 글자 조각 줄(명세 1.3 I2·I3): 화면 view model이 문장을 조각으로 넘기면 여기서 한 줄로 그린다.
// t = 평문(숫자 없음, view-models 테스트가 확인), m = 메타(시각·라벨·계산값·요약, data-pd-meta), v = API 값(Value: data-pd-value +
// data-pd-raw, derived면 data-pd-meta="derived"). 그래서 화면의 모든 숫자가 값·메타 표지 안에 놓인다.
import React from 'react';
import { Text } from './Text';
import type { TextTone } from './Text';
import { Value, unitText } from './Values';
import type { Formatter } from './Values';
import type { FontWeight, TextVariant } from './tokens';
import type { Provenanced } from '../data/adapters';

/** m의 strong: 계산값 숫자를 같은 줄의 API 값(Value, 600)과 같은 굵기로 보인다. */
export type Seg = { t: string } | { m: string; tag: string; strong?: boolean } | { v: Provenanced<number>; f: Formatter; unit?: boolean; derived?: boolean };
export type Line = readonly Seg[];

type ValueSize = 'display' | 'title' | 'body' | 'label';
const VALUE_SIZE: Readonly<Record<TextVariant, ValueSize>> = { display: 'display', title: 'title', body: 'body', label: 'label', meta: 'label' };

export type RichTextProps = { line: Line; variant?: TextVariant; weight?: FontWeight; tone?: TextTone; testID?: string };

export function RichText({ line, variant = 'body', weight, tone, testID }: RichTextProps) {
  return (
    <Text variant={variant} weight={weight} tone={tone} testID={testID}>
      {line.map((s, i) => {
        if ('t' in s) return s.t;
        if ('m' in s) return <Text key={i} variant={variant} weight={s.strong ? '600' : weight} tone={tone} dataSet={{ pdMeta: s.tag }}>{s.m}</Text>;
        return <Value key={i} v={s.v} format={s.f} size={VALUE_SIZE[variant]} showUnit={Boolean(s.unit)} derived={s.derived} />;
      })}
    </Text>
  );
}

/** 보이는 글자(테스트·접근 이름). 단위 간격은 Value와 같다. */
export const lineText = (line: Line): string =>
  line.map(s => ('t' in s ? s.t : 'm' in s ? s.m : s.f(s.v.value) + (s.unit ? unitText(s.v.unit) : ''))).join('');
