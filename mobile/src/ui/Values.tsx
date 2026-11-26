// 값 표시(6.2·7.2절). Value·KpiBlock·HeroValue는 Provenanced<number>만, SampleValue는 Example<number>만 받는다:
// 칩 없는 숫자나 예시 값을 실제 값 자리에 넣으면 컴파일 오류다. 글자는 넘겨받은 format.ts 함수 하나로 한 번만 만든다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { SPACE } from './tokens';
import type { TextVariant } from './tokens';
import { ProvenanceChip } from './Chips';
import { SrOnly } from './Live';
import { Text } from './Text';
import type { Provenanced, Unit } from '../data/adapters';
import type { Example, ExampleUnit } from '../data/examples';
import { chipA11yLabel } from '../lib/provenance';

export type Formatter = (value: number) => string;
type AnyUnit = Unit | ExampleUnit;

// 붙여 쓰는 단위("115.3원/kWh", "79.5%")와 띄어 쓰는 단위("664 MW").
const UNIT_GAP: Readonly<Record<AnyUnit, string>> = { '원/kWh': '', '%': '', '°C': '', MW: ' ', MWh: ' ', 'm/s': ' ', 천원: ' ', '': '' };
const UNIT_SPOKEN: Readonly<Record<AnyUnit, string>> = {
  '원/kWh': '원/킬로와트시', '%': '퍼센트', '°C': '도', MW: '메가와트', MWh: '메가와트시', 'm/s': '미터/초', 천원: '천 원', '': '',
};
/** 접근 이름용 읽기 형태: "664 메가와트". */
export const spokenValue = (text: string, unit: AnyUnit) => (UNIT_SPOKEN[unit] ? `${text} ${UNIT_SPOKEN[unit]}` : text);
/** 값 뒤에 붙는 단위 글자(간격 포함): "원/kWh"·"%"는 붙여, "MW"·"m/s"는 띄어 쓴다. */
export const unitText = (unit: AnyUnit) => (unit ? UNIT_GAP[unit] + unit : '');

type Size = 'display' | 'title' | 'body' | 'label';
// 값 옆 단위는 한 단계 작게(5.4절): 32 옆 15, 20 옆 13, 15·13 옆 13.
const UNIT_VARIANT: Readonly<Record<Size, TextVariant>> = { display: 'body', title: 'label', body: 'label', label: 'label' };
const SMALLER: Readonly<Record<Size, Size>> = { display: 'title', title: 'body', body: 'label', label: 'label' };

function UnitText({ unit, size }: { unit: AnyUnit; size: Size }) {
  if (!unit) return null;
  return <Text variant={UNIT_VARIANT[size]} weight={size === 'display' ? '500' : '400'} tone="text2" dataSet={{ pdMeta: 'unit' }}>{UNIT_GAP[unit] + unit}</Text>;
}

export type ValueProps = {
  v: Provenanced<number>; format: Formatter; size?: Size; showUnit?: boolean;
  /** API 값으로 계산한 값(예비력 = 공급능력 − 수요, 관심 경계 = 공급능력 ÷ 1.15). API 원값이 아니라서 data-pd-meta="derived". */
  derived?: boolean;
};
/** 실제 값. data-pd-value="{상태}", data-pd-raw=API 원값(I2·I3 검사가 보이는 글자와 비교한다). 계산값은 derived로 표시한다. */
export function Value({ v, format, size = 'body', showUnit = true, derived = false }: ValueProps) {
  return (
    <Text variant={size} weight={size === 'display' || size === 'title' ? '700' : '600'} nowrap
      dataSet={derived ? { pdMeta: 'derived' } : { pdValue: v.state, pdRaw: String(v.value) }}>
      {format(v.value)}{showUnit ? <UnitText unit={v.unit} size={size} /> : null}
    </Text>
  );
}

/** 예시 값: text2, 한 단계 작게, 화면 읽기에는 끝에 "(예시)". 단위가 열 머리에 있으면(표) showUnit을 끈다. */
export function SampleValue({ v, format, size = 'body', showUnit = true }: { v: Example<number>; format: Formatter; size?: Size; showUnit?: boolean }) {
  const shown = SMALLER[size];
  return (
    <Text variant={shown} weight={shown === 'title' ? '700' : '600'} tone="text2" nowrap dataSet={{ pdExample: 'true' }}>
      {format(v.value)}{showUnit ? <UnitText unit={v.unit} size={shown} /> : null}<SrOnly> (예시)</SrOnly>
    </Text>
  );
}

type BlockProps = {
  label: string; v: Provenanced<number>; format: Formatter; chip?: React.ReactNode; testID?: string;
  /** 라벨과 값을 한 줄에(들어가지 않으면 줄바꿈). 4.1절 전력 카드 "쓰는 전력(수요)  664 MW". */
  inline?: boolean;
};

function Block({ label, v, format, chip, testID, size, inline = false }: BlockProps & { size: 'display' | 'title' }) {
  // 라벨 → 값+단위 → 칩을 한 문자열로 읽히게 묶는다(role=group은 이름을 가질 수 있는 역할).
  const name = `${label} ${spokenValue(format(v.value), v.unit)}, ${chipA11yLabel(v.chip, v.source)}`;
  const labelText = <Text variant="label" weight="600" tone="text2">{label}</Text>;
  return (
    <View role="group" aria-label={name} testID={testID} style={styles.block}>
      {inline
        ? <View style={styles.inline}>{labelText}<Value v={v} format={format} size={size} /></View>
        : <>{labelText}<Value v={v} format={format} size={size} /></>}
      {chip ?? <ProvenanceChip chip={v.chip} source={v.source} />}
    </View>
  );
}

/** KPI 타일: 20/700 값. */
export const KpiBlock = (props: BlockProps) => <Block {...props} size="title" />;
/** 영웅 값: 32/700(폭 ≤ 360은 28). */
export const HeroValue = (props: BlockProps) => <Block {...props} size="display" />;

const styles = StyleSheet.create({
  block: { gap: SPACE.xs, alignItems: 'flex-start' },
  inline: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'baseline', columnGap: SPACE.sm, rowGap: SPACE.xs },
});
