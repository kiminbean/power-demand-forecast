// 지어낸 값의 유일한 집(명세 7.9절). 실제 발전소 이름을 쓰지 않는다. 화면에서는 ┆예시┆ 리본·칩과 SampleValue로만 보인다.
// Example<T>는 이 파일 안에서만 만들 수 있는 브랜드라 Provenanced<T>(실제 API 값) 자리에 넣을 수 없다.
declare const exampleBrand: unique symbol;

export type ExampleUnit = 'MW' | 'MWh' | '원/kWh' | '천원' | '';
export type ExampleFields<T> = { value: T; unit: ExampleUnit; label: string };
export type Example<T> = Readonly<ExampleFields<T>> & { readonly [exampleBrand]: true };

function example<T>(fields: ExampleFields<T>): Example<T> {
  return Object.freeze({ ...fields }) as unknown as Example<T>;
}

/** 가상 발전소(입찰 계산기의 용량 인자도 이 값을 쓴다). */
export const EXAMPLE_PLANT = Object.freeze({
  name: example({ value: '예시 풍력발전소', unit: '', label: '발전소 이름' }),
  capacityMw: example({ value: 50, unit: 'MW', label: '설비 용량' }),
});

/**
 * 예시 정산서(이중 정산): 하루전 낙찰 120 MWh × 150원/kWh + 실시간 편차 −6 MWh × 140원/kWh.
 * 계산은 정산 화면(F9)이 format.ts를 거쳐 한다(1,800만원 − 84만원 = 1,716만원).
 */
export const EXAMPLE_STATEMENT = Object.freeze({
  dayAheadMwh: example({ value: 120, unit: 'MWh', label: '하루전 낙찰량' }),
  dayAheadPrice: example({ value: 150, unit: '원/kWh', label: '하루전 가격' }),
  realtimeDeviationMwh: example({ value: -6, unit: 'MWh', label: '실시간 편차' }),
  realtimePrice: example({ value: 140, unit: '원/kWh', label: '실시간 가격' }),
});

/**
 * 가격이 정해지는 원리(메리트 오더, 4.2절) 예시 발전기 6개, 싼 것부터. 가격은 명세 그대로(풍력 A 0원 · 태양광 B 0원 · 바이오 C 80원 ·
 * LNG D 150원 · 유류 E 220원 · 유류 F 260원), 공급 가능량은 그림이 읽히도록 정한 예시다. 실제 발전소 이름이 아니다.
 */
export const EXAMPLE_MERIT_BLOCKS = Object.freeze(
  [
    { name: '풍력 A', mw: 100, price: 0 },
    { name: '태양광 B', mw: 80, price: 0 },
    { name: '바이오 C', mw: 40, price: 80 },
    { name: 'LNG D', mw: 200, price: 150 },
    { name: '유류 E', mw: 120, price: 220 },
    { name: '유류 F', mw: 80, price: 260 },
  ].map(b => Object.freeze({
    name: example({ value: b.name, unit: '', label: '예시 발전기' }),
    capacityMw: example({ value: b.mw, unit: 'MW', label: '공급 가능량' }),
    bidPrice: example({ value: b.price, unit: '원/kWh', label: '입찰 가격' }),
  })),
);
/** 위 블록에 맞춘 예시 수요: 풍력 A–LNG D(420 MW)로 모자라 유류 E가 마지막으로 필요하고 그 가격(220원/kWh)이 SMP가 된다. */
export const EXAMPLE_MERIT_DEMAND_MW = example({ value: 450, unit: 'MW', label: '예시 수요' });

/**
 * 예시 용량(50 MW)에 기대는 계산기 값(구간 물량, 기대 수입)을 예시로 표시한다(7.2절).
 * 값 자체는 API가 계산한 것이지만 가상 발전소를 전제로 하므로 실제 값처럼 보이면 안 된다.
 */
export function exampleFromCalculator(value: number, unit: ExampleUnit, label: string): Example<number> {
  return example({ value, unit, label });
}
