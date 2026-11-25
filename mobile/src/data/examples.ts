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

/** 가격이 정해지는 원리(메리트 오더) 블록 6개: 싼 발전기부터 쌓아 수요를 채운 마지막 블록의 값이 가격이 된다. */
export const EXAMPLE_MERIT_BLOCKS = Object.freeze(
  [
    { name: '가', mw: 120, price: 60 },
    { name: '나', mw: 100, price: 95 },
    { name: '다', mw: 90, price: 120 },
    { name: '라', mw: 80, price: 150 },
    { name: '마', mw: 60, price: 190 },
    { name: '바', mw: 50, price: 240 },
  ].map(b => Object.freeze({
    name: example({ value: `예시 발전기 ${b.name}`, unit: '', label: '발전기' }),
    capacityMw: example({ value: b.mw, unit: 'MW', label: '공급 가능량' }),
    bidPrice: example({ value: b.price, unit: '원/kWh', label: '입찰 가격' }),
  })),
);
/** 위 블록에 맞춘 예시 수요: 가–라(390 MW)로 모자라 마 블록이 가격(190원/kWh)을 정한다. */
export const EXAMPLE_MERIT_DEMAND_MW = example({ value: 400, unit: 'MW', label: '예시 수요' });

/**
 * 예시 용량(50 MW)에 기대는 계산기 값(구간 물량, 기대 수입)을 예시로 표시한다(7.2절).
 * 값 자체는 API가 계산한 것이지만 가상 발전소를 전제로 하므로 실제 값처럼 보이면 안 된다.
 */
export function exampleFromCalculator(value: number, unit: ExampleUnit, label: string): Example<number> {
  return example({ value, unit, label });
}
