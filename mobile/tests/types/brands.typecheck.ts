// 타입 수준 검사(명세 7.2절). node 테스트가 아니라 `tsc --noEmit -p .`(check.sh 3단계)가 이 파일을 본다.
// 아래 오류 기대 지시문 줄이 실제로 오류를 내지 않으면 tsc가 "Unused directive"(TS2578)로 실패한다.
import type { Provenanced } from '../../src/data/adapters';
import type { Example } from '../../src/data/examples';
import { EXAMPLE_PLANT, EXAMPLE_STATEMENT_AMOUNTS } from '../../src/data/examples';

declare const real: Provenanced<number>;
const takesReal = (v: Provenanced<number>) => v.value;
const takesExample = (v: Example<number>) => v.value;

takesReal(real);
takesExample(EXAMPLE_PLANT.capacityMw);
// @ts-expect-error 예시 값은 실제 값(Provenanced) 자리에 들어갈 수 없다
takesReal(EXAMPLE_PLANT.capacityMw);
// @ts-expect-error 실제 값은 예시(Example) 자리에 들어갈 수 없다
takesExample(real);
// 정산 예시 금액(F9)은 예시 값으로 계산한 예시다: 실제 값 자리에 들어갈 수 없다.
takesExample(EXAMPLE_STATEMENT_AMOUNTS.total);
// @ts-expect-error 정산 예시 금액은 실제 값(Provenanced) 자리에 들어갈 수 없다
takesReal(EXAMPLE_STATEMENT_AMOUNTS.total);
// @ts-expect-error 칩 없는 숫자는 Provenanced가 아니다
takesReal(42);
// @ts-expect-error adapters.ts 밖에서는 객체 리터럴로 브랜드를 만들 수 없다
takesReal({ value: 1, unit: 'MW', state: 'measured', chip: { kind: 'measured', text: '실측' }, source: 'x', time: { kind: '조회', kst: '' } });
