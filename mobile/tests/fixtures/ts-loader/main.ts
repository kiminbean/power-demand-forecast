// ts-loader 자체 시험용 모듈(앱 번들에 들어가지 않는다).
import { Platform } from 'react-native';
import { add } from './math';
import type { Shape } from './types';
import { pick } from './platform-pick';
import data from './data.json';

declare const PD_TEST_VALUE: string | undefined;

export const sum = add(2, 3);
export const os = Platform.OS;
export const picked = pick();
export const jsonValue = data.value;
export const injected = typeof PD_TEST_VALUE === 'string' ? PD_TEST_VALUE : 'absent';
export const list = [1, 2, 3].map(n => add(n, 1));

export function area(shape: Shape): number {
  return shape.w * shape.h;
}
