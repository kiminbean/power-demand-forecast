// 목이 없는 bare import: 로더가 조용히 {}를 주지 않고 실패해야 한다.
import { isAxiosError } from 'axios';

export const hasAxios = typeof isAxiosError === 'function';
