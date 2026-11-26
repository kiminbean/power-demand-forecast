// 입찰 계산기 요청(명세 4.2·6.3절). 서버는 계산을 한 번에 하나만 받는다(겹치면 503): 성향마다 자료 저장소 키를 따로 두어
// 5분 동안 다시 쓰고(저장소의 마운트 규칙), 요청은 앞 계산이 끝난 뒤에 보낸다. 503·네트워크 오류·시간 초과는 저장소가 1.5초 뒤
// 한 번만 다시 보내고, 그래도 실패하면 화면이 [다시 계산]을 준다. 폴링하지 않는다: 요청은 첫 마운트 · 성향 바꾸기 · [다시 계산] ·
// 오래된 입력의 다시 계산 한 번 · 5분 넘은 결과의 헤더 새로고침에서만 생긴다(BiddingTab.tsx).
import { createClient, ENDPOINTS } from '../../data/client';
import type { RiskLevel } from '../../data/client';
import type { OptimizedSegments } from '../../data/types';
import type { Loader } from '../../lib/resource';

/** 성향별 저장소 키(실제 쿼리는 클라이언트가 예시 용량과 함께 만든다). */
export const calcKey = (risk: RiskLevel) => `${ENDPOINTS.segments}?risk_level=${risk}`;

const noop = () => undefined;

/** 한 번에 하나만: 앞 작업이 끝나야(성공·실패 모두) 다음 작업을 시작한다. 기다리는 순서대로 보낸다. */
export function createSingleFlight() {
  let tail: Promise<unknown> = Promise.resolve();
  return function run<T>(task: () => Promise<T>): Promise<T> {
    const next = tail.then(task);
    tail = next.then(noop, noop);
    return next;
  };
}

export type SegmentsClient = { getOptimizedSegments(risk: RiskLevel): Promise<OptimizedSegments> };
const appGate = createSingleFlight();
const appClient = (signal: AbortSignal): SegmentsClient => createClient({ signal });
const aborted = () => Object.assign(new Error('aborted'), { name: 'AbortError' });

/** 저장소 loader. 저장소가 만든 10초 중단 신호를 쓰고, 차례를 기다리는 동안 중단되면 요청을 보내지 않는다. */
export function calcLoader(risk: RiskLevel, client: (signal: AbortSignal) => SegmentsClient = appClient,
  gate: ReturnType<typeof createSingleFlight> = appGate): Loader<OptimizedSegments> {
  return signal => gate(() => (signal.aborted ? Promise.reject(aborted()) : client(signal).getOptimizedSegments(risk)));
}
