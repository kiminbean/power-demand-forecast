// 설명 글(public/content/guide.ko.json, 명세 4.4절) 읽기 상태. 처음 필요할 때(시트를 열 때) 상대 경로로 한 번 읽고, 읽은 내용은
// 기준 경로별로 기억해 다음 시트에서 바로 쓴다. 실패하면 [다시 시도]가 새로 요청한다(로더가 실패한 약속을 지운다).
import { useCallback, useEffect, useState } from 'react';
import { useNav } from '../../app/Shell';
import { createGuideLoader } from '../../content/guide';
import type { GuideContent } from '../../content/guide';

const loader = createGuideLoader();
const loaded = new Map<string, GuideContent>();

export type GuideState = { status: 'loading' | 'ready' | 'error'; content: GuideContent | null; retry: () => void };
type Held = Omit<GuideState, 'retry'>;
const held = (base: string): Held => {
  const hit = loaded.get(base);
  return hit ? { status: 'ready', content: hit } : { status: 'loading', content: null };
};

export function useGuide(): GuideState {
  const { base } = useNav();
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<Held>(() => held(base));
  useEffect(() => {
    if (loaded.has(base)) {
      setState(held(base));
      return undefined;
    }
    let live = true;
    setState({ status: 'loading', content: null });
    loader.load(base).then(
      content => {
        loaded.set(base, content);
        if (live) setState({ status: 'ready', content });
      },
      () => {
        if (live) setState({ status: 'error', content: null });
      },
    );
    return () => {
      live = false;
    };
  }, [base, attempt]);
  const retry = useCallback(() => setAttempt(n => n + 1), []);
  return { ...state, retry };
}
