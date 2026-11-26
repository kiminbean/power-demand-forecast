// 키별 자료 상태(명세 6.3절): { data, fetchedAt, status, error, busy }. 같은 키는 한 번에 하나만 요청하고(중복 제거) 실패해도
// 마지막 좋은 값을 남긴다. 요청마다 AbortController 10초, 네트워크 오류·502·503·시간 초과에만 1.5초 뒤 한 번 더 보낸다
// (4xx·모양 오류는 다시 보내도 같다). 저장소가 모듈에 하나라서 화면과 시트(산출 방식·출처)가 같은 값을 본다.
// 시각(fetchedAt)은 기기 시계다: 나이(지금 − 조회)는 서버 시계 차이와 상관없고, KST 글자로 바꿀 때만 화면이 차이를 더한다.
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from 'react';

export const REQUEST_TIMEOUT_MS = 10_000;
export const RETRY_DELAY_MS = 1_500;
/** 화면이 보이는 동안 다시 받는 주기이자, 숨었다 보이거나 다시 들어왔을 때 새로 받는 나이(6.3절). */
export const REFRESH_EVERY_MS = 5 * 60_000;

export type ResourceStatus = 'idle' | 'loading' | 'ready' | 'error';
/** idle: 아직 요청 안 함 · loading: 첫 값을 기다림 · ready: 마지막 요청 성공 · error: 마지막 요청 실패(data는 남을 수 있다). */
export type Resource<T> = Readonly<{ data: T | null; fetchedAt: number | null; status: ResourceStatus; error: unknown; busy: boolean }>;
export type Loader<T> = (signal: AbortSignal) => Promise<T>;

export class RequestTimeoutError extends Error {
  readonly retryable = true;
  constructor() {
    super('요청 시간이 지났습니다');
    this.name = 'RequestTimeoutError';
  }
}

/** ApiNetworkError·ApiHttpError(502·503)·RequestTimeoutError가 retryable을 참으로 둔다. */
export const isRetryable = (error: unknown) =>
  typeof error === 'object' && error !== null && (error as { retryable?: unknown }).retryable === true;

type Timer = unknown;
export type StoreDeps = { now: () => number; setTimer: (fn: () => void, ms: number) => Timer; clearTimer: (timer: Timer) => void };
const defaultDeps = (): StoreDeps => ({
  now: () => Date.now(),
  setTimer: (fn, ms) => setTimeout(fn, ms),
  clearTimer: timer => clearTimeout(timer as ReturnType<typeof setTimeout>),
});

const IDLE: Resource<never> = Object.freeze({ data: null, fetchedAt: null, status: 'idle', error: null, busy: false });
type Outcome<T> = { ok: true; data: T } | { ok: false; error: unknown };

export function createResourceStore(deps: StoreDeps = defaultDeps()) {
  type Entry = { snap: Resource<unknown>; listeners: Set<() => void>; inflight: Promise<boolean> | null };
  const entries = new Map<string, Entry>();
  const entryOf = (key: string): Entry => {
    let e = entries.get(key);
    if (!e) {
      e = { snap: IDLE, listeners: new Set(), inflight: null };
      entries.set(key, e);
    }
    return e;
  };
  const patch = (e: Entry, next: Partial<Resource<unknown>>) => {
    e.snap = Object.freeze({ ...e.snap, ...next });
    e.listeners.forEach(fn => fn());
  };
  const wait = (ms: number) => new Promise<void>(resolve => {
    deps.setTimer(resolve, ms);
  });

  // 불러오는 쪽이 signal을 무시해도 10초 뒤에는 반드시 끝난다(경주). 끝나면 타이머를 지운다.
  async function attempt<T>(loader: Loader<T>): Promise<Outcome<T>> {
    const controller = new AbortController();
    let timer: Timer = null;
    const timeout = new Promise<never>((_, reject) => {
      timer = deps.setTimer(() => {
        reject(new RequestTimeoutError());
        controller.abort();
      }, REQUEST_TIMEOUT_MS);
    });
    try {
      return { ok: true, data: await Promise.race([loader(controller.signal), timeout]) };
    } catch (error) {
      return { ok: false, error };
    } finally {
      deps.clearTimer(timer);
    }
  }

  return {
    get<T>(key: string): Resource<T> {
      return entryOf(key).snap as Resource<T>;
    },
    subscribe(key: string, listener: () => void): () => void {
      const e = entryOf(key);
      e.listeners.add(listener);
      return () => e.listeners.delete(listener);
    },
    /** 진행 중이면 그 요청을 같이 기다린다. 성공하면 true. */
    load<T>(key: string, loader: Loader<T>): Promise<boolean> {
      const e = entryOf(key);
      if (e.inflight) return e.inflight;
      patch(e, { busy: true, status: e.snap.data === null ? 'loading' : e.snap.status });
      const run = async () => {
        let outcome = await attempt(loader);
        if (!outcome.ok && isRetryable(outcome.error)) {
          await wait(RETRY_DELAY_MS);
          outcome = await attempt(loader);
        }
        e.inflight = null;
        if (outcome.ok) patch(e, { data: outcome.data, fetchedAt: deps.now(), status: 'ready', error: null, busy: false });
        else patch(e, { status: 'error', error: outcome.error, busy: false });
        return outcome.ok;
      };
      e.inflight = run();
      return e.inflight;
    },
    now: () => deps.now(),
  };
}
export type ResourceStore = ReturnType<typeof createResourceStore>;

/** 앱의 저장소(화면·시트가 함께 쓴다). 테스트는 createResourceStore로 따로 만든다. */
export const appResources = createResourceStore();

/** 처음 마운트(다른 탭에서 돌아온 것 포함): 값이 없거나, 마지막이 실패였거나, 5분 넘은 것만 받는다. */
export function shouldLoadOnMount(snap: Resource<unknown>, now: number): boolean {
  if (snap.busy) return false;
  if (snap.data === null || snap.status === 'error') return true;
  return snap.fetchedAt === null || now - snap.fetchedAt >= REFRESH_EVERY_MS;
}

export type ResourceHandle<T> = Resource<T> & { reload: () => Promise<boolean> };

/**
 * loader를 주면 마운트 때 shouldLoadOnMount 규칙으로 받고 reload로 다시 받는다. 주지 않으면 엿보기만 한다(시트·패널:
 * 화면이 받은 값을 같이 보되 새 요청은 보내지 않는다).
 */
export function useResource<T>(key: string, loader?: Loader<T>, store: ResourceStore = appResources): ResourceHandle<T> {
  const subscribe = useCallback((fn: () => void) => store.subscribe(key, fn), [store, key]);
  const read = useCallback(() => store.get<T>(key), [store, key]);
  const snap = useSyncExternalStore(subscribe, read, read);
  const latest = useRef(loader);
  latest.current = loader;
  const canLoad = loader !== undefined;
  const reload = useCallback(() => (latest.current ? store.load(key, latest.current) : Promise.resolve(false)), [store, key]);
  useEffect(() => {
    if (canLoad && shouldLoadOnMount(store.get(key), store.now())) reload();
  }, [store, key, canLoad, reload]);
  // 값이 바뀔 때만 새 객체다(화면의 useMemo가 이 객체 하나를 의존으로 쓴다).
  return useMemo(() => ({ ...snap, reload }), [snap, reload]);
}

export type Visibility = { isVisible: () => boolean; watchVisible: (listener: () => void) => () => void };

/**
 * 화면이 보이는 동안 everyMs마다 refreshAll, 숨었다 다시 보이면 마지막 조회가 everyMs를 넘었을 때만 refreshAll.
 * 입찰 계산기처럼 폴링하지 않는 자료는 이 훅을 쓰지 않는다(4.2절).
 */
export function useRefreshWhileVisible(refreshAll: () => void, oldestFetchedAt: () => number | null, visibility: Visibility,
  everyMs: number = REFRESH_EVERY_MS, store: ResourceStore = appResources): void {
  const latest = useRef({ refreshAll, oldestFetchedAt });
  latest.current = { refreshAll, oldestFetchedAt };
  const { isVisible, watchVisible } = visibility;
  useEffect(() => {
    const id = setInterval(() => {
      if (isVisible()) latest.current.refreshAll();
    }, everyMs);
    const stop = watchVisible(() => {
      const oldest = latest.current.oldestFetchedAt();
      if (oldest !== null && store.now() - oldest >= everyMs) latest.current.refreshAll();
    });
    return () => {
      clearInterval(id);
      stop();
    };
  }, [isVisible, watchVisible, everyMs, store]);
}
