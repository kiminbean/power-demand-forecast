// History API 라우터(3.3절, 웹 전용). Metro는 웹 빌드에서 router.ts 대신 이 파일을 고른다(export 시그니처가 같다).
// 우리가 넣은 항목은 history.state에 { pd: 1 }(시트는 pdSheet: true도)를 남긴다. 닫기·헤더 ←는 이 표지로 history.back()과
// replaceState 중 하나를 고른다. 새로고침해도 브라우저가 history.state를 지키므로 규칙이 그대로다.
import { buildUrl, detectBase, parseLocation, sanitize } from './routes';
import type { CreateRouter, EntryMarks, HistoryHost, NavKind, RouteState, RouterApi, RouterListener } from './routes';

const BUNDLE_SCRIPT = 'script[src*="_expo/static/js/web/"]';

type EntryData = Record<string, unknown>;
const dataOf = (value: unknown): EntryData => (value && typeof value === 'object' ? { ...(value as EntryData) } : {});
const marksOf = (value: unknown): EntryMarks => {
  const data = dataOf(value);
  return { pushed: data.pd === 1, sheet: data.pdSheet === true };
};

function windowHost(): HistoryHost {
  if (typeof window === 'undefined') throw new Error('router.web: window is not available');
  return window as unknown as HistoryHost;
}

export const createRouter = ((options = {}): RouterApi => {
  const host = options.host ?? windowHost();
  const script = host.document ? host.document.querySelector(BUNDLE_SCRIPT) : null;
  const base = options.base ?? detectBase(script ? script.getAttribute('src') : null, host.location.pathname);
  const read = () => parseLocation(host.location.pathname, host.location.search, base);
  let state = read().state;
  // history.back() 뒤 popstate가 오기 전의 두 번째 back()은 버린다(Esc를 두 번 눌러도 시트 아래 탭까지 되돌아가지 않게).
  let backPending = false;
  const listeners = new Set<RouterListener>();
  const emit = (kind: NavKind) => listeners.forEach(listener => listener(state, kind));
  const canonicalize = () => {
    const parsed = read();
    state = parsed.state;
    if (!parsed.canonical) host.history.replaceState(host.history.state, '', buildUrl(state, base));
  };
  const onPop = () => {
    backPending = false;
    canonicalize();
    emit('pop');
  };

  return {
    base,
    current: () => state,
    entry: () => marksOf(host.history.state),
    start() {
      canonicalize();
      host.addEventListener('popstate', onPop);
      return () => host.removeEventListener('popstate', onPop);
    },
    push(next: RouteState, { sheet = false } = {}) {
      state = sanitize(next);
      host.history.pushState(sheet ? { pd: 1, pdSheet: true } : { pd: 1 }, '', buildUrl(state, base));
      emit('push');
    },
    replace(next: RouteState) {
      state = sanitize(next);
      const data = dataOf(host.history.state);
      if (state.sheet === null) delete data.pdSheet;
      host.history.replaceState(Object.keys(data).length ? data : null, '', buildUrl(state, base));
      emit('replace');
    },
    back() {
      if (backPending) return;
      backPending = true;
      host.history.back();
    },
    subscribe(listener: RouterListener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}) satisfies CreateRouter;
