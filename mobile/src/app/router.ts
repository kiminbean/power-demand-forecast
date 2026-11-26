// 메모리 스택 라우터(3.3절): 네이티브와 node 테스트용. 웹은 같은 이름의 router.web.ts(History API)를 Metro가 고른다.
// 항목마다 URL과 표지(우리가 넣었나 · 우리가 연 시트인가)를 들고, 브라우저 히스토리와 같은 규칙으로 앞 항목을 잘라 낸다.
import { buildUrl, parseLocation, sanitize } from './routes';
import type { CreateRouter, EntryMarks, NavKind, RouteState, RouterApi, RouterListener } from './routes';

type Entry = { url: string; marks: EntryMarks };

const splitUrl = (url: string): [string, string] => {
  const at = url.indexOf('?');
  return at < 0 ? [url, ''] : [url.slice(0, at), url.slice(at)];
};

export const createRouter = ((options = {}): RouterApi => {
  const base = options.base ?? '/';
  const [pathname, search] = splitUrl(options.initialUrl ?? base);
  let state = parseLocation(pathname, search, base).state;
  // 첫 항목은 정규형 URL로 둔다(웹의 replaceState와 같은 결과).
  const entries: Entry[] = [{ url: buildUrl(state, base), marks: { pushed: false, sheet: false } }];
  let index = 0;
  const listeners = new Set<RouterListener>();
  const emit = (kind: NavKind) => listeners.forEach(listener => listener(state, kind));

  return {
    base,
    current: () => state,
    entry: () => entries[index].marks,
    start: () => () => undefined,
    push(next: RouteState, { sheet = false } = {}) {
      state = sanitize(next);
      entries.splice(index + 1);
      entries.push({ url: buildUrl(state, base), marks: { pushed: true, sheet } });
      index = entries.length - 1;
      emit('push');
    },
    replace(next: RouteState) {
      state = sanitize(next);
      const marks = entries[index].marks;
      entries[index] = { url: buildUrl(state, base), marks: { pushed: marks.pushed, sheet: marks.sheet && state.sheet !== null } };
      emit('replace');
    },
    back() {
      if (index === 0) return;
      index -= 1;
      state = parseLocation(...splitUrl(entries[index].url), base).state;
      emit('pop');
    },
    subscribe(listener: RouterListener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}) satisfies CreateRouter;
