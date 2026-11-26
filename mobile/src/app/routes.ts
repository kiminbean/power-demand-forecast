// 경로 상태의 단일 원천(명세 3.1–3.3절). DOM이 없다: 기준 경로 감지, URL 읽기·만들기, 탭·시트·헤더 ← 규칙이 모두 순수 함수이고
// tests/routes.test.cjs가 고정한다. router.ts(메모리 스택: 네이티브·node 테스트)와 router.web.ts(History API)가 아래 RouterApi를
// 같은 모양으로 구현한다(둘 다 `satisfies CreateRouter`). Metro는 웹에서 .web.ts를, TypeScript는 router.ts를 본다.
import { GLOSSARY_TITLES, TOUR_NAME, routeTitle, screenAnnouncement, tourLabel, tourTitle } from '../content/strings';
import type { GlossaryId } from '../content/strings';
import type { IconName } from '../ui/icons';

export type RouteKey = 'smp' | 'bidding' | 'settlement' | 'guide';
/** 탭 4개(3.1절). 경로 깊이는 1이라 상대 경로 자원(content/guide.ko.json)이 어느 경로에서든 기준 폴더로 풀린다. */
export type RouteDef = { key: RouteKey; path: string; tab: string; icon: IconName; heading: string; refreshable: boolean };
export const ROUTES: Readonly<Record<RouteKey, RouteDef>> = {
  smp: { key: 'smp', path: '', tab: '오늘', icon: 'chart', heading: '오늘 제주 전력시장', refreshable: true },
  bidding: { key: 'bidding', path: 'bidding', tab: '입찰', icon: 'clock', heading: '입찰', refreshable: true },
  settlement: { key: 'settlement', path: 'settlement', tab: '정산', icon: 'list', heading: '정산', refreshable: false },
  guide: { key: 'guide', path: 'guide', tab: '알아보기', icon: 'book', heading: '알아보기', refreshable: false },
};
const ROUTE_KEYS = Object.keys(ROUTES) as RouteKey[];

export const RISKS = ['conservative', 'moderate', 'aggressive'] as const;
export type Risk = (typeof RISKS)[number];
export const GUIDE_TOPICS = ['glossary', 'reading', 'method', 'sources', 'about'] as const;
export type GuideTopic = (typeof GUIDE_TOPICS)[number];
export const GUIDE_TOPIC_TITLES: Readonly<Record<GuideTopic, string>> = {
  glossary: '용어 사전', reading: '숫자 읽는 법', method: '예측 방법과 정확도', sources: '데이터 출처와 연결 상태', about: '앱 정보',
};
/** 산출 방식 시트가 설명하는 카드(4.4절). */
export const METHOD_CARDS = [
  'today-price', 'today-power', 'today-forecast', 'today-accuracy', 'bid-deadline', 'bid-ladder', 'bid-merit', 'settle-sample',
] as const;
export type MethodCard = (typeof METHOD_CARDS)[number];
/** 출처와 기준 시각 표의 행(4.4절 표 순서). */
export const SOURCE_ROWS = ['day-ahead', 'forecast', 'accuracy', 'demand', 'supply', 'weather', 'market', 'bidding', 'status'] as const;
export type SourceRow = (typeof SOURCE_ROWS)[number];
export const SHEET_KINDS = ['method', 'sources', 'term', 'theme'] as const;
export type SheetKind = (typeof SHEET_KINDS)[number];
export const TOUR_STEPS = 5;
/** 거래시간 01h–24h. */
export const HOURS = { first: 1, last: 24 } as const;
const GLOSSARY_IDS = Object.keys(GLOSSARY_TITLES) as GlossaryId[];

export type SheetState =
  | { kind: 'method'; card: MethodCard }
  | { kind: 'sources'; row: SourceRow | null }
  | { kind: 'term'; id: GlossaryId }
  | { kind: 'theme' };

export type RouteState = {
  key: RouteKey;
  /** 오늘·입찰: 고른 거래시간(1–24). null이면 화면이 지금 거래시간을 쓴다. */
  hour: number | null;
  /** 입찰: 성향. */
  risk: Risk | null;
  /** 알아보기 하위 상태는 하나만: 둘러보기 단계 > 용어 한 개 > 주제. */
  tour: number | null;
  term: GlossaryId | null;
  topic: GuideTopic | null;
  /** 어느 경로에서든 ?sheet=로 연다. */
  sheet: SheetState | null;
};

export const rootState = (key: RouteKey): RouteState => ({ key, hour: null, risk: null, tour: null, term: null, topic: null, sheet: null });

const oneOf = <T extends string>(list: readonly T[], text: string | null | undefined): T | null =>
  (text !== null && text !== undefined && (list as readonly string[]).includes(text) ? (text as T) : null);
const intIn = (value: unknown, first: number, last: number): number | null =>
  (typeof value === 'number' && Number.isInteger(value) && value >= first && value <= last ? value : null);
const intOf = (text: string | undefined): number | null => (text !== undefined && /^\d{1,2}$/.test(text) ? Number(text) : null);

function validSheet(sheet: SheetState | null): SheetState | null {
  if (!sheet) return null;
  switch (sheet.kind) {
    case 'method': return oneOf(METHOD_CARDS, sheet.card) ? sheet : null;
    case 'sources': return { kind: 'sources', row: oneOf(SOURCE_ROWS, sheet.row) };
    case 'term': return oneOf(GLOSSARY_IDS, sheet.id) ? sheet : null;
    case 'theme': return { kind: 'theme' };
    default: return null;
  }
}

/** 경로마다 쓰는 값만 남긴다(오늘: hour · 입찰: hour, risk · 알아보기: 하위 상태 하나 · 모든 경로: sheet). 틀린 값은 버린다. */
export function sanitize(s: RouteState): RouteState {
  const out = rootState(s.key);
  if (s.key === 'smp' || s.key === 'bidding') out.hour = intIn(s.hour, HOURS.first, HOURS.last);
  if (s.key === 'bidding') out.risk = oneOf(RISKS, s.risk);
  if (s.key === 'guide') {
    out.tour = intIn(s.tour, 1, TOUR_STEPS);
    out.term = out.tour === null ? oneOf(GLOSSARY_IDS, s.term) : null;
    out.topic = out.tour === null && out.term === null ? oneOf(GUIDE_TOPICS, s.topic) : null;
  }
  out.sheet = validSheet(s.sheet);
  return out;
}

const normalizeBase = (base: string) => {
  const lead = base.startsWith('/') ? base : `/${base}`;
  return lead.endsWith('/') ? lead : `${lead}/`;
};

export function buildUrl(state: RouteState, base: string): string {
  const s = sanitize(state);
  const query: string[] = [];
  const put = (name: string, value: string | number | null) => {
    if (value !== null) query.push(`${name}=${encodeURIComponent(String(value))}`);
  };
  put('hour', s.hour);
  put('risk', s.risk);
  put('tour', s.tour);
  put('term', s.term);
  put('topic', s.topic);
  if (s.sheet) {
    put('sheet', s.sheet.kind);
    if (s.sheet.kind === 'method') put('card', s.sheet.card);
    if (s.sheet.kind === 'sources') put('row', s.sheet.row);
    if (s.sheet.kind === 'term') put('id', s.sheet.id);
  }
  return `${normalizeBase(base)}${ROUTES[s.key].path}${query.length ? `?${query.join('&')}` : ''}`;
}

// 같은 이름이 두 번이면 첫 값을 쓰고, 깨진 % 인코딩은 그 쌍만 버린다. 어느 쪽이든 정규형이 아니라서 URL은 고쳐진다.
function readQuery(search: string): Map<string, string> {
  const query = new Map<string, string>();
  for (const pair of search.replace(/^\?/, '').split('&')) {
    if (!pair) continue;
    const at = pair.indexOf('=');
    try {
      const name = decodeURIComponent((at < 0 ? pair : pair.slice(0, at)).replace(/\+/g, ' '));
      const value = decodeURIComponent((at < 0 ? '' : pair.slice(at + 1)).replace(/\+/g, ' '));
      if (!query.has(name)) query.set(name, value);
    } catch {
      continue;
    }
  }
  return query;
}

function readSheet(query: Map<string, string>): SheetState | null {
  switch (query.get('sheet')) {
    case 'method': {
      const card = oneOf(METHOD_CARDS, query.get('card'));
      return card ? { kind: 'method', card } : null;
    }
    case 'sources': return { kind: 'sources', row: oneOf(SOURCE_ROWS, query.get('row')) };
    case 'term': {
      const id = oneOf(GLOSSARY_IDS, query.get('id'));
      return id ? { kind: 'term', id } : null;
    }
    case 'theme': return { kind: 'theme' };
    default: return null;
  }
}

export type ParsedLocation = { state: RouteState; canonical: boolean };

/**
 * pathname + search → 경로 상태. 모르는 경로(/mobile/foo, 깊은 경로, 끝 슬래시 없는 /mobile)는 오늘의 첫 상태, 모르는 쿼리 값은
 * 버린다. canonical이 거짓이면 라우터가 replaceState로 정규형 URL로 바꾼다(조용히 틀린 상태를 보이지 않는다, 3.1절).
 */
export function parseLocation(pathname: string, search: string, base: string): ParsedLocation {
  const root = normalizeBase(base);
  const rel = pathname.startsWith(root) ? pathname.slice(root.length) : null;
  const key = rel === null ? null : ROUTE_KEYS.find(k => ROUTES[k].path === rel) ?? null;
  let state = rootState('smp');
  if (key !== null) {
    const query = readQuery(search);
    state = sanitize({
      key, hour: intOf(query.get('hour')), risk: oneOf(RISKS, query.get('risk')), tour: intOf(query.get('tour')),
      term: oneOf(GLOSSARY_IDS, query.get('term')), topic: oneOf(GUIDE_TOPICS, query.get('topic')), sheet: readSheet(query),
    });
  }
  return { state, canonical: pathname + search === buildUrl(state, root) };
}

const BUNDLE_MARK = '_expo/static/js/web/';
const MOBILE_PREFIX = '/mobile';

/**
 * 기준 경로(3.2절, 빌드 플래그 없음): 번들 script src의 `_expo/` 앞부분. multiplex(:10005·:10104)에서는 public_server.py가
 * src를 /mobile/_expo/…로 고쳐 주므로 '/mobile/', 단일 앱 모드에서는 '/'. 못 찾으면 pathname이 /mobile이거나 /mobile/로
 * 시작할 때만 '/mobile/'(startsWith('/mobile')만 보면 /mobile-x도 걸린다).
 */
export function detectBase(scriptSrc: string | null | undefined, pathname: string): string {
  if (scriptSrc && scriptSrc.includes(BUNDLE_MARK)) {
    const path = scriptSrc.replace(/^(?:[a-z][a-z\d+.-]*:)?\/\/[^/]*/i, '');
    const base = path.slice(0, path.indexOf(BUNDLE_MARK));
    if (base.startsWith('/') && base.endsWith('/') && !base.startsWith('//')) return base;
  }
  return pathname === MOBILE_PREFIX || pathname.startsWith(`${MOBILE_PREFIX}/`) ? `${MOBILE_PREFIX}/` : '/';
}

/** 셸이 쓰지 않는 탭(F6: 알아보기)과 등록되지 않은 시트는 모르는 값처럼 다룬다: 첫 탭의 첫 상태 / 시트 없음. */
export function restrict(s: RouteState, tabs: readonly RouteKey[], sheets: readonly SheetKind[]): RouteState {
  const reachable = tabs.includes(s.key) ? s : rootState(tabs[0] ?? 'smp');
  return reachable.sheet && !sheets.includes(reachable.sheet.kind) ? { ...reachable, sheet: null } : reachable;
}

export const hasSubState = (s: RouteState) => s.key === 'guide' && (s.tour !== null || s.term !== null || s.topic !== null);
/** 헤더 ←의 부모: 용어 한 개 → 용어 사전, 그 밖의 하위 상태 → 그 탭의 첫 상태. */
export const parentOf = (s: RouteState): RouteState =>
  (s.key === 'guide' && s.term !== null ? { ...rootState('guide'), topic: 'glossary' } : rootState(s.key));
/** 같은 화면인지(초점·알림·스크롤 기억의 단위). 거래시간·성향 선택과 시트는 화면을 바꾸지 않는다. */
export const screenId = (s: RouteState) => `${s.key}|${s.tour ?? ''}|${s.term ?? ''}|${s.topic ?? ''}`;

export function screenLabel(s: RouteState): string {
  if (s.key === 'guide' && s.tour !== null) return tourLabel(s.tour, TOUR_STEPS);
  if (s.key === 'guide' && s.term !== null) return GLOSSARY_TITLES[s.term];
  if (s.key === 'guide' && s.topic !== null) return GUIDE_TOPIC_TITLES[s.topic];
  return ROUTES[s.key].tab;
}
/** 하위 상태 헤더의 제목(15/700). 둘러보기는 단계 제목이 h1이라 헤더에는 이름만 둔다. */
export const headerTitle = (s: RouteState) => (s.tour !== null ? TOUR_NAME : screenLabel(s));
/** document.title: "오늘 · eXeco 제주 전력시장", "둘러보기 2/5 · eXeco 제주 전력시장". */
export const documentTitle = (s: RouteState) =>
  (s.key === 'guide' && s.tour !== null ? tourTitle(s.tour, TOUR_STEPS) : routeTitle(screenLabel(s)));
/** live region 알림: "입찰 화면". */
export const routeAnnouncement = (s: RouteState) => screenAnnouncement(screenLabel(s));

// ---------------------------------------------------------------- 라우터 계약
export type NavKind = 'init' | 'push' | 'replace' | 'pop';
/** 지금 히스토리 항목의 표지: 우리가 넣었나(state.pd), 우리가 연 시트 항목인가(state.pdSheet). */
export type EntryMarks = { pushed: boolean; sheet: boolean };
export type RouterListener = (state: RouteState, kind: NavKind) => void;

/** router.web.ts가 쓰는 브라우저 표면. 기본은 window, 테스트는 가짜를 넣는다. */
export type HistoryHost = {
  location: { pathname: string; search: string };
  history: {
    readonly state: unknown;
    pushState(data: unknown, unused: string, url?: string | null): void;
    replaceState(data: unknown, unused: string, url?: string | null): void;
    back(): void;
  };
  addEventListener(type: 'popstate', listener: () => void): void;
  removeEventListener(type: 'popstate', listener: () => void): void;
  document?: { querySelector(selector: string): { getAttribute(name: string): string | null } | null };
};

export interface RouterApi {
  readonly base: string;
  current(): RouteState;
  entry(): EntryMarks;
  /** 정규형이 아닌 첫 URL을 replaceState로 고치고 popstate를 듣기 시작한다. 멈추는 함수를 돌려준다(렌더 밖, 효과에서 부른다). */
  start(): () => void;
  push(next: RouteState, options?: { sheet?: boolean }): void;
  replace(next: RouteState): void;
  back(): void;
  subscribe(listener: RouterListener): () => void;
}
export type RouterOptions = { initialUrl?: string; base?: string; host?: HistoryHost };
export type CreateRouter = (options?: RouterOptions) => RouterApi;

/** 탭 누름: 지금 탭이면 히스토리를 건드리지 않고 'top'(셸이 맨 위로 올린다, iOS 방식), 아니면 그 탭의 첫 상태를 새 항목으로. */
export function pressTab(router: RouterApi, key: RouteKey): 'top' | 'pushed' {
  if (router.current().key === key) return 'top';
  router.push(rootState(key));
  return 'pushed';
}

/** 시트 열기: 새 항목(state.pdSheet) + ?sheet=…. 탭은 그대로라서 브라우저 뒤로가 시트만 닫는다. */
export function openSheet(router: RouterApi, sheet: SheetState): void {
  router.push({ ...router.current(), sheet }, { sheet: true });
}

/** 시트 닫기(닫기 버튼·스크림·Esc): 우리가 넣은 시트 항목이면 뒤로, 딥링크로 들어온 시트면 쿼리만 뺀다. */
export function closeSheet(router: RouterApi): void {
  const s = router.current();
  if (!s.sheet) return;
  if (router.entry().sheet) router.back();
  else router.replace({ ...s, sheet: null });
}

/** 헤더 ←(하위 상태): 우리가 넣은 항목이면 뒤로, 딥링크로 들어온 첫 화면이면 부모 경로로 바꾼다(앱 밖으로 나가지 않는다). */
export function headerBack(router: RouterApi): void {
  if (router.entry().pushed) router.back();
  else router.replace(parentOf(router.current()));
}
