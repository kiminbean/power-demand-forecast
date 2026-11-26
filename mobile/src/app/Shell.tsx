// 앱 셸(명세 3.3·3.4절): 헤더(App.tsx의 훅 없는 Header를 renderHeader로 받는다) · 오프라인 띠 · 화면 · 탭 막대(≥ 1024는 왼쪽
// 레일) · 시트 · polite 알림 하나. 화면이 바뀌면 document.title을 바꾸고 h1로 초점을 옮기고 "입찰 화면"처럼 알린다. 스크롤 위치는
// 화면마다 메모리에 두고 뒤로·앞으로(popstate) 때 되돌린다. 지금 탭을 다시 누르면 히스토리를 건드리지 않고 맨 위로 올린다.
// 화면이 쓰는 것: useNav(경로 상태·이동·시트), useClock(60초마다 바뀌는 KST 지금·서버 시계 차이), useShellData(새로고침 손잡이).
import { createContext, useCallback, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type React from 'react';
import { ScrollView, StyleSheet, View, useWindowDimensions } from 'react-native';
import type { TextStyle } from 'react-native';
import { createRouter } from './router';
import {
  ROUTES, buildUrl, closeSheet, documentTitle, hasSubState, headerBack, headerTitle, openSheet, pressTab, restrict, rootState,
  routeAnnouncement, screenId,
} from './routes';
import type { NavKind, RouteKey, RouteState, RouterApi, SheetKind, SheetState } from './routes';
import { BUILT_IN_SHEETS, SheetHost } from './SheetHost';
import type { SheetRegistry } from './SheetHost';
import { isOnline, setDocumentTitle, watchOnline, watchVisible } from './head';
import { useTheme } from './theme';
import { NowPill } from '../ui/Chips';
import { LiveRegion, OfflineBanner, useAnnounce } from '../ui/Live';
import { ErrorBoundary } from '../ui/Screen';
import { Rail, TabBar } from '../ui/TabBar';
import { Text, textStyle } from '../ui/Text';
import { LAYOUT, SPACE } from '../ui/tokens';
import { domOf, focusHeading, scrollBoxes, setInert, web } from '../ui/web';
import type { DomEl } from '../ui/web';
import { MINUTE_MS, clockSkew, kstNow } from '../lib/kst';
import type { ClockSkew } from '../lib/kst';
import {
  ONLINE_AGAIN_TEXT, REFRESH_FAILED_TEXT, REFRESH_NONE_REASON, REFRESH_OFFLINE_REASON, offlineText, refreshDoneText, refreshFailedText,
  screenReloadingText,
} from '../content/strings';

export type HeaderPalette = { background: string; border: string; wordmark: string; title: string; version: string; icon: string };
export type HeaderFonts = { wordmark?: TextStyle; title?: TextStyle; version?: TextStyle };
export type HeaderRefresh = { onPress: () => void; busy: boolean; disabledReason: string | null };
/** App.tsx의 Header가 받는 것. Header는 훅을 쓰지 않으므로 색·글자 꼴·상태를 모두 셸이 넘긴다. */
export type ShellHeaderProps = {
  title?: string;
  showBack: boolean;
  onBack: () => void;
  onHome: () => void;
  palette: HeaderPalette;
  fonts: HeaderFonts;
  refresh: HeaderRefresh;
};

export type Nav = {
  state: RouteState;
  /** 감지한 기준 경로('/mobile/' 또는 '/'). 상대 자원(content/guide.ko.json)의 주소에 쓴다. */
  base: string;
  /** 셸이 쓰는 탭. 아직 없는 탭(F6–F9의 알아보기)으로 가는 입구(둘러보기 배너·용어 사전 링크)는 그리지 않는다. */
  tabs: readonly RouteKey[];
  /** 새 히스토리 항목(둘러보기 단계·주제·용어). */
  push(next: RouteState): void;
  /** 같은 항목을 고친다(거래시간·성향 선택이 URL에 남는다). */
  replace(next: RouteState): void;
  openSheet(sheet: SheetState): void;
  closeSheet(): void;
};
export type Clock = { now: number; skew: ClockSkew | null; skewNotable: boolean; reportServerTime(serverMs: number, deviceMs?: number): void };
export type RefreshResult = { ok: true; at: number } | { ok: false; shownAt: number | null };
/** ≥ 1280 오른쪽 패널의 내용(4.6절: 그 화면의 4상태 범례·용어 3개·산출 방식 요약). 제목은 화면마다 다르다("입찰 읽는 법"). */
export type PanelContent = { title: string; body: React.ReactNode };
/** 화면이 셸에 알리는 것: 새로고침 손잡이와 마지막 성공 조회 시각(헤더 새로고침 결과 알림·오프라인 띠). */
export type ShellData = { refresh: () => Promise<RefreshResult>; fetchedAt: number | null };

const noop = () => undefined;
// 공급자 밖(VM 렌더 테스트·오류 화면)에서도 그릴 수 있게 아무것도 하지 않는 기본값을 둔다(useAnnounce·useTheme과 같은 규칙).
const NavContext = createContext<Nav>({ state: rootState('smp'), base: '/', tabs: [], push: noop, replace: noop, openSheet: noop, closeSheet: noop });
const ClockContext = createContext<Clock | null>(null);
const DataContext = createContext<(data: ShellData | null) => void>(noop);

export const useNav = () => useContext(NavContext);
export function useClock(): Clock {
  return useContext(ClockContext) ?? { now: kstNow(), skew: null, skewNotable: false, reportServerTime: noop };
}
/** 레거시 화면은 부르지 않는다: 그때 헤더 새로고침은 화면을 다시 그려 다시 받게 한다. */
export function useShellData(data: ShellData | null): void {
  const set = useContext(DataContext);
  const refresh = data ? data.refresh : null;
  const fetchedAt = data ? data.fetchedAt : null;
  useEffect(() => {
    set(refresh ? { refresh, fetchedAt } : null);
    return () => set(null);
  }, [set, refresh, fetchedAt]);
}

// 분이 바뀐 직후에 깨어난다. 1초 안의 차이는 같은 시계 차이로 본다(조회마다 다시 그리지 않는다).
const TICK_SLACK_MS = 50;
const SKEW_JITTER_MS = 1000;

function ClockProvider({ deviceNow, children }: { deviceNow: () => number; children: React.ReactNode }) {
  const [skew, setSkew] = useState<ClockSkew | null>(null);
  const [now, setNow] = useState(() => kstNow(null, deviceNow()));
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = () => {
      const t = kstNow(skew, deviceNow());
      setNow(t);
      if (timer !== undefined) clearTimeout(timer);
      timer = setTimeout(tick, MINUTE_MS - (t % MINUTE_MS) + TICK_SLACK_MS);
    };
    tick();
    const stopVisible = watchVisible(tick);
    return () => {
      if (timer !== undefined) clearTimeout(timer);
      stopVisible();
    };
  }, [skew, deviceNow]);
  const reportServerTime = useCallback((serverMs: number, deviceMs?: number) => {
    const next = clockSkew(serverMs, deviceMs ?? deviceNow());
    if (!next) return;
    setSkew(prev => (prev && prev.plausible === next.plausible && Math.abs(prev.skewMs - next.skewMs) < SKEW_JITTER_MS ? prev : next));
  }, [deviceNow]);
  const value = useMemo<Clock>(() => ({ now, skew, skewNotable: Boolean(skew && skew.notable), reportServerTime }),
    [now, skew, reportServerTime]);
  return <ClockContext.Provider value={value}>{children}</ClockContext.Provider>;
}

function useOnline(): boolean {
  const [online, setOnline] = useState(isOnline);
  useEffect(() => watchOnline(setOnline), []);
  return online;
}

// 레거시 화면은 자료가 늦게 와서 처음에는 기억한 위치만큼 길지 않을 수 있다: 몇 번 더 맞춘다. 그사이 사용자가 움직였거나
// 다른 화면으로 갔으면 그대로 둔다.
const RESTORE_RETRIES_MS = [150, 400, 1000];
function restoreScroll(root: DomEl | null, y: number, stillHere: () => boolean): void {
  if (y <= 0) return;
  let last = 0;
  const apply = () => {
    const box = scrollBoxes(root)[0];
    if (!box || !stillHere() || box.scrollTop >= y || (box.scrollTop !== 0 && box.scrollTop !== last)) return;
    box.scrollTop = y;
    last = box.scrollTop;
  };
  apply();
  RESTORE_RETRIES_MS.forEach(ms => setTimeout(apply, ms));
}

export type ShellProps = {
  /** 쓰는 탭(순서대로). 쓰지 않는 탭을 가리키는 URL은 첫 탭으로 바뀐다. */
  tabs: readonly RouteKey[];
  /** 자기 h1 행이 없는 레거시 화면: 셸이 main 랜드마크와 h1 행(제목 + NowPill)을 둘러 준다. 새 화면은 Screen이 그린다. */
  legacy?: readonly RouteKey[];
  renderRoute: (state: RouteState) => React.ReactNode;
  renderHeader: (props: ShellHeaderProps) => React.ReactNode;
  /** ≥ 1280 "이 화면 읽는 법" 패널(320). 내용을 주는 화면에만 생긴다(F6의 레거시 화면은 없음). */
  renderPanel?: (state: RouteState) => PanelContent | null;
  /** 화면 단계가 더하는 시트(산출 방식·출처·용어). 화면 테마 시트는 셸이 늘 가진다. */
  sheets?: SheetRegistry;
  /** 테스트가 넣는 라우터와 기기 시계. 기본은 플랫폼 라우터(웹: History API)와 Date.now. */
  router?: RouterApi;
  deviceNow?: () => number;
};

export function Shell({ deviceNow = Date.now, ...rest }: ShellProps) {
  return (
    <ErrorBoundary>
      <LiveRegion>
        <ClockProvider deviceNow={deviceNow}>
          <ShellBody {...rest} />
        </ClockProvider>
      </LiveRegion>
    </ErrorBoundary>
  );
}

const NO_KEYS: readonly RouteKey[] = [];

function ShellBody({ tabs, legacy = NO_KEYS, renderRoute, renderHeader, renderPanel, sheets, router: given }: Omit<ShellProps, 'deviceNow'>) {
  const { theme } = useTheme();
  const c = theme.color;
  const { width } = useWindowDimensions();
  const announce = useAnnounce();
  const [router] = useState<RouterApi>(() => given ?? createRouter());
  const [nav, setNav] = useState<{ state: RouteState; kind: NavKind }>(() => ({ state: router.current(), kind: 'init' }));
  const registry = useMemo<SheetRegistry>(() => ({ ...BUILT_IN_SHEETS, ...sheets }), [sheets]);
  const sheetKinds = useMemo(() => Object.keys(registry) as SheetKind[], [registry]);
  const shown = useMemo(() => restrict(nav.state, tabs, sheetKinds), [nav.state, tabs, sheetKinds]);
  const id = screenId(shown);
  const route = ROUTES[shown.key];
  const contentRef = useRef<View>(null);
  const headerRef = useRef<View>(null);
  const bodyRef = useRef<View>(null);
  const tabsRef = useRef<View>(null);
  const shownRef = useRef(shown);
  shownRef.current = shown;
  const lastId = useRef<string | null>(null);
  const scrollMemory = useRef(new Map<string, number>());
  const [data, setData] = useState<ShellData | null>(null);
  const [busy, setBusy] = useState(false);
  const [reloads, setReloads] = useState(0);
  const online = useOnline();
  const wasOnline = useRef(online);

  useEffect(() => router.start(), [router]);
  // 오프라인이 되고 풀릴 때 셸의 알림 하나로도 알린다(round 2: 띠의 role=status만으로는 글자가 이미 있어 읽히지 않을 수 있었다).
  useEffect(() => {
    if (wasOnline.current === online) return;
    wasOnline.current = online;
    announce(online ? ONLINE_AGAIN_TEXT : offlineText(data ? data.fetchedAt : null));
  }, [online, announce, data]);
  // 시트가 열린 동안 바깥(헤더·본문·탭 막대)은 inert(round 1 접근성: 화면 읽기 프로그램이 시트 밖으로 나갔다). 레이아웃 효과라서 시트가
  // 닫히며 연 버튼으로 초점을 돌려주기(시트의 일반 효과) 전에 풀린다.
  const sheetOpen = Boolean(shown.sheet && registry[shown.sheet.kind]);
  useLayoutEffect(() => {
    if (!sheetOpen) return undefined;
    const outside = [headerRef, bodyRef, tabsRef];
    outside.forEach(ref => setInert(ref, true));
    return () => outside.forEach(ref => setInert(ref, false));
  }, [sheetOpen]);
  useEffect(() => router.subscribe((state, kind) => {
    // 화면이 바뀌기 전(아직 옛 DOM)에 지금 화면의 스크롤 위치를 적어 둔다.
    const box = scrollBoxes(domOf(contentRef))[0];
    if (box) scrollMemory.current.set(screenId(shownRef.current), box.scrollTop);
    setNav({ state, kind });
  }), [router]);

  // 쓰지 않는 탭·등록되지 않은 시트를 가리키는 URL은 그 자리에서 고친다(조용히 틀린 상태를 보이지 않는다, 3.1절).
  useEffect(() => {
    if (buildUrl(shown, router.base) !== buildUrl(nav.state, router.base)) router.replace(shown);
  }, [router, shown, nav.state]);

  useEffect(() => {
    if (lastId.current === id) return;
    const first = lastId.current === null;
    lastId.current = id;
    setDocumentTitle(documentTitle(shown));
    if (first) return;
    const root = domOf(contentRef);
    focusHeading(root);
    if (nav.kind === 'pop') restoreScroll(root, scrollMemory.current.get(id) ?? 0, () => lastId.current === id);
    announce(routeAnnouncement(shown));
  }, [id, shown, nav.kind, announce]);

  const toTop = useCallback(() => {
    for (const box of scrollBoxes(domOf(contentRef))) box.scrollTop = 0;
  }, []);
  const onTab = useCallback((key: RouteKey) => {
    if (pressTab(router, key) === 'top') toTop();
  }, [router, toTop]);
  // 워드마크: 오늘로. 이미 오늘이면 탭을 다시 누른 것과 같다.
  const onHome = useCallback(() => {
    const s = router.current();
    if (s.key === 'smp' && !hasSubState(s)) toTop();
    else router.push(rootState('smp'));
  }, [router, toTop]);

  const disabledReason = !route.refreshable ? REFRESH_NONE_REASON : online ? null : REFRESH_OFFLINE_REASON;
  const onRefresh = useCallback(() => {
    if (disabledReason || busy) return;
    if (!data) {
      // 레거시 화면: 다시 그려서 다시 받게 한다. 결과(불러오는 중·오류)는 그 화면이 직접 보여 준다.
      setReloads(n => n + 1);
      announce(screenReloadingText(route.tab));
      return;
    }
    setBusy(true);
    data.refresh()
      .then(r => announce(r.ok ? refreshDoneText(r.at) : r.shownAt !== null ? refreshFailedText(r.shownAt) : REFRESH_FAILED_TEXT))
      .catch(() => announce(REFRESH_FAILED_TEXT))
      .finally(() => setBusy(false));
  }, [disabledReason, busy, data, announce, route.tab]);

  const navValue = useMemo<Nav>(() => ({
    state: shown,
    base: router.base,
    tabs,
    push: next => router.push(next),
    replace: next => router.replace(next),
    openSheet: sheet => openSheet(router, sheet),
    closeSheet: () => closeSheet(router),
  }), [router, shown, tabs]);
  const focusScreen = useCallback(() => {
    focusHeading(domOf(contentRef));
  }, []);

  const wide = width >= LAYOUT.railFrom;
  const items = useMemo(() => tabs.map(key => ({ key, label: ROUTES[key].tab, icon: ROUTES[key].icon })), [tabs]);
  const sub = hasSubState(shown);
  const header = renderHeader({
    title: sub ? headerTitle(shown) : undefined,
    showBack: sub,
    onBack: () => headerBack(router),
    onHome,
    // 헤더 아래 선은 셸 래퍼가 화면 끝까지 그린다(가로 모드 노치에서 헤더 바탕·선이 안전 영역까지 가게, round 1 접근성).
    palette: { background: c.surface, border: 'transparent', wordmark: c.text1, title: c.text1, version: c.text3, icon: c.text2 },
    fonts: { wordmark: textStyle('title', width, '700'), title: textStyle('body', width, '700'), version: textStyle('meta', width, '500') },
    refresh: { onPress: onRefresh, busy, disabledReason },
  });
  const page = renderRoute(shown);
  const panelWide = width >= LAYOUT.panelFrom;
  const panel = panelWide && renderPanel ? renderPanel(shown) : null;
  // 안전 영역(가로 모드 노치)은 셸 전체가 아니라 헤더·띠·본문·탭 막대 안쪽에 준다: 바탕은 화면 끝까지 가고 글자만 안으로 든다(round 1).
  return (
    <NavContext.Provider value={navValue}>
      <DataContext.Provider value={setData}>
        <View style={[styles.root, { backgroundColor: c.canvas }]}>
          <View ref={headerRef} style={{ backgroundColor: c.surface }} {...web({ dataSet: { pdSafe: 'top' } })}>
            <View {...web({ dataSet: { pdSafe: 'x' } })}>{header}</View>
            <View style={[styles.headerRule, { backgroundColor: c.border }]} />
          </View>
          {online ? null : <OfflineBanner text={offlineText(data ? data.fetchedAt : null)} />}
          <View ref={bodyRef} style={styles.body}>
            {wide ? <Rail items={items} active={shown.key} onSelect={onTab} /> : null}
            <View ref={contentRef} style={styles.content} {...web({ dataSet: { pdSafe: 'x' } })}>
              {/* 화면(하위 상태 포함)마다 새로 그린다: 새 화면은 맨 위에서 시작하고 스크롤은 뒤로·앞으로 때만 되돌린다. */}
              <ErrorBoundary key={`${id}:${reloads}`}>
                {legacy.includes(shown.key) ? <LegacyFrame title={route.heading}>{page}</LegacyFrame> : page}
              </ErrorBoundary>
            </View>
            {panel ? <ReadingPanel title={panel.title}>{panel.body}</ReadingPanel> : null}
            {/* 패널이 없는 화면(알아보기)도 ≥ 1280에서 본문 열이 같은 x에서 시작한다(round 1 D14: 탭을 바꾸면 160 px 뛰었다). */}
            {panelWide && !panel ? <View style={styles.panelSpace} testID="panel-space" /> : null}
          </View>
          {wide ? null : <View ref={tabsRef}><TabBar items={items} active={shown.key} onSelect={onTab} /></View>}
          <SheetHost sheet={shown.sheet} registry={registry} onClose={navValue.closeSheet} onNoOpener={focusScreen} />
        </View>
      </DataContext.Provider>
    </NavContext.Provider>
  );
}

/** 자기 h1이 없는 레거시 화면을 새 화면(F7–F9)이 올 때까지 감싼다: main 랜드마크 + h1(tabIndex=-1) + NowPill, 화면은 아래 칸. */
function LegacyFrame({ title, children }: { title: string; children: React.ReactNode }) {
  const { width } = useWindowDimensions();
  const gutter = width < LAYOUT.narrowBelow ? LAYOUT.gutterNarrow : LAYOUT.gutter;
  return (
    <View role="main" style={styles.legacy} testID="legacy-frame">
      <View style={[styles.legacyHead, { paddingHorizontal: gutter }]}>
        <Text role="heading" {...web({ 'aria-level': 1, tabIndex: -1 })} variant="title" style={styles.h1}>{title}</Text>
        <ClockNowPill />
      </View>
      <View style={styles.legacyBody}>{children}</View>
    </View>
  );
}

/** ≥ 1280 오른쪽 패널(320): complementary 랜드마크, 제목 h2, 자기 스크롤. 시트가 열리면 이 자리를 덮는다(Sheet가 오른쪽 400). */
function ReadingPanel({ title, children }: { title: string; children: React.ReactNode }) {
  const { theme } = useTheme();
  const titleId = `pd-panel-${useId().replace(/[^A-Za-z0-9_-]/g, '')}`;
  return (
    <View role="complementary" aria-labelledby={titleId} testID="reading-panel"
      style={[styles.panel, { backgroundColor: theme.color.surface, borderLeftColor: theme.color.border }]}>
      <ScrollView contentContainerStyle={styles.panelContent}>
        <Text role="heading" {...web({ 'aria-level': 2 })} nativeID={titleId} variant="body" weight="700">{title}</Text>
        {children}
      </ScrollView>
    </View>
  );
}

/** 셸이 조회 시각을 모르는 화면의 NowPill: KST 시각만(정산·알아보기와 같은 꼴). 60초마다 글자만 바뀌고 알리지 않는다. */
function ClockNowPill() {
  const { now, skewNotable } = useClock();
  return <NowPill now={now} fetchedAt={null} skewNotable={skewNotable} />;
}

const styles = StyleSheet.create({
  root: { flex: 1, width: '100%', minWidth: 0, minHeight: 0 },
  body: { flex: 1, flexDirection: 'row', minHeight: 0 },
  content: { flex: 1, minWidth: 0, minHeight: 0 },
  legacy: { flex: 1, width: '100%', maxWidth: LAYOUT.column, alignSelf: 'center', minHeight: 0 },
  legacyHead: {
    flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: SPACE.sm,
    paddingTop: SPACE.lg, paddingBottom: SPACE.sm,
  },
  h1: { flexShrink: 1 },
  legacyBody: { flex: 1, minHeight: 0 },
  panel: { width: LAYOUT.panel, borderLeftWidth: 1, minHeight: 0 },
  panelSpace: { width: LAYOUT.panel },
  headerRule: { position: 'absolute', left: 0, right: 0, bottom: 0, height: 1 },
  panelContent: { padding: SPACE.lg, gap: SPACE.md },
});
