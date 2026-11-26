'use strict';
// 라우터 계약 테스트(명세 3.1–3.3절, F6). routes.ts의 순수 함수(기준 경로, URL 읽기·만들기, 쓰지 않는 탭·시트 제한, 하위 상태,
// 제목·알림 글자)와 두 라우터를 같은 시나리오로 돌린다: router.ts(메모리 스택)와 router.web.ts(History API, 가짜 window).
// 가짜 window의 history.back()은 브라우저처럼 늦게 돈다(popstate는 settle() 때) — Esc 두 번 같은 경합을 그대로 재현한다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const routes = createTsLoader().load(path.join(SRC_ROOT, 'app/routes.ts'));
const { buildUrl, parseLocation, detectBase, restrict, rootState: root } = routes;
const APP = 'eXeco 제주 전력시장';
const split = url => {
  const at = url.indexOf('?');
  return at < 0 ? [url, ''] : [url.slice(0, at), url.slice(at)];
};
const parse = (url, base = '/mobile/') => parseLocation(...split(url), base);

test('detectBase: 번들 script src의 _expo/ 앞부분, 없으면 /mobile·/mobile/…만 /mobile/ (/mobile-x는 아님)', () => {
  assert.equal(detectBase('/mobile/_expo/static/js/web/AppEntry-1.js', '/mobile/bidding'), '/mobile/');
  assert.equal(detectBase('http://127.0.0.1:10104/mobile/_expo/static/js/web/AppEntry-1.js', '/x'), '/mobile/');
  assert.equal(detectBase('/_expo/static/js/web/AppEntry-1.js', '/bidding'), '/');
  assert.equal(detectBase('https://host:10000/_expo/static/js/web/AppEntry-1.js', '/mobile/'), '/');
  for (const [pathname, want] of [['/mobile', '/mobile/'], ['/mobile/', '/mobile/'], ['/mobile/settlement', '/mobile/'], ['/mobile-x', '/'], ['/', '/']]) {
    assert.equal(detectBase(null, pathname), want, pathname);
  }
  // 엑스포 번들이 아니거나 상대 경로 src면 pathname으로 판단한다.
  assert.equal(detectBase('/static/js/main.js', '/mobile/x'), '/mobile/');
  assert.equal(detectBase('_expo/static/js/web/a.js', '/'), '/');
});

test('parseLocation 경로: 4개 경로는 정규형, 모르는·깊은 경로와 끝 슬래시 없는 /mobile은 오늘(바꿔 써야 함)', () => {
  for (const [url, key] of [['/mobile/', 'smp'], ['/mobile/bidding', 'bidding'], ['/mobile/settlement', 'settlement'], ['/mobile/guide', 'guide']]) {
    assert.deepStrictEqual(parse(url), { state: root(key), canonical: true }, url);
  }
  for (const url of ['/mobile', '/mobile/foo', '/mobile/bidding/', '/mobile/bidding/x', '/mobile/index.html', '/mobile//bidding', '/mobile-x', '/']) {
    assert.deepStrictEqual(parse(url), { state: root('smp'), canonical: false }, url);
  }
  assert.deepStrictEqual(parse('/mobile/foo?hour=8&sheet=theme'), { state: root('smp'), canonical: false }, '모르는 경로는 쿼리도 버린다');
  // 단일 앱 모드(기준 경로 '/')
  assert.deepStrictEqual(parse('/', '/'), { state: root('smp'), canonical: true });
  assert.deepStrictEqual(parse('/bidding?hour=3', '/'), { state: { ...root('bidding'), hour: 3 }, canonical: true });
  assert.equal(parse('/mobile/', '/').canonical, false);
});

test('parseLocation 쿼리: 경로마다 쓰는 값만, 틀린 값은 기본값(정규형 아님), 같은 이름은 첫 값, 깨진 인코딩은 버림', () => {
  assert.deepStrictEqual(parse('/mobile/?hour=8'), { state: { ...root('smp'), hour: 8 }, canonical: true });
  assert.deepStrictEqual(parse('/mobile/?hour=08'), { state: { ...root('smp'), hour: 8 }, canonical: false });
  for (const bad of ['0', '25', '1.5', 'abc', '-1', '', '100', '+3']) {
    assert.deepStrictEqual(parse(`/mobile/?hour=${bad}`), { state: root('smp'), canonical: false }, `hour=${bad}`);
  }
  assert.deepStrictEqual(parse('/mobile/bidding?hour=19&risk=moderate'), { state: { ...root('bidding'), hour: 19, risk: 'moderate' }, canonical: true });
  assert.equal(parse('/mobile/bidding?risk=moderate&hour=19').canonical, false, '정규형 순서는 hour 다음 risk');
  assert.deepStrictEqual(parse('/mobile/bidding?risk=AGGRESSIVE'), { state: root('bidding'), canonical: false });
  assert.deepStrictEqual(parse('/mobile/?risk=moderate'), { state: root('smp'), canonical: false }, '오늘에는 성향이 없다');
  assert.deepStrictEqual(parse('/mobile/settlement?hour=3'), { state: root('settlement'), canonical: false }, '정산에는 거래시간이 없다');
  assert.deepStrictEqual(parse('/mobile/guide?tour=5'), { state: { ...root('guide'), tour: 5 }, canonical: true });
  for (const bad of ['0', '6']) assert.deepStrictEqual(parse(`/mobile/guide?tour=${bad}`).state, root('guide'));
  assert.deepStrictEqual(parse('/mobile/guide?term=smp').state, { ...root('guide'), term: 'smp' });
  assert.deepStrictEqual(parse('/mobile/guide?topic=sources').state, { ...root('guide'), topic: 'sources' });
  assert.deepStrictEqual(parse('/mobile/guide?tour=2&topic=glossary'), { state: { ...root('guide'), tour: 2 }, canonical: false }, '하위 상태는 하나');
  assert.deepStrictEqual(parse('/mobile/guide?term=smp&topic=glossary'), { state: { ...root('guide'), term: 'smp' }, canonical: false });
  assert.deepStrictEqual(parse('/mobile/guide?term=nope').state, root('guide'));
  assert.deepStrictEqual(parse('/mobile/?hour=8&hour=9'), { state: { ...root('smp'), hour: 8 }, canonical: false });
  assert.deepStrictEqual(parse('/mobile/?hour=%E0&x=1'), { state: root('smp'), canonical: false });
  assert.deepStrictEqual(parse('/mobile/?utm_source=x'), { state: root('smp'), canonical: false });
});

test('parseLocation 시트: method는 카드 8개 중 하나, sources의 row는 선택, term은 용어 12개, theme; 그 밖은 시트 없음', () => {
  const sheet = url => parse(url).state.sheet;
  assert.deepStrictEqual(sheet('/mobile/?sheet=method&card=today-price'), { kind: 'method', card: 'today-price' });
  assert.equal(sheet('/mobile/?sheet=method'), null);
  assert.equal(sheet('/mobile/?sheet=method&card=today-nope'), null);
  assert.deepStrictEqual(sheet('/mobile/bidding?sheet=sources'), { kind: 'sources', row: null });
  assert.deepStrictEqual(sheet('/mobile/bidding?sheet=sources&row=status'), { kind: 'sources', row: 'status' });
  assert.deepStrictEqual(parse('/mobile/?sheet=sources&row=x'), { state: { ...root('smp'), sheet: { kind: 'sources', row: null } }, canonical: false });
  assert.deepStrictEqual(sheet('/mobile/settlement?sheet=term&id=settlement'), { kind: 'term', id: 'settlement' });
  assert.equal(sheet('/mobile/?sheet=term&id=zzz'), null);
  assert.deepStrictEqual(sheet('/mobile/guide?sheet=theme'), { kind: 'theme' });
  assert.equal(sheet('/mobile/?sheet=alan'), null);
  assert.equal(routes.METHOD_CARDS.length, 8);
  assert.equal(routes.SOURCE_ROWS.length, 9);
});

test('buildUrl: 정규형 순서, parse(build(s)) = s, 경로가 쓰지 않는 값은 버림', () => {
  const states = [
    root('smp'), { ...root('smp'), hour: 24 }, { ...root('bidding'), hour: 1, risk: 'aggressive' }, root('settlement'),
    { ...root('guide'), tour: 5 }, { ...root('guide'), term: 'trade-hour' }, { ...root('guide'), topic: 'about' },
    { ...root('smp'), sheet: { kind: 'method', card: 'settle-sample' } }, { ...root('bidding'), hour: 7, sheet: { kind: 'sources', row: 'status' } },
    { ...root('guide'), topic: 'glossary', sheet: { kind: 'term', id: 'band' } }, { ...root('settlement'), sheet: { kind: 'theme' } },
  ];
  for (const s of states) {
    const url = buildUrl(s, '/mobile/');
    assert.deepStrictEqual(parse(url), { state: s, canonical: true }, url);
  }
  assert.equal(buildUrl({ ...root('bidding'), hour: 19, risk: 'moderate', sheet: { kind: 'method', card: 'bid-ladder' } }, '/mobile/'),
    '/mobile/bidding?hour=19&risk=moderate&sheet=method&card=bid-ladder');
  assert.equal(buildUrl({ ...root('settlement'), hour: 3, risk: 'moderate', tour: 2 }, '/mobile/'), '/mobile/settlement');
  assert.equal(buildUrl({ ...root('smp'), hour: 0 }, '/mobile/'), '/mobile/');
  assert.equal(buildUrl({ ...root('guide'), tour: 2, term: 'smp', topic: 'reading' }, '/mobile/'), '/mobile/guide?tour=2');
  assert.equal(buildUrl(root('smp'), '/'), '/');
  assert.equal(buildUrl(root('guide'), '/'), '/guide');
  assert.equal(buildUrl(root('bidding'), 'mobile'), '/mobile/bidding', '기준 경로 모양을 고친다');
});

test('restrict·하위 상태·제목: 쓰지 않는 탭은 첫 탭, 등록되지 않은 시트는 없음, 문서 제목·알림·헤더 제목·부모', () => {
  const tabs = ['smp', 'bidding', 'settlement'];
  assert.deepStrictEqual(restrict({ ...root('guide'), tour: 2 }, tabs, ['theme']), root('smp'));
  const sources = { ...root('bidding'), hour: 3, sheet: { kind: 'sources', row: null } };
  assert.deepStrictEqual(restrict(sources, tabs, ['theme']), { ...sources, sheet: null });
  const ok = { ...root('bidding'), sheet: { kind: 'theme' } };
  assert.equal(restrict(ok, tabs, ['theme']), ok, '바꿀 것이 없으면 같은 객체(효과가 다시 돌지 않는다)');

  const title = s => routes.documentTitle(s);
  assert.equal(title(root('smp')), `오늘 · ${APP}`);
  assert.equal(title({ ...root('bidding'), hour: 19, sheet: { kind: 'theme' } }), `입찰 · ${APP}`);
  assert.equal(title(root('settlement')), `정산 · ${APP}`);
  assert.equal(title(root('guide')), `알아보기 · ${APP}`);
  assert.equal(title({ ...root('guide'), tour: 2 }), `둘러보기 2/5 · ${APP}`);
  assert.equal(title({ ...root('guide'), topic: 'glossary' }), `용어 사전 · ${APP}`);
  assert.equal(title({ ...root('guide'), term: 'smp' }), `SMP(계통한계가격) · ${APP}`);
  assert.equal(routes.routeAnnouncement(root('bidding')), '입찰 화면');
  assert.equal(routes.routeAnnouncement({ ...root('guide'), tour: 3 }), '둘러보기 3/5 화면');

  assert.equal(routes.hasSubState(root('guide')), false);
  assert.equal(routes.hasSubState({ ...root('smp'), hour: 8, sheet: { kind: 'theme' } }), false, '거래시간·시트는 하위 상태가 아니다');
  for (const sub of [{ tour: 1 }, { term: 'smp' }, { topic: 'about' }]) assert.equal(routes.hasSubState({ ...root('guide'), ...sub }), true);
  assert.deepStrictEqual(routes.parentOf({ ...root('guide'), term: 'smp' }), { ...root('guide'), topic: 'glossary' });
  assert.deepStrictEqual(routes.parentOf({ ...root('guide'), tour: 3, sheet: { kind: 'theme' } }), root('guide'));
  assert.equal(routes.headerTitle({ ...root('guide'), tour: 4 }), '1분 둘러보기');
  assert.equal(routes.headerTitle({ ...root('guide'), topic: 'sources' }), '데이터 출처와 연결 상태');
  assert.equal(routes.screenId({ ...root('smp'), hour: 8, sheet: { kind: 'theme' } }), routes.screenId(root('smp')));
  assert.notEqual(routes.screenId({ ...root('guide'), tour: 1 }), routes.screenId({ ...root('guide'), tour: 2 }));

  const { ROUTES } = routes;
  assert.deepStrictEqual(Object.values(ROUTES).map(r => [r.key, r.path, r.tab, r.icon, r.refreshable]), [
    ['smp', '', '오늘', 'chart', true], ['bidding', 'bidding', '입찰', 'clock', true],
    ['settlement', 'settlement', '정산', 'list', false], ['guide', 'guide', '알아보기', 'book', false],
  ]);
  assert.equal(ROUTES.smp.heading, '오늘 제주 전력시장');
});

// ---------------------------------------------------------------- 두 라우터, 같은 규칙
function fakeWindow(url, script = '/mobile/_expo/static/js/web/AppEntry-abc.js') {
  const entries = [{ url, state: null }];
  let index = 0;
  const queue = [];
  const listeners = new Set();
  const clone = value => (value === undefined ? null : JSON.parse(JSON.stringify(value)));
  const host = {
    location: { get pathname() { return split(entries[index].url)[0]; }, get search() { return split(entries[index].url)[1]; } },
    history: {
      get state() { return entries[index].state; },
      pushState(state, _unused, next) { entries.splice(index + 1); entries.push({ url: next, state: clone(state) }); index = entries.length - 1; },
      replaceState(state, _unused, next) { entries[index] = { url: next, state: clone(state) }; },
      // 브라우저처럼 늦게: 위치와 state는 settle() 때 바뀌고 popstate가 그때 온다.
      back() { queue.push(() => { if (index > 0) { index -= 1; listeners.forEach(fn => fn()); } }); },
    },
    addEventListener: (_type, fn) => listeners.add(fn),
    removeEventListener: (_type, fn) => listeners.delete(fn),
    document: { querySelector: () => (script ? { getAttribute: () => script } : null) },
  };
  return { host, entries, listeners, settle: () => { while (queue.length) queue.shift()(); }, at: () => entries[index] };
}

const ROUTERS = {
  memory: url => {
    const router = createTsLoader().load(path.join(SRC_ROOT, 'app/router.ts')).createRouter({ initialUrl: url, base: '/mobile/' });
    return { router, settle: () => undefined, depth: null };
  },
  web: url => {
    const win = fakeWindow(url);
    const router = createTsLoader().load(path.join(SRC_ROOT, 'app/router.web.ts')).createRouter({ host: win.host });
    return { router, settle: win.settle, depth: () => win.entries.length, win };
  },
};
const urlOf = router => buildUrl(router.current(), router.base);

for (const [name, make] of Object.entries(ROUTERS)) {
  test(`${name} 라우터: 탭 다시 누름은 히스토리 없음, 다른 탭은 새 항목, 뒤로는 이전 탭`, () => {
    const { router, settle } = make('/mobile/bidding');
    router.start();
    const seen = [];
    router.subscribe((state, kind) => seen.push(`${kind} ${buildUrl(state, router.base)}`));
    assert.equal(router.base, '/mobile/');
    assert.deepStrictEqual(router.entry(), { pushed: false, sheet: false });
    assert.equal(routes.pressTab(router, 'bidding'), 'top');
    assert.deepStrictEqual(seen, [], '지금 탭은 히스토리를 건드리지 않는다');
    assert.equal(routes.pressTab(router, 'settlement'), 'pushed');
    assert.deepStrictEqual(router.entry(), { pushed: true, sheet: false });
    router.back();
    settle();
    assert.equal(urlOf(router), '/mobile/bidding');
    assert.deepStrictEqual(seen, ['push /mobile/settlement', 'pop /mobile/bidding']);
  });

  test(`${name} 라우터: 하위 상태(둘러보기·용어·주제)에서 지금 탭을 다시 누르면 그 탭의 첫 화면(새 항목), 첫 화면에서는 맨 위만`, () => {
    for (const sub of ['/mobile/guide?tour=3', '/mobile/guide?term=smp', '/mobile/guide?topic=sources']) {
      const { router, settle } = make(sub);
      router.start();
      const seen = [];
      router.subscribe((state, kind) => seen.push(`${kind} ${buildUrl(state, router.base)}`));
      assert.equal(routes.pressTab(router, 'guide'), 'pushed', sub);
      assert.deepStrictEqual(seen, ['push /mobile/guide'], sub);
      assert.equal(routes.pressTab(router, 'guide'), 'top', `${sub}: 첫 화면에서 다시 누르면 맨 위로만`);
      router.back();
      settle();
      assert.equal(urlOf(router), sub, '뒤로는 하위 상태로 돌아간다');
    }
    // 오늘·입찰의 거래시간·성향 선택은 하위 상태가 아니다: 다시 눌러도 히스토리 없음(G2).
    const { router } = make('/mobile/bidding?hour=19&risk=aggressive');
    router.start();
    assert.equal(routes.pressTab(router, 'bidding'), 'top');
    assert.equal(urlOf(router), '/mobile/bidding?hour=19&risk=aggressive');
  });

  test(`${name} 라우터: 연 시트는 뒤로로 닫힘(탭 그대로), Esc 두 번도 한 번만 뒤로, 딥링크 시트는 쿼리만 뺌`, () => {
    const { router, settle } = make('/mobile/bidding');
    router.start();
    routes.pressTab(router, 'settlement');
    routes.openSheet(router, { kind: 'theme' });
    assert.equal(urlOf(router), '/mobile/settlement?sheet=theme');
    assert.deepStrictEqual(router.entry(), { pushed: true, sheet: true });
    routes.closeSheet(router);
    routes.closeSheet(router);
    settle();
    assert.equal(urlOf(router), '/mobile/settlement', '시트만 닫히고 탭(정산)은 그대로');
    assert.deepStrictEqual(router.entry(), { pushed: true, sheet: false });
    router.back();
    settle();
    assert.equal(urlOf(router), '/mobile/bidding');

    const deep = make('/mobile/?hour=8&sheet=theme');
    deep.router.start();
    routes.closeSheet(deep.router);
    deep.settle();
    assert.equal(urlOf(deep.router), '/mobile/?hour=8');
    assert.deepStrictEqual(deep.router.entry(), { pushed: false, sheet: false });
    if (deep.depth) assert.equal(deep.depth(), 1, 'replaceState: 히스토리 길이 그대로');
  });

  test(`${name} 라우터: 헤더 ←는 우리가 넣은 하위 상태면 뒤로, 딥링크면 부모로 바꿔 씀(앱 밖으로 안 나감)`, () => {
    const { router, settle } = make('/mobile/guide');
    router.start();
    router.push({ ...root('guide'), term: 'smp' });
    routes.headerBack(router);
    settle();
    assert.equal(urlOf(router), '/mobile/guide');

    const deep = make('/mobile/guide?term=smp');
    deep.router.start();
    routes.headerBack(deep.router);
    deep.settle();
    assert.equal(urlOf(deep.router), '/mobile/guide?topic=glossary');
    assert.deepStrictEqual(deep.router.entry(), { pushed: false, sheet: false });
    routes.headerBack(deep.router);
    deep.settle();
    assert.equal(urlOf(deep.router), '/mobile/guide');
    if (deep.depth) assert.equal(deep.depth(), 1);
  });

  test(`${name} 라우터: 정규형이 아닌 첫 URL은 바로잡고, push·replace는 값을 정리한다`, () => {
    const { router, win } = make('/mobile/foo?hour=8');
    router.start();
    assert.equal(urlOf(router), '/mobile/');
    if (win) assert.deepStrictEqual(win.at(), { url: '/mobile/', state: null });
    router.push({ ...root('settlement'), hour: 5 });
    assert.equal(urlOf(router), '/mobile/settlement');
    router.replace({ ...root('settlement'), sheet: { kind: 'theme' } });
    assert.equal(urlOf(router), '/mobile/settlement?sheet=theme');
    assert.deepStrictEqual(router.entry(), { pushed: true, sheet: false }, 'replace는 시트 표지를 새로 달지 않는다');
  });
}

test('web 라우터: history.state 표지({pd:1}·{pd:1,pdSheet:true}), 새로고침 뒤에도 같은 규칙, popstate 구독·해제, 기준 경로 감지', () => {
  const win = fakeWindow('/mobile/');
  const make = () => createTsLoader().load(path.join(SRC_ROOT, 'app/router.web.ts')).createRouter({ host: win.host });
  const router = make();
  const stop = router.start();
  assert.equal(win.listeners.size, 1);
  routes.pressTab(router, 'bidding');
  assert.deepStrictEqual(win.at(), { url: '/mobile/bidding', state: { pd: 1 } });
  routes.openSheet(router, { kind: 'theme' });
  assert.deepStrictEqual(win.at(), { url: '/mobile/bidding?sheet=theme', state: { pd: 1, pdSheet: true } });
  stop();
  assert.equal(win.listeners.size, 0);
  // 새로고침: 같은 항목(state 유지)에서 새 라우터가 시작해도 닫기는 뒤로다.
  const reloaded = make();
  reloaded.start();
  assert.deepStrictEqual(reloaded.current().sheet, { kind: 'theme' });
  routes.closeSheet(reloaded);
  win.settle();
  assert.deepStrictEqual(win.at(), { url: '/mobile/bidding', state: { pd: 1 } });
  assert.equal(urlOf(reloaded), '/mobile/bidding');

  const loader = createTsLoader().load(path.join(SRC_ROOT, 'app/router.web.ts'));
  assert.equal(loader.createRouter({ host: fakeWindow('/bidding', '/_expo/static/js/web/a.js').host }).base, '/');
  assert.equal(loader.createRouter({ host: fakeWindow('/mobile', null).host }).base, '/mobile/');
  assert.equal(loader.createRouter({ host: fakeWindow('/mobile-x', null).host }).base, '/');
  assert.throws(() => loader.createRouter(), /window is not available/, 'window가 없으면 바로 실패한다');
});
