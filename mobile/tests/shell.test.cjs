'use strict';
// 셸 VM 렌더 테스트(명세 3.4절, F6). Shell.tsx를 ts-loader로 불러 메모리 라우터를 넣고 React 목으로 한 번 그린 뒤 트리를 본다:
// 폭별 탭 막대·레일, 안전 영역 표지, 레거시 화면을 감싼 main + h1 + NowPill, 새로고침 비활성 까닭, 쓰지 않는 탭·등록되지 않은 시트
// 제한, 테마 시트, 하위 상태 헤더, 탭·워드마크 누름. 효과(초점·알림·popstate)는 브라우저 점검(eval/steps/F6)이 따로 확인한다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, defaultMocks, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const SVG = { __esModule: true, default: 'Svg', Path: 'Path', Circle: 'Circle' };
function load(width = 390) {
  const rn = { ...defaultMocks()['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG } });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  return { ...at('app/Shell.tsx'), routes: at('app/routes.ts'), memory: at('app/router.ts'), strings: at('content/strings.ts') };
}
const h = (type, props, ...children) => ({ type, props: { ...(props || {}), children } });
function render(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return [];
  if (Array.isArray(node)) return node.flatMap(render);
  if (typeof node !== 'object') return [String(node)];
  const { type, props } = node;
  if (typeof type === 'function') {
    if (type.prototype && typeof type.prototype.render === 'function') {
      const instance = new type(props);
      instance.props = props;
      return render(instance.render());
    }
    return render(type(props));
  }
  const { children, ...rest } = props;
  return [{ type, props: rest, children: render(children) }];
}
function all(node, pred, out = []) {
  if (node && typeof node === 'object') {
    if (pred(node)) out.push(node);
    node.children.forEach(child => all(child, pred, out));
  }
  return out;
}
const textOf = node => (typeof node === 'string' ? node : node.children.map(textOf).join(''));
const one = (root, pred, what) => {
  const hits = all(root, pred);
  assert.equal(hits.length, 1, `${what}: ${hits.length}`);
  return hits[0];
};

// F6의 App.tsx와 같은 구성: 탭 3개, 세 화면 모두 레거시.
function shell(url, { width = 390, tabs = ['smp', 'bidding', 'settlement'], legacy = ['smp', 'bidding', 'settlement'], renderPanel } = {}) {
  const m = load(width);
  const router = m.memory.createRouter({ initialUrl: url, base: '/mobile/' });
  const seen = { header: null, routes: [] };
  const element = h(m.Shell, {
    tabs, legacy, router, renderPanel,
    renderRoute: state => {
      seen.routes.push(state);
      return h('Page', { name: state.key });
    },
    renderHeader: props => {
      seen.header = props;
      return h('HeaderStub', {});
    },
  });
  return { root: { type: 'root', props: {}, children: render(element) }, seen, router, m };
}

test('390: 아래 탭 막대(3개, 지금 탭 선택), 레일 없음, 셸 바깥 x · 헤더 top · 탭 막대 bottom 안전 영역, polite 하나', () => {
  const { root } = shell('/mobile/bidding');
  const bar = one(root, n => n.props.testID === 'tab-bar', '탭 막대');
  assert.equal(all(root, n => n.props.testID === 'rail').length, 0);
  assert.deepStrictEqual(all(bar, n => n.props.role === 'tab').map(t => [t.props['aria-label'], t.props['aria-selected']]),
    [['오늘', false], ['입찰', true], ['정산', false]]);
  assert.equal(bar.props.dataSet.pdSafe, 'bottom');
  assert.equal(all(root, n => n.props.dataSet && n.props.dataSet.pdSafe === 'x').length, 1);
  const top = one(root, n => n.props.dataSet && n.props.dataSet.pdSafe === 'top', '헤더 안전 영역');
  assert.equal(top.children[0].type, 'HeaderStub');
  assert.equal(all(root, n => n.props['aria-live'] === 'polite').length, 1);
});

test('≥ 1024: 왼쪽 레일(세로, 같은 3항목), 아래 탭 막대 없음', () => {
  for (const width of [1024, 1280]) {
    const { root } = shell('/mobile/', { width });
    const rail = one(root, n => n.props.testID === 'rail', `레일 ${width}`);
    assert.equal(all(root, n => n.props.testID === 'tab-bar').length, 0);
    assert.equal(one(rail, n => n.props.role === 'tablist', 'tablist').props['aria-orientation'], 'vertical');
    assert.deepStrictEqual(all(rail, n => n.props.role === 'tab').map(t => t.props['aria-label']), ['오늘', '입찰', '정산']);
  }
});

test('≥ 1280 패널: 화면이 내용을 주면 complementary + h2 제목 + 본문, 1280 미만이거나 내용이 없으면 없음', () => {
  const panel = () => ({ title: '입찰 읽는 법', body: h('PanelBody', {}) });
  const wide = shell('/mobile/bidding', { width: 1280, renderPanel: panel });
  const aside = one(wide.root, n => n.props.testID === 'reading-panel', '패널');
  assert.equal(aside.props.role, 'complementary');
  const title = one(aside, n => n.props.role === 'heading', '패널 제목');
  assert.deepStrictEqual([textOf(title), title.props['aria-level'], aside.props['aria-labelledby']], ['입찰 읽는 법', 2, title.props.nativeID]);
  one(aside, n => n.type === 'PanelBody', '패널 본문');
  for (const [width, render] of [[1279, panel], [1280, () => null], [1280, undefined]]) {
    assert.equal(all(shell('/mobile/bidding', { width, renderPanel: render }).root, n => n.props.testID === 'reading-panel').length, 0, `${width}`);
  }
});

test('레거시 화면: 셸이 main 하나 + h1(tabIndex=-1, 경로 제목) + NowPill(조회 없음)으로 감싸고, 새 화면은 그대로 둔다', () => {
  for (const [url, heading] of [['/mobile/', '오늘 제주 전력시장'], ['/mobile/bidding', '입찰'], ['/mobile/settlement', '정산']]) {
    const { root } = shell(url);
    const frame = one(root, n => n.props.testID === 'legacy-frame', `레거시 틀 ${url}`);
    assert.equal(frame.props.role, 'main');
    const h1 = one(frame, n => n.props.role === 'heading', 'h1');
    assert.deepStrictEqual([textOf(h1), h1.props['aria-level'], h1.props.tabIndex], [heading, 1, -1]);
    assert.equal(all(frame, n => n.props.testID === 'now-pill').length, 1);
    assert.match(textOf(one(frame, n => n.props.testID === 'now-pill', 'NowPill')), /^\d{2}:\d{2} KST$/, '조회 시각을 모르면 시각만');
    one(frame, n => n.type === 'Page', '화면 본문은 틀 안');
  }
  const { root } = shell('/mobile/', { legacy: ['bidding', 'settlement'] });
  assert.equal(all(root, n => n.props.testID === 'legacy-frame').length, 0, 'F7 이후 오늘은 Screen이 h1 행을 그린다');
  one(root, n => n.type === 'Page', '화면 본문');
});

test('헤더: 모든 탭에 새로고침, 정산은 숨기지 않고 비활성(까닭), 테마 팔레트와 토큰 글자 꼴을 넘긴다', () => {
  for (const [url, reason] of [['/mobile/', null], ['/mobile/bidding', null], ['/mobile/settlement', '이 화면은 새로 불러올 자료가 없습니다']]) {
    const { seen } = shell(url);
    assert.equal(seen.header.refresh.disabledReason, reason, url);
    assert.equal(seen.header.showBack, false);
    assert.equal(seen.header.title, undefined);
  }
  const { seen, m } = shell('/mobile/');
  assert.equal(m.strings.REFRESH_NONE_REASON, '이 화면은 새로 불러올 자료가 없습니다');
  assert.deepStrictEqual(Object.keys(seen.header.palette).sort(), ['background', 'border', 'icon', 'title', 'version', 'wordmark']);
  assert.deepStrictEqual([seen.header.fonts.wordmark.fontSize, seen.header.fonts.title.fontSize, seen.header.fonts.version.fontSize], [20, 15, 12]);
});

test('제한: 쓰지 않는 탭(F6의 알아보기)은 오늘로, 등록되지 않은 시트는 열지 않는다. 테마 시트는 셸이 늘 가진다', () => {
  const guide = shell('/mobile/guide?tour=2');
  assert.deepStrictEqual(guide.seen.routes.map(s => s.key), ['smp']);
  one(guide.root, n => n.props.testID === 'legacy-frame', '오늘 틀');
  const sources = shell('/mobile/bidding?sheet=sources');
  assert.equal(sources.seen.routes[0].sheet, null);
  assert.equal(all(sources.root, n => n.props.role === 'dialog').length, 0);
  const theme = shell('/mobile/bidding?sheet=theme');
  const dialog = one(theme.root, n => n.props.role === 'dialog', '테마 시트');
  assert.equal(dialog.props['aria-modal'], true);
  assert.equal(textOf(one(dialog, n => n.props.role === 'heading', '시트 제목')), '화면 테마');
  const group = one(dialog, n => n.props.role === 'radiogroup', 'radiogroup');
  assert.deepStrictEqual(all(group, n => n.props.role === 'radio').map(r => [r.props['aria-label'], r.props['aria-checked']]),
    [['시스템 설정 따름 (기본)', true], ['밝게', false], ['어둡게', false]]);
  assert.match(textOf(dialog), /이 기기에만 저장됩니다/);
});

test('하위 상태 헤더: ← + 제목(알아보기 탭을 쓰는 F10 구성), 첫 화면에는 ← 없음', () => {
  const tabs = ['smp', 'bidding', 'settlement', 'guide'];
  assert.deepStrictEqual([shell('/mobile/guide?topic=glossary', { tabs }).seen.header.showBack, shell('/mobile/guide?topic=glossary', { tabs }).seen.header.title],
    [true, '용어 사전']);
  assert.equal(shell('/mobile/guide?tour=3', { tabs }).seen.header.title, '1분 둘러보기');
  assert.equal(shell('/mobile/guide', { tabs }).seen.header.showBack, false);
});

test('누름: 지금 탭은 히스토리 없음, 다른 탭은 push, 워드마크는 오늘로(이미 오늘이면 push 없음)', () => {
  const { root, router, seen } = shell('/mobile/bidding');
  const tab = name => one(root, n => n.props.role === 'tab' && n.props['aria-label'] === name, name);
  tab('입찰').props.onPress();
  assert.deepStrictEqual(router.entry(), { pushed: false, sheet: false });
  tab('정산').props.onPress();
  assert.deepStrictEqual([router.current().key, router.entry().pushed], ['settlement', true]);
  seen.header.onHome();
  assert.equal(router.current().key, 'smp');
  const today = shell('/mobile/?hour=8');
  today.seen.header.onHome();
  assert.deepStrictEqual([today.router.current().hour, today.router.entry().pushed], [8, false]);
});
