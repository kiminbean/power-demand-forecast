'use strict';
// 기본 컴포넌트 VM 렌더 테스트(명세 6.2절, F4). src/ui/*.tsx를 ts-loader로 불러 React 목으로 한 번 그린 뒤
// 호스트 요소 트리(View·Text·Pressable·ScrollView)의 prop을 검사한다: 역할·이름·aria·data-pd-*·44 px 상자·키보드 이동.
// 실제 DOM 의미(RN-web이 무엇을 내보내는지)는 브라우저 탐침(scratchpad/mobile/eval/steps/F4/probe.md)이 따로 확인한다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, defaultMocks, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const SVG = { __esModule: true, default: 'Svg', Path: 'Path', Circle: 'Circle' };
function load(width = 390) {
  const rn = { ...defaultMocks()['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG } });
  const ui = name => loader.load(path.join(SRC_ROOT, 'ui', name));
  return {
    ...ui('Button.tsx'), ...ui('Chips.tsx'), ...ui('Card.tsx'), ...ui('Values.tsx'), ...ui('Controls.tsx'), ...ui('TabBar.tsx'),
    ...ui('Sheet.tsx'), ...ui('DataTable.tsx'), ...ui('Screen.tsx'), ...ui('Live.tsx'), ...ui('Tour.tsx'),
    provenance: loader.load(path.join(SRC_ROOT, 'lib/provenance.ts')),
    format: loader.load(path.join(SRC_ROOT, 'lib/format.ts')),
  };
}
const h = (type, props, ...children) => ({ type, props: { ...(props || {}), children } });

// 함수·클래스 컴포넌트를 펼쳐 호스트 요소만 남긴다. 글자는 문자열 노드다.
function render(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return [];
  if (Array.isArray(node)) return node.flatMap(render);
  if (typeof node !== 'object') return [String(node)];
  const { type, props } = node;
  if (typeof type === 'function') {
    if (type.prototype && typeof type.prototype.render === 'function') {
      const instance = new type(props);
      instance.props = props;
      if (props.__failed) instance.state = type.getDerivedStateFromError(new Error('probe'));
      return render(instance.render());
    }
    return render(type(props));
  }
  const { children, ...rest } = props;
  return [{ type, props: rest, children: render(children) }];
}
const tree = element => ({ type: 'root', props: {}, children: render(element) });
function all(node, pred, out = []) {
  if (node && typeof node === 'object') {
    if (pred(node)) out.push(node);
    node.children.forEach(child => all(child, pred, out));
  }
  return out;
}
const textOf = node => (typeof node === 'string' ? node : node.children.map(textOf).join(''));
const flat = style => (Array.isArray(style) ? Object.assign({}, ...style.map(flat)) : style || {});
const byRole = (root, role) => all(root, n => n.props.role === role);
const pressables = root => all(root, n => n.type === 'Pressable');
const HANGUL = /[가-힣]/;

function assertTarget(node, label) {
  const s = flat(node.props.style);
  const w = Math.max(s.minWidth || 0, s.width || 0);
  const hgt = Math.max(s.minHeight || 0, s.height || 0);
  assert.ok(w >= 44 || s.flex === 1, `${label}: width box ${w}`);
  assert.ok(hgt >= 44, `${label}: height box ${hgt}`);
}

// Provenanced 브랜드는 타입에만 있으므로 실행 중에는 같은 모양의 객체로 충분하다(브랜드 검사는 tests/types/*.typecheck.ts).
const pv = (ui, value, unit, chipKey, source, state = 'measured') =>
  ({ value, unit, state, source, chip: ui.provenance.CHIPS[chipKey], time: { kind: '조회', kst: '06:04 KST' } });

const NOW = Date.UTC(2026, 10, 25, 21, 3, 57); // 2026-11-26 06:03:57 KST

test('Button·IconButton·RefreshButton: role=button, 한국어 이름, 44 px, 비활성 까닭, 진행 중 aria-busy', () => {
  const ui = load();
  const root = tree([
    h(ui.Button, { label: '다시 시도', accessibilityLabel: '다시 시도', onPress: () => {} }),
    h(ui.Button, { label: '다시 불러오기', accessibilityLabel: '다시 불러오기', tone: 'primary', busy: true, onPress: () => {} }),
    h(ui.IconButton, { icon: 'info', accessibilityLabel: '산출 방식 보기: 확정 가격' }),
    h(ui.RefreshButton, { disabledReason: '이 화면은 새로 불러올 자료가 없습니다' }),
  ]);
  const [retry, busy, info, refresh] = pressables(root);
  pressables(root).forEach((p, i) => {
    assert.equal(p.props.role, 'button');
    assert.match(p.props['aria-label'], HANGUL);
    assertTarget(p, `button ${i}`);
  });
  assert.equal(retry.props['aria-label'], '다시 시도');
  assert.equal(busy.props['aria-busy'], true);
  assert.equal(busy.props.onPress, undefined, '진행 중에는 누름을 받지 않는다');
  assert.equal(all(info, n => n.props['aria-hidden'] === true).length, 1, '아이콘은 장식(aria-hidden)');
  assert.equal(refresh.props['aria-label'], '가격·전력 정보 새로고침 · 이 화면은 새로 불러올 자료가 없습니다');
  assert.equal(refresh.props.disabled, true);
  assert.equal(refresh.props.tabIndex, 0, '까닭 있는 비활성은 Tab으로 닿는다');
});

test('ProvenanceChip: 글자 + 글리프 + 모양, role=img 긴 이름, 누를 수 없음, 지연·끊김·잠김', () => {
  const ui = load();
  const { CHIPS } = ui.provenance;
  const root = tree([
    h(ui.ProvenanceChip, { chip: CHIPS.confirmedPrice, source: 'KPX 제주 하루전시장' }),
    h(ui.ProvenanceChip, { chip: CHIPS.estNotMeasured }),
    h(ui.ProvenanceChip, { lag: 'stale', at: NOW }),
    h(ui.ProvenanceChip, { lag: 'broken', at: NOW }),
    h(ui.ProvenanceChip, { chip: CHIPS.locked }),
  ]);
  const chips = byRole(root, 'img');
  assert.equal(chips.length, 5);
  assert.equal(pressables(root).length, 0, '칩은 누를 수 없다');
  assert.equal(chips[0].props['aria-label'], '상태: 확정 가격, 출처 KPX 제주 하루전시장');
  assert.equal(textOf(chips[0]), '● 확정 가격');
  assert.equal(chips[0].props.dataSet.pdChip, 'measured');
  assert.equal(flat(chips[0].props.style).borderStyle, 'solid');
  assert.equal(chips[1].props.dataSet.pdHatch, 'est', '추정 칩은 빗금');
  assert.equal(flat(chips[1].props.style).borderStyle, 'dotted');
  assert.equal(all(chips[1], n => n.props.dataSet && n.props.dataSet.pdNowrap === 'true').map(textOf).join('|'), '실측 아님');
  assert.equal(textOf(chips[2]), '지연 · 06:03 값');
  assert.equal(chips[2].props.dataSet.pdChip, 'stale');
  assert.equal(textOf(chips[3]), '06:03 값', '끊김에서는 실측 글리프를 거둔다');
  assert.equal(textOf(chips[4]), '잠김');
  assert.equal(all(chips[4], n => n.type === 'Svg').length, 1, '잠김은 lock 아이콘');
  chips.forEach(c => assert.equal(flat(c.props.style).minHeight, 24));
});

test('NowPill·StatusPill·SampleRibbon: 단계별 글자, 색만이 아닌 표지, 예시 리본', () => {
  const ui = load();
  const MIN = 60 * 1000;
  const pills = [null, NOW - MIN, NOW - 12 * MIN, NOW - 61 * MIN].map(fetchedAt => tree(h(ui.NowPill, { now: NOW, fetchedAt })).children[0]);
  assert.deepEqual(pills.map(textOf), ['06:03 KST', '06:03 KST · 조회 1분 전', '지연 · 05:51 조회 (12분 전)', '끊김 · 마지막 조회 05:02']);
  assert.deepEqual(pills.map(p => p.props.dataSet.pdStage), ['fresh', 'fresh', 'stale', 'broken']);
  assert.equal(all(pills[2], n => n.type === 'Svg').length, 1, '지연은 clock 아이콘');
  const pill = tree(h(ui.StatusPill, { tone: 'ok', text: '정상 · 넉넉합니다' })).children[0];
  assert.equal(textOf(pill), '정상 · 넉넉합니다');
  assert.ok(all(pill, n => n.type === 'View' && n !== pill).length >= 1, '색 점');
  const ribbon = tree(h(ui.SampleRibbon, {})).children[0];
  assert.equal(textOf(ribbon), '┆ 예시 · 실제 값 아님');
  assert.equal(ribbon.props.dataSet.pdChip, 'example');
});

test('Card·SourceFooter·RestrictedCard: h2 제목, [i] 이름, data-pd-card, role=note, 제한 카드에는 버튼 없음', () => {
  const ui = load();
  const card = tree(h(ui.Card, { title: '확정 가격', onMethod: () => {}, chips: 'chip' },
    h(ui.SourceFooter, { source: 'KPX 제주 하루전시장', time: '관측 시각 미제공 · 조회 06:04 KST', onSources: () => {}, context: '확정 가격' }))).children[0];
  assert.equal(card.props.dataSet.pdCard, 'true');
  const [title] = byRole(card, 'heading');
  assert.equal(title.props['aria-level'], 2);
  assert.equal(textOf(title), '확정 가격');
  const names = pressables(card).map(p => p.props['aria-label']);
  assert.deepEqual(names, ['산출 방식 보기: 확정 가격', '출처 표: 확정 가격']);
  pressables(card).forEach(p => assertTarget(p, p.props['aria-label']));
  const [note] = byRole(card, 'note');
  assert.ok(flat(note.props.style).minHeight >= 44, '출처 줄 최소 높이 44');
  assert.equal(all(note, n => n.props.dataSet && n.props.dataSet.pdMeta === 'time').map(textOf).join(''), '관측 시각 미제공 · 조회 06:04 KST');
  const restricted = tree(h(ui.RestrictedCard, { title: '내 발전소', body: '공개 데모에서 제공하지 않습니다' }));
  assert.equal(pressables(restricted).length, 0, '제한 카드에는 버튼이 없다');
  assert.equal(textOf(byRole(restricted, 'img')[0]), '잠김');
  const narrow = load(320);
  const small = tree(h(narrow.Card, { title: '확정 가격', chips: 'chip' })).children[0];
  const row = all(small, n => n.type === 'View' && flat(n.props.style).flexWrap === 'wrap' && flat(n.props.style).flex === 1)[0];
  assert.equal(flat(row.props.style).flexDirection, 'column', '360 미만에서는 칩이 제목 아래 줄로');
  assert.equal(flat(small.props.style).padding, 12);
});

test('Value·SampleValue·KpiBlock·HeroValue: data-pd-value·raw, 단위 띄어쓰기, 예시 덧말, 한 문자열 이름', () => {
  const ui = load();
  const demand = pv(ui, 664.3, 'MW', 'measured', 'KPX 제주 실시간');
  const priceV = pv(ui, 115.27, '원/kWh', 'confirmedPrice', 'KPX 제주 하루전시장', 'confirmed');
  const v1 = tree(h(ui.Value, { v: demand, format: ui.format.mw })).children[0];
  assert.equal(textOf(v1), '664 MW');
  assert.deepEqual(v1.props.dataSet, { pdValue: 'measured', pdRaw: '664.3', pdNowrap: 'true' });
  assert.equal(textOf(tree(h(ui.Value, { v: priceV, format: ui.format.price })).children[0]), '115.3원/kWh');
  const sample = { value: 468.6, unit: 'MW', label: '예시' };
  const s1 = tree(h(ui.SampleValue, { v: sample, format: ui.format.price })).children[0];
  assert.equal(s1.props.dataSet.pdExample, 'true');
  assert.equal(textOf(s1), '468.6 MW (예시)');
  assert.equal(flat(all(s1, n => textOf(n) === ' (예시)')[0].props.style).opacity, 0, '"(예시)"는 화면 읽기에만');
  const kpi = byRole(tree(h(ui.KpiBlock, { label: '수요', v: demand, format: ui.format.mw })), 'group')[0];
  assert.equal(kpi.props['aria-label'], '수요 664 메가와트, 상태: 실측, 출처 KPX 제주 실시간');
  const hero = byRole(tree(h(ui.HeroValue, { label: '확정 가격', v: priceV, format: ui.format.price })), 'group')[0];
  const big = all(hero, n => n.props.dataSet && n.props.dataSet.pdValue)[0];
  assert.equal(flat(big.props.style).fontSize, 32);
  const narrowHero = byRole(tree(h(load(360).HeroValue, { label: '확정 가격', v: priceV, format: ui.format.price })), 'group')[0];
  assert.equal(flat(all(narrowHero, n => n.props.dataSet && n.props.dataSet.pdValue)[0].props.style).fontSize, 28);
});

test('HourStepper: 버튼 2개 이름, 양 끝 비활성(까닭), 값은 메타, 값에 aria-live 없음(live region은 셸 하나)', () => {
  const ui = load();
  const seen = [];
  const mid = tree(h(ui.HourStepper, { hour: 7, onChange: v => seen.push(v) }));
  const [prev, next] = pressables(mid);
  assert.deepEqual([prev.props['aria-label'], next.props['aria-label']], ['이전 거래시간', '다음 거래시간']);
  prev.props.onPress();
  next.props.onPress();
  assert.deepEqual(seen, [6, 8]);
  assert.equal(all(mid, n => n.props.dataSet && n.props.dataSet.pdMeta === 'hour').map(textOf).join(''), '07h · 06–07시');
  assert.equal(all(mid, n => 'aria-live' in n.props).length, 0);
  const first = pressables(tree(h(ui.HourStepper, { hour: 1, onChange: () => {} })))[0];
  assert.equal(first.props.disabled, true);
  assert.equal(first.props['aria-label'], '이전 거래시간 · 첫 거래시간입니다');
  const last = pressables(tree(h(ui.HourStepper, { hour: 24, onChange: () => {} })))[1];
  assert.equal(last.props.disabled, true);
});

test('SegmentedControl: radiogroup > radio(aria-checked), roving tabindex, 화살표·Home·End', () => {
  const ui = load();
  const picked = [];
  const options = [{ value: 'chart', label: '차트' }, { value: 'table', label: '표' }, { value: 'both', label: '둘 다' }];
  const root = tree(h(ui.SegmentedControl, { label: '보기 방식', options, value: 'chart', onChange: v => picked.push(v) }));
  const [group] = byRole(root, 'radiogroup');
  assert.equal(group.props['aria-label'], '보기 방식');
  const radios = byRole(root, 'radio');
  assert.deepEqual(radios.map(r => [r.props['aria-checked'], r.props.tabIndex]), [[true, 0], [false, -1], [false, -1]]);
  radios.forEach(r => assertTarget(r, r.props['aria-label']));
  const key = k => ({ key: k, preventDefault: () => {} });
  radios[0].props.onKeyDown(key('ArrowRight'));
  radios[0].props.onKeyDown(key('ArrowLeft'));
  radios[0].props.onKeyDown(key('End'));
  radios[2].props.onKeyDown(key('Home'));
  radios[0].props.onKeyDown(key('ArrowDown'));
  radios[0].props.onKeyDown(key('a'));
  assert.deepEqual(picked, ['table', 'both', 'both', 'chart', 'table']);
});

test('TabBar·레일: nav "주요 메뉴" > tablist > tab(aria-selected), 탭 높이 56, 아래 탭만 안전 영역', () => {
  const ui = load();
  const items = [{ key: 'smp', label: '오늘', icon: 'chart' }, { key: 'bidding', label: '입찰', icon: 'clock' },
    { key: 'settlement', label: '정산', icon: 'list' }, { key: 'guide', label: '알아보기', icon: 'book' }];
  const selected = [];
  const bar = tree(h(ui.TabBar, { items, active: 'bidding', onSelect: k => selected.push(k) }));
  const [nav] = byRole(bar, 'navigation');
  assert.equal(nav.props['aria-label'], '주요 메뉴');
  assert.deepEqual(nav.props.dataSet, { pdSafe: 'bottom' });
  const tabs = byRole(bar, 'tab');
  assert.deepEqual(tabs.map(t => t.props['aria-label']), ['오늘', '입찰', '정산', '알아보기']);
  assert.deepEqual(tabs.map(t => [t.props['aria-selected'], t.props.tabIndex]), [[false, -1], [true, 0], [false, -1], [false, -1]]);
  tabs.forEach(t => assert.ok(flat(t.props.style).minHeight >= 56));
  tabs[3].props.onPress();
  assert.deepEqual(selected, ['guide']);
  const rail = tree(h(ui.Rail, { items, active: 'smp', onSelect: () => {} }));
  assert.equal(byRole(rail, 'tablist')[0].props['aria-orientation'], 'vertical');
  assert.equal(byRole(rail, 'navigation')[0].props.dataSet, undefined);
  assert.equal(flat(byRole(rail, 'navigation')[0].props.style).width, 240);
});

test('Sheet: dialog + aria-modal + aria-labelledby, 닫기 버튼, 스크림 클릭, 폭별 자리', () => {
  for (const [width, expect] of [[390, { left: 0, right: 0, bottom: 0 }], [768, { width: 640 }], [1280, { width: 400, right: 0, top: 0 }]]) {
    const ui = load(width);
    let closed = 0;
    const root = tree(h(ui.Sheet, { title: '산출 방식', onClose: () => { closed += 1; } }, 'body'));
    const [dialog] = byRole(root, 'dialog');
    assert.equal(dialog.props['aria-modal'], true);
    const [title] = byRole(dialog, 'heading');
    assert.equal(dialog.props['aria-labelledby'], title.props.nativeID);
    assert.equal(textOf(title), '산출 방식');
    const style = flat(dialog.props.style);
    Object.entries(expect).forEach(([k, v]) => assert.equal(style[k], v, `${width}: ${k}`));
    const [close] = pressables(dialog);
    assert.equal(close.props['aria-label'], '닫기');
    close.props.onPress();
    const scrim = all(root, n => n.props.dataSet && n.props.dataSet.pdScrim === 'true')[0];
    assert.equal(scrim.props['aria-hidden'], true, '스크림은 Tab 순서·접근성 트리 밖');
    scrim.props.onClick();
    assert.equal(closed, 2);
  }
});

test('StateView: 불러오는 중 aria-busy, 오류만 다시 시도, 제한 상태에는 재시도 없음, 최종 크기', () => {
  const ui = load();
  const loading = tree(h(ui.StateView, { kind: 'loading', minHeight: 132 })).children[0];
  assert.equal(loading.props['aria-busy'], true);
  assert.equal(flat(loading.props.style).minHeight, 132);
  const error = tree(h(ui.StateView, { kind: 'error', onRetry: () => {} }));
  assert.deepEqual(pressables(error).map(p => p.props['aria-label']), ['다시 시도']);
  assert.equal(pressables(tree(h(ui.StateView, { kind: 'restricted', onRetry: () => {} }))).length, 0);
  assert.equal(pressables(tree(h(ui.StateView, { kind: 'error' }))).length, 0, '성공할 수 없으면 버튼을 그리지 않는다');
});

test('DataTable·TermLink·ListRow: table 역할, 현재 행 aria-current, 링크 이름·높이', () => {
  const ui = load();
  const table = tree(h(ui.DataTable, {
    caption: '시간별 기준선 예측', columns: [{ key: 'h', title: '거래시간', meta: true }, { key: 'p', title: '가격', align: 'right' }],
    rows: [{ key: '7', cells: { h: '07h', p: 'x' }, current: true }, { key: '8', cells: { h: '08h', p: 'y' } }],
  }));
  assert.equal(byRole(table, 'table')[0].props['aria-label'], '시간별 기준선 예측');
  assert.deepEqual(['row', 'columnheader', 'cell'].map(r => byRole(table, r).length), [3, 2, 4]);
  assert.equal(byRole(table, 'row')[1].props['aria-current'], 'true');
  assert.equal(all(table, n => n.type === 'ScrollView')[0].props.horizontal, true);
  const term = pressables(tree(h(ui.TermLink, { label: 'SMP', onPress: () => {} })))[0];
  assert.deepEqual([term.props.role, term.props['aria-label']], ['link', '용어 풀이: SMP']);
  assertTarget(term, 'TermLink');
  const row = pressables(tree(h(ui.ListRow, { title: '용어 사전', detail: '열두 개 낱말', onPress: () => {} })))[0];
  assert.deepEqual([row.props.role, row.props['aria-label']], ['link', '용어 사전, 열두 개 낱말']);
  assert.equal(flat(row.props.style).minHeight, 56);
});

test('Screen·LiveRegion·OfflineBanner·ErrorBoundary·Tour: main, h1 tabIndex=-1, polite 하나, status, 오류 화면', () => {
  for (const [width, gutter] of [[390, 16], [320, 12]]) {
    const ui = load(width);
    const screen = tree(h(ui.Screen, { title: '오늘' }, 'card'));
    const [main] = byRole(screen, 'main');
    assert.equal(main.type, 'ScrollView');
    assert.equal(flat(main.props.contentContainerStyle).paddingHorizontal, gutter);
    const [h1] = byRole(screen, 'heading');
    assert.deepEqual([h1.props['aria-level'], h1.props.tabIndex, textOf(h1)], [1, -1, '오늘']);
  }
  const ui = load();
  const live = tree(h(ui.LiveRegion, {}, 'app'));
  const regions = all(live, n => 'aria-live' in n.props);
  assert.deepEqual(regions.map(r => r.props['aria-live']), ['polite']);
  assert.equal(flat(regions[0].props.style).opacity, 0);
  assert.equal(byRole(tree(h(ui.OfflineBanner, { text: '오프라인' })), 'status').length, 1);
  const failed = tree(h(ui.ErrorBoundary, { __failed: true }, 'app'));
  assert.equal(textOf(byRole(failed, 'heading')[0]), '화면을 그리는 중 문제가 생겼습니다');
  assert.deepEqual(pressables(failed).map(p => p.props['aria-label']), ['다시 불러오기']);
  assert.equal(textOf(tree(h(ui.ErrorBoundary, {}, 'app'))), 'app');
  const banner = tree(h(ui.TourBanner, { title: '1분 둘러보기', body: '설명', onStart: () => {}, onDismiss: () => {} }));
  assert.deepEqual(pressables(banner).map(p => p.props['aria-label']), ['둘러보기 시작', '닫기: 둘러보기 안내']);
  const step1 = tree(h(ui.TourStep, { step: 1, total: 5, body: '설명', onPrev: () => {}, onNext: () => {}, onSkip: () => {} }));
  assert.equal(all(step1, n => n.props.dataSet && n.props.dataSet.pdMeta === 'tour').map(textOf).join(''), '1 / 5');
  assert.equal(pressables(step1)[0].props.disabled, true);
  const step5 = tree(h(ui.TourStep, { step: 5, total: 5, body: '설명', onPrev: () => {}, onNext: () => {}, onSkip: () => {} }));
  assert.equal(pressables(step5)[1].props['aria-label'], '둘러보기 마치기');
});
