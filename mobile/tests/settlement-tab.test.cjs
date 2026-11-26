'use strict';
// 정산 화면(명세 4.3·6.3·7.3·7.9절, F9). 예시 정산서의 산수(120 MWh × 150원/kWh = 1,800만원 · −6 MWh × 140원/kWh = −84만원 ·
// 합계 1,716만원)와 예시 출처(화면의 숫자는 모두 data/examples.ts의 예시 값, 실제 값 표지 없음)를 고정하고, 정산 경로가 요청을 보내지
// 않는지(자료 클라이언트·저장소를 가져오지 않는다), 카드·패널·산출 방식 시트 값의 의미와 I2를 노드 수준에서 본다.
// 브라우저 확인(S1–S2, 6폭 × 2테마, 요청 0, ≥ 1280 패널)은 scratchpad/mobile/eval/steps/F9에 있다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const { createTsLoader, defaultMocks, MOBILE_ROOT, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const SVG = { __esModule: true, default: 'Svg', Svg: 'Svg', Path: 'Path', Circle: 'Circle', Ellipse: 'Ellipse', Rect: 'Rect', Line: 'Line', Defs: 'Defs', Pattern: 'Pattern' };

function load(width = 390, globals) {
  const mocks = defaultMocks();
  const rn = { ...mocks['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG }, globals });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  const tab = at('features/settlement/SettlementTab.tsx');
  return {
    model: at('features/settlement/model.ts'), examples: at('data/examples.ts'), format: at('lib/format.ts'), strings: at('content/strings.ts'),
    tokens: at('ui/tokens.ts'), card: at('features/settlement/SampleStatementCard.tsx'),
    SettlementTab: tab.default, SettlementPanel: tab.SettlementPanel, SettlementMethodFacts: tab.SettlementMethodFacts,
  };
}

// ---------------------------------------------------------------- 노드 수준 렌더(bidding-tab.test.cjs와 같은 방식)
const h = (type, props, ...children) => ({ type, props: { ...(props || {}), children } });
function render(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return [];
  if (Array.isArray(node)) return node.flatMap(render);
  if (typeof node !== 'object') return [String(node)];
  const { type, props } = node;
  if (typeof type === 'function') return render(type(props));
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
const ds = n => (n.props && n.props.dataSet) || {};
const byId = (root, id) => all(root, n => n.props.testID === id)[0];
const MARKS = ['pdValue', 'pdMeta', 'pdChip', 'pdExample'];
const noop = () => {};
const nb = text => text.replace(/~/g, '\u00A0');

/** I2(노드 수준): 숫자가 든 글자는 값·메타·칩·예시 표지 안에만(그림 role=img 안은 이름이 대신 읽힌다). */
function assertI2(root, label) {
  const stray = [];
  const walk = (node, marked) => {
    if (typeof node === 'string') {
      if (/\d/.test(node) && !marked) stray.push(node);
      return;
    }
    const here = marked || MARKS.some(k => ds(node)[k] !== undefined) || node.props.role === 'img';
    node.children.forEach(c => walk(c, here));
  };
  walk(root, false);
  assert.deepEqual(stray, [], `${label}: 표지 밖 숫자`);
}

/** 값으로 가져오는 상대 경로 파일 전부(형식만 가져오기는 번들에서 지워지므로 뺀다). */
function importGraph(entry, seen = new Set()) {
  if (seen.has(entry)) return seen;
  seen.add(entry);
  const sf = ts.createSourceFile(entry, fs.readFileSync(entry, 'utf8'), ts.ScriptTarget.Latest, true,
    entry.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  for (const st of sf.statements) {
    if (!(ts.isImportDeclaration(st) || ts.isExportDeclaration(st)) || !st.moduleSpecifier || !ts.isStringLiteral(st.moduleSpecifier)) continue;
    const clause = ts.isImportDeclaration(st) ? st.importClause : null;
    const named = clause && clause.namedBindings && ts.isNamedImports(clause.namedBindings) ? clause.namedBindings.elements : [];
    const typeOnly = ts.isImportDeclaration(st)
      ? Boolean(clause && (clause.isTypeOnly || (!clause.name && named.length > 0 && named.every(el => el.isTypeOnly))))
      : st.isTypeOnly;
    const spec = st.moduleSpecifier.text;
    if (typeOnly || !spec.startsWith('.')) continue;
    const base = path.resolve(path.dirname(entry), spec);
    const file = ['.ts', '.tsx', '/index.ts', '/index.tsx'].map(ext => base + ext).find(f => fs.existsSync(f));
    assert.ok(file, `${path.relative(MOBILE_ROOT, entry)}: ${spec}`);
    importGraph(file, seen);
  }
  return seen;
}

/** 숫자 리터럴(0 빼고)과 숫자가 든 글자 리터럴. 주석은 AST에 없다. */
function digitLiterals(file) {
  const sf = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true,
    file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const out = [];
  const visit = node => {
    if (ts.isNumericLiteral(node) && node.text !== '0') out.push(node.text);
    if ((ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateLiteralToken(node) || ts.isJsxText(node))
      && /\d/.test(node.text)) out.push(node.text);
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return out;
}

// ---------------------------------------------------------------- 1. 산수와 글자
test('예시 정산서 산수(4.3절): 120,000 kWh × 150 = 18,000,000원 · −6,000 kWh × 140 = −840,000원 · 합 17,160,000원 → 1,800만원 · −84만원 · 1,716만원', () => {
  const m = load();
  const st = m.examples.EXAMPLE_STATEMENT;
  const won = m.examples.EXAMPLE_STATEMENT_AMOUNTS;
  assert.equal(m.examples.KWH_PER_MWH, 1000);
  assert.deepEqual([st.dayAheadMwh.value, st.dayAheadPrice.value, st.realtimeDeviationMwh.value, st.realtimePrice.value], [120, 150, -6, 140]);
  assert.deepEqual([won.dayAhead.value, won.deviation.value, won.total.value], [18000000, -840000, 17160000]);
  assert.equal(won.total.value, won.dayAhead.value + won.deviation.value);
  assert.deepEqual([won.dayAhead.unit, won.deviation.unit, won.total.unit], ['원', '원', '원']);
  const f = m.format;
  assert.deepEqual([won.dayAhead, won.deviation, won.total].map(v => f.krw(v.value, 0)), ['1,800만원', '−84만원', '1,716만원']);
  // 기본 자릿수(입찰 기대 수입 "468.6만원")와 억 단위는 그대로다.
  assert.deepEqual([f.krw(won.total.value), f.krw(-840000), f.krw(123456789, 0), f.krw(99995000, 0), f.krw(null, 0)],
    ['1,716.0만원', '−84.0만원', '1.23억원', '1.00억원', '—']);
  const view = m.model.settlementView();
  assert.equal(view.answer, '정산은 판 전기값을 KPX가 계산해 주는 과정입니다. 아래 숫자는 계산 방식을 보여 주는 예시입니다.');
  // ~ = 줄바꿈하지 않는 빈칸(U+00A0): 값과 단위, 연산자와 값은 한 줄에 둔다(5.4절).
  assert.deepEqual(view.facts, [
    '단위: 1~MWh =~1,000~kWh (가격이 kWh당이라 MWh로 적은 양을 kWh로 바꿔 곱합니다)',
    '① 하루전시장에서 판 양: 120~MWh =~120,000~kWh, ×~150원/kWh =~18,000,000원 (1,800만원)',
    '② 실제 발전과의 차이: −6~MWh =~−6,000~kWh, ×~140원/kWh =~−840,000원 (−84만원)',
    '합계: 18,000,000원 −~840,000원 =~17,160,000원 (1,716만원)',
  ].map(nb));
  assert.equal(view.legend, '1,716만원 예시 정산서', '둘러보기 5단계의 ┆1,716만원 예시 정산서┆와 같은 글자');
  assert.equal(view.panelMethod[2], nb('①~+~②가 그날 받는 돈입니다. 가격이 kWh당이라 1~MWh는 1,000~kWh로 바꿔 곱합니다.'));
});

test('정산서 렌더: 행(라벨 / 양 × 가격 = 금액) · 합계 · 가상 발전소 줄 · 설명 두 줄, well 면, 머리 리본이 제목보다 먼저, 출처 줄 없음(7.3절)', () => {
  const m = load();
  const root = tree(h(m.card.SampleStatementCard, { onMethod: noop }));
  const row = id => byId(root, `settle-row-${id}`).children.map(textOf);
  assert.deepEqual(row('day-ahead'), ['① 하루전시장에서 판 양', '120 MWh (예시) × 150원/kWh (예시) = 1,800만원 (예시)']);
  assert.deepEqual(row('deviation'), ['② 실제 발전과의 차이', '−6 MWh (예시) × 140원/kWh (예시) = −84만원 (예시)']);
  assert.deepEqual(row('total'), ['합계', '1,716만원 (예시)']);
  assert.equal(textOf(byId(root, 'settle-plant')), '예시 풍력발전소 · 50 MW (예시) (가상) · 하루');
  const card = byId(root, 'settle-sample');
  assert.deepEqual([ds(card.children[0]).pdRibbon, textOf(card.children[0])], ['true', '┆ 예시 · 실제 값 아님'], '리본이 첫 자식');
  assert.equal(all(card, n => n.props.role === 'heading' && n.props['aria-level'] === 2).map(textOf)[0], '예시 정산서');
  assert.equal(all(card, n => n.props.role === 'note').length, 0, '예시에는 출처·시각이 없다');
  assert.equal(flat(byId(root, 'settle-statement').props.style).backgroundColor, m.tokens.THEMES.light.color.well);
  const notes = all(card, n => n.type === 'Text' && /단순화한 예시|KPX 정산 규정/.test(textOf(n))).map(textOf);
  assert.deepEqual(notes, ['①은 하루전 가격, ②는 실시간 가격으로 계산하는 단순화한 예시입니다.', '실제 규칙은 KPX 정산 규정을 따릅니다.']);
  assertI2(root, 'statement@390');
});

test('폭별(4.3·4.6절): 휴대폰 폭(600 미만, 320 포함)에서는 정산서 행이 두 줄(라벨 / 식 = 값)로 쌓이고, 600부터는 한 줄(식은 오른쪽)', () => {
  // 들어가는 만큼 줄을 바꾸게 두면 430에서 ① 행은 두 줄 · ② 행은 한 줄로 섞였다(F9 브라우저 점검): 폭 구간으로 정한다.
  for (const [width, stacked] of [[320, true], [359, true], [360, true], [390, true], [430, true], [599, true], [600, false], [768, false], [1280, false]]) {
    const m = load(width);
    const root = tree(h(m.card.SampleStatementCard, {}));
    for (const id of ['day-ahead', 'deviation']) {
      const row = byId(root, `settle-row-${id}`);
      const s = flat(row.props.style);
      assert.deepEqual([s.flexDirection, s.flexWrap], [stacked ? 'column' : 'row', 'wrap'], `${id}@${width}`);
      assert.deepEqual([flat(row.children[1].props.style).marginLeft, flat(row.children[1].props.style).textAlign], ['auto', 'right']);
    }
    assertI2(root, `statement@${width}`);
  }
});

// ---------------------------------------------------------------- 2. 숫자의 출처와 요청
test('숫자는 모두 examples.ts에서(F9 합격 확인): 실제 값 표지(data-pd-value) 없음, 예시 값은 examples.ts 객체 그대로, 메타 안 숫자는 NowPill 시각뿐', () => {
  const m = load();
  const root = tree(h(m.SettlementTab, {}));
  assertI2(root, 'settlement@390');
  assert.equal(all(root, n => ds(n).pdValue !== undefined).length, 0, '정산 화면에는 API 값이 없다');
  assert.deepEqual(all(root, n => ds(n).pdExample !== undefined).map(textOf),
    ['50 MW (예시)', '120 MWh (예시)', '150원/kWh (예시)', '1,800만원 (예시)', '−6 MWh (예시)', '140원/kWh (예시)', '−84만원 (예시)', '1,716만원 (예시)',
      // ①+② 그림(round 1 D14: 모든 폭에서 예시 정산서 카드 안, 그림 안 글자는 이름이 대신 읽힌다)
      '1,800만원', '−84만원', '1,716만원']);
  const metaDigits = all(root, n => ds(n).pdMeta !== undefined && /\d/.test(textOf(n))).map(n => ds(n).pdMeta);
  assert.deepEqual([...new Set(metaDigits)], ['now'], 'KST 시각(NowPill) 말고는 메타에 숫자가 없다');
  // view model은 examples.ts의 얼린 객체를 그대로 넘긴다(복사·재계산 없음).
  const st = m.examples.EXAMPLE_STATEMENT;
  const won = m.examples.EXAMPLE_STATEMENT_AMOUNTS;
  const view = m.model.settlementView();
  assert.deepEqual(view.rows.map(r => [r.quantity, r.unitPrice, r.amount]),
    [[st.dayAheadMwh, st.dayAheadPrice, won.dayAhead], [st.realtimeDeviationMwh, st.realtimePrice, won.deviation]]);
  assert.ok(view.rows[0].quantity === st.dayAheadMwh && view.rows[1].amount === won.deviation && view.total === won.total);
  assert.deepEqual(view.figure.bars.map(b => b.amount), [won.dayAhead, won.deviation, won.total]);
  assert.ok([...Object.values(st), ...Object.values(won), st, won].every(Object.isFrozen));
  // 정산 view model과 화면 입구에는 숫자가 없다(축의 원점 0만). 카드 파일의 숫자는 그림 치수(px)뿐이다.
  for (const rel of ['features/settlement/model.ts', 'features/settlement/SettlementTab.tsx']) {
    assert.deepEqual(digitLiterals(path.join(SRC_ROOT, rel)), [], rel);
  }
});

test('정산 경로 요청 0(6.3절): 정산 기능은 자료 클라이언트·저장소·services를 가져오지 않고, 화면·패널·시트 값을 그려도 fetch가 불리지 않는다', () => {
  const graph = [...importGraph(path.join(SRC_ROOT, 'features/settlement/SettlementTab.tsx'))].map(f => path.relative(SRC_ROOT, f).split(path.sep).join('/'));
  assert.ok(graph.includes('features/settlement/model.ts') && graph.includes('data/examples.ts'), graph.join(', '));
  for (const banned of ['data/client.ts', 'lib/resource.ts', 'data/adapters.ts']) assert.ok(!graph.includes(banned), `${banned} 가져오기`);
  assert.ok(!graph.some(f => f.startsWith('services/') || f.startsWith('screens/')), graph.join(', '));
  for (const rel of ['model.ts', 'SampleStatementCard.tsx', 'SettlementTab.tsx']) {
    assert.doesNotMatch(fs.readFileSync(path.join(SRC_ROOT, 'features/settlement', rel), 'utf8'), /\bfetch\s*\(|useResource|createClient|\/settlements|\/power-plants/, rel);
  }
  const calls = [];
  const m = load(1280, { fetch: (...args) => { calls.push(args); return new Promise(() => {}); } });
  tree([h(m.SettlementTab, {}), h(m.SettlementPanel, {}), h(m.SettlementMethodFacts, { card: 'settle-sample' })]);
  assert.equal(calls.length, 0);
  // 레거시 정산 화면 파일은 지우지 않고(P2) 저장소에 남는다. 번들에서는 App.tsx가 가져오지 않아 빠진다(마지막 테스트).
  assert.ok(fs.existsSync(path.join(SRC_ROOT, 'screens/SettlementScreen.tsx')));
});

// ---------------------------------------------------------------- 3. 의미
test('정산 화면 의미: h1 정산 · AnswerLine · 제한 카드(잠김 칩, 버튼·재시도 없음) · [i] 하나 · 용어 링크 넷 · 제출·저장 버튼 없음 · 누름 대상 44', () => {
  const m = load();
  const root = tree(h(m.SettlementTab, {}));
  assert.deepEqual(all(root, n => n.props.role === 'heading' && n.props['aria-level'] === 1).map(textOf), ['정산']);
  assert.deepEqual(all(root, n => n.props.role === 'main').length, 1);
  assert.equal(textOf(all(root, n => ds(n).pdAnswer === 'true')[0]), m.strings.SETTLEMENT_ANSWER);
  assert.equal(textOf(all(root, n => n.props.testID === 'now-pill')[0]).endsWith(' KST'), true, '조회가 없는 화면은 KST 시각만(3.4절)');
  const presses = all(root, n => n.type === 'Pressable');
  const names = presses.map(p => p.props['aria-label']);
  assert.deepEqual(names, ['산출 방식 보기: 예시 정산서', '용어 풀이: 정산', '용어 풀이: 하루전시장', '용어 풀이: 실시간시장', '용어 풀이: kWh와 MW']);
  assert.ok(!names.some(n => /제출|저장|등록|삭제|전송|보내기|다시 시도|완료/.test(n)), names.join(' / '));
  presses.forEach(p => {
    const s = flat(p.props.style);
    assert.ok(Math.max(s.minHeight || 0, s.height || 0) >= 44 && Math.max(s.minWidth || 0, s.width || 0) >= 44, `${p.props['aria-label']} 상자`);
  });
  const links = all(byId(root, 'settle-terms'), n => n.props.role === 'link');
  assert.equal(links.length, 4);
  // 44보다 짧은 "정산"은 상자 가운데에 둔다(왼쪽에 붙이면 다음 링크와의 간격만 넓어 보인다).
  assert.ok(links.every(l => flat(l.props.style).alignItems === 'center'));
  const restricted = byId(root, 'settle-restricted');
  assert.equal(all(restricted, n => n.type === 'Pressable').length, 0, '제한 카드에는 재시도·버튼이 없다(M8)');
  assert.deepEqual(all(restricted, n => ds(n).pdChip !== undefined).map(n => n.props['aria-label']), ['상태: 잠김']);
  assert.equal(all(restricted, n => n.props.role === 'heading').map(textOf)[0], '실제 정산 자료');
  assert.ok(textOf(restricted).includes('발전소별 실제 정산 자료는 공개 데모에서 제공하지 않습니다.'));
  assert.ok(textOf(restricted).includes('이 서비스의 내부 정산 계산도 KPX 정산이 아닌 모의 계산입니다.'));
  // 카드 순서(와이어프레임): 제한 카드 → 예시 정산서 → 용어 줄.
  const order = all(root, n => ['settle-restricted', 'settle-sample', 'settle-terms'].includes(n.props.testID)).map(n => n.props.testID);
  assert.deepEqual(order, ['settle-restricted', 'settle-sample', 'settle-terms']);
  // 용어 줄의 id는 용어 사전에 있다(시트가 열린다). "kWh와 MW"는 F9가 더한 13번째 용어다.
  assert.deepEqual(m.strings.SETTLEMENT_TERMS.map(t => [t.id, m.strings.GLOSSARY_TITLES[t.id]]),
    [['settlement', '정산'], ['dam', '하루전시장'], ['rtm', '실시간시장'], ['units', 'kWh와 MW']]);
});

test('≥ 1280 "정산 읽는 법" 패널: 표시 예(┆예시┆ 1,716만원 예시 정산서 · ‹잠김›) · 용어 셋 · 읽는 법 세 줄(그림은 카드 안, D14)', () => {
  const m = load(1280);
  const root = tree(h(m.SettlementPanel, {}));
  assertI2(root, 'panel');
  assert.deepEqual(all(root, n => ds(n).pdMeta === 'legend').map(textOf), ['1,716만원 예시 정산서', '실제 정산 자료 · 공개 데모에서 제공하지 않음']);
  assert.deepEqual(all(root, n => ds(n).pdChip !== undefined).map(n => n.props['aria-label']), ['상태: 예시', '상태: 잠김']);
  assert.deepEqual(all(root, n => n.type === 'Pressable').map(p => p.props['aria-label']), ['용어 풀이: 정산', '용어 풀이: 하루전시장', '용어 풀이: 실시간시장']);
  assert.deepEqual(all(root, n => ds(n).pdMeta === 'method').map(textOf), [...m.model.settlementView().panelMethod]);
  assert.equal(all(root, n => n.props.testID === 'settle-figure').length, 0, '패널에 그림을 두 번 두지 않는다');
});

test('①+② 식 그림(round 1 D14): 모든 폭에서 예시 정산서 카드 안(정산서 뒤, 설명 두 줄 앞), ①·합계는 예시 호박색 채움, ②는 진한 칸, y는 0부터', () => {
  for (const width of [320, 390, 768, 1280]) {
    const m = load(width);
    const card = byId(tree(h(m.card.SampleStatementCard, {})), 'settle-sample');
    const fig = byId(card, 'settle-figure');
    assert.deepEqual([fig.props.role, fig.props['aria-label']], ['img', '예시 정산서 그림: ① 하루전 1,800만원 + ② 실시간 차이 −84만원 = 합계 1,716만원'], `${width}`);
    assert.equal(fig.children[0].props['aria-hidden'], true, '그림 안은 이름이 대신 읽힌다');
    const bars = all(fig, n => n.props.testID === 'settle-figure-bar').map(b => flat(b.props.style));
    // 112 px 칸: ①은 0–112, ②는 1,716만원 높이(107)에서 ①의 끝(112)까지 5 px, 합계는 0–107(빈 상자 둘과 머리카락 선이 아니다).
    assert.deepEqual(bars.map(b => [b.bottom, b.height]), [[0, 112], [107, 5], [0, 107]], `${width}`);
    const t = m.tokens.THEMES.light;
    assert.deepEqual(bars.map(b => b.backgroundColor), [t.chart.exampleFill, t.chart.exampleMark, t.chart.exampleFill]);
    // ② 진한 칸: 밝은 테마는 예시 글자색 그대로, 어두운 테마는 휘도 0.6 아래(예시 글자색 #FCD34D는 0.68이라 어두운 바탕 위 밝은 덩어리, F13b 측정).
    const lum = hex => { const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).map(v => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; };
    assert.ok(t.chart.exampleMark === t.chip.example.text && lum(m.tokens.THEMES.dark.chart.exampleMark) < 0.6 && lum(m.tokens.THEMES.dark.chip.example.text) > 0.6);
    assert.ok(bars.every(b => b.borderColor === t.chip.example.text), '예시 글자색 외곽선(7.1절)');
    assert.deepEqual(all(fig, n => ds(n).pdExample !== undefined).map(textOf), ['1,800만원', '−84만원', '1,716만원']);
    assert.deepEqual(all(fig, n => n.type === 'Text' && /^[+=]$/.test(textOf(n))).map(textOf), ['+', '=']);
    const order = all(card, n => ['settle-statement', 'settle-figure'].includes(n.props.testID)).map(n => n.props.testID);
    assert.deepEqual(order, ['settle-statement', 'settle-figure'], `${width}: 정산서 뒤에 그림`);
  }
});

test('산출 방식 시트(settle-sample)의 이번 화면 값: 리본(예시 · 실제 값 아님) 뒤에 단위 환산과 원 단위 계산 네 줄, 다른 카드에는 없다', () => {
  const m = load();
  const root = tree(h(m.SettlementMethodFacts, { card: 'settle-sample' }));
  assertI2(root, 'facts');
  assert.equal(textOf(all(root, n => ds(n).pdRibbon === 'true')[0]), '┆ 예시 · 실제 값 아님');
  assert.deepEqual(all(root, n => ds(n).pdMeta === 'fact').map(textOf), [...m.model.settlementView().facts]);
  for (const card of ['today-price', 'bid-ladder', 'bid-merit']) assert.equal(tree(h(m.SettlementMethodFacts, { card })).children.length, 0, card);
});

// ---------------------------------------------------------------- 4. 설명 글과 입구
test('guide.ko.json(실제 파일): 로더 검사 통과(용어 13개), 정산·kWh와 MW 용어와 정산서 설명 글, 숫자 없음', () => {
  const guide = createTsLoader().load(path.join(SRC_ROOT, 'content/guide.ts'));
  const content = guide.parseGuide(JSON.parse(fs.readFileSync(path.join(MOBILE_ROOT, 'public/content/guide.ko.json'), 'utf8')));
  assert.equal(content.glossary.length, 13);
  const units = content.glossary.find(g => g.id === 'units');
  assert.ok(units.short.length <= guide.GLOSSARY_SHORT_MAX, units.short);
  assert.ok(units.body.join(' ').includes('메가와트시는 킬로와트시의 천 배입니다'));
  assert.ok(content.glossary.find(g => g.id === 'settlement').body.join(' ').includes('계산 방식을 보여 주는 예시'));
  assert.ok(content.cards['settle-sample'].body.join(' ').includes('실제 정산 자료가 아닙니다'));
});

test('App.tsx: 정산 경로는 새 탭(SettlementTab), 레거시 정산 화면을 가져오지 않고(번들에서 빠진다) 레거시 틀도 넘기지 않으며, 패널·시트 값을 등록한다', () => {
  const app = fs.readFileSync(path.join(MOBILE_ROOT, 'App.tsx'), 'utf8');
  assert.match(app, /case 'settlement':\s*return <SettlementTab \/>/);
  assert.doesNotMatch(app, /screens\/SettlementScreen/);
  assert.doesNotMatch(app, /\blegacy=/);
  assert.match(app, /<SettlementMethodFacts card=\{s\.card\} \/>/);
  assert.match(app, /state\.key === 'settlement' \? \{ title: SETTLEMENT_PANEL_TITLE, body: <SettlementPanel \/> \}/);
  assert.match(app, /case 'smp':\s*return <SMPForecastScreen/);
  assert.match(app, /case 'bidding':\s*return <BiddingTab \/>/);
});
