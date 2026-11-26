'use strict';
// 헤더 팔레트 테스트(명세 10절, F6). 기존 헤더 회귀 테스트와 같은 VM 방식(App.tsx + `export { Header };`, react · react-native의
// View·Text·TouchableOpacity·StyleSheet · src/theme/colors만 진짜, 나머지 가져오기는 {})으로 돌린다. 셸이 넘기는 palette·fonts·
// refresh·onHome을 줘도 기존 단언이 모두 성립하는지, 색이 그대로 칠해지는지, App.tsx에 적은 크기 숫자가 토큰과 같은지 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const { createTsLoader, SRC_ROOT } = require('./helpers/ts-loader.cjs');

function loadHeader() {
  const React = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    Component: class {},
  };
  React.default = React;
  const native = {
    View: 'View', Text: 'Text', TouchableOpacity: 'TouchableOpacity',
    SafeAreaView: 'SafeAreaView', Platform: {}, StyleSheet: { create: styles => styles },
  };
  const themeTs = fs.readFileSync(path.join(__dirname, '../src/theme/colors.ts'), 'utf8');
  const themeCode = ts.transpileModule(themeTs, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const themeModule = { exports: {} };
  vm.runInNewContext(themeCode, { module: themeModule, exports: themeModule.exports });
  const source = fs.readFileSync(path.join(__dirname, '../App.tsx'), 'utf8');
  const code = ts.transpileModule(source + '\nexport { Header };', {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, {
    module, exports: module.exports,
    require: name => (name === 'react' ? React : name === 'react-native' ? native : name.endsWith('src/theme/colors') ? themeModule.exports : {}),
  });
  return { Header: module.exports.Header, colors: themeModule.exports.colors };
}

const nodes = node => {
  if (Array.isArray(node)) return node.flatMap(nodes);
  if (!node || typeof node !== 'object') return [];
  return [node, ...nodes(node.props.children)];
};
const flat = style => (Array.isArray(style) ? Object.assign({}, ...style.map(flat)) : style || {});
const texts = tree => tree.filter(n => n.type === 'Text');
const labelsOf = tree => texts(tree).flatMap(n => n.props.children).join(' ');

const loader = createTsLoader({ mocks: { 'react-native-svg': { __esModule: true, default: 'Svg', Path: 'Path', Circle: 'Circle' } } });
const { THEMES, LAYOUT, HIT, SIZE, SPACE } = loader.load(path.join(SRC_ROOT, 'ui/tokens.ts'));
const { textStyle } = loader.load(path.join(SRC_ROOT, 'ui/Text.tsx'));
const { APP_VERSION, REFRESH_NONE_REASON } = loader.load(path.join(SRC_ROOT, 'content/strings.ts'));

// 셸(Shell.tsx)이 넘기는 것과 같은 모양: dark 테마 팔레트와 토큰 글자 꼴.
const dark = THEMES.dark.color;
const PALETTE = { background: dark.surface, border: dark.border, wordmark: dark.text1, title: dark.text1, version: dark.text3, icon: dark.text2 };
const FONTS = { wordmark: textStyle('title', 390, '700'), title: textStyle('body', 390, '700'), version: textStyle('meta', 390, '500') };

test('palette·fonts·refresh를 줘도 기존 헤더 단언이 모두 성립한다(중립 글자, v6.1, eXeco/제목, 첫 TouchableOpacity = 뒤로)', () => {
  const { Header } = loadHeader();
  let backCalls = 0;
  const refresh = { onPress: () => {}, busy: false, disabledReason: null };
  for (const props of [{}, { title: 'DAM 시뮬레이션', showBack: true, onBack: () => backCalls++ }]) {
    const tree = nodes(Header({ ...props, palette: PALETTE, fonts: FONTS, refresh, onHome: () => {} }));
    const labels = labelsOf(tree);
    assert.doesNotMatch(labels, /\b(live|online|connected)\b|실시간|연결됨/i);
    assert.match(labels, /v6\.1/);
    if (props.showBack) {
      assert.match(labels, /DAM 시뮬레이션/);
      tree.find(node => node.type === 'TouchableOpacity').props.onPress();
    } else {
      assert.match(labels, /eXeco/);
      assert.equal(tree.filter(node => node.type === 'TouchableOpacity').length, 0, '첫 화면에는 뒤로 버튼이 없다');
    }
  }
  assert.equal(backCalls, 1);
});

test('색: palette가 헤더 바탕·선·워드마크·버전·제목·아이콘에 칠해지고, 없으면 colors.ts의 옛 색', () => {
  const { Header, colors } = loadHeader();
  const main = nodes(Header({ palette: PALETTE, fonts: FONTS }));
  const bar = main[0];
  assert.equal(bar.props.role, 'banner');
  assert.equal(flat(bar.props.style).backgroundColor, PALETTE.background);
  assert.equal(flat(bar.props.style).borderBottomColor, PALETTE.border);
  const [wordmark, version] = texts(main);
  assert.equal(flat(wordmark.props.style).color, PALETTE.wordmark);
  assert.equal(flat(version.props.style).color, PALETTE.version);
  const sub = nodes(Header({ title: '용어 사전', showBack: true, palette: PALETTE, fonts: FONTS }));
  const [title] = texts(sub);
  assert.equal(flat(title.props.style).color, PALETTE.title);
  assert.equal(sub.find(n => n.props.name === 'back').props.color, PALETTE.icon);

  const plain = nodes(Header({}));
  assert.equal(flat(plain[0].props.style).backgroundColor, colors.surface);
  assert.equal(flat(plain[0].props.style).borderBottomColor, colors.border.primary);
  assert.deepStrictEqual(texts(plain).map(n => flat(n.props.style).color), [colors.brand.navy, colors.text.muted]);
});

test('글자 꼴: 셸이 넘긴 토큰 꼴(워드마크 20/700, 제목 15/700, 버전 12/500, tabular-nums)이 그대로 쓰인다', () => {
  const { Header } = loadHeader();
  const [wordmark, version] = texts(nodes(Header({ palette: PALETTE, fonts: FONTS })));
  assert.deepStrictEqual([flat(wordmark.props.style).fontSize, flat(wordmark.props.style).fontWeight], [20, '700']);
  assert.deepStrictEqual([flat(version.props.style).fontSize, flat(version.props.style).fontWeight], [12, '500']);
  assert.deepStrictEqual(flat(version.props.style).fontVariant, ['tabular-nums']);
  const [title] = texts(nodes(Header({ title: '둘러보기', showBack: true, fonts: FONTS })));
  assert.deepStrictEqual([flat(title.props.style).fontSize, flat(title.props.style).fontWeight], [15, '700']);
});

test('누름: 워드마크 → onHome(이름 "eXeco 처음 화면"), 뒤로 이름 "뒤로", 새로고침은 두 상태 모두 오른쪽 끝에 그대로 넘어간다', () => {
  const { Header } = loadHeader();
  let home = 0;
  const refresh = { onPress: () => {}, busy: true, disabledReason: REFRESH_NONE_REASON };
  const main = nodes(Header({ onHome: () => home++, refresh, palette: PALETTE }));
  const mark = main.find(n => n.props['aria-label'] === 'eXeco 처음 화면');
  assert.equal(mark.props.role, 'button');
  mark.props.onPress();
  assert.equal(home, 1);
  const sub = nodes(Header({ title: '용어 사전', showBack: true, onBack: () => {}, refresh, palette: PALETTE }));
  const back = sub.find(n => n.type === 'TouchableOpacity');
  assert.deepStrictEqual([back.props.role, back.props['aria-label']], ['button', '뒤로']);
  for (const tree of [main, sub]) {
    const right = tree[0].props.children[1];
    const last = right.props.children[right.props.children.length - 1];
    assert.deepStrictEqual(last.props, { onPress: refresh.onPress, busy: true, disabledReason: REFRESH_NONE_REASON, children: [] });
  }
  assert.equal(labelsOf(sub).includes('v6.1'), true, '하위 상태에서도 v6.1은 새로고침 앞에 있다');
});

test('App.tsx에 적은 숫자 = 토큰: 높이 LAYOUT.header, 누름 상자 HIT, 아이콘 SIZE.icon, 여백 SPACE, 버전 = APP_VERSION', () => {
  const { Header } = loadHeader();
  const main = nodes(Header({}));
  const bar = flat(main[0].props.style);
  assert.equal(bar.height, LAYOUT.header);
  assert.equal(bar.paddingHorizontal, SPACE.xs);
  const mark = flat(main.find(n => n.props['aria-label'] === 'eXeco 처음 화면').props.style);
  assert.deepStrictEqual([mark.minHeight, mark.minWidth], [HIT, HIT]);
  assert.equal(bar.paddingHorizontal + mark.paddingHorizontal, LAYOUT.gutter, '워드마크 글자는 화면 끝에서 16');
  const sub = nodes(Header({ title: '용어 사전', showBack: true }));
  const back = flat(sub.find(n => n.type === 'TouchableOpacity').props.style);
  assert.deepStrictEqual([back.width, back.height], [HIT, HIT]);
  assert.equal(sub.find(n => n.props.name === 'back').props.size, SIZE.icon);
  assert.equal(texts(main)[1].props.children.join(''), APP_VERSION);
});
