// tests/helpers/ts-loader.cjs가 뒤 단계 테스트(kst·format·provenance·view-models 등)의 바탕이므로 먼저 스스로 검증한다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, loadTs, defaultMocks } = require('./helpers/ts-loader.cjs');

const FIXTURES = path.join(__dirname, 'fixtures', 'ts-loader');

test('ts-loader resolves relative imports, erases type-only imports and reads JSON', () => {
  const main = loadTs(path.join(FIXTURES, 'main.ts'));
  assert.equal(main.sum, 5);
  assert.equal(main.jsonValue, 42);
  assert.equal(main.area({ w: 3, h: 4 }), 12);
  // 같은 realm에서 돌므로 배열 비교가 프로토타입 차이로 깨지지 않는다.
  assert.deepStrictEqual(main.list, [2, 3, 4]);
});

test('ts-loader picks the plain .ts platform file unless a platform is requested', () => {
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts')).picked, 'default');
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts'), { platform: 'web' }).picked, 'web');
});

test('ts-loader mocks react-native and lets a test replace one mock', () => {
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts')).os, 'test');
  const native = { ...defaultMocks()['react-native'], Platform: { OS: 'web', select: o => o.web } };
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts'), { mocks: { 'react-native': native } }).os, 'web');
});

test('ts-loader injects globals as parameters without leaking real environment', () => {
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts')).injected, 'absent');
  assert.equal(loadTs(path.join(FIXTURES, 'main.ts'), { globals: { PD_TEST_VALUE: 'given' } }).injected, 'given');
});

test('ts-loader renders TSX through the automatic JSX runtime mock', () => {
  const { Hello } = loadTs(path.join(FIXTURES, 'view.tsx'));
  const tree = Hello({ name: '제주' });
  assert.equal(tree.type, 'View');
  assert.equal(tree.props.accessibilityRole, 'header');
  const texts = tree.props.children.map(child => [child.type, ...child.props.children]);
  assert.deepStrictEqual(texts, [['Text', 'hello'], ['Text', '제주']]);
});

test('ts-loader fails fast on a bare import without a mock', () => {
  assert.throws(() => loadTs(path.join(FIXTURES, 'bad-import.ts')), /no mock for 'axios'/);
});

test('ts-loader fails fast on a missing file and keeps one module instance per loader', () => {
  assert.throws(() => loadTs(path.join(FIXTURES, 'does-not-exist.ts')), /file not found/);
  const loader = createTsLoader();
  const first = loader.load(path.join(FIXTURES, 'math.ts'));
  assert.equal(loader.load(path.join(FIXTURES, 'math')), first);
  assert.notEqual(createTsLoader().load(path.join(FIXTURES, 'math.ts')), first);
});
