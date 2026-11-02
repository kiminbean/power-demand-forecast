const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Reuse the existing screen harness, opening the market for review interactions.
const harnessSource = fs.readFileSync(path.join(__dirname, 'bidding-capacity-validation.test.cjs'), 'utf8')
  .split('\nconst result =')[0]
  .replace("index === 19 ?", "index === 9 ? 'open' : index === 19 ?")
  .replace('    useCallback: fn => fn,', '    useRef: initial => { const index = cursor++; if (!(index in state)) state[index] = { current: initial }; return state[index]; },\n    useCallback: fn => fn,')
  .replace('  return { optimize, render, text };', '  return { optimize, render, text, nodes };');
const harnessModule = { exports: {} };
vm.runInThisContext(`(function(require, module, __dirname) { ${harnessSource}\nmodule.exports = harness; })`)(require, harnessModule, __dirname);
const harness = harnessModule.exports;

function input(h, value) {
  const field = h.nodes(h.render(), n => n.type === 'TextInput' && n.props.placeholder === '50')[0];
  field.props.onChangeText(value);
}
function press(h, label) {
  const button = h.nodes(h.render(), n => n.type === 'TouchableOpacity' && h.text(n) === label)[0];
  assert.ok(button, `button ${label} available`);
  return button.props.onPress();
}
function reviewVisible(h) {
  return h.nodes(h.render(), n => n.type === 'Modal')[0].props.visible;
}
const response = capacity => ({ model_used: 'test', hourly_bids: [{ hour: 12, segments: [{ quantity_mw: capacity, price_krw_mwh: 100 }] }] });

test('review rejects invalid capacity and displays validation guidance', () => {
  for (const value of ['', '0', '501', 'abc', 'Infinity']) {
    const h = harness({ getOptimizedSegments: async () => response(50) });
    input(h, value);
    press(h, '입찰 제출');
    assert.equal(reviewVisible(h), false, value);
    assert.match(h.text(h.render()), /0\s*MW\s*초과/);
  }
});

test('review cannot use segments after capacity changes until reoptimized', async () => {
  const h = harness({ getOptimizedSegments: async capacity => response(capacity) });
  await h.optimize('50');
  input(h, '40');
  assert.doesNotMatch(h.text(h.render()), /최적화 완료/);
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), false);
  assert.match(h.text(h.render()), /최적화.*다시/);
  await h.optimize('40');
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), true);
});

test('review blocks segments exceeding capacity and permits corrected manual quantities', () => {
  const h = harness({});
  const fields = h.nodes(h.render(), n => n.type === 'TextInput' && n.props.selectTextOnFocus);
  fields[0].props.onChangeText('6');
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), false);
  assert.match(h.text(h.render()), /입찰량.*용량/);
  h.nodes(h.render(), n => n.type === 'TextInput' && n.props.selectTextOnFocus)[0].props.onChangeText('5');
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), true);
});

test('late optimization success or fallback cannot restore stale segments after an edit', async () => {
  for (const reject of [false, true]) {
    let finish;
    const pending = new Promise((resolve, rejectPromise) => { finish = reject ? rejectPromise : resolve; });
    const h = harness({ getOptimizedSegments: () => pending });
    const optimizing = h.optimize('50');
    input(h, '40');
    finish(reject ? Error('offline') : response(50));
    await optimizing;
    assert.doesNotMatch(h.text(h.render()), /최적화 완료|대체 알고리즘 사용/);
    press(h, '입찰 제출');
    assert.equal(reviewVisible(h), false);
    assert.match(h.text(h.render()), /최적화.*다시/);
  }
});

test('review permits valid decimal capacity with matching optimized segments', async () => {
  const h = harness({ getOptimizedSegments: async capacity => response(capacity) });
  await h.optimize('12.5');
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), true);
});

test('submission revalidates if capacity changes while review is open', () => {
  const h = harness({});
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), true);
  input(h, '0');
  press(h, '제출');
  assert.equal(reviewVisible(h), false);
  assert.match(h.text(h.render()), /0\s*MW\s*초과/);
  assert.doesNotMatch(h.text(h.render()), /입찰 검증 중/);
});

test('review rejects an optimized segment total exceeding capacity', async () => {
  const h = harness({ getOptimizedSegments: async () => response(51) });
  await h.optimize('50');
  press(h, '입찰 제출');
  assert.equal(reviewVisible(h), false);
  assert.match(h.text(h.render()), /입찰량.*용량/);
});
