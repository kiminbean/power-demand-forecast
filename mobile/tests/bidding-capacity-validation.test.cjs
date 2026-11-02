const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function harness(api) {
  const state = [];
  let cursor = 0;
  const React = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    useState(initial) {
      const index = cursor++;
      if (!(index in state)) state[index] = index === 19 ? [{ id: 'plant', name: '발전소', capacity: 1000, status: 'active', type: 'solar', installDate: '2020-01-01', contractType: 'ppa', location: { address: '' } }] : initial;
      return [state[index], value => { state[index] = typeof value === 'function' ? value(state[index]) : value; }];
    },
    useCallback: fn => fn,
    useEffect: () => {},
  };
  React.default = React;
  const native = { StyleSheet: { create: x => x }, Dimensions: { get: () => ({ width: 400 }) }, Platform: {}, Alert: { alert() {} } };
  for (const name of ['View', 'Text', 'ScrollView', 'TouchableOpacity', 'TextInput', 'RefreshControl', 'ActivityIndicator', 'Modal', 'Switch']) native[name] = name;
  const utils = { calculateEfficiency: () => 1, estimateDailyGeneration: () => 1, getEfficiencyStatus: () => ({}), formatCapacity: String, formatRevenue: String, mapWeatherCondition: () => 'clear' };
  const labels = { solar: {}, ppa: {}, south: {}, active: {}, maintenance: {}, paused: {} };
  const types = { PLANT_TYPE_LABELS: labels, CONTRACT_TYPE_LABELS: labels, ROOF_DIRECTION_LABELS: labels, PLANT_STATUS_LABELS: labels };
  const source = fs.readFileSync(path.join(__dirname, '../src/screens/BiddingScreen.tsx'), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports, console, require: name => name === 'react' ? React : name === 'react-native' ? native : name === '../services/api' ? { apiService: api } : name === '../utils/powerPlantUtils' ? utils : name === '../types/powerPlant' ? types : {} });
  function expand(node) {
    if (Array.isArray(node)) return node.map(expand);
    if (!node || typeof node !== 'object') return node;
    if (typeof node.type === 'function') return expand(node.type(node.props));
    return { ...node, props: { ...node.props, children: expand(node.props.children) } };
  }
  function render() { cursor = 0; return expand(module.exports.default({})); }
  function nodes(node, predicate) {
    if (Array.isArray(node)) return node.flatMap(n => nodes(n, predicate));
    if (!node || typeof node !== 'object') return [];
    return [...(predicate(node) ? [node] : []), ...nodes(node.props?.children, predicate)];
  }
  function text(node) {
    if (Array.isArray(node)) return node.map(text).join('');
    if (node && typeof node === 'object') return text(node.props.children);
    return node == null || typeof node === 'boolean' ? '' : String(node);
  }
  async function optimize(value) {
    let tree = render();
    const input = nodes(tree, n => n.type === 'TextInput' && n.props.placeholder === '50')[0];
    assert.ok(input, 'capacity input available');
    input.props.onChangeText(value);
    tree = render();
    const button = nodes(tree, n => n.type === 'TouchableOpacity' && text(n).includes('AI 최적화'))[0];
    assert.ok(button, 'optimization button available');
    await button.props.onPress();
    return render();
  }
  return { optimize, render, text };
}

const result = { model_used: 'test', hourly_bids: [{ hour: 12, segments: [{ quantity_mw: 1, price_krw_mwh: 100 }] }] };

test('invalid capacities show clear Korean guidance without requesting optimization or changing segments', async () => {
  for (const value of ['', ' ', '0', '-1', '500.01', '501', 'abc', '1abc', 'Infinity']) {
    let calls = 0;
    const h = harness({ getOptimizedSegments: async () => { calls++; return result; } });
    const before = h.text(h.render());
    const after = h.text(await h.optimize(value));
    assert.equal(calls, 0, `invalid value ${JSON.stringify(value)} must not reach API`);
    assert.match(after, /0\s*MW\s*초과/);
    assert.match(after, /500\s*MW\s*이하/);
    assert.doesNotMatch(after, /대체 알고리즘 사용/);
    assert.match(before, /50\.0/);
    assert.match(after, /50\.0/);
  }
});

test('valid boundaries and decimal values are sent without truncation', async () => {
  for (const [input, expected] of [['0.01', 0.01], ['50.5', 50.5], ['500', 500]]) {
    const calls = [];
    const h = harness({ getOptimizedSegments: async capacity => { calls.push(capacity); return result; } });
    const tree = await h.optimize(input);
    assert.deepEqual(calls, [expected]);
    assert.doesNotMatch(h.text(tree), /0\s*MW\s*초과/);
  }
});

test('invalid input after a successful optimization does not show a stale success badge', async () => {
  const h = harness({ getOptimizedSegments: async () => result });
  assert.match(h.text(await h.optimize('50')), /최적화 완료/);
  const output = h.text(await h.optimize('501'));
  assert.match(output, /500\s*MW\s*이하/);
  assert.doesNotMatch(output, /최적화 완료/);
});

test('failed API optimization falls back using the valid decimal capacity', async () => {
  const h = harness({ getOptimizedSegments: async () => { throw Error('offline'); } });
  const tree = await h.optimize('12.5');
  const output = h.text(tree);
  assert.match(output, /대체 알고리즘 사용/);
  assert.match(output, /12\.5/);
});
