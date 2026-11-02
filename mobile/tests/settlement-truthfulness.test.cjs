const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function harness(api) {
  let state = [], cursor = 0, effect, initialized = false;
  const React = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    useState(initial) { const i = cursor++; if (!(i in state)) state[i] = initial; return [state[i], value => { state[i] = typeof value === 'function' ? value(state[i]) : value; }]; },
    useCallback: fn => fn,
    useEffect: fn => { if (!initialized) effect = fn; },
  };
  React.default = React;
  const native = { StyleSheet: { create: x => x }, Dimensions: { get: () => ({ width: 400 }) }, Platform: {} };
  for (const name of ['View', 'Text', 'ScrollView', 'TouchableOpacity', 'RefreshControl', 'ActivityIndicator']) native[name] = name;
  const filename = path.join(__dirname, '../src/screens/SettlementScreen.tsx');
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports, console, require: name => name === 'react' ? React : name === 'react-native' ? native : { apiService: api } });
  const component = module.exports.default;
  function render() {
    cursor = 0;
    function expand(node) {
      if (node == null || typeof node !== 'object') return node;
      if (Array.isArray(node)) return node.map(expand);
      if (typeof node.type === 'function') return expand(node.type(node.props));
      return { ...node, props: { ...node.props, children: expand(node.props.children) } };
    }
    return expand(component());
  }
  function all(node, predicate, found = []) {
    if (Array.isArray(node)) { node.forEach(x => all(x, predicate, found)); return found; }
    if (node && typeof node === 'object') { if (predicate(node)) found.push(node); all(node.props?.children, predicate, found); }
    return found;
  }
  function text(tree) { return all(tree, x => x.type === 'Text').map(x => flatten(x.props.children)).join(' '); }
  function flatten(node) { return Array.isArray(node) ? node.map(flatten).join('') : typeof node === 'object' && node ? flatten(node.props.children) : node == null || typeof node === 'boolean' ? '' : String(node); }
  return { render, text, all, start() { render(); initialized = true; effect(); }, async flush() { await new Promise(resolve => setImmediate(resolve)); } };
}
const summary = { generation_revenue_million: 42, imbalance_charges_million: -2, forecast_accuracy_pct: 97, avg_da_smp: 100, avg_rt_smp: 99 };
const record = { date: '2026-01-02', generation_mwh: 10, revenue_million: 42, imbalance_million: -2, net_revenue_million: 40, accuracy_pct: 97 };
const service = (records, stats) => ({ getRecentSettlements: async () => records, getSettlementSummary: async () => stats });

test('loading has no invented financial figures', () => {
  const h = harness({ getRecentSettlements: () => new Promise(() => {}), getSettlementSummary: () => new Promise(() => {}) });
  h.start(); const tree = h.render();
  assert.ok(h.all(tree, x => x.type === 'ActivityIndicator').length);
  assert.doesNotMatch(h.text(tree), /1251|85\.2|82\.5|45\.2/);
});

test('failed initial request shows error and retry without invented values', async () => {
  let calls = 0;
  const h = harness({ getRecentSettlements: async () => { calls++; if (calls === 1) throw Error('offline'); return [record]; }, getSettlementSummary: async () => summary });
  h.start(); await h.flush(); let tree = h.render();
  assert.match(h.text(tree), /불러오지 못|오류/);
  assert.doesNotMatch(h.text(tree), /1251|85\.2|82\.5|45\.2/);
  const button = h.all(tree, x => x.type === 'TouchableOpacity' && h.text(x).includes('다시 시도'))[0];
  assert.ok(button); button.props.onPress(); await h.flush(); tree = h.render();
  assert.match(h.text(tree), /42/);
});

test('empty and missing fields do not produce sample figures or NaN', async () => {
  for (const [records, stats] of [[[], {}], [[{ date: '2026-01-02' }], {}], [null, null]]) {
    const h = harness(service(records, stats)); h.start(); await h.flush(); const tree = h.render();
    assert.doesNotMatch(h.text(tree), /1251|85\.2|82\.5|94\.5|NaN|45\.2/);
    assert.match(h.text(tree), /없|—|-/);
    assert.equal(h.all(tree, x => x.props?.style && Array.isArray(x.props.style) && x.props.style.some(y => y && typeof y.height === 'number')).length, 0);
  }
});

test('successful data is shown and failed refresh is clearly marked', async () => {
  let fail = false;
  const h = harness({ getRecentSettlements: async () => { if (fail) throw Error('offline'); return [record]; }, getSettlementSummary: async () => summary });
  h.start(); await h.flush(); let tree = h.render();
  assert.match(h.text(tree), /42/); assert.match(h.text(tree), /100/);
  const expand = h.all(tree, x => x.type === 'TouchableOpacity' && h.text(x).includes('거래 내역'))[0]; expand.props.onPress(); tree = h.render();
  assert.match(h.text(tree), /42\.0/);
  fail = true; tree.props.refreshControl.props.onRefresh(); await h.flush(); tree = h.render();
  assert.match(h.text(tree), /불러오지 못|오류/);
  assert.doesNotMatch(h.text(tree), /1251|85\.2|82\.5|45\.2/);
});
