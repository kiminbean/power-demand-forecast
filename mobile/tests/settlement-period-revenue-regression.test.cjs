const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

async function renderSettlement(records) {
  let state = [], cursor = 0, effect, initialized = false, requestedDays;
  const React = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    useState(initial) { const i = cursor++; if (!(i in state)) state[i] = initial; return [state[i], value => { state[i] = value; }]; },
    useCallback: fn => fn,
    useEffect: fn => { if (!initialized) effect = fn; },
  };
  React.default = React;
  const native = { StyleSheet: { create: x => x }, Dimensions: { get: () => ({ width: 400 }) }, Platform: {} };
  for (const name of ['View', 'Text', 'ScrollView', 'TouchableOpacity', 'RefreshControl', 'ActivityIndicator']) native[name] = name;
  const apiService = { getRecentSettlements: async days => { requestedDays = days; return records; }, getSettlementSummary: async () => ({}) };
  const source = fs.readFileSync(path.join(__dirname, '../src/screens/SettlementScreen.tsx'), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports, require: name => name === 'react' ? React : name === 'react-native' ? native : { apiService } });
  function expand(node) {
    if (Array.isArray(node)) return node.map(expand);
    if (!node || typeof node !== 'object') return node;
    if (typeof node.type === 'function') return expand(node.type(node.props));
    return { ...node, props: { ...node.props, children: expand(node.props.children) } };
  }
  function render() { cursor = 0; return expand(module.exports.default()); }
  render(); initialized = true; effect(); await new Promise(resolve => setImmediate(resolve));
  const tree = render();
  function nodes(node, predicate) {
    if (Array.isArray(node)) return node.flatMap(x => nodes(x, predicate));
    if (!node || typeof node !== 'object') return [];
    return [...(predicate(node) ? [node] : []), ...nodes(node.props?.children, predicate)];
  }
  function text(node) {
    if (Array.isArray(node)) return node.map(text).join('');
    if (node && typeof node === 'object') return text(node.props.children);
    return node == null || typeof node === 'boolean' ? '' : String(node);
  }
  return { tree, nodes, text, requestedDays };
}

const records = [
  { date: '2026-01-02', net_revenue_million: -12.5 },
  { date: '2026-01-03', net_revenue_million: 8 },
  { date: '2026-01-04', net_revenue_million: 0 },
];

test('period is a fixed seven-day description rather than nonfunctional buttons', async () => {
  const h = await renderSettlement(records);
  assert.equal(h.requestedDays, 7);
  assert.match(h.text(h.tree), /최근 7일/);
  const buttons = h.nodes(h.tree, n => n.type === 'TouchableOpacity');
  assert.ok(buttons.every(button => !/1주|1개월|분기/.test(h.text(button))));
});

test('daily net revenue displays signed amounts without misleading bars', async () => {
  const h = await renderSettlement(records);
  const text = h.text(h.tree);
  assert.match(text, /01\/02[^\n]*-12\.5/);
  assert.match(text, /01\/03[^\n]*\+8/);
  assert.match(text, /01\/04[^\n]*0/);
  assert.equal(h.nodes(h.tree, n => Array.isArray(n.props?.style) && n.props.style.some(s => s && typeof s.height === 'number')).length, 0);
});
