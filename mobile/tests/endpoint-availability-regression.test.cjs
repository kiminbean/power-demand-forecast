const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const { loadTheme } = require('./theme-loader.cjs');
const themeExports = loadTheme();

function screen(name, api, stored = null) {
  const state = [];
  let cursor = 0;
  const effects = [];
  const React = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    useState(initial) {
      const index = cursor++;
      if (!(index in state)) state[index] = initial;
      return [state[index], value => { state[index] = typeof value === 'function' ? value(state[index]) : value; }];
    },
    useCallback: fn => fn,
    useEffect: fn => effects.push(fn),
  };
  React.default = React;
  const native = { StyleSheet: { create: x => x }, Dimensions: { get: () => ({ width: 400 }) }, Platform: { OS: 'web' }, Alert: { alert() {} } };
  for (const n of ['View', 'Text', 'ScrollView', 'TouchableOpacity', 'TextInput', 'RefreshControl', 'ActivityIndicator', 'Modal', 'Switch']) native[n] = n;
  const utils = { calculateEfficiency: () => 1, estimateDailyGeneration: () => 1, getEfficiencyStatus: () => ({}), formatCapacity: String, formatRevenue: String, mapWeatherCondition: () => 'clear' };
  const labels = { solar: {}, ppa: {}, south: {}, active: {}, maintenance: {}, paused: {} };
  const types = { PLANT_TYPE_LABELS: labels, CONTRACT_TYPE_LABELS: labels, ROOF_DIRECTION_LABELS: labels, PLANT_STATUS_LABELS: labels };
  const source = fs.readFileSync(path.join(__dirname, '../src/screens', `${name}.tsx`), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports, console, Error, localStorage: { getItem: () => stored, setItem() {} }, require: dep => dep === 'react' ? React : dep === 'react-native' ? native : dep === '../services/api' ? { apiService: api } : dep === '../utils/powerPlantUtils' ? utils : dep === '../types/powerPlant' ? types : dep === './PowerPlantRegistrationScreen' ? { default: 'PowerPlantRegistrationScreen' } : dep.endsWith('theme/colors') ? themeExports : {} });
  function expand(node) {
    if (Array.isArray(node)) return node.map(expand);
    if (!node || typeof node !== 'object') return node;
    if (typeof node.type === 'function') return expand(node.type(node.props));
    return { ...node, props: { ...node.props, children: expand(node.props.children) } };
  }
  function render() { cursor = 0; effects.length = 0; return expand(module.exports.default({})); }
  function nodes(node, predicate) {
    if (Array.isArray(node)) return node.flatMap(n => nodes(n, predicate));
    if (!node || typeof node !== 'object') return [];
    return [...(predicate(node) ? [node] : []), ...nodes(node.props?.children, predicate)];
  }
  function text(node) {
    if (Array.isArray(node)) return node.map(text).join(' ');
    if (node && typeof node === 'object') return text(node.props.children);
    return node == null || typeof node === 'boolean' ? '' : String(node);
  }
  function start() {
    render();
    const effect = effects.find(fn => fn.toString().includes(name === 'BiddingScreen' ? 'loadPowerPlants' : 'fetchSettlementData'));
    assert.ok(effect, 'screen load effect');
    effect();
  }
  function retry(tree) {
    const button = nodes(tree, n => n.type === 'TouchableOpacity' && text(n).includes('다시 시도'))[0];
    assert.ok(button, 'retry available');
    button.props.onPress();
  }
  function savePlant(tree, plant) {
    const form = nodes(tree, n => typeof n.props?.onSave === 'function')[0];
    assert.ok(form, 'plant registration form');
    form.props.onSave(plant);
  }
  return { render, text, start, retry, savePlant, flush: () => new Promise(resolve => setImmediate(resolve)) };
}
const unavailable = () => { throw Error('API Error: 404'); };
const offline = () => { throw Error('API Error: 503'); };
const plant = { id: 'p', name: '개인 발전소', capacity: 500, status: 'active', type: 'solar', installDate: '2020-01-01', location: {} };

test('settlement 404 is unavailable, not empty or a permission assertion, and can retry', async () => {
  let fail = true;
  const h = screen('SettlementScreen', { getRecentSettlements: async () => fail ? unavailable() : [], getSettlementSummary: async () => ({}) });
  h.start(); await h.flush(); let tree = h.render();
  assert.match(h.text(tree), /정산.*이용할 수 없/);
  assert.doesNotMatch(h.text(tree), /정산 내역이 없습니다|권한/);
  fail = false; h.retry(tree); await h.flush(); tree = h.render();
  assert.match(h.text(tree), /정산 내역이 없습니다/);
});

test('settlement non-404 failures remain retryable errors', async () => {
  const h = screen('SettlementScreen', { getRecentSettlements: async () => offline(), getSettlementSummary: async () => ({}) });
  h.start(); await h.flush(); const tree = h.render();
  assert.match(h.text(tree), /정산 정보를 불러오지 못/);
  assert.doesNotMatch(h.text(tree), /정산 내역이 없습니다/);
  h.retry(tree);
});

test('plant 404 is unavailable, not empty or a permission assertion, and retry can load private data', async () => {
  let fail = true;
  const h = screen('BiddingScreen', { getPowerPlants: async () => fail ? unavailable() : [plant] });
  h.start(); await h.flush(); let tree = h.render();
  assert.match(h.text(tree), /발전소.*이용할 수 없/);
  assert.doesNotMatch(h.text(tree), /등록된 발전소가 없습니다|권한/);
  fail = false; h.retry(tree); await h.flush(); tree = h.render();
  assert.match(h.text(tree), /개인 발전소/);
  assert.doesNotMatch(h.text(tree), /발전소.*이용할 수 없/);
});

test('successful empty plant response alone reports no plants', async () => {
  const h = screen('BiddingScreen', { getPowerPlants: async () => [] });
  h.start(); await h.flush(); assert.match(h.text(h.render()), /등록된 발전소가 없습니다/);
});

test('plant created during an outage is labeled locally stored', async () => {
  const h = screen('BiddingScreen', { getPowerPlants: async () => unavailable() });
  h.start(); await h.flush(); h.savePlant(h.render(), plant);
  const tree = h.render();
  assert.match(h.text(tree), /개인 발전소/);
  assert.match(h.text(tree), /기기.*저장/);
});

test('failed plant request is retryable and any cached plants are labeled local', async () => {
  for (const cached of [null, JSON.stringify([plant])]) {
    const h = screen('BiddingScreen', { getPowerPlants: async () => offline() }, cached);
    h.start(); await h.flush(); const tree = h.render();
    assert.match(h.text(tree), /발전소.*불러오지 못/);
    assert.doesNotMatch(h.text(tree), /등록된 발전소가 없습니다/);
    if (cached) { assert.match(h.text(tree), /개인 발전소/); assert.match(h.text(tree), /기기.*저장|로컬/); }
    h.retry(tree);
  }
});
