const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

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
  const themeCode = ts.transpileModule(themeTs, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;
  const themeModule = { exports: {} };
  vm.runInNewContext(themeCode, { module: themeModule, exports: themeModule.exports });
  const source = fs.readFileSync(path.join(__dirname, '../App.tsx'), 'utf8');
  const code = ts.transpileModule(source + '\nexport { Header };', {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, {
    module, exports: module.exports,
    require: name => name === 'react' ? React : name === 'react-native' ? native : name.endsWith('src/theme/colors') ? themeModule.exports : {},
  });
  return module.exports.Header;
}

function nodes(node) {
  if (Array.isArray(node)) return node.flatMap(nodes);
  if (!node || typeof node !== 'object') return [];
  return [node, ...nodes(node.props.children)];
}

test('main and simulation headers stay neutral while preserving navigation', () => {
  const Header = loadHeader();
  let backCalls = 0;
  for (const props of [{}, { title: 'DAM 시뮬레이션', showBack: true, onBack: () => backCalls++ }]) {
    const tree = nodes(Header(props));
    const labels = tree.filter(node => node.type === 'Text').flatMap(node => node.props.children).join(' ');
    assert.doesNotMatch(labels, /\b(live|online|connected)\b|실시간|연결됨/i);
    assert.match(labels, /v6\.1/);
    if (props.showBack) {
      assert.match(labels, /DAM 시뮬레이션/);
      tree.find(node => node.type === 'TouchableOpacity').props.onPress();
    } else {
      assert.match(labels, /eXeco/);
    }
  }
  assert.equal(backCalls, 1);
});
