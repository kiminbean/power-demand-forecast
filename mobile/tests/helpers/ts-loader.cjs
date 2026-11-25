'use strict';
// 테스트 도우미: TS·TSX 소스를 저장소에 이미 있는 typescript로 CommonJS로 바꾼 뒤 node:vm에서 실행한다.
// - 상대 경로 import를 재귀로 푼다(확장자 순서 .ts → .tsx → .js → .cjs → .json, 다음 index.*).
// - 플랫폼 파일은 기본으로 고르지 않는다: './router'는 router.ts로 풀리고 router.web.ts는 쓰지 않는다.
//   웹 파일을 시험할 때만 { platform: 'web' }을 준다.
// - react · react/jsx-runtime · react-native는 목으로 준다. 목이 없는 bare import는 바로 실패한다(조용히 {}를 주지 않는다).
// - 코드는 현재 realm에서 돈다(vm.compileFunction). 새 context를 쓰면 배열·객체의 프로토타입이 달라
//   assert.deepStrictEqual이 실패하므로, 전역 바꿔치기는 함수 매개변수로 넣는다(options.globals).
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

const MOBILE_ROOT = path.resolve(__dirname, '..', '..');
const SRC_ROOT = path.join(MOBILE_ROOT, 'src');
const EXTENSIONS = ['.ts', '.tsx', '.js', '.cjs', '.json'];
const PLATFORM_EXTENSIONS = ['.ts', '.tsx', '.js'];

function flattenStyle(style) {
  if (Array.isArray(style)) return Object.assign({}, ...style.map(flattenStyle));
  return style && typeof style === 'object' ? style : {};
}

// 정적 렌더용 React 목: 컴포넌트를 함수로 부르면 요소 트리를 돌려준다(훅은 첫 렌더 값만).
function createReactMock() {
  const toChildren = value => (value === undefined ? [] : [].concat(value));
  const createElement = (type, props, ...children) => ({ type, props: { ...(props || {}), children } });
  const jsx = (type, props) => ({ type, props: { ...(props || {}), children: toChildren(props && props.children) } });
  const React = {
    createElement,
    Fragment: 'Fragment',
    Component: class {},
    memo: component => component,
    forwardRef: render => props => render(props, null),
    createContext: value => ({ _currentValue: value, Provider: 'Provider', Consumer: 'Consumer' }),
    useContext: context => context._currentValue,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => {}],
    useReducer: (reducer, initial, init) => [init ? init(initial) : initial, () => {}],
    useMemo: factory => factory(),
    useCallback: callback => callback,
    useRef: value => ({ current: value }),
    useEffect: () => {},
    useLayoutEffect: () => {},
    useId: () => ':pd0:',
    useSyncExternalStore: (subscribe, getSnapshot) => getSnapshot(),
  };
  React.default = React;
  const runtime = { jsx, jsxs: jsx, Fragment: React.Fragment };
  return { React, runtime };
}

function createReactNativeMock() {
  const host = ['View', 'Text', 'Pressable', 'ScrollView', 'Image', 'TextInput', 'SafeAreaView', 'TouchableOpacity', 'ActivityIndicator'];
  const native = Object.fromEntries(host.map(name => [name, name]));
  return Object.assign(native, {
    // 'test'는 실제 OS 이름이 아니다: OS 분기가 웹 쪽(document 등)으로 새지 않게 한다.
    Platform: { OS: 'test', select: options => ('native' in options ? options.native : options.default) },
    StyleSheet: { create: styles => styles, flatten: flattenStyle, hairlineWidth: 1, absoluteFillObject: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0 } },
    AccessibilityInfo: {
      announceForAccessibility: () => {},
      isReduceMotionEnabled: async () => false,
      addEventListener: () => ({ remove: () => {} }),
    },
    Appearance: { getColorScheme: () => 'light', addChangeListener: () => ({ remove: () => {} }) },
    Linking: { openURL: async () => true },
    useWindowDimensions: () => ({ width: 390, height: 844, scale: 1, fontScale: 1 }),
    useColorScheme: () => 'light',
  });
}

/** 기본 목 묶음. 테스트는 { ...defaultMocks()['react-native'], 바꿀 것 }처럼 펼쳐 일부만 바꾼다. */
function defaultMocks() {
  const { React, runtime } = createReactMock();
  return { react: React, 'react/jsx-runtime': runtime, 'react-native': createReactNativeMock() };
}

function isFile(file) {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

function resolveFile(base, platform) {
  const candidates = [];
  if (path.extname(base) && EXTENSIONS.includes(path.extname(base))) candidates.push(base);
  for (const dir of [base, path.join(base, 'index')]) {
    if (platform) candidates.push(...PLATFORM_EXTENSIONS.map(ext => `${dir}.${platform}${ext}`));
    candidates.push(...EXTENSIONS.map(ext => `${dir}${ext}`));
  }
  return candidates.find(isFile) || null;
}

function transpile(file, source) {
  const result = ts.transpileModule(source, {
    fileName: file,
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
  });
  const errors = (result.diagnostics || []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) {
    const first = errors[0];
    const where = first.file && first.start !== undefined ? first.file.getLineAndCharacterOfPosition(first.start) : null;
    const message = ts.flattenDiagnosticMessageText(first.messageText, '\n');
    throw new SyntaxError(`ts-loader: ${path.relative(MOBILE_ROOT, file)}${where ? `:${where.line + 1}` : ''} ${message}`);
  }
  return result.outputText;
}

/**
 * 모듈 캐시를 가진 로더를 만든다. 테스트마다 새 로더를 쓰면 모듈 상태가 섞이지 않는다.
 * @param {{ mocks?: Record<string, unknown>, globals?: Record<string, unknown>, platform?: string }} [options]
 */
function createTsLoader(options = {}) {
  const mocks = { ...defaultMocks(), ...(options.mocks || {}) };
  // process는 실제 환경 변수가 새지 않게 빈 env로 덮는다. __DEV__는 RN 전역이다.
  const globals = { process: { env: {} }, __DEV__: false, ...(options.globals || {}) };
  const globalNames = Object.keys(globals);
  const cache = new Map();

  function load(file) {
    const resolved = resolveFile(path.resolve(file), options.platform);
    if (!resolved) throw new Error(`ts-loader: file not found: ${file}`);
    return execute(resolved);
  }

  function requireFrom(parentFile, specifier) {
    if (typeof specifier !== 'string' || !specifier.trim()) {
      throw new TypeError(`ts-loader: invalid import in ${path.relative(MOBILE_ROOT, parentFile)}`);
    }
    if (Object.prototype.hasOwnProperty.call(mocks, specifier)) return mocks[specifier];
    let base = null;
    if (specifier.startsWith('./') || specifier.startsWith('../')) base = path.resolve(path.dirname(parentFile), specifier);
    else if (specifier.startsWith('@/')) base = path.join(SRC_ROOT, specifier.slice(2));
    if (!base) {
      throw new Error(`ts-loader: no mock for '${specifier}' (imported from ${path.relative(MOBILE_ROOT, parentFile)})`);
    }
    const resolved = resolveFile(base, options.platform);
    if (!resolved) {
      throw new Error(`ts-loader: cannot resolve '${specifier}' from ${path.relative(MOBILE_ROOT, parentFile)}`);
    }
    return execute(resolved);
  }

  function execute(file) {
    if (cache.has(file)) return cache.get(file).exports;
    const source = fs.readFileSync(file, 'utf8');
    const module = { exports: {} };
    // 순환 import는 CommonJS처럼 채우는 중인 exports를 돌려준다.
    cache.set(file, module);
    if (file.endsWith('.json')) {
      module.exports = JSON.parse(source);
      return module.exports;
    }
    const code = file.endsWith('.cjs') ? source : transpile(file, source);
    const params = ['exports', 'require', 'module', '__filename', '__dirname', ...globalNames];
    const fn = vm.compileFunction(code, params, { filename: file });
    const localRequire = specifier => requireFrom(file, specifier);
    fn(module.exports, localRequire, module, file, path.dirname(file), ...globalNames.map(name => globals[name]));
    return module.exports;
  }

  return { load, cache };
}

/** 한 번 쓰고 버리는 로더로 파일 하나를 불러온다. */
function loadTs(file, options) {
  return createTsLoader(options).load(file);
}

module.exports = { createTsLoader, loadTs, defaultMocks, MOBILE_ROOT, SRC_ROOT };
