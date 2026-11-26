// 설계 명세 7.10절 금지 목록을 "새 코드"에만 적용한다(레거시 화면은 범위 밖).
// 범위: App.tsx, src/{app,ui,lib,data,content,features}/**, src/screens/SMPForecastScreen.tsx, public/**.
// App.tsx와 SMPForecastScreen.tsx는 같은 자리에서 다시 쓰는 파일(F6·F7)이라, round 0 바이트(sha256)와 같은 동안은
// 레거시로 보고 한 바이트라도 바뀌면 자동으로 범위에 넣는다. 그래서 F0 시점의 범위는 공집합이다.
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const ROOT = path.resolve(__dirname, '..');
const NEW_DIRS = ['src/app', 'src/ui', 'src/lib', 'src/data', 'src/content', 'src/features', 'public'];
const REWRITTEN_IN_PLACE = {
  'App.tsx': '41e5870a0910866b5a8ae880c18d3436f5156890dbb8ed3e509c728c6a521dfd',
  'src/screens/SMPForecastScreen.tsx': 'a6fc68f89e5e1c23b8065e35e3818eccbbfb8be27e746ddbd62ee79a808c3784',
};
// 12절 F6 행: "이 단계에서는 입찰·정산 경로가 레거시 화면을 그대로 띄운다". 그 가져오기만 App.tsx에 잠시 허용했다.
// F8이 입찰을 새 탭으로 바꾸며 BiddingScreen을, F9가 정산을 새 탭으로 바꾸며 SettlementScreen을 지워 이제 비어 있다.
// 다시 항목을 넣으면 아래 테스트가 그 항목이 아직 쓰이는지 본다(낡은 허용이 남지 않게).
const TRANSITIONAL_IMPORTS = {};
const CODE_FILE = /\.(ts|tsx|js|jsx|cjs|mjs)$/;
const TEXT_FILE = /\.(ts|tsx|js|jsx|cjs|mjs|json|html|css|svg|txt|md)$/;

// 정규식 규칙은 주석을 걷어낸 글자에 적용한다(주석은 화면에 나오지 않는다).
const TEXT_RULES = [
  { id: 'math-random', re: /\bMath\.random\b/ },
  { id: 'hex-color', re: /#[0-9A-Fa-f]{3,8}(?![0-9A-Za-z_])/, except: ['src/ui/tokens.ts', 'public/index.html'] },
  { id: 'number-format', re: /\.toFixed\(|\btoLocaleString\(|\bIntl\./, except: ['src/lib/format.ts'] },
  { id: 'local-time', re: /\bget(?:Hours|Date|Month|FullYear|Day)\(|toISOString\(\)\.split|\bnew Date\(\s*['"`]/, except: ['src/lib/kst.ts'] },
  { id: 'font-size', re: /\bfontSize\s*:/, except: ['src/ui/Text.tsx', 'src/ui/tokens.ts'] },
  { id: 'dimensions-get', re: /\bDimensions\.get\b/ },
  { id: 'blocking-dialog', re: /(?<![\w$])(?:alert|confirm)\s*\(|\bAlert\./ },
  { id: 'console', re: /\bconsole\s*\./ },
  { id: 'phrase', re: /R[²2]\s*=\s*0\.95|BiLSTM|CatBoost|AI 최적화|AI 추천|신뢰도|현재 SMP|실시간 데이터|자동 입찰|등록되었습니다|제출 완료|임시 저장|운영자 계정/ },
  { id: 'private-path', re: /\/settlements\b|\/power-plants\b|\/resources\b|\/smp\/rtm\b|\/smp\/current\b/ },
];
// 이모지·기호 문자는 주석까지 포함한 원문에서 찾는다. 화살표(→ U+2192)는 허용한다(입찰 문구 "95%→5%").
const EMOJI = /[☀-➿️⌚⌛⏩-⏳⏸-⏺⬅-⬇⬛⬜⭐⭕]|[\u{1F000}-\u{1FAFF}]/u;
const BANNED_MODULES = [/^@expo\/vector-icons(\/|$)/, /^react-native-webview(\/|$)/, /^react-native-chart-kit(\/|$)/, /^axios(\/|$)/, /^@react-navigation\//, /^expo-linear-gradient(\/|$)/];
const BANNED_RN_NAMES = ['Animated', 'Modal', 'Switch', 'FlatList', 'SectionList', 'RefreshControl', 'Alert', 'TouchableOpacity'];

const sha256 = buffer => crypto.createHash('sha256').update(buffer).digest('hex');
const lineOf = (text, index) => text.slice(0, index).split('\n').length;

function walk(dir, out) {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory() && entry.name !== 'node_modules') walk(full, out);
    else if (entry.isFile() && TEXT_FILE.test(entry.name)) out.push(full);
  }
  return out;
}

/** 다시 쓴 자리 파일이 레거시 바이트와 달라졌는지(= 새 코드인지). */
function isRewritten(rel, buffer) {
  return sha256(buffer) !== REWRITTEN_IN_PLACE[rel];
}

function collectScope() {
  const files = [];
  for (const dir of NEW_DIRS) walk(path.join(ROOT, dir), files);
  for (const rel of Object.keys(REWRITTEN_IN_PLACE)) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs) && isRewritten(rel, fs.readFileSync(abs))) files.push(abs);
  }
  return files.map(abs => path.relative(ROOT, abs).split(path.sep).join('/')).sort();
}

function parse(rel, text) {
  const kind = rel.endsWith('.tsx') || rel.endsWith('.jsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  return ts.createSourceFile(rel, text, ts.ScriptTarget.Latest, true, kind);
}

// 주석을 같은 길이의 공백으로 바꾼다(줄 번호 유지). JSX 글자 안의 "//"는 주석이 아니므로 그 범위는 건너뛴다.
function stripComments(sf) {
  const text = sf.getFullText();
  const jsxText = [];
  const ranges = [];
  const visit = node => {
    if (node.kind === ts.SyntaxKind.JsxText) jsxText.push([node.pos, node.end]);
    ranges.push(...(ts.getLeadingCommentRanges(text, node.getFullStart()) || []), ...(ts.getTrailingCommentRanges(text, node.getEnd()) || []));
    node.getChildren(sf).forEach(visit);
  };
  visit(sf);
  const chars = text.split('');
  for (const range of ranges) {
    if (jsxText.some(([start, end]) => range.pos >= start && range.pos < end)) continue;
    for (let i = range.pos; i < range.end; i += 1) if (chars[i] !== '\n') chars[i] = ' ';
  }
  return chars.join('');
}

function resolveImport(rel, specifier) {
  if (specifier.startsWith('./') || specifier.startsWith('../')) return path.posix.normalize(path.posix.join(path.posix.dirname(rel), specifier));
  if (specifier.startsWith('@/')) return `src/${specifier.slice(2)}`;
  if (specifier.startsWith('src/')) return specifier;
  return null;
}

function importViolations(rel, sf, add) {
  const check = (node, specifier, typeOnly, names) => {
    if (BANNED_MODULES.some(re => re.test(specifier))) add(node, 'import', specifier);
    const target = resolveImport(rel, specifier);
    if (target !== null) {
      const bare = target.replace(/\.(tsx?|jsx?)$/, '');
      const transitional = (TRANSITIONAL_IMPORTS[rel] || []).includes(bare);
      if ((bare === 'src/screens' || bare.startsWith('src/screens/')) && bare !== 'src/screens/SMPForecastScreen' && !transitional) {
        add(node, 'import', specifier);
      }
      if ((bare === 'src/services' || bare.startsWith('src/services/')) && !typeOnly) add(node, 'import', specifier);
      if (bare === 'src/components/Ionicon') add(node, 'import', specifier);
    }
    if (specifier === 'react-native') {
      for (const name of names) {
        if (BANNED_RN_NAMES.includes(name) && !(name === 'TouchableOpacity' && rel === 'App.tsx')) add(node, 'import', `react-native ${name}`);
      }
    }
  };
  const visit = node => {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      const clause = node.importClause;
      const named = clause && clause.namedBindings && ts.isNamedImports(clause.namedBindings) ? clause.namedBindings.elements : [];
      const typeOnly = Boolean(clause && (clause.isTypeOnly || (!clause.name && named.length > 0 && named.every(el => el.isTypeOnly))));
      const names = named.filter(el => !el.isTypeOnly).map(el => (el.propertyName || el.name).text);
      check(node, node.moduleSpecifier.text, typeOnly, names);
      if (clause && node.moduleSpecifier.text === 'react-native') {
        const alias = clause.name ? clause.name.text : clause.namedBindings && ts.isNamespaceImport(clause.namedBindings) ? clause.namedBindings.name.text : null;
        if (alias) namespaceAliases.push(alias);
      }
    } else if (ts.isExportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      const names = node.exportClause && ts.isNamedExports(node.exportClause) ? node.exportClause.elements.map(el => (el.propertyName || el.name).text) : [];
      check(node, node.moduleSpecifier.text, node.isTypeOnly, names);
    } else if (ts.isCallExpression(node) && node.arguments.length === 1 && ts.isStringLiteral(node.arguments[0]) &&
      (node.expression.kind === ts.SyntaxKind.ImportKeyword || (ts.isIdentifier(node.expression) && node.expression.text === 'require'))) {
      check(node, node.arguments[0].text, false, []);
    }
    ts.forEachChild(node, visit);
  };
  const namespaceAliases = [];
  visit(sf);
  return namespaceAliases;
}

function scanSource(rel, text) {
  const violations = [];
  const push = (rule, line, detail) => violations.push({ rel, rule, line, detail });
  const rawMatch = EMOJI.exec(text);
  if (rawMatch) push('emoji', lineOf(text, rawMatch.index), rawMatch[0]);
  let code = text;
  let sf = null;
  if (CODE_FILE.test(rel)) {
    sf = parse(rel, text);
    code = stripComments(sf);
  } else if (rel.endsWith('.html')) {
    code = text.replace(/<!--[\s\S]*?-->/g, comment => comment.replace(/[^\n]/g, ' '));
  }
  for (const rule of TEXT_RULES) {
    if (rule.except && rule.except.includes(rel)) continue;
    const re = new RegExp(rule.re.source, `${rule.re.flags}g`);
    for (const match of code.matchAll(re)) push(rule.id, lineOf(code, match.index), match[0]);
  }
  if (!sf) return violations;
  const add = (node, rule, detail) => push(rule, sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1, detail);
  const aliases = importViolations(rel, sf, add);
  for (const alias of aliases) {
    const re = new RegExp(`\\b${alias}\\.(${BANNED_RN_NAMES.join('|')})\\b`, 'g');
    for (const match of code.matchAll(re)) {
      if (!(match[1] === 'TouchableOpacity' && rel === 'App.tsx')) push('import', lineOf(code, match.index), match[0]);
    }
  }
  const isTsx = rel.endsWith('.tsx');
  const visit = node => {
    // JSX 글자 안 숫자: 숫자가 든 문구는 src/content/strings.ts에서 가져온다.
    if (isTsx && ts.isJsxText(node) && /\d/.test(node.text)) add(node, 'jsx-digit', node.text.trim());
    if (isTsx && ts.isJsxExpression(node) && node.expression && ts.isJsxElement(node.parent)) {
      const e = node.expression;
      const literal = ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e) || ts.isNumericLiteral(e) ? e.text
        : ts.isTemplateExpression(e) ? [e.head.text, ...e.templateSpans.map(span => span.literal.text)].join('') : '';
      if (/\d/.test(literal)) add(node, 'jsx-digit', literal);
    }
    // 타이머로 상태 문구를 바꾸는 호출(가짜 "연결 중 → 연결됨" 류).
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      const name = ts.isIdentifier(callee) ? callee.text : ts.isPropertyAccessExpression(callee) ? callee.name.text : '';
      if ((name === 'setTimeout' || name === 'setInterval') && node.arguments.length > 0) {
        const body = node.arguments[0].getText(sf);
        if (/\bset[A-Z]\w*\(\s*['"`]/.test(body) || /['"`][^'"`]*[가-힣][^'"`]*['"`]/.test(body)) add(node, 'timer-status', name);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return violations;
}

const format = list => list.map(v => `${v.rel}:${v.line} ${v.rule} ${JSON.stringify(v.detail)}`).join('\n');

test('new code obeys the 7.10 ban list (scope is computed, empty until new files exist)', t => {
  const scope = collectScope();
  t.diagnostic(`source-bans scope files: ${scope.length}${scope.length ? ` (${scope.join(', ')})` : ''}`);
  const violations = scope.flatMap(rel => scanSource(rel, fs.readFileSync(path.join(ROOT, rel), 'utf8')));
  assert.equal(violations.length, 0, format(violations));
});

test('App.tsx and SMPForecastScreen.tsx enter the scope as soon as their bytes change', () => {
  for (const rel of Object.keys(REWRITTEN_IN_PLACE)) {
    const legacy = Buffer.from(fs.readFileSync(path.join(ROOT, rel)));
    if (!isRewritten(rel, legacy)) {
      assert.equal(isRewritten(rel, Buffer.concat([legacy, Buffer.from('\n')])), true, rel);
    }
  }
});

test('transitional legacy imports are still used (delete the entry when F8/F9 replace the tab)', () => {
  for (const [rel, targets] of Object.entries(TRANSITIONAL_IMPORTS)) {
    const text = fs.readFileSync(path.join(ROOT, rel), 'utf8');
    const sf = parse(rel, text);
    const imported = sf.statements
      .filter(st => ts.isImportDeclaration(st) && ts.isStringLiteral(st.moduleSpecifier))
      .map(st => resolveImport(rel, st.moduleSpecifier.text))
      .filter(Boolean)
      .map(target => target.replace(/\.(tsx?|jsx?)$/, ''));
    for (const target of targets) assert.ok(imported.includes(target), `${rel} no longer imports ${target}: remove it from TRANSITIONAL_IMPORTS`);
  }
});

const POSITIVE = [
  ['App.tsx', "import DashboardScreen from './src/screens/DashboardScreen';", ['import']],
  // F8 뒤로 레거시 입찰 화면, F9 뒤로 레거시 정산 화면은 App.tsx에서도 금지다(번들에서 빠져야 한다).
  ['App.tsx', "import BiddingScreen from './src/screens/BiddingScreen';", ['import']],
  ['App.tsx', "import SettlementScreen from './src/screens/SettlementScreen';", ['import']],
  ['src/features/settlement/x.tsx', "import SettlementScreen from '../../screens/SettlementScreen';", ['import']],
  ['src/features/l11.tsx', "import BiddingScreen from '../screens/BiddingScreen';", ['import']],
  ['src/lib/a.ts', 'export const x = Math.random();', ['math-random']],
  ['src/ui/b.tsx', "export const s = { color: '#1A2B3C' };", ['hex-color']],
  ['src/lib/c.ts', 'export const t = (n: number) => n.toFixed(1);', ['number-format']],
  ['src/lib/c2.ts', "export const f = new Intl.NumberFormat('ko-KR');", ['number-format']],
  ['src/lib/c3.ts', 'export const g = (n: number) => n.toLocaleString();', ['number-format']],
  ['src/lib/d.ts', 'export const h = new Date().getHours();', ['local-time']],
  ['src/lib/d2.ts', "export const d = new Date('2026-11-26T06:00:00');", ['local-time']],
  ['src/lib/d3.ts', "export const day = new Date().toISOString().split('T')[0];", ['local-time']],
  ['src/ui/e.tsx', 'export const s = StyleSheet.create({ a: { fontSize: 14 } });', ['font-size']],
  ['src/app/f.ts', "export const w = Dimensions.get('window').width;", ['dimensions-get']],
  ['src/app/g.ts', "window.confirm('x');", ['blocking-dialog']],
  ['src/app/g2.ts', "Alert.alert('x');", ['blocking-dialog']],
  ['src/data/h.ts', "console.log('x');", ['console']],
  ['public/content/guide.ko.json', '{ "a": "신뢰도 75%" }', ['phrase']],
  ['src/content/strings.ts', "export const m = 'GradientBoosting (R²=0.95)';", ['phrase']],
  ['src/data/i.ts', "fetch('/api/v1/settlements/summary');", ['private-path']],
  ['src/data/i2.ts', "fetch('/api/v1/smp/current?region=jeju');", ['private-path']],
  ['src/features/j.tsx', 'export const A = () => <Text>⚠ 주의</Text>;', ['emoji']],
  ['src/features/j2.ts', "export const robot = '\u{1F916}';", ['emoji']],
  ['src/features/k.tsx', 'export const A = () => <Text>3시간 56분</Text>;', ['jsx-digit']],
  ['src/features/k2.tsx', "export const A = () => <Text>{'50 MW'}</Text>;", ['jsx-digit']],
  ['src/features/l.ts', "import { getSMP } from '../services/api';", ['import']],
  ['src/features/l2.ts', "import X from '../screens/DashboardScreen';", ['import']],
  ['src/features/l3.tsx', "import { Animated } from 'react-native';", ['import']],
  ['src/features/l4.ts', "import axios from 'axios';", ['import']],
  ['src/features/l5.ts', "import { Ionicons } from '@expo/vector-icons';", ['import']],
  ['src/features/l6.ts', "import { Ionicon } from '../components/Ionicon';", ['import']],
  ['src/features/l7.ts', "import { NavigationContainer } from '@react-navigation/native';", ['import']],
  ['src/features/l8.tsx', "import { TouchableOpacity } from 'react-native';", ['import']],
  ['src/features/l9.ts', "import * as RN from 'react-native';\nexport const M = RN.Modal;", ['import']],
  ['src/features/l10.ts', "const W = require('react-native-webview');", ['import']],
  ['src/features/m.ts', "setTimeout(() => setStatus('connected'), 3000);", ['timer-status']],
  ['src/features/m2.ts', "setTimeout(() => { label = '연결됨'; }, 3000);", ['timer-status']],
  // JSX 글자 안의 "//"는 주석이 아니다: 뒤의 금지 문구가 걷히지 않고 잡혀야 한다.
  ['src/features/v2.tsx', 'export const A = () => <Text> // 주소 보기 현재 SMP</Text>;', ['phrase']],
];

test('every ban rule fires on a known-bad sample (positive controls)', () => {
  for (const [rel, text, expected] of POSITIVE) {
    const rules = [...new Set(scanSource(rel, text).map(v => v.rule))].sort();
    assert.deepStrictEqual(rules, expected, `${rel}: ${text}`);
  }
});

const NEGATIVE = [
  ['src/ui/tokens.ts', "export const ink = '#0B1220';\nexport const body = { fontSize: 16 };"],
  ['src/ui/Text.tsx', 'export const s = { fontSize: 16 };'],
  ['src/lib/format.ts', "export const f = (n: number) => n.toFixed(1);\nexport const g = new Intl.NumberFormat('ko-KR');"],
  ['src/lib/kst.ts', 'export const h = (ms: number) => new Date(ms).getUTCHours();'],
  ['public/index.html', '<meta name="theme-color" content="#F6F7F9" media="(prefers-color-scheme: light)">'],
  ['App.tsx', "import { TouchableOpacity } from 'react-native';\nimport SMPForecastScreen from './src/screens/SMPForecastScreen';"],
  ['src/data/types.ts', "import type { DayAheadSMP } from '../services/api';\nexport type { CurrentSMP } from '../services/api';"],
  ['src/data/o.ts', "import { type DayAheadSMP } from '../services/api';"],
  ['src/content/strings.ts', "export const s = '95%→5% · 3시간 56분 · 11-26(목) 10:00 KST';"],
  ['src/data/client.ts', 'const id = setTimeout(() => controller.abort(), 15000);\nsetTimeout(retry, 1500);'],
  ['src/features/r.tsx', "// console.log('x') 현재 SMP Math.random #FFF\n/* alert('x') */\nexport const A = () => <Text>{label}</Text>;"],
  ['src/ui/s.tsx', 'export const ok = (x: number, y: number): Array<number> => (x > 0 && y < 5 ? [x] : [y]);'],
  ['src/features/u.ts', 'export const d = new Date(0).getUTCDate();\nonConfirm();'],
  ['src/features/v.tsx', 'export const A = () => <Text> // 주소 보기</Text>;'],
];

test('allowed patterns and listed exceptions do not fire (negative controls)', () => {
  for (const [rel, text] of NEGATIVE) {
    assert.deepStrictEqual(scanSource(rel, text), [], `${rel}: ${text}`);
  }
});
