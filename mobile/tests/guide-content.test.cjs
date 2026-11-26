'use strict';
// 알아보기와 설명 글(명세 4.4·7.9·7.10절, F10). public/content/guide.ko.json(실제 파일)이 로더 검사를 통과하고 4.4절 구성(둘러보기 5단계 제목,
// 용어 13개와 "이 앱에서 보는 곳", 숫자 읽는 법의 표시별 설명, 단순화한 설명 목록, 모든 산출 방식 카드)을 갖췄는지, 알아보기 화면의
// 의미(h1·목록 6행·링크 이름·44/56 px)와 I2(숫자는 표지 안에만)를 노드 수준에서 본다. 앨런은 공개 빌드에 들어가지 않는다(P6).
// 응답 상태 매핑·둘러보기 5단계 범례·출처 행은 tests/view-models.test.cjs, 브라우저 확인(A1–A6)은 scratchpad/mobile/eval/steps/F10에 있다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const { createTsLoader, defaultMocks, MOBILE_ROOT, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const SVG = { __esModule: true, default: 'Svg', Svg: 'Svg', Path: 'Path', Circle: 'Circle', Ellipse: 'Ellipse', Rect: 'Rect', Line: 'Line', Defs: 'Defs', Pattern: 'Pattern' };
function load(width = 390) {
  const rn = { ...defaultMocks()['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG } });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  return { tab: at('features/guide/GuideTab.tsx'), guide: at('content/guide.ts'), strings: at('content/strings.ts'), routes: at('app/routes.ts'),
    model: at('features/guide/model.ts') };
}
const JSON_FILE = path.join(MOBILE_ROOT, 'public/content/guide.ko.json');
const content = m => m.guide.parseGuide(JSON.parse(fs.readFileSync(JSON_FILE, 'utf8')));

const h = (type, props, ...children) => ({ type, props: { ...(props || {}), children } });
function render(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return [];
  if (Array.isArray(node)) return node.flatMap(render);
  if (typeof node !== 'object') return [String(node)];
  const { type, props } = node;
  if (typeof type === 'function') return render(type(props));
  const { children, ...rest } = props;
  return [{ type, props: rest, children: render(children) }];
}
const tree = element => ({ type: 'root', props: {}, children: render(element) });
function all(node, pred, out = []) {
  if (node && typeof node === 'object') {
    if (pred(node)) out.push(node);
    node.children.forEach(child => all(child, pred, out));
  }
  return out;
}
const textOf = node => (typeof node === 'string' ? node : node.children.map(textOf).join(''));
const flat = style => (Array.isArray(style) ? Object.assign({}, ...style.map(flat)) : style || {});
const ds = n => (n.props && n.props.dataSet) || {};
const MARKS = ['pdValue', 'pdMeta', 'pdChip', 'pdExample'];
/** I2(노드 수준): 숫자가 든 글자는 값·메타·칩·예시 표지 안에만. */
function assertI2(root, label) {
  const stray = [];
  const walk = (node, marked) => {
    if (typeof node === 'string') {
      if (/\d/.test(node) && !marked) stray.push(node);
      return;
    }
    const here = marked || MARKS.some(k => ds(node)[k] !== undefined) || node.props.role === 'img';
    node.children.forEach(c => walk(c, here));
  };
  walk(root, false);
  assert.deepEqual(stray, [], `${label}: 표지 밖 숫자`);
}
const links = root => all(root, n => n.props.role === 'link');
const BANNED = /R[²2]\s*=\s*0\.95|BiLSTM|CatBoost|AI 최적화|AI 추천|신뢰도|현재 SMP|실시간 데이터|자동 입찰|등록되었습니다|제출 완료|임시 저장|운영자 계정|연결됨/;

test('guide.ko.json: 둘러보기 5단계 제목(4.4절 그대로) · 용어 13개(사전 순서, 한 줄 뜻 60자 이하) · 주제 넷 · 표시별 설명 9개 · 단순화한 설명 넷', () => {
  const m = load();
  const c = content(m);
  assert.deepEqual(c.tour.map(t => t.title), ['제주 전기값은 시간마다 바뀝니다', '오늘 값은 어제 같은 시간 값으로 짐작합니다', '발전소는 하루 전에 팝니다',
    '판 만큼 정산 받습니다', '숫자 옆 표시를 보세요']);
  assert.deepEqual(c.glossary.map(g => g.id), Object.keys(m.strings.GLOSSARY_TITLES));
  for (const g of c.glossary) assert.ok(g.short.length <= m.guide.GLOSSARY_SHORT_MAX, g.id);
  assert.deepEqual(Object.keys(c.topics).sort(), ['about', 'method', 'reading', 'sources']);
  assert.equal(c.topics.method.body[0], '지금 쓰는 방법: 어제 같은 시간 가격을 오늘 예측으로 씁니다(기준선).');
  assert.deepEqual(Object.keys(c.legend), [...m.guide.LEGEND_KEYS]);
  // 단순화한 설명(4.4·7.9절, P7): 마감 · 이중 정산 · 가격 원리 · 실시간시장 구간. 숫자는 서버 값으로만 붙는다(JSON에는 숫자 없음).
  assert.deepEqual(c.simplified.body.map(p => p.split(':')[0]), ['하루전시장 마감', '이중 정산', '가격 원리', '실시간시장 구간']);
  assert.doesNotMatch(fs.readFileSync(JSON_FILE, 'utf8'), BANNED);
});

test('D1: 둘러보기 2단계와 용어 사전 "기준선 예측"·예측 방법·차트 범례가 같은 방법(어제 같은 시간 값 → 오늘)을 말한다', () => {
  const m = load();
  const c = content(m);
  const step2 = c.tour[1];
  const baseline = c.glossary.find(g => g.id === 'baseline');
  // 어제 값을 "내일"에 쓰면 48시간 차이가 된다(round 1 멘토 major): 같은 날짜 표현만 허용한다.
  for (const [label, text] of [['tour[1].title', step2.title], ['tour[1].body', step2.body.join(' ')], ['glossary baseline', baseline.short],
    ['method', c.topics.method.body[0]], ['chart legend', m.strings.forecastLegendText('baseline')]]) {
    assert.match(text, /어제 같은 시간/, label);
    assert.doesNotMatch(text, /내일/, label);
  }
  for (const text of [step2.title, step2.body.join(' '), baseline.short, c.topics.method.body[0]]) assert.match(text, /오늘/, text);
  assert.match(step2.body.join(' '), /학습한 모델이 아닌 단순한 방법이라 평소 얼마나 빗나가는지도 함께 보여 줍니다/);
});

test('D7: 뜻·읽는 법이 먼저, 짝·검증 규칙은 끝의 흐린 "자세히" 한 줄 — 공급능력 뜻, "KPI 수요"·"±1 MW" 없음, 규칙은 detail에만', () => {
  const m = load();
  const c = content(m);
  const supply = c.glossary.find(g => g.id === 'supply');
  assert.equal(supply.short, '가동 중이거나 곧 돌릴 수 있는 발전기가 지금 낼 수 있는 최대 전력입니다');
  assert.match(supply.detail, /이 화면의 수요 값과 거의 같을 때만/);
  assert.doesNotMatch(supply.body.join(' '), /거의 같을 때만|미수신/);
  const raw = fs.readFileSync(JSON_FILE, 'utf8');
  assert.doesNotMatch(raw, /KPI 수요/);
  assert.doesNotMatch(m.strings.SOURCE_TEXT.supply, /±|일치 때만/);
  // 산출 방식 시트의 짝·검증 규칙(출처 표기 확인, 같은 응답에서 가져온 모델 이름)은 본문이 아니라 detail에 있다.
  assert.match(c.cards['today-power'].detail, /출처 표기로 실측인지 확인/);
  assert.doesNotMatch(c.cards['today-power'].body.join(' '), /출처 표기로 실측인지 확인|거의 같을 때만/);
  assert.match(c.cards['today-forecast'].detail, /같은 응답에서 가져옵니다/);
  assert.doesNotMatch(c.cards['today-forecast'].body.join(' '), /같은 응답에서 가져옵니다/);
  // 예비율 단계 이름은 앱 기준(정상·보통·낮음·위험)이고 KPX 공식 경보 단계가 아니라고 한 번 밝힌다(D9).
  const reserve = c.glossary.find(g => g.id === 'reserve');
  assert.match(reserve.body.join(' '), /정상·보통·낮음·위험.*KPX 공식 경보 단계가 아닙니다/);
  assert.doesNotMatch(raw, /관심 경계|관심 단계|관심 기준/);
  // 로더: detail도 숫자 없는 글자여야 한다.
  const bad = JSON.parse(raw);
  bad.cards['today-power'].detail = '수요 ±1 MW 일치 때만';
  assert.throws(() => m.guide.parseGuide(bad), /cards\.today-power\.detail no digits/);
});

test('산출 방식: 모든 카드 id(8개)에 설명 글 · 카드 제목 · 시트 제목이 있다(F10 합격 확인)', () => {
  const m = load();
  const c = content(m);
  assert.deepEqual([...m.routes.METHOD_CARDS], [...m.guide.METHOD_CARD_IDS]);
  for (const id of m.routes.METHOD_CARDS) {
    assert.ok(c.cards[id] && c.cards[id].title && c.cards[id].body.length > 0, id);
    assert.equal(m.strings.methodSheetTitle(id), `산출 방식 · ${m.strings.CARD_TITLES[id]}`, id);
  }
});

test('용어가 보이는 곳: 13개 모두 탭이 있는 경로를 가리킨다(정산 · kWh와 MW → 정산, 하루전시장 · 실시간시장 · 낙찰 가능성 → 입찰)', () => {
  const m = load();
  const where = m.model.TERM_WHERE;
  assert.deepEqual(Object.keys(where).sort(), Object.keys(m.strings.GLOSSARY_TITLES).sort());
  for (const key of Object.values(where)) assert.ok(m.routes.ROUTES[key].tab && key !== 'guide', key);
  assert.deepEqual(['dam', 'rtm', 'clearing', 'settlement', 'units', 'smp'].map(id => where[id]), ['bidding', 'bidding', 'bidding', 'settlement', 'settlement', 'smp']);
});

test('알아보기 첫 화면: h1 · AnswerLine · 둘러보기 입구 [시작하기] · 목록 6행(링크, 높이 56, 이름 = 보이는 글자) · 요청·제출 없음 · I2', () => {
  const m = load();
  const root = tree(h(m.tab.GuideHome, {}));
  const [h1] = all(root, n => n.props.role === 'heading' && n.props['aria-level'] === 1);
  assert.equal(textOf(h1), '알아보기');
  assert.match(textOf(root), /이 앱이 무엇을 보여 주는지, 숫자를 어떻게 읽는지 설명합니다\./);
  const buttons = all(root, n => n.props.role === 'button').map(b => b.props['aria-label']);
  assert.deepEqual(buttons, ['시작하기: 1분 둘러보기']);
  const rows = links(root);
  assert.deepEqual(rows.map(r => r.props['aria-label']), ['용어 사전, 용어 13개', '숫자 읽는 법', '예측 방법과 정확도', '데이터 출처와 연결 상태',
    '화면 테마, 시스템 설정 따름 (기본)', '앱 정보, eXeco v6.1 공개 데모']);
  for (const r of rows) {
    assert.equal(flat(r.props.style).minHeight, 56);
    assert.equal(typeof r.props.onKeyDown, 'function', `${r.props['aria-label']}: Enter로 연다`);
  }
  assert.match(textOf(root), /1분 둘러보기 · 5단계/);
  assertI2(root, '첫 화면');
});

test('용어 사전(?topic=glossary): 항목마다 이름 링크(→ 용어 화면) · 한 줄 뜻 · "이 앱에서 보는 곳: …" 링크, 숫자 든 이름은 메타', () => {
  const m = load();
  const c = content(m);
  const root = tree(h(m.tab.GlossaryList, { content: c }));
  const names = links(root).map(l => l.props['aria-label']);
  const titles = Object.values(m.strings.GLOSSARY_TITLES);
  assert.equal(names.length, titles.length * 2);
  titles.forEach((t, i) => {
    assert.equal(names[i * 2], `${t}, ${c.glossary[i].short}`);
    assert.equal(names[i * 2 + 1], `이 앱에서 보는 곳: ${m.routes.ROUTES[m.model.TERM_WHERE[c.glossary[i].id]].tab}`);
  });
  assert.ok(links(root).every(l => typeof l.props.onKeyDown === 'function' && flat(l.props.style).minHeight >= 44));
  assertI2(root, '용어 사전');
});

test('숫자 읽는 법(?topic=reading): 4상태 칩(실측·예측·추정·예시) · 일반 칩 둘 · 줄표 · 지연·끊김 · KST와 "07h · 06–07시" 풀이', () => {
  const m = load();
  const c = content(m);
  const root = tree(h(m.tab.ReadingBody, { content: c }));
  const chips = all(root, n => ds(n).pdChip).map(n => n.props['aria-label']);
  assert.deepEqual(chips, ['상태: 실측', '상태: 예측 · 기준선', '상태: 추정 · 실측 아님', '상태: 예시', '상태: 검증값', '상태: 시장 규칙 · 서버 시각',
    '상태: 지연', '상태: 끊김']);
  for (const key of m.guide.LEGEND_KEYS) assert.ok(textOf(root).includes(c.legend[key]), key);
  assert.ok(textOf(root).includes('07h · 06–07시 = 06:00–07:00에 거래하는 1시간'));
  assert.equal(all(root, n => n.props.role === 'heading' && n.props['aria-level'] === 2).length, 3);
  // 링크·버튼이 없는 글 화면도 키보드로 스크롤 영역에 들어오도록 끝에 이어 볼 주제 링크(axe scrollable-region-focusable).
  assert.deepEqual(links(root).map(l => l.props['aria-label']), ['용어 사전']);
  assertI2(root, '숫자 읽는 법');
});

/** 값으로 가져오는 상대 경로 파일 전부(형식만 가져오기는 번들에서 지워진다). */
function importGraph(entry, seen = new Set()) {
  if (seen.has(entry)) return seen;
  seen.add(entry);
  const sf = ts.createSourceFile(entry, fs.readFileSync(entry, 'utf8'), ts.ScriptTarget.Latest, true, entry.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  for (const st of sf.statements) {
    if (!ts.isImportDeclaration(st) || !ts.isStringLiteral(st.moduleSpecifier) || !st.moduleSpecifier.text.startsWith('.')) continue;
    const clause = st.importClause;
    const named = clause && clause.namedBindings && ts.isNamedImports(clause.namedBindings) ? clause.namedBindings.elements : [];
    if (clause && (clause.isTypeOnly || (!clause.name && named.length > 0 && named.every(el => el.isTypeOnly)))) continue;
    const base = path.resolve(path.dirname(entry), st.moduleSpecifier.text);
    const file = ['.web.ts', '.ts', '.tsx', '/index.ts', '/index.tsx'].map(ext => base + ext).find(f => fs.existsSync(f));
    if (file) importGraph(file, seen);
  }
  return seen;
}

test('앨런(P6): 공개 빌드는 src/services/alan·AlanChatScreen을 가져오지 않고, getLocalResponse에는 지어낸 숫자 대신 "앨런 서비스에 연결하지 못했습니다"', () => {
  const graph = [...importGraph(path.join(MOBILE_ROOT, 'App.tsx'))].map(f => path.relative(MOBILE_ROOT, f));
  assert.ok(graph.some(f => f.endsWith('features/guide/GuideTab.tsx')));
  assert.deepEqual(graph.filter(f => /services\/alan|screens\/alan|AlanChat/.test(f)), []);
  const alan = fs.readFileSync(path.join(SRC_ROOT, 'services/alan/alanApi.ts'), 'utf8');
  const local = alan.slice(alan.indexOf('private getLocalResponse'), alan.indexOf('getQuickActions'));
  assert.doesNotMatch(local, /49건|71건|131건|12\.3%|1,251|45\.3MWh|94\.5%|8\.2%/);
  assert.equal(local.match(/answer: '앨런 서비스에 연결하지 못했습니다\.'/g).length, 2);
});

test('App.tsx: 탭 4개(알아보기 추가) · guide 경로는 GuideTab · 출처 시트에 응답 상태 행 · 숫자 든 하위 상태 제목은 메타', () => {
  const app = fs.readFileSync(path.join(MOBILE_ROOT, 'App.tsx'), 'utf8');
  assert.match(app, /const TABS: readonly RouteKey\[\] = \['smp', 'bidding', 'settlement', 'guide'\];/);
  assert.match(app, /case 'guide':\s*return <GuideTab \/>/);
  assert.match(app, /extra=\{\[\.\.\.useBiddingSourceRows\(\), \.\.\.useStatusRows\(\)\]\}/);
  assert.match(app, /\{\.\.\.titleData\(title\)\}/);
  assert.match(app, /case 'smp':\s*return <SMPForecastScreen/);
});
