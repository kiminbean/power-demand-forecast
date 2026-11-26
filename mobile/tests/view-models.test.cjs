'use strict';
// 오늘 화면 view model 픽스처 재생(명세 4.1·6.3·7.4·7.6·7.8절, F7). round 0 스냅숏(2026-11-26 06:03:57 KST)과 상태 변형
// (fixtures/variants/variants.json의 T9)을 features/today/model.ts에 넣어 글자·상태·칩을 고정하고, 같은 결과를 카드 컴포넌트로 그려
// I2 규칙(숫자는 값·메타·칩 표지 안에만, 값은 칩과 출처 줄이 있는 카드 안에만)과 의미(이름·역할·44 px)를 노드 수준에서 본다.
// 입력은 화면(src/screens/SMPForecastScreen.tsx)과 같게 만든다: 파싱 → isAvailableDayAhead·describeDayAhead → 시장 상태 서버 시각으로
// 시계 차이 → 앱 시각. 브라우저 확인(첫 화면 상자·시트·T1–T9 캡처)은 scratchpad/mobile/eval/steps/F7에 있다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, defaultMocks, SRC_ROOT } = require('./helpers/ts-loader.cjs');
const { loadManifest, resolveVariant } = require('./helpers/fixtures.cjs');

const SVG = { __esModule: true, default: 'Svg', Svg: 'Svg', Path: 'Path', Circle: 'Circle', Rect: 'Rect', Line: 'Line', Defs: 'Defs', Pattern: 'Pattern' };
const EXPECTED = require('./fixtures/round0/expected-today.json');

/** forceTable: 차트/표 세그먼트와 "24시간 전체 보기"의 첫 렌더 값을 표·전체로 바꿔 그린다(React 목의 useState는 첫 값만 준다). */
function load(width = 390, { forceTable = false } = {}) {
  const mocks = defaultMocks();
  const rn = { ...mocks['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const react = forceTable
    ? { ...mocks.react, useState: init => [init === 'chart' ? 'table' : init === false ? true : typeof init === 'function' ? init() : init, () => {}] }
    : mocks.react;
  react.default = react;
  const loader = createTsLoader({ mocks: { react, 'react-native': rn, 'react-native-svg': SVG } });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  return {
    model: at('features/today/model.ts'), types: at('data/types.ts'), client: at('data/client.ts'), kst: at('lib/kst.ts'),
    sp: at('utils/smpPresentation.ts'), rich: at('ui/Rich.tsx'), format: at('lib/format.ts'), strings: at('content/strings.ts'),
    provenance: at('lib/provenance.ts'),
    ...at('features/today/PriceCard.tsx'), ...at('features/today/PowerCard.tsx'), ...at('features/today/ForecastCard.tsx'),
    ...at('features/today/AccuracyCard.tsx'), ...at('features/today/TodayParts.tsx'), ...at('ui/Controls.tsx'), ...at('ui/Values.tsx'),
    ...at('ui/Chips.tsx'), ...at('ui/Card.tsx'),
  };
}

const ROUTES = {
  dayAhead: ['/api/v1/smp/day-ahead', 'parseDayAhead'], forecast: ['/api/v1/smp-forecast', 'parseForecast'],
  modelInfo: ['/api/v1/model/info', 'parseModelInfo'], kpis: ['/api/v1/dashboard/kpis', 'parseKpis'],
  powerSupply: ['/api/v1/power-supply', 'parsePowerSupply'], market: ['/api/v1/market-status', 'parseMarketStatus'],
};

/**
 * 변형 → TodayInput. pending: 지연 주입(delay_ms) 경로는 아직 응답 전. advanceMs·failAfter: 첫 로드 뒤 시계를 앞당기고 그 뒤 요청이
 * 모두 실패한 상태(마지막 좋은 값은 남고 상태는 error) — variants.json의 after_load와 같은 뜻.
 */
function inputFor(m, id, { advanceMs = 0, failAfter = false, pending = true, selectedHour = null } = {}) {
  const v = resolveVariant(id);
  const device = v.clockMs;
  const res = {};
  for (const [key, [route, parser]] of Object.entries(ROUTES)) {
    const r = v.responses[route];
    if (pending && r.delayMs) res[key] = { data: null, fetchedAt: null, status: 'loading', error: null };
    else if (r.abort) res[key] = { data: null, fetchedAt: null, status: 'error', error: new m.client.ApiNetworkError(route) };
    else if (r.status >= 400) res[key] = { data: null, fetchedAt: null, status: 'error', error: new m.client.ApiHttpError(route, r.status) };
    else {
      try {
        res[key] = { data: m.types[parser](r.body), fetchedAt: device, status: 'ready', error: null };
      } catch (error) {
        res[key] = { data: null, fetchedAt: null, status: 'error', error };
      }
    }
  }
  const server = res.market.data ? m.kst.parseApiTime(res.market.data.current_time) : null;
  const skew = server === null ? null : m.kst.clockSkew(server, device);
  const offset = skew && skew.plausible ? skew.skewMs : 0;
  const app = r => ({ ...r, fetchedAt: r.fetchedAt === null ? null : r.fetchedAt + offset, status: failAfter && r.data !== null ? 'error' : r.status });
  const dayAheadSmp = res.dayAhead.data;
  return {
    input: {
      now: device + advanceMs + offset, dayAhead: app(res.dayAhead), priceAvailable: m.sp.isAvailableDayAhead(dayAheadSmp),
      kpxText: m.sp.describeDayAhead(dayAheadSmp), forecast: app(res.forecast), modelInfo: app(res.modelInfo), kpis: app(res.kpis),
      powerSupply: app(res.powerSupply), market: app(res.market), selectedHour,
    },
    skew,
  };
}
const viewOf = (m, id, opts) => m.model.todayView(inputFor(m, id, opts).input);

const chipLabel = (m, c) => (c.chip ? m.provenance.chipText(c.chip) : c.text || m.strings.staleChipText(c.at, c.lag));
/** expected-today.json과 같은 투영. */
function project(m, view) {
  const f = m.format;
  const P = view.price;
  const W = view.power;
  const F = view.forecast;
  const A = view.accuracy;
  const chip = c => chipLabel(m, c);
  return {
    answer: view.answer, heroState: view.heroState, nowPill: m.strings.nowPillText({ now: inputFor(m, 'today-base').input.now, fetchedAt: view.fetchedAt }),
    price: { value: f.price(P.price.value), unit: P.price.unit, chips: P.chips.map(chip), hourLine: P.hourLine, checking: P.checking,
      compare: P.compare.map(m.rich.lineText), footer: `${P.footer.source} · ${P.footer.time}` },
    power: { chips: W.chips.map(chip), demand: `${W.demand.label} ${f.mw(W.demand.value.value)} MW`, reserve: `${W.reserve.label} ${f.pct(W.reserve.value.value)}%`,
      level: W.level.text, levelNote: W.level.note, formula: m.rich.lineText(W.supply.formula), supply: f.mw(W.supply.supply.value),
      reserveMw: f.mw(W.supply.reserve.value), boundary: f.mw(W.supply.boundary.value), weather: m.rich.lineText(W.weather.line),
      weatherChip: chip(W.weather.chip), footer: `${W.footer.source} · ${W.footer.time} · ${W.footer.extra}` },
    forecast: { chips: F.chips.map(chip), legend: F.legend, selected: F.selected, stepLabel: F.stepLabel(F.selected), reading: m.rich.lineText(F.reading),
      summary: F.summary, uncertainty: F.uncertainty, periods: F.periods.map(p => [p.label, p.mean, p.min, p.max]), periodNote: F.periodNote,
      window: [F.window.from, F.window.to], footer: `${F.footer.source} · ${F.footer.time}` },
    accuracy: { main: A.documented.main, basis: A.documented.basis, example: A.documented.example, footer: `${A.footer.source} · ${A.footer.time}` },
    facts: Object.fromEntries(Object.entries(view.facts).map(([k, x]) => [k, { lines: x.lines, api: x.api || [], link: x.link && x.link.url }])),
    sources: view.sources.map(r => [r.id, r.item, r.source, r.basis, r.chip ? chip(r.chip) : r.status]),
  };
}

// ---------------------------------------------------------------- 렌더 도우미(ui-components.test.cjs와 같은 방식)
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

/** I2(노드 수준): 숫자가 든 글자 노드는 값·메타·칩·예시 표지 안에만. 값은 칩과 출처 줄(role=note)이 있는 카드 안에만, 원값은 숫자. */
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
  const walkValues = (node, card) => {
    if (typeof node === 'string') return;
    const inCard = ds(node).pdCard ? node : card;
    if (ds(node).pdValue !== undefined) {
      assert.ok(inCard, `${label}: 카드 밖 값 ${textOf(node)}`);
      assert.ok(all(inCard, n => ds(n).pdChip !== undefined).length > 0, `${label}: 칩 없는 카드의 값 ${textOf(node)}`);
      assert.ok(all(inCard, n => n.props.role === 'note').length > 0, `${label}: 출처 줄 없는 카드의 값 ${textOf(node)}`);
      assert.ok(Number.isFinite(Number(ds(node).pdRaw)), `${label}: 원값 ${ds(node).pdRaw}`);
    }
    node.children.forEach(c => walkValues(c, inCard));
  };
  walkValues(root, null);
}

/** 오늘 화면 본문(셸·Screen 틀 제외)을 그린다: 카드 넷 + 용어 줄 + (1280) 패널 본문. */
function renderToday(m, view, nowMs) {
  const noop = () => {};
  return tree([
    h(m.PriceCard, { view: view.price, onMethod: noop, onSources: noop, onRetry: noop }),
    h(m.PowerCard, { view: view.power, onMethod: noop, onSources: noop, onRetry: noop }),
    h(m.ForecastCard, { view: view.forecast, chartLabel: () => '기준선 예측 SMP 24시간', now: nowMs, onSelectHour: noop, onToNow: noop,
      onMethod: noop, onSources: noop, onRetry: noop }),
    h(m.AccuracyCard, { view: view.accuracy, onMethod: noop, onSources: noop, onRetry: noop }),
    h(m.TermRow, { onOpen: noop }),
    h(m.TodayPanelBody, { legend: view.legend, onTerm: noop }),
  ]);
}

const TODAY_VARIANTS = loadManifest().variants.filter(v => v.screen === 'today' && v.sweep === 'T9').map(v => v.id);

// ---------------------------------------------------------------- 1. 기준 스냅숏
test('기준 스냅숏 → expected-today.json(글자·칩·출처·시트 값·출처 표 행)', () => {
  const m = load();
  const view = viewOf(m, 'today-base');
  const { $doc, clock_kst: clock, ...expected } = EXPECTED;
  assert.equal(clock, '2026-11-26T06:03:57');
  assert.ok($doc.length > 0);
  assert.deepEqual(project(m, view), expected);
});

test('명세 F7 행의 기대 문자열이 모두 그 자리에 있다', () => {
  const m = load();
  const p = project(m, viewOf(m, 'today-base'));
  const has = (where, ...parts) => parts.forEach(s => assert.ok(where.includes(s), `${s} ∉ ${where}`));
  has(p.price.value, '115.3');
  has(p.price.hourLine, '06–07시', '거래시간 07h');
  has(p.price.compare.join(' '), '145.2', '29.9');
  has(p.power.demand, '664');
  has(p.power.reserve, '79.5');
  has(p.power.formula, '1,191', '527');
  assert.equal(p.power.boundary, '1,036');
  has(p.power.weather, '11.0', '3.6', '맑음');
  // D11: 문장·요약·표의 첫 줄은 시각 범위만, "07h" 번호는 카드 메타 줄(위 hourLine)·스텝퍼·표 칸에만.
  has(p.forecast.summary, '212.3', '07–08시', '102.3', '02–05시', '169.2');
  has(p.forecast.uncertainty, '326.4', '12–13시');
  for (const sentence of [p.answer, p.forecast.summary, p.forecast.uncertainty, ...p.price.compare]) assert.doesNotMatch(sentence, /\d{2}h/, sentence);
  assert.deepEqual(p.forecast.periods, [['밤·아침 00–08시', '130.6', '102.3', '212.3'], ['주간 08–17시', '177.1', '176.5', '178.7'], ['저녁 17–24시', '203.3', '201.3', '205.9']]);
  // D8: 카드에는 쉬운 말과 문서화 MAPE 하나만. API 수치(12.68 % 등)는 정확도 시트의 회색 "API 표기" 줄에만 있다.
  assert.equal(p.accuracy.main, '시간마다 평균 11.97원/kWh 빗나갑니다 (MAPE 11.75%)');
  assert.equal(p.accuracy.basis, 'walk-forward 평가(지난 자료만 보고 하루씩 맞혀 봄) · 기준선 자체(학습한 모델 아님)');
  assert.ok(!JSON.stringify(p.accuracy).includes('12.68') && !JSON.stringify(p.accuracy).includes('72.6'));
  assert.ok(!p.facts['today-forecast'].lines.join(' ').includes('MAPE'), '모델 이름 줄에 API MAPE 없음(I5)');
  assert.deepEqual(p.facts['today-accuracy'].api.map(l => l.slice(0, 8)), ['API 표기: ', 'API 표기: ']);
  assert.equal(p.answer, '지금 시간 제주 전기 도매가격은 1kWh에 115.3원이고, 낼 수 있는 전력이 쓰는 전력보다 527 MW 더 많습니다.');
  // 25.9 % 비율·MW 계산 예비율 차이는 카드가 아니라 산출 방식 시트에만(4.1절).
  assert.ok(!JSON.stringify([p.price, p.power]).includes('25.9'));
  assert.match(p.facts['today-price'].lines[1], /25\.9%/);
  assert.match(p.facts['today-power'].lines[0], /^화면 MW로 계산하면 79\.4% · KPX 발표값 79\.5%와 0\.1%p 차이/);
});

// ---------------------------------------------------------------- 2. 상태 변형(T9, 4.1절 상태표·AnswerLine 표)
test('불러오는 중·확정 가격 없음·정시 경계 유예와 지연', () => {
  const m = load();
  const loading = viewOf(m, 'today-loading-8s');
  assert.deepEqual([loading.heroState, loading.answer, loading.price.state, loading.power.state, loading.forecast.state],
    ['loading', '가격과 전력 정보를 불러오는 중입니다.', 'loading', 'loading', 'loading']);
  const none = viewOf(m, 'today-price-no-data');
  assert.deepEqual([none.heroState, none.price.text], ['unavailable', 'KPX 확정 가격 없음 · 지금 거래시간 값이 아직 없습니다']);
  assert.equal(none.answer, '지금 거래시간의 확정 가격이 아직 없습니다. 낼 수 있는 전력이 쓰는 전력보다 527 MW 더 많습니다.');
  assert.deepEqual(none.forecast.dots, [], '확정 가격 점 없음');
  assert.ok(!none.forecast.chips.some(c => c.chip && c.chip.text === '확정 가격 · 지금 시간만'));
  const grace = viewOf(m, 'today-hour-boundary-grace');
  assert.equal(grace.price.checking, '다음 거래시간 값 확인 중');
  assert.equal(chipLabel(m, grace.price.chips[0]), '● 확정 가격', '유예 동안 이전 값과 칩 유지');
  assert.equal(grace.answer, '06–07시 제주 전기 도매가격은 1kWh에 115.3원이었고, 낼 수 있는 전력이 쓰는 전력보다 527 MW 더 많습니다.');
  assert.equal(grace.recheckPrice, false, '정시 뒤에 받은 값이라 다시 받지 않는다');
  const late = viewOf(m, 'today-hour-boundary-stale');
  assert.deepEqual([late.price.checking, late.price.chips[0].lag, late.price.chips[0].text], [null, 'stale', '지연 · 07h 값']);
  assert.match(late.answer, /^06–07시 제주 전기 도매가격은 1kWh에 115\.3원이었고/);
  assert.equal(late.forecast.selected, 8, '선택 기본값은 지금 거래시간(08h)');
});

test('KPI 조회 12분(지연)·61분(끊김): NowPill·칩·AnswerLine 과거형, 끊김이면 단계 판정과 실측 ● 철회', () => {
  const m = load();
  const s = m.strings;
  const stale = viewOf(m, 'today-kpi-age-12m', { advanceMs: 12 * 60_000, failAfter: true });
  assert.equal(stale.stage, 'stale');
  assert.equal(s.nowPillText({ now: inputFor(m, 'today-base').input.now + 12 * 60_000, fetchedAt: stale.fetchedAt }), '지연 · 06:03 조회 (12분 전)');
  assert.equal(stale.answer, '06:03 기준 제주 전기 도매가격은 1kWh에 115.3원이었습니다.');
  assert.deepEqual(stale.power.chips.map(c => chipLabel(m, c)), ['지연 · 06:03 값']);
  assert.equal(stale.power.level.text, '정상 · 넉넉합니다', '지연(10–60분)에는 판정 유지');
  assert.equal(chipLabel(m, stale.price.chips[0]), '● 확정 가격', '가격은 거래시간이 맞으면 조회 나이로 지연 처리하지 않는다');
  const broken = viewOf(m, 'today-kpi-age-61m', { advanceMs: 61 * 60_000, failAfter: true });
  assert.equal(broken.stage, 'broken');
  assert.equal(broken.answer, '06:03 뒤로 새 자료를 받지 못했습니다. 아래 값은 그때 값입니다.');
  assert.deepEqual([broken.power.level, broken.power.judgeNote], [null, '새 자료를 받지 못해 예비율 단계 판정을 멈췄습니다']);
  assert.deepEqual(broken.power.chips.map(c => [c.lag, chipLabel(m, c)]), [['broken', '06:03 값']]);
  assert.deepEqual([broken.price.chips[0].lag, chipLabel(m, broken.price.chips[0]), broken.price.checking], ['broken', '06:03 값', null],
    '07:04:57: 07h는 바로 앞 거래시간이지만 끊김이라 "확인 중"을 쓰지 않는다');
  assert.equal(broken.power.weather.chip.lag, 'broken', '날씨 실측 ●도 거둔다');
  assert.ok(!JSON.stringify(broken.power.chips).includes('"measured"'));
});

test('일부 실패·전부 실패(서버·네트워크): 카드마다 자기 상태, 전부 실패는 원인과 함께 한 번', () => {
  const m = load();
  const partial = viewOf(m, 'today-partial-kpis-error');
  assert.deepEqual([partial.heroState, partial.power.state, partial.power.text], ['ready', 'error', '전력 정보를 불러오지 못했습니다 · 06:03 KST']);
  assert.equal(partial.answer, '지금 시간 제주 전기 도매가격은 1kWh에 115.3원입니다. 전력 정보를 불러오지 못했습니다.');
  assert.equal(partial.allFailed, null);
  for (const [id, cause] of [['today-all-error-server', '서버 응답 없음'], ['today-all-error-network', '네트워크 연결을 확인해 주세요']]) {
    const v = viewOf(m, id);
    // 값이 하나도 없으면 AnswerLine이 이미 "지금은 시장 정보를 불러올 수 없습니다."라서 안내 상자는 원인만(같은 문장 두 번 금지, round 1).
    assert.deepEqual(v.allFailed, { text: cause, showCards: false }, id);
    assert.deepEqual([v.answer, v.heroState], ['지금은 시장 정보를 불러올 수 없습니다.', 'error'], id);
  }
});

test('수요 출처 ②④⑤⑥: 머리 칩 ▨ 추정값 기준 + KPI별 칩, 판정 없음과 그 까닭, 공급능력 미수신', () => {
  const m = load();
  const cases = {
    'today-demand-src-2': [['▨ 추정 · 실측 아님', '▨ 추정 · 실측 아님'], '출처를 확인할 수 없어 예비율 단계를 판정하지 않습니다', 'API 서버 값 (출처 확인 불가)'],
    'today-demand-src-4': [['▨ 추정 · 실측 아님', '▨ 기본값 · 실측 아님'], 'API 서버가 실측 대신 기본 예비율(25%)을 보냈습니다', 'EPSIS 30일 평균 (API 서버 대체값)'],
    'today-demand-src-5': [['▨ 추정 · 실측 아님', '▨ 전국 값 · 제주 아님'], 'API 서버가 제주 대신 전국 예비율을 보냈습니다', '전국 값 × 0.018 비례 추정 (API 서버)'],
    'today-demand-src-6': [['▨ 시뮬레이션 · 실측 아님', '▨ 기본값 · 실측 아님'], 'API 서버가 실측 대신 기본값(수요 650 MW·예비율 25%)을 보냈습니다', 'API 서버 기본값 650 MW·25% (실측 아님)'],
  };
  for (const [id, [kpiChips, why, source]] of Object.entries(cases)) {
    const v = viewOf(m, id);
    assert.deepEqual(v.power.chips.map(c => chipLabel(m, c)), ['▨ 추정값 기준'], id);
    assert.deepEqual([v.power.demand.chip, v.power.reserve.chip].map(c => chipLabel(m, c)), kpiChips, id);
    assert.deepEqual([v.power.level, v.power.judgeNote, v.power.supply.kind, v.power.supply.text], [null, why, 'missing', '공급능력 미수신'], id);
    assert.equal(v.power.footer.source, `출처 ${source}`, id);
    assert.equal(v.answer, '지금 시간 제주 전기 도매가격은 1kWh에 115.3원입니다. 전력 여유는 실측 값이 없어 판정하지 않습니다.', id);
  }
  assert.equal(viewOf(m, 'today-demand-src-4').power.reserve.note, 'API 기본값 (실측 아님)');
});

test('공급능력 짝 불일치·기상 file/default/none·KPI 빈 값', () => {
  const m = load();
  const pair = viewOf(m, 'today-supply-pair-mismatch');
  assert.deepEqual([pair.power.supply.kind, pair.power.supply.text], ['missing', '공급능력 미수신 · 예비율은 KPX 발표값입니다']);
  assert.equal(pair.answer, '지금 시간 제주 전기 도매가격은 1kWh에 115.3원이고, 예비율은 79.5%로 넉넉합니다(앱 기준).');
  assert.deepEqual(pair.facts['today-power'].lines, [], 'MW 계산 차이는 짝이 맞을 때만');
  const file = viewOf(m, 'today-weather-file');
  assert.deepEqual([chipLabel(m, file.power.weather.chip), file.power.footer.extra], ['▨ 추정 · 실시간 아님', '날씨 기상청 과거 자료 24시간 평균 (API 서버) · 실시간 아님']);
  const def = viewOf(m, 'today-weather-default');
  assert.deepEqual([chipLabel(m, def.power.weather.chip), m.rich.lineText(def.power.weather.line), def.power.footer.extra],
    ['▨ 기본값 · 실측 아님', '날씨 5.0°C · 바람 3.0 m/s · 맑음', '날씨 API 서버 기본값 (실측 아님)']);
  const none = viewOf(m, 'today-weather-none');
  assert.equal(none.power.state, 'error', 'KPI 응답이 객체가 아니면 모양 오류: 카드 오류, 날씨 줄 없음');
  const empty = viewOf(m, 'today-kpis-nulls');
  assert.deepEqual([empty.power.chips, empty.power.demand.value, empty.power.reserve.value, empty.power.weather], [[], null, null, null]);
  assert.deepEqual([empty.power.judgeNote, empty.power.supply.text], ['예비율 값이 없어 단계를 판정하지 않습니다', '공급능력 미수신']);
  assert.equal(empty.answer, '지금 시간 제주 전기 도매가격은 1kWh에 115.3원입니다. 전력 값이 비어 있어 판정하지 않습니다.');
});

test('예측 대체값·모르는 모델·예측일 어제·0.00 입력(12–14h)·빈 예측', () => {
  const m = load();
  for (const id of ['today-forecast-fallback', 'today-forecast-fallback-epsis', 'today-forecast-unknown-model']) {
    const v = viewOf(m, id);
    assert.deepEqual([v.forecast.kind, chipLabel(m, v.forecast.chips[0]), v.forecast.view.band, v.forecast.uncertainty],
      ['fallback', '▨ 추정 · API 대체값 (모델 예측 아님)', null, null], id);
    assert.deepEqual([v.price.compare, v.forecast.dots], [[], []], `${id}: 대체값과는 비교하지 않는다`);
    assert.deepEqual([v.accuracy.documented, v.accuracy.unregistered], [null, '이 모델의 검증 수치는 앱에 등록되어 있지 않습니다'], id);
    assert.equal(m.rich.lineText(v.forecast.reading), 'API 대체값 145.2원', id);
  }
  const old = viewOf(m, 'today-forecast-date-yesterday');
  assert.deepEqual(old.forecast.chips.map(c => chipLabel(m, c)), ['┅ 예측 · 기준선', '예측일 지연']);
  assert.deepEqual([old.price.compare, old.forecast.dots], [[], []], '예측일 불일치: 비교 줄·확정 점 없음');
  const zero = viewOf(m, 'today-forecast-zero-12-14h', { selectedHour: 13 });
  assert.match(zero.forecast.summary, /\(0\.00 입력 3시간 제외\)\.$/);
  assert.equal(m.rich.lineText(zero.forecast.reading), '기준선 입력 0.00 · 예측 아님');
  assert.equal(zero.forecast.periodNote, '평균·최저·최고에서 0.00 입력 3시간을 뺐습니다');
  assert.equal(viewOf(m, 'today-base').forecast.periodNote, null, '0.00 입력 시간이 없으면 각주도 없다(round 1 멘토)');
  assert.ok(!zero.forecast.uncertainty.includes('12–13시'), '0.00 입력 시간은 가장 불확실한 시간에서도 뺀다');
  const day = zero.forecast.periods.find(p => p.label.startsWith('주간'));
  assert.notEqual(day.min, '0.0');
  assert.equal(zero.forecast.rows[12].point.kind, 'zero');
  const empty = viewOf(m, 'today-forecast-empty');
  assert.deepEqual([empty.forecast.state, empty.forecast.text], ['empty', '그릴 수 있는 예측 값이 없습니다']);
});

test('출처 표 행: 받는 중·실패·아직 조회하지 않음(표를 연 화면이 받기 시작하기 전 한 순간, round 1 D15)을 구분한다', () => {
  const m = load();
  const { input } = inputFor(m, 'today-base');
  const idle = { data: null, fetchedAt: null, status: 'idle', error: null };
  const loading = { ...idle, status: 'loading' };
  const failedR = { ...idle, status: 'error', error: new m.client.ApiNetworkError('/x') };
  const rows = m.model.todayView({ ...input, dayAhead: idle, forecast: loading, kpis: failedR, market: idle, priceAvailable: false }).sources;
  const basis = Object.fromEntries(rows.map(r => [r.id, r.basis]));
  assert.deepEqual([basis['day-ahead'], basis.forecast, basis.demand, basis.supply, basis.weather, basis.market],
    ['이번 방문에서 아직 조회하지 않음', '불러오는 중', '불러오지 못함', '불러오지 못함', '불러오지 못함', '이번 방문에서 아직 조회하지 않음']);
  assert.ok(rows.every(r => r.chip === null || r.id === 'accuracy'), '값이 없으면 상태 칩을 붙이지 않는다');
});

test('시계 차이 +3분(서버 시각 기준)·+13시간(서버 불신, 기기 시계)', () => {
  const m = load();
  const plus3 = inputFor(m, 'today-clock-skew-plus3m');
  // 서버 시각 06:03:57.768 − 기기 06:06:57.000 = −179.232초.
  assert.deepEqual([plus3.skew.plausible, plus3.skew.notable, plus3.skew.skewMs], [true, true, -179_232]);
  assert.equal(m.model.todayView(plus3.input).answer, EXPECTED.answer, '서버 시각 기준이면 기준 스냅숏과 같다');
  const plus13 = inputFor(m, 'today-clock-skew-plus13h');
  assert.equal(plus13.skew.plausible, false);
  const v = m.model.todayView(plus13.input);
  assert.deepEqual([v.price.chips[0].text, v.price.checking, v.forecast.selected], ['지연 · 07h 값', null, 20],
    '기기 시계(19:03)로 거래시간 20을 본다. 07h는 바로 앞 거래시간이 아니라 정시 뒤 5분 안이어도 유예하지 않는다');
  // 자정 경계: 00:01에 어제 24h 값은 유예(다음 거래시간 값 확인 중).
  const { input } = inputFor(m, 'today-base');
  const day = input.dayAhead.data;
  const midnight = Date.UTC(2026, 10, 26, 0, 1, 0) - 9 * 3600e3;
  const yesterday24 = { ...input, now: midnight, dayAhead: { ...input.dayAhead, data: { ...day, trade_day: '20261125', trade_hour: 24 }, fetchedAt: midnight } };
  yesterday24.priceAvailable = m.sp.isAvailableDayAhead(yesterday24.dayAhead.data);
  assert.equal(m.model.todayView(yesterday24).price.checking, '다음 거래시간 값 확인 중');
});

test('모든 T9 변형: 평문 조각에 숫자 없음, AnswerLine 있음, 선택 시간 유효', () => {
  const m = load();
  assert.ok(TODAY_VARIANTS.length >= 25);
  for (const id of TODAY_VARIANTS) {
    const opts = id === 'today-kpi-age-12m' ? { advanceMs: 12 * 60_000, failAfter: true } : id === 'today-kpi-age-61m' ? { advanceMs: 61 * 60_000, failAfter: true } : {};
    const v = viewOf(m, id, opts);
    assert.ok(v.answer.length > 0, id);
    const lines = [];
    if (v.price.state === 'ready') lines.push(...v.price.compare);
    if (v.power.state === 'ready') {
      if (v.power.supply.kind === 'paired') lines.push(v.power.supply.formula);
      if (v.power.weather) lines.push(v.power.weather.line);
    }
    if (v.forecast.state === 'ready') {
      lines.push(v.forecast.reading);
      assert.ok(v.forecast.selected >= 1 && v.forecast.selected <= 24, id);
    }
    for (const line of lines) for (const seg of line) if ('t' in seg) assert.doesNotMatch(seg.t, /\d/, `${id}: 평문 조각 "${seg.t}"`);
  }
});

// ---------------------------------------------------------------- 3. 시계·경계 도우미
test('정시 경계: HH:00:15·HH:02:00 재조회, 다른 탭에서 정시를 넘겨 돌아오면 한 번 다시 받기', () => {
  const m = load();
  const K = (h, mi, s = 0) => Date.UTC(2026, 10, 26, h, mi, s) - 9 * 3600e3;
  assert.equal(m.model.nextBoundaryCheck(K(6, 3, 57)), K(7, 0, 15));
  assert.equal(m.model.nextBoundaryCheck(K(7, 0, 0)), K(7, 0, 15));
  assert.equal(m.model.nextBoundaryCheck(K(7, 0, 15)), K(7, 2, 0));
  assert.equal(m.model.nextBoundaryCheck(K(7, 1, 30)), K(7, 2, 0));
  assert.equal(m.model.nextBoundaryCheck(K(23, 59, 30)), K(24, 0, 15));
  assert.equal(m.model.sinceHourStart(K(7, 4, 59)), 299_000);
  const { input } = inputFor(m, 'today-base');
  const later = { ...input, now: K(7, 1, 0) };
  assert.equal(m.model.todayView(later).recheckPrice, true, '06:03에 받은 07h 값, 지금 07:01');
  assert.equal(m.model.todayView(input).recheckPrice, false);
});

// ---------------------------------------------------------------- 4. 카드 렌더(노드 수준 I2·의미)
test('카드 렌더: 모든 T9 변형 × 390 차트·표 / 320 / 768에서 표지 밖 숫자 0, 값은 칩·출처 있는 카드 안', () => {
  const opts = id => (id === 'today-kpi-age-12m' ? { advanceMs: 12 * 60_000, failAfter: true } : id === 'today-kpi-age-61m' ? { advanceMs: 61 * 60_000, failAfter: true } : {});
  for (const [width, forceTable] of [[390, false], [390, true], [320, false], [768, false]]) {
    const m = load(width, { forceTable });
    for (const id of ['today-base', ...TODAY_VARIANTS]) {
      const { input } = inputFor(m, id, opts(id));
      const root = renderToday(m, m.model.todayView(input), input.now);
      assertI2(root, `${id}@${width}${forceTable ? ' 표' : ''}`);
    }
  }
});

test('줄바꿈 단위(F13c 갈림 탐침): 차트 범례의 짧은 괄호·80%, 판독 줄의 80% 구간 범위, 가격 카드 메타 줄의 시각 범위는 한 줄 조각', () => {
  const m = load();
  const { input } = inputFor(m, 'today-base');
  const root = renderToday(m, m.model.todayView(input), input.now);
  const nowrapOf = node => all(node, n => ds(n).pdNowrap === 'true').map(textOf);
  // 360에서 "(▲ 위쪽" / "잘림)", 320에서 판독 줄 "116.9–" / "174.6원"으로 갈렸다(F13c split_probe).
  const legend = nowrapOf(all(root, n => ds(n).pdMeta === 'legend')[0]);
  assert.ok(legend.includes('(어제 같은 시간 가격)') && legend.includes('80%'), legend.join(' | '));
  assert.ok(nowrapOf(all(root, n => n.props.testID === 'today-reading')[0]).includes('116.9–174.6원'), '범위는 두 값과 줄표·단위를 한 묶음으로');
  assert.ok(nowrapOf(all(root, n => ds(n).pdMeta === 'hour')[0]).includes('06–07시'));
});

test('카드 렌더 의미: today-hero data-state, [i] 4개·[출처 표] 4개 이름, 용어 링크 5개, 스텝퍼, 표 12행/24행, 44 px', () => {
  const m = load();
  const { input } = inputFor(m, 'today-base');
  const view = m.model.todayView(input);
  const root = renderToday(m, view, input.now);
  const hero = all(root, n => n.props.testID === 'today-hero')[0];
  assert.deepEqual([ds(hero).pdCard, ds(hero).state], ['true', 'ready']);
  const presses = all(root, n => n.type === 'Pressable');
  const names = presses.map(p => p.props['aria-label']);
  assert.deepEqual(names.filter(n => n.startsWith('산출 방식 보기')),
    ['산출 방식 보기: 지금 시간 전기값', '산출 방식 보기: 지금 제주 전력', '산출 방식 보기: 오늘 24시간 가격', '산출 방식 보기: 예측은 얼마나 맞나요?']);
  assert.deepEqual(names.filter(n => n.startsWith('출처 표')),
    ['출처 표: 지금 시간 전기값', '출처 표: 지금 제주 전력', '출처 표: 오늘 24시간 가격', '출처 표: 예측은 얼마나 맞나요?']);
  assert.deepEqual(names.filter(n => n.startsWith('용어 풀이')).slice(0, 5),
    ['용어 풀이: SMP', '용어 풀이: 하루전시장', '용어 풀이: 예비율', '용어 풀이: 공급능력', '용어 풀이: 80% 구간']);
  assert.ok(names.includes('이전 거래시간') && names.includes('다음 거래시간'));
  // round 1 D4: 지금 거래시간이어도 [지금으로]는 남고 까닭 있는 비활성(누른 버튼이 사라져 초점이 body로 떨어졌다).
  assert.ok(!names.includes('지금으로') && names.includes('지금으로 · 지금 거래시간입니다'), '지금이면 까닭 있는 비활성');
  presses.forEach(p => {
    const s = flat(p.props.style);
    assert.ok(Math.max(s.minHeight || 0, s.height || 0) >= 44, `${p.props['aria-label']} 높이`);
  });
  const chart = all(root, n => n.props.role === 'img' && String(n.props['aria-label']).startsWith('기준선 예측 SMP 24시간'));
  assert.equal(chart.length, 1);
  assert.match(chart[0].props['aria-label'], /가장 비싸고\(212\.3원\)/);
  const picked = load(390);
  const moved = m.model.todayView({ ...input, selectedHour: 9 });
  const r2 = renderToday(picked, moved, input.now);
  assert.ok(all(r2, n => n.type === 'Pressable').some(p => p.props['aria-label'] === '지금으로'));
  // 600 px 아래(390)의 시간별 표는 행 카드 목록이다(D2): 시간대 요약(4칸)만 표.
  const table = load(390, { forceTable: true });
  const rt = renderToday(table, table.model.todayView(input), input.now);
  assert.deepEqual(all(rt, n => n.props.role === 'table').map(t => t.props['aria-label']), ['시간대 요약']);
  const [list] = all(rt, n => n.props.role === 'list');
  assert.equal(list.props['aria-label'], '시간별 기준선 예측');
  const items = all(list, n => n.props.role === 'listitem');
  assert.equal(items.length, 24, 'forceTable은 "24시간 전체 보기"를 누른 상태');
  const now = items.filter(n => n.props['aria-current'] === 'true');
  assert.equal(now.length, 1);
  // 머리 = 시각 범위 / 둘째 줄 = 거래시간 번호(D11), 값 조각은 표 열 순서 거래시간 · 확정 · 기준선 · 80% 구간(D2), 확정 가격은 받은 시간에만.
  assert.equal(textOf(now[0]), '06–07시07h확정 가격 115.3 · 기준선 예측 145.2 · 80% 구간 116.9–174.6');
  assert.equal(textOf(items[0]), '00–01시01h기준선 예측 139.3 · 80% 구간 120.5–164.0', '확정 가격이 없는 시간은 그 조각을 뺀다');
  const nowrapItems = all(now[0], n => ds(n).pdNowrap === 'true').map(textOf);
  assert.ok(nowrapItems.includes('80% 구간 116.9–174.6') && nowrapItems.includes('06–07시'), '범위·시각은 한 줄 조각');
  // 600 이상(768)은 표: 열 순서 거래시간 · 확정 가격 · 기준선 예측 · 80% 구간.
  const wide = load(768, { forceTable: true });
  const rw = renderToday(wide, wide.model.todayView(input), input.now);
  const tables = all(rw, n => n.props.role === 'table');
  assert.deepEqual(tables.map(t => t.props['aria-label']), ['시간대 요약', '시간별 기준선 예측']);
  assert.deepEqual(all(tables[1], n => n.props.role === 'columnheader').map(textOf), ['거래시간', '확정 가격', '기준선 예측', '80% 구간']);
  const current = all(tables[1], n => n.props.role === 'row' && n.props['aria-current'] === 'true');
  assert.equal(current.length, 1);
  assert.deepEqual(all(current[0], n => n.props.role === 'cell').map(textOf), ['07h · 06–07시', '115.3', '145.2', '116.9–174.6'], '표 칸은 거래시간 번호와 시각 범위(D11 허용)');
  assert.equal(textOf(all(tables[1], n => n.props.role === 'row')[1]).split('—').length, 2, '확정 가격이 없는 표 칸은 "—"');
  const twelve = view.forecast.rows.slice(view.forecast.window.from - 1, view.forecast.window.to);
  assert.deepEqual([twelve.length, twelve[0].hour, twelve[11].hour], [12, 7, 18]);
  const tail = m.model.todayView({ ...input, selectedHour: 20 }).forecast.window;
  assert.deepEqual([tail.from, tail.to], [13, 24], '하루 끝을 넘으면 마지막 12행');
});

test('확정 가격 점은 이번 방문 동안 쌓인다(D10): 받은 거래시간마다 점 하나, 표·행 카드·판독 줄에도, 어제 값·대체값 예측·0.00 입력 시간은 빼고', () => {
  const m = load();
  const { input } = inputFor(m, 'today-base');
  const base = input.dayAhead.data;
  const seen = [
    { data: { ...base, trade_hour: 6, current_smp: 120.4 }, fetchedAt: input.now - 3_600_000, available: true },
    { data: { ...base, trade_day: '20261125', trade_hour: 9, current_smp: 99.9 }, fetchedAt: input.now - 86_400_000, available: true },
    { data: { ...base, trade_hour: 5, current_smp: 88.8 }, fetchedAt: input.now - 7_200_000, available: false },
  ];
  const v = m.model.todayView({ ...input, seen, selectedHour: 6 });
  assert.deepEqual(v.forecast.dots.map(d => [d.tradeHour, d.price.value, d.price.state]), [[6, 120.4, 'confirmed'], [7, 115.27, 'confirmed']],
    '어제(11-25) 값과 판정을 못 지난 값은 점이 아니다');
  assert.equal(m.rich.lineText(v.forecast.reading), '기준선 예측 119.1원 · 80% 구간 93.6–148.5원 · 확정 가격 120.4원', '고른 지난 시간의 확정 가격');
  assert.deepEqual(v.forecast.rows.filter(r => r.confirmed).map(r => r.hour), [6, 7]);
  assert.equal(chipLabel(m, v.forecast.chips[1]), '● 확정 가격 · 지금 시간만');
  // 대체값 예측이면 비교하지 않으므로 쌓인 값도 점으로 두지 않는다(7.8절 1).
  const fb = inputFor(m, 'today-forecast-fallback').input;
  assert.deepEqual(m.model.todayView({ ...fb, seen }).forecast.dots, []);
  // 띠가 그림 위로 넘치지 않으면 범례에 ▲ 뜻을 붙이지 않는다.
  assert.ok(!m.model.todayView({ ...input, forecast: { ...input.forecast, data: { ...input.forecast.data, q90: input.forecast.data.q50.slice() } } })
    .forecast.legend.includes('▲'));
});

test('오늘 카드 글 밀도(D17): 요약·불확실성은 본문 크기 한 문단, 범위 막대 이름표에 값을 다시 쓰지 않음, 둘러보기 배너는 제목 한 줄', () => {
  const m = load();
  const { input } = inputFor(m, 'today-base');
  const root = renderToday(m, m.model.todayView(input), input.now);
  const summary = all(root, n => n.props.testID === 'today-summary')[0];
  assert.equal(flat(summary.props.style).fontSize, 15);
  assert.equal(textOf(summary), '기준선 예측으로는 07–08시가 가장 비싸고(212.3원), 02–05시가 가장 쌉니다(102.3원). 하루 평균 169.2원. '
    + '가장 불확실한 시간: 12–13시 (80% 구간 폭 326.4원).');
  const bar = all(root, n => n.props.testID === 'range-bar')[0];
  assert.equal(all(bar, n => ds(n).pdValue !== undefined).length, 0, '수요 값은 수급 식에만');
});

test('전력 카드 폭별: 320은 KPI 위아래, 360 이상 나란히(라벨·값 한 줄), 날씨는 모든 폭에서 범위 막대 아래 자기 줄(D14)', () => {
  for (const [width, direction, weatherInRow] of [[320, 'column', false], [390, 'row', false], [768, 'row', false], [1280, 'row', false]]) {
    const m = load(width);
    const { input } = inputFor(m, 'today-base');
    const root = tree(h(m.PowerCard, { view: m.model.todayView(input).power, onMethod: () => {}, onSources: () => {}, onRetry: () => {} }));
    const groups = all(root, n => n.props.role === 'group');
    assert.deepEqual(groups.map(g => g.props['aria-label']), [
      '쓰는 전력(수요) 664 메가와트, 상태: 실측, 출처 KPX 제주 실시간 (API 서버)', '예비율 79.5 퍼센트, 상태: 실측, 출처 KPX 제주 실시간 (API 서버)']);
    const kpiRow = all(root, n => n.props.testID === 'today-kpis')[0];
    assert.equal(flat(kpiRow.props.style).flexDirection, direction, `${width} KPI 방향`);
    assert.equal(all(kpiRow, n => n.props.testID === 'today-weather').length > 0, weatherInRow, `${width} 날씨 칸`);
    // 라벨과 값이 한 줄(줄바꿈 허용) 묶음 안에 있다(4.1절 "쓰는 전력(수요)  664 MW").
    groups.forEach(g => assert.equal(flat(g.children[0].props.style).flexDirection, 'row'));
  }
});

// ---------------------------------------------------------------- 5. F7이 쓰는 공용 부품 동작
test('HourStepper wrap: 01h ‹ → 24h "(끝으로)", 24h › → 01h "(처음으로)", 양 끝도 활성, 값은 라벨 함수', () => {
  const m = load();
  const seen = [];
  const label = hr => m.strings.hourStepLabel(hr, hr === 7);
  const at = hour => all(tree(h(m.HourStepper, { hour, onChange: v => seen.push(v), wrap: true, label })), n => n.type === 'Pressable');
  const [prev1] = at(1);
  assert.equal(Boolean(prev1.props.disabled), false, '돌기 모드에서는 양 끝도 활성');
  assert.equal(prev1.props['aria-label'], '이전 거래시간', '비활성 까닭이 붙지 않는다');
  prev1.props.onPress();
  const [, next24] = at(24);
  next24.props.onPress();
  assert.deepEqual(seen, [24, 1]);
  const mid = tree(h(m.HourStepper, { hour: 7, onChange: () => {}, wrap: true, label }));
  assert.equal(textOf(all(mid, n => ds(n).pdMeta === 'hour')[0]), '07h · 06–07시 · 지금');
  // 알림 글자: 셸 live region에 보낸다(공급자 밖에서는 아무것도 하지 않으므로 글자 함수만 본다).
  assert.deepEqual([m.strings.WRAP_FIRST, m.strings.WRAP_LAST], ['(처음으로)', '(끝으로)']);
});

test('TermLink는 Enter로 열고(Space 아님) 라벨 숫자는 메타, KpiBlock inline, Value derived, 지연 칩 글자 바꾸기, [i] 상자 44', () => {
  const m = load();
  let opened = 0;
  const link = all(tree(h(m.TermLink, { label: '80% 구간', onPress: () => { opened += 1; } })), n => n.type === 'Pressable')[0];
  let prevented = 0;
  link.props.onKeyDown({ key: 'Enter', preventDefault: () => { prevented += 1; } });
  link.props.onKeyDown({ key: ' ', preventDefault: () => {} });
  assert.deepEqual([opened, prevented], [1, 1]);
  assert.equal(ds(all(link, n => n.type === 'Text')[0]).pdMeta, 'term');
  const { input } = inputFor(m, 'today-base');
  const view = m.model.todayView(input);
  const reserve = view.power.supply.reserve;
  const derived = all(tree(h(m.Value, { v: reserve, format: m.format.mw, derived: true })), n => n.type === 'Text')[0];
  assert.deepEqual([ds(derived).pdMeta, ds(derived).pdValue, ds(derived).pdRaw], ['derived', undefined, undefined]);
  const chip = tree(h(m.ProvenanceChip, { lag: 'stale', at: input.now, text: '지연 · 07h 값' }));
  assert.equal(textOf(chip), '지연 · 07h 값');
  const card = tree(h(m.Card, { title: '오늘 24시간 가격', onMethod: () => {} }));
  const info = all(card, n => n.type === 'Pressable')[0];
  assert.deepEqual([flat(info.props.style).width, flat(info.props.style).height], [44, 44], '[i] 상자는 44×44 그대로');
  const overhang = all(card, n => n.children.includes(info))[0];
  assert.equal(flat(overhang.props.style).marginVertical, -10, '머리 줄은 칩 높이만 차지하고 [i]는 여백으로 내민다');
  assert.equal(ds(all(card, n => n.props.role === 'heading')[0]).pdMeta, 'label', '숫자가 든 제목은 메타');
});

// ---------------------------------------------------------------- 알아보기(F10, 4.4절): 출처 행 · 응답 상태 매핑 · 둘러보기 5단계 범례
function loadGuide(width = 390) {
  const rn = { ...defaultMocks()['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG } });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  return { guide: at('features/guide/model.ts'), sheets: at('features/guide/sheets.tsx'), types: at('data/types.ts'), adapters: at('data/adapters.ts'),
    provenance: at('lib/provenance.ts'), kst: at('lib/kst.ts') };
}
const RS = '/api/v1/realtime-status';
const rsOf = (g, id) => {
  const r = resolveVariant(id).responses[RS];
  return r.status >= 400 ? { data: null, fetchedAt: null, status: 'error', error: new Error(String(r.status)) }
    : { data: g.types.parseRealtimeStatus(r.body), fetchedAt: 0, status: 'ready', error: null };
};
const statusOf = (g, id) => g.guide.statusView(rsOf(g, id)).rows.map(r => [r.item, r.status, r.tone]);

test('출처 카드 머리 칩(round 1 D15): 서버 응답 상태를 쓰되 표의 행을 하나라도 불러오지 못했으면 초록 "모두 응답" 대신 그 사실', () => {
  const g = loadGuide();
  const chip = (id, failed) => g.guide.sourcesChip(g.guide.statusView(rsOf(g, id)).overall, failed);
  assert.deepEqual(chip('guide-rs-base', false), { text: 'API 서버: 모두 응답', tone: 'ok' });
  assert.deepEqual(chip('guide-rs-base', true), { text: '일부 자료를 불러오지 못함', tone: 'watch' }, '칩과 표가 엇갈리지 않게');
  assert.deepEqual(chip('guide-rs-partial', true), { text: 'API 서버: 일부 응답', tone: 'watch' }, '서버가 이미 좋지 않다고 하면 서버 상태 그대로');
  assert.deepEqual(chip('guide-rs-disconnected', false), { text: 'API 서버: 응답 없음', tone: 'danger' });
  assert.equal(chip('guide-rs-error', true), null, '응답 상태를 모르면 칩이 없다(카드의 오류 안내가 말한다)');
});

test('응답 상태 매핑(4.4절): status·data_date_status·overall_status → 한국어, 목록 밖·빠진 부분은 "알 수 없음", 응답 시각은 timestamp(KST), "연결됨" 없음', () => {
  const g = loadGuide();
  const base = g.guide.statusView(rsOf(g, 'guide-rs-base'));
  assert.deepEqual(base.rows.map(r => [r.id, r.source, r.basis]), ['smp', 'supply', 'weather'].map(k => [`status:${k}`, 'API 서버 응답 상태(realtime-status)', '응답 시각 07:43']));
  assert.deepEqual(base.overall, { text: '모두 응답', tone: 'ok', at: g.kst.parseApiTime('2026-11-26T07:43:21.732618') });
  assert.deepEqual(statusOf(g, 'guide-rs-base'), [['응답 상태 · SMP', '응답 정상 · 과거 날짜 자료', 'ok'], ['응답 상태 · 수급', '응답 정상', 'ok'], ['응답 상태 · 기상', '응답 정상', 'ok']]);
  assert.deepEqual(statusOf(g, 'guide-rs-all-current')[0], ['응답 상태 · SMP', '응답 정상 · 오늘 자료', 'ok']);
  assert.deepEqual(statusOf(g, 'guide-rs-partial').map(r => [r[1], r[2]]), [['응답 정상 · 과거 날짜 자료', 'ok'], ['자료 없음', 'watch'], ['오류', 'danger']]);
  assert.deepEqual(g.guide.statusView(rsOf(g, 'guide-rs-disconnected')).overall.text, '응답 없음');
  assert.deepEqual(statusOf(g, 'guide-rs-unknown').map(r => [r[1], r[2]]), [['알 수 없음 · 날짜 미상', 'neutral'], ['알 수 없음', 'neutral'], ['알 수 없음', 'neutral']]);
  assert.equal(statusOf(g, 'guide-rs-future-date')[0][1], '응답 정상 · 미래 날짜 자료');
  assert.deepEqual(statusOf(g, 'guide-rs-no-date-status').map(r => r[1]), ['응답 정상', '자료 없음', '응답 정상']);
  // 목록 밖 값과 빠진 부분은 닫힌 쪽으로(4.4절).
  const odd = g.guide.statusView({ data: { timestamp: null, overall_status: 'great', smp_api: { status: 'connected!', data_date_status: 'today' }, power_supply_api: null,
    weather_api: { status: 'constructor', data_date_status: null } }, fetchedAt: 0, status: 'ready', error: null });
  assert.deepEqual(odd.rows.map(r => [r.status, r.tone, r.basis]), [['알 수 없음 · 날짜 알 수 없음', 'neutral', '응답 시각 미제공'],
    ['알 수 없음', 'neutral', '응답 시각 미제공'], ['알 수 없음', 'neutral', '응답 시각 미제공']]);
  assert.deepEqual([odd.overall.text, odd.overall.tone], ['알 수 없음', 'neutral']);
  // 값이 없으면 받는 중 · 실패(502) · 아직 조회하지 않음(표를 연 화면이 받기 시작하기 전 한 순간, D15), 상태 글자는 비운다.
  const pending = s => g.guide.statusView({ data: null, fetchedAt: null, status: s, error: null }).rows.map(r => [r.basis, r.status]);
  assert.deepEqual([pending('idle')[0], pending('loading')[0], g.guide.statusView(rsOf(g, 'guide-rs-error')).rows[0]].map(r => (Array.isArray(r) ? r : [r.basis, r.status])),
    [['이번 방문에서 아직 조회하지 않음', null], ['불러오는 중', null], ['불러오지 못함', null]]);
  const every = loadManifest().variants.filter(v => v.sweep === 'A').flatMap(v => g.guide.statusView(rsOf(g, v.id)).rows.map(r => `${r.item} ${r.status} ${r.basis}`));
  assert.ok(every.every(t => !/연결됨/.test(t)));
});

test('출처 행(4.4절): 오늘 8행 + 입찰 계산 + 응답 상태 3행, 응답 상태 행은 수신 상태 pill · [출처 표]의 row=status는 세 행을 함께 표시 · 표지 밖 숫자 0', () => {
  const m = load();
  const g = loadGuide();
  const today = viewOf(m, 'today-base').sources;
  const status = g.guide.statusView(rsOf(g, 'guide-rs-partial')).rows;
  const bidding = { id: 'bidding', item: '입찰 계산', source: 'API 서버 입찰 계산 (기준선 분위수)', basis: '조회 06:03', chip: null, status: null, link: null };
  const rows = [...today, bidding, ...status];
  assert.deepEqual(rows.map(r => r.id), ['day-ahead', 'forecast', 'accuracy', 'demand', 'supply', 'weather', 'market', 'bidding', 'status:smp', 'status:supply', 'status:weather']);
  // 768 이상은 표, 다만 ≥ 1024 시트(오른쪽 400)는 목록이고 알아보기 화면(page, 본문 640)은 표다.
  for (const [width, page, table] of [[390, false, 0], [768, false, 1], [1280, false, 0], [1280, true, 1]]) {
    const root = tree(h(loadGuide(width).sheets.SourcesTable, { rows, highlight: 'status', page }));
    assert.equal(all(root, n => n.props.role === 'table').length, table, `${width} page=${page}: 표/목록`);
    // 출처 표는 가로 스크롤 없이 칸이 폭을 나눈다(긴 글 칸이 넘치고 끝 칸이 잘리던 것, F10 측정).
    assert.equal(all(root, n => n.type === 'ScrollView' && n.props.horizontal).length, 0, `${width}: 가로 스크롤 없음`);
    assert.equal(all(root, n => n.props['aria-current'] === 'true').length, 3, `${width}: row=status`);
    assert.deepEqual(all(root, n => ds(n).pdPill).map(n => [ds(n).pdPill, textOf(n)]), [['ok', '응답 정상 · 과거 날짜 자료'], ['watch', '자료 없음'], ['danger', '오류']]);
    assertI2(root, `출처 표 ${width}`);
  }
  // 평가 기간은 한 줄로 묶는다(F8 관찰: 400 px 시트에서 "2025-" / "07-22~12-18").
  for (const width of [390, 768]) {
    const root = tree(h(loadGuide(width).sheets.SourcesTable, { rows, highlight: null }));
    assert.equal(all(root, n => ds(n).pdNowrap && textOf(n) === '2025-07-22~12-18').length, 1, `${width}: 평가 기간 한 줄`);
  }
});

test('둘러보기 5단계 범례(4.4절): 이번 방문 값(664 MW 실측 · 115.3 확정 · 145.2 기준선 예측, 가격은 시각 범위와 · 95% 낙찰 가능성 · 1,716만원 예시), 값이 없으면 표시와 설명만', () => {
  const m = load();
  const g = loadGuide();
  const view = viewOf(m, 'today-base');
  const seg = resolveVariant('bidding-base').responses['/api/v1/bidding/optimized-segments'].body;
  const first = g.adapters.adaptSegments(g.types.parseSegments(seg), 0).hours.find(x => x.tradeHour === 19).segments[0];
  const row = rows => rows.map(r => [g.provenance.chipText(r.chip), r.text]);
  // round 1 멘토: 실측·확정 행에 카드의 "● 실측"(수요)과 "● 확정 가격"을 함께 들고, 가격은 카드 문장과 같은 꼴("115.3원/kWh")로.
  assert.deepEqual(row(g.guide.tourLegend({ today: view.legend, forecastKind: view.forecast.kind, probability: first.probability.value, statementTotal: 17160000 })), [
    ['● 실측·확정', '664 MW 실측 · 115.3원/kWh 확정 (06–07시)'], ['┅ 예측 · 기준선', '145.2원/kWh 기준선 예측 (06–07시)'],
    ['▨ 추정', '95% 낙찰 가능성 (예시 발전소 1구간)'], ['┆ 예시', '1,716만원 예시 정산서']]);
  const blank = { demand: null, confirmed: null, forecast: null };
  const none = row(g.guide.tourLegend({ today: blank, forecastKind: null, probability: null, statementTotal: 17160000 }));
  assert.deepEqual(none.slice(0, 3).map(r => /\d/.test(r[1])), [false, false, false]);
  const priced = { ...blank, forecast: { value: 150, hour: 7 } };
  assert.deepEqual(row(g.guide.tourLegend({ today: priced, forecastKind: 'model', probability: null, statementTotal: 0 }))[1],
    ['┅ 예측 · 학습 모델', '학습 모델 예측']);
  assert.deepEqual(row(g.guide.tourLegend({ today: priced, forecastKind: 'fallback', probability: null, statementTotal: 0 }))[1],
    ['▨ 추정 · API 대체값 (모델 예측 아님)', 'API 대체값']);
  // 끊김(61분)이면 실측 ●를 거두므로 범례에도 수요 값을 들지 않는다.
  const broken = viewOf(m, 'today-kpi-age-61m', { advanceMs: 61 * 60_000, failAfter: true });
  assert.equal(broken.legend.demand, null);
});

test('예측 방법·앱 정보의 이번 방문 값: 정확도(+ R² 0.203, 기준선이 아니면 등록 안 됨) · 서버 시장 값 · 시계 차이 120초/12시간', () => {
  const m = load();
  const g = loadGuide();
  const info = m.types.parseModelInfo(resolveVariant('today-base').responses['/api/v1/model/info'].body);
  assert.equal(info.version, 'baseline-24h');
  const base = g.guide.methodAccuracy('baseline');
  // D8: R²에는 한 줄 풀이, API 수치는 이 카드가 아니라 [i] 시트의 "API 표기" 줄에만(오늘 view의 facts['today-accuracy'].api).
  assert.deepEqual([base.documented.main, base.documented.r2, base.api, base.footer.time], ['시간마다 평균 11.97원/kWh 빗나갑니다 (MAPE 11.75%)',
    'R² 0.203 · 가격이 오르내리는 폭의 약 20%를 설명합니다(1이면 전부)', undefined, '평가 기간 2025-07-22~12-18']);
  assert.equal(g.guide.methodAccuracy(null).documented.r2, base.documented.r2);
  const model = g.guide.methodAccuracy('model');
  assert.deepEqual([model.documented, model.unregistered], [null, '이 모델의 검증 수치는 앱에 등록되어 있지 않습니다']);
  const market = m.types.parseMarketStatus(resolveVariant('today-base').responses['/api/v1/market-status'].body);
  assert.equal(g.guide.marketFacts(market), '이번 방문의 서버 값: 하루전시장 마감 11-26(목) 10:00 KST · 입찰일 11-27(금) · 실시간시장 15분 단위');
  assert.equal(g.guide.marketFacts(null), null);
  const skew = ms => g.kst.clockSkew(ms, 0);
  assert.deepEqual([skew(3 * 60000), skew(13 * 3600000), skew(60000), null].map(s => g.guide.aboutClockLines(s)),
    [['기기 시계와 3분 차이 · 서버 시각 기준으로 표시'], ['서버 시각이 기기 시계와 12시간 넘게 달라 기기 시계로 표시합니다'], [], []]);
});
