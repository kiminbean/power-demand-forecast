'use strict';
// 차트(명세 4.1·4.2·5.3·6.2·7.5·7.8절, F5). chartMath.ts의 순수 계산과 StepChart·LadderChart·RangeBar의 VM 렌더를 고정한다:
// 06:04 지금선 x = 6.067/24 × 그림 폭, 계단 경계 i/24, y축 0부터, 0.00 입력 구간, 같은 값 구간 라벨, 360 미만 확률 라벨 솎기,
// 12 px 글자, 응답자 없음(세로 끌기 = 페이지 스크롤), 키보드 선택. 실제 DOM·터치·대비는 브라우저 탐침(scratchpad/mobile/eval/steps/F5)이 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, defaultMocks, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const SVG = { __esModule: true, default: 'Svg', Path: 'Path', Circle: 'Circle', Rect: 'Rect', Line: 'Line', Defs: 'Defs', Pattern: 'Pattern' };
const fixture = name => require(`./fixtures/round0/${name}`);
const FORECAST = fixture('_api_v1_smp-forecast.json');
const DAY_AHEAD = fixture('_api_v1_smp_day-ahead.json');
const SEGMENTS = fixture('_api_v1_bidding_optimized-segments_capacity_mw_50_risk_level_moderate.json');
const KPIS = fixture('_api_v1_dashboard_kpis.json');
const SUPPLY = fixture('_api_v1_power-supply.json');
const NOW = Date.UTC(2026, 10, 25, 21, 4, 0); // 2026-11-26 06:04:00 KST
const FETCHED = Date.UTC(2026, 10, 25, 21, 3, 57);
const NOW_FRACTION = (6 + 4 / 60) / 24; // 6.067/24

function load(windowWidth = 390) {
  const rn = { ...defaultMocks()['react-native'], useWindowDimensions: () => ({ width: windowWidth, height: 844, scale: 1, fontScale: 1 }) };
  const loader = createTsLoader({ mocks: { 'react-native': rn, 'react-native-svg': SVG } });
  const src = rel => loader.load(path.join(SRC_ROOT, rel));
  return {
    math: src('ui/charts/chartMath.ts'), strings: src('content/strings.ts'), tokens: src('ui/tokens.ts'),
    types: src('data/types.ts'), adapters: src('data/adapters.ts'),
    ...src('ui/charts/StepChart.tsx'), ...src('ui/charts/LadderChart.tsx'), ...src('ui/charts/RangeBar.tsx'),
  };
}
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
const byTestId = (root, id) => all(root, n => n.props.testID === id);
const metaTexts = (root, meta) => all(root, n => n.type === 'Text' && n.props.dataSet && n.props.dataSet.pdMeta === meta).map(textOf);
const close = (a, b, eps = 0.011) => Math.abs(a - b) <= eps;

function views(ui) {
  const fc = ui.adapters.adaptForecast(ui.types.parseForecast(FORECAST));
  const da = ui.adapters.adaptDayAhead(ui.types.parseDayAhead(DAY_AHEAD), FETCHED);
  const seg = ui.adapters.adaptSegments(ui.types.parseSegments(SEGMENTS), FETCHED);
  return { fc, confirmed: { tradeHour: da.tradeHour, price: da.price }, seg };
}
function zeroForecast(ui) {
  const raw = structuredClone(FORECAST);
  for (const i of [11, 12, 13]) { raw.q10[i] = 0; raw.q50[i] = 0; }
  return ui.adapters.adaptForecast(ui.types.parseForecast(raw));
}
// 계단 경로의 모든 x가 시 경계(i/24) 위에 있는지 본다.
function pathXs(d) {
  const xs = [];
  for (const m of d.matchAll(/([MHV])([-\d.]+)(?: ([-\d.]+))?/g)) if (m[1] !== 'V') xs.push(Number(m[2]));
  return xs;
}

test('niceScale: y축은 0부터, 맨 위 눈금은 50 단위 올림, 다섯 칸 안', () => {
  const { math } = load();
  assert.deepEqual(math.niceScale([...FORECAST.q50, ...FORECAST.q90]).ticks, [0, 100, 200, 300, 400]);
  const ladder = math.niceScale(SEGMENTS.hourly_bids[18].segments.map(s => s.price_krw_mwh));
  assert.deepEqual([ladder.bottom, ladder.top, ladder.ticks], [0, 250, [0, 50, 100, 150, 200, 250]]);
  assert.deepEqual(math.niceScale([]).ticks, [0, 50], '값이 없어도 0과 한 칸');
  assert.deepEqual(math.niceScale([null, NaN, Infinity, 12]).ticks, [0, 50]);
  const neg = math.niceScale([-30, 120]);
  assert.equal(neg.bottom, -50, '음수가 있으면 0 아래 눈금까지');
  assert.ok(neg.ticks.includes(0));
  for (const max of [1, 49.9, 50, 50.1, 244.1, 341.58, 999, 1191]) {
    const s = math.niceScale([max]);
    assert.equal(s.bottom, 0);
    assert.equal(s.top % 50, 0, `${max} → ${s.top}`);
    assert.ok(s.top >= max && s.ticks.length - 1 <= 5, `${max}: ${s.ticks}`);
  }
});

test('지금선: 06:04 KST → x = 6.067/24 × 그림 폭(296·358·398·608), 예측일이 오늘이 아니면 없음', () => {
  const { math } = load();
  const f = math.nowFraction(NOW, '2026-11-26');
  assert.equal(f, NOW_FRACTION);
  assert.equal(math.nowFraction(NOW, '2026-11-25'), null);
  assert.equal(math.nowFraction(null, '2026-11-26'), null);
  assert.equal(math.nowFraction(NOW, null), null);
  const scale = math.niceScale(FORECAST.q90);
  for (const [w, height] of [[296, 200], [358, 220], [398, 230], [608, 260]]) {
    const { plot, height: hh } = math.stepLayout(w, scale, false);
    assert.equal(hh, height, `${w} 높이`);
    const x = plot.left + f * plot.width;
    assert.ok(close((x - plot.left) / plot.width * 24, 6.0667, 1e-3), `${w}: ${x}`);
    assert.ok(close(plot.left + plot.width + 12, w, 1e-9), '오른쪽 여백 12');
  }
});

test('계단 경계: 모든 꺾임이 i/24 위치, 시간마다 한 칸, 0.00 입력·값 없음에서 선이 끊긴다', () => {
  const { math } = load();
  const scale = math.niceScale(FORECAST.q90);
  const { plot } = math.stepLayout(358, scale, false);
  const kinds = math.hourKinds(FORECAST.q50);
  const d = math.stepPath(FORECAST.q50, kinds, scale, plot);
  const bounds = Array.from({ length: 25 }, (_, k) => plot.left + (k / 24) * plot.width);
  pathXs(d).forEach(x => assert.ok(bounds.some(b => close(x, b)), `x ${x} not on an hour boundary`));
  assert.equal((d.match(/M/g) || []).length, 1, '24시간 이어진 선');
  assert.equal((d.match(/H/g) || []).length, 24, '시간마다 가로 한 칸');
  const firstY = Number(/^M[-\d.]+ ([-\d.]+)/.exec(d)[1]);
  assert.ok(close(firstY, math.yOf(FORECAST.q50[0], scale, plot)));
  const withGap = FORECAST.q50.slice();
  withGap[5] = null;
  assert.equal((math.stepPath(withGap, math.hourKinds(withGap), scale, plot).match(/M/g) || []).length, 2, '값 없음에서 끊김');
  assert.equal(math.hourAtX(plot.left, plot), 1);
  assert.equal(math.hourAtX(plot.left + plot.width - 0.1, plot), 24);
  assert.equal(math.hourAtX(plot.left + (6.5 / 24) * plot.width, plot), 7);
  assert.equal(math.hourAtX(-50, plot), 1);
  assert.equal(math.hourAtX(10000, plot), 24);
});

test('0.00 입력(12–14h): 빗금 구간 하나, 선·띠 끊김, 최고·최저·평균에서 빼고 "0.00 입력 3시간 제외"', () => {
  const { math, strings } = load();
  const q50 = FORECAST.q50.slice();
  const q10 = FORECAST.q10.slice();
  for (const i of [11, 12, 13]) { q50[i] = 0; q10[i] = 0; }
  const kinds = math.hourKinds(q50);
  assert.deepEqual(math.zeroRuns(kinds), [{ from: 12, to: 14 }]);
  const scale = math.niceScale(q50);
  const { plot } = math.stepLayout(358, scale, true);
  const line = math.stepPath(q50, kinds, scale, plot);
  assert.equal((line.match(/M/g) || []).length, 2);
  const inside = pathXs(line).filter(x => x > plot.left + (11 / 24) * plot.width + 0.01 && x < plot.left + (14 / 24) * plot.width - 0.01);
  assert.deepEqual(inside, [], '0.00 시간 안에는 선이 없다');
  assert.equal((math.bandPath(kinds, q10, FORECAST.q90, scale, plot).match(/Z/g) || []).length, 2, '띠도 끊긴다');
  const stats = math.forecastStats(q50);
  assert.equal(stats.zeroHours, 3);
  assert.equal(stats.counted, 21);
  assert.equal(strings.forecastSummaryText(stats, 'baseline'),
    '기준선 예측으로는 08h · 07–08시가 가장 비싸고(212.3원), 03–05h · 02–05시가 가장 쌉니다(102.3원). 하루 평균 168.1원 (0.00 입력 3시간 제외).');
  assert.deepEqual(math.widestBand(q50, q10, FORECAST.q90).runs, [{ from: 15, to: 15 }], '0.00 시간(13h 326.4원)의 띠 폭은 보지 않는다');
  assert.deepEqual(math.periodStats(q50).map(p => p.zeroHours), [0, 3, 0]);
  assert.equal(math.forecastStats([0, 0, null]), null, '그릴 값이 없으면 요약 없음');
  assert.equal(strings.forecastSummaryText(null, 'baseline'), '그릴 수 있는 예측 값이 없습니다.');
});

test('요약 문장(4.1절): 같은 값이 이어진 시간은 구간 라벨 하나, 떨어진 같은 값은 쉼표로', () => {
  const { math, strings } = load();
  const stats = math.forecastStats(FORECAST.q50);
  assert.deepEqual(stats.min.runs, [{ from: 3, to: 5 }]);
  assert.equal(strings.forecastSummaryText(stats, 'baseline'),
    '기준선 예측으로는 08h · 07–08시가 가장 비싸고(212.3원), 03–05h · 02–05시가 가장 쌉니다(102.3원). 하루 평균 169.2원.');
  assert.equal(strings.uncertaintyText(math.widestBand(FORECAST.q50, FORECAST.q10, FORECAST.q90)),
    '가장 불확실한 시간: 13h · 12–13시 (80% 구간 폭 326.4원)');
  // 화면 글자가 같으면(100.04·100.01 → "100.0") 같은 값으로 묶는다. 떨어진 구간은 따로 적는다.
  const q = Array.from({ length: 24 }, () => 150);
  q[2] = 100.04; q[3] = 100.01; q[14] = 99.96; q[20] = 210;
  const s2 = math.forecastStats(q);
  assert.deepEqual(s2.min.runs, [{ from: 3, to: 4 }, { from: 15, to: 15 }]);
  assert.equal(strings.hourGroupText(s2.min.runs), '03–04h · 02–04시, 15h · 14–15시');
  assert.ok(strings.forecastSummaryText(s2, 'fallback').startsWith('API 대체값으로는 '));
  assert.ok(strings.forecastSummaryText(s2, 'model').startsWith('학습 모델 예측으로는 '));
});

test('시간대 요약 숫자(4.1절 표 보기): 130.6/102.3/212.3 · 177.1/176.5/178.7 · 203.3/201.3/205.9', () => {
  const { price } = createTsLoader().load(path.join(SRC_ROOT, 'lib/format.ts'));
  const { periodStats } = load().math;
  assert.deepEqual(periodStats(FORECAST.q50).map(p => [p.from, p.to, p.label]), [[1, 8, '심야'], [9, 17, '주간'], [18, 24, '저녁']]);
  assert.deepEqual(periodStats(FORECAST.q50).map(p => [p.mean, p.min, p.max].map(price).join('/')),
    ['130.6/102.3/212.3', '177.1/176.5/178.7', '203.3/201.3/205.9']);
  assert.deepEqual(periodStats([]).map(p => [p.mean, p.min, p.max]), [[null, null, null], [null, null, null], [null, null, null]]);
});

test('계단 차트 계산(4.2절): 누적 물량 x, 0부터 높이, 1·4·7·10구간 솎기, 요약·판독 문구', () => {
  const { math, strings } = load();
  const items = SEGMENTS.hourly_bids[18].segments.map(s => ({ price: s.price_krw_mwh, probability: s.clearing_probability * 100, quantity: s.quantity_mw }));
  const scale = math.niceScale(items.map(i => i.price));
  const { plot, height } = math.ladderLayout(358, scale);
  assert.equal(height, 200);
  assert.deepEqual([math.ladderChartHeight(296), math.ladderChartHeight(398), math.ladderChartHeight(608)], [180, 200, 240]);
  const bars = math.ladderBars(items, scale, plot);
  assert.equal(bars.length, 10);
  assert.ok(close(bars[0].x0, plot.left) && close(bars[9].x1, plot.left + plot.width));
  bars.forEach(b => assert.ok(close(b.x1 - b.x0, plot.width / 10), '5 MW씩 같은 폭'));
  assert.ok(close(bars[9].y, math.yOf(244.1, scale, plot)));
  assert.ok(close(math.yOf(0, scale, plot), plot.top + plot.height), '막대 바닥 = 0');
  assert.deepEqual(math.ladderLabelIndices(10, true), [0, 3, 6, 9], '360 미만: 1·4·7·10구간');
  assert.deepEqual(math.ladderLabelIndices(10, false), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(math.ladderLabelIndices(11, true), [0, 3, 6, 9], '붙은 마지막 구간은 더하지 않는다');
  assert.deepEqual(math.ladderLabelIndices(12, true), [0, 3, 6, 9, 11]);
  assert.equal(strings.ladderSummaryText(math.ladderStats(items)), '10구간, 171.7–244.1원, 낙찰 가능성 95%에서 5%(추정)');
  const h0 = SEGMENTS.hourly_bids[0].segments[3];
  assert.equal(strings.ladderReadingText(3, h0.quantity_mw, h0.price_krw_mwh, h0.clearing_probability * 100), '4구간 · 5 MW · 134.5원 · 낙찰 가능성 62% (추정)');
  assert.deepEqual(math.mwTicks(50), [0, 10, 20, 30, 40, 50]);
  assert.equal(math.barAtX(plot.left + plot.width * 0.35, bars), 3);
  assert.equal(math.barAtX(0, bars), 0);
  assert.equal(math.barAtX(9999, bars), 9);
  assert.equal(math.ladderStats([]), null);
});

test('범위 막대 계산(4.1절): 0…공급능력, 수요 끝, 관심 경계 = 공급능력 ÷ 1.15, 식·경계 문장', () => {
  const { math, strings } = load();
  const g = math.rangeBarGeometry(1191, 664, 1191 / 1.15, 358);
  assert.ok(close(g.demandX, (664 / 1191) * 358) && close(g.boundaryX, 358 / 1.15));
  assert.equal(math.rangeBarGeometry(0, 664, 0, 358), null);
  assert.equal(math.rangeBarGeometry(1191, NaN, 1000, 358), null);
  assert.equal(math.rangeBarGeometry(500, 664, 434.8, 358).demandX, 358, '수요가 공급능력을 넘어도 막대 밖으로 나가지 않는다');
  assert.equal(strings.supplyFormulaText(1191, 664, 527), '공급능력 1,191 MW − 수요 664 MW = 예비력 527 MW');
  assert.equal(strings.watchBoundaryText(1191 / 1.15), '관심 경계 1,036 MW (예비율 15%)');
});

test('키·라벨 자리: ←/→는 양 끝에서 멈춤, Home·End, 라벨 상자는 차트 밖으로 나가지 않는다', () => {
  const { math } = load();
  assert.deepEqual(['ArrowLeft', 'ArrowRight', 'Home', 'End', 'a'].map(k => math.keyStep(k, 7, 1, 24)), [6, 8, 1, 24, null]);
  assert.deepEqual([math.keyStep('ArrowLeft', 1, 1, 24), math.keyStep('ArrowRight', 24, 1, 24)], [1, 24]);
  for (const x of [-20, 0, 150, 357, 400]) {
    const left = math.labelLeft(x, 40, 358);
    assert.ok(left >= 0 && left + 40 <= 358, `${x} → ${left}`);
  }
  assert.ok(math.pillLeft(40, 80, 51, 358) >= 12 + 51 + 4, 'pill은 단위 글자와 겹치지 않는다');
  assert.ok(math.pillLeft(357, 80, 51, 358) + 80 <= 358 - 12);
  assert.ok(math.textWidth('지금 06:04') > math.textWidth('06:04') && math.textWidth('MW') > math.textWidth('00'));
});

function stepTree(ui, props) {
  return tree(h(ui.StepChart, { forecast: views(ui).fc, label: '기준선 예측 SMP 24시간', now: NOW, width: 358, ...props })).children[0];
}

test('StepChart: role=img 요약 이름, 안쪽 aria-hidden, 지금선·pill·음영, 눈금 글자, 12 px, data-pd-meta', () => {
  const ui = load();
  const { confirmed } = views(ui);
  const root = stepTree(ui, { confirmed, selectedHour: 7, onSelectHour: () => {} });
  assert.equal(root.props.role, 'img');
  assert.equal(root.props['aria-label'],
    '기준선 예측 SMP 24시간: 기준선 예측으로는 08h · 07–08시가 가장 비싸고(212.3원), 03–05h · 02–05시가 가장 쌉니다(102.3원). '
    + '하루 평균 169.2원. 가장 불확실한 시간: 13h · 12–13시 (80% 구간 폭 326.4원). 점은 KPX 확정 가격 07h · 06–07시 115.3원.');
  const layer = root.children[0];
  assert.equal(layer.props['aria-hidden'], true);
  const [px, py, pw, ph] = root.props.dataSet.pdPlot.split(' ').map(Number);
  assert.equal(root.props.dataSet.pdScale, '0 400', 'y축 0부터');
  const [nowLine] = byTestId(root, 'step-now-line');
  assert.ok(close((nowLine.props.x1 - px) / pw, NOW_FRACTION, 1e-4), `지금선 ${nowLine.props.x1}`);
  assert.ok(close(nowLine.props.y1, py) && close(nowLine.props.y2, py + ph));
  assert.ok(close(byTestId(root, 'step-past')[0].props.width, nowLine.props.x1 - px), '지난 시간 음영 = 지금선 왼쪽');
  assert.deepEqual(metaTexts(root, 'now'), ['지금 06:04']);
  assert.deepEqual(metaTexts(root, 'axis'), ['0', '100', '200', '300', '400', '00', '06', '12', '18', '24시']);
  assert.deepEqual(metaTexts(root, 'unit'), ['원/kWh']);
  const texts = all(root, n => n.type === 'Text');
  texts.forEach(t => assert.ok(flat(t.props.style).fontSize >= 12, `글자 ${textOf(t)} ${flat(t.props.style).fontSize}`));
  assert.equal(new Set(texts.map(t => flat(t.props.style).fontSize)).size, 1, '차트 글자는 12 하나');
  const line = byTestId(root, 'step-line')[0];
  assert.equal(line.props.strokeDasharray, '6 4', '예측은 점선');
  assert.equal(line.props.stroke, ui.tokens.SERIES.smp);
  assert.equal(byTestId(root, 'step-band')[0].props.fill, ui.tokens.SERIES.band);
  const [dot] = byTestId(root, 'step-confirmed');
  assert.ok(close(dot.props.cx, px + (6.5 / 24) * pw), '확정 가격 점은 지금 거래시간 가운데');
  assert.equal(dot.props.fill, ui.tokens.THEMES.light.color.text1);
  // 라벨 상자는 모두 차트 상자 안이다(가로 넘침 0).
  all(root, n => n.type === 'View' && flat(n.props.style).position === 'absolute' && typeof flat(n.props.style).width === 'number')
    .forEach(v => { const s = flat(v.props.style); assert.ok(s.left >= 0 && s.left + s.width <= 358 + 0.01, JSON.stringify(s)); });
});

test('StepChart 상호작용: 초점·키·누름은 onSelectHour가 있을 때만, 응답자·터치 처리기 없음(세로 끌기 = 페이지 스크롤)', () => {
  const ui = load();
  const picked = [];
  const root = stepTree(ui, { selectedHour: 7, onSelectHour: v => picked.push(v) });
  assert.equal(root.props.tabIndex, 0);
  const key = k => ({ key: k, preventDefault: () => {} });
  ['ArrowRight', 'ArrowLeft', 'Home', 'End', 'Tab'].forEach(k => root.props.onKeyDown(key(k)));
  const [px, , pw] = root.props.dataSet.pdPlot.split(' ').map(Number);
  root.props.onClick({ clientX: 100 + px + (12.5 / 24) * pw, currentTarget: { getBoundingClientRect: () => ({ left: 100 }) } });
  assert.deepEqual(picked, [8, 6, 1, 24, 13]);
  const atStart = [];
  stepTree(ui, { selectedHour: 1, onSelectHour: v => atStart.push(v) }).props.onKeyDown(key('ArrowLeft'));
  assert.deepEqual(atStart, [1], '양 끝에서 멈춘다');
  assert.equal(byTestId(root, 'step-selected-mark').length, 1);
  assert.equal(byTestId(root, 'step-selected').length, 1);
  const banned = /^on(StartShouldSet|MoveShouldSet|Responder|Touch|Pointer|Gesture)|PanResponder/;
  const handlers = all(root, () => true).flatMap(n => Object.keys(n.props).filter(k => banned.test(k)));
  assert.deepEqual(handlers, []);
  const still = stepTree(ui, {});
  assert.equal(still.props.tabIndex, undefined, '보기 전용(둘러보기)은 초점을 받지 않는다');
  assert.equal(still.props.onClick, undefined);
  assert.equal(byTestId(still, 'step-selected-mark').length, 0);
});

test('StepChart 상태: 예측일 어제 → 지금선 없음, 대체값 → 띠 없음, 0.00 입력 → 빗금 열과 각주 한 번', () => {
  const ui = load();
  const stale = ui.adapters.adaptForecast(ui.types.parseForecast({ ...FORECAST, forecast_date: '2026-11-25' }));
  const s = stepTree(ui, { forecast: stale });
  assert.deepEqual([byTestId(s, 'step-now-line').length, byTestId(s, 'step-past').length, metaTexts(s, 'now').length], [0, 0, 0]);
  const fallback = ui.adapters.adaptForecast(ui.types.parseForecast({ ...FORECAST, model_used: '패턴 기반 대체값 (모델 예측 실패)' }));
  const f = stepTree(ui, { forecast: fallback });
  assert.equal(byTestId(f, 'step-band').length, 0);
  assert.match(f.props['aria-label'], /: API 대체값으로는 /);
  const z = stepTree(ui, { forecast: zeroForecast(ui) });
  const [px, , pw] = z.props.dataSet.pdPlot.split(' ').map(Number);
  const [col] = byTestId(z, 'step-zero');
  assert.ok(close(col.props.x, px + (11 / 24) * pw) && close(col.props.width, (3 / 24) * pw), '12–14h 한 열');
  assert.deepEqual(metaTexts(z, 'note'), [ui.strings.ZERO_INPUT_LABEL]);
  assert.equal(ui.strings.ZERO_INPUT_LABEL, '기준선 입력 0.00 · 예측 아님');
  assert.match(z.props['aria-label'], /하루 평균 168\.1원 \(0\.00 입력 3시간 제외\)\./);
  assert.equal(flat(z.props.style).height, 220, '각주 줄이 생겨도 차트 높이는 그대로');
  // 확정 가격 시간이 0.00 입력이면 점을 그리지 않는다.
  const { price } = views(ui).confirmed;
  assert.equal(byTestId(stepTree(ui, { forecast: zeroForecast(ui), confirmed: { tradeHour: 13, price } }), 'step-confirmed').length, 0);
});

test('StepChart 폭별 크기: 296×200 · 358×220 · 398×230 · 608×260, 모든 라벨이 상자 안', () => {
  const ui = load();
  for (const [w, height] of [[296, 200], [358, 220], [398, 230], [608, 260]]) {
    const root = stepTree(ui, { width: w, forecast: zeroForecast(ui) });
    assert.equal(flat(root.props.style).height, height);
    const svg = all(root, n => n.type === 'Svg')[0];
    assert.deepEqual([svg.props.width, svg.props.height], [w, height], 'SVG는 1:1(글자 축소 없음)');
    all(root, n => n.type === 'View' && flat(n.props.style).position === 'absolute' && typeof flat(n.props.style).width === 'number')
      .forEach(v => { const s = flat(v.props.style); assert.ok(s.left >= 0 && s.left + s.width <= w + 0.01, `${w}: ${JSON.stringify(s)}`); });
  }
});

test('재기 전 첫 그림 폭: 넓힌 차트 = 카드 폭(296·358·398·608), 범위 막대 = 카드 안쪽 폭', () => {
  for (const [windowWidth, card, inner] of [[320, 296, 270], [390, 358, 324], [430, 398, 364], [768, 608, 574], [1280, 608, 574]]) {
    const ui = load(windowWidth);
    const step = tree(h(ui.StepChart, { forecast: views(ui).fc, label: '기준선 예측 SMP 24시간', now: NOW })).children[0];
    assert.equal(all(step, n => n.type === 'Svg')[0].props.width, card, `${windowWidth} step`);
    const ladder = tree(h(ui.LadderChart, { hour: views(ui).seg.hours[18], label: '입찰 계단' })).children[0];
    assert.equal(all(ladder, n => n.type === 'Svg')[0].props.width, card, `${windowWidth} ladder`);
    const kpis = ui.types.parseKpis(KPIS);
    const sup = ui.adapters.adaptSupply(kpis, ui.types.parsePowerSupply(SUPPLY), FETCHED);
    const kpi = ui.adapters.adaptKpis(kpis, FETCHED);
    const bar = tree(h(ui.RangeBar, { supply: sup.supply, demand: kpi.demand, reserve: sup.reserve, boundary: sup.watchBoundary })).children[0];
    assert.equal(all(bar, n => n.type === 'Svg' && n.props.height > 12)[0].props.width, inner, `${windowWidth} range`);
  }
});

test('LadderChart: 요약 이름, 확률 글자(360 미만 1·4·7·10구간만), 단색 막대(불투명도 없음), 키·누름', () => {
  for (const [windowWidth, w, expected] of [[390, 358, ['95%', '88%', '77%', '65%', '53%', '42%', '31%', '20%', '9%', '5%']], [320, 296, ['95%', '65%', '31%', '5%']]]) {
    const ui = load(windowWidth);
    const picked = [];
    const root = tree(h(ui.LadderChart, { hour: views(ui).seg.hours[18], label: '19h · 18–19시 입찰 계단', width: w, onSelect: v => picked.push(v) })).children[0];
    assert.equal(root.props.role, 'img');
    assert.equal(root.props['aria-label'], '19h · 18–19시 입찰 계단: 10구간, 171.7–244.1원, 낙찰 가능성 95%에서 5%(추정)');
    assert.deepEqual(metaTexts(root, 'chart'), expected);
    assert.deepEqual(metaTexts(root, 'axis'), ['0', '50', '100', '150', '200', '250', '0', '10', '20', '30', '40', '50 MW']);
    const bars = byTestId(root, 'ladder-bar');
    assert.equal(bars.length, 10);
    bars.forEach(b => {
      assert.equal(b.props.fill, ui.tokens.SERIES.smp);
      assert.equal(b.props.opacity ?? b.props.fillOpacity, undefined, '불투명도로 확률을 나타내지 않는다');
    });
    all(root, n => n.type === 'Text').forEach(t => assert.ok(flat(t.props.style).fontSize >= 12));
    const key = k => ({ key: k, preventDefault: () => {} });
    root.props.onKeyDown(key('ArrowRight'));
    root.props.onKeyDown(key('End'));
    const [px, , pw] = root.props.dataSet.pdPlot.split(' ').map(Number);
    root.props.onClick({ clientX: px + pw * 0.35, currentTarget: { getBoundingClientRect: () => ({ left: 0 }) } });
    assert.deepEqual(picked, [0, 9, 3], '고른 것이 없을 때 → 는 1구간');
  }
});

test('RangeBar: role=img 식+경계 이름, 0…공급능력, 수요 채움·예비력 빗금, 관심 경계 눈금, 수요는 data-pd-value·계산값은 derived', () => {
  const ui = load();
  const kpi = ui.adapters.adaptKpis(ui.types.parseKpis(KPIS), FETCHED);
  const sup = ui.adapters.adaptSupply(ui.types.parseKpis(KPIS), ui.types.parsePowerSupply(SUPPLY), FETCHED);
  assert.equal(sup.status, 'paired');
  const root = tree(h(ui.RangeBar, { supply: sup.supply, demand: kpi.demand, reserve: sup.reserve, boundary: sup.watchBoundary, width: 358 })).children[0];
  assert.equal(root.props.role, 'img');
  assert.equal(root.props['aria-label'], '공급능력 1,191 메가와트 − 수요 664 메가와트 = 예비력 527 메가와트. 관심 경계 1,036 메가와트 (예비율 15%)');
  const [demand] = byTestId(root, 'range-demand');
  assert.ok(close(demand.props.width, (664 / 1191) * 358));
  assert.equal(demand.props.fill, ui.tokens.SERIES.demand);
  const [reserve] = byTestId(root, 'range-reserve');
  assert.equal(reserve.props.stroke, ui.tokens.SERIES.reserve);
  assert.match(reserve.props.fill, /^url\(#pd-reserve-/);
  assert.ok(close(byTestId(root, 'range-boundary')[0].props.x1, 358 / 1.15));
  // F7: 예비력(공급능력 − 수요)과 관심 경계(공급능력 ÷ 1.15)는 API 원값이 아니라서 I3 비교 대상(data-pd-value)에서 뺀다.
  const values = all(root, n => n.props.dataSet && n.props.dataSet.pdValue);
  assert.deepEqual(values.map(textOf), ['664 MW']);
  assert.deepEqual(values.map(v => [v.props.dataSet.pdValue, v.props.dataSet.pdRaw]), [['measured', '664']]);
  const derived = all(root, n => n.props.dataSet && n.props.dataSet.pdMeta === 'derived');
  assert.deepEqual(derived.map(textOf), ['527 MW', '1,036 MW']);
  assert.ok(derived.every(d => d.props.dataSet.pdRaw === undefined && d.props.dataSet.pdValue === undefined));
  assert.match(textOf(root), /관심 경계 1,036 MW \(예비율 15%\)/);
  all(root, n => n.type === 'Text').forEach(t => assert.ok(flat(t.props.style).fontSize >= 12));
});
