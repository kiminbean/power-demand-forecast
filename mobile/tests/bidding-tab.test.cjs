'use strict';
// 입찰 화면(명세 4.2·6.3·7.5·7.8절, F8). round 0 스냅숏(2026-11-26 06:03:57 KST)과 상태 변형(fixtures/variants/variants.json의 B6)을
// features/bidding/model.ts에 넣어 마감 상태 기계·카운트다운·계산 결과 글자를 고정하고, 계산기 요청 규칙(한 번에 하나 · 503은 1.5초 뒤
// 한 번만 · 폴링 없음)을 가짜 시계 저장소로, 카드 렌더를 노드 수준의 I2·의미로 본다. 같은 순간은 TZ 세 개에서 같은 글자여야 한다.
// 브라우저 확인(B1–B6, 6폭 × 2테마, 요청 목록, throttle)은 scratchpad/mobile/eval/steps/F8에 있다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { createTsLoader, defaultMocks, SRC_ROOT } = require('./helpers/ts-loader.cjs');
const { resolveVariant } = require('./helpers/fixtures.cjs');

const SVG = { __esModule: true, default: 'Svg', Svg: 'Svg', Path: 'Path', Circle: 'Circle', Rect: 'Rect', Line: 'Line', Defs: 'Defs', Pattern: 'Pattern' };
const KST = (y, mo, d, h, mi, s = 0) => Date.UTC(y, mo - 1, d, h, mi, s) - 9 * 3600e3;

/** forceTable: 차트/표 세그먼트의 첫 값을 표로. open: 가격 원리 카드를 펼친 채로(React 목의 useState는 첫 값만 준다). */
function load(width = 390, { forceTable = false, open = false } = {}) {
  const mocks = defaultMocks();
  const rn = { ...mocks['react-native'], useWindowDimensions: () => ({ width, height: 844, scale: 1, fontScale: 1 }) };
  const pick = init => (forceTable && init === 'chart' ? 'table' : open && init === false ? true : typeof init === 'function' ? init() : init);
  const react = forceTable || open ? { ...mocks.react, useState: init => [pick(init), () => {}] } : mocks.react;
  react.default = react;
  const loader = createTsLoader({ mocks: { react, 'react-native': rn, 'react-native-svg': SVG } });
  const at = rel => loader.load(path.join(SRC_ROOT, rel));
  return {
    model: at('features/bidding/model.ts'), calc: at('features/bidding/calculator.ts'), types: at('data/types.ts'), client: at('data/client.ts'),
    kst: at('lib/kst.ts'), format: at('lib/format.ts'), strings: at('content/strings.ts'), resource: at('lib/resource.ts'),
    provenance: at('lib/provenance.ts'), examples: at('data/examples.ts'), tokens: at('ui/tokens.ts'),
    ...at('features/bidding/DeadlineCard.tsx'), ...at('features/bidding/SampleBidCard.tsx'), ...at('features/bidding/MeritOrderCard.tsx'),
    ...at('ui/Card.tsx'),
  };
}

const ROUTES = {
  market: ['/api/v1/market-status', 'parseMarketStatus'], calc: ['/api/v1/bidding/optimized-segments', 'parseSegments'],
  forecast: ['/api/v1/smp-forecast', 'parseForecast'],
};
const empty = { data: null, fetchedAt: null, status: 'idle', error: null };

/**
 * 변형 → BiddingInput. 화면(BiddingTab)과 같게: 파싱 → 시장 상태 서버 시각으로 시계 차이 → 앱 시각. status_sequence는 저장소 규칙의
 * 결과로 푼다(503 한 번은 1.5초 뒤 다시 보내 성공, 두 번이면 오류). withForecast: 기준선 예측을 받은 뒤(round 1 D5 뒤로는 입찰 화면도
 * 같은 저장소로 받는다. 거짓이면 받는 중), forecastFailed: 변형의 예측 응답이 실패했다.
 */
function inputFor(m, id, { withForecast = false, advanceMs = 0, failAfter = false, ...rest } = {}) {
  const v = resolveVariant(id);
  const device = v.clockMs;
  const res = {};
  for (const [key, [route, parser]] of Object.entries(ROUTES)) {
    const r = v.responses[route];
    const seq = r.statusSequence || [];
    const status = seq.length >= 2 ? seq[1] : seq.length === 1 ? 200 : r.status;
    if (r.delayMs) res[key] = { data: null, fetchedAt: null, status: 'loading', error: null };
    else if (r.abort) res[key] = { data: null, fetchedAt: null, status: 'error', error: new m.client.ApiNetworkError(route) };
    else if (status >= 400) res[key] = { data: null, fetchedAt: null, status: 'error', error: new m.client.ApiHttpError(route, status) };
    else res[key] = { data: m.types[parser](r.body), fetchedAt: device, status: 'ready', error: null };
  }
  const server = res.market.data ? m.kst.parseApiTime(res.market.data.current_time) : null;
  const skew = server === null ? null : m.kst.clockSkew(server, device);
  const offset = skew && skew.plausible ? skew.skewMs : 0;
  const app = r => ({ ...r, fetchedAt: r.fetchedAt === null ? null : r.fetchedAt + offset, status: failAfter && r.data !== null ? 'error' : r.status });
  return {
    now: device + advanceMs + offset, market: app(res.market), calc: { ...app(res.calc), busy: false },
    forecast: withForecast ? res.forecast.data : null, forecastFailed: res.forecast.status === 'error', calcForecastDate: null, risk: null,
    selectedHour: null, recheck: 'idle', staleRetried: false,
    ...rest,
  };
}
const viewOf = (m, id, opts) => m.model.biddingView(inputFor(m, id, opts));
const chipText = (m, c) => (c.chip ? m.provenance.chipText(c.chip) : c.text);

/** 서버(api/v6_routes.py get_market_status)가 그 순간에 보낼 시장 상태: 10시 전 open, 거래일은 내일, 남은 시간은 다음 10:00까지. */
function marketAt(m, t) {
  const p = m.kst.kstParts(t);
  const pad = n => String(n).padStart(2, '0');
  const today = m.kst.kstDateKey(t);
  const next = m.kst.kstAt(today, '10:00', p.hour >= 10 ? 1 : 0);
  return m.types.parseMarketStatus({
    current_time: `${today}T${pad(p.hour)}:${pad(p.minute)}:${pad(p.second)}`,
    dam: { status: p.hour < 10 ? 'open' : 'closed', deadline: '10:00', trading_date: m.kst.addDays(today, 1), hours_remaining: Math.round((next - t) / 360e3) / 10 },
    rtm: { status: 'active', next_interval: '15분', interval_minutes: 15 },
  });
}
const at = (m, t, data, rest = {}) => m.model.biddingView({
  ...inputFor(m, 'bidding-base'), now: t, market: { data, fetchedAt: t, status: 'ready', error: null }, ...rest,
});

// ---------------------------------------------------------------- 1. 하루전 마감 상태 기계(4.2절 상태표)
test('06:03:57 접수 중: 3시간 56분 · 마감 11-26(목) 10:00 KST · 거래일 11-27(금) · 다음 구간 06:15 · 서버 시각 출처', () => {
  const m = load();
  const v = viewOf(m, 'bidding-base');
  const d = v.deadline;
  assert.deepEqual([d.phase, d.countdown, d.when, d.pill], ['open', '3시간 56분 남음', '마감 11-26(목) 10:00 KST · 거래일 11-27(금)', { tone: 'ok', text: '접수 중' }]);
  // round 1 멘토: 출처 줄이 이미 "API 서버 시장 상태"라 "(API 기준)"을 뺐다.
  assert.equal(d.rtm, '실시간시장은 15분 단위입니다 · 다음 구간 06:15 KST');
  assert.equal(d.status, null, '접수 중에는 제목 옆 상태 없음(pill은 카운트다운 아래)');
  assert.deepEqual(d.chips.map(c => chipText(m, c)), ['시장 규칙 · 서버 시각']);
  assert.equal(`${d.footer.source} · ${d.footer.time}`, '출처 API 서버 시장 상태 · 서버 시각 06:03 KST');
  assert.equal(v.answer, '발전소는 내일 만들 전기를 오늘 오전 10시까지 미리 팝니다. 마감까지 3시간 56분 남았습니다.');
  assert.equal(v.mismatch, false);
  // API의 next_interval("15분")은 쓰지 않는다: 지금 뒤 첫 15분 배수를 KST로 계산한다.
  assert.ok(!JSON.stringify(d).includes('"15분"'));
});

test('09:20 마감 임박 · 09:59:30 1분 미만 · 10:00:30·10:01 마감: 다음 마감까지 큰 숫자 · 제목 옆 «오늘 마감됨» · 다음 마감의 거래일(D13)', () => {
  const m = load();
  const soon = viewOf(m, 'bidding-imminent-0920');
  assert.deepEqual([soon.deadline.phase, soon.deadline.countdown, soon.deadline.pill], ['imminent', '40분 남음', { tone: 'watch', text: '마감 임박' }]);
  assert.equal(soon.answer, '발전소는 내일 만들 전기를 오늘 오전 10시까지 미리 팝니다. 마감까지 40분 남았습니다(마감 임박).');
  const last = at(m, KST(2026, 11, 26, 9, 59, 30), marketAt(m, KST(2026, 11, 26, 9, 59, 30)));
  assert.deepEqual([last.deadline.phase, last.deadline.countdown], ['imminent', '1분 미만 남음'], '"0분"이라 쓰지 않는다');
  assert.match(last.answer, /마감까지 1분 미만 남았습니다\(마감 임박\)\.$/);
  for (const view of [at(m, KST(2026, 11, 26, 10, 0, 30), marketAt(m, KST(2026, 11, 26, 10, 0, 30))), viewOf(m, 'bidding-closed-1001')]) {
    // 하루 14시간 동안 큰 숫자가 없던 카드(round 1 parity): 서버 시각으로 맞춘 지금에서 다음 마감(거래일 11-27 10:00)까지.
    assert.deepEqual([view.deadline.phase, view.deadline.countdownLead, view.deadline.countdown, view.deadline.when, view.deadline.pill, view.deadline.status],
      ['closed', '다음 마감까지', '23시간 59분', '다음 마감 11-27(금) 10:00 KST · 거래일 11-28(토)', null, { tone: 'neutral', text: '오늘 마감됨' }]);
    assert.equal(view.answer, '오늘 입찰은 마감됐습니다. 다음 마감은 내일 오전 10시입니다.');
  }
  const night = at(m, KST(2026, 11, 26, 23, 59, 30), marketAt(m, KST(2026, 11, 26, 23, 59, 30)));
  assert.deepEqual([night.deadline.phase, night.deadline.countdown], ['closed', '10시간'], '분이 0이면 "10시간"');
});

test('확인 중(첫 응답 전) · 오류(502) · 지연(11분, 카운트다운은 계속): 확인 중에는 "마감"이라 쓰지 않는다', () => {
  const m = load();
  const checking = viewOf(m, 'bidding-checking');
  assert.deepEqual([checking.deadline.phase, checking.deadline.pill.text, checking.deadline.countdown, checking.answer],
    ['checking', '확인 중', null, '입찰 마감 시각을 확인하는 중입니다.']);
  assert.ok(!/마감됐|마감 임박|«마감»/.test(JSON.stringify([checking.deadline, checking.answer])));
  const error = viewOf(m, 'bidding-market-error');
  assert.deepEqual([error.deadline.phase, error.deadline.note, error.deadline.countdown, error.answer],
    ['error', '시장 상태를 확인하지 못했습니다', null, '시장 상태를 확인하지 못해 남은 시간을 표시하지 않습니다.']);
  assert.equal(error.calc.state, 'ready', '시장 상태가 실패해도 계산 카드는 자기 상태를 가진다');
  const stale = viewOf(m, 'bidding-stale-11m', { advanceMs: 11 * 60_000, failAfter: true });
  assert.deepEqual(stale.deadline.chips.map(c => chipText(m, c)), ['시장 규칙 · 서버 시각', '지연 · 06:03 기준']);
  assert.deepEqual([stale.deadline.phase, stale.deadline.countdown], ['open', '3시간 45분 남음']);
  assert.equal(stale.answer, '발전소는 내일 만들 전기를 오늘 오전 10시까지 미리 팝니다. 마감까지 3시간 45분 남았습니다.', '접수 중 문장 유지');
  const broken = viewOf(m, 'bidding-stale-11m', { advanceMs: 61 * 60_000, failAfter: true });
  assert.deepEqual(broken.deadline.chips.map(c => [c.lag || null, chipText(m, c)]), [[null, '시장 규칙 · 서버 시각'], ['broken', '06:03 기준']]);
});

test('불일치(hours_remaining 1.0 대 3.9 · 10:00:30인데 open): 한 번 다시 조회하는 동안 확인 중, 그래도 다르면 서버 pill + 시각 확인 필요', () => {
  const m = load();
  for (const id of ['bidding-mismatch-hours', 'bidding-mismatch-open-after-deadline']) {
    for (const recheck of ['idle', 'pending']) {
      const v = viewOf(m, id, { recheck });
      assert.deepEqual([v.mismatch, v.deadline.phase, v.deadline.countdown, v.answer], [true, 'checking', null, '입찰 마감 시각을 확인하는 중입니다.'], `${id} ${recheck}`);
    }
    const done = viewOf(m, id, { recheck: 'done' });
    assert.deepEqual([done.deadline.phase, done.deadline.pill, done.deadline.countdown], ['mismatch', { tone: 'ok', text: '접수 중' }, null], id);
    assert.deepEqual(done.deadline.chips.map(c => chipText(m, c)), ['시장 규칙 · 서버 시각', '시각 확인 필요'], id);
    assert.equal(done.answer, '입찰 마감 시각을 확인하지 못해 남은 시간을 표시하지 않습니다.', id);
    assert.equal(done.deadline.when, '마감 11-26(목) 10:00 KST · 거래일 11-27(금)', id);
  }
  // 허용 차이 0.2시간 안(서버가 소수 첫째 자리로 반올림): 09:20의 0.7 대 0.667은 불일치가 아니다.
  assert.equal(viewOf(m, 'bidding-imminent-0920').mismatch, false);
  // 마감을 지나면(서버는 아직 open) 지금 시각만으로도 불일치가 된다: 화면이 다시 조회해 closed를 받는다.
  const passed = at(m, KST(2026, 11, 26, 10, 0, 5), marketAt(m, KST(2026, 11, 26, 9, 59, 30)));
  assert.equal(passed.mismatch, true);
});

test('같은 순간은 TZ=UTC·Asia/Seoul·America/Los_Angeles에서 같은 단계·카운트다운·마감·다음 구간 글자(7.5절)', () => {
  const helper = JSON.stringify(path.join(__dirname, 'helpers', 'ts-loader.cjs'));
  const src = file => JSON.stringify(path.join(SRC_ROOT, file));
  const instants = [KST(2026, 11, 26, 6, 3, 57), KST(2026, 11, 26, 9, 59, 30), KST(2026, 11, 26, 10, 0, 30), KST(2026, 11, 26, 23, 59, 30)];
  const script = `
    const { createTsLoader } = require(${helper});
    const loader = createTsLoader();
    const model = loader.load(${src('features/bidding/model.ts')});
    const kst = loader.load(${src('lib/kst.ts')});
    const types = loader.load(${src('data/types.ts')});
    const pad = n => String(n).padStart(2, '0');
    const out = ${JSON.stringify(instants)}.map(t => {
      const p = kst.kstParts(t);
      const today = kst.kstDateKey(t);
      const next = kst.kstAt(today, '10:00', p.hour >= 10 ? 1 : 0);
      const data = types.parseMarketStatus({ current_time: today + 'T' + pad(p.hour) + ':' + pad(p.minute) + ':' + pad(p.second),
        dam: { status: p.hour < 10 ? 'open' : 'closed', deadline: '10:00', trading_date: kst.addDays(today, 1), hours_remaining: Math.round((next - t) / 360e3) / 10 },
        rtm: { status: 'active', next_interval: '15분', interval_minutes: 15 } });
      const none = { data: null, fetchedAt: null, status: 'idle', error: null };
      const v = model.biddingView({ now: t, market: { data, fetchedAt: t, status: 'ready', error: null }, calc: { ...none, busy: false },
        forecast: null, calcForecastDate: null, risk: null, selectedHour: null, recheck: 'idle', staleRetried: false });
      return { phase: v.deadline.phase, countdown: v.deadline.countdown, when: v.deadline.when, rtm: v.deadline.rtm, answer: v.answer, localHour: new Date(t).getHours() };
    });
    process.stdout.write(JSON.stringify(out));`;
  const run = tz => JSON.parse(execFileSync(process.execPath, ['-e', script], { env: { ...process.env, TZ: tz }, encoding: 'utf8' }));
  const results = Object.fromEntries(['UTC', 'Asia/Seoul', 'America/Los_Angeles'].map(tz => [tz, run(tz)]));
  assert.notEqual(results.UTC[0].localHour, results['Asia/Seoul'][0].localHour, '양성 대조: 기기 시간대가 실제로 바뀌었다');
  const expected = [
    { phase: 'open', countdown: '3시간 56분 남음', when: '마감 11-26(목) 10:00 KST · 거래일 11-27(금)', next: '06:15' },
    { phase: 'imminent', countdown: '1분 미만 남음', when: '마감 11-26(목) 10:00 KST · 거래일 11-27(금)', next: '10:00' },
    { phase: 'closed', countdown: '23시간 59분', when: '다음 마감 11-27(금) 10:00 KST · 거래일 11-28(토)', next: '10:15' },
    { phase: 'closed', countdown: '10시간', when: '다음 마감 11-27(금) 10:00 KST · 거래일 11-28(토)', next: '00:00' },
  ];
  for (const [tz, rows] of Object.entries(results)) {
    rows.forEach((r, i) => {
      assert.deepEqual([r.phase, r.countdown, r.when], [expected[i].phase, expected[i].countdown, expected[i].when], `${tz} #${i}`);
      assert.ok(r.rtm.endsWith(`다음 구간 ${expected[i].next} KST`), `${tz} #${i} ${r.rtm}`);
    });
    assert.deepEqual(rows.map(({ localHour, ...rest }) => rest), results.UTC.map(({ localHour, ...rest }) => rest), tz);
  }
});

// ---------------------------------------------------------------- 2. 예시 발전소 입찰 계산(4.2·7.8절)
test('계산 기대 문자열: 19h · 18–19시 · 11-27 · 171.7–244.1 · 95%→5% · 합계 468.6만원 · 1구간 0h 56.1만원 · 칩·출처', () => {
  const m = load();
  const s = m.strings;
  const v = viewOf(m, 'bidding-base');
  const c = v.calc;
  assert.equal(c.state, 'ready');
  assert.deepEqual([c.selected, s.bidHourLabel(c.selected, c.tradingDate), c.first, c.last], [19, '19h · 18–19시 · 11-27', 1, 24], '기본 거래시간 19h(와이어프레임)');
  assert.equal(s.bidHourLabel(c.selected, null), '19h · 18–19시', '320에서는 날짜를 카드 부제로');
  assert.equal(s.bidDayText(c.tradingDate), '거래일 11-27(금)');
  assert.equal(c.summary, '50 MW를 10구간으로 나눠 171.7–244.1원에 내는 계산입니다.');
  assert.equal(c.trend, '가격이 낮은 구간일수록 낙찰 가능성(추정)이 높습니다 (95%→5%).');
  assert.equal(c.notReal, '낙찰 가능성은 기준선 분위수로 계산한 값이며 실제 낙찰 확률이 아닙니다.');
  assert.deepEqual([m.provenance.chipText(c.chip.chip), c.chip.source], ['▨ 추정 · 기준선 분위수', 'API 서버 입찰 계산 (기준선 분위수)']);
  assert.equal(`${c.footer.source} · ${c.footer.time}`, '출처 API 서버 입찰 계산 · 기준선 · 조회 06:03 KST');
  assert.deepEqual([m.format.krwFromThousand(c.total.revenue.value), m.format.mw(c.total.quantity.value), c.total.quantity.label], ['468.6만원', '50', '합계 물량']);
  assert.deepEqual(c.rows.map(r => r.segment), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']);
  assert.deepEqual([m.format.price(c.rows[0].price.value), m.format.pct(c.rows[0].probability.value, 0)], ['171.7', '95']);
  const first = m.model.biddingView(inputFor(m, 'bidding-base', { selectedHour: 1 })).calc;
  assert.deepEqual([first.selected, m.format.krwFromThousand(first.rows[0].revenue.value), m.format.price(first.rows[0].price.value)], [1, '56.1만원', '118.1']);
  assert.equal(c.chartLabel, '18–19시 입찰 계단', '접근 이름은 문장이라 시각 범위만(D11)');
  // 기대 수입은 표에만: 요약·추세·주의 문장 어디에도 없다.
  assert.ok(![c.summary, c.trend, c.notReal, v.answer].join(' ').includes('만원'));
  const facts = v.facts['bid-ladder'].lines;
  // D6: API 원문 낱말은 한국어 이름 뒤 괄호("API 값 …", "API 이름: …")에만. 성향 글자는 세그먼트와 같은 표(RISK_LABELS)에서 온다.
  assert.deepEqual(facts, ['계산 방법: 분위수 기반 확률 낙찰 모델 (API 이름: Quantile-based Probabilistic Clearing Model)',
    'API 모델 이름: 기준선 예측 (전날 같은 시간 제주 하루전 가격)', 'API 거래일 11-27(금) · 성향 균형 (API 값 moderate) · 예시 용량 50 MW']);
  assert.deepEqual(v.facts['bid-deadline'].lines, ['API 표기: 하루전시장 접수 중 (API 값 open) · 마감 10:00 · 거래일 2026-11-27 · 남은 시간 3.9시간',
    '서버 시각으로 다시 계산한 남은 시간 3.9시간 (차이가 0.2시간을 넘으면 남은 시간을 숨깁니다)']);
  assert.deepEqual(v.sources.map(r => [r.id, r.item, r.source, r.basis, chipText(m, r.chip)]),
    [['bidding', '입찰 계산', 'API 서버 입찰 계산 (기준선 분위수)', '조회 06:03', '▨ 추정']]);
  // round 1 D15: 출처 표는 계산을 부르지 않는다(서버가 한 번에 하나만 계산한다). 받기 전이면 어디서 계산되는지를 쓴다.
  const pendingCalc = st => viewOf(m, 'bidding-base', { calc: { ...empty, status: st, busy: false } }).sources[0].basis;
  assert.deepEqual(['idle', 'loading', 'error'].map(pendingCalc), ['입찰 화면을 열면 계산합니다', '불러오는 중', '불러오지 못함']);
});

test('가격 근거 줄(7.8절 2): 엿본 예측이 있고 거래일과 다르며 q50이 0.05 안일 때만, D+1 기준선(P5)이면 사라진다', () => {
  const m = load();
  const present = viewOf(m, 'bidding-price-basis-present', { withForecast: true }).calc;
  assert.equal(present.basis, '가격 근거: 11-26 기준선(=11-25 확정 가격) · 거래일 11-27과 하루 차이');
  for (const hour of [1, 8, 13, 24]) {
    assert.equal(viewOf(m, 'bidding-price-basis-present', { withForecast: true, selectedHour: hour }).calc.basis, present.basis, `${hour}h`);
  }
  assert.equal(viewOf(m, 'bidding-price-basis-absent', { withForecast: true }).calc.basis, null, 'D+1 기준선: q50이 달라 줄이 없다');
  // round 1 D5: 입찰 화면이 예측을 같은 저장소로 받는다. 받는 중이면 줄이 없고(받으면 생긴다), 받지 못했으면 확인할 수 없다는 것을 쓴다.
  assert.equal(viewOf(m, 'bidding-price-basis-present').calc.basis, null, '예측을 받는 중: 아직 쓰지 않는다');
  assert.equal(viewOf(m, 'bidding-price-basis-present', { forecastFailed: true }).calc.basis, '가격 근거 확인 전 · 서버 기준선 예측 사용 · 거래일 11-27');
  assert.equal(viewOf(m, 'bidding-price-basis-present', { withForecast: true, forecastFailed: true }).calc.basis, present.basis,
    '앞서 받은 예측이 있으면(다시 받기만 실패) 그 예측으로 확인한다');
  assert.equal(m.strings.priceBasisPendingText('model', '2026-11-28'), '가격 근거 확인 전 · 서버 학습 모델 예측 사용 · 거래일 11-28');
  assert.equal(viewOf(m, 'bidding-price-basis-forecast-error').calc.basis, '가격 근거 확인 전 · 서버 기준선 예측 사용 · 거래일 11-27', '변형(B6): 예측 502');
  // 날짜 뒤 '과/와'와 날 수: 11-22와, 11-28과, 이틀이면 "2일 차이".
  assert.equal(m.strings.priceBasisText('2026-11-21', '2026-11-22', 'baseline'), '가격 근거: 11-21 기준선(=11-20 확정 가격) · 거래일 11-22와 하루 차이');
  assert.equal(m.strings.priceBasisText('2026-11-26', '2026-11-28', 'model'), '가격 근거: 11-26 학습 모델 예측 · 거래일 11-28과 2일 차이');
});

test('계산 상태: 503 한 번(다시 보내 성공) · 두 번([다시 계산]) · 500 · 400(버튼 없음) · 8초 지연 · 빈 값', () => {
  const m = load();
  assert.equal(viewOf(m, 'bidding-calc-503-once').calc.state, 'ready');
  assert.deepEqual(viewOf(m, 'bidding-calc-503-twice').calc, { state: 'error', text: '다른 요청을 처리하는 중입니다 · 잠시 뒤 다시 계산해 주세요', retry: true });
  assert.deepEqual(viewOf(m, 'bidding-calc-500').calc, { state: 'error', text: '계산 결과를 불러오지 못했습니다', retry: true });
  const bad = { ...inputFor(m, 'bidding-base'), calc: { ...empty, status: 'error', error: new m.client.ApiHttpError('/api/v1/bidding/optimized-segments', 400), busy: false } };
  assert.deepEqual(m.model.biddingView(bad).calc, { state: 'error', text: '계산 조건 오류', retry: false });
  const shape = { ...bad, calc: { ...bad.calc, error: new m.types.ApiShapeError('/x', '$', 'object') } };
  assert.deepEqual(m.model.biddingView(shape).calc, { state: 'error', text: '계산 결과를 읽지 못했습니다', retry: true });
  assert.deepEqual(viewOf(m, 'bidding-calc-delay-8s').calc, { state: 'loading' });
  const none = viewOf(m, 'bidding-calc-empty').calc;
  assert.deepEqual([none.state, none.text], ['empty', '계산된 입찰 구간이 없습니다']);
});

test('오래된 입력: 계산의 거래일 ≠ 시장 상태 거래일, 또는 계산 뒤 예측일이 바뀜 → 다시 계산하는 동안 스켈레톤, 그래도 다르면 시각 확인 필요', () => {
  const m = load();
  const base = inputFor(m, 'bidding-base');
  const old = { ...base, calc: { ...base.calc, data: { ...base.calc.data, trading_date: '2026-11-26' } } };
  assert.equal(m.model.calcInputStale(old), true);
  assert.deepEqual(m.model.biddingView(old).calc, { state: 'loading' }, '처음에는 스켈레톤(한 번 다시 계산)');
  assert.deepEqual(m.model.biddingView({ ...old, staleRetried: true, calc: { ...old.calc, busy: true } }).calc, { state: 'loading' });
  const kept = m.model.biddingView({ ...old, staleRetried: true }).calc;
  assert.deepEqual([kept.state, kept.headChips.map(c => chipText(m, c))], ['ready', ['시각 확인 필요']]);
  assert.equal(m.model.calcInputStale(base), false);
  const forecast = inputFor(m, 'bidding-base', { withForecast: true }).forecast;
  assert.equal(m.model.calcInputStale({ ...base, forecast, calcForecastDate: '2026-11-25' }), true, '계산 뒤 예측일이 11-25 → 11-26');
  assert.equal(m.model.calcInputStale({ ...base, forecast, calcForecastDate: '2026-11-26' }), false);
});

// ---------------------------------------------------------------- 3. 계산기 요청 규칙(4.2·6.3절)
function fakeClock(start = 1_000_000) {
  let now = start;
  let seq = 0;
  const timers = new Map();
  const flush = async () => {
    for (let i = 0; i < 20; i += 1) await new Promise(res => setImmediate(res));
  };
  return {
    deps: { now: () => now, setTimer: (fn, ms) => { const id = ++seq; timers.set(id, { fn, at: now + ms }); return id; }, clearTimer: id => timers.delete(id) },
    get now() { return now; },
    pending: () => timers.size,
    flush,
    async advance(ms) {
      const end = now + ms;
      await flush();
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = due[1].at;
        due[1].fn();
        await flush();
      }
      now = end;
      await flush();
    },
  };
}

const SEGMENTS = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/round0/_api_v1_bidding_optimized-segments_capacity_mw_50_risk_level_moderate.json'), 'utf8'));

/** 손으로 끝내는 가짜 계산기: 부를 때마다 기다리는 요청 하나를 만든다. 동시에 몇 개가 열려 있었는지 센다. */
function fakeCalculator(m) {
  const calls = [];
  const open = [];
  let most = 0;
  const client = () => ({
    getOptimizedSegments: risk => new Promise((resolve, reject) => {
      calls.push(risk);
      const entry = { risk, resolve: () => { open.splice(open.indexOf(entry), 1); resolve(m.types.parseSegments(SEGMENTS)); },
        reject: status => { open.splice(open.indexOf(entry), 1); reject(new m.client.ApiHttpError('/api/v1/bidding/optimized-segments', status)); } };
      open.push(entry);
      most = Math.max(most, open.length);
    }),
  });
  return { client, calls, open, most: () => most };
}

test('계산기: 성향을 바꿔도 한 번에 하나만(앞 계산이 끝난 뒤 보낸다), 성향별 5분 기억, 기다리는 동안 중단되면 보내지 않는다', async () => {
  const m = load();
  const clock = fakeClock();
  const store = m.resource.createResourceStore(clock.deps);
  const gate = m.calc.createSingleFlight();
  const fake = fakeCalculator(m);
  const loader = risk => m.calc.calcLoader(risk, fake.client, gate);
  const a = store.load(m.calc.calcKey('moderate'), loader('moderate'));
  const b = store.load(m.calc.calcKey('aggressive'), loader('aggressive'));
  const c = store.load(m.calc.calcKey('conservative'), loader('conservative'));
  await clock.flush();
  assert.deepEqual(fake.calls, ['moderate'], '뒤 둘은 앞 계산이 끝나기를 기다린다');
  fake.open[0].resolve();
  await clock.flush();
  assert.deepEqual(fake.calls, ['moderate', 'aggressive']);
  fake.open[0].resolve();
  await clock.flush();
  fake.open[0].resolve();
  assert.deepEqual(await Promise.all([a, b, c]), [true, true, true]);
  assert.deepEqual([fake.calls, fake.most()], [['moderate', 'aggressive', 'conservative'], 1], '서버에 동시에 둘 이상 보내지 않는다');
  // 같은 성향을 5분 안에 다시 고르면 기억한 값(요청 없음), 5분이 지나면 다시 받는다.
  const key = m.calc.calcKey('moderate');
  assert.equal(m.resource.shouldLoadOnMount(store.get(key), store.now()), false);
  await clock.advance(5 * 60_000);
  assert.equal(m.resource.shouldLoadOnMount(store.get(key), store.now()), true);
  assert.equal(fake.calls.length, 3, '시간이 흘러도 스스로 다시 보내지 않는다(폴링 없음)');
  const ctl = new AbortController();
  ctl.abort();
  await assert.rejects(loader('moderate')(ctl.signal), { name: 'AbortError' });
  assert.equal(fake.calls.length, 3);
  assert.equal(m.calc.calcKey('aggressive'), '/api/v1/bidding/optimized-segments?risk_level=aggressive');
});

test('계산기 503: 1.5초 뒤 한 번만 자동으로 다시 보내고, 그래도 503이면 [다시 계산](세 번째 요청은 누를 때만), 400은 다시 보내지 않는다', async () => {
  const m = load();
  const clock = fakeClock();
  const store = m.resource.createResourceStore(clock.deps);
  const fake = fakeCalculator(m);
  const gate = m.calc.createSingleFlight();
  const key = m.calc.calcKey('moderate');
  const loader = m.calc.calcLoader('moderate', fake.client, gate);
  const once = store.load(key, loader);
  await clock.flush();
  fake.open[0].reject(503);
  await clock.advance(1499);
  assert.equal(fake.calls.length, 1, '1.5초 전에는 다시 보내지 않는다');
  await clock.advance(1);
  fake.open[0].resolve();
  assert.equal(await once, true);
  assert.equal(fake.calls.length, 2);
  // 두 번 연속 503 → 오류, 화면은 "다른 요청을 처리하는 중입니다" + [다시 계산].
  const twice = store.load(key, loader);
  await clock.flush();
  fake.open[0].reject(503);
  await clock.advance(1500);
  fake.open[0].reject(503);
  assert.equal(await twice, false);
  const snap = store.get(key);
  assert.deepEqual([snap.status, snap.error.status, fake.calls.length], ['error', 503, 4]);
  await clock.advance(60 * 60_000);
  assert.equal(fake.calls.length, 4, '오류 뒤에도 스스로 다시 보내지 않는다');
  const base = inputFor(m, 'bidding-base');
  const view = m.model.biddingView({ ...base, calc: { ...base.calc, data: null, fetchedAt: null, status: 'error', error: snap.error } });
  assert.deepEqual(view.calc, { state: 'error', text: '다른 요청을 처리하는 중입니다 · 잠시 뒤 다시 계산해 주세요', retry: true });
  const press = store.load(key, loader);
  await clock.flush();
  fake.open[0].resolve();
  assert.equal(await press, true);
  assert.equal(fake.calls.length, 5);
  const four = store.load(m.calc.calcKey('aggressive'), m.calc.calcLoader('aggressive', fake.client, gate));
  await clock.flush();
  fake.open[0].reject(400);
  assert.equal(await four, false);
  await clock.advance(10_000);
  assert.equal(fake.calls.length, 6, '400은 다시 보내지 않는다');
});

// ---------------------------------------------------------------- 4. 카드 렌더(노드 수준 I2·의미)
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

/** I2(노드 수준): 숫자가 든 글자는 값·메타·칩·예시 표지 안에만. 값은 칩과 출처 줄(role=note)이 있는 카드 안에만, 원값은 숫자. */
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
      assert.ok(inCard && all(inCard, n => ds(n).pdChip !== undefined).length > 0, `${label}: 칩 없는 카드의 값 ${textOf(node)}`);
      assert.ok(all(inCard, n => n.props.role === 'note').length > 0, `${label}: 출처 줄 없는 카드의 값 ${textOf(node)}`);
      assert.ok(Number.isFinite(Number(ds(node).pdRaw)), `${label}: 원값 ${ds(node).pdRaw}`);
    }
    node.children.forEach(c => walkValues(c, inCard));
  };
  walkValues(root, null);
}

const noop = () => {};
function renderBidding(m, view) {
  return tree([
    h(m.DeadlineCard, { view: view.deadline, onMethod: noop, onSources: noop, onRetry: noop }),
    h(m.SampleBidCard, { view: view.calc, risk: 'moderate', onRisk: noop, onHour: noop, onRetry: noop, onMethod: noop, onSources: noop }),
    h(m.MeritOrderCard, { onMethod: noop }),
    h(m.RestrictedCard, { title: m.strings.MY_PLANT_TITLE, body: m.strings.MY_PLANT_BODY }),
  ]);
}
const B6 = ['bidding-base', 'bidding-checking', 'bidding-imminent-0920', 'bidding-closed-1001', 'bidding-mismatch-hours', 'bidding-mismatch-open-after-deadline',
  'bidding-market-error', 'bidding-stale-11m', 'bidding-calc-503-once', 'bidding-calc-503-twice', 'bidding-calc-500', 'bidding-calc-delay-8s',
  'bidding-price-basis-present', 'bidding-price-basis-absent', 'bidding-calc-empty'];

test('카드 렌더: B6 변형 × 390 차트·표·펼침 / 320 / 768에서 표지 밖 숫자 0, 값은 칩·출처 있는 카드 안, 늘 "KPX에 아무것도 보내지 않습니다"', () => {
  for (const [width, opts] of [[390, {}], [390, { forceTable: true }], [390, { open: true }], [320, {}], [768, { forceTable: true, open: true }]]) {
    const m = load(width, opts);
    for (const id of B6) {
      for (const recheck of ['idle', 'done']) {
        const extra = id === 'bidding-stale-11m' ? { advanceMs: 11 * 60_000, failAfter: true } : {};
        const root = renderBidding(m, m.model.biddingView(inputFor(m, id, { withForecast: true, recheck, ...extra })));
        assertI2(root, `${id}@${width}${JSON.stringify(opts)}/${recheck}`);
        assert.ok(textOf(root).includes('이 화면은 KPX에 아무것도 보내지 않습니다.'), `${id}@${width}`);
      }
    }
  }
});

test('카드 렌더 의미: [i] 셋 · [출처 표] 둘 · 성향 radio 셋 · 스텝퍼(양 끝 멈춤) · 계단 이름은 "10구간"으로 시작 · 제출·저장 버튼 없음 · 44 px', () => {
  const m = load();
  const view = viewOf(m, 'bidding-base', { withForecast: true });
  const root = renderBidding(m, view);
  const presses = all(root, n => n.type === 'Pressable');
  const names = presses.map(p => p.props['aria-label']);
  assert.deepEqual(names.filter(n => n.startsWith('산출 방식 보기')),
    ['산출 방식 보기: 하루전시장 입찰 마감', '산출 방식 보기: 예시 발전소로 입찰 계산해 보기', '산출 방식 보기: 가격은 어떻게 정해지나요?']);
  assert.deepEqual(names.filter(n => n.startsWith('출처 표')), ['출처 표: 하루전시장 입찰 마감', '출처 표: 예시 발전소로 입찰 계산해 보기']);
  const radios = presses.filter(p => p.props.role === 'radio').map(p => [p.props['aria-label'], p.props['aria-checked']]);
  assert.deepEqual(radios, [['보수적', false], ['균형', true], ['공격적', false], ['차트', true], ['표', false]]);
  assert.deepEqual(all(root, n => n.props.role === 'radiogroup').map(g => g.props['aria-label']), ['성향', '보기 방식']);
  assert.ok(names.includes('이전 거래시간') && names.includes('다음 거래시간'));
  assert.ok(!names.some(n => /제출|저장|등록|삭제|전송|보내기|입찰하기/.test(n)), names.join(' / '));
  presses.forEach(p => {
    const s = flat(p.props.style);
    assert.ok(Math.max(s.minHeight || 0, s.height || 0) >= 44, `${p.props['aria-label']} 높이`);
  });
  const ladder = all(root, n => n.props.role === 'img' && String(n.props['aria-label']).startsWith('10구간'));
  assert.equal(ladder.length, 1, '하니스 B3가 "10구간"으로 찾는다');
  assert.equal(ladder[0].props['aria-label'], '10구간, 171.7–244.1원, 낙찰 가능성 95%에서 5%(추정) · 18–19시 입찰 계단');
  const merit = all(root, n => n.type === 'Pressable' && String(n.props['aria-label']).startsWith('가격'));
  assert.deepEqual(merit.map(p => [p.props['aria-label'], p.props['aria-expanded']]), [['가격은 어떻게 정해지나요? 펼치기', false]]);
  assert.equal(all(root, n => n.props.testID === 'merit-body').length, 0, '처음에는 접혀 있다');
  const basis = all(root, n => n.props.testID === 'bid-basis')[0];
  assert.equal(textOf(basis), '가격 근거: 11-26 기준선(=11-25 확정 가격) · 거래일 11-27과 하루 차이');
  // 날짜·값 범위는 붙임표·줄표에서 줄이 갈리지 않게 묶는다(5.4절, 320에서 "11-" / "27과"로 갈렸다).
  const nowrap = node => all(node, n => ds(n).pdNowrap === 'true').map(textOf);
  // 짧은 괄호는 통째로 묶는다(round 1: "(MAPE" / "11.75%)"가 갈렸다): 안의 날짜도 함께 한 줄.
  assert.deepEqual(nowrap(basis), ['11-26', '(=11-25 확정 가격)', '11-27']);
  assert.deepEqual(nowrap(all(root, n => ds(n).pdMeta === 'deadline')[0]), ['11-26(목)', '10:00 KST', '11-27(금)'], '시각도 "10:00" / "KST"로 갈리지 않는다');
  assert.deepEqual(nowrap(all(root, n => ds(n).pdMeta === 'summary')[0]), ['50 MW', '171.7–244.1원'], '값+단위도 한 줄');
  // F13c 갈림 탐침: 320에서 실시간시장 줄이 "06:15" / "KST"로, 추세 문장의 괄호가 갈릴 수 있었다(keepTogether 밖이었다).
  // round 2: 이름표와 시각을 한 묶음으로("다음 구간" / "06:15 KST"가 갈리지 않게).
  assert.deepEqual(nowrap(all(root, n => ds(n).pdMeta === 'rtm')[0]), ['다음 구간 06:15 KST']);
  assert.ok(nowrap(all(root, n => ds(n).pdMeta === 'summary')[1]).includes('(95%→5%)'));
  const countdown = all(root, n => n.props.testID === 'bid-countdown')[0];
  assert.deepEqual([textOf(countdown), ds(countdown).pdMeta, flat(countdown.props.style).fontSize], ['3시간 56분 남음', 'countdown', 32]);
  // 스텝퍼는 양 끝에서 멈춘다(입찰에는 돌기 규칙이 없다, 하니스 B2): 01h에서 ‹ 비활성(까닭), 24h에서 › 비활성.
  for (const [hour, disabled] of [[1, '이전 거래시간 · 첫 거래시간입니다'], [24, '다음 거래시간 · 마지막 거래시간입니다']]) {
    const r = renderBidding(m, viewOf(m, 'bidding-base', { selectedHour: hour }));
    assert.ok(all(r, n => n.type === 'Pressable').some(p => p.props['aria-label'] === disabled), `${hour}h`);
  }
  // 제한 카드: 잠김 칩, 버튼 없음.
  const restricted = all(root, n => n.props.testID === undefined && ds(n).pdCard && textOf(n).includes('내 발전소 관리와'))[0];
  assert.equal(all(restricted, n => n.type === 'Pressable').length, 0);
  assert.ok(textOf(restricted).includes('잠김'));
});

test('D6: 보이는 글자에 API 원문 성향·영어 계산 방법이 맨몸으로 없다(한국어 이름 뒤 "API 값"·"API 이름" 괄호에만)', () => {
  const m = load();
  const bare = /(?<!API 값 )\b(conservative|moderate|aggressive)\b/;
  for (const [risk, label] of [['conservative', '보수적'], ['moderate', '균형'], ['aggressive', '공격적']]) {
    const input = inputFor(m, 'bidding-base');
    const v = m.model.biddingView({ ...input, risk, calc: { ...input.calc, data: { ...input.calc.data, risk_level: risk } } });
    const lines = Object.values(v.facts).flatMap(f => f.lines);
    assert.ok(lines.includes(`API 거래일 11-27(금) · 성향 ${label} (API 값 ${risk}) · 예시 용량 50 MW`), risk);
    for (const line of lines) assert.doesNotMatch(line, bare, line);
    assert.ok(lines.filter(l => /Quantile/.test(l)).every(l => /\(API 이름: Quantile-based Probabilistic Clearing Model\)$/.test(l)), risk);
    // 카드(마감·계단·가격 원리·내 발전소)에는 영어 API 낱말이 아예 없다: 성향 세그먼트 글자는 같은 표(RISK_LABELS)에서 온다.
    const root = renderBidding(m, v);
    assert.doesNotMatch(textOf(root), /\b(conservative|moderate|aggressive|Quantile|open|closed)\b/, risk);
    const radios = all(root, n => n.props.role === 'radio').map(r => r.props['aria-label'] || textOf(r));
    assert.deepEqual(radios.filter(l => /보수적|균형|공격적/.test(l)), ['보수적', '균형', '공격적'], risk);
  }
  assert.equal(m.strings.calcMethodText('Some New Method'), '계산 방법: 한국어 이름 없음 (API 이름: Some New Method)');
  assert.equal(m.strings.calcMethodText('constructor'), '계산 방법: 한국어 이름 없음 (API 이름: constructor)', '표의 자기 키만 본다');
  assert.match(m.strings.calcInputText('2026-11-27', 'toString', 50), /성향 알 수 없음 \(API 값 toString\)/);
  assert.equal(m.strings.calcMethodText('기준선 분위수'), '계산 방법: 기준선 분위수');
  assert.equal(m.strings.calcMethodText(null), '계산 방법: API 표기 없음');
  assert.match(m.strings.deadlineApiText('weird', '10:00', '2026-11-27', 3.9), /^API 표기: 하루전시장 알 수 없음 \(API 값 weird\) · /);
});

test('표 보기(600 이상): 구간 10행 + 합계 행, 가격은 값(원값), 낙찰 가능성은 계산값, 물량·기대 수입은 예시, 기대 수입 머리에 ┆예시┆ [▨ 추정]', () => {
  const m = load(768, { forceTable: true });
  const root = renderBidding(m, viewOf(m, 'bidding-base'));
  const table = all(root, n => n.props.role === 'table')[0];
  assert.equal(table.props['aria-label'], '18–19시 구간별 입찰 계산');
  const rows = all(table, n => n.props.role === 'row');
  assert.equal(rows.length, 12, '머리 + 10구간 + 합계');
  const heads = all(rows[0], n => n.props.role === 'columnheader');
  assert.deepEqual(heads.map(hd => textOf(all(hd, n => n.type === 'Text')[0])), ['구간', '물량(MW, 예시)', '가격(원/kWh)', '낙찰 가능성(추정)', '기대 수입(입찰가)']);
  assert.deepEqual(all(heads[4], n => ds(n).pdChip).map(n => n.props['aria-label']), ['상태: 예시', '상태: 추정, 출처 API 서버 입찰 계산 (기준선 분위수)']);
  const cells = all(rows[1], n => n.props.role === 'cell' || n.props.role === 'rowheader');
  assert.deepEqual(cells.map(textOf), ['1', '5 (예시)', '171.7', '95%', '81.6만원 (예시)']);
  assert.deepEqual([ds(all(cells[2], n => ds(n).pdValue)[0]).pdRaw, all(cells[3], n => ds(n).pdValue).length, ds(all(cells[3], n => n.type === 'Text')[0]).pdMeta],
    ['171.7', 0, 'derived'], '가격만 API 원값, 낙찰 가능성(×100)은 계산값');
  assert.ok(all(cells[1], n => ds(n).pdExample).length === 1 && all(cells[4], n => ds(n).pdExample).length === 1);
  assert.deepEqual(all(rows[11], n => n.props.role === 'cell' || n.props.role === 'rowheader').map(textOf), ['합계', '50 (예시)', '—', '—', '468.6만원 (예시)']);
  // 첫 열은 행 머리(round 2): 칸을 옮겨 다니는 화면 읽기에서 값마다 구간 이름이 함께 읽힌다.
  assert.deepEqual(all(rows[11], n => n.props.role === 'rowheader').map(textOf), ['합계']);
});

test('행 카드(600 미만, D2): 표 대신 목록 11행(10구간 + 합계), 값은 모두 보이고 조각마다 한 줄, 기대 수입 칩은 목록 위 한 줄', () => {
  for (const width of [320, 390]) {
    const m = load(width, { forceTable: true });
    const root = renderBidding(m, viewOf(m, 'bidding-base'));
    assert.equal(all(root, n => n.props.role === 'table').length, 0, `${width}: 가로로 밀리는 표 없음`);
    const [list] = all(root, n => n.props.role === 'list');
    assert.equal(list.props['aria-label'], '18–19시 구간별 입찰 계산');
    const items = all(list, n => n.props.role === 'listitem');
    assert.equal(items.length, 11);
    assert.equal(textOf(items[0]), '1구간물량(MW, 예시) 5 (예시) · 가격(원/kWh) 171.7 · 낙찰 가능성(추정) 95% · 기대 수입(입찰가) 81.6만원 (예시)');
    assert.equal(textOf(items[10]), '합계물량(MW, 예시) 50 (예시) · 기대 수입(입찰가) 468.6만원 (예시)', '합계 행의 빈 칸(가격·낙찰 가능성)은 행 카드에서 뺀다');
    // 값 표지는 표와 같다: 가격은 API 원값(data-pd-value), 낙찰 가능성은 계산값, 물량·기대 수입은 예시.
    assert.equal(ds(all(items[0], n => ds(n).pdValue)[0]).pdRaw, '171.7');
    assert.equal(all(items[0], n => ds(n).pdExample).length, 2);
    const pieces = all(items[0], n => ds(n).pdNowrap === 'true').map(textOf);
    assert.ok(pieces.includes('가격(원/kWh) 171.7') && pieces.includes('낙찰 가능성(추정) 95%'), pieces.join(' | '));
    const chipsLine = all(root, n => n.props.testID === 'bid-cards')[0].children[0];
    assert.equal(textOf(chipsLine), '기대 수입(입찰가)┆ 예시▨ 추정');
  }
});

test('성향 줄은 모든 상태에서 카드의 같은 자리(D4): 계산 중·오류·빈 값·준비 모두 리본 다음 [발전소 줄, 거래시간·성향 칸, …]의 둘째', () => {
  const m = load();
  const states = { ready: viewOf(m, 'bidding-base').calc, loading: { state: 'loading' }, error: { state: 'error', text: '계산 결과를 불러오지 못했습니다', retry: true },
    empty: { state: 'empty', text: '계산된 입찰 구간이 없습니다', footer: { source: '출처 x', time: '조회 06:05 KST' } } };
  for (const [name, view] of Object.entries(states)) {
    const root = tree(h(m.SampleBidCard, { view, risk: 'moderate', onRisk: noop, onHour: noop, onRetry: noop, onMethod: noop, onSources: noop }));
    const body = all(root, n => n.type === 'Provider')[0];
    const slot = body.children[1];
    const groups = all(slot, n => n.props.role === 'radiogroup');
    assert.deepEqual(groups.map(g => g.props['aria-label']), ['성향'], `${name}: 둘째 자리는 거래시간·성향 칸`);
    // 성향 줄은 그 칸의 마지막 자식이다(준비 상태에서만 거래시간 줄이 앞에 끼어든다: 성향 세그먼트는 옮겨지지 않는다).
    assert.equal(all(slot.children[slot.children.length - 1], n => n.props.role === 'radiogroup').length, 1, name);
  }
});

test('계단 차트 범례·판독 칩(D12)과 마감 뒤 카드(D13): 기준선·80% 구간 값, 고른 구간 칩, 큰 숫자 위 "다음 마감까지", 제목 옆 «오늘 마감됨»', () => {
  const m = load();
  const view = viewOf(m, 'bidding-base');
  const root = renderBidding(m, view);
  // round 2: 기호 + 이름(파선 · 띠 · 빗금 계단) 한 줄 범례.
  assert.deepEqual(all(all(root, n => n.props.testID === 'bid-legend')[0], n => ds(n).pdMeta === 'legend').map(textOf),
    ['기준선 205.9원', '80% 구간 175.2–239.3원', '입찰 가격(추정)']);
  assert.deepEqual(m.strings.ladderKeyItems('baseline', 205.9, 175.2, 239.3).map(k => k.swatch), ['dash', 'band', 'hatch']);
  assert.equal(all(root, n => n.props.testID === 'bid-reading').length, 0, '고르기 전에는 판독 칩 없음');
  const closed = viewOf(m, 'bidding-closed-1001');
  const card = tree(h(m.DeadlineCard, { view: closed.deadline, onMethod: noop, onSources: noop, onRetry: noop }));
  const countdown = all(card, n => n.props.testID === 'bid-countdown')[0];
  assert.deepEqual([textOf(countdown), flat(countdown.props.style).fontSize, ds(countdown).pdNowrap], ['23시간 59분', 32, 'true']);
  assert.ok(textOf(card).includes('다음 마감까지23시간 59분다음 마감 11-27(금) 10:00 KST · 거래일 11-28(토)'), textOf(card));
  const pills = all(card, n => ds(n).pdPill);
  assert.deepEqual(pills.map(textOf), ['오늘 마감됨'], '«마감» pill은 날짜 줄 밑이 아니라 제목 옆 하나');
  // 마감 뒤 예시 계산의 날짜는 마감된 거래일이라고 밝힌다(round 1 integrity).
  const after = tree(h(m.SampleBidCard, { view: view.calc, risk: 'moderate', onRisk: noop, onHour: noop, onRetry: noop, onMethod: noop, onSources: noop, closed: true }));
  assert.equal(textOf(all(after, n => n.props.testID === 'bid-plant')[0]), '예시 풍력발전소 · 50 MW (예시) (가상) · 마감된 거래일 11-27 기준');
});

test('가격 원리(펼침): 예시 6개를 싼 순서로 쌓은 그림 + 예시 수요선, 마지막 발전기 유류 E 220원, 글 대안 6행, 펼침 상태 aria-expanded', () => {
  const m = load(390, { open: true });
  const blocks = m.examples.EXAMPLE_MERIT_BLOCKS.map(b => [b.name.value, b.bidPrice.value]);
  assert.deepEqual(blocks, [['풍력 A', 0], ['태양광 B', 0], ['바이오 C', 80], ['LNG D', 150], ['유류 E', 220], ['유류 F', 260]], '명세 4.2 그대로');
  assert.ok(blocks.every(([, p], i) => i === 0 || p >= blocks[i - 1][1]), '싼 것부터');
  const ex = m.model.meritExample();
  assert.equal(ex.marginal, 4);
  assert.equal(ex.summary, '예시 수요 450 MW를 채우는 마지막 발전기(유류 E)의 가격 220원/kWh가 이 시간 모두의 가격(SMP)이 됩니다.');
  const root = tree(h(m.MeritOrderCard, { onMethod: noop }));
  const toggle = all(root, n => n.type === 'Pressable' && String(n.props['aria-label']).startsWith('가격'))[0];
  assert.deepEqual([toggle.props['aria-label'], toggle.props['aria-expanded']], ['가격은 어떻게 정해지나요? 접기', true]);
  const chart = all(root, n => n.props.testID === 'merit-chart')[0];
  assert.deepEqual([chart.props.role, chart.props['aria-label']], ['img', ex.summary]);
  const bars = all(chart, n => n.props.testID === 'merit-bar');
  assert.equal(bars.length, 6);
  // 예시 호박색 한 벌(round 2: 두 테마에서 같은 채움, 외곽선만 테마별). 공급자 밖 기본 테마는 light.
  const S = m.tokens.SERIES;
  const edge = m.tokens.THEMES.light.chart.exampleEdge;
  assert.deepEqual(bars.map(b => b.props.fill === S.exampleMark), [false, false, false, false, true, false], '마지막 발전기만 진하게 채운다');
  assert.ok(bars.every(b => b.props.height >= 3 && b.props.stroke === edge), '0원 발전기도 보인다');
  // 수요선은 마지막 발전기 가격 글자 위아래 두 토막(글자를 가로지르지 않는다).
  const demand = all(chart, n => n.props.testID === 'merit-demand');
  const label = all(chart, n => n.type === 'Text' && textOf(n) === '220원')[0];
  assert.equal(demand.length, 2);
  assert.ok(demand.every(l => l.props.x1 === demand[0].props.x1 && l.props.y2 > l.props.y1));
  assert.ok(label, '마지막 발전기 가격 글자');
  const list = all(root, n => n.props.testID === 'merit-list')[0];
  assert.deepEqual(list.children.map(textOf), ['풍력 A · 100 MW (예시) · 0원/kWh (예시)', '태양광 B · 80 MW (예시) · 0원/kWh (예시)',
    '바이오 C · 40 MW (예시) · 80원/kWh (예시)', 'LNG D · 200 MW (예시) · 150원/kWh (예시)', '유류 E · 120 MW (예시) · 220원/kWh (예시)',
    '유류 F · 80 MW (예시) · 260원/kWh (예시)']);
  assert.deepEqual(all(root, n => ds(n).pdChip).map(n => n.props['aria-label']), ['상태: 예시', '상태: 단순화한 설명']);
});

test('guide.ko.json(실제 파일)이 로더 검사를 통과하고, 입찰 카드 설명 글이 4.2절 문장 그대로다', () => {
  const guide = createTsLoader().load(path.join(SRC_ROOT, 'content/guide.ts'));
  const content = guide.parseGuide(JSON.parse(fs.readFileSync(path.join(__dirname, '../public/content/guide.ko.json'), 'utf8')));
  assert.deepEqual(content.cards['bid-merit'].body,
    ['필요한 전기를 채우는 마지막 발전기의 가격이 그 시간 모두의 가격(SMP)이 됩니다.', '단순화한 설명입니다. 실제 시장 규칙은 KPX 규정을 따릅니다.']);
  assert.ok(content.cards['bid-ladder'].body.join(' ').includes('KPX에 아무것도 보내지 않습니다'));
  assert.ok(content.cards['bid-deadline'].body.join(' ').includes('서버의 시장 상태'));
});

test('App.tsx: 입찰 경로는 새 탭(BiddingTab), 레거시 입찰 화면을 가져오지 않는다(번들에서 빠진다)', () => {
  const app = fs.readFileSync(path.join(__dirname, '../App.tsx'), 'utf8');
  assert.match(app, /case 'bidding':\s*return <BiddingTab \/>/);
  assert.doesNotMatch(app, /screens\/BiddingScreen/);
  assert.match(app, /case 'smp':\s*return <SMPForecastScreen/);
});
