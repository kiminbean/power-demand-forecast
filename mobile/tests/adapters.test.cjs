'use strict';
// src/data/{types,client,adapters,examples}.ts와 src/content/guide.ts(명세 6.3·7.2·7.8절, 4.4절 로더).
// round 0 스냅숏(2026-11-26 06:03:57 KST)을 재생해 값·단위·칩·출처·시각과 모양 검사, 허용 목록 GET만 쓰는지 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createTsLoader, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const loader = createTsLoader();
const load = rel => loader.load(path.join(SRC_ROOT, rel));
const types = load('data/types.ts');
const a = load('data/adapters.ts');
const { createClient, ENDPOINTS, ApiHttpError, ApiNetworkError } = load('data/client.ts');
const ex = load('data/examples.ts');
const f = load('lib/format.ts');
const s = load('content/strings.ts');
const guide = load('content/guide.ts');

const R0 = path.join(__dirname, 'fixtures', 'round0');
const fx = name => JSON.parse(fs.readFileSync(path.join(R0, name), 'utf8'));
const KST = (y, mo, d, h, mi, sec = 0) => Date.UTC(y, mo - 1, d, h, mi, sec) - 9 * 3600e3;
const FETCHED = KST(2026, 11, 26, 6, 4, 2);
const SEGMENTS_FILE = '_api_v1_bidding_optimized-segments_capacity_mw_50_risk_level_moderate.json';
const ALLOW_LIST = ['/health', '/api/v1/health', '/api/v1/smp-forecast', '/api/v1/market-status', '/api/v1/dashboard/kpis',
  '/api/v1/bidding/optimized-segments', '/api/v1/model/info', '/api/v1/smp/current', '/api/v1/smp/day-ahead', '/api/v1/power-supply',
  '/api/v1/realtime-status', '/smp/rtm/model/info', '/smp/rtm/predict', '/api/v1/resources'];

test('day-ahead: confirmed price 115.3 원/kWh for trade hour 07h · 06–07시; unavailable otherwise', () => {
  const view = a.adaptDayAhead(types.parseDayAhead(fx('_api_v1_smp_day-ahead.json')), FETCHED);
  assert.equal(view.status, 'ready');
  const { price } = view;
  assert.deepEqual([f.price(price.value), price.unit, price.state, price.chip.text, price.source, price.time.kind, price.time.kst, price.note],
    ['115.3', '원/kWh', 'confirmed', '확정 가격', 'KPX 제주 하루전시장', '거래시간', '07h · 06–07시', '조회 06:04 KST']);
  assert.deepEqual([view.tradeHour, view.tradeDay], [7, '2026-11-26']);
  assert.ok(Object.isFrozen(price) && Object.isFrozen(price.time));
  const base = fx('_api_v1_smp_day-ahead.json');
  for (const patch of [{ status: 'unavailable' }, { current_smp: null }, { trade_hour: 0 }, { trade_hour: 25 }, { trade_day: '2026-02-30' }]) {
    assert.deepEqual(a.adaptDayAhead(types.parseDayAhead({ ...base, ...patch }), FETCHED), { status: 'unavailable' }, JSON.stringify(patch));
  }
});

test('forecast: baseline kind from the same response, made 06:03, band only for non-fallback', () => {
  const raw = fx('_api_v1_smp-forecast.json');
  const view = a.adaptForecast(types.parseForecast(raw));
  assert.deepEqual([view.kind, view.shortName, view.forecastDate, view.q50.chip.text, view.q50.state, view.q50.source, view.q50.time.kst, view.q50.note],
    ['baseline', '기준선', '2026-11-26', '예측 · 기준선', 'forecast', 'API 서버 기준선 예측', '만든 시각 06:03 KST', '예측일 2026-11-26']);
  assert.equal(view.modelUsed, raw.model_used);
  assert.equal(view.q50.value.length, 24);
  assert.ok(view.band && view.band.q10.length === 24);
  const fallback = a.adaptForecast(types.parseForecast({ ...raw, model_used: 'EPSIS Fallback (Model Unavailable)' }));
  assert.deepEqual([fallback.kind, fallback.q50.state, fallback.q50.chip.text, fallback.band], ['fallback', 'estimated', '추정 · API 대체값 (모델 예측 아님)', null]);
});

test('KPIs: demand 664 MW and reserve 79.5 % measured (③), level 정상, live weather 11.0 °C · 3.6 m/s · 맑음', () => {
  const view = a.adaptKpis(types.parseKpis(fx('_api_v1_dashboard_kpis.json')), FETCHED);
  assert.deepEqual([f.mw(view.demand.value), view.demand.unit, view.demand.state, view.demand.chip.text, view.demand.source],
    ['664', 'MW', 'measured', '실측', 'KPX 제주 실시간 (API 서버)']);
  assert.equal(view.demand.time.kst, '관측 시각 미제공 · 조회 06:04 KST');
  assert.deepEqual([f.pct(view.reserveRate.value), view.reserveRate.unit, view.reserveRate.note], ['79.5', '%', undefined]);
  assert.equal(view.level.label, '정상 · 넉넉합니다');
  const w = view.weather;
  assert.deepEqual([w.kind, f.num(w.temperature.value), w.temperature.unit, f.num(w.windSpeed.value), w.condition.value, w.temperature.chip.text, w.temperature.source],
    ['live', '11.0', '°C', '3.6', '맑음', '실측', '기상청 초단기실황 (API 서버)']);
  const sim = a.adaptKpis(types.parseKpis({ ...fx('_api_v1_dashboard_kpis.json'), data_source: 'Simulated' }), FETCHED);
  assert.deepEqual([sim.demand.chip.text, sim.reserveRate.state, sim.reserveRate.note, sim.level, sim.weather.kind],
    ['시뮬레이션 · 실측 아님', 'default', ' · API 기본값 (실측 아님)', null, 'default']);
  const noWeather = a.adaptKpis(types.parseKpis({ ...fx('_api_v1_dashboard_kpis.json'), weather: null }), FETCHED);
  assert.equal(noWeather.weather, null);
});

test('supply pair: 1,191 − 664 = 527 MW, watch boundary 1,036 MW; mismatch → 공급능력 미수신', () => {
  const kpis = types.parseKpis(fx('_api_v1_dashboard_kpis.json'));
  const supply = types.parsePowerSupply(fx('_api_v1_power-supply.json'));
  const view = a.adaptSupply(kpis, supply, FETCHED);
  assert.equal(view.status, 'paired');
  assert.deepEqual([f.mw(view.supply.value), f.mw(view.reserve.value), f.mw(view.watchBoundary.value)], ['1,191', '527', '1,036']);
  assert.deepEqual([view.supply.chip.text, view.supply.note], ['실측', '수급 자료의 지금 시각 값']);
  const off = { ...supply, data: supply.data.map(r => (r.hour === 6 ? { ...r, demand: 700 } : r)) };
  assert.deepEqual(a.adaptSupply(kpis, off, FETCHED), { status: 'missing', text: '공급능력 미수신' });
});

test('market status and model info: server time, skew, trading date, API-stated accuracy line', () => {
  const market = a.adaptMarketStatus(types.parseMarketStatus(fx('_api_v1_market-status.json')), KST(2026, 11, 26, 6, 0, 57));
  assert.equal(market.serverTime, KST(2026, 11, 26, 6, 3, 57) + 768);
  assert.deepEqual([market.skew.notable, market.skew.plausible, market.tradingDate, market.dam.deadline, market.serverText],
    [true, true, '2026-11-27', '10:00', '서버 시각 06:03 KST']);
  assert.equal(f.kstDayLabel(market.tradingDate), '11-27(금)');
  const info = a.adaptModelInfo(types.parseModelInfo(fx('_api_v1_model_info.json')));
  assert.deepEqual([info.apiMape.value, info.apiCoverage.value, info.apiMape.chip.text], [12.68, 72.64, 'API 표기']);
  assert.equal(s.apiAccuracyText(info.apiMape.value, info.apiCoverage.value),
    'API 표기: 검증 MAPE 12.68% (최근 150일 재현) · 80% 구간 적중률 72.6% (최근 30일 보류 검증)');
  assert.equal(s.accuracyText(), '시간마다 평균 11.97원/kWh 빗나갑니다 (MAPE 11.75%)');
  assert.equal(s.accuracyBasisText(), 'walk-forward 2025-07-22~12-18 · 기준선 자체(학습한 모델 아님)');
});

test('bidding calculator: hour is 0-based, q50 matches the forecast, revenue in thousand won (56.1만원 · 468.6만원)', () => {
  const raw = fx(SEGMENTS_FILE);
  const view = a.adaptSegments(types.parseSegments(raw), FETCHED);
  const forecast = fx('_api_v1_smp-forecast.json');
  assert.deepEqual(view.hours.map(h => h.tradeHour), Array.from({ length: 24 }, (_, i) => i + 1));
  view.hours.forEach((h, i) => assert.equal(f.price(h.baselineQ50), f.price(forecast.q50[i]), `hour ${i}`));
  const seg = view.hours[0].segments[0];
  assert.deepEqual([f.price(seg.price.value), seg.price.chip.text, f.pct(seg.probability.value, 0), seg.probability.note, f.krwFromThousand(seg.revenue.value)],
    ['118.1', '추정 · 기준선 분위수', '95', '실제 낙찰 확률 아님', '56.1만원']);
  assert.equal(f.krwFromThousand(view.hours[18].revenue.value), '468.6만원');
  assert.deepEqual([seg.quantity.unit, seg.quantity.label, view.tradingDate], ['MW', '구간 물량', '2026-11-27']);
  const bad = { ...raw, hourly_bids: raw.hourly_bids.map((h, i) => (i === 0 ? { ...h, hour: 24 } : h)) };
  assert.throws(() => a.adaptSegments(types.parseSegments(bad), FETCHED), types.ApiShapeError);
});

test('shape checkers fail fast with ApiShapeError instead of defaulting', () => {
  const forecast = fx('_api_v1_smp-forecast.json');
  const cases = [
    () => types.parseForecast({ ...forecast, q50: 'x' }),
    () => types.parseForecast({ ...forecast, q50: forecast.q50.slice(1) }),
    () => types.parseForecast({ ...forecast, model_used: null }),
    () => types.parseKpis([]),
    () => types.parseKpis({ ...fx('_api_v1_dashboard_kpis.json'), current_demand_mw: '664' }),
    () => types.parseMarketStatus({ current_time: '2026-11-26T06:03:57' }),
    () => types.parsePowerSupply({ current_hour: 6, data: [{ hour: '6' }] }),
    () => types.parseDayAhead(null),
    () => types.parseRealtimeStatus({ overall_status: 1 }),
  ];
  for (const run of cases) assert.throws(run, types.ApiShapeError);
  assert.throws(() => types.parseForecast({ ...forecast, q50: 'x' }), /smp-forecast q50/);
  const rs = types.parseRealtimeStatus(JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'f0-live', '_api_v1_realtime-status.json'), 'utf8')));
  assert.deepEqual([rs.overall_status, rs.smp_api.status, rs.smp_api.data_date_status, rs.weather_api.data_date_status], ['all_connected', 'connected', 'historical', null]);
  assert.equal(types.parseRealtimeStatus({ overall_status: 'partial' }).smp_api, null);
});

function fakeFetch(responder) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    return responder(url);
  };
  return { calls, impl };
}
const okJson = body => ({ ok: true, status: 200, json: async () => body });

test('client: GET only, no headers, allow-listed paths, method names as the old api.ts', async () => {
  const bodies = {
    [ENDPOINTS.dayAhead]: fx('_api_v1_smp_day-ahead.json'), [ENDPOINTS.forecast]: fx('_api_v1_smp-forecast.json'),
    [ENDPOINTS.modelInfo]: fx('_api_v1_model_info.json'), [ENDPOINTS.kpis]: fx('_api_v1_dashboard_kpis.json'),
    [ENDPOINTS.powerSupply]: fx('_api_v1_power-supply.json'), [ENDPOINTS.marketStatus]: fx('_api_v1_market-status.json'),
    [ENDPOINTS.segments]: fx(SEGMENTS_FILE), [ENDPOINTS.realtimeStatus]: { overall_status: 'all_connected' },
  };
  const { calls, impl } = fakeFetch(url => okJson(bodies[new URL(url, 'http://local.test').pathname]));
  const client = createClient({ baseUrl: '', fetchImpl: impl });
  await client.getDayAheadSMP();
  await client.getSMPForecast();
  await client.getModelInfo();
  await client.getDashboardKPIs();
  await client.getPowerSupply();
  await client.getMarketStatus();
  const seg = await client.getOptimizedSegments('moderate');
  await client.getRealtimeStatus();
  assert.equal(seg.hourly_bids.length, 24);
  assert.equal(calls.length, 8);
  for (const { url, init } of calls) {
    assert.ok(ALLOW_LIST.includes(url.split('?')[0]), url);
    assert.deepEqual(Object.keys(init).sort(), ['method', 'signal']);
    assert.equal(init.method, 'GET');
  }
  assert.equal(calls[6].url, '/api/v1/bidding/optimized-segments?capacity_mw=50&risk_level=moderate');
  assert.throws(() => client.getOptimizedSegments('yolo'), TypeError);
  const prefixed = createClient({ baseUrl: 'http://127.0.0.1:10104/', fetchImpl: impl });
  await prefixed.getMarketStatus();
  assert.equal(calls.at(-1).url, 'http://127.0.0.1:10104/api/v1/market-status');
});

test('client errors: 503 retryable, 404 not, network → ApiNetworkError, abort passes through, bad JSON → shape error', async () => {
  const status = code => createClient({ baseUrl: '', fetchImpl: async () => ({ ok: false, status: code, json: async () => ({}) }) });
  await assert.rejects(status(503).getMarketStatus(), e => e instanceof ApiHttpError && e.status === 503 && e.retryable === true);
  await assert.rejects(status(404).getMarketStatus(), e => e instanceof ApiHttpError && e.retryable === false);
  const down = createClient({ baseUrl: '', fetchImpl: async () => { throw new TypeError('Failed to fetch'); } });
  await assert.rejects(down.getMarketStatus(), e => e instanceof ApiNetworkError && e.retryable === true);
  const abort = Object.assign(new Error('aborted'), { name: 'AbortError' });
  const aborted = createClient({ baseUrl: '', fetchImpl: async () => { throw abort; } });
  await assert.rejects(aborted.getMarketStatus(), e => e === abort);
  const garbage = createClient({ baseUrl: '', fetchImpl: async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('NaN'); } }) });
  await assert.rejects(garbage.getMarketStatus(), types.ApiShapeError);
});

test('examples: the only home of invented values, frozen, 50 MW virtual plant and 1,716만원 sample statement', () => {
  assert.deepEqual([ex.EXAMPLE_PLANT.name.value, ex.EXAMPLE_PLANT.capacityMw.value, ex.EXAMPLE_PLANT.capacityMw.unit], ['예시 풍력발전소', 50, 'MW']);
  const st = ex.EXAMPLE_STATEMENT;
  const won = st.dayAheadMwh.value * 1000 * st.dayAheadPrice.value + st.realtimeDeviationMwh.value * 1000 * st.realtimePrice.value;
  assert.equal(f.krw(won), '1,716.0만원');
  assert.equal(ex.EXAMPLE_MERIT_BLOCKS.length, 6);
  assert.ok(Object.isFrozen(ex.EXAMPLE_PLANT.capacityMw) && Object.isFrozen(ex.EXAMPLE_MERIT_BLOCKS));
  assert.equal(ex.exampleFromCalculator(5, 'MW', '구간 물량').label, '구간 물량');
});

const GUIDE_OK = () => {
  const sec = title => ({ title, body: [`${title} 설명입니다.`] });
  return {
    version: 1,
    tour: ['하나', '둘', '셋', '넷', '다섯'].map(sec),
    glossary: Object.keys(s.GLOSSARY_TITLES).map(id => ({ id, short: '짧은 뜻입니다', body: ['풀이입니다.'] })),
    topics: Object.fromEntries(guide.TOPIC_IDS.map(id => [id, sec(`주제 ${id}`)])),
    cards: Object.fromEntries(guide.METHOD_CARD_IDS.map(id => [id, sec(`산출 방식 ${id}`)])),
  };
};

test('guide loader: shape (keys, kinds, lengths, no digits), relative URL, one request per base, retry after failure', async () => {
  assert.equal(guide.parseGuide(GUIDE_OK()).glossary.length, 12);
  const broken = [
    j => { j.version = 2; }, j => { j.tour.pop(); }, j => { j.glossary[0].id = 'nope'; }, j => { j.glossary[1].id = j.glossary[0].id; },
    j => { j.glossary[0].short = '가'.repeat(61); }, j => { delete j.cards['bid-merit']; }, j => { j.topics.extra = j.topics.about; },
    j => { j.tour[0].body = []; }, j => { j.tour[0].title = '마감 10시'; }, j => { j.cards['today-price'].body = ['']; },
  ];
  for (const mutate of broken) {
    const j = GUIDE_OK();
    mutate(j);
    assert.throws(() => guide.parseGuide(j), e => e instanceof guide.GuideLoadError && e.reason === 'shape', mutate.toString());
  }
  assert.equal(guide.guideUrl('/mobile/'), '/mobile/content/guide.ko.json');
  assert.equal(guide.guideUrl('/'), '/content/guide.ko.json');
  let fail = true;
  const urls = [];
  const gl = guide.createGuideLoader(async url => {
    urls.push(url);
    return fail ? { ok: false, status: 404, json: async () => ({}) } : okJson(GUIDE_OK());
  });
  await assert.rejects(gl.load('/mobile/'), e => e instanceof guide.GuideLoadError && e.reason === 'http');
  fail = false;
  const first = gl.load('/mobile/');
  assert.equal(gl.load('/mobile/'), first);
  assert.equal((await first).tour.length, 5);
  assert.deepEqual(urls, ['/mobile/content/guide.ko.json', '/mobile/content/guide.ko.json']);
});
