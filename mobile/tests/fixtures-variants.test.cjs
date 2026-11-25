// F0 픽스처 검증: round 0 스냅숏이 온전한지, 상태 변형이 모두 풀리고 이름대로의 조건을 실제로 만드는지 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { FIXTURES, loadManifest, resolveVariant, kstEpochMs } = require('./helpers/fixtures.cjs');

const ROUND0 = path.join(FIXTURES, 'round0');

test('round 0 snapshot: 14 responses with the recorded HTTP status and snapshot time', () => {
  assert.equal(fs.readFileSync(path.join(ROUND0, 'snapshot_time.txt'), 'utf8').trim(), '2026-11-26 06:03:57 KST');
  const lines = fs.readFileSync(path.join(ROUND0, 'status.txt'), 'utf8').trim().split('\n');
  assert.equal(lines.length, 14);
  for (const line of lines) {
    const m = /^(\d{3}) (\S+) -> (\S+) \(\s*(\d+) B\)$/.exec(line);
    assert.ok(m, line);
    const [, status, , file, bytes] = m;
    const body = fs.readFileSync(path.join(ROUND0, file));
    assert.equal(body.length, Number(bytes), file);
    if (status === '200') assert.equal(typeof JSON.parse(body.toString('utf8')), 'object', file);
    else assert.match(body.toString('utf8'), /Error code: 404/, file);
  }
});

test('F0 realtime-status fixture has the filtered public shape', () => {
  const body = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'f0-live', '_api_v1_realtime-status.json'), 'utf8'));
  assert.deepStrictEqual(Object.keys(body).sort(), ['overall_status', 'power_supply_api', 'smp_api', 'timestamp', 'weather_api']);
  assert.ok(['all_connected', 'partial', 'disconnected'].includes(body.overall_status));
  for (const name of ['smp_api', 'power_supply_api', 'weather_api']) {
    assert.ok(['connected', 'no_data', 'error', 'unknown'].includes(body[name].status), name);
  }
  assert.doesNotThrow(() => kstEpochMs(body.timestamp));
});

test('every variant resolves: unique id, known keys, existing JSON pointers', () => {
  const manifest = loadManifest();
  const ids = manifest.variants.map(v => v.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const id of ids) {
    const resolved = resolveVariant(id, manifest);
    assert.ok(['today', 'bidding', 'guide'].includes(resolved.variant.screen), id);
    assert.ok(Number.isFinite(resolved.clockMs), id);
  }
  // 11.5절 상태 목록에서 픽스처가 필요한 묶음이 모두 있다.
  for (const sweep of ['T9', 'B6', 'A', 'A6', 'G10']) assert.ok(manifest.variants.some(v => v.sweep === sweep), sweep);
});

const at = (id, route) => resolveVariant(id).responses[route];
const kpis = id => at(id, '/api/v1/dashboard/kpis').body;
const LIVE_WEATHER = /KMA Weather Crawler|KMA Real-time API|기상청 \(실시간\)/;

test('provenance variants carry the data_source each 7.4 row needs', () => {
  assert.equal(kpis('today-base').data_source.includes('Jeju Crawler'), true);
  assert.equal(kpis('today-demand-src-2').data_source, null);
  assert.match(kpis('today-demand-src-4').data_source, /EPSIS Demand/);
  const national = kpis('today-demand-src-5').data_source;
  assert.ok(national.includes('KPX') && !/Jeju Crawler|EPSIS Demand/.test(national));
  assert.match(kpis('today-demand-src-6').data_source, /Simulated/);
  assert.ok(/KMA Weather \(파일\)/.test(kpis('today-weather-file').data_source) && !LIVE_WEATHER.test(kpis('today-weather-file').data_source));
  const fallback = kpis('today-weather-default');
  assert.ok(!/KMA Weather \(파일\)/.test(fallback.data_source) && !LIVE_WEATHER.test(fallback.data_source));
  assert.deepStrictEqual(fallback.weather, { temperature: 5.0, wind_speed: 3.0, humidity: 60.0, condition: '맑음' });
  assert.equal(kpis('today-weather-none'), null);
});

test('supply pair, forecast and price variants create their conditions', () => {
  const pairGap = id => {
    const supply = at(id, '/api/v1/power-supply').body;
    const row = supply.data.find(r => r.hour === supply.current_hour);
    return Math.abs(row.demand - kpis(id).current_demand_mw);
  };
  assert.ok(pairGap('today-base') <= 1);
  assert.ok(pairGap('today-supply-pair-mismatch') > 1);
  assert.equal(at('today-price-no-data', '/api/v1/smp/day-ahead').body.status, 'no_data');
  const zero = at('today-forecast-zero-12-14h', '/api/v1/smp-forecast').body;
  assert.deepStrictEqual([11, 12, 13].map(i => zero.q50[i]), [0, 0, 0]);
  assert.equal(at('today-forecast-date-yesterday', '/api/v1/smp-forecast').body.forecast_date, '2026-11-25');
  const basisGaps = id => {
    const forecast = at(id, '/api/v1/smp-forecast').body;
    return at(id, '/api/v1/bidding/optimized-segments').body.hourly_bids.map((h, i) => Math.abs(h.smp_forecast.q50 - forecast.q50[i]));
  };
  assert.ok(basisGaps('bidding-price-basis-present').every(gap => gap <= 0.05));
  assert.ok(basisGaps('bidding-price-basis-absent').every(gap => gap > 0.05));
});

test('market and clock variants match the moments the spec names', () => {
  const market = id => at(id, '/api/v1/market-status').body;
  const skewSeconds = id => (resolveVariant(id).clockMs - kstEpochMs(market(id).current_time)) / 1000;
  assert.equal(skewSeconds('today-base'), 0);
  assert.equal(Math.round(skewSeconds('today-clock-skew-plus3m')), 180);
  assert.equal(Math.round(skewSeconds('today-clock-skew-plus13h')), 13 * 3600);
  assert.deepStrictEqual([market('bidding-imminent-0920').dam.status, market('bidding-imminent-0920').dam.hours_remaining], ['open', 0.7]);
  assert.equal(market('bidding-closed-1001').dam.status, 'closed');
  assert.equal(skewSeconds('bidding-closed-1001'), 0);
  assert.ok(Math.abs(market('bidding-mismatch-hours').dam.hours_remaining - 3.94) > 0.2);
  assert.deepStrictEqual(at('bidding-calc-503-twice', '/api/v1/bidding/optimized-segments').statusSequence, [503, 503]);
  assert.equal(at('guide-content-404', 'content/guide.ko.json').status, 404);
});
