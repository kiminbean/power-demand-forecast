const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const source = p => fs.readFileSync(path.join(__dirname, '../src', p), 'utf8');
function loadPresentation() {
  const code = ts.transpileModule(source('utils/smpPresentation.ts'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports });
  return module.exports;
}
test('nullable and zero values are formatted without fallback figures', () => {
  const { formatNullableNumber } = loadPresentation();
  for (const value of [null, undefined, NaN, Infinity]) assert.equal(formatNullableNumber(value, 1), '—');
  assert.equal(formatNullableNumber(0, 1), '0.0');
  assert.equal(formatNullableNumber(42.6, 1), '42.6');
});
test('day-ahead requires a valid KPX trade date and 1–24 trade hour', () => {
  const { isAvailableDayAhead, describeDayAhead } = loadPresentation();
  const quote = { status: 'available', market_type: 'day_ahead', current_smp: 0, trade_day: '20260228', trade_hour: 24, data_source: 'KPX', source_url: null };
  assert.equal(isAvailableDayAhead(quote), true);
  assert.match(describeDayAhead(quote), /2026-02-28.*24.*KPX/);
  for (const bad of [
    { trade_day: '20260230' }, { trade_day: null }, { trade_day: '2026-13-01' },
    { trade_hour: 0 }, { trade_hour: 25 }, { trade_hour: 1.5 }, { trade_hour: null },
    { current_smp: null }, { current_smp: Infinity }, { status: 'no_data' }, { market_type: 'unknown' },
  ]) assert.equal(isAvailableDayAhead({ ...quote, ...bad }), false);
  assert.equal(isAvailableDayAhead({ ...quote, trade_day: '2026-02-28', trade_hour: 1 }), true);
  assert.match(describeDayAhead(null), /데이터 없음/);
});
test('SMP provenance never equates same date with live and shows date/source', () => {
  const { describeCurrentSmp } = loadPresentation();
  assert.match(describeCurrentSmp({ current_smp: null, data_date_status: 'no_data', trade_day: null, data_source: null }), /없/);
  const historical = describeCurrentSmp({ current_smp: 100, data_date_status: 'historical', trade_day: '2025-01-01', data_source: 'KPX' });
  assert.match(historical, /2025-01-01/); assert.match(historical, /KPX/); assert.match(historical, /과거/);
  for (const status of ['current_date', 'unknown', 'future']) {
    const label = describeCurrentSmp({ current_smp: 0, data_date_status: status, trade_day: '2026-01-02', data_source: 'prediction' });
    assert.match(label, /prediction/); assert.doesNotMatch(label, /실시간|현재 SMP/);
  }
  assert.doesNotMatch(describeCurrentSmp({ current_smp: 4 }), /실시간|현재 SMP/);
});
test('dated latest SMP is separate from the nullable current price and retains trade hour', () => {
  const { describeCurrentSmp, publishedSmpPrice } = loadPresentation();
  const data = { current_smp: null, data_date_status: 'historical', latest_smp: { price: 0, trade_day: '20260102', trade_hour: 24, market: 'unknown', data_source: 'KPX', region: 'jeju' } };
  assert.equal(publishedSmpPrice(data), 0);
  assert.match(describeCurrentSmp(data), /20260102.*24h.*시장 미상.*KPX/);
  assert.equal(publishedSmpPrice({ current_smp: null, latest_smp: null }), null);
  assert.equal(publishedSmpPrice({ current_smp: null, latest_smp: { ...data.latest_smp, price: Infinity } }), null);
});
test('current SMP consumers handle null and distinguish forecasts and demo', () => {
  const api = source('services/api.ts');
  assert.match(api, /current_smp: number \| null/);
  assert.match(api, /comparison: \{[\s\S]*?\} \| null/);
  assert.match(api, /smp_change_pct: number \| null/);
  assert.match(api, /daily_revenue_million: number \| null/);
  const dashboard = source('screens/DashboardScreen.tsx');
  const execo = source('screens/ExecoDashboardScreen.tsx');
  assert.doesNotMatch(dashboard, /current_smp\s*\|\|\s*95|daily_revenue_million\s*\|\|\s*38/);
  assert.doesNotMatch(execo, /current_smp\s*\?\?\s*95|title="현재 SMP"|실시간 데이터/);
  const forecast = source('screens/SMPForecastScreen.tsx');
  assert.doesNotMatch(forecast, /현재 SMP|generateMockSMPData/);
  assert.match(forecast, /예측 SMP/);
  const chat = source('screens/alan/AlanChatScreen.tsx');
  assert.doesNotMatch(chat, /currentSMP\.current_smp\.toFixed|kpis\.current_smp\.toFixed|kpis\.daily_revenue_million\.toFixed|현재 SMP:/);
  assert.match(chat, /describeCurrentSmp/);
  assert.doesNotMatch(source('services/alan/alanApi.ts'), /현재 제주 SMP는 71/);
  assert.doesNotMatch(source('screens/BiddingScreen.tsx'), /현재 SMP/);
});
test('the active mobile SMP screen displays an independently fetched day-ahead quote', () => {
  const app = fs.readFileSync(path.join(__dirname, '../App.tsx'), 'utf8');
  assert.match(app, /case 'smp':\s*return <SMPForecastScreen/);
  const screen = source('screens/SMPForecastScreen.tsx');
  assert.match(screen, /getDayAheadSMP\(\)/);
  assert.match(screen, /describeDayAhead\(dayAheadSmp\)/);
  assert.match(screen, /isAvailableDayAhead\(dayAheadSmp\)/);
});
