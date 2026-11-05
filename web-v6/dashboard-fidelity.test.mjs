import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';

import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

const dir = await mkdtemp(join(process.env.TMPDIR, 'dashboard-test-'));
await build({
  entryPoints: ['src/pages/Dashboard.tsx'], bundle: true, platform: 'node', format: 'cjs',
  outfile: join(dir, 'dashboard.cjs'),
  plugins: [{ name: 'fixtures', setup(b) {
    b.onResolve({ filter: /^(react|react-dom|lucide-react|clsx)$/ }, args => ({ path: require.resolve(args.path), external: true }));
    b.onResolve({ filter: /hooks\/useApi$/ }, () => ({ path: 'hooks', namespace: 'fixture' }));
    b.onResolve({ filter: /components\/Charts\// }, () => ({ path: 'chart', namespace: 'fixture' }));
    b.onLoad({ filter: /.*/, namespace: 'fixture' }, ({ path }) => ({ loader: 'js', contents: path === 'chart'
      ? 'export default function Chart(){return null}'
      : `export const useDashboardKPIs=()=>globalThis.dashboardFixtures.kpis;
         export const useSMPForecast=()=>globalThis.dashboardFixtures.forecast;
         export const useResources=()=>globalThis.dashboardFixtures.resources;
         export const useMarketStatus=()=>globalThis.dashboardFixtures.market;
         export const useDayAheadSMP=()=>globalThis.dashboardFixtures.dayAhead;
         export const usePowerSupply=()=>globalThis.dashboardFixtures.powerSupply;
         export const useModelInfo=()=>globalThis.dashboardFixtures.modelInfo;
         export const useAutoRefresh=()=>{};` }));
  }}],
});
const { default: Dashboard } = require(join(dir, 'dashboard.cjs'));
const state = (data, error = null) => ({ data, error, refetch: () => {} });
const dayAheadQuote = { status: 'available', current_smp: 123.4, trade_day: '20261105', trade_hour: 17, market_type: 'day_ahead', data_source: 'KPX 제주 하루전시장', source_url: null };
function render(kpis, resources, error = null, dayAhead = dayAheadQuote, modelInfo = { status: 'active (성능 재검증 전)', version: 'v3.2', mape: null }) {
  globalThis.dashboardFixtures = {
    kpis: state(kpis, error), resources: state(resources, error),
    forecast: state({ q50: Array(24).fill(999), confidence: null, forecast_date: '2026-11-05' }),
    market: state(null, error), dayAhead: state(dayAhead, error), powerSupply: state(null, error), modelInfo: state(modelInfo),
  };
  return renderToStaticMarkup(React.createElement(Dashboard));
}
test('SMP card shows the day-ahead quote, never a forecast, and provenance is visible', () => {
  const html = render({ current_smp: null, data_source: 'KPX Jeju Crawler' }, []);
  assert.match(html, /제주 하루전시장 SMP \(17h\)[\s\S]*123\.4/);
  // 예측값은 "예측 최고 SMP" 알림에서만 한 번 나온다.
  assert.equal(html.match(/999\.0/g)?.length, 1);
  assert.match(html, /예측 최고 SMP[\s\S]*999\.0/);
  assert.match(html, /KPX Jeju Crawler/); assert.match(html, /KPX 제주 하루전시장/);
});
test('no hardcoded reserve, revenue, accuracy or model version is shown', () => {
  const html = render({ current_smp: null, supply_reserve_rate: 135.6, utilization_pct: 61.2, daily_revenue_million: null }, []);
  assert.match(html, /현재 예비율 135\.6%/); assert.match(html, /61\.2%/);
  assert.match(html, /산출 불가/); assert.match(html, /재검증 전/); assert.match(html, /v3\.2 사용 중/);
  for (const fake of [/127%/, /245\.8/, /94\.5/, /v3\.1/, /14:00-16:00/, /123\.3/]) assert.doesNotMatch(html, fake);
});
test('missing day-ahead quote is shown as absent, not substituted', () => {
  const html = render({ current_smp: 888.8 }, [], null, { status: 'no_data', current_smp: null, trade_day: null, trade_hour: null, market_type: 'day_ahead', data_source: null, source_url: null });
  assert.doesNotMatch(html, /888\.8/); assert.match(html, /제주 하루전시장 SMP[\s\S]*—/);
});
test('solar simulation is labeled and absent data is not fabricated', () => {
  const html = render({ current_smp: 0 }, [{ type: 'solar', current_output: 0 }]);
  assert.match(html, /태양광 발전 \(시뮬레이션\)/); assert.match(html, /0\.0/);
  const missing = render(null, null, 'API unavailable, showing demo data');
  assert.doesNotMatch(missing, /45\.2/); assert.match(missing, /API/); assert.match(missing, /데모/);
});
test('render time is not presented as an observation timestamp', () => {
  const html = render(null, null);
  assert.doesNotMatch(html, /업데이트:/); assert.match(html, /관측 시각.*제공되지/);
});
after(async () => { await rm(dir, { recursive: true, force: true }); });
