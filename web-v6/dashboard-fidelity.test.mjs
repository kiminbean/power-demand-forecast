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
         export const useAutoRefresh=()=>{};` }));
  }}],
});
const { default: Dashboard } = require(join(dir, 'dashboard.cjs'));
const state = (data, error = null) => ({ data, error, refetch: () => {} });
function render(kpis, resources, error = null) {
  globalThis.dashboardFixtures = { kpis: state(kpis, error), resources: state(resources, error), forecast: state({ q50: Array(24).fill(999), confidence: 0.9 }), market: state(null, error) };
  return renderToStaticMarkup(React.createElement(Dashboard));
}
test('KPI SMP is not substituted with a forecast and backend provenance is visible', () => {
  const html = render({ current_smp: 123.4, data_source: 'EPSIS SMP (파일)' }, []);
  assert.match(html, /123\.4/); assert.doesNotMatch(html, /999\.0/); assert.match(html, /EPSIS SMP/);
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
