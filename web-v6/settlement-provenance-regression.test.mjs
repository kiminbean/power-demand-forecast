import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dir = await mkdtemp(join(process.env.TMPDIR, 'settlement-provenance-'));
await build({
  entryPoints: ['src/pages/Settlement.tsx'], bundle: true, platform: 'node', format: 'cjs',
  outfile: join(dir, 'settlement.cjs'),
  plugins: [{ name: 'fixtures', setup(b) {
    b.onResolve({ filter: /^(react|react-dom|lucide-react|clsx)$/ }, args => ({ path: require.resolve(args.path), external: true }));
    b.onResolve({ filter: /recharts/ }, () => ({ path: 'charts', namespace: 'fixture' }));
    b.onResolve({ filter: /hooks\/useApi$/ }, () => ({ path: 'hooks', namespace: 'fixture' }));
    b.onResolve({ filter: /contexts\/ThemeContext$/ }, () => ({ path: 'theme', namespace: 'fixture' }));
    b.onLoad({ filter: /.*/, namespace: 'fixture' }, ({ path }) => ({ loader: 'js', contents: path === 'hooks'
      ? 'export const useSettlements=()=>({data: [{revenue: 999}]}); export const useSettlementSummary=()=>({data: {revenue: 999}});'
      : path === 'theme' ? 'export const useTheme=()=>({isDark:false});'
      : 'export const XAxis=()=>null,YAxis=()=>null,CartesianGrid=()=>null,Tooltip=()=>null,ResponsiveContainer=()=>null,Legend=()=>null,BarChart=()=>null,Bar=()=>null,ComposedChart=()=>null,Line=()=>null,Area=()=>null,Cell=()=>null;' }));
  }}],
});
const { default: Settlement } = require(join(dir, 'settlement.cjs'));
test('settlement page discloses its generated figures even when API data is available', () => {
  const html = renderToStaticMarkup(React.createElement(Settlement));
  assert.match(html, /데모 데이터/);
  assert.match(html, /실제 정산 데이터가 아닙니다/);
});
after(async () => { await rm(dir, { recursive: true, force: true }); });
