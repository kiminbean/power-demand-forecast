import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const chrome = process.env.CHROME_BIN || ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find(existsSync);
let server, browser, baseURL;
const forecasts = { q10: Array(24).fill(180), q50: Array(24).fill(200), q90: Array(24).fill(220), model_used: 'test', confidence: 1, created_at: '2026-11-03T00:00:00Z' };
const kpis = { total_capacity_mw: 100, current_output_mw: 50, utilization_pct: 50, current_demand_mw: 600, supply_reserve_rate: 25, current_smp: 777, smp_change_pct: 8, daily_revenue_million: null, revenue_change_pct: 0, grid_frequency: 60, renewable_ratio_pct: 20, weather: { temperature: 20, wind_speed: 3, humidity: 60, condition: '맑음' }, data_source: 'KPI' };
const fixtures = {
  '/api/v1/dashboard/kpis': kpis,
  '/api/v1/resources': [],
  '/api/v1/smp-forecast': forecasts,
  '/api/v1/power-supply': { current_hour: 12, data: [], data_source: 'test' },
  '/api/v1/realtime-status': { overall_status: 'partial' },
  '/api/v1/market-status': { dam: { status: 'open' }, rtm: { status: 'active' } },
  '/api/v1/model/info': { type: 'test', version: '1', mape: 1 },
};
const sample = (status, value = 123.4) => ({ current_smp: null, hour: 12, region: 'jeju', comparison: null, data_date_status: status, trade_day: '20261102', data_source: 'EPSIS test', latest_smp: value === null ? null : { price: value, trade_day: '20261102', trade_hour: 4, data_source: 'EPSIS test', market: 'unknown', region: 'jeju' } });

before(async () => {
  assert.ok(chrome, 'Chrome executable required');
  server = await createServer({ base: '/v7/', server: { port: 0, host: '127.0.0.1' } });
  await server.listen();
  baseURL = `http://127.0.0.1:${server.httpServer.address().port}/v7/`;
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox'] });
});
after(async () => { await browser?.close(); await server?.close(); });

async function setup(component, initial) {
  const page = await browser.newPage();
  let current = initial;
  let dayAhead = { status: 'available', current_smp: 145.6, trade_day: '20261103', trade_hour: 14, market_type: 'day_ahead', data_source: 'EPSIS day ahead', source_url: 'https://example.com/source' };
  let mode = 'success';
  await page.evaluateOnNewDocument(() => {
    const original = window.setInterval;
    window.setInterval = (fn, delay, ...args) => {
      if (delay === 60000 || delay === 300000) window.smpRefresh = () => fn(...args);
      return original(fn, delay, ...args);
    };
  });
  await page.setRequestInterception(true);
  page.on('request', request => {
    const url = new URL(request.url());
    if (!url.pathname.startsWith('/api/')) {
      if (url.hostname === '127.0.0.1') request.continue(); else request.abort();
      return;
    }
    if (mode === 'fail' && ['/api/v1/smp/current', '/api/v1/smp/day-ahead', '/api/v1/dashboard/kpis', '/api/v1/smp-forecast'].includes(url.pathname)) {
      request.respond({ status: 503, body: 'unavailable' });
      return;
    }
    const response = url.pathname === '/api/v1/smp/current' ? current : url.pathname === '/api/v1/smp/day-ahead' ? dayAhead : fixtures[url.pathname] ?? {};
    request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(response) });
  });
  await page.goto(baseURL, { waitUntil: 'networkidle0' });
  if (component !== 'ExecoDashboard') {
    await page.evaluate(async componentName => {
      const { default: React } = await import('/v7/node_modules/.vite/deps/react.js');
      const { default: ReactDOM } = await import('/v7/node_modules/.vite/deps/react-dom_client.js');
      const { ThemeProvider } = await import('/v7/src/contexts/ThemeContext.tsx');
      const { default: Component } = await import(`/v7/src/pages/${componentName}.tsx`);
      const target = document.createElement('div');
      document.body.replaceChildren(target);
      ReactDOM.createRoot(target).render(React.createElement(ThemeProvider, null, React.createElement(Component)));
    }, component);
  }
  await page.waitForFunction(() => document.body.innerText.includes('SMP'));
  return { page, setMode: next => { mode = next; }, setCurrent: next => { current = next; }, setDayAhead: next => { dayAhead = next; } };
}
const body = page => page.$eval('body', element => element.innerText);

for (const component of ['ExecoDashboard', 'Dashboard', 'SMPPrediction']) {
  test(`${component}: date provenance and nullable comparisons are explicit`, async () => {
    const { page, setCurrent } = await setup(component, sample('historical'));
    try {
      await page.waitForFunction(() => document.body.innerText.includes('123.4'));
      let text = await body(page);
      assert.match(text, /과거/);
      assert.match(text, /20261102/);
      assert.match(text, /EPSIS test/);
      assert.match(text, /공표 SMP/);
      assert.match(text, /제주/);
      assert.match(text, /04h/);
      assert.doesNotMatch(text, /현재 SMP \(실시간\)/);
      assert.doesNotMatch(text, /777\.0/);
      setCurrent(sample('current_date', 0));
      await page.evaluate(() => window.smpRefresh());
      await page.waitForFunction(() => document.body.innerText.includes('0.0') && document.body.innerText.includes('당일'));
      text = await body(page);
      assert.match(text, /실시간.*확인.*불가|실시간 여부.*확인/);
    } finally { await page.close(); }
  });
  test(`${component}: null and failed refresh never fall back to forecasts or mock SMP`, async () => {
    const { page, setCurrent, setMode } = await setup(component, sample('future'));
    try {
      await page.waitForFunction(() => document.body.innerText.includes('123.4'));
      assert.match(await body(page), /미래/);
      setCurrent(sample('unknown', 124.5));
      await page.evaluate(() => window.smpRefresh());
      await page.waitForFunction(() => document.body.innerText.includes('124.5'));
      assert.match(await body(page), /날짜.*확인.*불가/);
      setCurrent(sample('no_data', null));
      await page.evaluate(() => window.smpRefresh());
      await page.waitForFunction(() => !document.body.innerText.includes('124.5') && document.body.innerText.includes('SMP 데이터 없음'));
      setCurrent(sample('historical'));
      await page.evaluate(() => window.smpRefresh());
      await page.waitForFunction(() => document.body.innerText.includes('123.4'));
      setMode('fail');
      await page.evaluate(() => window.smpRefresh());
      await page.waitForFunction(() => !document.body.innerText.includes('123.4') && document.body.innerText.includes('SMP 데이터 없음'));
      const text = await body(page);
      assert.doesNotMatch(text, /777\.0|114\.8|102\.5/);
      if (component !== 'ExecoDashboard') assert.match(text, /데모.*예측|예측.*데모/);
    } finally { await page.close(); }
  });
}

test('day-ahead card is separate, sourced and clears on no data or failure', async () => {
  const { page, setDayAhead, setMode } = await setup('SMPPrediction', sample('historical'));
  try {
    await page.waitForFunction(() => document.querySelector('[data-testid="day-ahead-smp"]')?.innerText.includes('145.6'));
    const card = () => page.$eval('[data-testid="day-ahead-smp"]', element => element.innerText);
    let text = await card();
    assert.match(text, /제주 하루전시장 SMP.*Day-ahead/);
    assert.match(text, /20261103/);
    assert.match(text, /14h/);
    assert.match(text, /EPSIS day ahead/);
    assert.doesNotMatch(text, /실시간|현재/);
    assert.equal(await page.$eval('[data-testid="day-ahead-smp"] a', element => element.href), 'https://example.com/source');
    setDayAhead({ status: 'no_data', current_smp: null, trade_day: null, trade_hour: null, market_type: 'day_ahead', data_source: null, source_url: null });
    await page.evaluate(() => window.smpRefresh());
    await page.waitForFunction(() => document.querySelector('[data-testid="day-ahead-smp"]')?.innerText.includes('SMP 데이터 없음'));
    assert.doesNotMatch(await card(), /145\.6/);
    setDayAhead({ status: 'available', current_smp: 145.6, trade_day: '20261103', trade_hour: 14, market_type: 'day_ahead', data_source: 'EPSIS', source_url: null });
    await page.evaluate(() => window.smpRefresh());
    await page.waitForFunction(() => document.querySelector('[data-testid="day-ahead-smp"]')?.innerText.includes('145.6'));
    setMode('fail');
    await page.evaluate(() => window.smpRefresh());
    await page.waitForFunction(() => document.querySelector('[data-testid="day-ahead-smp"]')?.innerText.includes('SMP 데이터 없음'));
    assert.doesNotMatch(await card(), /145\.6/);
  } finally { await page.close(); }
});
