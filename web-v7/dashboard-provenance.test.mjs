import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const chrome = process.env.CHROME_BIN || ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find(existsSync);
let server, browser, baseURL;
const kpis = {
  current_demand_mw: 600, supply_reserve_rate: 25, current_smp: 100, smp_change_pct: 0,
  grid_frequency: 60, renewable_ratio_pct: 20,
  weather: { temperature: 20, wind_speed: 3, humidity: 60, condition: '맑음' },
  data_source: 'EPSIS SMP (파일) + EPSIS Demand (파일) + KMA Weather (파일) + Renewable ML: model',
};
const resources = [{ id: 'solar', name: 'solar', type: 'solar', capacity: 100, current_output: 30, utilization: 30, status: 'active', location: '제주' }];
const fixtures = {
  '/api/v1/dashboard/kpis': kpis,
  '/api/v1/resources': resources,
  '/api/v1/smp-forecast': { q10: [], q50: [], q90: [], model_used: 'test', confidence: 1, created_at: '' },
  '/api/v1/power-supply': { current_hour: 12, data: [], data_source: 'Simulated Pattern + Renewable ML' },
  '/api/v1/realtime-status': { overall_status: 'partial' },
};

before(async () => {
  assert.ok(chrome, 'Chrome executable required');
  server = await createServer({ base: '/v7/', server: { port: 0, host: '127.0.0.1' } });
  await server.listen();
  baseURL = `http://127.0.0.1:${server.httpServer.address().port}/v7/`;
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox'] });
});
after(async () => { await browser?.close(); await server?.close(); });

async function setup(initialMode = 'success', source = kpis.data_source) {
  const page = await browser.newPage();
  let mode = initialMode;
  const pending = [];
  await page.evaluateOnNewDocument(() => {
    const original = window.setInterval;
    window.setInterval = (fn, delay, ...args) => {
      if (delay === 60000) window.dashboardRefresh = () => fn(...args);
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
    const controlled = ['/api/v1/dashboard/kpis', '/api/v1/resources'].includes(url.pathname);
    if (controlled && mode === 'hold') { pending.push(request); return; }
    if (controlled && mode === 'fail') { request.respond({ status: 503, body: 'unavailable' }); return; }
    const data = url.pathname === '/api/v1/dashboard/kpis' ? { ...kpis, data_source: source } : fixtures[url.pathname] ?? {};
    request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(data) });
  });
  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('main');
  return { page, setMode: next => { mode = next; }, pending };
}
const body = page => page.$eval('body', element => element.innerText);
const release = (pending, success) => Promise.all(pending.splice(0).map(request => {
  const path = new URL(request.url()).pathname;
  return request.respond(success ? { status: 200, contentType: 'application/json', body: JSON.stringify(fixtures[path]) } : { status: 503, body: 'unavailable' });
}));

test('initial loading discloses placeholders, then response provenance and simulated fields', async () => {
  const { page, pending } = await setup('hold');
  try {
    await page.waitForFunction(() => document.body.innerText.includes('데이터 확인 중'));
    let text = await body(page);
    assert.match(text, /로딩 중.*현재 지표.*발전소 현황/s);
    assert.doesNotMatch(text, /EPSIS 실시간/);
    await page.waitForFunction(() => window.dashboardRefresh !== undefined);
    while (pending.length < 2) await new Promise(resolve => setTimeout(resolve, 20));
    await release(pending, true);
    await page.waitForFunction(() => document.body.innerText.includes('EPSIS SMP (파일)'));
    text = await body(page);
    assert.match(text, /EPSIS Demand \(파일\)/);
    assert.match(text, /주파수.*시뮬레이션/);
    assert.match(text, /재생에너지.*예측/);
    assert.match(text, /발전소.*시뮬레이션/s);
    assert.match(text, /Simulated Pattern/);
    assert.doesNotMatch(text, /EPSIS 실시간/);
  } finally { await page.close(); }
});

test('unknown KPI source is not represented as live', async () => {
  const { page } = await setup('success', null);
  try {
    await page.waitForFunction(() => document.body.innerText.includes('출처 정보 없음'));
    const text = await body(page);
    assert.doesNotMatch(text, /EPSIS 실시간/);
  } finally { await page.close(); }
});

test('failed refresh replaces prior values with disclosed demo data until recovery', async () => {
  const { page, setMode, pending } = await setup();
  try {
    await page.waitForFunction(() => document.body.innerText.includes('EPSIS SMP (파일)'));
    setMode('fail');
    await page.evaluate(() => window.dashboardRefresh());
    await page.waitForFunction(() => document.querySelector('.demo-notice')?.textContent.includes('발전소 현황'));
    let text = await body(page);
    assert.match(text, /데모 데이터.*현재 지표.*발전소 현황/s);
    assert.doesNotMatch(text, /KPI 출처: EPSIS SMP/);
    setMode('hold');
    await page.evaluate(() => window.dashboardRefresh());
    while (pending.length < 2) await new Promise(resolve => setTimeout(resolve, 20));
    text = await body(page);
    assert.match(text, /데모 데이터.*현재 지표.*발전소 현황/s);
    await release(pending, true);
    await page.waitForFunction(() => document.body.innerText.includes('KPI 출처: EPSIS SMP'));
    assert.doesNotMatch(await body(page), /API 연결 실패/);
  } finally { await page.close(); }
});

test('failure disclosure remains during a pending refresh and clears on recovery', async () => {
  const { page, setMode, pending } = await setup('fail');
  try {
    await page.waitForFunction(() => document.querySelector('.demo-notice')?.textContent.includes('발전소 현황'));
    let text = await body(page);
    assert.match(text, /데모 데이터.*현재 지표.*발전소 현황/s);
    setMode('hold');
    await page.evaluate(() => window.dashboardRefresh());
    while (pending.length < 2) await new Promise(resolve => setTimeout(resolve, 20));
    text = await body(page);
    assert.match(text, /데모 데이터.*현재 지표.*발전소 현황/s);
    await release(pending, true);
    await page.waitForFunction(() => !document.querySelector('.demo-notice')?.textContent.includes('API 연결 실패'));
    text = await body(page);
    assert.match(text, /EPSIS SMP \(파일\)/);
    assert.doesNotMatch(text, /API 연결 실패/);
  } finally { await page.close(); }
});
