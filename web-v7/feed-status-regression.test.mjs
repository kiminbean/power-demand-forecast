import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const chrome = process.env.CHROME_BIN || ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find(existsSync);

for (const [status, label] of [['all_connected', '전체 연결'], ['partial', '일부 연결'], ['all_error', '연결 오류']]) {
  test(`monitored feed ${status} excludes the unavailable resources endpoint`, async () => {
    assert.ok(chrome, 'Chrome executable required');
    const server = await createServer({ base: '/v7/', server: { port: 0, host: '127.0.0.1' } });
    let browser;
    try {
      await server.listen();
      browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox'] });
      const page = await browser.newPage();
      await page.setRequestInterception(true);
      page.on('request', request => {
        const url = new URL(request.url());
        if (!url.pathname.startsWith('/api/')) {
          if (url.hostname === '127.0.0.1') request.continue(); else request.abort();
          return;
        }
        if (url.pathname === '/api/v1/resources') {
          request.respond({ status: 404, body: 'Not Found' });
          return;
        }
        const fixtures = {
          '/api/v1/realtime-status': { overall_status: status },
          '/api/v1/dashboard/kpis': { current_demand_mw: 600, supply_reserve_rate: 25, current_smp: 100, smp_change_pct: 0, grid_frequency: 60, renewable_ratio_pct: 20, weather: { temperature: 20, wind_speed: 3, humidity: 60, condition: '맑음' } },
          '/api/v1/smp-forecast': { q10: [], q50: [], q90: [], model_used: 'test', confidence: 1, created_at: '' },
          '/api/v1/power-supply': { current_hour: 12, data: [] },
        };
        request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fixtures[url.pathname] ?? {}) });
      });
      await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/v7/`, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => document.querySelector('.demo-notice')?.textContent.includes('발전소 현황'));
      await page.waitForFunction(expected => document.querySelector('header')?.innerText.includes(expected), {}, label);
      const header = await page.$eval('header', element => element.innerText);
      const notice = await page.$eval('.demo-notice', element => element.innerText);
      assert.match(header, new RegExp(`모니터링 피드: ${label}`));
      assert.match(notice, /API 연결 실패.*데모 데이터.*발전소 현황/);
    } finally {
      await browser?.close();
      await server.close();
    }
  });
}
