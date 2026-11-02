import { test, after, before } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createServer } from 'vite';
import puppeteer from 'puppeteer-core';

const executablePath = process.env.CHROME_BIN || [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].find(existsSync);
let server;
let browser;
let baseURL;

before(async () => {
  assert.ok(executablePath, 'Set CHROME_BIN to a Chrome/Chromium executable');
  server = await createServer({ base: '/v7/', server: { port: 0, host: '127.0.0.1' } });
  await server.listen();
  baseURL = `http://127.0.0.1:${server.httpServer.address().port}/v7/`;
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox'] });
});

after(async () => {
  await browser?.close();
  await server?.close();
});

async function openDashboard(width) {
  const page = await browser.newPage();
  await page.setViewport({ width, height: 844, isMobile: width < 600, hasTouch: width < 600 });
  await page.setRequestInterception(true);
  page.on('request', request => {
    const url = new URL(request.url());
    if (url.pathname.startsWith('/api/') || url.hostname !== '127.0.0.1') request.abort();
    else request.continue();
  });
  await page.goto(baseURL, { waitUntil: 'networkidle0' });
  await page.waitForSelector('main');
  return page;
}

for (const width of [320, 390, 768, 1280, 1440]) {
  test(`dashboard fits viewport at ${width}px`, async () => {
    const page = await openDashboard(width);
    try {
      const dimensions = await page.evaluate(() => ({
        viewport: document.documentElement.clientWidth,
        document: document.documentElement.scrollWidth,
        body: document.body.scrollWidth,
      }));
      assert.ok(dimensions.document <= dimensions.viewport + 1 && dimensions.body <= dimensions.viewport + 1, JSON.stringify(dimensions));
    } finally {
      await page.close();
    }
  });
}

test('plant details work by touch and keyboard without overflowing the page', async () => {
  const page = await openDashboard(390);
  try {
    const marker = await page.$('button[aria-label="대정풍력 상세 정보"]');
    assert.ok(marker, 'plant marker should be an accessible button');
    await marker.tap();
    assert.match(await page.$eval('.plant-details', element => element.textContent), /대정풍력.*45.*MW/s);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), true);
    await marker.focus();
    await page.keyboard.press('Enter');
    assert.doesNotMatch(await page.$eval('.plant-details', element => element.textContent), /대정풍력/);
    // Nearby markers overlap on a small map; the picker must reach both plants.
    await page.select('.plant-picker select', 'wind_003');
    assert.match(await page.$eval('.plant-details', element => element.textContent), /가시리풍력.*30.*MW/s);
    await page.select('.plant-picker select', 'ess_004');
    assert.match(await page.$eval('.plant-details', element => element.textContent), /가시리풍력 연계 ESS.*15.*MW/s);
    await page.select('.plant-picker select', 'wind_001');
    assert.match(await page.$eval('.plant-details', element => element.textContent), /한경풍력.*21.*MW/s);
  } finally {
    await page.close();
  }
});

test('API failure is disclosed and assets use the deployment prefix', async () => {
  const page = await openDashboard(390);
  try {
    assert.match(await page.$eval('.demo-notice', element => element.textContent), /데모 데이터.*현재 지표.*발전소 현황.*전력수급 차트/s);
    const images = await page.$$eval('img', elements => elements.map(element => ({ src: new URL(element.src).pathname, loaded: element.complete && element.naturalWidth > 0 })));
    for (const name of ['jeju-map.png', 'logo-light.png', 'check-icon.svg']) {
      assert.ok(images.some(image => image.src === `/v7/${name}` && image.loaded), `${name}: ${JSON.stringify(images)}`);
    }
  } finally {
    await page.close();
  }
});
