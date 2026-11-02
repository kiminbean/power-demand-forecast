import { test, before, after } from 'node:test';
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

async function openDashboard(width = 390) {
  const page = await browser.newPage();
  await page.setViewport({ width, height: 844, isMobile: true, hasTouch: true });
  await page.setRequestInterception(true);
  page.on('request', request => {
    const url = new URL(request.url());
    if (url.pathname.startsWith('/api/') || url.hostname !== '127.0.0.1') request.abort();
    else request.continue();
  });
  await page.goto(baseURL, { waitUntil: 'networkidle0' });
  await page.waitForSelector('.plant-map');
  return page;
}

for (const width of [320, 390]) {
  test(`touch selects both overlapping plants at ${width}px`, async () => {
    const page = await openDashboard(width);
    try {
      const top = await page.$('button[aria-label="가시리풍력 연계 ESS 상세 정보"]');
      const covered = await page.$('button[aria-label="가시리풍력 상세 정보"]');
      await top.scrollIntoView();
      const topBox = await top.boundingBox();
      const coveredBox = await covered.boundingBox();
      assert.ok(topBox.x < coveredBox.x + coveredBox.width && coveredBox.x < topBox.x + topBox.width &&
        topBox.y < coveredBox.y + coveredBox.height && coveredBox.y < topBox.y + topBox.height,
      'test markers must actually overlap');
      await top.tap();
      assert.match(await page.$eval('.plant-details', el => el.textContent), /가시리풍력 연계 ESS.*15.*MW/s);
      const group = await page.$('[role="group"][aria-label="겹친 발전소 선택"]');
      assert.ok(group, 'tap on overlapping markers should expose nearby plant choices');
      const choices = await group.$$('button');
      assert.ok(choices.length >= 2);
      const choice = await group.$('button[aria-label="가시리풍력 선택"]');
      assert.ok(choice, 'covered marker must have an accessible choice');
      await choice.tap();
      assert.match(await page.$eval('.plant-details', el => el.textContent), /가시리풍력.*30.*MW/s);
      assert.equal(await page.$eval('.plant-picker select', el => el.value), 'wind_003');
      assert.equal(await page.$eval('button[aria-label="가시리풍력 상세 정보"]', el => el.getAttribute('aria-pressed')), 'true');
    } finally {
      await page.close();
    }
  });
}

test('overlap choices are keyboard accessible and picker remains usable', async () => {
  const page = await openDashboard();
  try {
    const marker = await page.$('button[aria-label="가시리풍력 연계 ESS 상세 정보"]');
    await marker.focus();
    await page.keyboard.press('Enter');
    const choice = await page.$('[role="group"][aria-label="겹친 발전소 선택"] button[aria-label="가시리풍력 선택"]');
    assert.ok(choice);
    await choice.focus();
    await page.keyboard.press('Enter');
    assert.match(await page.$eval('.plant-details', el => el.textContent), /가시리풍력.*30.*MW/s);
    await page.select('.plant-picker select', 'wind_001');
    assert.match(await page.$eval('.plant-details', el => el.textContent), /한경풍력.*21.*MW/s);
  } finally {
    await page.close();
  }
});

test('favicon resolves and loads under the deployment prefix', async () => {
  const page = await openDashboard();
  try {
    const icon = await page.$eval('link[rel="icon"]', el => ({ url: el.href, type: el.type }));
    assert.equal(new URL(icon.url).pathname, '/v7/logo-dark.png');
    assert.equal(icon.type, 'image/png');
    await page.setCacheEnabled(false);
    const response = await page.goto(icon.url);
    assert.equal(response.status(), 200);
    assert.match(response.headers()['content-type'], /^image\/png/);
  } finally {
    await page.close();
  }
});
