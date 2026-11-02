// Run against a local Expo web server, e.g. EXPO_WEB_URL=http://localhost:19079 node --test tests/chat-short-viewport.browser.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const puppeteer = require('puppeteer');

const url = process.env.EXPO_WEB_URL || 'http://localhost:19079/';
const executablePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function withPage(width, height, fn) {
  const browser = await puppeteer.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width, height });
    await page.goto(url, { waitUntil: 'networkidle2' });
    await fn(page);
  } finally {
    await browser.close();
  }
}

async function openChat(page) {
  await page.click('[role="button"][aria-label="Alan AI 채팅 열기"]');
  await page.waitForSelector('input[placeholder="무엇이든 물어보세요"]');
}

test('short viewport chat composer is visible and receives pointer hits', async () => {
  await withPage(320, 568, async page => {
    await openChat(page);
    const geometry = await page.evaluate(() => {
      const icon = [...document.querySelectorAll('[dir="auto"]')].find(el => el.textContent === '🎤');
      const button = icon.parentElement;
      const input = document.querySelector('input[placeholder="무엇이든 물어보세요"]');
      const r = button.getBoundingClientRect();
      const i = input.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      const style = getComputedStyle(input);
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      const placeholderWidth = ctx.measureText(input.placeholder).width;
      const availableWidth = input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      return { button: r.toJSON(), input: i.toJSON(), hitButton: !!hit && button.contains(hit), placeholderWidth, availableWidth };
    });
    assert.ok(geometry.button.bottom <= 568 && geometry.input.bottom <= 568, JSON.stringify(geometry));
    assert.ok(geometry.hitButton, `microphone center hit another element: ${JSON.stringify(geometry)}`);
    assert.ok(geometry.placeholderWidth <= geometry.availableWidth, `placeholder is clipped: ${JSON.stringify(geometry)}`);
    await page.type('input[placeholder="무엇이든 물어보세요"]', 'hello');
    await page.mouse.click(geometry.button.x + geometry.button.width / 2, geometry.button.y + geometry.button.height / 2);
    await page.waitForSelector('input[placeholder="메시지를 입력하세요..."]');
    assert.match(await page.evaluate(() => document.body.innerText), /hello/);
  });
});

test('bottom navigation exposes named tabs with selected state', async () => {
  await withPage(320, 568, async page => {
    const tabs = await page.$$eval('[role="tab"]', elements => elements.map(e => ({ name: e.getAttribute('aria-label'), selected: e.getAttribute('aria-selected') })));
    assert.deepEqual(tabs, [
      { name: 'SMP예측', selected: 'true' },
      { name: '입찰관리', selected: 'false' },
      { name: '정산', selected: 'false' },
    ]);
    await page.click('[role="tab"][aria-label="입찰관리"]');
    assert.equal(await page.$eval('[role="tab"][aria-label="입찰관리"]', el => el.getAttribute('aria-selected')), 'true');
    assert.equal(await page.$eval('[role="tab"][aria-label="SMP예측"]', el => el.getAttribute('aria-selected')), 'false');
    await openChat(page);
    const selected = await page.$$eval('[role="tab"][aria-selected="true"]', elements => elements.length);
    assert.equal(selected, 0, 'no main tab is selected while Alan is displayed');
  });
});
