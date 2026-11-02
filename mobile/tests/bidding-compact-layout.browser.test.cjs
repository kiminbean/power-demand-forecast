// Run against Expo web: EXPO_WEB_URL=http://localhost:19079 node --test tests/bidding-compact-layout.browser.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const puppeteer = require('puppeteer');

const url = process.env.EXPO_WEB_URL || 'http://localhost:19079/';
const executablePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function withBidding(fn) {
  const browser = await puppeteer.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 320, height: 568 });
    await page.setRequestInterception(true);
    page.on('request', request => {
      if (new URL(request.url()).pathname === '/api/v1/power-plants') {
        request.respond({ status: 404, contentType: 'application/json', body: '{}' });
      } else request.continue();
    });
    await page.goto(url, { waitUntil: 'networkidle2' });
    await page.click('[role="tab"][aria-label="입찰관리"]');
    await page.waitForFunction(() => document.body.innerText.includes('발전소 정보를 현재 이용할 수 없습니다.'));
    await fn(page);
  } finally {
    await browser.close();
  }
}

test('Alan never covers the VPP switch as bidding scrolls', async () => {
  await withBidding(async page => {
    const result = await page.evaluate(() => {
      const alan = document.querySelector('[aria-label="Alan AI 채팅 열기"]');
      const toggle = document.querySelector('[role="switch"]');
      if (!toggle || !alan) throw new Error('VPP switch or Alan button missing');
      const scroller = [...document.querySelectorAll('*')].find(el => {
        const style = getComputedStyle(el);
        return ['auto', 'scroll'].includes(style.overflowY) && el.contains(toggle) && el.scrollHeight > el.clientHeight;
      });
      if (!scroller) throw new Error('bidding scroller missing');
      const overlaps = [];
      for (let y = 0; y <= scroller.scrollHeight - scroller.clientHeight; y += 4) {
        scroller.scrollTop = y;
        const a = alan.getBoundingClientRect();
        const b = toggle.getBoundingClientRect();
        if (a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top) overlaps.push(y);
      }
      scroller.scrollTop = 0;
      toggle.scrollIntoView({ block: 'center' });
      const r = toggle.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return { overlaps, hitSwitch: toggle.contains(hit), switchRect: r.toJSON() };
    });
    assert.deepEqual(result.overlaps, [], `Alan overlaps VPP switch while scrolling: ${JSON.stringify(result)}`);
    assert.ok(result.hitSwitch, `VPP switch must receive pointer hits: ${JSON.stringify(result)}`);
    const before = await page.$eval('[role="switch"]', el => el.checked);
    await page.click('[role="switch"]');
    const after = await page.$eval('[role="switch"]', el => el.checked);
    assert.notEqual(after, before, 'VPP switch changes state');
    await page.click('[aria-label="Alan AI 채팅 열기"]');
    await page.waitForSelector('input[placeholder="무엇이든 물어보세요"]');
  });
});

test('unavailable plant message has no one-character orphan line', async () => {
  await withBidding(async page => {
    const lines = await page.evaluate(() => {
      const message = '발전소 정보를 현재 이용할 수 없습니다.';
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode()) && node.textContent !== message) {}
      if (!node) throw new Error('unavailable message missing');
      const chars = new Map();
      for (let i = 0; i < node.length; i++) {
        const range = document.createRange();
        range.setStart(node, i); range.setEnd(node, i + 1);
        const top = range.getBoundingClientRect().top;
        chars.set(top, (chars.get(top) || '') + node.textContent[i]);
      }
      return [...chars.values()].map(line => line.trim()).filter(Boolean);
    });
    assert.ok(lines.every(line => line.length > 2), `orphaned text line: ${JSON.stringify(lines)}`);
    assert.ok(!lines.some(line => /^다\./.test(line)), `split Korean word: ${JSON.stringify(lines)}`);
  });
});
