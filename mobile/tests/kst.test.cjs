'use strict';
// src/lib/kst.ts(명세 7.5절): 오프셋 없는 API 시각은 KST, 계산은 고정 +9시간, 기기 시간대와 무관해야 한다.
// 마지막 테스트는 같은 순간을 TZ=UTC·Asia/Seoul·America/Los_Angeles 하위 프로세스에서 계산해 결과가 모두 같은지 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { loadTs, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const kst = loadTs(path.join(SRC_ROOT, 'lib/kst.ts'));
const KST = (y, mo, d, h, mi, s = 0) => Date.UTC(y, mo - 1, d, h, mi, s) - 9 * 3600e3;
const SNAP = KST(2026, 11, 26, 6, 3, 57);

test('parseApiTime reads offset-less API times as KST and honours explicit offsets', () => {
  assert.equal(kst.parseApiTime('2026-11-26T06:03:57'), SNAP);
  assert.equal(kst.parseApiTime('2026-11-26T06:03:57.768171'), SNAP + 768);
  assert.equal(kst.parseApiTime(' 2026-11-26 06:03:57 '), SNAP);
  assert.equal(kst.parseApiTime('2026-11-25T21:03:57Z'), SNAP);
  assert.equal(kst.parseApiTime('2026-11-26T06:03:57+09:00'), SNAP);
  assert.equal(kst.parseApiTime('2026-11-25T13:03:57-0800'), SNAP);
  assert.equal(kst.parseApiTime('2026-11-26T06:03'), KST(2026, 11, 26, 6, 3));
  for (const bad of ['', '2026-02-30T00:00:00', '2026-11-26T24:00:00', '2026-13-01T00:00:00', '26-11-26T06:03', 'now', null, undefined, SNAP]) {
    assert.equal(kst.parseApiTime(bad), null, String(bad));
  }
});

test('day keys, trade hour, now line and countdown follow KST only', () => {
  assert.equal(kst.parseDayKey('20261126'), '2026-11-26');
  assert.equal(kst.parseDayKey('2026-11-27'), '2026-11-27');
  for (const bad of ['20260230', '2026/11/26', '', null, 20261126]) assert.equal(kst.parseDayKey(bad), null, String(bad));
  assert.equal(kst.kstDayStart('2026-11-26'), KST(2026, 11, 26, 0, 0));
  assert.equal(kst.kstDateKey(SNAP), '2026-11-26');
  assert.equal(kst.kstDateKey(KST(2026, 11, 26, 23, 59, 59)), '2026-11-26');
  assert.equal(kst.kstDateKey(KST(2026, 11, 27, 0, 0)), '2026-11-27');
  assert.equal(kst.tradeHourAt(SNAP), 7);
  assert.equal(kst.tradeHourAt(KST(2026, 11, 26, 0, 0)), 1);
  assert.equal(kst.tradeHourAt(KST(2026, 11, 26, 23, 59, 30)), 24);
  // 06:04 지금선 x = 6.067/24 × 폭(F5 합격 확인과 같은 식, 초는 버린다)
  assert.equal((kst.nowLineFraction(KST(2026, 11, 26, 6, 4)) * 24).toFixed(3), '6.067');
  assert.equal(kst.nowLineFraction(KST(2026, 11, 26, 6, 4, 59)), kst.nowLineFraction(KST(2026, 11, 26, 6, 4)));
  assert.equal(kst.msUntilKstTime(SNAP, '10:00'), (3 * 3600 + 56 * 60 + 3) * 1000);
  assert.equal(kst.msUntilKstTime(KST(2026, 11, 26, 10, 0, 30), '10:00'), -30000);
  for (const bad of ['10시', '24:00', '9:00', '']) assert.equal(kst.msUntilKstTime(SNAP, bad), null, bad);
  for (const fn of ['kstParts', 'kstDateKey', 'tradeHourAt', 'nowLineFraction']) assert.equal(kst[fn](NaN), null, fn);
});

test('clock skew: notable above 120 s, implausible above 12 h (server time ignored)', () => {
  const plus3m = kst.clockSkew(SNAP + 3 * 60e3, SNAP);
  assert.deepEqual(plus3m, { skewMs: 180000, plausible: true, notable: true });
  assert.deepEqual(kst.clockSkew(SNAP + 90e3, SNAP), { skewMs: 90000, plausible: true, notable: false });
  const plus13h = kst.clockSkew(SNAP + 13 * 3600e3, SNAP);
  assert.equal(plus13h.plausible, false);
  assert.equal(plus13h.notable, false);
  assert.equal(kst.kstNow(plus3m, SNAP), SNAP + 180000);
  assert.equal(kst.kstNow(plus13h, SNAP), SNAP);
  assert.equal(kst.kstNow(null, SNAP), SNAP);
  assert.equal(kst.clockSkew(NaN, SNAP), null);
});

// 7.5절: 4개 순간 × 3개 시간대에서 영웅 거래시간·카운트다운·지금선·마감 전후·NowPill·날짜 글자가 같아야 한다.
const INSTANTS = [SNAP, KST(2026, 11, 26, 9, 59, 30), KST(2026, 11, 26, 10, 0, 30), KST(2026, 11, 26, 23, 59, 30)];
const EXPECTED = [
  { hero: '07h · 06–07시', countdown: '3시간 56분', open: true, nowLine: (6 + 3 / 60) / 24, pill: '06:03 KST · 조회 1분 전', date: '11-26(목)' },
  { hero: '10h · 09–10시', countdown: '0분', open: true, nowLine: (9 + 59 / 60) / 24, pill: '09:59 KST · 조회 1분 전', date: '11-26(목)' },
  { hero: '11h · 10–11시', countdown: '—', open: false, nowLine: (10 + 0 / 60) / 24, pill: '10:00 KST · 조회 1분 전', date: '11-26(목)' },
  { hero: '24h · 23–24시', countdown: '—', open: false, nowLine: (23 + 59 / 60) / 24, pill: '23:59 KST · 조회 1분 전', date: '11-26(목)' },
];

test('same instants give identical KST results under TZ=UTC, Asia/Seoul and America/Los_Angeles', () => {
  const helper = JSON.stringify(path.join(__dirname, 'helpers', 'ts-loader.cjs'));
  const src = file => JSON.stringify(path.join(SRC_ROOT, file));
  const script = `
    const { createTsLoader } = require(${helper});
    const loader = createTsLoader();
    const kst = loader.load(${src('lib/kst.ts')});
    const f = loader.load(${src('lib/format.ts')});
    const s = loader.load(${src('content/strings.ts')});
    const out = ${JSON.stringify(INSTANTS)}.map(t => {
      const left = kst.msUntilKstTime(t, '10:00');
      return { hero: f.tradeHour(kst.tradeHourAt(t)), countdown: f.duration(left), open: left > 0, nowLine: kst.nowLineFraction(t),
        pill: s.nowPillText({ now: t, fetchedAt: t - 60000 }), date: f.kstDate(t), localHour: new Date(t).getHours() };
    });
    process.stdout.write(JSON.stringify(out));`;
  const run = tz => JSON.parse(execFileSync(process.execPath, ['-e', script], { env: { ...process.env, TZ: tz }, encoding: 'utf8' }));
  const results = Object.fromEntries(['UTC', 'Asia/Seoul', 'America/Los_Angeles'].map(tz => [tz, run(tz)]));
  // 양성 대조: 기기 시간대가 실제로 바뀌었는지(지역 시각 읽기는 시간대마다 다르다).
  assert.notEqual(results.UTC[0].localHour, results['Asia/Seoul'][0].localHour);
  assert.notEqual(results.UTC[0].localHour, results['America/Los_Angeles'][0].localHour);
  for (const [tz, rows] of Object.entries(results)) {
    const kstOnly = rows.map(({ localHour, ...rest }) => rest);
    assert.deepStrictEqual(kstOnly, EXPECTED, tz);
  }
});
