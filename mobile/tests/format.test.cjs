'use strict';
// src/lib/format.ts(명세 6.3절): 화면 숫자·시각의 유일한 포맷 도우미. 6.3절 예를 모두 고정하고, null·NaN·±Infinity 규칙이
// 기존 formatNullableNumber와 같은지 본다.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const loader = createTsLoader();
const f = loader.load(path.join(SRC_ROOT, 'lib/format.ts'));
const { formatNullableNumber } = loader.load(path.join(SRC_ROOT, 'utils/smpPresentation.ts'));
const KST = (y, mo, d, h, mi, s = 0) => Date.UTC(y, mo - 1, d, h, mi, s) - 9 * 3600e3;

test('6.3 examples: price, MW, percent, money, delta', () => {
  assert.equal(f.price(115.27), '115.3');
  assert.equal(f.price(145.2), '145.2');
  assert.equal(f.price(0), '0.0');
  assert.equal(f.mw(1191.0), '1,191');
  assert.equal(f.mw(664.0), '664');
  assert.equal(f.mw(1191 / 1.15), '1,036');
  assert.equal(f.mw(1191 - 664), '527');
  assert.equal(f.pct(79.5, 1), '79.5');
  assert.equal(f.pct(72.64), '72.6');
  assert.equal(f.krwFromThousand(4686.04), '468.6만원');
  assert.equal(f.krwFromThousand(560.98), '56.1만원');
  assert.equal(f.krw(123456789), '1.23억원');
  assert.equal(f.krw(100000000), '1.00억원');
  assert.equal(f.krw(99990000), '9,999.0만원');
  assert.equal(f.krw(17160000), '1,716.0만원');
  assert.equal(f.krw(-840000), '−84.0만원');
  assert.equal(f.krwFromThousand(0), '0.0만원');
  assert.equal(f.delta(3.2), '+3.2');
  assert.equal(f.delta(-17.2), '−17.2');
  assert.equal(f.delta(29.93), '+29.9');
  assert.equal(f.delta(0), '0.0');
  assert.equal(f.delta(-0.04), '0.0');
  assert.ok(!f.delta(-3).includes('+-'));
  assert.equal(f.num(-1234.56, 1), '−1,234.6');
  assert.equal(f.num(-0.04, 1), '0.0');
});

test('6.3 examples: trade hour labels use two digits (07h · 06–07시)', () => {
  assert.equal(f.tradeHour(7), '07h · 06–07시');
  assert.equal(f.tradeHour(1), '01h · 00–01시');
  assert.equal(f.tradeHour(24), '24h · 23–24시');
  assert.equal(f.tradeHourRange(3, 5), '03–05h · 02–05시');
  assert.equal(f.tradeHourRange(8, 8), '08h · 07–08시');
  for (const bad of [0, 25, 7.5, NaN, null, undefined]) assert.equal(f.tradeHour(bad), '—', String(bad));
  assert.equal(f.tradeHourRange(5, 3), '—');
});

test('6.3 examples: KST clock, date, age and duration (minutes floored)', () => {
  assert.equal(f.kstTime(KST(2026, 11, 26, 6, 4, 10)), '06:04 KST');
  assert.equal(f.kstTime(KST(2026, 11, 26, 6, 3, 57)), '06:03 KST');
  assert.equal(f.kstClock(KST(2026, 11, 26, 23, 59, 59)), '23:59');
  assert.equal(f.kstDate(KST(2026, 11, 26, 6, 4)), '11-26(목)');
  assert.equal(f.kstDayLabel('2026-11-27'), '11-27(금)');
  assert.equal(f.kstDayLabel('20261127'), '11-27(금)');
  assert.equal(f.kstDayLabel('bad'), '—');
  assert.equal(f.age(61000), '1분 전');
  assert.equal(f.age(12 * 60000 + 5000), '12분 전');
  assert.equal(f.age(65 * 60000), '1시간 5분 전');
  assert.equal(f.age(30000), '방금');
  assert.equal(f.age(-5000), '방금');
  assert.equal(f.duration((3 * 3600 + 56 * 60 + 3) * 1000), '3시간 56분');
  assert.equal(f.duration(45000), '0분');
  assert.equal(f.duration(3600000), '1시간');
  assert.equal(f.duration(-1), '—');
});

test('null, NaN and ±Infinity are always "—", the same rule as formatNullableNumber', () => {
  const numeric = ['num', 'price', 'mw', 'pct', 'krw', 'krwFromThousand', 'delta', 'kstClock', 'kstTime', 'kstDate', 'age', 'duration'];
  for (const value of [null, undefined, NaN, Infinity, -Infinity]) {
    assert.equal(formatNullableNumber(value), '—');
    for (const name of numeric) assert.equal(f[name](value), '—', `${name}(${value})`);
  }
  assert.equal(f.DASH, formatNullableNumber(null));
});
