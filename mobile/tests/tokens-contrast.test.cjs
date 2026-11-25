'use strict';
// src/ui/tokens.ts(명세 5절, 게이트 M4): 실제로 쓰는 글자·바탕 쌍을 두 테마에서 WCAG 식으로 계산한다(글자 4.5:1, 비글자 3:1).
// 빗금·음영처럼 반투명 덧칠이 있는 바탕은 덧칠을 섞은 색으로 잰다. 표는 진단 출력으로 남긴다(증거: 대비 표).
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { loadTs, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const tokens = loadTs(path.join(SRC_ROOT, 'ui/tokens.ts'));
const { ICONS } = loadTs(path.join(SRC_ROOT, 'ui/icons.ts'));

function parseColor(text) {
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].replace(/./g, ch => ch + ch) : hex[1];
    return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)).concat(1);
  }
  const rgba = /^rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*(\.\d+|0?\.\d+|1|0)\s*\)$/.exec(text);
  if (rgba) return [Number(rgba[1]), Number(rgba[2]), Number(rgba[3]), Number(rgba[4])];
  throw new Error(`unparsable color ${text}`);
}
const over = (top, base) => [0, 1, 2].map(i => top[i] * top[3] + base[i] * (1 - top[3])).concat(1);
function luminance([r, g, b]) {
  const lin = c => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}
const ratio = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
const resolve = (theme, key) => key.split('.').reduce((node, part) => {
  if (node === null || node === undefined || !(part in node)) throw new Error(`no token ${key}`);
  return node[part];
}, theme);

function measure(themeName, pair) {
  const theme = tokens.THEMES[themeName];
  let bg = parseColor(resolve(theme, pair.bg));
  if (pair.overlay) bg = over(parseColor(resolve(theme, pair.overlay)), bg);
  const fg = over(parseColor(resolve(theme, pair.fg)), bg);
  return ratio(fg, bg);
}

// 그림자처럼 한 테마에서만 쓰는 값은 null이 될 수 있어 잎은 종류 대신 '값'으로 비교한다.
const shape = node => (node && typeof node === 'object' ? Object.fromEntries(Object.keys(node).sort().map(k => [k, shape(node[k])])) : 'leaf');

test('light and dark define the same token keys; data colors are identical in both themes', () => {
  assert.deepStrictEqual(shape(tokens.THEMES.light), shape(tokens.THEMES.dark));
  assert.deepStrictEqual(tokens.THEMES.light.series, tokens.THEMES.dark.series);
  for (const name of ['light', 'dark']) {
    const walk = node => Object.values(node).forEach(v => (v && typeof v === 'object' ? walk(v) : typeof v === 'string' && /^(#|rgba)/.test(v) && parseColor(v)));
    walk(tokens.THEMES[name]);
  }
});

test('every used text/background pair meets 4.5:1 (text) or 3:1 (non-text) in both themes', t => {
  const failures = [];
  t.diagnostic('| 테마 | 글자·선 | 바탕 | 대비 | 기준 | 쓰임 |');
  t.diagnostic('|---|---|---|---|---|---|');
  for (const themeName of ['light', 'dark']) {
    for (const pair of tokens.CONTRAST_PAIRS) {
      const value = measure(themeName, pair);
      const bg = pair.overlay ? `${pair.bg} + ${pair.overlay}` : pair.bg;
      t.diagnostic(`| ${themeName} | ${pair.fg} | ${bg} | ${value.toFixed(2)} | ${pair.min} | ${pair.use} |`);
      if (value < pair.min) failures.push(`${themeName} ${pair.fg} on ${bg}: ${value.toFixed(2)} < ${pair.min}`);
    }
  }
  assert.deepEqual(failures, []);
  assert.ok(tokens.CONTRAST_PAIRS.length >= 37, `pairs ${tokens.CONTRAST_PAIRS.length}`);
});

test('spec anchor values: 5.1 table minimums and 5.3 data-colour minimums on cards', () => {
  const on = (theme, fg, bg) => measure(theme, { fg, bg });
  assert.ok(on('light', 'color.text1', 'color.surface') >= 15.88 && on('dark', 'color.text1', 'color.surface') >= 14.14);
  assert.ok(on('light', 'color.text2', 'color.surface') >= 9.21 && on('dark', 'color.text2', 'color.surface') >= 10.44);
  assert.equal(on('light', 'color.onPrimary', 'color.primary').toFixed(2), '5.17');
  assert.ok(on('dark', 'color.onPrimary', 'color.primary') >= 7.36);
  assert.equal(on('light', 'color.nowPillText', 'color.nowPillBg').toFixed(2), '7.58');
  assert.equal(Math.min(on('light', 'series.smp', 'color.surface'), on('dark', 'series.smp', 'color.surface')).toFixed(2), '3.63');
  assert.equal(Math.min(on('light', 'series.watch', 'color.surface'), on('dark', 'series.watch', 'color.surface')).toFixed(2), '3.41');
});

test('informational (not gated): decorative chip borders and text over the 22 % zero-input hatch', t => {
  // 칩의 상태는 글자·글리프·테두리 모양이 함께 전한다(5.2절)라 테두리 색은 장식이다. 0.00 빗금 위에는 글자를 두지 않는다(F5).
  for (const themeName of ['light', 'dark']) {
    for (const kind of ['measured', 'forecast', 'estimated', 'example']) {
      t.diagnostic(`info ${themeName} chip.${kind}.border on color.surface: ${measure(themeName, { fg: `chip.${kind}.border`, bg: 'color.surface' }).toFixed(2)}`);
    }
    const hatch = measure(themeName, { fg: 'color.text3', bg: 'color.surface', overlay: 'chart.zeroHatch' });
    t.diagnostic(`info ${themeName} color.text3 on surface + chart.zeroHatch: ${hatch.toFixed(2)} (그 위에 글자 두지 않음)`);
  }
});

test('type scale has exactly five rendered sizes with a 12 px floor; icon set has the 13 spec names', () => {
  const sizes = new Set(Object.values(tokens.TYPE_SCALE).map(v => v.fontSize).concat(tokens.DISPLAY_COMPACT.fontSize));
  assert.deepEqual([...sizes].sort((x, y) => x - y), [12, 13, 15, 20, 28, 32]);
  assert.equal(Math.min(...sizes), 12);
  assert.deepEqual(Object.keys(ICONS).sort(),
    ['alert', 'back', 'book', 'chart', 'chev', 'clock', 'close', 'contrast', 'info', 'list', 'lock', 'offline', 'refresh']);
  for (const [name, shapes] of Object.entries(ICONS)) {
    for (const sh of shapes) assert.ok(sh.kind === 'circle' || /^[MmLlHhVvAaZz0-9 .,-]+$/.test(sh.d), `${name} path uses only basic commands`);
  }
});
