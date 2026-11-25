// 설계 명세 8.1–8.3절(웹 셸)을 고정한다: mobile/public/index.html 템플릿과 app.json의 web 필드,
// 그리고 설치된 Expo CLI(SDK 50, @expo/cli 0.17.13)가 이 둘로 만드는 index.html의 머리 부분.
// 실제 export 결과(파비콘 링크 1개, 번들 script, 복사본에 덮이지 않음)는 F2 단계 증거에서 확인한다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const TEMPLATE_PATH = path.join(ROOT, 'public', 'index.html');
const template = fs.readFileSync(TEMPLATE_PATH, 'utf8');
const appJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'app.json'), 'utf8'));

// 5.1절 색 역할 중 index.html이 JS 전에 쓰는 값(canvas·surface·border·text1·text3·primary).
const SHELL_TOKENS = {
  light: { canvas: '#F6F8FB', surface: '#FFFFFF', border: '#E2E8F0', text: '#0F172A', text3: '#56657A', focus: '#2563EB' },
  dark: { canvas: '#0B1220', surface: '#121B31', border: '#24324D', text: '#F1F5F9', text3: '#94A3B8', focus: '#60A5FA' },
};
const WEB_TITLE = 'eXeco 제주 전력시장';
const BUNDLE = '_expo/static/js/web/AppEntry-0123456789abcdef.js';

function metas(html, name) {
  const re = new RegExp(`<meta\\s+name="${name}"([^>]*)>`, 'g');
  return [...html.matchAll(re)].map(m => ({
    media: (m[1].match(/media="([^"]*)"/) || [])[1],
    content: (m[1].match(/content="([^"]*)"/) || [])[1],
  }));
}

function count(html, needle) {
  return html.split(needle).length - 1;
}

// CSS 블록 하나의 --pd-* 변수를 읽는다.
function cssVars(block) {
  return Object.fromEntries([...block.matchAll(/--pd-([a-z0-9]+):\s*([^;]+);/g)].map(m => [m[1], m[2].trim()]));
}

function cssBlock(selectorRe) {
  const m = template.match(selectorRe);
  assert.ok(m, `CSS 블록을 찾지 못함: ${selectorRe}`);
  return cssVars(m[1]);
}

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

test('template: lang="ko", title placeholder, viewport-fit=cover without zoom blocking', () => {
  assert.match(template, /^<!DOCTYPE html>\n<html lang="ko">\n/);
  assert.equal(count(template, '<html'), 1);
  assert.ok(!template.includes('%LANG_ISO_CODE%'), 'lang은 템플릿에 고정한다(8.5절)');
  assert.equal(count(template, '<title>%WEB_TITLE%</title>'), 1);
  const viewport = metas(template, 'viewport');
  assert.equal(viewport.length, 1);
  assert.deepEqual(viewport[0].content.split(',').map(s => s.trim()),
    ['width=device-width', 'initial-scale=1', 'viewport-fit=cover']);
  assert.doesNotMatch(template, /maximum-scale|user-scalable|shrink-to-fit|http-equiv|httpEquiv/i);
  assert.deepEqual(metas(template, 'color-scheme').map(m => m.content), ['light dark']);
  assert.equal(metas(template, 'description').length, 1);
});

test('template: exactly two theme-color metas, each with media, equal to the canvas tokens', () => {
  const tc = metas(template, 'theme-color');
  assert.deepEqual(tc, [
    { media: '(prefers-color-scheme: light)', content: SHELL_TOKENS.light.canvas },
    { media: '(prefers-color-scheme: dark)', content: SHELL_TOKENS.dark.canvas },
  ]);
});

test('template: CSS variables match section 5.1 and the two dark selectors agree', () => {
  const light = cssBlock(/\n {4}:root \{([^}]*)\}/);
  const darkMedia = cssBlock(/@media \(prefers-color-scheme: dark\) \{ :root:not\(\[data-theme="light"\]\) \{([^}]*)\} \}/);
  const darkForced = cssBlock(/:root\[data-theme="dark"\] \{([^}]*)\}/);
  for (const [scheme, vars] of [['light', light], ['dark', darkMedia], ['dark', darkForced]]) {
    for (const [key, hex] of Object.entries(SHELL_TOKENS[scheme])) {
      assert.equal(vars[key], hex, `${scheme} --pd-${key}`);
    }
  }
  assert.deepEqual(darkMedia, darkForced, '앱 안 "어둡게"와 OS 어둡게가 같은 값을 써야 한다');
  assert.match(template, /:root \{[^}]*color-scheme: light; \}/);
  assert.equal(count(template, 'color-scheme: dark;'), 2);
  // 스켈레톤 안내 글자(13 px, text3)와 noscript 글자(text)는 두 테마 바탕에서 AA를 넘는다.
  for (const scheme of ['light', 'dark']) {
    const t = SHELL_TOKENS[scheme];
    assert.ok(contrast(t.text3, t.canvas) >= 4.5, `${scheme} text3/canvas ${contrast(t.text3, t.canvas).toFixed(2)}`);
    assert.ok(contrast(t.text, t.canvas) >= 4.5, `${scheme} text/canvas`);
  }
});

test('template: skeleton lives inside #root, noscript outside, no fixed overlay or external resources', () => {
  const body = template.slice(template.indexOf('<body>'), template.indexOf('</body>'));
  assert.ok(body.indexOf('<noscript>') >= 0 && body.indexOf('<noscript>') < body.indexOf('<div id="root">'));
  assert.match(body, /<noscript>[^<]*JavaScript[^<]*<\/noscript>/);
  assert.match(body, /<div id="root"><div class="pd-boot" aria-busy="true">/);
  const root = body.slice(body.indexOf('<div id="root">'));
  for (const cls of ['pd-boot-bar', 'pd-boot-card', 'pd-boot-msg', 'pd-boot-slow', 'pd-boot-tabs']) {
    assert.ok(root.includes(`class="${cls}"`), cls);
  }
  assert.equal(count(body, '<div'), count(body, '</div>'), '#root 안 div가 짝이 맞아야 한다');
  assert.ok(root.trimEnd().endsWith('</div></div>'), '스켈레톤 뒤에 #root가 닫힌다');
  assert.doesNotMatch(template, /position:\s*fixed/);
  assert.doesNotMatch(template, /https?:\/\/|<link\b|@import|@font-face|favicon/);
  assert.doesNotMatch(template, /<script\s+src=/, '번들 script는 export가 붙인다');
  assert.ok(!fs.existsSync(path.join(ROOT, 'public', 'favicon.ico')),
    'public/favicon.ico가 있으면 Expo가 파비콘 링크를 넣지 않는다(8.1절)');
});

test('template: slow-connection line appears once after 12 s even with reduced motion', () => {
  assert.match(template, /\.pd-boot-slow \{ opacity: 0; animation: pd-show 0s linear 12s forwards !important; \}/);
  assert.match(template, /@keyframes pd-show \{ to \{ opacity: 1; \} \}/);
  assert.match(template, /@media \(prefers-reduced-motion: reduce\) \{ \*, \*::before, \*::after \{ animation: none !important; transition: none !important; \} \}/);
  // 줄인 움직임 규칙(*)보다 .pd-boot-slow가 뒤에 오고 구체성이 높아야 12초 표시가 살아남는다.
  assert.ok(template.indexOf('.pd-boot-slow { opacity') > template.indexOf('prefers-reduced-motion'));
});

test('template: inline theme script only applies a stored light/dark choice', () => {
  const scripts = [...template.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  assert.equal(scripts.length, 1);
  const s = scripts[0];
  assert.match(s, /^\s*try \{/);
  assert.match(s, /catch \(e\) \{\}\s*$/);
  assert.match(s, /localStorage\.getItem\('pd-theme'\)/);
  assert.match(s, /if \(t === 'light' \|\| t === 'dark'\) document\.documentElement\.setAttribute\('data-theme', t\);/);
});

test('app.json: web fields of section 8.2, no fields that make Expo add duplicate metas', () => {
  const { expo } = appJson;
  assert.equal(expo.userInterfaceStyle, 'automatic');
  assert.equal(expo.name, 'RE-BMS', 'expo.name은 건드리지 않는다(P8)');
  assert.equal(expo.version, '5.0.0', 'version은 건드리지 않는다(P8)');
  assert.deepEqual(expo.web, { favicon: './src/assets/favicon.png', name: WEB_TITLE, shortName: 'eXeco', lang: 'ko' });
  assert.ok(fs.existsSync(path.join(ROOT, expo.web.favicon)), 'web.favicon 원본 파일');
  // web.themeColor는 media 없는 세 번째 theme-color를, web.description은 두 번째 description을 만든다
  // (@expo/cli webTemplate.js addMeta). 템플릿의 meta 하나만 남긴다.
  assert.ok(!('themeColor' in expo.web));
  assert.ok(!('description' in expo.web));
});

test('installed Expo CLI turns the template into the expected shell head and script', async () => {
  const resolve = id => require.resolve(id, { paths: [ROOT] });
  const { createTemplateHtmlFromExpoConfigAsync } = require(resolve('@expo/cli/build/src/start/server/webTemplate'));
  const { serializeHtmlWithAssets } = require(resolve('@expo/cli/build/src/start/server/metro/serializeHtml'));
  const shell = await createTemplateHtmlFromExpoConfigAsync(ROOT, { scripts: [], cssLinks: [] });
  const html = serializeHtmlWithAssets({
    mode: 'production',
    resources: [{ type: 'js', filename: BUNDLE, metadata: { isAsync: false } }],
    template: shell,
    baseUrl: '',
  });
  assert.equal(count(html, '<html lang="ko">'), 1);
  assert.equal(count(html, `<title>${WEB_TITLE}</title>`), 1);
  assert.doesNotMatch(html, /%[A-Z_]+%/);
  assert.deepEqual(metas(html, 'theme-color').map(m => m.media),
    ['(prefers-color-scheme: light)', '(prefers-color-scheme: dark)']);
  assert.equal(metas(html, 'description').length, 1);
  assert.equal(count(html, `<script src="/${BUNDLE}" defer></script>`), 1);
  assert.ok(html.indexOf(`<script src="/${BUNDLE}"`) > html.indexOf('class="pd-boot-tabs"'), '번들은 스켈레톤 뒤, body 끝에 붙는다');
  assert.equal(count(html, '<script src='), 1);
});
