import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

test('active map presents static plant capacity rather than invented generation', async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
  try {
    const { default: App } = await server.ssrLoadModule('/src/App.tsx');
    const html = renderToStaticMarkup(React.createElement(App));
    assert.match(html, /발전소 위치 및 설비용량/);
    assert.match(html, /원 크기: 설비용량 기준/);
    assert.doesNotMatch(html, /발전량 히트맵 표시/);
    const source = await readFile(new URL('./src/pages/ExecoDashboard.tsx', import.meta.url), 'utf8');
    assert.doesNotMatch(source, /generationRate|plant\.generation/);
    assert.match(source, /실시간 발전량 정보 없음/);
  } finally {
    await server.close();
  }
});
