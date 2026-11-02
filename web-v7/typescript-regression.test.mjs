import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createServer } from 'vite';

// The dashboard is maintained separately; check the files in this regression's scope.
test('settlement, bidding and API files have no TypeScript diagnostics', () => {
  let output = '';
  try {
    execFileSync(process.execPath, ['node_modules/typescript/bin/tsc', '--noEmit'], { encoding: 'utf8' });
  } catch (error) {
    output = error.stdout || '';
    if (!output) throw error;
  }
  const diagnostics = output.split('\n').filter(line => /^src\/(hooks\/useApi\.ts|pages\/(Bidding|Settlement)\.tsx|services\/api\.ts)\(/.test(line));
  assert.deepEqual(diagnostics, []);
});

test('plant status updates are sent to the API', async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
  const originalFetch = globalThis.fetch;
  try {
    const { apiService } = await server.ssrLoadModule('/src/services/api.ts');
    let request;
    globalThis.fetch = async (url, options) => {
      request = { url, ...options };
      return { ok: true, json: async () => ({ id: 'plant-1', status: 'maintenance' }) };
    };
    const result = await apiService.updatePowerPlant('plant-1', { status: 'maintenance' });
    assert.equal(request.url, '/api/v1/power-plants/plant-1');
    assert.equal(request.method, 'PUT');
    assert.deepEqual(JSON.parse(request.body), { status: 'maintenance' });
    assert.equal(result.status, 'maintenance');
  } finally {
    globalThis.fetch = originalFetch;
    await server.close();
  }
});
