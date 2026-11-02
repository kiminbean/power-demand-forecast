const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function loadClient(publicUrl, platform = 'ios') {
  const requests = [];
  const env = publicUrl === undefined ? {} : { EXPO_PUBLIC_API_URL: publicUrl };
  const modules = {};
  function load(file) {
    if (modules[file]) return modules[file].exports;
    const module = { exports: {} };
    modules[file] = module;
    const source = fs.readFileSync(path.join(__dirname, '..', 'src', file + '.ts'), 'utf8');
    const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
    const context = {
      module, exports: module.exports, process: { env }, console,
      fetch: async (url, options) => {
        requests.push({ url, options });
        return { ok: true, json: async () => ({ status: 'healthy' }) };
      },
      require: (name) => {
        if (name === 'react-native') return { Platform: { OS: platform } };
        if (name === '../config/environment') return load('config/environment');
        throw new Error(`Unexpected dependency: ${name}`);
      },
    };
    vm.runInNewContext(code, context, { filename: file });
    return module.exports;
  }
  return { environment: load('config/environment'), api: load('services/api'), requests };
}

test('environment configuration passes TypeScript semantic checks', () => {
  const file = path.join(__dirname, '..', 'src', 'config', 'environment.ts');
  const program = ts.createProgram([file], { noEmit: true, skipLibCheck: true, strict: true });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  assert.deepEqual(diagnostics.map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
});

test('public API URL configures both native platforms and normalizes trailing slashes', async () => {
  for (const platform of ['ios', 'android']) {
    const { environment, api, requests } = loadClient(' https://example.test/backend/// ', platform);
    assert.equal(environment.API_URL, 'https://example.test/backend');
    assert.equal(environment.config.apiUrl, environment.API_URL);
    await api.apiService.healthCheck();
    await api.apiService.getPowerPlant('plant-1');
    assert.deepEqual(requests.map((request) => request.url), [
      'https://example.test/backend/health',
      'https://example.test/backend/api/v1/power-plants/plant-1',
    ]);
  }
});

test('web defaults to root-relative same-origin API paths even under /mobile/', async () => {
  for (const value of [undefined, '   ']) {
    const { api, requests } = loadClient(value, 'web');
    await api.apiService.healthCheck();
    await api.apiService.getPowerPlant('plant-1');
    assert.deepEqual(requests.map((request) => request.url), [
      '/health', '/api/v1/power-plants/plant-1',
    ]);
  }
});

test('web honors an explicitly configured API base URL', async () => {
  const { api, requests } = loadClient('https://example.test/backend/', 'web');
  await api.apiService.healthCheck();
  assert.equal(requests[0].url, 'https://example.test/backend/health');
});

test('native platforms use the approved Funnel origin when no override is set', async () => {
  for (const platform of ['ios', 'android']) {
    for (const value of [undefined, '   ']) {
      const { api, requests } = loadClient(value, platform);
      await api.apiService.healthCheck();
      await api.apiService.getPowerPlant('plant-1');
      assert.deepEqual(requests.map((request) => request.url), [
        'https://ib-macbookpro.tail918684.ts.net:10000/health',
        'https://ib-macbookpro.tail918684.ts.net:10000/api/v1/power-plants/plant-1',
      ]);
    }
  }
});
