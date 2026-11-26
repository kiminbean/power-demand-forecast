'use strict';
// src/lib/provenance.ts(명세 7.4절): 표의 모든 행, 공급능력 ±1 MW 짝, 예비율 단계 경계, 기상 4종류(API 기본값 5.0/3.0/60/맑음 포함),
// SMP 예측 종류(API 코드에서 읽은 문자열). 문구는 대시보드(src/dashboard/app_v4.py)와 한 글자도 달라서는 안 된다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { loadTs, SRC_ROOT, MOBILE_ROOT } = require('./helpers/ts-loader.cjs');

const p = loadTs(path.join(SRC_ROOT, 'lib/provenance.ts'));
const R0 = path.join(__dirname, 'fixtures', 'round0');
const readJson = name => JSON.parse(fs.readFileSync(path.join(R0, name), 'utf8'));
const KPIS = readJson('_api_v1_dashboard_kpis.json');
const SUPPLY = readJson('_api_v1_power-supply.json');
const withSource = src => ({ ...KPIS, data_source: src });

test('7.4 demand/reserve table: every row, first match from the top', () => {
  const rows = [
    { src: undefined, case: 2, demand: '추정 · 실측 아님', reserve: '추정 · 실측 아님', note: ' · 실측 아님', source: 'API 서버 값 (출처 확인 불가)', judge: false, why: null },
    { src: KPIS.data_source, case: 3, demand: '실측', reserve: '실측', note: '', source: 'KPX 제주 실시간 (API 서버)', judge: true, why: null },
    { src: 'EPSIS Demand (30d avg) + KMA Weather (파일)', case: 4, demand: '추정 · 실측 아님', reserve: '기본값 · 실측 아님', note: ' · API 기본값 (실측 아님)',
      source: 'EPSIS 30일 평균 (API 서버 대체값)', judge: false, why: 'API 서버가 실측 대신 기본 예비율(25%)을 보냈습니다' },
    { src: 'KPX Public API (national) × 0.018', case: 5, demand: '추정 · 실측 아님', reserve: '전국 값 · 제주 아님', note: ' · 전국 예비율 (제주 값 아님)',
      source: '전국 값 × 0.018 비례 추정 (API 서버)', judge: false, why: 'API 서버가 제주 대신 전국 예비율을 보냈습니다' },
    { src: 'Simulated', case: 6, demand: '시뮬레이션 · 실측 아님', reserve: '기본값 · 실측 아님', note: ' · API 기본값 (실측 아님)',
      source: 'API 서버 기본값 650 MW·25% (실측 아님)', judge: false, why: 'API 서버가 실측 대신 기본값(수요 650 MW·예비율 25%)을 보냈습니다' },
  ];
  for (const row of rows) {
    const got = p.demandProvenance(row.src === undefined ? { ...KPIS, data_source: undefined } : withSource(row.src));
    assert.deepEqual(
      { case: got.case, demand: got.demandChip.text, reserve: got.reserveChip.text, note: got.reserveNote, source: got.source, judge: got.judge, why: got.why },
      { case: row.case, demand: row.demand, reserve: row.reserve, note: row.note, source: row.source, judge: row.judge, why: row.why }, String(row.src));
  }
  for (const src of [null, 42, '', '   ', ['KPX Jeju Crawler']]) assert.equal(p.demandProvenance(withSource(src)).case, 2, JSON.stringify(src));
  assert.equal(p.demandProvenance(null).case, 2);
  assert.equal(p.demandProvenance(withSource('KPX Jeju Crawler + EPSIS Demand')).case, 3);
  assert.equal(p.demandProvenance(withSource('EPSIS Demand + KPX')).case, 4);
  assert.equal(p.demandProvenance(withSource('KPX Jeju Crawler')).demandChip.kind, 'measured');
  assert.equal(p.demandProvenance(withSource('Simulated')).demandChip.kind, 'estimated');
});

test('supplyPair uses only the current-hour row, within ±1 MW of KPI demand, and only for case ③', () => {
  const pair = p.supplyPair(KPIS, SUPPLY);
  assert.equal(pair.ok, true);
  assert.deepEqual([pair.hour, pair.supplyMw, pair.demandMw, pair.reserveMw], [6, 1191, 664, 527]);
  assert.equal(pair.watchBoundaryMw, 1191 / 1.15);
  const rowDemand = d => ({ ...SUPPLY, data: SUPPLY.data.map(r => (r.hour === 6 ? { ...r, demand: d } : r)) });
  assert.equal(p.supplyPair(KPIS, rowDemand(665.0)).ok, true);
  assert.equal(p.supplyPair(KPIS, rowDemand(663.0)).ok, true);
  assert.equal(p.supplyPair(KPIS, rowDemand(665.01)).ok, false);
  assert.equal(p.supplyPair(KPIS, rowDemand(662.99)).ok, false);
  assert.equal(p.supplyPair({ ...KPIS, current_demand_mw: 664.3 }, rowDemand(665.3)).ok, true);
  // 지금 시각이 7시면 6시 행이 맞아도 쓰지 않는다(다른 시각 행은 패턴 값).
  const at7 = { ...SUPPLY, current_hour: 7 };
  assert.ok(Math.abs(SUPPLY.data.find(r => r.hour === 7).demand - 664) > 1);
  assert.deepEqual(p.supplyPair(KPIS, at7), { ok: false, reason: '공급능력 미수신' });
  assert.equal(p.supplyPair(withSource('Simulated'), SUPPLY).ok, false);
  assert.equal(p.supplyPair(KPIS, { ...SUPPLY, data: SUPPLY.data.filter(r => r.hour !== 6) }).ok, false);
  assert.equal(p.supplyPair(KPIS, { ...SUPPLY, data: SUPPLY.data.map(r => (r.hour === 6 ? { ...r, supply: null } : r)) }).ok, false);
  assert.equal(p.supplyPair({ ...KPIS, current_demand_mw: null }, SUPPLY).ok, false);
  assert.equal(p.supplyPair(KPIS, null).ok, false);
});

test('reserveLevel boundaries (≥15 정상, <15 보통, <10 낮음, <5 위험): KPX 공식 경보 단계 낱말은 쓰지 않는다(D9)', () => {
  const label = r => (p.reserveLevel(r) || {}).label;
  assert.equal(label(79.5), '정상 · 넉넉합니다');
  assert.equal(label(15), '정상 · 넉넉합니다');
  assert.equal(label(14.99), '보통 · 여유가 줄고 있습니다');
  assert.equal(label(10), '보통 · 여유가 줄고 있습니다');
  assert.equal(label(9.99), '낮음 · 여유가 부족합니다');
  assert.equal(label(5), '낮음 · 여유가 부족합니다');
  // KPX 전력수급 위기경보 단계(준비·관심·주의·경계·심각)와 겹치는 이름이 없다.
  for (const r of [79.5, 12, 7, 1]) assert.doesNotMatch(p.reserveLevel(r).name, /준비|관심|주의|경계|심각/, String(r));
  assert.equal(label(4.99), '위험 · 공급이 빠듯합니다');
  assert.equal(label(-3), '위험 · 공급이 빠듯합니다');
  assert.deepEqual([p.reserveLevel(79.5).tone, p.reserveLevel(12).tone, p.reserveLevel(7).tone, p.reserveLevel(1).tone], ['ok', 'watch', 'caution', 'danger']);
  for (const bad of [null, NaN, Infinity, '79.5']) assert.equal(p.reserveLevel(bad), null, String(bad));
});

test('weather: live · file · default (API 5.0/3.0/60/맑음 without a source token) · none', () => {
  const live = p.weatherProvenance(KPIS);
  assert.deepEqual([live.kind, live.chip.text, live.source], ['live', '실측', '기상청 초단기실황 (API 서버)']);
  for (const token of ['KMA Weather Crawler', 'KMA Real-time API', '기상청 (실시간)']) assert.equal(p.weatherProvenance(withSource(`KPX Jeju Crawler + ${token}`)).kind, 'live');
  const file = p.weatherProvenance(withSource('KPX Jeju Crawler (실시간) + KMA Weather (파일)'));
  assert.deepEqual([file.kind, file.chip.text, file.source], ['file', '추정 · 실시간 아님', '기상청 과거 자료 24시간 평균 (API 서버) · 실시간 아님']);
  // API 기본값: 값은 그럴듯하지만 출처 토큰이 없다 → 실측으로 보이면 안 된다.
  const apiDefault = { ...withSource('KPX Jeju Crawler (실시간)'), weather: { temperature: 5.0, wind_speed: 3.0, humidity: 60, condition: '맑음' } };
  const def = p.weatherProvenance(apiDefault);
  assert.deepEqual([def.kind, def.chip.text, def.source], ['default', '기본값 · 실측 아님', 'API 서버 기본값 (실측 아님)']);
  assert.equal(p.weatherProvenance(withSource(null)).kind, 'default');
  for (const none of [null, undefined, 'x', [KPIS]]) assert.deepEqual(p.weatherProvenance(none), { kind: 'none', chip: null, source: null });
});

// API가 실제로 보내는 model_used 문자열(src/smp/models/serving.py 18–20행 + api/v6_routes.py 63–66·309·365행).
const API_MODEL_STRINGS = {
  baseline: ['기준선 예측 (전날 같은 시간 제주 하루전 가격)', '기준선 예측 (전날 같은 시간 제주 하루전 가격) · 검증 MAPE 12.68%'],
  model: [
    'BiLSTM+Attention v4.1 (제주 하루전 SMP)', 'BiLSTM+Attention v4.1 (제주 하루전 SMP) · 검증 MAPE 16.82%',
    'BiLSTM+Attention v3.2 Optuna (성능 재검증 전)',
  ],
  fallback: ['패턴 기반 대체값 (모델 예측 실패)', 'EPSIS Fallback (Model Unavailable)', 'SMP 예측 모델'],
};
const API_SOURCES = {
  'src/smp/models/serving.py': ['"BiLSTM+Attention v4.1 (제주 하루전 SMP)"', '"기준선 예측 (전날 같은 시간 제주 하루전 가격)"', '"BiLSTM+Attention v3.2 Optuna (성능 재검증 전)"'],
  'api/v6_routes.py': ["'SMP 예측 모델'", 'f"{label} · 검증 MAPE {mape:.2f}%"', '"패턴 기반 대체값 (모델 예측 실패)"', '"EPSIS Fallback (Model Unavailable)"'],
};

test('smpForecastKind classifies every API model string; unknown names fail closed (estimated)', () => {
  for (const [kind, names] of Object.entries(API_MODEL_STRINGS)) {
    for (const name of names) assert.equal(p.smpForecastKind({ model_used: name }), kind, name);
  }
  for (const odd of ['', '  ', null, 42, 'unknown model', 'BiLSTM+Attention v9.9 (새 모델)', 'v4.1', '기준선 대체값']) {
    assert.equal(p.smpForecastKind({ model_used: odd }), 'fallback', String(odd));
  }
  assert.equal(p.smpForecastKind(null), 'fallback');
  assert.equal(p.forecastChip('baseline').text, '예측 · 기준선');
  assert.equal(p.forecastChip('model').text, '예측 · 학습 모델');
  assert.equal(p.forecastChip('fallback').text, '추정 · API 대체값 (모델 예측 아님)');
  assert.equal(p.forecastShortName(API_MODEL_STRINGS.baseline[1]), '기준선');
  assert.equal(p.forecastShortName(API_MODEL_STRINGS.model[1]), 'v4.1');
  assert.equal(p.forecastShortName(API_MODEL_STRINGS.model[2]), 'v3.2 · 재검증 전');
});

test('pinned model strings still appear in the API source (drift guard when the repo is reachable)', t => {
  const repo = path.resolve(MOBILE_ROOT, '..');
  for (const [rel, needles] of Object.entries(API_SOURCES)) {
    const file = path.join(repo, rel);
    if (!fs.existsSync(file)) {
      t.diagnostic(`${rel} not reachable from ${MOBILE_ROOT}; pinned strings checked by classification only`);
      continue;
    }
    const text = fs.readFileSync(file, 'utf8');
    for (const needle of needles) assert.ok(text.includes(needle), `${rel} no longer contains ${needle}`);
  }
});

test('chips: glyph + text, long accessible name, documented accuracy constant', () => {
  assert.equal(p.chipText(p.CHIPS.confirmedPrice), '● 확정 가격');
  assert.equal(p.chipText(p.CHIPS.forecastBaseline), '┅ 예측 · 기준선');
  assert.equal(p.chipText(p.CHIPS.estQuantile), '▨ 추정 · 기준선 분위수');
  assert.equal(p.chipText(p.CHIPS.exampleRibbon), '┆ 예시 · 실제 값 아님');
  assert.equal(p.chipText(p.CHIPS.marketRule), '시장 규칙 · 서버 시각');
  assert.equal(p.chipA11yLabel(p.CHIPS.confirmedPrice, 'KPX 제주 하루전시장'), '상태: 확정 가격, 출처 KPX 제주 하루전시장');
  assert.ok(Object.isFrozen(p.CHIPS) && Object.isFrozen(p.CHIPS.measured));
  assert.deepEqual({ ...p.DOCUMENTED_SMP_ACCURACY }, {
    maeKrwPerKwh: 11.97, mapePct: 11.75, r2: 0.203, method: 'walk-forward', period: '2025-07-22~12-18', basis: '기준선 자체(학습한 모델 아님)',
  });
});
