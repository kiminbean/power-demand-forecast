'use strict';
// 픽스처 재생 도우미: round 0 스냅숏과 fixtures/variants/variants.json의 상태 변형을 경로별 응답으로 풀어 준다.
// 뒤 단계의 view-model·resource 테스트가 쓰고, Playwright sweep(F1)은 같은 JSON을 직접 읽는다.
const fs = require('node:fs');
const path = require('node:path');

const FIXTURES = path.resolve(__dirname, '..', 'fixtures');
const VARIANTS_DIR = path.join(FIXTURES, 'variants');
const INJECT_KEYS = new Set(['route', 'status', 'delay_ms', 'abort', 'body', 'set', 'status_sequence']);
const AFTER_LOAD_KEYS = new Set(['advance_ms', 'inject', 'offline']);
const KST_OFFSET_MS = 9 * 60 * 60 * 1000;

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function loadManifest() {
  return readJson(path.join(VARIANTS_DIR, 'variants.json'));
}

function baseBody(manifest, route) {
  const rel = manifest.routes[route];
  if (!rel) throw new Error(`fixtures: no base fixture for route ${route}`);
  return readJson(path.resolve(VARIANTS_DIR, rel));
}

/** 오프셋 없는 KST 시각 'YYYY-MM-DDTHH:MM:SS'를 epoch ms로(문자열 Date 파싱 없이). */
function kstEpochMs(kst) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?$/.exec(String(kst));
  if (!m) throw new Error(`fixtures: not a KST timestamp without offset: ${kst}`);
  const [y, mo, d, h, mi, s] = m.slice(1).map(Number);
  return Date.UTC(y, mo - 1, d, h, mi, s) - KST_OFFSET_MS;
}

function decodePointer(pointer) {
  if (typeof pointer !== 'string' || !pointer.startsWith('/')) throw new Error(`fixtures: bad JSON pointer ${pointer}`);
  return pointer.slice(1).split('/').map(part => part.replace(/~1/g, '/').replace(/~0/g, '~'));
}

const hasSlot = (parent, key) => parent !== null && typeof parent === 'object' &&
  (Array.isArray(parent) ? /^\d+$/.test(key) && Number(key) < parent.length : Object.prototype.hasOwnProperty.call(parent, key));

/** set 맵을 기준 본문 복사본에 덮어쓴다. 없는 경로는 오타로 보고 바로 실패한다. */
function applySet(body, setMap, label) {
  const copy = structuredClone(body);
  for (const [pointer, value] of Object.entries(setMap)) {
    const parts = decodePointer(pointer);
    let parent = copy;
    for (const part of parts.slice(0, -1)) {
      if (!hasSlot(parent, part)) throw new Error(`fixtures: ${label} ${pointer} has no '${part}'`);
      parent = parent[part];
    }
    const last = parts[parts.length - 1];
    if (!hasSlot(parent, last)) throw new Error(`fixtures: ${label} ${pointer} has no '${last}'`);
    parent[last] = structuredClone(value);
  }
  return copy;
}

function checkKeys(object, allowed, label) {
  for (const key of Object.keys(object)) if (!allowed.has(key)) throw new Error(`fixtures: ${label} unknown key '${key}'`);
}

/**
 * 변형 하나를 { clockKst, clockMs, responses, afterLoad }로 푼다.
 * responses[route] = { status, body, delayMs?, abort?, statusSequence? } (기준 경로는 모두 200 + 기준 본문으로 시작)
 */
function resolveVariant(id, manifest = loadManifest()) {
  const variant = manifest.variants.find(v => v.id === id);
  if (!variant) throw new Error(`fixtures: unknown variant ${id}`);
  const responses = {};
  for (const route of Object.keys(manifest.routes)) responses[route] = { status: 200, body: baseBody(manifest, route) };
  for (const item of variant.inject) {
    checkKeys(item, INJECT_KEYS, `${id} inject`);
    if (item.route === '*') throw new Error(`fixtures: ${id} '*' is only allowed after load`);
    const entry = responses[item.route] || (responses[item.route] = { status: 200, body: null });
    if (Object.prototype.hasOwnProperty.call(item, 'body')) entry.body = structuredClone(item.body);
    if (item.set) entry.body = applySet(entry.body, item.set, `${id} ${item.route}`);
    if (item.status !== undefined) entry.status = item.status;
    if (item.delay_ms !== undefined) entry.delayMs = item.delay_ms;
    if (item.abort) entry.abort = true;
    if (item.status_sequence) entry.statusSequence = [...item.status_sequence];
  }
  let afterLoad = null;
  if (variant.after_load) {
    checkKeys(variant.after_load, AFTER_LOAD_KEYS, `${id} after_load`);
    (variant.after_load.inject || []).forEach(item => checkKeys(item, INJECT_KEYS, `${id} after_load.inject`));
    afterLoad = structuredClone(variant.after_load);
  }
  const clockKst = (variant.clock || manifest.default_clock).kst;
  return { id, variant, clockKst, clockMs: kstEpochMs(clockKst), responses, afterLoad };
}

module.exports = { FIXTURES, loadManifest, baseBody, resolveVariant, applySet, kstEpochMs };
