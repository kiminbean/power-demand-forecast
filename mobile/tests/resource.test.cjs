'use strict';
// src/lib/resource.ts(명세 6.3절, F7): 가짜 fetch·가짜 타이머로 키별 상태·중복 제거·재시도 규칙·시간 초과·마지막 좋은 값을 고정한다.
// 재시도는 네트워크 오류·502·503·시간 초과에만 1.5초 뒤 한 번, 4xx·모양 오류는 다시 보내지 않는다. 시간 초과는 10초.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { createTsLoader, SRC_ROOT } = require('./helpers/ts-loader.cjs');

const loader = createTsLoader();
const r = loader.load(path.join(SRC_ROOT, 'lib/resource.ts'));
const { ApiHttpError, ApiNetworkError } = loader.load(path.join(SRC_ROOT, 'data/client.ts'));
const { ApiShapeError } = loader.load(path.join(SRC_ROOT, 'data/types.ts'));

// 손으로 돌리는 시계: advance(ms)가 그 사이 타이머를 시각 순서대로 부르고, 사이사이 마이크로태스크를 비운다.
function fakeClock(start = 1_000_000) {
  let now = start;
  let seq = 0;
  const timers = new Map();
  const flush = async () => {
    for (let i = 0; i < 20; i += 1) await new Promise(res => setImmediate(res));
  };
  return {
    deps: {
      now: () => now,
      setTimer: (fn, ms) => {
        const id = ++seq;
        timers.set(id, { fn, at: now + ms });
        return id;
      },
      clearTimer: id => timers.delete(id),
    },
    get now() { return now; },
    pending: () => timers.size,
    flush,
    async advance(ms) {
      const end = now + ms;
      await flush();
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = due[1].at;
        due[1].fn();
        await flush();
      }
      now = end;
      await flush();
    },
  };
}

// 부를 때마다 다음 결과를 하나씩 쓰는 가짜 loader. 'hang'은 signal이 끊길 때까지 끝나지 않는다.
function script(results) {
  const calls = [];
  const fn = signal => {
    calls.push({ signal });
    const next = results.shift();
    if (next === 'hang') {
      return new Promise((_, reject) => signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))));
    }
    return next instanceof Error ? Promise.reject(next) : Promise.resolve(next);
  };
  return { fn, calls };
}

test('첫 요청: idle → loading(busy) → ready, fetchedAt은 받은 시각, 구독자에게 알린다', async () => {
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  const seen = [];
  store.subscribe('k', () => seen.push(store.get('k').status));
  assert.deepEqual(store.get('k'), { data: null, fetchedAt: null, status: 'idle', error: null, busy: false });
  // 응답이 250 ms 뒤에 온다: fetchedAt은 요청 시각이 아니라 받은 시각이다.
  const fn = () => new Promise(res => clock.deps.setTimer(() => res({ v: 1 }), 250));
  const started = clock.now;
  const done = store.load('k', fn);
  assert.deepEqual([store.get('k').status, store.get('k').busy], ['loading', true]);
  await clock.advance(250);
  assert.equal(await done, true);
  assert.deepEqual(store.get('k'), { data: { v: 1 }, fetchedAt: started + 250, status: 'ready', error: null, busy: false });
  assert.deepEqual(seen, ['loading', 'ready']);
  assert.equal(clock.pending(), 0, '시간 초과 타이머를 지운다');
});

test('같은 키는 한 번에 하나만: 진행 중에 다시 부르면 같은 약속을 돌려주고 loader는 한 번만 불린다', async () => {
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  const { fn, calls } = script([{ v: 1 }, { v: 2 }]);
  const a = store.load('k', fn);
  const b = store.load('k', fn);
  assert.equal(a, b);
  await clock.advance(10);
  assert.equal(calls.length, 1);
  await store.load('k', fn);
  assert.equal(calls.length, 2, '끝난 뒤에는 새로 보낸다');
  assert.deepEqual(store.get('k').data, { v: 2 });
});

test('재시도: 네트워크 오류·502·503·시간 초과는 1.5초 뒤 한 번만, 4xx·모양 오류는 다시 보내지 않는다', async () => {
  const cases = [
    [new ApiNetworkError('/api/v1/smp/day-ahead'), 2],
    [new ApiHttpError('/api/v1/smp/day-ahead', 502), 2],
    [new ApiHttpError('/api/v1/smp/day-ahead', 503), 2],
    [new ApiHttpError('/api/v1/smp/day-ahead', 404), 1],
    [new ApiHttpError('/api/v1/smp/day-ahead', 500), 1],
    [new ApiShapeError('/api/v1/smp/day-ahead', 'status', 'string'), 1],
  ];
  for (const [error, expectedCalls] of cases) {
    const clock = fakeClock();
    const store = r.createResourceStore(clock.deps);
    const { fn, calls } = script([error, error]);
    const done = store.load('k', fn);
    await clock.advance(1_499);
    assert.equal(calls.length, 1, `${error.name} ${error.status ?? ''}: 1.5초 전에는 다시 보내지 않는다`);
    await clock.advance(1);
    assert.equal(calls.length, expectedCalls, `${error.name} ${error.status ?? ''}`);
    await clock.advance(5_000);
    assert.equal(await done, false);
    assert.equal(store.get('k').status, 'error');
    assert.equal(store.get('k').error, error);
  }
  // 재시도가 성공하면 ready.
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  const { fn, calls } = script([new ApiHttpError('/x', 503), { v: 9 }]);
  const done = store.load('k', fn);
  await clock.advance(1_500);
  assert.equal(await done, true);
  assert.equal(calls.length, 2);
  assert.deepEqual([store.get('k').status, store.get('k').data], ['ready', { v: 9 }]);
});

test('시간 초과 10초: signal을 끊고 RequestTimeoutError, 재시도도 10초면 error. signal을 무시하는 loader도 끝난다', async () => {
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  const { fn, calls } = script(['hang', 'hang']);
  const done = store.load('k', fn);
  await clock.advance(9_999);
  assert.equal(calls[0].signal.aborted, false);
  await clock.advance(1);
  assert.equal(calls[0].signal.aborted, true, '10초에 끊는다');
  await clock.advance(1_500);
  assert.equal(calls.length, 2, '시간 초과는 한 번 다시 보낸다');
  await clock.advance(10_000);
  assert.equal(await done, false);
  assert.equal(store.get('k').error.name, 'RequestTimeoutError');
  // signal을 보지 않는 loader(끝나지 않는 약속)도 경주로 끝난다.
  const deaf = r.createResourceStore(clock.deps);
  const never = deaf.load('n', () => new Promise(() => {}));
  await clock.advance(21_500);
  assert.equal(await never, false);
  assert.equal(deaf.get('n').status, 'error');
});

test('실패해도 마지막 좋은 값과 그 시각은 남는다(지연 표시는 화면이 나이로 한다), 다시 성공하면 갱신', async () => {
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  const { fn } = script([{ v: 1 }, new ApiHttpError('/x', 404), { v: 3 }]);
  await store.load('k', fn);
  const firstAt = store.get('k').fetchedAt;
  await clock.advance(60_000);
  const reloading = store.load('k', fn);
  assert.deepEqual([store.get('k').status, store.get('k').busy, store.get('k').data], ['ready', true, { v: 1 }], '새로고침 중에도 값은 그대로');
  await reloading;
  assert.deepEqual([store.get('k').status, store.get('k').data, store.get('k').fetchedAt], ['error', { v: 1 }, firstAt]);
  await clock.advance(60_000);
  await store.load('k', fn);
  assert.deepEqual([store.get('k').status, store.get('k').data, store.get('k').fetchedAt], ['ready', { v: 3 }, clock.now]);
});

test('마운트 규칙: 값 없음·마지막 실패·5분 넘음만 받는다, 진행 중이면 받지 않는다', () => {
  const at = 10_000_000;
  const snap = over => ({ data: { v: 1 }, fetchedAt: at, status: 'ready', error: null, busy: false, ...over });
  assert.equal(r.shouldLoadOnMount(snap({ data: null, fetchedAt: null, status: 'idle' }), at), true);
  assert.equal(r.shouldLoadOnMount(snap({ status: 'error' }), at + 1), true);
  assert.equal(r.shouldLoadOnMount(snap(), at + r.REFRESH_EVERY_MS - 1), false);
  assert.equal(r.shouldLoadOnMount(snap(), at + r.REFRESH_EVERY_MS), true);
  assert.equal(r.shouldLoadOnMount(snap({ busy: true, data: null }), at), false);
  assert.deepEqual([r.REQUEST_TIMEOUT_MS, r.RETRY_DELAY_MS, r.REFRESH_EVERY_MS], [10_000, 1_500, 300_000]);
});

test('키는 서로 섞이지 않고, 구독을 끊으면 더 알리지 않는다', async () => {
  const clock = fakeClock();
  const store = r.createResourceStore(clock.deps);
  let n = 0;
  const off = store.subscribe('a', () => { n += 1; });
  await store.load('b', script([{ v: 'b' }]).fn);
  assert.equal(n, 0);
  assert.equal(store.get('a').status, 'idle');
  off();
  await store.load('a', script([{ v: 'a' }]).fn);
  assert.equal(n, 0);
  assert.deepEqual(store.get('a').data, { v: 'a' });
});
