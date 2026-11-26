// 긴 설명 글 로더(명세 4.4절). Metro가 한글을 \uXXXX(한 글자 6바이트)로 바꾸므로 둘러보기·용어 사전·산출 방식 본문은
// 번들 대신 public/content/guide.ko.json에 두고, 처음 필요할 때 상대 경로로 한 번 읽는다. 모양(키·종류·길이)이 틀리면 실패다.
import { GLOSSARY_TITLES } from './strings';
import type { GlossaryId } from './strings';

/** 경로 깊이가 1이라 어느 경로에서든 기준 폴더로 풀린다(3.1절). */
export const GUIDE_PATH = 'content/guide.ko.json';
export const TOUR_STEPS = 5;
export const TOPIC_IDS = ['reading', 'method', 'sources', 'about'] as const;
export const METHOD_CARD_IDS = [
  'today-price', 'today-power', 'today-forecast', 'today-accuracy', 'bid-deadline', 'bid-ladder', 'bid-merit', 'settle-sample',
] as const;
/** 숫자 읽는 법(4.4절)의 표시별 설명: 4상태 · 일반 칩 · 줄표 · 지연 · 끊김 · 시각. */
export const LEGEND_KEYS = ['measured', 'forecast', 'estimated', 'example', 'plain', 'missing', 'stale', 'broken', 'time'] as const;
export const GLOSSARY_SHORT_MAX = 60;
const TEXT_MAX = 600;
const PARAGRAPHS_MAX = 12;

export type TopicId = (typeof TOPIC_IDS)[number];
export type MethodCardId = (typeof METHOD_CARD_IDS)[number];
export type LegendKey = (typeof LEGEND_KEYS)[number];
/** detail: 짝·검증 규칙 같은 구현 설명 한 줄(D7). 뜻과 읽는 법(body) 뒤에 흐린 "자세히" 줄로 그린다. 없어도 된다. */
export type GuideSection = { title: string; body: string[]; detail?: string };
export type GlossaryEntry = { id: GlossaryId; short: string; body: string[]; detail?: string };
export type GuideContent = {
  version: 1; tour: GuideSection[]; glossary: GlossaryEntry[];
  topics: Record<TopicId, GuideSection>; cards: Record<MethodCardId, GuideSection>;
  /** 표시별 설명 한 문장씩. simplified: 예측 방법 화면의 "단순화한 설명" 목록(마감 · 이중 정산 · 가격 원리 · 실시간시장 구간, P7). */
  legend: Record<LegendKey, string>; simplified: GuideSection;
};

export class GuideLoadError extends Error {
  readonly reason: 'http' | 'network' | 'shape';
  constructor(reason: 'http' | 'network' | 'shape', detail: string) {
    super(`설명을 불러오지 못했습니다 (${reason}: ${detail})`);
    this.name = 'GuideLoadError';
    this.reason = reason;
  }
}

type Rec = Record<string, unknown>;
const bad = (path: string, expected: string): never => {
  throw new GuideLoadError('shape', `${path} ${expected}`);
};
const rec = (v: unknown, path: string): Rec => (typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Rec) : bad(path, 'object'));

/** 숫자는 화면 값에서만 나온다: JSON 글자에 숫자가 있으면 실패(4.4절). */
function text(v: unknown, path: string, max = TEXT_MAX): string {
  if (typeof v !== 'string' || !v.trim()) return bad(path, 'non-empty string');
  if (v.length > max) return bad(path, `≤ ${max} chars`);
  if (/[0-9]/.test(v)) return bad(path, 'no digits');
  return v;
}

function paragraphs(v: unknown, path: string): string[] {
  if (!Array.isArray(v) || v.length < 1 || v.length > PARAGRAPHS_MAX) return bad(path, `1–${PARAGRAPHS_MAX} paragraphs`);
  return v.map((p, i) => text(p, `${path}[${i}]`));
}

/** 있으면 글자 규칙(숫자 없음·길이)을 지켜야 한다. 없으면 키 자체를 두지 않는다. */
const detail = (v: unknown, path: string): { detail?: string } => (v === undefined ? {} : { detail: text(v, path) });

function section(v: unknown, path: string): GuideSection {
  const r = rec(v, path);
  return { title: text(r.title, `${path}.title`), body: paragraphs(r.body, `${path}.body`), ...detail(r.detail, `${path}.detail`) };
}

/** 키 집합이 정확히 같아야 한다(빠진 항목·모르는 항목 모두 실패). */
function keyed<K extends string, V>(v: unknown, keys: readonly K[], path: string, read: (x: unknown, at: string) => V): Record<K, V> {
  const r = rec(v, path);
  const actual = Object.keys(r).sort().join(',');
  if (actual !== [...keys].sort().join(',')) bad(path, `keys ${keys.join(',')}`);
  return Object.fromEntries(keys.map(k => [k, read(r[k], `${path}.${k}`)])) as Record<K, V>;
}

export function parseGuide(json: unknown): GuideContent {
  const r = rec(json, '$');
  if (r.version !== 1) bad('version', '=== 1');
  if (!Array.isArray(r.tour) || r.tour.length !== TOUR_STEPS) bad('tour', `${TOUR_STEPS} steps`);
  const tour = (r.tour as unknown[]).map((s, i) => section(s, `tour[${i}]`));
  const ids = Object.keys(GLOSSARY_TITLES) as GlossaryId[];
  if (!Array.isArray(r.glossary) || r.glossary.length !== ids.length) bad('glossary', `${ids.length} entries`);
  const glossary = (r.glossary as unknown[]).map((g, i): GlossaryEntry => {
    const e = rec(g, `glossary[${i}]`);
    if (typeof e.id !== 'string' || !ids.includes(e.id as GlossaryId)) bad(`glossary[${i}].id`, 'known term id');
    return {
      id: e.id as GlossaryId, short: text(e.short, `glossary[${i}].short`, GLOSSARY_SHORT_MAX), body: paragraphs(e.body, `glossary[${i}].body`),
      ...detail(e.detail, `glossary[${i}].detail`),
    };
  });
  if (new Set(glossary.map(g => g.id)).size !== ids.length) bad('glossary', 'unique ids');
  return {
    version: 1, tour, glossary, topics: keyed(r.topics, TOPIC_IDS, 'topics', section), cards: keyed(r.cards, METHOD_CARD_IDS, 'cards', section),
    legend: keyed(r.legend, LEGEND_KEYS, 'legend', (x, at) => text(x, at)), simplified: section(r.simplified, 'simplified'),
  };
}

/** base는 감지한 기준 경로('/mobile/' 또는 '/'). */
export function guideUrl(base: string): string {
  return `${base.endsWith('/') ? base : `${base}/`}${GUIDE_PATH}`;
}

type GuideFetch = (url: string, init: { method: 'GET' }) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;

/** 기준 경로마다 한 번만 읽는다. 실패하면 기억을 지워 [다시 시도]가 새로 요청하게 한다. */
export function createGuideLoader(fetchImpl: GuideFetch = (url, init) => fetch(url, init)) {
  const cache = new Map<string, Promise<GuideContent>>();
  async function fetchGuide(url: string): Promise<GuideContent> {
    let response: Awaited<ReturnType<GuideFetch>>;
    try {
      response = await fetchImpl(url, { method: 'GET' });
    } catch {
      throw new GuideLoadError('network', url);
    }
    if (!response.ok) throw new GuideLoadError('http', String(response.status));
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new GuideLoadError('shape', 'JSON');
    }
    return parseGuide(body);
  }
  return {
    load(base: string): Promise<GuideContent> {
      const url = guideUrl(base);
      const hit = cache.get(url);
      if (hit) return hit;
      const pending = fetchGuide(url);
      cache.set(url, pending);
      pending.catch(() => cache.delete(url));
      return pending;
    },
  };
}
