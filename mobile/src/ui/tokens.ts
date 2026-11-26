// 디자인 토큰의 단일 원천(명세 5절). 새 코드에서 색 hex는 이 파일에만 둔다(7.10절 금지 검사 예외).
// 가져오기가 없어야 한다: 헤더 VM 테스트와 대비 테스트가 이 파일을 그대로 읽는다. 실제로 쓰는 글자·바탕 쌍 목록(M4)은
// ui/contrastPairs.ts에 있다(round 2: 앱이 쓰지 않는 시험 자료라 번들에서 뺐다).
export type ThemeName = 'light' | 'dark';
export type ColorRole = 'canvas' | 'surface' | 'well' | 'border' | 'borderStrong' | 'text1' | 'text2' | 'text3' | 'link'
  | 'primary' | 'onPrimary' | 'nowPillBg' | 'nowPillText' | 'scrim';
export type ChipKind = 'measured' | 'forecast' | 'estimated' | 'example' | 'plain';
export type PillTone = 'ok' | 'watch' | 'caution' | 'danger' | 'neutral';
export type ChipTone = { text: string; bg: string; border: string | null; borderStyle: 'solid' | 'dashed' | 'dotted'; hatch: string | null };

// 데이터 색은 두 테마에서 같다(5.3절, 대시보드 SERIES). 경보 빨강은 공개 화면에 쓰지 않아 두지 않는다.
// 80% 띠는 8 % 면 + 1 px 테두리(round 1 D10): 13 % 면이 그림을 덮어 기준선 점선이 묻혔다. 테두리는 자홍 그대로의 1 px 점선이다(round 2
// 접근성: 45 % 실선은 2.04:1·1.62:1로 M4 3:1에 못 미쳤다. 점선이라 기준선 파선보다 옅게 읽힌다).
export const SERIES = {
  demand: '#3B82F6', supply: '#64748B', reserve: '#16A34A', smp: '#C026D3', band: 'rgba(192,38,211,.08)', bandEdge: '#C026D3',
  watch: '#B45309',
  // 예시 그림(가격 원리 계단·정산 ①+② 그림)의 호박색 한 벌(round 2 parity: 테마마다 크림·밤색 ↔ 회색·노랑으로 색이 바뀌었다). 채움은 두 테마에서
  // 같고 외곽선만 테마의 chart.exampleEdge(3:1)다.
  exampleFill: 'rgba(245,158,11,.25)', exampleMark: '#F59E0B',
} as const;

export type Theme = {
  color: Record<ColorRole, string>;
  elevation: { card: string | null; sheet: string | null; sheetBorder: string | null };
  chip: Record<ChipKind, ChipTone>;
  pill: Record<PillTone, { text: string; bg: string }>;
  series: typeof SERIES;
  /** exampleEdge: 예시 그림 막대의 외곽선(밝은 테마 amber-700, 어두운 테마 amber-300). 채움은 SERIES.exampleFill·exampleMark(round 2). */
  chart: { pastShade: string; zeroHatch: string; exampleEdge: string };
};

const light: Theme = {
  color: {
    canvas: '#F6F8FB', surface: '#FFFFFF', well: '#F1F5F9', border: '#E2E8F0', borderStrong: '#7B8798', text1: '#0F172A',
    text2: '#334155', text3: '#56657A', link: '#1D4ED8', primary: '#2563EB', onPrimary: '#FFFFFF', nowPillBg: '#475569',
    nowPillText: '#FFFFFF', scrim: 'rgba(15,23,42,.40)',
  },
  elevation: { card: '0 1px 2px rgba(15,23,42,.06)', sheet: '0 8px 24px rgba(15,23,42,.16)', sheetBorder: null },
  chip: {
    measured: { text: '#334155', bg: '#FFFFFF', border: '#94A3B8', borderStyle: 'solid', hatch: null },
    forecast: { text: '#1D4ED8', bg: '#EFF6FF', border: '#3B82F6', borderStyle: 'dashed', hatch: null },
    estimated: { text: '#334155', bg: '#F1F5F9', border: '#94A3B8', borderStyle: 'dotted', hatch: 'rgba(100,116,139,.22)' },
    example: { text: '#92400E', bg: '#FFFBEB', border: 'rgba(217,119,6,.55)', borderStyle: 'dotted', hatch: null },
    plain: { text: '#334155', bg: '#F1F5F9', border: null, borderStyle: 'solid', hatch: null },
  },
  pill: {
    ok: { text: '#166534', bg: '#DCFCE7' }, watch: { text: '#92400E', bg: '#FEF3C7' }, caution: { text: '#9A3412', bg: '#FFEDD5' },
    danger: { text: '#B91C1C', bg: '#FEE2E2' }, neutral: { text: '#334155', bg: '#F1F5F9' },
  },
  series: SERIES,
  // 지난 시간 음영 = text3 6 %, 0.00 입력 빗금 = text3 22 %(5.3절)
  chart: { pastShade: 'rgba(86,101,122,.06)', zeroHatch: 'rgba(86,101,122,.22)', exampleEdge: '#B45309' },
};

// dark는 그림자 대신 면 단계로 구분하고, 시트는 1 px borderStrong 테두리를 쓴다(5.1절).
const dark: Theme = {
  color: {
    canvas: '#0B1220', surface: '#121B31', well: '#1A2439', border: '#24324D', borderStrong: '#6B7A93', text1: '#F1F5F9',
    text2: '#CBD5E1', text3: '#94A3B8', link: '#60A5FA', primary: '#60A5FA', onPrimary: '#0B1220', nowPillBg: '#475569',
    nowPillText: '#FFFFFF', scrim: 'rgba(0,0,0,.55)',
  },
  elevation: { card: null, sheet: null, sheetBorder: '#6B7A93' },
  chip: {
    measured: { text: '#CBD5E1', bg: '#121B31', border: '#64748B', borderStyle: 'solid', hatch: null },
    forecast: { text: '#93C5FD', bg: '#1B2C49', border: '#60A5FA', borderStyle: 'dashed', hatch: null },
    estimated: { text: '#CBD5E1', bg: '#1A2439', border: '#64748B', borderStyle: 'dotted', hatch: 'rgba(148,163,184,.22)' },
    example: { text: '#FCD34D', bg: '#2D2B2C', border: 'rgba(245,158,11,.55)', borderStyle: 'dotted', hatch: null },
    plain: { text: '#CBD5E1', bg: '#1A2439', border: null, borderStyle: 'solid', hatch: null },
  },
  pill: {
    ok: { text: '#4ADE80', bg: '#1A363C' }, watch: { text: '#FBBF24', bg: '#33322F' }, caution: { text: '#FB923C', bg: '#332C33' },
    danger: { text: '#F87171', bg: '#37293B' }, neutral: { text: '#CBD5E1', bg: '#1A2439' },
  },
  series: SERIES,
  chart: { pastShade: 'rgba(148,163,184,.06)', zeroHatch: 'rgba(148,163,184,.22)', exampleEdge: '#FCD34D' },
};

export const THEMES: Readonly<Record<ThemeName, Theme>> = { light, dark };

// 글자: 시스템 글꼴, 화면당 5크기(5.4절). 줄높이는 RN이 px만 받으므로 배수를 px로 바꿔 둔다.
export const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif';
export type TextVariant = 'display' | 'title' | 'body' | 'label' | 'meta';
export type FontWeight = '400' | '500' | '600' | '700';
export const TYPE_SCALE: Readonly<Record<TextVariant, { fontSize: number; lineHeight: number; weights: readonly FontWeight[] }>> = {
  display: { fontSize: 32, lineHeight: 37, weights: ['700'] },
  title: { fontSize: 20, lineHeight: 26, weights: ['700'] },
  body: { fontSize: 15, lineHeight: 23, weights: ['400', '500', '600', '700'] },
  label: { fontSize: 13, lineHeight: 18, weights: ['400', '600'] },
  meta: { fontSize: 12, lineHeight: 17, weights: ['500', '600'] },
};
/** 폭 360 이하에서 display는 28로 줄인다. */
export const DISPLAY_COMPACT = { maxWidth: 360, fontSize: 28, lineHeight: 32 } as const;

export const SPACE = { xs: 4, sm: 8, md: 12, lg: 16, xl: 20, xxl: 24, x3: 32, x4: 48 } as const;
/** mark: 상태 pill의 작은 네모 표지(둥근 ●는 실측·확정 표시라서 쓰지 않는다, round 2). */
export const RADIUS = { card: 14, control: 10, pill: 999, sheet: 20, mark: 2 } as const;
export const MOTION = { sheetMs: 200, shimmerMs: 1200 } as const;
/** 모든 상호작용 요소의 DOM 상자 최소 크기(M2). */
export const HIT = 44;
/**
 * rowCardsBelow: 이 폭 아래에서는 카드 안 표(오늘 시간별·입찰 구간별)를 행 카드로 쌓는다(round 1 D2, 휴대폰 = 600 미만, 4.6절).
 * twoUpFrom–railFrom: 오늘의 가격·전력 카드를 두 칸으로(round 1 D14, 태블릿).
 */
export const LAYOUT = {
  header: 56, tabBar: 56, offlineBanner: 40, gutter: 16, gutterNarrow: 12, narrowBelow: 360, rail: 240, column: 640,
  railFrom: 1024, panel: 320, panelFrom: 1280, sheetPanel: 400, sheetMaxHeight: 0.85, sheetCenterFrom: 600, rowCardsBelow: 600, twoUpFrom: 768,
  /** 오늘 AnswerLine이 두 줄에 드는 폭(그 아래는 세 줄, round 2 CLS). */
  answerTwoLinesFrom: 390,
} as const;
/** 오늘 AnswerLine과 가격 카드 비교 문단이 다 받은 뒤 차지하는 줄 수(390 미만 세 줄, 그 밖 두 줄): 받는 동안 그만큼 자리를 잡는다(round 2 CLS). */
export const answerLines = (width: number) => (width < LAYOUT.answerTwoLinesFrom ? 3 : 2);
/**
 * 오늘 카드 스켈레톤 높이(StateView minHeight, round 2 CLS): 다 받은 카드 본문 높이를 폭 구간마다 잰 값(F14 card_heights.py, round 0
 * 스냅숏 꼴의 실측 자료). 구간 [360 미만, 390 미만, 600 미만, 768–1023(가격·전력 두 칸), 그 밖]. 불러오는 동안과 받은 뒤의 높이가 같아야
 * 아래가 밀리지 않는다.
 */
export const SKELETON = {
  hero: [222, 191, 173, 355, 173], power: [386, 281, 263, 355, 263], forecast: [677, 669, 617, 536, 536], accuracy: [228, 190, 190, 127, 127],
} as const;
/** SKELETON의 칸: 구간 경계는 LAYOUT의 배치 경계 그대로다(좁은 폭 · AnswerLine 두 줄 · 휴대폰 행 카드 · 태블릿 두 칸). */
export function skeletonBucket(width: number): number {
  if (width >= LAYOUT.twoUpFrom && width < LAYOUT.railFrom) return 3;
  const below = [LAYOUT.narrowBelow, LAYOUT.answerTwoLinesFrom, LAYOUT.rowCardsBelow].findIndex(edge => width < edge);
  return below === -1 ? 4 : below;
}
/** 부품 치수(6.2절): 칩·pill 높이 24, 목록 행 56, 탭 위 표시 3, 아이콘 크기. cellMinNarrow: 360 미만 시간대 요약 표(4칸이 272 px에 든다). */
export const SIZE = {
  chip: 24, listRow: 56, tabMark: 3, dot: 8, icon: 22, chipIcon: 14, buttonIcon: 20, skeletonLine: 12, stateMin: 96, cellMin: 72, cellMinNarrow: 56,
  /** 행 카드의 시간·구간 칸 폭(행 카드 왼쪽 열, "06–07시"·"10구간"이 한 줄). */
  rowHead: 68,
  /** 넘치는 표의 오른쪽 가장자리 흐림 폭. */
  edgeFade: 24,
  /** 표의 선택 행 왼쪽 표시(4.1절 "왼쪽 2 px primary"). */
  selectedEdge: 2,
} as const;

/**
 * 차트 치수(4.1·4.2·6.2절). 차트 폭은 카드 폭(296·358·398·608)이고 높이는 그 폭 구간으로 고른다.
 * glyph는 12 px 시스템 글꼴의 글자 폭 추정값이다. 실제보다 넉넉하게 잡아 라벨 상자가 글자보다 좁아지지 않게 한다.
 */
/**
 * stepIntervals: 오늘 차트의 y 칸 수 상한(round 1 D10: y 위쪽은 기준선·확정 가격으로 정해 눈금이 적다). clipMark: 띠가 그림 위로 넘친
 * 시간의 ▲ 표시 높이. barStrokeLadder: 입찰 계단 외곽선(D12, 막대를 채우지 않고 빗금 + 외곽선).
 */
export const CHART = {
  edge: 12, tickGap: 4, topRow: 26, xRow: 24, xLabelTop: 6, noteRow: 22, ladderTop: 40, labelGap: 2,
  line: 2, axis: 1, grid: 1, nowLine: 1.5, dash: '6 4', dotR: 5, dotRing: 2, selMark: 3, barGap: 1,
  pill: 20, pillTop: 3, pillPad: 8, hatch: 6, hatchLine: 2, swatch: 12, yIntervals: 5, stepIntervals: 4, clipMark: 6, barStrokeLadder: 1.5,
  /**
   * round 2: 띠가 넘친 시간의 ▲ 줄(그림 밖 12), 잘린 띠 끝을 흐리는 높이(10), 띠 테두리 점선, 범례 기호 상자(16×12)와 그 안의 작은 꼴
   * (파선 · 점 반지름 · 띠 높이: 차트의 것을 상자에 맞게 줄였다).
   */
  clipRow: 12, clipFade: 10, bandDash: '1 2', keyW: 16, keyH: 12, keyDash: '4 3', keyDot: 4, keyBand: 7,
  glyph: { hangul: 12.5, wide: 11.5, space: 3.5, other: 7.5 },
  stepHeights: [{ below: 320, height: 200 }, { below: 380, height: 220 }, { below: 500, height: 230 }, { below: Infinity, height: 260 }],
  ladderHeights: [{ below: 320, height: 180 }, { below: 500, height: 200 }, { below: Infinity, height: 240 }],
  bar: 24, barTick: 4, triangle: 7, barStroke: 1.5,
} as const;
