// 디자인 토큰의 단일 원천(명세 5절). 새 코드에서 색 hex는 이 파일에만 둔다(7.10절 금지 검사 예외).
// 가져오기가 없어야 한다: 헤더 VM 테스트와 대비 테스트가 이 파일을 그대로 읽는다.
export type ThemeName = 'light' | 'dark';
export type ColorRole = 'canvas' | 'surface' | 'well' | 'border' | 'borderStrong' | 'text1' | 'text2' | 'text3' | 'link'
  | 'primary' | 'onPrimary' | 'nowPillBg' | 'nowPillText' | 'scrim';
export type ChipKind = 'measured' | 'forecast' | 'estimated' | 'example' | 'plain';
export type PillTone = 'ok' | 'watch' | 'caution' | 'danger' | 'neutral';
export type ChipTone = { text: string; bg: string; border: string | null; borderStyle: 'solid' | 'dashed' | 'dotted'; hatch: string | null };

// 데이터 색은 두 테마에서 같다(5.3절, 대시보드 SERIES). 경보 빨강은 공개 화면에 쓰지 않아 두지 않는다.
export const SERIES = {
  demand: '#3B82F6', supply: '#64748B', reserve: '#16A34A', smp: '#C026D3', band: 'rgba(192,38,211,.13)', watch: '#B45309',
} as const;

export type Theme = {
  color: Record<ColorRole, string>;
  elevation: { card: string | null; sheet: string | null; sheetBorder: string | null };
  chip: Record<ChipKind, ChipTone>;
  pill: Record<PillTone, { text: string; bg: string }>;
  series: typeof SERIES;
  chart: { pastShade: string; zeroHatch: string };
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
  chart: { pastShade: 'rgba(86,101,122,.06)', zeroHatch: 'rgba(86,101,122,.22)' },
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
  chart: { pastShade: 'rgba(148,163,184,.06)', zeroHatch: 'rgba(148,163,184,.22)' },
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
export const RADIUS = { card: 14, control: 10, pill: 999, sheet: 20 } as const;
export const MOTION = { sheetMs: 200, shimmerMs: 1200 } as const;
/** 모든 상호작용 요소의 DOM 상자 최소 크기(M2). */
export const HIT = 44;
export const LAYOUT = {
  header: 56, tabBar: 56, offlineBanner: 40, gutter: 16, gutterNarrow: 12, narrowBelow: 360, rail: 240, column: 640,
  railFrom: 1024, panel: 320, panelFrom: 1280, sheetPanel: 400, sheetMaxHeight: 0.85, sheetCenterFrom: 600,
} as const;
/** 부품 치수(6.2절): 칩·pill 높이 24, 목록 행 56, 탭 위 표시 3, 아이콘 크기. */
export const SIZE = {
  chip: 24, listRow: 56, tabMark: 3, dot: 8, icon: 22, chipIcon: 14, buttonIcon: 20, skeletonLine: 12, stateMin: 96, cellMin: 72,
  /** 표의 선택 행 왼쪽 표시(4.1절 "왼쪽 2 px primary"). */
  selectedEdge: 2,
} as const;

/**
 * 차트 치수(4.1·4.2·6.2절). 차트 폭은 카드 폭(296·358·398·608)이고 높이는 그 폭 구간으로 고른다.
 * glyph는 12 px 시스템 글꼴의 글자 폭 추정값이다. 실제보다 넉넉하게 잡아 라벨 상자가 글자보다 좁아지지 않게 한다.
 */
export const CHART = {
  edge: 12, tickGap: 4, topRow: 26, xRow: 24, xLabelTop: 6, noteRow: 22, ladderTop: 40, labelGap: 2,
  line: 2, axis: 1, grid: 1, nowLine: 1.5, dash: '6 4', dotR: 5, dotRing: 2, selMark: 3, barGap: 1,
  pill: 20, pillTop: 3, pillPad: 8, hatch: 6, hatchLine: 2, swatch: 12, yIntervals: 5,
  glyph: { hangul: 12.5, wide: 11.5, space: 3.5, other: 7.5 },
  stepHeights: [{ below: 320, height: 200 }, { below: 380, height: 220 }, { below: 500, height: 230 }, { below: Infinity, height: 260 }],
  ladderHeights: [{ below: 320, height: 180 }, { below: 500, height: 200 }, { below: Infinity, height: 240 }],
  bar: 24, barTick: 4, triangle: 7, barStroke: 1.5,
} as const;

// 실제로 쓰는 글자·바탕 쌍(M4). tokens-contrast 테스트가 두 테마에서 모두 계산한다. 경로는 Theme 객체 기준.
export type ContrastPair = { fg: string; bg: string; overlay?: string; min: 4.5 | 3; use: string };
const onAll = (fg: string, use: string, min: 4.5 | 3 = 4.5): ContrastPair[] =>
  ['color.canvas', 'color.surface', 'color.well'].map(bg => ({ fg, bg, min, use }));
const chipPairs = (['measured', 'forecast', 'estimated', 'example', 'plain'] as const).map((k): ContrastPair => ({
  fg: `chip.${k}.text`, bg: `chip.${k}.bg`, overlay: k === 'estimated' ? `chip.${k}.hatch` : undefined, min: 4.5, use: `칩 ${k}`,
}));
const pillPairs = (['ok', 'watch', 'caution', 'danger', 'neutral'] as const).map((k): ContrastPair => ({
  fg: `pill.${k}.text`, bg: `pill.${k}.bg`, min: 4.5, use: `pill ${k}`,
}));
const seriesPairs = (['demand', 'supply', 'reserve', 'smp', 'watch'] as const).map((k): ContrastPair => ({
  fg: `series.${k}`, bg: 'color.surface', min: 3, use: `차트 계열 ${k} (카드 위)`,
}));
// 차트 덧칠 위의 선·점(F5): 계단선·확정 가격 점·선택 고리·지금선이 80% 띠나 지난 시간 음영 위에 놓여도 3:1을 넘는다.
const chartPairs: ContrastPair[] = [
  { fg: 'series.smp', bg: 'color.surface', overlay: 'series.band', min: 3, use: '계단선 · 80% 띠 위' },
  { fg: 'series.smp', bg: 'color.surface', overlay: 'chart.pastShade', min: 3, use: '계단선 · 지난 시간 음영 위' },
  { fg: 'color.text1', bg: 'color.surface', overlay: 'series.band', min: 3, use: '확정 가격 점 · 띠 위' },
  { fg: 'color.primary', bg: 'color.surface', overlay: 'series.band', min: 3, use: '선택한 시간 고리 · 띠 위' },
  { fg: 'color.borderStrong', bg: 'color.surface', overlay: 'chart.pastShade', min: 3, use: '지금선·x축 · 지난 시간 음영 위' },
];
export const CONTRAST_PAIRS: readonly ContrastPair[] = [
  ...onAll('color.text1', '제목·값·AnswerLine'),
  ...onAll('color.text2', '본문·라벨·예시 값'),
  ...onAll('color.text3', '출처·시각·단위·메타'),
  ...onAll('color.link', 'TermLink·[출처 표]·[i]'),
  { fg: 'color.text3', bg: 'color.surface', overlay: 'chart.pastShade', min: 4.5, use: '차트 지난 시간 음영 위 축 글자' },
  { fg: 'color.primary', bg: 'color.surface', min: 4.5, use: '활성 탭 라벨' },
  { fg: 'color.onPrimary', bg: 'color.primary', min: 4.5, use: '주 버튼 글자' },
  { fg: 'color.nowPillText', bg: 'color.nowPillBg', min: 4.5, use: 'NowPill·차트 지금 pill' },
  ...chipPairs,
  ...pillPairs,
  ...onAll('color.borderStrong', '입력·세그먼트 외곽, 차트 축', 3),
  ...onAll('color.primary', '포커스 링·활성 탭 표시', 3),
  ...seriesPairs,
  ...chartPairs,
];
