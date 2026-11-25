// 아이콘 한 벌(명세 5.6절): 대시보드 세트 docs/2026-11-23-dashboard-ui-top1/icons.json(24×24, stroke 1.8, 둥근 끝·이음, 채움 없음)에서
// 쓰는 7개를 그대로 옮기고, 같은 규칙의 새 아이콘 6개를 더한다. lock의 <rect>는 Path·Circle만 쓰려고 같은 모양의 path로 바꿨다.
export type IconShape = { kind: 'path'; d: string } | { kind: 'circle'; cx: number; cy: number; r: number };

const p = (d: string): IconShape => ({ kind: 'path', d });
const c = (cx: number, cy: number, r: number): IconShape => ({ kind: 'circle', cx, cy, r });

export const ICONS = {
  chart: [p('M3 3v18h18'), p('M7 15l4-4 3 3 5-6')],
  clock: [c(12, 12, 9), p('M12 7v5l3 2')],
  list: [p('M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01')],
  refresh: [p('M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5')],
  info: [c(12, 12, 9), p('M12 11v5M12 8h.01')],
  alert: [p('M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'), p('M12 9v4M12 17h.01')],
  chev: [p('M9 6l6 6-6 6')],
  back: [p('M15 6l-6 6 6 6')],
  close: [p('M6 6l12 12M18 6 6 18')],
  lock: [p('M7 11h10a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2z'), p('M8 11V7a4 4 0 0 1 8 0v4')],
  book: [p('M4 4h11a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3z'), p('M4 17a3 3 0 0 1 3-3h11')],
  contrast: [c(12, 12, 9), p('M12 3v18')],
  offline: [p('M3 3l18 18'), p('M8.5 16.5a5 5 0 0 1 7 0M5 12.9a10 10 0 0 1 5.2-2.7M19 12.9a10 10 0 0 0-2.5-1.7M12 20h.01')],
} as const satisfies Record<string, readonly IconShape[]>;

export type IconName = keyof typeof ICONS;
export const ICON_VIEWBOX = '0 0 24 24';
export const ICON_STROKE_WIDTH = 1.8;
