// 실제로 쓰는 글자·바탕 쌍(M4, 명세 5절). tests/tokens-contrast가 두 테마에서 모두 계산한다. 경로는 Theme 객체(ui/tokens.ts) 기준이다.
// 앱 코드는 이 파일을 가져오지 않는다: 시험 자료라서 번들에 넣지 않는다(round 2 F14: tokens.ts에 있으면 한국어 쓰임 글자까지 번들에 실렸다).
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
  { fg: 'series.bandEdge', bg: 'color.surface', min: 3, use: '80% 띠 테두리(1 px 점선, round 2)' },
  { fg: 'series.bandEdge', bg: 'color.surface', overlay: 'chart.pastShade', min: 3, use: '80% 띠 테두리 · 지난 시간 음영 위' },
  { fg: 'chart.exampleEdge', bg: 'color.surface', min: 3, use: '예시 그림 막대 외곽선(round 2)' },
  { fg: 'chart.exampleEdge', bg: 'color.surface', overlay: 'series.exampleFill', min: 3, use: '예시 그림 외곽선 · 옅은 호박 채움 위' },
];
export const CONTRAST_PAIRS: readonly ContrastPair[] = [
  ...onAll('color.text1', '제목·값·AnswerLine'),
  ...onAll('color.text2', '본문·라벨·예시 값'),
  ...onAll('color.text3', '출처·시각·단위·메타'),
  ...onAll('color.link', 'TermLink·[출처 표]·[i]'),
  { fg: 'color.text3', bg: 'color.surface', overlay: 'chart.pastShade', min: 4.5, use: '차트 지난 시간 음영 위 축 글자' },
  { fg: 'color.primary', bg: 'color.surface', min: 4.5, use: '활성 탭 라벨' },
  { fg: 'color.onPrimary', bg: 'color.primary', min: 4.5, use: '주 버튼 글자' },
  { fg: 'color.nowPillText', bg: 'color.nowPillBg', min: 4.5, use: '차트 지금 pill (h1 행 NowPill은 round 2부터 well 위 text1)' },
  ...chipPairs,
  ...pillPairs,
  ...onAll('color.borderStrong', '입력·세그먼트 외곽, 차트 축', 3),
  ...onAll('color.primary', '포커스 링·활성 탭 표시', 3),
  ...seriesPairs,
  ...chartPairs,
];
