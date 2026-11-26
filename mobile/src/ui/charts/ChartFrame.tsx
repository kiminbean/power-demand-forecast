// 차트 공용 부품(F5): 폭 재기, 그림 위에 겹쳐 쓰는 12 px 라벨, 빗금 무늬, 카드 폭으로 넓히기, 키·누름을 받는 틀.
// 차트는 응답자(responder)·터치 처리기를 달지 않는다: 세로로 끌면 페이지가 스크롤되고, 탭은 브라우저 click으로만 받는다.
import React, { useCallback, useId, useState } from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import type { LayoutChangeEvent } from 'react-native';
import { Pattern, Rect } from 'react-native-svg';
import { CHART, LAYOUT, TYPE_SCALE } from '../tokens';
import type { FontWeight } from '../tokens';
import { CARD_BORDER, cardPadding } from '../Card';
import { Text } from '../Text';
import type { TextTone } from '../Text';
import { keyOf, prevent, web, webStyle } from '../web';
import { keyStep } from './chartMath';

/** 라벨 상자 높이 = meta 글자 줄높이(12 px 글자). */
export const LABEL_HEIGHT = TYPE_SCALE.meta.lineHeight;

/**
 * 차트 상자 폭. 재기 전에는 화면 폭에서 짐작한 폭(본문 열 최대 640 − 화면 여백, 카드 안쪽이면 카드 여백·테두리도 뺀다)으로
 * 그려 첫 그림이 실제와 같게 한다. 고정 폭(테스트·둘러보기)을 주면 재지 않는다.
 */
export function useChartWidth(fixed?: number, insideCard = false): { width: number; onLayout?: (e: LayoutChangeEvent) => void } {
  const { width: windowWidth } = useWindowDimensions();
  const [measured, setMeasured] = useState(0);
  const onLayout = useCallback((e: LayoutChangeEvent) => {
    const next = Math.floor(e.nativeEvent.layout.width);
    setMeasured(prev => (prev === next ? prev : next));
  }, []);
  if (fixed !== undefined) return { width: fixed };
  const gutter = windowWidth < LAYOUT.narrowBelow ? LAYOUT.gutterNarrow : LAYOUT.gutter;
  const inset = insideCard ? 2 * (cardPadding(windowWidth) + CARD_BORDER) : 0;
  return { width: measured || Math.max(0, Math.min(windowWidth, LAYOUT.column) - 2 * gutter - inset), onLayout };
}

/** SVG 무늬 id(문서 전체에서 하나). useId 글자의 콜론은 url(#…)에 쓰지 않는다. */
export const useSvgId = (prefix: string) => `${prefix}-${useId().replace(/[^A-Za-z0-9_-]/g, '')}`;

/** 45° 빗금 무늬(0.00 입력 열, 예비력 구간). */
export function Hatch({ id, color }: { id: string; color: string }) {
  return (
    <Pattern id={id} patternUnits="userSpaceOnUse" width={CHART.hatch} height={CHART.hatch} patternTransform="rotate(45)">
      <Rect x={0} y={0} width={CHART.hatchLine} height={CHART.hatch} fill={color} />
    </Pattern>
  );
}

export type ChartLabelProps = {
  text: string; left: number; top: number; width: number; meta: string;
  align?: 'left' | 'center' | 'right'; tone?: TextTone; weight?: FontWeight; testID?: string;
};
/** 그림 위 12 px 라벨(눈금·단위·각주·확률). 숫자가 들어가므로 data-pd-meta를 단다(I2). */
export function ChartLabel({ text, left, top, width, meta, align = 'center', tone = 'text3', weight = '500', testID }: ChartLabelProps) {
  return (
    <View style={[styles.label, { left, top, width }]} testID={testID}>
      <Text variant="meta" weight={weight} tone={tone} nowrap dataSet={{ pdMeta: meta }} style={{ textAlign: align }}>{text}</Text>
    </View>
  );
}

/** 차트를 카드 안쪽 여백과 테두리 끝까지 넓혀 카드 폭으로 그린다. Card의 바로 아래 자식으로 둔다. */
export function ChartBleed({ children }: { children: React.ReactNode }) {
  const { width } = useWindowDimensions();
  return <View style={{ marginHorizontal: -(cardPadding(width) + CARD_BORDER) }}>{children}</View>;
}

type PickOptions = { min: number; max: number; current: number | null; pick: (value: number) => void; fromX: (x: number) => number | null };
/**
 * 초점을 받는 차트 틀의 웹 처리기: ←/→(양 끝에서 멈춤)·Home·End와 click(탭)만. 고른 것이 없으면 → 는 처음, ← 는 처음.
 * click의 x는 틀 상자 기준이다(currentTarget). 터치 이동은 건드리지 않아 세로로 끌면 페이지가 스크롤된다.
 */
export function pickHandlers({ min, max, current, pick, fromX }: PickOptions): object {
  const onKeyDown = (e: unknown) => {
    const next = keyStep(keyOf(e), current ?? min - 1, min, max);
    if (next === null) return;
    prevent(e);
    pick(next);
  };
  const onClick = (e: unknown) => {
    const ev = (e ?? {}) as { clientX?: unknown; currentTarget?: { getBoundingClientRect?: () => { left: number } } };
    const target = ev.currentTarget;
    if (typeof ev.clientX !== 'number' || !target || typeof target.getBoundingClientRect !== 'function') return;
    const value = fromX(ev.clientX - target.getBoundingClientRect().left);
    if (value !== null) pick(value);
  };
  return web({ tabIndex: 0, onKeyDown, onClick });
}

export const chartStyles = StyleSheet.create({
  // overflow hidden: 라벨이 실수로 차트 밖으로 나가도 문서 가로 넘침을 만들지 않는다(초점 링은 틀 밖에 그려져 잘리지 않는다).
  frame: { width: '100%', overflow: 'hidden' },
  layer: { position: 'absolute', left: 0, top: 0, right: 0, bottom: 0 },
  pointer: webStyle({ cursor: 'pointer' }),
});

const styles = StyleSheet.create({
  label: { position: 'absolute', height: LABEL_HEIGHT },
});
