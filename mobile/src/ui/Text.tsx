// 글자 크기는 이 컴포넌트의 variant로만 정한다(명세 5.4절, raw fontSize 금지). 숫자는 모두 tabular-nums.
import React from 'react';
import { Text as RNText, useWindowDimensions } from 'react-native';
import type { StyleProp, TextProps as RNTextProps, TextStyle } from 'react-native';
import { DISPLAY_COMPACT, FONT_FAMILY, TYPE_SCALE } from './tokens';
import type { ColorRole, FontWeight, TextVariant } from './tokens';
import { useTheme } from '../app/theme';

export type TextTone = Extract<ColorRole, 'text1' | 'text2' | 'text3' | 'link' | 'primary' | 'onPrimary' | 'nowPillText'>;
type OwnStyle = Omit<TextStyle, 'fontSize' | 'fontWeight' | 'lineHeight' | 'fontFamily'>;

export type TextProps = Omit<RNTextProps, 'style'> & {
  variant?: TextVariant;
  /** variant가 허용하지 않는 굵기는 그 variant의 첫 굵기로 바꾼다(화면당 굵기 4개 규칙). */
  weight?: FontWeight;
  tone?: TextTone;
  /** 칩·pill처럼 토큰 묶음이 정한 글자색. 주면 tone보다 우선한다. */
  color?: string;
  style?: StyleProp<OwnStyle>;
  dataSet?: Record<string, string>;
  /** 값+단위, "실측 아님" 같은 부정 표현을 한 줄로 묶는다(index.html의 [data-pd-nowrap]). */
  nowrap?: boolean;
};

export function textStyle(variant: TextVariant, width: number, weight?: FontWeight): TextStyle {
  const scale = TYPE_SCALE[variant];
  const compact = variant === 'display' && width <= DISPLAY_COMPACT.maxWidth;
  return {
    fontFamily: FONT_FAMILY,
    fontSize: compact ? DISPLAY_COMPACT.fontSize : scale.fontSize,
    lineHeight: compact ? DISPLAY_COMPACT.lineHeight : scale.lineHeight,
    fontWeight: weight && scale.weights.includes(weight) ? weight : scale.weights[0],
    fontVariant: ['tabular-nums'],
  };
}

// h1은 라우트가 바뀔 때 초점을 받아야 해서(3.3절) ref를 넘긴다.
export const Text = React.forwardRef<RNText, TextProps>(function Text(
  { variant = 'body', weight, tone = 'text1', color, style, dataSet, nowrap, ...rest }, ref) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const data = nowrap ? { ...dataSet, pdNowrap: 'true' } : dataSet;
  // RN 타입에는 dataSet이 없지만 react-native-web은 data-* 속성으로 내보낸다.
  const webOnly = data ? ({ dataSet: data } as object) : {};
  return <RNText ref={ref} {...rest} {...webOnly} style={[textStyle(variant, width, weight), { color: color ?? theme.color[tone] }, style]} />;
});

/** 숫자가 든 이름 글자(카드 제목 "오늘 24시간 가격", 버튼 "24시간 전체 보기", 열 머리 "80% 구간")는 값이 아니라 이름이라
 * data-pd-meta="label"로 둔다. I2 검사(scan_i2.js)는 값·메타·칩·예시 밖의 숫자를 잡는다. */
export const labelData = (text: string): Record<string, string> | undefined => (/\d/.test(text) ? { pdMeta: 'label' } : undefined);

const NEGATION = /(\S+ 아님)/;
/** "실측 아님"·"값 아님" 같은 부정 표현만 한 줄로 묶는다(5.4절). 나머지는 keep-all 규칙으로 줄을 바꾼다. */
export function keepNegations(text: string, props: Omit<TextProps, 'children' | 'nowrap'>): React.ReactNode[] {
  return text.split(NEGATION).map((part, i) => (i % 2 ? <Text key={i} {...props} nowrap>{part}</Text> : part));
}

const TOGETHER = new RegExp(`(${[
  '\\d{4}-\\d{2}-\\d{2}(?:~\\d{2}-\\d{2})?', // 2026-11-26, 2025-07-22~12-18
  '\\d{2}:\\d{2}–\\d{2}:\\d{2}', // 06:00–07:00
  '\\d[\\d.,]*–\\d[\\d.,]*(?:원|시|h)?', // 171.7–244.1원, 07–08시, 03–05h
  '−?\\d[\\d.,]* ~ −?\\d[\\d.,]*원?', // −11.1 ~ 315.9원(음수가 든 범위, round 2)
  '\\d{2}-\\d{2}(?:\\([가-힣]\\))?', // 11-27(금)
  '(?:(?:조회|서버 시각|만든 시각|다음 구간|마지막 조회|응답 시각) )?\\d{2}:\\d{2}(?: KST)?', // (조회) 06:04 KST: 이름표와 시각이 갈리지 않게(round 2)
  '\\([^()]{1,16}\\)', // (MAPE 11.75%), (80% 구간 폭 326.4원): 15 px에서도 320 카드 폭(270) 안에 드는 길이까지만(16자 ≈ 200 px)
  '\\d[\\d.,]*\\s?(?:MW|원\\/kWh|만원|억원|원|%|°C|m\\/s)', // 684 MW, 12원, 15%
  '관측 시각 미제공',
].join('|')})`);
/**
 * 줄바꿈 단위(5.4절, round 1 멘토·parity): 값 범위("171.7–244.1원"), 시각 범위("07–08시"), 날짜("11-27(금)", "2026-11-26", 평가 기간
 * "2025-07-22~12-18"), 시각("06:04 KST"), 짧은 괄호("(MAPE 11.75%)"), 값+단위("684 MW"), "관측 시각 미제공"은 한 줄로 묶는다.
 * 브라우저는 줄표·붙임표 뒤나 괄호 안 빈칸에서 줄을 바꿀 수 있다(F8 관찰: "2025-" / "07-22~12-18", round 1: "(MAPE" / "11.75%)").
 */
export function keepTogether(text: string, props: Omit<TextProps, 'children' | 'nowrap'>): React.ReactNode[] {
  return text.split(TOGETHER).map((part, i) => (i % 2 ? <Text key={i} {...props} nowrap>{part}</Text> : part));
}
