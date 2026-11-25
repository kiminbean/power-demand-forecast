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

export function Text({ variant = 'body', weight, tone = 'text1', color, style, dataSet, nowrap, ...rest }: TextProps) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const data = nowrap ? { ...dataSet, pdNowrap: 'true' } : dataSet;
  // RN 타입에는 dataSet이 없지만 react-native-web은 data-* 속성으로 내보낸다.
  const webOnly = data ? ({ dataSet: data } as object) : {};
  return <RNText {...rest} {...webOnly} style={[textStyle(variant, width, weight), { color: color ?? theme.color[tone] }, style]} />;
}
