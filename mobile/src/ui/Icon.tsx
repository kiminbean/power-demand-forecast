// 선 아이콘(명세 5.6절). 장식이 기본이라 보조 기술에서 숨기고, 이름은 감싼 버튼이 가진다.
// @expo/vector-icons(TTF 400 KB)와 src/components/Ionicon.tsx는 쓰지 않는다.
import React from 'react';
import { View } from 'react-native';
import Svg, { Circle, Path } from 'react-native-svg';
import { ICONS, ICON_STROKE_WIDTH, ICON_VIEWBOX } from './icons';
import type { IconName } from './icons';
import { useTheme } from '../app/theme';

export type IconProps = { name: IconName; size?: number; color?: string };

export function Icon({ name, size = 24, color }: IconProps) {
  const { theme } = useTheme();
  const stroke = color ?? theme.color.text2;
  return (
    <View aria-hidden importantForAccessibility="no-hide-descendants" accessibilityElementsHidden style={{ width: size, height: size }}>
      <Svg width={size} height={size} viewBox={ICON_VIEWBOX} fill="none" stroke={stroke} strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round" strokeLinejoin="round">
        {ICONS[name].map((shape, index) =>
          shape.kind === 'path'
            ? <Path key={index} d={shape.d} />
            : <Circle key={index} cx={shape.cx} cy={shape.cy} r={shape.r} />)}
      </Svg>
    </View>
  );
}
