// 누르는 것의 공통 규칙(6.2절): role=button, 한국어 이름은 타입에서 필수, DOM 상자 ≥ 44×44(M2),
// 비활성은 aria-disabled, 진행 중 aria-busy. RN-web은 hitSlop을 DOM에 반영하지 않으므로 상자 자체를 키운다.
import React from 'react';
import { Pressable, StyleSheet } from 'react-native';
import type { View } from 'react-native';
import { HIT, RADIUS, SIZE, SPACE } from './tokens';
import { Icon } from './Icon';
import type { IconName } from './icons';
import { Text } from './Text';
import { web } from './web';
import { useTheme } from '../app/theme';

export type PressBase = {
  /** 보조 기술이 읽는 한국어 이름. 기호만인 이름은 쓰지 않는다(M5). */
  accessibilityLabel: string;
  onPress?: () => void;
  disabled?: boolean;
  /** 비활성 까닭. 이름 뒤에 " · {까닭}"으로 붙고, 까닭이 있으면 Tab으로 닿을 수 있게 남긴다. */
  disabledReason?: string;
  busy?: boolean;
  testID?: string;
  pressRef?: React.Ref<View>;
};
export type ButtonTone = 'primary' | 'secondary' | 'ghost';
export type ButtonProps = PressBase & { label: string; tone?: ButtonTone; icon?: IconName };
export type IconButtonProps = PressBase & { icon: IconName; color?: string };

function pressProps({ accessibilityLabel, onPress, disabled, disabledReason, busy, testID }: PressBase) {
  const name = disabled && disabledReason ? `${accessibilityLabel} · ${disabledReason}` : accessibilityLabel;
  // 진행 중에는 누름을 받지 않는다(같은 요청을 겹쳐 보내지 않는다).
  return {
    role: 'button' as const, 'aria-label': name, 'aria-busy': busy ? true : undefined, disabled, testID,
    onPress: busy ? undefined : onPress, ...web(disabled && disabledReason ? { tabIndex: 0 } : {}),
  };
}

export function Button(props: ButtonProps) {
  const { theme } = useTheme();
  const c = theme.color;
  const { label, tone = 'secondary', icon, disabled, pressRef } = props;
  const fg = disabled ? c.text3 : tone === 'primary' ? c.onPrimary : tone === 'ghost' ? c.link : c.text1;
  const box = tone === 'ghost' ? null
    : tone === 'primary' && !disabled ? { backgroundColor: c.primary }
      : { backgroundColor: disabled ? c.well : c.surface, borderWidth: 1, borderColor: c.borderStrong };
  return (
    <Pressable ref={pressRef} {...pressProps(props)} style={[styles.button, tone === 'ghost' ? styles.ghost : null, box]}>
      {icon ? <Icon name={icon} size={SIZE.buttonIcon} color={fg} /> : null}
      <Text variant="body" weight="600" color={fg}>{label}</Text>
    </Pressable>
  );
}

export function IconButton(props: IconButtonProps) {
  const { theme } = useTheme();
  const tint = props.disabled ? theme.color.text3 : props.color ?? theme.color.text2;
  return (
    <Pressable ref={props.pressRef} {...pressProps(props)} style={styles.icon}>
      <Icon name={props.icon} size={SIZE.icon} color={tint} />
    </Pressable>
  );
}

export const REFRESH_LABEL = '가격·전력 정보 새로고침';
/** 헤더 새로고침은 모든 탭에서 같은 자리에 있고, 새로 불러올 것이 없으면 숨기지 않고 비활성으로 둔다(3.4절). */
export function RefreshButton({ onPress, busy, disabledReason }: { onPress?: () => void; busy?: boolean; disabledReason?: string | null }) {
  return (
    <IconButton icon="refresh" accessibilityLabel={REFRESH_LABEL} onPress={onPress} busy={busy}
      disabled={Boolean(disabledReason)} disabledReason={disabledReason ?? undefined} testID="refresh-button" />
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: HIT, minWidth: HIT, paddingHorizontal: SPACE.lg, borderRadius: RADIUS.control,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: SPACE.sm,
  },
  ghost: { paddingHorizontal: SPACE.md },
  icon: { width: HIT, height: HIT, borderRadius: RADIUS.control, alignItems: 'center', justifyContent: 'center' },
});
