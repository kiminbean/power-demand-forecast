// 1분 둘러보기(4.4절). 배너는 덮개가 아니라 화면 안 카드다. 단계의 h1은 Screen 제목이 맡고(단계마다 h1 하나),
// 여기서는 진행 "2 / 5" 글자·설명·이동 버튼을 그린다. 글은 guide.ko.json에서 오고 숫자는 strings.ts가 만든다.
import React from 'react';
import { StyleSheet, View } from 'react-native';
import { RADIUS, SIZE, SPACE } from './tokens';
import { Button } from './Button';
import { Icon } from './Icon';
import { Text } from './Text';
import { useTheme } from '../app/theme';
import { tourProgressText } from '../content/strings';

export type TourBannerProps = { title: string; body: string; onStart: () => void; onDismiss: () => void };

export function TourBanner({ title, body, onStart, onDismiss }: TourBannerProps) {
  const { theme } = useTheme();
  return (
    <View style={[styles.banner, { backgroundColor: theme.color.surface, borderColor: theme.color.border }]} testID="tour-banner">
      <Icon name="book" size={SIZE.icon} color={theme.color.link} />
      <View style={styles.grow}>
        <Text variant="body" weight="700">{title}</Text>
        <Text variant="label" tone="text2">{body}</Text>
        <View style={styles.actions}>
          <Button tone="primary" label="둘러보기" accessibilityLabel="둘러보기 시작" onPress={onStart} />
          <Button tone="ghost" label="닫기" accessibilityLabel="닫기: 둘러보기 안내" onPress={onDismiss} />
        </View>
      </View>
    </View>
  );
}

export type TourStepProps = {
  step: number;
  total: number;
  body: string;
  onPrev: () => void;
  onNext: () => void;
  onSkip: () => void;
};

export function TourStep({ step, total, body, onPrev, onNext, onSkip }: TourStepProps) {
  const last = step >= total;
  return (
    <View style={styles.step} testID="tour-step">
      <Text variant="meta" weight="600" tone="text3" dataSet={{ pdMeta: 'tour' }}>{tourProgressText(step, total)}</Text>
      <Text variant="body" tone="text2">{body}</Text>
      <View style={styles.actions}>
        <Button label="이전" accessibilityLabel="이전 단계" onPress={onPrev} disabled={step <= 1} disabledReason="첫 단계입니다" />
        <Button tone="primary" label={last ? '마치기' : '다음'} accessibilityLabel={last ? '둘러보기 마치기' : '다음 단계'} onPress={onNext} />
        <Button tone="ghost" label="건너뛰기" accessibilityLabel="둘러보기 건너뛰기" onPress={onSkip} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: { flexDirection: 'row', gap: SPACE.md, padding: SPACE.lg, borderRadius: RADIUS.card, borderWidth: 1 },
  grow: { flex: 1, minWidth: 0, gap: SPACE.xs },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACE.sm, marginTop: SPACE.xs },
  step: { gap: SPACE.md },
});
