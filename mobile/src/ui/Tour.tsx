// 1분 둘러보기(4.4절). 배너는 덮개가 아니라 화면 안 카드다(round 1: 한 줄 제목 + 버튼, 오늘 화면에서는 차트 뒤). 단계의 h1은 Screen 제목이 맡고(단계마다 h1 하나),
// 여기서는 진행 "2 / 5" 글자·설명(children)·이동 버튼을 그린다. 버튼은 설명 바로 아래라 단계 예시가 길어도 첫 화면에 있다.
// 마지막 단계는 [이전] [직접 둘러보기](→ 오늘)이고 건너뛸 단계가 없어 [건너뛰기]가 없다. 글은 guide.ko.json, 숫자는 strings.ts.
import type React from 'react';
import { StyleSheet, View } from 'react-native';
import { RADIUS, SIZE, SPACE } from './tokens';
import { Button } from './Button';
import { Icon } from './Icon';
import { Text, labelData } from './Text';
import { useTheme } from '../app/theme';
import { TOUR_FINISH, TOUR_NAME, tourProgressText } from '../content/strings';

export type TourBannerProps = { title: string; onStart: () => void; onDismiss: () => void };

export function TourBanner({ title, onStart, onDismiss }: TourBannerProps) {
  const { theme } = useTheme();
  return (
    <View style={[styles.banner, { backgroundColor: theme.color.surface, borderColor: theme.color.border }]} testID="tour-banner">
      <Icon name="book" size={SIZE.icon} color={theme.color.link} />
      <View style={styles.grow}>
        {/* "1분 둘러보기"의 숫자는 이름표라 메타로 둔다(I2). */}
        <Text variant="body" weight="700" dataSet={labelData(title)}>{title}</Text>
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
  /** 진행 글자 옆 NowPill(round 2: h1 행에 두면 단계 제목 길이에 따라 줄이 바뀌어 머리가 뛰었다). */
  pill?: React.ReactNode;
  onPrev: () => void;
  /** 마지막 단계에서는 [직접 둘러보기](부모가 오늘로 보낸다). */
  onNext: () => void;
  onSkip: () => void;
  children?: React.ReactNode;
};

export function TourStep({ step, total, pill, onPrev, onNext, onSkip, children }: TourStepProps) {
  const last = step >= total;
  return (
    <View style={styles.step} testID="tour-step">
      <View style={styles.progress}>
        <Text variant="meta" weight="600" tone="text3" dataSet={{ pdMeta: 'tour' }}>{tourProgressText(step, total)}</Text>
        {pill}
      </View>
      {children}
      <View style={styles.actions}>
        <Button label="이전" accessibilityLabel="이전 단계" onPress={onPrev} disabled={step <= 1} disabledReason="첫 단계입니다" />
        <Button tone="primary" label={last ? TOUR_FINISH : '다음'} accessibilityLabel={last ? TOUR_FINISH : '다음 단계'} onPress={onNext} />
        {last ? null : <Button tone="ghost" label="건너뛰기" accessibilityLabel={`건너뛰기: ${TOUR_NAME} 닫기`} onPress={onSkip} />}
      </View>
    </View>
  );
}

/** 단계 예시 뒤의 짧은 이동 줄(round 2: 320×568에서 2–4단계 예시가 300–500 px 넘쳐 [다음]을 누르려면 위로 돌아가야 했다). 휴대폰 폭에서만. */
export function TourStepEnd({ step, total, onPrev, onNext }: Pick<TourStepProps, 'step' | 'total' | 'onPrev' | 'onNext'>) {
  const last = step >= total;
  return (
    <View style={styles.actions} testID="tour-step-end">
      <Button label="이전" accessibilityLabel="이전 단계" onPress={onPrev} disabled={step <= 1} disabledReason="첫 단계입니다" />
      <Button tone="primary" label={last ? TOUR_FINISH : '다음'} accessibilityLabel={last ? TOUR_FINISH : '다음 단계'} onPress={onNext} />
    </View>
  );
}

const styles = StyleSheet.create({
  banner: { flexDirection: 'row', gap: SPACE.md, padding: SPACE.lg, borderRadius: RADIUS.card, borderWidth: 1 },
  grow: { flex: 1, minWidth: 0, gap: SPACE.xs },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: SPACE.sm, marginTop: SPACE.xs },
  step: { gap: SPACE.md },
  progress: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: SPACE.sm },
});
