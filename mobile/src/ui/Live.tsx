// 알림(3.4절): 셸에 polite live region 하나(M5). 새로고침 결과·라우트 이름·스텝퍼 값·오류·지연 글자가 모두 이 하나로 간다.
// 같은 글자를 다시 알릴 때도 읽히도록 매번 새 key로 글자 노드를 바꾼다.
import React, { createContext, useCallback, useContext, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { LAYOUT, SIZE, SPACE } from './tokens';
import { Icon } from './Icon';
import { Text } from './Text';
import { SR_ONLY, web } from './web';
import { useTheme } from '../app/theme';

export type Announce = (text: string) => void;
const AnnounceContext = createContext<Announce>(() => undefined);
/** 공급자 밖(테스트·오류 화면)에서는 아무것도 하지 않는 함수를 준다. */
export const useAnnounce = () => useContext(AnnounceContext);

export function LiveRegion({ children }: { children?: React.ReactNode }) {
  const [message, setMessage] = useState({ text: '', n: 0 });
  const announce = useCallback<Announce>(text => setMessage(m => ({ text, n: m.n + 1 })), []);
  return (
    <AnnounceContext.Provider value={announce}>
      {children}
      <View aria-live="polite" style={SR_ONLY} testID="live-region" {...web({ dataSet: { pdMeta: 'live' } })}>
        <Text key={message.n} variant="meta">{message.text}</Text>
      </View>
    </AnnounceContext.Provider>
  );
}

/** 화면 읽기 프로그램에만 들리는 덧말(예: SampleValue의 "(예시)"). */
export function SrOnly({ children }: { children: string }) {
  return <Text variant="meta" style={SR_ONLY}>{children}</Text>;
}

/** navigator.onLine === false일 때 헤더 아래 띠(7.6절). role=status라 스스로 알린다. */
export function OfflineBanner({ text }: { text: string }) {
  const { theme } = useTheme();
  const tone = theme.pill.watch;
  return (
    <View role="status" style={[styles.banner, { backgroundColor: tone.bg }]} testID="offline-banner">
      <Icon name="offline" size={SIZE.buttonIcon} color={tone.text} />
      <Text variant="label" weight="600" color={tone.text} dataSet={{ pdMeta: 'offline' }}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: { minHeight: LAYOUT.offlineBanner, flexDirection: 'row', alignItems: 'center', gap: SPACE.sm, paddingHorizontal: LAYOUT.gutter },
});
