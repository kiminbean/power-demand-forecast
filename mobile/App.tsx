/**
 * eXeco 제주 전력시장 — 공개 모바일 웹앱의 입구(명세 3·6.1절). 셸(라우터·탭·시트·알림·테마)은 src/app/Shell.tsx가 맡고,
 * 이 파일에는 훅 없는 Header와 라우트 스위치만 둔다.
 *
 * 헤더 회귀 테스트(tests/header-neutral-status-regression.test.cjs)가 이 파일 끝에 `export { Header };`를 덧붙여 VM에서 돌린다.
 * 그 VM은 react · react-native(View·Text·TouchableOpacity·StyleSheet) · src/theme/colors만 진짜로 주고 나머지 가져오기는 {}다.
 * 그래서 모듈 최상위에서는 StyleSheet.create 말고 아무것도 부르지 않고, Header는 가져온 값을 읽지 않는다: 색·글자 꼴·상태는
 * 셸이 props로 넘기고, 없으면 colors.ts의 옛 색을 쓴다. 크기 숫자는 토큰을 읽을 수 없어 그대로 적고 tests/header-palette가 맞춰 본다.
 */
import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { Pressable, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { colors } from './src/theme/colors';
import { Shell } from './src/app/Shell';
import type { HeaderFonts, HeaderPalette, ShellHeaderProps } from './src/app/Shell';
import { applyDocumentTheme } from './src/app/head';
import type { RouteKey, RouteState } from './src/app/routes';
import { ThemeProvider } from './src/app/theme';
import { RefreshButton } from './src/ui/Button';
import { Icon } from './src/ui/Icon';
import SMPForecastScreen from './src/screens/SMPForecastScreen';
import BiddingScreen from './src/screens/BiddingScreen';
import SettlementScreen from './src/screens/SettlementScreen';

// 헤더 버전 표기(P8 전까지 v6.1). src/content/strings.ts의 APP_VERSION과 같아야 한다.
const HEADER_VERSION = 'v6.1';
// F6: 탭 3개(알아보기는 F10). 세 화면 모두 아직 자기 h1이 없는 레거시 화면이라 셸이 h1 행을 둘러 준다(F7·F8·F9가 하나씩 뺀다).
const TABS: readonly RouteKey[] = ['smp', 'bidding', 'settlement'];
const LEGACY: readonly RouteKey[] = ['smp', 'bidding', 'settlement'];
// 셸이 테마 팔레트를 넘기지 않을 때(헤더 VM 테스트)의 옛 색.
const LEGACY_PALETTE: HeaderPalette = {
  background: colors.surface, border: colors.border.primary, wordmark: colors.brand.navy,
  title: colors.text.primary, version: colors.text.muted, icon: colors.text.primary,
};
const NO_FONTS: HeaderFonts = {};
// 레거시 입찰 화면이 부르는 이동은 DAM·RTM 시뮬레이션 화면뿐인데, 그 화면은 공개 빌드에서 뺐다(4.5절). 넘기지 않으면 가짜 결과를
// window.alert로 띄우므로 아무것도 하지 않는 이동을 넘긴다(F8이 입찰 탭을 바꾸면 사라진다).
const LEGACY_BIDDING_NAVIGATION = { navigate: () => undefined, goBack: () => undefined };

type HeaderProps = Partial<ShellHeaderProps>;

/** 모든 탭에서 같은 헤더(3.4절): 워드마크 + v6.1 | 새로고침. 하위 상태: ←(첫 TouchableOpacity) + 제목 | v6.1 + 새로고침. */
function Header({ title, showBack, onBack, onHome, palette = LEGACY_PALETTE, fonts = NO_FONTS, refresh }: HeaderProps) {
  const version = <Text style={[fonts.version, { color: palette.version }]}>{HEADER_VERSION}</Text>;
  return (
    <View role="banner" style={[styles.header, { backgroundColor: palette.background, borderBottomColor: palette.border }]}>
      <View style={styles.side}>
        {showBack ? (
          <TouchableOpacity onPress={onBack} role="button" aria-label="뒤로" style={styles.target}>
            <Icon name="back" size={22} color={palette.icon} />
          </TouchableOpacity>
        ) : (
          <Pressable onPress={onHome} role="button" aria-label="eXeco 처음 화면" style={styles.wordmark}>
            <Text style={[fonts.wordmark, { color: palette.wordmark }]}>eXeco</Text>
          </Pressable>
        )}
        {title ? <Text numberOfLines={1} style={[styles.title, fonts.title, { color: palette.title }]}>{title}</Text> : null}
        {showBack ? null : version}
      </View>
      <View style={styles.side}>
        {showBack ? version : null}
        {refresh ? <RefreshButton onPress={refresh.onPress} busy={refresh.busy} disabledReason={refresh.disabledReason} /> : null}
      </View>
    </View>
  );
}

/** 라우트 스위치: 화면 본문만 고른다(헤더·탭·시트는 셸). */
function renderRoute(state: RouteState) {
  switch (state.key) {
    case 'smp':
      return <SMPForecastScreen />;
    case 'bidding':
      return <BiddingScreen webNavigation={LEGACY_BIDDING_NAVIGATION} />;
    case 'settlement':
      return <SettlementScreen />;
    default:
      return null;
  }
}

export default function App() {
  return (
    <ThemeProvider onDocumentTheme={applyDocumentTheme}>
      <StatusBar style="auto" />
      <Shell tabs={TABS} legacy={LEGACY} renderRoute={renderRoute} renderHeader={Header} />
    </ThemeProvider>
  );
}

// 크기는 토큰과 같다: 높이 LAYOUT.header 56, 누름 상자 HIT 44, 여백 SPACE.xs 4 · SPACE.md 12(글자가 화면 끝에서 16).
const styles = StyleSheet.create({
  header: {
    height: 56, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 4, borderBottomWidth: 1,
  },
  side: { flexDirection: 'row', alignItems: 'center', gap: 4, flexShrink: 1, minWidth: 0 },
  target: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  wordmark: { minHeight: 44, minWidth: 44, paddingHorizontal: 12, justifyContent: 'center' },
  title: { flexShrink: 1 },
});
