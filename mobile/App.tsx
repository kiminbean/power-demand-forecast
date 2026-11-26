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
import type { HeaderFonts, HeaderPalette, PanelContent, ShellHeaderProps } from './src/app/Shell';
import type { SheetRegistry } from './src/app/SheetHost';
import { applyDocumentTheme } from './src/app/head';
import type { RouteKey, RouteState, SourceRow } from './src/app/routes';
import { ThemeProvider } from './src/app/theme';
import {
  BIDDING_PANEL_TITLE, GLOSSARY_TITLES, SETTLEMENT_PANEL_TITLE, SOURCES_SHEET_TITLE, TODAY_PANEL_TITLE, methodSheetTitle,
} from './src/content/strings';
import BiddingTab, { BiddingMethodFacts, BiddingPanel, useBiddingSourceRows } from './src/features/bidding/BiddingTab';
import SettlementTab, { SettlementMethodFacts, SettlementPanel } from './src/features/settlement/SettlementTab';
import { MethodSheetBody, TermSheetBody } from './src/features/guide/sheets';
import { RefreshButton } from './src/ui/Button';
import { Icon } from './src/ui/Icon';
import SMPForecastScreen, { TodayMethodFacts, TodayPanel, TodaySourcesSheet } from './src/screens/SMPForecastScreen';

// 헤더 버전 표기(P8 전까지 v6.1). src/content/strings.ts의 APP_VERSION과 같아야 한다.
const HEADER_VERSION = 'v6.1';
// 버전 글자의 숫자는 값이 아니라 이름이다(I2: data-pd-meta). web()은 헤더 VM 테스트에서 {}라서 부르지 않고 객체로 둔다.
const VERSION_DATA: object = { dataSet: { pdMeta: 'version' } };
// 탭 3개(알아보기는 F10). 오늘(F7)·입찰(F8)·정산(F9) 모두 새 화면(자기 h1 행)이라 셸의 레거시 틀은 더 쓰지 않는다.
const TABS: readonly RouteKey[] = ['smp', 'bidding', 'settlement'];

/** 출처와 기준 시각 표(4.4절): 오늘 행 + 입찰 계산 행. 응답 상태 행은 F10이 더한다. */
function SourcesSheet({ row }: { row: SourceRow | null }) {
  return <TodaySourcesSheet row={row} extra={useBiddingSourceRows()} />;
}
// 화면 단계가 더하는 시트(3.1·4.4절). 산출 방식은 설명 글(guide.ko.json) + 그 카드 화면의 이번 값(각 화면 모듈이 자기 카드만 그린다).
// 용어 시트의 "용어 사전 전체" 링크는 알아보기 탭이 생기는 F10이 onGlossary로 단다.
const SHEETS: SheetRegistry = {
  method: {
    title: s => methodSheetTitle(s.card),
    render: s => (
      <MethodSheetBody card={s.card}
        facts={<><TodayMethodFacts card={s.card} /><BiddingMethodFacts card={s.card} /><SettlementMethodFacts card={s.card} /></>} />
    ),
  },
  sources: { title: () => SOURCES_SHEET_TITLE, render: s => <SourcesSheet row={s.row} /> },
  term: { title: s => GLOSSARY_TITLES[s.id], render: s => <TermSheetBody id={s.id} /> },
};
// ≥ 1280 "이 화면 읽는 법" 패널(4.6절): 화면마다 제목이 다르다(정산은 ①+② 식 그림, 4.3절).
const renderPanel = (state: RouteState): PanelContent | null => {
  if (state.key === 'smp') return { title: TODAY_PANEL_TITLE, body: <TodayPanel /> };
  if (state.key === 'bidding') return { title: BIDDING_PANEL_TITLE, body: <BiddingPanel /> };
  return state.key === 'settlement' ? { title: SETTLEMENT_PANEL_TITLE, body: <SettlementPanel /> } : null;
};
// 셸이 테마 팔레트를 넘기지 않을 때(헤더 VM 테스트)의 옛 색.
const LEGACY_PALETTE: HeaderPalette = {
  background: colors.surface, border: colors.border.primary, wordmark: colors.brand.navy,
  title: colors.text.primary, version: colors.text.muted, icon: colors.text.primary,
};
const NO_FONTS: HeaderFonts = {};

type HeaderProps = Partial<ShellHeaderProps>;

/** 모든 탭에서 같은 헤더(3.4절): 워드마크 + v6.1 | 새로고침. 하위 상태: ←(첫 TouchableOpacity) + 제목 | v6.1 + 새로고침. */
function Header({ title, showBack, onBack, onHome, palette = LEGACY_PALETTE, fonts = NO_FONTS, refresh }: HeaderProps) {
  const version = <Text {...VERSION_DATA} style={[fonts.version, { color: palette.version }]}>{HEADER_VERSION}</Text>;
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
      return <BiddingTab />;
    case 'settlement':
      return <SettlementTab />;
    default:
      return null;
  }
}

export default function App() {
  return (
    <ThemeProvider onDocumentTheme={applyDocumentTheme}>
      <StatusBar style="auto" />
      <Shell tabs={TABS} renderRoute={renderRoute} renderHeader={Header} renderPanel={renderPanel} sheets={SHEETS} />
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
