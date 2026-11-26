// 테마 선택(명세 8.4절): 시스템(기본)·밝게·어둡게. 레이아웃과 데이터 색은 두 테마에서 같다.
// 문서(data-theme·color-scheme·theme-color 메타)에 쓰는 일은 F6의 head.web.ts가 맡고, 여기서는 적용할 값만 계산한다.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type React from 'react';
import { useColorScheme } from 'react-native';
import { THEMES } from '../ui/tokens';
import type { Theme, ThemeName } from '../ui/tokens';

export type ThemeMode = 'system' | ThemeName;
export const THEME_STORAGE_KEY = 'pd-theme';

type StorageLike = { getItem(key: string): string | null; setItem(key: string, value: string): void; removeItem(key: string): void };

/** 저장소 접근 자체가 막힌 환경(사생활 모드·네이티브)에서도 던지지 않는다. */
function defaultStorage(): StorageLike | null {
  try {
    return (globalThis as { localStorage?: StorageLike }).localStorage ?? null;
  } catch {
    return null;
  }
}

/** 저장된 값이 정확히 light·dark일 때만 따르고, 그 밖(없음·오타·예외)은 system. */
export function readThemeMode(store: StorageLike | null = defaultStorage()): ThemeMode {
  try {
    const value = store ? store.getItem(THEME_STORAGE_KEY) : null;
    return value === 'light' || value === 'dark' ? value : 'system';
  } catch {
    return 'system';
  }
}

/** 저장에 실패해도 그 세션 동안의 선택은 유지된다. 저장했으면 true. */
export function writeThemeMode(mode: ThemeMode, store: StorageLike | null = defaultStorage()): boolean {
  try {
    if (!store) return false;
    if (mode === 'system') store.removeItem(THEME_STORAGE_KEY);
    else store.setItem(THEME_STORAGE_KEY, mode);
    return true;
  } catch {
    return false;
  }
}

/** OS 변화는 system일 때만 반영한다. */
export function resolveScheme(mode: ThemeMode, os: string | null | undefined): ThemeName {
  if (mode !== 'system') return mode;
  return os === 'dark' ? 'dark' : 'light';
}

export type DocumentTheme = { dataTheme: ThemeName | null; colorScheme: ThemeName | null; themeColor: { light: string; dark: string } };

/** 시스템이면 data-theme을 지우고 두 theme-color 메타를 index.html의 원래 값(각 테마 canvas)으로 되돌린다. */
export function documentTheme(mode: ThemeMode): DocumentTheme {
  if (mode === 'system') {
    return { dataTheme: null, colorScheme: null, themeColor: { light: THEMES.light.color.canvas, dark: THEMES.dark.color.canvas } };
  }
  const canvas = THEMES[mode].color.canvas;
  return { dataTheme: mode, colorScheme: mode, themeColor: { light: canvas, dark: canvas } };
}

/** setMode는 이 기기에 저장했으면 true를 돌려준다(저장소를 못 쓰면 그 방문 동안만 적용, 테마 시트가 그대로 알린다). */
export type ThemeContextValue = { mode: ThemeMode; scheme: ThemeName; theme: Theme; setMode: (mode: ThemeMode) => boolean };

// 공급자 밖(테스트·오류 화면)에서도 그릴 수 있게 밝은 테마를 기본값으로 둔다.
const ThemeContext = createContext<ThemeContextValue>({ mode: 'system', scheme: 'light', theme: THEMES.light, setMode: () => false });

type ThemeProviderProps = { children?: React.ReactNode; onDocumentTheme?: (value: DocumentTheme) => void };

export function ThemeProvider({ children, onDocumentTheme }: ThemeProviderProps) {
  const [mode, setModeState] = useState<ThemeMode>(() => readThemeMode());
  const os = useColorScheme();
  const scheme = resolveScheme(mode, os);
  useEffect(() => {
    if (onDocumentTheme) onDocumentTheme(documentTheme(mode));
  }, [mode, onDocumentTheme]);
  const setMode = useCallback((next: ThemeMode) => {
    const saved = writeThemeMode(next);
    setModeState(next);
    return saved;
  }, []);
  const value = useMemo(() => ({ mode, scheme, theme: THEMES[scheme], setMode }), [mode, scheme, setMode]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
