// 문서 머리(8.4·8.5절, 웹): 라우트별 document.title, 테마(data-theme · color-scheme · theme-color 두 메타), 그리고 셸이 듣는
// 브라우저 상태(온라인 여부, 숨었던 탭이 다시 보임). <html lang="ko">는 index.html 템플릿에 고정이다. head.ts와 export가 같다.
import type { DocumentTheme } from './theme';

const doc = (): Document | null => (typeof document === 'undefined' ? null : document);

export function setDocumentTitle(title: string): void {
  const d = doc();
  if (d && d.title !== title) d.title = title;
}

/** 시스템이면 data-theme을 지우고 theme-color 두 메타를 각 테마의 canvas로 되돌린다(값은 theme.tsx의 documentTheme이 정한다). */
export function applyDocumentTheme(theme: DocumentTheme): void {
  const d = doc();
  if (!d) return;
  const root = d.documentElement;
  if (theme.dataTheme) root.setAttribute('data-theme', theme.dataTheme);
  else root.removeAttribute('data-theme');
  root.style.colorScheme = theme.colorScheme ?? '';
  d.querySelectorAll('meta[name="theme-color"]').forEach(meta => {
    const dark = (meta.getAttribute('media') ?? '').includes('dark');
    meta.setAttribute('content', dark ? theme.themeColor.dark : theme.themeColor.light);
  });
}

export function isOnline(): boolean {
  return !(typeof navigator !== 'undefined' && navigator.onLine === false);
}

export function watchOnline(listener: (online: boolean) => void): () => void {
  if (typeof window === 'undefined') return () => undefined;
  const update = () => listener(isOnline());
  window.addEventListener('online', update);
  window.addEventListener('offline', update);
  return () => {
    window.removeEventListener('online', update);
    window.removeEventListener('offline', update);
  };
}

/** 화면이 보이는가(배경 탭에서는 5분 새로고침을 건너뛴다, 6.3절). */
export function isVisible(): boolean {
  const d = doc();
  return !d || d.visibilityState !== 'hidden';
}

/** 숨었던 탭이 다시 보일 때 부른다(배경 탭에서 늦게 돈 타이머가 남긴 지난 시각을 바로잡는다). */
export function watchVisible(listener: () => void): () => void {
  const d = doc();
  if (!d) return () => undefined;
  const onChange = () => {
    if (d.visibilityState === 'visible') listener();
  };
  d.addEventListener('visibilitychange', onChange);
  return () => d.removeEventListener('visibilitychange', onChange);
}
