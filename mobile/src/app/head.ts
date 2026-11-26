// 문서 머리(8.4·8.5절)의 네이티브·node 테스트 판: 문서가 없으니 아무것도 하지 않는다. 웹은 같은 이름의 head.web.ts를 쓴다.
import type { DocumentTheme } from './theme';

export function setDocumentTitle(_title: string): void {
  return undefined;
}
export function applyDocumentTheme(_theme: DocumentTheme): void {
  return undefined;
}
export const isOnline = (): boolean => true;
export const watchOnline = (_listener: (online: boolean) => void): (() => void) => () => undefined;
export const watchVisible = (_listener: () => void): (() => void) => () => undefined;
export const isVisible = (): boolean => true;
