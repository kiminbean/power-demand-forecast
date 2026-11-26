// 오늘 화면의 둘러보기 배너는 첫 방문에만(명세 4.4절). localStorage 'pd-tour-seen'(try/catch). 저장소를 못 쓰면 그 방문 동안만
// 닫힌 것을 기억한다. 둘러보기를 연 것도 본 것으로 친다(둘러보기 화면이 markTourSeen을 부른다).
const TOUR_SEEN_KEY = 'pd-tour-seen';
let tourSeenThisVisit = false;
type StorageLike = { getItem(k: string): string | null; setItem(k: string, v: string): void };

const storage = (): StorageLike | null => {
  try {
    return (globalThis as { localStorage?: StorageLike }).localStorage ?? null;
  } catch {
    return null;
  }
};

export function tourSeen(): boolean {
  if (tourSeenThisVisit) return true;
  try {
    const s = storage();
    return s ? s.getItem(TOUR_SEEN_KEY) === '1' : false;
  } catch {
    return false;
  }
}

export function markTourSeen(): void {
  tourSeenThisVisit = true;
  try {
    const s = storage();
    if (s) s.setItem(TOUR_SEEN_KEY, '1');
  } catch {
    // 저장하지 못해도 이번 방문 동안은 닫힌 채로 둔다.
  }
}
