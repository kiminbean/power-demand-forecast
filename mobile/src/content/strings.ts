// 숫자가 든 짧은 UI 문구(명세 6.1절). .tsx의 JSX 글자에는 숫자를 직접 쓰지 않고 여기서 가져온다(7.10절).
// 숫자·시각은 모두 format.ts를 거친다. 긴 설명 글은 public/content/guide.ko.json(guide.ts가 읽는다)에 둔다.
import { age, kstClock, kstTime, num, pct } from '../lib/format';
import { MINUTE_MS } from '../lib/kst';
import { DOCUMENTED_SMP_ACCURACY } from '../lib/provenance';

export const APP_NAME = 'eXeco 제주 전력시장';
export const APP_VERSION = 'v6.1';

/** 마지막 성공 조회가 10분을 넘으면 지연, 60분을 넘으면 끊김(7.6절). */
export const STALE_AFTER_MS = 10 * MINUTE_MS;
export const BROKEN_AFTER_MS = 60 * MINUTE_MS;
export type Freshness = 'fresh' | 'stale' | 'broken';
export function freshness(ageMs: number): Freshness {
  if (ageMs > BROKEN_AFTER_MS) return 'broken';
  return ageMs > STALE_AFTER_MS ? 'stale' : 'fresh';
}

export const lookupAt = (fetchedAt: number) => `조회 ${kstTime(fetchedAt)}`;
/** KPX·기상 값: 게시 시각이 없어 조회 시각만 쓴다. */
export const observedLookup = (fetchedAt: number) => `관측 시각 미제공 · 조회 ${kstTime(fetchedAt)}`;
export const createdAtText = (createdAt: number | null) => `만든 시각 ${kstTime(createdAt)}`;
export const serverTimeText = (serverTime: number | null) => `서버 시각 ${kstTime(serverTime)}`;
export const forecastDateText = (dayKey: string | null) => `예측일 ${dayKey ?? '미제공'}`;

export type NowPillInput = { now: number; fetchedAt: number | null; skewNotable?: boolean };
/** NowPill 글자: "06:04 KST · 조회 1분 전" / "지연 · 06:04 조회 (12분 전)" / "끊김 · 마지막 조회 06:04" / 조회 없는 화면 "06:04 KST". */
export function nowPillText({ now, fetchedAt, skewNotable = false }: NowPillInput): string {
  const tail = skewNotable ? ' · 서버 시각 기준' : '';
  if (fetchedAt === null) return kstTime(now) + tail;
  const elapsed = now - fetchedAt;
  const stage = freshness(elapsed);
  if (stage === 'broken') return `끊김 · 마지막 조회 ${kstClock(fetchedAt)}${tail}`;
  if (stage === 'stale') return `지연 · ${kstClock(fetchedAt)} 조회 (${age(elapsed)})${tail}`;
  return `${kstTime(now)} · 조회 ${age(elapsed)}${tail}`;
}

/** 지연이면 "지연 · 06:04 값", 끊김이면 실측 글리프를 거두고 "06:04 값". */
export const staleChipText = (fetchedAt: number, stage: Exclude<Freshness, 'fresh'>) =>
  (stage === 'stale' ? `지연 · ${kstClock(fetchedAt)} 값` : `${kstClock(fetchedAt)} 값`);
export const offlineText = (lastData: number) => `오프라인 · 마지막 자료 ${kstTime(lastData)}`;
export const refreshDoneText = (at: number) => `${kstTime(at)} 기준으로 새로 불러왔습니다`;
export const refreshFailedText = (shown: number) => `새로고침 실패 · ${kstClock(shown)} 값 표시 중`;
export const clockSkewText = (skewMs: number) => `기기 시계와 ${num(Math.abs(skewMs) / MINUTE_MS, 0)}분 차이`;

/** 문서화된 정확도(7.7절). 수치는 provenance.ts의 상수 하나에서만 온다. */
export const accuracyText = () =>
  `시간마다 평균 ${num(DOCUMENTED_SMP_ACCURACY.maeKrwPerKwh, 2)}원/kWh 빗나갑니다 (MAPE ${num(DOCUMENTED_SMP_ACCURACY.mapePct, 2)}%)`;
export const accuracyBasisText = () =>
  `${DOCUMENTED_SMP_ACCURACY.method} ${DOCUMENTED_SMP_ACCURACY.period} · ${DOCUMENTED_SMP_ACCURACY.basis}`;
/** 예측 방법 화면에서만 더한다. */
export const accuracyR2Text = () => `R² ${num(DOCUMENTED_SMP_ACCURACY.r2, 3)}`;
/** API가 스스로 밝힌 수치는 회색 "API 표기" 줄로만. */
export const apiAccuracyText = (mape: number | null, coverage: number | null) =>
  `API 표기: 검증 MAPE ${pct(mape, 2)}% (최근 150일 재현) · 80% 구간 적중률 ${pct(coverage, 1)}% (최근 30일 보류 검증)`;

/** 0.00 입력 시간(7.8절 4). */
export const ZERO_INPUT_LABEL = '기준선 입력 0.00 · 예측 아님';
export const ZERO_INPUT_CELL = '— (입력 0.00)';
export const zeroInputExcludedText = (hours: number) => `0.00 입력 ${num(hours, 0)}시간 제외`;

export const tourProgressText = (step: number, total: number) => `${num(step, 0)} / ${num(total, 0)}`;
export const routeTitle = (screen: string) => `${screen} · ${APP_NAME}`;
export const tourTitle = (step: number, total: number) => `둘러보기 ${num(step, 0)}/${num(total, 0)} · ${APP_NAME}`;

/** 용어 사전 12항목의 제목. 숫자가 든 제목이 있어 JSON(숫자 금지)이 아니라 여기 둔다. 뜻풀이는 guide.ko.json. */
export const GLOSSARY_TITLES = Object.freeze({
  smp: 'SMP(계통한계가격)',
  dam: '하루전시장',
  rtm: '실시간시장',
  'trade-hour': '거래시간(01h–24h)',
  demand: '수요',
  supply: '공급능력',
  reserve: '예비율과 예비력',
  baseline: '기준선 예측',
  band: '80% 구간',
  error: '평균 오차(MAE·MAPE)',
  clearing: '낙찰 가능성',
  settlement: '정산',
});
export type GlossaryId = keyof typeof GLOSSARY_TITLES;
