// 알아보기 view model(명세 4.4·7.6·7.7·7.9절). 응답 상태(/realtime-status) 행, 둘러보기 5단계 범례, 용어가 보이는 화면, 예측 방법·앱 정보의
// 이번 방문 값을 만드는 순수 함수다. React·DOM이 없고 컴포넌트는 배치만 한다. tests/view-models.test.cjs가 픽스처로 글자를 고정한다.
import type { SourceRowItem } from './sheets';
import type { RouteKey } from '../../app/routes';
import type { MarketStatus, ModelInfo, RealtimeStatus } from '../../data/types';
import { kstAt, parseApiTime, parseDayKey } from '../../lib/kst';
import type { ClockSkew } from '../../lib/kst';
import { CHIPS, forecastChip } from '../../lib/provenance';
import type { Chip, ForecastKind } from '../../lib/provenance';
import type { AccuracyCardView, Fetched } from '../today/model';
import type { PillTone } from '../../ui/tokens';
import {
  ABOUT_SKEW_IMPLAUSIBLE, ACCURACY_SOURCE, ACCURACY_UNREGISTERED, FORECAST_SUBJECT_SHORT, RS_DATE, RS_DATE_OTHER, RS_ITEMS, RS_OVERALL, RS_PART, RS_SOURCE, SOURCE_PENDING,
  aboutSkewText, accuracyBasisPeriod, accuracyBasisText, accuracyExampleText, accuracyR2Text, accuracyText, apiAccuracyText, legendConfirmedText, legendForecastText, legendProbabilityText, legendStatementText, marketFactsText, rsBasisText,
} from '../../content/strings';
import type { GlossaryId } from '../../content/strings';

const own = <T extends object>(table: T, key: unknown): key is keyof T =>
  typeof key === 'string' && Object.prototype.hasOwnProperty.call(table, key);
/** 목록 밖 값(빈 값·오타·새 상태)은 닫힌 쪽: "알 수 없음"(4.4절). */
export const overallText = (s: unknown) => (own(RS_OVERALL, s) ? RS_OVERALL[s] : RS_PART.unknown);
export const partText = (s: unknown) => (own(RS_PART, s) ? RS_PART[s] : RS_PART.unknown);
export const dateText = (s: unknown) => (own(RS_DATE, s) ? RS_DATE[s] : RS_DATE_OTHER);
const TONES = { connected: 'ok', all_connected: 'ok', no_data: 'watch', partial: 'watch', error: 'danger', disconnected: 'danger' } as const;
/** 수신 상태 pill 색(5.2절). 글자가 같은 뜻을 늘 함께 준다. 모르는 값은 회색. */
export const statusTone = (s: unknown): PillTone => (own(TONES, s) ? TONES[s] : 'neutral');

const PARTS = [['smp', 'smp_api'], ['supply', 'power_supply_api'], ['weather', 'weather_api']] as const;
export type StatusView = { rows: SourceRowItem[]; overall: { text: string; tone: PillTone; at: number | null } | null };

/** 출처 표의 응답 상태 3행(SMP · 수급 · 기상)과 전체 상태. 값이 없으면 받는 중 · 실패 · 이번 방문에 조회하지 않음을 구분한다. */
export function statusView(r: Fetched<RealtimeStatus>): StatusView {
  const d = r.data;
  if (d === null) {
    const basis = r.status === 'error' ? SOURCE_PENDING.error : r.status === 'idle' ? SOURCE_PENDING.idle : SOURCE_PENDING.loading;
    return { overall: null, rows: PARTS.map(([k]) => ({ id: `status:${k}`, item: RS_ITEMS[k], source: RS_SOURCE, basis, chip: null, status: null, link: null })) };
  }
  const at = parseApiTime(d.timestamp);
  const rows = PARTS.map(([k, key]): SourceRowItem => {
    const part = d[key];
    // 날짜 상태는 SMP에만 있는 선택 필드다. 빠진 부분은 상태를 모르는 것으로 본다.
    const date = k === 'smp' && part && part.data_date_status !== null ? dateText(part.data_date_status) : null;
    const status = part ? partText(part.status) : RS_PART.unknown;
    return { id: `status:${k}`, item: RS_ITEMS[k], source: RS_SOURCE, basis: rsBasisText(at), chip: null,
      status: date ? `${status} · ${date}` : status, tone: statusTone(part ? part.status : null), link: null };
  });
  return { rows, overall: { text: overallText(d.overall_status), tone: statusTone(d.overall_status), at } };
}

/** 용어 사전 항목의 "이 앱에서 보는 곳 ›"(4.4절). */
export const TERM_WHERE: Readonly<Record<GlossaryId, RouteKey>> = Object.freeze({
  smp: 'smp', dam: 'bidding', rtm: 'bidding', 'trade-hour': 'smp', demand: 'smp', supply: 'smp', reserve: 'smp', baseline: 'smp',
  band: 'smp', error: 'smp', clearing: 'bidding', settlement: 'settlement', units: 'settlement',
});

export type LegendRow = { chip: Chip; text: string };
export type TourLegendInput = {
  /** 지금 거래시간의 확정 가격(오늘 화면 값, 거래시간이 어긋나면 null). */
  confirmed: number | null;
  /** 같은 응답의 예측 종류와 지금 거래시간 q50. */
  forecast: { kind: ForecastKind | null; value: number | null };
  /** 예시 발전소 1구간의 낙찰 가능성(입찰 계산, 추정). */
  probability: number | null;
  statementTotal: number;
};
/**
 * 둘러보기 5단계 "숫자 옆 표시를 보세요"(4.4절): 이번 방문의 실제 값으로 네 표시를 보인다. 값이 없으면 숫자 없이 표시와 설명만.
 * 예측이 기준선이 아니면(학습 모델·대체값) 그 종류의 칩과 이름만 쓴다(기준선 문구에 다른 값을 넣지 않는다).
 */
export function tourLegend(i: TourLegendInput): LegendRow[] {
  const kind = i.forecast.kind;
  const forecast: LegendRow = kind && kind !== 'baseline'
    ? { chip: forecastChip(kind), text: FORECAST_SUBJECT_SHORT[kind] }
    : { chip: CHIPS.forecastBaseline, text: legendForecastText(kind === 'baseline' ? i.forecast.value : null) };
  return [
    { chip: CHIPS.confirmedPrice, text: legendConfirmedText(i.confirmed) },
    forecast,
    { chip: CHIPS.estShort, text: legendProbabilityText(i.probability) },
    { chip: CHIPS.example, text: legendStatementText(i.statementTotal) },
  ];
}

/**
 * 예측 방법 화면의 정확도(4.4·7.7절): 오늘 화면의 정확도 카드와 같고 여기서만 R²를 더한다. kind는 이번 방문에 받은 예측의 종류다.
 * 받지 않았으면(알아보기로 바로 들어온 방문) 이 화면이 설명하는 기준선의 문서화 수치를 그대로 보인다(수치 줄이 "기준선 자체"라고 밝힌다).
 * 기준선이 아닌 예측을 받았으면 문서화 수치를 쓰지 않는다.
 */
export function methodAccuracy(kind: ForecastKind | null, info: ModelInfo | null): AccuracyCardView {
  const baseline = kind === null || kind === 'baseline';
  return {
    state: 'ready',
    documented: baseline ? { main: accuracyText(), basis: accuracyBasisText(), example: accuracyExampleText(), r2: accuracyR2Text() } : null,
    unregistered: baseline ? null : ACCURACY_UNREGISTERED,
    api: info ? apiAccuracyText(info.mape, info.coverage) : null,
    footer: { source: ACCURACY_SOURCE, time: accuracyBasisPeriod() },
  };
}

/** 예측 방법 화면의 이번 방문 서버 값: 하루전 마감(입찰일 하루 전 dam.deadline) · 입찰일 · 실시간시장 구간 길이. 시장 상태를 받지 않았으면 null. */
export function marketFacts(market: MarketStatus | null): string | null {
  if (!market) return null;
  const trading = parseDayKey(market.dam.trading_date);
  const minutes = market.rtm.interval_minutes;
  return marketFactsText(trading ? kstAt(trading, market.dam.deadline, -1) : null, trading,
    typeof minutes === 'number' && Number.isFinite(minutes) && minutes > 0 ? minutes : null);
}

/** 앱 정보의 시계 줄(7.6절): 120초 넘게 다르면 "기기 시계와 N분 차이 · 서버 시각 기준으로 표시", 12시간 넘으면 서버 시각을 믿지 않는다. */
export function aboutClockLines(skew: ClockSkew | null): string[] {
  if (!skew) return [];
  if (!skew.plausible) return [ABOUT_SKEW_IMPLAUSIBLE];
  return skew.notable ? [aboutSkewText(skew.skewMs)] : [];
}
