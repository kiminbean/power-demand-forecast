// 출처 판정(명세 7.1·7.4·7.7절). 문구는 대시보드와 한 글자도 다르지 않게 둔다:
// 수요·예비율은 src/dashboard/app_v4.py 3352–3366행, 기상은 _fmt_weather_provenance, 단계는 3296–3316행.
import type { ChipKind } from '../ui/tokens';

export type Chip = Readonly<{ kind: ChipKind; text: string }>;
export const CHIP_GLYPH: Readonly<Record<ChipKind, string>> = { measured: '●', forecast: '┅', estimated: '▨', example: '┆', plain: '' };
const chip = (kind: ChipKind, text: string): Chip => Object.freeze({ kind, text });

// 4상태 문법(7.1절). "●"는 지금 잰 값과 확정 가격에만, 시장 규칙·정확도·제한은 글리프 없는 일반 칩.
export const CHIPS = Object.freeze({
  confirmedPrice: chip('measured', '확정 가격'),
  confirmedPoint: chip('measured', '확정 가격 1점'),
  measured: chip('measured', '실측'),
  forecastBaseline: chip('forecast', '예측 · 기준선'),
  forecastModel: chip('forecast', '예측 · 학습 모델'),
  estNotMeasured: chip('estimated', '추정 · 실측 아님'),
  estQuantile: chip('estimated', '추정 · 기준선 분위수'),
  estModelQuantile: chip('estimated', '추정 · 학습 모델 분위수'),
  estShort: chip('estimated', '추정'),
  estApiFallback: chip('estimated', '추정 · API 대체값 (모델 예측 아님)'),
  estNotRealtime: chip('estimated', '추정 · 실시간 아님'),
  estBasis: chip('estimated', '추정값 기준'),
  simNotMeasured: chip('estimated', '시뮬레이션 · 실측 아님'),
  defaultNotMeasured: chip('estimated', '기본값 · 실측 아님'),
  national: chip('estimated', '전국 값 · 제주 아님'),
  example: chip('example', '예시'),
  exampleRibbon: chip('example', '예시 · 실제 값 아님'),
  exampleVirtual: chip('example', '예시 · 가상 발전소'),
  verified: chip('plain', '검증값'),
  apiStated: chip('plain', 'API 표기'),
  marketRule: chip('plain', '시장 규칙 · 서버 시각'),
  locked: chip('plain', '잠김'),
  checkTime: chip('plain', '시각 확인 필요'),
  simplified: chip('plain', '단순화한 설명'),
});

/** 보이는 칩 글자(글리프 + 문구). */
export const chipText = (c: Chip) => (CHIP_GLYPH[c.kind] ? `${CHIP_GLYPH[c.kind]} ${c.text}` : c.text);
/** 칩의 긴 접근 이름: "상태: 확정 가격, 출처 KPX 제주 하루전시장" */
export const chipA11yLabel = (c: Chip, source?: string | null) => `상태: ${c.text}${source ? `, 출처 ${source}` : ''}`;

type Rec = Record<string, unknown>;
const isRecord = (value: unknown): value is Rec => typeof value === 'object' && value !== null && !Array.isArray(value);
const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export type DemandCase = 2 | 3 | 4 | 5 | 6;
export type DemandProvenance = Readonly<{
  case: DemandCase;
  demandState: 'measured' | 'estimated';
  reserveState: 'measured' | 'estimated' | 'default' | 'national';
  demandChip: Chip;
  reserveChip: Chip;
  /** 예비율 값 뒤에 붙는 덧말(대시보드 reserve_note). */
  reserveNote: string;
  source: string;
  /** 예비율 단계 판정을 하는가(③일 때만). */
  judge: boolean;
  /** 판정하지 않는 까닭(대시보드 reserve_why). */
  why: string | null;
}>;

const DEMAND_CASES: Readonly<Record<DemandCase, DemandProvenance>> = {
  2: { case: 2, demandState: 'estimated', reserveState: 'estimated', demandChip: CHIPS.estNotMeasured, reserveChip: CHIPS.estNotMeasured,
    reserveNote: ' · 실측 아님', source: 'API 서버 값 (출처 확인 불가)', judge: false, why: null },
  3: { case: 3, demandState: 'measured', reserveState: 'measured', demandChip: CHIPS.measured, reserveChip: CHIPS.measured,
    reserveNote: '', source: 'KPX 제주 실시간 (API 서버)', judge: true, why: null },
  4: { case: 4, demandState: 'estimated', reserveState: 'default', demandChip: CHIPS.estNotMeasured, reserveChip: CHIPS.defaultNotMeasured,
    reserveNote: ' · API 기본값 (실측 아님)', source: 'EPSIS 30일 평균 (API 서버 대체값)', judge: false,
    why: 'API 서버가 실측 대신 기본 예비율(25%)을 보냈습니다' },
  5: { case: 5, demandState: 'estimated', reserveState: 'national', demandChip: CHIPS.estNotMeasured, reserveChip: CHIPS.national,
    reserveNote: ' · 전국 예비율 (제주 값 아님)', source: '전국 값 × 0.018 비례 추정 (API 서버)', judge: false,
    why: 'API 서버가 제주 대신 전국 예비율을 보냈습니다' },
  6: { case: 6, demandState: 'estimated', reserveState: 'default', demandChip: CHIPS.simNotMeasured, reserveChip: CHIPS.defaultNotMeasured,
    reserveNote: ' · API 기본값 (실측 아님)', source: 'API 서버 기본값 650 MW·25% (실측 아님)', judge: false,
    why: 'API 서버가 실측 대신 기본값(수요 650 MW·예비율 25%)을 보냈습니다' },
};

/** /dashboard/kpis의 data_source를 위에서부터 첫 일치로 본다(대시보드 7.2절 ②–⑥). */
export function demandProvenance(kpis: unknown): DemandProvenance {
  const src = isRecord(kpis) ? kpis.data_source : undefined;
  if (typeof src !== 'string' || !src.trim()) return DEMAND_CASES[2];
  if (src.includes('Jeju Crawler')) return DEMAND_CASES[3];
  if (src.includes('EPSIS Demand')) return DEMAND_CASES[4];
  if (src.includes('KPX')) return DEMAND_CASES[5];
  return DEMAND_CASES[6];
}

export const PAIR_TOLERANCE_MW = 1;
/** 관심 경계 = 공급능력 ÷ 1.15 (예비율 15 %가 되는 수요). */
export const WATCH_RATIO = 1.15;
export const SUPPLY_MISSING = '공급능력 미수신';

export type SupplyPair =
  | Readonly<{ ok: true; hour: number; supplyMw: number; demandMw: number; reserveMw: number; watchBoundaryMw: number }>
  | Readonly<{ ok: false; reason: typeof SUPPLY_MISSING }>;

/**
 * /power-supply에서 hour === current_hour인 행 하나만 본다. 그 행의 demand가 KPI 수요와 ±1 MW 안이고 수요가 ③일 때만
 * 그 행의 supply를 실측 공급능력으로 쓴다. 다른 시각 행(패턴)은 쓰지 않는다. 반올림은 표시할 때 1회.
 */
export function supplyPair(kpis: unknown, powerSupply: unknown): SupplyPair {
  const missing = { ok: false, reason: SUPPLY_MISSING } as const;
  if (!isRecord(kpis) || !isRecord(powerSupply) || demandProvenance(kpis).case !== 3) return missing;
  const demand = kpis.current_demand_mw;
  const hour = powerSupply.current_hour;
  const rows = powerSupply.data;
  if (!isFiniteNumber(demand) || typeof hour !== 'number' || !Number.isInteger(hour) || !Array.isArray(rows)) return missing;
  const row = rows.find(r => isRecord(r) && r.hour === hour);
  if (!isRecord(row) || !isFiniteNumber(row.supply) || !isFiniteNumber(row.demand) || row.supply <= 0) return missing;
  // 1e-9: 664.3과 665.3의 차이가 부동소수로 1.0000000000000568이 되는 것을 경계 안으로 둔다.
  if (Math.abs(row.demand - demand) > PAIR_TOLERANCE_MW + 1e-9) return missing;
  return { ok: true, hour, supplyMw: row.supply, demandMw: demand, reserveMw: row.supply - demand, watchBoundaryMw: row.supply / WATCH_RATIO };
}

export type ReserveTone = 'ok' | 'watch' | 'caution' | 'danger';
export type ReserveLevel = Readonly<{ tone: ReserveTone; name: string; verdict: string; label: string }>;
const LEVELS: ReadonlyArray<{ min: number; tone: ReserveTone; name: string; verdict: string }> = [
  { min: 15, tone: 'ok', name: '정상', verdict: '넉넉합니다' },
  { min: 10, tone: 'watch', name: '관심', verdict: '여유가 줄고 있습니다' },
  { min: 5, tone: 'caution', name: '주의', verdict: '여유가 부족합니다' },
  { min: -Infinity, tone: 'danger', name: '위험', verdict: '공급이 빠듯합니다' },
];

/** 예비율 단계(대시보드 7.4절). 앱 기준이며 KPX 공식 경보 단계가 아니다. 판정 여부는 호출하는 쪽이 정한다(③·끊김 아님). */
export function reserveLevel(rate: unknown): ReserveLevel | null {
  if (!isFiniteNumber(rate)) return null;
  const level = LEVELS.find(l => rate >= l.min) ?? LEVELS[LEVELS.length - 1];
  return { tone: level.tone, name: level.name, verdict: level.verdict, label: `${level.name} · ${level.verdict}` };
}

export type WeatherKind = 'live' | 'file' | 'default' | 'none';
export type WeatherProvenance = Readonly<{ kind: WeatherKind; chip: Chip | null; source: string | null }>;
const WEATHER: Readonly<Record<WeatherKind, WeatherProvenance>> = {
  live: { kind: 'live', chip: CHIPS.measured, source: '기상청 초단기실황 (API 서버)' },
  file: { kind: 'file', chip: CHIPS.estNotRealtime, source: '기상청 과거 자료 24시간 평균 (API 서버) · 실시간 아님' },
  // API는 실시간·파일이 모두 실패하면 5.0 °C·3.0 m/s·60 %·맑음을 출처 표시 없이 보낸다(api/v6_routes.py 433–464행).
  default: { kind: 'default', chip: CHIPS.defaultNotMeasured, source: 'API 서버 기본값 (실측 아님)' },
  none: { kind: 'none', chip: null, source: null },
};
const LIVE_WEATHER_TOKENS = ['KMA Weather Crawler', 'KMA Real-time API', '기상청 (실시간)'];

/** 기상 값의 출처 종류(대시보드 _fmt_weather_provenance). 값이 유한하다는 것만으로 실측이라 하지 않는다. */
export function weatherProvenance(kpis: unknown): WeatherProvenance {
  if (!isRecord(kpis)) return WEATHER.none;
  const src = typeof kpis.data_source === 'string' ? kpis.data_source : '';
  if (src.includes('KMA Weather (파일)')) return WEATHER.file;
  if (LIVE_WEATHER_TOKENS.some(token => src.includes(token))) return WEATHER.live;
  return WEATHER.default;
}

export type ForecastKind = 'baseline' | 'model' | 'fallback';
const FALLBACK_TOKENS = ['대체값', 'Fallback', 'fallback'];
const BASELINE_TOKENS = ['기준선', '전날 같은 시간', '어제 같은 시각', 'naive', 'baseline'];
// 학습 모델 이름: src/smp/models/serving.py 18·20행(V41_LABEL·V32_LABEL)에 api/v6_routes.py _forecast_label이
// " · 검증 MAPE x.xx%"를 붙인 꼴. 아키텍처 이름 글자는 7.10절 금지 문구라 쓰지 않고 "+Attention 버전 (설명)"으로 알아본다.
const KNOWN_MODELS = [
  /^[A-Za-z]+\+Attention v4\.1 \(제주 하루전 SMP\)( · 검증 MAPE \d+\.\d{2}%)?$/,
  /^[A-Za-z]+\+Attention v3\.2 Optuna \(성능 재검증 전\)( · 검증 MAPE \d+\.\d{2}%)?$/,
];

/** /smp-forecast의 model_used로 예측 종류를 정한다. 대체값 표시가 있거나 모르는 이름이면 닫힌 쪽(추정)이다. */
export function smpForecastKind(forecast: unknown): ForecastKind {
  const used = isRecord(forecast) ? forecast.model_used : undefined;
  if (typeof used !== 'string' || !used.trim()) return 'fallback';
  const name = used.trim();
  if (FALLBACK_TOKENS.some(token => name.includes(token))) return 'fallback';
  const lower = name.toLowerCase();
  if (BASELINE_TOKENS.some(token => lower.includes(token.toLowerCase()))) return 'baseline';
  return KNOWN_MODELS.some(re => re.test(name)) ? 'model' : 'fallback';
}

export function forecastChip(kind: ForecastKind): Chip {
  if (kind === 'baseline') return CHIPS.forecastBaseline;
  return kind === 'model' ? CHIPS.forecastModel : CHIPS.estApiFallback;
}

/** 출처 줄용 짧은 이름(칩·줄이 390 px에서 넘치지 않게). 원문 전체는 산출 방식 시트에 둔다. */
export function forecastShortName(modelUsed: string): string {
  const name = modelUsed.trim();
  if (name.includes('기준선')) return '기준선';
  if (/\bv4\.1\b/.test(name)) return 'v4.1';
  if (/\bv3\.2\b/.test(name)) return 'v3.2 · 재검증 전';
  return name.length > 12 ? `${name.slice(0, 12)}…` : name;
}

/**
 * 문서화된 SMP 정확도(7.7절, 원천: 프로젝트 CLAUDE.md · docs/2026-11-06-competitor-benchmark.md).
 * 기준선(전날 같은 시간 가격) 자체의 walk-forward 수치이며 smpForecastKind가 baseline일 때만 보인다.
 */
export const DOCUMENTED_SMP_ACCURACY = Object.freeze({
  maeKrwPerKwh: 11.97,
  mapePct: 11.75,
  r2: 0.203,
  method: 'walk-forward',
  period: '2025-07-22~12-18',
  basis: '기준선 자체(학습한 모델 아님)',
});
