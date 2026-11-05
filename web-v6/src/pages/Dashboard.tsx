/**
 * Dashboard Page - RE-BMS v6.0
 * Main Command Center with KPIs and Real-time Data
 */

import {
  Zap,
  TrendingUp,
  Sun,
  Wind,
  Battery,
  Activity,
  Clock,
  DollarSign,
  BarChart3,
} from 'lucide-react';
import KPICard from '../components/Cards/KPICard';
import SMPChart from '../components/Charts/SMPChart';
import PowerSupplyChart from '../components/Charts/PowerSupplyChart';
import {
  useDashboardKPIs,
  useSMPForecast,
  useMarketStatus,
  useResources,
  useAutoRefresh,
  useDayAheadSMP,
  usePowerSupply,
  useModelInfo,
} from '../hooks/useApi';
import type { PowerChartData } from '../types';
import clsx from 'clsx';

// KPX 예비율 경보 기준 (Streamlit 대시보드와 동일)
function reserveLevel(rate: number) {
  if (rate < 5) return '위험';
  if (rate < 10) return '주의';
  if (rate < 15) return '관심';
  return '정상';
}

interface MarketBadgeProps {
  market: 'DAM' | 'RTM';
  status: string;
  deadline?: string;
  hoursRemaining?: number;
}

function MarketBadge({ market, status, deadline, hoursRemaining }: MarketBadgeProps) {
  const isOpen = status === 'open' || status === 'active';
  const isClosing = hoursRemaining !== undefined && hoursRemaining <= 2;

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-text-muted">{market}</span>
        <div
          className={clsx(
            'status-dot',
            isOpen ? (isClosing ? 'status-warning' : 'status-success') : 'status-danger'
          )}
        />
      </div>
      <div className="text-lg font-bold text-text-primary">
        {isOpen ? '거래 가능' : '마감'}
      </div>
      {deadline && (
        <div className="text-sm text-text-muted mt-1">
          마감: {deadline}
          {hoursRemaining !== undefined && hoursRemaining > 0 && (
            <span className="ml-1 text-warning">({hoursRemaining}시간 남음)</span>
          )}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { data: kpis, error: kpiError, refetch: refetchKPIs } = useDashboardKPIs();
  const { data: smpForecast, error: forecastError, refetch: refetchSMP } = useSMPForecast();
  const { data: marketStatus, error: marketError } = useMarketStatus();
  const { data: resources, error: resourcesError } = useResources();
  const { data: dayAhead, error: dayAheadError, refetch: refetchDayAhead } = useDayAheadSMP();
  const { data: powerSupply, error: powerSupplyError } = usePowerSupply();
  const { data: modelInfo } = useModelInfo();

  // Auto refresh every 60 seconds
  useAutoRefresh(() => {
    refetchKPIs();
    refetchSMP();
    refetchDayAhead();
  }, 60000);

  const powerChartData: PowerChartData[] = powerSupply?.data.map((d) => ({
    time: d.time,
    demand: d.demand,
    supply: d.supply,
    reserve: d.supply - d.demand,
    solar: d.solar,
    wind: d.wind,
    forecast: d.is_forecast,
  })) ?? [];
  // 응답 일부가 빠져도 페이지가 깨지지 않도록 숫자일 때만 쓴다.
  const reserveRate = !kpiError && typeof kpis?.supply_reserve_rate === 'number' ? kpis.supply_reserve_rate : undefined;
  const utilization = !kpiError && typeof kpis?.utilization_pct === 'number' ? kpis.utilization_pct : undefined;
  // 예측 배열의 인덱스 i는 i시~i+1시 구간(거래시간 i+1)이며 v5·모바일과 같이 시작 시각으로 표기한다.
  const forecastPeak = smpForecast && !forecastError && smpForecast.q50.length > 0
    ? smpForecast.q50.reduce((best, v, i) => (v > best.value ? { hour: i, value: v } : best), { hour: 0, value: smpForecast.q50[0] })
    : null;

  // Calculate resource stats
  const resourceStats = resources
    ? {
        totalCapacity: resources.reduce((sum, r) => sum + r.capacity, 0),
        currentOutput: resources.reduce((sum, r) => sum + r.current_output, 0),
        solarOutput: resources
          .filter((r) => r.type === 'solar')
          .reduce((sum, r) => sum + r.current_output, 0),
        windOutput: resources
          .filter((r) => r.type === 'wind')
          .reduce((sum, r) => sum + r.current_output, 0),
      }
    : null;

  // v7·모바일과 같은 KPX 제주 하루전시장 현재 거래시간 가격
  const currentSMP = !dayAheadError && dayAhead?.status === 'available' ? dayAhead.current_smp : null;
  const solarOutput = resourcesError ? undefined : resourceStats?.solarOutput;
  const windOutput = resourcesError ? undefined : resourceStats?.windOutput;
  const errors = [kpiError && 'KPI', forecastError && 'SMP 예측', marketError && '시장 상태', resourcesError && '발전소', powerSupplyError && '전력수급'].filter(Boolean);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">대시보드</h1>
          <p className="text-text-muted mt-1">제주 전력 시스템 현황 (일부 시뮬레이션 및 예시)</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
            <Clock className="w-4 h-4 text-text-muted flex-shrink-0" />
            <span className="text-sm text-text-muted whitespace-nowrap">
              관측 시각이 제공되지 않음
            </span>
          </div>
          <button
            onClick={() => {
              refetchKPIs();
              refetchSMP();
            }}
            className="btn-secondary flex items-center gap-2 whitespace-nowrap"
          >
            <Activity className="w-4 h-4" />
            새로고침
          </button>
        </div>
      </div>

      {errors.length > 0 && (
        <div role="alert" className="p-3 rounded-lg bg-warning/10 text-warning">
          API 오류 ({errors.join(', ')}): 일부 화면에 데모 데이터가 표시됩니다.
        </div>
      )}
      {kpis?.data_source && !kpiError && (
        <p className="text-xs text-text-muted">KPI 데이터 출처: {kpis.data_source} · SMP: {dayAhead?.data_source ?? '확인 불가'}</p>
      )}

      {/* Market Status */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <MarketBadge
          market="DAM"
          status={marketStatus?.dam.status ?? 'open'}
          deadline={marketStatus?.dam.deadline ?? '10:00'}
          hoursRemaining={marketStatus?.dam.hours_remaining ?? 4}
        />
        <MarketBadge
          market="RTM"
          status={marketStatus?.rtm.status ?? 'active'}
          deadline={marketStatus?.rtm.next_interval ?? '15분 후'}
        />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title={dayAhead?.trade_hour != null ? `제주 하루전시장 SMP (${String(dayAhead.trade_hour).padStart(2, '0')}h)` : '제주 하루전시장 SMP'}
          value={currentSMP != null ? currentSMP.toFixed(1) : '—'}
          unit="원/kWh"
          icon={DollarSign}
          color="smp"
        />
        <KPICard
          title="총 발전량 (시뮬레이션)"
          value={resourcesError ? '—' : resourceStats?.currentOutput.toFixed(1) ?? '—'}
          unit="MW"
          icon={Zap}
          color="primary"
        />
        <KPICard
          title="태양광 발전 (시뮬레이션)"
          value={solarOutput?.toFixed(1) ?? '—'}
          unit="MW"
          icon={Sun}
          color="solar"
        />
        <KPICard
          title="풍력 발전 (시뮬레이션)"
          value={windOutput?.toFixed(1) ?? '—'}
          unit="MW"
          icon={Wind}
          color="wind"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Power Supply Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-text-primary">전력수급 현황</h2>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs bg-success/20 text-success rounded">
                {powerChartData.length > 0 ? '패턴·예측' : '데모 데이터'}
              </span>
            </div>
          </div>
          {powerSupply?.data_source && (
            <p className="text-xs text-text-muted mb-2">출처: {powerSupply.data_source}</p>
          )}
          <PowerSupplyChart data={powerChartData} height={280} />
        </div>

        {/* SMP Forecast Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-text-primary">24시간 SMP 예측</h2>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs bg-primary/20 text-primary rounded">
                {forecastError ? '데모 데이터' : smpForecast?.model_used ?? '데이터 없음'}
              </span>
              {smpForecast?.forecast_date && !forecastError && (
                <span className="text-xs text-text-muted">예측일 {smpForecast.forecast_date}</span>
              )}
              {smpForecast?.confidence != null && (
                <span className="text-xs text-text-muted">
                  신뢰도: {(smpForecast.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>
          <SMPChart forecast={smpForecast} height={280} />
        </div>
      </div>

      {/* Bottom Section - Stats & Alerts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Resource Summary */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">발전소 현황 (출력 시뮬레이션)</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-wind/10 rounded-lg flex items-center justify-center">
                  <Wind className="w-5 h-5 text-wind" />
                </div>
                <div>
                  <div className="text-text-primary font-medium">풍력 발전소</div>
                  <div className="text-sm text-text-muted">
                    {resourcesError || !resources ? '정보 없음' : `${resources.filter((r) => r.type === 'wind').length}개 운영중`}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-wind font-bold">
                  {windOutput?.toFixed(1) ?? '—'} MW
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-solar/10 rounded-lg flex items-center justify-center">
                  <Sun className="w-5 h-5 text-solar" />
                </div>
                <div>
                  <div className="text-text-primary font-medium">태양광 발전소</div>
                  <div className="text-sm text-text-muted">
                    {resourcesError || !resources ? '정보 없음' : `${resources.filter((r) => r.type === 'solar').length}개`}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-solar font-bold">
                  {solarOutput?.toFixed(1) ?? '—'} MW
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Daily Stats */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">일일 실적</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-text-muted">예상 수익</span>
                <span className="text-success font-mono">
                  {!kpiError && typeof kpis?.daily_revenue_million === 'number' ? `${kpis.daily_revenue_million.toFixed(1)}백만원` : '산출 불가'}
                </span>
              </div>
              <p className="text-xs text-text-muted">실제 발전량·정산 데이터가 없어 산출하지 않습니다.</p>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-text-muted">설비 가동률 (시뮬레이션)</span>
                <span className="text-primary font-mono">
                  {utilization != null ? `${utilization.toFixed(1)}%` : '—'}
                </span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full"
                  style={{ width: `${utilization != null ? Math.min(100, utilization) : 0}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-text-muted">SMP 예측 정확도</span>
                <span className="text-smp font-mono">
                  {typeof modelInfo?.mape === 'number' ? `MAPE ${modelInfo.mape.toFixed(2)}%` : '재검증 전'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Alerts */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-text-primary">알림</h3>
            <span className="px-2 py-1 text-xs bg-success/20 text-success rounded">
              {reserveRate != null ? reserveLevel(reserveRate) : '확인 불가'}
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-success/10 rounded-lg border border-success/20">
              <Battery className="w-5 h-5 text-success mt-0.5" />
              <div>
                <div className="text-text-primary text-sm font-medium">
                  {reserveRate != null ? `예비율 ${reserveLevel(reserveRate)}` : '예비율 확인 불가'}
                </div>
                <div className="text-text-muted text-xs">
                  {reserveRate != null ? `현재 예비율 ${reserveRate.toFixed(1)}%` : 'KPI 데이터 없음'}
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-primary/10 rounded-lg border border-primary/20">
              <BarChart3 className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <div className="text-text-primary text-sm font-medium">
                  SMP 예측 모델
                </div>
                <div className="text-text-muted text-xs">
                  {modelInfo ? `${modelInfo.version} 사용 중 · ${modelInfo.status}` : '모델 정보 없음'}
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-background rounded-lg border border-border">
              <TrendingUp className="w-5 h-5 text-text-muted mt-0.5" />
              <div>
                <div className="text-text-primary text-sm font-medium">
                  예측 최고 SMP
                </div>
                <div className="text-text-muted text-xs">
                  {forecastPeak ? `${forecastPeak.hour}시 ${forecastPeak.value.toFixed(1)}원/kWh${smpForecast?.forecast_date ? ` (${smpForecast.forecast_date})` : ''}` : '예측 데이터 없음'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
