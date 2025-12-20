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
} from '../hooks/useApi';
import clsx from 'clsx';

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
        <span className="text-sm font-medium text-gray-400">{market}</span>
        <div
          className={clsx(
            'status-dot',
            isOpen ? (isClosing ? 'status-warning' : 'status-success') : 'status-danger'
          )}
        />
      </div>
      <div className="text-lg font-bold text-white">
        {isOpen ? '거래 가능' : '마감'}
      </div>
      {deadline && (
        <div className="text-sm text-gray-400 mt-1">
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
  const { data: kpis, refetch: refetchKPIs } = useDashboardKPIs();
  const { data: smpForecast, refetch: refetchSMP } = useSMPForecast();
  const { data: marketStatus } = useMarketStatus();
  const { data: resources } = useResources();

  // Auto refresh every 60 seconds
  useAutoRefresh(() => {
    refetchKPIs();
    refetchSMP();
  }, 60000);

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

  // Get current SMP from forecast
  const currentHour = new Date().getHours();
  const currentSMP = smpForecast?.q50?.[currentHour] ?? kpis?.current_smp ?? 0;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">대시보드</h1>
          <p className="text-gray-400 mt-1">제주 전력 시스템 실시간 현황</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">
              마지막 업데이트: {new Date().toLocaleTimeString('ko-KR')}
            </span>
          </div>
          <button
            onClick={() => {
              refetchKPIs();
              refetchSMP();
            }}
            className="btn-secondary flex items-center gap-2"
          >
            <Activity className="w-4 h-4" />
            새로고침
          </button>
        </div>
      </div>

      {/* Market Status */}
      <div className="grid grid-cols-2 gap-4">
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
      <div className="grid grid-cols-4 gap-4">
        <KPICard
          title="현재 SMP"
          value={currentSMP.toFixed(1)}
          unit="원/kWh"
          change={kpis?.smp_change_pct}
          icon={DollarSign}
          color="smp"
        />
        <KPICard
          title="총 발전량"
          value={resourceStats?.currentOutput.toFixed(1) ?? kpis?.current_output_mw ?? 0}
          unit="MW"
          change={kpis?.revenue_change_pct}
          icon={Zap}
          color="primary"
        />
        <KPICard
          title="태양광 발전"
          value={resourceStats?.solarOutput.toFixed(1) ?? 45.2}
          unit="MW"
          icon={Sun}
          color="solar"
        />
        <KPICard
          title="풍력 발전"
          value={resourceStats?.windOutput.toFixed(1) ?? 123.3}
          unit="MW"
          icon={Wind}
          color="wind"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-2 gap-6">
        {/* Power Supply Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">전력수급 현황</h2>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs bg-success/20 text-success rounded">
                실시간
              </span>
            </div>
          </div>
          <PowerSupplyChart data={[]} height={280} />
        </div>

        {/* SMP Forecast Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">24시간 SMP 예측</h2>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs bg-primary/20 text-primary rounded">
                AI 예측
              </span>
              {smpForecast && (
                <span className="text-xs text-gray-400">
                  신뢰도: {(smpForecast.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>
          <SMPChart forecast={smpForecast} height={280} />
        </div>
      </div>

      {/* Bottom Section - Stats & Alerts */}
      <div className="grid grid-cols-3 gap-6">
        {/* Resource Summary */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">발전소 현황</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-wind/10 rounded-lg flex items-center justify-center">
                  <Wind className="w-5 h-5 text-wind" />
                </div>
                <div>
                  <div className="text-white font-medium">풍력 발전소</div>
                  <div className="text-sm text-gray-400">
                    {resources?.filter((r) => r.type === 'wind').length ?? 14}개 운영중
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-wind font-bold">
                  {resourceStats?.windOutput.toFixed(1) ?? 123.3} MW
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 bg-background rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-solar/10 rounded-lg flex items-center justify-center">
                  <Sun className="w-5 h-5 text-solar" />
                </div>
                <div>
                  <div className="text-white font-medium">태양광 발전소</div>
                  <div className="text-sm text-gray-400">
                    {resources?.filter((r) => r.type === 'solar').length ?? 6}개 운영중
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-solar font-bold">
                  {resourceStats?.solarOutput.toFixed(1) ?? 45.2} MW
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Daily Stats */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">일일 실적</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">예상 수익</span>
                <span className="text-success font-mono">
                  {(kpis?.daily_revenue_million ?? 245.8).toFixed(1)}백만원
                </span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-success rounded-full"
                  style={{ width: '78%' }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">설비 가동률</span>
                <span className="text-primary font-mono">
                  {(kpis?.utilization_pct ?? 74.9).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full"
                  style={{ width: `${kpis?.utilization_pct ?? 74.9}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">예측 정확도</span>
                <span className="text-smp font-mono">94.5%</span>
              </div>
              <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                  className="h-full bg-smp rounded-full"
                  style={{ width: '94.5%' }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Alerts */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">알림</h3>
            <span className="px-2 py-1 text-xs bg-success/20 text-success rounded">
              정상
            </span>
          </div>
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-success/10 rounded-lg border border-success/20">
              <Battery className="w-5 h-5 text-success mt-0.5" />
              <div>
                <div className="text-white text-sm font-medium">
                  예비율 정상 범위
                </div>
                <div className="text-gray-400 text-xs">현재 예비율 127%</div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-primary/10 rounded-lg border border-primary/20">
              <BarChart3 className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <div className="text-white text-sm font-medium">
                  SMP 예측 모델 업데이트
                </div>
                <div className="text-gray-400 text-xs">v3.1 모델 적용됨</div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 bg-background rounded-lg border border-border">
              <TrendingUp className="w-5 h-5 text-gray-400 mt-0.5" />
              <div>
                <div className="text-white text-sm font-medium">
                  오후 피크 예상
                </div>
                <div className="text-gray-400 text-xs">14:00-16:00 SMP 상승 예상</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
