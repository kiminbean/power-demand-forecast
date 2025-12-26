/**
 * Settlement Page - RE-BMS v7.0
 * KPX Jeju Pilot Settlement with Gemini-Verified Rules
 * - 3-Tier Penalty System (±8%, ±15%, >15%)
 * - DA-RT Dual Settlement
 * - Capacity Payment (CP)
 */

import { useState } from 'react';
import {
  Calendar,
  Download,
  AlertTriangle,
  CheckCircle,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Clock,
  TrendingUp,
  Shield,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  ComposedChart,
  Line,
  Area,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { useSettlements, useSettlementSummary } from '../hooks/useApi';
import { useTheme } from '../contexts/ThemeContext';
import clsx from 'clsx';

export default function Settlement() {
  const { data: settlements, isLoading: settlementsLoading } = useSettlements(7);
  const { data: summary, isLoading: summaryLoading } = useSettlementSummary();
  const [period, setPeriod] = useState<'week' | 'month' | 'quarter'>('week');
  const { isDark } = useTheme();

  // Theme-aware chart colors
  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tier1: '#22c55e',  // Green - No penalty
    tier2: '#f59e0b',  // Yellow - Mild penalty
    tier3: '#ef4444',  // Red - Severe penalty
    cp: '#6366f1',     // Indigo - Capacity Payment
    da: '#3b82f6',     // Blue - DA SMP
    rt: '#8b5cf6',     // Purple - RT SMP
  };

  // Process settlement data for charts
  const chartData = settlements?.map((record) => ({
    date: record.date,
    displayDate: record.date.slice(5), // MM-DD format
    revenue: record.revenue_million,
    imbalance: record.imbalance_million,
    cp: record.capacity_payment_million,
    netRevenue: record.net_revenue_million,
    accuracy: record.accuracy_pct,
    daSmp: record.avg_da_smp,
    rtSmp: record.avg_rt_smp,
    tier1: record.hours_tier1,
    tier2: record.hours_tier2,
    tier3: record.hours_tier3,
    zeroRisk: record.hours_zero_risk,
    generation: record.generation_mwh,
    cleared: record.cleared_mwh,
  })) || [];

  // Tier distribution for pie chart
  const tierData = summary ? [
    { name: 'Tier 1 (±8%)', value: summary.total_hours_tier1, fill: chartColors.tier1 },
    { name: 'Tier 2 (±8~15%)', value: summary.total_hours_tier2, fill: chartColors.tier2 },
    { name: 'Tier 3 (>±15%)', value: summary.total_hours_tier3, fill: chartColors.tier3 },
  ] : [];

  // Custom tooltip for revenue chart
  const RevenueTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-xl min-w-[200px]">
          <p className="text-text-primary font-medium mb-2">{data.date}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-success">발전수익:</span>
              <span className="font-mono">{data.revenue?.toFixed(1)}백만</span>
            </div>
            <div className="flex justify-between">
              <span className="text-primary">용량정산(CP):</span>
              <span className="font-mono">{data.cp?.toFixed(2)}백만</span>
            </div>
            <div className="flex justify-between">
              <span className={data.imbalance >= 0 ? 'text-success' : 'text-danger'}>
                불균형:
              </span>
              <span className="font-mono">
                {data.imbalance >= 0 ? '+' : ''}{data.imbalance?.toFixed(2)}백만
              </span>
            </div>
            <hr className="border-border my-1" />
            <div className="flex justify-between font-bold">
              <span className="text-text-primary">순수익:</span>
              <span className="font-mono">{data.netRevenue?.toFixed(1)}백만</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom tooltip for tier chart
  const TierTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-xl min-w-[180px]">
          <p className="text-text-primary font-medium mb-2">{data.date}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-success">Tier 1 (±8%):</span>
              <span className="font-mono">{data.tier1}시간</span>
            </div>
            <div className="flex justify-between">
              <span className="text-warning">Tier 2 (±8~15%):</span>
              <span className="font-mono">{data.tier2}시간</span>
            </div>
            <div className="flex justify-between">
              <span className="text-danger">Tier 3 ({'>'}±15%):</span>
              <span className="font-mono">{data.tier3}시간</span>
            </div>
            <hr className="border-border my-1" />
            <div className="flex justify-between">
              <span className="text-purple-400">0원 리스크:</span>
              <span className="font-mono">{data.zeroRisk}시간</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  if (settlementsLoading || summaryLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        <span className="ml-3 text-text-muted">정산 데이터 로딩 중...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">정산</h1>
          <p className="text-text-muted mt-1">KPX 제주 시범사업 이중 정산 (Gemini 검증)</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Period Selector */}
          <div className="flex items-center bg-card rounded-lg border border-border p-1">
            {(['week', 'month', 'quarter'] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded-md transition-colors whitespace-nowrap',
                  period === p
                    ? 'bg-primary text-text-primary'
                    : 'text-text-muted hover:text-text-primary'
                )}
              >
                {p === 'week' && '1주'}
                {p === 'month' && '1개월'}
                {p === 'quarter' && '분기'}
              </button>
            ))}
          </div>
          <button className="btn-secondary flex items-center gap-2 whitespace-nowrap">
            <Calendar className="w-4 h-4" />
            <span className="hidden sm:inline">기간 선택</span>
          </button>
          <button className="btn-primary flex items-center gap-2 whitespace-nowrap">
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">리포트 다운로드</span>
          </button>
        </div>
      </div>

      {/* Summary KPI Cards - Row 1 */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {/* Generation Revenue */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">발전 수익</span>
            <TrendingUp className="w-4 h-4 text-success" />
          </div>
          <div className="text-2xl font-bold text-success">
            {summary?.generation_revenue_million.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">백만</span>
          </div>
          <div className="flex items-center text-xs text-success mt-1">
            <ArrowUpRight className="w-3 h-3" />
            {summary?.generation_change_pct.toFixed(1)}%
          </div>
        </div>

        {/* Capacity Payment */}
        <div className="card bg-gradient-to-br from-indigo-500/10 to-transparent">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">용량정산(CP)</span>
            <Zap className="w-4 h-4 text-primary" />
          </div>
          <div className="text-2xl font-bold text-primary">
            {summary?.capacity_payment_million.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">백만</span>
          </div>
          <div className="text-xs text-text-muted mt-1">
            CP 지급률 기반
          </div>
        </div>

        {/* Imbalance Charges */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">불균형 정산</span>
            <AlertTriangle className="w-4 h-4 text-warning" />
          </div>
          <div className={clsx(
            'text-2xl font-bold',
            (summary?.imbalance_charges_million || 0) >= 0 ? 'text-success' : 'text-danger'
          )}>
            {(summary?.imbalance_charges_million || 0) >= 0 ? '' : ''}{summary?.imbalance_charges_million.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">백만</span>
          </div>
          <div className={clsx(
            'flex items-center text-xs mt-1',
            (summary?.imbalance_change_pct || 0) >= 0 ? 'text-success' : 'text-danger'
          )}>
            {(summary?.imbalance_change_pct || 0) >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
            {Math.abs(summary?.imbalance_change_pct || 0).toFixed(1)}%
          </div>
        </div>

        {/* Net Revenue */}
        <div className="card bg-gradient-to-br from-primary/10 to-transparent">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">순 수익</span>
            <CheckCircle className="w-4 h-4 text-primary" />
          </div>
          <div className="text-2xl font-bold text-primary">
            {summary?.net_revenue_million.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">백만</span>
          </div>
          <div className="flex items-center text-xs text-primary mt-1">
            <ArrowUpRight className="w-3 h-3" />
            {summary?.net_change_pct.toFixed(1)}%
          </div>
        </div>

        {/* Forecast Accuracy */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">예측 정확도</span>
            <Shield className="w-4 h-4 text-success" />
          </div>
          <div className={clsx(
            'text-2xl font-bold',
            (summary?.forecast_accuracy_pct || 0) >= 95 ? 'text-success' :
            (summary?.forecast_accuracy_pct || 0) >= 90 ? 'text-warning' : 'text-danger'
          )}>
            {summary?.forecast_accuracy_pct.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">%</span>
          </div>
          <div className="flex items-center text-xs text-success mt-1">
            <ArrowUpRight className="w-3 h-3" />
            {summary?.accuracy_change_pct.toFixed(1)}%
          </div>
        </div>

        {/* Zero Risk Hours */}
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">0원 리스크</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-400">
            {summary?.total_hours_zero_risk}
            <span className="text-sm text-text-muted ml-1">시간</span>
          </div>
          <div className="text-xs text-text-muted mt-1">
            RT-SMP 0원 시간대
          </div>
        </div>
      </div>

      {/* DA/RT SMP Info Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">평균 DA-SMP</span>
          </div>
          <div className="text-xl font-bold text-blue-400">
            {summary?.avg_da_smp.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">원/kWh</span>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">평균 RT-SMP</span>
          </div>
          <div className="text-xl font-bold text-purple-400">
            {summary?.avg_rt_smp.toFixed(1)}
            <span className="text-sm text-text-muted ml-1">원/kWh</span>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">낙찰량</span>
          </div>
          <div className="text-xl font-bold text-text-primary">
            {(summary?.total_cleared_mwh || 0).toLocaleString()}
            <span className="text-sm text-text-muted ml-1">MWh</span>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-text-muted">실발전량</span>
          </div>
          <div className="text-xl font-bold text-text-primary">
            {(summary?.total_actual_mwh || 0).toLocaleString()}
            <span className="text-sm text-text-muted ml-1">MWh</span>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Trend */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">수익 추이 (발전 + CP)</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis
                  dataKey="displayDate"
                  stroke={chartColors.axis}
                  fontSize={11}
                  tickLine={false}
                />
                <YAxis stroke={chartColors.axis} fontSize={12} tickLine={false} />
                <Tooltip content={<RevenueTooltip />} />
                <Legend
                  wrapperStyle={{ paddingTop: 10 }}
                  formatter={(value) => <span className="text-text-muted text-sm">{value}</span>}
                />
                <Area
                  type="monotone"
                  dataKey="revenue"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="url(#revenueGradient)"
                  name="발전 수익"
                />
                <Bar
                  dataKey="cp"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                  name="용량정산(CP)"
                />
                <Line
                  type="monotone"
                  dataKey="netRevenue"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="순 수익"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Tier Distribution */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">페널티 구간 분포</h3>
          <div className="h-[300px] flex">
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={tierData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, value }) => `${value}h`}
                    labelLine={false}
                  >
                    {tierData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="w-48 flex flex-col justify-center space-y-3">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-success"></div>
                <span className="text-sm text-text-muted">Tier 1 (±8%)</span>
                <span className="text-sm font-mono ml-auto">{summary?.total_hours_tier1}h</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-warning"></div>
                <span className="text-sm text-text-muted">Tier 2 (±8~15%)</span>
                <span className="text-sm font-mono ml-auto">{summary?.total_hours_tier2}h</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-danger"></div>
                <span className="text-sm text-text-muted">Tier 3 ({'>'}±15%)</span>
                <span className="text-sm font-mono ml-auto">{summary?.total_hours_tier3}h</span>
              </div>
              <hr className="border-border" />
              <div className="text-xs text-text-muted">
                <p><strong>CP 지급률:</strong></p>
                <p>Tier 1: 100%, Tier 2: 50%, Tier 3: 0%</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Daily Tier Breakdown */}
      <div className="card">
        <h3 className="text-lg font-semibold text-text-primary mb-4">일별 페널티 구간</h3>
        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="displayDate"
                stroke={chartColors.axis}
                fontSize={11}
                tickLine={false}
              />
              <YAxis stroke={chartColors.axis} fontSize={12} tickLine={false} />
              <Tooltip content={<TierTooltip />} />
              <Legend
                wrapperStyle={{ paddingTop: 10 }}
                formatter={(value) => <span className="text-text-muted text-sm">{value}</span>}
              />
              <Bar dataKey="tier1" stackId="a" fill={chartColors.tier1} name="Tier 1 (±8%)" />
              <Bar dataKey="tier2" stackId="a" fill={chartColors.tier2} name="Tier 2 (±8~15%)" />
              <Bar dataKey="tier3" stackId="a" fill={chartColors.tier3} name="Tier 3 (>±15%)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Settlement Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">일별 정산 내역</h3>
          <span className="text-sm text-text-muted">{chartData.length}일</span>
        </div>
        <div className="overflow-x-auto max-h-[400px]">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              <tr className="text-text-muted border-b border-border">
                <th className="text-left py-3 px-4">날짜</th>
                <th className="text-right py-3 px-4">발전량</th>
                <th className="text-right py-3 px-4">DA-SMP</th>
                <th className="text-right py-3 px-4">RT-SMP</th>
                <th className="text-right py-3 px-4">발전수익</th>
                <th className="text-right py-3 px-4">CP</th>
                <th className="text-right py-3 px-4">불균형</th>
                <th className="text-right py-3 px-4">순수익</th>
                <th className="text-center py-3 px-4">Tier 분포</th>
              </tr>
            </thead>
            <tbody>
              {[...chartData].reverse().map((row, idx) => (
                <tr
                  key={idx}
                  className="border-b border-border/50 hover:bg-background/50 transition-colors"
                >
                  <td className="py-3 px-4 font-medium text-text-primary">{row.date}</td>
                  <td className="py-3 px-4 text-right font-mono">{row.generation?.toLocaleString()} MWh</td>
                  <td className="py-3 px-4 text-right font-mono text-blue-400">{row.daSmp?.toFixed(1)}</td>
                  <td className="py-3 px-4 text-right font-mono text-purple-400">{row.rtSmp?.toFixed(1)}</td>
                  <td className="py-3 px-4 text-right font-mono text-success">{row.revenue?.toFixed(1)}백만</td>
                  <td className="py-3 px-4 text-right font-mono text-primary">{row.cp?.toFixed(2)}백만</td>
                  <td className={clsx(
                    'py-3 px-4 text-right font-mono',
                    (row.imbalance || 0) >= 0 ? 'text-success' : 'text-danger'
                  )}>
                    {(row.imbalance || 0) >= 0 ? '+' : ''}{row.imbalance?.toFixed(2)}백만
                  </td>
                  <td className="py-3 px-4 text-right font-mono font-bold text-text-primary">
                    {row.netRevenue?.toFixed(1)}백만
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center gap-1">
                      <span className="px-1.5 py-0.5 text-xs rounded bg-success/20 text-success">{row.tier1}</span>
                      <span className="px-1.5 py-0.5 text-xs rounded bg-warning/20 text-warning">{row.tier2}</span>
                      <span className="px-1.5 py-0.5 text-xs rounded bg-danger/20 text-danger">{row.tier3}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* KPX Rules Info */}
      <div className="card bg-gradient-to-r from-blue-500/10 to-purple-500/10">
        <h3 className="text-lg font-semibold text-text-primary mb-3">KPX 제주 시범사업 정산 규정</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-success mb-2">Tier 1 (±8% 이내)</h4>
            <ul className="text-text-muted space-y-1">
              <li>• 불균형 페널티 없음</li>
              <li>• CP 100% 지급</li>
              <li>• 목표 운영 구간</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-warning mb-2">Tier 2 (±8~15%)</h4>
            <ul className="text-text-muted space-y-1">
              <li>• 경미한 불균형 페널티</li>
              <li>• CP 50% 지급</li>
              <li>• 개선 필요 구간</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-danger mb-2">Tier 3 ({'>'}±15%)</h4>
            <ul className="text-text-muted space-y-1">
              <li>• 가중 불균형 페널티</li>
              <li>• CP 0% 지급</li>
              <li>• 위험 경고 구간</li>
            </ul>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-border">
          <p className="text-xs text-text-muted">
            <strong>참고:</strong> RT-SMP 0원 리스크는 태양광 피크 시간대(11-14시)에 재생에너지 출력제한으로 인해 발생할 수 있습니다.
            DA-RT 가격 차이가 큰 시간대에는 불균형 정산 영향이 증가합니다.
          </p>
        </div>
      </div>
    </div>
  );
}
