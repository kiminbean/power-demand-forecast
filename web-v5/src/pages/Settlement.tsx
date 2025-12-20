/**
 * Mobile Settlement Page
 */

import { useState } from 'react';
import { TrendingUp, TrendingDown, Calendar, ChevronDown, ChevronUp } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { useSettlements, useSettlementSummary } from '../hooks/useApi';
import { useTheme } from '../contexts/ThemeContext';
import type { SettlementRecord } from '../types';
import clsx from 'clsx';

export default function Settlement() {
  const { data: settlements } = useSettlements(7);
  const { data: summary } = useSettlementSummary();
  const [showDetails, setShowDetails] = useState(false);
  const { isDark } = useTheme();

  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1e2530' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
  };

  // Transform settlements for chart
  const chartData = settlements?.map((s: SettlementRecord) => ({
    date: s.date.slice(5), // MM-DD format
    revenue: s.net_revenue_million,
    quantity: s.generation_mwh,
    isProfit: s.net_revenue_million > 0,
  })) ?? [];

  // Calculate totals
  const totalRevenue = settlements?.reduce((sum: number, s: SettlementRecord) => sum + s.net_revenue_million, 0) ?? 0;
  const totalQuantity = settlements?.reduce((sum: number, s: SettlementRecord) => sum + s.generation_mwh, 0) ?? 0;
  const avgAccuracy = settlements && settlements.length > 0
    ? settlements.reduce((sum: number, s: SettlementRecord) => sum + s.accuracy_pct, 0) / settlements.length
    : 0;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { date: string; revenue: number; quantity: number } }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-text-primary font-bold mb-1">{data.date}</p>
          <p className="text-smp font-mono">{data.revenue.toFixed(1)}백만원</p>
          <p className="text-xs text-text-muted">{data.quantity.toFixed(0)} MWh</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-text-primary">정산</h1>
          <p className="text-xs text-text-muted mt-0.5">최근 7일 정산 현황</p>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1 bg-card rounded-lg border border-border">
          <Calendar className="w-4 h-4 text-text-muted" />
          <span className="text-xs text-text-muted">7일</span>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="card bg-brand text-white">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm opacity-80">순 수익 합계</p>
            <p className="text-3xl font-bold mt-1">
              {totalRevenue.toFixed(1)}
            </p>
            <p className="text-sm opacity-80">백만원</p>
          </div>
          <div className="text-right">
            {summary && (
              <div className={clsx(
                'flex items-center gap-1 text-sm',
                summary.net_change_pct >= 0 ? 'text-green-300' : 'text-red-300'
              )}>
                {summary.net_change_pct >= 0 ? (
                  <TrendingUp className="w-4 h-4" />
                ) : (
                  <TrendingDown className="w-4 h-4" />
                )}
                <span>{Math.abs(summary.net_change_pct).toFixed(1)}%</span>
              </div>
            )}
            <p className="text-xs opacity-60 mt-1">전주 대비</p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="card text-center">
          <p className="text-xs text-text-muted">총 발전량</p>
          <p className="text-xl font-bold text-text-primary">{totalQuantity.toFixed(0)}</p>
          <p className="text-xs text-text-muted">MWh</p>
        </div>
        <div className="card text-center">
          <p className="text-xs text-text-muted">예측 정확도</p>
          <p className="text-xl font-bold text-success">{avgAccuracy.toFixed(1)}%</p>
          <p className="text-xs text-text-muted">평균</p>
        </div>
      </div>

      {/* Revenue Chart */}
      <div className="card">
        <h2 className="text-sm font-semibold text-text-primary mb-3">일별 순수익</h2>
        <div className="h-[180px] -mx-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="date"
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
              />
              <YAxis
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
                tickFormatter={(v: number) => `${v}M`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
                {chartData.map((entry: { isProfit: boolean }, index: number) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isProfit ? '#22c55e' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Metrics */}
      {summary && (
        <div className="grid grid-cols-3 gap-3">
          <div className="card text-center">
            <p className="text-xs text-text-muted">발전수익</p>
            <p className="text-lg font-bold text-success">{summary.generation_revenue_million.toFixed(0)}M</p>
          </div>
          <div className="card text-center">
            <p className="text-xs text-text-muted">불균형</p>
            <p className="text-lg font-bold text-danger">{summary.imbalance_charges_million.toFixed(1)}M</p>
          </div>
          <div className="card text-center">
            <p className="text-xs text-text-muted">정확도</p>
            <p className="text-lg font-bold text-primary">{summary.forecast_accuracy_pct.toFixed(1)}%</p>
          </div>
        </div>
      )}

      {/* Transaction Details (Collapsible) */}
      <div className="card">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center justify-between w-full"
        >
          <span className="text-sm font-semibold text-text-primary">거래 내역</span>
          {showDetails ? (
            <ChevronUp className="w-5 h-5 text-text-muted" />
          ) : (
            <ChevronDown className="w-5 h-5 text-text-muted" />
          )}
        </button>

        {showDetails && settlements && (
          <div className="mt-4 space-y-2">
            {settlements.map((s: SettlementRecord, idx: number) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-background rounded-lg"
              >
                <div>
                  <p className="text-sm font-medium text-text-primary">{s.date}</p>
                  <p className="text-xs text-text-muted">{s.generation_mwh.toFixed(0)} MWh</p>
                </div>
                <div className="text-right">
                  <p className={clsx(
                    'text-sm font-bold',
                    s.net_revenue_million >= 0 ? 'text-success' : 'text-danger'
                  )}>
                    {s.net_revenue_million >= 0 ? '+' : ''}{s.net_revenue_million.toFixed(2)}M
                  </p>
                  <p className="text-xs text-text-muted">정확도 {s.accuracy_pct.toFixed(1)}%</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
