/**
 * Settlement Page - RE-BMS v6.0
 * Revenue and Imbalance Settlement Management
 */

import { useState } from 'react';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Calendar,
  Download,
  AlertTriangle,
  CheckCircle,
  ArrowUpRight,
  ArrowDownRight,
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
} from 'recharts';
import { useSettlements, useSettlementSummary } from '../hooks/useApi';
import clsx from 'clsx';

// Generate demo settlement data
function generateDemoData() {
  const data = [];
  const baseDate = new Date();

  for (let i = 30; i >= 0; i--) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() - i);

    const generation = 150 + Math.random() * 50;
    const smp = 80 + Math.random() * 40;
    const revenue = generation * smp * 24 / 1000000; // in millions
    const forecastAccuracy = 88 + Math.random() * 10;
    const imbalance = (100 - forecastAccuracy) / 100 * revenue * (Math.random() > 0.5 ? 1 : -1);

    data.push({
      date: date.toISOString().split('T')[0],
      displayDate: `${date.getMonth() + 1}/${date.getDate()}`,
      generation: Math.round(generation * 24),
      revenue: Math.round(revenue * 10) / 10,
      imbalance: Math.round(imbalance * 100) / 100,
      netRevenue: Math.round((revenue + imbalance) * 10) / 10,
      accuracy: Math.round(forecastAccuracy * 10) / 10,
      avgSmp: Math.round(smp),
    });
  }

  return data;
}

export default function Settlement() {
  const { data: settlements } = useSettlements(30);
  const { data: summary } = useSettlementSummary();
  const [period, setPeriod] = useState<'week' | 'month' | 'quarter'>('month');

  // Use demo data if no API data
  const settlementData = generateDemoData();

  // Calculate period totals
  const periodData = period === 'week' ? settlementData.slice(-7) :
                     period === 'month' ? settlementData.slice(-30) :
                     settlementData;

  const totals = {
    generation: periodData.reduce((sum, d) => sum + d.generation, 0),
    revenue: periodData.reduce((sum, d) => sum + d.revenue, 0),
    imbalance: periodData.reduce((sum, d) => sum + d.imbalance, 0),
    netRevenue: periodData.reduce((sum, d) => sum + d.netRevenue, 0),
    avgAccuracy: periodData.reduce((sum, d) => sum + d.accuracy, 0) / periodData.length,
  };

  // Custom tooltip
  const ChartTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-xl min-w-[180px]">
          <p className="text-white font-medium mb-2">{data.date}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-success">수익:</span>
              <span className="font-mono">{data.revenue.toFixed(1)}백만</span>
            </div>
            <div className="flex justify-between">
              <span className={data.imbalance >= 0 ? 'text-success' : 'text-danger'}>
                불균형:
              </span>
              <span className="font-mono">
                {data.imbalance >= 0 ? '+' : ''}{data.imbalance.toFixed(2)}백만
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-white">순수익:</span>
              <span className="font-mono font-bold">{data.netRevenue.toFixed(1)}백만</span>
            </div>
            <hr className="border-border my-1" />
            <div className="flex justify-between">
              <span className="text-gray-400">발전량:</span>
              <span className="font-mono">{data.generation.toLocaleString()} MWh</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">예측정확도:</span>
              <span className="font-mono">{data.accuracy.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">정산</h1>
          <p className="text-gray-400 mt-1">수익 및 불균형 정산 현황</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Period Selector */}
          <div className="flex items-center bg-card rounded-lg border border-border p-1">
            {(['week', 'month', 'quarter'] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded-md transition-colors',
                  period === p
                    ? 'bg-primary text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                {p === 'week' && '1주'}
                {p === 'month' && '1개월'}
                {p === 'quarter' && '분기'}
              </button>
            ))}
          </div>
          <button className="btn-secondary flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            기간 선택
          </button>
          <button className="btn-primary flex items-center gap-2">
            <Download className="w-4 h-4" />
            리포트 다운로드
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-5 gap-4">
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">발전 수익</span>
            <div className="flex items-center text-success text-xs">
              <ArrowUpRight className="w-3 h-3" />
              5.2%
            </div>
          </div>
          <div className="text-2xl font-bold text-success">
            {totals.revenue.toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">백만원</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">불균형 정산</span>
            <div className={clsx(
              'flex items-center text-xs',
              totals.imbalance >= 0 ? 'text-success' : 'text-danger'
            )}>
              {totals.imbalance >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
              {Math.abs(totals.imbalance / totals.revenue * 100).toFixed(1)}%
            </div>
          </div>
          <div className={clsx(
            'text-2xl font-bold',
            totals.imbalance >= 0 ? 'text-success' : 'text-danger'
          )}>
            {totals.imbalance >= 0 ? '+' : ''}{totals.imbalance.toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">백만원</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">순 수익</span>
            <div className="flex items-center text-primary text-xs">
              <ArrowUpRight className="w-3 h-3" />
              6.8%
            </div>
          </div>
          <div className="text-2xl font-bold text-primary">
            {totals.netRevenue.toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">백만원</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">총 발전량</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {(totals.generation / 1000).toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">GWh</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">예측 정확도</span>
            {totals.avgAccuracy >= 90 ? (
              <CheckCircle className="w-4 h-4 text-success" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-warning" />
            )}
          </div>
          <div className={clsx(
            'text-2xl font-bold',
            totals.avgAccuracy >= 90 ? 'text-success' : 'text-warning'
          )}>
            {totals.avgAccuracy.toFixed(1)}
            <span className="text-sm text-gray-400 ml-1">%</span>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Revenue Trend */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">수익 추이</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={periodData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="displayDate"
                  stroke="#9ca3af"
                  fontSize={11}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Legend
                  wrapperStyle={{ paddingTop: 10 }}
                  formatter={(value) => <span className="text-gray-400 text-sm">{value}</span>}
                />
                <Area
                  type="monotone"
                  dataKey="revenue"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="url(#revenueGradient)"
                  name="발전 수익"
                />
                <Line
                  type="monotone"
                  dataKey="netRevenue"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="순 수익"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Imbalance Analysis */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">불균형 정산 분석</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={periodData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="displayDate"
                  stroke="#9ca3af"
                  fontSize={11}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Legend
                  wrapperStyle={{ paddingTop: 10 }}
                  formatter={(value) => <span className="text-gray-400 text-sm">{value}</span>}
                />
                <Bar
                  dataKey="imbalance"
                  radius={[4, 4, 0, 0]}
                  name="불균형"
                >
                  {periodData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.imbalance >= 0 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Settlement Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">일별 정산 내역</h3>
          <span className="text-sm text-gray-400">{periodData.length}일</span>
        </div>
        <div className="overflow-x-auto max-h-[400px]">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              <tr className="text-gray-400 border-b border-border">
                <th className="text-left py-3 px-4">날짜</th>
                <th className="text-right py-3 px-4">발전량 (MWh)</th>
                <th className="text-right py-3 px-4">평균 SMP</th>
                <th className="text-right py-3 px-4">발전 수익</th>
                <th className="text-right py-3 px-4">불균형</th>
                <th className="text-right py-3 px-4">순 수익</th>
                <th className="text-right py-3 px-4">예측 정확도</th>
              </tr>
            </thead>
            <tbody>
              {[...periodData].reverse().map((row, idx) => (
                <tr
                  key={idx}
                  className="border-b border-border/50 hover:bg-background/50 transition-colors"
                >
                  <td className="py-3 px-4 font-medium text-white">{row.date}</td>
                  <td className="py-3 px-4 text-right font-mono">{row.generation.toLocaleString()}</td>
                  <td className="py-3 px-4 text-right font-mono text-smp">{row.avgSmp} 원</td>
                  <td className="py-3 px-4 text-right font-mono text-success">{row.revenue.toFixed(1)} 백만</td>
                  <td className={clsx(
                    'py-3 px-4 text-right font-mono',
                    row.imbalance >= 0 ? 'text-success' : 'text-danger'
                  )}>
                    {row.imbalance >= 0 ? '+' : ''}{row.imbalance.toFixed(2)} 백만
                  </td>
                  <td className="py-3 px-4 text-right font-mono font-bold text-white">{row.netRevenue.toFixed(1)} 백만</td>
                  <td className="py-3 px-4 text-right">
                    <span className={clsx(
                      'px-2 py-1 text-xs rounded font-mono',
                      row.accuracy >= 95 ? 'bg-success/20 text-success' :
                      row.accuracy >= 90 ? 'bg-primary/20 text-primary' :
                      'bg-warning/20 text-warning'
                    )}>
                      {row.accuracy.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
