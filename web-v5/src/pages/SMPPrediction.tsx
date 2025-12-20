/**
 * Mobile SMP Prediction Page
 */

import { useState } from 'react';
import { TrendingUp, TrendingDown, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useSMPForecast, useModelInfo, useAutoRefresh } from '../hooks/useApi';
import { useTheme } from '../contexts/ThemeContext';
import clsx from 'clsx';

export default function SMPPrediction() {
  const { data: forecast, loading, refetch } = useSMPForecast();
  const { data: modelInfo } = useModelInfo();
  const [showDetails, setShowDetails] = useState(false);
  const { isDark } = useTheme();

  useAutoRefresh(refetch, 300000);

  const currentHour = new Date().getHours();

  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    background: isDark ? '#0e1117' : '#f8fafc',
    tooltipBg: isDark ? '#1e2530' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
  };

  // Transform data for chart
  const chartData = forecast?.q50.map((value, index) => ({
    hour: `${String(index).padStart(2, '0')}시`,
    hourNum: index,
    q10: forecast.q10[index],
    q50: value,
    q90: forecast.q90[index],
    isCurrent: index === currentHour,
  })) ?? [];

  // Statistics
  const stats = forecast ? {
    current: forecast.q50[currentHour],
    min: Math.min(...forecast.q50),
    max: Math.max(...forecast.q50),
    avg: Math.round(forecast.q50.reduce((a, b) => a + b, 0) / forecast.q50.length),
    peakHour: forecast.q50.indexOf(Math.max(...forecast.q50)),
    lowHour: forecast.q50.indexOf(Math.min(...forecast.q50)),
  } : null;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-text-primary font-bold mb-1">{data.hour}</p>
          <p className="text-smp font-mono text-lg">{data.q50.toFixed(1)} 원</p>
          <p className="text-xs text-text-muted">
            {data.q10.toFixed(0)} ~ {data.q90.toFixed(0)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-text-primary">SMP 예측</h1>
          {modelInfo && (
            <p className="text-xs text-text-muted mt-0.5">
              {modelInfo.type} {modelInfo.version} · MAPE {modelInfo.mape}%
            </p>
          )}
        </div>
        <button
          onClick={() => refetch()}
          disabled={loading}
          className="p-2 rounded-lg bg-secondary border border-border active:scale-95 transition-transform"
        >
          <RefreshCw className={clsx('w-5 h-5 text-text-muted', loading && 'animate-spin')} />
        </button>
      </div>

      {/* Current SMP Card */}
      {stats && (
        <div className="card bg-brand text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm opacity-80">현재 SMP ({currentHour}시)</p>
              <p className="text-4xl font-bold mt-1">{stats.current.toFixed(1)}</p>
              <p className="text-sm opacity-80">원/kWh</p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-1 text-sm opacity-80">
                <TrendingUp className="w-4 h-4" />
                <span>최고 {stats.max.toFixed(0)} ({stats.peakHour}시)</span>
              </div>
              <div className="flex items-center gap-1 text-sm opacity-80 mt-1">
                <TrendingDown className="w-4 h-4" />
                <span>최저 {stats.min.toFixed(0)} ({stats.lowHour}시)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="card">
        <h2 className="text-sm font-semibold text-text-primary mb-3">24시간 예측</h2>
        <div className="h-[200px] -mx-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="smpGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="rangeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="hour"
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
                interval={5}
              />
              <YAxis
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
                domain={['auto', 'auto']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="q90"
                stroke="transparent"
                fill="url(#rangeGradient)"
              />
              <Area
                type="monotone"
                dataKey="q10"
                stroke="transparent"
                fill={chartColors.background}
              />
              <Area
                type="monotone"
                dataKey="q50"
                stroke="#fbbf24"
                strokeWidth={2}
                fill="url(#smpGradient)"
              />
              <ReferenceLine
                x={`${String(currentHour).padStart(2, '0')}시`}
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Statistics Grid */}
      {stats && (
        <div className="grid grid-cols-3 gap-3">
          <div className="card text-center">
            <p className="text-xs text-text-muted">평균</p>
            <p className="text-lg font-bold text-text-primary">{stats.avg}</p>
          </div>
          <div className="card text-center">
            <p className="text-xs text-text-muted">신뢰도</p>
            <p className="text-lg font-bold text-primary">
              {((forecast?.confidence ?? 0.87) * 100).toFixed(0)}%
            </p>
          </div>
          <div className="card text-center">
            <p className="text-xs text-text-muted">업데이트</p>
            <p className="text-lg font-bold text-text-primary">
              {forecast?.created_at
                ? new Date(forecast.created_at).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
                : '-'}
            </p>
          </div>
        </div>
      )}

      {/* Hourly Details (Collapsible) */}
      <div className="card">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center justify-between w-full"
        >
          <span className="text-sm font-semibold text-text-primary">시간대별 상세</span>
          {showDetails ? (
            <ChevronUp className="w-5 h-5 text-text-muted" />
          ) : (
            <ChevronDown className="w-5 h-5 text-text-muted" />
          )}
        </button>

        {showDetails && (
          <div className="mt-4 max-h-[300px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="text-text-muted border-b border-border">
                  <th className="text-left py-2">시간</th>
                  <th className="text-right py-2">예측</th>
                  <th className="text-right py-2">범위</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((row) => (
                  <tr
                    key={row.hourNum}
                    className={clsx(
                      'border-b border-border/50',
                      row.isCurrent && 'bg-brand/10'
                    )}
                  >
                    <td className="py-2 font-mono">
                      {row.hour}
                      {row.isCurrent && (
                        <span className="ml-1 text-[10px] text-danger font-sans">현재</span>
                      )}
                    </td>
                    <td className="text-right py-2 font-mono text-smp font-bold">
                      {row.q50.toFixed(1)}
                    </td>
                    <td className="text-right py-2 font-mono text-text-muted text-xs">
                      {row.q10.toFixed(0)}~{row.q90.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
