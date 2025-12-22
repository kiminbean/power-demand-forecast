/**
 * SMP Forecast Chart Component - RE-BMS v6.0
 * Theme-aware chart with light/dark mode support
 */

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import type { SMPForecast } from '../../types';
import { useTheme } from '../../contexts/ThemeContext';

interface SMPChartProps {
  forecast: SMPForecast | null;
  height?: number;
  showConfidenceInterval?: boolean;
}

export default function SMPChart({
  forecast,
  height = 300,
  showConfidenceInterval = true,
}: SMPChartProps) {
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    background: isDark ? '#0e1117' : '#f8fafc',
    text: isDark ? '#ffffff' : '#0f172a',
    textMuted: isDark ? '#9ca3af' : '#64748b',
    textSecondary: isDark ? '#d1d5db' : '#475569',
  };

  if (!forecast) {
    return (
      <div
        className="flex items-center justify-center bg-card rounded-xl"
        style={{ height }}
      >
        <p className="text-text-muted">Loading SMP data...</p>
      </div>
    );
  }

  // Transform data for chart
  const chartData = forecast.q50.map((value, index) => ({
    hour: `${String(index).padStart(2, '0')}:00`,
    hourNum: index,
    q10: forecast.q10[index],
    q50: value,
    q90: forecast.q90[index],
  }));

  // Find current hour
  const currentHour = new Date().getHours();

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-xl">
          <p className="text-text-primary font-medium mb-2">{label}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-text-muted">예측 SMP:</span>
              <span className="text-smp font-mono">{data.q50.toFixed(1)} 원/kWh</span>
            </div>
            {showConfidenceInterval && (
              <>
                <div className="flex justify-between gap-4">
                  <span className="text-text-muted">상한 (90%):</span>
                  <span className="text-text-secondary font-mono">{data.q90.toFixed(1)} 원/kWh</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-text-muted">하한 (10%):</span>
                  <span className="text-text-secondary font-mono">{data.q10.toFixed(1)} 원/kWh</span>
                </div>
              </>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="smpGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />

          <XAxis
            dataKey="hour"
            stroke={colors.axis}
            fontSize={12}
            tickLine={false}
            interval={2}
          />

          <YAxis
            stroke={colors.axis}
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) => `${value}`}
            domain={['auto', 'auto']}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{ paddingTop: 10 }}
            formatter={(value) => (
              <span className="text-text-muted text-sm">{value}</span>
            )}
          />

          {/* Confidence Interval (Q10-Q90 band) */}
          {showConfidenceInterval && (
            <Area
              type="monotone"
              dataKey="q90"
              stroke="transparent"
              fill="url(#confidenceGradient)"
              name="신뢰구간 (10%-90%)"
            />
          )}

          {showConfidenceInterval && (
            <Area
              type="monotone"
              dataKey="q10"
              stroke="transparent"
              fill={colors.background}
              name=""
            />
          )}

          {/* Main SMP Line (Q50) */}
          <Area
            type="monotone"
            dataKey="q50"
            stroke="#fbbf24"
            strokeWidth={2}
            fill="url(#smpGradient)"
            name="SMP 예측 (중앙값)"
            dot={false}
            activeDot={{ r: 6, stroke: '#fbbf24', strokeWidth: 2, fill: colors.background }}
          />

          {/* Current hour reference line */}
          <ReferenceLine
            x={`${String(currentHour).padStart(2, '0')}:00`}
            stroke="#ef4444"
            strokeDasharray="5 5"
            label={{
              value: '현재',
              position: 'top',
              fill: '#ef4444',
              fontSize: 11,
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
