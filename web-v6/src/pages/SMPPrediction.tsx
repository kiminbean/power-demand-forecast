/**
 * SMP Prediction Page - RE-BMS v6.0
 * AI-powered SMP forecasting with detailed analysis
 */

import { useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  Info,
  Download,
  RefreshCw,
  Brain,
} from 'lucide-react';
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
  BarChart,
  Bar,
} from 'recharts';
import { useSMPForecast, useModelInfo, useAutoRefresh } from '../hooks/useApi';
import clsx from 'clsx';

export default function SMPPrediction() {
  const { data: forecast, loading, refetch } = useSMPForecast();
  const { data: modelInfo } = useModelInfo();
  const [selectedHour, setSelectedHour] = useState<number | null>(null);

  useAutoRefresh(refetch, 300000); // Refresh every 5 minutes

  const currentHour = new Date().getHours();

  // Transform forecast data for charts
  const chartData = forecast?.q50.map((value, index) => ({
    hour: `${String(index).padStart(2, '0')}:00`,
    hourNum: index,
    q10: forecast.q10[index],
    q50: value,
    q90: forecast.q90[index],
    range: forecast.q90[index] - forecast.q10[index],
    isPast: index < currentHour,
    isCurrent: index === currentHour,
  })) ?? [];

  // Calculate statistics
  const stats = forecast ? {
    min: Math.min(...forecast.q50),
    max: Math.max(...forecast.q50),
    avg: forecast.q50.reduce((a, b) => a + b, 0) / forecast.q50.length,
    current: forecast.q50[currentHour],
    peakHour: forecast.q50.indexOf(Math.max(...forecast.q50)),
    lowHour: forecast.q50.indexOf(Math.min(...forecast.q50)),
  } : null;

  // Custom tooltip for main chart
  const MainTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-4 shadow-xl min-w-[200px]">
          <p className="text-white font-bold text-lg mb-2">{label}</p>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-smp">예측 SMP:</span>
              <span className="text-white font-mono font-bold">{data.q50.toFixed(1)} 원/kWh</span>
            </div>
            <hr className="border-border" />
            <div className="flex justify-between">
              <span className="text-gray-400">상한 (90%):</span>
              <span className="text-gray-300 font-mono">{data.q90.toFixed(1)} 원/kWh</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">하한 (10%):</span>
              <span className="text-gray-300 font-mono">{data.q10.toFixed(1)} 원/kWh</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">변동폭:</span>
              <span className="text-gray-300 font-mono">{data.range.toFixed(1)} 원</span>
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
          <h1 className="text-2xl font-bold text-white">SMP 예측</h1>
          <p className="text-gray-400 mt-1">AI 기반 24시간 SMP 가격 예측</p>
        </div>
        <div className="flex items-center gap-3">
          {modelInfo && (
            <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
              <Brain className="w-4 h-4 text-primary" />
              <span className="text-sm text-gray-400">
                {modelInfo.type} {modelInfo.version}
              </span>
              <span className="text-xs text-success">MAPE: {modelInfo.mape}%</span>
            </div>
          )}
          <button
            onClick={() => refetch()}
            className="btn-secondary flex items-center gap-2"
            disabled={loading}
          >
            <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
            새로고침
          </button>
          <button className="btn-primary flex items-center gap-2">
            <Download className="w-4 h-4" />
            CSV 다운로드
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-6 gap-4">
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">현재 SMP</p>
            <div className="text-2xl font-bold text-smp">{stats.current.toFixed(1)}</div>
            <p className="text-xs text-gray-500">원/kWh</p>
          </div>
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">평균 예측</p>
            <div className="text-2xl font-bold text-white">{stats.avg.toFixed(1)}</div>
            <p className="text-xs text-gray-500">원/kWh</p>
          </div>
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">최고가</p>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4 text-danger" />
              <span className="text-2xl font-bold text-danger">{stats.max.toFixed(1)}</span>
            </div>
            <p className="text-xs text-gray-500">{String(stats.peakHour).padStart(2, '0')}:00</p>
          </div>
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">최저가</p>
            <div className="flex items-center gap-1">
              <TrendingDown className="w-4 h-4 text-success" />
              <span className="text-2xl font-bold text-success">{stats.min.toFixed(1)}</span>
            </div>
            <p className="text-xs text-gray-500">{String(stats.lowHour).padStart(2, '0')}:00</p>
          </div>
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">신뢰도</p>
            <div className="text-2xl font-bold text-primary">
              {((forecast?.confidence ?? 0.87) * 100).toFixed(0)}%
            </div>
            <p className="text-xs text-gray-500">AI 모델</p>
          </div>
          <div className="card">
            <p className="text-sm text-gray-400 mb-1">업데이트</p>
            <div className="text-sm font-medium text-white">
              {forecast?.created_at
                ? new Date(forecast.created_at).toLocaleTimeString('ko-KR')
                : '-'}
            </div>
            <p className="text-xs text-gray-500">최근 예측</p>
          </div>
        </div>
      )}

      {/* Main Chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">24시간 SMP 예측 차트</h2>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 px-2 py-1 text-xs bg-smp/20 text-smp rounded">
              <Activity className="w-3 h-3" />
              실시간
            </span>
          </div>
        </div>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="smpMainGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="confidenceAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

              <XAxis
                dataKey="hour"
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
              />

              <YAxis
                stroke="#9ca3af"
                fontSize={12}
                tickLine={false}
                domain={['auto', 'auto']}
                tickFormatter={(value) => `${value}`}
              />

              <Tooltip content={<MainTooltip />} />

              <Legend
                wrapperStyle={{ paddingTop: 20 }}
                formatter={(value) => <span className="text-gray-400 text-sm">{value}</span>}
              />

              {/* Confidence interval */}
              <Area
                type="monotone"
                dataKey="q90"
                stroke="transparent"
                fill="url(#confidenceAreaGradient)"
                name="신뢰구간 상한 (90%)"
              />
              <Area
                type="monotone"
                dataKey="q10"
                stroke="transparent"
                fill="#0e1117"
                name="신뢰구간 하한 (10%)"
              />

              {/* Main prediction line */}
              <Area
                type="monotone"
                dataKey="q50"
                stroke="#fbbf24"
                strokeWidth={3}
                fill="url(#smpMainGradient)"
                name="SMP 예측 (중앙값)"
                dot={false}
                activeDot={{ r: 8, stroke: '#fbbf24', strokeWidth: 2, fill: '#0e1117' }}
              />

              {/* Current hour marker */}
              <ReferenceLine
                x={`${String(currentHour).padStart(2, '0')}:00`}
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{
                  value: '현재',
                  position: 'top',
                  fill: '#ef4444',
                  fontSize: 12,
                  fontWeight: 'bold',
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Hourly Detail & Range Analysis */}
      <div className="grid grid-cols-2 gap-6">
        {/* Hourly Table */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">시간대별 상세</h3>
          <div className="max-h-[300px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="text-gray-400 border-b border-border">
                  <th className="text-left py-2 px-2">시간</th>
                  <th className="text-right py-2 px-2">예측</th>
                  <th className="text-right py-2 px-2">하한</th>
                  <th className="text-right py-2 px-2">상한</th>
                  <th className="text-right py-2 px-2">변동폭</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((row) => (
                  <tr
                    key={row.hourNum}
                    className={clsx(
                      'border-b border-border/50 hover:bg-background/50 cursor-pointer transition-colors',
                      row.isCurrent && 'bg-primary/10',
                      row.isPast && 'text-gray-500'
                    )}
                    onClick={() => setSelectedHour(row.hourNum)}
                  >
                    <td className="py-2 px-2 font-mono">
                      {row.hour}
                      {row.isCurrent && (
                        <span className="ml-2 text-[10px] text-danger">현재</span>
                      )}
                    </td>
                    <td className="text-right py-2 px-2 font-mono text-smp font-bold">
                      {row.q50.toFixed(1)}
                    </td>
                    <td className="text-right py-2 px-2 font-mono text-gray-400">
                      {row.q10.toFixed(1)}
                    </td>
                    <td className="text-right py-2 px-2 font-mono text-gray-400">
                      {row.q90.toFixed(1)}
                    </td>
                    <td className="text-right py-2 px-2 font-mono text-gray-400">
                      {row.range.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Uncertainty Range Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">불확실성 분석</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="hour"
                  stroke="#9ca3af"
                  fontSize={10}
                  tickLine={false}
                  interval={2}
                />
                <YAxis
                  stroke="#9ca3af"
                  fontSize={12}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e2530',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar
                  dataKey="range"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                  name="예측 변동폭 (원)"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 p-3 bg-background rounded-lg">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-primary mt-0.5" />
              <div className="text-sm text-gray-400">
                변동폭이 클수록 예측 불확실성이 높습니다.
                입찰 시 보수적인 가격 전략을 고려하세요.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
