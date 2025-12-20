/**
 * Power Supply/Demand Chart Component - RE-BMS v6.0
 */

import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import type { PowerChartData } from '../../types';

interface PowerSupplyChartProps {
  data: PowerChartData[];
  height?: number;
  showForecast?: boolean;
}

// Generate demo data if not provided
function generateDemoData(): PowerChartData[] {
  const now = new Date();
  const data: PowerChartData[] = [];

  for (let i = -12; i <= 12; i++) {
    const time = new Date(now.getTime() + i * 60 * 60 * 1000);
    const hour = time.getHours();
    const isForecast = i > 0;

    // Simulate demand pattern
    const baseDemand = 650 + Math.sin((hour - 6) * Math.PI / 12) * 100;
    const demand = baseDemand + (Math.random() - 0.5) * 30;

    // Solar follows sun pattern
    const solarFactor = hour >= 6 && hour <= 18 ? Math.sin((hour - 6) * Math.PI / 12) : 0;
    const solar = solarFactor * 150 + (Math.random() - 0.5) * 20;

    // Wind is more variable
    const wind = 80 + Math.random() * 60;

    // Supply = demand + reserve
    const supply = demand + 200 + Math.random() * 50;

    data.push({
      time: time.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
      demand: Math.round(demand),
      supply: Math.round(supply),
      reserve: Math.round(supply - demand),
      solar: Math.max(0, Math.round(solar)),
      wind: Math.round(wind),
      forecast: isForecast,
    });
  }

  return data;
}

export default function PowerSupplyChart({
  data,
  height = 350,
  showForecast = true,
}: PowerSupplyChartProps) {
  const chartData = data && data.length > 0 ? data : generateDemoData();

  // Split data for actual vs forecast styling
  const currentIndex = chartData.findIndex((d) => d.forecast);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isForecast = data.forecast;

      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-xl">
          <div className="flex items-center gap-2 mb-2">
            <p className="text-white font-medium">{label}</p>
            {isForecast && (
              <span className="px-1.5 py-0.5 text-[10px] bg-warning/20 text-warning rounded">
                예측
              </span>
            )}
          </div>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-supply">공급능력:</span>
              <span className="text-white font-mono">{data.supply} MW</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-demand">전력수요:</span>
              <span className="text-white font-mono">{data.demand} MW</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">예비전력:</span>
              <span className="text-white font-mono">{data.reserve} MW</span>
            </div>
            <hr className="border-border my-1" />
            <div className="flex justify-between gap-4">
              <span className="text-solar">태양광:</span>
              <span className="text-white font-mono">{data.solar} MW</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-wind">풍력:</span>
              <span className="text-white font-mono">{data.wind} MW</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="supplyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="demandGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            fontSize={12}
            tickLine={false}
            interval={3}
          />

          <YAxis
            stroke="#9ca3af"
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) => `${value}`}
            domain={['dataMin - 50', 'dataMax + 50']}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{ paddingTop: 10 }}
            formatter={(value) => (
              <span className="text-gray-400 text-sm">{value}</span>
            )}
          />

          {/* Supply Area */}
          <Area
            type="monotone"
            dataKey="supply"
            stroke="#22c55e"
            strokeWidth={2}
            fill="url(#supplyGradient)"
            name="공급능력"
            dot={false}
          />

          {/* Demand Area */}
          <Area
            type="monotone"
            dataKey="demand"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#demandGradient)"
            name="전력수요"
            dot={false}
          />

          {/* Solar Line */}
          <Line
            type="monotone"
            dataKey="solar"
            stroke="#fbbf24"
            strokeWidth={2}
            name="태양광"
            dot={false}
            strokeDasharray={showForecast ? undefined : "3 3"}
          />

          {/* Wind Line */}
          <Line
            type="monotone"
            dataKey="wind"
            stroke="#06b6d4"
            strokeWidth={2}
            name="풍력"
            dot={false}
            strokeDasharray={showForecast ? undefined : "3 3"}
          />

          {/* Current time reference line */}
          {currentIndex > 0 && (
            <ReferenceLine
              x={chartData[currentIndex]?.time}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{
                value: '현재',
                position: 'top',
                fill: '#ef4444',
                fontSize: 11,
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
