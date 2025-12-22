/**
 * eXeco Dashboard - Jeju Power Grid Real-time Monitoring
 * Design based on Figma eXeco_main (100% identical)
 */

import React, { useState, useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Line,
  ReferenceLine,
} from 'recharts';

// Inline SVG for Jeju Island map (simplified outline)
const JejuMapSVG = () => (
  <svg viewBox="0 0 400 200" className="w-full h-full" style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}>
    <defs>
      <linearGradient id="jejuGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#e8f5e9" />
        <stop offset="50%" stopColor="#c8e6c9" />
        <stop offset="100%" stopColor="#a5d6a7" />
      </linearGradient>
      <filter id="glow">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>
    {/* Jeju main island outline */}
    <path
      d="M 60 100
         C 70 60, 120 40, 180 35
         C 220 32, 260 38, 300 50
         C 340 62, 360 85, 355 110
         C 350 140, 320 160, 270 165
         C 220 170, 160 168, 120 155
         C 80 142, 55 125, 60 100 Z"
      fill="url(#jejuGradient)"
      stroke="#4caf50"
      strokeWidth="2"
    />
    {/* Hallasan mountain indication */}
    <ellipse cx="200" cy="95" rx="35" ry="25" fill="#81c784" opacity="0.6" />
    <text x="200" y="100" textAnchor="middle" fontSize="10" fill="#2e7d32" fontWeight="bold">한라산</text>
    {/* Cities */}
    <circle cx="190" cy="55" r="3" fill="#1976d2" />
    <text x="190" y="48" textAnchor="middle" fontSize="8" fill="#1976d2">제주시</text>
    <circle cx="220" cy="145" r="3" fill="#1976d2" />
    <text x="220" y="158" textAnchor="middle" fontSize="8" fill="#1976d2">서귀포시</text>
  </svg>
);

// Types
interface PowerPlant {
  id: string;
  name: string;
  type: 'solar' | 'wind' | 'ess';
  capacity: number;
  generation: number;
  x: number;
  y: number;
  size: number;
}

interface ChartData {
  time: string;
  windForecast: number;
  solarForecast: number;
  windActual: number;
  solarActual: number;
  demandForecast: number;
  supplyForecast: number;
  demandActual: number;
  supplyActual: number;
  baseLoad: number;
}

// Generate mock chart data
const generateChartData = (): ChartData[] => {
  const data: ChartData[] = [];
  const baseDate = new Date('2024-12-19T03:00:00');

  for (let i = 0; i <= 21; i++) {
    const time = new Date(baseDate.getTime() + i * 60 * 60 * 1000);
    const hour = time.getHours();

    const solarMultiplier = hour >= 6 && hour <= 18
      ? Math.sin((hour - 6) * Math.PI / 12)
      : 0;

    const windMultiplier = 0.6 + Math.sin(hour * 0.3) * 0.2;
    const demandMultiplier = 0.75 + (hour >= 8 && hour <= 11 ? 0.15 : 0) + (hour >= 17 && hour <= 21 ? 0.25 : 0);

    const baseLoad = 800;
    const windForecast = 300 * windMultiplier;
    const solarForecast = 150 * solarMultiplier;
    const demandForecast = 1300 * demandMultiplier;

    data.push({
      time: `12/19 ${String(hour).padStart(2, '0')}:00`,
      baseLoad,
      windForecast: Math.round(windForecast),
      solarForecast: Math.round(solarForecast),
      windActual: Math.round(windForecast * (0.85 + Math.random() * 0.3)),
      solarActual: Math.round(solarForecast * (0.8 + Math.random() * 0.4)),
      demandForecast: Math.round(demandForecast),
      supplyForecast: Math.round(demandForecast * 1.1),
      demandActual: Math.round(demandForecast * (0.95 + Math.random() * 0.1)),
      supplyActual: Math.round(demandForecast * 1.05),
    });
  }

  return data;
};

// Power plants data - positioned on SVG viewBox (400x200)
// Percentages: x% of 400, y% of 200
const powerPlants: PowerPlant[] = [
  // Solar (red) - #ff4a4a - East side
  { id: 's1', name: '성산태양광', type: 'solar', capacity: 20, generation: 12.5, x: 340, y: 70, size: 18 },
  { id: 's2', name: '서귀포태양광', type: 'solar', capacity: 15, generation: 8.2, x: 250, y: 150, size: 14 },

  // Wind (blue) - #4a89ff - Various locations
  { id: 'w1', name: '한림풍력', type: 'wind', capacity: 30, generation: 22.5, x: 90, y: 85, size: 16 },
  { id: 'w2', name: '김녕풍력', type: 'wind', capacity: 15, generation: 12.1, x: 280, y: 55, size: 12 },
  { id: 'w3', name: '행원풍력', type: 'wind', capacity: 10, generation: 7.5, x: 315, y: 45, size: 10 },
  { id: 'w4', name: '대정풍력', type: 'wind', capacity: 20, generation: 14.8, x: 100, y: 130, size: 14 },
  { id: 'w5', name: '표선풍력', type: 'wind', capacity: 18, generation: 11.2, x: 300, y: 130, size: 12 },

  // ESS (yellow) - #ffbd00
  { id: 'e1', name: '제주ESS1', type: 'ess', capacity: 25, generation: 5.2, x: 175, y: 50, size: 12 },
  { id: 'e2', name: '서귀포ESS', type: 'ess', capacity: 20, generation: 3.8, x: 200, y: 145, size: 10 },
  { id: 'e3', name: '한경ESS', type: 'ess', capacity: 15, generation: 2.1, x: 70, y: 110, size: 10 },
  { id: 'e4', name: '조천ESS', type: 'ess', capacity: 15, generation: 0.6, x: 240, y: 60, size: 10 },
];

// Plant marker component with hover
const PlantMarker: React.FC<{
  plant: PowerPlant;
  onHover: (plant: PowerPlant | null) => void;
  isHovered: boolean;
}> = ({ plant, onHover, isHovered }) => {
  const colors = {
    solar: '#ff4a4a',
    wind: '#4a89ff',
    ess: '#ffbd00',
  };

  return (
    <div
      className="absolute rounded-full cursor-pointer transition-transform hover:scale-110"
      style={{
        left: plant.x,
        top: plant.y,
        width: plant.size,
        height: plant.size,
        backgroundColor: colors[plant.type],
        transform: `translate(-50%, -50%)`,
      }}
      onMouseEnter={() => onHover(plant)}
      onMouseLeave={() => onHover(null)}
    >
      {isHovered && (
        <div
          className="absolute bg-white rounded-lg shadow-lg p-3 z-50 whitespace-nowrap"
          style={{
            left: plant.size + 10,
            top: -10,
            minWidth: 150,
          }}
        >
          <div className="font-bold text-sm text-gray-900">{plant.name}</div>
          <div className="text-xs text-gray-600 mt-1">
            용량: {plant.capacity} MW
          </div>
          <div className="text-xs mt-1">
            발전량: <span className="font-bold text-green-600">{plant.generation.toFixed(1)} MW</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default function ExecoDashboard() {
  const [hoveredPlant, setHoveredPlant] = useState<PowerPlant | null>(null);
  const chartData = useMemo(() => generateChartData(), []);

  // Current time index for reference line
  const currentTimeIndex = 9; // 12:00

  return (
    <div className="min-h-screen bg-white">
      {/* Header - #04265e */}
      <header
        className="h-20 px-6 flex items-center justify-between"
        style={{ backgroundColor: '#04265e', borderBottom: '0.4px solid #d8d8d8' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-3xl font-bold text-white tracking-tight">eXeco</span>
          <span className="text-sm text-white/70">v7.0</span>
        </div>
        <button className="w-8 h-8 flex flex-col justify-center items-center gap-1.5 hover:opacity-80 transition-opacity">
          <span className="w-5 h-0.5 bg-white rounded"></span>
          <span className="w-5 h-0.5 bg-white rounded"></span>
          <span className="w-5 h-0.5 bg-white rounded"></span>
        </button>
      </header>

      {/* Main Content */}
      <main className="p-6 flex flex-col gap-6" style={{ height: 'calc(100vh - 80px)' }}>
        {/* KPI Cards Row */}
        <div className="flex gap-6">
          {/* 현재 수요 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-tight">현재 수요</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-tight">707.4</span>
                <span className="text-xl font-bold text-black leading-9">MW</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-lg text-[#272727]">예비율 94.5%</span>
              </div>
            </div>
          </div>

          {/* 현재 SMP */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-tight">현재 SMP (제주)</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold leading-[50px] tracking-tight" style={{ color: '#0048ff' }}>114.8</span>
                <span className="text-xl font-bold leading-9" style={{ color: '#0048ff' }}>원</span>
              </div>
              <div className="rounded-lg px-3.5 py-1 flex items-center gap-1" style={{ backgroundColor: 'rgba(0,72,255,0.1)' }}>
                <span className="text-lg" style={{ color: '#0048ff' }}>- 6.0원(-5.0%)</span>
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="#0048ff" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              </div>
            </div>
          </div>

          {/* 재생에너지 비율 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-tight">재생에너지 비율</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold leading-[50px] tracking-tight" style={{ color: '#ff1d1d' }}>15.6</span>
                <span className="text-xl font-bold leading-9" style={{ color: '#ff1d1d' }}>%</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: '#ffeaea' }}>
                <span className="text-lg" style={{ color: '#ff1d1d' }}>태양광+풍력</span>
              </div>
            </div>
          </div>

          {/* 계통 주파수 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-tight">계통 주파수</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-tight">60.01</span>
                <span className="text-xl font-bold text-black leading-9">Hz</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: 'rgba(0,197,21,0.1)' }}>
                <span className="text-lg" style={{ color: '#00c515' }}>정상</span>
              </div>
            </div>
          </div>

          {/* 기상 현황 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-tight">기상 현황</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-tight">3</span>
                <span className="text-xl font-bold text-black leading-9">°C</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-lg text-[#272727]">풍속 7.2 m/s</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="flex gap-6 flex-1">
          {/* Left - Chart Section */}
          <div className="flex-[2] flex flex-col gap-2.5">
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex-1 flex flex-col gap-6">
              <div className="flex items-center gap-2.5">
                <span className="text-2xl font-bold text-black tracking-tight">제주 전력수급 현황</span>
                <span className="text-base text-black">실측vs예측(MW)</span>
              </div>

              <div className="bg-white rounded-[14px] p-6 flex-1 flex flex-col gap-2.5">
                {/* Legend */}
                <div className="flex flex-wrap gap-2.5 justify-end text-[10px]">
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-4 bg-gradient-to-b from-blue-300 to-blue-100 rounded"></div>
                    <span>풍력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-4 bg-gradient-to-b from-yellow-300 to-yellow-100 rounded"></div>
                    <span>태양광(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-4 bg-blue-500 rounded"></div>
                    <span>풍력(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-4 bg-yellow-400 rounded"></div>
                    <span>태양광(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 bg-green-500"></div>
                    <span>전력수요(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 border-t-2 border-dashed border-gray-400"></div>
                    <span>공급능력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 bg-green-700"></div>
                    <span>전력수요(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 bg-green-400"></div>
                    <span>공급능력(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 bg-cyan-400"></div>
                    <span>예비전력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-1 bg-cyan-600"></div>
                    <span>예비전력(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-gray-200 rounded"></div>
                    <span>예측 신뢰구간</span>
                  </div>
                </div>

                {/* Chart */}
                <div className="flex-1">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        interval={2}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                        domain={[0, 2500]}
                        ticks={[0, 500, 1000, 1500, 2000, 2500]}
                        label={{ value: '전력(MW)', angle: -90, position: 'insideLeft', fontSize: 12 }}
                      />
                      <Tooltip />

                      {/* Current time reference line (orange) */}
                      <ReferenceLine
                        x={chartData[currentTimeIndex]?.time}
                        stroke="#ff9500"
                        strokeWidth={2}
                      />

                      {/* Base load - green gradient area */}
                      <Area
                        type="monotone"
                        dataKey="supplyActual"
                        stackId="1"
                        stroke="#22c55e"
                        fill="url(#greenGradient)"
                        fillOpacity={0.8}
                      />

                      {/* Wind - blue area */}
                      <Area
                        type="monotone"
                        dataKey="windActual"
                        stackId="2"
                        stroke="#3b82f6"
                        fill="#93c5fd"
                        fillOpacity={0.7}
                      />

                      {/* Solar - yellow area */}
                      <Area
                        type="monotone"
                        dataKey="solarActual"
                        stackId="2"
                        stroke="#eab308"
                        fill="#fde047"
                        fillOpacity={0.7}
                      />

                      {/* Demand lines */}
                      <Line
                        type="monotone"
                        dataKey="demandActual"
                        stroke="#16a34a"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="demandForecast"
                        stroke="#4ade80"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                      />

                      {/* Gradient definitions */}
                      <defs>
                        <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#86efac" stopOpacity={0.9} />
                          <stop offset="100%" stopColor="#dcfce7" stopOpacity={0.5} />
                        </linearGradient>
                      </defs>
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Map and Stats */}
          <div className="flex-1 flex flex-col gap-6">
            {/* Jeju Map */}
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex flex-col gap-6">
              <div className="flex items-center h-[52px]">
                <span className="text-2xl font-bold text-black tracking-tight">발전량 히트맵 표시</span>
              </div>

              <div className="flex items-end justify-between relative">
                {/* Legend */}
                <div className="bg-white rounded-lg p-3.5 shadow-md flex flex-col gap-3.5 z-10">
                  <span className="text-sm font-bold text-black">발전소 유형</span>
                  <div className="flex flex-col gap-2 text-sm">
                    <div className="flex items-center gap-1">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#ff4a4a' }}></div>
                      <span>태양광</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#4a89ff' }}></div>
                      <span>풍력</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#ffbd00' }}></div>
                      <span>ESS</span>
                    </div>
                  </div>
                </div>

                {/* Jeju Map with markers */}
                <div className="relative w-[400px] h-[200px]">
                  <JejuMapSVG />
                  {/* Power Plant Markers */}
                  {powerPlants.map((plant) => (
                    <PlantMarker
                      key={plant.id}
                      plant={plant}
                      onHover={setHoveredPlant}
                      isHovered={hoveredPlant?.id === plant.id}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Plant Statistics */}
            <div className="bg-[#f8f8f8] rounded-lg p-6 flex-1 flex flex-col justify-between">
              <div className="flex flex-col gap-3.5">
                <span className="text-xl font-bold text-black tracking-tight">발전소 현황</span>
                <div className="bg-white rounded-[14px] px-6 py-3.5 flex flex-col gap-2">
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-14">태양광</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold">6개소 | 5MW</span>
                      <span className="text-sm">(발전량 : 0.0MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-14">풍력</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold">14개소|220MW</span>
                      <span className="text-sm">(발전량 : 79.9MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-14">ESS</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold">4개소 | 75MW</span>
                      <span className="text-sm">(충방전 : 11.7MW)</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Weather Info */}
              <div className="flex flex-col gap-1">
                <span className="text-sm text-black w-14">기상 정보</span>
                <div className="flex gap-3.5 text-sm font-medium text-[#232323]">
                  <span>일사량 (0w/m²)</span>
                  <span>|</span>
                  <span>풍향 SE</span>
                  <span>|</span>
                  <span>운량 67%</span>
                  <span>|</span>
                  <span>습도 49%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-xs text-[#232323] py-2">
          <p>제주 전력 지도 v7.0 | Powered by AI | © 2025 Power Demand Forecast Team</p>
          <p className="mt-1">데이터 출처: EPSIS, 기상청 AMOS | 모델: LSTM + Quantile Regression</p>
        </div>
      </main>
    </div>
  );
}
