/**
 * eXeco Dashboard - Jeju Power Grid Real-time Monitoring
 * Design 100% identical to Figma eXeco_main (node-id=3316-358)
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

// Local assets (downloaded from Figma)
const JEJU_MAP = '/jeju-map.png';
const EXECO_LOGO = '/execo-logo.png';
const MENU_ICON = '/menu-icon.png';
const CHECK_ICON = '/check-icon.png';

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

// Power plants data - EXACT positions from Figma (node-id 3316:572 ~ 3316:583)
// Map container: w-[385px] h-[255px]
const powerPlants: PowerPlant[] = [
  // Solar (red #ff4a4a)
  { id: 's1', name: '성산태양광', type: 'solar', capacity: 20, generation: 0, x: 249, y: 41, size: 34 },
  { id: 's2', name: '서귀포태양광', type: 'solar', capacity: 15, generation: 0, x: 300, y: 185, size: 21 },

  // Wind (blue #4a89ff)
  { id: 'w1', name: '한림풍력', type: 'wind', capacity: 30, generation: 22.5, x: 174, y: 74.5, size: 23 },
  { id: 'w2', name: '김녕풍력', type: 'wind', capacity: 15, generation: 12.1, x: 216, y: 63, size: 10 },
  { id: 'w3', name: '행원풍력', type: 'wind', capacity: 10, generation: 7.5, x: 148, y: 105, size: 10 },
  { id: 'w4', name: '대정풍력', type: 'wind', capacity: 20, generation: 14.8, x: 197, y: 195, size: 10 },
  { id: 'w5', name: '표선풍력', type: 'wind', capacity: 18, generation: 11.2, x: 424, y: 142, size: 10 },

  // ESS (yellow #ffbd00)
  { id: 'e1', name: '제주ESS1', type: 'ess', capacity: 25, generation: 5.2, x: 329, y: 38, size: 10 },
  { id: 'e2', name: '서귀포ESS', type: 'ess', capacity: 20, generation: 3.8, x: 306, y: 43, size: 10 },
  { id: 'e3', name: '한경ESS', type: 'ess', capacity: 15, generation: 2.1, x: 232, y: 195, size: 10 },
  { id: 'e4', name: '조천ESS', type: 'ess', capacity: 15, generation: 0.6, x: 369, y: 172, size: 10 },
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
        className="h-[80px] px-6 flex items-center justify-between"
        style={{ backgroundColor: '#04265e', borderBottom: '0.4px solid #d8d8d8' }}
      >
        <div className="h-[34px] w-[109px]">
          <img src={EXECO_LOGO} alt="eXeco" className="h-full w-full object-contain" />
        </div>
        <button className="w-6 h-6">
          <img src={MENU_ICON} alt="Menu" className="w-full h-full" />
        </button>
      </header>

      {/* Main Content */}
      <main className="p-6 flex flex-col gap-6" style={{ height: 'calc(100vh - 80px)' }}>
        {/* KPI Cards Row */}
        <div className="flex gap-6">
          {/* 현재 수요 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-[-0.8px]">현재 수요</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-[-1.28px]">707.4</span>
                <span className="text-xl font-bold text-black leading-9 tracking-[-0.8px]">MW</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-lg text-[#272727] tracking-[-0.72px]">예비율 94.5%</span>
              </div>
            </div>
          </div>

          {/* 현재 SMP */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-[-0.8px]">현재 SMP (제주)</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold leading-[50px] tracking-[-1.28px]" style={{ color: '#0048ff' }}>114.8</span>
                <span className="text-xl font-bold leading-9 tracking-[-0.8px]" style={{ color: '#0048ff' }}>원</span>
              </div>
              <div className="rounded-lg px-3.5 py-1 flex items-center gap-1" style={{ backgroundColor: 'rgba(0,72,255,0.1)' }}>
                <span className="text-lg tracking-[-0.72px]" style={{ color: '#0048ff' }}>- 6.0원(-5.0%)</span>
                <img src={CHECK_ICON} alt="" className="w-6 h-6" />
              </div>
            </div>
          </div>

          {/* 재생에너지 비율 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-[-0.8px]">재생에너지 비율</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold leading-[50px] tracking-[-1.28px]" style={{ color: '#ff1d1d' }}>15.6</span>
                <span className="text-xl font-bold leading-9 tracking-[-0.8px]" style={{ color: '#ff1d1d' }}>%</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: '#ffeaea' }}>
                <span className="text-lg tracking-[-0.72px]" style={{ color: '#ff1d1d' }}>태양광+풍력</span>
              </div>
            </div>
          </div>

          {/* 계통 주파수 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-[-0.8px]">계통 주파수</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-[-1.28px]">60.01</span>
                <span className="text-xl font-bold text-black leading-9 tracking-[-0.8px]">Hz</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: 'rgba(0,197,21,0.1)' }}>
                <span className="text-lg tracking-[-0.72px]" style={{ color: '#00c515' }}>정상</span>
              </div>
            </div>
          </div>

          {/* 기상 현황 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-xl font-medium text-black tracking-[-0.8px]">기상 현황</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[32px] font-bold text-black leading-[50px] tracking-[-1.28px]">3</span>
                <span className="text-xl font-bold text-black leading-9 tracking-[-0.8px]">°C</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-lg text-[#272727] tracking-[-0.72px]">풍속 7.2 m/s</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="flex gap-6 flex-1">
          {/* Left - Chart Section */}
          <div className="w-[1300px] flex flex-col gap-2.5">
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex-1 flex flex-col gap-6">
              <div className="flex items-center gap-2.5">
                <span className="text-2xl font-bold text-black tracking-[-0.96px]">제주 전력수급 현황</span>
                <span className="text-base text-black tracking-[-0.64px]">실측vs예측(MW)</span>
              </div>

              <div className="bg-white rounded-[14px] p-6 flex-1 flex flex-col gap-2.5">
                {/* Legend */}
                <div className="flex flex-wrap gap-2.5 justify-end text-[10px] font-medium tracking-[-0.4px]">
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-[17.5px] bg-gradient-to-b from-blue-300 to-blue-100 rounded"></div>
                    <span>풍력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-[17.5px] bg-gradient-to-b from-yellow-300 to-yellow-100 rounded"></div>
                    <span>태양광(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-[17.5px] bg-blue-500 rounded"></div>
                    <span>풍력(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-[17.5px] bg-yellow-400 rounded"></div>
                    <span>태양광(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-green-500"></div>
                    <span>전력수요(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 border-t-2 border-dashed border-gray-400"></div>
                    <span>공급능력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-green-700"></div>
                    <span>전력수요(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-green-400"></div>
                    <span>공급능력(실측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-cyan-400"></div>
                    <span>예비전력(예측)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-8 h-2 bg-cyan-600"></div>
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
                <span className="text-2xl font-bold text-black tracking-[-0.96px]">발전량 히트맵 표시</span>
              </div>

              <div className="flex items-end justify-between relative">
                {/* Legend */}
                <div className="bg-white rounded-lg p-3.5 shadow-[2px_2px_8px_0px_rgba(0,0,0,0.14)] flex flex-col gap-3.5 z-10">
                  <span className="text-sm font-bold text-black tracking-[-0.56px]">발전소 유형</span>
                  <div className="flex flex-col gap-2 text-sm tracking-[-0.56px]">
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

                {/* Jeju Map with markers - EXACT Figma dimensions */}
                <div className="relative w-[385px] h-[255px]" style={{ mixBlendMode: 'darken' }}>
                  <img
                    src={JEJU_MAP}
                    alt="Jeju Island Map"
                    className="absolute h-[106.51%] left-[-3.12%] max-w-none top-0 w-[109.09%]"
                  />
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
                <span className="text-xl font-bold text-black tracking-[-0.8px]">발전소 현황</span>
                <div className="bg-white rounded-[14px] px-6 py-3.5 flex flex-col gap-2">
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-[54px] tracking-[-0.72px]">태양광</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold tracking-[-0.72px]">6개소 | 5MW</span>
                      <span className="text-sm tracking-[-0.56px]">(발전량 : 0.0MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-[54px] tracking-[-0.72px]">풍력</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold tracking-[-0.72px]">14개소|220MW</span>
                      <span className="text-sm tracking-[-0.56px]">(발전량 : 79.9MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3.5">
                    <span className="text-lg text-black w-[54px] tracking-[-0.72px]">ESS</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-lg font-bold tracking-[-0.72px]">4개소 | 75MW</span>
                      <span className="text-sm tracking-[-0.56px]">(충방전 : 11.7MW)</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Weather Info */}
              <div className="flex flex-col gap-1">
                <span className="text-sm text-black w-[54px] tracking-[-0.56px]">기상 정보</span>
                <div className="flex gap-3.5 text-sm font-medium text-[#232323] tracking-[-0.56px]">
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
        <div className="text-center text-xs text-[#232323] tracking-[-0.48px] leading-[10px]">
          <p>제주 전력 지도 v7.0 | Powered by AI | © 2025 Power Demand Forecast Team</p>
          <p className="mt-2">데이터 출처: EPSIS, 기상청 AMOS | 모델: LSTM + Quantile Regression</p>
        </div>
      </main>
    </div>
  );
}
