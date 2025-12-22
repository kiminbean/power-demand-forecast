/**
 * eXeco Dashboard - Jeju Power Grid Real-time Monitoring
 * Design 100% identical to Figma eXeco_main (node-id=3316-358)
 */

import React, { useState, useMemo } from 'react';
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Line,
  Legend,
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

// v6 스타일 간단한 차트 데이터 (공급능력, 전력수요, 태양광, 풍력)
interface ChartData {
  time: string;
  supply: number;   // 공급능력
  demand: number;   // 전력수요
  solar: number;    // 태양광
  wind: number;     // 풍력
}

// 실제 제주도 전력 데이터 기반 (2024년 12월 패턴)
const generateChartData = (): ChartData[] => {
  // 2024-12-22 기준 실측 데이터 패턴 (EPSIS 제주 시간별 전력거래량)
  const demandPattern = [
    531, 511, 491, 504, 510, 536, 549, 573, 636, 662, 681, 747, 736, 705, 718, 624, 542, 549, 587, 579, 562, 544
  ];

  // 풍력 발전량 패턴 (겨울철 제주)
  const windPattern = [
    180, 175, 168, 172, 178, 185, 165, 142, 128, 115, 108, 95, 88, 92, 105, 118, 135, 158, 172, 185, 192, 188
  ];

  // 태양광 발전량 패턴 (겨울철)
  const solarPattern = [
    0, 0, 0, 0, 0, 0, 5, 28, 65, 98, 125, 142, 148, 138, 112, 72, 25, 0, 0, 0, 0, 0
  ];

  return demandPattern.map((demand, i) => ({
    time: `${String(i + 1).padStart(2, '0')}:00`,
    demand,
    supply: Math.round(demand * 1.15), // 공급예비율 15%
    solar: solarPattern[i],
    wind: windPattern[i],
  }));
};

// Jeju island coordinate bounds - calibrated to match the map image
// Reference points:
// - 제주시 (북쪽 중앙): lat 33.50, lng 126.53
// - 서귀포시 (남쪽 중앙): lat 33.25, lng 126.56
// - 성산 (동쪽 끝): lat 33.46, lng 126.94
// - 한경 (서쪽 끝): lat 33.34, lng 126.17
const JEJU_BOUNDS = {
  // 지도 이미지의 실제 표시 범위에 맞춤
  minLat: 33.12,  // 지도 하단
  maxLat: 33.60,  // 지도 상단
  minLng: 126.08, // 지도 좌측
  maxLng: 127.02, // 지도 우측
};

// Map container dimensions (enlarged)
const MAP_WIDTH = 480;
const MAP_HEIGHT = 320;

// Convert lat/lng to pixel position
const latLngToPixel = (lat: number, lng: number): { x: number; y: number } => {
  const x = ((lng - JEJU_BOUNDS.minLng) / (JEJU_BOUNDS.maxLng - JEJU_BOUNDS.minLng)) * MAP_WIDTH;
  const y = ((JEJU_BOUNDS.maxLat - lat) / (JEJU_BOUNDS.maxLat - JEJU_BOUNDS.minLat)) * MAP_HEIGHT;
  return { x, y };
};

// Power plants data from data/jeju_plants/jeju_power_plants.json
// Using actual latitude/longitude coordinates
const powerPlantsData = [
  // Wind farms (주요 풍력단지)
  { id: 'wind_001', name: '한경풍력', type: 'wind' as const, capacity: 21.0, lat: 33.339417, lng: 126.169222 },
  { id: 'wind_002', name: '상명풍력', type: 'wind' as const, capacity: 21.0, lat: 33.339250, lng: 126.289556 },
  { id: 'wind_003', name: '가시리풍력', type: 'wind' as const, capacity: 30.0, lat: 33.3576, lng: 126.7461 },
  { id: 'wind_004', name: '김녕풍력', type: 'wind' as const, capacity: 30.0, lat: 33.5560, lng: 126.7480 },
  { id: 'wind_005', name: '행원풍력', type: 'wind' as const, capacity: 8.4, lat: 33.5490, lng: 126.7990 },
  { id: 'wind_006', name: '삼달풍력', type: 'wind' as const, capacity: 33.0, lat: 33.3790, lng: 126.8630 },
  { id: 'wind_007', name: '성산1풍력', type: 'wind' as const, capacity: 12.0, lat: 33.4399, lng: 126.9229 },
  { id: 'wind_008', name: '성산2풍력', type: 'wind' as const, capacity: 8.0, lat: 33.4350, lng: 126.9180 },
  { id: 'wind_011', name: '탐라해상풍력', type: 'wind' as const, capacity: 30.0, lat: 33.2942, lng: 126.1631 },
  { id: 'wind_014', name: '제주음풍력', type: 'wind' as const, capacity: 21.0, lat: 33.4800, lng: 126.5500 },
  { id: 'wind_015', name: '동복-북촌풍력', type: 'wind' as const, capacity: 30.0, lat: 33.5300, lng: 126.7200 },
  { id: 'wind_016', name: '대정풍력', type: 'wind' as const, capacity: 45.0, lat: 33.2200, lng: 126.2800 },

  // Solar farms (주요 태양광)
  { id: 'solar_001', name: '구좌태양광', type: 'solar' as const, capacity: 0.75, lat: 33.5200, lng: 126.8500 },
  { id: 'solar_002', name: '삼도태양광', type: 'solar' as const, capacity: 1.7, lat: 33.5100, lng: 126.5200 },
  { id: 'solar_003', name: '용담태양광', type: 'solar' as const, capacity: 0.42, lat: 33.5050, lng: 126.5100 },
  { id: 'solar_005', name: '한경태양광', type: 'solar' as const, capacity: 0.5, lat: 33.3400, lng: 126.1800 },
  { id: 'solar_006', name: '표선태양광', type: 'solar' as const, capacity: 1.0, lat: 33.3200, lng: 126.8300 },

  // ESS facilities
  { id: 'ess_001', name: '제주 변동성완화 ESS', type: 'ess' as const, capacity: 30, lat: 33.5100, lng: 126.5400 },
  { id: 'ess_002', name: '제주 피크저감 ESS', type: 'ess' as const, capacity: 20, lat: 33.4500, lng: 126.5600 },
  { id: 'ess_003', name: '한경풍력 연계 ESS', type: 'ess' as const, capacity: 10, lat: 33.3400, lng: 126.1700 },
  { id: 'ess_004', name: '가시리풍력 연계 ESS', type: 'ess' as const, capacity: 15, lat: 33.3576, lng: 126.7500 },
];

// Convert to PowerPlant format with calculated pixel positions
const powerPlants: PowerPlant[] = powerPlantsData.map(plant => {
  const { x, y } = latLngToPixel(plant.lat, plant.lng);

  // Generation is simulated (겨울철 오후 기준)
  // 풍력: 60-70% 가동, 태양광: 정오 기준 최대, ESS: 충방전 30%
  const generationRate = plant.type === 'ess' ? 0.3 : (plant.type === 'solar' ? 0.85 : 0.65);
  const generation = Math.round(plant.capacity * generationRate * 10) / 10;

  // Size based on generation (1.5x scale: min 12, max 30)
  // 발전량 기준으로 크기 차별화
  const baseSize = 12; // 최소 크기 (기존 8 * 1.5 = 12)
  const maxSize = 30;  // 최대 크기 (기존 20 * 1.5 = 30)
  const size = Math.max(baseSize, Math.min(maxSize, baseSize + (generation / 5)));

  return {
    id: plant.id,
    name: plant.name,
    type: plant.type,
    capacity: plant.capacity,
    generation,
    x,
    y,
    size,
  };
});

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
          className="absolute bg-white rounded-xl shadow-lg p-4 z-50 whitespace-nowrap"
          style={{
            left: plant.size + 12,
            top: -15,
            minWidth: 200,
          }}
        >
          <div className="font-bold text-xl text-gray-900">{plant.name}</div>
          <div className="text-base text-gray-600 mt-2">
            용량: {plant.capacity} MW
          </div>
          <div className="text-base mt-1">
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
        {/* KPI Cards Row - Font sizes 1.2x */}
        <div className="flex gap-6">
          {/* 현재 수요 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">현재 수요</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">707.4</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">MW</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-xl text-[#272727] tracking-[-0.72px]">예비율 94.5%</span>
              </div>
            </div>
          </div>

          {/* 현재 SMP */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">현재 SMP (제주)</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold leading-[60px] tracking-[-1.28px]" style={{ color: '#0048ff' }}>114.8</span>
                <span className="text-2xl font-bold leading-9 tracking-[-0.8px]" style={{ color: '#0048ff' }}>원</span>
              </div>
              <div className="rounded-lg px-3.5 py-1 flex items-center gap-1" style={{ backgroundColor: 'rgba(0,72,255,0.1)' }}>
                <span className="text-xl tracking-[-0.72px]" style={{ color: '#0048ff' }}>- 6.0원(-5.0%)</span>
                <img src={CHECK_ICON} alt="" className="w-7 h-7" />
              </div>
            </div>
          </div>

          {/* 재생에너지 비율 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">재생에너지 비율</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold leading-[60px] tracking-[-1.28px]" style={{ color: '#ff1d1d' }}>15.6</span>
                <span className="text-2xl font-bold leading-9 tracking-[-0.8px]" style={{ color: '#ff1d1d' }}>%</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: '#ffeaea' }}>
                <span className="text-xl tracking-[-0.72px]" style={{ color: '#ff1d1d' }}>태양광+풍력</span>
              </div>
            </div>
          </div>

          {/* 계통 주파수 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">계통 주파수</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">60.01</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">Hz</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: 'rgba(0,197,21,0.1)' }}>
                <span className="text-xl tracking-[-0.72px]" style={{ color: '#00c515' }}>정상</span>
              </div>
            </div>
          </div>

          {/* 기상 현황 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">기상 현황</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">3</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">°C</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-xl text-[#272727] tracking-[-0.72px]">풍속 7.2 m/s</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid - 7:3 ratio */}
        <div className="flex gap-6 flex-1">
          {/* Left - Chart Section (70%) */}
          <div className="flex-[7] flex flex-col gap-2.5">
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex-1 flex flex-col gap-6">
              <div className="flex items-center gap-2.5">
                <span className="text-3xl font-bold text-black tracking-[-0.96px]">제주 전력수급 현황</span>
                <span className="text-lg text-black tracking-[-0.64px]">실측vs예측(MW)</span>
              </div>

              {/* v6 스타일 간단한 차트 */}
              <div className="bg-white rounded-[14px] p-6 flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
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

                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

                    <XAxis
                      dataKey="time"
                      stroke="#6b7280"
                      fontSize={14}
                      tickLine={false}
                      interval={2}
                    />

                    <YAxis
                      stroke="#6b7280"
                      fontSize={14}
                      tickLine={false}
                      domain={[0, 900]}
                      tickFormatter={(value) => `${value}`}
                    />

                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        fontSize: '14px'
                      }}
                      formatter={(value: number, name: string) => {
                        const labels: Record<string, string> = {
                          supply: '공급능력',
                          demand: '전력수요',
                          solar: '태양광',
                          wind: '풍력',
                        };
                        return [`${value} MW`, labels[name] || name];
                      }}
                      labelFormatter={(label) => `${label}`}
                    />

                    <Legend
                      wrapperStyle={{ paddingTop: 10, fontSize: '14px' }}
                      formatter={(value) => {
                        const labels: Record<string, string> = {
                          supply: '공급능력',
                          demand: '전력수요',
                          solar: '태양광',
                          wind: '풍력',
                        };
                        return labels[value] || value;
                      }}
                    />

                    {/* 현재 시간 표시선 */}
                    <ReferenceLine
                      x={chartData[currentTimeIndex]?.time}
                      stroke="#ef4444"
                      strokeDasharray="5 5"
                      label={{ value: '현재', position: 'top', fill: '#ef4444', fontSize: 12 }}
                    />

                    {/* 공급능력 - 녹색 영역 */}
                    <Area
                      type="monotone"
                      dataKey="supply"
                      stroke="#22c55e"
                      strokeWidth={2}
                      fill="url(#supplyGradient)"
                      name="supply"
                      dot={false}
                    />

                    {/* 전력수요 - 파란색 영역 */}
                    <Area
                      type="monotone"
                      dataKey="demand"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      fill="url(#demandGradient)"
                      name="demand"
                      dot={false}
                    />

                    {/* 태양광 - 노란색 라인 */}
                    <Line
                      type="monotone"
                      dataKey="solar"
                      stroke="#fbbf24"
                      strokeWidth={2}
                      name="solar"
                      dot={false}
                    />

                    {/* 풍력 - 청록색 라인 */}
                    <Line
                      type="monotone"
                      dataKey="wind"
                      stroke="#06b6d4"
                      strokeWidth={2}
                      name="wind"
                      dot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Right Panel - Map and Stats (30%) */}
          <div className="flex-[3] flex flex-col gap-6">
            {/* Jeju Map */}
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex flex-col gap-6">
              <div className="flex items-center">
                <span className="text-3xl font-bold text-black tracking-[-0.96px]">발전량 히트맵 표시</span>
              </div>

              {/* Map Container - Centered */}
              <div className="flex flex-col items-center gap-4">
                {/* Jeju Map with markers - Centered, exact fit */}
                <div className="relative" style={{ width: MAP_WIDTH, height: MAP_HEIGHT }}>
                  <img
                    src={JEJU_MAP}
                    alt="Jeju Island Map"
                    className="w-full h-full object-cover"
                    style={{ mixBlendMode: 'multiply' }}
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

                {/* Legend - Below map */}
                <div className="bg-white rounded-lg p-3.5 shadow-[2px_2px_8px_0px_rgba(0,0,0,0.14)] flex gap-6">
                  <span className="text-base font-bold text-black tracking-[-0.56px]">발전소 유형</span>
                  <div className="flex gap-4 text-base tracking-[-0.56px]">
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ff4a4a' }}></div>
                      <span>태양광</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#4a89ff' }}></div>
                      <span>풍력</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ffbd00' }}></div>
                      <span>ESS</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Plant Statistics - Font sizes 1.5x */}
            <div className="bg-[#f8f8f8] rounded-lg p-6 flex-1 flex flex-col justify-between">
              <div className="flex flex-col gap-4">
                <span className="text-4xl font-bold text-black tracking-[-0.8px]">발전소 현황</span>
                <div className="bg-white rounded-[14px] px-6 py-4 flex flex-col gap-3">
                  <div className="flex items-center gap-4">
                    <span className="text-3xl text-black w-[100px] tracking-[-0.72px]">태양광</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-3xl font-bold tracking-[-0.72px]">5개소 | 4.4MW</span>
                      <span className="text-2xl tracking-[-0.56px]">(발전량 : 142MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-3xl text-black w-[100px] tracking-[-0.72px]">풍력</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-3xl font-bold tracking-[-0.72px]">12개소 | 290MW</span>
                      <span className="text-2xl tracking-[-0.56px]">(발전량 : 185MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-3xl text-black w-[100px] tracking-[-0.72px]">ESS</span>
                    <div className="flex items-center gap-2 text-[#232323]">
                      <span className="text-3xl font-bold tracking-[-0.72px]">4개소 | 75MW</span>
                      <span className="text-2xl tracking-[-0.56px]">(충방전 : 22.5MW)</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Weather Info */}
              <div className="flex flex-col gap-1">
                <span className="text-base text-black tracking-[-0.56px]">기상 정보</span>
                <div className="flex gap-3.5 text-base font-medium text-[#232323] tracking-[-0.56px]">
                  <span>일사량 (320W/m²)</span>
                  <span>|</span>
                  <span>풍향 NW</span>
                  <span>|</span>
                  <span>운량 42%</span>
                  <span>|</span>
                  <span>습도 58%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-[#232323] tracking-[-0.48px] leading-4">
          <p>제주 전력 지도 v7.0 | Powered by AI | © 2025 Power Demand Forecast Team</p>
          <p className="mt-2">데이터 출처: EPSIS, 기상청 AMOS | 모델: LSTM + Quantile Regression</p>
        </div>
      </main>
    </div>
  );
}
