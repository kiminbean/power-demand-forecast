/**
 * eXeco Dashboard - Jeju Power Grid Real-time Monitoring
 * Design 100% identical to Figma eXeco_main (node-id=3316-358)
 * 실시간 데이터: FastAPI 백엔드 (port 8506) 연동
 */

import React, { useState, useMemo, useRef } from 'react';
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
import { useDashboardKPIs, useSMPForecast, useResources, usePowerSupply, useAutoRefresh, useRealtimeStatus, useCurrentSMP, useDayAheadSMP } from '../hooks/useApi';
import { PublishedSMPCard, DayAheadSMPCard } from '../components/Cards/PublishedSMPCards';

// Local assets
const JEJU_MAP = `${import.meta.env.BASE_URL}jeju-map.png`;
const LOGO = `${import.meta.env.BASE_URL}logo-light.png`;  // v6 로고 (다크 배경용)


// Types
interface PowerPlant {
  id: string;
  name: string;
  type: 'solar' | 'wind' | 'ess';
  capacity: number;

  x: number;
  y: number;
  size: number;
}

// v6 스타일 간단한 차트 데이터 (공급능력, 전력수요, 태양광, 풍력, 실측/예측 구분)
interface ChartData {
  time: string;
  supply: number;        // 공급능력
  demand: number;        // 전력수요
  solar: number;         // 태양광
  wind: number;          // 풍력
  is_forecast: boolean;  // 예측값 여부
  // 실측/예측 분리 표시용 (차트에서 영역 분리)
  demandActual?: number;
  demandForecast?: number;
  supplyActual?: number;
  supplyForecast?: number;
}

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

  // 원 크기는 실측 발전량이 아닌 설비용량을 나타냄
  const baseSize = 12; // 최소 크기 (기존 8 * 1.5 = 12)
  const maxSize = 30;  // 최대 크기 (기존 20 * 1.5 = 30)
  const size = Math.max(baseSize, Math.min(maxSize, baseSize + (plant.capacity / 5)));

  return {
    id: plant.id,
    name: plant.name,
    type: plant.type,
    capacity: plant.capacity,

    x,
    y,
    size,
  };
});

// Plant markers support pointer, touch and keyboard selection.
const PlantMarker: React.FC<{
  plant: PowerPlant;
  onSelect: (plant: PowerPlant) => void;
  isSelected: boolean;
}> = ({ plant, onSelect, isSelected }) => {
  const colors = { solar: '#ff4a4a', wind: '#4a89ff', ess: '#ffbd00' };
  return (
    <button
      type="button"
      data-plant-id={plant.id}
      aria-label={`${plant.name} 상세 정보`}
      aria-pressed={isSelected}
      className="plant-marker absolute rounded-full cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-black"
      style={{
        left: `${plant.x / MAP_WIDTH * 100}%`,
        top: `${plant.y / MAP_HEIGHT * 100}%`,
        width: plant.size,
        height: plant.size,
        backgroundColor: colors[plant.type],
        transform: 'translate(-50%, -50%)',
      }}
      onClick={() => onSelect(plant)}
    />
  );
};

export default function ExecoDashboard() {
  const [hoveredPlant, setHoveredPlant] = useState<PowerPlant | null>(null);
  const [overlappingPlants, setOverlappingPlants] = useState<PowerPlant[]>([]);
  const mapRef = useRef<HTMLDivElement>(null);

  const selectMarker = (selected: PowerPlant) => {
    const markers = Array.from(mapRef.current?.querySelectorAll<HTMLButtonElement>('.plant-marker') ?? []);
    const selectedRect = markers.find(marker => marker.dataset.plantId === selected.id)?.getBoundingClientRect();
    const overlappingIds = new Set(markers.filter(marker => {
      if (!selectedRect) return false;
      const rect = marker.getBoundingClientRect();
      return rect.left < selectedRect.right && rect.right > selectedRect.left &&
        rect.top < selectedRect.bottom && rect.bottom > selectedRect.top;
    }).map(marker => marker.dataset.plantId));
    const nearby = powerPlants.filter(plant => overlappingIds.has(plant.id));
    setOverlappingPlants(nearby.length > 1 ? nearby : []);
    setHoveredPlant(current => current?.id === selected.id ? null : selected);
  };

  // API 훅 - 실시간 데이터 로드
  const { data: kpis, refetch: refetchKPIs, error: kpisError } = useDashboardKPIs();
  const { refetch: refetchSMP } = useSMPForecast();
  const { data: publishedSMP, refetch: refetchPublishedSMP } = useCurrentSMP();
  const { data: dayAheadSMP, refetch: refetchDayAheadSMP } = useDayAheadSMP();
  const { data: resources, refetch: refetchResources, error: resourcesError } = useResources();
  const { data: powerSupply, refetch: refetchPowerSupply, error: powerSupplyError } = usePowerSupply();
  const { data: realtimeStatus } = useRealtimeStatus();

  // 60초마다 자동 새로고침
  useAutoRefresh(() => {
    refetchKPIs();
    refetchSMP();
    refetchPublishedSMP();
    refetchDayAheadSMP();
    refetchResources();
    refetchPowerSupply();
  }, 60000);

  // Current time index for reference line (API에서 현재 시간 사용)
  const currentHour = powerSupply?.current_hour ?? new Date().getHours();
  const currentTimeIndex = currentHour;

  // 차트 데이터 변환 (API 데이터 → 차트 형식 + 실측/예측 분리)
  // 현재 시간 데이터는 양쪽에 포함시켜 자연스러운 연결 유지
  const chartData = useMemo<ChartData[]>(() => {
    if (!powerSupply?.data) {
      // API 데이터 없으면 기본값
      return [];
    }
    const currentHr = powerSupply.current_hour;
    return powerSupply.data.map((d) => {
      // 현재 시간은 양쪽에 포함 (연결점)
      const isCurrentHour = d.hour === currentHr;
      const isActual = !d.is_forecast || isCurrentHour;
      const isForecast = d.is_forecast || isCurrentHour;

      return {
        time: d.time,
        supply: d.supply,
        demand: d.demand,
        solar: d.solar,
        wind: d.wind,
        is_forecast: d.is_forecast,
        // 실측/예측 분리 (현재 시간은 양쪽 모두 포함하여 연결)
        demandActual: isActual ? d.demand : undefined,
        demandForecast: isForecast ? d.demand : undefined,
        supplyActual: isActual ? d.supply : undefined,
        supplyForecast: isForecast ? d.supply : undefined,
      };
    });
  }, [powerSupply]);

  // KPI 데이터 (API 또는 기본값) - 실제 EPSIS/KMA 데이터 기반
  const currentDemand = kpis?.current_demand_mw ?? 685.0;  // 현재 전력 수요 (MW)
  const supplyReserveRate = kpis?.supply_reserve_rate ?? 25.0;  // 공급 예비율 (%)

  const gridFrequency = kpis?.grid_frequency ?? 60.01;  // 계통 주파수 (Hz)
  const renewableRatio = kpis?.renewable_ratio_pct ?? 24.6;  // 재생에너지 비율 (%)
  const weather = kpis?.weather ?? { temperature: 5.5, wind_speed: 3.2, humidity: 58.0, condition: '맑음' };

  // 발전소 통계 (resources에서)
  const plantStats = useMemo(() => {
    if (!resources) return { solar: { count: 5, capacity: 4.4, output: 142 }, wind: { count: 12, capacity: 290, output: 185 }, ess: { count: 4, capacity: 75, output: 22.5 } };
    const solarPlants = resources.filter(r => r.type === 'solar');
    const windPlants = resources.filter(r => r.type === 'wind');
    const essPlants = resources.filter(r => r.type === 'ess');
    return {
      solar: { count: solarPlants.length, capacity: Math.round(solarPlants.reduce((s, r) => s + r.capacity, 0) * 10) / 10, output: Math.round(solarPlants.reduce((s, r) => s + r.current_output, 0) * 10) / 10 },
      wind: { count: windPlants.length, capacity: Math.round(windPlants.reduce((s, r) => s + r.capacity, 0)), output: Math.round(windPlants.reduce((s, r) => s + r.current_output, 0) * 10) / 10 },
      ess: { count: essPlants.length, capacity: Math.round(essPlants.reduce((s, r) => s + r.capacity, 0)), output: Math.round(essPlants.reduce((s, r) => s + r.current_output, 0) * 10) / 10 },
    };
  }, [resources]);

  // 데이터 소스 표시 (API 연결 상태)
  const fallbackSections = [kpisError && '현재 지표', resourcesError && '발전소 현황', powerSupplyError && '전력수급 차트'].filter(Boolean);
  const loadingSections = [!kpis && !kpisError && '현재 지표', !resources && !resourcesError && '발전소 현황'].filter(Boolean);
  const dataSource = kpisError ? '데모 데이터' : kpis ? 'KPI API 데이터' : '데이터 확인 중';

  // API 연결 상태 아이콘 색상
  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'connected': return '#00c515';  // 녹색
      case 'error': return '#ff1d1d';       // 빨간색
      default: return '#fbbf24';            // 노란색 (unknown)
    }
  };

  // 모니터링 피드 연결 상태 라벨
  const getOverallStatusLabel = (status: string | undefined) => {
    switch (status) {
      case 'all_connected': return '전체 연결';
      case 'partial': return '일부 연결';
      case 'all_error': return '연결 오류';
      default: return '확인 중';
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header - #04265e */}
      <header
        className="dashboard-header min-h-[80px] px-6 flex items-center justify-between"
        style={{ backgroundColor: '#04265e', borderBottom: '0.4px solid #d8d8d8' }}
      >
        <div className="h-[40px]">
          <img src={LOGO} alt="RE-BMS" className="h-full object-contain" />
        </div>
        <div className="flex items-center gap-4">
          {/* API 연결 상태 표시 */}
          <div className="flex items-center gap-3 bg-white/10 rounded-lg px-3 py-1.5">
            <div className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: getStatusColor(realtimeStatus?.smp_api?.status) }}
                title={`SMP API: ${realtimeStatus?.smp_api?.status || 'unknown'}`}
              />
              <span className="text-xs text-white/80">SMP</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: getStatusColor(realtimeStatus?.power_supply_api?.status) }}
                title={`전력수급 API: ${realtimeStatus?.power_supply_api?.status || 'unknown'}`}
              />
              <span className="text-xs text-white/80">전력</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: getStatusColor(realtimeStatus?.weather_api?.status) }}
                title={`기상청 API: ${realtimeStatus?.weather_api?.status || 'unknown'}`}
              />
              <span className="text-xs text-white/80">기상</span>
            </div>
            <div className="h-4 w-px bg-white/30" />
            <span
              className="text-xs font-medium"
              style={{ color: getStatusColor(realtimeStatus?.overall_status === 'all_connected' ? 'connected' : realtimeStatus?.overall_status === 'partial' ? 'unknown' : 'error') }}
            >
              모니터링 피드: {getOverallStatusLabel(realtimeStatus?.overall_status)}
            </span>
          </div>
          <span className="text-xs text-white/70">{dataSource}</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-main p-6 flex flex-col gap-6">
        {fallbackSections.length > 0 && (
          <div role="status" className="demo-notice">
            API 연결 실패로 데모 데이터를 표시 중입니다: {fallbackSections.join(', ')}
          </div>
        )}
        {loadingSections.length > 0 && (
          <div role="status" className="demo-notice">
            로딩 중 임시 값을 표시합니다: {loadingSections.join(', ')}
          </div>
        )}
        {kpis && !kpisError && (
          <div className="text-sm text-gray-700">
            KPI 출처: {kpis.data_source || '출처 정보 없음'} · 출처가 명시되지 않은 항목은 기본값일 수 있음 · 주파수는 시뮬레이션 · 재생에너지 비율은 예측 기반
          </div>
        )}
        {/* KPI Cards Row - 실시간 데이터 */}
        <div className="dashboard-kpis flex gap-6">
          {/* 현재 수요 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">현재 수요</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">{currentDemand.toFixed(1)}</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">MW</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-xl text-[#272727] tracking-[-0.72px]">예비율 {supplyReserveRate.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* 공표 SMP */}
          <PublishedSMPCard data={publishedSMP} variant="kpi" />

          {/* 계통 주파수 - 실시간 데이터 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">계통 주파수</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">{gridFrequency.toFixed(2)}</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">Hz</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: gridFrequency >= 59.8 && gridFrequency <= 60.2 ? 'rgba(0,197,21,0.1)' : 'rgba(255,29,29,0.1)' }}>
                <span className="text-xl tracking-[-0.72px]" style={{ color: gridFrequency >= 59.8 && gridFrequency <= 60.2 ? '#00c515' : '#ff1d1d' }}>
                  {gridFrequency >= 59.8 && gridFrequency <= 60.2 ? '정상' : '주의'}
                </span>
              </div>
            </div>
          </div>

          {/* 재생에너지 비율 - 실시간 데이터 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">재생에너지 비율</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold leading-[60px] tracking-[-1.28px]" style={{ color: '#ff1d1d' }}>{renewableRatio.toFixed(1)}</span>
                <span className="text-2xl font-bold leading-9 tracking-[-0.8px]" style={{ color: '#ff1d1d' }}>%</span>
              </div>
              <div className="rounded-lg px-3.5 py-1" style={{ backgroundColor: '#ffeaea' }}>
                <span className="text-xl tracking-[-0.72px]" style={{ color: '#ff1d1d' }}>태양광+풍력</span>
              </div>
            </div>
          </div>

          {/* 기상 현황 - 실시간 KMA 데이터 */}
          <div className="flex-1 bg-[#f8f8f8] rounded-lg px-6 py-8 flex flex-col items-center justify-center gap-3.5">
            <span className="text-2xl font-medium text-black tracking-[-0.8px]">기상 현황</span>
            <div className="flex items-center gap-3.5 px-3.5">
              <div className="flex items-end gap-1">
                <span className="text-[38px] font-bold text-black leading-[60px] tracking-[-1.28px]">{weather.temperature.toFixed(0)}</span>
                <span className="text-2xl font-bold text-black leading-9 tracking-[-0.8px]">°C</span>
              </div>
              <div className="bg-white rounded-lg px-3.5 py-1">
                <span className="text-xl text-[#272727] tracking-[-0.72px]">풍속 {weather.wind_speed.toFixed(1)} m/s</span>
              </div>
            </div>
          </div>
        </div>

        <DayAheadSMPCard data={dayAheadSMP} variant="kpi" />

        {/* Main Content Grid - 7:3 ratio */}
        <div className="dashboard-grid flex gap-6 flex-1">
          {/* Left - Chart Section (70%) */}
          <div className="dashboard-chart flex-[7] flex flex-col gap-2.5">
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex-1 flex flex-col gap-6">
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="text-3xl font-bold text-black tracking-[-0.96px]">제주 전력수급 현황</span>
                <span className="text-lg text-black tracking-[-0.64px]">시간별 패턴 및 예측(MW)</span>
              </div>
              <span className="text-sm text-gray-700">전력수급 출처: {powerSupplyError ? '데모 데이터' : powerSupply?.data_source || '데이터 확인 중'} · 과거 시간도 패턴 기반이며 태양광·풍력은 예측값</span>

              {/* v6 스타일 간단한 차트 - 실측/예측 구분 */}
              <div className="bg-white rounded-[14px] p-6" style={{ height: '640px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 25, right: 30, left: 0, bottom: 10 }}>
                    <defs>
                      {/* 실측 데이터 그라데이션 (진한 색) */}
                      <linearGradient id="supplyActualGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1} />
                      </linearGradient>
                      <linearGradient id="demandActualGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                      </linearGradient>
                      {/* 예측 데이터 그라데이션 (연한 색) */}
                      <linearGradient id="supplyForecastGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#86efac" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#86efac" stopOpacity={0.05} />
                      </linearGradient>
                      <linearGradient id="demandForecastGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#93c5fd" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#93c5fd" stopOpacity={0.05} />
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
                      formatter={(value, name) => {
                        if (value === undefined) return null;
                        const labels: Record<string, string> = {
                          supplyActual: '공급능력 (패턴)',
                          supplyForecast: '공급능력 (예측)',
                          demandActual: '전력수요 (패턴)',
                          demandForecast: '전력수요 (예측)',
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
                          supplyActual: '공급능력(패턴)',
                          supplyForecast: '공급능력(예측)',
                          demandActual: '전력수요(패턴)',
                          demandForecast: '전력수요(예측)',
                          solar: '태양광',
                          wind: '풍력',
                        };
                        return labels[value] || value;
                      }}
                    />

                    {/* 현재 시간 표시선 - 실측/예측 경계 */}
                    <ReferenceLine
                      x={chartData[currentTimeIndex]?.time}
                      stroke="#ef4444"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      label={{
                        value: '현재',
                        position: 'insideTopRight',
                        fill: '#ef4444',
                        fontSize: 14,
                        fontWeight: 'bold',
                        offset: 10
                      }}
                    />

                    {/* 공급능력 - 실측 (진한 녹색, 실선) */}
                    <Area
                      type="monotone"
                      dataKey="supplyActual"
                      stroke="#22c55e"
                      strokeWidth={2}
                      fill="url(#supplyActualGradient)"
                      name="supplyActual"
                      dot={false}
                      connectNulls={false}
                    />

                    {/* 공급능력 - 예측 (연한 녹색, 점선) */}
                    <Area
                      type="monotone"
                      dataKey="supplyForecast"
                      stroke="#86efac"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      fill="url(#supplyForecastGradient)"
                      name="supplyForecast"
                      dot={false}
                      connectNulls={false}
                    />

                    {/* 전력수요 - 실측 (진한 파란색, 실선) */}
                    <Area
                      type="monotone"
                      dataKey="demandActual"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      fill="url(#demandActualGradient)"
                      name="demandActual"
                      dot={false}
                      connectNulls={false}
                    />

                    {/* 전력수요 - 예측 (연한 파란색, 점선) */}
                    <Area
                      type="monotone"
                      dataKey="demandForecast"
                      stroke="#93c5fd"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      fill="url(#demandForecastGradient)"
                      name="demandForecast"
                      dot={false}
                      connectNulls={false}
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
          <div className="dashboard-side flex-[3] flex flex-col gap-6">
            {/* Jeju Map */}
            <div className="bg-[#f8f8f8] rounded-[14px] p-6 flex flex-col gap-6">
              <div className="flex items-center">
                <span className="text-3xl font-bold text-black tracking-[-0.96px]">발전소 위치 및 설비용량</span>
              </div>

              {/* Map Container - Centered */}
              <div className="flex flex-col items-center gap-4">
                <span className="text-sm text-gray-600">원 크기: 설비용량 기준 · 실시간 발전량 정보 없음</span>
                {/* Jeju Map with markers - Centered, exact fit */}
                <div ref={mapRef} className="plant-map relative" style={{ maxWidth: MAP_WIDTH, aspectRatio: `${MAP_WIDTH} / ${MAP_HEIGHT}` }}>
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
                      onSelect={selectMarker}
                      isSelected={hoveredPlant?.id === plant.id}
                    />
                  ))}
                </div>

                {overlappingPlants.length > 1 && (
                  <div role="group" aria-label="겹친 발전소 선택" className="flex flex-wrap justify-center gap-2">
                    <span className="w-full text-center text-sm text-gray-600">겹친 발전소 선택</span>
                    {overlappingPlants.map(plant => (
                      <button key={plant.id} type="button" aria-label={`${plant.name} 선택`}
                        aria-pressed={hoveredPlant?.id === plant.id}
                        className="min-h-[44px] rounded-md border border-gray-500 bg-white px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-black"
                        onClick={() => setHoveredPlant(plant)}>{plant.name}</button>
                    ))}
                  </div>
                )}
                <label className="plant-picker">
                  발전소 선택
                  <select value={hoveredPlant?.id ?? ''} onChange={event => { setOverlappingPlants([]); setHoveredPlant(powerPlants.find(plant => plant.id === event.target.value) ?? null); }}>
                    <option value="">발전소를 선택하세요</option>
                    {powerPlants.map(plant => <option key={plant.id} value={plant.id}>{plant.name}</option>)}
                  </select>
                </label>
                <div className="plant-details" role="status" aria-live="polite">
                  {hoveredPlant ? <><strong>{hoveredPlant.name}</strong> · 용량: {hoveredPlant.capacity} MW · 실시간 발전량 정보 없음</> : '지도에서 발전소를 선택하세요.'}
                </div>

                {/* Legend - Below map (1.2배 크기) */}
                <div className="bg-white rounded-lg p-4 shadow-[2px_2px_8px_0px_rgba(0,0,0,0.14)] flex flex-wrap gap-4">
                  <span className="text-lg font-bold text-black tracking-[-0.64px]">발전소 유형</span>
                  <div className="flex gap-5 text-lg tracking-[-0.64px]">
                    <div className="flex items-center gap-1.5">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#ff4a4a' }}></div>
                      <span>태양광</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#4a89ff' }}></div>
                      <span>풍력</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#ffbd00' }}></div>
                      <span>ESS</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Plant Statistics - 발전량 히트맵과 동일 크기 - 실시간 데이터 */}
            <div className="bg-[#f8f8f8] rounded-lg p-6 flex-1 flex flex-col justify-between">
              <div className="flex flex-col gap-4">
                <span className="text-3xl font-bold text-black tracking-[-0.96px]">발전소 현황</span>
                <span className="text-sm text-gray-700">발전소 출력 및 충방전은 시뮬레이션 값</span>
                <div className="bg-white rounded-[14px] px-5 py-3 flex flex-col gap-2">
                  <div className="flex items-center gap-3">
                    <span className="text-xl text-black w-[70px] tracking-[-0.64px]">태양광</span>
                    <div className="flex flex-wrap items-center gap-2 text-[#232323]">
                      <span className="text-xl font-bold tracking-[-0.64px]">{plantStats.solar.count}개소 | {plantStats.solar.capacity}MW</span>
                      <span className="text-lg tracking-[-0.56px]">(발전량 : {plantStats.solar.output}MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xl text-black w-[70px] tracking-[-0.64px]">풍력</span>
                    <div className="flex flex-wrap items-center gap-2 text-[#232323]">
                      <span className="text-xl font-bold tracking-[-0.64px]">{plantStats.wind.count}개소 | {plantStats.wind.capacity}MW</span>
                      <span className="text-lg tracking-[-0.56px]">(발전량 : {plantStats.wind.output}MW)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xl text-black w-[70px] tracking-[-0.64px]">ESS</span>
                    <div className="flex flex-wrap items-center gap-2 text-[#232323]">
                      <span className="text-xl font-bold tracking-[-0.64px]">{plantStats.ess.count}개소 | {plantStats.ess.capacity}MW</span>
                      <span className="text-lg tracking-[-0.56px]">(충방전 : {plantStats.ess.output}MW)</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Weather Info - 실시간 KMA 데이터 */}
              <div className="flex flex-col gap-1">
                <span className="text-base text-black tracking-[-0.56px]">기상 정보</span>
                <div className="flex flex-wrap gap-3.5 text-base font-medium text-[#232323] tracking-[-0.56px]">
                  <span>기온 {weather.temperature.toFixed(1)}°C</span>
                  <span>|</span>
                  <span>풍속 {weather.wind_speed.toFixed(1)}m/s</span>
                  <span>|</span>
                  <span>{weather.condition}</span>
                  <span>|</span>
                  <span>습도 {weather.humidity.toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-[#232323] tracking-[-0.48px] leading-4">
          <p>제주 전력 지도 v7.0 | Powered by AI | © 2025 Power Demand Forecast Team</p>
          <p className="mt-2">데이터 출처와 예측·시뮬레이션 여부는 각 항목의 안내를 확인하세요.</p>
        </div>
      </main>
    </div>
  );
}
