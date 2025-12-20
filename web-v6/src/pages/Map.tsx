/**
 * Jeju Map Page - RE-BMS v6.0
 * Interactive map showing power plant locations
 */

import { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import { DivIcon } from 'leaflet';
import { Sun, Wind, Zap, MapPin } from 'lucide-react';
import { useResources } from '../hooks/useApi';
import clsx from 'clsx';
import 'leaflet/dist/leaflet.css';

// Custom marker icons
const windIcon = new DivIcon({
  className: 'custom-marker',
  html: `<div style="background: #06b6d4; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
      <path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/>
      <path d="M9.6 4.6A2 2 0 1 1 11 8H2"/>
      <path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>
    </svg>
  </div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

const solarIcon = new DivIcon({
  className: 'custom-marker',
  html: `<div style="background: #fbbf24; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
    </svg>
  </div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
});

// Default Jeju coordinates
const JEJU_CENTER: [number, number] = [33.4, 126.55];
const JEJU_ZOOM = 10;

// Demo plant data with coordinates
const demoPlants = [
  { id: '1', name: '가시리풍력', type: 'wind', capacity: 15.0, output: 12.3, lat: 33.3823, lng: 126.7632, location: '서귀포시' },
  { id: '2', name: '김녕풍력', type: 'wind', capacity: 12.0, output: 9.8, lat: 33.5575, lng: 126.7631, location: '제주시' },
  { id: '3', name: '한경풍력', type: 'wind', capacity: 21.0, output: 18.2, lat: 33.3343, lng: 126.1727, location: '제주시' },
  { id: '4', name: '삼달풍력', type: 'wind', capacity: 6.1, output: 5.1, lat: 33.3489, lng: 126.8347, location: '서귀포시' },
  { id: '5', name: '월정태양광', type: 'solar', capacity: 5.0, output: 3.8, lat: 33.5556, lng: 126.7889, location: '제주시' },
  { id: '6', name: '성산태양광', type: 'solar', capacity: 8.0, output: 6.2, lat: 33.4586, lng: 126.9312, location: '서귀포시' },
  { id: '7', name: '대정풍력', type: 'wind', capacity: 18.0, output: 15.1, lat: 33.2234, lng: 126.2512, location: '서귀포시' },
  { id: '8', name: '한림풍력', type: 'wind', capacity: 10.5, output: 8.7, lat: 33.4123, lng: 126.2645, location: '제주시' },
  { id: '9', name: '표선태양광', type: 'solar', capacity: 4.5, output: 3.2, lat: 33.3256, lng: 126.8234, location: '서귀포시' },
  { id: '10', name: '조천풍력', type: 'wind', capacity: 8.0, output: 6.5, lat: 33.5234, lng: 126.6312, location: '제주시' },
];

export default function Map() {
  useResources(); // Load resources data
  const [selectedPlant, setSelectedPlant] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'solar' | 'wind'>('all');

  // Use demo data with coordinates
  const plants = demoPlants.filter(p => filter === 'all' || p.type === filter);

  // Calculate stats
  const stats = {
    total: plants.reduce((sum, p) => sum + p.output, 0),
    wind: plants.filter(p => p.type === 'wind').reduce((sum, p) => sum + p.output, 0),
    solar: plants.filter(p => p.type === 'solar').reduce((sum, p) => sum + p.output, 0),
    count: plants.length,
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">제주 전력 지도</h1>
          <p className="text-gray-400 mt-1">재생에너지 발전소 실시간 현황</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Stats */}
          <div className="flex items-center gap-4 px-4 py-2 bg-card rounded-lg border border-border">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-success" />
              <span className="text-sm text-gray-400">총 발전:</span>
              <span className="text-white font-bold">{stats.total.toFixed(1)} MW</span>
            </div>
            <div className="w-px h-4 bg-border" />
            <div className="flex items-center gap-2">
              <Wind className="w-4 h-4 text-wind" />
              <span className="text-white font-mono">{stats.wind.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Sun className="w-4 h-4 text-solar" />
              <span className="text-white font-mono">{stats.solar.toFixed(1)}</span>
            </div>
          </div>

          {/* Filter */}
          <div className="flex items-center bg-card rounded-lg border border-border p-1">
            {(['all', 'wind', 'solar'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded-md transition-colors',
                  filter === f
                    ? 'bg-primary text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                {f === 'all' && '전체'}
                {f === 'wind' && '풍력'}
                {f === 'solar' && '태양광'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Map and Sidebar */}
      <div className="flex-1 flex gap-4">
        {/* Map Container */}
        <div className="flex-1 rounded-xl overflow-hidden border border-border">
          <MapContainer
            center={JEJU_CENTER}
            zoom={JEJU_ZOOM}
            style={{ height: '100%', width: '100%' }}
            className="bg-background"
          >
            <TileLayer
              attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />

            {/* Plant Markers */}
            {plants.map((plant) => (
              <Marker
                key={plant.id}
                position={[plant.lat, plant.lng]}
                icon={plant.type === 'wind' ? windIcon : solarIcon}
                eventHandlers={{
                  click: () => setSelectedPlant(plant.id),
                }}
              >
                <Popup className="custom-popup">
                  <div className="p-2">
                    <h3 className="font-bold text-lg">{plant.name}</h3>
                    <div className="mt-2 space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">유형:</span>
                        <span>{plant.type === 'wind' ? '풍력' : '태양광'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">용량:</span>
                        <span>{plant.capacity} MW</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">현재 발전:</span>
                        <span className="font-bold text-green-600">{plant.output} MW</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">가동률:</span>
                        <span>{((plant.output / plant.capacity) * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            ))}

            {/* Output circles */}
            {plants.map((plant) => (
              <Circle
                key={`circle-${plant.id}`}
                center={[plant.lat, plant.lng]}
                radius={plant.output * 150}
                pathOptions={{
                  color: plant.type === 'wind' ? '#06b6d4' : '#fbbf24',
                  fillColor: plant.type === 'wind' ? '#06b6d4' : '#fbbf24',
                  fillOpacity: 0.2,
                  weight: 1,
                }}
              />
            ))}
          </MapContainer>
        </div>

        {/* Sidebar - Plant List */}
        <div className="w-80 card overflow-hidden flex flex-col">
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold text-white">발전소 목록</h3>
            <p className="text-sm text-gray-400">{plants.length}개 발전소</p>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {plants.map((plant) => (
              <div
                key={plant.id}
                onClick={() => setSelectedPlant(plant.id)}
                className={clsx(
                  'p-3 rounded-lg cursor-pointer transition-colors mb-2',
                  selectedPlant === plant.id
                    ? 'bg-primary/20 border border-primary/50'
                    : 'bg-background hover:bg-background/80'
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={clsx(
                    'w-10 h-10 rounded-lg flex items-center justify-center',
                    plant.type === 'wind' ? 'bg-wind/10' : 'bg-solar/10'
                  )}>
                    {plant.type === 'wind' ? (
                      <Wind className="w-5 h-5 text-wind" />
                    ) : (
                      <Sun className="w-5 h-5 text-solar" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="text-white font-medium text-sm">{plant.name}</div>
                    <div className="flex items-center gap-1 text-xs text-gray-400">
                      <MapPin className="w-3 h-3" />
                      {plant.location}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-success font-bold text-sm">{plant.output} MW</div>
                    <div className="text-xs text-gray-400">/ {plant.capacity} MW</div>
                  </div>
                </div>
                {/* Progress bar */}
                <div className="mt-2 h-1.5 bg-card rounded-full overflow-hidden">
                  <div
                    className={clsx(
                      'h-full rounded-full',
                      plant.type === 'wind' ? 'bg-wind' : 'bg-solar'
                    )}
                    style={{ width: `${(plant.output / plant.capacity) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
