/**
 * Power Plant Type Definitions
 * For small-scale solar/wind/ESS plant registration
 */

export type PlantType = 'solar' | 'wind' | 'ess';
export type ContractType = 'net_metering' | 'ppa';
export type RoofDirection = 'south' | 'east' | 'west' | 'flat';
export type PlantStatus = 'active' | 'maintenance' | 'paused';

export interface PowerPlant {
  id: string;
  name: string;                    // e.g., "우리집 태양광 1호"
  type: PlantType;                 // 태양광, 풍력, ESS
  capacity: number;                // kW (e.g., 3)
  installDate: string;             // ISO date (e.g., "2024-01-15")
  contractType: ContractType;
  location: {
    address: string;
    lat?: number;
    lng?: number;
  };
  roofDirection?: RoofDirection;   // 남향, 동향, 서향, 평지
  status?: PlantStatus;            // 운영 상태 (기본값: active)
  createdAt: string;
  updatedAt: string;
}

export interface PowerPlantWithEstimates extends PowerPlant {
  efficiency: number;              // 0.0 ~ 1.0 (based on age)
  estimatedDailyKwh: number;       // Weather-adjusted
  estimatedMonthlyKwh: number;
  estimatedRevenue: number;        // Based on current SMP
}

export interface PowerPlantCreate {
  name: string;
  type: PlantType;
  capacity: number;
  installDate: string;
  contractType: ContractType;
  location: {
    address: string;
    lat?: number;
    lng?: number;
  };
  roofDirection?: RoofDirection;
  status?: PlantStatus;  // Operating status (default: active)
}

export interface PowerPlantUpdate extends Partial<PowerPlantCreate> {
  id: string;
}

// Display helpers
export const PLANT_TYPE_LABELS: Record<PlantType, { label: string; icon: string }> = {
  solar: { label: '태양광', icon: '☀️' },
  wind: { label: '풍력', icon: '💨' },
  ess: { label: 'ESS', icon: '🔋' },
};

export const CONTRACT_TYPE_LABELS: Record<ContractType, { label: string; description: string }> = {
  net_metering: { label: '상계거래', description: '전기요금 차감' },
  ppa: { label: 'PPA', description: '현금 수익' },
};

export const ROOF_DIRECTION_LABELS: Record<RoofDirection, string> = {
  south: '남향',
  east: '동향',
  west: '서향',
  flat: '평지',
};

export const PLANT_STATUS_LABELS: Record<PlantStatus, { label: string; color: string; icon: string }> = {
  active: { label: '운영중', color: '#10b981', icon: '✓' },
  maintenance: { label: '점검중', color: '#f59e0b', icon: '🔧' },
  paused: { label: '중지', color: '#9ca3af', icon: '⏸' },
};
