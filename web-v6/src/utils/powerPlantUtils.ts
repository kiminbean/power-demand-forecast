/**
 * Power Plant Utility Functions
 * eXeco Web v6.2.0
 * Calculation and estimation utilities for small-scale power plants
 */

import type { WeatherCondition, RoofDirection } from '../types';

// Seasonal peak sun hours (PSH) for Jeju, Korea
const SEASONAL_PSH: Record<string, number> = {
  spring: 4.0,  // March-May
  summer: 3.5,  // June-August (cloudy, typhoon season)
  fall: 4.0,    // September-November
  winter: 3.0,  // December-February
};

// Weather impact on solar generation
const WEATHER_FACTORS: Record<WeatherCondition, number> = {
  clear: 1.0,
  partly_cloudy: 0.7,
  cloudy: 0.4,
  rainy: 0.15,
};

// Roof direction efficiency factors (relative to south-facing)
const DIRECTION_FACTORS: Record<RoofDirection, number> = {
  south: 1.0,
  east: 0.85,
  west: 0.85,
  flat: 0.92,
};

/**
 * Calculate current season based on month
 */
function getCurrentSeason(): 'spring' | 'summer' | 'fall' | 'winter' {
  const month = new Date().getMonth();
  if (month >= 2 && month <= 4) return 'spring';
  if (month >= 5 && month <= 7) return 'summer';
  if (month >= 8 && month <= 10) return 'fall';
  return 'winter';
}

/**
 * Calculate efficiency degradation based on installation date
 * Year 1: 3% degradation (LID - Light Induced Degradation)
 * Year 2+: 0.6% per year
 */
export function calculateEfficiency(installDate: string): number {
  const install = new Date(installDate);
  const now = new Date();
  const yearDiff = now.getFullYear() - install.getFullYear();
  const monthDiff = now.getMonth() - install.getMonth();
  const totalYears = yearDiff + (monthDiff / 12);

  if (totalYears <= 0) return 1.0;

  // Year 1: 3% degradation (LID)
  // Year 2+: 0.6% per year
  const degradation = totalYears <= 1
    ? totalYears * 3  // LID degradation (up to 3% in first year)
    : 3 + (totalYears - 1) * 0.6;  // LID + annual degradation

  // Minimum efficiency is 80% (after ~30 years)
  return Math.max(0.8, (100 - degradation) / 100);
}

/**
 * Estimate daily generation for a solar plant
 */
export function estimateDailyGeneration(
  capacityKw: number,
  efficiency: number,
  weather: WeatherCondition = 'clear',
  direction: RoofDirection = 'south'
): number {
  const season = getCurrentSeason();
  const psh = SEASONAL_PSH[season];
  const weatherFactor = WEATHER_FACTORS[weather];
  const directionFactor = DIRECTION_FACTORS[direction];

  // Daily generation = Capacity × PSH × Efficiency × Weather × Direction
  return capacityKw * psh * efficiency * weatherFactor * directionFactor;
}

/**
 * Estimate monthly generation
 */
export function estimateMonthlyGeneration(
  capacityKw: number,
  efficiency: number,
  weather: WeatherCondition = 'clear',
  direction: RoofDirection = 'south'
): number {
  const dailyGen = estimateDailyGeneration(capacityKw, efficiency, weather, direction);
  return dailyGen * 30;
}

/**
 * Estimate revenue based on generation and SMP price
 */
export function estimateRevenue(dailyKwh: number, smpPrice: number): number {
  return dailyKwh * smpPrice;
}

/**
 * Get efficiency status and color
 */
export function getEfficiencyStatus(efficiency: number): { status: string; color: string } {
  if (efficiency >= 0.95) {
    return { status: '우수', color: '#22c55e' };  // green
  } else if (efficiency >= 0.90) {
    return { status: '양호', color: '#3b82f6' };  // blue
  } else if (efficiency >= 0.85) {
    return { status: '보통', color: '#f59e0b' };  // orange
  } else {
    return { status: '점검 필요', color: '#ef4444' };  // red
  }
}

/**
 * Format capacity for display
 */
export function formatCapacity(capacityKw: number): string {
  if (capacityKw >= 1000) {
    return `${(capacityKw / 1000).toFixed(1)} MW`;
  }
  return `${capacityKw.toFixed(1)} kW`;
}

/**
 * Format revenue for display
 */
export function formatRevenue(revenue: number): string {
  if (revenue >= 10000) {
    return `${(revenue / 10000).toFixed(1)}만원`;
  }
  return `${revenue.toFixed(0)}원`;
}

/**
 * Map weather icon/code to weather condition
 */
export function mapWeatherCondition(icon: string): WeatherCondition {
  const iconLower = icon.toLowerCase();
  if (iconLower.includes('rain') || iconLower.includes('비')) {
    return 'rainy';
  } else if (iconLower.includes('cloud') || iconLower.includes('흐림')) {
    return 'cloudy';
  } else if (iconLower.includes('partly') || iconLower.includes('구름많음')) {
    return 'partly_cloudy';
  }
  return 'clear';
}
