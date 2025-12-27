/**
 * Power Plant Utility Functions
 * Based on Gemini consultation for Korean electricity market
 */

// Weather types for generation estimation
export type WeatherCondition = 'clear' | 'partly_cloudy' | 'cloudy' | 'rain' | 'snow';

// Weather impact factors (verified by Gemini)
const WEATHER_FACTORS: Record<WeatherCondition, number> = {
  clear: 1.0,           // 100% - optimal
  partly_cloudy: 0.7,   // 70%
  cloudy: 0.4,          // 40%
  rain: 0.15,           // 15%
  snow: 0.05,           // 5%
};

// Seasonal generation factors for Korea (hours of peak sun)
const SEASONAL_HOURS: Record<string, number> = {
  spring: 4.0,    // March-May
  summer: 3.5,    // June-August (high temp reduces efficiency)
  fall: 4.0,      // September-November
  winter: 3.0,    // December-February
};

// Roof direction efficiency factors
const DIRECTION_FACTORS: Record<string, number> = {
  south: 1.0,     // Optimal
  east: 0.85,
  west: 0.85,
  flat: 0.95,
};

/**
 * Calculate efficiency degradation based on installation date
 * - Year 1: 3% degradation (LID - Light Induced Degradation)
 * - Year 2+: 0.6% per year
 * - Minimum efficiency: 80% (20-year guarantee)
 */
export function calculateEfficiency(installDate: string): number {
  const install = new Date(installDate);
  const now = new Date();

  // Calculate years since installation
  const yearDiff = now.getFullYear() - install.getFullYear();
  const monthDiff = now.getMonth() - install.getMonth();
  const totalYears = yearDiff + (monthDiff / 12);

  if (totalYears <= 0) return 1.0;

  // First year: 3% degradation, subsequent years: 0.6% per year
  const degradation = totalYears <= 1
    ? totalYears * 3
    : 3 + (totalYears - 1) * 0.6;

  // Minimum 80% efficiency (manufacturer guarantee)
  return Math.max(0.8, (100 - degradation) / 100);
}

/**
 * Get the current season based on date
 */
export function getCurrentSeason(): 'spring' | 'summer' | 'fall' | 'winter' {
  const month = new Date().getMonth() + 1;
  if (month >= 3 && month <= 5) return 'spring';
  if (month >= 6 && month <= 8) return 'summer';
  if (month >= 9 && month <= 11) return 'fall';
  return 'winter';
}

/**
 * Estimate daily generation in kWh
 * Formula: Capacity (kW) x Peak Sun Hours x Efficiency x Weather Factor x Direction Factor
 */
export function estimateDailyGeneration(
  capacityKw: number,
  efficiency: number,
  weather: WeatherCondition = 'clear',
  direction: string = 'south'
): number {
  const season = getCurrentSeason();
  const peakSunHours = SEASONAL_HOURS[season];
  const weatherFactor = WEATHER_FACTORS[weather];
  const directionFactor = DIRECTION_FACTORS[direction] || 1.0;

  // Daily generation = Capacity × Hours × Efficiency × Weather × Direction
  const dailyKwh = capacityKw * peakSunHours * efficiency * weatherFactor * directionFactor;

  return Math.round(dailyKwh * 10) / 10; // Round to 1 decimal
}

/**
 * Estimate monthly generation (assume 30 days with average weather)
 */
export function estimateMonthlyGeneration(
  capacityKw: number,
  efficiency: number,
  direction: string = 'south'
): number {
  // Use average weather factor (weighted: 40% clear, 30% partly cloudy, 20% cloudy, 10% rain)
  const avgWeatherFactor = 0.4 * 1.0 + 0.3 * 0.7 + 0.2 * 0.4 + 0.1 * 0.15;

  const season = getCurrentSeason();
  const peakSunHours = SEASONAL_HOURS[season];
  const directionFactor = DIRECTION_FACTORS[direction] || 1.0;

  const dailyKwh = capacityKw * peakSunHours * efficiency * avgWeatherFactor * directionFactor;
  const monthlyKwh = dailyKwh * 30;

  return Math.round(monthlyKwh);
}

/**
 * Estimate revenue based on generation and SMP price
 * For PPA users: actual cash income
 * For Net Metering users: bill credit (equivalent value)
 */
export function estimateRevenue(
  dailyKwh: number,
  smpPrice: number, // won/kWh
  contractType: 'net_metering' | 'ppa' = 'ppa'
): number {
  // Daily revenue
  const dailyRevenue = dailyKwh * smpPrice;

  // Monthly revenue (30 days)
  const monthlyRevenue = dailyRevenue * 30;

  // Net metering users get bill credit (same value but displayed differently)
  return Math.round(monthlyRevenue);
}

/**
 * Get efficiency status text and color
 */
export function getEfficiencyStatus(efficiency: number): {
  text: string;
  color: string;
  yearText: string;
} {
  const effPercent = Math.round(efficiency * 100);

  // Estimate years based on efficiency
  let years = 0;
  if (efficiency >= 0.97) years = 1;
  else if (efficiency >= 0.94) years = Math.round((100 - effPercent - 3) / 0.6) + 1;
  else years = Math.round((100 - effPercent - 3) / 0.6) + 1;

  let color = '#22c55e'; // green
  if (efficiency < 0.9) color = '#f97316'; // orange
  if (efficiency < 0.85) color = '#ef4444'; // red

  return {
    text: `${effPercent}%`,
    color,
    yearText: years <= 1 ? '1년차' : `${years}년차`,
  };
}

/**
 * Map Korean weather condition to WeatherCondition type
 */
export function mapWeatherCondition(koreanWeather: string): WeatherCondition {
  const weatherMap: Record<string, WeatherCondition> = {
    '맑음': 'clear',
    '구름조금': 'partly_cloudy',
    '구름많음': 'cloudy',
    '흐림': 'cloudy',
    '비': 'rain',
    '눈': 'snow',
    '소나기': 'rain',
  };

  return weatherMap[koreanWeather] || 'partly_cloudy';
}

/**
 * Get weather display info
 */
export function getWeatherDisplay(weather: WeatherCondition): {
  icon: string;
  text: string;
  factor: number;
} {
  const displays: Record<WeatherCondition, { icon: string; text: string }> = {
    clear: { icon: '☀️', text: '맑음' },
    partly_cloudy: { icon: '⛅', text: '약간 흐림' },
    cloudy: { icon: '☁️', text: '흐림' },
    rain: { icon: '🌧️', text: '비' },
    snow: { icon: '❄️', text: '눈' },
  };

  return {
    ...displays[weather],
    factor: WEATHER_FACTORS[weather],
  };
}

/**
 * Format capacity for display
 */
export function formatCapacity(capacityKw: number): string {
  if (capacityKw >= 1000) {
    return `${(capacityKw / 1000).toFixed(1)} MW`;
  }
  return `${capacityKw} kW`;
}

/**
 * Format revenue for display
 */
export function formatRevenue(revenue: number): string {
  if (revenue >= 1000000) {
    return `${(revenue / 1000000).toFixed(1)}백만원`;
  }
  if (revenue >= 10000) {
    return `${(revenue / 10000).toFixed(1)}만원`;
  }
  return `${revenue.toLocaleString()}원`;
}
