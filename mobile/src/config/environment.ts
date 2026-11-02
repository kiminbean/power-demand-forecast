/**
 * API configuration for RE-BMS Mobile.
 * Web defaults to the current origin; native builds default to the public Funnel.
 * Set EXPO_PUBLIC_API_URL to override either default (origin or origin plus base path).
 * Expo embeds EXPO_PUBLIC_* values in the bundle, so never put secrets here.
 */
export const NATIVE_API_URL = 'https://ib-macbookpro.tail918684.ts.net:10000';
declare const process: { env: { EXPO_PUBLIC_API_URL?: string } };
export const API_URL_OVERRIDE = process.env.EXPO_PUBLIC_API_URL?.trim().replace(/\/+$/, '') || undefined;
export const API_URL = API_URL_OVERRIDE || '';

// Environment type for conditional logic
export type Environment = 'local' | 'docker' | 'production';
export const CURRENT_ENV: Environment = 'docker';

// ============================================
// Derived Configuration (no need to modify)
// ============================================

const environmentFlags = (environment: Environment) => ({
  isProduction: environment === 'production',
  isDocker: environment === 'docker',
  isLocal: environment === 'local',
});

export const config = {
  apiUrl: API_URL,
  environment: CURRENT_ENV,
  ...environmentFlags(CURRENT_ENV),
  
  // API Timeout settings
  timeout: {
    default: 10000,  // 10 seconds
    long: 30000,     // 30 seconds for predictions
  },
  
  // Retry configuration
  retry: {
    maxAttempts: 3,
    backoffMs: 1000,
  },
};

export default config;
