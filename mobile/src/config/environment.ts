/**
 * Environment Configuration for RE-BMS Mobile App
 * 
 * Change these values based on your deployment:
 * - LOCAL: Development on the same machine
 * - DOCKER: Docker deployment with ngrok or port forwarding
 * - PRODUCTION: Production server with custom domain
 */

// ============================================
// CONFIGURATION - MODIFY THIS SECTION
// ============================================

/**
 * API Server URL Configuration
 * 
 * For local development:
 *   API_URL = 'http://localhost:8000'
 * 
 * For Docker with ngrok:
 *   API_URL = 'https://xxxx-xxx-xxx.ngrok-free.app'
 * 
 * For Docker with port forwarding:
 *   API_URL = 'http://YOUR_PUBLIC_IP:8000'
 * 
 * For production:
 *   API_URL = 'https://api.your-domain.com'
 */
export const API_URL = 'http://localhost:8000';

// Environment type for conditional logic
export type Environment = 'local' | 'docker' | 'production';
export const CURRENT_ENV: Environment = 'local';

// ============================================
// Derived Configuration (no need to modify)
// ============================================

export const config = {
  apiUrl: API_URL,
  environment: CURRENT_ENV,
  isProduction: CURRENT_ENV === 'production',
  isDocker: CURRENT_ENV === 'docker',
  isLocal: CURRENT_ENV === 'local',
  
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
