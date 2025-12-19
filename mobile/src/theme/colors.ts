/**
 * RE-BMS Mobile Theme Colors
 * Command Center Dark Theme
 */

export const colors = {
  // Background
  background: {
    primary: '#0e1117',
    secondary: '#1a1f2c',
    tertiary: '#2d3748',
    card: '#1e2530',
  },

  // Text
  text: {
    primary: '#ffffff',
    secondary: '#a0aec0',
    muted: '#718096',
    inverse: '#0e1117',
  },

  // Brand
  brand: {
    primary: '#6366f1',    // Indigo
    secondary: '#8b5cf6',  // Purple
    accent: '#10b981',     // Emerald
  },

  // Status
  status: {
    success: '#10b981',
    warning: '#fbbf24',
    danger: '#ef4444',
    info: '#3b82f6',
  },

  // Chart Colors
  chart: {
    smp: '#6366f1',
    smpBand: 'rgba(99, 102, 241, 0.2)',
    generation: '#10b981',
    bid: '#fbbf24',
    solar: '#fbbf24',
    wind: '#3b82f6',
  },

  // Segment Colors (10 segments gradient)
  segments: [
    '#10b981', // 1 - Green (lowest price)
    '#34d399',
    '#6ee7b7',
    '#a7f3d0',
    '#fef3c7', // 5 - Yellow-ish (mid)
    '#fde68a',
    '#fcd34d',
    '#fbbf24',
    '#f59e0b',
    '#ef4444', // 10 - Red (highest price)
  ],

  // Border
  border: {
    primary: '#374151',
    secondary: '#4b5563',
    focus: '#6366f1',
  },

  // Gradient
  gradient: {
    card: ['#1a1f2c', '#2d3748'],
    header: ['#1a1f2c', '#0e1117'],
    accent: ['#6366f1', '#8b5cf6'],
  },
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const borderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
};

export const fontSize = {
  xs: 10,
  sm: 12,
  md: 14,
  lg: 16,
  xl: 18,
  xxl: 24,
  xxxl: 32,
};

export default { colors, spacing, borderRadius, fontSize };
