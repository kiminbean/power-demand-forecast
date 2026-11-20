/**
 * eXeco Light DS — shared design tokens (mobile)
 * Brand navy #1E3A8A on light surfaces.
 * Light surfaces: bg #F1F5F9, card #FFFFFF, border #E2E8F0
 * Accents: brand blue #2563EB, amber #F59E0B, success #16A34A, danger #EF4444
 */

export const colors = {
  // Brand
  navy: '#1E3A8A',

  // Background / surfaces (light)
  bg: '#F1F5F9',
  surface: '#FFFFFF',
  surfaceAlt: '#F8FAFC',
  background: {
    primary: '#F1F5F9',
    secondary: '#FFFFFF',
    tertiary: '#F8FAFC',
    card: '#FFFFFF',
  },

  // Text
  text: {
    primary: '#1E293B',
    secondary: '#64748B',
    muted: '#94A3B8',
    inverse: '#F8FAFC',
  },

  // Brand / interactive accents
  brand: {
    primary: '#2563EB', // brand blue — interactive accent on light surfaces
    secondary: '#1D4ED8', // deep blue
    accent: '#2563EB',
    navy: '#1E3A8A',
  },

  // Status
  status: {
    success: '#16A34A',
    warning: '#F59E0B',
    danger: '#EF4444',
    info: '#2563EB',
  },

  // Semantic accents (flat, for direct reference)
  accent: '#16A34A', // used as success-green across screens ("green" mappings)
  success: '#16A34A',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#2563EB',

  // Chart Colors
  chart: {
    smp: '#FCD34D',
    smpBand: 'rgba(252, 211, 77, 0.35)',
    generation: '#16A34A',
    bid: '#F59E0B',
    solar: '#F59E0B',
    wind: '#2563EB',
  },

  // Segment Colors (10 segments price gradient)
  segments: [
    '#059669', // 1 - Green (lowest price)
    '#10B981',
    '#34D399',
    '#6EE7B7',
    '#FDE68A', // 5 - Yellow-ish (mid)
    '#FCD34D',
    '#FBBF24',
    '#F59E0B',
    '#F97316',
    '#EF4444', // 10 - Red (highest price)
  ],

  // Border
  border: {
    primary: '#E2E8F0',
    secondary: '#CBD5E1',
    focus: '#2563EB',
  },

  // Gradient
  gradient: {
    card: ['#FFFFFF', '#F8FAFC'],
    header: ['#FFFFFF', '#F8FAFC'],
    accent: ['#1D4ED8', '#2563EB'],
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
  sm: 6,
  md: 10,
  lg: 14,
  xl: 20,
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

// Typography rhythm: big bold numbers, small labels
export const typography = {
  number: { fontSize: 32, fontWeight: '700' as const },
  numberSmall: { fontSize: 20, fontWeight: '700' as const },
  label: { fontSize: 11, fontWeight: '600' as const, color: colors.text.secondary },
  title: { fontSize: 18, fontWeight: '700' as const, color: colors.text.primary },
  body: { fontSize: 14, fontWeight: '400' as const, color: colors.text.primary },
};

export default { colors, spacing, borderRadius, fontSize, typography };
