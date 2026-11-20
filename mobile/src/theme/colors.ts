/**
 * eXeco DS v1 — shared design tokens (mobile)
 * Brand navy #04265E on dark command-center surfaces.
 * Dark surfaces: bg #0B1220, card #121B31, border rgba(148,163,184,0.16)
 * Accents: emerald #10B981, success #22C55E, warning #F59E0B, danger #EF4444, info #3B82F6
 */

export const colors = {
  // Brand
  navy: '#04265E',

  // Background / surfaces (dark)
  bg: '#0B1220',
  surface: '#121B31',
  surfaceAlt: '#1B2540',
  background: {
    primary: '#0B1220',
    secondary: '#121B31',
    tertiary: '#1B2540',
    card: '#121B31',
  },

  // Text
  text: {
    primary: '#F1F5F9',
    secondary: '#94A3B8',
    muted: '#64748B',
    inverse: '#0B1220',
  },

  // Brand / interactive accents
  brand: {
    primary: '#10B981', // emerald — interactive accent on dark surfaces
    secondary: '#3B82F6', // info blue
    accent: '#10B981',
    navy: '#04265E',
  },

  // Status
  status: {
    success: '#22C55E',
    warning: '#F59E0B',
    danger: '#EF4444',
    info: '#3B82F6',
  },

  // Semantic accents (flat, for direct reference)
  accent: '#10B981',
  success: '#22C55E',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#3B82F6',

  // Chart Colors
  chart: {
    smp: '#10B981',
    smpBand: 'rgba(16, 185, 129, 0.2)',
    generation: '#22C55E',
    bid: '#F59E0B',
    solar: '#F59E0B',
    wind: '#3B82F6',
  },

  // Segment Colors (10 segments price gradient)
  segments: [
    '#10B981', // 1 - Green (lowest price)
    '#34D399',
    '#6EE7B7',
    '#A7F3D0',
    '#FEF3C7', // 5 - Yellow-ish (mid)
    '#FDE68A',
    '#FCD34D',
    '#FBBF24',
    '#F59E0B',
    '#EF4444', // 10 - Red (highest price)
  ],

  // Border
  border: {
    primary: 'rgba(148, 163, 184, 0.16)',
    secondary: 'rgba(148, 163, 184, 0.28)',
    focus: '#10B981',
  },

  // Gradient
  gradient: {
    card: ['#121B31', '#1B2540'],
    header: ['#0B1220', '#121B31'],
    accent: ['#10B981', '#3B82F6'],
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
