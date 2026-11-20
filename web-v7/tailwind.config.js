/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Theme-aware colors using CSS variables
        background: 'var(--color-background)',
        secondary: 'var(--color-secondary)',
        card: 'var(--color-card)',
        elevated: 'var(--color-elevated)',
        border: 'var(--color-border)',
        text: {
          primary: 'var(--color-text-primary)',
          secondary: 'var(--color-text-secondary)',
          muted: 'var(--color-text-muted)',
        },
        // eXeco DS v1 static colors (same in both themes)
        brand: '#04265E',
        primary: '#3B82F6',
        accent: '#10B981',
        warning: '#F59E0B',
        danger: '#EF4444',
        success: '#22C55E',
        smp: '#F59E0B',
        demand: '#3B82F6',
        supply: '#22C55E',
        solar: '#F59E0B',
        wind: '#06B6D4',
      },
      fontFamily: {
        sans: ['Pretendard Variable', 'Pretendard', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
