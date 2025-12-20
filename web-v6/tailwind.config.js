/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0e1117',
        secondary: '#1a1f2c',
        card: '#1e2530',
        border: '#374151',
        primary: '#6366f1',
        accent: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        success: '#22c55e',
        smp: '#fbbf24',
        demand: '#3b82f6',
        supply: '#22c55e',
        solar: '#fbbf24',
        wind: '#06b6d4',
      },
      fontFamily: {
        sans: ['Pretendard', 'Noto Sans KR', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
