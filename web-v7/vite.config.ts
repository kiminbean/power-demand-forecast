import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // 테스트가 src/pages/Dashboard.tsx, SMPPrediction.tsx를 템플릿 리터럴로
  // 동적 import하므로 시작 시 스캐너가 이들 의존성을 찾지 못한다. 미리
  // 포함하지 않으면 브라우저 로드 중 dep 재최적화 → 페이지 리로드로
  // puppeteer 실행 컨텍스트가 파괴된다(CI 콜드 캐시에서 재현).
  optimizeDeps: {
    include: ['react', 'react-dom/client', 'react-router-dom', 'recharts', 'lucide-react', 'clsx', 'leaflet', 'react-leaflet'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
