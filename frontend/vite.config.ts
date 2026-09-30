import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies /api and /health to the FastAPI service so the browser
// sees a single origin. That keeps CORS out of the way in development and makes
// the built frontend deployable behind any reverse proxy.
const apiTarget = process.env.FDM_API_URL ?? 'http://127.0.0.1:8002'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/health': { target: apiTarget, changeOrigin: true },
    },
  },
  preview: { port: 4173 },
})
