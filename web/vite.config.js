import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api/* to the FastAPI backend so the frontend can call it without CORS fuss.
    // Use 127.0.0.1 (not localhost) so Node doesn't resolve to IPv6 ::1 while
    // uvicorn listens on IPv4 — that mismatch causes proxy 404 / ECONNREFUSED.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
