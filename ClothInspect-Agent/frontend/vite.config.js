import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 构建产物由 FastAPI 后端托管（backend 的 spa_fallback 会兜底到 index.html）
export default defineConfig({
  plugins: [vue()],
  base: '/',
  server: {
    port: 5173,
    host: '127.0.0.1',
    // 开发模式：把 /api 与 /media 代理到后端，避免跨域
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/media': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200
  }
})
