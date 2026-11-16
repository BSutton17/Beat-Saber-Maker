import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'

const port = fs.existsSync('config.json') ? JSON.parse(fs.readFileSync('config.json', 'utf8')).port || 5175 : 5175

export default defineConfig({
  root: 'client',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: { port: 5173, proxy: { '/api': `http://127.0.0.1:${port}` } },
})
