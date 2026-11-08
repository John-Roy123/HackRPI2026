import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import { photoApi } from './server/photoApi.js'

const root = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [photoApi({ root })],
  server: {
    host: true,
    // Uploads and description edits shouldn't trigger page reloads.
    watch: { ignored: ['**/images/**', '**/descriptions.json'] },
  },
  preview: {
    host: true,
  },
})
