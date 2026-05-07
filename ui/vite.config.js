import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /predict, /health etc. to Flask so you can also use relative URLs
    proxy: {
      "/predict": "http://localhost:5001",
      "/health":  "http://localhost:5001",
      "/intents": "http://localhost:5001",
      "/history": "http://localhost:5001",
    },
  },
})
