import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Dev server — proxies /api requests to local FastAPI backend
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Output into frontend/dist — FastAPI serves this in production
    outDir: "dist",
    emptyOutDir: true,
  },
});
