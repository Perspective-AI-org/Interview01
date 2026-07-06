import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development, proxy /api calls to the FastAPI backend on :8000 so the
// SPA and API share an origin. In production the backend serves the built
// assets directly, so no proxy is needed.
export default defineConfig({
  plugins: [react()],
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
    outDir: "dist",
  },
});
