import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "src/public",
  publicDir: false,
  plugins: [react()],
  build: {
    outDir: "../../public",
    emptyOutDir: false,
    target: "es2020"
  }
});
