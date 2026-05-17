import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "src/public",
  base: "./",
  publicDir: false,
  plugins: [react()],
  build: {
    // From react-uploader/src/public to faceattendance/uploader.
    // Moodle serves this compiled bundle through recorder.php.
    outDir: "../../../uploader",
    emptyOutDir: false,
    target: "es2020"
  }
});
