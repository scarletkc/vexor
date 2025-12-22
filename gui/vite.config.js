const { defineConfig } = require("vite");
const vue = require("@vitejs/plugin-vue");
const path = require("path");

module.exports = defineConfig({
  plugins: [vue()],
  base: "./",
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true
  },
  server: {
    port: 5173,
    strictPort: true
  }
});
