import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/",
  build: {
    outDir: "../docs",
    emptyOutDir: false,
    rollupOptions: {
      input: {
        home: resolve(import.meta.dirname, "index.html"),
        notFound: resolve(import.meta.dirname, "404.html"),
        history: resolve(import.meta.dirname, "history/first-48-hours/index.html")
      }
    }
  }
});
