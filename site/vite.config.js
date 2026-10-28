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
        about: resolve(import.meta.dirname, "about/index.html"),
        paper: resolve(import.meta.dirname, "paper/index.html"),
        community: resolve(import.meta.dirname, "community/index.html"),
        install: resolve(import.meta.dirname, "install/index.html"),
        benchmarks: resolve(import.meta.dirname, "benchmarks/index.html"),
        notFound: resolve(import.meta.dirname, "404.html"),
        history: resolve(import.meta.dirname, "history/first-48-hours/index.html")
      }
    }
  }
});
