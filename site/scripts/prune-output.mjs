// Runs after `vite build`. The build keeps ../docs (it also holds the handbook), so the files of earlier builds
// would stay beside the new ones: remove the hashed bundles that no built page references, and the captures and
// fonts that have left site/public. Published files with stable names in docs/assets are never touched.
import { readdir, readFile, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = resolve(siteRoot, "..", "docs");
const hashedBundle = /^[\w.-]+-[\w-]{8}\.(?:js|css)$/;

async function names(dir) {
  try {
    return (await readdir(dir, { withFileTypes: true })).filter((entry) => entry.isFile()).map((entry) => entry.name);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
}

async function pages(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) found.push(...await pages(path));
    else if (entry.name.endsWith(".html")) found.push(path);
  }
  return found;
}

const referenced = new Set();
for (const page of await pages(outRoot)) {
  for (const match of (await readFile(page, "utf8")).matchAll(/\/assets\/([\w.-]+\.(?:js|css))/g)) referenced.add(match[1]);
}
const removed = [];
for (const name of await names(join(outRoot, "assets"))) {
  if (hashedBundle.test(name) && !referenced.has(name)) removed.push(join("assets", name));
}
for (const dir of ["ui", "fonts"]) {
  const kept = new Set(await names(join(siteRoot, "public", dir)));
  for (const name of await names(join(outRoot, dir))) {
    if (!kept.has(name)) removed.push(join(dir, name));
  }
}
await Promise.all(removed.map((path) => rm(join(outRoot, path))));
if (removed.length) console.log(`pruned from docs: ${removed.sort().join(", ")}`);
